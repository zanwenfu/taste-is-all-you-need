from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from taste.brains.planner_transport import (
    PLANNER_RECEIPT_BRANCH,
    LLMPlannerTransport,
    PlannerCompletionError,
    PlannerReceiptError,
    load_planner_transport_evidence,
    planner_transport_intent_path,
    planner_transport_outcome_path,
)
from taste.llm import MODEL_PLANNER
from taste.memstore import Store
from taste.pricing import call_cost, ensure_priced, max_call_cost_usd, table_sha

from .fakes import FakeLLM, FakeTurn


class ReadyFakeLLM(FakeLLM):
    def __init__(self, turns: list[FakeTurn] | None = None) -> None:
        super().__init__(turns, model=MODEL_PLANNER)
        self.max_attempts = 1
        self.ready_models: list[tuple[str, ...]] = []

    def ensure_ready(self, *models: str) -> None:
        self.ready_models.append(models)

    def call(self, **kwargs: Any) -> Any:
        completion = super().call(**kwargs)
        return replace(
            completion,
            provider=ensure_priced(completion.model).provider,
        )


@pytest.fixture
def store(tmp_path: Path):
    opened = Store.open(tmp_path / "repo", "planner-transport-test")
    yield opened
    opened.close()


def make_transport(
    llm: Any,
    store: Store,
    *,
    control=None,
    journal=None,
    mutation_lock: threading.RLock | None = None,
    **kwargs: Any,
) -> LLMPlannerTransport:
    branch = control or store.branch("central-control", producer="planner-transport-test")
    receipt_branch = journal or store.branch(
        PLANNER_RECEIPT_BRANCH,
        producer="planner-transport-test",
    )
    lock = mutation_lock or threading.RLock()
    return LLMPlannerTransport(
        llm,
        store=store,
        control=branch,
        journal=receipt_branch,
        mutation_lock=lock,
        **kwargs,
    )


def evidence(transport: LLMPlannerTransport, request_id: str):
    found = load_planner_transport_evidence(transport.journal.head, request_id)
    assert found is not None
    return found


def test_llm_transport_preflights_tool_free_call_and_prices_exact_usage(store: Store) -> None:
    llm = ReadyFakeLLM(
        [
            FakeTurn(
                text='{"schema":"proposal"}',
                input_tokens=120,
                output_tokens=30,
                cache_read_tokens=80,
                cache_creation_tokens=20,
            )
        ]
    )
    transport = make_transport(llm, store, max_tokens=2048)

    result = transport.complete(request_id="attempt-1", system="planner system", prompt="world")

    assert result.text == '{"schema":"proposal"}'
    assert llm.ready_models == [(MODEL_PLANNER,)]
    assert llm.call_count == 1
    call = llm.calls[0]
    assert call == {
        "model": MODEL_PLANNER,
        "system": "planner system",
        "messages": [{"role": "user", "content": "world"}],
        "tools": None,
        "max_tokens": 2048,
        "temperature": 0.0,
        "role": "planner",
    }
    usage = result.telemetry.usage
    assert usage is not None
    assert usage.to_dict()["cache_write_tokens"] == 20
    expected_billed, expected_work = call_cost(
        MODEL_PLANNER,
        input_tokens=120,
        output_tokens=30,
        cache_read_tokens=80,
        cache_write_tokens=20,
    )
    assert result.telemetry.billed_usd == pytest.approx(expected_billed)
    assert result.telemetry.work_usd == pytest.approx(expected_work)
    assert result.telemetry.pricing_table_sha == table_sha()
    receipt = evidence(transport, "attempt-1")
    assert receipt.status == "completed"
    assert receipt.response == result.text


@pytest.mark.parametrize(
    ("turn", "match"),
    [
        (FakeTurn(text="{}", stop_reason="max_tokens"), "did not finish"),
        (FakeTurn(text="{}", tool_calls=[("surprise", {})]), "zero tool calls"),
        (FakeTurn(text="   "), "non-empty text"),
    ],
)
def test_charged_invalid_completion_is_receipted_and_replayed_without_call(
    store: Store,
    turn: FakeTurn,
    match: str,
) -> None:
    first_llm = ReadyFakeLLM([turn])
    first = make_transport(first_llm, store)
    with pytest.raises(PlannerCompletionError, match=match) as raised:
        first.complete(request_id="attempt-invalid", system="system", prompt="prompt")
    assert raised.value.telemetry.cost_known is True
    receipt = evidence(first, "attempt-invalid")
    assert receipt.status == "invalid"

    replacement_llm = ReadyFakeLLM([])
    replacement = make_transport(
        replacement_llm,
        store,
        control=first.control,
        journal=first.journal,
        mutation_lock=first.mutation_lock,
    )
    with pytest.raises(PlannerCompletionError, match=match) as replayed:
        replacement.complete(request_id="attempt-invalid", system="system", prompt="prompt")
    assert replayed.value.telemetry == raised.value.telemetry
    assert replacement_llm.call_count == 0
    assert replacement_llm.ready_models == []


def test_unauditable_usage_is_terminal_and_cost_unknown(store: Store) -> None:
    class BadUsageLLM:
        def __init__(self) -> None:
            self.calls = 0

        def ensure_ready(self, *models: str) -> None:
            pass

        def call(self, **kwargs: Any) -> Any:
            self.calls += 1
            return SimpleNamespace(
                model=MODEL_PLANNER,
                provider="fake",
                usage=SimpleNamespace(input_tokens=3),
                text_blocks=("{}",),
                tool_calls=(),
                stop_reason="end_turn",
            )

    llm = BadUsageLLM()
    transport = make_transport(llm, store)
    with pytest.raises(PlannerCompletionError, match="unauditable") as raised:
        transport.complete(request_id="bad-usage", system="system", prompt="prompt")
    assert raised.value.telemetry.cost_known is False
    assert evidence(transport, "bad-usage").status == "invalid"


def test_provider_model_mismatch_is_terminal_and_cost_unknown(store: Store) -> None:
    class WrongProviderLLM(ReadyFakeLLM):
        def call(self, **kwargs: Any) -> Any:
            return replace(super().call(**kwargs), provider="openai")

    llm = WrongProviderLLM([FakeTurn(text="{}")])
    transport = make_transport(llm, store)

    with pytest.raises(PlannerCompletionError, match="unauditable") as raised:
        transport.complete(
            request_id="wrong-provider",
            system="system",
            prompt="prompt",
        )

    assert raised.value.telemetry.source == "provider_model_mismatch"
    assert raised.value.telemetry.cost_known is False
    assert raised.value.telemetry.model == MODEL_PLANNER
    assert raised.value.telemetry.provider == "openai"
    receipt = evidence(transport, "wrong-provider")
    assert receipt.status == "invalid"
    assert receipt.telemetry == raised.value.telemetry

    replacement_llm = ReadyFakeLLM([])
    replacement = make_transport(
        replacement_llm,
        store,
        control=transport.control,
        journal=transport.journal,
        mutation_lock=transport.mutation_lock,
    )
    with pytest.raises(PlannerCompletionError) as replayed:
        replacement.complete(
            request_id="wrong-provider",
            system="system",
            prompt="prompt",
        )
    assert replayed.value.telemetry == raised.value.telemetry
    assert replacement_llm.call_count == 0
    assert replacement_llm.ready_models == []


def test_kill_after_intent_leaves_pending_attempt_that_never_recalls_provider(
    tmp_path: Path,
) -> None:
    class KilledLLM:
        def __init__(self) -> None:
            self.calls = 0

        def ensure_ready(self, *models: str) -> None:
            pass

        def call(self, **kwargs: Any) -> Any:
            self.calls += 1
            raise KeyboardInterrupt

    root = tmp_path / "repo"
    first_store = Store.open(root, "planner-crash")
    killed = KilledLLM()
    first_control = first_store.branch("central-control", producer="planner-transport-test")
    first_lock = threading.RLock()
    first = make_transport(
        killed,
        first_store,
        control=first_control,
        mutation_lock=first_lock,
    )
    with pytest.raises(KeyboardInterrupt):
        first.complete(request_id="crash-intent", system="system", prompt="prompt")
    assert evidence(first, "crash-intent").status == "pending"
    first_control.close()
    first_store.close()

    second_store = Store.open(root, "planner-crash")
    try:
        replacement_llm = ReadyFakeLLM([])
        second_control = second_store.branch("central-control", producer="planner-transport-test")
        replacement = make_transport(
            replacement_llm,
            second_store,
            control=second_control,
            mutation_lock=threading.RLock(),
        )
        with pytest.raises(PlannerCompletionError, match="surviving pending") as raised:
            replacement.complete(
                request_id="crash-intent",
                system="system",
                prompt="prompt",
            )
        assert raised.value.telemetry.cost_known is False
        assert replacement_llm.call_count == 0
        assert replacement_llm.ready_models == []
        second_control.close()
    finally:
        second_store.close()


def test_kill_after_authoritative_intent_before_control_mirror_never_calls_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    first_store = Store.open(root, "planner-intent-mirror-crash")
    first_llm = ReadyFakeLLM([FakeTurn(text="must not be called")])
    first = make_transport(first_llm, first_store)
    original_commit = first._commit_record

    def crash_after_journal_intent(branch, path: str, raw: dict[str, Any], reason: str) -> str:
        recorded = original_commit(branch, path, raw, reason)
        if branch is first.journal and path.endswith("/intent.json"):
            raise KeyboardInterrupt
        return recorded

    monkeypatch.setattr(first, "_commit_record", crash_after_journal_intent)
    with pytest.raises(KeyboardInterrupt):
        first.complete(request_id="intent-mirror-crash", system="system", prompt="prompt")
    assert first_llm.call_count == 0
    assert evidence(first, "intent-mirror-crash").status == "pending"
    assert load_planner_transport_evidence(first.control.head, "intent-mirror-crash") is None
    first_store.close()

    second_store = Store.open(root, "planner-intent-mirror-crash")
    try:
        replacement_llm = ReadyFakeLLM([])
        replacement = make_transport(replacement_llm, second_store)
        with pytest.raises(PlannerCompletionError, match="surviving pending"):
            replacement.complete(
                request_id="intent-mirror-crash",
                system="system",
                prompt="prompt",
            )
        assert replacement_llm.call_count == 0
        mirrored = load_planner_transport_evidence(
            replacement.control.head,
            "intent-mirror-crash",
        )
        assert mirrored is not None
        assert mirrored.status == "pending"
    finally:
        second_store.close()


def test_kill_after_completed_receipt_replays_without_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    first_store = Store.open(root, "planner-crash")
    first_llm = ReadyFakeLLM([FakeTurn(text="exact response")])
    first_control = first_store.branch("central-control", producer="planner-transport-test")
    first = make_transport(
        first_llm,
        first_store,
        control=first_control,
        mutation_lock=threading.RLock(),
    )
    original_commit = first._commit_record

    def crash_after_completed_write(branch, path: str, raw: dict[str, Any], reason: str) -> str:
        recorded = original_commit(branch, path, raw, reason)
        if branch is first.journal and raw.get("status") == "completed":
            raise KeyboardInterrupt
        return recorded

    monkeypatch.setattr(first, "_commit_record", crash_after_completed_write)
    with pytest.raises(KeyboardInterrupt):
        first.complete(request_id="crash-receipt", system="system", prompt="prompt")
    assert first_llm.call_count == 1
    first_control.close()
    first_store.close()

    second_store = Store.open(root, "planner-crash")
    try:
        replacement_llm = ReadyFakeLLM([])
        second_control = second_store.branch("central-control", producer="planner-transport-test")
        replacement = make_transport(
            replacement_llm,
            second_store,
            control=second_control,
            mutation_lock=threading.RLock(),
        )
        replayed = replacement.complete(
            request_id="crash-receipt",
            system="system",
            prompt="prompt",
        )
        assert replayed.text == "exact response"
        assert replayed.telemetry.cost_known is True
        assert replacement_llm.call_count == 0
        assert replacement_llm.ready_models == []
        second_control.close()
    finally:
        second_store.close()


def test_receipt_is_bound_to_exact_prompt_model_and_config(store: Store) -> None:
    transport = make_transport(ReadyFakeLLM([FakeTurn(text="response")]), store)
    transport.complete(request_id="bound", system="system", prompt="prompt")

    replacement = make_transport(
        ReadyFakeLLM([]),
        store,
        control=transport.control,
        journal=transport.journal,
        mutation_lock=transport.mutation_lock,
    )
    with pytest.raises(PlannerReceiptError, match="reused with different"):
        replacement.complete(request_id="bound", system="system", prompt="changed")


@pytest.mark.parametrize("damage", ["tampered", "malformed"])
def test_tampered_or_malformed_receipt_bytes_fail_closed(store: Store, damage: str) -> None:
    transport = make_transport(ReadyFakeLLM([FakeTurn(text="response")]), store)
    transport.complete(request_id="tamper", system="system", prompt="prompt")
    outcome_path = planner_transport_outcome_path("tamper")
    if damage == "tampered":
        raw = json.loads(transport.control.read(outcome_path) or "")
        raw["response"] = "forged"
        transport.control.write(outcome_path, json.dumps(raw) + "\n")
    else:
        transport.control.write(outcome_path, "{not-json\n")
    transport.control.checkpoint(f"fault injection: {damage} receipt")

    replacement_llm = ReadyFakeLLM([])
    replacement = make_transport(
        replacement_llm,
        store,
        control=transport.control,
        journal=transport.journal,
        mutation_lock=transport.mutation_lock,
    )
    with pytest.raises(PlannerReceiptError, match="rewrote"):
        replacement.complete(request_id="tamper", system="system", prompt="prompt")
    assert replacement_llm.call_count == 0


@pytest.mark.parametrize("damage", ["unexpected_path", "outcome_without_intent"])
def test_malformed_authority_tree_blocks_every_new_provider_call(
    store: Store,
    damage: str,
) -> None:
    replacement_llm = ReadyFakeLLM([FakeTurn(text="must not be called")])
    transport = make_transport(replacement_llm, store)
    if damage == "unexpected_path":
        path = f".taste/planner/transport/unexpected-{damage}.json"
    else:
        path = planner_transport_outcome_path("orphan-outcome")
    transport.journal.checkpoint(
        f"fault injection: {damage}",
        records={path: {"request_id": "orphan-outcome"}},
    )

    with pytest.raises(PlannerReceiptError, match=r"unexpected path|no durable intent"):
        transport.complete(request_id="new-call", system="system", prompt="prompt")
    assert replacement_llm.call_count == 0
    assert replacement_llm.ready_models == []


def test_deleted_authoritative_receipt_pair_fails_closed_after_store_reopen(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    first_store = Store.open(root, "planner-delete")
    control = first_store.branch("central-control", producer="planner-transport-test")
    first_llm = ReadyFakeLLM([FakeTurn(text="response")])
    first = make_transport(
        first_llm,
        first_store,
        control=control,
        mutation_lock=threading.RLock(),
    )
    first.complete(request_id="deleted", system="system", prompt="prompt")
    journal = first.journal
    journal.path(planner_transport_intent_path("deleted")).unlink()
    journal.path(planner_transport_outcome_path("deleted")).unlink()
    journal.checkpoint("fault injection: delete entire authoritative planner receipt pair")
    control.close()
    first_store.close()

    second_store = Store.open(root, "planner-delete")
    second_control = second_store.branch("central-control", producer="planner-transport-test")
    try:
        replacement_llm = ReadyFakeLLM([])
        replacement = make_transport(
            replacement_llm,
            second_store,
            control=second_control,
            mutation_lock=threading.RLock(),
        )
        with pytest.raises(PlannerReceiptError, match="deleted"):
            replacement.complete(request_id="deleted", system="system", prompt="prompt")
        assert replacement_llm.call_count == 0
    finally:
        second_control.close()
        second_store.close()


def test_delete_then_restore_receipts_still_fails_closed_after_reopen(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    first_store = Store.open(root, "planner-delete-restore")
    control = first_store.branch("central-control", producer="planner-transport-test")
    first = make_transport(
        ReadyFakeLLM([FakeTurn(text="response")]),
        first_store,
        control=control,
        mutation_lock=threading.RLock(),
    )
    first.complete(request_id="restored", system="system", prompt="prompt")
    journal = first.journal
    paths = (
        planner_transport_intent_path("restored"),
        planner_transport_outcome_path("restored"),
    )
    saved = {path: journal.read(path) or "" for path in paths}
    for path in paths:
        journal.path(path).unlink()
    journal.checkpoint("fault injection: delete authoritative planner receipts")
    for path, raw in saved.items():
        journal.write(path, raw)
    journal.checkpoint("fault injection: restore valid authoritative planner receipts")
    control.close()
    first_store.close()

    second_store = Store.open(root, "planner-delete-restore")
    second_control = second_store.branch("central-control", producer="planner-transport-test")
    try:
        replacement_llm = ReadyFakeLLM([])
        replacement = make_transport(
            replacement_llm,
            second_store,
            control=second_control,
            mutation_lock=threading.RLock(),
        )
        with pytest.raises(PlannerReceiptError, match="deleted"):
            replacement.complete(request_id="restored", system="system", prompt="prompt")
        assert replacement_llm.call_count == 0
    finally:
        second_control.close()
        second_store.close()


@pytest.mark.parametrize("rollback_target", ["before_intent", "pending_intent"])
def test_valid_control_rollback_cannot_make_receipt_recallable(
    tmp_path: Path, rollback_target: str
) -> None:
    root = tmp_path / "repo"
    first_store = Store.open(root, "planner-rollback")
    control = first_store.branch("central-control", producer="planner-transport-test")
    before = control.head
    first = make_transport(
        ReadyFakeLLM([FakeTurn(text="response")]),
        first_store,
        control=control,
        mutation_lock=threading.RLock(),
    )
    first.complete(request_id="rollback", system="system", prompt="prompt")
    pending = next(
        state
        for state in control.history()
        if state.meta.reason == "planner transport intent: rollback"
    )
    control.rollback(
        before if rollback_target == "before_intent" else pending,
        f"fault injection: valid rollback to {rollback_target}",
    )
    control.close()
    first_store.close()

    second_store = Store.open(root, "planner-rollback")
    second_control = second_store.branch("central-control", producer="planner-transport-test")
    try:
        replacement_llm = ReadyFakeLLM([])
        replacement = make_transport(
            replacement_llm,
            second_store,
            control=second_control,
            mutation_lock=threading.RLock(),
        )
        with pytest.raises(PlannerReceiptError, match="deleted"):
            replacement.complete(request_id="rollback", system="system", prompt="prompt")
        assert replacement_llm.call_count == 0
    finally:
        second_control.close()
        second_store.close()


def test_raw_control_ref_rewind_reconciles_from_journal_without_provider_call(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    first_store = Store.open(root, "planner-raw-control-rewind")
    first_llm = ReadyFakeLLM([FakeTurn(text="exact response")])
    first = make_transport(first_llm, first_store)
    before_call = first.control.head
    first.complete(request_id="raw-control-rewind", system="system", prompt="prompt")
    charged_head = first.control.head
    assert first_store.backend.cas_update_ref(
        first.control.ref,
        before_call.id,
        charged_head.id,
    )
    first.control.backend.reset_hard_to_head()
    first_store.close()

    second_store = Store.open(root, "planner-raw-control-rewind")
    try:
        replacement_llm = ReadyFakeLLM([])
        replacement = make_transport(replacement_llm, second_store)
        replayed = replacement.complete(
            request_id="raw-control-rewind",
            system="system",
            prompt="prompt",
        )
        assert replayed.text == "exact response"
        assert replacement_llm.call_count == 0
        mirrored = load_planner_transport_evidence(
            replacement.control.head,
            "raw-control-rewind",
        )
        assert mirrored is not None
        assert mirrored.status == "completed"
    finally:
        second_store.close()


def test_raw_journal_ref_rollback_fails_against_surviving_control_mirror(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    first_store = Store.open(root, "planner-raw-journal-rewind")
    first_llm = ReadyFakeLLM([FakeTurn(text="old response"), FakeTurn(text="charged response")])
    first = make_transport(first_llm, first_store)
    first.complete(request_id="older", system="system", prompt="old prompt")
    older_journal = first.journal.head
    first.complete(request_id="rewound", system="system", prompt="prompt")
    charged_journal = first.journal.head
    assert first_store.backend.cas_update_ref(
        first.journal.ref,
        older_journal.id,
        charged_journal.id,
    )
    first.journal.backend.reset_hard_to_head()
    first_store.close()

    second_store = Store.open(root, "planner-raw-journal-rewind")
    try:
        replacement_llm = ReadyFakeLLM([])
        replacement = make_transport(replacement_llm, second_store)
        with pytest.raises(PlannerReceiptError, match="absent from authority"):
            replacement.complete(
                request_id="brand-new-after-rewind",
                system="system",
                prompt="prompt",
            )
        assert replacement_llm.call_count == 0
    finally:
        second_store.close()


def test_deleted_journal_ref_fails_against_surviving_control_after_reopen(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    first_store = Store.open(root, "planner-delete-journal-ref")
    first = make_transport(
        ReadyFakeLLM([FakeTurn(text="charged response")]),
        first_store,
    )
    first.complete(request_id="deleted-journal", system="system", prompt="prompt")
    journal_ref = first.journal.ref
    first_store.close()

    second_store = Store.open(root, "planner-delete-journal-ref")
    second_store.backend.delete_ref(journal_ref)
    assert second_store.backend.ref_sha(journal_ref) is None
    try:
        replacement_llm = ReadyFakeLLM([])
        replacement = make_transport(replacement_llm, second_store)
        with pytest.raises(PlannerReceiptError, match="absent from authority"):
            replacement.complete(
                request_id="deleted-journal",
                system="system",
                prompt="prompt",
            )
        assert replacement_llm.call_count == 0
    finally:
        second_store.close()


def test_billed_exposure_and_one_shot_budget_binding_are_conservative(store: Store) -> None:
    llm = ReadyFakeLLM([FakeTurn(text="response")])
    llm.budget_usd = 100.0
    llm.cap_on = "billed"
    transport = make_transport(llm, store, max_tokens=2048)
    expected = max_call_cost_usd(
        MODEL_PLANNER,
        max_output_tokens=2048,
        max_attempts=1,
        cap_on="billed",
    )
    assert transport.max_billed_call_usd() == pytest.approx(expected)

    with pytest.raises(PlannerReceiptError, match="bind_budget"):
        transport.complete(request_id="budgeted", system="system", prompt="prompt")
    with pytest.raises(PlannerReceiptError, match="does not cover"):
        transport.bind_budget(remaining_usd=expected / 2)
    transport.bind_budget(remaining_usd=expected)
    transport.complete(request_id="budgeted", system="system", prompt="prompt")
    admission = evidence(transport, "budgeted").admission
    assert admission["remaining_usd"] == pytest.approx(expected)
    assert admission["max_billed_call_usd"] == pytest.approx(expected)


def test_planner_transport_refuses_hidden_provider_retries(store: Store) -> None:
    llm = ReadyFakeLLM([FakeTurn(text="response")])
    llm.max_attempts = 2
    transport = make_transport(llm, store)

    with pytest.raises(PlannerReceiptError, match="exactly 1"):
        transport.complete(request_id="retrying", system="system", prompt="prompt")

    assert llm.call_count == 0
