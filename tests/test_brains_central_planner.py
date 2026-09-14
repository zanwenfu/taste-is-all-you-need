from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from taste.brains.central_planner import (
    PLANNER_ROOT,
    CentralPlanner,
    Goal,
    InvalidPlannerOutput,
    PlannerIdentityConflict,
    PlannerStateError,
    PlannerTransportError,
    PlanningRequest,
    RejectedPlannerOperation,
    StalePlanningWorld,
)
from taste.brains.contract import Contract
from taste.brains.planner_transport import (
    PLANNER_RECEIPT_BRANCH,
    LLMPlannerTransport,
    PlannerCompletion,
    PlannerCompletionError,
    PlannerTelemetry,
    PlannerUsage,
)
from taste.brains.records import (
    ArtifactRef,
    ArtifactSpec,
    Assignment,
    CriteriaRevision,
    Criterion,
    WorkerReport,
    contract_digest,
)
from taste.brains.supervisor import CentralSupervisor, ProcessExit
from taste.brains.worker_runtime import WORKER_REPORT_PATH
from taste.llm import MODEL_PLANNER
from taste.memstore import Store, Verdict
from taste.pricing import call_cost, ensure_priced, table_sha
from taste.providers.base import Usage


class FakeTransport:
    def __init__(self, responder: Callable[[str, str, str], Any]) -> None:
        self.responder = responder
        self.calls: list[tuple[str, str, str]] = []

    def complete(self, *, request_id: str, system: str, prompt: str) -> Any:
        self.calls.append((request_id, system, prompt))
        result = self.responder(request_id, system, prompt)
        if isinstance(result, str):
            return PlannerCompletion(result, PlannerTelemetry.synthetic_zero())
        return result


class UnusedLauncher:
    def launch(self, spec):  # pragma: no cover - a prepared run is stopped before launch
        raise AssertionError("planner tests never launch a worker")

    def recover(self, spec):
        return None


class NeverProcess:
    pid = 1001
    process_group_id = 1001
    launch_token = "never"
    process_identity = "never"

    def poll(self):
        return ProcessExit(exit_code=0, reaped=True)

    def ready(self):
        return False

    def terminate_tree(self, grace_seconds):
        return ProcessExit(exit_code=0, reaped=True)


@pytest.fixture
def store(tmp_path: Path):
    opened = Store.open(tmp_path / "repo", "planner-test")
    integration = opened.branch("integration", producer="central-integration")
    integration.close()
    yield opened
    opened.close()


@pytest.fixture
def goal() -> Goal:
    return Goal(
        goal_id="ship-parser",
        task="Build and validate a parser",
        success_criteria=("parser.py exists", "the parser tests pass"),
        budget_usd=12.5,
        metadata={"owner": "research"},
    )


def request_from_prompt(prompt: str) -> PlanningRequest:
    return PlanningRequest.from_dict(json.loads(prompt)["request"])


def assignment_for(
    request: PlanningRequest,
    *,
    assignment_id: str = "build-parser",
    worker: str = "worker-parser",
    output_id: str = "parser-source",
    output_path: str = "parser.py",
    depends_on: tuple[str, ...] = (),
    inputs: tuple[ArtifactRef, ...] = (),
    attempt: int = 0,
    target_budget_usd: float | None = 1.0,
    monitor_budget_usd: Any | None = 0.25,
) -> Assignment:
    contract = Contract(
        identity=worker,
        task=f"Produce {output_path}",
        inputs=tuple(item.path for item in inputs),
        outputs=(output_path,),
        success_criteria=(f"{output_path} is complete and tested",),
        budget_usd=target_budget_usd,
    )
    resources: dict[str, Any] = {"wall_timeout_seconds": 60}
    if monitor_budget_usd is not None:
        resources["monitor_budget_usd"] = monitor_budget_usd
    return Assignment(
        assignment_id=assignment_id,
        generation=request.generation,
        attempt=attempt,
        contract=contract,
        contract_digest=contract_digest(contract),
        base_state_id=request.world.integration_state_id,
        depends_on=depends_on,
        inputs=inputs,
        outputs=(ArtifactSpec(artifact_id=output_id, path=output_path),),
        resources=resources,
    )


def proposal(
    prompt: str,
    *assignments: Assignment,
    complete: bool = False,
    completion_reason: str = "",
    assessment: list[dict[str, Any]] | None = None,
    changes: dict[str, Any] | None = None,
) -> str:
    payload = json.loads(prompt)
    result = dict(payload["required_output_shape"])
    if assessment is None:
        # A completion claim has to account for every standing obligation, so
        # the default mirrors what a real planner would have to state rather
        # than letting tests declare completion for free.
        verdict = "met" if complete else "not_met"
        evidence = "satisfied by the delivered work" if complete else "work has not started"
        assessment = [
            {"criterion_id": item["criterion_id"], "verdict": verdict, "evidence": evidence}
            for item in payload.get("standing_criteria", ())
        ]
    result.update(
        assignments=[item.to_dict() for item in assignments],
        rationale="decompose by independently owned product artifacts",
        complete=complete,
        completion_reason=completion_reason,
        assessment=assessment,
        metadata={"confidence": 0.8},
    )
    result.update(changes or {})
    return json.dumps(result, sort_keys=True)


def one_assignment_response(_call_id: str, _system: str, prompt: str) -> str:
    request = request_from_prompt(prompt)
    return proposal(prompt, assignment_for(request))


def exact_telemetry() -> PlannerTelemetry:
    usage = PlannerUsage(
        input_tokens=120,
        output_tokens=30,
        cache_read_tokens=80,
        cache_write_tokens=20,
        reasoning_tokens=10,
    )
    billed, work = call_cost(
        MODEL_PLANNER,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        reasoning_tokens=usage.reasoning_tokens,
    )
    return PlannerTelemetry(
        source="llm_completion",
        cost_known=True,
        requested_model=MODEL_PLANNER,
        model=MODEL_PLANNER,
        provider="anthropic",
        usage=usage,
        billed_usd=billed,
        work_usd=work,
        pricing_table_sha=table_sha(),
        pricing_as_of=ensure_priced(MODEL_PLANNER).as_of,
    )


def planner(store: Store, response=one_assignment_response, **kwargs):
    transport = FakeTransport(response)
    return CentralPlanner(store, transport=transport, **kwargs), transport


def planner_files(store: Store) -> list[str]:
    return [
        path
        for path in store.view("central-control").head.files()
        if path.startswith(f"{PLANNER_ROOT}/")
    ]


def test_goal_is_strict_frozen_and_round_trips(goal: Goal) -> None:
    assert Goal.from_json(goal.to_json()) == goal
    with pytest.raises(TypeError):
        goal.metadata["owner"] = "changed"  # type: ignore[index]
    raw = goal.to_dict()
    raw["unknown"] = True
    with pytest.raises(ValueError, match="unknown"):
        Goal.from_dict(raw)
    with pytest.raises(ValueError, match="non-empty"):
        Goal(goal_id="x", task="task", success_criteria=())
    with pytest.raises(ValueError, match="finite"):
        Goal(goal_id="x", task="task", success_criteria=("done",), budget_usd=float("nan"))


def test_initial_plan_is_exact_durable_and_replayed_without_model_call(
    store: Store, goal: Goal
) -> None:
    central, transport = planner(store)
    plan = central.plan(goal)

    assert plan.generation == 1
    assert plan.parent_plan_id is None
    assert plan.based_on_state_id == plan.observed_heads["central-control"]
    assert plan.assignments[0].base_state_id == plan.observed_heads["integration"]
    assert central.current_plan(goal.goal_id) == plan
    assert len(transport.calls) == 1
    assert any(path.endswith("/request.json") for path in planner_files(store))
    assert any(path.endswith("/result.json") for path in planner_files(store))
    assert any("/plans/" in path for path in planner_files(store))

    replacement = FakeTransport(lambda *_: (_ for _ in ()).throw(AssertionError("called")))
    recovered = CentralPlanner(
        store,
        transport=replacement,
        control=central.control,
    )
    assert recovered.plan(goal) == plan
    assert recovered.revise(goal, operation_id="initial") == plan
    assert replacement.calls == []

    [audit] = central.planner_attempts(goal.goal_id)
    assert audit.status == "accepted"
    assert audit.telemetry.source == "legacy_string_zero"
    assert central.planner_cost(goal.goal_id) == 0.0


def test_budgeted_goal_rejects_text_only_transport_as_cost_unknown(
    store: Store,
    goal: Goal,
) -> None:
    class TextOnlyTransport:
        def complete(self, *, request_id: str, system: str, prompt: str) -> str:
            return one_assignment_response(request_id, system, prompt)

    central = CentralPlanner(store, transport=TextOnlyTransport())

    with pytest.raises(InvalidPlannerOutput, match="exact cost telemetry"):
        central.plan(goal)

    [audit] = central.planner_attempts(goal.goal_id)
    assert audit.status == "rejected"
    assert audit.telemetry.source == "legacy_string_transport"
    assert not audit.telemetry.cost_known
    assert central.planner_cost(goal.goal_id, currency="billed") is None


def test_unbudgeted_goal_retains_text_only_transport_compatibility(store: Store) -> None:
    goal = Goal(
        goal_id="unbudgeted-text-transport",
        task="Build and validate a parser",
        success_criteria=("parser.py exists",),
    )

    class TextOnlyTransport:
        def complete(self, *, request_id: str, system: str, prompt: str) -> str:
            return one_assignment_response(request_id, system, prompt)

    central = CentralPlanner(store, transport=TextOnlyTransport())
    plan = central.plan(goal)

    assert plan.generation == 1
    [audit] = central.planner_attempts(goal.goal_id)
    assert audit.telemetry.source == "legacy_string_zero"
    assert central.planner_cost(goal.goal_id, currency="billed") == 0.0


def test_rich_completion_telemetry_is_durable_auditable_and_replayed(
    store: Store, goal: Goal
) -> None:
    telemetry = exact_telemetry()
    transport = FakeTransport(
        lambda request_id, system, prompt: PlannerCompletion(
            one_assignment_response(request_id, system, prompt),
            telemetry,
        )
    )
    central = CentralPlanner(store, transport=transport)

    plan = central.plan(goal)

    [audit] = central.planner_attempts(goal.goal_id)
    assert audit.status == "accepted"
    assert audit.telemetry == telemetry
    assert central.planner_cost(goal.goal_id, currency="work") == pytest.approx(telemetry.work_usd)
    assert central.planner_cost(goal.goal_id, currency="billed") == pytest.approx(
        telemetry.billed_usd
    )
    outcome_path = next(path for path in planner_files(store) if path.endswith("/outcome.json"))
    assert store.view("central-control").head.record(outcome_path)["telemetry"] == (
        telemetry.to_dict()
    )

    replacement = FakeTransport(
        lambda *_args: (_ for _ in ()).throw(AssertionError("transport replayed"))
    )
    recovered = CentralPlanner(store, transport=replacement, control=central.control)
    assert recovered.plan(goal) == plan
    assert replacement.calls == []


@pytest.mark.parametrize("record_name", ["request.json", "intent.json", "outcome.json"])
@pytest.mark.parametrize("damage", ["delete", "rewrite"])
def test_planner_cost_fails_closed_if_historical_attempt_evidence_is_damaged(
    store: Store,
    goal: Goal,
    record_name: str,
    damage: str,
) -> None:
    telemetry = exact_telemetry()
    transport = FakeTransport(
        lambda request_id, system, prompt: PlannerCompletion(
            one_assignment_response(request_id, system, prompt),
            telemetry,
        )
    )
    central = CentralPlanner(store, transport=transport)
    central.plan(goal)
    assert central.planner_cost(goal.goal_id) == pytest.approx(telemetry.work_usd)

    target = next(path for path in planner_files(store) if path.endswith(f"/{record_name}"))
    if damage == "delete":
        central.control.path(target).unlink()
    else:
        central.control.write(target, '{"tampered":true}\n')
    central.control.checkpoint(f"fault injection: {damage} planner {record_name}")

    with pytest.raises(PlannerStateError, match=f"{damage}d|rewrote"):
        central.planner_cost(goal.goal_id)


def test_planner_attempt_history_cache_skips_unchanged_head_and_rejects_rewind(
    store: Store,
    goal: Goal,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    central, _transport = planner(store, one_assignment_response)
    central.plan(goal)
    expected = central.planner_attempts(goal.goal_id)
    audited_head = central.control.head
    assert audited_head.meta.parents

    original_rev_list = store.backend.rev_list_first_parent
    rev_list_calls = 0

    def counted_rev_list(commit: str, limit: int | None = None) -> list[str]:
        nonlocal rev_list_calls
        rev_list_calls += 1
        return original_rev_list(commit, limit)

    monkeypatch.setattr(store.backend, "rev_list_first_parent", counted_rev_list)
    assert central.planner_attempts(goal.goal_id) == expected
    assert rev_list_calls == 0

    parent_id = audited_head.meta.parents[0]
    assert store.backend.cas_update_ref(central.control.ref, parent_id, audited_head.id)
    central.control.backend.reset_hard_to_head()
    with pytest.raises(PlannerStateError, match="head rewound or diverged"):
        central.planner_attempts(goal.goal_id)


def test_production_receipt_pair_cache_tracks_both_heads_without_rescanning(
    store: Store,
    goal: Goal,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoCallLLM:
        max_attempts = 1

        def ensure_ready(self, *models: str) -> None:
            raise AssertionError(models)

        def call(self, **kwargs: Any) -> Any:
            raise AssertionError(kwargs)

    control = store.branch("central-control", producer="central-planner-test")
    journal = store.branch(PLANNER_RECEIPT_BRANCH, producer="planner-receipt-test")
    lock = threading.RLock()
    transport = LLMPlannerTransport(
        NoCallLLM(),
        store=store,
        control=control,
        journal=journal,
        mutation_lock=lock,
    )
    central = CentralPlanner(
        store,
        transport=transport,
        control=control,
        mutation_lock=lock,
    )
    assert central.planner_attempts(goal.goal_id) == ()

    with monkeypatch.context() as exact_hit:
        exact_hit.setattr(
            store.backend,
            "ls_files",
            lambda _commit: (_ for _ in ()).throw(
                AssertionError("exact paired-receipt cache hit reconstructed fingerprints")
            ),
        )
        assert central._paired_transport_evidences() == ()

    def unexpected_rescan() -> tuple[Any, ...]:
        raise AssertionError("paired receipt histories were rescanned")

    monkeypatch.setattr(transport, "_paired_evidences", unexpected_rescan)
    assert central.planner_attempts(goal.goal_id) == ()

    control.checkpoint("unrelated control advance", records={"audit/control.txt": "ok"})
    journal.checkpoint("unrelated journal advance", records={"audit/journal.txt": "ok"})
    assert central.planner_attempts(goal.goal_id) == ()

    audited_journal = journal.head
    assert audited_journal.meta.parents
    assert store.backend.cas_update_ref(
        journal.ref,
        audited_journal.meta.parents[0],
        audited_journal.id,
    )
    journal.backend.reset_hard_to_head()
    with pytest.raises(PlannerStateError, match="head rewound or diverged"):
        central.planner_attempts(goal.goal_id)


def test_charged_invalid_proposal_retains_exact_telemetry(store: Store, goal: Goal) -> None:
    telemetry = exact_telemetry()
    transport = FakeTransport(
        lambda _request_id, _system, _prompt: PlannerCompletion("not JSON", telemetry)
    )
    central = CentralPlanner(store, transport=transport)

    with pytest.raises(InvalidPlannerOutput):
        central.plan(goal)

    [audit] = central.planner_attempts(goal.goal_id)
    assert audit.status == "rejected"
    assert audit.telemetry == telemetry
    assert central.planner_cost(goal.goal_id) == pytest.approx(telemetry.work_usd)


def test_charged_transport_validation_error_retains_response_and_telemetry(
    store: Store, goal: Goal
) -> None:
    telemetry = exact_telemetry()

    def invalid_completion(_request_id: str, _system: str, _prompt: str) -> Any:
        raise PlannerCompletionError(
            "model stopped at max_tokens",
            telemetry=telemetry,
            response='{"partial":true}',
        )

    central, _ = planner(store, invalid_completion)
    with pytest.raises(PlannerTransportError, match="max_tokens"):
        central.plan(goal)

    [audit] = central.planner_attempts(goal.goal_id)
    assert audit.status == "rejected"
    assert audit.category == "transport_terminal"
    assert audit.telemetry == telemetry
    assert audit.response_digest is not None
    assert central.planner_cost(goal.goal_id) == pytest.approx(telemetry.work_usd)
    with pytest.raises(RejectedPlannerOperation, match="transport_terminal"):
        central.plan(goal)


def test_goal_identity_cannot_be_silently_overwritten(store: Store, goal: Goal) -> None:
    central, _ = planner(store)
    central.plan(goal)
    changed = Goal(
        goal_id=goal.goal_id,
        task="A different objective",
        success_criteria=goal.success_criteria,
        budget_usd=goal.budget_usd,
        metadata=goal.metadata,
    )
    with pytest.raises(PlannerIdentityConflict, match="different objective bytes"):
        central.plan(changed)


def test_historical_operation_identity_cannot_be_reused_after_control_rollback(
    store: Store, goal: Goal
) -> None:
    central, transport = planner(store)
    central.plan(goal)
    goal_state = next(
        state
        for state in central.control.history()
        if state.meta.reason == f"planner goal: {goal.goal_id}"
    )
    central.control.rollback(goal_state, "simulate central rollback after accepted plan")

    with pytest.raises(PlannerStateError, match="current plan history"):
        central.plan(goal)
    assert len(transport.calls) == 1


def test_transport_failure_is_durable_and_same_operation_retries(store: Store, goal: Goal) -> None:
    attempts = 0

    def flaky(_call_id: str, _system: str, prompt: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary model outage")
        return one_assignment_response(_call_id, _system, prompt)

    central, transport = planner(store, flaky)
    with pytest.raises(PlannerTransportError, match="temporary model outage"):
        central.plan(goal)

    plan = central.plan(goal)
    assert plan.generation == 1
    assert len(transport.calls) == 2
    assert transport.calls[0][0] != transport.calls[1][0]
    outcomes = [path for path in planner_files(store) if path.endswith("/outcome.json")]
    assert len(outcomes) == 2
    first = store.view("central-control").head.record(sorted(outcomes)[0])
    assert first["status"] == "transport_error"
    assert first["detail"] == "temporary model outage"
    assert first["telemetry"]["cost_known"] is False
    attempts_audit = central.planner_attempts(goal.goal_id)
    assert [item.telemetry.cost_known for item in attempts_audit] == [False, True]
    assert central.planner_cost(goal.goal_id) is None


def test_crash_after_call_intent_reuses_exact_attempt(store: Store, goal: Goal) -> None:
    class CrashTransport:
        def __init__(self) -> None:
            self.call_id = ""

        def complete(self, *, request_id: str, system: str, prompt: str) -> str:
            self.call_id = request_id
            raise KeyboardInterrupt

    crashed = CrashTransport()
    central = CentralPlanner(store, transport=crashed)
    with pytest.raises(KeyboardInterrupt):
        central.plan(goal)
    [pending] = central.planner_attempts(goal.goal_id)
    assert pending.status == "pending"
    assert pending.telemetry.cost_known is False
    assert central.planner_cost(goal.goal_id) is None

    replacement = FakeTransport(one_assignment_response)
    recovered = CentralPlanner(store, transport=replacement, control=central.control)
    plan = recovered.plan(goal)
    assert plan.generation == 1
    assert replacement.calls[0][0] == crashed.call_id
    intents = [path for path in planner_files(store) if path.endswith("/intent.json")]
    assert len(intents) == 1


def test_ambiguous_production_receipt_terminally_blocks_automatic_respend(
    store: Store, goal: Goal
) -> None:
    class KilledLLM:
        def __init__(self) -> None:
            self.calls = 0
            self.ready = 0

        def ensure_ready(self, *models: str) -> None:
            self.ready += 1

        def call(self, **kwargs: Any) -> Any:
            self.calls += 1
            raise KeyboardInterrupt

    killed = KilledLLM()
    lock = threading.RLock()
    control = store.branch("central-control", producer="central-planner-test")
    transport = LLMPlannerTransport(
        killed,
        store=store,
        control=control,
        journal=store.branch(PLANNER_RECEIPT_BRANCH, producer="planner-receipt-test"),
        mutation_lock=lock,
    )
    central = CentralPlanner(
        store,
        transport=transport,
        control=control,
        mutation_lock=lock,
    )
    with pytest.raises(KeyboardInterrupt):
        central.plan(goal)
    assert killed.calls == 1

    replacement = KilledLLM()
    central.transport = LLMPlannerTransport(
        replacement,
        store=store,
        control=control,
        journal=transport.journal,
        mutation_lock=lock,
    )
    with pytest.raises(PlannerTransportError, match="surviving pending"):
        central.plan(goal)
    assert replacement.calls == 0
    assert replacement.ready == 0
    [audit] = central.planner_attempts(goal.goal_id)
    assert audit.status == "rejected"
    assert audit.category == "transport_terminal"
    assert audit.telemetry.cost_known is False
    assert central.planner_cost(goal.goal_id) is None

    with pytest.raises(RejectedPlannerOperation, match="transport_terminal"):
        central.plan(goal)
    assert replacement.calls == 0


def test_crash_after_transport_receipt_is_known_cost_and_replays_without_call(
    store: Store,
    goal: Goal,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ProposalLLM:
        max_attempts = 1

        def __init__(self) -> None:
            self.calls = 0
            self.ready = 0

        def ensure_ready(self, *models: str) -> None:
            self.ready += 1

        def call(self, **kwargs: Any) -> Any:
            self.calls += 1
            response = one_assignment_response(
                "ignored", PLANNER_SYSTEM, kwargs["messages"][0]["content"]
            )
            return SimpleNamespace(
                model=MODEL_PLANNER,
                provider="anthropic",
                usage=Usage(input_tokens=120, output_tokens=30),
                text_blocks=(response,),
                tool_calls=(),
                stop_reason="end_turn",
            )

    from taste.brains.central_planner import PLANNER_SYSTEM

    llm = ProposalLLM()
    lock = threading.RLock()
    control = store.branch("central-control", producer="central-planner-test")
    transport = LLMPlannerTransport(
        llm,
        store=store,
        control=control,
        journal=store.branch(PLANNER_RECEIPT_BRANCH, producer="planner-receipt-test"),
        mutation_lock=lock,
    )
    original_commit = transport._commit_record

    def crash_after_completed_receipt(branch, path: str, raw: dict[str, Any], reason: str) -> str:
        recorded = original_commit(branch, path, raw, reason)
        if branch is transport.journal and raw.get("status") == "completed":
            raise KeyboardInterrupt
        return recorded

    monkeypatch.setattr(transport, "_commit_record", crash_after_completed_receipt)
    central = CentralPlanner(
        store,
        transport=transport,
        control=control,
        mutation_lock=lock,
    )
    with pytest.raises(KeyboardInterrupt):
        central.plan(goal)

    [audit] = central.planner_attempts(goal.goal_id)
    assert audit.status == "pending"
    assert audit.category == "transport_completed"
    assert audit.telemetry.cost_known
    assert central.planner_cost(goal.goal_id, currency="billed") == pytest.approx(
        audit.telemetry.billed_usd
    )

    replacement = ProposalLLM()
    central.transport = LLMPlannerTransport(
        replacement,
        store=store,
        control=control,
        journal=transport.journal,
        mutation_lock=lock,
    )
    recovered = central.plan(goal)
    assert recovered.generation == 1
    assert replacement.calls == 0
    assert replacement.ready == 0
    [finished] = central.planner_attempts(goal.goal_id)
    assert finished.status == "accepted"
    assert finished.telemetry == audit.telemetry


def test_raw_control_ref_rewind_cannot_erase_cost_or_repeat_provider_call(
    tmp_path: Path,
    goal: Goal,
) -> None:
    class ProposalLLM:
        max_attempts = 1

        def __init__(self) -> None:
            self.calls = 0

        def ensure_ready(self, *models: str) -> None:
            pass

        def call(self, **kwargs: Any) -> Any:
            self.calls += 1
            response = one_assignment_response(
                "ignored",
                PLANNER_SYSTEM,
                kwargs["messages"][0]["content"],
            )
            return SimpleNamespace(
                model=MODEL_PLANNER,
                provider="anthropic",
                usage=Usage(input_tokens=120, output_tokens=30),
                text_blocks=(response,),
                tool_calls=(),
                stop_reason="end_turn",
            )

    from taste.brains.central_planner import PLANNER_SYSTEM

    root = tmp_path / "repo"
    first_store = Store.open(root, "planner-control-ref-rewind")
    first_store.branch("integration", producer="central-integration").close()
    control = first_store.branch("central-control", producer="central-planner-test")
    journal = first_store.branch(PLANNER_RECEIPT_BRANCH, producer="planner-receipt-test")
    first_llm = ProposalLLM()
    lock = threading.RLock()
    first = CentralPlanner(
        first_store,
        transport=LLMPlannerTransport(
            first_llm,
            store=first_store,
            control=control,
            journal=journal,
            mutation_lock=lock,
        ),
        control=control,
        mutation_lock=lock,
    )
    first.plan(goal)
    assert first_llm.calls == 1
    request_path = next(
        path
        for path in control.head.files()
        if path.startswith(f"{PLANNER_ROOT}/operations/") and path.endswith("/request.json")
    )
    request = PlanningRequest.from_json(control.read(request_path) or "")
    attempt_state = next(
        state
        for state in control.history()
        if state.meta.reason == f"planner attempt: {request.request_id}"
    )
    charged_control = control.head
    assert first_store.backend.cas_update_ref(
        control.ref,
        attempt_state.id,
        charged_control.id,
    )
    control.backend.reset_hard_to_head()
    first_store.close()

    second_store = Store.open(root, "planner-control-ref-rewind")
    second_control = second_store.branch("central-control", producer="central-planner-test")
    second_journal = second_store.branch(
        PLANNER_RECEIPT_BRANCH,
        producer="planner-receipt-test",
    )
    replacement_llm = ProposalLLM()
    replacement_lock = threading.RLock()
    recovered = CentralPlanner(
        second_store,
        transport=LLMPlannerTransport(
            replacement_llm,
            store=second_store,
            control=second_control,
            journal=second_journal,
            mutation_lock=replacement_lock,
        ),
        control=second_control,
        mutation_lock=replacement_lock,
    )
    try:
        [pending] = recovered.planner_attempts(goal.goal_id)
        assert pending.status == "pending"
        assert pending.category == "transport_completed"
        assert pending.telemetry.cost_known
        billed = recovered.planner_cost(goal.goal_id, currency="billed")
        assert billed == pytest.approx(pending.telemetry.billed_usd)

        replayed = recovered.plan(goal)
        assert replayed.generation == 1
        assert replacement_llm.calls == 0
        [accepted] = recovered.planner_attempts(goal.goal_id)
        assert accepted.status == "accepted"
        assert recovered.planner_cost(goal.goal_id, currency="billed") == pytest.approx(billed)
    finally:
        second_store.close()


def test_raw_control_ref_rewind_past_request_blocks_operation_recreation(
    store: Store,
    goal: Goal,
) -> None:
    class ProposalLLM:
        max_attempts = 1

        def __init__(self) -> None:
            self.calls = 0

        def ensure_ready(self, *models: str) -> None:
            pass

        def call(self, **kwargs: Any) -> Any:
            self.calls += 1
            response = one_assignment_response(
                "ignored",
                PLANNER_SYSTEM,
                kwargs["messages"][0]["content"],
            )
            return SimpleNamespace(
                model=MODEL_PLANNER,
                provider="anthropic",
                usage=Usage(input_tokens=120, output_tokens=30),
                text_blocks=(response,),
                tool_calls=(),
                stop_reason="end_turn",
            )

    from taste.brains.central_planner import PLANNER_SYSTEM

    lock = threading.RLock()
    control = store.branch("central-control", producer="central-planner-test")
    journal = store.branch(PLANNER_RECEIPT_BRANCH, producer="planner-receipt-test")
    before_request = control.head
    first_llm = ProposalLLM()
    first = CentralPlanner(
        store,
        transport=LLMPlannerTransport(
            first_llm,
            store=store,
            control=control,
            journal=journal,
            mutation_lock=lock,
        ),
        control=control,
        mutation_lock=lock,
    )
    first.plan(goal)
    assert first_llm.calls == 1
    charged_control = control.head
    assert store.backend.cas_update_ref(control.ref, before_request.id, charged_control.id)
    control.backend.reset_hard_to_head()

    replacement_llm = ProposalLLM()
    recovered = CentralPlanner(
        store,
        transport=LLMPlannerTransport(
            replacement_llm,
            store=store,
            control=control,
            journal=journal,
            mutation_lock=lock,
        ),
        control=control,
        mutation_lock=lock,
    )
    [orphaned] = recovered.planner_attempts(goal.goal_id)
    assert orphaned.status == "orphaned"
    assert orphaned.category == "orphaned_transport_completed"
    assert orphaned.telemetry.cost_known
    assert recovered.planner_cost(goal.goal_id, currency="billed") == pytest.approx(
        orphaned.telemetry.billed_usd
    )

    with pytest.raises(PlannerStateError, match="exact control request is absent"):
        recovered.plan(goal)
    assert replacement_llm.calls == 0


def test_crash_recovery_reuses_attempt_after_reopening_central_lease(
    tmp_path: Path, goal: Goal
) -> None:
    class CrashTransport:
        def __init__(self) -> None:
            self.call_id = ""

        def complete(self, *, request_id: str, system: str, prompt: str) -> str:
            self.call_id = request_id
            raise KeyboardInterrupt

    root = tmp_path / "repo"
    first_store = Store.open(root, "planner-restart")
    first_store.branch("integration", producer="central-integration").close()
    crashed = CrashTransport()
    first = CentralPlanner(first_store, transport=crashed)
    with pytest.raises(KeyboardInterrupt):
        first.plan(goal)
    first_store.close()

    second_store = Store.open(root, "planner-restart")
    replacement = FakeTransport(one_assignment_response)
    recovered = CentralPlanner(second_store, transport=replacement)
    plan = recovered.plan(goal)

    assert plan.generation == 1
    assert replacement.calls[0][0] == crashed.call_id
    intents = [path for path in planner_files(second_store) if path.endswith("/intent.json")]
    assert len(intents) == 1
    second_store.close()


@pytest.mark.parametrize(
    "response, match",
    [
        (lambda prompt: "```json\n{}\n```", "not valid JSON"),
        (
            lambda prompt: proposal(
                prompt,
                assignment_for(request_from_prompt(prompt)),
                changes={"unknown": True},
            ),
            "unknown",
        ),
        (
            lambda prompt: proposal(
                prompt,
                assignment_for(request_from_prompt(prompt)),
                changes={"request_id": "sha256:" + "f" * 64},
            ),
            "does not echo",
        ),
    ],
)
def test_invalid_output_is_preserved_and_operation_is_terminally_rejected(
    store: Store, goal: Goal, response, match: str
) -> None:
    transport = FakeTransport(lambda _id, _system, prompt: response(prompt))
    central = CentralPlanner(store, transport=transport)
    with pytest.raises(InvalidPlannerOutput, match=match):
        central.plan(goal)
    assert central.current_plan(goal.goal_id) is None
    paths = planner_files(store)
    result_path = next(path for path in paths if path.endswith("/result.json"))
    outcome_path = next(path for path in paths if path.endswith("/outcome.json"))
    assert store.view("central-control").head.record(result_path)["status"] == "rejected"
    assert store.view("central-control").head.record(outcome_path)["response"] == response(
        transport.calls[0][2]
    )

    with pytest.raises(RejectedPlannerOperation, match="invalid_output"):
        central.plan(goal)
    assert len(transport.calls) == 1


def test_duplicate_json_keys_are_rejected(store: Store, goal: Goal) -> None:
    def duplicate(_id: str, _system: str, prompt: str) -> str:
        valid = proposal(prompt, assignment_for(request_from_prompt(prompt)))
        return valid[:-1] + ',"rationale":"shadow"}'

    central, _ = planner(store, duplicate)
    with pytest.raises(InvalidPlannerOutput, match="duplicate key"):
        central.plan(goal)


def test_world_movement_during_call_rejects_and_retains_response(store: Store, goal: Goal) -> None:
    captured = ""

    def move_world(_id: str, _system: str, prompt: str) -> str:
        nonlocal captured
        request = request_from_prompt(prompt)
        captured = proposal(prompt, assignment_for(request))
        branch = store.branch("late-worker", from_state=store.view("integration").head)
        branch.write("late.txt", "arrived during planning\n")
        branch.checkpoint("late durable work")
        branch.close()
        return captured

    central, _ = planner(store, move_world)
    with pytest.raises(StalePlanningWorld, match="during planning call"):
        central.plan(goal)
    assert central.current_plan(goal.goal_id) is None
    outcome_path = next(path for path in planner_files(store) if path.endswith("/outcome.json"))
    outcome = store.view("central-control").head.record(outcome_path)
    assert outcome["status"] == "rejected"
    assert outcome["category"] == "stale_world"
    assert outcome["response"] == captured


def test_verdict_note_movement_without_head_change_is_fenced(store: Store, goal: Goal) -> None:
    observed = store.branch("observed-worker", from_state=store.view("integration").head)
    state = observed.checkpoint("state awaiting monitor verdict")
    observed.close()

    def add_verdict(_id: str, _system: str, prompt: str) -> str:
        request = request_from_prompt(prompt)
        response = proposal(prompt, assignment_for(request))
        store.judge(
            state,
            Verdict(status="fail", by="monitor", detail="new exact-state failure"),
        )
        return response

    central, _ = planner(store, add_verdict)
    with pytest.raises(StalePlanningWorld, match="observation"):
        central.plan(goal)
    assert store.view("observed-worker").head.id == state.id


def test_global_picture_contains_live_dirty_inbox_manifest_and_provenance(
    store: Store, goal: Goal
) -> None:
    producer = store.branch("producer", from_state=store.view("integration").head)
    producer.write("published.txt", "published bytes\n")
    producer.publish("published", "published.txt", description="reusable partial work")
    published = producer.checkpoint("publish partial work")
    producer.write("scratch.txt", "uncheckpointed but visible\n")
    producer.intend("extend the partial result")
    store.send("producer", {"kind": "artifact_request", "artifact_id": "published"})

    seen: dict[str, Any] = {}

    def inspect(_id: str, _system: str, prompt: str) -> str:
        request = request_from_prompt(prompt)
        observation = next(
            item for item in request.world.to_dict()["observations"] if item["branch"] == "producer"
        )
        seen.update(observation)
        return proposal(prompt, assignment_for(request))

    central, _ = planner(store, inspect)
    plan = central.plan(goal)
    assert plan.generation == 1
    assert seen["head_state_id"] == published.id
    assert seen["live"] is True
    assert seen["dirty_paths"] == ["scratch.txt"]
    assert "scratch.txt" in seen["pending_diff"]
    assert seen["intent"] == "extend the partial result"
    assert seen["inbox"][0]["body"]["artifact_id"] == "published"
    assert seen["manifest"]["entries"]["published"]["blob"] == published.blob("published.txt")
    assert seen["origins"][0]["origin_state_id"] == published.id
    assert seen["observation_digest"].startswith("sha256:")
    producer.close()


def test_foreign_control_event_after_transport_failure_fences_retry(
    store: Store, goal: Goal
) -> None:
    central, transport = planner(
        store,
        lambda _id, _system, _prompt: (_ for _ in ()).throw(OSError("offline")),
    )
    with pytest.raises(PlannerTransportError):
        central.plan(goal)

    central.control.checkpoint(
        "supervisor external event",
        records={".taste/supervisor/external.json": {"event": "spawned"}},
    )
    transport.responder = one_assignment_response
    with pytest.raises(StalePlanningWorld, match="foreign-state"):
        central.plan(goal)
    assert len(transport.calls) == 1


def test_output_paths_and_artifact_ids_have_one_owner(store: Store, goal: Goal) -> None:
    def conflicting(_id: str, _system: str, prompt: str) -> str:
        request = request_from_prompt(prompt)
        first = assignment_for(request, assignment_id="one", worker="worker-one")
        second = assignment_for(
            request,
            assignment_id="two",
            worker="worker-two",
            output_id="other-id",
            output_path="parser.py",
        )
        return proposal(prompt, first, second)

    central, _ = planner(store, conflicting)
    with pytest.raises(InvalidPlannerOutput, match="multiple owners"):
        central.plan(goal)


def test_artifact_id_cannot_name_two_different_outputs(store: Store, goal: Goal) -> None:
    def conflicting(_id: str, _system: str, prompt: str) -> str:
        request = request_from_prompt(prompt)
        first = assignment_for(request, assignment_id="one", worker="worker-one")
        second = assignment_for(
            request,
            assignment_id="two",
            worker="worker-two",
            output_id="parser-source",
            output_path="parser_test.py",
        )
        return proposal(prompt, first, second)

    central, _ = planner(store, conflicting)
    with pytest.raises(InvalidPlannerOutput, match="multiple owners"):
        central.plan(goal)


def test_planner_rejects_reusing_an_observed_worker_execution_branch(
    store: Store, goal: Goal
) -> None:
    used = store.branch("worker-used", from_state=store.view("integration").head)
    used.write("old.txt", "prior generation bytes\n")
    used.checkpoint("prior worker attempt")
    used.close()

    def reuse(_id: str, _system: str, prompt: str) -> str:
        request = request_from_prompt(prompt)
        assert json.loads(prompt)["rules"]["fresh_worker_execution_branch"] is True
        return proposal(prompt, assignment_for(request, worker="worker-used"))

    central, _ = planner(store, reuse)
    with pytest.raises(InvalidPlannerOutput, match="fresh branch identity"):
        central.plan(goal)


def test_input_must_bind_bytes_reachable_in_observed_world(store: Store, goal: Goal) -> None:
    source = store.branch("source", from_state=store.view("integration").head)
    source.write("grammar.txt", "expr := atom\n")
    state = source.checkpoint("publish grammar")
    source.close()

    def with_input(_id: str, _system: str, prompt: str) -> str:
        request = request_from_prompt(prompt)
        artifact = ArtifactRef(
            artifact_id="grammar",
            branch="source",
            state_id=state.id,
            path="grammar.txt",
            blob_id=state.blob("grammar.txt") or "",
        )
        return proposal(prompt, assignment_for(request, inputs=(artifact,)))

    central, _ = planner(store, with_input)
    plan = central.plan(goal)
    assert plan.assignments[0].inputs[0].blob_id == state.blob("grammar.txt")


def test_input_from_state_created_after_snapshot_is_rejected(store: Store, goal: Goal) -> None:
    source = store.branch("source", from_state=store.view("integration").head)
    source.write("grammar.txt", "v1\n")
    first = source.checkpoint("grammar v1")
    source.close()

    def future_input(_id: str, _system: str, prompt: str) -> str:
        request = request_from_prompt(prompt)
        writer = store.branch("source")
        writer.write("grammar.txt", "v2\n")
        future = writer.checkpoint("grammar v2 during planning")
        writer.close()
        artifact = ArtifactRef(
            artifact_id="grammar",
            branch="source",
            state_id=future.id,
            path="grammar.txt",
            blob_id=future.blob("grammar.txt") or "",
        )
        return proposal(prompt, assignment_for(request, inputs=(artifact,)))

    assert first.id != store.view("integration").head.id
    central, _ = planner(store, future_input)
    with pytest.raises(StalePlanningWorld):
        central.plan(goal)


def test_contract_io_must_match_structured_artifacts(store: Store, goal: Goal) -> None:
    def mismatch(_id: str, _system: str, prompt: str) -> str:
        request = request_from_prompt(prompt)
        item = assignment_for(request)
        wrong = Contract(
            identity=item.worker,
            task=item.contract.task,
            outputs=("different.py",),
            success_criteria=item.contract.success_criteria,
            budget_usd=item.contract.budget_usd,
        )
        item = Assignment(
            assignment_id=item.assignment_id,
            generation=item.generation,
            attempt=item.attempt,
            contract=wrong,
            contract_digest=contract_digest(wrong),
            base_state_id=item.base_state_id,
            outputs=item.outputs,
            resources=item.resources,
        )
        return proposal(prompt, item)

    central, _ = planner(store, mismatch)
    with pytest.raises(InvalidPlannerOutput, match="contract outputs are not exact"):
        central.plan(goal)


def test_replan_receives_exact_durable_worker_failure(store: Store, goal: Goal) -> None:
    shared_lock = threading.RLock()
    supervisor = CentralSupervisor(store, launcher=UnusedLauncher())
    supervisor.integration.write("shared.txt", "base\n")
    base = supervisor.integration.checkpoint("shared conflict base")
    left = store.branch("conflict-left", from_state=base)
    left.write("shared.txt", "left\n")
    left.checkpoint("left version")
    left.close()
    right = store.branch("conflict-right", from_state=base)
    right.write("shared.txt", "right\n")
    right.checkpoint("right version")
    right.close()
    first_transport = FakeTransport(one_assignment_response)
    central = CentralPlanner(
        store,
        transport=first_transport,
        control=supervisor.control,
        mutation_lock=shared_lock,
    )
    first = central.plan(goal)
    run = supervisor.prepare(first.assignments[0], wall_timeout_seconds=30)
    stopped = supervisor.stop(run.run_id, reason="worker crashed during parsing")
    assert stopped.terminal_reason == "worker_crashed_during_parsing"
    worker = store.branch(first.assignments[0].worker)
    worker.write("parser.py", "# partial parser preserved after SDK failure\n")
    failed_state = worker.checkpoint("durable failed worker state")
    partial_output = ArtifactRef(
        artifact_id="parser-source",
        branch=first.assignments[0].worker,
        state_id=failed_state.id,
        path="parser.py",
        blob_id=failed_state.blob("parser.py") or "",
    )
    report = WorkerReport(
        report_id="failed-parser-report",
        run_id=run.run_id,
        assignment_id=first.assignments[0].assignment_id,
        worker=first.assignments[0].worker,
        generation=first.generation,
        attempt=first.assignments[0].attempt,
        contract_digest=first.assignments[0].contract_digest,
        base_state_id=first.assignments[0].base_state_id,
        final_state_id=failed_state.id,
        at=failed_state.meta.created_at,
        completed=False,
        terminal_reason="sdk_error",
        outputs=(partial_output,),
        summary="worker stream ended before producing parser.py",
    )
    worker.write(WORKER_REPORT_PATH, report.to_json())
    worker.checkpoint("durable failed worker report")
    worker.close()
    assert supervisor.collect(run.run_id, active_generation=1) == report

    assert supervisor.integration.merge(store.view("conflict-left"), reason="take left").ok
    conflict = supervisor.integration.merge(store.view("conflict-right"), reason="take right")
    assert not conflict.ok
    assert [item.path for item in conflict.conflicts] == ["shared.txt"]

    seen: dict[str, Any] = {}

    def recovery(_id: str, _system: str, prompt: str) -> str:
        request = request_from_prompt(prompt)
        seen["outcomes"] = request.world.to_dict()["outcomes"]
        seen["observations"] = request.world.to_dict()["observations"]
        retry = assignment_for(
            request,
            assignment_id="rebuild-parser",
            worker="worker-parser-retry",
            attempt=1,
        )
        return proposal(prompt, retry)

    recovery_transport = FakeTransport(recovery)
    central.transport = recovery_transport
    revised = central.revise(goal, operation_id="recover-crashed-parser")
    assert revised.generation == 2
    assert revised.parent_plan_id == first.plan_id
    assert revised.assignments[0].attempt == 1
    assert len(seen["outcomes"]) == 1
    observed = seen["outcomes"][0]["run"]
    assert observed["run_id"] == run.run_id
    assert observed["phase"] == "report_accepted"
    assert observed["terminal_reason"] == "worker_crashed_during_parsing"
    observed_report = seen["outcomes"][0]["report"]
    assert observed_report["completed"] is False
    assert observed_report["terminal_reason"] == "sdk_error"
    integration_observation = next(
        item for item in seen["observations"] if item["branch"] == "integration"
    )
    assert [item["path"] for item in integration_observation["conflicts"]] == ["shared.txt"]
    assert "conflicts=1" in integration_observation["abstract"]
    supervisor.close()


def test_complete_plan_has_no_assignments(store: Store, goal: Goal) -> None:
    transport = FakeTransport(
        lambda _id, _system, prompt: proposal(
            prompt,
            complete=True,
            completion_reason="all exact success criteria are already met",
        )
    )
    central = CentralPlanner(store, transport=transport)
    plan = central.plan(goal)
    assert plan.complete
    assert plan.assignments == ()


def test_non_complete_empty_plan_is_rejected(store: Store, goal: Goal) -> None:
    central, _ = planner(store, lambda _id, _system, prompt: proposal(prompt))
    with pytest.raises(InvalidPlannerOutput, match="active plans must have assignments"):
        central.plan(goal)


@pytest.mark.parametrize(
    "invalid_kind",
    ["missing", "zero", "boolean", "nonfinite"],
)
def test_budgeted_goal_rejects_missing_or_invalid_worker_target_cap(
    store: Store, goal: Goal, invalid_kind: str
) -> None:
    def invalid_target_cap(_id: str, _system: str, prompt: str) -> str:
        rules = json.loads(prompt)["rules"]
        assert rules["budgeted_assignment_requires_positive_finite_contract_budget_usd"] is True
        request = request_from_prompt(prompt)
        target_budget = None if invalid_kind == "missing" else 0.0
        item = assignment_for(request, target_budget_usd=target_budget)
        raw = json.loads(proposal(prompt, item))
        contract = raw["assignments"][0]["contract"]
        if invalid_kind == "missing":
            del contract["budget_usd"]
        elif invalid_kind == "boolean":
            contract["budget_usd"] = False
        elif invalid_kind == "nonfinite":
            contract["budget_usd"] = float("inf")
        return json.dumps(raw)

    central, _ = planner(store, invalid_target_cap)
    with pytest.raises(InvalidPlannerOutput):
        central.plan(goal)


@pytest.mark.parametrize(
    "invalid_kind",
    ["missing", "zero", "boolean", "nonfinite"],
)
def test_budgeted_goal_rejects_missing_or_invalid_monitor_cap(
    store: Store, goal: Goal, invalid_kind: str
) -> None:
    def invalid_monitor_cap(_id: str, _system: str, prompt: str) -> str:
        rules = json.loads(prompt)["rules"]
        assert (
            rules["budgeted_assignment_requires_separate_positive_finite_monitor_budget_usd"]
            is True
        )
        request = request_from_prompt(prompt)
        monitor_budget: Any | None
        if invalid_kind == "missing":
            monitor_budget = None
        elif invalid_kind == "zero":
            monitor_budget = 0.0
        elif invalid_kind == "boolean":
            monitor_budget = False
        else:
            monitor_budget = 0.25
        item = assignment_for(request, monitor_budget_usd=monitor_budget)
        raw = json.loads(proposal(prompt, item))
        if invalid_kind == "nonfinite":
            raw["assignments"][0]["resources"]["monitor_budget_usd"] = float("inf")
        return json.dumps(raw)

    central, _ = planner(store, invalid_monitor_cap)
    with pytest.raises(InvalidPlannerOutput):
        central.plan(goal)


@pytest.mark.parametrize("target_budget_usd", [None, 0.0])
def test_unbudgeted_goal_preserves_optional_worker_and_monitor_caps(
    store: Store, goal: Goal, target_budget_usd: float | None
) -> None:
    unbudgeted = Goal(
        goal_id=goal.goal_id,
        task=goal.task,
        success_criteria=goal.success_criteria,
        metadata=goal.metadata,
    )

    def without_caps(_id: str, _system: str, prompt: str) -> str:
        rules = json.loads(prompt)["rules"]
        assert rules["goal_has_global_budget"] is False
        assert rules["budgeted_assignment_requires_positive_finite_contract_budget_usd"] is False
        assert (
            rules["budgeted_assignment_requires_separate_positive_finite_monitor_budget_usd"]
            is False
        )
        request = request_from_prompt(prompt)
        return proposal(
            prompt,
            assignment_for(
                request,
                target_budget_usd=target_budget_usd,
                monitor_budget_usd=None,
            ),
        )

    central, _ = planner(store, without_caps)
    plan = central.plan(unbudgeted)
    assert plan.assignments[0].contract.budget_usd == target_budget_usd
    assert "monitor_budget_usd" not in plan.assignments[0].resources


@pytest.mark.parametrize("invalid_monitor_budget", [0.0, False, float("inf")])
def test_unbudgeted_goal_rejects_invalid_optional_monitor_cap(
    store: Store, goal: Goal, invalid_monitor_budget: Any
) -> None:
    unbudgeted = Goal(
        goal_id=goal.goal_id,
        task=goal.task,
        success_criteria=goal.success_criteria,
        metadata=goal.metadata,
    )

    def invalid_optional_cap(_id: str, _system: str, prompt: str) -> str:
        assert (
            json.loads(prompt)["rules"]["monitor_budget_usd_if_present_must_be_positive_finite"]
            is True
        )
        request = request_from_prompt(prompt)
        item = assignment_for(request, target_budget_usd=None)
        raw = json.loads(proposal(prompt, item))
        raw["assignments"][0]["resources"]["monitor_budget_usd"] = invalid_monitor_budget
        return json.dumps(raw)

    central, _ = planner(store, invalid_optional_cap)
    with pytest.raises(InvalidPlannerOutput):
        central.plan(unbudgeted)


def test_corrupt_current_pointer_fails_closed(store: Store, goal: Goal) -> None:
    central, _ = planner(store)
    central.plan(goal)
    current = next(path for path in planner_files(store) if "/current/" in path)
    central.control.write(current, '{"schema":"wrong"}\n')
    central.control.checkpoint("inject corrupt current pointer")
    with pytest.raises(PlannerStateError, match="current plan pointer"):
        central.current_plan(goal.goal_id)


@pytest.mark.parametrize("damage", ["delete", "rewrite", "rollback"])
def test_current_plan_history_rejects_deletion_rewrite_or_valid_rollback(
    store: Store, goal: Goal, damage: str
) -> None:
    def two_generations(_id: str, _system: str, prompt: str) -> PlannerCompletion:
        request = request_from_prompt(prompt)
        if request.generation == 1:
            response = proposal(prompt, assignment_for(request))
        else:
            response = proposal(
                prompt,
                complete=True,
                completion_reason="the exact goal is complete",
            )
        return PlannerCompletion(response, exact_telemetry())

    central, _ = planner(store, two_generations)
    first = central.plan(goal)
    generation_one_state = central.control.head
    second = central.revise(goal, operation_id="finish-goal")
    assert first.generation == 1
    assert second.generation == 2
    pointer_path = next(path for path in planner_files(store) if "/current/" in path)
    generation_one_pointer = generation_one_state.read(pointer_path)
    assert generation_one_pointer is not None

    if damage == "delete":
        central.control.path(pointer_path).unlink()
        central.control.checkpoint("fault injection: delete current plan pointer")
    elif damage == "rewrite":
        central.control.write(pointer_path, generation_one_pointer)
        central.control.checkpoint("fault injection: rewrite pointer to generation one")
    else:
        central.control.rollback(
            generation_one_state,
            "fault injection: valid rollback to generation one",
        )

    with pytest.raises(PlannerStateError, match="current plan history"):
        central.current_plan(goal.goal_id)


def test_an_omitted_contract_digest_is_derived_but_a_wrong_one_still_fails() -> None:
    """A planner cannot compute a sha256 by hand, so the parser derives it.

    Measured: the first real planner call returned a well-formed proposal and
    was rejected for five missing fields, ``contract_digest`` among them. The
    prompt had asked for "the complete taste.brains/Assignment/1 wire schema"
    and shown an empty list, so the model was guessing a strict schema from its
    name -- and one of the fields it had to guess was a hash.

    Deriving it keeps the integrity property rather than trading it away: a
    digest the planner *supplies* is left exactly as sent, so a wrong one still
    fails. Silently correcting it would turn a tamper signal into a shrug.
    """
    contract = Contract(
        identity="worker-derived",
        task="produce derived.py",
        outputs=("derived.py",),
        success_criteria=("derived.py exists",),
    )
    base = {
        "schema": Assignment.SCHEMA,
        "assignment_id": "derived-assignment",
        "generation": 1,
        "attempt": 0,
        "contract": contract.to_dict(),
        "base_state_id": "0" * 40,
        "depends_on": [],
        "inputs": [],
        "outputs": [
            {
                "schema": "taste.brains/ArtifactSpec/1",
                "artifact_id": "derived",
                "path": "derived.py",
                "kind": "file",
                "description": "",
                "required": True,
                "disposition": "present",
                "metadata": {},
            }
        ],
        "model": "claude-sonnet-5",
        "resources": {},
        "metadata": {},
    }
    exact = contract_digest(contract)
    fill = CentralPlanner._with_derived_digest

    filled = fill(dict(base))
    assert filled["contract_digest"] == exact
    assert Assignment.from_dict(filled).contract_digest == exact

    supplied = fill({**base, "contract_digest": exact})
    assert supplied["contract_digest"] == exact

    forged = "sha256:" + "0" * 64
    kept = fill({**base, "contract_digest": forged})
    assert kept["contract_digest"] == forged, "a supplied digest is never rewritten"
    with pytest.raises(ValueError):
        Assignment.from_dict(kept)


def test_the_prompt_shows_the_assignment_shape_instead_of_naming_it(
    store: Store, goal: Goal
) -> None:
    """Naming a strict schema is not the same as showing it.

    The first real planner call omitted ``attempt``, ``contract_digest``,
    ``depends_on``, ``model`` and ``resources`` -- five of eleven fields --
    because the prompt said "use the complete wire schema" and then showed
    ``"assignments": []``.
    """
    captured: dict[str, Any] = {}

    def capture(_request_id: str, _system: str, prompt: str) -> str:
        captured.update(json.loads(prompt))
        request = request_from_prompt(prompt)
        return proposal(prompt, assignment_for(request))

    central = CentralPlanner(store, transport=FakeTransport(capture))
    central.plan(goal)

    exemplar = captured["required_output_shape"]["assignments"][0]
    required = set(captured["assignment_schema"]["required"])

    assert required <= set(exemplar), "the exemplar must satisfy the schema it ships with"
    for field_name in ("attempt", "depends_on", "model", "resources", "metadata"):
        assert field_name in exemplar, f"{field_name} was guessed wrong for want of an example"
    # The parser derives it, so the model is told not to send one.
    assert "contract_digest" not in required
    # ``worker`` is a property of the contract, not a wire field: naming it as
    # one is how a plan acquires a key that parsing then refuses.
    assert "worker" not in exemplar

# ------------------------------------------------------ append-only criteria
#
# The planner is judged against the criteria it is handed, and it is also the
# party that proposes revisions to them.  That is exactly the shape where a
# stop condition can be talked down: reach ``complete`` by releasing whatever
# was not met.  These tests pin the seam where that becomes impossible --
# criteria are carried *inside* the request identity, so a plan is
# cryptographically bound to the obligations it was judged against.


def test_planning_a_goal_seeds_its_genesis_criteria(store: Store, goal: Goal) -> None:
    planner = CentralPlanner(store, transport=FakeTransport(one_assignment_response))
    planner.plan(goal)
    revision = planner.criteria(goal.goal_id)
    assert revision is not None
    assert revision.sequence == 0
    assert tuple(item.text for item in revision.criteria) == goal.success_criteria
    assert all(item.parent_id is None for item in revision.criteria)


def test_live_criteria_are_inside_the_request_identity(store: Store, goal: Goal) -> None:
    seen: list[PlanningRequest] = []

    def respond(_call_id: str, _system: str, prompt: str) -> str:
        request = request_from_prompt(prompt)
        seen.append(request)
        return proposal(prompt, assignment_for(request))

    planner = CentralPlanner(store, transport=FakeTransport(respond))
    planner.plan(goal)

    request = seen[0]
    assert request.criteria is not None
    assert tuple(item.text for item in request.criteria.criteria) == goal.success_criteria
    # The digest covers them: the same request with different obligations is a
    # different request, so a plan cannot be moved onto a weaker bar.
    other = Goal(
        goal_id=goal.goal_id,
        task=goal.task,
        success_criteria=("parser.py exists",),
        budget_usd=goal.budget_usd,
        metadata=goal.metadata,
    )
    weaker = CriteriaRevision.genesis(other, at=request.created_at)
    moved = PlanningRequest.create(
        operation_id=request.operation_id,
        generation=request.generation,
        created_at=request.created_at,
        goal=request.goal,
        world=request.world,
        parent_plan=request.parent_plan,
        criteria=weaker,
    )
    assert moved.request_id != request.request_id


def test_a_recovered_request_replays_the_same_identity(store: Store, goal: Goal) -> None:
    planner = CentralPlanner(store, transport=FakeTransport(one_assignment_response))
    first = planner.plan(goal)
    reopened = CentralPlanner(store, transport=FakeTransport(one_assignment_response))
    again = reopened.plan(goal)
    assert again.plan_id == first.plan_id


def test_a_request_whose_criteria_were_swapped_fails_its_own_digest(
    store: Store, goal: Goal
) -> None:
    seen: list[PlanningRequest] = []

    def respond(_call_id: str, _system: str, prompt: str) -> str:
        request = request_from_prompt(prompt)
        seen.append(request)
        return proposal(prompt, assignment_for(request))

    planner = CentralPlanner(store, transport=FakeTransport(respond))
    planner.plan(goal)
    raw = seen[0].to_dict()
    weaker = Goal(
        goal_id=goal.goal_id,
        task=goal.task,
        success_criteria=("parser.py exists",),
        budget_usd=goal.budget_usd,
        metadata=goal.metadata,
    )
    raw["criteria"] = CriteriaRevision.genesis(weaker, at=seen[0].created_at).to_dict()
    with pytest.raises(ValueError, match="request_id"):
        PlanningRequest.from_dict(raw)


def test_a_refinement_is_carried_and_keeps_its_parent(store: Store, goal: Goal) -> None:
    planner = CentralPlanner(store, transport=FakeTransport(one_assignment_response))
    planner.plan(goal)
    base = planner.criteria(goal.goal_id)
    assert base is not None
    parent = base.criteria[1]
    child = Criterion.derive(
        goal_id=goal.goal_id,
        text="`pytest -q tests/test_parser.py` exits 0",
        parent_id=parent.criterion_id,
    )
    revised = planner.refine_criteria(
        goal.goal_id, (child,), reason="name the exact command"
    )
    assert revised.sequence == 1
    ids = {item.criterion_id for item in revised.criteria}
    assert parent.criterion_id in ids and child.criterion_id in ids
    assert planner.criteria(goal.goal_id) == revised


def test_criteria_cannot_be_dropped_through_the_planner(store: Store, goal: Goal) -> None:
    planner = CentralPlanner(store, transport=FakeTransport(one_assignment_response))
    planner.plan(goal)
    base = planner.criteria(goal.goal_id)
    assert base is not None
    with pytest.raises(ValueError, match=r"append-only|already standing|parent"):
        planner.refine_criteria(
            goal.goal_id,
            (Criterion.derive(goal_id=goal.goal_id, text=base.criteria[0].text),),
            reason="restate the first obligation as if it were new",
        )


def _live_ids(prompt: str) -> list[str]:
    payload = json.loads(prompt)
    return [item["criterion_id"] for item in payload["standing_criteria"]]


def _assessed(prompt: str, verdict: str = "met", evidence: str = "parser.py is present") -> list[dict[str, Any]]:
    return [
        {"criterion_id": cid, "verdict": verdict, "evidence": evidence}
        for cid in _live_ids(prompt)
    ]


def test_the_prompt_shows_every_standing_criterion_with_its_parentage(
    store: Store, goal: Goal
) -> None:
    prompts: list[str] = []

    def respond(_call_id: str, _system: str, prompt: str) -> str:
        prompts.append(prompt)
        request = request_from_prompt(prompt)
        return proposal(prompt, assignment_for(request), assessment=_assessed(prompt, "not_met", "not started"))

    planner = CentralPlanner(store, transport=FakeTransport(respond))
    planner.plan(goal)

    payload = json.loads(prompts[0])
    standing = payload["standing_criteria"]
    assert [item["text"] for item in standing] == list(goal.success_criteria)
    assert all("criterion_id" in item and "parent_id" in item for item in standing)
    assert payload["rules"]["criteria_are_append_only"] is True


def test_completion_requires_every_standing_criterion_to_be_assessed(
    store: Store, goal: Goal
) -> None:
    def respond(_call_id: str, _system: str, prompt: str) -> str:
        partial = _assessed(prompt)[:1]
        return proposal(
            prompt,
            complete=True,
            completion_reason="the parser is built",
            assessment=partial,
        )

    planner = CentralPlanner(store, transport=FakeTransport(respond))
    with pytest.raises(InvalidPlannerOutput):
        planner.plan(goal)


def test_completion_is_refused_while_any_criterion_is_unmet(store: Store, goal: Goal) -> None:
    def respond(_call_id: str, _system: str, prompt: str) -> str:
        items = _assessed(prompt)
        items[-1] = {**items[-1], "verdict": "not_met", "evidence": "tests still fail"}
        return proposal(
            prompt,
            complete=True,
            completion_reason="calling it done",
            assessment=items,
        )

    planner = CentralPlanner(store, transport=FakeTransport(respond))
    with pytest.raises(InvalidPlannerOutput):
        planner.plan(goal)


def test_an_assessment_of_an_unknown_criterion_is_refused(store: Store, goal: Goal) -> None:
    def respond(_call_id: str, _system: str, prompt: str) -> str:
        items = _assessed(prompt)
        items.append(
            {
                "criterion_id": "sha256:" + "0" * 64,
                "verdict": "met",
                "evidence": "a criterion nobody set",
            }
        )
        return proposal(
            prompt, complete=True, completion_reason="done", assessment=items
        )

    planner = CentralPlanner(store, transport=FakeTransport(respond))
    with pytest.raises(InvalidPlannerOutput):
        planner.plan(goal)


def test_a_fully_assessed_completion_is_accepted(store: Store, goal: Goal) -> None:
    def respond(_call_id: str, _system: str, prompt: str) -> str:
        return proposal(
            prompt,
            complete=True,
            completion_reason="every standing criterion is met",
            assessment=_assessed(prompt),
        )

    planner = CentralPlanner(store, transport=FakeTransport(respond))
    plan = planner.plan(goal)
    assert plan.complete
    assert plan.assessment is not None
    assert {item["criterion_id"] for item in plan.assessment} == set(
        item.criterion_id for item in planner.criteria(goal.goal_id).criteria
    )


def test_an_active_plan_may_leave_criteria_unassessed(store: Store, goal: Goal) -> None:
    planner = CentralPlanner(store, transport=FakeTransport(one_assignment_response))
    plan = planner.plan(goal)
    assert not plan.complete
