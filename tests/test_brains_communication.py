"""Typed cross-brain messages preserve identity, ordering, and failures."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from taste.brains.communication import (
    AcceptanceReceipt,
    Communicator,
    GenerationFence,
    GenerationStatus,
    IdempotencyConflict,
    InboxMessage,
    InboxOrderError,
    InvalidAcceptanceReceipt,
    Message,
    MessageWireError,
    artifact_request_dedup_key,
    group_artifact_requests,
)
from taste.brains.records import ArtifactRef
from taste.memstore import Store
from taste.memstore.backend import EMPTY_TREE


@pytest.fixture
def store(tmp_path: Path) -> Store:
    opened = Store.open(tmp_path / "repo", "communication-test")
    yield opened
    opened.close()


def message(**changes: Any) -> Message:
    values = {
        "idempotency_key": "plan-7.notify-worker-1",
        "kind": "assignment_ready",
        "sender": "central",
        "recipient": "worker-1",
        "generation": 7,
        "request_id": "request-7",
        "payload": {"assignment_id": "assignment-1", "flags": ["exact"]},
    }
    return Message.create(**{**values, **changes})


def receipt(
    item,  # type: ignore[no-untyped-def]
    disposition: str,
    current_generation: int,
    *,
    durable_ref: str = "3" * 40,
) -> AcceptanceReceipt:
    return AcceptanceReceipt.for_message(
        item,
        disposition=disposition,
        current_generation=current_generation,
        durable_ref=durable_ref,
    )


class DurableLedger:
    """A tiny idempotent acceptance boundary backed by a memstore branch."""

    def __init__(self, store: Store) -> None:
        self.branch = store.branch("acceptance-ledger", producer="central")
        self.calls = 0

    def close(self) -> None:
        self.branch.close()

    def __call__(  # type: ignore[no-untyped-def]
        self, item, disposition: str, current_generation: int
    ) -> AcceptanceReceipt:
        self.calls += 1
        path = f"receipts/{item.message.message_id.removeprefix('sha256:')}.json"
        wanted = {
            "inbox_id": item.inbox_id,
            "message_id": item.message.message_id,
            "message_generation": item.message.generation,
            "current_generation": current_generation,
            "disposition": disposition,
        }
        existing = self.branch.read(path)
        if existing is None:
            self.branch.write(path, json.dumps(wanted, sort_keys=True) + "\n")
            state = self.branch.checkpoint(f"accept communication: {item.message.message_id}")
        else:
            assert json.loads(existing) == wanted
            state = self.branch.head
        return receipt(
            item,
            disposition,
            current_generation,
            durable_ref=state.id,
        )


def append_raw_inbox(store: Store, recipient: str, body: str) -> str:
    """Append a deliberately non-Store wire commit for fault injection."""
    ref = f"{store.INBOX_REF}/{store.session}/{recipient}"
    with store.backend.lock():
        head = store.backend.ref_sha(ref)
        inbox_id = store.backend.commit_tree(EMPTY_TREE, [head] if head else [], body)
        assert store.backend.cas_update_ref(ref, inbox_id, head)
    return inbox_id


def test_message_wire_round_trips_exact_artifact_and_is_deeply_immutable() -> None:
    artifact = ArtifactRef(
        artifact_id="parser",
        branch="producer",
        state_id="1" * 40,
        path="out/parser.py",
        blob_id="a" * 64,
        metadata={"checks": ["unit", {"reviewed": True}]},
    )
    original = message(
        kind="artifact_ready",
        artifact_ref=artifact,
        reply_to_id="sha256:" + "b" * 64,
    )

    decoded = Message.from_json(original.to_json())

    assert decoded == original
    assert decoded.artifact_ref == artifact
    with pytest.raises(TypeError):
        decoded.payload["new"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        decoded.artifact_ref.metadata["new"] = True  # type: ignore[index,union-attr]
    with pytest.raises(FrozenInstanceError):
        decoded.kind = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda raw: raw.update(extra=True), "unknown fields"),
        (lambda raw: raw.pop("payload"), "missing required"),
        (lambda raw: raw.update(schema="taste.brains/Message/2"), "schema must"),
        (lambda raw: raw.update(generation=True), "integer >= 1"),
        (lambda raw: raw.update(message_id="sha256:" + "0" * 64), "does not match"),
    ],
)
def test_message_rejects_non_exact_wire(mutate, match: str) -> None:  # type: ignore[no-untyped-def]
    raw = message().to_dict()
    mutate(raw)

    with pytest.raises(MessageWireError, match=match):
        Message.from_dict(raw)


def test_message_json_rejects_duplicate_keys_and_nonfinite_numbers() -> None:
    with pytest.raises(MessageWireError, match="duplicate key"):
        Message.from_json(message().to_json().replace('"payload":', '"payload": {}, "payload":'))
    nonfinite = message().to_dict()
    nonfinite["payload"] = {"bad": float("nan")}
    with pytest.raises(MessageWireError, match="non-JSON number"):
        Message.from_json(json.dumps(nonfinite))


def test_stable_message_id_is_sender_and_idempotency_identity() -> None:
    first = message(payload={"value": 1})
    exact_retry = message(payload={"value": 1})
    conflicting_retry = message(payload={"value": 2}, recipient="worker-2", generation=8)

    assert first.message_id == exact_retry.message_id == conflicting_retry.message_id
    assert message(idempotency_key="different").message_id != first.message_id
    assert message(sender="other-central").message_id != first.message_id


def test_send_retry_reuses_one_exact_durable_inbox_entry(store: Store) -> None:
    communicator = Communicator(store)
    first = communicator.send(message())
    retry = communicator.send(message())

    assert retry == first
    assert len(communicator.pending("worker-1")) == 1
    assert len(communicator.history("worker-1")) == 1


def test_conflicting_idempotency_reuse_is_rejected_even_after_ack(store: Store) -> None:
    communicator = Communicator(store)
    sent = communicator.send(message())
    communicator.accept(
        sent,
        current_generation=7,
        boundary=lambda item, disposition, generation: receipt(
            item, disposition, generation
        ),
    )

    conflicting = message(recipient="worker-2", payload={"different": True})
    with pytest.raises(IdempotencyConflict, match="different message bytes"):
        communicator.send(conflicting)

    assert communicator.pending("worker-1") == ()
    assert communicator.pending("worker-2") == ()
    assert len(communicator.history("worker-1")) == 1


def test_crash_before_send_publishes_nothing_and_retry_sends_once(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    communicator = Communicator(store)
    original = store.send

    def crash_before(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("killed before Store.send")

    monkeypatch.setattr(store, "send", crash_before)
    with pytest.raises(RuntimeError, match=r"before Store\.send"):
        communicator.send(message())
    assert store.inbox("worker-1") == []

    monkeypatch.setattr(store, "send", original)
    sent = communicator.send(message())
    assert [raw["id"] for raw in store.inbox("worker-1")] == [sent.inbox_id]


def test_crash_after_store_send_is_recovered_by_history_scan(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    communicator = Communicator(store)
    original = store.send

    def crash_after(*args: Any, **kwargs: Any) -> str:
        original(*args, **kwargs)
        raise RuntimeError("killed after Store.send")

    monkeypatch.setattr(store, "send", crash_after)
    with pytest.raises(RuntimeError, match=r"after Store\.send"):
        communicator.send(message())
    monkeypatch.setattr(store, "send", original)

    recovered = Communicator(store).send(message())

    assert recovered.message.message_id == message().message_id
    assert len(Communicator(store).history("worker-1")) == 1


def test_concurrent_same_identity_send_serializes_to_one_entry(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    first_store = Store.open(root, "concurrent-communication")
    second_store = Store.open(root, "concurrent-communication")
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(Communicator(opened).send, message())
                for opened in (first_store, second_store)
            ]
        sent = [future.result(timeout=5) for future in futures]
        assert sent[0].inbox_id == sent[1].inbox_id
        assert len(Communicator(first_store).history("worker-1")) == 1
    finally:
        first_store.close()
        second_store.close()


def test_untyped_oldest_message_blocks_without_dropping_later_typed_message(
    store: Store,
) -> None:
    untyped_id = store.send("worker-1", {"legacy": True}, sender="central")
    typed = Communicator(store).send(message())

    with pytest.raises(MessageWireError, match="untyped message"):
        Communicator(store).pending("worker-1")
    with pytest.raises(MessageWireError, match="untyped message"):
        Communicator(store).accept(
            typed,
            current_generation=7,
            boundary=lambda item, disposition, generation: receipt(
                item, disposition, generation
            ),
        )

    assert [raw["id"] for raw in store.inbox("worker-1")] == [untyped_id, typed.inbox_id]


@pytest.mark.parametrize(
    "corrupt,match",
    [
        ('{"sender":"central",', "not valid JSON"),
        (
            '{"sender":"central","sender":"central",'
            '"at":"2026-09-10T12:00:00+00:00","body":{}}',
            "duplicate key",
        ),
    ],
)
def test_corrupt_predecessor_cannot_be_skipped_by_cumulative_ack(
    store: Store,
    corrupt: str,
    match: str,
) -> None:
    corrupt_id = append_raw_inbox(store, "worker-1", corrupt)
    typed = message()
    typed_id = store.send("worker-1", typed.to_dict(), sender=typed.sender)
    item = InboxMessage(
        inbox_id=typed_id,
        received_at="2026-09-10T12:00:00+00:00",
        message=typed,
    )
    communicator = Communicator(store)
    boundary_called = False

    def boundary(*_args):  # type: ignore[no-untyped-def]
        nonlocal boundary_called
        boundary_called = True
        pytest.fail("a later message crossed a corrupt predecessor")

    # The memstore primitive is strict too: no caller may accidentally advance
    # a cumulative cursor past a predecessor it could not decode.
    with pytest.raises(ValueError, match="corrupt"):
        store.inbox("worker-1")
    with pytest.raises(MessageWireError, match=match):
        communicator.pending("worker-1")
    with pytest.raises(MessageWireError, match=match):
        communicator.accept(item, current_generation=7, boundary=boundary)

    seen_ref = f"{store.SEEN_REF}/{store.session}/worker-1"
    inbox_ref = f"{store.INBOX_REF}/{store.session}/worker-1"
    assert not boundary_called
    assert store.backend.ref_sha(seen_ref) is None
    assert store.backend.ref_sha(inbox_ref) == typed_id
    assert store.backend.parents_of(typed_id) == [corrupt_id]


def test_outer_sender_mismatch_is_rejected_and_remains_pending(store: Store) -> None:
    exact = message()
    envelope_id = store.send("worker-1", exact.to_dict(), sender="impostor")

    with pytest.raises(MessageWireError, match="sender does not match"):
        Communicator(store).pending("worker-1")

    assert store.inbox("worker-1")[0]["id"] == envelope_id


def test_acceptance_boundary_runs_before_exact_ack(store: Store) -> None:
    communicator = Communicator(store)
    item = communicator.send(message())
    observed_pending: list[str] = []

    def boundary(  # type: ignore[no-untyped-def]
        received, disposition: str, current_generation: int
    ) -> AcceptanceReceipt:
        observed_pending.extend(raw["id"] for raw in store.inbox("worker-1"))
        return receipt(received, disposition, current_generation)

    accepted = communicator.accept(item, current_generation=7, boundary=boundary)

    assert observed_pending == [item.inbox_id]
    assert accepted.disposition == "accepted"
    assert communicator.pending("worker-1") == ()


def test_boundary_failure_or_wrong_receipt_never_advances_cursor(store: Store) -> None:
    communicator = Communicator(store)
    item = communicator.send(message())

    def failed_boundary(*args: Any) -> AcceptanceReceipt:
        raise RuntimeError("durable write failed")

    with pytest.raises(RuntimeError, match="durable write failed"):
        communicator.accept(item, current_generation=7, boundary=failed_boundary)
    assert [pending.inbox_id for pending in communicator.pending("worker-1")] == [item.inbox_id]

    wrong = message(idempotency_key="other")
    wrong_item = Communicator(store).send(wrong)
    with pytest.raises(InvalidAcceptanceReceipt, match="exactly bind"):
        communicator.accept(
            item,
            current_generation=7,
            boundary=lambda _item, _disposition, _generation: AcceptanceReceipt(
                inbox_id=wrong_item.inbox_id,
                message_id=wrong.message_id,
                recipient="worker-1",
                message_generation=7,
                current_generation=7,
                disposition="accepted",
                durable_ref="4" * 40,
            ),
        )
    assert len(communicator.pending("worker-1")) == 2


def test_only_oldest_message_can_cross_acceptance_boundary(store: Store) -> None:
    communicator = Communicator(store)
    first = communicator.send(message(idempotency_key="first"))
    second = communicator.send(message(idempotency_key="second"))
    calls = 0

    def boundary(  # type: ignore[no-untyped-def]
        item, disposition: str, current_generation: int
    ) -> AcceptanceReceipt:
        nonlocal calls
        calls += 1
        return receipt(item, disposition, current_generation)

    with pytest.raises(InboxOrderError, match="not the exact oldest"):
        communicator.accept(second, current_generation=7, boundary=boundary)
    assert calls == 0

    communicator.accept(first, current_generation=7, boundary=boundary)
    communicator.accept(second, current_generation=7, boundary=boundary)
    assert calls == 2
    assert communicator.pending("worker-1") == ()


def test_crash_after_durable_acceptance_before_ack_replays_without_loss(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    communicator = Communicator(store)
    item = communicator.send(message())
    ledger = DurableLedger(store)
    original = store.mark_inbox_seen

    def crash_before_ack(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("killed before inbox ack")

    monkeypatch.setattr(store, "mark_inbox_seen", crash_before_ack)
    with pytest.raises(RuntimeError, match="before inbox ack"):
        communicator.accept(item, current_generation=7, boundary=ledger)
    durable_state = ledger.branch.head.id
    assert [raw["id"] for raw in store.inbox("worker-1")] == [item.inbox_id]
    ledger.close()

    monkeypatch.setattr(store, "mark_inbox_seen", original)
    restarted_ledger = DurableLedger(store)
    recovered = Communicator(store).accept(
        Communicator(store).pending("worker-1")[0],
        current_generation=7,
        boundary=restarted_ledger,
    )
    try:
        assert recovered.durable_ref == durable_state
        assert restarted_ledger.calls == 1
        assert Communicator(store).pending("worker-1") == ()
    finally:
        restarted_ledger.close()


def test_crash_after_ack_is_observed_as_complete_on_restart(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    communicator = Communicator(store)
    item = communicator.send(message())
    ledger = DurableLedger(store)
    original = store.mark_inbox_seen

    def crash_after_ack(branch: str, inbox_id: str) -> None:
        original(branch, inbox_id)
        raise RuntimeError("killed after inbox ack")

    monkeypatch.setattr(store, "mark_inbox_seen", crash_after_ack)
    with pytest.raises(RuntimeError, match="after inbox ack"):
        communicator.accept(item, current_generation=7, boundary=ledger)
    ledger.close()

    assert Communicator(store).pending("worker-1") == ()
    assert len(Communicator(store).history("worker-1")) == 1


def test_stale_generation_cannot_be_accepted_but_is_durably_retired(
    store: Store,
) -> None:
    communicator = Communicator(store)
    stale = communicator.send(message(generation=6))
    calls = 0

    def boundary(  # type: ignore[no-untyped-def]
        item, disposition: str, current_generation: int
    ) -> AcceptanceReceipt:
        nonlocal calls
        calls += 1
        return receipt(item, disposition, current_generation)

    assert communicator.generation_status(stale.message, 7) is GenerationStatus.STALE
    with pytest.raises(GenerationFence, match="is stale"):
        communicator.accept(stale, current_generation=7, boundary=boundary)
    assert calls == 0
    assert len(communicator.pending("worker-1")) == 1

    stale_receipt = communicator.retire_stale(
        stale,
        current_generation=7,
        boundary=boundary,
    )
    assert stale_receipt.disposition == "stale"
    assert communicator.pending("worker-1") == ()
    assert communicator.history("worker-1") == (stale,)


def test_future_generation_remains_pending_until_it_is_current(store: Store) -> None:
    communicator = Communicator(store)
    future = communicator.send(message(generation=8))

    def never_called(*_args):  # type: ignore[no-untyped-def]
        pytest.fail("future message crossed durable boundary")

    assert communicator.generation_status(future.message, 7) is GenerationStatus.FUTURE
    with pytest.raises(GenerationFence, match="is future"):
        communicator.accept(future, current_generation=7, boundary=never_called)
    with pytest.raises(GenerationFence, match="only stale"):
        communicator.retire_stale(future, current_generation=7, boundary=never_called)
    assert communicator.pending("worker-1") == (future,)

    communicator.accept(
        future,
        current_generation=8,
        boundary=lambda item, disposition, generation: receipt(
            item, disposition, generation
        ),
    )
    assert communicator.pending("worker-1") == ()


def test_artifact_request_key_deduplicates_consumers_not_requirements(store: Store) -> None:
    first = Message.artifact_request(
        idempotency_key="consumer-a.needs-parser",
        sender="consumer-a",
        recipient="central",
        generation=3,
        artifact_id="parser",
        requirement={"path": "out/parser.py", "tests": ["unit"]},
    )
    second = Message.artifact_request(
        idempotency_key="consumer-b.needs-parser",
        sender="consumer-b",
        recipient="central",
        generation=3,
        artifact_id="parser",
        requirement={"tests": ["unit"], "path": "out/parser.py"},
    )
    changed_generation = Message.artifact_request(
        idempotency_key="consumer-c.needs-parser",
        sender="consumer-c",
        recipient="central",
        generation=4,
        artifact_id="parser",
        requirement={"path": "out/parser.py", "tests": ["unit"]},
    )
    changed_requirement = Message.artifact_request(
        idempotency_key="consumer-d.needs-parser",
        sender="consumer-d",
        recipient="central",
        generation=3,
        artifact_id="parser",
        requirement={"path": "out/parser.py", "tests": ["integration"]},
    )
    communicator = Communicator(store)
    items = tuple(communicator.send(item) for item in (first, second))

    assert artifact_request_dedup_key(first) == artifact_request_dedup_key(second)
    assert artifact_request_dedup_key(changed_generation) != artifact_request_dedup_key(first)
    assert artifact_request_dedup_key(changed_requirement) != artifact_request_dedup_key(first)
    groups = group_artifact_requests(items)
    assert len(groups) == 1
    assert next(iter(groups.values())) == items


def test_artifact_request_key_rejects_ambiguous_payload() -> None:
    request = Message.create(
        idempotency_key="bad-artifact-request",
        kind="artifact_request",
        sender="worker-1",
        recipient="central",
        generation=1,
        payload={"artifact_id": "parser", "requirement": {}, "surprise": True},
    )
    with pytest.raises(MessageWireError, match="unknown fields"):
        artifact_request_dedup_key(request)


def test_semantic_message_policy_is_injected_on_send_and_receive(store: Store) -> None:
    def assignments_only(value: Message) -> None:
        if value.kind != "assignment_ready":
            raise MessageWireError("policy rejected message kind")

    communicator = Communicator(store, message_policy=assignments_only)
    communicator.send(message())
    with pytest.raises(MessageWireError, match="policy rejected"):
        communicator.send(message(idempotency_key="artifact", kind="artifact_ready"))

    raw = message(idempotency_key="raw-artifact", kind="artifact_ready")
    store.send(raw.recipient, raw.to_dict(), sender=raw.sender)
    with pytest.raises(MessageWireError, match="policy rejected"):
        communicator.pending("worker-1")
