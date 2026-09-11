"""Production central communication never loses an accepted inbox item."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("claude_agent_sdk", reason="central runtime imports the brain layer")

from taste.brains.central_communication import (
    ACCEPTANCE_INDEX_SCHEMA,
    CentralCommunication,
    CommunicationLedgerCorruption,
    SharedCommunicationStateError,
    central_recipient_for_goal,
)
from taste.brains.central_planner import Goal
from taste.brains.communication import (
    MESSAGE_SCHEMA,
    Communicator,
    Message,
    MessageWireError,
    artifact_request_dedup_key,
)
from taste.brains.records import PlanRevision
from taste.memstore import Store


class Context:
    def __init__(self, root: Path) -> None:
        self.store = Store.open(root, "central-communication-test")
        self.control = self.store.branch("central-control", producer="central")
        self.lock = threading.RLock()
        self.goal = Goal(
            goal_id="goal-1",
            task="finish the exact product",
            success_criteria=("the exact requested product is delivered",),
        )
        self.recipient = central_recipient_for_goal(self.goal.goal_id)

    def plan(self, generation: int) -> PlanRevision:
        return PlanRevision(
            plan_id=f"plan-generation-{generation}",
            generation=generation,
            goal_id=self.goal.goal_id,
            based_on_state_id=self.control.head.id,
            observed_heads={self.control.name: self.control.head.id},
            assignments=(),
            created_at=f"2030-01-0{generation}T00:00:00Z",
        )

    def hook(self, **kwargs: Any) -> CentralCommunication:
        return CentralCommunication(
            self.store,
            goal_id=self.goal.goal_id,
            control=self.control,
            control_lock=self.lock,
            **kwargs,
        )

    def close(self) -> None:
        self.control.close()
        self.store.close()


@pytest.fixture
def context(tmp_path: Path):
    value = Context(tmp_path / "repo")
    yield value
    value.close()


def worker_message(
    key: str,
    *,
    kind: str = "progress",
    generation: int = 1,
    sender: str = "worker-1",
    payload: dict[str, Any] | None = None,
) -> Message:
    return Message.create(
        idempotency_key=key,
        kind=kind,
        sender=sender,
        recipient="central",
        generation=generation,
        payload={"detail": key} if payload is None else payload,
    )


def send(context: Context, message: Message):  # type: ignore[no-untyped-def]
    if message.recipient == "central":
        raw = message.to_dict()
        raw["recipient"] = context.recipient
        message = Message.from_dict(raw)
    return Communicator(context.store).send(message)


def test_crash_after_acceptance_checkpoint_before_ack_replays_exactly(
    context: Context,
) -> None:
    item = send(context, worker_message("worker.blocked", kind="blocked"))
    armed = True

    def fault(boundary: str, payload: Any) -> None:
        nonlocal armed
        if boundary == "acceptance_checkpoint" and armed:
            armed = False
            raise RuntimeError("crash after acceptance checkpoint")

    crashing = context.hook(fault_injector=fault)
    with pytest.raises(RuntimeError, match="crash after acceptance"):
        crashing.signals(goal=context.goal, plan=context.plan(1), runs=())

    assert Communicator(context.store).pending(context.recipient) == (item,)
    index_before = context.control.head.record(crashing._index_path)
    assert index_before["schema"] == ACCEPTANCE_INDEX_SCHEMA
    assert len(index_before["entries"]) == 1

    restarted = context.hook()
    signals = restarted.signals(goal=context.goal, plan=context.plan(1), runs=())

    assert len(signals) == 1
    assert signals[0].kind == "blocked"
    assert signals[0].requires_replan is True
    assert signals[0].metadata["inbox_id"] == item.inbox_id
    assert signals[0].metadata["message"]["message_id"] == item.message.message_id
    assert Communicator(context.store).pending(context.recipient) == ()
    assert context.control.head.record(restarted._index_path) == index_before


def test_stale_is_retired_current_is_signalled_and_future_remains_pending(
    context: Context,
) -> None:
    stale = send(context, worker_message("stale", generation=1))
    current = send(
        context,
        worker_message("current", kind="replan_request", generation=2),
    )
    future = send(context, worker_message("future", kind="failure", generation=3))
    hook = context.hook()

    signals = hook.signals(goal=context.goal, plan=context.plan(2), runs=())

    assert [signal.kind for signal in signals] == ["replan_request"]
    assert signals[0].metadata["inbox_id"] == current.inbox_id
    assert Communicator(context.store).pending(context.recipient) == (future,)
    index = context.control.head.record(hook._index_path)
    assert [entry["inbox_id"] for entry in index["entries"]] == [
        stale.inbox_id,
        current.inbox_id,
    ]
    assert [entry["disposition"] for entry in index["entries"]] == [
        "stale",
        "accepted",
    ]
    assert [entry["sequence"] for entry in index["entries"]] == [1, 2]

    next_signals = hook.signals(goal=context.goal, plan=context.plan(3), runs=())
    assert [signal.kind for signal in next_signals] == ["failure"]
    assert next_signals[0].metadata["inbox_id"] == future.inbox_id
    assert Communicator(context.store).pending(context.recipient) == ()


def test_future_oldest_blocks_later_current_message(context: Context) -> None:
    future = send(context, worker_message("future-first", generation=2))
    current = send(context, worker_message("current-second", generation=1))

    assert context.hook().signals(
        goal=context.goal,
        plan=context.plan(1),
        runs=(),
    ) == ()
    assert Communicator(context.store).pending(context.recipient) == (future, current)


def test_plan_generation_cannot_regress_behind_acceptance_ledger(context: Context) -> None:
    send(context, worker_message("accepted-generation-2", generation=2))
    hook = context.hook()
    hook.signals(goal=context.goal, plan=context.plan(2), runs=())

    with pytest.raises(CommunicationLedgerCorruption, match="generation regressed"):
        hook.signals(goal=context.goal, plan=context.plan(1), runs=())


@pytest.mark.parametrize("wire", ["untyped", "malformed_typed"])
def test_untyped_or_malformed_oldest_blocks_without_ledger_or_ack(
    context: Context,
    wire: str,
) -> None:
    body = (
        {"kind": "legacy", "detail": "not typed"}
        if wire == "untyped"
        else {"schema": MESSAGE_SCHEMA, "kind": "blocked"}
    )
    corrupt_id = context.store.send(context.recipient, body, sender="worker-1")
    valid_message = worker_message("valid-behind-corruption", kind="blocked")
    valid_raw = valid_message.to_dict()
    valid_raw["recipient"] = context.recipient
    valid_id = context.store.send(
        context.recipient, valid_raw, sender=valid_message.sender
    )
    hook = context.hook()

    with pytest.raises(MessageWireError):
        hook.signals(goal=context.goal, plan=context.plan(1), runs=())

    assert [item["id"] for item in context.store.inbox(context.recipient)] == [
        corrupt_id,
        valid_id,
    ]
    assert context.control.head.read(hook._index_path) is None


def test_semantically_malformed_artifact_request_is_not_accepted(
    context: Context,
) -> None:
    malformed = Message.create(
        idempotency_key="bad-artifact-request",
        kind="artifact_request",
        sender="worker-1",
        recipient=context.recipient,
        generation=1,
        payload={"artifact_id": "parser", "requirement": {}, "surprise": True},
    )
    item = Communicator(context.store).send(malformed)
    hook = context.hook()

    with pytest.raises(MessageWireError, match="unknown fields"):
        hook.signals(goal=context.goal, plan=context.plan(1), runs=())

    assert Communicator(context.store).pending(context.recipient) == (item,)
    assert context.control.head.read(hook._index_path) is None


@pytest.mark.parametrize("damage", ["delete", "tamper"])
def test_deleted_or_tampered_receipt_fails_closed(
    context: Context,
    damage: str,
) -> None:
    send(context, worker_message("accepted", kind="blocked"))
    hook = context.hook()
    hook.signals(goal=context.goal, plan=context.plan(1), runs=())
    index = context.control.head.record(hook._index_path)
    receipt_path = index["entries"][0]["path"]
    if damage == "delete":
        context.control.path(receipt_path).unlink()
    else:
        context.control.write(receipt_path, json.dumps({"tampered": True}) + "\n")
    context.control.checkpoint(f"fault injection: {damage} acceptance receipt")

    with pytest.raises(CommunicationLedgerCorruption):
        hook.signals(goal=context.goal, plan=context.plan(1), runs=())


def test_acked_message_is_replayed_until_generation_changes(context: Context) -> None:
    item = send(context, worker_message("one-shot", kind="blocked"))
    first_hook = context.hook()
    first = tuple(
        first_hook.signals(goal=context.goal, plan=context.plan(1), runs=())
    )
    assert Communicator(context.store).pending(context.recipient) == ()

    same_process = tuple(
        first_hook.signals(goal=context.goal, plan=context.plan(1), runs=())
    )
    restarted = tuple(
        context.hook().signals(goal=context.goal, plan=context.plan(1), runs=())
    )

    assert first == same_process == restarted
    assert first[0].metadata["inbox_id"] == item.inbox_id
    assert context.hook().signals(
        goal=context.goal,
        plan=context.plan(2),
        runs=(),
    ) == ()


def test_control_ref_rollback_cannot_forget_an_acknowledged_message(
    context: Context,
) -> None:
    before_acceptance = context.control.head.id
    item = send(context, worker_message("rollback-after-ack", kind="blocked"))
    hook = context.hook()
    assert hook.signals(goal=context.goal, plan=context.plan(1), runs=())
    accepted_head = context.control.head.id
    assert accepted_head != before_acceptance
    assert Communicator(context.store).pending(context.recipient) == ()

    assert context.store.backend.cas_update_ref(
        context.control.ref,
        before_acceptance,
        accepted_head,
    )
    context.control.backend.reset_hard_to_head()

    with pytest.raises(
        CommunicationLedgerCorruption,
        match="acknowledged inbox entry has no durable acceptance receipt",
    ):
        hook.signals(goal=context.goal, plan=context.plan(1), runs=())
    assert item in Communicator(context.store).history(context.recipient)


def test_partial_control_ref_rollback_cannot_truncate_acknowledged_prefix(
    context: Context,
) -> None:
    hook = context.hook()
    first = send(context, worker_message("first-ack", kind="blocked"))
    assert hook.signals(goal=context.goal, plan=context.plan(1), runs=())
    first_acceptance_head = context.control.head.id
    second = send(context, worker_message("second-ack", kind="failure"))
    assert len(hook.signals(goal=context.goal, plan=context.plan(1), runs=())) == 2
    second_acceptance_head = context.control.head.id
    assert Communicator(context.store).pending(context.recipient) == ()

    assert context.store.backend.cas_update_ref(
        context.control.ref,
        first_acceptance_head,
        second_acceptance_head,
    )
    context.control.backend.reset_hard_to_head()

    with pytest.raises(
        CommunicationLedgerCorruption,
        match="does not exactly cover the inbox cursor",
    ):
        context.hook().signals(goal=context.goal, plan=context.plan(1), runs=())
    assert Communicator(context.store).history(context.recipient) == (first, second)


@pytest.mark.parametrize(
    "kind",
    ["artifact_ready", "blocked", "failure", "replan_request", "unknown_notice"],
)
def test_safe_default_makes_every_accepted_kind_actionable(
    context: Context,
    kind: str,
) -> None:
    item = send(context, worker_message(f"kind.{kind}", kind=kind))

    signals = context.hook().signals(
        goal=context.goal,
        plan=context.plan(1),
        runs=(),
    )

    assert len(signals) == 1
    assert signals[0].kind == kind
    assert signals[0].requires_replan is True
    assert signals[0].metadata["replan_policy"] == "all_accepted_messages"
    assert signals[0].metadata["message"]["message_id"] == item.message.message_id
    assert signals[0].metadata["receipt"]["durable_ref"] == signals[0].metadata[
        "acceptance_state_id"
    ]


def test_equal_artifact_requests_become_one_deduplicated_producer_signal(
    context: Context,
) -> None:
    first_message = Message.artifact_request(
        idempotency_key="consumer-a.needs-parser",
        sender="consumer-a",
        recipient=context.recipient,
        generation=1,
        artifact_id="parser",
        requirement={"path": "out/parser.py", "checks": ["unit"]},
    )
    second_message = Message.artifact_request(
        idempotency_key="consumer-b.needs-parser",
        sender="consumer-b",
        recipient=context.recipient,
        generation=1,
        artifact_id="parser",
        requirement={"checks": ["unit"], "path": "out/parser.py"},
    )
    first = send(context, first_message)
    second = send(context, second_message)

    signals = context.hook().signals(
        goal=context.goal,
        plan=context.plan(1),
        runs=(),
    )

    assert len(signals) == 1
    signal = signals[0]
    key = artifact_request_dedup_key(first_message)
    assert signal.kind == "artifact_request"
    assert signal.signal_id.endswith(key.removeprefix("sha256:"))
    assert signal.metadata["artifact_request_dedup_key"] == key
    group = signal.metadata["artifact_request_group"]
    assert [entry["inbox_id"] for entry in group] == [first.inbox_id, second.inbox_id]
    assert [entry["message"]["sender"] for entry in group] == [
        "consumer-a",
        "consumer-b",
    ]
    assert signal.requires_replan is True


def test_goal_derived_recipient_prevents_cross_goal_consumption(context: Context) -> None:
    other_goal = Goal(
        goal_id="goal-2",
        task="a distinct product with the same worker names",
        success_criteria=("the other exact product is delivered",),
    )
    other_recipient = central_recipient_for_goal(other_goal.goal_id)
    other_message = Message.create(
        idempotency_key="goal-2.worker-1.blocked",
        kind="blocked",
        sender="worker-1",
        recipient=other_recipient,
        generation=1,
        payload={"same_assignment_id": "assignment-1"},
    )
    other_item = Communicator(context.store).send(other_message)

    assert context.hook().signals(
        goal=context.goal,
        plan=context.plan(1),
        runs=(),
    ) == ()
    assert Communicator(context.store).pending(other_recipient) == (other_item,)

    other_plan = PlanRevision(
        plan_id="other-plan-generation-1",
        generation=1,
        goal_id=other_goal.goal_id,
        based_on_state_id=context.control.head.id,
        observed_heads={context.control.name: context.control.head.id},
        assignments=(),
        created_at="2030-02-01T00:00:00Z",
    )
    other_hook = CentralCommunication(
        context.store,
        goal_id=other_goal.goal_id,
        control=context.control,
        control_lock=context.lock,
    )
    signals = other_hook.signals(goal=other_goal, plan=other_plan, runs=())
    assert len(signals) == 1
    assert signals[0].metadata["message"]["recipient"] == other_recipient
    assert Communicator(context.store).pending(other_recipient) == ()


def test_adapter_requires_exact_shared_store_branch_and_lock(tmp_path: Path) -> None:
    first = Context(tmp_path / "first")
    second = Context(tmp_path / "second")
    try:
        with pytest.raises(SharedCommunicationStateError):
            CentralCommunication(
                first.store,
                goal_id=first.goal.goal_id,
                control=second.control,
                control_lock=first.lock,
            )
        with pytest.raises(SharedCommunicationStateError):
            CentralCommunication(
                first.store,
                goal_id=first.goal.goal_id,
                control=first.control,
                control_lock=first.lock,
                communicator=Communicator(second.store),
            )
        with pytest.raises(SharedCommunicationStateError):
            CentralCommunication(
                first.store,
                goal_id=first.goal.goal_id,
                control=first.control,
                control_lock=first.lock,
                recipient="central",
            )
        hook = first.hook()
        with pytest.raises(SharedCommunicationStateError):
            hook.assert_shared(
                store=first.store,
                control=first.control,
                control_lock=threading.RLock(),
            )
    finally:
        second.close()
        first.close()
