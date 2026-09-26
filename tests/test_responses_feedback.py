"""Typed inbox and verdict acceptance across Responses, memory and cursors."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from taste.brains.communication import Communicator, Message
from taste.brains.contract import CONTRACT_PATH, Contract
from taste.brains.records import Assignment, contract_digest
from taste.brains.responses_conversation import ResponsesConversation
from taste.brains.responses_feedback import ResponsesFeedback
from taste.brains.responses_session import ResponsesConflict, ResponsesSession
from taste.brains.worker_protocol import ASSIGNMENT_PATH, ContractMismatch, assignment_run_id
from taste.memstore import Store, Verdict
from tests.test_azure_openai import config, httpx
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_openai_responses import message, response
from tests.test_responses_session import binding

sdk_transport = _sdk_transport


@pytest.fixture
def worker(tmp_path):
    store = Store.open(tmp_path / "repo", "feedback")
    branch = store.branch("worker")
    brief = Contract("worker", "perform the task", success_criteria=("the result is correct",))
    assignment = Assignment("task", 2, 0, brief, contract_digest(brief), branch.head.id,
                            model=binding().model)
    branch.write(CONTRACT_PATH, brief.to_json())
    branch.write(ASSIGNMENT_PATH, assignment.to_json())
    branch.checkpoint("prepared assignment")
    limits = binding(run_id=assignment_run_id(assignment), max_calls=10)
    session = ResponsesSession.create(store.backend.common_dir / "responses", limits, config())
    conversation = ResponsesConversation(branch, session, system="Return a structured worker claim.", tools={})
    conversation.observe("contract", brief.brief())
    worker = SimpleNamespace(store=store, branch=branch, assignment=assignment, session=session,
                             conversation=conversation, communicator=Communicator(store))
    worker.feedback = ResponsesFeedback(conversation, assignment)
    yield worker
    worker.session.close()
    store.close()


def claim(*ids, verdicts=None, **changes):
    return {"status": "completed", "summary": "Finished", "evidence": ["Checked result"],
            "accepted_inbox_ids": list(ids), "accepted_verdicts": verdicts or {}, **changes}


def install(sdk_transport, handler):
    def handle(wire):
        raw = handler(wire)
        if not isinstance(raw, str):
            raw = json.dumps(raw)
        return httpx.Response(200, json=response(model="gpt-6-sol", output=[message(raw)]),
                              headers={"x-ms-served-model": binding().model})
    sent, _ = sdk_transport(handle)
    return sent


def send(worker, key, *, generation=2):
    return worker.communicator.send(Message.create(
        idempotency_key=key, kind="feedback", sender="central", recipient="worker",
        generation=generation, payload={"instruction": "incorporate " + key},
    ))


def reopen(worker):
    worker.session.close()
    worker.session = ResponsesSession.open(worker.session.directory, worker.session.binding, config())
    worker.conversation = ResponsesConversation(worker.branch, worker.session,
                                                system="Return a structured worker claim.", tools={})
    worker.feedback = ResponsesFeedback(worker.conversation, worker.assignment)


def test_observation_is_not_acceptance_and_receipt_excludes_later_inputs(worker, sdk_transport):
    first = send(worker, "first")
    install(sdk_transport, lambda _wire: claim(first.inbox_id))
    worker.feedback.observe_pending()
    assert worker.communicator.pending("worker") == (first,)

    async def scenario():
        await worker.conversation.step()
        receipt = worker.conversation.completed_turn()
        later = send(worker, "later")
        worker.feedback.observe_pending()
        assert "inbox." + later.inbox_id not in receipt.submitted_inputs
        assert "inbox." + later.inbox_id not in worker.conversation.completed_turn().submitted_inputs
        worker.feedback.accept_latest()
        assert worker.communicator.pending("worker") == (later,)
        with pytest.raises(TypeError):
            receipt.submitted_inputs["forged"] = "input"

    asyncio.run(scenario())


def test_ack_for_a_message_that_arrived_during_the_call_is_rejected(worker, sdk_transport):
    late = []

    def answer(_wire):
        late.append(send(worker, "too-late"))
        return claim(late[0].inbox_id)

    sent = install(sdk_transport, answer)

    async def scenario():
        await worker.conversation.step()
        worker.feedback.observe_pending()
        with pytest.raises(ContractMismatch, match="never submitted"):
            worker.feedback.accept_latest()
        assert worker.communicator.pending("worker") == tuple(late)

    asyncio.run(scenario())
    assert len(sent) == 1


@pytest.mark.parametrize("failure_point", ["claim", "acceptance", "cursor"])
def test_acceptance_recovery_keeps_exact_model_evidence_and_never_pays_again(
        worker, sdk_transport, monkeypatch, failure_point):
    first = send(worker, "first")
    sent = install(sdk_transport, lambda _wire: claim(first.inbox_id))
    worker.feedback.observe_pending()

    async def scenario():
        await worker.conversation.step()
        original_turn = worker.branch.turn

        def fail_turn(**event):
            original_turn(**event)
            target = "worker_claim" if failure_point == "claim" else "inbox_accepted"
            if event.get("kind") == target:
                raise OSError("durable acceptance acknowledgement lost")

        def fail_cursor(*_args):
            raise OSError("cursor not persisted")

        with monkeypatch.context() as patch:
            if failure_point == "cursor":
                patch.setattr(worker.store, "mark_inbox_seen", fail_cursor)
            else:
                patch.setattr(worker.branch, "turn", fail_turn)
            with pytest.raises(OSError):
                worker.feedback.accept_latest()
        assert worker.communicator.pending("worker") == (first,)
        worker.branch.checkpoint("interrupted acceptance")
        reopen(worker)
        worker.feedback.reconcile()
        assert not worker.communicator.pending("worker")
        worker.feedback.accept_latest()

    asyncio.run(scenario())
    assert len(sent) == 1


def test_out_of_order_acknowledgements_cannot_skip_an_older_pending_item(worker, sdk_transport):
    first, later = send(worker, "first"), send(worker, "later")
    replies = iter((claim(later.inbox_id, status="continue"), claim(first.inbox_id)))
    sent = install(sdk_transport, lambda _wire: next(replies))
    worker.feedback.observe_pending()

    async def scenario():
        await worker.conversation.step()
        worker.feedback.accept_latest()
        assert worker.communicator.pending("worker") == (first, later)
        worker.conversation.observe("continue", "Continue and process remaining feedback.")
        await worker.conversation.step()
        worker.feedback.accept_latest()
        assert not worker.communicator.pending("worker")

    asyncio.run(scenario())
    assert len(sent) == 2


def test_stale_prefix_is_retired_and_future_generation_is_never_submitted(worker):
    stale = send(worker, "stale", generation=1)
    current = send(worker, "current")
    future = send(worker, "future", generation=3)
    behind = send(worker, "behind-future")
    worker.feedback.observe_pending()
    assert worker.communicator.pending("worker") == (current, future, behind)
    context = json.dumps(worker.conversation.messages)
    assert current.inbox_id in context
    assert stale.inbox_id not in context
    assert future.inbox_id not in context
    assert behind.inbox_id not in context


def test_acknowledged_history_is_reexposed_after_worker_context_rollback(worker, sdk_transport):
    before = worker.branch.checkpoint("before feedback")
    first = send(worker, "first")
    sent = install(sdk_transport, lambda _wire: claim(first.inbox_id))
    worker.feedback.observe_pending()

    async def scenario():
        await worker.conversation.step()
        worker.feedback.accept_latest()
        worker.branch.checkpoint("accepted feedback")
        assert not worker.communicator.pending("worker")
        worker.branch.rollback(before, "restore earlier context")
        reopen(worker)
        worker.feedback.observe_pending()
        assert first.inbox_id in json.dumps(worker.conversation.messages)
        await worker.conversation.step()
        worker.feedback.accept_latest()

    asyncio.run(scenario())
    assert len(sent) == 2  # rollback requires new reasoning; neither turn is a duplicate dispatch
    assert worker.session.known_cost_usd == pytest.approx(0.000732)


def test_racing_verdict_is_not_in_the_submitted_prefix(worker, sdk_transport):
    state = worker.branch.head
    worker.store.judge(state, Verdict("pass", by="monitor", detail="first"))
    worker.feedback.observe_pending()
    sent = install(sdk_transport, lambda _wire: claim(verdicts={state.id: 1}))

    async def scenario():
        await worker.conversation.step()
        worker.store.judge(state, Verdict("fail", by="monitor", detail="arrived later"))
        worker.feedback.accept_latest()
        assert worker.branch.unacked_verdicts()[-1].detail == "arrived later"
        assert worker.branch._acked()[state.id] == 1

    asyncio.run(scenario())
    assert len(sent) == 1


def test_verdict_ack_must_not_exceed_what_the_model_received(worker, sdk_transport):
    state = worker.branch.head
    worker.store.judge(state, Verdict("pass", by="monitor", detail="first"))
    worker.feedback.observe_pending()
    install(sdk_transport, lambda _wire: claim(verdicts={state.id: 2}))

    async def scenario():
        await worker.conversation.step()
        worker.store.judge(state, Verdict("fail", by="monitor", detail="later"))
        with pytest.raises(ContractMismatch, match="never submitted"):
            worker.feedback.accept_latest()
        assert len(worker.branch.unacked_verdicts()) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("bad", [
    "not JSON", "```json\n{}\n```", '{"status":"completed","status":"blocked"}',
    json.dumps(claim(evidence=[])), json.dumps(claim(accepted_inbox_ids=["not-an-id"])),
    json.dumps(claim(accepted_verdicts={"a" * 40: True})),
    json.dumps({**claim(), "extra": "field"}),
])
def test_invalid_claims_do_not_advance_any_cursor(worker, sdk_transport, bad):
    first = send(worker, "first")
    worker.feedback.observe_pending()
    install(sdk_transport, lambda _wire: bad)

    async def scenario():
        await worker.conversation.step()
        with pytest.raises(ContractMismatch):
            worker.feedback.accept_latest()
        assert worker.communicator.pending("worker") == (first,)

    asyncio.run(scenario())


def test_a_forged_acceptance_marker_without_a_provider_receipt_is_rejected(worker):
    first = send(worker, "first")
    worker.branch.turn(kind="worker_claim", request_id="response.never-sent")
    with pytest.raises(ResponsesConflict, match="completed Responses turn"):
        worker.feedback.reconcile()
    assert worker.communicator.pending("worker") == (first,)
