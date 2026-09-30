"""Cross-check provider receipts against conversation publication crash windows."""

import asyncio
from copy import deepcopy

import pytest

from taste.benchmarks.goal_trajectory import _worker_calls
from taste.benchmarks.worker_trajectory import worker_trajectory
from taste.brains.goal_entrypoint import GoalInputError
from tests.test_responses_conversation import install, make, tool
from tests.test_responses_conversation import sdk_transport as _sdk_transport
from tests.test_responses_conversation import worker as _worker

worker, sdk_transport = _worker, _sdk_transport


def test_paid_reply_before_conversation_publication_is_retained_without_inventing_effects(
        worker, sdk_transport, monkeypatch):
    sent = install(sdk_transport, [tool()])
    conversation = make(worker)
    original = worker.session.record_conversation_event

    def fail(events, event):
        if event["kind"] == "responses_completion":
            raise OSError("crash after paid receipt before conversation publication")
        return original(events, event)

    monkeypatch.setattr(worker.session, "record_conversation_event", fail)
    with pytest.raises(OSError):
        asyncio.run(conversation.step())
    rows = worker.session.conversation_audit()
    calls = worker.session.call_evidence()
    trace = worker_trajectory(rows, run_id=worker.session.binding.run_id)
    gaps = []
    _worker_calls(trace, rows, calls, gaps, worker.session.binding.run_id)
    assert len(sent) == 1 and gaps[0].startswith("worker_reply_unpublished:")
    assert not worker.branch.path("artifact.txt").exists()
    assert trace["steps"][-1]["tool_calls"][0]["arguments"] == {"text": "correct"}
    assert trace["steps"][-1]["extra"]["memory_published"] is False
    assert not any(s.get("observation") for s in trace["steps"])
    assert trace["steps"][-1]["metrics"]["cost_usd"] == worker.session.known_cost_usd
    calls[0]["completion"]["text_blocks"].append("altered detached evidence")
    assert "altered detached evidence" not in worker.session.call_evidence()[0]["completion"]["text_blocks"]


def test_changed_conversation_cannot_override_a_real_provider_receipt(worker, sdk_transport):
    install(sdk_transport, [tool()])
    conversation = make(worker)
    asyncio.run(conversation.step())
    rows = worker.session.conversation_audit()
    trace = worker_trajectory(rows, run_id=worker.session.binding.run_id)
    altered = deepcopy(rows)
    for row in altered:
        if row["event"]["kind"] == "responses_completion":
            row["event"]["completion"]["text_blocks"] = ["unearned claim"]
    with pytest.raises(GoalInputError, match="provider receipt"):
        _worker_calls(trace, altered, worker.session.call_evidence(), [], worker.session.binding.run_id)
