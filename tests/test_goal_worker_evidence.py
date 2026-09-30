"""Cross-check provider receipts against conversation publication crash windows."""

import asyncio
from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from taste.benchmarks.goal_trajectory import _worker_calls, _workers
from taste.benchmarks.worker_trajectory import worker_trajectory
from taste.brains.goal_entrypoint import GoalInputError
from taste.brains.supervisor import CentralSupervisor
from taste.memstore import Store
from tests.test_azure_worker_policy import assignment
from tests.test_brains_central_host import NoLaunchLauncher
from tests.test_responses_conversation import install, make, tool
from tests.test_responses_conversation import sdk_transport as _sdk_transport
from tests.test_responses_conversation import worker as _worker

worker, sdk_transport = _worker, _sdk_transport


def test_prepared_but_unlaunched_worker_is_recorded_without_false_missing_evidence(tmp_path, sdk_transport):
    sent, _ = sdk_transport(lambda *_: pytest.fail("an unlaunched worker cannot call a model"))
    with (closing(Store.open(tmp_path / "repo", "no-launch")) as store,
          CentralSupervisor(store, launcher=NoLaunchLauncher()) as supervisor):
        source = replace(assignment(), base_state_id=supervisor.integration.head.id)
        prepared = supervisor.prepare(source, wall_timeout_seconds=20)
        stopped = supervisor.stop(prepared.run_id, "budget_blocked")
        assert stopped.recovery_status == "complete"
        gaps = []
        traces, cost = _workers(SimpleNamespace(store=store), supervisor.runs(), gaps)
        assert not gaps and not sent and cost == 0
        assert len(traces) == 1 and traces[0]["extra"]["not_launched"]
        assert traces[0]["extra"]["run_id"] == prepared.run_id
        assert not any(s["source"] == "agent" for s in traces[0]["steps"])


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
