"""Evidence survives memory rollback and the model/tool publication crash windows."""

import asyncio
import json

import pytest

from taste.brains import responses_audit
from taste.brains.responses_conversation import ToolOutcome
from taste.brains.responses_session import ResponsesFenced, ResponsesSession
from tests.test_azure_openai import config
from tests.test_openai_responses import message
from tests.test_responses_conversation import install, make, tool
from tests.test_responses_conversation import sdk_transport as _sdk_transport
from tests.test_responses_conversation import worker as _worker

sdk_transport = _sdk_transport
worker = _worker


def events(worker, kind):
    return [r for r in worker.session.conversation_audit() if r["event"]["kind"] == "responses_" + kind]


def test_rollback_preserves_external_effects_and_exact_observations_across_restart(worker, sdk_transport, tmp_path):
    install(sdk_transport, [tool(identifier="same")], [tool("second", identifier="same")])
    outside = tmp_path / "external.txt"

    async def execute(_effect, call):
        with outside.open("a") as handle:
            handle.write(call.arguments["text"] + "\n")
        return ToolOutcome("observed: " + call.arguments["text"] + "\r\n\u2603")

    conversation = make(worker, execute)
    conversation.observe("instruction", "do the work")
    before = worker.branch.checkpoint("before any effect")

    async def run():
        await conversation.step()
        worker.branch.checkpoint("first attempt")
        worker.branch.rollback(before, "retry reasoning, effects persist")
        await conversation.step()
    asyncio.run(run())
    snapshot = worker.session.conversation_audit()
    worker.branch.checkpoint("second attempt")
    worker.session.close()
    worker.session = ResponsesSession.open(worker.session.directory, worker.session.binding, config())
    make(worker, execute)
    assert worker.session.conversation_audit() == snapshot
    assert outside.read_text() == "correct\nsecond\n"
    assert [r["event"]["result"]["content"] for r in events(worker, "tool_result")] == [
        "observed: correct\r\n\u2603", "observed: second\r\n\u2603"]
    assert len({r["event"]["effect_id"] for r in events(worker, "tool_intent")}) == 2
    assert all(row["published"] for row in snapshot)
    snapshot[-1]["event"]["result"]["content"] = "tampered detached copy"
    assert "tampered detached copy" not in json.dumps(worker.session.conversation_audit())


@pytest.mark.parametrize("boundary", ["before_memory", "after_memory", "after_publish"])
def test_tool_result_persistence_windows_preserve_evidence_without_replaying(worker, sdk_transport, monkeypatch, boundary):
    install(sdk_transport, [tool()], [message("finished")])
    effects = []

    async def execute(effect, _call):
        effects.append(effect)
        return ToolOutcome("exact result", True)

    conversation = make(worker, execute)
    original = worker.branch.turn

    def fail(**event):
        if event["kind"] == "responses_tool_result":
            if boundary != "before_memory":
                original(**event)
            if boundary == "after_publish":
                row = events(worker, "tool_result")[-1]
                worker.session.publish_conversation_event(row["id"])
            raise OSError("lost acknowledgement")
        original(**event)

    monkeypatch.setattr(worker.branch, "turn", fail)
    with pytest.raises(OSError):
        asyncio.run(conversation.step())
    receipt = events(worker, "tool_result")[0]
    assert receipt["event"]["result"] == {"content": "exact result", "is_error": True}
    assert receipt["published"] == (boundary == "after_publish")
    monkeypatch.setattr(worker.branch, "turn", original)
    worker.branch.checkpoint("crash boundary")
    worker.session.close()
    worker.session = ResponsesSession.open(worker.session.directory, worker.session.binding, config())
    resumed = make(worker, execute)
    if boundary == "before_memory":
        with pytest.raises(ResponsesFenced, match="unknown outcome"):
            asyncio.run(resumed.step())
        assert events(worker, "tool_result")[0]["published"] is False
    else:
        asyncio.run(resumed.step())
        assert events(worker, "tool_result")[0]["published"] is True
    assert len(effects) == 1


def test_full_audit_blocks_dispatch_before_an_unrecorded_paid_call(worker, sdk_transport, monkeypatch):
    sent = install(sdk_transport, [tool()])
    conversation = make(worker)
    monkeypatch.setattr(responses_audit, "MAX_EVENTS", 1)
    with pytest.raises(responses_audit.ResponsesAuditError, match="capacity"):
        asyncio.run(conversation.step())
    assert not sent and not worker.branch.path("artifact.txt").exists()


def test_recording_failure_prevents_tool_effect_even_after_paid_reply(worker, sdk_transport, monkeypatch):
    sent = install(sdk_transport, [tool()])
    conversation = make(worker)
    original = worker.session.record_conversation_event

    def fail(prefix, event):
        if event["kind"] == "responses_tool_intent":
            raise OSError("audit disk unavailable")
        return original(prefix, event)

    monkeypatch.setattr(worker.session, "record_conversation_event", fail)
    with pytest.raises(OSError):
        asyncio.run(conversation.step())
    assert len(sent) == 1 and not worker.branch.path("artifact.txt").exists()
    assert len(events(worker, "completion")) == 1
    assert not events(worker, "tool_intent")


def test_legacy_context_cannot_be_advertised_as_a_complete_audit(worker, sdk_transport):
    sent = install(sdk_transport, [message("done")])
    make(worker)
    # Simulate an older journal which has memory but no conversation evidence.
    with worker.session._db:
        worker.session._db.execute("DELETE FROM conversation_events")
    with pytest.raises(responses_audit.ResponsesAuditError, match="no matching"):
        make(worker)
    assert not sent
