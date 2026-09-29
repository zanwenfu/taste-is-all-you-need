"""Real Responses turns projected into ATIF without promoting internal claims."""

import asyncio
import json

import pytest

from taste.benchmarks.worker_trajectory import worker_trajectory
from taste.brains.responses_conversation import ToolOutcome
from tests.test_openai_responses import message
from tests.test_responses_conversation import install, make, tool
from tests.test_responses_conversation import sdk_transport as _sdk_transport
from tests.test_responses_conversation import worker as _worker

sdk_transport = _sdk_transport
worker = _worker


def export(worker):
    return worker_trajectory(worker.session.conversation_audit(), run_id=worker.session.binding.run_id)


def save_case(tmp_path, trace, calls):
    # Also consumed by the separate pinned Harbor + Errata compatibility check.
    (tmp_path / "trajectory.case.json").write_text(json.dumps({"trajectory": trace,
        "expected_reply": "", "expected_calls": calls}, ensure_ascii=False))


def test_rollback_reused_provider_ids_and_exact_results_are_preserved(worker, sdk_transport, tmp_path):
    install(sdk_transport, [tool(identifier="same")], [tool("second", identifier="same")],
            [message(' {"status":"completed","summary":"internal claim"} ')])
    observed = 'exit 1\r\n\u2603\n{"encoding":"base64","content":"/w==","dropped_bytes":40}'

    async def execute(_effect, _call):
        return ToolOutcome(observed, True)

    conversation = make(worker, execute)
    conversation.observe("task", "continue the developer conversation")
    before = worker.branch.checkpoint("before task work")

    async def run():
        await conversation.step()
        worker.branch.checkpoint("discarded reasoning")
        worker.branch.rollback(before, "retry")
        await conversation.step()
        await conversation.step()
    asyncio.run(run())
    trace = export(worker)
    calls = [call for s in trace["steps"] for call in s.get("tool_calls", [])]
    results = [r for s in trace["steps"] for r in s.get("observation", {}).get("results", [])]
    assert len(calls) == len(results) == 2
    assert calls[0]["tool_call_id"] != calls[1]["tool_call_id"]
    assert [r["source_call_id"] for r in results] == [c["tool_call_id"] for c in calls]
    assert all(r["content"] == observed and r["extra"]["is_error"] for r in results)
    assert all(s["extra"]["is_sidechain"] for s in trace["steps"] if s["source"] == "agent")
    assert trace["extra"]["incomplete_worker_trace"] is False
    save_case(tmp_path, trace, [{"args": {"text": text}, "result": observed} for text in ("correct", "second")])
    calls[0]["arguments"]["text"] = "changed projection"
    assert "changed projection" not in json.dumps(worker.session.conversation_audit())


@pytest.mark.parametrize("fail", ["tool", "publication", "provider"])
def test_interrupted_work_has_no_invented_observation_or_final_reply(worker, sdk_transport, monkeypatch, tmp_path, fail):
    install(sdk_transport, [tool()])

    async def execute(_effect, _call):
        if fail == "tool":
            raise OSError("outcome lost")
        return ToolOutcome("saved but unpublished", False)

    conversation = make(worker, execute)
    if fail == "publication":
        original = worker.branch.turn

        def turn(**event):
            if event["kind"] == "responses_tool_result":
                raise OSError("memory publication failed")
            original(**event)
        monkeypatch.setattr(worker.branch, "turn", turn)
    elif fail == "provider":
        async def failed(*_args, **_kwargs):
            raise OSError("model reply lost")
        monkeypatch.setattr(worker.session, "complete", failed)
    with pytest.raises(OSError):
        asyncio.run(conversation.step())
    trace = export(worker)
    assert trace["extra"]["incomplete_worker_trace"]
    observations = [r for s in trace["steps"] for r in s.get("observation", {}).get("results", [])]
    if fail == "publication":
        assert observations[0]["content"] == "saved but unpublished"
        assert observations[0]["extra"]["memory_published"] is False
    else:
        assert not observations
    save_case(tmp_path, trace, [] if fail == "provider" else [{"args": {"text": "correct"},
        "result": "saved but unpublished" if fail == "publication" else ""}])
