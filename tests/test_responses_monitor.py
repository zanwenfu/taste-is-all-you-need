"""Azure SDK -> durable call receipts -> monitor sidecar/report boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import subprocess
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from taste.brains.contract import CONTRACT_PATH, Contract
from taste.brains.monitor import MonitorBrain
from taste.brains.monitor_judge import (
    JUDGEMENT_SCHEMA,
    TERMINAL_JUDGEMENT_SCHEMA,
    MonitorResponseError,
)
from taste.brains.responses_monitor import ResponsesMonitorJudge
from taste.brains.responses_session import ResponsesConflict, ResponsesFenced, ResponsesSession
from taste.brains.worker_protocol import ModelCallAccounting
from taste.llm import LLM, InfraFailure
from taste.memstore import Store
from tests.test_azure_openai import config, httpx
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_openai_responses import function_call, message, response
from tests.test_responses_session import binding

sdk_transport = _sdk_transport


def verdict(*, terminal=False):
    result = {"schema": TERMINAL_JUDGEMENT_SCHEMA if terminal else JUDGEMENT_SCHEMA,
              "severity": "fine", "reason": "The recorded work meets the criteria.",
              "evidence": ["The artifact contains the requested value."], "suggestion": ""}
    if terminal:
        result.update(resolved_finding_ids=[], unresolved_finding_ids=[])
    return json.dumps(result)


def reply(_wire, *, terminal=False, text=None, **overrides):
    return httpx.Response(200, json=response(
        model="gpt-6-sol", output=[message(text if text is not None else verdict(terminal=terminal))],
        **overrides), headers={"x-ms-served-model": binding().model})


@pytest.fixture
def worker(tmp_path):
    store = Store.open(tmp_path / "repo", "monitor-test")
    branch = store.branch("worker")
    contract = Contract("worker", "write the artifact", outputs=("artifact.txt",),
                        success_criteria=("the artifact contains correct",))
    branch.write(CONTRACT_PATH, contract.to_json())
    branch.write("artifact.txt", "correct")
    branch.checkpoint("prepared")
    directory = store.backend.common_dir / "azure-monitor"
    limits = binding(role="monitor", run_id="worker-run.test.monitor", max_calls=10)
    judge = ResponsesMonitorJudge.create(directory, limits, config())
    yield SimpleNamespace(store=store, branch=branch, contract=contract, directory=directory,
                          limits=limits, judge=judge)
    store.close()


def monitor(worker, judge=None):
    return MonitorBrain(worker.store, worker.contract, judge or worker.judge, batch_size=1)


def observe(worker, value="first"):
    worker.branch.turn(kind="tool_result", content=value)


def reopened(worker):
    return ResponsesMonitorJudge.open(worker.directory, worker.limits, config())


def test_real_sdk_incremental_and_terminal_judgements_use_owned_threads_and_monitor_role(
        worker, sdk_transport, monkeypatch):
    roles = []
    original = LLM.call

    def call(llm, **kwargs):
        roles.append(kwargs["role"])
        assert 0 < kwargs["timeout_seconds"] <= 60
        return original(llm, **kwargs)

    monkeypatch.setattr(LLM, "call", call)

    def handle(wire):
        payload = json.loads(wire.content)
        assert payload["model"] == "gpt-6-sol"
        assert payload["max_output_tokens"] == 128
        assert not any(item.get("role") == "assistant" for item in payload["input"])
        return reply(wire, terminal=TERMINAL_JUDGEMENT_SCHEMA in wire.content.decode())

    sent, _ = sdk_transport(handle)
    observe(worker)
    first = monitor(worker)
    asyncio.run(first.drain(None, final=True))
    state = worker.branch.checkpoint("work finished")
    # A new MonitorBrain and a new judge reuse durable state, despite the
    # different thread and event loop in the next runtime operation.
    second = monitor(worker, reopened(worker))
    assessment = asyncio.run(second.certify_terminal(state, context={"completed": True}))
    assert assessment.acceptable, assessment.failure
    assert second.report()["current_state"] == state.id
    assert roles == ["monitor", "monitor"]
    assert len(sent) == 2
    assert second.report()["model_calls"] == 2
    assert second.report()["cost_usd"] == pytest.approx(0.000732)
    assert second.report()["cost_known"]
    assert reopened(worker).call_accounting() == ModelCallAccounting(0.000732, 2, 0)


@pytest.mark.parametrize("after_commit", [False, True])
def test_lost_incremental_sidecar_recovers_paid_reply_without_redispatch(
        worker, sdk_transport, monkeypatch, after_commit):
    sent, _ = sdk_transport(reply)
    observe(worker)
    first = monitor(worker)
    original = first._save_state

    def fail():
        if after_commit:
            original()
        raise OSError("sidecar write interrupted")

    monkeypatch.setattr(first, "_save_state", fail)
    with pytest.raises(OSError):
        asyncio.run(first.cycle(None, final=True))
    assert first.report()["cost_usd"] == pytest.approx(0.000366)
    resumed = monitor(worker, reopened(worker))
    asyncio.run(resumed.drain(None, final=True))
    assert len(sent) == 1
    assert resumed.report()["judgements"] == 1
    assert resumed.report()["model_calls"] == 1
    assert resumed.report()["cost_usd"] == pytest.approx(0.000366)
    assert not resumed.pending_actions


def test_later_recovery_pins_the_original_elapsed_time_before_paying(worker, sdk_transport, monkeypatch):
    sent, _ = sdk_transport(reply)
    elapsed = ["1s since this worker started"]
    monkeypatch.setattr("taste.brains.monitor_judge._elapsed_since", lambda _started: elapsed[0])
    observe(worker)
    first = monitor(worker)

    def fail():
        raise OSError("crash after provider receipt before monitor sidecar")

    monkeypatch.setattr(first, "_save_state", fail)
    with pytest.raises(OSError):
        asyncio.run(first.cycle(None, final=True))
    elapsed[0] = "10.0m since this worker started"
    resumed = monitor(worker, reopened(worker))
    asyncio.run(resumed.drain(None, final=True))
    assert len(sent) == 1, "the same observation was billed again merely because wall time advanced"
    assert resumed.report()["cost_usd"] == pytest.approx(0.000366)


def test_crash_after_observation_is_pinned_but_before_dispatch_preserves_original_prompt(
        worker, sdk_transport, monkeypatch):
    sent, _ = sdk_transport(reply)
    elapsed = ["1s since this worker started"]
    monkeypatch.setattr("taste.brains.monitor_judge._elapsed_since", lambda _started: elapsed[0])
    observe(worker)
    original = ResponsesSession.pin_context

    def fail(session, *args):
        original(session, *args)
        raise OSError("observation committed before dispatch")

    with monkeypatch.context() as patch:
        patch.setattr(ResponsesSession, "pin_context", fail)
        with pytest.raises(OSError):
            asyncio.run(monitor(worker).cycle(None, final=True))
    assert sent == []
    assert worker.judge.call_accounting() == ModelCallAccounting(0, 0, 0)
    elapsed[0] = "10.0m since this worker started"
    asyncio.run(monitor(worker, reopened(worker)).drain(None, final=True))
    assert len(sent) == 1
    assert b"1s since this worker started" in sent[0].content
    assert b"10.0m since this worker started" not in sent[0].content


@pytest.mark.parametrize("valid_digest", [False, True])
def test_corrupted_pinned_evidence_cannot_be_used_as_a_monitor_observation(
        worker, sdk_transport, monkeypatch, valid_digest):
    sent, _ = sdk_transport(reply)
    observe(worker)
    first = monitor(worker)

    def fail():
        raise OSError("lost sidecar")

    monkeypatch.setattr(first, "_save_state", fail)
    with pytest.raises(OSError):
        asyncio.run(first.cycle(None, final=True))
    with sqlite3.connect(worker.directory / "calls.sqlite3") as db:
        key, value = db.execute("SELECT key,value FROM meta WHERE key LIKE 'context:%'").fetchone()
        envelope = json.loads(value)
        evidence = json.loads(envelope["value"]["payload"])
        evidence["events"] = []
        envelope["value"]["payload"] = json.dumps(evidence)
        if valid_digest:
            canonical = json.dumps(envelope["value"], sort_keys=True, ensure_ascii=True,
                                   allow_nan=False, separators=(",", ":"))
            envelope["digest"] = hashlib.sha256(canonical.encode()).hexdigest()
        db.execute("UPDATE meta SET value=? WHERE key=?", (json.dumps(envelope), key))
    with pytest.raises(ResponsesConflict):
        asyncio.run(monitor(worker, reopened(worker)).cycle(None, final=True))
    assert len(sent) == 1


@pytest.mark.parametrize("after_commit", [False, True])
def test_lost_terminal_sidecar_recovers_exact_assessment_and_cost(
        worker, sdk_transport, monkeypatch, after_commit):
    sent, _ = sdk_transport(lambda wire: reply(wire, terminal=True))
    first = monitor(worker)
    state = worker.branch.head
    original = first._save_state

    def fail():
        if after_commit:
            original()
        raise OSError("terminal assessment not acknowledged")

    monkeypatch.setattr(first, "_save_state", fail)
    with pytest.raises(OSError):
        asyncio.run(first.certify_terminal(state, context={"completed": True}))
    resumed = monitor(worker, reopened(worker))
    assessment = asyncio.run(resumed.certify_terminal(state, context={"completed": True}))
    assert assessment.acceptable, assessment.failure
    assert len(sent) == 1
    assert resumed.report()["model_calls"] == 1
    assert resumed.report()["cost_usd"] == pytest.approx(0.000366)
    with pytest.raises(RuntimeError, match="different context"):
        asyncio.run(resumed.certify_terminal(state, context={"completed": False}))
    assert len(sent) == 1


@pytest.mark.parametrize("bad_reply", ["invalid_json", "continuation", "fenced_json", "incomplete", "tool"])
def test_rejected_verdict_remains_paid_and_replays_without_a_second_charge(worker, sdk_transport, bad_reply):
    def handle(wire):
        if bad_reply == "incomplete":
            return reply(wire, status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        if bad_reply == "tool":
            return httpx.Response(200, json=response(model="gpt-6-sol", output=[function_call()]),
                                  headers={"x-ms-served-model": worker.limits.model})
        text = {"invalid_json": "not a verdict", "continuation": verdict()[1:],
                "fenced_json": "```json\n" + verdict() + "\n```"}[bad_reply]
        return reply(wire, text=text)

    sent, _ = sdk_transport(handle)
    observe(worker)
    first = monitor(worker)
    for current in (first, monitor(worker, reopened(worker))):
        with pytest.raises(MonitorResponseError):
            asyncio.run(current.cycle(None, final=True))
        assert current.report()["judgements"] == 0
        assert current.report()["cost_known"]
        assert current.report()["cost_usd"] == pytest.approx(0.000366)
        assert current.report()["model_calls"] == 1
    assert len(sent) == 1


def test_a_monitor_whose_reply_is_lost_can_give_it_up_and_judge_again(worker, sdk_transport):
    # One unanswered request to the monitor closed its journal, and with it
    # ended the worker's whole run, though the worker had done nothing wrong.
    answers = iter([None])

    def handle(wire):
        if next(answers, "answered") is None:
            raise httpx.ReadTimeout("no answer", request=wire)
        return reply(wire)

    sent, _ = sdk_transport(handle)
    observe(worker)
    first = monitor(worker)
    with pytest.raises(InfraFailure):
        asyncio.run(first.cycle(None, final=True))
    assert first.forfeit_lost_reply()
    assert not first.forfeit_lost_reply(), "nothing is left to give up"
    asyncio.run(first.cycle(None, final=True))
    report = first.report()
    assert report["judgements"] == 1 and report["worst"] == "fine"
    # The lost request was charged; the same judgement was asked as a new
    # call. Its exact cost is not known, and nothing is left open.
    assert len(sent) == 2 and sent[0].content == sent[1].content
    assert not report["cost_known"] and report["known_cost_usd"] == pytest.approx(0.000366)
    assert (report["lost_model_calls"], report["unknown_model_calls"]) == (1, 0)
    assert 0 < report["lost_exposure_usd"] < 0.05
    # After a restart the same observation is answered from the receipt.
    resumed = monitor(worker, reopened(worker))
    asyncio.run(resumed.cycle(None, final=True))
    assert len(sent) == 2 and resumed.report()["judgements"] == 1


def test_a_lost_final_certification_is_not_recorded_as_a_refusal(worker, sdk_transport):
    # The certifier's lost reply was recorded as its verdict on that state: a
    # refusal, which a second asking returned unchanged. The work was then
    # refused for want of a reply, and another worker sent to certify it again.
    lost = []

    def handle(wire):
        terminal = TERMINAL_JUDGEMENT_SCHEMA in wire.content.decode()
        if terminal and not lost:
            lost.append(wire)
            raise httpx.ReadTimeout("no answer", request=wire)
        return reply(wire, terminal=terminal)

    sent, _ = sdk_transport(handle)
    observe(worker)
    brain = monitor(worker)
    asyncio.run(brain.drain(None, final=True))
    state = worker.branch.checkpoint("work finished")
    with pytest.raises(InfraFailure):
        asyncio.run(brain.certify_terminal(state, context={"completed": True}))
    assert brain.report()["terminal_assessment"] is None, "no verdict was recorded for the lost reply"
    assert brain.forfeit_lost_reply()
    assessment = asyncio.run(brain.certify_terminal(state, context={"completed": True}))
    assert assessment.acceptable, assessment.failure
    assert len(sent) == 3 and len(lost) == 1


def test_a_lost_monitor_reply_not_given_up_fences_new_observations(worker, sdk_transport):
    def fail(wire):
        raise httpx.ReadError("private provider failure", request=wire)

    sent, _ = sdk_transport(fail)
    observe(worker)
    first = monitor(worker)
    with pytest.raises(InfraFailure):
        asyncio.run(first.cycle(None, final=True))
    report = first.report()
    assert not report["cost_known"]
    assert report["cost_usd"] is None
    assert report["known_cost_usd"] == 0.0
    assert report["unknown_model_calls"] == report["model_calls"] == 1
    assert 0 < report["unknown_exposure_usd"] < 0.05
    observe(worker, "new evidence")
    resumed = monitor(worker, reopened(worker))
    with pytest.raises(ResponsesFenced):
        asyncio.run(resumed.cycle(None, final=True))
    assert len(sent) == 1


def test_a_judge_without_a_journal_has_nothing_to_give_up(worker):
    class Plain:
        def __call__(self, _contract, _batch, _view):
            raise AssertionError("not judged in this test")

    assert monitor(worker, Plain()).forfeit_lost_reply() is False


def test_monitor_crash_during_dispatch_leaves_durable_unknown_spend(worker, sdk_transport):
    sent, _ = sdk_transport(reply)
    script = '''
import json, os, sys
from pathlib import Path
from taste.brains.responses_monitor import ResponsesMonitorJudge
from taste.brains.responses_session import ResponsesBinding
from taste.llm import LLM
from tests.test_azure_openai import config
LLM.call = lambda *args, **kwargs: os._exit(37)
judge = ResponsesMonitorJudge.open(Path(sys.argv[1]), ResponsesBinding(**json.loads(sys.argv[2])), config())
judge._completion(system="strict monitor", prompt="pinned observation")
'''
    result = subprocess.run([sys.executable, "-c", script, str(worker.directory),
                             json.dumps(asdict(worker.limits))], capture_output=True, text=True, timeout=20)
    assert result.returncode == 37, result.stderr
    second = reopened(worker)
    accounting = second.call_accounting()
    assert (accounting.known_cost_usd, accounting.completed_calls, accounting.unknown_calls) == (0, 0, 1)
    # Charged by the small request it sent; the model's whole window is $5.25.
    assert 0 < accounting.unknown_exposure_usd < 0.05
    with pytest.raises(ResponsesFenced):
        second._completion(system="strict monitor", prompt="new observation")
    assert sent == []


def test_monitor_role_is_immutable_and_cannot_reuse_a_worker_journal(tmp_path, sdk_transport, monkeypatch):
    sent, _ = sdk_transport(reply)
    directory = tmp_path / "old-worker"
    limits = binding()
    first = ResponsesSession.create(directory, limits, config())
    first.close()
    with sqlite3.connect(directory / "calls.sqlite3") as db:
        identity = json.loads(db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
        assert "role" not in identity["binding"]  # existing /1 worker identity is preserved
    with pytest.raises(ResponsesConflict, match="monitor role"):
        ResponsesMonitorJudge.open(directory, limits, config())
    with pytest.raises(ResponsesConflict, match="binding changed"):
        ResponsesMonitorJudge.open(directory, replace(limits, role="monitor"), config())
    assert sent == []


def test_synchronous_judge_cannot_deadlock_its_callers_event_loop(worker, sdk_transport):
    sent, _ = sdk_transport(reply)

    async def scenario():
        with pytest.raises(ResponsesConflict, match="async API"):
            worker.judge._completion(system="strict monitor", prompt="observation")

    asyncio.run(scenario())
    assert sent == []
    assert worker.judge.call_accounting() == ModelCallAccounting(0, 0, 0)
