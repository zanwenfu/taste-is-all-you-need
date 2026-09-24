"""Real Responses SDK, durable brain memory, tool effects and monitor seams."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from taste.brains.contract import CONTRACT_PATH, Contract
from taste.brains.monitor import Judgement, MonitorBrain, Severity, TerminalDecision
from taste.brains.responses_conversation import ResponsesConversation, ResponsesTool, ToolOutcome
from taste.brains.responses_session import ResponsesConflict, ResponsesFenced, ResponsesSession
from taste.memstore import Store
from tests.test_azure_openai import config, httpx
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_openai_responses import function_call, message, response
from tests.test_responses_session import binding
from tests.test_terminal_broker import Environment, broker

sdk_transport = _sdk_transport


@pytest.fixture
def worker(tmp_path):
    store = Store.open(tmp_path / "repo", "conversation")
    branch = store.branch("worker")
    contract = Contract("worker", "produce the artifact", success_criteria=("artifact is correct",))
    branch.write(CONTRACT_PATH, contract.to_json())
    branch.checkpoint("prepared contract")
    session = ResponsesSession.create(store.backend.common_dir / "responses", binding(max_calls=10), config())
    yield SimpleNamespace(store=store, branch=branch, session=session, contract=contract)
    session.close()
    store.close()


def install(sdk_transport, *outputs):
    pending = iter(outputs)

    def handle(_wire):
        return httpx.Response(200, json=response(model="gpt-6-sol", output=next(pending)),
                              headers={"x-ms-served-model": binding().model})

    sent, _ = sdk_transport(handle)
    return sent


def make(worker, execute=None, **overrides):
    async def write(effect_id, call):
        worker.branch.write("artifact.txt", call.arguments["text"])
        return ToolOutcome("artifact saved: " + effect_id)

    def validate(arguments):
        if set(arguments) != {"text"} or not isinstance(arguments["text"], str):
            raise ValueError("write_artifact requires exactly one text argument")

    tools = {"write_artifact": ResponsesTool(
        "Write the fixed artifact", {"type": "object", "properties": {"text": {"type": "string"}},
                                     "required": ["text"], "additionalProperties": False},
        validate, execute or write,
    )}
    return ResponsesConversation(worker.branch, worker.session,
                                 **{"system": "Work precisely.", "tools": tools, **overrides})


def tool(text="correct", identifier="call_one", **overrides):
    return function_call(json.dumps({"text": text}), name="write_artifact", call_id=identifier, **overrides)


def test_native_tool_round_trip_checkpoint_restart_and_monitor_certification(worker, sdk_transport):
    reasoning = {"type": "reasoning", "id": "reason_1", "summary": [], "encrypted_content": "opaque-context"}
    sent = install(sdk_transport, [reasoning, tool()], [message("completed")])
    conversation = make(worker)
    conversation.observe("contract", worker.contract.brief())

    async def first():
        reply = await conversation.step()
        assert reply.stop_reason == "tool_use"

    asyncio.run(first())
    checkpoint = worker.branch.checkpoint("tool results and native model context")
    directory, limits = worker.session.directory, worker.session.binding
    worker.session.close()
    resumed = ResponsesSession.open(directory, limits, config())
    worker.session = resumed
    conversation = make(worker)

    class Judge:
        def __call__(self, _contract, batch, _view):
            assert not any(item["kind"] in {"responses_binding", "responses_request"} for item in batch)
            return Judgement(Severity.FINE, "observed worker behavior", cost_usd=0.0)

        def judge_terminal(self, _contract, state, _context, findings):
            assert state.read("artifact.txt") == "correct"
            return TerminalDecision(Judgement(Severity.FINE, "checked artifact", cost_usd=0.0),
                                    resolved_finding_ids=tuple(item["id"] for item in findings))

    async def finish():
        reply = await conversation.step()
        assert reply.summary_text == "completed"
        assert await conversation.step() == reply  # final answer is replayable, not another paid call
        monitor = MonitorBrain(worker.store, worker.contract, Judge(), batch_size=2)
        await monitor.drain(None, final=True)
        final = worker.branch.checkpoint("complete work with all observations judged")
        assessment = await monitor.certify_terminal(final, context={"artifact": "artifact.txt"})
        assert assessment.acceptable, assessment.failure

    try:
        asyncio.run(finish())
        assert len(sent) == 2
        wire = json.loads(sent[1].content)
        assert [item["encrypted_content"] for item in wire["input"] if item.get("type") == "reasoning"] == ["opaque-context"]
        assert len([item for item in wire["input"] if item.get("type") == "function_call"]) == 1
        results = [item for item in wire["input"] if item.get("type") == "function_call_output"]
        assert len(results) == 1 and results[0]["call_id"] == "call_one"
        assert checkpoint.read("artifact.txt") == "correct"
        assert resumed.known_cost_usd == pytest.approx(0.000732)
    finally:
        resumed.close()


@pytest.mark.parametrize("after_write", [False, True])
def test_lost_memory_reply_is_recovered_without_paying_again(worker, sdk_transport, monkeypatch, after_write):
    sent = install(sdk_transport, [message("saved answer")])
    conversation = make(worker)
    conversation.observe("contract", "produce the artifact")
    original = worker.branch.turn

    def fail(**event):
        if event.get("kind") == "responses_completion":
            if after_write:
                original(**event)
            raise OSError("projection boundary failure")
        return original(**event)

    monkeypatch.setattr(worker.branch, "turn", fail)
    with pytest.raises(OSError):
        asyncio.run(conversation.step())
    monkeypatch.setattr(worker.branch, "turn", original)
    worker.branch.checkpoint("capture interrupted projection")
    resumed = make(worker)

    async def recover():
        assert (await resumed.step()).summary_text == "saved answer"

    # Reopening the session also changes event loops, as a real restart would.
    worker.session.close()
    worker.session = ResponsesSession.open(worker.session.directory, worker.session.binding, config())
    resumed = make(worker)
    try:
        asyncio.run(recover())
        assert len(sent) == 1
    finally:
        worker.session.close()


@pytest.mark.parametrize("failure", ["intent", "effect", "result_before", "result_after"])
def test_tool_crash_windows_never_repeat_a_possibly_applied_effect(worker, sdk_transport, monkeypatch, failure):
    sent = install(sdk_transport, [tool()], [message("finished")])
    effects = []

    async def execute(effect_id, call):
        effects.append(effect_id)
        worker.branch.write("artifact.txt", call.arguments["text"])
        if failure == "effect":
            raise OSError("lost tool outcome")
        return ToolOutcome("saved")

    conversation = make(worker, execute)
    conversation.observe("contract", "produce")
    original = worker.branch.turn

    def fail(**event):
        if failure == "intent" and event.get("kind") == "responses_tool_intent":
            raise OSError("tool intent not saved")
        if failure.startswith("result_") and event.get("kind") == "responses_tool_result":
            if failure == "result_after":
                original(**event)
            raise OSError("tool result persistence failed")
        original(**event)

    monkeypatch.setattr(worker.branch, "turn", fail)
    with pytest.raises(OSError):
        asyncio.run(conversation.step())
    monkeypatch.setattr(worker.branch, "turn", original)
    worker.branch.checkpoint("interrupted tool boundary")
    worker.session.close()
    worker.session = ResponsesSession.open(worker.session.directory, worker.session.binding, config())
    recovered = make(worker, execute)

    async def resume():
        if failure in {"effect", "result_before"}:
            with pytest.raises(ResponsesFenced, match="unknown outcome"):
                await recovered.step()
        else:
            await recovered.step()

    try:
        asyncio.run(resume())
        assert len(effects) == 1
        assert worker.branch.path("artifact.txt").read_text() == "correct"
        assert len(sent) == (2 if failure == "result_after" else 1)
    finally:
        worker.session.close()


def test_multiple_tools_are_serial_and_arguments_are_admitted_before_handler(worker, sdk_transport):
    sent = install(sdk_transport, [tool(identifier="first"), function_call('{}', name="write_artifact", call_id="bad"),
                                  function_call('{}', name="undeclared", call_id="unknown"), tool(identifier="last")])
    effects = []

    async def execute(effect_id, call):
        events = list(worker.branch.view.pending_turns())
        assert events[-1]["kind"] == "responses_tool_intent"
        if effects:
            assert any(event.get("effect_id") == effects[-1][0] and event["kind"] == "responses_tool_result" for event in events)
        effects.append((effect_id, call.id))
        await asyncio.sleep(0)
        return ToolOutcome("ok")

    conversation = make(worker, execute)
    conversation.observe("contract", "produce")
    asyncio.run(conversation.step())
    assert [item[1] for item in effects] == ["first", "last"]
    assert len(set(item[0] for item in effects)) == 2
    results = [item["content"][0] for item in conversation.messages if isinstance(item["content"], list)
               and item["content"][0].get("type") == "tool_result"]
    assert [item["is_error"] for item in results] == [False, True, True, False]
    assert len(sent) == 1


def test_active_response_blocks_competing_drivers_and_feedback(worker, sdk_transport):
    entered, release = threading.Event(), threading.Event()

    def handle(_wire):
        entered.set()
        assert release.wait(10)
        return httpx.Response(200, json=response(model=binding().model))

    sent, _ = sdk_transport(handle)
    first, second = make(worker), make(worker)
    first.observe("contract", "produce")

    async def scenario():
        active = asyncio.create_task(first.step())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            with pytest.raises(ResponsesConflict, match="owns"):
                await second.step()
            with pytest.raises(ResponsesConflict, match="active"):
                second.observe("new", "changed instructions")
            with pytest.raises(ResponsesConflict, match="owns"):
                make(worker)
            release.set()
            await active
        finally:
            release.set()
            await asyncio.gather(active, return_exceptions=True)

    asyncio.run(scenario())
    assert len(sent) == 1


def test_rollback_discards_context_but_keeps_spending_and_new_request_identity(worker, sdk_transport):
    sent = install(sdk_transport, [message("first")], [message("second")])
    conversation = make(worker)
    conversation.observe("contract", "produce")
    before = worker.branch.checkpoint("before thinking")

    async def scenario():
        assert (await conversation.step()).summary_text == "first"
        worker.branch.write("artifact.txt", "discarded")
        worker.branch.checkpoint("first approach")
        worker.branch.rollback(before, "reconsider")
        assert (await conversation.step()).summary_text == "second"

    asyncio.run(scenario())
    assert len(sent) == 2 and worker.session.known_cost_usd == pytest.approx(0.000732)
    assert not worker.branch.path("artifact.txt").exists()
    assert "first" not in json.dumps(json.loads(sent[1].content)["input"])


def test_fenced_or_expired_run_cannot_execute_a_saved_model_tool(worker, sdk_transport, monkeypatch):
    install(sdk_transport, [tool()])
    conversation = make(worker)
    conversation.observe("contract", "produce")
    original = conversation._append

    def expire(kind, **event):
        original(kind, **event)
        if kind == "completion":
            monkeypatch.setattr("taste.brains.responses_conversation.time.time", lambda: worker.session.binding.deadline_unix + 1)

    monkeypatch.setattr(conversation, "_append", expire)
    with pytest.raises(ResponsesFenced, match="deadline"):
        asyncio.run(conversation.step())
    assert not worker.branch.path("artifact.txt").exists()
    assert not any(event["kind"] == "responses_tool_intent" for event in worker.branch.view.pending_turns())


def test_ledger_location_and_context_binding_cannot_be_substituted(worker):
    conversation = make(worker)
    conversation.observe("contract", "produce")
    with pytest.raises(ResponsesConflict, match="binding"):
        make(worker, system="a different task")
    with pytest.raises(ResponsesConflict, match="different content"):
        conversation.observe("contract", "different")
    wrong = ResponsesSession.create(worker.branch.worktree / "journal", replace(worker.session.binding, run_id="wrong"), config())
    try:
        with pytest.raises(ResponsesConflict, match="controller storage"):
            ResponsesConversation(worker.branch, wrong, system="s", tools={})
    finally:
        wrong.close()


def test_terminal_tool_cancellation_retains_both_memory_and_environment_ownership(worker, sdk_transport, tmp_path):
    from taste.brains.terminal_broker import TerminalFenced, TerminalRequest

    sent = install(sdk_transport, [tool()])
    environment = Environment()
    environment.release.clear()
    environment.stop_release.clear()
    terminal = broker(tmp_path, environment)

    async def execute(effect_id, call):
        result = await terminal.execute(TerminalRequest(effect_id, "worker", call.arguments["text"], "/task", 3))
        return ToolOutcome(result.stdout.decode(errors="replace"))

    conversation = make(worker, execute)
    conversation.observe("contract", "produce")

    async def scenario():
        active = asyncio.create_task(conversation.step())
        try:
            assert await asyncio.to_thread(environment.entered.wait, 5)
            active.cancel()
            assert await asyncio.to_thread(environment.stop_entered.wait, 5)
            for _ in range(3):
                active.cancel()
                await asyncio.sleep(0.01)
                assert not active.done()
                with pytest.raises(TerminalFenced):
                    terminal.close()
                with pytest.raises(ResponsesConflict, match="owns"):
                    make(worker, execute)
            environment.stop_release.set()
            with pytest.raises(asyncio.CancelledError):
                await active
            assert terminal.phase == "stopped"
            with pytest.raises(ResponsesFenced, match="unknown outcome"):
                await make(worker, execute).step()
            assert len(environment.calls) == environment.stop_calls == 1
        finally:
            environment.release.set()
            environment.stop_release.set()
            await asyncio.gather(active, return_exceptions=True)
            await terminal.abort()

    try:
        asyncio.run(scenario())
        assert len(sent) == 1
    finally:
        terminal.close()


def test_process_death_after_tool_effect_cannot_become_an_automatic_retry(tmp_path, sdk_transport):
    root, marker = tmp_path / "death-repo", tmp_path / "external-effect"
    limits = binding(max_calls=10)
    script = r'''
import asyncio, json, os, sys, httpx
from pathlib import Path
from types import SimpleNamespace
from taste.brains.responses_session import ResponsesBinding, ResponsesSession
from taste.memstore import Store
from tests.test_azure_openai import config
from tests.test_openai_responses import response
from tests.test_responses_conversation import make, tool
limits=ResponsesBinding(**json.loads(sys.argv[3]))
original=httpx.Client
class Client(original):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response(model=limits.model, output=[tool()]))))
httpx.Client=Client
store=Store.open(Path(sys.argv[1]), 'death')
branch=store.branch('worker')
session=ResponsesSession.create(store.backend.common_dir/'responses', limits, config())
worker=SimpleNamespace(store=store, branch=branch, session=session)
async def die(effect_id, call):
    with open(sys.argv[2], 'ab', buffering=0) as target:
        target.write(b'one persistent effect\n')
        os.fsync(target.fileno())
    os._exit(46)
conversation=make(worker, die)
conversation.observe('contract', 'produce')
asyncio.run(conversation.step())
'''
    child = subprocess.run([sys.executable, "-c", script, str(root), str(marker), json.dumps(asdict(limits))],
                           capture_output=True, text=True, timeout=20)
    assert child.returncode == 46, child.stderr
    sent = install(sdk_transport, [message("must never be requested")])
    store = Store.open(root, "death")
    session = ResponsesSession.open(store.backend.common_dir / "responses", limits, config())

    async def forbidden(*_args):
        raise AssertionError("must never repeat a possibly applied tool")

    try:
        recovered = make(SimpleNamespace(store=store, branch=store.branch("worker"), session=session), forbidden)
        with pytest.raises(ResponsesFenced, match="unknown outcome"):
            asyncio.run(recovered.step())
        assert sent == []
        assert marker.read_bytes() == b"one persistent effect\n"
        assert session.known_cost_usd == pytest.approx(0.000366)
    finally:
        session.close()
        store.close()
