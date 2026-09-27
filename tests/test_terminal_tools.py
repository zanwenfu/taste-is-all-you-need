"""Real Responses receipts -> authenticated RPC -> terminal receipt -> model pages."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from taste.brains.responses_conversation import ResponsesConversation
from taste.brains.responses_session import ResponsesFenced
from taste.brains.terminal_broker import TerminalResult
from taste.brains.terminal_service import TerminalUnavailable
from taste.brains.terminal_tools import TerminalTools
from taste.providers.base import ToolCall
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_openai_responses import function_call, message
from tests.test_responses_conversation import install
from tests.test_responses_conversation import worker as _worker
from tests.test_terminal_service import rig as _rig

sdk_transport = _sdk_transport
worker = _worker
rig = _rig


def call(**changes):
    return ToolCall("provider_call", "terminal_exec", {"command": "do task", "cwd": "/tmp", "timeout_seconds": 3, **changes})


def test_pages_preserve_partial_utf8_and_truncation_without_another_command(rig):
    async def scenario():
        async with rig() as r:
            r.env.result = TerminalResult(7, b"a" * 4095 + "🌍".encode() + b"tail", b"\xff\0", 8_000_000, 0)
            tools = TerminalTools(r.a)
            result = await tools.execute("effect_one", call())
            first = json.loads(result.content)
            assert result.is_error and first["request_id"] == "effect_one"
            assert first["stdout"]["encoding"] == "base64"
            assert base64.b64decode(first["stdout"]["content"]) == r.env.result.stdout[:4096]
            assert first["stdout"]["dropped_bytes"] == 8_000_000 and not first["stdout"]["captured_eof"]
            page = await tools.read("page_effect", ToolCall("read", "read_terminal_output", {
                "request_id": "effect_one", "stream": "stdout", "offset": 4095, "limit": 8}))
            assert json.loads(page.content)["content"] == "🌍tail"
            assert json.loads(page.content)["captured_eof"] and len(r.env.calls) == 1
            assert r.credentials[0].token not in result.content + page.content + tools.instructions()
            assert r.credentials[0].socket_path not in tools.instructions()
    asyncio.run(scenario())


def test_worst_case_control_characters_stay_inside_tool_outcome_limit(rig):
    async def scenario():
        async with rig() as r:
            r.env.result = TerminalResult(0, b"\0" * 1048576, b"\1" * 1048576, 22, 33)
            result = await TerminalTools(r.a).execute("effect_one", call())
            assert len(result.content.encode()) < 65536
            assert json.loads(result.content)["stdout"]["captured_bytes"] == 1048576
    asyncio.run(scenario())


@pytest.mark.parametrize("changes", [{"cwd": "relative"}, {"command": "nul\0"},
                                     {"timeout_seconds": True}, {"timeout_seconds": 6}])
def test_bad_terminal_arguments_are_refused_before_rpc(rig, changes):
    async def scenario():
        async with rig() as r:
            with pytest.raises(ValueError):
                await TerminalTools(r.a).execute("effect_one", call(**changes))
            assert not r.env.calls
    asyncio.run(scenario())


def test_native_responses_turn_carries_durable_terminal_evidence_without_credentials(worker, rig, sdk_transport):
    sent = install(sdk_transport,
        [function_call(json.dumps(call().arguments), name="terminal_exec", call_id="terminal_call")],
        [message("completed")])

    async def scenario():
        async with rig() as r:
            tools = TerminalTools(r.a)
            conversation = ResponsesConversation(worker.branch, worker.session, system=tools.instructions(), tools=tools.tools())
            conversation.observe("task", "run the admitted terminal command")
            before = worker.branch.checkpoint("before external effect")
            await conversation.step()
            first = worker.branch.checkpoint("terminal receipt in worker memory")
            events = [item for item in first.transcript.turns if item.get("kind") == "responses_tool_result"]
            assert len(events) == 1
            effect_id = events[0]["effect_id"]
            assert json.loads(events[0]["result"]["content"])["request_id"] == effect_id
            assert await r.a.lookup(effect_id) == r.env.result
            await conversation.step()
            assert len(sent) == 2 and len(r.env.calls) == 1
            wire = json.loads(sent[1].content)
            results = [item for item in wire["input"] if item.get("type") == "function_call_output"]
            assert json.loads(results[0]["output"])["request_id"] == effect_id
            assert r.credentials[0].token not in json.dumps(wire)
            worker.branch.rollback(before, reason="reconsider reasoning only")
            assert await r.a.lookup(effect_id) == r.env.result
            assert await tools.execute(effect_id, call())
            assert len(r.env.calls) == 1
    asyncio.run(scenario())


def test_unknown_terminal_outcome_keeps_pending_responses_intent_and_cannot_replay(worker, rig, sdk_transport):
    sent = install(sdk_transport, [function_call(json.dumps(call().arguments), name="terminal_exec", call_id="terminal_call")])

    async def scenario():
        async with rig() as r:
            r.env.error = ConnectionError("lost terminal acknowledgement")
            tools = TerminalTools(r.a)
            conversation = ResponsesConversation(worker.branch, worker.session, system=tools.instructions(), tools=tools.tools())
            conversation.observe("task", "run the admitted terminal command")
            with pytest.raises(TerminalUnavailable):
                await conversation.step()
            worker.branch.checkpoint("retain unknown tool intent")
            reopened = ResponsesConversation(worker.branch, worker.session, system=tools.instructions(), tools=tools.tools())
            with pytest.raises(ResponsesFenced, match="unknown outcome"):
                await reopened.step()
            assert len(sent) == len(r.env.calls) == 1 and r.env.stopped
    asyncio.run(scenario())
