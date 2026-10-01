"""Real Responses receipts -> authenticated RPC -> terminal receipt -> model pages."""

from __future__ import annotations

import asyncio
import json

import pytest

from taste.brains import terminal_tools
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


def test_result_reads_like_a_shell_and_long_output_keeps_both_ends(rig):
    async def scenario():
        async with rig() as r:
            body = "".join(f"line {number}\n" for number in range(4000)).encode()
            r.env.result = TerminalResult(7, body, b"warning\n", 8_000_000, 0)
            tools = TerminalTools(r.a)
            result = await tools.execute("effect_one", call())
            text = result.content
            assert result.is_error and text.startswith("exit 7\nline 0\nline 1\n")
            # The end of long output is what a test run's verdict lives in.
            assert "line 3999\n[stderr]\nwarning\n" in text
            cut = f"... [cut: {len(body) - (terminal_tools.SHOWN_CHARS - len('warning' + chr(10))):,} more characters of stdout;"
            assert cut in text and 'request_id "effect_one"' in text
            assert "[cut: 8,000,000 more bytes of stdout were beyond what is kept" in text
            assert len(text) < terminal_tools.SHOWN_CHARS + 600
            # The cut part is read by page, without running anything again.
            page = await tools.read("page_effect", ToolCall("read", "read_terminal_output", {
                "request_id": "effect_one", "stream": "stdout", "offset": 14, "limit": 14}))
            assert page.content == (f"stdout bytes 14-28 of {len(body):,} kept for request effect_one; "
                                    "continue from offset 28\nline 2\nline 3\n")
            last = await tools.read("page_effect", ToolCall("read", "read_terminal_output", {
                "request_id": "effect_one", "stream": "stdout", "offset": len(body) - 10}))
            assert last.content.endswith("line 3999\n") and "8,000,000 more bytes were beyond" in last.content
            assert len(r.env.calls) == 1
            assert r.credentials[0].token not in text + page.content + tools.instructions()
            assert r.credentials[0].socket_path not in tools.instructions()
    asyncio.run(scenario())


def test_short_output_is_exact_and_undecodable_bytes_are_marked(rig):
    async def scenario():
        async with rig() as r:
            r.env.result = TerminalResult(0, "héllo 🌍\n".encode() + b"\xff\n", b"")
            result = await TerminalTools(r.a).execute("effect_one", call())
            assert result.content == "exit 0\nhéllo 🌍\n\ufffd\n" and not result.is_error
    asyncio.run(scenario())


@pytest.mark.parametrize("cause,opening", [
    ("timeout", "timed out after 3s: the command and its child processes were killed (exit 137). Output before that:\n"),
    ("cancelled", "cancelled: the command and its child processes were killed (exit 137).\n"),
])
def test_ended_command_says_so_before_its_partial_output(rig, cause, opening):
    async def scenario():
        async with rig() as r:
            r.env.result = TerminalResult(137, b"compiling...\n", b"", terminated=cause)
            result = await TerminalTools(r.a).execute("effect_one", call())
            assert result.is_error and result.content == opening + "compiling...\n"
    asyncio.run(scenario())


def test_worst_case_control_characters_stay_inside_tool_outcome_limit(rig):
    async def scenario():
        async with rig() as r:
            r.env.result = TerminalResult(0, "🌍".encode() * 200_000, b"\xff" * 1048576, 22, 33)
            result = await TerminalTools(r.a).execute("effect_one", call())
            assert len(result.content.encode()) < 65536
    asyncio.run(scenario())


def test_working_directory_and_timeout_have_task_defaults(rig):
    async def scenario():
        async with rig() as r:
            tools = TerminalTools(r.a, workdir="/workspace/app")
            await tools.execute("effect_one", ToolCall("c1", "terminal_exec", {"command": "ls"}))
            await tools.execute("effect_two", ToolCall("c2", "terminal_exec", {"command": "ls", "cwd": "src/../lib"}))
            await tools.execute("effect_three", ToolCall("c3", "terminal_exec",
                                                        {"command": "ls", "cwd": "/etc", "timeout_seconds": 2}))
            assert [(item.cwd, item.timeout_seconds) for item in r.env.calls] == [
                ("/workspace/app", 5), ("/workspace/app/lib", 5), ("/etc", 2)]
            assert "cwd defaults to /workspace/app" in tools.instructions()
            assert tools.tools()["terminal_exec"].input_schema["required"] == ["command"]
    asyncio.run(scenario())


@pytest.mark.parametrize("changes", [{"cwd": ""}, {"cwd": 7}, {"command": "nul\0"}, {"command": ""},
                                     {"timeout_seconds": True}, {"timeout_seconds": 6},
                                     {"timeout_seconds": 0}, {"shell": "bash"}])
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
            assert events[0]["result"]["content"] == "exit 0\nstdout\x00\ufffd\n[stderr]\nstderr\n"
            assert await r.a.lookup(effect_id) == r.env.result
            await conversation.step()
            assert len(sent) == 2 and len(r.env.calls) == 1
            wire = json.loads(sent[1].content)
            results = [item for item in wire["input"] if item.get("type") == "function_call_output"]
            assert results[0]["output"] == events[0]["result"]["content"]
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
