"""SWE-agent runs unchanged on Taste: its loop, environment class, tools and templates, with Taste's model
and shell. The session wrapper is run in a real bash; the rest against a scripted host."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import subprocess

import pytest
from swerex.exceptions import BashIncorrectSyntaxError, CommandTimeoutError, NonZeroExitCodeError
from swerex.runtime.abstract import (
    BashAction,
    Command,
    CreateBashSessionRequest,
    ReadFileRequest,
    UploadRequest,
    WriteFileRequest,
)

from taste.agents import HOSTED_AGENTS, AgentExit, HostedStop, ModelReply, ShellResult, hosted_agent
from taste.agents.protocols import SWE_AGENT_STATE_COMMANDS, protocol
from taste.agents.swe_agent import (
    FILES,
    PS1,
    SYNTAX_ERROR,
    SYNTAX_EXIT,
    SweAgent,
    TasteRuntime,
    request_items,
    responses_tool,
    session_command,
    start_command,
)

PATCH = "diff --git a/parser.py b/parser.py\n-x\n+y\n"
MARKER = "<<SWE_AGENT_SUBMISSION>>"


def reply(name, arguments, *, text="", cost=0.01):
    output = []
    if text:
        output.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
    output.append({"type": "function_call", "call_id": f"call_{name}", "name": name,
                   "arguments": json.dumps(arguments)})
    return ModelReply(tuple(output), "completed", None, {"input_tokens": 100, "output_tokens": 20,
                                                         "cache_read_tokens": 50}, "gpt-test", cost)


def answer(shown):
    """The task container, as far as SWE-agent's setup and two steps look at it."""
    if shown == "ls":
        return ShellResult("app\nbin\nroot\n", 0)
    if shown == "pwd":
        return ShellResult("/app\n", 0)
    if shown == "echo $PATH":
        return ShellResult("/root/tools/submit/bin:/root/tools/edit_anthropic/bin:/usr/bin\n", 0)
    if shown.endswith("read_file /root/state.json"):
        return ShellResult('{"working_dir": "/app"}', 0)
    if shown.endswith("read_file /root/model.patch"):
        return ShellResult(PATCH, 0)
    if shown.endswith("read_file /app/parser.py"):
        return ShellResult("y\n", 0)
    if shown == "submit":
        return ShellResult(f"{MARKER}\n{PATCH}{MARKER}\n", 0)
    if shown.startswith("str_replace_editor view"):
        return ShellResult("\x1b[1mHere's the result of running `cat -n` on /app/parser.py:\x1b[0m\n     1\tx\n\n", 0)
    return ShellResult("", 0)


class ScriptedHost:
    def __init__(self, replies=()):
        self.replies = list(replies)
        self.asked, self.ran = [], []

    def ask(self, *, messages, tools, effort):
        self.asked.append({"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools), "effort": effort})
        return self.replies.pop(0)

    def run(self, command, *, cwd, timeout_seconds, shown=None):
        self.ran.append({"command": command, "cwd": cwd, "timeout": timeout_seconds, "shown": shown})
        return answer(shown)


def runtime_of(host, **options):
    return TasteRuntime(host, **options)


def call(coroutine):
    return asyncio.run(coroutine)


# -- the shell session, run in a real bash ------------------------------------------------------


def shell(command):
    done = subprocess.run(["/bin/sh", "-c", command], capture_output=True, text=True, timeout=30)
    return done.stdout + done.stderr, done.returncode


def test_the_session_carries_variables_functions_aliases_and_directory_from_one_command_to_the_next(tmp_path):
    directory = str(tmp_path / "session")
    shell(start_command([f"export PS1='{PS1}'", "export PS2=''", "export PS0=''"], directory=directory))
    output, code = shell(session_command(
        "cd /tmp && export A=1 && B=2 && f() { echo fn; } && alias hi='echo hello'", directory=directory))
    assert code == 0, output
    output, code = shell(session_command("pwd; echo $A $B; f; hi; echo $PS1", directory=directory))
    assert (output, code) == ("/tmp\n1 2\nfn\nhello\nSHELLPS1PREFIX\n", 0)


def test_a_session_command_starts_from_an_empty_environment_and_sees_nothing_of_the_wrapper(tmp_path):
    directory = str(tmp_path / "session")
    shell(start_command([], directory=directory))
    output, _ = shell(session_command("env | sort | cut -d= -f1 | tr '\\n' ' '; compgen -v | grep -c '^__taste'",
                                      directory=directory))
    names, count = output.rsplit(" ", 1)
    assert "HOME" not in names.split() and "PATH" not in names.split() and count.strip() == "0"


def test_a_commands_exit_code_is_its_own_and_its_heredoc_survives(tmp_path):
    directory = str(tmp_path / "session")
    shell(start_command([], directory=directory))
    assert shell(session_command("grep -q nothing /dev/null", directory=directory)) == ("", 1)
    heredoc = "cat <<'EOF' > " + str(tmp_path / "out.txt") + "\nline one\nEOF\ncat " + str(tmp_path / "out.txt")
    assert shell(session_command(heredoc, directory=directory)) == ("line one\n", 0)


def test_a_command_with_a_syntax_error_does_not_run(tmp_path):
    directory = str(tmp_path / "session")
    shell(start_command([], directory=directory))
    output, code = shell(session_command(f"touch {tmp_path}/ran; if then fi", directory=directory))
    assert code == SYNTAX_EXIT and output.startswith(SYNTAX_ERROR + "\n") and "syntax error" in output
    assert not (tmp_path / "ran").exists()


def test_a_command_too_long_for_one_terminal_command_is_written_in_parts_then_run(tmp_path):
    directory = str(tmp_path / "session")
    shell(start_command([], directory=directory))
    text = "x" * 70_000
    command = f"printf '%s' '{text}' | wc -c"

    class Real(ScriptedHost):
        def run(self, command, **options):
            super().run(command, **options)
            output, code = shell(command)
            return ShellResult(output, code)

    host = Real()
    observed = call(runtime_of(host, session_dir=directory).run_in_session(
        BashAction(command=command, timeout=30, check="silent")))
    assert (observed.output, observed.exit_code) == ("70000", 0)
    assert all(len(item["command"].encode()) <= 60_000 for item in host.ran)
    assert [item["shown"].startswith("# swe-rex: ") for item in host.ran] == [True] * (len(host.ran) - 1) + [False]
    assert host.ran[-1]["shown"] == command and len(host.ran) >= 3


def test_the_same_command_is_the_same_text_wherever_it_runs():
    assert session_command("ls -la") == session_command("ls -la")
    encoded = base64.b64encode(b"ls -la").decode()
    assert encoded in session_command("ls -la") and session_command("ls -la").startswith("exec env -i bash -c ")


# -- the runtime ----------------------------------------------------------------------------------


def test_a_session_command_returns_its_output_stripped_of_control_characters_and_its_exit_code():
    host = ScriptedHost()
    observed = call(runtime_of(host).run_in_session(BashAction(command="str_replace_editor view /app/x.py",
                                                               timeout=450, check="silent")))
    assert observed.output.startswith("Here's the result") and observed.output.endswith("1\tx")
    assert observed.exit_code == 0
    assert host.ran[0]["shown"] == "str_replace_editor view /app/x.py" and host.ran[0]["cwd"] == "/"
    assert host.ran[0]["command"] == session_command("str_replace_editor view /app/x.py")


def test_a_timed_out_command_and_a_syntax_error_raise_as_swe_rex_raises_them():
    class Host(ScriptedHost):
        def run(self, command, **options):
            super().run(command, **options)
            if options["shown"] == "sleep 999":
                return ShellResult("partial", -1, timed_out=True)
            return ShellResult(f"{SYNTAX_ERROR}\nbash: line 1: syntax error near unexpected token `fi'\n",
                               SYNTAX_EXIT)

    runtime = runtime_of(Host())
    with pytest.raises(CommandTimeoutError):
        call(runtime.run_in_session(BashAction(command="sleep 999", timeout=5, check="silent")))
    with pytest.raises(BashIncorrectSyntaxError) as raised:
        call(runtime.run_in_session(BashAction(command="if then fi", timeout=5, check="silent")))
    assert "syntax error" in raised.value.extra_info["bash_stderr"]


def test_a_failing_command_raises_only_when_asked_to():
    class Fails(ScriptedHost):
        def run(self, command, **options):
            super().run(command, **options)
            return ShellResult("no", 2)

    runtime = runtime_of(Fails())
    assert call(runtime.run_in_session(BashAction(command="false", check="silent"))).exit_code == 2
    assert call(runtime.run_in_session(BashAction(command="false", check="ignore"))).exit_code is None
    with pytest.raises(NonZeroExitCodeError):
        call(runtime.run_in_session(BashAction(command="false", check="raise")))
    with pytest.raises(NonZeroExitCodeError):
        call(runtime.execute(Command(command="which nothing", shell=True, check=True)))


def test_files_are_read_written_and_uploaded_with_commands_shown_as_the_runtimes():
    class Files(ScriptedHost):
        def run(self, command, **options):
            super().run(command, **options)
            return ShellResult("", 66) if options["shown"].endswith("/missing") else answer(options["shown"])

    host = Files()
    runtime = runtime_of(host)
    assert call(runtime.read_file(ReadFileRequest(path="/root/state.json"))).content == '{"working_dir": "/app"}'
    with pytest.raises(FileNotFoundError):
        call(runtime.read_file(ReadFileRequest(path="/root/missing")))
    call(runtime.write_file(WriteFileRequest(path="/root/state.json", content="{}")))
    assert base64.b64encode(b"{}").decode() in host.ran[-1]["command"]
    call(runtime.upload(UploadRequest(source_path=str(FILES / "tools" / "submit"), target_path="/root/tools/submit")))
    upload = host.ran[-1]
    assert upload["shown"] == "# swe-rex: upload tools/submit -> /root/tools/submit"
    # Only the files' contents and names: the same text on every machine.
    assert str(FILES) not in upload["command"]
    assert base64.b64encode((FILES / "tools/submit/bin/submit").read_bytes()).decode() in upload["command"]
    call(runtime.execute(Command(command="which submit", shell=True, check=True, env={"PATH": "/x:/y"})))
    assert host.ran[-1]["command"].startswith("env -i PATH=/x:/y /bin/sh -c 'which submit'")


def test_a_stopped_host_makes_the_runtime_dead_so_nothing_more_runs():
    class Stops(ScriptedHost):
        def run(self, command, **options):
            raise HostedStop("deadline")

    runtime = runtime_of(Stops())
    assert call(runtime.is_alive()).is_alive
    with pytest.raises(HostedStop):
        call(runtime.run_in_session(BashAction(command="ls", check="silent")))
    assert not call(runtime.is_alive()).is_alive


def test_the_session_starts_with_swe_rexs_startup_commands():
    host = ScriptedHost()
    call(runtime_of(host).create_session(CreateBashSessionRequest(startup_source=["/root/.bashrc"])))
    started = host.ran[0]
    assert started["shown"] == ("# swe-rex: start the shell session: source /root/.bashrc ; "
                                f"export PS1='{PS1}' ; export PS2='' ; export PS0=''")
    assert started["command"].startswith(f"exec env -i PS1={PS1} PS2= PS0= HISTFILE=/dev/null bash -i -c ")


# -- the model ------------------------------------------------------------------------------------


def test_upstreams_chat_messages_become_responses_items():
    messages = [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "Fix it."},
                {"role": "assistant", "content": "(empty)", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]},
                {"role": "tool", "content": "OBSERVATION:\na.py", "tool_call_id": "c1"}]
    assert request_items(messages) == [
        {"role": "system", "content": "Be helpful."}, {"role": "user", "content": "Fix it."},
        {"role": "assistant", "content": "(empty)"},
        {"type": "function_call", "call_id": "c1", "name": "bash", "arguments": '{"command": "ls"}'},
        {"type": "function_call_output", "call_id": "c1", "output": "OBSERVATION:\na.py"}]


def test_a_chat_tool_is_sent_as_a_non_strict_responses_tool():
    tool = {"type": "function", "function": {"name": "submit", "description": "submits", "parameters": {
        "type": "object", "properties": {}, "required": []}}}
    assert responses_tool(tool) == {"type": "function", "name": "submit", "description": "submits",
                                    "parameters": {"type": "object", "properties": {}, "required": []},
                                    "strict": False}


# -- the agent ------------------------------------------------------------------------------------


def run_agent(host, effort="medium"):
    return SweAgent(effort=effort).run("Make the parser keep tabs.", host, cwd="/app", model_name="azure/gpt-test")


def test_swe_agents_own_loop_sets_up_works_and_submits_through_the_host():
    host = ScriptedHost([reply("str_replace_editor", {"command": "view", "path": "/app/parser.py"}, text="Look."),
                         reply("submit", {})])
    result = run_agent(host)
    assert isinstance(result, AgentExit)
    assert (result.exit_status, result.submission, result.model_calls) == ("Submitted", PATCH, 2)
    shown = [item["shown"] for item in host.ran]
    first_ask = shown.index("str_replace_editor view /app/parser.py")
    # The setup: the session, the repository, the tools; all before the first model call.
    assert shown[0].startswith("# swe-rex: start the shell session")
    assert "cd /app && export ROOT=$(pwd -P)" in shown[:first_ask]
    assert [item for item in shown[:first_ask] if item.startswith("# swe-rex: upload")] == [
        "# swe-rex: upload tools/registry -> /root/tools/registry",
        "# swe-rex: upload tools/edit_anthropic -> /root/tools/edit_anthropic",
        "# swe-rex: upload tools/submit -> /root/tools/submit"]
    # Each step: the action, then the state command and the state file.
    assert shown[first_ask:first_ask + 3] == ["str_replace_editor view /app/parser.py", "_state_anthropic",
                                              "# swe-rex: read_file /root/state.json"]
    first, second = host.asked
    assert first["effort"] == "medium" and [tool["name"] for tool in first["tools"]] == [
        "bash", "str_replace_editor", "submit"]
    assert all(tool["strict"] is False for tool in first["tools"])
    assert first["messages"][0] == {"role": "system",
                                    "content": "You are a helpful assistant that can interact with a computer to solve tasks."}
    assert first["messages"][1]["content"].startswith("<uploaded_files>\n/app\n</uploaded_files>")
    assert second["messages"][2] == {"role": "assistant", "content": "Look."}
    native = [message["content"][0]["item"] for message in second["messages"][3:]]
    assert native[0]["type"] == "function_call" and native[0]["name"] == "str_replace_editor"
    assert native[1] == {"type": "function_call_output", "call_id": "call_str_replace_editor",
                         "output": "OBSERVATION:\nHere's the result of running `cat -n` on /app/parser.py:\n     1\tx"}


def test_the_record_of_a_run_follows_the_protocol_that_replays_it():
    host = ScriptedHost([reply("submit", {})])
    run_agent(host)
    shown = [item["shown"] for item in host.ran]
    step = shown[shown.index("submit"):]

    class Run:
        def __init__(self, shown):
            self.command, self.output, self.returncode = shown, answer(shown).output, 0

    runs = [Run(item) for item in step]
    assert protocol("swe-agent").submission_index(runs) == 0
    assert protocol("swe-agent").submission_text(runs[0].output) == PATCH
    assert set(SweAgent(effort="low").agent_config("m").tools.state_commands) == SWE_AGENT_STATE_COMMANDS


def test_a_stop_swe_agent_catches_ends_the_run_and_nothing_runs_after_it():
    class Stops(ScriptedHost):
        def ask(self, **request):
            if self.asked:
                raise HostedStop("budget")
            return super().ask(**request)

    host = Stops([reply("str_replace_editor", {"command": "view", "path": "/app/parser.py"})])
    with pytest.raises(HostedStop, match="budget"):
        run_agent(host)
    assert host.ran[-1]["shown"] == "# swe-rex: read_file /root/state.json"


def test_upstreams_cost_limit_ends_the_run_as_upstream_ends_it():
    host = ScriptedHost([reply("str_replace_editor", {"command": "view", "path": "/app/parser.py"}, cost=3.5)])
    result = run_agent(host)
    # Upstream submits what the repository holds after an error: not the agent's own submission.
    assert result.exit_status == "submitted (exit_cost)" and result.submission == PATCH and result.model_calls == 1


def test_the_agent_is_named_by_version_configuration_and_digest():
    assert HOSTED_AGENTS["swe-agent"] == ("taste.agents.swe_agent", "SweAgent")
    agent = hosted_agent("swe-agent", effort="low")
    identity = agent.identity()
    assert identity["name"] == "swe-agent" and identity["version"] == "1.1.0" and identity["config"] == "tool_use"
    assert identity["commit"] == "1ac158175178c9c40ed903fb59cc7aa3c87c75ae" and len(identity["config_sha256"]) == 64
    assert identity == SweAgent(effort="high").identity()


def test_the_bundles_registry_library_is_there_for_the_editor_to_import():
    assert (FILES / "tools/registry/lib/registry.py").is_file()
    assert "from registry import registry" in (FILES / "tools/edit_anthropic/bin/str_replace_editor").read_text()
