"""mini-swe-agent runs unchanged on Taste: its loop, prompts and parsing, with Taste's model and shell.

The comparison test runs mini-swe-agent's own loop twice over the same model
replies: once with its own Responses model class (LiteLLM replaced by a stub
at its import), once with Taste's. The requests reaching the provider and
every message the agent keeps must be the same.
"""

from __future__ import annotations

import copy
import importlib
import json
import platform
import subprocess
import sys
import types

import pytest
from minisweagent.exceptions import FormatError, Submitted
from minisweagent.models.utils.actions_toolcall_response import BASH_TOOL_RESPONSE_API

from taste.agents import AgentExit, HostedStop, ModelReply, ShellResult
from taste.agents.mini_swe_agent import (
    MiniSweAgent,
    TasteEnvironment,
    TasteResponsesModel,
    canonical_messages,
)
from taste.providers._openai import OpenAIProvider, _to_tool

SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def reply(*items, status="completed", reason=None, cost=0.01):
    return ModelReply(output=tuple(items), status=status, incomplete_reason=reason,
                      usage={"input_tokens": 10, "output_tokens": 5}, model="gpt-5.6-luna-2026-07-09",
                      cost_usd=cost)


def call(command, call_id):
    return {"type": "function_call", "id": "fc_" + call_id, "call_id": call_id, "name": "bash",
            "arguments": json.dumps({"command": command}), "status": "completed"}


def text(words):
    return {"type": "message", "id": "msg_" + str(abs(hash(words))), "role": "assistant",
            "status": "completed", "content": [{"type": "output_text", "text": words, "annotations": []}]}


def thought(number):
    return {"type": "reasoning", "id": f"rs_{number}", "encrypted_content": f"opaque-{number}",
            "summary": []}


class ScriptedHost:
    def __init__(self, replies=(), outputs=None):
        self.replies = list(replies)
        self.outputs = dict(outputs or {})
        self.asked, self.ran, self.shown = [], [], []

    def ask(self, *, messages, tools, effort):
        self.asked.append({"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools),
                           "effort": effort})
        return self.replies.pop(0)

    def run(self, command, *, cwd, timeout_seconds, shown=None):
        self.ran.append((command, cwd, timeout_seconds))
        self.shown.append(shown)
        for key, result in self.outputs.items():
            if key in command:
                return result
        return ShellResult("", 0)


def environment(host, **config):
    return TasteEnvironment(host=host, cwd="/app", env={"PAGER": "cat", "LESS": "-R"}, **config)


# -- the environment: upstream's LocalEnvironment, with the command sent to the task's container


def test_a_command_runs_in_a_fresh_shell_with_errors_folded_into_its_output():
    host = ScriptedHost(outputs={"make": ShellResult("building\nerror: no rule\n", 2)})
    assert environment(host).execute({"command": "make"}) == {
        "output": "building\nerror: no rule\n", "returncode": 2, "exception_info": ""}
    assert host.ran == [("export PAGER=cat LESS=-R; {\nmake\n} 2>&1", "/app", 30)]
    assert host.shown == ["make"]


def test_a_heredoc_survives_the_wrapping():
    host = ScriptedHost()
    environment(host).execute({"command": "cat <<'EOF' > a.py\nprint(1)\nEOF"})
    assert host.ran[0][0].endswith("{\ncat <<'EOF' > a.py\nprint(1)\nEOF\n} 2>&1")


def test_upstreams_time_limit_applies_unless_a_call_names_its_own():
    host = ScriptedHost()
    shell = environment(host)
    shell.execute({"command": "true"})
    shell.execute({"command": "true"}, timeout=7)
    environment(host, timeout=5).execute({"command": "true"})
    shell.execute({"command": "pwd"}, cwd="/tmp")
    assert [(cwd, limit) for _, cwd, limit in host.ran] == [("/app", 30), ("/app", 7), ("/app", 5), ("/tmp", 30)]


def test_a_timeout_reads_as_it_does_under_upstreams_local_environment():
    host = ScriptedHost(outputs={"sleep": ShellResult("partial\n", -9, timed_out=True)})
    out = environment(host).execute({"command": "sleep 99"})
    expected = subprocess.TimeoutExpired("sleep 99", 30)
    assert out == {"output": "partial\n", "returncode": -1,
                   "exception_info": f"An error occurred while executing the command: {expected}",
                   "extra": {"exception_type": "TimeoutExpired", "exception": str(expected)}}


def test_the_submission_sentinel_ends_the_run_as_upstream_does():
    host = ScriptedHost(outputs={SENTINEL: ShellResult(f"{SENTINEL}\nall done\n", 0)})
    with pytest.raises(Submitted) as raised:
        environment(host).execute({"command": f"echo {SENTINEL}"})
    assert raised.value.messages[0] == {"role": "exit", "content": "all done\n",
                                        "extra": {"exit_status": "Submitted", "submission": "all done\n"}}
    failed = ScriptedHost(outputs={SENTINEL: ShellResult(f"{SENTINEL}\n", 1)})
    assert environment(failed).execute({"command": f"echo {SENTINEL}; false"})["returncode"] == 1


def test_a_stopped_host_ends_the_run_instead_of_becoming_an_output():
    class Stopped(ScriptedHost):
        def run(self, command, *, cwd, timeout_seconds, shown=None):
            raise HostedStop("the trial's spending cap was reached")

    with pytest.raises(HostedStop):
        environment(Stopped()).execute({"command": "ls"})


def test_template_values_describe_the_machine_but_not_the_hosts_environment(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "not-for-prompts")
    values = environment(ScriptedHost()).get_template_vars()
    uname = platform.uname()
    assert (values["system"], values["release"], values["machine"]) == (uname.system, uname.release, uname.machine)
    assert values["cwd"] == "/app" and "AZURE_OPENAI_API_KEY" not in values


# -- the model: upstream's Responses model class, with Taste's model session as the transport


def test_the_agents_input_items_reach_the_provider_exactly_as_it_wrote_them():
    items = [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": "Please solve this issue: x"},
             thought(1), text("Let me look."), call("ls", "c1"),
             {"type": "function_call_output", "call_id": "c1", "output": "{\"returncode\": 0}"},
             {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Tool call error"}]}]
    assert OpenAIProvider(api_key="unused")._to_input(canonical_messages(items)) == items


def test_the_agents_own_tool_definition_is_sent_unchanged():
    assert _to_tool(BASH_TOOL_RESPONSE_API) == BASH_TOOL_RESPONSE_API
    # Taste's own tools keep their translation.
    assert _to_tool({"name": "t", "description": "d", "input_schema": {"type": "object"}})["strict"] is False


def model(host, **config):
    return TasteResponsesModel(host=host, effort="low", model_name="azure/gpt-5.6-luna", **config)


def test_a_reply_becomes_the_agents_message_with_its_commands_and_cost():
    host = ScriptedHost([reply(thought(1), text("Let me look."), call("ls -la", "c1"), cost=0.002)])
    message = model(host).query([{"role": "system", "content": "s"}, {"role": "user", "content": "t",
                                                                      "extra": {"note": 1}}])
    assert message["object"] == "response"
    assert [item["type"] for item in message["output"]] == ["reasoning", "message", "function_call"]
    assert message["extra"]["actions"] == [{"command": "ls -la", "tool_call_id": "c1"}]
    assert message["extra"]["cost"] == 0.002
    asked = host.asked[0]
    assert asked["effort"] == "low" and asked["tools"] == [BASH_TOOL_RESPONSE_API]
    assert asked["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]


def test_a_reply_without_a_command_is_a_format_error_that_keeps_its_cost():
    host = ScriptedHost([reply(text("I think we are done."), cost=0.003)])
    with pytest.raises(FormatError) as raised:
        model(host).query([{"role": "user", "content": "t"}])
    extra = raised.value.messages[0]["extra"]
    assert extra["cost"] == 0.003 and extra["response"]["object"] == "response"


def test_a_cut_off_reply_is_reported_as_upstream_reports_one():
    host = ScriptedHost([reply(status="incomplete", reason="max_output_tokens")])
    with pytest.raises(FormatError) as raised:
        model(host, format_error_template="{{ finish_reason }}").query([{"role": "user", "content": "t"}])
    assert raised.value.messages[0]["content"][0]["text"] == "length"


def test_only_the_reasoning_effort_of_the_model_options_is_taken():
    host = ScriptedHost([reply(call("ls", "c1"))])
    chosen = model(host, model_kwargs={"drop_params": True, "timeout": 3600})
    chosen.query([{"role": "user", "content": "t"}])
    assert host.asked[0]["effort"] == "low"
    with pytest.raises(ValueError, match="temperature"):
        model(ScriptedHost(), model_kwargs={"temperature": 0.2})


# -- the whole loop, against mini-swe-agent's own model class


SCRIPT = [
    reply(thought(1), text("Let me look around."), call("ls", "c1"), cost=0.002),
    reply(text("Nothing to change, I believe."), cost=0.001),
    reply(thought(2), call("cat a.py", "c2"), call("python a.py", "c3"), cost=0.002),
    reply(call(f"echo {SENTINEL}", "c4"), cost=0.001),
]
OUTPUTS = {"ls": ShellResult("a.py\n", 0), "cat a.py": ShellResult("print(1)\n", 0),
           "python a.py": ShellResult("1\n", 0), SENTINEL: ShellResult(f"{SENTINEL}\n", 0)}


class _Upstream:
    """A reply in the shape LiteLLM's Responses object has for mini-swe-agent."""

    def __init__(self, scripted):
        self.output = [dict(item) for item in scripted.output]
        self.status = scripted.status
        self.incomplete_details = ({"reason": scripted.incomplete_reason}
                                   if scripted.incomplete_reason else None)
        self.usage = dict(scripted.usage)
        self.cost = scripted.cost_usd
        self.dump = {"object": "response", "model": scripted.model, "status": scripted.status,
                     "output": copy.deepcopy(self.output), "usage": dict(scripted.usage),
                     "incomplete_details": self.incomplete_details}

    def model_dump(self, mode=None):
        return copy.deepcopy(self.dump)


@pytest.fixture
def upstream_model_class(monkeypatch):
    sent = []
    stub = types.ModuleType("litellm")

    class Refused(Exception):
        pass

    stub.exceptions = types.SimpleNamespace(
        UnsupportedParamsError=Refused, NotFoundError=Refused, PermissionDeniedError=Refused,
        ContextWindowExceededError=Refused, AuthenticationError=Refused)
    stub.cost_calculator = types.SimpleNamespace(completion_cost=lambda response, model: response.cost)
    stub.utils = types.SimpleNamespace(register_model=lambda value: None)
    script = [copy.deepcopy(item) for item in SCRIPT]

    def responses(*, model, input, tools, **options):
        sent.append({"input": copy.deepcopy(input), "tools": copy.deepcopy(tools), "options": options})
        return _Upstream(script.pop(0))

    stub.responses = responses
    monkeypatch.setitem(sys.modules, "litellm", stub)
    for name in ("minisweagent.models.litellm_model", "minisweagent.models.litellm_response_model"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module("minisweagent.models.litellm_response_model")
    yield module.LitellmResponseModel, sent
    for name in ("minisweagent.models.litellm_model", "minisweagent.models.litellm_response_model"):
        sys.modules.pop(name, None)


def without_times(messages):
    clean = copy.deepcopy(list(messages))
    for message in clean:
        message.get("extra", {}).pop("timestamp", None)
    return clean


def test_the_agent_sends_and_keeps_the_same_things_with_taste_as_with_its_own_model(upstream_model_class):
    from minisweagent.agents.default import DefaultAgent

    upstream_class, upstream_sent = upstream_model_class
    agent = MiniSweAgent(effort="low")
    configs = agent.configs()

    theirs_host = ScriptedHost(outputs=OUTPUTS)
    theirs = DefaultAgent(
        upstream_class(**{**configs["model"], "model_name": "azure/gpt-5.6-luna",
                          "model_kwargs": {"reasoning": {"effort": "low"}}}),
        TasteEnvironment(host=theirs_host, cwd="/app", **configs["environment"]), **configs["agent"])
    theirs_result = theirs.run("Fix the bug.")

    ours_host = ScriptedHost([copy.deepcopy(item) for item in SCRIPT], OUTPUTS)
    ours = agent.run("Fix the bug.", ours_host, cwd="/app", model_name="azure/gpt-5.6-luna")

    assert theirs_result["exit_status"] == ours.exit_status == "Submitted"
    provider = OpenAIProvider(api_key="unused")
    assert [provider._to_input(item["messages"]) for item in ours_host.asked] == [
        item["input"] for item in upstream_sent]
    assert all(item["tools"] == [BASH_TOOL_RESPONSE_API] for item in upstream_sent)
    assert all(item["tools"] == [BASH_TOOL_RESPONSE_API] for item in ours_host.asked)
    assert without_times(ours.messages) == without_times(theirs.messages)
    assert ours_host.ran == theirs_host.ran
    assert ours.model_calls == theirs.n_calls == 4


def test_the_run_reports_how_it_ended_and_what_it_said(tmp_path):
    host = ScriptedHost([reply(call(f"echo {SENTINEL} && echo done", "c1"))],
                        {SENTINEL: ShellResult(f"{SENTINEL}\nthe fix is in a.py\n", 0)})
    result = MiniSweAgent(effort="medium").run("Fix it.", host, cwd="/app", model_name="m")
    assert isinstance(result, AgentExit)
    assert (result.exit_status, result.submission) == ("Submitted", "the fix is in a.py\n")
    assert host.asked[0]["effort"] == "medium"
    first = host.asked[0]["messages"]
    # Rendered by upstream's Jinja templates, which drop a template's final newline.
    assert first[0] == {"role": "system", "content": "You are a helpful assistant that can interact with a computer."}
    assert first[1]["content"].startswith("Please solve this issue: Fix it.")


def test_a_stopped_host_ends_the_agent_and_is_not_swallowed():
    class Stopped(ScriptedHost):
        def ask(self, **_):
            raise HostedStop("deadline")

    with pytest.raises(HostedStop):
        MiniSweAgent(effort="low").run("Fix it.", Stopped(), cwd="/app", model_name="m")


def test_the_upstream_configuration_is_named_by_version_and_digest():
    agent = MiniSweAgent(effort="low")
    identity = agent.identity()
    assert identity["name"] == "mini-swe-agent" and identity["version"] == "2.4.6"
    assert identity["config"] == "mini" and identity["config_sha256"] == MiniSweAgent(effort="high").identity()["config_sha256"]
    assert len(identity["config_sha256"]) == 64
