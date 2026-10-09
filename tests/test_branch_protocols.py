"""Branching an agent that works in SWE-agent's manner: commands before its first call, state reads after
each action, a marker in its submission's output. No model, no container."""

from __future__ import annotations

import asyncio
import json

import pytest

from taste.agents import HostedStop, ShellResult
from taste.agents.protocols import SWE_AGENT_MARKER, protocol
from taste.brains.branch_replay import (
    REJECT_EXIT,
    SCRIPT_SCHEMA,
    ReplayHost,
    ReplayScript,
    check_branch,
    digest,
    message_shas,
    rebuild,
    reply_to_dict,
    request_sha,
)
from tests.test_branch_replay import TOOL, reply

TASK = "Make the parser keep tabs."
NOTE = "Note: an attempt that continued from here was rejected by a reviewer: tabs are still dropped."
REJECTION = "Submission rejected by a reviewer: tabs are still dropped."
SETUP = ["# swe-rex: start the bash session", "export LANG=C.UTF-8", "cd /app && export ROOT=$(pwd -P)",
         "# swe-rex: upload tools/submit -> /root/tools/submit"]
STATE = "_state_anthropic"
READ_STATE = "# swe-rex: read_file /root/state.json"
READ_PATCH = "# swe-rex: read_file /root/model.patch"
PATCH = "diff --git a/parser.py b/parser.py\n-x\n+y\n"
OUTPUTS = {**{command: ("", 0) for command in SETUP},
           STATE: ("", 0), READ_STATE: ('{"working_dir": "/app"}', 0), READ_PATCH: (PATCH, 0),
           "make test": ("FAILED test_tabs", 1),
           "sed -i 's/x/y/' parser.py && make test": ("4 passed", 0),
           "submit": (f"{SWE_AGENT_MARKER}\n{PATCH}{SWE_AGENT_MARKER}", 0)}
BASE = [reply("make test"), reply("sed -i 's/x/y/' parser.py && make test"), reply("submit")]


def wrap(command):
    return "bash -c " + json.dumps(command)


class SweLike:
    """SWE-agent's order of calls: setup, then per step the action, the state command and a state read;
    after an action whose output holds the marker, a read of the patch, and the run ends."""

    def __init__(self):
        self.seen = []

    def _run(self, host, command):
        return host.run(wrap(command), cwd="/", timeout_seconds=30, shown=command)

    def run(self, host, limit=8):
        for command in SETUP:
            self._run(host, command)
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": TASK}]
        for _ in range(limit):
            answer = host.ask(messages=list(messages), tools=[TOOL], effort="low")
            messages.extend(dict(item) for item in answer.output)
            call = next(item for item in answer.output if item["type"] == "function_call")
            result = self._run(host, json.loads(call["arguments"])["command"])
            self.seen.append(result)
            self._run(host, STATE)
            self._run(host, READ_STATE)
            if SWE_AGENT_MARKER in result.output:
                return "Submitted", self._run(host, READ_PATCH).output
            messages.append({"type": "function_call_output", "call_id": call["call_id"],
                             "output": f"OBSERVATION:\n{result.output}"})
        return "LimitsExceeded", ""


class Live:
    def __init__(self, replies=()):
        self.replies, self.asked, self.ran, self.stopped = list(replies), [], [], None

    def ask(self, *, messages, tools, effort):
        self.asked.append(request_sha(messages, tools, effort))
        return self.replies.pop(0)

    def run(self, command, *, cwd, timeout_seconds, shown=None):
        self.ran.append(shown)
        return ShellResult(*OUTPUTS[shown])

    def stop(self, reason, *, now=True):
        self.stopped = reason


class Recorder:
    """What crossed the host, as a replay script: commands before the first call are its setup."""

    def __init__(self, inner):
        self.inner, self.setup, self.steps, self.previous, self.stopped = inner, [], [], [], None

    def ask(self, *, messages, tools, effort):
        answer = self.inner.ask(messages=messages, tools=tools, effort=effort)
        shas = message_shas(messages)
        base = next((index for index, (a, b) in enumerate(zip(self.previous, shas, strict=False)) if a != b),
                    min(len(self.previous), len(shas)))
        self.previous = shas
        calls = [{"id": item["call_id"], "name": item["name"], "arguments": json.loads(item["arguments"])}
                 for item in answer.output if item["type"] == "function_call"]
        self.steps.append({"step": len(self.steps) + 1, "request_sha": request_sha(messages, tools, effort),
                           "messages": {"base": base, "added": shas[base:]}, "reply": reply_to_dict(answer),
                           "text": ["Next step."], "calls": calls, "runs": [], "replayed": False})
        return answer

    def run(self, command, *, cwd, timeout_seconds, shown=None):
        result = self.inner.run(command, cwd=cwd, timeout_seconds=timeout_seconds, shown=shown)
        (self.steps[-1]["runs"] if self.steps else self.setup).append({
            "command": shown, "executed": command, "cwd": cwd, "timeout_seconds": timeout_seconds,
            "output": result.output, "returncode": result.returncode, "timed_out": result.timed_out,
            "output_exact": True})
        return result

    def script(self):
        return ReplayScript.from_dict({"schema": SCRIPT_SCHEMA, "task": TASK, "setup": self.setup,
                                       "steps": self.steps, "exit": None,
                                       "source": {"effort": "low", "tools_sha256": digest([TOOL]),
                                                  "agent": {"name": "swe-agent"}}})


@pytest.fixture
def script():
    recorder = Recorder(Live(BASE))
    assert SweLike().run(recorder) == ("Submitted", PATCH)
    return recorder.script()


def branch(script, step, live, **options):
    rows = []
    host = ReplayHost(live, script, step, record=lambda kind, payload: rows.append((kind, payload)), **options)
    return host, rows


def test_commands_before_the_first_call_are_the_scripts_setup(script):
    assert [run.command for run in script.setup] == SETUP
    assert [len(step.runs) for step in script.steps] == [3, 3, 4]
    again = ReplayScript.from_dict(script.to_dict())
    assert again.setup == script.setup and again.to_dict() == script.to_dict()
    assert script.to_dict()["submission_step"] == 3


def test_a_script_without_setup_reads_and_writes_as_before(script):
    value = script.to_dict()
    value.pop("setup")
    value["source"]["agent"] = {"name": "mini-swe-agent"}
    value["steps"] = []
    assert "setup" not in ReplayScript.from_dict(value).to_dict()


def test_the_setup_and_the_prefix_are_answered_from_the_record_then_the_agent_goes_live(script):
    live = Live([reply("make test"), reply("submit")])
    host, rows = branch(script, 2, live)
    assert SweLike().run(host) == ("Submitted", PATCH)
    # Nothing of the setup or of steps 1..2 ran: all of it came from the record.
    assert live.ran == ["make test", STATE, READ_STATE, "submit", STATE, READ_STATE, READ_PATCH]
    replayed = [payload["step"] for kind, payload in rows if kind == "replay_output"]
    assert replayed == [0] * len(SETUP) + [1, 1, 1, 2, 2, 2]
    assert ("prefix", {"steps": 2}) in rows and ("live", {"live": True, "context": "matched"}) in rows


def test_an_agent_that_skips_a_setup_command_is_unfaithful(script):
    class Skips(SweLike):
        def _run(self, host, command):
            return ShellResult("", 0) if command == SETUP[1] else super()._run(host, command)

    host, rows = branch(script, 2, Live())
    with pytest.raises(HostedStop, match="branch_unfaithful"):
        Skips().run(host)
    assert next(payload for kind, payload in rows if kind == "unfaithful")["step"] == 0


def test_a_rebuild_runs_the_setup_before_the_steps(script):
    executed = []

    async def execute(number, run):
        executed.append((number, run.command))
        return run.output, run.returncode, run.timed_out

    account = asyncio.run(rebuild(script, 1, execute))
    assert executed == [(0, command) for command in SETUP] + [(1, "make test"), (1, STATE), (1, READ_STATE)]
    assert account["faithful"] and account["commands"] == len(SETUP) + 3
    assert [row["step"] for row in account["rows"]][:len(SETUP)] == [0] * len(SETUP)


def test_a_note_goes_after_the_action_not_after_the_state_reads(script):
    live = Live([reply("sed -i 's/x/y/' parser.py && make test"), reply("submit")])
    host, rows = branch(script, 1, live, override="append", note=NOTE)
    agent = SweLike()
    assert agent.run(host)[0] == "Submitted"
    assert agent.seen[0].output == "FAILED test_tabs\n" + NOTE and agent.seen[0].returncode == 1
    shown = [payload for kind, payload in rows if kind == "replay_output" and payload["step"] == 1]
    assert [payload["override"] for payload in shown] == ["append", "", ""]
    assert shown[2]["output"] == '{"working_dir": "/app"}'


def test_a_rejected_submission_keeps_the_agent_working_and_its_patch_read_is_dropped(script):
    live = Live([reply("sed -i 's/x/y/' parser.py && make test"), reply("submit")])
    host, rows = branch(script, 3, live, override="reject", note=REJECTION)
    agent = SweLike()
    assert agent.run(host) == ("Submitted", PATCH)
    rejected = agent.seen[2]
    assert (rejected.output, rejected.returncode) == (REJECTION, REJECT_EXIT)
    # The state command and read after the action came from the record; the patch read never happened.
    step3 = [payload["command"] for kind, payload in rows if kind == "replay_output" and payload["step"] == 3]
    assert step3 == ["submit", STATE, READ_STATE]
    assert ("prefix", {"steps": 3, "dropped": 1}) in rows
    assert ("live", {"live": True, "context": "unchecked"}) in rows
    assert live.ran[:3] == ["sed -i 's/x/y/' parser.py && make test", STATE, READ_STATE]


def test_overrides_follow_the_agents_protocol(script):
    check_branch(script, 3, mode="restore", override="reject", note=REJECTION)
    check_branch(script, 2, mode="rebuild", override="append", note=NOTE)
    with pytest.raises(ValueError, match="not one"):
        check_branch(script, 2, mode="restore", override="reject", note=REJECTION)
    with pytest.raises(ValueError, match="is the submission"):
        check_branch(script, 3, mode="restore", override="append", note=NOTE)
    value = script.to_dict()
    value["steps"][0]["runs"] = value["steps"][0]["runs"][1:]  # an exit: only the state reads ran
    with pytest.raises(ValueError, match="ran none"):
        check_branch(ReplayScript.from_dict(value), 1, mode="restore", override="append", note=NOTE)


def test_a_rebuild_needs_the_setups_exact_commands_too(script):
    value = script.to_dict()
    value["setup"][1]["executed"] = None
    with pytest.raises(ValueError, match="exact commands"):
        check_branch(ReplayScript.from_dict(value), 1, mode="rebuild")
    check_branch(ReplayScript.from_dict(value), 1, mode="restore")


def test_each_agents_submission_text():
    assert protocol("swe-agent").submission_text(OUTPUTS["submit"][0]) == PATCH
    assert protocol("swe-agent").submission_text("no marker") == ""
    sentinel = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nfixed tabs\n"
    assert protocol("mini-swe-agent").submission_text(sentinel) == "fixed tabs\n"
    assert protocol(None).submission_text(sentinel) == "fixed tabs\n"
