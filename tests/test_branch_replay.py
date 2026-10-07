"""A recorded run branched at step k, with a small agent loop and hosts of our own: no model, no container."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from taste.agents import HostedStop, ModelReply, ShellResult
from taste.brains.branch_replay import (
    OUTPUT_FLOOR,
    REJECT_EXIT,
    SCRIPT_SCHEMA,
    SENTINEL,
    ReplayHost,
    ReplayScript,
    check_branch,
    compare,
    digest,
    message_shas,
    rebuild,
    rebuild_timeout,
    reply_from_completion,
    reply_to_dict,
    request_sha,
    similarity,
    submitted,
)

TOOL = {"type": "function", "name": "bash", "parameters": {"type": "object"}}
TASK = "Make the parser tests pass."
NOTE = "Note: an attempt that continued from here was rejected by a reviewer: the parser drops tabs."


def reply(*commands, text="Next step."):
    output = [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]
    output += [{"type": "function_call", "call_id": f"call_{index}", "name": "bash",
                "arguments": json.dumps({"command": command})} for index, command in enumerate(commands)]
    return ModelReply(output=tuple(output), status="completed", incomplete_reason=None,
                      usage={"input_tokens": 100, "output_tokens": 20}, model="gpt-test", cost_usd=0.25)


def wrap(command):
    return "{\n" + command + "\n} 2>&1"


class Agent:
    """An agent loop in mini-swe-agent's manner: ask, run each command, stop at its submission."""

    def __init__(self, task=TASK):
        self.task, self.costs, self.seen = task, [], []

    def run(self, host, limit=8):
        messages = [{"role": "system", "content": "You are a careful engineer."},
                    {"role": "user", "content": "Please solve this issue: " + self.task}]
        for _ in range(limit):
            answer = host.ask(messages=list(messages), tools=[TOOL], effort="low")
            self.costs.append(answer.cost_usd)
            messages.extend(dict(item) for item in answer.output)
            for item in answer.output:
                if item["type"] != "function_call":
                    continue
                command = json.loads(item["arguments"])["command"]
                result = host.run(wrap(command), cwd="/app", timeout_seconds=30, shown=command)
                self.seen.append(result)
                messages.append({"type": "function_call_output", "call_id": item["call_id"],
                                 "output": f"<returncode>{result.returncode}</returncode>\n{result.output}"})
                if submitted(result.output, result.returncode):
                    return "Submitted", messages
        return "LimitsExceeded", messages


class Live:
    """The live side: scripted replies, and outputs by command."""

    def __init__(self, replies=(), outputs=None):
        self.replies, self.outputs = list(replies), outputs or {}
        self.asked, self.ran, self.stopped = [], [], None

    def ask(self, *, messages, tools, effort):
        self.asked.append(request_sha(messages, tools, effort))
        return self.replies.pop(0)

    def run(self, command, *, cwd, timeout_seconds, shown=None):
        self.ran.append(shown)
        output, code = self.outputs[shown]
        return ShellResult(output, code)

    def stop(self, reason, *, now=True):
        self.stopped = reason


class Recorder:
    """A host in front of another that writes down what crossed it, as a replay script's steps."""

    def __init__(self, inner):
        self.inner, self.steps, self.previous = inner, [], []
        self.stopped = None

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
        self.steps[-1]["runs"].append({"command": shown, "executed": command, "cwd": cwd,
                                       "timeout_seconds": timeout_seconds, "output": result.output,
                                       "returncode": result.returncode, "timed_out": result.timed_out,
                                       "output_exact": True})
        return result

    def script(self):
        return ReplayScript.from_dict({"schema": SCRIPT_SCHEMA, "task": TASK, "steps": self.steps, "exit": None,
                                       "source": {"effort": "low", "tools_sha256": digest([TOOL]),
                                                  "agent": {"name": "mini-swe-agent"}}})


OUTPUTS = {"make test": ("FAILED test_parse\n", 1),
           "sed -i 's/x/y/' parser.py && make test": ("4 passed\n", 0),
           f"echo {SENTINEL}": (f"{SENTINEL}\n", 0)}
BASE = [reply("make test"), reply("sed -i 's/x/y/' parser.py && make test"), reply(f"echo {SENTINEL}")]


@pytest.fixture
def recorded():
    """A finished run of three steps (a failing test, a fix, a submission) and its replay script."""
    recorder = Recorder(Live(BASE, OUTPUTS))
    status, messages = Agent().run(recorder)
    assert status == "Submitted"
    return recorder.script(), messages


def branch(script, step, *, replies=(), outputs=None, **options):
    events = []
    live = Live(replies, outputs if outputs is not None else OUTPUTS)
    host = ReplayHost(live, script, step, record=lambda kind, payload: events.append((kind, payload)), **options)
    return host, live, events


def test_a_faithful_branch_answers_k_steps_from_the_record_then_goes_live(recorded):
    script, original = recorded
    host, live, events = branch(script, 2, replies=[reply(f"echo {SENTINEL}")])
    agent = Agent()
    status, messages = agent.run(host)
    assert status == "Submitted" and messages == original
    # Two steps from the record: no model call, no command, and they cost nothing.
    assert live.ran == [f"echo {SENTINEL}"] and len(live.asked) == 1
    assert agent.costs == [0.0, 0.0, 0.25]
    # The first live request is the one the recorded run sent at step 3.
    assert live.asked == [script.steps[2].request_sha]
    kinds = [kind for kind, _ in events]
    assert kinds == ["replay", "replay_output", "replay", "replay_output", "prefix", "live"]
    assert events[-1][1] == {"live": True, "context": "matched"}
    assert events[0][1]["recorded_cost_usd"] == 0.25 and events[0][1]["step"] == 1
    assert host.faithful and host.prefix_complete and host.replayed == 2


def test_a_request_that_differs_from_the_record_stops_the_branch_before_any_model_call(recorded):
    script, _ = recorded
    host, live, events = branch(script, 2, replies=[reply("ls")])
    with pytest.raises(HostedStop, match="branch_unfaithful"):
        Agent(task="Make the lexer tests pass.").run(host)
    assert live.asked == [] and live.ran == []
    [(kind, payload)] = events
    assert kind == "unfaithful" and payload["reason"] == "request" and payload["step"] == 1
    # The task is the agent's second message: the diagnosis names it.
    assert payload["detail"]["first_differing_message"] == 1 and payload["detail"]["role"] == "user"
    assert not host.faithful


def test_a_command_that_differs_from_the_record_stops_the_branch(recorded):
    script, _ = recorded
    value = script.to_dict()
    value["steps"][1]["runs"][0]["executed"] = wrap("sed -i 's/x/z/' parser.py && make test")
    host, live, events = branch(ReplayScript.from_dict(value), 2)
    with pytest.raises(HostedStop, match="branch_unfaithful"):
        Agent().run(host)
    assert live.ran == [] and live.asked == []
    assert events[-1][0] == "unfaithful" and events[-1][1]["reason"] == "command"
    assert events[-1][1]["step"] == 2 and events[-1][1]["detail"]["command_matches"] is False


def test_append_shows_step_ks_output_followed_by_the_note(recorded):
    script, _ = recorded
    host, live, events = branch(script, 1, override="append", note=NOTE,
                                replies=[reply("sed -i 's/x/y/' parser.py && make test"), reply(f"echo {SENTINEL}")])
    agent = Agent()
    status, messages = agent.run(host)
    assert status == "Submitted"
    assert agent.seen[0] == ShellResult("FAILED test_parse\n" + NOTE, 1)
    # The system and task messages, the reply's two items, then the command's output.
    assert messages[4]["output"].endswith("FAILED test_parse\n" + NOTE)
    # The next request differs from the record by design: it is not checked against it.
    assert events[2] == ("prefix", {"steps": 1}) and events[3] == ("live", {"live": True, "context": "unchecked"})
    assert events[1][1]["override"] == "append" and live.ran == ["sed -i 's/x/y/' parser.py && make test",
                                                                   f"echo {SENTINEL}"]


def test_reject_replaces_the_submission_with_a_failing_rejection_so_the_agent_works_on(recorded):
    script, _ = recorded
    rejection = "Submission rejected by a reviewer: tabs are still dropped (tests/test_tabs.py fails)."
    host, live, events = branch(script, 3, override="reject", note=rejection,
                                replies=[reply("make test"), reply(f"echo {SENTINEL}")])
    agent = Agent()
    status, _ = agent.run(host)
    assert status == "Submitted"
    assert agent.seen[2] == ShellResult(rejection, REJECT_EXIT)
    assert not submitted(agent.seen[2].output, agent.seen[2].returncode)
    assert len(live.asked) == 2 and live.ran == ["make test", f"echo {SENTINEL}"]
    assert [kind for kind, _ in events][-2:] == ["prefix", "live"]


def test_live_off_stops_the_agent_right_after_the_prefix(recorded):
    script, _ = recorded
    host, live, events = branch(script, 2, live_after=False)
    with pytest.raises(HostedStop, match="branch_live_off"):
        Agent().run(host)
    assert live.asked == [] and live.ran == []
    assert events[-1] == ("live", {"live": False, "context": "matched"})


def test_a_branch_whose_files_were_not_brought_back_replays_but_never_goes_live(recorded):
    script, _ = recorded
    host, live, events = branch(script, 2, faithful=False, replies=[reply("ls")])
    with pytest.raises(HostedStop, match="branch_unfaithful"):
        Agent().run(host)
    assert live.asked == [] and host.replayed == 2
    assert [kind for kind, _ in events][-1] == "prefix"


def test_a_branch_from_the_start_checks_the_first_request_and_goes_live(recorded):
    script, _ = recorded
    host, live, events = branch(script, 0, replies=list(BASE))
    status, _ = Agent().run(host)
    assert status == "Submitted" and len(live.asked) == 3
    assert events == [("live", {"live": True, "context": "matched"})]


def test_another_agent_given_only_the_files_starts_fresh_and_unchecked(recorded):
    """The replay off: a checker, say, with its own task, given the files of the last step."""
    script, _ = recorded
    host, live, events = branch(script, 0, context=False, replies=[reply(f"echo {SENTINEL}")])
    status, messages = Agent(task="Check whether the parser change is done.").run(host)
    assert status == "Submitted" and len(messages) == 5 and len(live.asked) == 1
    assert events == [("live", {"live": True, "context": "unchecked"})]
    with pytest.raises(ValueError, match="needs the replay"):
        check_branch(script, 3, mode="rebuild", override="reject", note=NOTE, replay=False)
    check_branch(script, 3, mode="rebuild", replay=False)


def test_the_script_says_which_branches_it_can_serve(recorded):
    script, _ = recorded
    check_branch(script, 3, mode="rebuild", override="reject", note=NOTE)
    for step, options, match in [
            (4, {}, "between 0 and 3"),
            (2, {"override": "reject", "note": NOTE}, "not one"),
            (3, {"override": "append", "note": NOTE}, "use reject"),
            (2, {"override": "append"}, "needs its note"),
            (0, {"override": "append", "note": NOTE}, "ran none")]:
        with pytest.raises(ValueError, match=match):
            check_branch(script, step, mode="rebuild", **options)
    value = script.to_dict()
    value["steps"][0]["runs"][0]["executed"] = None
    unexact = ReplayScript.from_dict(value)
    check_branch(unexact, 2, mode="restore")
    with pytest.raises(ValueError, match="exact commands"):
        check_branch(unexact, 2, mode="rebuild")


def test_a_script_round_trips_and_rebuilds_its_request_messages(recorded):
    script, original = recorded
    again = ReplayScript.from_dict(json.loads(json.dumps(script.to_dict())))
    assert again == script and again.to_dict()["submission_step"] == 3
    # The digests of each request's messages are carried step to step: step 3
    # asked with everything but its own reply's two items and their output.
    assert script.message_shas(3) == message_shas(original[:-3])
    with pytest.raises(ValueError, match="replay script"):
        ReplayScript.from_dict({**script.to_dict(), "schema": "other"})


def test_rebuilt_outputs_are_compared_by_their_normalized_lines():
    assert similarity("", "") == 1.0
    assert similarity("ran 5 tests in 0.12s\nOK\n", "ran 5 tests in 3.40s\nOK\n") == 1.0
    assert similarity("commit 3f9a2b7c1d\n", "commit 0aa917f3e2\n") == 1.0
    assert similarity("a\nb\nc\nd\n", "a\nb\nc\ne\n") == 0.75
    assert similarity("yes\n", "no\n") == 0.0
    run = ReplayScript.from_dict({"schema": SCRIPT_SCHEMA, "task": TASK, "source": {}, "exit": None, "steps": [
        {"step": 1, "request_sha": "0" * 64, "messages": {"base": 0, "added": []},
         "reply": reply_to_dict(reply("x")), "text": [], "calls": [], "runs": [
             {"command": "x", "executed": "x", "cwd": "/", "timeout_seconds": 5, "output": "4 passed\n",
              "returncode": 0, "timed_out": False, "output_exact": True}]}]}).steps[0].runs[0]
    assert compare(run, "4 passed\n", 0, False)["divergent"] is False
    assert compare(run, "4 passed\n", 1, False)["divergent"] is True
    assert compare(run, "4 passed\n", 0, True)["divergent"] is True
    assert compare(run, "2 failed\n", 0, False)["similarity"] < OUTPUT_FLOOR
    # A step a branch changed is rebuilt against what the container printed, not what the agent was shown.
    rejected = replace(run, output="Submission rejected.", returncode=REJECT_EXIT,
                       printed={"output": "4 passed\n", "returncode": 0, "timed_out": False})
    assert compare(rejected, "4 passed\n", 0, False)["divergent"] is False
    assert type(rejected).from_dict(rejected.to_dict()) == rejected
    # Cut off by its time limit when recorded: whether it is cut off again, and its exit code, depend
    # on the host's speed, so it is matched by its output alone.
    cut = replace(run, returncode=137, timed_out=True)
    assert compare(cut, "4 passed\n", 0, False)["divergent"] is False
    assert compare(cut, "4 passed\n", 137, True)["divergent"] is False
    assert compare(cut, "2 failed\n", 137, True)["divergent"] is True
    # A command the record shows finishing gets twice its limit, within the terminal's maximum;
    # one it shows cut off keeps its own, so its effects are cut off as before.
    assert rebuild_timeout(run, 600) == 10.0 and rebuild_timeout(run, 8) == 8.0
    assert rebuild_timeout(cut, 600) == 5.0


def test_a_rebuild_stops_once_more_commands_diverge_than_it_tolerates(recorded):
    script, _ = recorded
    ran = []

    def execute_with(changed):
        async def execute(number, run):
            ran.append(number)
            output, code = OUTPUTS[run.command]
            return output, changed.get(run.command, code), False
        return execute

    account = asyncio.run(rebuild(script, 3, execute_with({})))
    assert account["faithful"] and account["commands"] == 3 and account["divergent"] == 0
    ran.clear()
    account = asyncio.run(rebuild(script, 3, execute_with({"make test": 0})))
    assert not account["faithful"] and account["commands"] == 1 and ran == [1]
    account = asyncio.run(rebuild(script, 3, execute_with({"make test": 0}), tolerance=1))
    assert account["faithful"] and account["commands"] == 3 and account["divergent"] == 1


def test_the_digests_and_replies_are_the_hosted_workers_own():
    from taste.brains.hosted_worker import _reply, _sha
    from taste.brains.responses_session import _completion, _completion_payload
    from taste.providers._openai import _NATIVE

    messages = [{"role": "user", "content": "Fix it."}]
    assert request_sha(messages, [TOOL], "low") == _sha({"messages": messages, "tools": [TOOL], "effort": "low"})
    payload = {"text_blocks": ["ok"], "tool_calls": [], "stop_reason": "max_tokens", "model": "m",
               "provider": "openai", "usage": {"input_tokens": 3, "output_tokens": 2, "cache_read_tokens": 0,
                                               "cache_write_tokens": 0, "reasoning_tokens": 1},
               "transcript_blocks": [{"type": "text", "text": "ok"},
                                     {"type": _NATIVE, "item": {"type": "reasoning", "id": "rs_1"}}],
               "effective_sampling": {}, "provenance": {}}
    completion = _completion(payload)
    assert reply_from_completion(_completion_payload(completion), 0.5) == _reply(completion, 0.5)
