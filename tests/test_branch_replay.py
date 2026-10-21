"""A recorded run branched at step k, with a small agent loop and hosts of our own: no model, no container."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from taste.agents import HostedStop, ModelReply, ShellResult, agentrewind
from taste.brains.branch_replay import (
    OUTPUT_FLOOR,
    REJECT_EXIT,
    SCRIPT_SCHEMA,
    SENTINEL,
    BranchPolicy,
    RecordedRun,
    ReplayHost,
    ReplayScript,
    check_branch,
    compare,
    digest,
    host_ordered,
    measured,
    measured_at,
    message_shas,
    prompt_bound,
    rebuild,
    rebuild_timeout,
    replay_message_shas,
    replay_sha,
    reply_from_completion,
    reply_to_dict,
    request_sha,
    similarity,
    submitted,
)
from taste.pricing import request_token_bound

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
        shas = replay_message_shas(messages)
        base = next((index for index, (a, b) in enumerate(zip(self.previous, shas, strict=False)) if a != b),
                    min(len(self.previous), len(shas)))
        self.previous = shas
        calls = [{"id": item["call_id"], "name": item["name"], "arguments": json.loads(item["arguments"])}
                 for item in answer.output if item["type"] == "function_call"]
        self.steps.append({"step": len(self.steps) + 1, "request_sha": replay_sha(messages, tools, effort),
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
                                                  "agent": {"name": "mini-swe-agent"}, "model": "gpt-test"}})


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


class RewindAgent(Agent):
    """The rewind arm's agent: its two tools and its guidance on every call, its notebook from call ``at_call``
    on, after the messages that call restores (as ``taste.agents.mini_swe_agent`` adds them)."""

    def __init__(self, notes=(), at_call=None, **options):
        super().__init__(**options)
        self.notes, self.at_call, self.calls, self.note_at = list(notes), at_call, 0, None

    def run(self, host, limit=8):
        agent = self

        class Adding:
            def ask(self, *, messages, tools, effort):
                agent.calls += 1
                sent = [dict(messages[0], content=agentrewind.with_guidance(messages[0]["content"])), *messages[1:]]
                if agent.at_call is not None and agent.calls >= agent.at_call:
                    agent.note_at = len(sent) if agent.note_at is None else agent.note_at
                    sent.insert(agent.note_at, agentrewind.notebook_message(agent.notes))
                return host.ask(messages=sent, tools=[*tools, *agentrewind.tools()], effort=effort)

            def run(self, command, **options):
                return host.run(command, **options)

        return super().run(Adding(), limit)


class Seeing(Live):
    """A live side that keeps the messages it was sent."""

    def __init__(self, *args, **options):
        super().__init__(*args, **options)
        self.sent = []

    def ask(self, *, messages, tools, effort):
        self.sent.append((list(messages), list(tools)))
        return super().ask(messages=messages, tools=tools, effort=effort)


def test_a_run_replays_whether_or_not_the_agent_adds_the_rewind_arms_tools_and_guidance(recorded):
    """Round one of the rewind arm replays a run recorded without its tools; a later round may replay one
    recorded with them, or with another notebook. A replay compares requests without them."""
    script, _ = recorded
    host, live, events = branch(script, 2, replies=[reply(f"echo {SENTINEL}")])
    status, _ = RewindAgent().run(host)
    assert status == "Submitted" and host.faithful and events[-1][1] == {"live": True, "context": "matched"}
    assert live.asked != [script.steps[2].request_sha]  # what was sent carries the tools and the guidance
    # A run recorded by the rewind arm's agent, with a notebook from its second call, replays for a plain agent.
    recorder = Recorder(Live(BASE, OUTPUTS))
    assert RewindAgent(notes=[{"memory_summary": "Run the tests first."}], at_call=2).run(recorder)[0] == "Submitted"
    plain, _, events = branch(recorder.script(), 2, replies=[reply(f"echo {SENTINEL}")])
    assert Agent().run(plain)[0] == "Submitted" and plain.faithful and events[-1][1]["context"] == "matched"
    assert recorder.script().message_shas(3) == script.message_shas(3)


def test_a_rewind_re_decides_step_k_with_the_notebook_after_the_messages_it_restores(recorded):
    """Rewinding to llm-2 brings back step 1 (its files and its context) and makes step 2's call again, live,
    with the notebook after the messages step 2's call was sent."""
    script, original = recorded
    notes = [{"memory_summary": "Do not run make test before reading the parser.", "target_record_uid": "llm-2",
              "reason": "rejected twice"}]
    events = []
    live = Seeing([reply("sed -i 's/x/y/' parser.py && make test"), reply(f"echo {SENTINEL}")], OUTPUTS)
    host = ReplayHost(live, script, 1, record=lambda kind, payload: events.append((kind, payload)),
                      override="rewind", note=agentrewind.notebook(notes))
    status, _ = RewindAgent(notes=notes, at_call=2).run(host)
    assert status == "Submitted" and host.faithful and events[-1][1] == {"live": True, "context": "matched"}
    messages, tools = live.sent[0]
    restored = message_shas(original[:len(script.message_shas(2))])
    assert messages[-1] == agentrewind.notebook_message(notes)
    assert replay_message_shas(messages) == restored and [tool["name"] for tool in tools] == [
        "bash", "backtrack_candidates", "backtrack_commit"]
    # The notebook stays where it went in: after step 2's restored messages, before what followed.
    later, _ = live.sent[1]
    assert later[len(messages) - 1] == agentrewind.notebook_message(notes)
    # Step 1's command was replayed, not run, and no command of step 1 was changed.
    assert live.ran == ["sed -i 's/x/y/' parser.py && make test", f"echo {SENTINEL}"]
    assert all(payload.get("override", "") == "" for kind, payload in events if kind == "replay_output")


def test_a_rewind_branch_says_what_it_carries_and_needs_its_notebook():
    notes = [{"memory_summary": "Read the parser first.", "target_record_uid": "llm-2", "reason": None}]
    rewind = {"notes": notes, "restart_usd": 0.01}
    policy = BranchPolicy("/s/script.json", "a" * 64, 1, override="rewind", note=agentrewind.notebook(notes),
                          rewind=rewind)
    assert BranchPolicy.from_dict(policy.to_dict()) == policy and policy.to_dict()["rewind"] == rewind
    # The rewind arm's tools without a rewind: round one, carrying on in place after the rejection.
    first = BranchPolicy("/s/script.json", "a" * 64, 3, override="reject", note=NOTE,
                         rewind={"notes": [], "restart_usd": 0.01})
    assert BranchPolicy.from_dict(first.to_dict()) == first
    for options, match in [({"override": "rewind", "note": "other text", "rewind": rewind}, "notebook"),
                           ({"override": "rewind", "note": agentrewind.notebook(notes)}, "rewind"),
                           ({"rewind": {"notes": [{"memory_summary": ""}], "restart_usd": 0.0}}, "note"),
                           ({"rewind": {"notes": [], "restart_usd": -1.0}}, "restart"),
                           ({"rewind": {"notes": []}}, "rewind")]:
        with pytest.raises(ValueError, match=match):
            BranchPolicy("/s/script.json", "a" * 64, 1, **options)


def test_a_rewind_may_go_back_to_any_call_and_a_rebuild_skips_the_tools_calls(recorded):
    script, _ = recorded
    notebook = agentrewind.notebook([{"memory_summary": "m"}])
    for step in (0, 1, 2):
        check_branch(script, step, mode="rebuild", override="rewind", note=notebook)
    with pytest.raises(ValueError, match="needs its note"):
        check_branch(script, 1, mode="rebuild", override="rewind")
    # A call of the two tools changed no file: a rebuild runs none, and needs no exact text for them.
    value = script.to_dict()
    listing = agentrewind.tool_command("backtrack_candidates", {"reason": "stuck"})
    value["steps"][1]["runs"].insert(0, {"command": listing, "executed": None, "cwd": "/app", "timeout_seconds": 30,
                                         "output": "[]", "returncode": 0, "timed_out": False, "output_exact": True})
    with_tools = ReplayScript.from_dict(value)
    check_branch(with_tools, 3, mode="rebuild")
    ran = []

    async def execute(number, run):
        ran.append(run.command)
        return run.output, run.returncode, run.timed_out

    account = asyncio.run(rebuild(with_tools, 3, execute))
    assert account["faithful"] and listing not in ran and len(ran) == account["commands"] == 3


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


def size(value):
    return len(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode())


def test_a_request_is_bounded_by_its_measured_start_and_the_bytes_added_since(recorded):
    """A model's window is in tokens, and a request is only measured once it is answered: what an earlier,
    measured request shares with this one is bounded by its measure, the messages added since by their bytes."""
    script, original = recorded
    whole = prompt_bound(original, [TOOL], "low")
    assert whole == request_token_bound(size({"messages": original, "tools": [TOOL], "effort": "low"}))
    # Step 2's request, as the recorded run measured it (100 prompt tokens on gpt-test).
    earlier = measured_at(script, 2, "gpt-test")
    shared = len(script.message_shas(2))
    assert earlier[2] == 100 and 0 < shared < len(original)
    assert prompt_bound(original, [TOOL], "low", earlier) == 100 + request_token_bound(size(original[shared:]))
    assert prompt_bound(original, [TOOL], "low", earlier) < whole
    # Another model's measure, other tools or effort, or a request that does not begin with the measured one:
    # its bytes alone bound it.
    assert measured_at(script, 2, "claude-sonnet-5-5") is None and measured_at(script, 0, "gpt-test") is None
    for tools, effort in (([TOOL, TOOL], "low"), ([TOOL], "medium")):
        assert prompt_bound(original, tools, effort, earlier) == prompt_bound(original, tools, effort)
    changed = [dict(original[0], content="Another system prompt."), *original[1:]]
    assert prompt_bound(changed, [TOOL], "low", earlier) == prompt_bound(changed, [TOOL], "low")
    assert prompt_bound(original[:shared - 1], [TOOL], "low", earlier) == prompt_bound(original[:shared - 1],
                                                                                         [TOOL], "low")
    # A live call's own measure bounds the next request the same way.
    live = measured(original[:4], [TOOL], "low", 5_000)
    assert prompt_bound(original, [TOOL], "low", live) == 5_000 + request_token_bound(size(original[4:]))


def test_a_recorded_step_without_a_whole_usage_gives_no_measure(recorded):
    script, _ = recorded
    for usage in ({"output_tokens": 20}, {"input_tokens": "100"}, {"input_tokens": 100, "cache_read_tokens": -1}):
        steps = list(script.steps)
        steps[1] = replace(steps[1], reply=replace(steps[1].reply, usage=usage))
        assert measured_at(replace(script, steps=tuple(steps)), 2, "gpt-test") is None
    cached = list(script.steps)
    cached[1] = replace(cached[1], reply=replace(cached[1].reply, usage={
        "input_tokens": 10, "cache_read_tokens": 80, "cache_write_tokens": 30, "output_tokens": 20}))
    assert measured_at(replace(script, steps=tuple(cached)), 2, "gpt-test")[2] == 120


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
    # Cut off by its time limit when recorded: where a limit cuts a command depends on the host's
    # speed, so neither its exit code nor its output is compared; its similarity is still recorded,
    # and the commands after it, compared as usual, show whether the files differ.
    cut = replace(run, returncode=137, timed_out=True)
    assert compare(cut, "4 passed\n", 0, False)["divergent"] is False
    assert compare(cut, "4 passed\n", 137, True)["divergent"] is False
    found = compare(cut, "2 failed\n", 137, True)
    assert found["divergent"] is False and found["similarity"] == 0.0 and found["recorded_timed_out"] is True
    # A command the record shows finishing gets twice its limit, within the terminal's maximum;
    # one it shows cut off keeps its own, so its effects are cut off as before.
    assert rebuild_timeout(run, 600) == 10.0 and rebuild_timeout(run, 8) == 8.0
    assert rebuild_timeout(cut, 600) == 5.0


# Commands agents ran that cut a directory walk to its first lines (recorded runs of SWE-bench Pro and
# Multilingual tasks whose rebuilds on another host printed other lines).
CUT_WALKS = [
    "cd /app && find src -iname '*Unverified*' -o -iname '*DeviceMeta*'; rg \"New login|UnverifiedSession|unverified"
    " session\" src -n | head -100; rg \"interface ExtendedDevice|type ExtendedDevice|IMyDevice\" src -n | head -100",
    "cd /app && grep -R \"BOOLEAN_FLAG_TYPE\\|VARIANT_FLAG_TYPE\\|NextPageToken\" -n internal | head -80; grep -R"
    " \"type Store interface\\|type Store.*interface\" -n internal",
    "cd /app && grep -R -nE 'DecodeHooks|decode hook|mapstructure' --include='*.go' . | head -100",
    "cd /app && cat go.mod | grep -E 'otel|zap|grpc' | head -30; grep -R 'type .*Request\\|CreateFlag\\|UpdateFlag' -n"
    " rpc internal/server/middleware/grpc | head -60",
    "cd /app && rg -n \"cloudsql.*CA|CA.*cloudsql|missing.*CA|CACert\" api lib | head -30; grep -n 'sqladmin' go.mod",
    "cd /app && grep -RInE \"041|languages\" openlibrary | grep -Ei 'marc|import|language' | head -120; find . -iname"
    " '*marc*' -o -iname '*import*test*' | head -100",
    "grep -RniE 'EVALSHA|FCALL|MONITOR' src/server.c src/script* src 2>/dev/null | head -120",
    "find . -maxdepth 3 -type d | head; grep -R -n -E 'client_ip|remote_ip' caddyconfig/httpcaddyfile | head -80",
    "find test/transform -type f -name '*.java' | grep -i builder | head",
    "ls -la && grep -RIn \"replace\" crates | head -n 80",
    "cd /app && sed -n '230,275p' internal/cmd/grpc.go; grep -R \"List(ctx.*Namespace\" -n internal | head -50",
    "grep -rl TODO . | tail -5",
    "cd /app && rg --files | head -40",
    "fd -e py src | head",
    "LC_ALL=C grep -r -n x . 2>&1 | head -n 20 || true",
]
NOT_CUT_WALKS = [
    # The order is the files', or nothing cuts the walk.
    "grep -n x a.py b.py | head", "cat a.py | head -20", "ls -R | head", "git grep -n x | head",
    "grep -R x . | sort | head", "rg x | sort -u | head -5", "rg --sort path x | head", "rg x | wc -l | head",
    "grep -R x .", "find . -name '*.py'", "grep -e -r a.py | head", "grep --regexp -r a.py | head",
    # It writes, or runs something that could.
    "grep -R x . > hits.txt; head hits.txt", "grep -R x . | tee hits.txt | head", "grep -R x . >> log | head",
    "sed -i 's/a/b/' f; grep -R x . | head", "find . -name '*.pyc' -delete | head", "find . -exec rm {} + | head",
    "find . -type f | xargs grep x | head", "grep -R x . | head; rm -rf build", "rg --pre ./run x | head",
    "fd -x rm | head", "sort -o out f; grep -R x . | head", "uniq a b; grep -R x . | head",
    "git checkout -- a.py; grep -R x . | head", "python - <<'PY'\nprint(1)\nPY\ngrep -R x . | head",
    "echo $(rm x); grep -R x . | head", "echo `rm x`; grep -R x . | head", "grep -R x <(cat f) | head",
    "grep -R x . | head &", "sed -n 's/a/b/w out' f; grep -R x . | head", "awk 'NR<5' f; grep -R x . | head",
    # A newline or a comment does not hide a second command.
    "grep -R x . | head\nrm -rf y", "grep -R x . | head # top hits\nrm -rf y", "grep -R x . | head;\nmake",
    # Quotes left open, or operators read as words, are not read as a walk.
    "grep -R 'x . | head", "grep -R '|' . | head", 1, None,
]


def test_a_command_is_host_ordered_when_it_only_reads_and_cuts_a_directory_walk():
    for command in CUT_WALKS:
        assert host_ordered(command), command
    for command in NOT_CUT_WALKS:
        assert not host_ordered(command), command


def recorded_run(command, output, *, returncode=0, executed=None, timed_out=False):
    return RecordedRun(command, wrap(command) if executed is None else executed, "/app", 30.0, output, returncode,
                       timed_out, True)


def test_a_cut_off_walk_that_printed_other_lines_is_not_divergent_when_its_count_matches():
    command = "cd /app && grep -rn TODO src | head -n 3"
    run = recorded_run(command, "src/a.py:1:TODO a\nsrc/b.py:2:TODO b\nsrc/c.py:3:TODO c\n")
    other = "src/d.py:4:TODO d\nsrc/e.py:5:TODO e\nsrc/a.py:1:TODO a\n"
    row = compare(run, other, 0, False)
    assert row["similarity"] < OUTPUT_FLOOR and row["divergent"] is False and row["host_ordered"] is True
    # The same command in mini-swe-agent's environment, its variables exported first.
    exported = replace(run, executed="export PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1; "
                                     + wrap(command))
    assert compare(exported, other, 0, False)["host_ordered"] is True
    # Fewer lines, another exit code or a time-out is a difference in the files, as before.
    assert compare(run, "src/d.py:4:TODO d\nsrc/a.py:1:TODO a\n", 0, False)["divergent"] is True
    assert compare(run, other, 1, False)["divergent"] is True
    assert compare(run, other, 0, True)["divergent"] is True
    # A command that is not a cut-off walk, or that ran as something more than the agent's command, is compared
    # as every command is.
    plain = recorded_run("cat notes.txt | head -n 3", run.output)
    assert compare(plain, other, 0, False)["divergent"] is True
    wrapped = replace(run, executed="rm -rf /app/build; " + wrap(command))
    assert compare(wrapped, other, 0, False)["divergent"] is True
    # Rows of commands that match, or that differ for another reason, are the rows the rule gave before.
    same = compare(run, run.output, 0, False)
    assert same == {"returncode": 0, "recorded_returncode": 0, "timed_out": False, "recorded_timed_out": False,
                    "similarity": 1.0, "divergent": False}
    assert "host_ordered" not in compare(run, other, 1, False)


def test_the_rule_for_cut_off_walks_only_lets_through_what_diverged_by_its_lines_alone():
    # Against the rule as it was: no row it found faithful is divergent now, and a row it found divergent is
    # faithful now only when it says host_ordered.
    outputs = ["", "a.py:1:x\n", "a.py:1:x\nb.py:2:x\n", "b.py:2:x\nc.py:3:x\n", "c.py:3:x\nd.py:4:x\n",
               "Binary file e matches\n"]
    commands = ["grep -rn x . | head -2", "grep -n x a.py | head -2", "grep -rn x . > out", "find . | head -2"]
    for command in commands:
        for recorded_output in outputs:
            for recorded_code in (0, 1):
                run = recorded_run(command, recorded_output, returncode=recorded_code)
                for output in outputs:
                    for code in (0, 1):
                        for timed_out in (False, True):
                            row = compare(run, output, code, timed_out)
                            before = code != recorded_code or timed_out or row["similarity"] < OUTPUT_FLOOR
                            assert row["divergent"] <= before
                            assert (row["divergent"] != before) == bool(row.get("host_ordered"))


def test_a_rebuild_goes_on_past_a_cut_off_walk_and_still_stops_at_a_difference():
    command = "grep -rn TODO src | head -n 2"
    steps = [{"step": 1, "request_sha": "0" * 64, "messages": {"base": 0, "added": []},
              "reply": reply_to_dict(reply(command)), "text": [], "calls": [],
              "runs": [recorded_run(command, "src/a.py:1:TODO a\nsrc/b.py:2:TODO b\n").to_dict()]},
             {"step": 2, "request_sha": "1" * 64, "messages": {"base": 0, "added": []},
              "reply": reply_to_dict(reply("make test")), "text": [], "calls": [],
              "runs": [recorded_run("make test", "1 failed\n", returncode=1).to_dict()]}]
    script = ReplayScript.from_dict({"schema": SCRIPT_SCHEMA, "task": TASK, "source": {}, "exit": None,
                                     "steps": steps})

    def execute_with(tests):
        async def execute(number, run):
            return ("src/c.py:3:TODO c\nsrc/d.py:4:TODO d\n", 0, False) if number == 1 else tests
        return execute

    account = asyncio.run(rebuild(script, 2, execute_with(("1 failed\n", 1, False))))
    assert account["faithful"] and account["commands"] == 2 and account["divergent"] == 0
    assert account["rows"][0]["host_ordered"] is True and "host_ordered" not in account["rows"][1]
    account = asyncio.run(rebuild(script, 2, execute_with(("4 passed\n", 0, False))))
    assert not account["faithful"] and account["commands"] == 2 and account["rows"][-1]["divergent"]


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


def test_the_rewind_arms_agent_is_given_its_notes_and_the_call_they_go_in_at():
    notes = [{"memory_summary": "Read the parser first.", "target_record_uid": "llm-3", "reason": None}]
    rewound = BranchPolicy("/s/script.json", "a" * 64, 2, override="rewind", note=agentrewind.notebook(notes),
                           rewind={"notes": notes, "restart_usd": 0.0})
    assert rewound.agent_rewind() == {"notes": notes, "at_call": 3}
    first = BranchPolicy("/s/script.json", "a" * 64, 3, override="reject", note=NOTE,
                         rewind={"notes": [], "restart_usd": 0.0})
    assert first.agent_rewind() == {"notes": [], "at_call": None}
    assert BranchPolicy("/s/script.json", "a" * 64, 3, override="reject", note=NOTE).agent_rewind() is None
