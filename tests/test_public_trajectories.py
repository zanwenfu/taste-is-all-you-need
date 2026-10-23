"""Public mini-swe-agent trajectories: which commands ran, in order, and what each printed.

Every fixture here is synthetic, built in the shape of one published format
(no leaderboard content is copied into the repo). The formats:

- text actions (``mini-swe-agent-1`` and the undeclared July 2025 dialect):
  the command is the single fenced block in the assistant's text, its
  result the next user message;
- tool calls (``mini-swe-agent-1.1``, chat completions): ``extra.actions``
  on the assistant message, one ``tool`` message per call;
- the Responses API (``mini-swe-agent-1.1``): a response object, one
  ``function_call_output`` per call.

The last test runs the parser over real downloaded samples when
``PUBLIC_TRAJ_SURVEY_DIR`` points at them.
"""

from __future__ import annotations

import json
import os
from itertools import pairwise
from pathlib import Path

import pytest

from taste.benchmarks.public_trajectories import (
    CURRENT_SENTINEL,
    PublicTrajectory,
    RecordedCommand,
    early_sentinel_indices,
    parse_trajectory,
    replay_commands,
)

ENV = {"PAGER": "cat", "MANPAGER": "cat", "LESS": "-R", "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1"}
SUBMIT = f"echo {CURRENT_SENTINEL} && git add -A && git diff --cached"
PATCH = "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"

# -- text-action fixtures (the scaffold's v1 templates, rendered by hand) ------------------------


def say(*blocks: str, fence: str = "bash") -> str:
    """An assistant reply with one fenced block per command."""
    body = "".join(f"```{fence}\n{b}\n```\n" for b in blocks)
    return f"THOUGHT: next step.\n\n{body}"


def observation(rc: int, out: str) -> str:
    if len(out) < 10000:
        return f"<returncode>{rc}</returncode>\n<output>\n{out}</output>"
    return (
        f"<returncode>{rc}</returncode>\n<warning>\nThe output of your last command was too long.\n</warning>"
        f"<output_head>\n{out[:5000]}\n</output_head>\n<elided_chars>\n{len(out) - 10000} characters elided\n"
        f"</elided_chars>\n<output_tail>\n{out[-5000:]}\n</output_tail>"
    )


def timeout_message(command: str, partial: str) -> str:
    return (
        f"The last command <command>{command}</command> timed out and has been killed.\n"
        f"The output of the command was:\n <output>\n{partial}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )


def format_error(n: int) -> str:
    return f"Please always provide EXACTLY ONE action in triple backticks, found {n} actions.\n\nPlease format..."


def text_run(
    turns: list[tuple[str, str]], *, exit_status: str = "Submitted", submission: str = PATCH,
    version: str | None = "1.17.2", declared: bool = True, config: bool = True, prompt: str = SUBMIT,
    tail: list[dict] | None = None,
) -> dict:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Fix it. To submit run:\n```bash\n{prompt}\n```"},
    ]
    for reply, result in turns:
        messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": result}]
    messages += tail or []
    info: dict = {
        "exit_status": exit_status, "submission": submission,
        "model_stats": {"instance_cost": 0.42, "api_calls": len(turns)},
    }
    if version:
        info["mini_version"] = version
    if config:
        info["config"] = {
            "agent": {"step_limit": 250, "cost_limit": 3.0, "action_regex": r"```bash\s*\n(.*?)\n```",
                      "format_error_template": "Please always provide EXACTLY ONE action in triple backticks, "
                                               "found {{actions|length}} actions.",
                      "timeout_template": "The last command <command>{{action['action']}}</command> timed out"},
            "model": {"model_name": "provider/model"},
            "environment": {"image": "docker.io/swebench/sweb.eval.x86_64.pkg_1776_pkg-1:latest",
                            "cwd": "/testbed", "env": ENV, "forward_env": [], "timeout": 60},
            "agent_type": "minisweagent.run.extra.swebench.ProgressTrackingAgent",
            "model_type": "minisweagent.models.litellm_model.LitellmModel",
            "environment_type": "minisweagent.environments.docker.DockerEnvironment",
        }
    raw = {"info": info, "messages": messages, "instance_id": "pkg__pkg-1"}
    if declared:
        raw["trajectory_format"] = "mini-swe-agent-1"
    return raw


# -- tool-call fixtures (mini-swe-agent-1.1) --------------------------------------------------------


def tool_turn(*commands: str, start: int = 0) -> dict:
    ids = [f"call_{start + i}" for i in range(len(commands))]
    return {
        "role": "assistant", "content": "",
        "tool_calls": [{"id": i, "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": c})}}
                       for i, c in zip(ids, commands, strict=True)],
        "extra": {"actions": [{"command": c, "tool_call_id": i} for i, c in zip(ids, commands, strict=True)], "cost": 0.01},
    }


def tool_result(call_id: str, rc: int, out: str, *, exception: str = "", exception_type: str = "") -> dict:
    shown = (f"<exception>{exception}</exception>\n" if exception else "") + observation(rc, out)
    extra = {"raw_output": out, "returncode": rc, "timestamp": 1.0, "exception_info": exception}
    if exception_type:
        extra["exception_type"] = exception_type
    return {"role": "tool", "tool_call_id": call_id, "content": shown, "extra": extra}


def exit_message(status: str, submission: str = "") -> dict:
    return {"role": "exit", "content": submission or status, "extra": {"exit_status": status, "submission": submission}}


def tool_run(messages: list[dict], *, exit_status: str = "Submitted", submission: str = PATCH,
             environment: dict | None = None) -> dict:
    head = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Fix it."}]
    return {
        "info": {
            "model_stats": {"instance_cost": 1.5, "api_calls": 7},
            "config": {
                "agent": {"step_limit": 250, "cost_limit": 3.0},
                "agent_type": "minisweagent.run.benchmarks.swebench.ProgressTrackingAgent",
                "model": {"model_name": "provider/model"},
                "model_type": "minisweagent.models.litellm_model.LitellmModel",
                "environment": environment or {"image": "img", "cwd": "/testbed", "env": ENV, "timeout": 60,
                                               "interpreter": ["bash", "-c"]},
                "environment_type": "minisweagent.environments.docker.DockerEnvironment",
            },
            "mini_version": "2.1.0", "exit_status": exit_status, "submission": submission,
        },
        "messages": head + messages,
        "trajectory_format": "mini-swe-agent-1.1",
        "instance_id": "pkg__pkg-1",
    }


TIMED_OUT = "An error occurred while executing the command: Command '['docker', 'exec']' timed out after 60 seconds"


# -- text actions -----------------------------------------------------------------------------------


def test_text_actions_yield_every_executed_command_in_order_with_its_result() -> None:
    long = "".join(f"line {i}\n" for i in range(3000))
    raw = text_run([
        (say("ls -la"), observation(0, "a\nb\n")),
        ("I am thinking out loud, no command.", format_error(0)),
        (say("cat setup.py", "cat setup.cfg"), format_error(2)),
        (say("grep -rn x ."), observation(0, long)),
        (say("python -m pytest -x"), timeout_message("python -m pytest -x", "collected 12 items")),
        (say("python -c 'import sys; sys.exit(3)'"), observation(3, "")),
        (say(SUBMIT), PATCH),
    ])
    traj = parse_trajectory(raw)

    assert [c.command for c in traj.commands] == [
        "ls -la", "grep -rn x .", "python -m pytest -x", "python -c 'import sys; sys.exit(3)'", SUBMIT,
    ]
    assert [c.index for c in traj.commands] == [0, 1, 2, 3, 4]
    assert [c.returncode for c in traj.commands] == [0, 0, None, 3, None]
    assert [c.timed_out for c in traj.commands] == [False, False, True, False, False]
    assert [c.output_truncated for c in traj.commands] == [False, True, False, False, False]
    assert traj.commands[0].output == "a\nb\n"
    assert traj.commands[2].output == "collected 12 items"
    assert traj.unexecuted_turns == 2
    assert (traj.submit_index, traj.sentinel, traj.exit_status, traj.submission) == (4, CURRENT_SENTINEL, "Submitted", PATCH)
    assert traj.commands[4].output == PATCH  # what the scaffold kept: its output minus the sentinel line
    assert (traj.instance_id, traj.mini_version, traj.trajectory_format) == ("pkg__pkg-1", "1.17.2", "mini-swe-agent-1")
    assert (traj.timeout, traj.cwd, traj.env, traj.step_limit, traj.cost_limit) == (60, "/testbed", ENV, 250, 3.0)
    assert traj.environment_class == "minisweagent.environments.docker.DockerEnvironment"
    assert traj.interpreter == ("bash", "-lc")  # v1's DockerEnvironment hard-codes a login shell
    assert traj.image.endswith("pkg_1776_pkg-1:latest")
    assert (traj.api_calls, traj.cost_usd) == (7, 0.42)
    assert replay_commands(traj) == [c.command for c in traj.commands]


def test_a_truncated_text_observation_keeps_the_head_and_tail_the_scaffold_showed() -> None:
    out = "H" * 5000 + "M" * 777 + "T" * 5000
    traj = parse_trajectory(text_run([(say("cat big.log"), observation(0, out)), (say(SUBMIT), PATCH)]))
    first = traj.commands[0]
    assert first.output_truncated and first.returncode == 0
    assert first.output == "H" * 5000 + "T" * 5000


def test_a_timed_out_text_command_names_itself_and_must_be_the_parsed_one() -> None:
    raw = text_run([(say("sleep 100"), timeout_message("sleep 999", "")), (say(SUBMIT), PATCH)])
    with pytest.raises(ValueError, match="action regex"):
        parse_trajectory(raw)


def test_old_sentinels_are_rewritten_on_the_submit_command_only() -> None:
    old = "echo MINI_SWE_AGENT_FINAL_OUTPUT && git add -A && git diff --cached"
    raw = text_run(
        [(say("grep -rn MINI_SWE_AGENT_FINAL_OUTPUT docs"), observation(1, "")), (say(old), PATCH)],
        version="1.7.0", config=False, prompt=old,
    )
    traj = parse_trajectory(raw)
    assert traj.sentinel == "MINI_SWE_AGENT_FINAL_OUTPUT" and traj.submit_index == 1
    assert replay_commands(traj) == [
        "grep -rn MINI_SWE_AGENT_FINAL_OUTPUT docs",
        f"echo {CURRENT_SENTINEL} && git add -A && git diff --cached",
    ]
    # Without a recorded config nothing is invented: the replay falls back to its own defaults.
    assert (traj.timeout, traj.cwd, traj.env, traj.environment_class, traj.interpreter) == (None, "", {}, "", ())
    assert traj.step_limit is None and traj.cost_limit is None


def test_the_undeclared_micro_dialect_is_parsed_with_the_regex_its_record_proves() -> None:
    """July 2025 runs record no version. Before 22 July the scaffold also split on bare and
    trailing fences, so a stray fence after a block made three actions, not one (and nothing ran)."""
    micro = "echo MICRO_SWE_AGENT_FINAL_OUTPUT && git add -A && git diff --cached"
    stray = "THOUGHT: check.\n\n```bash\npython - << 'PY'\nprint(1)\nPY\n```\n```bash\n ```\n"
    raw = text_run(
        [(say("ls"), observation(0, "x\n")), (stray, format_error(3)), (say(micro), PATCH)],
        version=None, declared=False, config=False, prompt=micro,
    )
    traj = parse_trajectory(raw, mini_version="1.0.0")
    assert [c.command for c in traj.commands] == ["ls", micro]
    assert traj.unexecuted_turns == 1 and traj.trajectory_format == "" and traj.mini_version == "1.0.0"
    assert replay_commands(traj)[-1] == f"echo {CURRENT_SENTINEL} && git add -A && git diff --cached"


def test_a_configured_action_regex_is_the_one_used() -> None:
    raw = text_run([
        ("Run it:\n<bash_code>ls</bash_code>", observation(0, "x\n")),
        ("Two:\n<bash_code>a</bash_code><bash_code>b</bash_code>", format_error(2)),
        (f"<bash_code>{SUBMIT}</bash_code>", PATCH),
    ])
    raw["info"]["config"]["agent"]["action_regex"] = "<bash_code>(.*?)</bash_code>"
    traj = parse_trajectory(raw)
    assert [c.command for c in traj.commands] == ["ls", SUBMIT] and traj.unexecuted_turns == 1


def test_a_text_run_that_hit_its_limits_never_submitted() -> None:
    raw = text_run(
        [(say("ls"), observation(0, "x\n")), (say("pytest"), observation(1, "F\n"))],
        exit_status="LimitsExceeded", submission="", tail=[{"role": "user", "content": ""}],
    )
    traj = parse_trajectory(raw)
    assert traj.submit_index is None and traj.sentinel is None and traj.exit_status == "LimitsExceeded"
    assert replay_commands(traj) == ["ls", "pytest"]


@pytest.mark.parametrize("mutate, message", [
    (lambda raw: raw["messages"].pop(), "no recorded result"),                    # crash mid-command
    (lambda raw: raw["messages"].__setitem__(3, {"role": "user", "content": format_error(1)}), "action regex"),
    (lambda raw: raw["messages"].__setitem__(2, {"role": "assistant", "content": say("a", "b")}), "action regex"),
    (lambda raw: raw["info"].__setitem__("submission", "something else"), "submission"),
])
def test_text_structures_that_cannot_be_read_unambiguously_are_refused(mutate, message) -> None:
    raw = text_run([(say("ls"), observation(0, "x\n")), (say(SUBMIT), PATCH)])
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        parse_trajectory(raw)


# -- tool calls -------------------------------------------------------------------------------------


def test_tool_calls_keep_several_per_turn_in_order_and_pair_results_by_id() -> None:
    raw = tool_run([
        tool_turn("ls", "cat a.py", "cat b.py", start=0),
        tool_result("call_0", 0, "a.py\nb.py\n"), tool_result("call_1", 0, "x = 1\n"), tool_result("call_2", 1, "missing\n"),
        {"role": "user", "content": "Tool call error: no tool calls", "extra": {"interrupt_type": "FormatError"}},
        tool_turn("pip install -e .", start=3),
        tool_result("call_3", -1, "", exception=TIMED_OUT, exception_type="TimeoutExpired"),
        tool_turn("cat big.log", start=4), tool_result("call_4", 0, "y" * 12000),
        tool_turn(f"echo {CURRENT_SENTINEL} && cat patch.txt", start=5),
        exit_message("Submitted", PATCH),
    ])
    traj = parse_trajectory(raw)
    assert [c.command for c in traj.commands] == [
        "ls", "cat a.py", "cat b.py", "pip install -e .", "cat big.log", f"echo {CURRENT_SENTINEL} && cat patch.txt",
    ]
    assert [c.returncode for c in traj.commands] == [0, 0, 1, None, 0, 0]
    assert [c.timed_out for c in traj.commands] == [False, False, False, True, False, False]
    assert [c.output_truncated for c in traj.commands] == [False, False, False, False, True, False]
    assert traj.commands[4].output == "y" * 12000  # the full raw output, though the model saw head and tail
    assert traj.unexecuted_turns == 1 and traj.submit_index == 5 and traj.sentinel == CURRENT_SENTINEL
    assert traj.interpreter == ("bash", "-c") and traj.trajectory_format == "mini-swe-agent-1.1"
    assert (traj.api_calls, traj.cost_usd, traj.mini_version) == (7, 1.5, "2.1.0")


def test_results_out_of_order_or_missing_are_refused() -> None:
    swapped = tool_run([
        tool_turn("a", "b"), tool_result("call_1", 0, ""), tool_result("call_0", 0, ""),
        tool_turn(SUBMIT, start=2), exit_message("Submitted", PATCH),
    ])
    with pytest.raises(ValueError, match="tool_call_id"):
        parse_trajectory(swapped)
    crashed = tool_run([tool_turn("make"), exit_message("RuntimeError")], exit_status="RuntimeError", submission="")
    with pytest.raises(ValueError, match="no recorded result"):
        parse_trajectory(crashed)


def test_recorded_actions_must_be_the_tool_calls_the_model_made() -> None:
    turn = tool_turn("ls")
    turn["extra"]["actions"][0]["command"] = "ls -la"
    raw = tool_run([turn, tool_result("call_0", 0, ""), tool_turn(SUBMIT, start=1), exit_message("Submitted", PATCH)])
    with pytest.raises(ValueError, match="tool calls"):
        parse_trajectory(raw)


def test_in_the_submitting_turn_earlier_calls_ran_unrecorded_and_later_ones_never_ran() -> None:
    raw = tool_run([tool_turn("git diff > patch.txt", SUBMIT, "rm -rf /tmp/x"), exit_message("Submitted", PATCH)])
    traj = parse_trajectory(raw)
    assert [c.command for c in traj.commands] == ["git diff > patch.txt", SUBMIT]
    assert traj.commands[0].returncode is None and traj.commands[0].output == ""
    assert traj.submit_index == 1 and replay_commands(traj) == ["git diff > patch.txt", SUBMIT]


def test_a_tool_run_that_hit_its_limits_never_submitted() -> None:
    raw = tool_run(
        [tool_turn("ls"), tool_result("call_0", 0, "x\n"), exit_message("LimitsExceeded")],
        exit_status="LimitsExceeded", submission="",
    )
    traj = parse_trajectory(raw)
    assert traj.submit_index is None and traj.sentinel is None and replay_commands(traj) == ["ls"]


def test_the_responses_api_shape_reads_the_same() -> None:
    def response(call_id: str, command: str) -> dict:
        return {
            "id": f"resp_{call_id}", "object": "response", "status": "completed",
            "output": [{"type": "reasoning", "id": "rs", "summary": []},
                       {"type": "function_call", "id": "fc", "call_id": call_id, "name": "bash",
                        "arguments": json.dumps({"command": command})}],
            "extra": {"actions": [{"command": command, "tool_call_id": call_id}], "cost": 0.02},
        }

    def output(call_id: str, rc: int, out: str, exception: str = "", exception_type: str = "") -> dict:
        extra = {"raw_output": out, "returncode": rc, "timestamp": 1.0, "exception_info": exception}
        if exception_type:
            extra["exception_type"] = exception_type
        return {"type": "function_call_output", "call_id": call_id, "output": observation(rc, out), "extra": extra}

    raw = tool_run([
        response("c1", "ls"), output("c1", 0, "x\n"),
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Tool call error"}],
         "extra": {"interrupt_type": "FormatError"}},
        response("c2", "sleep 90"), output("c2", -1, "", TIMED_OUT, "TimeoutExpired"),
        response("c3", SUBMIT), exit_message("Submitted", PATCH),
    ])
    raw["info"]["config"]["model_type"] = "minisweagent.models.litellm_response_model.LitellmResponseModel"
    traj = parse_trajectory(raw)
    assert [c.command for c in traj.commands] == ["ls", "sleep 90", SUBMIT]
    assert [c.returncode for c in traj.commands] == [0, None, 0]
    assert traj.commands[1].timed_out and traj.unexecuted_turns == 1 and traj.submit_index == 2


# -- whole-file refusals and helpers ----------------------------------------------------------------


def test_swe_agent_trajectories_and_unknown_formats_are_refused_loudly() -> None:
    swe_agent = {"trajectory": [{"action": "ls", "observation": "x"}], "history": [], "info": {"swe_agent_version": "1.0.1"},
                 "replay_config": "{}", "environment": "pkg__pkg-1"}
    with pytest.raises(ValueError, match="SWE-agent"):
        parse_trajectory(swe_agent)
    unknown = tool_run([tool_turn(SUBMIT), exit_message("Submitted", PATCH)])
    unknown["trajectory_format"] = "mini-swe-agent-2"
    with pytest.raises(ValueError, match="trajectory_format"):
        parse_trajectory(unknown)


def test_commands_that_could_end_a_replay_early_are_named() -> None:
    raw = text_run([
        (say(SUBMIT), timeout_message(SUBMIT, f"{CURRENT_SENTINEL}\n")),  # timed out: the original went on
        (say("ls"), observation(0, "x\n")),
        (say(SUBMIT), PATCH),
    ])
    traj = parse_trajectory(raw)
    assert traj.submit_index == 2 and early_sentinel_indices(traj) == (0,)


def test_a_hand_built_trajectory_without_a_known_sentinel_will_not_replay_a_submission() -> None:
    commands = (RecordedCommand(index=0, command="cat out.txt", returncode=None, output=PATCH,
                                output_truncated=False, timed_out=False),)
    traj = PublicTrajectory(
        instance_id="pkg__pkg-1", mini_version="1.7.0", trajectory_format="mini-swe-agent-1", exit_status="Submitted",
        submission=PATCH, commands=commands, submit_index=0, sentinel=None, unexecuted_turns=0, timeout=60,
        env={}, cwd="/testbed", environment_class="", step_limit=None, cost_limit=None, api_calls=1, cost_usd=0.0,
    )
    with pytest.raises(ValueError, match="sentinel"):
        replay_commands(traj)


# -- real samples ---------------------------------------------------------------------------------


SURVEY = os.environ.get("PUBLIC_TRAJ_SURVEY_DIR", "")


def _recorded_results(raw: dict) -> tuple[int, int]:
    """(results recorded, format errors recorded), counted from the file without the parser's rules."""
    messages = raw["messages"]

    def text(m: dict) -> str:
        c = m.get("content") if m.get("content") is not None else m.get("output", "")
        return c if isinstance(c, str) else "".join(b.get("text", "") for b in c)

    submitted = 1 if raw["info"].get("exit_status") == "Submitted" else 0
    if raw.get("trajectory_format") == "mini-swe-agent-1.1":
        results = sum(1 for m in messages if m.get("role") == "tool" or m.get("type") == "function_call_output")
        errors = sum(1 for m in messages if (m.get("extra") or {}).get("interrupt_type") == "FormatError")
        return results + submitted, errors
    after_reply = [text(m) for prev, m in pairwise(messages) if prev.get("role") == "assistant"]
    results = sum(1 for t in after_reply if t.startswith(("<returncode>", "The last command")))
    errors = sum(1 for t in after_reply if t.startswith("Please always provide EXACTLY ONE action"))
    return results + submitted, errors


@pytest.mark.skipif(not SURVEY or not Path(SURVEY).is_dir(), reason="set PUBLIC_TRAJ_SURVEY_DIR to the downloaded samples")
def test_every_downloaded_sample_parses_to_one_command_per_recorded_result() -> None:
    files = sorted(p for p in Path(SURVEY).iterdir() if p.name.endswith((".traj.json", ".traj")))
    assert files
    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "trajectory" in raw and "messages" not in raw:
            with pytest.raises(ValueError, match="SWE-agent"):
                parse_trajectory(raw)
            continue
        traj = parse_trajectory(raw)
        results, errors = _recorded_results(raw)
        assert len(traj.commands) == results, path.name
        assert traj.unexecuted_turns == errors, path.name
        assert (traj.submit_index is not None) == (traj.exit_status == "Submitted"), path.name
        if traj.submit_index is not None:
            assert traj.submit_index == len(traj.commands) - 1, path.name
            assert CURRENT_SENTINEL in replay_commands(traj)[-1], path.name
