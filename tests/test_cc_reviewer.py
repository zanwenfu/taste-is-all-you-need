"""The Claude Code reviewer: the checker's review run as ``claude -p`` on the subscription, confined to one container.

No model and no network: ``claude`` is a fake that plays scripted runs (``tests.cc_fake_claude``) through the real
hook, and ``docker`` is a stand-in that answers from a folder and runs nothing (``tests.cc_docker_shim``).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from taste.agents import ShellResult
from taste.agents.checker import (
    REFUSED,
    SCHEMA,
    CheckerSettings,
    _observation,
    checker_task,
    parse_submission,
    system_prompt,
)
from taste.recovery_study import cc_review_driver
from taste.recovery_study.cc_reviewer import (
    CORRECTION,
    LAST_STEP,
    NO_STEPS,
    TOKEN_NAME,
    VERDICT_SCHEMA,
    ReviewFailed,
    ReviewSettings,
    Runner,
    StopJob,
    allow_rule,
    child_env,
    claude_settings,
    command_from_script,
    command_line,
    container_command,
    container_script,
    read_token,
    redact,
    reviewer_system_prompt,
    reviewer_task,
)
from tests import cc_docker_shim, cc_fake_claude

TOKEN = "fake-oauth-token-0123456789abcdef"
# A key-shaped fake, built at run time so that no key-shaped text is in the repository.
KEYISH = "sk-" + "ant-api03-fake"
TASK = checker_task("Make add() return the sum of its two arguments.", "I fixed add.", "diff --git a/calc.py ...")
DONE = {"verdict": "done", "confidence": 0.9}
NOT_DONE = {"verdict": "not_done", "confidence": 0.8, "unmet_requirement": "add() must return the sum.",
            "evidence": {"check": "Called add(2, 3).", "command": "python3 -c 'from calc import add; print(add(2,3))'",
                         "output": "-1", "expected": "5", "observed": "-1"}}


def success(structured, turns=2, cost=0.004):
    return {"subtype": "success", "is_error": False, "num_turns": turns + 1, "total_cost_usd": cost,
            "structured_output": structured, "result": json.dumps(structured), "duration_ms": 1000}


@pytest.fixture
def kit(tmp_path):
    """A bin folder with the fake claude and the docker stand-in first on PATH, a toy folder, a runner."""
    bin_dir, toy = tmp_path / "bin", tmp_path / "toy"
    bin_dir.mkdir()
    toy.mkdir()
    (toy / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (toy / "README.md").write_text("add() returns the sum.\n")
    cc_docker_shim.install(bin_dir, toy, tmp_path / "docker.jsonl")
    environ = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "LANG": "C.UTF-8",
               "ANTHROPIC_API_KEY": KEYISH, "AWS_SECRET_ACCESS_KEY": "nope",
               "CLAUDE_CODE_USE_BEDROCK": "1", "OPENAI_API_KEY": "nope", "ANTHROPIC_BASE_URL": "http://x"}

    class Kit:
        def __init__(self):
            self.tmp, self.bin, self.toy, self.environ = tmp_path, bin_dir, toy, environ

        def runner(self, scenario, **settings):
            cc_fake_claude.install(bin_dir, scenario)
            return Runner(TOKEN, ReviewSettings(claude=str(bin_dir / "claude"), docker="docker", **settings),
                          environ=environ)

        def calls(self):
            path = bin_dir / "calls.jsonl"
            return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

        def docker_calls(self):
            path = tmp_path / "docker.jsonl"
            return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

        def review(self, scenario, **settings):
            return self.runner(scenario, **settings).review("c0ffee", "/app", TASK, tmp_path / "work")

    return Kit()


def no_token_anywhere(root):
    for path in Path(root).rglob("*"):
        if path.is_file():
            assert TOKEN.encode() not in path.read_bytes(), f"the token is in {path}"


# -- what the reviewer is told ---------------------------------------------------------------------------------


def test_the_prompts_are_the_checkers_but_for_the_verdict_tools_name():
    settings = ReviewSettings()
    mine = reviewer_system_prompt("/app", settings).splitlines()
    theirs = system_prompt("/app", CheckerSettings()).splitlines()
    changed = [(a, b) for a, b in zip(mine, theirs, strict=True) if a != b]
    assert changed == [("When you have decided, call StructuredOutput once with:",
                        "When you have decided, call submit_verdict once with:")]
    task = reviewer_task(TASK)
    assert task.replace("call StructuredOutput.", "call submit_verdict.") == TASK
    assert VERDICT_SCHEMA["required"] == ["verdict", "confidence"]
    assert "StructuredOutput" in LAST_STEP and "StructuredOutput" in NO_STEPS and "{error}" in CORRECTION


def test_version_2_prompt_is_kept_too():
    mine = reviewer_system_prompt("/testbed", ReviewSettings(checker_version="2"))
    assert "fails because of the task, not the work" in mine and "call StructuredOutput once" in mine


# -- the container command ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("command,output,code", [
    ("printf 'hello\\n'", "hello\n", 0),
    ("true", "", 0),
    ("echo out; echo err >&2; exit 3", "out\nerr\n", 3),
    ("i=0; while [ $i -lt 400 ]; do echo line$i; i=$((i+1)); done",
     "".join(f"line{i}\n" for i in range(400)), 0),
    ("cat <<'EOF'\nheredoc 'quoted' \"too\"\nEOF", "heredoc 'quoted' \"too\"\n", 0),
])
def test_the_container_script_shows_what_the_checker_shows(command, output, code):
    settings = ReviewSettings(output_chars=500)
    done = subprocess.run(["/bin/sh", "-c", container_script(command, settings, timeout=False)],
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0
    expected = _observation(ShellResult(output, code), CheckerSettings(output_chars=500))
    assert done.stdout.rstrip("\n") == expected.rstrip("\n")


def test_the_container_script_reports_a_command_at_its_time_limit(tmp_path):
    fake = tmp_path / "timeout"
    fake.write_text("#!/bin/sh\nsleep 1\nprintf 'partial\\n'\nexit 124\n")
    fake.chmod(0o755)
    settings = ReviewSettings(command_seconds=1)
    done = subprocess.run(["/bin/sh", "-c", container_script("sleep 60", settings)], capture_output=True, text=True,
                          timeout=30, env={"PATH": f"{tmp_path}:/usr/bin:/bin"})
    assert done.stdout == "exit code: none (ended after 1 seconds; its output so far is below)\noutput:\npartial\n"


@pytest.mark.parametrize("command", ["ls -la", "cat 'a b.txt' \"c\"", "echo $(whoami) `id` ${HOME}",
                                     "python3 - <<'EOF'\nprint('x')\nEOF", "cd /x; ls | wc -l && echo it's"])
def test_the_command_is_read_back_from_its_script(command):
    assert command_from_script(container_script(command, ReviewSettings())) == command
    assert command_from_script(container_script(command, ReviewSettings(), timeout=False)) == command
    assert command_from_script("echo hi") is None


def test_every_command_is_one_docker_exec_into_the_container():
    line = container_command("cat a; ls ~ $(id)", "c0ffee", "/app", ReviewSettings())
    rule = allow_rule("c0ffee", "/app")
    assert rule == "Bash(docker exec -w /app c0ffee /bin/sh -c *)"
    assert line.startswith("docker exec -w /app c0ffee /bin/sh -c '")
    # The whole script is one single-quoted word: nothing in it is expanded or run by the host's shell.
    import shlex
    words = shlex.split(line)
    assert words[:7] == ["docker", "exec", "-w", "/app", "c0ffee", "/bin/sh", "-c"] and len(words) == 8
    assert allow_rule("c0ffee", "/app", user="1000") == "Bash(docker exec -w /app -u 1000 c0ffee /bin/sh -c *)"


# -- the token ----------------------------------------------------------------------------------------------------


def test_the_token_is_read_from_its_one_line(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"ANTHROPIC_API_KEY={KEYISH}\nexport {TOKEN_NAME}='{TOKEN}'\nOTHER=1\n")
    assert read_token(env_file, environ={}) == TOKEN
    assert read_token(None, environ={TOKEN_NAME: "from-env"}) == "from-env"
    env_file.write_text(f"ANTHROPIC_API_KEY={KEYISH}\n")
    with pytest.raises(ValueError) as caught:
        read_token(env_file, environ={})
    assert KEYISH not in str(caught.value)


def test_the_token_goes_only_in_the_environment(tmp_path):
    settings = ReviewSettings()
    paths = {"settings": tmp_path / "s.json", "system": tmp_path / "p.txt"}
    args = command_line(settings, paths, container="c0ffee", cwd="/app", user=None, session="s-1")
    assert TOKEN not in " ".join(args) and "--bare" not in args and "--dangerously-skip-permissions" not in args
    env = child_env(TOKEN, home=tmp_path, config=tmp_path, settings=settings,
                    environ={"PATH": "/bin", "ANTHROPIC_API_KEY": KEYISH, "ANTHROPIC_AUTH_TOKEN": "t",
                             "ANTHROPIC_BASE_URL": "http://x", "CLAUDE_CODE_USE_BEDROCK": "1",
                             "AWS_PROFILE": "p", "OPENAI_API_KEY": "o", "CLAUDE_CODE_RETRY_WATCHDOG": "1",
                             "CLAUDE_CODE_EFFORT_LEVEL": "max"})
    assert env[TOKEN_NAME] == TOKEN and [k for k, v in env.items() if v == TOKEN] == [TOKEN_NAME]
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK",
                 "AWS_PROFILE", "OPENAI_API_KEY", "CLAUDE_CODE_RETRY_WATCHDOG", "CLAUDE_CODE_EFFORT_LEVEL"):
        assert name not in env
    assert env["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"] == "1" and env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"
    assert redact(f"x {TOKEN} {KEYISH}_DEF y", TOKEN) == "x [redacted] [redacted] y"


def test_the_settings_allow_bash_only_as_docker_exec(tmp_path):
    settings = claude_settings("c0ffee", "/app", None, tmp_path / "hook.json", ReviewSettings())
    assert settings["permissions"]["allow"] == ["Bash(docker exec -w /app c0ffee /bin/sh -c *)"]
    assert {"Read", "Edit", "Write", "WebFetch", "WebSearch", "mcp__*"} <= set(settings["permissions"]["deny"])
    assert settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"].endswith("|| exit 2")
    assert TOKEN not in json.dumps(settings)


# -- one review ---------------------------------------------------------------------------------------------------


def test_a_review_runs_its_commands_in_the_container_and_gives_the_checkers_record(kit):
    scenario = [{"execute": True, "turns": [[{"name": "Bash", "input": {"command": "cat calc.py",
                                                                         "description": "read"}}],
                                            [{"name": "StructuredOutput", "input": DONE}]],
                 "result": success(DONE)}]
    review = kit.review(scenario)
    reply = review.reply
    assert parse_submission(json.dumps(reply)) == reply
    assert reply["schema"] == SCHEMA and reply["verdict"] == "done" and reply["ended"] == "verdict"
    assert (reply["steps"], reply["commands"], reply["refused"]) == (2, 1, 0)
    assert reply["tokens"] == {"input": 2 * 10_100, "output": 100} and reply["cost_usd"] == 0.004
    assert review.hook["token_in_env"] is False
    (call,) = kit.calls()
    assert call["token_present"] is True and TOKEN not in json.dumps(call["argv"])
    assert "ANTHROPIC_API_KEY" not in call["env_keys"] and "AWS_SECRET_ACCESS_KEY" not in call["env_keys"]
    assert call["prompt"].startswith("A coding agent was given the task below")
    args = call["argv"]
    for flag, value in (("--model", "claude-haiku-5-5"), ("--max-turns", "30"), ("--tools", "Bash"),
                        ("--output-format", "json"), ("--effort", "medium"), ("--max-budget-usd", "2"),
                        ("--allowedTools", "Bash(docker exec -w /app c0ffee /bin/sh -c *)")):
        assert args[args.index(flag) + 1] == value
    rewritten = call["hooks"][0]["out"]["hookSpecificOutput"]["updatedInput"]
    assert rewritten["command"].startswith("docker exec -w /app c0ffee /bin/sh -c ")
    # The hook approves exactly the command it confined to the container (Nov 15), and nothing it denies.
    assert call["hooks"][0]["out"]["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert rewritten["timeout"] == 150_000 and rewritten["run_in_background"] is False
    assert call["hooks"][0]["response"].startswith("exit code: 0\noutput:\ndef add(a, b):")
    # Preflight, then cat; and when the container started, read before and after the review.
    assert [entry.get("command") for entry in kit.docker_calls() if entry["argv0"] == ["exec"]] == [None, "cat calc.py"]
    assert [entry.get("inspect") for entry in kit.docker_calls() if entry["argv0"] == ["inspect"]] == ["c0ffee", "c0ffee"]
    no_token_anywhere(kit.tmp)


def test_commands_that_name_hidden_tests_are_refused_and_never_run(kit):
    scenario = [{"execute": True, "turns": [[{"name": "Bash", "input": {"command": "cat /tests/test_calc.py"}},
                                             {"name": "Bash", "input": {"command": "ls", "run_in_background": True}},
                                             {"name": "Bash", "input": {"command": "  "}},
                                             {"name": "Read", "input": {"file_path": "/etc/hosts"}}],
                                            [{"name": "StructuredOutput", "input": DONE}]],
                 "result": success(DONE)}]
    review = kit.review(scenario)
    assert (review.reply["commands"], review.reply["refused"]) == (0, 1)
    assert (review.hook["denied"], review.hook["other_tools"]) == (2, 1)
    responses = [entry["response"] for entry in kit.calls()[0]["hooks"] if entry["event"] == "PreToolUse"]
    assert responses[0] == REFUSED.format(name="/tests")
    assert "foreground" in responses[1] and "needs a command" in responses[2] and "no tool named" in responses[3]
    assert [entry for entry in kit.docker_calls() if entry.get("command")] == []


def test_the_last_step_is_announced_and_its_commands_are_not_run(kit):
    bash = [{"name": "Bash", "input": {"command": "ls"}}]
    scenario = [{"execute": True, "turns": [bash, bash, bash],
                 "result": {"subtype": "error_max_turns", "is_error": True, "num_turns": 4, "total_cost_usd": 0.01}}]
    review = kit.review(scenario, max_steps=3)
    hooks = kit.calls()[0]["hooks"]
    posts = [entry["out"] for entry in hooks if entry["event"] == "PostToolBatch"]
    assert posts[0] == {} and posts[1]["hookSpecificOutput"]["additionalContext"] == LAST_STEP
    assert posts[2] == {"continue": False, "stopReason": "no steps are left"}
    last = [entry for entry in hooks if entry["event"] == "PreToolUse"][-1]
    assert last["response"] == NO_STEPS
    reply = review.reply
    assert reply["verdict"] is None and reply["confidence"] is None and reply["ended"] == "steps"
    assert (reply["steps"], reply["commands"]) == (3, 2)
    assert kit.calls()[0]["argv"][kit.calls()[0]["argv"].index("--max-turns") + 1] == "3"


def test_the_turn_limit_and_the_budget_end_a_review_without_a_verdict(kit):
    review = kit.review([{"turns": [], "result": {"subtype": "error_max_turns", "is_error": True, "num_turns": 31}}])
    assert review.reply["ended"] == "steps" and review.reply["verdict"] is None
    kit.tmp.joinpath("bin", "count").unlink()
    review = kit.review([{"turns": [], "result": {"subtype": "error_max_budget_usd", "is_error": True}}])
    assert review.reply["ended"] == "usd"


def test_the_time_limit_stops_the_whole_process_group(kit):
    started = time.monotonic()
    review = kit.review([{"turns": [], "sleep": 60, "result": success(DONE)}], max_seconds=1, kill_grace_seconds=1)
    assert time.monotonic() - started < 30
    assert review.reply["ended"] == "seconds" and review.reply["verdict"] is None
    assert review.claude_code["runs"][0]["timed_out"] is True


def test_a_verdict_the_checker_would_not_accept_is_sent_back_once_corrected(kit):
    fenced = {**NOT_DONE, "unmet_requirement": "```python\nreturn a + b\n```"}
    scenario = [{"turns": [[{"name": "StructuredOutput", "input": fenced}]], "result": success(fenced, 1)},
                {"turns": [[{"name": "StructuredOutput", "input": NOT_DONE}]], "result": success(NOT_DONE, 2, 0.006)}]
    review = kit.review(scenario)
    first, second = kit.calls()
    assert "--session-id" in first["argv"] and "--resume" in second["argv"]
    session = first["argv"][first["argv"].index("--session-id") + 1]
    assert second["argv"][second["argv"].index("--resume") + 1] == session
    assert second["prompt"].startswith("Your verdict was not accepted: unmet_requirement must say what is wrong")
    assert second["argv"][second["argv"].index("--max-turns") + 1] == "29"
    assert review.reply["verdict"] == "not_done" and review.reply["evidence"]["observed"] == "-1"
    assert review.reply["steps"] == 2 and review.claude_code["corrections"] == 1 and review.reply["cost_usd"] == 0.006


@pytest.mark.parametrize("result,stderr,kind", [
    ({"subtype": "success", "is_error": True, "result": "You've hit your session limit · resets 7pm"}, "",
     "usage_limit"),
    ({"subtype": "success", "is_error": True, "result": "API Error: Request rejected (429)"}, "", "rate_limit"),
    ({"subtype": "error_during_execution", "is_error": True, "result": "x", "api_error_status": 429}, "",
     "rate_limit"),
    (None, "Invalid auth token · Fix external auth token", "auth"),
    ({"subtype": "success", "is_error": True, "result": "Credit balance is too low"}, "", "spend_limit"),
])
def test_a_limit_or_a_login_failure_stops_the_job(kit, result, stderr, kind):
    run = {"turns": [], "exit": 1, "stderr": stderr}
    if result is not None:
        run["result"] = result
    with pytest.raises(StopJob) as caught:
        kit.review([run])
    assert caught.value.kind == kind and TOKEN not in str(caught.value)


def test_other_errors_fail_the_review_without_stopping(kit):
    with pytest.raises(ReviewFailed):
        kit.review([{"turns": [], "exit": 1, "result": {"subtype": "error_during_execution", "is_error": True,
                                                         "result": "Repeated 529 Overloaded errors"}}])


def test_a_tool_call_the_hook_never_saw_stops_the_job(kit):
    scenario = [{"skip_hook": True, "turns": [[{"name": "Bash", "input": {"command": "ls"}}]],
                 "result": success(DONE)}]
    with pytest.raises(StopJob) as caught:
        kit.review(scenario)
    assert caught.value.kind == "hook_bypassed"
    # A run that then fails does not hide it.
    kit.tmp.joinpath("bin", "count").unlink()
    with pytest.raises(StopJob) as caught:
        kit.review([{**scenario[0], "exit": 1, "result": {"subtype": "error_during_execution", "is_error": True,
                                                          "result": "boom"}}])
    assert caught.value.kind == "hook_bypassed"


def test_a_container_without_timeout_gets_commands_without_it(kit, tmp_path):
    runner = kit.runner([{"turns": [[{"name": "Bash", "input": {"command": "ls"}}],
                                    [{"name": "StructuredOutput", "input": DONE}]], "result": success(DONE)}])
    plain = tmp_path / "bin" / "docker"
    plain.write_text("#!/bin/sh\nexit 0\n")
    review = runner.review("c0ffee", "/app", TASK, tmp_path / "work")
    rewritten = kit.calls()[0]["hooks"][0]["out"]["hookSpecificOutput"]["updatedInput"]["command"]
    assert "timeout -k" not in rewritten and review.claude_code["container_timeout"] is False
    assert command_from_script(__import__("shlex").split(rewritten)[-1]) == "ls"


def test_a_container_that_is_not_there_fails_before_any_model_call(kit, tmp_path):
    runner = kit.runner([{"turns": [], "result": success(DONE)}])
    broken = tmp_path / "bin" / "docker"
    broken.write_text("#!/bin/sh\necho 'Error: No such container' >&2\nexit 1\n")
    with pytest.raises(ReviewFailed):
        runner.review("c0ffee", "/app", TASK, tmp_path / "work")
    assert kit.calls() == []


TWO_COMMANDS = [{"execute": True, "turns": [[{"name": "Bash", "input": {"command": "cat calc.py"}}],
                                            [{"name": "Bash", "input": {"command": "ls"}}],
                                            [{"name": "StructuredOutput", "input": DONE}]], "result": success(DONE, 3)}]


def test_a_review_whose_commands_reached_the_container_is_kept(kit):
    review = kit.review(TWO_COMMANDS)
    assert review.reply["verdict"] == "done" and review.reply["commands"] == 2
    assert [entry.get("inspect") for entry in kit.docker_calls() if "inspect" in entry] == ["c0ffee", "c0ffee"]


def test_a_daemon_lost_mid_review_leaves_no_verdict(kit):
    # Preflight (1st exec) and the first command (2nd) reach the container; from the 3rd on Docker answers with its own
    # error. The model still gives a verdict, which the harness would accept: the review must not be kept.
    (kit.tmp / "docker.jsonl.fail_from").write_text("3")
    with pytest.raises(ReviewFailed, match="transport_failed: 1 of 2 commands"):
        kit.review(TWO_COMMANDS)
    assert [entry.get("failed") for entry in kit.docker_calls() if entry.get("argv0") == ["exec"]] == [None, None, True]


def test_a_container_restarted_mid_review_leaves_no_verdict(kit):
    (kit.tmp / "docker.jsonl.started").write_text("2026-11-15T01:00:00Z\n2026-11-15T03:30:00Z\n")
    with pytest.raises(ReviewFailed, match="container_restarted"):
        kit.review(TWO_COMMANDS)


def test_a_container_that_cannot_be_inspected_fails_before_any_model_call(kit, tmp_path):
    runner = kit.runner(TWO_COMMANDS)
    docker = tmp_path / "bin" / "docker"
    docker.write_text("#!/bin/sh\nif [ \"$1\" = inspect ]; then echo 'Error: No such object' >&2; exit 1; fi\n"
                      "echo has:timeout\nexit 0\n")
    with pytest.raises(ReviewFailed, match="could not be inspected"):
        runner.review("c0ffee", "/app", TASK, tmp_path / "work")
    assert kit.calls() == []


def test_a_review_whose_commands_have_no_tool_use_ids_fails_closed(kit):
    # If Claude Code sent its hook no tool-use IDs, no result could be checked: the review is not kept.
    with pytest.raises(ReviewFailed, match="transport_unverified: 2 commands went into the container, the hook knows 0"):
        kit.review([{**TWO_COMMANDS[0], "no_ids": True}])


def test_a_verdict_whose_commands_have_no_result_in_the_transcript_fails_closed(kit):
    with pytest.raises(ReviewFailed, match="transport_unverified: 2 of 2 commands have no result"):
        kit.review([{**TWO_COMMANDS[0], "no_results": True}])


def test_only_the_token_line_of_an_env_file_reaches_the_reviewer(kit, tmp_path):
    # The run reads the repository's own .env, which holds other keys too: only the token's value is taken, and only
    # into the claude process's environment; no other variable of the file or of this process reaches it.
    others = {"ANTHROPIC_API_KEY": KEYISH + "-other", "AWS_ACCESS_KEY_ID": "AKIA" + "FAKEFAKEFAKEFAKE",
              "AWS_SECRET_ACCESS_KEY": "fake-aws-secret-0123456789", "AWS_BEARER_TOKEN_BEDROCK": "fake-bedrock-0123456789",
              "OPENAI_API_KEY": "sk-" + "proj-fake-0123456789"}
    env_file = tmp_path / "dot.env"
    env_file.write_text("".join(f"{k}={v}\n" for k, v in others.items()) + f"export CLAUDE_CODE_OAUTH_TOKEN='{TOKEN}'\n"
                        + "AWS_DEFAULT_REGION=us-east-1\n")
    token = read_token(str(env_file), environ={})
    assert token == TOKEN
    cc_fake_claude.install(kit.bin, TWO_COMMANDS)
    runner = Runner(token, ReviewSettings(claude=str(kit.bin / "claude"), docker="docker"), environ=kit.environ)
    runner.review("c0ffee", "/app", TASK, tmp_path / "work")
    (call,) = kit.calls()
    assert call["token_present"] is True
    leaked = [k for k in call["env_keys"] if k.startswith(("ANTHROPIC_", "AWS_", "OPENAI_"))]
    assert leaked == [], leaked
    for path in tmp_path.rglob("*"):
        if path.is_file() and path != env_file:
            data = path.read_bytes()
            assert not any(value.encode() in data for value in others.values()), f"a value of the env file is in {path}"
    no_token_anywhere(tmp_path / "work")


def test_the_driver_keeps_a_lost_review_as_a_failure_to_retry(kit, tmp_path):
    (kit.tmp / "docker.jsonl.fail_from").write_text("3")
    runner = kit.runner(TWO_COMMANDS)
    items, results = manifest(tmp_path, 1), tmp_path / "results.jsonl"
    summary = cc_review_driver.drive(items, results, tmp_path / "w", runner, concurrency=1, log=lambda *_: None)
    record = cc_review_driver.read_results(results)["run-0"]
    assert summary["failed"] == 1 and record["reply"] is None and "transport_failed" in record["error"]
    assert cc_review_driver.verdicts(results) == {"done": 0, "not_done": 0, "none": 0, "failed": 1}


# -- the driver ---------------------------------------------------------------------------------------------------


def manifest(tmp_path, n):
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps({"run": f"run-{i}", "container": f"c{i}", "cwd": "/app",
                                        "task": "Make add() return the sum.", "final_message": "done",
                                        "submission": ""}) + "\n" for i in range(n)))
    return cc_review_driver.load_manifest(path)


def test_the_driver_resumes_where_it_stopped_and_labels_the_family(kit, tmp_path):
    runner = kit.runner([{"turns": [[{"name": "StructuredOutput", "input": DONE}]], "result": success(DONE, 1)}])
    items, results = manifest(tmp_path, 3), tmp_path / "out" / "results.jsonl"
    first = cc_review_driver.drive(items, results, tmp_path / "w", runner, limit=2, log=lambda *_: None)
    assert first["reviewed"] == 2, first
    assert len(kit.calls()) == 2, kit.calls()
    second = cc_review_driver.drive(items, results, tmp_path / "w", runner, log=lambda *_: None)
    assert second["reviewed"] == 1 and len(kit.calls()) == 3
    records = cc_review_driver.read_results(results)
    assert sorted(records) == ["run-0", "run-1", "run-2"]
    record = records["run-1"]
    assert record["reviewer"]["family"] == "claude" and record["reviewer"]["model"] == "claude-haiku-5-5"
    assert record["reviewer"]["claude_code_version"] == "9.9.9 (Claude Code)"
    assert parse_submission(json.dumps(record["reply"]))["verdict"] == "done"
    assert cc_review_driver.verdicts(results) == {"done": 3, "not_done": 0, "none": 0, "failed": 0}
    assert cc_review_driver.drive(items, results, tmp_path / "w", runner, log=lambda *_: None)["reviewed"] == 0
    no_token_anywhere(tmp_path)


def test_the_driver_stops_the_whole_job_on_a_usage_limit(kit, tmp_path):
    runner = kit.runner([{"turns": [], "exit": 1,
                          "result": {"subtype": "success", "is_error": True,
                                     "result": "You've hit your weekly limit · resets Nov 18"}}])
    items, results = manifest(tmp_path, 5), tmp_path / "results.jsonl"
    summary = cc_review_driver.drive(items, results, tmp_path / "w", runner, concurrency=1, log=lambda *_: None)
    assert summary["stopped"]["kind"] == "usage_limit" and summary["left"] == 4
    assert len(kit.calls()) == 1  # no retry, and no further review started
    assert cc_review_driver.read_results(results) == {}  # the stopped run is left to do, not recorded
    stop = json.loads((tmp_path / "stop.json").read_text())
    assert stop["kind"] == "usage_limit" and stop["run"] == "run-0"


def test_the_driver_records_failures_and_retries_them_only_when_asked(kit, tmp_path):
    runner = kit.runner([{"turns": [], "exit": 1, "result": {"subtype": "error_during_execution",
                                                             "is_error": True, "result": "boom"}},
                         {"turns": [[{"name": "StructuredOutput", "input": DONE}]], "result": success(DONE, 1)}])
    items, results = manifest(tmp_path, 2), tmp_path / "results.jsonl"
    summary = cc_review_driver.drive(items, results, tmp_path / "w", runner, concurrency=1, log=lambda *_: None)
    assert summary["failed"] == 1 and summary["reviewed"] == 1
    assert cc_review_driver.drive(items, results, tmp_path / "w", runner, log=lambda *_: None)["reviewed"] == 0
    again = cc_review_driver.drive(items, results, tmp_path / "w", runner, retry_failed=True, log=lambda *_: None)
    assert again["reviewed"] == 1
    assert cc_review_driver.verdicts(results)["failed"] == 0


def test_the_driver_stops_after_failures_in_a_row(kit, tmp_path):
    runner = kit.runner([{"turns": [], "exit": 1, "result": {"subtype": "error_during_execution",
                                                             "is_error": True, "result": "boom"}}])
    summary = cc_review_driver.drive(manifest(tmp_path, 6), tmp_path / "r.jsonl", tmp_path / "w", runner,
                                     concurrency=1, max_failures=3, log=lambda *_: None)
    assert summary["stopped"]["kind"] == "failures" and summary["failed"] == 3 and summary["left"] == 3


def test_the_manifest_takes_the_checkers_task_text_or_its_parts(tmp_path):
    item = {"run": "r", "container": "c", "cwd": "/app", "task": "Do it.", "final_message": "",
            "submission": "", "requirement": "Do it."}
    text = cc_review_driver.task_text(item)
    assert "<requirement>\nDo it.\n</requirement>" in text and text.endswith("then call submit_verdict.")
    assert reviewer_task(text).endswith("then call StructuredOutput.")
    (tmp_path / "t.md").write_text(TASK)
    assert cc_review_driver.task_text({"task_file": str(tmp_path / "t.md")}) == TASK
    path = tmp_path / "m.jsonl"
    path.write_text(json.dumps({"run": "a", "container": "c", "cwd": "/app", "task_text": TASK}) + "\n"
                    + json.dumps({"run": "a", "container": "d", "cwd": "/app", "task_text": TASK}) + "\n")
    with pytest.raises(ValueError):
        cc_review_driver.load_manifest(path)


def test_the_cli_shows_its_command_with_placeholders(tmp_path, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cc_review", Path(__file__).parents[1] / "scripts" / "cc_review.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "m.jsonl").write_text(json.dumps({"run": "a", "container": "c", "cwd": "/app",
                                                  "task_text": TASK}) + "\n")
    assert module.main(["--manifest", str(tmp_path / "m.jsonl"), "--results", str(tmp_path / "r.jsonl"),
                        "--work-dir", str(tmp_path / "w"), "--show-command"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["argv"][:2] == ["claude", "-p"] and "<container>" in shown["allow"]
    assert os.environ.get(TOKEN_NAME, "\x00") not in json.dumps(shown)
