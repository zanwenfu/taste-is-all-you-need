"""Tests for studies.claude_code.driver: command lines, environments, turns and the project folder; run() only with a fake Claude Code."""

import json

import pytest

from studies.claude_code import driver, score, settings
from studies.claude_code.driver import (
    LS_ONE,
    PYTHON_B,
    TAG_FILE,
    TOOLS,
    check_isolated,
    command,
    dry_run,
    environment,
    plan_session,
    prepare,
    run,
)

STUB = "studies.claude_code.hooks.rule_list_hook:stub_reader"
FAKE = "studies.claude_code.hooks.mem0_hook:fake_from_env"
EXTRA = {"rules": {"reader": STUB}, "mem0": {"memory": FAKE}}


def _plan(tmp_path, arm="floor", tasks=5, **options):
    return plan_session(arm, tmp_path / arm, seed=1, tasks=tasks, **EXTRA.get(arm, {}), **options)


class FakeClaude:
    """Stands in for claude -p: records each call and answers with a few stream-json lines, one Bash call among them."""

    def __init__(self, cost=0.1, result=True, bash_command="python3 -B x.py; ls -la", on_call=None):
        self.cost, self.result, self.bash_command, self.on_call, self.calls = cost, result, bash_command, on_call, []

    def __call__(self, argv, *, env, cwd, stdin_line, timeout_s, on_line, stderr_path):
        self.calls.append({"argv": argv, "env": env, "cwd": cwd, "message": json.loads(stdin_line), "tag_file": (cwd / TAG_FILE).exists()})
        if self.on_call:
            self.on_call(self, stderr_path)
        number = len(self.calls)
        on_line(json.dumps({"type": "system", "subtype": "init", "session_id": "sess-1"}) + "\n")
        on_line(json.dumps({"type": "assistant", "parent_tool_use_id": None, "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": f"call-{number}", "name": "Bash", "input": {"command": self.bash_command}}]}}) + "\n")
        if self.result:
            on_line(json.dumps({"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": self.cost,
                                "session_id": "sess-1", "num_turns": 2, "permission_denials": []}) + "\n")
        return 0


def _run(plan, fake, **options):
    prepare(plan, isolated=False)
    settings_for = {"per_call_usd": 1.0, "session_usd": 10.0, "caller": fake, "base_env": {driver.SUBSCRIPTION_TOKEN: TOKEN},
                    "say": lambda message: None}
    return run(plan, execute=True, **(settings_for | options))


TOKEN = "sk-ant-oat01-" + "t" * 40  # a stand-in subscription token, never a real one


def test_run_refuses_without_the_subscription_token(tmp_path):
    plan = _plan(tmp_path)
    with pytest.raises(RuntimeError, match="subscription"):
        _run(plan, FakeClaude(), base_env={"ANTHROPIC_API_KEY": "sk-ant-api03-" + "a" * 40})


def test_run_drops_every_other_credential_and_never_saves_the_token(tmp_path):
    plan, fake = _plan(tmp_path), FakeClaude()
    others = {name: "x" * 30 for name in driver.OTHER_CREDENTIALS}
    _run(plan, fake, base_env={driver.SUBSCRIPTION_TOKEN: TOKEN, **others})
    sent = fake.calls[0]["env"]
    assert sent[driver.SUBSCRIPTION_TOKEN] == TOKEN
    assert not any(name in sent for name in driver.OTHER_CREDENTIALS)
    for path in (plan.stream, plan.record):
        assert TOKEN not in path.read_text()


# Command lines and environments

def test_command_never_has_bare_and_resumes_only_when_told():
    for resume in (None, "sess-1"):
        argv = command(budget_usd=1.5, resume=resume, model="claude-sonnet-5-5")
        assert "--bare" not in argv
        assert ("--resume" in argv) == (resume is not None)
    argv = command(budget_usd=1.5)
    assert argv[:2] == ["claude", "-p"] and "--verbose" in argv
    assert argv[argv.index("--input-format") + 1] == "stream-json" and argv[argv.index("--output-format") + 1] == "stream-json"
    start = argv.index("--allowedTools") + 1
    assert tuple(argv[start:start + len(TOOLS)]) == TOOLS == ("Bash", "Edit", "Write", "Read", "Glob", "Grep")
    assert argv[argv.index("--max-budget-usd") + 1] == "1.50" and argv[argv.index("--permission-mode") + 1] == driver.PERMISSION_MODE


def test_every_arm_sends_the_same_command_lines_and_never_bare(tmp_path):
    lines = {arm: [call["argv"] for call in dry_run(_plan(tmp_path, arm), 2.0)["calls"]] for arm in settings.ARMS}
    assert all(argvs == lines["floor"] for argvs in lines.values())
    assert not any("--bare" in argv for argvs in lines.values() for argv in argvs)
    assert "--resume" not in lines["floor"][0] and all("--resume" in argv for argv in lines["floor"][1:])


def test_environment_by_arm(tmp_path):
    for arm in settings.ARMS:
        env = environment(arm, config_dir=tmp_path, **EXTRA.get(arm, {}))
        assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "100000" and env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "50"
        assert env["DISABLE_AUTOUPDATER"] == "1" and env["CLAUDE_CONFIG_DIR"] == str(tmp_path)
        assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == ("0" if arm == "native" else "1")
    assert environment("rules", config_dir=tmp_path, reader=STUB)["CC_STUDY_RULE_READER"] == STUB
    assert environment("mem0", config_dir=tmp_path, memory=FAKE)["CC_STUDY_MEM0_CLIENT"] == FAKE


def test_memory_arms_need_their_reader_or_client(tmp_path):
    with pytest.raises(ValueError):
        environment("rules", config_dir=tmp_path)
    with pytest.raises(ValueError):
        environment("mem0", config_dir=tmp_path)
    with pytest.raises(ValueError):
        environment("floor", config_dir=tmp_path, window=50_000)


# Turns

def test_rules_are_told_in_the_first_task_and_before_the_fourth(tmp_path):
    plan = _plan(tmp_path)
    assert [turn.id for turn in plan.turns] == ["task_01", "task_02", "task_03", "rule_ls", "task_04", "task_05"]
    first, rule = plan.turns[0], plan.turns[3]
    assert first.prompt.startswith(f"{PYTHON_B} {driver.tag_rule()}") and first.prompt.endswith(first.task.prompt)
    assert first.tells == ("python_B", "session_tag") and set(first.tells) <= set(score.RULES)
    assert (rule.kind, rule.prompt, rule.tells, rule.task) == ("rule", LS_ONE, ("ls_one",), None)
    assert all(turn.prompt == turn.task.prompt for turn in plan.turns[1:] if turn.kind == "task")


def test_the_ceiling_restates_only_rules_already_told(tmp_path):
    floor, ceiling = _plan(tmp_path, "floor"), _plan(tmp_path, "restated")
    assert ceiling.turns[0].prompt == floor.turns[0].prompt
    second, fourth = ceiling.turns[1].prompt, ceiling.turns[4].prompt
    assert second.startswith(floor.turns[1].prompt) and PYTHON_B in second and f"echo {ceiling.tag}" in second and LS_ONE not in second
    assert LS_ONE in fourth


def test_the_tag_sentence_follows_the_tag_file_choice(tmp_path):
    kept = _plan(tmp_path, tag_removed=False)
    assert "removed" not in kept.turns[0].prompt and "removed" in _plan(tmp_path).turns[0].prompt


def test_dry_run_writes_nothing(tmp_path):
    plan = _plan(tmp_path)
    shown = dry_run(plan, 2.0)
    assert not plan.out.exists()
    assert json.loads(shown["calls"][0]["stdin"])["message"] == {"role": "user", "content": plan.turns[0].prompt}
    assert shown["env"] == plan.env


def test_main_without_flags_prints_the_plan_and_writes_nothing(tmp_path, capsys):
    assert driver.main(["--arm", "floor", "--seed", "1", "--tasks", "4", "--out", str(tmp_path / "s")]) == 0
    assert json.loads(capsys.readouterr().out)["arm"] == "floor"
    assert not (tmp_path / "s").exists()


# The project folder

def test_prepare_writes_the_project(tmp_path):
    plan = _plan(tmp_path, "rules")
    prepare(plan, isolated=False)
    assert "ls -la" in (plan.project / "CLAUDE.md").read_text()
    assert (plan.project / TAG_FILE).read_text().strip() == plan.tag
    hooks = json.loads((plan.project / ".claude" / "settings.json").read_text())["hooks"]
    assert "UserPromptSubmit" in hooks and plan.hooks_dir.is_dir()
    assert (plan.out / "claude_config").is_dir()
    with pytest.raises(FileExistsError):
        prepare(plan, isolated=False)


def test_a_project_under_claude_instructions_or_a_repository_is_refused(tmp_path):
    for number, name in enumerate(("CLAUDE.md", "CLAUDE.local.md", ".git")):
        root = tmp_path / f"case{number}"
        (root / "runs").mkdir(parents=True)
        if name == ".git":
            (root / name).mkdir()
        else:
            (root / name).write_text("rules")
        with pytest.raises(ValueError, match=name):
            check_isolated(root / "runs" / "s1" / "project")


# Running, with a fake Claude Code only

def test_run_needs_execute_and_writes_nothing_without_it(tmp_path):
    plan = _plan(tmp_path)
    with pytest.raises(RuntimeError):
        run(plan, per_call_usd=1.0, session_usd=5.0, caller=FakeClaude())
    assert not plan.out.exists()


def test_run_sends_each_turn_resuming_the_session(tmp_path):
    plan, fake = _plan(tmp_path), FakeClaude()
    record = _run(plan, fake)
    assert len(fake.calls) == len(plan.turns) == 6
    assert "--resume" not in fake.calls[0]["argv"]
    assert all(call["argv"][call["argv"].index("--resume") + 1] == "sess-1" for call in fake.calls[1:])
    assert not any("--bare" in call["argv"] for call in fake.calls)
    assert [call["message"]["message"]["content"] for call in fake.calls] == [turn.prompt for turn in plan.turns]
    assert all(call["cwd"] == plan.project for call in fake.calls)
    assert fake.calls[0]["env"]["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert fake.calls[0]["tag_file"] and not fake.calls[1]["tag_file"]
    assert all((plan.project / turn.task.folder).is_dir() for turn in plan.turns if turn.task)
    assert record["ended_by"] is None and record["spent_usd"] == pytest.approx(0.6)
    assert json.loads(plan.record.read_text())["calls"][-1]["session_id"] == "sess-1"


def test_the_saved_stream_scores(tmp_path):
    plan = _plan(tmp_path)
    _run(plan, FakeClaude())
    lines = plan.stream.read_text().splitlines()
    assert sum(json.loads(line)["type"] == "study_turn" for line in lines) == 6
    summary = score.score_session(lines, plan.tag)
    assert summary["scores"]["python_B"]["applies"] == 6 == summary["scores"]["python_B"]["obeyed"]
    assert (summary["scores"]["ls_one"]["applies"], summary["scores"]["ls_one"]["obeyed"]) == (3, 0)
    assert summary["valid"] is True


def test_run_stops_when_the_session_budget_is_spent(tmp_path):
    plan, fake = _plan(tmp_path), FakeClaude(cost=0.6)
    record = _run(plan, fake, session_usd=1.5)
    assert len(fake.calls) == 3 and record["ended_by"] == "budget"
    assert [call["argv"][call["argv"].index("--max-budget-usd") + 1] for call in fake.calls] == ["1.00", "0.90", "0.30"]


def test_run_stops_on_a_call_with_no_result(tmp_path):
    plan, fake = _plan(tmp_path), FakeClaude(result=False)
    assert _run(plan, fake)["ended_by"] == "no_result" and len(fake.calls) == 1


def test_run_stops_when_a_hook_logs_an_error(tmp_path):
    def fail(fake, stderr_path):
        (stderr_path.parent / "hooks" / "rule_list.jsonl").write_text(json.dumps({"error": "reader down"}) + "\n")

    plan, fake = _plan(tmp_path), FakeClaude(on_call=fail)
    assert _run(plan, fake)["ended_by"] == "hook_error" and len(fake.calls) == 1


def test_secret_values_never_reach_the_saved_files(tmp_path):
    secret = "sk-test-0123456789abcdef"

    def leak(fake, stderr_path):
        stderr_path.write_text(f"warning: key {secret}\n")

    plan, fake = _plan(tmp_path), FakeClaude(bash_command=f"echo {secret}", on_call=leak)
    _run(plan, fake, base_env={"FAKE_API_KEY": secret, "HOME": "/home/someone", driver.SUBSCRIPTION_TOKEN: TOKEN})
    for path in (plan.stream, plan.record, plan.out / "stderr.log"):
        text = path.read_text()
        assert secret not in text
    assert "[redacted]" in plan.stream.read_text() and "/home/someone" not in plan.record.read_text()


def test_secret_values_are_picked_by_name():
    env = {"ANTHROPIC_API_KEY": "a" * 20, "AWS_SESSION_TOKEN": "b" * 30, "PATH": "/usr/bin:/bin", "SHORT_KEY": "abc"}
    assert driver.secret_values(env) == ["b" * 30, "a" * 20]
    assert driver.redact("x" + "a" * 20 + "y", driver.secret_values(env)) == "x[redacted]y"
