"""Tests for studies.claude_code.swe_series: task loading, the series plan, the "1" check, the docker command lines, and a whole series
run against a fake Claude Code and a fake docker (no container, no model)."""

import json
import subprocess

import pytest

from studies.claude_code import swe_series as s

TOKEN = "sk-ant-oat01-" + "t" * 40  # a stand-in, never a real token
COMMIT = "0ea40e09d1b35bcb69ff4d9cecf3d0defa4b36e8"


def _task(root, n, repo="ansible"):
    folder = root / f"instance_{repo}__{repo}-{n:08x}{COMMIT[8:]}-v30a923fb5c164d6cd18280c02422f75e611e8fb2"
    (folder / "environment").mkdir(parents=True)
    (folder / "tests").mkdir()
    (folder / "environment" / "Dockerfile").write_text(f"FROM ghcr.io/scaleapi/swe-bench_pro-v2:{folder.name}\n")
    (folder / "instruction.md").write_text(f"A code repository is available in the `/app` directory. Fix bug {n}.\n")
    (folder / "tests" / "test.sh").write_text("exit 0\n")
    return folder


@pytest.fixture
def tasks(tmp_path):
    root = tmp_path / "tasks"
    return [_task(root, n) for n in range(1, 6)] + [_task(root, 9, repo="flipt")]


def _plan(tmp_path, tasks, arm="floor", count=3):
    folders = s.pick_tasks(tasks[0].parent, "ansible", count, seed=1)
    return s.plan_series(arm, folders, tmp_path / "out", model="claude-sonnet-5-5", claude_binary=tmp_path / "claude")


def test_tasks_load_with_their_image_and_instruction(tasks):
    task = s.load_task(tasks[0])
    assert task.image == f"ghcr.io/scaleapi/swe-bench_pro-v2:{tasks[0].name}"
    assert task.instruction.startswith("A code repository") and task.short == "ansible-00000001"


def test_picking_is_seeded_one_repository_and_skips_what_it_should(tasks):
    root = tasks[0].parent
    first = s.pick_tasks(root, "ansible", 3, seed=1)
    assert first == s.pick_tasks(root, "ansible", 3, seed=1) and all("ansible" in p.name for p in first)
    assert all("00000002" not in p.name for p in s.pick_tasks(root, "ansible", 4, seed=1, avoid=("00000002",)))
    with pytest.raises(ValueError):
        s.pick_tasks(root, "flipt", 2, seed=1)


def test_the_requirement_is_told_once_or_every_round_or_never(tmp_path, tasks):
    floor = _plan(tmp_path, tasks, "floor")
    assert floor.rounds[0].prompt.startswith(s.REQUIREMENT) and floor.rounds[0].tells_requirement
    assert all(s.REQUIREMENT not in r.prompt and not r.tells_requirement for r in floor.rounds[1:])
    assert all(r.prompt.startswith(s.REQUIREMENT) for r in _plan(tmp_path, tasks, "restated").rounds)
    assert all(s.REQUIREMENT not in r.prompt for r in _plan(tmp_path, tasks, "plain").rounds)


@pytest.mark.parametrize("text, strict, line, token", [
    ("Fixed the bug in vars.py; tests pass.\n\n1", True, True, True),
    ("Done.\n**1**", True, True, True),
    ("Done.\n`1`.", True, True, True),
    ("1\n\nFixed the bug.", False, True, True),
    ("Fixed it. 1", False, False, True),
    ("Fixed 1 bug in line 12 of v1.2/a-1.py.", False, False, True),
    ("Fixed line 12 in v1.2/a-1.py and file_1.", False, False, False),
    ("All 10 tests pass.", False, False, False),
    ("", False, False, False),
    (None, False, False, False),
])
def test_the_one_is_read_by_rules_fixed_in_advance(text, strict, line, token):
    assert s.hand_back_one(text) == {"strict": strict, "line": line, "token": token}


def test_command_lines_pass_the_token_by_name_only_and_never_bare(tmp_path, tasks):
    plan = _plan(tmp_path, tasks)
    name = s.container_name(plan, plan.rounds[0])
    start = s.start_argv(plan, plan.rounds[0], name)
    assert start[:3] == ["docker", "run", "-d"] and start[-2:] == [plan.rounds[0].task.image, "infinity"]
    assert f"{(tmp_path / 'claude').resolve()}:{s.CLAUDE_IN_CONTAINER}:ro" in start
    argv = s.claude_argv(plan, name, budget_usd=8, resume="sess-1")
    assert argv[:3] == ["docker", "exec", "-i"] and "--bare" not in argv
    flags = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
    assert "CLAUDE_CODE_OAUTH_TOKEN" in flags and not any(f.startswith("CLAUDE_CODE_OAUTH_TOKEN=") for f in flags)
    assert "CLAUDE_CODE_DISABLE_AUTO_MEMORY=1" in flags and f"CLAUDE_CONFIG_DIR={s.CONFIG_IN_CONTAINER}" in flags
    assert argv[argv.index("--resume") + 1] == "sess-1" and argv[argv.index("--model") + 1] == "claude-sonnet-5-5"
    native = s.plan_series("native", [tasks[0]], tmp_path / "n", model=None, claude_binary=tmp_path / "claude")
    assert "CLAUDE_CODE_DISABLE_AUTO_MEMORY=0" in s.exec_env(native)


class FakeDocker:
    """Answers docker run/exec/cp/rm; a patch per round; a reward per round; records every command."""

    def __init__(self, rewards=(1, 0, 1), fail_start=False):
        self.rewards, self.fail_start, self.calls, self.graded = list(rewards), fail_start, [], 0

    def __call__(self, argv, **options):
        self.calls.append(argv)
        out, code = "", 0
        if argv[:2] == ["docker", "run"]:
            code = 1 if self.fail_start else 0
        elif argv[:2] == ["docker", "exec"] and "git diff --cached" in argv[-1]:
            out = "diff --git a/x.py b/x.py\n+fix\n"
        elif argv[:2] == ["docker", "exec"] and "test.sh" in argv[-1]:
            out = f"{self.rewards[self.graded]}\n"
            self.graded += 1
        return subprocess.CompletedProcess(argv, code, stdout=out, stderr="no such image" if code else "")


class FakeClaude:
    def __init__(self, replies, cost=0.5):
        self.replies, self.cost, self.calls = list(replies), cost, []

    def __call__(self, argv, *, env, cwd, stdin_line, timeout_s, on_line, stderr_path):
        self.calls.append({"argv": argv, "env": env, "message": json.loads(stdin_line)})
        n = len(self.calls)
        on_line(json.dumps({"type": "system", "subtype": "init", "session_id": "sess-1"}) + "\n")
        on_line(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": self.replies[n - 1],
                            "total_cost_usd": self.cost * n, "session_id": "sess-1", "num_turns": 30}) + "\n")
        return 0


def test_a_series_runs_grades_and_scores_each_round(tmp_path, tasks):
    plan = _plan(tmp_path, tasks)
    docker, claude = FakeDocker(), FakeClaude(["Fixed it.\n\n1", "Fixed it.\n1", "Fixed the parser, all tests pass."])
    record = s.run_series(plan, execute=True, per_call_usd=8, session_usd=25, caller=claude, runner=docker,
                          base_env={"CLAUDE_CODE_OAUTH_TOKEN": TOKEN, "ANTHROPIC_API_KEY": "sk-ant-api03-" + "a" * 40}, say=lambda m: None)
    assert [r["one"]["strict"] for r in record["rounds"]] == [True, True, False]
    assert [r["grade_reward"] for r in record["rounds"]] == [1, 0, 1]
    assert record["spent_usd"] == pytest.approx(1.5) and record["ended_by"] is None
    assert "--resume" not in claude.calls[0]["argv"] and all("sess-1" in c["argv"] for c in claude.calls[1:])
    assert all("ANTHROPIC_API_KEY" not in c["env"] for c in claude.calls)
    assert claude.calls[0]["message"]["message"]["content"].startswith(s.REQUIREMENT)
    assert (plan.out / "patches" / "round_01.diff").read_text().startswith("diff --git")
    assert sum(argv[:3] == ["docker", "rm", "-f"] for argv in docker.calls) == 6  # before and after each round
    for path in (plan.stream, plan.record):
        assert TOKEN not in path.read_text()


def test_a_series_refuses_without_the_subscription_token(tmp_path, tasks):
    with pytest.raises(RuntimeError, match="subscription"):
        s.run_series(_plan(tmp_path, tasks), execute=True, per_call_usd=8, session_usd=25, caller=FakeClaude([]), runner=FakeDocker(),
                     base_env={"ANTHROPIC_API_KEY": "sk-ant-api03-" + "a" * 40})
    with pytest.raises(RuntimeError, match="execute=True"):
        s.run_series(_plan(tmp_path, tasks), per_call_usd=8, session_usd=25)


def test_a_series_stops_when_a_container_will_not_start(tmp_path, tasks):
    claude = FakeClaude(["1"])
    record = s.run_series(_plan(tmp_path, tasks), execute=True, per_call_usd=8, session_usd=25, caller=claude,
                          runner=FakeDocker(fail_start=True), base_env={"CLAUDE_CODE_OAUTH_TOKEN": TOKEN}, say=lambda m: None)
    assert record["ended_by"].startswith("container") and not claude.calls
