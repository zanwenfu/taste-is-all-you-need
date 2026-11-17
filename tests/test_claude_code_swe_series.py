"""Tests for studies.claude_code.swe_series: tasks, the plan (rules once, sessions, arms), the two rule checks, the docker command
lines, and whole series against a fake Claude Code and a fake docker (no container, no model)."""

import json
import subprocess

import pytest

from studies.claude_code import swe_series as s

TOKEN = "sk-ant-oat01-" + "t" * 40  # a stand-in, never a real token
NO_PULL = lambda image: None  # noqa: E731  (tests never start a real docker pull)
COMMIT = "0ea40e09d1b35bcb69ff4d9cecf3d0defa4b36e8"


def _task(root, n, repo="ansible", gold_lines=10):
    folder = root / f"instance_{repo}__{repo}-{n:08x}{COMMIT[8:]}-v30a923fb5c164d6cd18280c02422f75e611e8fb2"
    for sub in ("environment", "tests", "solution"):
        (folder / sub).mkdir(parents=True)
    (folder / "environment" / "Dockerfile").write_text(f"FROM ghcr.io/scaleapi/swe-bench_pro-v2:{folder.name}\n")
    (folder / "instruction.md").write_text(f"A code repository is available in the `/app` directory. Fix bug {n}.\n")
    (folder / "tests" / "test.sh").write_text("exit 0\n")
    (folder / "solution" / "gold_patch.diff").write_text("x\n" * gold_lines)
    return folder


@pytest.fixture
def tasks(tmp_path):
    root = tmp_path / "tasks"
    return [_task(root, n, gold_lines=100 - 7 * n) for n in range(1, 11)] + [_task(root, 99, repo="flipt")]


def _plan(tmp_path, tasks, arm="shipped", count=10, sessions=(1, 6), **options):
    folders = s.pick_tasks(tasks[0].parent, "ansible", count, seed=3)
    extra = {"code": tmp_path / "code", "token_file": tmp_path / "token"} if arm == "ours" else {}
    return s.plan_series(arm, folders, tmp_path / arm, model="claude-sonnet-5-5", claude_binary=tmp_path / "claude", sessions=sessions,
                         **(extra | options))


def test_tasks_load_and_run_from_the_smallest_reference_patch_to_the_largest(tasks):
    task = s.load_task(tasks[0])
    assert task.image == f"ghcr.io/scaleapi/swe-bench_pro-v2:{tasks[0].name}" and task.short == "ansible-00000001" and task.size == 93
    picked = s.pick_tasks(tasks[0].parent, "ansible", 10, seed=3)
    assert picked == s.pick_tasks(tasks[0].parent, "ansible", 10, seed=3)
    sizes = [s.load_task(p).size for p in picked]
    assert sizes == sorted(sizes) and all("ansible" in p.name for p in picked)
    assert all("00000002" not in p.name for p in s.pick_tasks(tasks[0].parent, "ansible", 9, seed=3, avoid=("00000002",)))
    with pytest.raises(ValueError):
        s.pick_tasks(tasks[0].parent, "flipt", 2, seed=3)


def test_the_rules_are_told_once_in_plain_words_and_sessions_break_where_planned(tmp_path, tasks):
    shipped = _plan(tmp_path, tasks)
    assert shipped.rounds[0].prompt.startswith(s.OPENING) and s.REMEMBER not in shipped.rounds[0].prompt
    assert all(s.OPENING not in r.prompt for r in shipped.rounds[1:])
    assert [r.number for r in shipped.rounds if r.new_session] == [1, 6]
    assert _plan(tmp_path, tasks, "remember").rounds[0].prompt.startswith(f"{s.OPENING} {s.REMEMBER}")
    assert all(r.prompt.startswith(s.OPENING) for r in _plan(tmp_path, tasks, "restated").rounds)
    assert all(s.OPENING not in r.prompt for r in _plan(tmp_path, tasks, "plain").rounds)
    assert s.OPENING == 'Two rules for all my tasks. Always output "1" when you hand back your answer to me. Don\'t modify the tests unless I ask.'
    with pytest.raises(ValueError, match="ours arm needs"):
        s.plan_series("ours", tasks[:1], tmp_path / "x", model=None, claude_binary=tmp_path / "claude")


@pytest.mark.parametrize("text, strict, line, token", [
    ("Fixed the bug in vars.py; tests pass.\n\n1", True, True, True),
    ("Done.\n**1**", True, True, True),
    ("Done.\n`1`.", True, True, True),
    ("1\n\nFixed the bug.", False, True, True),
    ("Fixed it. 1", False, False, True),
    ("Fixed line 12 in v1.2/a-1.py and file_1.", False, False, False),
    ("All 10 tests pass.", False, False, False),
    ("", False, False, False),
    (None, False, False, False),
])
def test_rule_1_is_read_by_rules_fixed_in_advance(text, strict, line, token):
    assert s.hand_back_one(text) == {"strict": strict, "line": line, "token": token}


@pytest.mark.parametrize("path, is_test", [
    ("test/units/executor/test_task_executor.py", True), ("lib/ansible/galaxy/collection/__init__.py", False),
    ("conftest.py", True), ("pkg/server/server_test.go", True), ("src/app.spec.ts", True), ("tests/fixtures/data.json", False),
    ("lib/ansible/test_helpers.py", True), ("lib/ansible/testing.py", False), ("web/__tests__/a.js", True),
])
def test_rule_2_uses_the_verifiers_own_notion_of_a_test(path, is_test):
    assert s.is_test_code(path) is is_test


def test_rule_2_reads_the_files_a_patch_touches():
    patch = ("diff --git a/lib/x.py b/lib/x.py\n+fix\n"
             "diff --git a/test/units/test_x.py b/test/units/test_x.py\n+assert\n")
    assert s.patch_paths(patch) == ["lib/x.py", "test/units/test_x.py"]
    assert s.tests_rule(patch) == {"kept": False, "test_files": ["test/units/test_x.py"]}
    assert s.tests_rule("diff --git a/lib/x.py b/lib/x.py\n")["kept"] and s.tests_rule("")["kept"]


def test_command_lines_follow_the_locked_protocol_and_pass_the_token_by_name_only(tmp_path, tasks):
    plan = _plan(tmp_path, tasks, proxy="http://172.30.0.1:8888", network="cc-internal")
    name = s.container_name(plan, plan.rounds[0])
    start = s.start_argv(plan, plan.rounds[0], name)
    assert start[start.index("--network") + 1] == "cc-internal" and start[-2:] == [plan.rounds[0].task.image, "infinity"]
    assert not any(s.TOKEN_IN_CONTAINER in a for a in start)  # only the ours arm mounts the token file
    argv = s.claude_argv(plan, name, budget_usd=3, resume="sess-1")
    flags = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
    assert "CLAUDE_CODE_OAUTH_TOKEN" in flags and not any(f.startswith("CLAUDE_CODE_OAUTH_TOKEN=") for f in flags)
    assert "HTTPS_PROXY=http://172.30.0.1:8888" in flags and "CLAUDE_CODE_DISABLE_AUTO_MEMORY=0" in flags  # auto memory on
    assert argv[argv.index("--disallowedTools") + 1: argv.index("--disallowedTools") + 3] == ["WebFetch", "WebSearch"]
    assert "--settings" not in argv and "--bare" not in argv and argv[argv.index("--resume") + 1] == "sess-1"
    for excluded in s.PROJECT_MEMORY:
        assert f"':(exclude){excluded}'" in s.PATCH_SCRIPT


def test_the_ours_arm_mounts_its_hooks_memory_and_reader(tmp_path, tasks):
    plan = _plan(tmp_path, tasks, "ours")
    name = s.container_name(plan, plan.rounds[0])
    start = " ".join(s.start_argv(plan, plan.rounds[0], name))
    assert f"{(tmp_path / 'token').resolve()}:{s.TOKEN_IN_CONTAINER}:ro" in start and f":{s.CODE_IN_CONTAINER}:ro" in start
    argv = s.claude_argv(plan, name, budget_usd=3, resume=None)
    assert argv[argv.index("--settings") + 1] == s.HOOKS_IN_CONTAINER
    flags = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
    assert "CC_STUDY_RULE_READER=studies.claude_code.readers:claude_reader" in flags
    hooks = s.hook_settings()["hooks"]
    assert "--git" in hooks["UserPromptSubmit"][0]["hooks"][0]["command"] and hooks["SessionStart"][0]["matcher"] == "compact"


class FakeDocker:
    """Answers docker run/exec/cp/rm/rmi and tar; one patch per task (it touches a test on tasks listed in touch_tests); rewards in order."""

    def __init__(self, rewards=None, fail_start=False, touch_tests=(2,), project_memory=False):
        self.rewards, self.fail_start, self.touch_tests, self.project_memory = list(rewards or [1] * 20), fail_start, touch_tests, project_memory
        self.calls, self.graded, self.patched, self.carried_in = [], 0, 0, 0

    def __call__(self, argv, **options):
        self.calls.append(argv)
        out, code = "", 0
        script = argv[-1] if argv[:2] == ["docker", "exec"] else ""
        if argv[:2] == ["docker", "run"]:
            code = 1 if self.fail_start else 0
        elif "git diff --cached" in script:
            self.patched += 1
            out = "diff --git a/lib/x.py b/lib/x.py\n+fix\n"
            if self.patched in self.touch_tests:
                out += "diff --git a/test/units/test_x.py b/test/units/test_x.py\n+t\n"
        elif "test.sh" in script:
            out = f"{self.rewards[self.graded]}\n"
            self.graded += 1
        elif script == s.MEMORY_OUT:
            out = b"TAR" if self.project_memory else b""
        elif argv[:2] == ["tar", "tf"]:
            out = "CLAUDE.md\n"
        elif argv[:3] == ["docker", "exec", "-i"] and "tar" in argv:
            self.carried_in += 1
        return subprocess.CompletedProcess(argv, code, stdout=out, stderr="no such image" if code else "")


class FakeClaude:
    """claude -p: a running total that restarts with each new session (no --resume), as Claude Code's does."""

    def __init__(self, replies, cost=0.2, on_call=None, compact_on=()):
        self.replies, self.cost, self.on_call, self.calls, self.in_session = list(replies), cost, on_call, [], 0
        self.compact_on = set(compact_on)

    def __call__(self, argv, *, env, cwd, stdin_line, timeout_s, on_line, stderr_path):
        self.calls.append({"argv": argv, "env": env, "message": json.loads(stdin_line)["message"]["content"], "timeout": timeout_s})
        self.in_session = self.in_session + 1 if "--resume" in argv else 1
        if self.on_call:
            self.on_call(len(self.calls), cwd)
        on_line(json.dumps({"type": "system", "subtype": "init", "session_id": f"sess-{len(self.calls) if '--resume' not in argv else 'x'}"}) + "\n")
        if len(self.calls) in self.compact_on:
            on_line(json.dumps({"type": "system", "subtype": "compact_boundary", "compact_metadata": {"pre_tokens": 80000}}) + "\n")
        on_line(json.dumps({"type": "assistant", "message": {"content": [], "usage": {"input_tokens": 5, "cache_read_input_tokens": 40000,
                                                                                         "cache_creation_input_tokens": 995}}}) + "\n")
        on_line(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": self.replies[len(self.calls) - 1],
                            "total_cost_usd": self.cost * self.in_session, "session_id": "sess-a" if len(self.calls) <= 5 else "sess-b",
                            "num_turns": 20}) + "\n")
        return 0


def _run(plan, docker, claude, **options):
    return s.run_series(plan, execute=True, per_call_usd=3, session_usd=5, caller=claude, runner=docker,
                        base_env={"CLAUDE_CODE_OAUTH_TOKEN": TOKEN, "ANTHROPIC_API_KEY": "sk-ant-api03-" + "a" * 40},
                        say=lambda m: None, **({"prefetch": NO_PULL} | options))


def test_a_series_runs_two_sessions_checks_both_rules_and_grades_in_fresh_containers(tmp_path, tasks):
    plan = _plan(tmp_path, tasks)

    def remember(n, cwd):  # Claude Code saves the rules to its auto memory during task 3
        if n == 3:
            folder = cwd / "claude_config" / "projects" / "-app" / "memory"
            folder.mkdir(parents=True)
            (folder / "MEMORY.md").write_text('- Always output "1" when handing back. Do not modify the tests unless asked.\n')

    replies = ["Fixed.\n\n1"] * 5 + ["Fixed."] * 5
    docker, claude = FakeDocker(rewards=[1, 0, 1, 1, 1, 1, 1, 0, 1, 1]), FakeClaude(replies, on_call=remember)
    record = _run(plan, docker, claude)
    rounds = record["rounds"]
    assert [r["rule_1"]["strict"] for r in rounds] == [True] * 5 + [False] * 5
    assert [r["rule_2"]["kept"] for r in rounds] == [True, False] + [True] * 8
    assert [r["grade_reward"] for r in rounds] == [1, 0, 1, 1, 1, 1, 1, 0, 1, 1]
    resumed = ["--resume" in c["argv"] for c in claude.calls]
    assert resumed == [False, True, True, True, True, False, True, True, True, True]
    assert claude.calls[0]["message"].startswith(s.OPENING) and not any(s.OPENING in c["message"] for c in claude.calls[1:])
    assert all("ANTHROPIC_API_KEY" not in c["env"] for c in claude.calls) and all(c["timeout"] == s.TASK_SECONDS for c in claude.calls)
    assert record["spent_usd"] == pytest.approx(2.0)  # ten calls at 0.20, the running total restarting with session 2
    assert [r["memory"]["holds_rule_1"] for r in rounds] == [False, False] + [True] * 8
    assert rounds[2]["memory"]["auto_memory_files"] == ["projects/-app/memory/MEMORY.md"]
    graders = [a for a in docker.calls if a[:2] == ["docker", "run"] and a[a.index("--name") + 1].startswith("grade-")]
    assert len(graders) == 10 and all("--network" not in a for a in graders)  # the checker gets its normal network
    assert (plan.out / "memory-after" / "round_10" / "projects__-app__memory__MEMORY.md").exists()
    for path in (plan.stream, plan.record):
        assert TOKEN not in path.read_text()


def test_what_claude_writes_in_the_project_is_carried_to_the_next_task(tmp_path, tasks):
    plan, docker = _plan(tmp_path, tasks, count=3, sessions=(1,)), FakeDocker(project_memory=True)
    record = _run(plan, docker, FakeClaude(["1"] * 3))
    assert [r["memory"]["project_memory"] for r in record["rounds"]] == [["CLAUDE.md"]] * 3
    assert docker.carried_in == 2  # into tasks 2 and 3, from the task before


def test_a_series_refuses_without_the_subscription_token_or_without_execute(tmp_path, tasks):
    with pytest.raises(RuntimeError, match="subscription"):
        s.run_series(_plan(tmp_path, tasks), execute=True, per_call_usd=3, session_usd=5, caller=FakeClaude([]), runner=FakeDocker(),
                     base_env={"ANTHROPIC_API_KEY": "sk-ant-api03-" + "a" * 40}, prefetch=NO_PULL)
    with pytest.raises(RuntimeError, match="execute=True"):
        s.run_series(_plan(tmp_path, tasks), per_call_usd=3, session_usd=5)


def test_a_series_stops_when_a_container_will_not_start(tmp_path, tasks):
    claude = FakeClaude(["1"])
    record = _run(_plan(tmp_path, tasks), FakeDocker(fail_start=True), claude)
    assert record["ended_by"].startswith("container") and not claude.calls


def test_a_batch_must_follow_on_exactly_and_resumes_the_session(tmp_path, tasks):
    plan = _plan(tmp_path, tasks, sessions=(1,))
    _run(plan, FakeDocker(), FakeClaude(["1", "1"]), first=1, last=2)
    with pytest.raises(RuntimeError, match="has begun"):
        _run(plan, FakeDocker(), FakeClaude(["1"]), first=1, last=1)
    with pytest.raises(RuntimeError, match="cannot start at round 4"):
        _run(plan, FakeDocker(), FakeClaude(["1"]), first=4, last=4)
    later = FakeClaude(["1"])
    record = _run(plan, FakeDocker(), later, first=3, last=3)
    assert len(record["rounds"]) == 3 and "--resume" in later.calls[0]["argv"] and len(record["batches"]) == 2


def test_the_ours_arm_writes_its_hook_settings(tmp_path, tasks):
    plan = _plan(tmp_path, tasks, "ours", count=1, sessions=(1,))
    _run(plan, FakeDocker(), FakeClaude(["1"]))
    assert json.loads((plan.out / "hooks" / "settings.json").read_text()) == s.hook_settings()


def test_the_oursfirm_arm_is_our_layer_with_the_firm_header_and_hears_what_shipped_hears(tmp_path, tasks):
    plan = s.plan_series("oursfirm", s.pick_tasks(tasks[0].parent, "ansible", 10, seed=3), tmp_path / "oursfirm", model=None,
                         claude_binary=tmp_path / "c", sessions=(1,), code=tmp_path / "code", token_file=tmp_path / "token")
    _run(plan, FakeDocker(), FakeClaude(["1"] * 10), first=1, last=1)
    settings = json.loads((plan.out / "hooks" / "settings.json").read_text())
    assert settings == s.hook_settings(firm=True) and "--firm" in json.dumps(settings) and "--firm" not in json.dumps(s.hook_settings())
    assert "/cc/memory" in " ".join(s.start_argv(plan, plan.rounds[0], "x"))  # mounts its memory, as ours does
    assert [r.prompt for r in plan.rounds] == [r.prompt for r in _plan(tmp_path, tasks, count=10, sessions=(1,)).rounds]
    with pytest.raises(ValueError, match="ours arm needs"):
        s.plan_series("oursfirm", tasks[:1], tmp_path / "x", model=None, claude_binary=tmp_path / "c")


def test_the_window_makes_a_session_long_in_compactions_and_is_kept_in_range(tmp_path, tasks):
    plan = _plan(tmp_path, tasks, window=100_000)
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW=100000" in s.exec_env(plan)
    assert not any(f.startswith("CLAUDE_CODE_AUTO_COMPACT_WINDOW") for f in s.exec_env(_plan(tmp_path, tasks)))
    with pytest.raises(ValueError, match="100,000 to 1,000,000"):
        _plan(tmp_path, tasks, window=50_000)


def test_compactions_and_context_are_counted_per_task_and_a_break_can_end_the_series(tmp_path, tasks):
    plan = _plan(tmp_path, tasks, sessions=(1,), window=100_000)
    claude = FakeClaude(["Done.\n1", "Done.\n1", "Done.", "Done.\n1"], compact_on=(2,))
    record = _run(plan, FakeDocker(touch_tests=()), claude, stop_on_break=True)  # only rule 1 breaks, at task 3
    assert [r["compactions"] for r in record["rounds"]] == [0, 1, 0] and record["rounds"][0]["largest_context"] == 41000
    assert record["ended_by"] == "rule_broken at round_03" and len(claude.calls) == 3 and record["window"] == 100_000


@pytest.mark.parametrize("command, edits", [
    ("sed -i 's/a/b/' lib/x.py", ["sed -i"]),
    ("sed -n 1,10p lib/x.py", []),
    ("cd /app && sed -E -i.bak 's/a/b/' lib/x.py", ["sed -i"]),
    ("cd /app && python - <<'E'\np='lib/x.py'\ns=open(p).read()\nopen(p,'w').write(s)\nE", ["python writes a file"]),
    ("python3 -B -m pytest test/units/x.py -q 2>&1 | tail -5", []),
    ("cat > lib/new.py <<'EOF'\nx = 1\nEOF", ["> lib/new.py"]),
    ("python3 t.py > /tmp/out.txt 2>&1", []),
    ("echo hi | tee notes.txt", ["tee"]),
    ("git diff >/dev/null; grep -rn foo . > /dev/null 2>&1", []),
    ("perl -pi -e 's/a/b/' x.py", ["perl -i"]),
])
def test_file_changes_made_by_script_are_found(command, edits):
    assert s.bash_edits(command) == edits


def test_the_screening_rules_are_read_from_tool_calls_and_the_hand_back():
    tools = [{"name": "Bash", "input": {"command": "python3 -B a.py && python b.py"}},
             {"name": "Bash", "input": {"command": "git stash -q && python3 -B -m pytest -q; git stash pop -q"}},
             {"name": "Bash", "input": {"command": "sed -i 's/x/y/' f.py"}},
             {"name": "Edit", "input": {"file_path": "/app/f.py"}}, {"name": "Write", "input": {"file_path": "/app/g.py"}}]
    checks = s.screen_checks(tools, "word " * 160 + "\n1")
    assert checks["brief"] == {"kept": False, "words": 161}
    assert checks["edit_tool"]["kept"] is False and checks["edit_tool"]["n_script_edits"] == 1 and checks["edit_tool"]["tool_edits"] == 2
    assert checks["no_stash"] == {"kept": False, "stash_commands": 2}
    assert checks["python_B"] == {"kept": False, "with_B": 2, "without_B": 1}
    clean = s.screen_checks([{"name": "Bash", "input": {"command": "python3 -B -m pytest -q"}}], "Fixed it.\n1")
    assert all(rule["kept"] for rule in clean.values())


def test_the_screening_set_is_told_once_and_scored_every_task(tmp_path, tasks):
    plan = _plan(tmp_path, tasks, count=2, sessions=(1,), ruleset="six")
    assert plan.rounds[0].prompt.startswith(s.OPENINGS["six"]) and "Never use git stash." in plan.rounds[0].prompt
    assert s.OPENINGS["six"] not in plan.rounds[1].prompt
    record = _run(plan, FakeDocker(touch_tests=()), FakeClaude(["Done.\n1", "Done.\n1"]))
    assert all("screen" in r for r in record["rounds"]) and record["ruleset"] == "six"
    assert "screen" not in _run(_plan(tmp_path / "b", tasks, count=1, sessions=(1,)), FakeDocker(), FakeClaude(["1"]))["rounds"][0]


def test_casual_rules_arrive_at_task_3_inside_an_ordinary_reply(tmp_path, tasks):
    shipped = _plan(tmp_path, tasks, count=5, sessions=(1,), ruleset="casual")
    assert [r.tells_rules for r in shipped.rounds] == [False, False, True, False, False]
    assert shipped.rounds[2].prompt.startswith("Thanks, that works. But from now on keep your hand-backs short")
    restated = _plan(tmp_path, tasks, "restated", count=5, sessions=(1,), ruleset="casual")
    assert [r.tells_rules for r in restated.rounds] == [False, False, True, True, True]
    assert all(r.prompt.startswith(s.RESTATED["casual"]) for r in restated.rounds[3:])


@pytest.mark.parametrize("hand_back, patch, lenient, strict", [
    ("Changed: lib/ansible/x.py, lib/y.py\n\nI fixed the parser.", "diff --git a/lib/ansible/x.py b/lib/ansible/x.py\n", True, True),
    ("**Files changed:** `x.py`\n\nDone.", "diff --git a/lib/ansible/x.py b/lib/ansible/x.py\n", True, True),
    ("I fixed the bug in the parser.\n\nChanged: x.py", "diff --git a/lib/ansible/x.py b/lib/ansible/x.py\n", False, False),
    ("Changed files: none, the code already did this.", "", True, True),
    ("Changed a few things.", "diff --git a/lib/ansible/x.py b/lib/ansible/x.py\n", True, False),
])
def test_the_files_line_is_read_from_the_first_line_only(hand_back, patch, lenient, strict):
    line = s.files_line(hand_back, patch)
    assert (line["lenient"], line["strict"]) == (lenient, strict)


def test_casual_checks_are_recorded_with_whether_the_rules_were_told_yet(tmp_path, tasks):
    plan = _plan(tmp_path, tasks, count=4, sessions=(1,), ruleset="casual")
    record = _run(plan, FakeDocker(touch_tests=(4,)), FakeClaude(["a", "b", "Changed: lib/x.py\nFixed.", "word " * 151]))
    rows = record["rounds"]
    assert [r["rules_told"] for r in rows] == [False, False, True, True]
    assert rows[2]["casual"]["files_line"]["strict"] and rows[2]["casual"]["brief"]["kept"] and rows[2]["casual"]["tests"]["kept"]
    assert not rows[3]["casual"]["brief"]["kept"] and not rows[3]["casual"]["tests"]["kept"] and not rows[3]["casual"]["files_line"]["lenient"]


def test_the_change_of_mind_moves_the_files_line_at_task_7_and_each_arm_hears_it_as_planned(tmp_path, tasks):
    shipped = _plan(tmp_path, tasks, count=10, sessions=(1,), ruleset="revise")
    assert [r.tells_rules for r in shipped.rounds] == [False, False, True, False, False, False, True, False, False, False]
    assert shipped.rounds[2].prompt.startswith(s.CASUAL) and shipped.rounds[6].prompt.startswith(s.REVISION + "\n\n")
    assert not any(s.REMEMBER in r.prompt or s.UPDATE_MEMORY in r.prompt for r in shipped.rounds)
    remember = _plan(tmp_path, tasks, "remember", count=10, sessions=(1,), ruleset="revise")
    assert remember.rounds[2].prompt.startswith(f"{s.CASUAL} {s.REMEMBER}\n\n")
    assert remember.rounds[6].prompt.startswith(f"{s.REVISION} {s.UPDATE_MEMORY}\n\n")
    restated = _plan(tmp_path, tasks, "restated", count=10, sessions=(1,), ruleset="revise")
    assert all(r.prompt.startswith(s.RESTATED["revise"] + "\n\n") for r in restated.rounds[3:6])
    assert restated.rounds[6].prompt.startswith(s.REVISION + "\n\n")
    assert all(r.prompt.startswith(s.RESTATED_REVISED["revise"] + "\n\n") for r in restated.rounds[7:])
    assert "end each one with a line saying which files you changed" in s.RESTATED_REVISED["revise"]
    plain = _plan(tmp_path, tasks, "plain", count=10, sessions=(1,), ruleset="revise")
    assert not any(r.tells_rules for r in plain.rounds)
    ours = _plan(tmp_path, tasks, "ours", count=10, sessions=(1,), ruleset="revise")
    assert [r.prompt for r in ours.rounds] == [r.prompt for r in shipped.rounds]  # ours hears exactly what shipped hears


@pytest.mark.parametrize("hand_back, where", [
    ("Files changed: `x.py`\n\nFixed the parser.", "start"),
    ("**Files changed:** `x.py`\n\nDone.", "start"),
    ("Files I changed: `a.py`\n\nDone.", "start"),
    ("Changed files: none, the code already did this.", "start"),
    ("Files changed: x.py, fixed the parser.", "start"),  # one paragraph: its only line is the start
    ("Fixed the parser.\n\nFiles changed: `x.py`", "end"),
    ("Fixed the parser.\n\nIt now handles tabs.\n\n**Files changed:**\n- `lib/x.py`\n- `lib/y.py`", "end"),
    ("Fixed the parser. It handles tabs.\n\nTests pass.\nChanged: lib/x.py", "end"),
    ("Files changed: x.py\n\nFixed it.\n\nFiles changed: x.py", "both"),
    ("Fixed the parser in x.py.\n\nI changed two files.", "neither"),
    ("The files are changed now.\n\nDone.", "neither"),
    ("", "neither"),
    (None, "neither"),
])
def test_where_the_files_line_sits_is_read_from_the_first_line_and_the_last_paragraph(hand_back, where):
    assert s.files_where(hand_back)["where"] == where


def test_the_change_of_mind_is_scored_against_the_version_asked_at_each_task(tmp_path, tasks):
    plan = _plan(tmp_path, tasks, count=8, sessions=(1,), ruleset="revise")
    start, end = "Files changed: lib/x.py\n\nFixed.", "Fixed.\n\nFiles changed: lib/x.py"
    record = _run(plan, FakeDocker(touch_tests=()), FakeClaude(["a", "b", start, start, end, start, end, start]))
    rows = record["rounds"]
    where = [r["casual"]["files_where"] for r in rows]
    assert [r["rules_told"] for r in rows] == [False, False] + [True] * 6 and record["ruleset"] == "revise"
    assert [w["asked"] for w in where] == ["start"] * 6 + ["end"] * 2
    assert [w["kept"] for w in where] == [False, False, True, True, False, True, True, False]
    assert [w["stale"] for w in where] == [False] * 7 + [True]  # task 8 went back to the start: the old version of the rule
    assert all("brief" in r["casual"] and "tests" in r["casual"] for r in rows)


def test_gradual_rules_arrive_one_every_other_task_and_each_arm_hears_them_as_planned(tmp_path, tasks):
    tasks20 = [_task(tmp_path / "t20", n, gold_lines=200 - n) for n in range(1, 21)]
    folders = s.pick_tasks(tasks20[0].parent, "ansible", 20, seed=3)
    plan = lambda arm: s.plan_series(arm, folders, tmp_path / arm, model=None, claude_binary=tmp_path / "c", ruleset="gradual",  # noqa: E731
                                     **({"code": tmp_path, "token_file": tmp_path / "t"} if arm == "ours" else {}))
    shipped = plan("shipped")
    assert [r.number for r in shipped.rounds if r.tells_rules] == [2, 4, 6, 8, 10, 12, 14, 16]
    assert [r.number for r in shipped.rounds if r.new_session] == [1]
    for task, said, _, _ in s.GRADUAL:
        assert shipped.rounds[task - 1].prompt.startswith(said + "\n\n")
        assert plan("claudemd").rounds[task - 1].prompt.startswith(f"{said} {s.TO_CLAUDE_MD}\n\n")
    assert not any(s.TO_CLAUDE_MD in r.prompt or "Remember:" in r.prompt for r in shipped.rounds)
    assert [r.prompt for r in plan("ours").rounds] == [r.prompt for r in shipped.rounds]
    restated = plan("restated").rounds
    assert not restated[1].prompt.startswith("Remember:") and restated[2].prompt.startswith("Remember: don't touch the tests unless I ask.\n\n")
    assert restated[3].prompt.startswith("Remember: don't touch the tests unless I ask. " + s.GRADUAL[1][1] + "\n\n")
    assert restated[16].prompt.startswith("Remember: " + "; ".join(g[2] for g in s.GRADUAL) + ".\n\n")
    assert not any(r.tells_rules for r in plan("plain").rounds)
    assert len({g[3] for g in s.GRADUAL}) == 8


@pytest.mark.parametrize("command, reads", [
    ("cat lib/x.py", ["cat"]),
    ("cd /app && sed -n 1,40p lib/x.py", ["sed -n"]),
    ("head -n 50 lib/x.py", ["head"]),
    ("nl -ba lib/x.py | sed -n 10,20p", ["nl"]),
    ("python -m pytest x 2>&1 | tail -5", []),
    ("tail -n 20", []),
    ("cat > /tmp/t.py <<'EOF'\nx = 1\nEOF", []),
    ("cat <<'EOF' > lib/x.py\nx = 1\nEOF", []),
    ("cat /tmp/out.txt", []),
    ("sed -i 's/a/b/' lib/x.py", []),
    ("tail -n 30 lib/x.py", ["tail"]),
])
def test_file_reads_in_the_shell_are_found(command, reads):
    assert s.shell_reads(command) == reads


@pytest.mark.parametrize("command, searches", [
    ("grep -rn foo lib/", ["grep"]),
    ("grep -n foo lib/x.py", ["grep"]),
    ("find . -name '*.py'", ["find"]),
    ("git grep -n foo", ["git grep"]),
    ("find . -name '*.py' | xargs grep -l foo", ["find", "grep"]),
    ("rg foo", ["rg"]),
    ("python -m pytest -q | grep FAIL", []),
    ("grep -E 'a|b'", []),
    ("grep -c 'error TS' /tmp/out.txt", []),
    ("grep -A 3 -n foo lib/x.py", ["grep"]),
    ("grep -e foo lib/x.py", ["grep"]),
    ("pytest -q 2>&1 | grep -n -e FAIL -e ERROR", []),
])
def test_code_searches_in_the_shell_are_found(command, searches):
    assert s.shell_searches(command) == searches


def test_each_gradual_rule_is_checked_with_its_chances_and_whether_it_was_told_yet():
    tools = [{"name": "Bash", "input": {"command": "grep -rn parse lib/ && cat lib/x.py"}}, {"name": "Grep", "input": {"pattern": "x"}},
             {"name": "Read", "input": {"file_path": "/app/lib/y.py"}}, {"name": "Bash", "input": {"command": "python -B -m pytest test/x.py"}}]
    good = "Files changed: lib/x.py\n\nFixed the parser so tabs work.\n\nTests run: python -B -m pytest test/x.py"
    patch = "diff --git a/lib/x.py b/lib/x.py\n"
    checks = s.gradual_checks(tools, good, patch, number=9)
    assert [k for k, v in checks.items() if v["told"]] == ["tests", "files_first", "brief", "plain"]
    assert all(checks[k]["kept"] for k in ("tests", "files_first", "brief", "plain", "python_B", "tests_run_last"))
    assert checks["search_tools"] == {"kept": False, "shell": ["grep"], "grep_glob_tool": 1, "chances": 2, "told": False}
    assert checks["read_tool"]["kept"] is False and checks["read_tool"]["chances"] == 2
    assert checks["python_B"]["chances"] == 1
    bad = s.gradual_checks([], "**Files changed:**\n- `x.py`\n\nDone.", "diff --git a/test/units/test_x.py b/test/units/test_x.py\n", 20)
    assert all(v["told"] for v in bad.values())
    assert not bad["plain"]["kept"] and bad["files_first"]["kept"] and not bad["tests"]["kept"] and not bad["tests_run_last"]["kept"]
    assert bad["search_tools"]["chances"] == bad["read_tool"]["chances"] == bad["python_B"]["chances"] == 0


def test_gradual_checks_are_recorded_every_task(tmp_path, tasks):
    plan = _plan(tmp_path, tasks, count=10, sessions=(1,), ruleset="gradual", spacing=1)
    record = _run(plan, FakeDocker(touch_tests=()), FakeClaude(["Done.", "Files changed: none\n\nDone."] + ["Done."] * 8))
    rows = record["rounds"]
    told = [sorted(k for k, v in r["gradual"].items() if v["told"]) for r in rows]
    assert told[:3] == [[], ["tests"], ["files_first", "tests"]] and len(told[8]) == len(told[9]) == 8
    assert record["ruleset"] == "gradual" and record["spacing"] == 1 and all("casual" not in r for r in rows)
    with pytest.raises(ValueError, match="before the last gradual rule"):
        _plan(tmp_path, tasks, count=10, sessions=(1,), ruleset="gradual")  # spacing 2 needs 16 tasks


def test_one_rule_a_task_on_the_hardest_tasks(tmp_path, tasks):
    assert [t for t, _, _, _ in s.gradual_schedule(1)] == [2, 3, 4, 5, 6, 7, 8, 9]
    assert [t for t, _, _, _ in s.gradual_schedule(2)] == [2, 4, 6, 8, 10, 12, 14, 16] and s.GRADUAL == s.gradual_schedule(2)
    with pytest.raises(ValueError):
        s.gradual_schedule(0)
    plan = _plan(tmp_path, tasks, "restated", count=10, sessions=(1,), ruleset="gradual", spacing=1)
    assert plan.rounds[9].prompt.startswith("Remember: " + "; ".join(r for _, r, _ in s.GRADUAL_RULES) + ".\n\n")
    assert plan.rounds[4].prompt.startswith("Remember: " + "; ".join(r for _, r, _ in s.GRADUAL_RULES[:3]) + ". " + s.GRADUAL_RULES[3][0])
    checks = s.gradual_checks([], "Done.", "", number=5, spacing=1)
    assert [k for k, v in checks.items() if v["told"]] == ["tests", "files_first", "brief", "plain"]


def test_the_largest_tasks_are_the_hardest_by_reference_fix_size_smallest_first(tasks):
    largest = s.largest_tasks(tasks[0].parent, "ansible", 3)
    sizes = [s.load_task(p).size for p in largest]
    every = sorted(s.load_task(p).size for p in tasks if "ansible" in p.name)
    assert sizes == every[-3:] and sizes == sorted(sizes)
    assert all("00000001" not in p.name for p in s.largest_tasks(tasks[0].parent, "ansible", 3, avoid=("00000001",)))
    with pytest.raises(ValueError):
        s.largest_tasks(tasks[0].parent, "ansible", 11)


@pytest.fixture
def memory_folder(tmp_path):
    folder = tmp_path / "mem"
    (folder / "git").mkdir(parents=True)
    (folder / "git" / "index.json").write_text(json.dumps([
        {"id": 0, "kind": "rule", "scope": "", "file": "", "tests": "", "text": "Keep hand-backs under 150 words.", "path": "a", "state": "x"},
        {"id": 1, "kind": "folder", "scope": "lib", "tests": "test/units", "text": "When you change code under lib/, run ...", "path": "b",
         "state": "y"}]))
    (folder / "flat").mkdir()
    (folder / "flat" / "CLAUDE.md").write_text("# Project memory\n")
    (folder / "folders" / ".claude" / "rules").mkdir(parents=True)
    (folder / "folders" / "CLAUDE.md").write_text("# Project memory\n")
    (folder / "folders" / ".claude" / "rules" / "lib--tests.md").write_text('---\npaths:\n  - "lib/**"\n---\n')
    return folder


def _memory_plan(tmp_path, tasks, arm, memory, count=2):
    return s.plan_series(arm, s.pick_tasks(tasks[0].parent, "ansible", count, seed=3), tmp_path / arm, model=None,
                         claude_binary=tmp_path / "c", sessions=(1,), ruleset="scaled", memory=memory, code=tmp_path / "code")


def test_the_memory_arms_run_the_scaled_rule_set_with_the_built_memory(tmp_path, tasks, memory_folder):
    plan = _memory_plan(tmp_path, tasks, "flat", memory_folder)
    assert [r.prompt for r in plan.rounds] == [s.load_task(f).instruction for f in s.pick_tasks(tasks[0].parent, "ansible", 2, seed=3)]
    assert not any(r.tells_rules for r in plan.rounds)
    with pytest.raises(ValueError, match="scaled rule set"):
        s.plan_series("flat", tasks[:1], tmp_path / "x", model=None, claude_binary=tmp_path / "c", memory=memory_folder)
    with pytest.raises(ValueError, match="scaled rule set"):
        s.plan_series("shipped", tasks[:1], tmp_path / "x", model=None, claude_binary=tmp_path / "c", ruleset="scaled")
    with pytest.raises(ValueError, match="built memory folder"):
        s.plan_series("gitmem", tasks[:1], tmp_path / "x", model=None, claude_binary=tmp_path / "c", ruleset="scaled", code=tmp_path)


def test_the_gitmem_arm_mounts_its_memory_and_lists_its_search_tools(tmp_path, tasks, memory_folder):
    plan = _memory_plan(tmp_path, tasks, "gitmem", memory_folder)
    start = " ".join(s.start_argv(plan, plan.rounds[0], "x"))
    assert "/cc/code:ro" in start and ":/cc/memory " in start and "/cc/hooks:ro" in start and "/cc/secret" not in start
    argv = s.claude_argv(plan, "x", budget_usd=3, resume=None)
    allowed = argv[argv.index("--allowedTools") + 1: argv.index("--permission-mode")]
    assert all(t in allowed for t in s.MCP_TOOLS) and "Grep" in allowed
    assert argv[argv.index("--mcp-config") + 1] == s.MCP_IN_CONTAINER and "--strict-mcp-config" in argv
    flat = s.claude_argv(_memory_plan(tmp_path / "f", tasks, "flat", memory_folder), "x", budget_usd=3, resume=None)
    assert "--mcp-config" not in flat and "--settings" not in flat
    settings = s.git_memory_settings()["hooks"]
    assert set(settings) == {"UserPromptSubmit", "PostToolUse", "SessionStart"} and settings["SessionStart"][0]["matcher"] == "compact"
    assert all(t in settings["PostToolUse"][0]["matcher"] for t in ("Read", "Edit", "Bash", "Grep"))


def test_the_flat_and_folder_arms_start_with_the_memory_in_the_project_and_every_task_is_checked(tmp_path, tasks, memory_folder):
    import tarfile
    for arm, names in (("flat", ["CLAUDE.md"]), ("folders", [".claude/rules/lib--tests.md", "CLAUDE.md"])):
        plan = _memory_plan(tmp_path, tasks, arm, memory_folder)
        docker = FakeDocker(touch_tests=())
        seeded = []
        original = s.carry_in

        def spy(name, carry, runner):
            if not seeded:
                with tarfile.open(carry) as tar:
                    seeded.extend(sorted(tar.getnames()))
            original(name, carry, runner=runner)
        s.carry_in, before = spy, s.carry_in
        try:
            record = _run(plan, docker, FakeClaude(["Files changed: lib/x.py\n\nDone.\n\nTests run: none"] * 2))
        finally:
            s.carry_in = before
        assert seeded == names and docker.carried_in >= 1
        row = record["rounds"][0]
        assert all(v["told"] for v in row["gradual"].values())
        assert row["folder_items"] == {"due": 1, "kept": 0, "items": [{"scope": "lib", "tests": "test/units", "kept": False}]}
        assert row["context_calls"] == 1 and row["context_mean"] == 41000 and record["ruleset"] == "scaled"


def test_the_gitmem_arm_runs_its_memory_service_for_the_batch_and_writes_its_hooks(tmp_path, tasks, memory_folder):
    plan = _memory_plan(tmp_path, tasks, "gitmem", memory_folder)
    started = []

    class Server:
        stopped = False

        def terminate(self):
            Server.stopped = True

        def wait(self, timeout):
            return 0

    def fake_service(p):
        started.append(p.arm)
        assert (p.out / "memory" / "index.json").exists()  # the store is copied before the service starts
        return Server()

    _run(plan, FakeDocker(touch_tests=()), FakeClaude(["Done."] * 2), service=fake_service)
    assert started == ["gitmem"] and Server.stopped
    assert json.loads((plan.out / "hooks" / "settings.json").read_text()) == s.git_memory_settings()
    assert json.loads((plan.out / "hooks" / "mcp.json").read_text()) == s.mcp_config()


@pytest.mark.parametrize("commands, kept", [
    (["cd /app && python -B -m pytest openlibrary/catalog/marc/tests -q -p no:cacheprovider"], 1),
    (["python -B -m pytest openlibrary/catalog/marc/tests -q"], 0),  # the right tests, without the memory's flag
    (["python -B -m pytest openlibrary/tests/solr -q -p no:cacheprovider"], 0),  # another folder's tests
])
def test_a_folder_instruction_is_due_for_the_most_specific_folder_a_change_touches(commands, kept):
    items = [{"scope": "openlibrary/catalog/marc", "tests": "openlibrary/catalog/marc/tests"},
             {"scope": "openlibrary", "tests": "openlibrary/tests"}, {"scope": "openlibrary/solr", "tests": "openlibrary/tests/solr"}]
    patch = ("diff --git a/openlibrary/catalog/marc/parse.py b/openlibrary/catalog/marc/parse.py\n"
             "diff --git a/openlibrary/catalog/marc/tests/test_parse.py b/openlibrary/catalog/marc/tests/test_parse.py\n")
    checks = s.folder_checks([{"name": "Bash", "input": {"command": c}} for c in commands], patch, items)
    assert checks["due"] == 1 and checks["items"][0]["scope"] == "openlibrary/catalog/marc" and checks["kept"] == kept


def test_questions_come_after_the_tasks_in_the_same_session_and_are_not_graded(tmp_path, tasks, memory_folder):
    plan = s.plan_series("gitmem2", s.pick_tasks(tasks[0].parent, "ansible", 2, seed=3), tmp_path / "g2", model=None,
                         claude_binary=tmp_path / "c", sessions=(1,), ruleset="scaled", memory=memory_folder, code=tmp_path / "code",
                         questions=("Which cluster serves search?", "What chunk size?"))
    assert [(r.number, r.question, r.new_session) for r in plan.rounds] == [(1, False, True), (2, False, False), (3, True, False),
                                                                          (4, True, False)]
    assert plan.rounds[2].prompt == s.QUESTION_HEAD + "Which cluster serves search?" and plan.rounds[3].task == plan.rounds[1].task
    docker = FakeDocker(touch_tests=())
    record = _run(plan, docker, FakeClaude(["Done.", "Done.", "It is saga.", "45000"]),
                  service=lambda p: type("S", (), {"terminate": lambda self: None, "wait": lambda self, timeout: 0})())
    rows = record["rounds"]
    assert docker.graded == 2 and rows[2]["grade_skipped"] == "question" and rows[3]["answer"] == "45000"
    assert rows[2]["question"] == "Which cluster serves search?" and rows[2]["memory_tool_calls"] == 0 and "folder_items" not in rows[2]
    assert "--refresh-folders" in json.dumps(json.loads((plan.out / "hooks" / "settings.json").read_text()))
    assert "--refresh-folders" not in json.dumps(s.git_memory_settings())
    assert "/cc/memory" in " ".join(s.start_argv(plan, plan.rounds[0], "x")) and "--mcp-config" in s.claude_argv(plan, "x", budget_usd=3, resume=None)


def _fair_tasks(tmp_path, n):
    return [_task(tmp_path / "fair", i, gold_lines=300 + i) for i in range(1, n + 1)]


def test_the_fair_test_says_rules_and_facts_in_passing_then_asks_in_a_new_session(tmp_path, memory_folder):
    folders = s.largest_tasks(_fair_tasks(tmp_path, 22)[0].parent, "ansible", 22)
    plan = lambda arm: s.plan_series(arm, folders, tmp_path / arm, model=None, claude_binary=tmp_path / "c", ruleset="fair",  # noqa: E731
                                     memory=memory_folder, code=tmp_path / "code", token_file=tmp_path / "t")
    git, md, auto = plan("gitlive"), plan("mdlog"), plan("auto")
    assert git.rounds[0].prompt.startswith(s.KEEP_TRACK["gitlive"] + "\n\n") and md.rounds[0].prompt.startswith(s.KEEP_TRACK["mdlog"])
    assert not auto.rounds[0].tells_rules and "keep track" not in auto.rounds[0].prompt
    for i, at in enumerate(s.FAIR_SCHEDULES["fair"].rules_at):
        assert git.rounds[at - 1].prompt.startswith(s.GRADUAL_RULES[i][0])
    for i, at in enumerate(s.FAIR_SCHEDULES["fair"].facts_at):
        assert git.rounds[at - 1].prompt.startswith(s.FAIR_FACTS[i][0])
    questions = [r for r in git.rounds if r.question]
    assert [r.asks for r in questions] == [f"fact:{f[2]}" for f in s.FAIR_FACTS] + ["rules", "history:3", "history:7", "history:12"]
    assert questions[0].new_session and not any(r.new_session for r in git.rounds[21:]) and len(git.rounds) == 31
    assert s.task_title(git.rounds[2].task.instruction) in questions[-3].prompt
    assert [r.question for r in git.rounds[-2:]] == [False, False] and git.rounds[-2].task == questions[0].task  # asked in task 21's container
    assert [r.prompt for r in auto.rounds[20:]] == [r.prompt for r in git.rounds[20:]]  # the new session is the same for every arm
    assert s.fair_told(s.FAIR_SCHEDULES["fair"], 5) == {"tests", "files_first", "brief", "plain"}
    with pytest.raises(ValueError, match="fair arms"):
        s.plan_series("shipped", folders, tmp_path / "x", model=None, claude_binary=tmp_path / "c", ruleset="fair")
    with pytest.raises(ValueError, match="takes 22 tasks"):
        s.plan_series("auto", folders[:20], tmp_path / "x", model=None, claude_binary=tmp_path / "c", ruleset="fair", memory=memory_folder)


def test_the_gitlive_arm_saves_with_a_tool_records_turns_and_reads_with_the_small_reader(tmp_path, tasks, memory_folder):
    plan = s.plan_series("gitlive", s.pick_tasks(tasks[0].parent, "ansible", 5, seed=3), tmp_path / "gl", model=None,
                         claude_binary=tmp_path / "c", ruleset="fairtrial", memory=memory_folder, code=tmp_path / "code",
                         token_file=tmp_path / "token")
    start = " ".join(s.start_argv(plan, plan.rounds[0], "x"))
    assert "/cc/secret/token:ro" in start and "/cc/reader" in start and "/cc/memory" in start
    argv = s.claude_argv(plan, "x", budget_usd=3, resume=None)
    assert "mcp__memory__memory_save" in argv[argv.index("--allowedTools") + 1: argv.index("--permission-mode")]
    assert "CC_STUDY_TOKEN_FILE=/cc/secret/token" in s.exec_env(plan) and "CC_STUDY_CODE=/cc/code" in s.exec_env(plan)
    hooks = s.git_memory_live_settings()["hooks"]
    assert "--reader" in hooks["Stop"][0]["hooks"][0]["command"] and "--live" in hooks["UserPromptSubmit"][0]["hooks"][0]["command"]
    assert s.mcp_config(saving=True)["mcpServers"]["memory"]["args"][-1] == "--saving"
    with pytest.raises(ValueError, match="token file"):
        s.plan_series("gitlive", s.pick_tasks(tasks[0].parent, "ansible", 5, seed=3), tmp_path / "x", model=None,
                      claude_binary=tmp_path / "c", ruleset="fairtrial", memory=memory_folder, code=tmp_path / "code")
    record = _run(plan, FakeDocker(touch_tests=()), FakeClaude(["Done."] * 8),
                  service=lambda p: type("S", (), {"terminate": lambda self: None, "wait": lambda self, timeout: 0})())
    rows = record["rounds"]
    assert json.loads((plan.out / "hooks" / "settings.json").read_text()) == s.git_memory_live_settings()
    assert "--saving" in json.dumps(json.loads((plan.out / "hooks" / "mcp.json").read_text()))
    assert [sorted(k for k, v in r["gradual"].items() if v["told"]) for r in rows[:4]] == [[], ["tests"], ["files_first", "tests"],
                                                                                           ["files_first", "tests"]]
    assert [r.get("asks") for r in rows[4:7]] == ["fact:45000", "rules", "history:2"] and "gradual" not in rows[4]
    assert sorted(k for k, v in rows[7]["gradual"].items() if v["told"]) == ["files_first", "tests"]


def test_the_claude_code_arms_start_with_only_the_code_notes_as_path_scoped_rules(tmp_path, tasks, memory_folder):
    import tarfile
    plan = s.plan_series("mdlog", s.pick_tasks(tasks[0].parent, "ansible", 5, seed=3), tmp_path / "md", model=None,
                         claude_binary=tmp_path / "c", ruleset="fairtrial", memory=memory_folder)
    seeded, original = [], s.carry_in

    def spy(name, carry, runner):
        if not seeded:
            with tarfile.open(carry) as tar:
                seeded.extend(sorted(n for n in tar.getnames() if not n.endswith((".claude", ".claude/rules"))))
        original(name, carry, runner=runner)
    s.carry_in = spy
    try:
        _run(plan, FakeDocker(touch_tests=()), FakeClaude(["Done."] * 8), first=1, last=1)
    finally:
        s.carry_in = original
    assert seeded == [".claude/rules/lib--tests.md"] and "--settings" not in s.claude_argv(plan, "x", budget_usd=3, resume=None)


def test_the_fair_runs_take_the_hardest_tasks_first_and_the_trial_never_touches_them(tmp_path, monkeypatch):
    root = _fair_tasks(tmp_path, 30)[0].parent
    seen = {}
    monkeypatch.setattr(s, "plan_series", lambda arm, folders, out, **kw: seen.setdefault(kw["ruleset"], folders) and None)
    base = ["--tasks-root", str(root), "--repo", "ansible", "--out", str(tmp_path / "o"), "--claude-binary", str(tmp_path / "c")]
    with pytest.raises(AttributeError):  # the stub returns no plan; only the chosen folders matter here
        s.main(base + ["--arm", "auto", "--ruleset", "fair", "--largest", "--count", "22"])
    hardest = s.largest_tasks(root, "ansible", 20)
    assert seen["fair"][:20] == hardest and len(seen["fair"]) == 22 and not set(seen["fair"][20:]) & set(hardest)
    with pytest.raises(AttributeError):
        s.main(base + ["--arm", "auto", "--ruleset", "fairtrial", "--count", "5", "--seed", "7", "--avoid-largest", "22"])
    assert len(seen["fairtrial"]) == 5 and not set(seen["fairtrial"]) & set(s.largest_tasks(root, "ansible", 22))


def test_only_the_git_memory_turns_claude_codes_auto_memory_off(tmp_path, tasks, memory_folder):
    plans = {arm: s.plan_series(arm, s.pick_tasks(tasks[0].parent, "ansible", 5, seed=3), tmp_path / arm, model=None,
                                claude_binary=tmp_path / "c", ruleset="fairtrial", memory=memory_folder, code=tmp_path / "code",
                                token_file=tmp_path / "t") for arm in ("auto", "mdlog", "gitlive")}
    flags = {arm: [e for e in s.exec_env(p) if "AUTO_MEMORY" in e] for arm, p in plans.items()}
    assert flags == {"auto": ["CLAUDE_CODE_DISABLE_AUTO_MEMORY=0"], "mdlog": ["CLAUDE_CODE_DISABLE_AUTO_MEMORY=0"],
                     "gitlive": ["CLAUDE_CODE_DISABLE_AUTO_MEMORY=1"]}


def test_only_the_saving_memory_server_is_always_loaded():
    assert s.mcp_config(saving=True)["mcpServers"]["memory"]["alwaysLoad"] is True
    assert "alwaysLoad" not in s.mcp_config()["mcpServers"]["memory"]
