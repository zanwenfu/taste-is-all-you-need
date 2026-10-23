"""Public mini-swe-agent runs, replayed under the instrument: the parts that decide fidelity.

Hermetic: no Docker, no network. The replay's container seam is replaced by a
router that runs the recorded commands with the host's bash in a real git
workspace, so the shadow log observes real trees.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from taste.benchmarks import public_replay
from taste.benchmarks.public_replay import (
    NetworkedAgentProvider,
    StateLogExecutor,
    build_edit_paths,
    make_replay_execute,
    patch_files,
    same_patch,
    state_record,
)
from taste.benchmarks.public_trajectories import PublicTrajectory, RecordedCommand
from taste.replay import SuiteProbe, SuiteRun
from taste.shadow import load_timeline

PATCH = """diff --git a/pkg/mod.py b/pkg/mod.py
index 1111111..2222222 100644
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1 +1 @@
-x = 1
+x = 2
diff --git a/tox.ini b/tox.ini
index 3333333..4444444 100644
--- a/tox.ini
+++ b/tox.ini
@@ -1 +1 @@
-pytest
+pytest -rA
"""


def test_a_patch_splits_per_file_without_index_lines() -> None:
    files = patch_files(PATCH)
    assert sorted(files) == ["pkg/mod.py", "tox.ini"]
    assert "index " not in files["pkg/mod.py"] and "+x = 2" in files["pkg/mod.py"]


def test_patches_match_despite_blob_ids_and_trailing_space_but_not_content() -> None:
    other_blobs = PATCH.replace("1111111..2222222", "aaaaaaa..bbbbbbb").replace("+x = 2", "+x = 2   ")
    assert same_patch(PATCH, other_blobs) == (True, [])
    changed = PATCH.replace("+x = 2", "+x = 3")
    assert same_patch(PATCH, changed) == (False, ["pkg/mod.py"])


def test_the_image_builds_own_edits_are_ignored_and_only_they_are() -> None:
    """The original saw sphinx's tox.ini edit in its git diff; the replay's baseline already holds it."""
    without_tox = PATCH.split("diff --git a/tox.ini")[0]
    assert same_patch(PATCH, without_tox) == (False, ["tox.ini"])
    assert same_patch(PATCH, without_tox, ignore=("tox.ini",)) == (True, [])
    assert same_patch(PATCH, "", ignore=("tox.ini",)) == (False, ["pkg/mod.py"])
    # A directory entry covers what is under it, and nothing that merely shares a prefix.
    assert same_patch(PATCH, without_tox, ignore=("build/",))[0] is False


def test_build_edits_parse_from_porcelain_including_renames_and_untracked_dirs() -> None:
    porcelain = " M tox.ini\n?? build/\nR  old.py -> new.py\n M \"spaced name.txt\"\n\n"
    assert build_edit_paths(porcelain) == ("tox.ini", "build/", "new.py", "spaced name.txt")


@dataclass
class _Raw:
    output: str
    exit_code: int = 1
    timed_out: bool = False


def test_a_state_record_keeps_failures_unreported_members_and_why_apart() -> None:
    run = SuiteRun(statuses={"t::a": "pass", "t::b": "fail", "t::c": "error", "t::d": "skip"}, exit_code=1)
    record = state_record("abc", run, _Raw("....FAILED t::b\nImportError in c"), 3.14159)
    assert record["passed"] == 1 and record["failed"] == ["t::b"] and record["unreported"] == ["t::c"]
    assert record["skipped"] == ["t::d"] and record["ran"] is True and record["seconds"] == 3.14
    assert "ImportError" in record["output_tail"]
    clean = state_record("abc", SuiteRun(statuses={"t::a": "pass"}), _Raw("1 passed", exit_code=0), 1.0)
    assert clean["output_tail"] == "" and clean["failed"] == [] and clean["unreported"] == []
    # The suite never ran (the patch did not apply): infrastructure, said so.
    hole = state_record("abc", SuiteRun(statuses={"t::a": "error"}, infra_error="patch failed: x"), None, 0.5)
    assert hole["ran"] is False and hole["infra_error"] == "patch failed: x" and hole["timed_out"] is None


def test_the_state_log_keeps_the_raw_result_of_the_suite_command_only(tmp_path: Path, monkeypatch) -> None:
    class _Sandbox:
        workdir = "/testbed"

        def exec(self, command, *, timeout=600):
            if command == "run-suite":
                return _Raw("E   SyntaxError: invalid syntax", exit_code=2)
            return _Raw("", exit_code=0)

    class _Executor:
        """Shaped like SandboxProbeExecutor.run: suite command, then a cleanup command."""

        def __init__(self, sandbox, memory, base_commit) -> None:
            self.sandbox = sandbox

        def run(self, sha, suite):
            self.sandbox.exec(suite.command, timeout=suite.timeout)
            self.sandbox.exec("rm -f /tmp/patch")
            return SuiteRun(statuses=dict.fromkeys(suite.members, "error"), exit_code=2, infra_error="no results for any graded test")

        def close(self) -> None:
            pass

    monkeypatch.setattr(public_replay, "SandboxProbeExecutor", _Executor)
    log = tmp_path / "states.jsonl"
    executor = StateLogExecutor(_Sandbox(), None, "base", log_path=log)
    suite = SuiteProbe(name="p2p", command="run-suite", members=("t::a", "t::b"))
    executor.run("sha1", suite)
    (record,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert record["ran"] is True and record["exit_code"] == 2 and record["unreported"] == ["t::a", "t::b"]
    assert "SyntaxError" in record["output_tail"]  # the suite's output, not the cleanup's


def test_only_the_agent_container_gets_the_network() -> None:
    opened = []

    class _Provider:
        def open(self, *, key, image="", network_mode="none"):
            opened.append((key, network_mode))
            return key

    provider = NetworkedAgentProvider(_Provider())
    provider.open(key="agent:cell", image="img")
    provider.open(key="django__django-1", image="img")
    provider.open(key="grade:django__django-1", image="img", network_mode="bridge")
    assert opened == [("agent:cell", "bridge"), ("django__django-1", "none"), ("grade:django__django-1", "bridge")]


# ---------------------------------------------------------------- the replay loop


@dataclass
class _Exec:
    exit_code: int
    stdout: str
    stderr: str = ""
    timed_out: bool = False
    duration_s: float = 0.0


class _HostBash:
    """Runs each command with the host's bash in the workspace: real trees for the shadow log.

    Like the container's sandbox, it flags exit 124/137 as a timeout and reports its duration;
    a command over the limit is killed and returns 137."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.skipped: list[str] = []

    def exec(self, command: str, *, timeout: int = 600) -> _Exec:
        import time as _time

        started = _time.monotonic()
        try:
            done = subprocess.run(["bash", "-c", command], cwd=self.workspace, capture_output=True, text=True, timeout=timeout)
            code, out, err = done.returncode, done.stdout, done.stderr
        except subprocess.TimeoutExpired:
            code, out, err = 137, "", ""
        return _Exec(exit_code=code, stdout=out, stderr=err, timed_out=code in (124, 137),
                     duration_s=_time.monotonic() - started)


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "pkg").mkdir(parents=True)
    (ws / "pkg" / "mod.py").write_text("x = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t.co", "commit", "-qm", "base"], cwd=ws, check=True)
    return ws


def _trajectory(commands: list[tuple[str, int]], **overrides) -> PublicTrajectory:
    recorded = tuple(
        RecordedCommand(index=i, command=c, returncode=rc, output="", output_truncated=False, timed_out=False)
        for i, (c, rc) in enumerate(commands)
    )
    fields = dict(
        instance_id="pkg__pkg-1", mini_version="2.0.0", trajectory_format="test", exit_status="LimitsExceeded",
        submission="", commands=recorded, submit_index=None, sentinel=None, unexecuted_turns=0, timeout=30,
        env={"PAGER": "cat"}, cwd="/testbed", environment_class="docker", step_limit=None, cost_limit=None,
        api_calls=len(recorded), cost_usd=0.0,
    )
    fields.update(overrides)
    return PublicTrajectory(**fields)


def _ctx(tmp_path: Path, ws: Path) -> SimpleNamespace:
    return SimpleNamespace(
        workspace=ws, gitdir=ws / ".git" / "taste", agent_sandbox=SimpleNamespace(workdir="/testbed"),
        instance=SimpleNamespace(problem_statement="fix it", instance_id="pkg__pkg-1"), session="", router=None,
        shadow_ref="", llm_stats=None,
    )


def test_a_replay_reexecutes_the_recorded_commands_and_observes_each_tree_change(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(public_replay, "close_sandbox", lambda sandbox: None)
    ws = _workspace(tmp_path)
    trajectory = _trajectory([
        ("cat pkg/mod.py", 0),                       # read: no observation
        ("printf 'x = (\\n' > pkg/mod.py", 0),        # a broken edit
        ("python3 -c 'import sys; sys.exit(3)'", 0),  # recorded 0, replays 3: a fidelity miss, counted
        ("printf 'x = 2\\n' > pkg/mod.py", 0),        # the repair
        ("ls pkg", 0),
    ])
    ctx = _ctx(tmp_path, ws)
    execute = make_replay_execute(lambda cell: trajectory, source="sub", router_factory=lambda c: _HostBash(ws))
    result = execute(SimpleNamespace(key="pkg__pkg-1__MSWE__t1"), ctx)

    lines = [json.loads(line) for line in (ctx.gitdir / "replay.jsonl").read_text().splitlines()]
    assert [line["i"] for line in lines] == [0, 1, 2, 3, 4]
    assert [line["observed"] for line in lines] == [False, True, False, True, False]
    assert [line["rc_match"] for line in lines] == [True, True, False, True, True]
    manifest = json.loads((ctx.gitdir / "manifest.json").read_text())
    assert manifest["commands_replayed"] == 5 and manifest["observations"] == 2 and manifest["shadow_errors"] == 0
    assert manifest["submitted"] is False and ctx.replay_submission is None
    timeline = load_timeline(ctx.gitdir, ctx.session)
    assert [c.trigger for c in timeline] == ["run", "tool", "tool", "final"]
    assert [c.step_id for c in timeline[1:3]] == ["cmd-002", "cmd-004"]  # 1-based command numbers
    assert result.status == "failed" and result.failure_reason == "LimitsExceeded"

    out = tmp_path / "states.json.gz"
    assert public_replay.export_states(ws, timeline, out) == 4
    import gzip

    with gzip.open(out, "rt") as handle:
        states = json.load(handle)["states"]
    assert "+x = (" in states[1]["patch"] and "+x = 2" in states[2]["patch"] and states[0]["patch"] == ""


def test_a_replay_ends_where_the_original_submitted(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("minisweagent")
    monkeypatch.setattr(public_replay, "close_sandbox", lambda sandbox: None)
    ws = _workspace(tmp_path)
    submit = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached"
    trajectory = _trajectory(
        [("printf 'x = 2\\n' > pkg/mod.py", 0), (submit, 0)],
        exit_status="Submitted", submit_index=1, sentinel="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
    )
    ctx = _ctx(tmp_path, ws)
    execute = make_replay_execute(lambda cell: trajectory, router_factory=lambda c: _HostBash(ws))
    result = execute(SimpleNamespace(key="k"), ctx)
    assert result.status == "completed"
    assert "+x = 2" in ctx.replay_submission
    manifest = json.loads((ctx.gitdir / "manifest.json").read_text())
    assert manifest["submitted"] is True and manifest["commands_replayed"] == 2
    assert (ctx.gitdir / "replay_submission.diff").read_text() == ctx.replay_submission


def test_a_file_only_the_image_build_changed_is_excused_only_when_it_is_exactly_the_builds_edit() -> None:
    """Today's sphinx image commits its own setup.py pins; July 2025's left them uncommitted and runs submitted them."""
    agent = PATCH.split("diff --git a/tox.ini")[0]
    build = "diff --git a/tox.ini b/tox.ini\n--- a/tox.ini\n+++ b/tox.ini\n@@ -1 +1 @@\n-pytest\n+pytest -rA\n"
    assert same_patch(PATCH, agent, build=build) == (True, [])
    # The original's tox.ini change is not the build's: not excused.
    other = PATCH.replace("+pytest -rA", "+pytest -x")
    assert same_patch(other, agent, build=build) == (False, ["tox.ini"])
    # The agent edited a build-edited file too: compared as is, and differs.
    both = agent + build.replace("+pytest -rA", "+pytest -rA -q")
    assert same_patch(PATCH, both, build=build) == (False, ["tox.ini"])


def test_a_text_action_run_ends_on_the_sentinel_whatever_the_exit_status(tmp_path: Path, monkeypatch) -> None:
    """v1 accepted the sentinel on any exit status; today's environment wants a zero exit."""
    monkeypatch.setattr(public_replay, "close_sandbox", lambda sandbox: None)
    ws = _workspace(tmp_path)
    submit = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached; false"
    trajectory = _trajectory(
        [("printf 'x = 2\\n' > pkg/mod.py", 0), (submit, 1)],
        trajectory_format="mini-swe-agent-1", exit_status="Submitted", submit_index=1,
        sentinel="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
    )
    ctx = _ctx(tmp_path, ws)
    execute = make_replay_execute(lambda cell: trajectory, router_factory=lambda c: _HostBash(ws))
    result = execute(SimpleNamespace(key="k"), ctx)
    assert result.status == "completed" and "+x = 2" in ctx.replay_submission
    # The same command in a tool-call run does not end it: v2 demanded exit 0 as well.
    ws2 = _workspace(tmp_path / "v2")
    tool_run = _trajectory(
        [("printf 'x = 2\\n' > pkg/mod.py", 0), (submit, 1)],
        trajectory_format="mini-swe-agent-1.1", exit_status="Submitted", submit_index=1,
        sentinel="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
    )
    ctx2 = _ctx(tmp_path / "v2", ws2)
    make_replay_execute(lambda cell: tool_run, router_factory=lambda c: _HostBash(ws2))(SimpleNamespace(key="k"), ctx2)
    assert ctx2.replay_submission is None


def test_a_run_recorded_with_a_plain_shell_replays_outside_the_project_environment(tmp_path: Path, monkeypatch) -> None:
    """v2.0-2.3 ran ``docker exec -e K=V ... bash -c CMD`` with no BASH_ENV: the image's
    variables and the run's, no login shell, no conda activation (base Python 3.11
    where the project's env had 3.6). The replay must run them the same way."""
    from taste.benchmarks.public_replay import needs_plain_shell, plain_shell

    monkeypatch.setattr(public_replay, "close_sandbox", lambda sandbox: None)
    ws = _workspace(tmp_path)
    probe = 'echo "$PATH|$TZ|$PAGER|$HOME|${OUTER_ONLY:-unset}"'
    trajectory = _trajectory([(probe, 0)], interpreter=("bash", "-c"), env={"PAGER": "cat"})
    assert needs_plain_shell(trajectory)
    assert not needs_plain_shell(_trajectory([(probe, 0)], interpreter=("bash", "-c"), env={"BASH_ENV": "/root/.bashrc"}))
    assert not needs_plain_shell(_trajectory([(probe, 0)], interpreter=("bash", "-lc")))
    assert "bash -c" in plain_shell("ls", {"PATH": "/bin"}, {})

    ctx = _ctx(tmp_path, ws)
    monkeypatch.setenv("OUTER_ONLY", "leaked")
    execute = make_replay_execute(
        lambda cell: trajectory, router_factory=lambda c: _HostBash(ws), image_env={"PATH": "/usr/bin:/bin", "TZ": "Etc/UTC"},
    )
    execute(SimpleNamespace(key="k"), ctx)
    (line,) = [json.loads(x) for x in (ctx.gitdir / "replay.jsonl").read_text().splitlines()]
    assert line["replay_output_head"].strip() == "/usr/bin:/bin|Etc/UTC|cat|/root|unset"
    manifest = json.loads((ctx.gitdir / "manifest.json").read_text())
    assert manifest["shell"] == "plain"


def test_a_command_is_parsed_as_its_own_shell_string_as_the_scaffold_ran_it(tmp_path: Path, monkeypatch) -> None:
    """An unterminated heredoc only warns under ``bash -c CMD`` (the record says exit 0 and the
    file was written); spliced raw into a larger script it swallows the rest and the tree differs."""
    monkeypatch.setattr(public_replay, "close_sandbox", lambda sandbox: None)
    ws = _workspace(tmp_path)
    unterminated = "cat > NOTES.md << 'EOF'\n# notes\nall done"
    trajectory = _trajectory([(unterminated, 0), ("ls NOTES.md", 0)], interpreter=("bash", "-lc"))
    ctx = _ctx(tmp_path, ws)
    make_replay_execute(lambda cell: trajectory, router_factory=lambda c: _HostBash(ws))(SimpleNamespace(key="k"), ctx)
    lines = [json.loads(x) for x in (ctx.gitdir / "replay.jsonl").read_text().splitlines()]
    assert [line["replay_rc"] for line in lines] == [0, 0]
    assert (ws / "NOTES.md").read_text() == "# notes\nall done\n"
    assert lines[0]["observed"] is True


def test_a_generated_timestamp_is_the_only_thing_normalised() -> None:
    """Django's makemigrations stamps the wall-clock time into the file it writes; a replay months later cannot match it."""
    def migration(stamp: str, field: str = "name") -> str:
        return ("diff --git a/app/migrations/0001_initial.py b/app/migrations/0001_initial.py\nnew file mode 100644\n"
                "--- /dev/null\n+++ b/app/migrations/0001_initial.py\n@@ -0,0 +1,3 @@\n"
                f"+# Generated by Django 3.2 on {stamp}\n+\n+fields = ['{field}']\n")

    assert same_patch(migration("2025-08-08 02:59"), migration("2026-10-23 07:42")) == (True, [])
    assert same_patch(migration("2025-08-08 02:59"), migration("2026-10-23 07:42", field="title"))[0] is False


def test_a_run_whose_record_shows_no_git_author_replays_without_one(tmp_path: Path, monkeypatch) -> None:
    """Today's rebuilt images configure a git author; the 2025 images had none, so a recorded
    `git commit` (or `git stash`) failed then. Only the record can say which image a run saw."""
    from taste.benchmarks.public_replay import record_shows_no_git_author

    monkeypatch.setattr(public_replay, "close_sandbox", lambda sandbox: None)
    refused = "Author identity unknown\n\n*** Please tell me who you are.\n"
    without = _trajectory([("git commit -qam x", 128)])
    without = PublicTrajectory(**{**without.__dict__, "commands": (RecordedCommand(
        index=0, command="git commit -qam x", returncode=128, output=refused, output_truncated=False, timed_out=False),)})
    assert record_shows_no_git_author(without)
    assert not record_shows_no_git_author(_trajectory([("git commit -qam x", 0)]))

    issued: list[str] = []
    ws = _workspace(tmp_path)
    ctx = _ctx(tmp_path, ws)
    ctx.agent_sandbox = SimpleNamespace(workdir="/testbed", exec=lambda command, **kw: issued.append(command))
    make_replay_execute(lambda cell: without, router_factory=lambda c: _HostBash(ws))(SimpleNamespace(key="k"), ctx)
    assert issued and "--unset-all user.name" in issued[0] and "--unset-all user.email" in issued[0]
    assert json.loads((ctx.gitdir / "manifest.json").read_text())["git_author"] == "removed: the record shows none"


def test_an_inner_timeout_is_an_exit_status_and_only_a_real_one_is_a_timeout(tmp_path: Path, monkeypatch) -> None:
    """The sandbox flags any exit 124/137 as a timeout; an agent's own `timeout 1 ...` exits 124
    in a second, which the record shows as exit 124. Only reaching the run's limit is a timeout,
    and one side timing out while the other does not is a mismatch, never "not compared"."""
    monkeypatch.setattr(public_replay, "close_sandbox", lambda sandbox: None)
    ws = _workspace(tmp_path)
    trajectory = _trajectory([("exit 124", 124), ("sleep 6", 0), ("true", 0)], timeout=4)
    recorded = list(trajectory.commands)
    recorded[2] = RecordedCommand(index=2, command="true", returncode=None, output="", output_truncated=False, timed_out=True)
    trajectory = PublicTrajectory(**{**trajectory.__dict__, "commands": tuple(recorded)})
    ctx = _ctx(tmp_path, ws)
    make_replay_execute(lambda cell: trajectory, router_factory=lambda c: _HostBash(ws))(SimpleNamespace(key="k"), ctx)
    lines = [json.loads(x) for x in (ctx.gitdir / "replay.jsonl").read_text().splitlines()]
    assert (lines[0]["replay_rc"], lines[0]["replay_timed_out"], lines[0]["rc_match"]) == (124, False, True)
    assert (lines[1]["replay_timed_out"], lines[1]["rc_match"]) == (True, False)  # only the replay hit the limit
    assert (lines[2]["replay_timed_out"], lines[2]["rc_match"]) == (False, False)  # only the original did
    assert not any(line["transport_error"] for line in lines)


def test_a_transport_failure_is_recorded_on_its_command(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(public_replay, "close_sandbox", lambda sandbox: None)
    ws = _workspace(tmp_path)
    host = _HostBash(ws)
    calls = {"n": 0}
    real = host.exec

    def flaky(command, *, timeout=600):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("daemon hiccup")
        return real(command, timeout=timeout)

    host.exec = flaky
    trajectory = _trajectory([("true", 0), ("printf 'x = 2\\n' > pkg/mod.py", 0), ("true", 0)])
    ctx = _ctx(tmp_path, ws)
    make_replay_execute(lambda cell: trajectory, router_factory=lambda c: host)(SimpleNamespace(key="k"), ctx)
    lines = [json.loads(x) for x in (ctx.gitdir / "replay.jsonl").read_text().splitlines()]
    assert [line["transport_error"] for line in lines] == [False, True, False]
    assert "daemon hiccup" in lines[1]["exception_info"]
    manifest = json.loads((ctx.gitdir / "manifest.json").read_text())
    assert manifest["transport_errors"] == 1


def test_a_broken_state_keeps_every_error_line_of_its_full_output() -> None:
    """The analysis asks whether anything the agent saw contained the break's own error, so a
    broken state keeps every distinct error line of the suite's output, not only the tail's,
    with the file and line a syntax error names."""
    output = "\n".join([
        "Creating test database...",
        'File "/testbed/django/db/models/sql/compiler.py", line 139',
        "    having_group_by = self.having.get_group_by_cols() if self.having else ()",
        "NameError: name 'having_group_by' is not defined",
        *[f"test_{i} (aggregation.tests.T) ... ok" for i in range(400)],
        '  File "/testbed/pylint/lint/pylinter.py", line 614',
        "    if file.endswith('.py') and not _is_ignored_file(",
        "    ^",
        "E   SyntaxError: invalid syntax",
        "NameError: name 'having_group_by' is not defined",
        "django.db.transaction.TransactionManagementError: An error occurred in the current transaction.",
    ])
    run = SuiteRun(statuses={"t::a": "error"}, exit_code=1, infra_error="no results for any graded test")
    record = state_record("abc", run, _Raw(output), 1.0, tail_chars=200)
    assert record["exceptions"] == [
        "NameError: name 'having_group_by' is not defined",
        "SyntaxError: invalid syntax",
        "django.db.transaction.TransactionManagementError: An error occurred in the current transaction.",
    ]
    assert record["error_locations"] == ['File "/testbed/pylint/lint/pylinter.py", line 614']
    assert record["output_head"].startswith("Creating test database")
    clean = state_record("abc", SuiteRun(statuses={"t::a": "pass"}), _Raw("NameError: x", exit_code=0), 1.0)
    assert clean["exceptions"] == [] and clean["output_head"] == ""


def test_a_file_mode_is_not_content() -> None:
    """Today's rebuilt images ship some files executable where the 2025 images did not, so a copy or a
    move carries another mode into the patch; the lines are what a test reads."""
    body = "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
    plain = "diff --git a/pkg/mod.py b/pkg/mod.py\nindex 1..2 100644\n" + body
    moded = "diff --git a/pkg/mod.py b/pkg/mod.py\nold mode 100755\nnew mode 100644\nindex 1..2\n" + body
    assert same_patch(plain, moded) == (True, [])
    new_a = "diff --git a/b.bak b/b.bak\nnew file mode 100644\n--- /dev/null\n+++ b/b.bak\n@@ -0,0 +1 @@\n+x\n"
    assert same_patch(new_a, new_a.replace("100644", "100755")) == (True, [])
    assert same_patch(new_a, new_a.replace("+x", "+y"))[0] is False


def test_a_state_with_a_non_utf8_file_is_exported_byte_exact(tmp_path: Path) -> None:
    """A tree holding a non-UTF-8 file (sphinx-7462 under one model wrote 10 MB of one) must export,
    and its patch must come back byte for byte so the state can be re-tested."""
    import base64
    import gzip

    ws = _workspace(tmp_path)
    (ws / "notes.txt").write_bytes(b"caf\xe9 \x8f\n")
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t.co", "commit", "-qm", "x"], cwd=ws, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ws, capture_output=True, text=True).stdout.strip()
    timeline = [SimpleNamespace(seq=1, sha=head, step_id="cmd-001", trigger="tool", files=("notes.txt",))]
    out = tmp_path / "states.json.gz"
    assert public_replay.export_states(ws, timeline, out) == 1
    with gzip.open(out, "rt") as handle:
        (state,) = json.load(handle)["states"]
    assert "patch" not in state and b"caf\xe9 \x8f" in base64.b64decode(state["patch_b64"])


def test_error_locations_are_read_from_real_pytest_output() -> None:
    """pytest prints a syntax error's location with its `E` prefix, and in the
    `(file.py, line N)` form on its import-failure lines: both must be read, or every
    syntax-error break in a pytest project reads as unseen whatever the agent saw."""
    from taste.benchmarks.public_replay import error_evidence

    output = "\n".join([
        "ConftestImportFailure: SyntaxError: invalid syntax (pylinter.py, line 614) (from /testbed/tests/conftest.py)",
        "ImportError while loading conftest '/testbed/tests/conftest.py'.",
        "tests/conftest.py:15: in <module>",
        "    from pylint.lint import PyLinter",
        'E     File "/testbed/pylint/lint/pylinter.py", line 614',
        "E       if file.endswith(\".py\") and not _is_ignored_file(",
        "E       ^",
        "E   SyntaxError: invalid syntax",
        "E   IndentationError: unexpected indent (expand_modules.py, line 70)",
    ])
    exceptions, locations = error_evidence(output)
    assert "SyntaxError: invalid syntax" in exceptions
    assert "IndentationError: unexpected indent (expand_modules.py, line 70)" in exceptions
    assert 'File "/testbed/pylint/lint/pylinter.py", line 614' in locations
    assert 'File "pylinter.py", line 614' in locations
    assert 'File "expand_modules.py", line 70' in locations


def test_error_lines_are_read_through_colour_codes_and_from_pytest_summary_lines() -> None:
    """astropy and sphinx force coloured pytest output; and a summary line `FAILED t - Err: msg` is
    often the only place a test's error is printed."""
    from taste.benchmarks.public_replay import error_evidence, strip_colour

    coloured = "\n".join([
        "\x1b[31mE   AttributeError: 'Row' object has no attribute 'strip'\x1b[0m",
        "\x1b[31mFAILED\x1b[0m astropy/io/ascii/tests/test_rst.py::\x1b[1mtest_write_normal\x1b[0m - "
        "TypeError: unsupported operand\x1b[0m",
        "ERROR tests/test_x.py::test_y - ImportError: cannot import name 'Random'",
    ])
    assert strip_colour("\x1b[32mPASSED\x1b[0m t") == "PASSED t"
    exceptions, _ = error_evidence(coloured)
    assert exceptions == ["AttributeError: 'Row' object has no attribute 'strip'", "TypeError: unsupported operand",
                          "ImportError: cannot import name 'Random'"]


def test_rescanning_a_cell_adds_the_error_lines_a_coloured_output_hid(tmp_path: Path) -> None:
    """Recorded states keep their output's head and tail; a rescan reads them with today's rules and
    keeps the original file beside it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("driver", Path(__file__).resolve().parent.parent / "scripts" / "public_replay.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    cell = tmp_path / "cells" / "sub" / "astropy__astropy-1"
    cell.mkdir(parents=True)
    states = [
        {"sha": "a", "exceptions": [], "error_locations": [], "output_head": "", "output_tail": ""},
        {"sha": "b", "exceptions": ["NameError: x"], "error_locations": [], "output_head": "",
         "output_tail": "\x1b[31mE   AttributeError: 'Row' object has no attribute 'strip'\x1b[0m\n"},
    ]
    (cell / "states.jsonl").write_text("".join(json.dumps(s) + "\n" for s in states))
    (cell / "status.json").write_text(json.dumps({"state": "done"}))
    changed = driver.rescan_evidence(tmp_path)
    assert changed == {"cells": 1, "states": 1}
    after = [json.loads(x) for x in (cell / "states.jsonl").read_text().splitlines()]
    assert after[0]["exceptions"] == [] and after[1]["exceptions"] == [
        "NameError: x", "AttributeError: 'Row' object has no attribute 'strip'"]
    assert after[1]["evidence_rescanned"] is True
    assert (cell / "states.recorded.jsonl").read_text().count("NameError") == 1
    assert driver.rescan_evidence(tmp_path) == {"cells": 0, "states": 0}  # idempotent
