"""A trial's final change: git's diff of the task's files against the commit they started from (ISSTA plan, item 13).

The restarts' final code was never kept, so where the reviewer was right by the task text nobody can say whether
keeping going kept the gap it named or a restart closed it. With ``final_change_chars`` a trial's owner reads the base
commit before anything changes the files and, once the agent has ended, the paths and the diff against it, into the
trial's record. Here the reads run against real git repositories (a shell stands in for the task's terminal), and
through the Harbor agent with Harbor replaced, as tests.test_harbor_agent does.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from taste.benchmarks import final_change
from taste.benchmarks.harbor_settings import TrialSettings
from taste.brains.terminal_broker import MAX_TERMINAL_OUTPUT_BYTES, TerminalResult
from taste.recovery_study import jobs, records
from tests.test_harbor_agent import harbor_agent as _harbor_agent

harbor_agent = _harbor_agent
ALONE = {"model": "gpt-5.6-luna", "agent": "mini-swe-agent", "services": "none"}


class Shell:
    """The task's terminal, as a shell on this machine: each command in a fresh /bin/sh, its streams kept apart."""

    def __init__(self, home, limit=MAX_TERMINAL_OUTPUT_BYTES):
        self.temporary = home / "tmp"
        self.temporary.mkdir(exist_ok=True)
        self.environment = {**os.environ, "HOME": str(home), "GIT_CONFIG_NOSYSTEM": "1",
                            "TMPDIR": str(self.temporary)}
        self.limit, self.requests = limit, []

    def execute(self, request):
        self.requests.append(request)
        done = subprocess.run(["/bin/sh", "-c", request.command], cwd=request.cwd, capture_output=True,
                              timeout=request.timeout_seconds, env=self.environment)
        out, err = done.stdout, done.stderr
        return TerminalResult(done.returncode, out[:self.limit], err[:self.limit], max(0, len(out) - self.limit),
                              max(0, len(err) - self.limit))


def git(repo, *arguments):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *arguments], cwd=repo,
                          check=True, capture_output=True, text=True, env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1",
                                                                          "HOME": str(repo.parent)}).stdout


@pytest.fixture
def repo(tmp_path):
    """A task repository at its base commit: two modules and an ignore rule for build output."""
    path = tmp_path / "app"
    path.mkdir()
    git(path, "init", "-q")
    (path / "parser.py").write_text("def parse(text):\n    return text.split()\n")
    (path / "util.py").write_text("X = 1\n")
    (path / ".gitignore").write_text("build/\n")
    git(path, "add", "-A")
    git(path, "commit", "-qm", "base")
    return path


def agent_works(repo):
    """What an agent leaves: an edit, a deletion, a commit of its own, a new file, a binary, ignored build output, and
    a file written again as it was (git would refresh the index for it, given the chance)."""
    (repo / "parser.py").write_text("def parse(text):\n    if not text:\n        return []\n    return text.split()\n")
    (repo / "util.py").unlink()
    (repo / "dates.py").write_text("OFFSET = '+02:00'\n")
    git(repo, "add", "dates.py")
    git(repo, "commit", "-qm", "the agent's own commit")
    (repo / "test_parse.py").write_text("from parser import parse\n\ndef test_empty():\n    assert parse('') == []\n")
    (repo / "logo.bin").write_bytes(bytes(range(256)) * 4)
    (repo / "build").mkdir()
    (repo / "build" / "out.o").write_text("compiled\n")
    stat = (repo / ".gitignore").stat()
    (repo / ".gitignore").write_text("build/\n")
    os.utime(repo / ".gitignore", ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))


def test_the_change_is_the_diff_against_the_base_commit_with_new_files_and_without_ignored_ones(tmp_path, repo):
    shell = Shell(tmp_path)
    base, why = final_change.base_commit(shell, str(repo))
    assert why == "" and base == git(repo, "rev-parse", "HEAD").strip()
    agent_works(repo)
    index = (repo / ".git" / "index").read_bytes()
    found = final_change.final_change(shell, str(repo), base, 50_000)
    assert found["read"] and found["complete"] and not found["cut"] and found["cap_chars"] == 50_000
    # The agent's own commit moved HEAD; its change is in the diff all the same, which is against the base.
    assert found["head_moved"] and found["head"] == git(repo, "rev-parse", "HEAD").strip()
    assert found["paths"] == ["A\tdates.py", "M\tparser.py", "D\tutil.py", "?\tlogo.bin", "?\ttest_parse.py"]
    assert found["new_files"] == 2 and found["paths_more"] == 0
    diff = found["diff"]
    assert "+    if not text:" in diff and "+OFFSET = '+02:00'" in diff and "deleted file mode" in diff
    assert "diff --git a/test_parse.py b/test_parse.py" in diff and "+def test_empty():" in diff
    assert "Binary files /dev/null and b/logo.bin differ" in diff
    assert "build/out.o" not in diff and "compiled" not in diff
    assert found["diff_chars"] == len(diff)
    # The repository is as the agent left it, the index too (git refreshed its copy, which is gone).
    assert (repo / ".git" / "index").read_bytes() == index and list(shell.temporary.iterdir()) == []
    assert all(request.actor_id == final_change.ACTOR for request in shell.requests)


def test_a_long_change_is_kept_up_to_the_cap_and_says_so(tmp_path, repo):
    shell = Shell(tmp_path)
    base, _ = final_change.base_commit(shell, str(repo))
    (repo / "parser.py").write_text("".join(f"line_{number} = {number}\n" for number in range(2000)))
    found = final_change.final_change(shell, str(repo), base, 1000)
    assert found["cut"] and found["complete"] and len(found["diff"]) == 1000 and found["diff_chars"] > 20_000
    assert found["paths"] == ["M\tparser.py"]


def test_output_the_terminal_dropped_is_a_cut_change_and_an_unfinished_read_says_why(tmp_path, repo):
    base, _ = final_change.base_commit(Shell(tmp_path), str(repo))
    (repo / "parser.py").write_text("".join(f"line_{number} = {number}\n" for number in range(2000)))
    found = final_change.final_change(Shell(tmp_path, limit=4096), str(repo), base, 50_000)
    assert found["read"] and not found["complete"] and found["cut"] and found["dropped_bytes"] > 0
    assert found["paths"] == ["M\tparser.py"]
    stopped = final_change.read_change(TerminalResult(128, b"", b"fatal: not a git repository"), base, 5000)
    assert stopped == {"read": False, "reason": "fatal: not a git repository", "base": base, "cap_chars": 5000}


def test_outside_a_git_repository_there_is_no_base_and_nothing_to_read(tmp_path):
    (tmp_path / "plain").mkdir()
    base, why = final_change.base_commit(Shell(tmp_path), str(tmp_path / "plain"))
    assert base is None and "not a git repository" in why

    class Broken:
        def execute(self, request):
            raise ConnectionError("the terminal is gone")

    assert final_change.base_commit(Broken(), "/app") == (None, "the base read failed: ConnectionError")


def test_only_a_full_commit_name_reaches_the_command():
    for bad in ("HEAD", "abc123", "a" * 40 + "; rm -rf /", None):
        with pytest.raises(ValueError, match="commit"):
            final_change.change_command(bad)
    assert "a" * 40 in final_change.change_command("a" * 40)


# -- the setting, the trial's record


def test_the_setting_is_off_by_default_bounded_and_disclosed_only_when_set():
    off = TrialSettings.from_options(ALONE)
    assert off.final_change_chars == 0 and "final_change_chars" not in off.disclosure()
    on = TrialSettings.from_options({**ALONE, "final_change_chars": "60000"})
    assert on.final_change_chars == 60_000 and on.disclosure()["final_change_chars"] == 60_000
    for bad in ("-1", "999", "1000001"):
        with pytest.raises(ValueError, match="final_change_chars"):
            TrialSettings.from_options({**ALONE, "final_change_chars": bad})
    # Each job sets it for itself: a later round never inherits it from the trial it goes on from.
    assert "final_change_chars" not in jobs.inherited({"final_change_chars": 60000, "worker_effort": "low"})


def owner(harbor_agent, tmp_path, **options):
    agent = harbor_agent.TasteAgent(tmp_path, model_name="azure/gpt-6-luna", agent="mini-swe-agent", services="none",
                                    worker_python="/usr/bin/python3", **options)
    agent._workdir = str(tmp_path / "app")
    return agent


def test_the_trial_owner_reads_the_base_first_and_the_change_after_the_final_save(harbor_agent, tmp_path, repo):
    shell = Shell(tmp_path)
    agent = owner(harbor_agent, tmp_path, final_change_chars="60000")
    agent._change_base = asyncio.run(agent._read_base(shell))
    agent_works(repo)
    summary = {"trial": "t0ken", "configuration": {}}
    asyncio.run(agent._save_checkpoints(shell, None, summary, "t0ken"))
    found = summary["final_change"]
    assert found["read"] and found["head_moved"] and "?\ttest_parse.py" in found["paths"]
    assert [request.request_id for request in shell.requests] == ["final-change-base", "final-change-read"]


def test_without_the_setting_the_owner_runs_nothing_and_records_nothing(harbor_agent, tmp_path, repo):
    shell = Shell(tmp_path)
    agent = owner(harbor_agent, tmp_path)
    assert asyncio.run(agent._read_base(shell)) == (None, "")
    summary = {"trial": "t0ken", "configuration": {}}
    asyncio.run(agent._save_checkpoints(shell, None, summary, "t0ken"))
    assert "final_change" not in summary and shell.requests == []


def test_a_change_that_cannot_be_read_is_recorded_and_the_trial_goes_to_its_verifier(harbor_agent, tmp_path):
    agent = owner(harbor_agent, tmp_path, final_change_chars="5000")
    summary = {"trial": "t0ken", "configuration": {}}
    agent._change_base = (None, "/app is not a git repository with a commit when the trial starts")
    asyncio.run(agent._save_checkpoints(SimpleNamespace(), None, summary, "t0ken"))
    assert summary["final_change"] == {"read": False, "cap_chars": 5000,
                                       "reason": "/app is not a git repository with a commit when the trial starts"}

    class Broken:
        def execute(self, request):
            raise ConnectionError("the terminal is gone")

    agent._change_base = ("a" * 40, "")
    asyncio.run(agent._save_checkpoints(Broken(), None, summary, "t0ken"))
    assert summary["final_change"] == {"read": False, "reason": "the read failed: ConnectionError", "base": "a" * 40,
                                       "cap_chars": 5000}


def test_a_driver_reads_the_change_from_the_trials_record(tmp_path):
    from tests.recovery_fakes import trial

    change = {"read": True, "diff": "diff --git a/x b/x\n", "diff_chars": 19, "cap_chars": 5000, "cut": False}
    name = trial(tmp_path / "jobs", "job-1", "task-a", reward=1.0, trials_root=tmp_path / "trials")
    path = tmp_path / "jobs" / "job-1" / name / "result.json"
    result = json.loads(path.read_text())
    # A trial that read none has no such field: an earlier study's summaries are as they were.
    assert "final_change" not in records.trial(path.parent, tmp_path / "trials")
    result["agent_result"]["metadata"]["taste"]["final_change"] = change
    path.write_text(json.dumps(result))
    assert records.trial(path.parent, tmp_path / "trials")["final_change"] == change
