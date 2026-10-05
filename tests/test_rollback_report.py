"""The rollback report reads a trial's ledger and control records, and nothing else."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "rollback_report", Path(__file__).resolve().parents[1] / "scripts" / "rollback_report.py")
rollback_report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rollback_report)


def _ledger(root, events):
    terminal = root / "controller" / "terminal"
    terminal.mkdir(parents=True)
    connection = sqlite3.connect(terminal / "terminal.sqlite3")
    with connection:
        connection.execute("CREATE TABLE events (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL)")
        for kind, payload in events:
            connection.execute("INSERT INTO events(kind,payload) VALUES (?,?)", (kind, json.dumps(payload)))
    connection.close()
    (terminal / "checkpoints").mkdir()
    (terminal / "checkpoints" / ("a" * 64 + ".tar")).write_bytes(b"x" * 3000)


def _memory(root, records):
    repository = root / "agent-state" / "workspace"
    repository.mkdir(parents=True)

    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-C", str(repository), *args],
                       check=True, capture_output=True)

    git("init", "-q")
    git("checkout", "-q", "-b", "mem/session/central-control")
    for path, value in records.items():
        target = repository / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))
    git("add", "-A")
    git("commit", "-q", "-m", "records")


def _result(job, name, task, token, reward):
    trial = job / name
    trial.mkdir(parents=True)
    trial.joinpath("result.json").write_text(json.dumps({
        "task_name": f"terminal-bench/{task}", "trial_name": name,
        "verifier_result": {"rewards": {"reward": reward}},
        "agent_result": {"metadata": {"taste": {"trial": token, "audit_flags": []}}}}))


def test_the_report_counts_checkpoints_and_restores_with_the_planners_reason(tmp_path, capsys, monkeypatch):
    trials, job = tmp_path / "trials", tmp_path / "jobs" / "pilot"
    token = "f" * 32
    _ledger(trials / token, [
        ("checkpoint", {"checkpoint_id": "initial", "bytes": 10240, "added": 0, "changed": 0, "deleted": 0,
                        "partial": False}),
        ("checkpoint", {"checkpoint_id": "after_1", "bytes": 20480, "added": 2, "changed": 1, "deleted": 0,
                        "partial": False}),
        ("checkpoint_failed", {"checkpoint_id": "after_2", "failed": True, "error": "OSError"}),
        ("restored", {"operation_id": "undo_1", "checkpoint_id": "initial", "exact": True, "seconds": 2.5,
                      "removed": 1, "from_image": 3, "deleted": 0, "mismatches": []}),
        ("restore_failed", {"operation_id": "undo_2", "checkpoint_id": "after_1", "failed": True,
                            "error": "TerminalConflict"}),
        ("checkpoint_files_discarded", {"bytes": 40960}),
    ])
    _memory(trials / token, {
        ".taste/environment/goals/g/restores/b.json": {"generation": 4, "at": "2026-10-05T01:00:00Z", "label": "cp2",
                                                       "reason": "a later run broke it", "result": {"failed": True}},
        ".taste/environment/goals/g/restores/a.json": {"generation": 3, "at": "2026-10-05T00:59:00Z", "label": "cp1",
                                                       "reason": "tests passed before run 2", "result": {"exact": True}},
        ".taste/central-runtime/other.json": {"not": "an environment record"},
    })
    _result(job, "kv-store-grpc__1", "kv-store-grpc", token, 1.0)
    _result(job, "polyglot-c-py__1", "polyglot-c-py", "e" * 32, 0.0)
    # A trial with a ledger but no memory repository to read.
    _ledger(trials / ("d" * 32), [("checkpoint", {"checkpoint_id": "initial", "bytes": 10240})])
    _result(job, "sanitize-git-repo__1", "sanitize-git-repo", "d" * 32, 1.0)
    output = tmp_path / "report.json"
    monkeypatch.chdir(tmp_path)
    # A relative trials path works as well as an absolute one.
    assert rollback_report.main(["--job", "jobs/pilot", "--trials", "trials", "--json", str(output)]) == 0
    report = json.loads(output.read_text())
    summary = report["summary"]
    assert summary["trials"] == 3 and summary["with_ledger"] == 2 and summary["solved"] == 2
    assert summary["trials_without_control_branch"] == 1 and summary["discarded_bytes_total"] == 40960
    assert (summary["checkpoints"], summary["checkpoints_failed"]) == (4, 1)
    assert (summary["restores"], summary["restores_exact"], summary["restores_failed"]) == (2, 1, 1)
    assert summary["restore_seconds_max"] == 2.5 and summary["store_bytes_total"] == 6000
    row = next(item for item in report["rows"] if item["task"] == "kv-store-grpc")
    assert row["rollbacks"] == [
        {"generation": 3, "to": "cp1", "reason": "tests passed before run 2", "result": "exact"},
        {"generation": 4, "to": "cp2", "reason": "a later run broke it", "result": "failed"}]
    printed = capsys.readouterr().out
    assert "g3 to cp1 (exact)" in printed and "no control branch" in printed
