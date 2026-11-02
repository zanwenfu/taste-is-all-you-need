"""What a report reads about a crash: what ran on both sides of it, and whether any paid call went twice."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

from taste.benchmarks.crash_evidence import ledger_mark, read_evidence


def ledger(path: Path, commands):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE events (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL)")
    db.execute("INSERT INTO events(kind,payload) VALUES ('created','{}')")
    for command in commands:
        db.execute("INSERT INTO events(kind,payload) VALUES ('intent',?)",
                   (json.dumps({"command": command, "cwd": "/app"}),))
        db.execute("INSERT INTO events(kind,payload) VALUES ('completed','{}')")
    db.commit()
    db.close()


def add(path: Path, commands):
    db = sqlite3.connect(path)
    for command in commands:
        db.execute("INSERT INTO events(kind,payload) VALUES ('intent',?)",
                   (json.dumps({"command": command, "cwd": "/app"}),))
    db.commit()
    db.close()


def journal(path: Path, requests):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE calls (id TEXT PRIMARY KEY, request TEXT NOT NULL, status TEXT NOT NULL, "
               "result TEXT, digest TEXT, error_type TEXT)")
    for number, (request, status) in enumerate(requests):
        db.execute("INSERT INTO calls(id,request,status) VALUES (?,?,?)", (f"call-{number}", request, status))
    db.commit()
    db.close()


def test_commands_are_counted_on_both_sides_of_the_kill_and_the_same_command_again_is_seen(tmp_path):
    path = tmp_path / "terminal.sqlite3"
    ledger(path, ["ls", "make"])
    mark = ledger_mark(path)
    add(path, ["ls", "make test", "ls"])
    found = read_evidence(path, tmp_path / "none", mark)
    assert found["commands_before"] == 2 and found["commands_after"] == 3
    assert found["commands_repeated"] == 2, "ls ran before the kill and twice after it"


def test_the_same_paid_request_twice_is_counted_across_every_journal_and_a_refused_one_is_not(tmp_path):
    workspace = tmp_path / "workspace"
    journal(workspace / ".git/taste.one/worker/calls.sqlite3",
            [("first", "completed"), ("second", "completed"), ("third", "failed")])
    journal(workspace / ".git/taste.two/worker/calls.sqlite3",
            [("second", "completed"), ("third", "failed"), ("fourth", "unknown")])
    journal(workspace / ".git/taste.two/monitor/calls.sqlite3", [("fourth", "completed")])
    found = read_evidence(tmp_path / "none", workspace, 0)
    assert found["paid_calls"] == 5 and found["paid_calls_repeated"] == 2 and found["paid_calls_cut_off"] == 1


def test_missing_records_are_reported_as_missing_not_as_clean(tmp_path):
    found = read_evidence(tmp_path / "none", tmp_path / "none", 0)
    assert found["commands_before"] is None and found["commands_repeated"] is None
    assert found["paid_calls"] == 0 and found["paid_calls_repeated"] == 0
    assert found["planner_calls_read"] is False


def test_the_same_prompt_sent_to_the_planner_twice_is_counted(tmp_path):
    git = ("git", "-C", str(tmp_path / "repo"))

    def intent(number, prompt, cost_known=True):
        path = tmp_path / "repo" / ".taste/planner/transport" / f"attempt-{number}" / "intent.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"binding": {"system_sha256": "s", "prompt_sha256": prompt}}))
        if cost_known is not None:
            (path.parent / "outcome.json").write_text(json.dumps({"telemetry": {"cost_known": cost_known}}))

    (tmp_path / "repo").mkdir()
    subprocess.run(("git", "init", "-q", "-b", "mem/terminal-trial/planner-receipts", str(tmp_path / "repo")), check=True)
    intent(1, "first")
    intent(2, "second", cost_known=None)
    intent(3, "first", cost_known=False)
    subprocess.run((*git, "add", "-A", "-f"), check=True)
    subprocess.run((*git, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x"), check=True)
    found = read_evidence(tmp_path / "none", tmp_path / "repo", 0)
    assert found["planner_calls_read"] is True
    assert found["paid_calls"] == 3 and found["paid_calls_repeated"] == 1
    assert found["paid_calls_cut_off"] == 2, "no outcome, and an outcome of unknown cost"
