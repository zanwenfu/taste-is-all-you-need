"""What a report needs about a crash, read from the records a trial keeps anyway.

The terminal ledger says which commands ran on each side of the kill. Exactly
the same command after it is work done again: harmless or not, it is counted.
The call journals and the planner's receipts hold every request sent to a model.
The same request paid for twice would be the failure a durable record exists to
prevent, so it is counted across every journal, not within one. A call the kill
cut off is counted apart: it went out, may be billed, and its answer is lost, so
the decision it was for is asked again in a new request that is no repeat of it.
A record that cannot be read is reported as None, never as a clean zero.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path

# A call in one of these states may have been billed.
PAID = ("completed", "unknown", "lost", "pending")


def _reader(path):
    return sqlite3.connect(Path(path).absolute().as_uri() + "?mode=ro", uri=True)


def ledger_mark(path):
    """The ledger's position now, or None if it cannot be read."""
    try:
        with _reader(path) as ledger:
            return ledger.execute("SELECT max(seq) FROM events").fetchone()[0]
    except sqlite3.Error:
        return None


def commands_since(path, mark):
    """How many commands the ledger has admitted since ``mark``."""
    try:
        with _reader(path) as ledger:
            return ledger.execute("SELECT count(*) FROM events WHERE kind='intent' AND seq>?", (mark,)).fetchone()[0]
    except sqlite3.Error:
        return 0


def _commands(path, mark):
    empty = {"commands_before": None, "commands_after": None, "commands_repeated": None}
    if mark is None:
        return empty
    try:
        with _reader(path) as ledger:
            rows = ledger.execute("SELECT seq,payload FROM events WHERE kind='intent' ORDER BY seq").fetchall()
    except sqlite3.Error:
        return empty
    seen = [(seq, json.loads(payload)) for seq, payload in rows]
    keys = {(item["command"], item["cwd"]) for seq, item in seen if seq <= mark}
    after = [(item["command"], item["cwd"]) for seq, item in seen if seq > mark]
    return {"commands_before": len(seen) - len(after), "commands_after": len(after),
            "commands_repeated": sum(key in keys for key in after)}


def _journal_requests(workspace):
    requests = []
    for path in glob.glob(os.path.join(glob.escape(str(workspace)), ".git", "*", "*", "calls.sqlite3")):
        with _reader(path) as journal:
            marks = ",".join("?" * len(PAID))
            requests += [(hashlib.sha256(request.encode()).hexdigest(), status != "completed")
                         for request, status in journal.execute(
                             f"SELECT request,status FROM calls WHERE status IN ({marks})", PAID)]
    return requests


def _planner_requests(workspace):
    git = ("git", "-c", "safe.directory=*", "-C", str(workspace))

    def read(*arguments):
        return subprocess.run((*git, *arguments), capture_output=True, text=True, timeout=30, check=True).stdout

    refs = [line for line in read("for-each-ref", "--format=%(refname)", "refs/heads/mem").split()
            if line.endswith("/planner-receipts")]
    if len(refs) != 1:
        return []
    names = set(read("ls-tree", "-r", "--name-only", refs[0], ".taste/planner/transport/").split())
    requests = []
    for name in sorted(names):
        if name.endswith("/intent.json"):
            binding = json.loads(read("show", f"{refs[0]}:{name}"))["binding"]
            ending = name.removesuffix("intent.json") + "outcome.json"
            known = ending in names and json.loads(read("show", f"{refs[0]}:{ending}"))["telemetry"]["cost_known"]
            requests.append((binding["system_sha256"] + binding["prompt_sha256"], not known))
    return requests


def _paid_calls(workspace):
    requests = _journal_requests(workspace)
    try:
        requests += _planner_requests(workspace)
        planner_read = True
    except (subprocess.SubprocessError, OSError, ValueError, KeyError):
        planner_read = False
    sent = Counter(request for request, _ in requests)
    return {"paid_calls": len(requests), "paid_calls_repeated": len(requests) - len(sent),
            "paid_calls_cut_off": sum(cut_off for _, cut_off in requests), "planner_calls_read": planner_read}


def read_evidence(ledger, workspace, mark):
    return {**_commands(ledger, mark), **_paid_calls(workspace)}
