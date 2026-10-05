#!/usr/bin/env python3
"""What the coordinator's checkpoints and rollbacks did in a Harbor job's trials.

    sudo python3 scripts/rollback_report.py --job /root/tb/jobs/<job> [--trials /var/lib/taste-trials] [--json out.json]

For each trial: the verifier's reward; from the controller's terminal ledger,
every checkpoint (its size and counts, or its failure) and every restore (to
which checkpoint, exact or not, how long, or its failure); from the
coordinator's control branch, the checkpoint the plan named and the reason it
gave. Per job: totals, restore times and the checkpoint stores' size on disk.
Reads Harbor's result files, each trial's ledger (opened read-only) and its
memory repository (with git, read-only); writes nothing but the JSON asked for.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import subprocess
from pathlib import Path

LEDGER_KINDS = ("checkpoint", "checkpoint_failed", "restored", "restore_failed")
ENVIRONMENT_PREFIX = ".taste/environment/"


def _reward(result):
    value = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    return float(value) if isinstance(value, (int, float)) else None


def ledger_events(root):
    """The checkpoint and restore events of one trial's terminal ledger, in order."""
    path = root / "controller" / "terminal" / "terminal.sqlite3"
    if not path.is_file():
        return None
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        marks = ",".join("?" * len(LEDGER_KINDS))
        return [(kind, json.loads(payload)) for kind, payload in connection.execute(
            f"SELECT kind,payload FROM events WHERE kind IN ({marks}) ORDER BY seq", LEDGER_KINDS)]
    finally:
        connection.close()


def _git(repository, *args):
    result = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(repository), *args],
                            capture_output=True, text=True, timeout=60, check=False)
    return result.stdout if result.returncode == 0 else ""


def control_records(root):
    """The environment records on the trial's control branch: checkpoints and restores, by path."""
    repository = root / "agent-state" / "workspace"
    if not repository.is_dir():
        return {}
    refs = [line for line in _git(repository, "for-each-ref", "--format=%(refname)", "refs/heads/mem").split()
            if line.endswith("/central-control")]
    if len(refs) != 1:
        return {}
    paths = [line for line in _git(repository, "ls-tree", "-r", "--name-only", refs[0], ENVIRONMENT_PREFIX).split()
             if line.endswith(".json")]
    return {path: json.loads(_git(repository, "show", f"{refs[0]}:{path}")) for path in paths}


def store_bytes(root):
    store = root / "controller" / "terminal" / "checkpoints"
    return sum(item.stat().st_size for item in store.iterdir() if item.is_file()) if store.is_dir() else 0


def trial_row(result, trials_root):
    taste = ((result.get("agent_result") or {}).get("metadata") or {}).get("taste") or {}
    row = {"task": str(result.get("task_name", "")).rsplit("/", 1)[-1], "trial": result.get("trial_name"),
           "reward": _reward(result), "exception": (result.get("exception_info") or {}).get("exception_type"),
           "audit_flags": list(taste.get("audit_flags") or ()), "ledger": False}
    token = taste.get("trial")
    root = Path(trials_root) / str(token) if token else None
    events = ledger_events(root) if root else None
    if events is None:
        return row
    records = control_records(root)
    restores = sorted((item for path, item in records.items() if "/restores/" in path),
                      key=lambda item: (item["generation"], item["at"]))
    row.update(
        ledger=True, store_bytes=store_bytes(root),
        checkpoints=[{"checkpoint_id": payload["checkpoint_id"], "failed": kind == "checkpoint_failed",
                      **{name: payload.get(name) for name in ("bytes", "added", "changed", "deleted", "partial",
                                                              "error") if name in payload}}
                     for kind, payload in events if kind.startswith("checkpoint")],
        restores=[{"checkpoint_id": payload["checkpoint_id"], "failed": kind == "restore_failed",
                   **{name: payload.get(name) for name in ("exact", "seconds", "removed", "from_image",
                                                           "deleted", "mismatches", "error") if name in payload}}
                  for kind, payload in events if kind.startswith("restore")],
        rollbacks=[{"generation": item["generation"], "to": item["label"], "reason": item["reason"],
                    "result": "failed" if item["result"].get("failed") else
                    "exact" if item["result"].get("exact") else "with differences"} for item in restores],
    )
    return row


def summarize(rows):
    graded = [row for row in rows if row["ledger"]]
    restores = [item for row in graded for item in row["restores"]]
    checkpoints = [item for row in graded for item in row["checkpoints"]]
    seconds = [item["seconds"] for item in restores if not item["failed"] and item.get("seconds") is not None]
    sizes = [item["bytes"] for item in checkpoints if not item["failed"] and item.get("bytes") is not None]
    return {
        "trials": len(rows), "with_ledger": len(graded),
        "solved": sum(1 for row in rows if row["reward"] == 1.0),
        "exceptions": sum(1 for row in rows if row["exception"]),
        "checkpoints": len(checkpoints), "checkpoints_failed": sum(1 for item in checkpoints if item["failed"]),
        "checkpoints_partial": sum(1 for item in checkpoints if item.get("partial")),
        "checkpoint_bytes_max": max(sizes, default=0), "checkpoint_bytes_median": statistics.median(sizes) if sizes else 0,
        "trials_with_rollback": sum(1 for row in graded if row["restores"]),
        "restores": len(restores), "restores_exact": sum(1 for item in restores if item.get("exact")),
        "restores_failed": sum(1 for item in restores if item["failed"]),
        "restore_seconds_max": max(seconds, default=0), "restore_seconds_median": statistics.median(seconds) if seconds else 0,
        "store_bytes_total": sum(row.get("store_bytes", 0) for row in graded),
        "store_bytes_max": max((row.get("store_bytes", 0) for row in graded), default=0),
    }


def markdown(rows, summary):
    lines = ["| Task | Reward | Checkpoints (failed) | Rollbacks | Store MB | Flags |", "| --- | --- | --- | --- | --- | --- |"]
    for row in sorted(rows, key=lambda item: (item["task"], item["trial"] or "")):
        if not row["ledger"]:
            lines.append(f"| {row['task']} | {row['reward']} | no ledger | | | {', '.join(row['audit_flags'])} |")
            continue
        failed = sum(1 for item in row["checkpoints"] if item["failed"])
        rollbacks = "; ".join(f"g{item['generation']} to {item['to']} ({item['result']})" for item in row["rollbacks"])
        lines.append(f"| {row['task']} | {row['reward']} | {len(row['checkpoints'])} ({failed}) | {rollbacks or '-'} | "
                     f"{row['store_bytes'] / 1e6:.1f} | {', '.join(row['audit_flags'])} |")
    lines.append("")
    lines.append(json.dumps(summary, indent=1))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job", required=True, type=Path, help="the Harbor job directory")
    parser.add_argument("--trials", default="/var/lib/taste-trials", help="Taste's trial records")
    parser.add_argument("--json", type=Path, help="also write every row and the summary here")
    args = parser.parse_args(argv)
    rows = [trial_row(json.loads(path.read_text()), args.trials) for path in sorted(args.job.glob("*/result.json"))]
    summary = summarize(rows)
    print(markdown(rows, summary))
    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
