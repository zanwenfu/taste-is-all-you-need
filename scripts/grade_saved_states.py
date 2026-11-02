#!/usr/bin/env python3
"""Run a task's own verifier on every state a supervised trial saved.

    sudo python3 scripts/grade_saved_states.py plan --job <job> --trials <root> --out plan.jsonl
    sudo python3 scripts/grade_saved_states.py run --plan plan.jsonl --tasks <selected> --out <jobs> -n 16
    sudo python3 scripts/grade_saved_states.py report --plan plan.jsonl --out <jobs> [--json]

``plan`` lists each saved state of each trial, in the order taken, from the
trial's terminal ledger: its checkpoints, and its final save when it made one.
A failed or partial save cannot be graded and is counted, not listed as a job.
``run`` starts one Harbor job per state with taste.benchmarks.saved_state_agent,
which restores the state in a fresh container and ends, so that the verifier
grades exactly those files. No model is called. A job that exists is left as it
is, so a stopped run picks up where it was.
``report`` answers two questions. Control: does the verifier give a run's final
save the reward the run itself got? If not, grading saved states is unsound for
that task. Result: of the runs the verifier failed at their end, how many had
an earlier saved state that passed?
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollback_report import _reward, ledger_events

AGENT = "taste.benchmarks.saved_state_agent:SavedStateAgent"


def _results(job):
    for path in sorted(Path(job).glob("*/result.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


def plan_rows(job, trials_root):
    """(one row per gradable state, in order; what could not be graded, by reason)."""
    rows, skipped = [], Counter()
    for result in _results(job):
        taste = ((result.get("agent_result") or {}).get("metadata") or {}).get("taste") or {}
        token = taste.get("trial")
        common = {"job": Path(job).name, "trial": result.get("trial_name"),
                  "task": str(result.get("task_name", "")).rsplit("/", 1)[-1], "reward": _reward(result)}
        root = Path(trials_root) / str(token) if token else None
        events = ledger_events(root) if root else None
        if events is None:
            skipped["no_ledger"] += 1
            continue
        store = root / "controller" / "terminal" / "checkpoints"
        order = 0
        for kind, payload in events:
            if kind == "checkpoint_failed":
                skipped["failed"] += 1
            elif kind == "checkpoint":
                if payload.get("partial"):
                    skipped["partial"] += 1
                    continue
                state = store / f"{payload['checkpoint_id']}.json"
                if not state.is_file() or not (store / f"{payload['tar_sha256']}.tar").is_file():
                    skipped["tar_missing"] += 1
                    continue
                order += 1
                rows.append({**common, "order": order, "state": str(state), "kind": "checkpoint",
                             "checkpoint_id": payload["checkpoint_id"], "bytes": payload.get("bytes")})
        final = (taste.get("final_checkpoint") or {})
        if final.get("saved") and not final.get("partial"):
            rows.append({**common, "order": order + 1, "state": str(Path(final["directory"]) / "manifest.json"),
                         "kind": "final", "checkpoint_id": "final", "bytes": final.get("bytes")})
        elif final:
            skipped["final_unusable"] += 1
    return rows, skipped


def job_name(row):
    trial = re.sub(r"[^A-Za-z0-9_-]", "-", str(row["trial"]))
    return f"state-{trial}-{row['order']:02d}-{row['kind']}"


def grade_one(row, tasks, out, harbor):
    name = job_name(row)
    if (Path(out) / name).exists():
        return name, "exists"
    result = subprocess.run(
        [harbor, "run", "-p", str(Path(tasks) / row["task"]), "-a", AGENT, "--ak", f"saved_state={row['state']}",
         "-o", str(out), "--job-name", name, "-n", "1", "-q"],
        capture_output=True, text=True, check=False)
    return name, "ok" if result.returncode == 0 else result.stderr[-400:]


def graded(rows, out):
    """Each row with the verifier's reward on its state; None when it could not be graded."""
    table = []
    for row in rows:
        results = list(_results(Path(out) / job_name(row)))
        result = results[0] if len(results) == 1 else {}
        restored = ((result.get("agent_result") or {}).get("metadata") or {}).get("saved_state") or {}
        table.append({**row, "state_reward": _reward(result) if restored.get("exact") else None})
    return table


def summarize(table):
    """The control over final saves, then the result over runs the verifier failed at their end."""
    finals = [row for row in table if row["kind"] == "final" and row["state_reward"] is not None]
    agree = [row for row in finals if row["state_reward"] == row["reward"]]
    by_run = {}
    for row in table:
        by_run.setdefault((row["job"], row["trial"]), []).append(row)
    failed = [states for states in by_run.values() if states[0]["reward"] != 1.0]
    held = [states for states in failed
            if any(row["kind"] == "checkpoint" and row["state_reward"] == 1.0 for row in states)]
    return {"states": len(table), "states_ungraded": sum(1 for row in table if row["state_reward"] is None),
            "runs": len(by_run), "runs_failed_at_end": len(failed), "failed_runs_with_an_earlier_passing_state": len(held),
            "control_final_saves": len(finals), "control_agree": len(agree),
            "control_disagree": sorted(f"{row['task']}/{row['trial']}" for row in finals if row not in agree),
            "held_by_task": dict(Counter(states[0]["task"] for states in held))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--job", required=True, action="append")
    plan.add_argument("--trials", required=True)
    plan.add_argument("--out", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--tasks", required=True)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("-n", type=int, default=16)
    run.add_argument("--harbor", default="harbor")
    report = commands.add_parser("report")
    report.add_argument("--plan", required=True, type=Path)
    report.add_argument("--out", required=True, type=Path)
    report.add_argument("--json", type=Path)
    arguments = parser.parse_args(argv)

    if arguments.command == "plan":
        rows, skipped = [], Counter()
        for job in arguments.job:
            found, left = plan_rows(job, arguments.trials)
            rows += found
            skipped += left
        arguments.out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        print(json.dumps({"states": len(rows), "skipped": dict(skipped)}))
        return 0
    rows = [json.loads(line) for line in arguments.plan.read_text(encoding="utf-8").splitlines() if line]
    if arguments.command == "run":
        with concurrent.futures.ThreadPoolExecutor(arguments.n) as pool:
            jobs = [pool.submit(grade_one, row, arguments.tasks, arguments.out, arguments.harbor) for row in rows]
            for done in concurrent.futures.as_completed(jobs):
                name, status = done.result()
                print(name, status, flush=True)
        return 0
    summary = summarize(graded(rows, arguments.out))
    print(json.dumps(summary, indent=1))
    if arguments.json:
        arguments.json.write_text(json.dumps(graded(rows, arguments.out), indent=0), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
