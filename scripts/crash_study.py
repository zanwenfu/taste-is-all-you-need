#!/usr/bin/env python3
"""Kill real runs mid-task and see whether they continue and still solve (docs/studies/system-layer.md).

Each task runs once under the supervised arm's settings, and its goal is killed at a
point drawn for that task, between a fifth and four fifths of its usual run. The three
unkilled runs of that task are the comparison.

    sudo python3 scripts/crash_study.py plan --job <job> --out plan.json
    sudo python3 scripts/crash_study.py run --plan plan.json --code <dir> --tasks <dir> --jobs <dir> -n 24
    sudo python3 scripts/crash_study.py report --plan plan.json --jobs <dir> --out report.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

SEED = "crash-study-1"
FLOOR_SECONDS = 30.0
SETTINGS = {"agent": "mini-swe-agent", "spend_cap_usd": 2, "worker_spend_cap_usd": 2, "monitor_spend_cap_usd": 1,
            "worker_max_calls": 1000, "monitor_max_calls": 400, "max_commands": 2000, "rollback": "on",
            "keep_checkpoints": "on", "final_checkpoint": "sl1", "checkpoint_mib": 4096}


def kill_second(task, usual_seconds):
    """A point inside the task's usual run, repeatable, and late enough for a run to have started."""
    fraction = random.Random(f"{SEED}:{task}").uniform(0.2, 0.8)
    return max(FLOOR_SECONDS, round(usual_seconds * fraction, 1))


def _results(job):
    for path in Path(job).glob("*/result.json"):
        result = json.loads(path.read_text())
        if "trial_name" in result:
            yield result


def _reward(result):
    return ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")


def _taste(result):
    return ((result.get("agent_result") or {}).get("metadata") or {}).get("taste") or {}


def plan(arguments):
    tasks = {}
    for result in _results(arguments.job):
        name = result["task_name"].rsplit("/", 1)[-1]
        tasks.setdefault(name, []).append((_taste(result).get("seconds") or 0, _reward(result) == 1.0))
    rows = [{"task": name, "crash_after_seconds": kill_second(name, statistics.median(s for s, _ in runs)),
             "before": sum(solved for _, solved in runs) / len(runs)} for name, runs in sorted(tasks.items())]
    Path(arguments.out).write_text(json.dumps(rows, indent=0))
    print(json.dumps({"tasks": len(rows), "median_kill_seconds": statistics.median(
        row["crash_after_seconds"] for row in rows)}))


def job_name(row):
    return "crash-" + row["task"]


def run(arguments):
    rows = json.loads(Path(arguments.plan).read_text())
    environment = {**os.environ, "MODEL": "azure/gpt-6-luna", "JOBS": str(arguments.jobs),
                   "SECRETS": "/root/taste-secrets/bedrock.env", "TASTE_TRIALS_ROOT": arguments.trials,
                   "TASTE_CHECKPOINTS_ROOT": "/data/taste-checkpoints"}

    def one(row):
        job = job_name(row)
        if (Path(arguments.jobs) / job).exists():
            return job, "exists"
        options = [item for key, value in {**SETTINGS, "crash_after_seconds": row["crash_after_seconds"]}.items()
                   for item in ("--ak", f"{key}={value}")]
        # No retries: a run that a kill left broken must be counted, not run again.
        started = subprocess.run([f"{arguments.code}/infra/azure/run-harbor.sh", job, arguments.tasks, "-i", row["task"],
                                  "-k", "1", "-n", "1", "--max-retries", "0", *options],
                                 env=environment, capture_output=True, text=True, check=False)
        if started.returncode != 0:
            return job, started.stderr[-300:]
        while subprocess.run(["systemctl", "is-active", "--quiet", f"taste-harbor-{job}"], check=False).returncode == 0:
            time.sleep(5)
        return job, "ok"

    with concurrent.futures.ThreadPoolExecutor(arguments.n) as pool:
        for job, status in pool.map(one, rows):
            print(job, status, flush=True)


def _difference(rows, resamples):
    differences = [(row["reward"] == 1.0) - row["before"] for row in rows]
    if not differences:
        return {"tasks": 0}
    draw = random.Random(0)
    means = sorted(sum(draw.choices(differences, k=len(differences))) / len(differences) for _ in range(resamples))
    return {"tasks": len(differences), "mean_difference": sum(differences) / len(differences),
            "interval_95": [means[int(resamples * 0.025)], means[int(resamples * 0.975) - 1]],
            "better": sum(value > 0 for value in differences), "worse": sum(value < 0 for value in differences),
            "same": sum(value == 0 for value in differences)}


def recovered(row):
    """A new process started, the goal reached an end of its own, and the task was graded."""
    return bool(row["crash"].get("resumed")) and row["stop"] is not None and row["reward"] is not None


def summarize(rows, resamples=10_000):
    landed = [row for row in rows if row["crash"].get("landed")]
    resumed = [row for row in landed if row["crash"].get("resumed")]
    ungraded = [row for row in landed if row["reward"] is None]
    failed = [row for row in landed if not recovered(row)]
    with_evidence = [row for row in landed if row["crash"].get("paid_calls_repeated") is not None]
    seconds = [row["crash"]["recovery_seconds"] for row in resumed if row["crash"].get("recovery_seconds") is not None]
    return {"runs": len(rows), "kills_landed": len(landed), "ended_before_the_kill": len(rows) - len(landed),
            "resumed": len(resumed), "not_resumed": len(landed) - len(resumed), "ungraded": len(ungraded),
            "failed_recoveries": len(failed),
            "passed_without_recovering": sum(row["reward"] == 1.0 for row in failed),
            "solved_after_a_kill": sum(row["reward"] == 1.0 and recovered(row) for row in landed),
            "evidence_missing": len(landed) - len(with_evidence),
            "commands_repeated": sum(row["crash"].get("commands_repeated") or 0 for row in with_evidence),
            "paid_calls_repeated": sum(row["crash"].get("paid_calls_repeated") or 0 for row in with_evidence),
            "paid_calls_cut_off": sum(row["crash"].get("paid_calls_cut_off") or 0 for row in with_evidence),
            "median_recovery_seconds": statistics.median(seconds) if seconds else None,
            "landed_tasks": _difference(landed, resamples), "all_tasks": _difference(rows, resamples)}


def report(arguments):
    rows = []
    for row in json.loads(Path(arguments.plan).read_text()):
        results = list(_results(Path(arguments.jobs) / job_name(row)))
        result = results[-1] if results else {}
        rows.append({"task": row["task"], "before": row["before"], "reward": _reward(result),
                     "exception": (result.get("exception_info") or {}).get("exception_type"),
                     "cost": (result.get("agent_result") or {}).get("cost_usd"),
                     "stop": _taste(result).get("stop_reason"), "flags": _taste(result).get("audit_flags"),
                     "crash": _taste(result).get("crash") or {}})
    summary = summarize(rows)
    summary["model_usd"] = sum(row["cost"] or 0 for row in rows)
    Path(arguments.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=0))
    print(json.dumps(summary, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("plan")
    make.add_argument("--job", required=True)
    make.add_argument("--out", required=True)
    start = commands.add_parser("run")
    for name in ("plan", "code", "tasks", "jobs"):
        start.add_argument("--" + name, required=True)
    start.add_argument("--trials", default="/data/taste-trials-crash")
    start.add_argument("-n", type=int, default=24)
    show = commands.add_parser("report")
    for name in ("plan", "jobs", "out"):
        show.add_argument("--" + name, required=True)
    arguments = parser.parse_args(argv)
    {"plan": plan, "run": run, "report": report}[arguments.command](arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
