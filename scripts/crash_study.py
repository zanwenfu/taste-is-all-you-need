#!/usr/bin/env python3
"""Kill real runs mid-task and see whether they continue and still solve (docs/studies/system-layer.md).

Each task runs once under the supervised arm's settings, and its goal is killed at a
point drawn for that task, between a fifth and four fifths of its usual run. The three
unkilled runs of that task are the comparison.

    sudo python3 scripts/crash_study.py plan --job <job> --out plan.json
    sudo python3 scripts/crash_study.py run --plan plan.json --code <dir> --tasks <dir> --jobs <dir>
    sudo python3 scripts/crash_study.py report --plan plan.json --jobs <dir> --trials <dir> --out report.json
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
import threading
import time
from pathlib import Path

SEED = "crash-study-1"
FLOOR_SECONDS = 30.0
# Why 16 at once and a start every 10 seconds. The first study ran 24 with no such
# fault, through two Harbor processes. The first run of this one launched 24 Harbor
# processes and 69 jobs through systemd, each with its own host check, and 10 runs
# broke in its first 25 minutes on timeouts from the service manager: Docker waiting
# for a scope, ``systemctl stop``, a resumed goal's launch, which took 146 seconds.
# The cause under them was not measured. The rate of starts is what the
# two runs differed in, so it is what is limited, and the gate below watches the
# manager itself. 16 keeps most of the speed; a broken run costs more than the wait.
CONCURRENT = 16
START_SPACING_SECONDS = 10.0
SLOW_MANAGER_SECONDS = 2.0
POLL_SECONDS = 15.0
# The ends a goal reaches by itself. Any other stop says something stopped it.
GOAL_ENDS = frozenset({"complete", "wall_clock", "generation_bound", "budget_blocked", "spend_cap"})
FAILURE_FLAGS = ("goal_service_failed", "settlement_failed", "no_settled_evidence")
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


def crash_record(result, trials):
    """The kill's record: from the trial's metadata, or from its own file when that was lost."""
    taste = _taste(result)
    crash = taste.get("crash") or {}
    path = Path(trials) / str(taste.get("trial")) / "controller" / "crash.json"
    if taste.get("trial") and not crash.get("landed") and path.exists():
        crash = {**crash, **json.loads(path.read_text())}
    return crash


def _ended_cleanly(row):
    failed = any(flag.startswith(FAILURE_FLAGS) for flag in row.get("flags") or ())
    return row["stop"] in GOAL_ENDS and row["reward"] is not None and not row["exception"] and not failed


def classify(row):
    """Exactly one of four. Whatever is not provably fine counts against the system or the run."""
    if row["crash"].get("landed"):
        return "recovered" if row["crash"].get("resumed") and _ended_cleanly(row) else "failed_recovery"
    return "ended_first" if _ended_cleanly(row) else "broken"


class LaunchGate:
    """Holds each start until its turn, and while the service manager is slow to answer."""

    def __init__(self, spacing, quick, clock=time.monotonic, sleep=time.sleep):
        self.spacing, self.quick, self.clock, self.sleep = spacing, quick, clock, sleep
        self.lock = threading.Lock()
        self.turn = None

    def wait(self):
        with self.lock:
            while self.turn is not None and self.clock() < self.turn:
                self.sleep(self.turn - self.clock())
            while not self.quick():
                self.sleep(self.spacing)
            self.turn = self.clock() + self.spacing


def manager_is_quick():
    started = time.monotonic()
    try:
        subprocess.run(["systemctl", "show", "--property=Version"], capture_output=True, timeout=30, check=False)
    except subprocess.TimeoutExpired:
        return False
    return time.monotonic() - started < SLOW_MANAGER_SECONDS


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
    gate = LaunchGate(arguments.spacing, manager_is_quick)
    checked = threading.Event()

    def one(row):
        job = job_name(row)
        if (Path(arguments.jobs) / job).exists():
            return job, "exists"
        options = [item for key, value in {**SETTINGS, "crash_after_seconds": row["crash_after_seconds"]}.items()
                   for item in ("--ak", f"{key}={value}")]
        gate.wait()
        # One host check is enough: the host does not change between jobs.
        skip = {"SKIP_HOST_CHECK": "1"} if checked.is_set() else {}
        # No retries: a run that a kill left broken must be counted, not run again.
        started = subprocess.run([f"{arguments.code}/infra/azure/run-harbor.sh", job, arguments.tasks, "-i", row["task"],
                                  "-k", "1", "-n", "1", "--max-retries", "0", *options],
                                 env={**environment, **skip}, capture_output=True, text=True, check=False)
        if started.returncode != 0:
            return job, started.stderr[-300:]
        checked.set()
        while subprocess.run(["systemctl", "is-active", "--quiet", f"taste-harbor-{job}"], check=False).returncode == 0:
            time.sleep(POLL_SECONDS)
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


def summarize(rows, resamples=10_000):
    landed = [row for row in rows if row["crash"].get("landed")]
    resumed = [row for row in landed if row["crash"].get("resumed")]
    ungraded = [row for row in landed if row["reward"] is None]
    classes = {name: [row for row in rows if classify(row) == name]
               for name in ("recovered", "failed_recovery", "ended_first", "broken")}
    failed = classes["failed_recovery"]
    with_evidence = [row for row in landed if row["crash"].get("paid_calls_repeated") is not None]
    seconds = [row["crash"]["recovery_seconds"] for row in resumed if row["crash"].get("recovery_seconds") is not None]
    return {"runs": len(rows), "kills_landed": len(landed), "recovered": len(classes["recovered"]),
            "ended_before_the_kill": len(classes["ended_first"]), "broken": len(classes["broken"]),
            "broken_tasks": {row["task"]: row["exception"] or row["stop"] or str(row.get("flags"))
                             for row in classes["broken"]},
            "resumed": len(resumed), "not_resumed": len(landed) - len(resumed), "ungraded": len(ungraded),
            "failed_recoveries": len(failed), "failed_tasks": [row["task"] for row in failed],
            "passed_without_recovering": sum(row["reward"] == 1.0 for row in failed),
            "solved_after_a_kill": sum(row["reward"] == 1.0 for row in classes["recovered"]),
            "evidence_missing": len(landed) - len(with_evidence),
            "commands_repeated": sum(row["crash"].get("commands_repeated") or 0 for row in with_evidence),
            "paid_calls_repeated": sum(row["crash"].get("paid_calls_repeated") or 0 for row in with_evidence),
            "paid_calls_cut_off": sum(row["crash"].get("paid_calls_cut_off") or 0 for row in with_evidence),
            "median_recovery_seconds": statistics.median(seconds) if seconds else None,
            "landed_tasks": _difference(landed, resamples),
            "recovered_tasks": _difference(classes["recovered"], resamples),
            # A broken run says nothing about recovery, so it is reported above and compared with nothing.
            "all_tasks": _difference([row for row in rows if classify(row) != "broken"], resamples)}


def report(arguments):
    rows = []
    for row in json.loads(Path(arguments.plan).read_text()):
        results = list(_results(Path(arguments.jobs) / job_name(row)))
        result = results[-1] if results else {}
        rows.append({"task": row["task"], "before": row["before"], "reward": _reward(result),
                     "exception": (result.get("exception_info") or {}).get("exception_type"),
                     "cost": (result.get("agent_result") or {}).get("cost_usd"),
                     "stop": _taste(result).get("stop_reason"), "flags": _taste(result).get("audit_flags"),
                     "crash": crash_record(result, arguments.trials)})
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
    start.add_argument("-n", type=int, default=CONCURRENT)
    start.add_argument("--spacing", type=float, default=START_SPACING_SECONDS)
    show = commands.add_parser("report")
    for name in ("plan", "jobs", "out"):
        show.add_argument("--" + name, required=True)
    show.add_argument("--trials", default="/data/taste-trials-crash")
    arguments = parser.parse_args(argv)
    {"plan": plan, "run": run, "report": report}[arguments.command](arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
