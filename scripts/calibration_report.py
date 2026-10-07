"""Which tasks suit the recovery study: per-task results of calibration runs.

    python scripts/calibration_report.py /root/recovery-bench/jobs/cal-pro-a [more job dirs] \\
        [--trials /var/lib/taste-trials] [--low 0.2] [--high 0.8] [--min-steps 15] \\
        [--json out.json] [--keep kept.txt]

Calibration runs one agent alone (`--ak services=none`) a few times on every
candidate task. For each task, over all its trials in the job directories
given: graded attempts and how many the verifier passed (reward 1), the
agent's steps (model calls) in each run, dollars and wall-clock per trial. A
task is kept when the share it solved is within [--low, --high] and its runs
have at least --min-steps steps (the median over its runs).

A trial the verifier never graded (Docker or Harbor failed before a reward was
written) is not an attempt: it says nothing about the task's difficulty. It is
listed with its exception, to be run again. A trial the benchmark's time limit
cut off is graded as usual and counts.

Only Harbor's result files and Taste's settled trial records are read, as in
the go/no-go report. Dollars are the record's spending over every role; a
trial without a settled record falls back to the cost Harbor's result names,
and its steps are unknown.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("go_no_go_report", Path(__file__).resolve().parent / "go_no_go_report.py")
_records = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_records)


def _seconds(span):
    """Seconds between a Harbor timing's start and finish, if both were recorded."""
    try:
        start, end = (datetime.fromisoformat(str(span[key]).replace("Z", "+00:00"))
                      for key in ("started_at", "finished_at"))
    except (KeyError, TypeError, ValueError):
        return None
    return round((end - start).total_seconds(), 3)


def agent_steps(trials_root, token):
    """Model calls of each agent run in a settled record, in the record's order.

    A step is one model reply the agent received (an agent-sourced step and its
    `llm_call_count`); coordinator and monitor calls are not the agent's. None
    when the record is missing.
    """
    path = Path(trials_root) / str(token) / "controller" / "trajectory.json" if token else None
    if path is None or not path.is_file():
        return None
    runs = []
    for sub in json.loads(path.read_text()).get("subagent_trajectories", ()):
        if sub["agent"]["name"] in _records._ROLE_OF:
            continue
        runs.append(sum(1 if step.get("llm_call_count") is None else int(step["llm_call_count"])
                        for step in sub["steps"] if step.get("source") == "agent"))
    return runs


def calibration_row(result, trials_root):
    """One trial: the go/no-go fields plus steps, dollars and wall-clock."""
    row = _records.trial_row(result, trials_root)
    taste = ((result.get("agent_result") or {}).get("metadata") or {}).get("taste") or {}
    row["benchmark"] = str(result.get("task_name", "")).rpartition("/")[0]
    row["graded"] = row["reward"] is not None
    row["steps"] = agent_steps(trials_root, taste.get("trial"))
    if row["settled"]:
        row["usd"], row["usd_source"] = round(math.fsum(row["cost_usd"].values()), 8), "record"
    else:
        cost = (result.get("agent_result") or {}).get("cost_usd")
        row["usd"], row["usd_source"] = (cost, "harbor") if isinstance(cost, (int, float)) else (None, None)
    row["trial_seconds"] = _seconds(result)
    row["agent_seconds"] = _seconds(result.get("agent_execution") or {})
    row["verifier_seconds"] = _seconds(result.get("verifier") or {})
    return row


def load_jobs(job_dirs, trials_root):
    rows = []
    for job_dir in job_dirs:
        for path in sorted(Path(job_dir).glob("*/result.json")):
            row = calibration_row(json.loads(path.read_text()), trials_root)
            row["job"] = Path(job_dir).name
            rows.append(row)
    return rows


def _median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def summarize_task(task, rows, low, high, min_steps):
    graded = [row for row in rows if row["graded"]]
    solved = sum(row["solved"] for row in graded)
    steps = [total for row in graded for total in (row["steps"] or [])]
    known_usd = [row["usd"] for row in rows if row["usd"] is not None]
    summary = {
        "task": task, "benchmark": rows[0]["benchmark"], "attempts": len(graded), "solved": solved,
        "solve_rate": solved / len(graded) if graded else None,
        "steps_per_run": steps, "median_steps": _median(steps),
        "usd_total": round(math.fsum(known_usd), 6), "usd_unknown_trials": len(rows) - len(known_usd),
        "usd_per_trial": round(math.fsum(known_usd) / len(known_usd), 6) if known_usd else None,
        # A recovery's budget is one full run of its task: the median of the
        # calibration's graded runs, in dollars and in wall-clock.
        "median_usd": _median([row["usd"] for row in graded]),
        "median_trial_seconds": _median([row["trial_seconds"] for row in graded]),
        "max_trial_seconds": max((row["trial_seconds"] for row in graded if row["trial_seconds"] is not None),
                                 default=None),
        "median_agent_seconds": _median([row["agent_seconds"] for row in graded]),
        "ungraded": [{"trial": row["trial"], "exception": row["exception"]} for row in rows if not row["graded"]],
        "timeouts": sum(row["exception"] == "AgentTimeoutError" for row in graded),
    }
    summary["kept"], summary["reason"] = keep(summary, low, high, min_steps)
    return summary


def keep(summary, low, high, min_steps):
    """Whether a task is mid-difficulty with long enough runs, and if not, why."""
    if not summary["attempts"]:
        return False, "no graded attempt"
    if not low <= summary["solve_rate"] <= high:
        return False, f"solved {summary['solved']}/{summary['attempts']}"
    if summary["median_steps"] is None:
        return False, "steps unknown"
    if summary["median_steps"] < min_steps:
        return False, f"median {summary['median_steps']:g} steps"
    return True, ""


def report(job_dirs, trials_root, low=0.2, high=0.8, min_steps=15):
    rows = load_jobs(job_dirs, trials_root)
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    tasks = [summarize_task(task, by_task[task], low, high, min_steps) for task in sorted(by_task)]
    outcomes = defaultdict(int)
    for task in tasks:
        outcomes[f"{task['solved']}/{task['attempts']}"] += 1
    known = [row["usd"] for row in rows if row["usd"] is not None]
    return {
        "rule": {"low": low, "high": high, "min_steps": min_steps},
        "jobs": [str(path) for path in job_dirs],
        "trials": len(rows), "graded": sum(row["graded"] for row in rows),
        "tasks": len(tasks), "kept": sum(task["kept"] for task in tasks),
        "outcomes": dict(sorted(outcomes.items())),
        "usd_total": round(math.fsum(known), 6), "usd_unknown_trials": len(rows) - len(known),
        "median_steps": _median([task["median_steps"] for task in tasks]),
        "median_trial_seconds": _median([row["trial_seconds"] for row in rows if row["graded"]]),
        "per_task": tasks,
        "ungraded": [{"task": row["task"], "trial": row["trial"], "job": row["job"], "exception": row["exception"]}
                     for row in rows if not row["graded"]],
    }


def _fmt(value, spec):
    return "" if value is None else format(value, spec)


def markdown(result):
    rule = result["rule"]
    lines = [f"{result['tasks']} tasks, {result['graded']} of {result['trials']} trials graded; "
             f"kept {result['kept']} (solved {rule['low']:g}-{rule['high']:g}, median steps >= {rule['min_steps']}). "
             f"Spent ${result['usd_total']:.4f}" + (f" ({result['usd_unknown_trials']} trials of unknown cost)"
                                                  if result["usd_unknown_trials"] else "") + ".",
             "", "Outcomes (solved/attempts: tasks): " + ", ".join(f"{key}: {value}"
                                                                 for key, value in result["outcomes"].items()),
             "", "| Task | Solved | Steps per run | Median $ | Median s | Kept | Why not |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for task in result["per_task"]:
        steps = ", ".join(str(value) for value in task["steps_per_run"]) or "?"
        lines.append(f"| {task['task']} | {task['solved']}/{task['attempts']} | {steps} | "
                     f"{_fmt(task['median_usd'], '.4f')} | {_fmt(task['median_trial_seconds'], '.0f')} | "
                     f"{'yes' if task['kept'] else ''} | {task['reason']} |")
    if result["ungraded"]:
        lines += ["", "Not graded, to run again:"]
        lines += [f"- {item['task']} ({item['job']}/{item['trial']}): {item['exception'] or 'no reward'}"
                  for item in result["ungraded"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("jobs", nargs="+", type=Path, help="Harbor job directories")
    parser.add_argument("--trials", default="/var/lib/taste-trials", help="Taste's trial records")
    parser.add_argument("--low", type=float, default=0.2, help="lowest share solved that is kept")
    parser.add_argument("--high", type=float, default=0.8, help="highest share solved that is kept")
    parser.add_argument("--min-steps", type=float, default=15, help="fewest median steps per run that is kept")
    parser.add_argument("--json", type=Path, help="also write the full result here")
    parser.add_argument("--keep", type=Path, help="write the kept task names here, one per line")
    arguments = parser.parse_args(argv)
    result = report(arguments.jobs, arguments.trials, arguments.low, arguments.high, arguments.min_steps)
    if arguments.json:
        arguments.json.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    if arguments.keep:
        arguments.keep.write_text("".join(task["task"] + "\n" for task in result["per_task"] if task["kept"]))
    print(markdown(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
