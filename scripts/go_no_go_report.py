"""Compare two arms of Harbor trials of one agent: alone, and under Taste.

    python scripts/go_no_go_report.py --arm alone=/root/tb/jobs/gng-alone \\
        --arm taste=/root/tb/jobs/gng-taste [--trials /var/lib/taste-trials] [--json out.json]

For each trial: the task's verifier reward (1 is solved), Taste's settled
record of what it spent by role (coordinator, agent, monitor), how each agent
run ended, and the audit flags. Per arm: tasks solved over task-runs, dollars
per trial and per solved task by role, agent runs stopped by their monitor,
and submissions the certifier refused in trials the verifier passed. Between
the two arms, paired by task: tasks where each arm solved more of its runs,
and an exact two-sided sign test over those tasks.

Only Harbor's result files and Taste's settled trial records are read. A trial
without a settled record is counted, its verifier reward kept, and its costs
reported as unknown.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROLES = ("coordinator", "agent", "monitor")
_ROLE_OF = {"taste-coordinator": "coordinator", "taste-monitor": "monitor"}


def _reward(result):
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    value = rewards.get("reward")
    return float(value) if isinstance(value, (int, float)) else None


def trial_row(result, trials_root):
    """One Harbor trial as the report needs it."""
    taste = ((result.get("agent_result") or {}).get("metadata") or {}).get("taste") or {}
    row = {"task": str(result.get("task_name", "")).rsplit("/", 1)[-1], "trial": result.get("trial_name"),
           "reward": _reward(result), "solved": _reward(result) == 1.0,
           "exception": (result.get("exception_info") or {}).get("exception_type"),
           "audit_flags": list(taste.get("audit_flags") or ()), "stop_reason": taste.get("stop_reason"),
           "services": (taste.get("configuration") or {}).get("services", "all"),
           "cost_usd": {role: None for role in ROLES}, "agent_runs": [], "settled": False}
    token = taste.get("trial")
    path = Path(trials_root) / str(token) / "controller" / "trajectory.json" if token else None
    if path is None or not path.is_file():
        return row
    nested = json.loads(path.read_text())
    costs = defaultdict(float)
    for sub in nested.get("subagent_trajectories", ()):
        role = _ROLE_OF.get(sub["agent"]["name"], "agent")
        costs[role] += math.fsum(step["metrics"]["cost_usd"] for step in sub["steps"]
                                 if (step.get("metrics") or {}).get("cost_usd") is not None)
        if role == "agent":
            extra = sub.get("extra", {})
            row["agent_runs"].append({"run_id": extra.get("run_id"), "phase": extra.get("phase"),
                                      "exit": (extra.get("exit") or {}).get("exit_status"),
                                      "stopped_by": (extra.get("exit") or {}).get("stopped_by") or ""})
    row["cost_usd"] = {role: round(costs[role], 8) for role in ROLES}
    row["settled"] = True
    return row


def load_arm(job_dir, trials_root):
    rows = []
    for path in sorted(Path(job_dir).glob("*/result.json")):
        rows.append(trial_row(json.loads(path.read_text()), trials_root))
    return rows


def sign_test(wins, losses):
    """Exact two-sided sign test over tasks where the arms differ."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarize(rows):
    settled = [row for row in rows if row["settled"]]
    solved = sum(row["solved"] for row in rows)
    spend = {role: math.fsum(row["cost_usd"][role] for row in settled) for role in ROLES}
    total = math.fsum(spend.values())
    runs = [run for row in rows for run in row["agent_runs"]]
    return {
        "trials": len(rows), "solved": solved, "settled": len(settled),
        "solve_rate": solved / len(rows) if rows else None,
        "spend_usd": {**{role: round(value, 6) for role, value in spend.items()}, "total": round(total, 6)},
        "usd_per_trial": round(total / len(settled), 6) if settled else None,
        "usd_per_solved_task": round(total / solved, 6) if solved else None,
        "agent_runs": len(runs),
        # How the agent's runs ended, and what stopped those Taste stopped
        # (its monitor, a spending cap, a request over 1 MiB, the deadline).
        "exits": dict(sorted(Counter(run["exit"] or "none" for run in runs).items())),
        "stops": dict(sorted(Counter(run["stopped_by"] for run in runs if run["stopped_by"]).items())),
        "stopped_by_monitor": sum(run["stopped_by"].startswith("monitor_") for run in runs),
        # Supervised trials only (alone, nothing certifies and nothing is
        # delivered): a submission the certifier refused although the task's
        # own verifier passed the trial. An upper bound on wrong refusals: a
        # later run may have changed the container after the refusal.
        "refused_submissions_in_solved": sum(
            run["exit"] == "Submitted" and run["phase"] != "delivered"
            for row in rows if row["solved"] and row["services"] != "none" for run in row["agent_runs"]),
        # Taste closed the goal as complete and the task's verifier failed it:
        # a completion claimed wrongly, the opposite error to a wrong refusal.
        "claimed_complete": sum(row["stop_reason"] == "complete" for row in rows),
        "claimed_complete_unsolved": sum(row["stop_reason"] == "complete" and not row["solved"]
                                         for row in rows),
        "audit_flagged": sum(bool(row["audit_flags"]) for row in rows),
        "exceptions": sum(bool(row["exception"]) for row in rows),
    }


def compare(base_rows, other_rows):
    """Per task, which arm solved more of its runs; sign test over tasks that differ."""
    def rates(rows):
        by_task = defaultdict(list)
        for row in rows:
            by_task[row["task"]].append(row["solved"])
        return {task: sum(values) / len(values) for task, values in by_task.items()}

    base, other = rates(base_rows), rates(other_rows)
    tasks = sorted(set(base) & set(other))
    wins = sum(other[task] > base[task] for task in tasks)
    losses = sum(other[task] < base[task] for task in tasks)
    return {"tasks": len(tasks), "other_better": wins, "base_better": losses,
            "same": len(tasks) - wins - losses, "sign_test_p": round(sign_test(wins, losses), 6),
            "per_task": {task: {"base": base[task], "other": other[task]} for task in tasks}}


def report(arms, trials_root):
    loaded = {name: load_arm(path, trials_root) for name, path in arms}
    names = [name for name, _ in arms]
    result = {"arms": {name: summarize(rows) for name, rows in loaded.items()},
              "trials": {name: rows for name, rows in loaded.items()}}
    if len(names) == 2:
        result["paired"] = {"base": names[0], "other": names[1], **compare(loaded[names[0]], loaded[names[1]])}
    return result


def markdown(result):
    lines = ["| Arm | Solved | Trials | $/trial | $/solved | Coordinator $ | Agent $ | Monitor $ |"
             " Monitor stops | Refused, verifier passed | Flagged |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, arm in result["arms"].items():
        spend = arm["spend_usd"]
        lines.append(f"| {name} | {arm['solved']} | {arm['trials']} | {arm['usd_per_trial']} | "
                     f"{arm['usd_per_solved_task']} | {spend['coordinator']} | {spend['agent']} | "
                     f"{spend['monitor']} | {arm['stopped_by_monitor']} | {arm['refused_submissions_in_solved']} | "
                     f"{arm['audit_flagged']} |")
    if "paired" in result:
        paired = result["paired"]
        lines += ["", f"Paired by task ({paired['tasks']} tasks): {paired['other']} better on "
                      f"{paired['other_better']}, {paired['base']} better on {paired['base_better']}, "
                      f"same on {paired['same']}; exact sign test p = {paired['sign_test_p']}."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", action="append", required=True, help="name=harbor job directory")
    parser.add_argument("--trials", default="/var/lib/taste-trials", help="Taste's trial records")
    parser.add_argument("--json", type=Path, help="also write the full result here")
    arguments = parser.parse_args(argv)
    arms = [tuple(item.split("=", 1)) for item in arguments.arm]
    result = report(arms, arguments.trials)
    if arguments.json:
        arguments.json.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(markdown(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
