"""What a finished trial says, from Harbor's result and Taste's settled record.

Harbor writes ``<job>/<trial>/result.json``: the verifier's reward, the
agent's settings (``config.agent.kwargs``), the task's path, timestamps and any
exception; and the trial's flat record, ``<job>/<trial>/agent/trajectory.json``,
whose last reply that is not a sidechain is what the trial handed back (for a
checker trial, the checker's submission). Taste writes the trial's settled
record under its token, ``<trials root>/<token>/controller/trajectory.json``:
every model call with its cost and tokens, every command the agent ran with its
output and exit code, and how the agent's run ended. Rewards are read from
Harbor; cost, tokens and steps from Taste's record, and Harbor's own figures
only for a trial without one.

Time has one clock, Harbor's: a trial's seconds are its agent's execution
(``agent_execution``, start to finish), the same span the calibration report's
``median_agent_seconds`` measures, so a budget and what is charged against it
agree. A branch is charged its seconds less the time branching spent bringing
its prefix back (``prefix_seconds``); both are kept.

A run's steps are numbered as the study numbers them, by the trajectory reader
(``taste.agents.trajectory_reader.steps_from_trajectory``): one per model reply
the agent received; branching numbers its steps the same way.

Branching (work package A) adds to the trial's metadata
(``agent_result.metadata.taste``) what this study reads about a branch; its
keys are the constants below, the only place these names appear.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from taste.agents.checker import VERDICTS, parse_submission
from taste.agents.trajectory_reader import final_message, steps_from_trajectory
from taste.recovery_study.commands import diff_paths, gaming_flags, written_paths

# A: {"faithful": bool, "unfaithful_at": step, "checkpoint": the id it saved,
# "changed_paths": the paths its checkpoint's manifest lists,
# "prefix_seconds": the time spent bringing the prefix back}.
BRANCH = "branch"
# An audit flag, starting with this, that also marks a branch unfaithful.
UNFAITHFUL_FLAG = "branch_unfaithful"
FLAT_RECORD = Path("agent") / "trajectory.json"
TASTE_ROLES = frozenset({"taste-coordinator", "taste-monitor"})
SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def number(raw):
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw) if math.isfinite(raw) else None


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def reward(result):
    return number(((result.get("verifier_result") or {}).get("rewards") or {}).get("reward"))


def _moment(text):
    return datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)


def agent_seconds(result):
    """The agent's execution time by Harbor's clock, the study's one clock; None if it never ran."""
    execution = result.get("agent_execution") or {}
    try:
        return round((_moment(execution["finished_at"]) - _moment(execution["started_at"])).total_seconds(), 3)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def charged(seconds, prefix):
    """What a trial's time counts against a budget: its seconds less its prefix's, never below 0."""
    return None if seconds is None else round(max(0.0, seconds - (prefix or 0.0)), 3)


def settled(token, trials_root):
    """A trial's settled record, or None."""
    if not token or not trials_root:
        return None
    return read_json(Path(trials_root) / str(token) / "controller" / "trajectory.json")


def agent_trajectory(nested):
    """The agent's run in a settled record: the last run of a worker that is not Taste's own role."""
    runs = [sub for sub in (nested or {}).get("subagent_trajectories") or ()
            if (sub.get("agent") or {}).get("name") not in TASTE_ROLES]
    return runs[-1] if runs else None


def agent_view(nested):
    """What the agent was given and what it handed back: task, steps, final message, submission, exit."""
    run = agent_trajectory(nested)
    if run is None:
        return None
    task = next((str(step.get("message") or "") for step in run.get("steps") or () if step.get("source") == "user"), "")
    taken = steps_from_trajectory(run)
    ending = (run.get("extra") or {}).get("exit") or {}
    submission = ""
    if taken and ending.get("exit_status") == "Submitted":
        lines = taken[-1].output.lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == SENTINEL:
            submission = "".join(lines[1:])
    return {"task": task, "steps": taken, "final_message": final_message(run), "submission": submission,
            "exit_status": ending.get("exit_status"), "stopped_by": ending.get("stopped_by") or ""}


def spend(nested, agent_result):
    """(dollars, prompt tokens, completion tokens): the settled record's, else Harbor's."""
    if nested is None:
        return (number(agent_result.get("cost_usd")), agent_result.get("n_input_tokens"),
                agent_result.get("n_output_tokens"))
    final = nested.get("final_metrics") or {}
    metrics = [step["metrics"] for sub in nested.get("subagent_trajectories") or ()
               for step in sub.get("steps") or () if isinstance(step.get("metrics"), dict)]
    known = number((final.get("extra") or {}).get("known_cost_usd"))
    cost = known if known is not None else math.fsum(number(m.get("cost_usd")) or 0.0 for m in metrics)
    totals = []
    for name in ("prompt_tokens", "completion_tokens"):
        total = number(final.get("total_" + name))
        totals.append(int(total) if total is not None else sum(int(number(m.get(name)) or 0) for m in metrics))
    return round(cost, 8), totals[0], totals[1]


def handed_back(trial_dir):
    """The trial's last reply that is not a sidechain, from Harbor's flat record; None without one."""
    flat = read_json(Path(trial_dir) / FLAT_RECORD)
    replies = [step for step in (flat or {}).get("steps") or ()
               if step.get("source") == "agent" and (step.get("extra") or {}).get("is_sidechain") is False]
    return str(replies[-1].get("message") or "") if replies else None


def checker_submission(trial_dir):
    """A checker trial's submission (``taste.agents.checker.parse_submission``), or None."""
    text = handed_back(trial_dir)
    if not text:
        return None
    try:
        return parse_submission(text)
    except ValueError:
        return None


def finished(result):
    return bool(result.get("finished_at")) or result.get("verifier_result") is not None \
        or result.get("exception_info") is not None


def trial(trial_dir, trials_root):
    """One finished trial, summarised for the drivers' state (no long texts but a checker's verdict)."""
    trial_dir = Path(trial_dir)
    result = read_json(trial_dir / "result.json") or {}
    agent = result.get("agent_result") or {}
    taste = (agent.get("metadata") or {}).get("taste") or {}
    config = result.get("config") or {}
    task_path = (config.get("task") or {}).get("path")
    branch = taste.get(BRANCH) if isinstance(taste.get(BRANCH), dict) else {}
    audit = [str(flag) for flag in taste.get("audit_flags") or ()]
    faithful = branch.get("faithful") if isinstance(branch.get("faithful"), bool) else None
    if any(flag.startswith(UNFAITHFUL_FLAG) for flag in audit):
        faithful = False
    nested = settled(taste.get("trial"), trials_root)
    view = agent_view(nested) or {}
    cost, prompt, completion = spend(nested, agent)
    seconds, prefix = agent_seconds(result), number(branch.get("prefix_seconds"))
    graded = reward(result)
    changed = branch.get("changed_paths")
    return {
        "trial": result.get("trial_name") or trial_dir.name,
        "task": Path(task_path).name if task_path else str(result.get("task_name", "")).rsplit("/", 1)[-1],
        "tasks_dir": str(Path(task_path).parent) if task_path else None,
        "reward": graded, "solved": None if graded is None else graded == 1.0,
        "exception": (result.get("exception_info") or {}).get("exception_type"),
        "token": taste.get("trial"), "settled": nested is not None,
        "model": (config.get("agent") or {}).get("model_name"),
        "settings": dict((config.get("agent") or {}).get("kwargs") or {}),
        "cost_usd": cost, "prompt_tokens": prompt, "completion_tokens": completion,
        # Raw: the agent's execution by Harbor's clock; charged: without the prefix's time.
        "seconds": seconds, "prefix_seconds": prefix, "charged_seconds": charged(seconds, prefix),
        "faithful": faithful, "unfaithful_at": branch.get("unfaithful_at"), "checkpoint": branch.get("checkpoint"),
        "changed_paths": [str(path) for path in changed] if isinstance(changed, list) else None,
        "exit_status": view.get("exit_status"), "steps": len(view["steps"]) if view else None,
        "checker": checker_submission(trial_dir),
        "written_paths": sorted({path for step in view.get("steps", ()) for path in written_paths(step.command)}),
        "submission_paths": diff_paths(view.get("submission", "")),
        "skip_marker": "skip_marker" in gaming_flags((), view.get("submission", "")),
        "audit_flags": audit,
    }


def usable(summary):
    """A graded trial whose prefix, if it had one, was faithful: a sample of the outcome."""
    return summary["reward"] is not None and summary["faithful"] is not False


def verdict(summary):
    """A checker trial's submission, when it holds a verdict on a faithful copy of the final state.

    The Harbor reward of a checker trial grades the copy and is not used.
    """
    found = summary["checker"]
    if found and found.get("verdict") in VERDICTS and summary["faithful"] is not False:
        return found
    return None


def job_trials(job_dir, trials_root):
    """The finished trials of one Harbor job, in name order."""
    found = []
    for path in sorted(Path(job_dir).glob("*/result.json")):
        result = read_json(path)
        if isinstance(result, dict) and finished(result):
            found.append(trial(path.parent, trials_root))
    return found


def job_finished(job_dir):
    """Harbor has closed the job (its own result.json names when it finished)."""
    result = read_json(Path(job_dir) / "result.json")
    return bool(isinstance(result, dict) and result.get("finished_at"))


def fresh_runs(job_dirs, trials_root):
    """Graded runs from scratch by task: calibration and base runs, which give v(0) and budgets."""
    by_task = defaultdict(list)
    for job_dir in job_dirs:
        for summary in job_trials(job_dir, trials_root):
            if summary["reward"] is not None:
                by_task[summary["task"]].append(summary)
    return dict(by_task)


def budget(runs):
    """One base run's dollars and agent seconds for a task: the medians over its runs from scratch."""
    dollars = [run["cost_usd"] for run in runs if run["cost_usd"] is not None]
    seconds = [run["seconds"] for run in runs if run["seconds"] is not None]
    if not dollars or not seconds:
        return None
    return {"usd": round(statistics.median(dollars), 6), "seconds": round(statistics.median(seconds), 1),
            "runs": len(runs)}


def calibration_budgets(path):
    """Per task, from ``scripts/calibration_report.py --json``: kept, and one run's median dollars and time.

    The time is the agent's execution by Harbor's clock (``median_agent_seconds``),
    the clock every trial is charged by; a task without it has no time here.
    """
    report = read_json(path)
    if not isinstance(report, dict):
        raise ValueError(f"{path} is not a calibration report")
    found = {}
    for task in report.get("per_task") or ():
        seconds = task.get("median_agent_seconds")
        found[task["task"]] = {"kept": bool(task.get("kept")), "usd": task.get("median_usd"), "seconds": seconds,
                               "runs": task.get("attempts")}
    return found
