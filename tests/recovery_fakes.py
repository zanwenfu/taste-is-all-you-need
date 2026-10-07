"""Synthetic Harbor jobs and Taste records for the recovery study's tests.

``trial`` writes one finished Harbor trial (its result.json, and the flat
record whose last reply is what the trial handed back) and, given steps, the
settled record Taste keeps for it. ``FakeHarbor`` runs the job specs a driver
emits by writing their trials, with outcomes from a rule the test gives, so a
driver can be run to its end in one process.
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from taste.agents.checker import SCHEMA

SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
BASE_SETTINGS = {"agent": "mini-swe-agent", "services": "none", "reply_reserve_seconds": "10",
                 "worker_python": "/home/bugbash/venv/bin/python"}
NOT_DONE = {"unmet_requirement": "Empty input must parse to an empty list; it raises.",
            "evidence": {"check": "Parsed an empty string, as the task's first example does.",
                         "command": "python -c 'import app; print(app.parse(\"\"))'",
                         "output": "Traceback (most recent call last):\nValueError: empty",
                         "expected": "[] on standard output", "observed": "a ValueError traceback"}}
_tokens = itertools.count(1)


def agent_steps(count, *, tests_pass_at=(), large_edit_at=(), submit=True):
    """(command, return code, output, message) for ``count`` steps; the last one submits."""
    found = []
    for index in range(1, count + 1):
        if submit and index == count:
            found.append((f"echo {SENTINEL} && git diff", 0,
                          f"{SENTINEL}\ndiff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n+fix\n",
                          "I fixed the bug and ran the tests."))
        elif index in tests_pass_at:
            found.append(("cd /testbed && python -m pytest tests -q", 0, "3 passed in 0.12s", f"Run the tests ({index})."))
        elif index in large_edit_at:
            body = "\n".join(f"line_{number} = {number}" for number in range(25))
            found.append((f"cat > src/app.py <<'EOF'\n{body}\nEOF", 0, "", f"Rewrite the module ({index})."))
        else:
            found.append((f"grep -n thing src/app.py  # look {index}", 0, "12: thing", f"Look ({index})."))
    return found


def submission(verdict, *, confidence=0.8, cost=0.05):
    """A checker's submission text, valid for ``taste.agents.checker.parse_submission``."""
    value = {"schema": SCHEMA, "verdict": verdict, **(NOT_DONE if verdict == "not_done" else {}),
             "confidence": confidence, "ended": "verdict", "steps": 6, "commands": 5, "refused": 0,
             "cost_usd": cost, "tokens": {"input": 1000, "output": 100}}
    return json.dumps(value)


def record(trials_root, token, steps, *, task="Fix the bug.", exit_status="Submitted", cost=0.02,
           tokens=(1000, 100)):
    """The settled record of a trial: a fixed plan and one hosted agent run with these steps."""
    run = [{"step_id": 1, "source": "user", "message": task}]
    for index, (command, code, output, message) in enumerate(steps, start=1):
        call = f"call_{index}"
        run.append({"step_id": index + 1, "source": "agent", "message": message,
                    "tool_calls": [{"tool_call_id": call, "function_name": "bash", "arguments": {"command": command}}],
                    "observation": {"results": [{"source_call_id": call, "content": output,
                                                 "extra": {"command": command, "returncode": code}}]},
                    "metrics": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1], "cached_tokens": 0,
                                "cost_usd": cost / max(1, len(steps))}})
    nested = {"schema_version": "ATIF-v1.7", "session_id": token,
              "steps": [{"step_id": 1, "source": "user", "message": task}],
              "subagent_trajectories": [
                  {"agent": {"name": "taste-coordinator"},
                   "steps": [{"step_id": 1, "source": "system", "message": "fixed plan",
                              "metrics": {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}}]},
                  {"agent": {"name": "mini-swe-agent"}, "steps": run,
                   "extra": {"exit": {"exit_status": exit_status, "stopped_by": ""}}}],
              "final_metrics": {"extra": {"known_cost_usd": cost}, "total_cost_usd": cost,
                                "total_prompt_tokens": tokens[0] * len(steps),
                                "total_completion_tokens": tokens[1] * len(steps)}}
    path = Path(trials_root) / token / "controller" / "trajectory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nested))


def trial(jobs_dir, job, task, *, reward, trials_root, steps=None, settings=None, model="azure/gpt-6-luna",
          tasks_dir="/study/tasks", seconds=120.0, cost=0.02, exit_status="Submitted", branch=None,
          handed_back=None, exception=None, name=None, task_text="Fix the bug.", audit_flags=()):
    """One finished Harbor trial; with steps, its settled Taste record; with ``handed_back``, its final reply.

    ``seconds`` is the agent's execution by Harbor's clock (None: it never
    ran). Taste's own count of the run's time differs from it on purpose: the
    study's clock is Harbor's.
    """
    token = f"tok{next(_tokens):06d}"
    name = name or f"{task}__{token[-6:]}"
    taste = {"trial": token, "seconds": None if seconds is None else seconds + 17.0, "audit_flags": list(audit_flags)}
    began = datetime(2026, 10, 7, 10, 0, 5, tzinfo=UTC)
    execution = None if seconds is None else {
        "started_at": began.isoformat().replace("+00:00", "Z"),
        "finished_at": (began + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")}
    if branch is not None:
        taste["branch"] = branch
    if steps is not None:
        record(trials_root, token, steps, task=task_text, exit_status=exit_status, cost=cost)
    result = {"task_name": task, "trial_name": name,
              "config": {"task": {"path": f"{tasks_dir}/{task}"},
                         "agent": {"model_name": model, "kwargs": dict(settings or {})}},
              "agent_result": {"cost_usd": cost, "n_input_tokens": 0, "n_output_tokens": 0,
                               "metadata": {"taste": taste}},
              "verifier_result": None if reward is None else {"rewards": {"reward": reward}},
              "exception_info": None if exception is None else {"exception_type": exception},
              "started_at": "2026-10-07T10:00:00Z", "finished_at": "2026-10-07T11:00:00Z",
              "agent_execution": execution}
    directory = Path(jobs_dir) / job / name
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(json.dumps(result))
    if handed_back is not None:
        (directory / "agent").mkdir()
        (directory / "agent" / "trajectory.json").write_text(json.dumps({"steps": [
            {"step_id": 1, "source": "user", "message": "review"},
            {"step_id": 2, "source": "agent", "message": "ran a check", "extra": {"is_sidechain": True}},
            {"step_id": 3, "source": "agent", "message": handed_back, "extra": {"is_sidechain": False}}]}))
    return name


def finish(jobs_dir, job):
    """Harbor's job-level result, written when the job ends."""
    path = Path(jobs_dir) / job / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"started_at": "2026-10-07T10:00:00Z", "finished_at": "2026-10-07T11:00:00Z"}))


def base_jobs(jobs_dir, trials_root, runs, *, calibration=(), task="task-a", model="azure/gpt-6-luna"):
    """A base job of failed runs (each a step list, or (steps, exit status)) and a calibration job.

    ``calibration`` is a list of (reward, dollars, seconds). Returns the base
    run names.
    """
    names = []
    for steps in runs:
        steps, status = steps if isinstance(steps, tuple) else (steps, "Submitted")
        names.append(trial(jobs_dir, "base", task, reward=0.0, trials_root=trials_root, steps=steps,
                           settings=BASE_SETTINGS, model=model, exit_status=status))
    for reward, cost, seconds in calibration:
        trial(jobs_dir, "calibration", task, reward=reward, trials_root=trials_root, steps=agent_steps(3),
              settings=BASE_SETTINGS, model=model, cost=cost, seconds=seconds)
    finish(jobs_dir, "base")
    finish(jobs_dir, "calibration")
    return names


def parse(spec):
    """A job spec's task, attempts and settings."""
    argv = list(spec.argv)
    return argv[argv.index("-i") + 1], spec.attempts, spec.settings()


class FakeHarbor:
    """Runs job specs by writing their trials; ``outcome(spec, task, settings, harbor)`` gives each trial's fields."""

    def __init__(self, trials_root, outcome):
        self.trials_root, self.outcome = Path(trials_root), outcome
        self.started, self.counts = [], defaultdict(int)

    def run(self, specs, *, upto=None, already=0):
        """Write each spec's trials from number ``already`` up to ``upto`` (all), closing finished jobs."""
        for spec in specs:
            task, attempts, settings = parse(spec)
            env = dict(spec.env)
            stop = attempts if upto is None else min(upto, attempts)
            for _ in range(already, stop):
                fields = self.outcome(spec, task, settings, self)
                trial(env["JOBS"], spec.name, task, trials_root=self.trials_root, settings=settings,
                      model=env["MODEL"], **fields)
            if stop >= attempts:
                finish(env["JOBS"], spec.name)
                self.started.append(spec.name)


def branches(success, *, state=None, unfaithful_from=None):
    """Map outcomes: the i-th branch from step k succeeds when ``success(k, i)``.

    A checkpoint trial (``branch_live=off``) records the checkpoint it saved
    and is graded ``state(k)``. Steps from ``unfaithful_from`` on cannot be
    replayed faithfully.
    """
    def outcome(spec, task, settings, harbor):
        step = int(settings["branch_step"])
        faithful = unfaithful_from is None or step < unfaithful_from
        if settings.get("branch_live") == "off":
            return {"reward": state(step) if state else 0.0,
                    "branch": {"faithful": faithful, "checkpoint": settings.get("branch_checkpoint")
                               if faithful else None}}
        index = harbor.counts[(task, step)]
        harbor.counts[(task, step)] += 1
        return {"reward": 1.0 if faithful and success(step, index) else 0.0, "branch": {"faithful": faithful}}
    return outcome


def recoveries(verdict=lambda spec: "not_done", reward=lambda spec: 0.0, status=lambda spec: "Submitted",
               faithful=lambda spec: True, prefix=None, check_usd=0.05):
    """Recovery outcomes. A checker trial hands back ``verdict(spec)`` and costs ``check_usd`` ($0.05) and 50 s; a trial
    that saves a final state costs nothing and lists the changed paths; an agent trial submits
    (``status``), costs $0.20 and 150 s and is graded ``reward``. With ``prefix``, a branch trial and a
    checker trial record that many of their seconds as bringing their prefix back
    (``{"agent": s, "checker": s}``)."""
    prefix = prefix or {}

    def outcome(spec, task, settings, harbor):
        if settings.get("agent") == "checker":
            return {"reward": 1.0, "handed_back": submission(verdict(spec)), "cost": check_usd, "seconds": 50.0,
                    "branch": {"faithful": faithful(spec), **_prefix(prefix.get("checker"))}}
        if settings.get("branch_live") == "off":
            return {"reward": 0.0, "cost": 0.0, "seconds": 30.0,
                    "branch": {"faithful": True, "checkpoint": settings["branch_checkpoint"],
                               "changed_paths": ["src/app.py"]}}
        return {"reward": reward(spec), "steps": agent_steps(5), "cost": 0.2, "seconds": 150.0,
                "exit_status": status(spec),
                "branch": {"faithful": True, **_prefix(prefix.get("agent"))} if "branch" in settings else None}
    return outcome


def _prefix(seconds):
    return {} if seconds is None else {"prefix_seconds": seconds}
