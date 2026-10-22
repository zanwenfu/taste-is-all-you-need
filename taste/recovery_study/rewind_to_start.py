"""A rewind arm's way back to the start, graded at last, and the episodes it ended let go on.

A commit with no round left queues a round that only grades the state the agent chose (``grade_only``: its files
rebuilt from the record, no model call). Chosen at the run's first call, that round has no step to replay, so until
``harbor_agent`` let such a branch through, its trial stopped as one whose hosted agent never took a step
(``HostedAgentNotStarted``), never graded: tried twice, the episode ended ``agent_failed`` with the round taken off.
``reopen`` voids those jobs (``Driver.void``: no attempt and no results, kept on record) and puts the round back as it
was queued, for the driver to start again on code that grades it; ``host_order.move_code`` points the state's later
jobs at that code.
"""

from __future__ import annotations

from taste.recovery_study.jobs import JobSpec
from taste.recovery_study.recovery_driver import AGENTREWIND, new_check

REASON = ("a grade-only round back to the run's first call replays no step, and the harness stopped its trial as one "
          "whose agent never started (harbor_agent._check_started) instead of grading it")
NOT_STARTED = "HostedAgentNotStarted"


def _from_the_start(job):
    settings = JobSpec.from_dict(job["spec"]).settings()
    return settings.get("branch_live") == "off" and settings.get("branch_step") == "0"


def stopped(driver, names):
    """Whether these jobs are a grade-only round from the start that the harness stopped: all finished, none voided,
    each from step 0 with live off, no trial graded, and one at least stopped as never started."""
    jobs = [driver.jobs[name] for name in names]
    results = [summary for job in jobs for summary in job.get("results") or ()]
    return (bool(jobs) and all(job["complete"] and not job.get("void") and _from_the_start(job) for job in jobs)
            and all(summary["reward"] is None for summary in results)
            and any(summary.get("exception") == NOT_STARTED for summary in results))


def reopen(driver):
    """Void the jobs of every rewind episode that ended ``agent_failed`` on such a round, and put the round back.

    Returns the job names voided and the episodes reopened as (run, arm, episode, round). Running it again does
    nothing more."""
    voided, reopened = [], []
    for run_id, run in driver.runs.items():
        arm = (run.get("arms") or {}).get(AGENTREWIND)
        for index, episode in enumerate((arm or {}).get("episodes") or ()):
            if episode["end"] != "agent_failed" or not episode["rounds"] or not episode.get("notes"):
                continue
            number = len(episode["rounds"]) + 1
            names = [name for name, job in driver.jobs.items()
                     if job.get("purpose") == "round" and job.get("run") == run_id and job.get("arm") == AGENTREWIND
                     and job.get("episode") == index and job.get("round") == number]
            if not stopped(driver, names):
                continue
            [target] = {driver.jobs[name]["rewind_target"] for name in names}
            for name in names:
                driver.void(name, REASON)
            voided += names
            # As _rewind_round queued it; the voided jobs stay its own, so the next is named after them.
            episode["rounds"].append({"agent": None, "agent_jobs": list(names), "check": new_check(),
                                      "rewind": {"target": target, "notes": episode["notes"]}, "grade_only": True})
            episode["end"] = None
            reopened.append((run_id, AGENTREWIND, index, number))
    return {"voided": voided, "reopened": reopened}
