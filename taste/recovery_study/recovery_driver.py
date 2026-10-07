"""The recoveries: from each failed run the checker rejects, four ways to go on at equal cost.

For each failed base run that submitted, the checker (``taste.agents.checker``)
first checks its final state. A run it finds ``not_done`` is recovered in each
way asked for, ``repeats`` times, from the same failed run:

- ``retry``: a fresh trial, the task text only;
- ``retry_feedback``: a fresh trial, the task text followed by
  "A previous attempt was rejected by a reviewer: <feedback>";
- ``rewind``: a branch from step r, the recorded output of step r followed by
  "Note: an attempt that continued from here was rejected by a reviewer:
  <feedback>", for each source of r: the map's oracle r*, the trajectory
  reader's reading (rewind to the step before the one it names) and the rules.
  Sources that choose the same step share its trials;
- ``continue``: a branch from step T, the submission, whose output is replaced
  by "Submission rejected by a reviewer: <feedback>" with a non-zero exit;
- ``continue_bare`` (an ablation of continue): the same, with only the
  checker's bare rejection.

The feedback is the checker's own rendering of its findings
(``taste.agents.checker.feedback``).

A check of a trial's final state is a checker trial: the checker alone
(``--ak agent=checker --ak services=none``, GPT-6 Sol), given the text
``checker_task`` builds from what the checked trial's agent was given and
handed back, on a copy of that trial's final files. In ``restore`` mode the
copy is made first by a trial that rebuilds the final state and saves it
(``branch_live=off``); its manifest's changed paths go into the checker's text
(for work outside a git repository), the checker restores it, and the next
round continues from it. In ``rebuild`` mode the checker trial rebuilds it.

Every recovery gets the same budget, one base run of the task: the median
dollars and agent time of its calibration runs (``scripts/calibration_report.py``),
passed to each trial through Taste's equal caps (``spend_cap_usd``,
``agent_timeout_sec``). After a recovery trial submits, a checker trial checks
it; on ``not_done`` the next round continues from its submission
(``branch_override=reject``) with the new feedback, until the checker says
``done``, the trial ends without submitting, or the budget cannot pay for
another round (a round needs a tenth of the budget left, ``min_round_share``).
A plain retry stays plain: its next round is another fresh trial, told
nothing (``retry_rounds=continue`` continues it with feedback instead).

The budget is the agent's: its trials' dollars and time are charged to it,
round after round. The checker's are added up beside it and reported, never
charged: the same checker serves every recovery, and with a cheap agent model
one check can cost more than a whole base run, which would leave no recovery a
second round. Time has one clock, Harbor's agent execution, for the budget
(the calibration's median) and for every recovery and checker trial; a branch
is charged without the time branching spent bringing its prefix back
(``prefix_seconds``), and both the raw and the charged seconds are kept.
Every trial is graded by the hidden tests; the episode's outcome is its last
trial's grade, and no grade decides anything.

State (``kind`` taste.recovery_runs/1), beside the shared fields of ``driver``:

    runs.<run>.budget               {"usd", "seconds", "runs", "source"}
    runs.<run>.check                a check, with "reply": the checker's submission, once read
    runs.<run>.sources              {"oracle" | "reader" | "rules": rewind step}
    runs.<run>.arms.<arm>           {"recovery", "step", "first": [round-one jobs],
                                     "episodes": [{"rounds": [round], "end": why it ended}]}
    round                           {"agent": [job, trial], "agent_jobs": [later rounds' jobs], "check": check}
    check                           {"checkpoints": [final-state jobs], "jobs": [checker jobs],
                                     "verdict": [job, trial] of the checker's verdict}
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from taste.agents.checker import BARE_FEEDBACK, checker_task, feedback
from taste.agents.trajectory_reader import rewind_point
from taste.recovery_study import jobs, records, search
from taste.recovery_study.commands import gaming_flags
from taste.recovery_study.driver import COMMON, Driver

RECOVERIES = ("retry", "retry_feedback", "rewind", "continue", "continue_bare")
SOURCES = ("oracle", "reader", "rules")
SLUG = {"retry": "rt", "retry_feedback": "rf", "continue": "ct", "continue_bare": "cb"}
NOTES = {"retry_feedback": "A previous attempt was rejected by a reviewer: {feedback}",
         "rewind": "Note: an attempt that continued from here was rejected by a reviewer: {feedback}",
         "continue": "Submission rejected by a reviewer: {feedback}"}
# Taste's defaults (TrialSettings) for the time a trial holds back: for the
# handoff to the verifier, for a closing reply, and the least time a plan needs.
HANDOFF_SECONDS, REPLY_RESERVE_SECONDS, PLAN_SECONDS = 150.0, 210.0, 90.0
ENDS = ("done", "budget", "no_submission", "agent_failed", "check_failed")

DEFAULTS = {
    **COMMON,
    "prefix": "rec",
    "mode": "restore",          # or "rebuild": how a check gets its copy of the final state
    "calibration_report": "",   # scripts/calibration_report.py --json: budgets, and the tasks kept
    "checker_model": jobs.CHECKER_MODEL,
    "checker_effort": "medium",
    "checker_ak": {},           # settings every checker trial adds or overrides
    "recoveries": ["retry", "retry_feedback", "rewind", "continue"],
    "rewind_sources": list(SOURCES),
    "oracle": "highest_v",      # or "latest": which of the map's r* the oracle rewinds to
    "repeats": 3,
    "check_attempts": 2,
    "prefix_allowance": 300.0,  # seconds a branch's deadline adds for bringing its prefix back
    "min_round_usd": 0.0,       # dollars a round needs left, at least; and at least this share of the budget:
    "min_round_share": 0.1,
    "min_round_seconds": 0.0,   # 0: the trial's reply reserve and plan time, and a minute of work
    "retry_rounds": "retry",    # rounds after a plain retry: fresh trials told nothing, or "continue" with feedback
}


def readings(path):
    """The trajectory reader's readings by run, from a JSON file.

    ``{"<run>": <reading>}`` or a list of readings with a ``"run"``; a reading is
    ``read_trajectory``'s ``{"step", "reason", "confidence"}`` (or a bare step).
    """
    if path is None or not Path(path).exists():
        return {}
    raw = json.loads(Path(path).read_text())
    items = raw.items() if isinstance(raw, dict) else ((item.get("run"), item) for item in raw)
    found = {}
    for run_id, item in items:
        reading = item if isinstance(item, dict) else {"step": item}
        if type(reading.get("step")) is int:
            found[str(run_id)] = reading
    return found


def note(recovery, submission):
    """What the agent is shown, after the checker's ``not_done`` submission."""
    if recovery == "continue_bare":
        return BARE_FEEDBACK
    return NOTES[recovery].format(feedback=feedback(submission, "evidence"))


def handoff_seconds(settings):
    return float(settings.get("handoff_seconds", HANDOFF_SECONDS))


def round_seconds(settings, configured=0.0):
    """The least agent time a round can use: its reply reserve and plan time, and a minute of work."""
    if configured:
        return float(configured)
    reserve = float(settings.get("reply_reserve_seconds", REPLY_RESERVE_SECONDS))
    return reserve + max(60.0, float(settings.get("plan_seconds", PLAN_SECONDS))) + 60.0


def new_check():
    return {"checkpoints": [], "jobs": [], "verdict": None}


class RecoveryDriver(Driver):
    KIND = "taste.recovery_runs/1"
    DEFAULTS = DEFAULTS

    def __init__(self, path, data):
        super().__init__(path, data)
        self.oracle, self.reader, self._calibrated = {}, {}, None

    @classmethod
    def validate(cls, settings):
        super().validate(settings)
        if not str(settings["checker_model"]).strip():
            raise ValueError("checker_model must name a model")
        for name, allowed in (("mode", jobs.MODES), ("checker_effort", jobs.EFFORTS),
                              ("oracle", ("highest_v", "latest")), ("retry_rounds", ("continue", "retry"))):
            if settings[name] not in allowed:
                raise ValueError(f"{name} must be one of: " + ", ".join(allowed))
        unknown = (set(settings["recoveries"]) - set(RECOVERIES)) | (set(settings["rewind_sources"]) - set(SOURCES))
        if unknown:
            raise ValueError("unknown recoveries or rewind sources: " + ", ".join(sorted(unknown)))
        if settings["repeats"] < 1 or settings["check_attempts"] < 1:
            raise ValueError("repeats and check_attempts must be positive")
        if not 0 <= settings["min_round_share"] < 1:
            raise ValueError("min_round_share must be at least 0 and below 1")

    def use(self, oracle=None, reader=None):
        """The map's results (``MapDriver.summary``) and the trajectory reader's readings."""
        self.oracle, self.reader = oracle or {}, reader or {}

    def calibration(self):
        if self._calibrated is None:
            path = self.settings["calibration_report"]
            self._calibrated = records.calibration_budgets(path) if path else {}
        return self._calibrated

    def _add_run(self, summary, job_dir):
        before = set(self.runs)
        super()._add_run(summary, job_dir)
        for run_id in set(self.runs) - before:
            run = self.runs[run_id]
            if not run.get("skipped") and run["base"]["exit_status"] != "Submitted":
                run["skipped"] = f"the base run did not submit ({run['base']['exit_status']})"
            if run.get("skipped"):
                continue
            found = self._budget(run)
            if isinstance(found, str):
                run["skipped"] = found
                continue
            run.update(budget=found, check={**new_check(), "reply": None}, sources={}, arms={})

    def _budget(self, run):
        """One base run's dollars and agent time for the run's task, or why there is none."""
        if self.settings["calibration_report"]:
            task = self.calibration().get(run["task"])
            if task is None:
                return "the task is not in the calibration report"
            if not task["kept"]:
                return "the calibration did not keep the task"
            if task["usd"] and task["seconds"]:
                return {"usd": float(task["usd"]), "seconds": float(task["seconds"]), "runs": task["runs"],
                        "source": "calibration report"}
        calibrated = self.fresh(run["task"], ("calibration",))
        found = records.budget(calibrated) or records.budget(self.fresh(run["task"], ("base",)))
        if found is None:
            return "no runs of the task give its budget"
        return {**found, "source": "calibration runs" if calibrated else "base runs"}

    # Deciding --------------------------------------------------------------

    def advance(self):
        """Decide each run's next jobs and record them; the job specs this call started."""
        before = len(self.new)
        for run_id, run in sorted(self.runs.items()):
            if run.get("skipped") or not self._initial_check(run_id, run):
                continue
            self._sources(run_id, run)
            for name, recovery, step in self._arms(run):
                arm = run["arms"].setdefault(name, {
                    "recovery": recovery, "step": step, "first": [],
                    "episodes": [{"rounds": [], "end": None} for _ in range(self.settings["repeats"])]})
                self._advance_arm(run_id, run, name, arm)
        return self.new[before:]

    def _initial_check(self, run_id, run):
        """True once the checker has rejected the base run's final state."""
        check = run["check"]
        if check["reply"] is None:
            source = {"token": run["base"]["token"], "steps": run["steps"], "settings": run["settings"]}
            found = self._check(run_id, run, check, source, ("check",), initial=True)
            if not found:
                if found is False:
                    run["status"] = "check_failed"
                return False
            check["verdict"], check["reply"] = found, self.trial(found)["checker"]
        run["status"] = "rejected" if check["reply"]["verdict"] == "not_done" else "accepted"
        return check["reply"]["verdict"] == "not_done"

    def _sources(self, run_id, run):
        values, sources = self.settings, run["sources"]
        if "rules" in values["rewind_sources"] and "rules" not in sources:
            sources["rules"] = run["rules"]["rules"]
        mapped = self.oracle.get(run_id) or {}
        if "oracle" in values["rewind_sources"] and "oracle" not in sources and mapped.get("status") in search.FINAL:
            sources["oracle"] = mapped["rewind"][values["oracle"]]
        reading = self.reader.get(run_id)
        if "reader" in values["rewind_sources"] and "reader" not in sources and reading is not None:
            sources["reader"] = max(0, min(rewind_point(reading), run["steps"] - 1))

    def _arms(self, run):
        """(arm, recovery, step) for every recovery this run can start now."""
        arms = []
        for recovery in self.settings["recoveries"]:
            if recovery == "rewind":
                for step in sorted(set(run["sources"].values())):
                    arms.append((f"rewind@{step}", "rewind", step))
            else:
                arms.append((recovery, recovery, run["steps"] if recovery.startswith("continue") else 0))
        return arms

    def _advance_arm(self, run_id, run, name, arm):
        values = self.settings
        taken = {tuple(episode["rounds"][0]["agent"]) for episode in arm["episodes"] if episode["rounds"]}
        pool = [(job, summary["trial"]) for job, summary in self.results(arm["first"])
                if records.usable(summary) and (job, summary["trial"]) not in taken]
        for episode in arm["episodes"]:
            if not episode["rounds"] and not episode["end"] and pool:
                episode["rounds"].append({"agent": list(pool.pop(0)), "agent_jobs": [], "check": new_check()})
        waiting = [episode for episode in arm["episodes"] if not episode["rounds"] and not episode["end"]]
        if waiting and not self.pending(arm["first"]):
            room = values["attempts_factor"] * values["repeats"] - self.attempts(arm["first"])
            if room <= 0:
                for episode in waiting:
                    episode["end"] = "agent_failed"
            else:
                job = jobs.job_name(values["prefix"], run["key"], _slug(name), f"f{len(arm['first']) + 1}")
                self._emit_first(run_id, run, name, arm, job, min(len(waiting), room))
                arm["first"].append(job)
        for index, episode in enumerate(arm["episodes"]):
            self._advance_episode(run_id, run, name, arm, index, episode)

    # Checks ----------------------------------------------------------------

    def final_state(self, check):
        """The saved copy of a checked trial's final state: (checkpoint, changed paths), or None."""
        for name in check["checkpoints"]:
            job = self.jobs[name]
            for summary in job["results"] if job["complete"] else ():
                if summary["exception"] is None and summary["faithful"] is not False \
                        and summary["checkpoint"] in (None, name):
                    return name, summary["changed_paths"]
        return None

    def _check(self, run_id, run, check, source, label, **meta):
        """Advance one check of a trial's final state: [job, trial] of its verdict, None while it is
        being made, False when it cannot be.

        ``source`` is the checked trial: its token, steps and settings.
        """
        values = self.settings
        found = [(job, summary["trial"]) for job, summary in self.results(check["jobs"]) if records.verdict(summary)]
        if found:
            return list(found[0])
        if self.pending(check["jobs"]) or self.pending(check["checkpoints"]):
            return None
        if len(check["jobs"]) >= values["check_attempts"]:
            return False
        view = records.agent_view(records.settled(source["token"], values["trials_root"]))
        if view is None or not view["task"].strip():
            return False
        script, export = self.script(source["token"])
        state = self.final_state(check)
        if values["mode"] == "restore" and state is None and len(check["checkpoints"]) < values["check_attempts"]:
            name = jobs.job_name(values["prefix"], run["key"], *label, f"s{len(check['checkpoints']) + 1}")
            settings = {**jobs.inherited(source["settings"], drop=jobs.CAPS), **values["ak"],
                        **jobs.branch(script, source["steps"], mode="rebuild", checkpoint=name, live=False)}
            self.emit(self.harbor_job(name, run, settings, 1, prepare=[export]), run=run_id,
                      purpose="final_state", **meta)
            check["checkpoints"].append(name)
            return None
        checkpoint, paths = state or (None, None)
        name = jobs.job_name(values["prefix"], run["key"], *label, f"c{len(check['jobs']) + 1}")
        text = checker_task(view["task"], view["final_message"], view["submission"], changed_paths=paths)
        path = jobs.write_text(self.work_dir() / "checker" / f"{name}.md", text)
        files = jobs.branch(script, source["steps"], mode="restore" if checkpoint else "rebuild",
                            checkpoint=checkpoint, live=False)
        settings = {**jobs.checker_trial(path, files, effort=values["checker_effort"]), **values["checker_ak"]}
        self.emit(self.harbor_job(name, run, settings, 1, model=values["checker_model"], prepare=[export]),
                  run=run_id, purpose="check", **meta)
        check["jobs"].append(name)
        return None

    # Episodes --------------------------------------------------------------

    def spent(self, episode):
        """What an episode has spent so far: dollars, seconds, tokens, and the parts.

        Its trials, and every checker trial of its checks; not the trials that
        rebuilt a final state for a check, which a harness restoring a
        snapshot would not run. Seconds are Harbor's agent execution times:
        ``seconds`` is what the budget is charged (a branch's without its
        prefix's time), ``raw_seconds`` the times as measured, and
        ``prefix_seconds`` the difference.
        """
        total = {name: 0.0 for name in ("agent_usd", "check_usd", "agent_seconds", "check_seconds",
                                        "raw_seconds", "prefix_seconds")}
        tokens = 0
        for step in episode["rounds"]:
            trials = [("agent", self.trial(step["agent"]))] if step.get("agent") else []
            trials += [("check", summary) for _, summary in self.results(step["check"]["jobs"])]
            for kind, summary in trials:
                total[kind + "_usd"] += summary["cost_usd"] or 0.0
                total[kind + "_seconds"] += summary["charged_seconds"] or 0.0
                total["raw_seconds"] += summary["seconds"] or 0.0
                total["prefix_seconds"] += (summary["seconds"] or 0.0) - (summary["charged_seconds"] or 0.0)
                tokens += (summary["prompt_tokens"] or 0) + (summary["completion_tokens"] or 0)
        return {"usd": total["agent_usd"] + total["check_usd"],
                "seconds": total["agent_seconds"] + total["check_seconds"], "tokens": tokens, **total}

    def remaining(self, run, episode):
        """What the budget has left for the agent: the checker's spending is kept beside it, not charged."""
        spent = self.spent(episode)
        return run["budget"]["usd"] - spent["agent_usd"], run["budget"]["seconds"] - spent["agent_seconds"]

    def _room(self, run, episode, settings):
        usd, seconds = self.remaining(run, episode)
        least = max(self.settings["min_round_usd"], self.settings["min_round_share"] * run["budget"]["usd"])
        return usd > least and seconds >= round_seconds(settings, self.settings["min_round_seconds"])

    def _advance_episode(self, run_id, run, name, arm, index, episode):
        while episode["rounds"] and not episode["end"]:
            current = episode["rounds"][-1]
            if current["agent"] is None:
                if self.pending(current["agent_jobs"]):
                    return
                found = [(job, summary["trial"]) for job, summary in self.results(current["agent_jobs"])
                         if records.usable(summary)]
                if found:
                    current["agent"] = list(found[0])
                    continue
                if len(current["agent_jobs"]) >= 2:
                    episode["end"] = "agent_failed"
                    episode["rounds"].pop()
                    return
                self._emit_round(run_id, run, name, arm, index, episode)
                return
            agent = self.trial(current["agent"])
            if agent["exit_status"] != "Submitted":
                episode["end"] = "no_submission"
                return
            if not self._room(run, episode, agent["settings"]):
                episode["end"] = "budget"
                return
            check = current["check"]
            if check["verdict"] is None:
                source = {"token": agent["token"], "steps": agent["steps"], "settings": agent["settings"]}
                found = self._check(run_id, run, check, source, (_slug(name), f"e{index + 1}r{len(episode['rounds'])}"),
                                    arm=name, episode=index)
                if found is False:
                    episode["end"] = "check_failed"
                if not found:
                    return
                check["verdict"] = found
                continue
            if self.trial(check["verdict"])["checker"]["verdict"] == "done":
                episode["end"] = "done"
                return
            if not self._room(run, episode, agent["settings"]):
                episode["end"] = "budget"
                return
            episode["rounds"].append({"agent": None, "agent_jobs": [], "check": new_check()})

    # Recovery trials -------------------------------------------------------

    def _note(self, job, text):
        return str(jobs.write_text(self.work_dir() / "notes" / f"{job}.txt", text))

    def _caps(self, usd, seconds, settings, branch):
        """Taste's equal caps for a trial with this much budget left."""
        extra = handoff_seconds(settings) + (self.settings["prefix_allowance"] if branch else 0.0)
        return jobs.caps(usd, seconds + extra)

    def _emit_first(self, run_id, run, name, arm, job, attempts):
        """Round one of an arm's episodes, as one job of ``attempts`` trials."""
        values, reply = self.settings, run["check"]["reply"]
        recovery, step = arm["recovery"], arm["step"]
        base = {**run["settings"], **values["ak"]}
        prepare, settings = [], dict(base)
        if recovery == "retry_feedback" or (recovery == "rewind" and step == 0):
            settings.update(jobs.task_suffix(self._note(job, note(recovery, reply))))
        elif recovery != "retry":
            script, export = self.script(run["base"]["token"])
            prepare = [export]
            if recovery == "rewind":
                override = "append"
                checkpoint = ((self.oracle.get(run_id) or {}).get("checkpoints") or {}).get(str(step))
            else:
                override, checkpoint = "reject", (self.final_state(run["check"]) or (None, None))[0]
            settings.update(jobs.branch(script, step, mode="restore" if checkpoint else "rebuild",
                                        checkpoint=checkpoint, live=True, override=override,
                                        note=self._note(job, note(recovery, reply))))
        settings.update(self._caps(run["budget"]["usd"], run["budget"]["seconds"], base, branch=bool(prepare)))
        self.emit(self.harbor_job(job, run, settings, attempts, prepare=prepare),
                  run=run_id, arm=name, purpose="first", recovery=recovery, step=step)

    def _emit_round(self, run_id, run, name, arm, index, episode):
        """A later round: on from the last trial's submission, with its checker's feedback."""
        values = self.settings
        previous = episode["rounds"][-2]
        agent, submission = self.trial(previous["agent"]), self.trial(previous["check"]["verdict"])["checker"]
        usd, seconds = self.remaining(run, episode)
        current = episode["rounds"][-1]
        job = jobs.job_name(values["prefix"], run["key"], _slug(name), f"e{index + 1}r{len(episode['rounds'])}",
                            f"a{len(current['agent_jobs']) + 1}")
        if arm["recovery"] == "retry" and values["retry_rounds"] == "retry":
            base = {**run["settings"], **values["ak"]}
            settings, prepare = {**base, **self._caps(usd, seconds, base, branch=False)}, []
        else:
            base = {**jobs.inherited(agent["settings"], drop=jobs.CAPS), **values["ak"]}
            script, export = self.script(agent["token"])
            checkpoint = (self.final_state(previous["check"]) or (None, None))[0]
            text = note("continue_bare" if arm["recovery"] == "continue_bare" else "continue", submission)
            settings = {**base, **jobs.branch(script, agent["steps"], mode="restore" if checkpoint else "rebuild",
                                              checkpoint=checkpoint, live=True, override="reject",
                                              note=self._note(job, text)),
                        **self._caps(usd, seconds, base, branch=True)}
            prepare = [export]
        self.emit(self.harbor_job(job, run, settings, 1, prepare=prepare), run=run_id, arm=name,
                  purpose="round", episode=index, round=len(episode["rounds"]))
        current["agent_jobs"].append(job)

    # Reporting -------------------------------------------------------------

    def outcome(self, episode):
        """An episode's result: its last trial's reward, and what all its rounds spent."""
        trials = [self.trial(step["agent"]) for step in episode["rounds"] if step.get("agent")]
        if not trials or not episode["end"]:
            return None
        last, spent = trials[-1], self.spent(episode)
        paths = sorted(set(last["written_paths"]) | set(last["submission_paths"]))
        verdicts = []
        for step in episode["rounds"]:
            if step.get("agent") and step["check"]["verdict"]:
                reply = self.trial(step["check"]["verdict"])["checker"]
                verdicts.append({"verdict": reply["verdict"], "confidence": reply.get("confidence"),
                                 "solved": self.trial(step["agent"])["solved"]})
        flags = gaming_flags(paths) + (["skip_marker"] if last["skip_marker"] else [])
        return {"end": episode["end"], "rounds": len(trials), "reward": last["reward"], "solved": last["solved"],
                **{key: round(value, 6) for key, value in spent.items()},
                "verdicts": verdicts, "changed_paths": paths, "flags": flags}

    def summary(self):
        """Per run: budget, the initial verdict, rewind sources, and each arm's episode outcomes."""
        out = {}
        for run_id, run in sorted(self.runs.items()):
            base = {"task": run["task"], "base": run["base"]}
            if run.get("skipped"):
                out[run_id] = {**base, "status": "skipped", "skipped": run["skipped"]}
                continue
            arms = {name: {"recovery": arm["recovery"], "step": arm["step"],
                           "episodes": [self.outcome(episode) for episode in arm["episodes"]]}
                    for name, arm in run["arms"].items()}
            out[run_id] = {**base, "status": run.get("status", "checking"), "steps": run["steps"],
                           "budget": run["budget"], "features": run["features"], "rules": run["rules"],
                           "verdict": run["check"]["reply"], "sources": run["sources"], "arms": arms,
                           "check_usd": round(math.fsum(s["cost_usd"] or 0.0
                                                        for _, s in self.results(run["check"]["jobs"])), 6)}
        return out


def _slug(arm):
    return SLUG.get(arm, arm.replace("rewind@", "rw"))
