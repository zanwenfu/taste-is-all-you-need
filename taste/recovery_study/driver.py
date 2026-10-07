"""What the map and recovery drivers share: their state, their jobs and their runs.

A driver is run, its jobs are started from the shell script it writes, and
once they have ended it is run again. Each run reads the jobs that have
finished, decides the next ones and records them before they are written out,
so stopping it anywhere loses nothing and a job is never started twice.

A failed base run enters a study as a run: its task, its steps, the settings
it ran with, where its record is and where its replay script will be written.
Runs from scratch of each task (calibration and base runs) give v(0) and the
budget of one base run.
"""

from __future__ import annotations

import os
from pathlib import Path

from taste.recovery_study import jobs, records, rules, state

# A driver's exit status: new jobs were written; none, but some are running;
# every run is finished.
STARTED, RUNNING, FINISHED = 0, 2, 3

COMMON = {
    "prefix": "study",          # every job name starts with it; one prefix per state file
    "model": jobs.WORKER_MODEL,
    "jobs_dir": "/root/study/jobs",
    "trials_root": "/var/lib/taste-trials",
    "work_dir": "",             # replay scripts, notes, checker inputs; empty: <state>.d
    "tasks_dir": "",            # only for base trials whose result names no task path
    "run_harbor": jobs.RUN_HARBOR,
    "export_template": jobs.EXPORT_TEMPLATE,
    "concurrent": 8,            # -n of each job
    "env": {},                  # more environment for run-harbor.sh (EXTRA_PATH, TASTE_SOURCE, ...)
    "ak": {},                   # settings every agent trial adds or overrides
    "min_steps": 2,
    "large_edit_lines": rules.LARGE_EDIT_LINES,
    "attempts_factor": 2,       # attempts started for an outcome, at most this many times those wanted
}


class Driver:
    KIND = ""
    DEFAULTS = COMMON

    def __init__(self, path, data):
        self.path, self.data, self.new = Path(path), data, []

    @classmethod
    def open(cls, path, given=None):
        stored = state.load(path, cls.KIND)
        data = stored or {"kind": cls.KIND, "created": state.now(), "sources": {"base": [], "calibration": []},
                          "fresh": {}, "runs": {}, "jobs": {}}
        data["settings"] = state.settings(stored and stored["settings"], given or {}, cls.DEFAULTS)
        cls.validate(data["settings"])
        return cls(path, data)

    @classmethod
    def validate(cls, settings):
        """Refuse settings no job could be built from, before any job is."""
        if not str(settings["model"]).strip():
            raise ValueError("model must name a model")
        if jobs.safe(settings["prefix"], len(settings["prefix"])) != settings["prefix"]:
            raise ValueError("prefix: letters, digits, - and _ only")
        for name in ("concurrent", "min_steps", "large_edit_lines", "attempts_factor"):
            if type(settings[name]) is not int or settings[name] < 1:
                raise ValueError(f"{name} must be a positive whole number")

    def save(self):
        state.save(self.path, self.data)

    @property
    def settings(self):
        return self.data["settings"]

    @property
    def runs(self):
        return self.data["runs"]

    @property
    def jobs(self):
        return self.data["jobs"]

    def work_dir(self):
        return Path(self.settings["work_dir"] or self.path.with_name(self.path.stem + ".d"))

    # Jobs ------------------------------------------------------------------

    def emit(self, spec, **meta):
        if spec.name in self.jobs:
            raise RuntimeError(f"job {spec.name} was already started")
        self.jobs[spec.name] = {"spec": spec.to_dict(), "emitted": state.now(), "complete": False,
                                "results": None, **meta}
        self.new.append(spec)
        return spec

    def refresh(self):
        """Read every started job that has finished; the number read."""
        read = 0
        for name, job in self.jobs.items():
            if job["complete"]:
                continue
            directory = Path(job["spec"]["env"]["JOBS"]) / name
            if not directory.is_dir():
                continue
            found = records.job_trials(directory, self.settings["trials_root"])
            if len(found) >= job["spec"]["attempts"] or records.job_finished(directory):
                job.update(complete=True, results=found, finished=state.now())
                read += 1
        return read

    def close(self, name):
        """Take a job as finished with what it has: one that crashed, or was never started."""
        job = self.jobs[name]
        directory = Path(job["spec"]["env"]["JOBS"]) / name
        found = records.job_trials(directory, self.settings["trials_root"]) if directory.is_dir() else []
        job.update(complete=True, results=found, finished=state.now(), closed=True)

    def pending(self, names):
        return any(not self.jobs[name]["complete"] for name in names)

    def attempts(self, names):
        return sum(self.jobs[name]["spec"]["attempts"] for name in names)

    def results(self, names):
        """(job, summary) for every finished trial of these jobs, in the order they were started."""
        return [(name, summary) for name in names if self.jobs[name]["complete"]
                for summary in self.jobs[name]["results"]]

    def trial(self, reference):
        name, trial = reference
        return next(summary for summary in self.jobs[name]["results"] if summary["trial"] == trial)

    def harbor_job(self, name, run, settings, attempts, *, model=None, prepare=()):
        values = self.settings
        return jobs.harbor_job(name, tasks_dir=run["tasks_dir"], task=run["task"], attempts=attempts,
                               settings=settings, model=model or values["model"], jobs_dir=values["jobs_dir"],
                               run_harbor=values["run_harbor"], concurrent=values["concurrent"],
                               env=values["env"], prepare=prepare)

    def script(self, token):
        """Where a trial's replay script is written, and the command that writes it."""
        path = self.work_dir() / "scripts" / f"{token}.json"
        record = Path(self.settings["trials_root"]) / str(token)
        return str(path), jobs.export_script(record, path, self.settings["export_template"])

    def launcher(self, specs, max_active, title=""):
        return jobs.launcher(specs, prefix=self.settings["prefix"], max_active=max_active, title=title)

    # Runs ------------------------------------------------------------------

    def add_sources(self, base_dirs=(), calibration_dirs=()):
        sources = self.data["sources"]
        for kind, dirs in (("base", base_dirs), ("calibration", calibration_dirs)):
            for directory in dirs:
                if str(directory) not in sources[kind]:
                    sources[kind].append(str(directory))
        fresh = self.data["fresh"]
        for kind in ("calibration", "base"):
            for directory in sources[kind]:
                for path in sorted(Path(directory).glob("*/result.json")):
                    key = str(path.parent)
                    if key in fresh:
                        continue
                    result = records.read_json(path)
                    if not isinstance(result, dict) or not records.finished(result):
                        continue
                    summary = records.trial(path.parent, self.settings["trials_root"])
                    fresh[key] = {"kind": kind, "job_dir": str(directory),
                                  **{name: summary[name] for name in ("trial", "task", "reward", "solved", "cost_usd",
                                                                      "seconds", "model", "exit_status")}}
                    if kind == "base" and summary["reward"] is not None and not summary["solved"]:
                        self._add_run(summary, directory)

    def fresh(self, task, kinds=("calibration", "base")):
        return [run for run in self.data["fresh"].values()
                if run["task"] == task and run["kind"] in kinds and run["reward"] is not None]

    def v0(self, task):
        """(successes, runs) of the task from scratch, by calibration and base runs."""
        runs = self.fresh(task)
        return [sum(1 for run in runs if run["solved"]), len(runs)]

    def _add_run(self, summary, job_dir):
        run_id = summary["trial"]
        if run_id in self.runs:
            if self.runs[run_id].get("job_dir") == str(job_dir):
                return
            run_id = f"{Path(job_dir).name}/{run_id}"
        self.runs[run_id] = self.base_run(run_id, summary, job_dir)

    def base_run(self, run_id, summary, job_dir):
        """A failed base run as a study run, or a record of why it is left out."""
        values = self.settings
        nested = records.settled(summary["token"], values["trials_root"])
        view = records.agent_view(nested)
        tasks_dir = summary["tasks_dir"] or values["tasks_dir"]
        skipped = None
        if view is None:
            skipped = "no settled record of the agent's run"
        elif len(view["steps"]) < values["min_steps"]:
            skipped = f"{len(view['steps'])} steps"
        elif jobs.short_model(summary["model"]) != jobs.short_model(values["model"]):
            skipped = f"base model {summary['model']} is not this study's {values['model']}"
        elif not tasks_dir:
            skipped = "no tasks directory"
        run = {"key": jobs.run_key(run_id), "task": summary["task"], "tasks_dir": tasks_dir,
               "job_dir": str(job_dir), "model": summary["model"],
               "base": {name: summary[name] for name in ("trial", "token", "reward", "cost_usd", "seconds",
                                                         "exit_status", "steps")}}
        if skipped:
            return {**run, "skipped": skipped}
        taken = view["steps"]
        script, _ = self.script(summary["token"])
        return {**run, "steps": len(taken), "settings": jobs.inherited(summary["settings"]), "script": script,
                "rules": rules.rewind_points(taken, values["large_edit_lines"]), "features": rules.features(taken)}


def pairs(items):
    """NAME=VALUE arguments as a dict, or None when none were given (the stored ones stand)."""
    if not items:
        return None
    found = {}
    for item in items:
        name, separator, raw = item.partition("=")
        if not separator or not name:
            raise ValueError(f"expected NAME=VALUE, got {item!r}")
        found[name] = raw
    return found


def finish(driver, launcher=None, max_active=4, title=""):
    """Save the state, write the launcher, and say how the study stands.

    The launcher lists every job not yet finished, in the order they were
    decided, and skips those already started; so a job recorded by a run that
    stopped before writing its launcher is started by the next one.
    """
    driver.save()
    if launcher:
        unfinished = [jobs.JobSpec.from_dict(job["spec"]) for job in driver.jobs.values() if not job["complete"]]
        path = Path(launcher)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(driver.launcher(unfinished, max_active, title))
        os.chmod(path, 0o755)
    if driver.new:
        return STARTED
    return RUNNING if any(not job["complete"] for job in driver.jobs.values()) else FINISHED
