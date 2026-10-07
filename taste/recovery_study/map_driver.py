"""The map: for each failed base run, the step after which no branch of it succeeds.

A probe of step k is K branches from the state at step k, graded by the
task's hidden tests; ``search`` decides which steps to probe. In ``restore``
mode (the design's) a probe first runs one checkpoint trial, which rebuilds
the state at step k, saves it and ends there (``branch_live=off``); its K
branches then restore that checkpoint. While a probe runs, the checkpoints of
the two steps the search can want next are made too (``prefetch``), so that
the next probe can start as soon as this one ends. In ``rebuild`` mode every
branch rebuilds its prefix.

A checkpoint trial is graded too: the hidden tests' verdict on the state at
step k ("state_reward"), which shows runs that passed and then broke.

State (``kind`` taste.recovery_map/1), beside the shared fields of ``driver``:

    runs.<run>.scan                 steps of the full scan, if the run is in it
    runs.<run>.probes.<k>           {"target": branches wanted, "jobs": [branch jobs],
                                     "checkpoints": [checkpoint jobs]}
    jobs.<job>                      {"run", "step", "purpose": "probe" | "checkpoint",
                                     "checkpoint": id, "spec", "complete", "results"}
"""

from __future__ import annotations

import hashlib
import math

from taste.recovery_study import jobs, records, search
from taste.recovery_study.driver import COMMON, Driver

DEFAULTS = {
    **COMMON,
    "prefix": "map",
    "k": 8,                     # branches in a probe
    "extra": 8,                 # added to each side of the boundary
    "mode": "restore",          # or "rebuild"
    "prefetch": True,
    "checkpoint_attempts": 2,
    "scan_every": 0,            # a full scan probes every N steps ...
    "scan_fraction": 0.0,       # ... of this share of the runs, chosen by a salted hash of the run
    "seed": "recovery-map",
}


class MapDriver(Driver):
    KIND = "taste.recovery_map/1"
    DEFAULTS = DEFAULTS

    @classmethod
    def validate(cls, settings):
        super().validate(settings)
        if settings["mode"] not in jobs.MODES:
            raise ValueError("mode must be rebuild or restore")
        if settings["k"] < 1 or settings["extra"] < 0 or settings["checkpoint_attempts"] < 1 \
                or settings["scan_every"] < 0 or not 0 <= settings["scan_fraction"] <= 1:
            raise ValueError("k and checkpoint_attempts must be positive, extra and scan_every not negative, "
                             "scan_fraction within [0, 1]")

    def _add_run(self, summary, job_dir):
        super()._add_run(summary, job_dir)
        values = self.settings
        for run_id, run in self.runs.items():
            if "scan" in run or run.get("skipped"):
                continue
            draw = int(hashlib.sha256(f"{values['seed']}:{run_id}".encode()).hexdigest(), 16) / 16 ** 64
            every = values["scan_every"]
            run["scan"] = list(range(every, run["steps"], every)) if every and draw < values["scan_fraction"] else []
            run["probes"] = {}

    # What the probes found ------------------------------------------------

    def _checkpoint(self, entry):
        """The checkpoint of a probe's step that is ready; or (None, failed)."""
        for name in entry["checkpoints"]:
            job = self.jobs[name]
            if job["complete"] and any(summary["exception"] is None and summary["faithful"] is not False
                                       and summary["checkpoint"] in (None, job["checkpoint"])
                                       for summary in job["results"]):
                return job["checkpoint"], False
        finished = [name for name in entry["checkpoints"] if self.jobs[name]["complete"]]
        return None, len(finished) >= self.settings["checkpoint_attempts"]

    def probes(self, run):
        """Every probe a run has been given work for, as the search reads it."""
        values, found = self.settings, {}
        for key, entry in run["probes"].items():
            if entry["target"] <= 0:
                continue
            usable = [summary for _, summary in self.results(entry["jobs"]) if records.usable(summary)]
            attempts = self.attempts(entry["jobs"])
            checkpoint, failed = (self._checkpoint(entry) if values["mode"] == "restore" else ("", False))
            pending = self.pending(entry["jobs"]) or (checkpoint is None and not failed and self.pending(entry["checkpoints"]))
            exhausted = failed or attempts >= values["attempts_factor"] * entry["target"]
            found[int(key)] = search.Probe(successes=sum(1 for s in usable if s["solved"]), trials=len(usable),
                                           pending=pending, exhausted=exhausted,
                                           unusable=not pending and not usable and exhausted)
        return found

    # Deciding --------------------------------------------------------------

    def _entry(self, run, step):
        return run["probes"].setdefault(str(step), {"target": 0, "jobs": [], "checkpoints": []})

    def advance(self):
        """Decide each run's next jobs and record them; the job specs this call started."""
        values, before = self.settings, len(self.new)
        for run_id, run in sorted(self.runs.items()):
            if run.get("skipped"):
                continue
            found = search.plan(run["steps"], self.probes(run), k=values["k"], extra=values["extra"],
                                scan=run["scan"])
            for step, wanted in found.targets.items():
                entry = self._entry(run, step)
                entry["target"] = max(entry["target"], wanted)
            for key, entry in sorted(run["probes"].items(), key=lambda item: int(item[0])):
                if entry["target"] > 0:
                    self._feed(run_id, run, int(key), entry)
            if values["mode"] == "restore" and values["prefetch"]:
                for step in found.next_steps:
                    self._make_checkpoint(run_id, run, step)
        return self.new[before:]

    def _feed(self, run_id, run, step, entry):
        values = self.settings
        if self.pending(entry["jobs"]):
            return
        usable = [summary for _, summary in self.results(entry["jobs"]) if records.usable(summary)]
        room = values["attempts_factor"] * entry["target"] - self.attempts(entry["jobs"])
        count = min(entry["target"] - len(usable), room)
        if count <= 0:
            return
        checkpoint = None
        if values["mode"] == "restore":
            checkpoint, _ = self._checkpoint(entry)
            if checkpoint is None:
                self._make_checkpoint(run_id, run, step)
                return
        name = jobs.job_name(values["prefix"], run["key"], f"s{step}", f"b{len(entry['jobs']) + 1}")
        settings = {**run["settings"], **values["ak"],
                    **jobs.branch(run["script"], step, mode="restore" if checkpoint else "rebuild",
                                  checkpoint=checkpoint, live=True)}
        self.emit(self.harbor_job(name, run, settings, count, prepare=[self.script(run["base"]["token"])[1]]),
                  run=run_id, step=step, purpose="probe", checkpoint=checkpoint)
        entry["jobs"].append(name)

    def _make_checkpoint(self, run_id, run, step):
        values, entry = self.settings, self._entry(run, step)
        ready, failed = self._checkpoint(entry)
        if ready or failed or self.pending(entry["checkpoints"]) \
                or len(entry["checkpoints"]) >= values["checkpoint_attempts"]:
            return
        name = jobs.job_name(values["prefix"], run["key"], f"s{step}", f"c{len(entry['checkpoints']) + 1}")
        settings = {**run["settings"], **values["ak"],
                    **jobs.branch(run["script"], step, mode="rebuild", checkpoint=name, live=False)}
        self.emit(self.harbor_job(name, run, settings, 1, prepare=[self.script(run["base"]["token"])[1]]),
                  run=run_id, step=step, purpose="checkpoint", checkpoint=name)
        entry["checkpoints"].append(name)

    # Reporting -------------------------------------------------------------

    def summary(self):
        """Each run's curve, decisive step, best rewind points and spend."""
        values, out = self.settings, {}
        for run_id, run in sorted(self.runs.items()):
            base = {"task": run["task"], "base": run["base"]}
            if run.get("skipped"):
                out[run_id] = {**base, "status": "skipped", "skipped": run["skipped"]}
                continue
            found = search.result(run["steps"], self.probes(run), v0=self.v0(run["task"]), k=values["k"],
                                  extra=values["extra"], scan=run["scan"])
            states, checkpoints = {}, {}
            for key, entry in run["probes"].items():
                ready, _ = self._checkpoint(entry)
                if ready:
                    checkpoints[key] = ready
                rewards = [summary["reward"] for _, summary in self.results(entry["checkpoints"])
                           if summary["reward"] is not None]
                if rewards:
                    states[int(key)] = rewards[0]
            for point in found["curve"]:
                if point["step"] in states:
                    point["state_reward"] = states[point["step"]]
            names = [name for entry in run["probes"].values() for name in (*entry["jobs"], *entry["checkpoints"])]
            trials = [summary for _, summary in self.results(names)]
            out[run_id] = {
                **base, **found, "rules": run["rules"], "features": run["features"], "checkpoints": checkpoints,
                "state_rewards": {str(step): reward for step, reward in sorted(states.items())},
                "spend": {"usd": round(math.fsum(t["cost_usd"] or 0.0 for t in trials), 6),
                          "seconds": round(math.fsum(t["seconds"] or 0.0 for t in trials), 1),
                          "trials": len(trials), "branches": len(self.results([n for e in run["probes"].values() for n in e["jobs"]])),
                          "not_usable": sum(1 for t in trials if not records.usable(t)),
                          "jobs": len(names), "jobs_pending": sum(1 for n in names if not self.jobs[n]["complete"])},
            }
        return out
