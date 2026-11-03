"""The blind re-check: the judged final states checked again, the checker told the original task alone.

A recovery's check is told what its agent was given and what it said. retry_feedback's task holds the reviewer's
earlier rejection, and every agent's final message and submission are its own account of its work, which after a
rejection may answer it. So the checker that ends an episode may lean on what it said before rather than on the work.
Here the same checker (the recovery state's model, effort, settings and route, unless restated at creation) checks the
same final states again, told the base run's task with no final message and no submission:

- every judged round of the arms in ``arms`` (by default those told the reviewer's reasons);
- a seeded sample of ``control_sample`` judged rounds of the ``control`` arms (by default plain retry, told nothing):
  what the missing account alone changes, and whatever else differs from the first checks, such as a restated route.

Each state comes from the same save where this host still keeps it (``kept_saves``: the checkpoint IDs under its
checkpoints root, one per line); else, with ``rebuild``, it is rebuilt from the record and checked against it first,
as the recovery's checks were, and one that cannot be is left out. The recovery state is only read.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from pathlib import Path

from taste.agents.checker import checker_task
from taste.benchmarks import branch_trial
from taste.recovery_study import jobs, records, state
from taste.recovery_study.driver import own_final
from taste.recovery_study.recovery_driver import (AGENTREWIND, DEFAULTS as RECOVERY_DEFAULTS, MESSAGE, NORESTORE,
                                                  RecoveryDriver, _slug, new_check)

# The arms told the reviewer's reasons, and the arm told nothing.
REASONS = ["retry_feedback", "rewind", "continue", "continue_both", AGENTREWIND, NORESTORE]
CONTROL = ["retry"]
# The recovery state's settings a re-check takes, so that the same checker checks the same states the same way.
INHERITED = ("model", "checker_model", "checker_effort", "checker_ak", "checker_env", "check_attempts", "mode",
             "trials_root", "jobs_dir", "run_harbor", "export_template", "env", "concurrent", "ak", "tasks_dir")

DEFAULTS = {
    **RECOVERY_DEFAULTS,
    "prefix": "blind",
    "source": "",           # the recovery state whose judged rounds are checked again
    "kept_saves": "",       # the checkpoint IDs this host keeps, one per line
    "arms": REASONS,
    "control": CONTROL,
    "control_sample": 200,
    "sample_seed": 0,
    "rebuild": True,        # rebuild a pruned save from the record; False: leave its round out
}


class RecheckDriver(RecoveryDriver):
    KIND = "taste.recovery_rechecks/1"
    DEFAULTS = DEFAULTS

    def __init__(self, path, data):
        super().__init__(path, data)
        self._kept, self._tasks = None, {}

    @classmethod
    def open(cls, path, given=None):
        """A re-check state; a new one takes the recovery state's checker settings (those given stand) and its
        judged rounds."""
        given = {name: raw for name, raw in (given or {}).items() if raw is not None}
        if state.load(path, cls.KIND) is not None:
            return super().open(path, given)
        source = given.get("source")
        theirs = state.load(source, RecoveryDriver.KIND) if source and Path(source).is_file() else None
        if theirs is None:
            raise ValueError("source must name the recovery state to re-check")
        first = {name: theirs["settings"][name] for name in INHERITED if name in theirs["settings"]}
        driver = super().open(path, {**first, **given})
        driver.data["source_sha256"] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        driver.data["restated"] = {name: {"first": first[name], "now": given[name]} for name in sorted(first)
                                   if name in given and given[name] != first[name]}
        driver.build(theirs)
        return driver

    @classmethod
    def validate(cls, settings):
        super().validate(settings)
        if not settings["source"]:
            raise ValueError("source must name the recovery state to re-check")
        if not settings["kept_saves"] or not Path(settings["kept_saves"]).is_file():
            raise ValueError("kept_saves must name the file of the checkpoint IDs this host keeps")
        # Every arm an episode can be in: the recoveries asked for, and the twins their forks make.
        unknown = sorted((set(settings["arms"]) | set(settings["control"])) - set(MESSAGE))
        if unknown:
            raise ValueError("unknown arms: " + ", ".join(unknown))
        if set(settings["arms"]) & set(settings["control"]):
            raise ValueError("an arm is either re-checked whole or sampled as a control, not both")
        for name in ("control_sample", "sample_seed"):
            if isinstance(settings[name], bool) or not isinstance(settings[name], int) or settings[name] < 0:
                raise ValueError(f"{name} must be a whole number, 0 or more")
        if not isinstance(settings["rebuild"], bool):
            raise ValueError("rebuild must be true or false")

    @property
    def items(self):
        return self.data["items"]

    def kept(self):
        if self._kept is None:
            self._kept = set(Path(self.settings["kept_saves"]).read_text().split())
        return self._kept

    def _saved(self, checkpoint):
        """A save the recovery state made is restored only where this host keeps it; this state's own always."""
        if checkpoint and not checkpoint.startswith(self.settings["prefix"] + "-") and checkpoint not in self.kept():
            return None
        return checkpoint

    def original_task(self, run):
        """The base run's task: what the plain retry is given, without anything a recovery added."""
        token = run["base"]["token"]
        if token not in self._tasks:
            view = records.agent_view(records.settled(token, self.settings["trials_root"]))
            self._tasks[token] = view["task"] if view else ""
        return self._tasks[token]

    def checker_text(self, run, view, paths):
        return checker_task(self.original_task(run), "", "", changed_paths=paths)

    # The re-checks ---------------------------------------------------------

    def build(self, theirs):
        """One re-check for every judged round of ``arms`` and for a seeded sample of those of ``control``."""
        values = self.settings
        whole, pool = [], []
        for run_id, run in sorted(theirs["runs"].items()):
            for name, arm in sorted((run.get("arms") or {}).items()):
                for index, episode in enumerate(arm["episodes"]):
                    for number, step in enumerate(episode["rounds"], 1):
                        if not (step.get("agent") and step["check"]["verdict"]):
                            continue
                        found = (run_id, name, arm["recovery"], index, number)
                        if arm["recovery"] in values["arms"]:
                            whole.append(found)
                        elif arm["recovery"] in values["control"] and (values["rebuild"] or self._saves(theirs, step)[1]):
                            # The control is a sample of the rounds that can be checked again.
                            pool.append(found)
        sample = random.Random(values["sample_seed"]).sample(pool, min(values["control_sample"], len(pool)))
        self.data["items"] = {}
        for found in whole + sorted(sample):
            self._add_item(theirs, *found, sample=found in sample)

    def _saves(self, theirs, step):
        """A judged round's own save as its first check restored it, or would have had this host kept it ([ID,
        changed paths] or None), and the saves of its final state this host keeps."""
        first, (name, trial) = step["check"], step["agent"]
        agent = next(summary for summary in theirs["jobs"][name]["results"] if summary["trial"] == trial)
        own, whole = first.get("own"), own_final(agent)
        if own is None and whole["final_saved"] and whole["final_checkpoint"]:
            own = [branch_trial.final_checkpoint_name(whole["final_checkpoint"], agent["token"]),
                   whole["final_changed_paths"]]
        return own, [save for save in ([own[0]] if own else []) + list(first["checkpoints"]) if save in self.kept()]

    def _add_item(self, theirs, run_id, name, recovery, index, number, *, sample):
        run = theirs["runs"][run_id]
        step = run["arms"][name]["episodes"][index]["rounds"][number - 1]
        first = step["check"]
        for job in (step["agent"][0], first["verdict"][0], *first["checkpoints"]):
            self.jobs.setdefault(job, {**copy.deepcopy(theirs["jobs"][job]), "taken_from": self.settings["source"]})
        self.runs.setdefault(run_id, copy.deepcopy({key: raw for key, raw in run.items() if key != "arms"}))
        own, kept = self._saves(theirs, step)
        check = {**new_check(), "checkpoints": [job for job in first["checkpoints"] if job in kept]}
        if own and own[0] in kept:
            check["own"] = list(own)
        told = records.agent_view(records.settled(self.trial(step["agent"])["token"], self.settings["trials_root"]))
        self.items[f"{run_id}:{name}:e{index + 1}r{number}"] = {
            "run": run_id, "arm": name, "recovery": recovery, "episode": index, "round": number, "sample": sample,
            "agent": list(step["agent"]), "informed": list(first["verdict"]),
            "label": [_slug(name), f"e{index + 1}r{number}"],
            "first_saves": ([own[0]] if own else []) + list(first["checkpoints"]),
            "told_task_differs": bool(told) and told["task"] != self.original_task(self.runs[run_id]),
            "check": check, "verdict": None, "end": None if kept or self.settings["rebuild"] else "save_gone"}

    def advance(self):
        before = len(self.new)
        for item_id, item in self.items.items():
            if item["verdict"] or item["end"]:
                continue
            check = item["check"]
            if not check["jobs"] and not check.get("own") and not self.pending(check["checkpoints"]) \
                    and self.final_state(check) is None \
                    and len(self.counted(check["checkpoints"])) >= self.settings["check_attempts"]:
                # Not one rebuild came out faithful: what is left to check is not the state that was judged.
                item["end"] = "no_faithful_state"
                continue
            agent = self.trial(item["agent"])
            source = {"token": agent["token"], "steps": agent["steps"], "settings": agent["settings"],
                      **own_final(agent)}
            found = self._check(item["run"], self.runs[item["run"]], check, source, tuple(item["label"]),
                                recheck=item_id)
            if found is False:
                item["end"] = "check_failed"
            elif found:
                item["verdict"] = found
        return self.new[before:]

    # Reporting -------------------------------------------------------------

    def summary(self):
        """Per re-check: its round, the hidden tests' verdict, the first check's and the blind one's, and the state
        checked: "same save" as the first check, "rebuilt" from the record and checked against it, "rebuilt in the
        check" (a partial save, as the first check did), or None before a verdict."""
        out = {}
        for item_id, item in self.items.items():
            first = self.trial(item["informed"])["checker"]
            blind = self.trial(item["verdict"])["checker"] if item["verdict"] else None
            where = None
            if item["verdict"]:
                settings = jobs.JobSpec.from_dict(self.jobs[item["verdict"][0]]["spec"]).settings()
                restored = settings.get(jobs.SETTING["checkpoint"]) if settings[jobs.SETTING["mode"]] == "restore" \
                    else None
                where = "same save" if restored in item["first_saves"] else "rebuilt" if restored \
                    else "rebuilt in the check"
            out[item_id] = {
                **{key: item[key] for key in ("run", "arm", "recovery", "episode", "round", "sample", "end",
                                              "told_task_differs")},
                "solved": self.trial(item["agent"])["solved"],
                "informed": first["verdict"], "informed_confidence": first.get("confidence"),
                "blind": blind and blind["verdict"], "blind_confidence": blind and blind.get("confidence"),
                "state": where,
                "check_usd": round(math.fsum(summary["cost_usd"] or 0.0
                                             for _, summary in self.results(item["check"]["jobs"])), 6)}
        return out
