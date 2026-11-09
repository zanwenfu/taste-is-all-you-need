"""Each episode's final state checked against one requirement of its task, the one the task text says was unmet.

On rejected passing work, keeping going beats starting over by the hidden tests. Where the task text says the reviewer
was right, the tests cannot say whether keeping going kept the gap the reviewer named or a restart closed it: the
restarts' final code was not kept (ISSTA plan, item 13; review of draft 3, finding 3). Here the same checker (the
recovery state's model, effort, settings and route, unless restated at creation) checks final states again, told the
base run's task with no final message and no submission, and one requirement of it: the passage ``requirements`` gives
for the run, quoted from the task text by its readers. Its verdict and evidence are on that requirement alone:

- every ended episode of the arms in ``arms`` (by default keep going and start over, told the reviewer's reasons): the
  final state of the trial that ended it, which is the episode's outcome;
- with ``check_base``, each run's base state too, the work the reviewer rejected.

Each state comes from its trial's own save where this host keeps it (``kept_saves``; the recovery state's
``keep_final_saves`` keeps them from its prune), else, with ``rebuild``, it is rebuilt from the record and checked
against it first, as the recovery's checks were; one that cannot be is left out. The checker is the study's own,
unchanged: only its task text names the requirement (``requirement_task``). The recovery state is only read.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from taste.agents.checker import checker_task
from taste.recovery_study import jobs
from taste.recovery_study.driver import own_final
from taste.recovery_study.recheck_driver import DEFAULTS as RECHECK_DEFAULTS
from taste.recovery_study.recheck_driver import RecheckDriver
from taste.recovery_study.recovery_driver import RecoveryDriver, _slug, new_check

# The checker's task text ends with this line; the requirement goes just before it.
CLOSING = "Gather your evidence in the container, then call submit_verdict."
REQUIREMENT_TEMPLATE = """\
One requirement of the task is in question:
<requirement>
{requirement}
</requirement>
Check this requirement alone. Your verdict is on it: "done" if the work meets it, "not_done" if a check you ran \
shows it unmet; for not_done, the unmet requirement is this one. The task's other requirements do not change the \
verdict.
"""
# A requirement is a short passage of the task text, not the task.
REQUIREMENT_CHARS = 2000

DEFAULTS = {
    **RECHECK_DEFAULTS,
    "prefix": "req",
    "requirements": "",     # JSON: {"<run>": "<requirement>"}, or a list of {"run", "requirement"}
    "arms": ["continue", "retry_feedback"],
    "control": [],
    "control_sample": 0,
    "check_base": False,    # also check each run's base state, the work the reviewer rejected
}


def requirements(path):
    """Each run's requirement, from a JSON file: ``{"<run>": "<text>"}`` or a list of ``{"run", "requirement"}``."""
    raw = json.loads(Path(path).read_text())
    items = raw.items() if isinstance(raw, dict) else ((item.get("run"), item.get("requirement")) for item in raw)
    found = {}
    for run_id, text in items:
        if not isinstance(run_id, str) or not isinstance(text, str) or not text.strip():
            raise ValueError("a requirement is a run's name and the requirement's text")
        if len(text) > REQUIREMENT_CHARS:
            raise ValueError(f"{run_id}: a requirement is a passage of the task, at most {REQUIREMENT_CHARS} characters")
        if run_id in found:
            raise ValueError(f"{run_id}: one requirement per run")
        found[run_id] = text.strip()
    return found


def requirement_task(text, requirement):
    """The checker's task text (``checker_task``) with one requirement named before its closing line."""
    if not text.endswith("\n" + CLOSING):
        raise ValueError("the checker's task text no longer ends with its closing line")
    return text[:-len(CLOSING)] + REQUIREMENT_TEMPLATE.format(requirement=requirement) + "\n" + CLOSING


def _quoted(requirement, task):
    """Whether the requirement is a passage of the task text, spacing aside."""
    return " ".join(requirement.split()) in " ".join(task.split())


class RequirementDriver(RecheckDriver):
    KIND = "taste.recovery_requirements/1"
    DEFAULTS = DEFAULTS

    @classmethod
    def validate(cls, settings):
        super().validate(settings)
        if not settings["requirements"] or not Path(settings["requirements"]).is_file():
            raise ValueError("requirements must name the file of each run's requirement")
        if not settings["arms"] and not settings["check_base"]:
            raise ValueError("nothing to check: name arms, or check_base")
        if not isinstance(settings["check_base"], bool):
            raise ValueError("check_base must be true or false")

    def checker_text(self, run, view, paths):
        return requirement_task(checker_task(self.original_task(run), "", "", changed_paths=paths),
                                run["requirement"])

    # The checks -------------------------------------------------------------

    def build(self, theirs):
        """One check for the final state of every ended episode of ``arms``, and with ``check_base`` for every base
        state, of the runs the requirements name."""
        wanted = requirements(self.settings["requirements"])
        unknown = sorted(set(wanted) - set(theirs["runs"]))
        if unknown:
            raise ValueError("requirements name runs the recovery state does not hold: " + ", ".join(unknown))
        running = sorted(f"{run_id}:{name}" for run_id in wanted
                         for name, arm in (theirs["runs"][run_id].get("arms") or {}).items()
                         if arm["recovery"] in self.settings["arms"] and not all(e["end"] for e in arm["episodes"]))
        if running:
            raise ValueError("these arms have episodes still running; finish the recovery state first: "
                             + ", ".join(running))
        self.data["items"] = {}
        for run_id in sorted(wanted):
            run = theirs["runs"][run_id]
            if run.get("skipped"):
                continue
            self._take_run(theirs, run_id, wanted[run_id])
            if self.settings["check_base"]:
                self._add_base(theirs, run_id)
            for name, arm in sorted((run.get("arms") or {}).items()):
                if arm["recovery"] not in self.settings["arms"]:
                    continue
                for index, episode in enumerate(arm["episodes"]):
                    last = RecoveryDriver._last_round(episode)
                    if last:
                        self._add_episode(theirs, run_id, name, arm["recovery"], index, episode, last[0])

    def _take_run(self, theirs, run_id, requirement):
        run = copy.deepcopy({key: raw for key, raw in theirs["runs"][run_id].items() if key != "arms"})
        self.runs[run_id] = {**run, "requirement": requirement}
        self.runs[run_id]["requirement_in_task"] = _quoted(requirement, self.original_task(self.runs[run_id]))

    def _take_jobs(self, theirs, names):
        for job in names:
            self.jobs.setdefault(job, {**copy.deepcopy(theirs["jobs"][job]), "taken_from": self.settings["source"]})

    def _item(self, item_id, own, kept, first, **fields):
        """An item whose state is its own save where kept, else its first check's saves where kept."""
        check = {**new_check(), "checkpoints": [job for job in first["checkpoints"] if job in kept]}
        if own and own[0] in kept:
            check["own"] = list(own)
        self.items[item_id] = {**fields, "first": list(first["verdict"]) if first.get("verdict") else None,
                               "first_saves": ([own[0]] if own else []) + list(first["checkpoints"]),
                               "check": check, "verdict": None,
                               "end": None if kept or self.settings["rebuild"] else "save_gone"}

    def _add_episode(self, theirs, run_id, name, recovery, index, episode, last):
        first, number = last["check"], episode["rounds"].index(last) + 1
        self._take_jobs(theirs, (last["agent"][0], *([first["verdict"][0]] if first["verdict"] else []),
                                 *first["checkpoints"]))
        own, kept = self._saves(theirs, last)
        self._item(f"{run_id}:{name}:e{index + 1}", own, kept, first, run=run_id, arm=name, recovery=recovery,
                   episode=index, round=number, base=False, ended=episode["end"], agent=list(last["agent"]),
                   label=[_slug(name), f"e{index + 1}r{number}"])

    def _add_base(self, theirs, run_id):
        run = theirs["runs"][run_id]
        first = run.get("check") or new_check()
        self._take_jobs(theirs, (*([first["verdict"][0]] if first.get("verdict") else []), *first["checkpoints"]))
        base = run["base"]
        own = first.get("own")
        if own is None and base.get("final_saved") and base.get("final_checkpoint"):
            own = [self._save_name(True, base["final_checkpoint"], base["token"]), base.get("final_changed_paths")]
        kept = [save for save in ([own[0]] if own else []) + list(first["checkpoints"]) if save in self.kept()]
        self._item(f"{run_id}:base", own, kept, first, run=run_id, arm="base", recovery="base", episode=None,
                   round=None, base=True, ended=None, agent=None, label=["base"])

    def _source(self, item):
        """The checked trial: its token, steps and settings, and its own save."""
        if item["base"]:
            run = self.runs[item["run"]]
            return {"token": run["base"]["token"], "steps": run["steps"], "settings": run["settings"],
                    **{key: run["base"].get(key) for key in ("final_saved", "final_checkpoint", "final_changed_paths")}}
        agent = self.trial(item["agent"])
        return {"token": agent["token"], "steps": agent["steps"], "settings": agent["settings"], **own_final(agent)}

    def advance(self):
        before = len(self.new)
        for item_id, item in self.items.items():
            if item["verdict"] or item["end"]:
                continue
            check = item["check"]
            if not check["jobs"] and not check.get("own") and not self.pending(check["checkpoints"]) \
                    and self.final_state(check) is None \
                    and len(self.counted(check["checkpoints"])) >= self.settings["check_attempts"]:
                # Not one rebuild came out faithful: what is left to check is not the state that ended the episode.
                item["end"] = "no_faithful_state"
                continue
            found = self._check(item["run"], self.runs[item["run"]], check, self._source(item), tuple(item["label"]),
                                requirement=item_id)
            if found is False:
                item["end"] = "check_failed"
            elif found:
                item["verdict"] = found
        return self.new[before:]

    # Reporting -------------------------------------------------------------

    def summary(self):
        """Per check: the state checked (an episode's outcome or a base state) and what the hidden tests said of it,
        the requirement, the checker's verdict on it with its evidence, the first check's verdict on the same state
        if it had one, and how the state was brought back ("same save", "rebuilt", "rebuilt in the check")."""
        out = {}
        for item_id, item in self.items.items():
            run = self.runs[item["run"]]
            reply = self.trial(item["verdict"])["checker"] if item["verdict"] else None
            first = self.trial(item["first"])["checker"] if item["first"] else None
            graded = run["base"] if item["base"] else self.trial(item["agent"])
            change = graded.get("final_change") or {}
            where = None
            if item["verdict"]:
                settings = jobs.JobSpec.from_dict(self.jobs[item["verdict"][0]]["spec"]).settings()
                restored = settings.get(jobs.SETTING["checkpoint"]) if settings[jobs.SETTING["mode"]] == "restore" \
                    else None
                where = "same save" if restored in item["first_saves"] else "rebuilt" if restored \
                    else "rebuilt in the check"
            out[item_id] = {
                **{key: item[key] for key in ("run", "arm", "recovery", "episode", "round", "base", "ended", "end")},
                "solved": graded.get("reward") == 1.0, "first_verdict": first and first["verdict"],
                "requirement": run["requirement"], "requirement_in_task": run["requirement_in_task"],
                "verdict": reply and reply["verdict"], "confidence": reply and reply.get("confidence"),
                "unmet_requirement": reply and reply.get("unmet_requirement"),
                "evidence": reply and reply.get("evidence"), "state": where,
                "final_change": {key: change.get(key) for key in ("diff_chars", "cut", "head_moved")} if change else None,
                "check_usd": round(math.fsum(summary["cost_usd"] or 0.0
                                             for _, summary in self.results(item["check"]["jobs"])), 6)}
        return out


def verdicts(summary):
    """The checks by arm: how many, how many have a verdict, met (done), unmet (not_done), and those left out."""
    found = {}
    for row in summary.values():
        arm = found.setdefault(row["recovery"], {"checks": 0, "checked": 0, "met": 0, "unmet": 0, "left_out": 0})
        arm["checks"] += 1
        arm["checked"] += row["verdict"] is not None
        arm["met"] += row["verdict"] == "done"
        arm["unmet"] += row["verdict"] == "not_done"
        arm["left_out"] += row["end"] is not None
    return found
