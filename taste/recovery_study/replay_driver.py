"""Each episode's final code rebuilt from its record, with no model call, and read as a git diff.

The restarts' final code was not kept (the container saves went with a deleted disk), but every trial's record holds
its commands in order with their outputs (``taste.benchmarks.replay_export``). Here the trial that ended each episode
of a finished recovery state (its outcome) is rebuilt: its replay script exported from the restored record
(``taste.recovery_study.archive_restore``), then run again in a fresh task container up to its last step, the agent's
context answered from the record and no model asked (``branch_live`` off: the Oct 20 re-grade's and the recovery
driver's own final-state job). Each rebuild trial then

- saves the rebuilt files as a checkpoint named by its job (``save_states``), which a later check restores, and whose
  manifest lists the files it changed;
- reads the final change against the task's base commit (``final_change_chars``: git's diff and changed paths,
  ``taste.benchmarks.final_change``);
- is graded by the task's hidden tests.

A rebuild stands for its episode ("kept") only when it is faithful (every recorded output came back) and passes the
three checks of ``replay_checks``: the same grade as recorded, the same changed files where the record lists them,
git's mode-only lines not counted as changes. One that does not is tried again, up to ``tries`` times; an episode
whose record cannot be rebuilt is lost, and the loss is tabled by model and arm (``replay_checks.losses``). A trial
whose record stops in the middle of a command (``dropped_steps``) is rebuilt to the last whole step and ends
"short_record": its rebuilt state is not its final one.

The items are the ended episodes of the arms in ``arms`` (by default keep going and start over, told the reviewer's
reasons), and with ``base`` each run's base state (the work the reviewer rejected), of the runs in ``only_runs`` (all
by default), narrowed to the ids in ``episodes`` or a seeded ``sample`` spread over the arms. An item's id is the
requirement driver's: ``<run>:<arm>:e<N>`` or ``<run>:base``. The recovery state is only read; the records are read
from ``records_root`` (by default the recovery state's trials root, where the restore puts them).

The rebuild trials' settings are the recorded trial's, without its caps (a rebuild needs no budget and gets the
task's own time), without its task suffix (a replayed agent is given its record's task), without anything a job sets
itself (``jobs.OWN``) and without the names in ``drop``; then ``ak``. The environment must name a route file
(``SECRETS``) of made-up keys, never the recovery state's own: no model can be billed even by a fault.

State (``kind`` taste.recovery_replays/1), beside the shared fields of ``driver``:

    items.<id>      {"run", "arm", "recovery", "episode", "round", "base", "ended", "label", "trial", "token",
                     "steps", "model", "task", "tasks_dir", "settings",
                     "recorded": {"reward", "image_id", "changed_paths", "changed_from"},
                     "script": {"path", "sha256", "steps", "dropped_steps", "workdir", "rebuildable"} or None,
                     "step", "short", "tries": [job], "results": [one per finished try], "end": None or why}
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import shlex
import sqlite3
from pathlib import Path

from taste.benchmarks import final_change
from taste.recovery_study import archive_restore, jobs, replay_checks, state
from taste.recovery_study.driver import COMMON, Driver
from taste.recovery_study.recovery_driver import MESSAGE, RecoveryDriver, _slug

# The recovery state's settings a replay takes: the same model, trials root and jobs folder. Not its export command:
# that names the code folder of the host it ran on.
INHERITED = ("model", "trials_root", "jobs_dir")
# The export command of this code, wherever it is installed (the driver writes each script itself first; the job's
# own export only runs if that script is gone).
EXPORT_TEMPLATE = f"PYTHONPATH={shlex.quote(str(Path(__file__).resolve().parents[2]))} {jobs.EXPORT_TEMPLATE}"
ITEM = re.compile(r"^(?P<run>.+):(?:(?P<arm>[^:]+):e(?P<episode>[1-9][0-9]*)|base)$")
MAX_TRIES = 6

DEFAULTS = {
    **COMMON,
    "prefix": "replay",
    "source": "",               # the finished recovery state whose episodes are rebuilt; only read
    "records_root": "",         # where the recorded trials' records are (the restore's --out); "": the source's
    "arms": ["continue", "retry_feedback"],
    "base": False,              # also each run's base state, the work the reviewer rejected
    "only_runs": [],            # the runs taken; empty: all
    "episodes": [],             # the items taken, by id; empty: every item of the runs and arms
    "sample": 0,                # 0: all; else this many items, seeded, spread over the arms
    "sample_seed": 0,
    "tries": 3,                 # rebuilds of one item at most, until one is kept
    "final_change_chars": 60000,  # each rebuild reads its final change, kept up to this many characters; 0: not read
    "save_states": True,        # each rebuild saves its files as a checkpoint named by its job
    "drop": [],                 # settings left out of the recorded trial's (an older host's own)
    "export_template": EXPORT_TEMPLATE,
}


def export_record(record):
    """A trial's replay script, exported from its record (``taste.benchmarks.replay_export``): (bytes, script)."""
    from taste.benchmarks.replay_export import encode_script, export_script, trial_files

    journal, ledger, parent, trial = trial_files(record)
    script = export_script(journal, ledger=ledger, parent=parent, trial=trial)
    return encode_script(script), script


def _write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def _route(settings):
    return (settings.get("env") or {}).get("SECRETS")


class ReplayDriver(Driver):
    KIND = "taste.recovery_replays/1"
    DEFAULTS = DEFAULTS

    @classmethod
    def open(cls, path, given=None):
        """A replay state; a new one takes the recovery state's model, trials root and jobs folder (those given
        stand) and its items."""
        given = {name: raw for name, raw in (given or {}).items() if raw is not None}
        if state.load(path, cls.KIND) is not None:
            return super().open(path, given)
        source = given.get("source")
        theirs = state.load(source, RecoveryDriver.KIND) if source and Path(source).is_file() else None
        if theirs is None:
            raise ValueError("source must name the finished recovery state whose episodes are rebuilt")
        first = {name: theirs["settings"][name] for name in INHERITED if name in theirs["settings"]}
        if not given.get("records_root"):
            given["records_root"] = theirs["settings"].get("trials_root") or COMMON["trials_root"]
        routes = {_route(theirs["settings"]), (theirs["settings"].get("checker_env") or {}).get("SECRETS")}
        if (given.get("env") or {}).get("SECRETS") in routes - {None}:
            raise ValueError("env SECRETS is the recovery state's own model route: give a file of made-up keys")
        driver = super().open(path, {**first, **given})
        driver.data["source_sha256"] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        driver.build(theirs)
        return driver

    @classmethod
    def validate(cls, settings):
        super().validate(settings)
        if not settings["source"]:
            raise ValueError("source must name the finished recovery state whose episodes are rebuilt")
        if not _route(settings):
            raise ValueError("env must name SECRETS, a route file of made-up keys: a rebuild asks no model")
        unknown = sorted(set(settings["arms"]) - set(MESSAGE))
        if unknown:
            raise ValueError("unknown arms: " + ", ".join(unknown))
        if not settings["arms"] and not settings["base"]:
            raise ValueError("nothing to rebuild: name arms, or base")
        for name in ("base", "save_states"):
            if not isinstance(settings[name], bool):
                raise ValueError(f"{name} must be true or false")
        for name in ("sample", "sample_seed"):
            if isinstance(settings[name], bool) or not isinstance(settings[name], int) or settings[name] < 0:
                raise ValueError(f"{name} must be a whole number, 0 or more")
        if isinstance(settings["tries"], bool) or not isinstance(settings["tries"], int) \
                or not 1 <= settings["tries"] <= MAX_TRIES:
            raise ValueError(f"tries must be 1 to {MAX_TRIES}")
        chars = settings["final_change_chars"]
        if isinstance(chars, bool) or not isinstance(chars, int) or (
                chars and not final_change.LEAST_CHARS <= chars <= final_change.MOST_CHARS):
            raise ValueError(f"final_change_chars must be 0 (not read) or "
                             f"{final_change.LEAST_CHARS:,}-{final_change.MOST_CHARS:,}")
        for item in settings["episodes"]:
            if not ITEM.match(str(item)):
                raise ValueError(f"not an item id (<run>:<arm>:e<N> or <run>:base): {item!r}")
        if settings["episodes"] and settings["sample"]:
            raise ValueError("name the episodes or a sample, not both")

    @property
    def items(self):
        return self.data["items"]

    # The items -------------------------------------------------------------

    def build(self, theirs):
        """One item for the trial that ended each ended episode of ``arms`` (and with ``base`` each base state) of
        the runs asked for, narrowed to ``episodes`` or a ``sample``."""
        values = self.settings
        unknown = sorted(set(values["only_runs"]) - set(theirs["runs"]))
        if unknown:
            raise ValueError("only_runs names runs the recovery state does not hold: " + ", ".join(unknown))
        taken = sorted(values["only_runs"] or theirs["runs"])
        running = sorted(f"{run_id}:{name}" for run_id in taken
                         for name, arm in (theirs["runs"][run_id].get("arms") or {}).items()
                         if arm["recovery"] in values["arms"] and not all(e["end"] for e in arm["episodes"]))
        if running:
            raise ValueError("these arms have episodes still running; finish the recovery state first: "
                             + ", ".join(running))
        found = {}
        self.data["base_tokens"] = {}
        for run_id in taken:
            run = theirs["runs"][run_id]
            if run.get("skipped"):
                continue
            self.data["base_tokens"][run_id] = run["base"].get("token")
            if values["base"]:
                found[f"{run_id}:base"] = self._base(theirs, run_id, run)
            for name, arm in sorted((run.get("arms") or {}).items()):
                if arm["recovery"] not in values["arms"]:
                    continue
                for index, episode in enumerate(arm["episodes"]):
                    found[f"{run_id}:{name}:e{index + 1}"] = self._episode(theirs, run_id, run, name, arm, index,
                                                                           episode)
        if values["episodes"]:
            missing = sorted(set(values["episodes"]) - set(found))
            if missing:
                raise ValueError("these items are not in the recovery state: " + ", ".join(missing))
            found = {item_id: found[item_id] for item_id in values["episodes"]}
        elif values["sample"]:
            found = {item_id: found[item_id] for item_id in self._sample(found)}
        self.data["items"] = dict(sorted(found.items()))

    def _sample(self, found):
        """``sample`` items, seeded, taken from each arm in turn so that every arm has its share."""
        by_arm = {}
        for item_id, item in sorted(found.items()):
            by_arm.setdefault(item["recovery"], []).append(item_id)
        rng = random.Random(self.settings["sample_seed"])
        for ids in by_arm.values():
            rng.shuffle(ids)
        picked, arms = [], sorted(by_arm)
        while len(picked) < self.settings["sample"] and any(by_arm.values()):
            for arm in arms:
                if by_arm[arm] and len(picked) < self.settings["sample"]:
                    picked.append(by_arm[arm].pop(0))
        return picked

    @staticmethod
    def _summary(theirs, reference):
        job, trial = reference
        return next(summary for summary in theirs["jobs"][job]["results"] if summary["trial"] == trial)

    @staticmethod
    def _check_save(theirs, check):
        """The changed paths of a save a study's check rebuilt of this state: faithful, named by its job and whole (a
        save over its size cap lists only part of what changed)."""
        for name in (check or {}).get("checkpoints") or ():
            job = theirs["jobs"].get(name) or {}
            for summary in job.get("results") or ():
                if summary.get("faithful") is not False and summary.get("checkpoint") == name \
                        and not summary.get("checkpoint_partial") and isinstance(summary.get("changed_paths"), list):
                    return summary["changed_paths"]
        return None

    def _recorded(self, theirs, reward, own, check, image=None, *, whole=False):
        """What the record says of the state: its grade, its image, and the files it changed, from the trial's own
        save when that saved whole (``whole``), else from a save the study's check rebuilt of it, else none."""
        if whole and isinstance(own, list):
            listed, where = own, "own save"
        else:
            listed = self._check_save(theirs, check)
            where = "check save" if listed is not None else None
        return {"reward": reward, "image_id": image, "changed_paths": listed, "changed_from": where}

    def _item(self, run_id, run, **fields):
        return {"run": run_id, "task": run["task"], "tasks_dir": run.get("tasks_dir"), "script": None, "step": None,
                "short": None, "tries": [], "results": [], "end": None, **fields}

    def _base(self, theirs, run_id, run):
        base = run["base"]
        return self._item(run_id, run, arm="base", recovery="base", episode=None, round=None, base=True, ended=None,
                          label=[run["key"], "base"], trial=[None, base.get("trial")], token=base.get("token"),
                          steps=run.get("steps", base.get("steps")), model=run.get("model"),
                          settings=dict(run.get("settings") or {}),
                          recorded=self._recorded(theirs, base.get("reward"), base.get("final_changed_paths"),
                                                  run.get("check"), whole=bool(base.get("final_saved"))))

    def _episode(self, theirs, run_id, run, name, arm, index, episode):
        fields = {"arm": name, "recovery": arm["recovery"], "episode": index, "base": False, "ended": episode["end"],
                  "label": [run["key"], _slug(name), f"e{index + 1}"]}
        last = RecoveryDriver._last_round(episode)
        if not last:
            return self._item(run_id, run, **fields, round=None, trial=None, token=None, steps=None, model=None,
                              settings={}, recorded=self._recorded(theirs, None, None, None)) | {"end": "no_trial"}
        last = last[0]
        summary = self._summary(theirs, last["agent"])
        return self._item(run_id, run, **fields, round=episode["rounds"].index(last) + 1, trial=list(last["agent"]),
                          token=summary.get("token"), steps=summary.get("steps"), model=summary.get("model"),
                          settings=dict(summary.get("settings") or {}),
                          recorded=self._recorded(theirs, summary.get("reward"), summary.get("final_changed_paths"),
                                                  last.get("check"), summary.get("image_id"),
                                                  whole=bool(summary.get("final_saved"))))

    def tokens(self):
        """The trials whose records the replays and their checks read: each item's, and the base trial of each run
        an item is of (whose task text a requirement check is told)."""
        found = [item["token"] for item in self.items.values() if item["token"]]
        bases = self.data.get("base_tokens", {})
        found += [bases[run_id] for run_id in sorted({item["run"] for item in self.items.values()})
                  if bases.get(run_id)]
        return list(dict.fromkeys(found))

    # The rebuilds ------------------------------------------------------------

    def records_root(self):
        return Path(self.settings["records_root"])

    def script_path(self, token):
        return self.work_dir() / "scripts" / f"{token}.json"

    def _read_script(self, item, *, write=True):
        """The item's replay script: read where written, else exported from its record (and written, unless
        ``write`` is false). (script info, None) or (None, (end, why))."""
        path, record = self.script_path(item["token"]), self.records_root() / str(item["token"])
        if path.is_file():
            raw = path.read_bytes()
            script = json.loads(raw)
        else:
            if not record.is_dir():
                return None, ("no_record", f"no record at {record}")
            try:
                raw, script = export_record(record)
            except (ValueError, OSError, KeyError, sqlite3.Error) as error:
                return None, ("export_failed", f"{type(error).__name__}: {str(error)[:300]}")
            if write:
                _write(path, raw)
        source = script.get("source") or {}
        steps = script.get("steps") or []
        runs = [run for step in steps for run in step.get("runs") or ()] + list(script.get("setup") or ())
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "steps": len(steps),
                "dropped_steps": int(source.get("dropped_steps") or 0), "workdir": source.get("workdir"),
                "rebuildable": all(run.get("executed") is not None for run in runs)}, None

    def _place(self, item, script):
        """The step a rebuild ends at: the trial's last, or the last whole step its record holds."""
        if not script["rebuildable"]:
            return "not_rebuildable", "the record has no exact command texts (no terminal ledger)"
        if not item["steps"]:
            return "no_steps", "the trial took no step"
        item["step"] = min(int(item["steps"]), script["steps"])
        item["short"] = item["step"] < int(item["steps"]) or script["dropped_steps"] > 0
        return None, None

    def spec(self, item, number):
        """The job of an item's ``number``-th rebuild."""
        values = self.settings
        name = jobs.job_name(values["prefix"], *item["label"], f"t{number}")
        drop = jobs.CAPS | {jobs.SETTING["suffix"]} | set(values["drop"])
        settings = {**jobs.inherited(item["settings"], drop=drop), **values["ak"],
                    **jobs.branch(item["script"]["path"], item["step"], mode="rebuild",
                                  checkpoint=name if values["save_states"] else None, live=False)}
        if values["final_change_chars"]:
            settings["final_change_chars"] = values["final_change_chars"]
        record = self.records_root() / str(item["token"])
        prepare = [jobs.export_script(record, item["script"]["path"], values["export_template"])]
        return jobs.harbor_job(name, tasks_dir=values["tasks_dir"] or item["tasks_dir"], task=item["task"],
                               attempts=1, settings=settings, model=values["model"], jobs_dir=values["jobs_dir"],
                               run_harbor=values["run_harbor"], concurrent=1,
                               env={**values["env"], "TASTE_TRIALS_ROOT": values["trials_root"]}, prepare=prepare)

    def advance(self):
        """Start each item's next rebuild, or end it: kept, lost, or never rebuildable."""
        before = len(self.new)
        for item_id, item in self.items.items():
            if item["end"]:
                continue
            if item["script"] is None:
                script, ended = self._read_script(item)
                if ended is None:
                    item["script"] = script
                    ended = self._place(item, script)
                if ended[0]:
                    item["end"], item["why"] = ended
                    continue
            if self.pending(item["tries"]):
                continue
            self._read(item)
            rows = item["results"]
            if any(row["kept"] for row in rows):
                item["end"] = "kept"
            elif item["short"] and any(row["faithful"] for row in rows):
                item["end"] = "short_record"
            elif len(self.counted(item["tries"])) >= self.settings["tries"]:
                item["end"] = self._lost(rows)
            else:
                spec = self.spec(item, len(item["tries"]) + 1)
                self.emit(spec, item=item_id, purpose="replay", attempt=len(item["tries"]) + 1)
                item["tries"].append(spec.name)
        return self.new[before:]

    def _read(self, item):
        read = {row["job"] for row in item["results"]}
        for name in item["tries"]:
            job = self.jobs[name]
            if not job["complete"] or job.get("void") or name in read:
                continue
            found = job["results"] or []
            item["results"].append(self.judge(item, name, found[0]) if found else
                                   {"job": name, "trial": None, "faithful": False, "kept": False, "why": "no trial"})

    def judge(self, item, name, summary):
        """One finished rebuild against its record: faithful, graded alike, the same changed files."""
        faithful = summary.get("faithful") is True
        grade = replay_checks.grade(item["recorded"]["reward"], summary.get("reward"))
        if summary.get("checkpoint_partial"):
            files = {"judged": False, "equal": None, "why": "the rebuild's save is partial (over its size cap)"}
        else:
            files = replay_checks.changed_files(item["recorded"]["changed_paths"], summary.get("changed_paths"),
                                                item["script"]["workdir"])
        cost = summary.get("cost_usd") or 0.0
        if not faithful:
            why = "unfaithful"
        elif summary.get("reward") is None or summary.get("exception"):
            why = "not_graded"
        elif cost > 0:
            why = "model_cost"  # a rebuild asks no model: a cost means one was asked
        elif grade["equal"] is False:
            why = "grade_differs"
        elif files["judged"] and not files["equal"]:
            why = "files_differ"
        else:
            why = None
        recorded_image, image = item["recorded"]["image_id"], summary.get("image_id")
        partial = summary.get("checkpoint_partial")
        return {"job": name, "trial": summary.get("trial"), "token": summary.get("token"), "faithful": faithful,
                "unfaithful_at": summary.get("unfaithful_at"), "exception": summary.get("exception"),
                "reward": summary.get("reward"), "grade": grade, "files": files,
                "image": {"recorded": recorded_image, "rebuilt": image,
                          "same": None if not (recorded_image and image) else recorded_image == image},
                "checkpoint": summary.get("checkpoint") if summary.get("checkpoint") and not partial else None,
                "checkpoint_partial": partial, "changed_paths": summary.get("changed_paths"),
                "change": replay_checks.change_view(summary.get("final_change")),
                "prefix_seconds": summary.get("prefix_seconds"), "seconds": summary.get("seconds"),
                "cost_usd": summary.get("cost_usd"), "why": why, "kept": why is None and not item["short"]}

    @staticmethod
    def _lost(rows):
        """Why an item's tries are spent with none kept: the last faithful try's failed check, else unfaithful."""
        faithful = [row for row in rows if row["faithful"]]
        return faithful[-1]["why"] if faithful else "never_faithful"

    # Reporting -----------------------------------------------------------------

    def summary(self):
        """Per item: what was rebuilt and how it ended; the kept try's (else the last try's) checks, saved state and
        final change."""
        out = {}
        for item_id, item in self.items.items():
            rows = item["results"]
            chosen = next((row for row in rows if row["kept"]), rows[-1] if rows else None)
            script = item["script"] or {}
            recorded = item["recorded"]
            out[item_id] = {
                **{key: item[key] for key in ("run", "arm", "recovery", "episode", "round", "base", "ended", "task",
                                              "model", "token", "steps", "step", "short", "end")},
                "why": item.get("why"), "kept": item["end"] == "kept", "tries": len(item["tries"]),
                "recorded": {"reward": recorded["reward"], "changed_from": recorded["changed_from"],
                             "changed_paths": None if recorded["changed_paths"] is None
                             else len(recorded["changed_paths"])},
                "dropped_steps": script.get("dropped_steps"), "workdir": script.get("workdir"),
                "script_sha256": script.get("sha256"),
                **{key: (chosen or {}).get(key) for key in ("faithful", "unfaithful_at", "reward", "grade", "files",
                                                            "image", "checkpoint", "changed_paths", "change",
                                                            "cost_usd", "why")},
                "try_whys": [row["why"] for row in rows]}
            if chosen is None:
                out[item_id]["why"] = item.get("why")
        return out

    def model_cost(self):
        """Every rebuild's model dollars: 0 when no model was asked, as none should be."""
        return round(sum(row.get("cost_usd") or 0.0 for item in self.items.values() for row in item["results"]), 6)

    # The dry run ---------------------------------------------------------------

    def walk(self, requirements=None):
        """Every stage of every item, as it would run, with no container, no job and nothing written: the records
        to restore, the export, the first rebuild's command, the checks and the requirement check's input."""
        lines, counts = [], {}

        def count(key):
            counts[key] = counts.get(key, 0) + 1

        for item_id, stored in self.items.items():
            item = copy.deepcopy(stored)
            count(f"items {item['recovery']}")
            lines.append(f"[{item_id}] {item['recovery']}, episode ended {item['ended']}, trial {item['token']}, "
                         f"{item['steps']} steps, model {item['model']}")
            if item["end"]:
                lines.append(f"  ended before any stage: {item['end']}")
                count(f"ended {item['end']}")
                continue
            record = self.records_root() / str(item["token"])
            missing = archive_restore.on_disk(self.records_root(), [item["token"]]).get(item["token"])
            parent = (record / "branch-script.json").is_file()
            lines.append(f"  1 restore: {record}: " + ("whole" if not missing else "missing " + ", ".join(missing))
                         + (", with its branch script" if parent else ""))
            count("records whole" if not missing else "records missing")
            script, ended = self._read_script(item, write=False)
            if ended:
                lines.append(f"  2 export: {ended[0]}: {ended[1]}")
                count(f"export {ended[0]}")
                script = {"path": str(self.script_path(item["token"])), "sha256": None, "steps": item["steps"] or 0,
                          "dropped_steps": 0, "workdir": None, "rebuildable": True}
            else:
                lines.append(f"  2 export: {script['steps']} steps, dropped {script['dropped_steps']}, workdir "
                             f"{script['workdir']}, rebuildable {script['rebuildable']} -> {script['path']}")
                count("exported")
            item["script"] = script
            why = self._place(item, script)
            if why[0]:
                lines.append(f"  3 rebuild: none: {why[0]}: {why[1]}")
                count(f"rebuild {why[0]}")
                continue
            if item["short"]:
                count("short record")
            lines.append(f"  3 rebuild, try 1 of {self.settings['tries']} at step {item['step']}"
                         + (" (short record)" if item["short"] else "") + ":")
            lines.append("    " + self.spec(item, 1).shell())
            recorded = item["recorded"]
            listed = recorded["changed_paths"]
            lines.append(f"  4 checks: recorded grade {recorded['reward']}; changed files "
                         + (f"{len(listed)} listed ({recorded['changed_from']})" if listed is not None
                            else "not listed (not judged)") + f"; image {recorded['image_id'] or 'not recorded'}")
            if requirements is not None:
                text = requirements.get(item["run"])
                lines.append("  5 requirement: " + (repr(text[:120]) if text else "none named for this run"))
        return lines, dict(sorted(counts.items()))
