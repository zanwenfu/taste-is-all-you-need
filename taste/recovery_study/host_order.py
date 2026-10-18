"""Trials a rebuild stopped at a cut-off directory walk, taken back, and the episodes they ended let go on.

Until the fidelity rule read the lines of a cut-off directory walk as the host's order (``branch_replay``'s
``host_ordered``), a rebuild stopped at one as at a difference in the files, before the trial's agent ran: an arm
whose round-one trials all stopped so ended ``agent_failed``, and a check whose final state could not be brought
back, ``check_failed``. ``spoiled`` reads a trial's own record (the replay script it was given and its terminal's
ledger) and compares each command its rebuild ran again, under the rule as it is now. ``reopen`` voids each
finished job whose trials were all spoiled so (``Driver.void``: no attempt and no results, kept on record) and
reopens the episodes those jobs ended, for the driver to start them again; ``move_code`` points the state's later
jobs at another installed copy of the code.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path

from taste.benchmarks import branch_trial
from taste.brains.branch_replay import ReplayScript, compare
from taste.recovery_study import records, state

REASON = ("its rebuild stopped at a cut-off directory walk that printed the lines this host lists first; the "
          "fidelity rule now compares such a walk by its exit code and line count (branch_replay.host_ordered)")
LEDGER = Path("controller") / "terminal" / "terminal.sqlite3"


def _ran(path):
    """What a trial's terminal ran, in order: (the text it ran, its output as the worker read it, code, how it ended)."""
    with contextlib.closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
        rows = connection.execute("SELECT payload, stdout, stderr, code, terminated FROM requests ORDER BY rowid")
        return [(json.loads(payload).get("command"),
                 (stdout or b"").decode("utf-8", errors="replace") + (stderr or b"").decode("utf-8", errors="replace"),
                 code, terminated)
                for payload, stdout, stderr, code, terminated in rows]


def spoiled(summary, trials_root):
    """Whether a trial's rebuild stopped only at a cut-off directory walk.

    True when its record shows a rebuild that stopped (``unfaithful`` for ``rebuild``), and every command the
    terminal ran for it is the record's command, in order, and matches the record under the rule as it is now, the
    last by ``host_ordered``. False whenever the record cannot show that.
    """
    token = summary.get("token")
    if summary.get("faithful") is not False or not token:
        return False
    account = branch_trial.replay_outcome(records.settled(token, trials_root))
    if (not isinstance(account, dict) or account.get("mode") != "rebuild"
            or (account.get("unfaithful") or {}).get("reason") != "rebuild"):
        return False
    root = Path(trials_root) / str(token)
    try:
        script = ReplayScript.from_bytes((root / branch_trial.BRANCH_SCRIPT).read_bytes())
        ran = _ran(root / LEDGER)
    except (OSError, ValueError, sqlite3.Error):
        return False
    step = account.get("step")
    if type(step) is not int or not 0 < step <= len(script.steps):
        return False
    commands = [*script.setup, *(run for item in script.steps[:step] for run in item.runs)]
    if not ran or len(ran) != (account.get("rebuild") or {}).get("commands") or len(ran) > len(commands):
        return False
    rows = []
    for run, (command, output, code, ended) in zip(commands, ran, strict=False):
        if command != run.executed or type(code) is not int:
            return False
        rows.append(compare(run, output, code, ended == "timeout"))
    return not any(row["divergent"] for row in rows) and bool(rows[-1].get("host_ordered"))


def reopen(driver, is_spoiled=None):
    """Void every finished job whose trials were all spoiled, and reopen the episodes they ended.

    A round-one job: the arm's episodes that ended ``agent_failed`` with no round are reopened. A check's job (a
    final state's save or the checker's trial): the episode it ended ``check_failed`` is reopened. A later round's
    job is only reported (``left``): its episode ended with the round taken off. Returns the job names voided, the
    episodes reopened as (run, arm, episode, where) and the jobs left. Running it again does nothing more.
    """
    is_spoiled = is_spoiled or (lambda summary: spoiled(summary, driver.settings["trials_root"]))

    def all_spoiled(name):
        job = driver.jobs[name]
        results = job.get("results") or []
        return job["complete"] and not job.get("void") and bool(results) and all(map(is_spoiled, results))

    voided, reopened = [], []
    for run_id, run in driver.runs.items():
        for arm_name, arm in (run.get("arms") or {}).items():
            first = [name for name in arm["first"] if all_spoiled(name)]
            for name in first:
                driver.void(name, REASON)
            voided += first
            for index, episode in enumerate(arm["episodes"]):
                if first and episode["end"] == "agent_failed" and not episode["rounds"]:
                    episode["end"] = None
                    reopened.append((run_id, arm_name, index, "first"))
                for number, current in enumerate(episode["rounds"]):
                    check = current["check"]
                    spoiling = [name for name in (*check["checkpoints"], *check["jobs"]) if all_spoiled(name)]
                    for name in spoiling:
                        driver.void(name, REASON)
                    voided += spoiling
                    if spoiling and episode["end"] == "check_failed" and number == len(episode["rounds"]) - 1:
                        episode["end"] = None
                        reopened.append((run_id, arm_name, index, "check"))
    # A later round's job: its episode may have ended with the round taken off, so it is found among the jobs.
    left = [name for name, job in driver.jobs.items() if job.get("purpose") == "round" and all_spoiled(name)]
    return {"voided": voided, "reopened": reopened, "left": left}


def move_code(driver, code):
    """Point the state's later jobs at the code installed at ``code``: its run-harbor.sh, and the export
    template's PYTHONPATH. Returns the settings changed, as {name: [before, after]}."""
    code = Path(code)
    harbor = code / "infra" / "azure" / "run-harbor.sh"
    if not harbor.is_file():
        raise ValueError(f"{code} holds no infra/azure/run-harbor.sh")
    before = driver.settings["run_harbor"]
    old = Path(before).parents[2]
    template = driver.settings["export_template"]
    if f"PYTHONPATH={old} " not in template:
        raise ValueError(f"the export template does not run the code at {old}")
    changes = {"run_harbor": [before, str(harbor)],
               "export_template": [template, template.replace(f"PYTHONPATH={old} ", f"PYTHONPATH={code} ")]}
    for name, (_, after) in changes.items():
        driver.settings[name] = after
    return changes


def amend(driver, what):
    """Keep on the state what was changed, and when: its ``amendments``."""
    driver.data.setdefault("amendments", []).append({"at": state.now(), **what})
