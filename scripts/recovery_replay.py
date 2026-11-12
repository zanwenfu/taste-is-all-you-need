"""Rebuild each episode's final code from its record, with no model call, and read it as a git diff.

    sudo python3 scripts/recovery_replay.py --state /data/replay/i13.json --source /data/study/n7/cpa1-rec.json \\
        --env SECRETS=/root/taste-secrets-dry/dry-anthropic.env [--records-root /data/taste-trials] \\
        [--arms continue,retry_feedback] [--base true] [--only-runs a,b | --only-runs @runs.txt] \\
        [--episodes @ids.txt | --sample 5 --sample-seed 0] [--tries 3] [--final-change-chars 60000] \\
        [--save-states true] [--drop name,...] [--ak NAME=VALUE] [--prefix i13p] [--jobs-dir DIR] \\
        [--launcher /data/replay/next.sh | --spool /data/study/spool] [--max-trials 24] [--loop 30] \\
        [--json /data/replay/i13-summary.json] [--tokens-out /data/replay/i13-tokens.txt] [--dry-run]
    python3 scripts/recovery_replay.py --tables /data/replay/i13.json [other replay states ...]

For each ended episode of the arms (and with --base true each run's base state), the trial that ended it is rebuilt
in a fresh task container from its replay script, up to its last step, no model asked, the rebuilt files saved as a
checkpoint and read as a diff, and checked: faithful, the same hidden-test grade, the same changed files, mode-only
lines ignored; up to --tries times (``taste.recovery_study.replay_driver``). The records must be restored first
(``scripts/restore_trial_records.py`` with the tokens --tokens-out writes). The environment must name SECRETS, a route
file of made-up keys. The recovery state is only read.

--dry-run walks every stage of every item (records, export, the first rebuild's command, the checks, and with
--requirements the requirement each run's check would be told) and prints it; it starts nothing and writes no state.

Use as for the recoveries: run the launcher (or let the host's scheduler take the spool), wait, run this again; with
--loop SECONDS it runs again by itself until every item has ended. Exit status: 0 new jobs were written; 2 none, but
jobs are still running; 3 every item has ended. Prints the loss table by model and arm.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste.recovery_study import replay_checks, state
from taste.recovery_study.driver import (
    MAX_TRIALS,
    RUNNING,
    STARTED,
    finish,
    harbor_unit_active,
    pairs,
    progress,
)
from taste.recovery_study.replay_driver import ReplayDriver


def truth(raw):
    """A true/false flag's value as given (true, false, 1, 0)."""
    value = str(raw).strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise argparse.ArgumentTypeError("give true or false")
    return value in {"true", "1"}


def listed(raw):
    """A comma-separated list, or @FILE with one entry per line; None when the flag is absent."""
    if raw is None:
        return None
    if raw.startswith("@"):
        return [line.strip() for line in Path(raw[1:]).read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")]
    return [name for name in raw.split(",") if name]


OPTIONS = {"prefix": str, "source": str, "records_root": str, "base": truth, "tries": int,
           "final_change_chars": int, "save_states": truth, "sample": int, "sample_seed": int, "jobs_dir": str,
           "trials_root": str, "work_dir": str, "tasks_dir": str, "run_harbor": str, "model": str}
LISTS = ("arms", "only_runs", "episodes", "drop")


def tables(paths):
    """The loss table of these replay states together."""
    rows = []
    for path in paths:
        data = state.load(path, ReplayDriver.KIND)
        if data is None:
            raise SystemExit(f"{path}: no replay state")
        rows += ReplayDriver(path, data).summary().values()
    return replay_checks.loss_text(rows)


def once(arguments, given):
    """One run of the driver: read, decide, write the jobs; (exit status, driver)."""
    with state.locked(arguments.state):
        driver = ReplayDriver.open(arguments.state, given)
        for name in arguments.close:
            driver.close(name)
        driver.refresh(unit_active=harbor_unit_active)
        driver.advance()
        status = finish(driver, arguments.launcher, arguments.max_trials,
                        f"replays {arguments.state}: {len(driver.new)} new jobs, {state.now()}", spool=arguments.spool)
    if arguments.json:
        arguments.json.write_text(json.dumps(driver.summary(), indent=1, sort_keys=True) + "\n")
    if arguments.tokens_out:
        arguments.tokens_out.write_text("".join(token + "\n" for token in driver.tokens()))
    return status, driver


def main(argv=None, *, sleep=time.sleep, run=subprocess.run):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", type=Path, help="the replays' state file (JSON)")
    for name, kind in OPTIONS.items():
        parser.add_argument("--" + name.replace("_", "-"), dest=name, type=kind, default=None)
    for name in LISTS:
        parser.add_argument("--" + name.replace("_", "-"), dest=name, help="comma-separated, or @FILE")
    parser.add_argument("--ak", action="append", help="NAME=VALUE: a setting every rebuild trial adds or overrides")
    parser.add_argument("--env", action="append", help="NAME=VALUE: environment for run-harbor.sh (SECRETS)")
    parser.add_argument("--close", action="append", default=[], help="take this job as finished with what it has")
    parser.add_argument("--launcher", type=Path, help="write the jobs to start to this shell script")
    parser.add_argument("--spool", type=Path, help="queue the jobs to start here, for the host's scheduler")
    parser.add_argument("--max-trials", type=int, default=MAX_TRIALS,
                        help="the launcher's limit on the running trials (each running job's -n)")
    parser.add_argument("--loop", type=float, default=0, help="run again every SECONDS until every item has ended")
    parser.add_argument("--json", type=Path, help="also write every item's summary here")
    parser.add_argument("--tokens-out", type=Path, help="write the trial tokens whose records are needed here")
    parser.add_argument("--dry-run", action="store_true", help="print every stage; start and write nothing")
    parser.add_argument("--requirements", type=Path, help="with --dry-run: each run's requirement, as the "
                                                           "requirement check takes them")
    parser.add_argument("--tables", type=Path, nargs="+", help="print the loss table of these replay states")
    arguments = parser.parse_args(argv)
    if arguments.tables:
        print(tables(arguments.tables), end="")
        return 0
    if arguments.state is None:
        parser.error("--state is required")
    given = {name: getattr(arguments, name) for name in OPTIONS}
    given.update({name: listed(getattr(arguments, name)) for name in LISTS})
    given.update(ak=pairs(arguments.ak), env=pairs(arguments.env))
    if arguments.dry_run:
        if arguments.launcher or arguments.spool or arguments.loop or arguments.close:
            parser.error("a dry run starts nothing: no --launcher, --spool, --loop or --close")
        driver = ReplayDriver.open(arguments.state, given)  # not saved
        wanted = None
        if arguments.requirements:
            from taste.recovery_study.requirement_driver import requirements
            wanted = requirements(arguments.requirements)
        lines, counts = driver.walk(wanted)
        print("\n".join(lines))
        print("\ndry run, nothing started or written: " + json.dumps(counts))
        if arguments.tokens_out:
            arguments.tokens_out.write_text("".join(token + "\n" for token in driver.tokens()))
            print(f"tokens to restore: {len(driver.tokens())} -> {arguments.tokens_out}")
        return 0
    if arguments.loop and not (arguments.launcher or arguments.spool):
        parser.error("--loop needs --launcher or --spool to start the jobs")
    while True:
        status, driver = once(arguments, given)
        print(replay_checks.loss_text(driver.summary().values()), end="")
        cost = driver.model_cost()
        print(f"\n{progress(driver, arguments.launcher, arguments.spool)} Model cost of the rebuilds: ${cost:.6f}"
              + (" -- NOT ZERO: a rebuild asked a model; stop and find why." if cost else ""))
        if not arguments.launcher and not arguments.spool:
            for spec in driver.new:
                print(spec.shell())
        if not arguments.loop or status not in (STARTED, RUNNING) or cost:
            return status
        if arguments.launcher and status == STARTED:
            run(["sh", str(arguments.launcher)], check=False)
        else:
            sleep(arguments.loop)
        given = {}  # the stored settings stand


if __name__ == "__main__":
    raise SystemExit(main())
