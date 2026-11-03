"""Check a recovery state's judged final states again, the checker told the original task alone.

    sudo python3 scripts/recovery_recheck.py --state /root/study/blind.json --source /root/study/rec.json \\
        --kept-saves /root/study/kept-saves.txt [--prefix blind1] \\
        [--arms retry_feedback,continue,continue_both] [--control retry] [--control-sample 200] [--sample-seed 0] \\
        [--rebuild true|false] [--checker-env SECRETS=/root/taste-secrets/bedrock.env] \\
        [--launcher /root/study/blind-next.sh | --spool /root/study/spool] [--max-trials 24] \\
        [--json /root/study/blind-summary.json]

The kept-saves file lists the checkpoint IDs under the host's checkpoints root, one per line
(``ls /var/lib/taste-checkpoints``). A new state takes the recovery state's checker settings (model, effort, settings,
route); those given here stand over them, each such change recorded as restated
(``taste.recovery_study.recheck_driver``). The recovery state is only read.

Use as for the recoveries: run the launcher, wait, run this again. Exit status: 0 new jobs were written; 2 none, but
jobs are still running; 3 every re-check is finished.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste.recovery_study import state
from taste.recovery_study.driver import MAX_TRIALS, finish, harbor_unit_active, pairs, progress
from taste.recovery_study.recheck_driver import RecheckDriver

OPTIONS = {"prefix": str, "source": str, "kept_saves": str, "control_sample": int, "sample_seed": int,
           "rebuild": lambda raw: truth(raw), "check_attempts": int, "concurrent": int, "work_dir": str}


def truth(raw):
    """A true/false flag's value as given (true, false, 1, 0)."""
    value = str(raw).strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise argparse.ArgumentTypeError("give true or false")
    return value in {"true", "1"}


def names(raw):
    return None if raw is None else [name for name in raw.split(",") if name]


def table(summary):
    """Per arm: the re-checks, those with a blind verdict, and how the blind verdicts stand against the first."""
    lines = ["| Arm | Re-checks | Checked | Same verdict | done -> not_done | not_done -> done | Left out |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    arms = sorted({row["recovery"] for row in summary.values()})
    for arm in arms:
        rows = [row for row in summary.values() if row["recovery"] == arm]
        moved = Counter((row["informed"], row["blind"]) for row in rows if row["blind"])
        ends = Counter(row["end"] for row in rows if row["end"])
        same = sum(count for (first, blind), count in moved.items() if first == blind)
        left = ", ".join(f"{end} {count}" for end, count in sorted(ends.items())) or "0"
        lines.append(f"| {arm} | {len(rows)} | {sum(moved.values())} | {same} | {moved['done', 'not_done']} | "
                     f"{moved['not_done', 'done']} | {left} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", required=True, type=Path, help="the re-check's state file (JSON)")
    for name, kind in OPTIONS.items():
        parser.add_argument("--" + name.replace("_", "-"), dest=name, type=kind, default=None)
    parser.add_argument("--arms", help="comma-separated: the arms whose every judged round is checked again")
    parser.add_argument("--control", help="comma-separated: the arms whose judged rounds are sampled as the control")
    parser.add_argument("--checker-env", action="append",
                        help="NAME=VALUE: environment every checker job adds (the reviewer's model route)")
    parser.add_argument("--env", action="append", help="NAME=VALUE: environment for run-harbor.sh")
    parser.add_argument("--close", action="append", default=[], help="take this job as finished with what it has")
    parser.add_argument("--launcher", type=Path, help="write the jobs to start to this shell script")
    parser.add_argument("--spool", type=Path, help="queue the jobs to start here, for the host's scheduler")
    parser.add_argument("--max-trials", type=int, default=MAX_TRIALS,
                        help="the launcher's limit on the running trials (each running job's -n)")
    parser.add_argument("--json", type=Path, help="also write every re-check's summary here")
    arguments = parser.parse_args(argv)
    given = {name: getattr(arguments, name) for name in OPTIONS}
    given.update(arms=names(arguments.arms), control=names(arguments.control),
                 checker_env=pairs(arguments.checker_env), env=pairs(arguments.env))
    with state.locked(arguments.state):
        driver = RecheckDriver.open(arguments.state, given)
        for name in arguments.close:
            driver.close(name)
        driver.refresh(unit_active=harbor_unit_active)
        driver.advance()
        status = finish(driver, arguments.launcher, arguments.max_trials,
                        f"blind re-checks {arguments.state}: {len(driver.new)} new jobs, {state.now()}",
                        spool=arguments.spool)
        summary = driver.summary()
    if arguments.json:
        arguments.json.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(table(summary), end="")
    print("\n" + progress(driver, arguments.launcher, arguments.spool))
    if not arguments.launcher and not arguments.spool:
        for spec in driver.new:
            print(spec.shell())
    return status


if __name__ == "__main__":
    raise SystemExit(main())
