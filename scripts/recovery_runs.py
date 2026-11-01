"""Recover failed runs the checker rejects, four ways at equal cost, round by round.

    sudo python3 scripts/recovery_runs.py --state /root/study/recoveries.json \\
        --base-job /root/study/jobs/base-luna --calibration-report /root/study/calibration.json \\
        [--map /root/study/map.json] [--reader /root/study/readings.json] \\
        [--recoveries retry,retry_feedback,rewind,continue] [--rewind-sources oracle,reader,rules] \\
        [--oracle highest_v|latest] [--repeats 3] [--prefix rec1] [--model gpt-6-luna] \\
        [--checker-model gpt-6-sol] [--checker-effort medium] [--mode restore|rebuild] \\
        [--budget-runs 2] [--checks-from /root/study/recoveries-first.json] \\
        [--first-only retry_bare,continue_bare] [--checker-env SECRETS=/root/taste-secrets/azure.env] \\
        [--jobs-dir /root/study/jobs] [--trials /var/lib/taste-trials] \\
        [--launcher /root/study/rec-next.sh] [--max-trials 24] [--json /root/study/rec-summary.json]

Every failed base trial that submitted is a run. A checker trial
(``taste.agents.checker``) checks its final state; a run it rejects is
recovered by plain retry, retry with feedback, rewind with feedback (to the
step each source chooses: the map's r* with --map, the trajectory reader's with
--reader, the rules') and continue with feedback (and the bare-feedback
ablation, ``continue_bare``, if listed), each --repeats times. Each recovery has
one base run's dollars and agent time: the task's median_usd and
median_agent_seconds in the calibration report (``scripts/calibration_report.py
--json``), which also says which tasks are kept; without one, the medians of
the --calibration jobs' runs. After each round a checker trial checks again
and, until it accepts or the budget is spent, the next round continues from
the submission with the new feedback; a plain retry's next round is another
fresh trial told nothing, unless --retry-rounds continue.

With --checks-from another recovery state (a second budget, --budget-runs),
each run's initial check is taken from that state, its verdict, saved final
state and checker trial, and none is made: both budgets recover the same
rejected runs. A run that state has not checked yet waits for it. On a host
that does not keep the checkpoints an earlier study saved (the map's, and those
checks), --prior-checkpoints rebuild rebuilds those states from the record.

The reader file is JSON: {"<run>": <reading>} or a list of readings with a
"run", a reading being ``read_trajectory``'s {"step", "reason", "confidence"};
the rewind is to the step before the one it names (``rewind_point``).

Use as for the map: run the launcher, wait, run this again. Exit status: 0 new
jobs were written; 2 none, but jobs are still running; 3 every run is finished.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste.benchmarks import branch_trial
from taste.recovery_study import state
from taste.recovery_study.driver import MAX_TRIALS, finish, harbor_unit_active, pairs, progress
from taste.recovery_study.map_driver import MapDriver
from taste.recovery_study.recovery_driver import RECOVERIES, SOURCES, RecoveryDriver, readings

OPTIONS = {"prefix": str, "model": str, "mode": str, "calibration_report": str, "checker_model": str,
           "checker_effort": str, "oracle": str, "repeats": int, "budget_runs": float, "check_attempts": int, "prefix_allowance": float,
           "min_round_usd": float, "min_round_share": float, "min_round_seconds": float, "retry_rounds": str, "jobs_dir": str,
           "trials_root": str, "work_dir": str, "tasks_dir": str, "run_harbor": str, "export_template": str,
           "concurrent": int, "min_steps": int, "large_edit_lines": int, "attempts_factor": int, "checks_from": str,
           "prior_checkpoints": str, "arm_order_seed": int, "tests_feedback_chars": int, "tests_code_chars": int,
           "tests_full_chars": int, "save_final": lambda raw: truth(raw), "max_active_runs": int}


def truth(raw):
    """A true/false flag's value as given (true, false, 1, 0)."""
    value = str(raw).strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise argparse.ArgumentTypeError("give true or false")
    return value in {"true", "1"}


def table(summary):
    lines = ["| Run | Task | Status | Budget $ / s | Rewind steps | Arm: episodes ended, solved, $ |",
             "| --- | --- | --- | --- | --- | --- |"]
    for run_id, run in summary.items():
        if run["status"] == "skipped":
            lines.append(f"| {run_id} | {run['task']} | skipped: {run['skipped']} | | | |")
            continue
        arms = []
        for name, arm in run["arms"].items():
            done = [item for item in arm["episodes"] if item]
            arms.append(f"{name}: {len(done)}/{len(arm['episodes'])}, {sum(1 for item in done if item['solved'])}, "
                        f"{sum(item['usd'] for item in done):.3f}")
        budget = run["budget"]
        lines.append(f"| {run_id} | {run['task']} | {run['status']} | {budget['usd']:.3f} / {budget['seconds']:.0f} | "
                     f"{json.dumps(run['sources'], sort_keys=True)} | {'; '.join(arms)} |")
    return "\n".join(lines) + "\n"


def prune(driver):
    """Delete the released saves' directories under the state's checkpoints root, each once (recorded as pruned)."""
    root = Path(driver.settings["env"].get("TASTE_CHECKPOINTS_ROOT") or branch_trial.CHECKPOINTS_ROOT)
    pruned = driver.data.setdefault("pruned", [])
    done = set(pruned)
    for name in driver.released_saves():
        if name in done:
            continue
        directory = branch_trial.checkpoint_directory(name, root)
        if directory.parent == root and directory.is_dir() and not directory.is_symlink():
            shutil.rmtree(directory)
        pruned.append(name)
    driver.save()
    print(f"saves pruned: {len(pruned) - len(done)} now, {len(pruned)} in all", file=sys.stderr)


def names(raw):
    """A comma-separated list as given: None when the flag is absent (the stored list stands), [] when it is empty."""
    return None if raw is None else [name for name in raw.split(",") if name]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", required=True, type=Path, help="the study's state file (JSON)")
    parser.add_argument("--base-job", action="append", default=[], help="a Harbor job of base runs")
    parser.add_argument("--calibration", action="append", default=[], help="a Harbor job of calibration runs")
    parser.add_argument("--map", type=Path, help="the map's state file, for the oracle's rewind points")
    parser.add_argument("--reader", type=Path, help="the trajectory reader's readings (JSON)")
    for name, kind in OPTIONS.items():
        flag = "--trials" if name == "trials_root" else "--" + name.replace("_", "-")
        parser.add_argument(flag, dest=name, type=kind, default=None)
    parser.add_argument("--recoveries", help="comma-separated: " + ", ".join(RECOVERIES) + " (\"\": none, a state that "
                        "only checks the base runs)")
    parser.add_argument("--rewind-sources", help="comma-separated: " + ", ".join(SOURCES))
    parser.add_argument("--first-only", help="comma-separated: arms whose episodes are round one only")
    parser.add_argument("--ak", action="append", help="NAME=VALUE: a setting every agent trial adds or overrides")
    parser.add_argument("--checker-ak", action="append", help="NAME=VALUE: a setting every checker trial adds")
    parser.add_argument("--checker-env", action="append",
                        help="NAME=VALUE: environment every checker job adds to --env (the reviewer's model route)")
    parser.add_argument("--env", action="append", help="NAME=VALUE: environment for run-harbor.sh")
    parser.add_argument("--close", action="append", default=[], help="take this job as finished with what it has")
    parser.add_argument("--launcher", type=Path, help="write the jobs to start to this shell script")
    parser.add_argument("--spool", type=Path, help="queue the jobs to start here, for the host's scheduler")
    parser.add_argument("--max-trials", type=int, default=MAX_TRIALS,
                        help="the launcher's limit on the study's running trials (each running job's -n)")
    parser.add_argument("--json", type=Path, help="also write every run's summary here")
    parser.add_argument("--prune-saves", action="store_true",
                        help="delete the saved final states nothing in this state will restore again "
                             "(released_saves), under its checkpoints root; each once")
    arguments = parser.parse_args(argv)
    given = {name: getattr(arguments, name) for name in OPTIONS}
    given.update(ak=pairs(arguments.ak), checker_ak=pairs(arguments.checker_ak), env=pairs(arguments.env),
                 checker_env=pairs(arguments.checker_env), recoveries=names(arguments.recoveries),
                 rewind_sources=names(arguments.rewind_sources), first_only=names(arguments.first_only))
    unknown = (set(given["recoveries"] or ()) | set(given["first_only"] or ())) - set(RECOVERIES)
    unknown |= set(given["rewind_sources"] or ()) - set(SOURCES)
    if unknown:
        parser.error("unknown recoveries or rewind sources: " + ", ".join(sorted(unknown)))
    oracle = MapDriver.open(arguments.map).summary() if arguments.map and arguments.map.exists() else {}
    with state.locked(arguments.state):
        driver = RecoveryDriver.open(arguments.state, given)
        driver.use(oracle=oracle, reader=readings(arguments.reader))
        for name in arguments.close:
            driver.close(name)
        driver.add_sources(arguments.base_job, arguments.calibration)
        driver.refresh(unit_active=harbor_unit_active)
        driver.advance()
        status = finish(driver, arguments.launcher, arguments.max_trials,
                        f"recoveries {arguments.state}: {len(driver.new)} new jobs, {state.now()}",
                        spool=arguments.spool)
        if arguments.prune_saves:
            prune(driver)
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
