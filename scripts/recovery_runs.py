"""Recover failed runs the checker rejects, four ways at equal cost, round by round.

    sudo python3 scripts/recovery_runs.py --state /root/study/recoveries.json \\
        --base-job /root/study/jobs/base-luna --calibration-report /root/study/calibration.json \\
        [--map /root/study/map.json] [--reader /root/study/readings.json] \\
        [--recoveries retry,retry_feedback,rewind,continue] [--rewind-sources oracle,reader,rules] \\
        [--oracle highest_v|latest] [--repeats 3] [--prefix rec1] [--model gpt-6-luna] \\
        [--checker-model gpt-6-sol] [--checker-effort medium] [--mode restore|rebuild] \\
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

The reader file is JSON: {"<run>": <reading>} or a list of readings with a
"run", a reading being ``read_trajectory``'s {"step", "reason", "confidence"};
the rewind is to the step before the one it names (``rewind_point``).

Use as for the map: run the launcher, wait, run this again. Exit status: 0 new
jobs were written; 2 none, but jobs are still running; 3 every run is finished.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste.recovery_study import state
from taste.recovery_study.driver import MAX_TRIALS, finish, pairs, progress
from taste.recovery_study.map_driver import MapDriver
from taste.recovery_study.recovery_driver import RECOVERIES, SOURCES, RecoveryDriver, readings

OPTIONS = {"prefix": str, "model": str, "mode": str, "calibration_report": str, "checker_model": str,
           "checker_effort": str, "oracle": str, "repeats": int, "check_attempts": int, "prefix_allowance": float,
           "min_round_usd": float, "min_round_share": float, "min_round_seconds": float, "retry_rounds": str, "jobs_dir": str,
           "trials_root": str, "work_dir": str, "tasks_dir": str, "run_harbor": str, "export_template": str,
           "concurrent": int, "min_steps": int, "large_edit_lines": int, "attempts_factor": int}


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
    parser.add_argument("--recoveries", help="comma-separated: " + ", ".join(RECOVERIES))
    parser.add_argument("--rewind-sources", help="comma-separated: " + ", ".join(SOURCES))
    parser.add_argument("--ak", action="append", help="NAME=VALUE: a setting every agent trial adds or overrides")
    parser.add_argument("--checker-ak", action="append", help="NAME=VALUE: a setting every checker trial adds")
    parser.add_argument("--env", action="append", help="NAME=VALUE: environment for run-harbor.sh")
    parser.add_argument("--close", action="append", default=[], help="take this job as finished with what it has")
    parser.add_argument("--launcher", type=Path, help="write the jobs to start to this shell script")
    parser.add_argument("--max-trials", type=int, default=MAX_TRIALS,
                        help="the launcher's limit on the study's running trials (each running job's -n)")
    parser.add_argument("--json", type=Path, help="also write every run's summary here")
    arguments = parser.parse_args(argv)
    given = {name: getattr(arguments, name) for name in OPTIONS}
    given.update(ak=pairs(arguments.ak), checker_ak=pairs(arguments.checker_ak), env=pairs(arguments.env),
                 recoveries=arguments.recoveries.split(",") if arguments.recoveries else None,
                 rewind_sources=arguments.rewind_sources.split(",") if arguments.rewind_sources else None)
    unknown = set(given["recoveries"] or ()) - set(RECOVERIES)
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
        driver.refresh()
        driver.advance()
        status = finish(driver, arguments.launcher, arguments.max_trials,
                        f"recoveries {arguments.state}: {len(driver.new)} new jobs, {state.now()}")
        summary = driver.summary()
    if arguments.json:
        arguments.json.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(table(summary), end="")
    print("\n" + progress(driver, arguments.launcher))
    if not arguments.launcher:
        for spec in driver.new:
            print(spec.shell())
    return status


if __name__ == "__main__":
    raise SystemExit(main())
