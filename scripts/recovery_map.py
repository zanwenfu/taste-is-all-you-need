"""Map where failed runs become unwinnable: branches from their steps, by binary search.

    sudo python3 scripts/recovery_map.py --state /root/study/map.json \\
        --base-job /root/study/jobs/base-luna --calibration /root/study/jobs/calib-luna \\
        [--prefix map1] [--model gpt-6-luna] [--k 8] [--extra 8] [--mode restore|rebuild] \\
        [--scan-every 5 --scan-fraction 0.2] [--jobs-dir /root/study/jobs] [--trials /var/lib/taste-trials] \\
        [--ak NAME=VALUE ...] [--env NAME=VALUE ...] [--launcher /root/study/map-next.sh] \\
        [--max-trials 24] [--json /root/study/map-summary.json]

Every failed trial of the base jobs is a run. Each time this is run it reads
the jobs that have finished, decides each run's next probes (K branches from
one step; K + extra on both sides of the boundary once found), records them in
the state file and writes them to the launcher. Run the launcher as root; it
starts the jobs, at most --max-trials of the study's trials at once, and returns when
none is running. Then run this again. Settings given when the state file is
first written are the study's; a later run may repeat them, not change them.

Per run it prints the status, the last winnable and first lost step, the
decisive step d, the best rewind points r* (highest v, and latest winnable)
and the spend; --json writes every run's curve with Wilson intervals.
Exit status: 0 new jobs were written; 2 none, but jobs are still running;
3 every run is finished.
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

OPTIONS = {"prefix": str, "model": str, "k": int, "extra": int, "mode": str, "checkpoint_attempts": int,
           "scan_every": int, "scan_fraction": float, "seed": str, "jobs_dir": str, "trials_root": str,
           "work_dir": str, "tasks_dir": str, "run_harbor": str, "export_template": str, "concurrent": int,
           "min_steps": int, "large_edit_lines": int, "attempts_factor": int}


def table(summary):
    lines = ["| Run | Task | Steps | Status | Low | High | d | r* (v) | r* (latest) | Probes | $ | Pending jobs |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run_id, run in summary.items():
        if run["status"] == "skipped":
            lines.append(f"| {run_id} | {run['task']} | | skipped: {run['skipped']} | | | | | | | | |")
            continue
        lines.append(f"| {run_id} | {run['task']} | {run['steps']} | {run['status']} | {run['low']} | {run['high']} | "
                     f"{'' if run['decisive_step'] is None else run['decisive_step']} | {run['rewind']['highest_v']} | "
                     f"{run['rewind']['latest']} | {len(run['curve']) - 1} | {run['spend']['usd']:.4f} | "
                     f"{run['spend']['jobs_pending']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", required=True, type=Path, help="the study's state file (JSON)")
    parser.add_argument("--base-job", action="append", default=[], help="a Harbor job of base runs")
    parser.add_argument("--calibration", action="append", default=[], help="a Harbor job of calibration runs")
    for name, kind in OPTIONS.items():
        flag = "--trials" if name == "trials_root" else "--" + name.replace("_", "-")
        parser.add_argument(flag, dest=name, type=kind, default=None)
    parser.add_argument("--no-prefetch", dest="prefetch", action="store_false", default=None,
                        help="make a probe's checkpoint only when the probe is chosen")
    parser.add_argument("--ak", action="append", help="NAME=VALUE: a setting every branch adds or overrides")
    parser.add_argument("--env", action="append", help="NAME=VALUE: environment for run-harbor.sh")
    parser.add_argument("--close", action="append", default=[], help="take this job as finished with what it has")
    parser.add_argument("--launcher", type=Path, help="write the jobs to start to this shell script")
    parser.add_argument("--max-trials", type=int, default=MAX_TRIALS,
                        help="the launcher's limit on the study's running trials (each running job's -n)")
    parser.add_argument("--json", type=Path, help="also write every run's summary here")
    arguments = parser.parse_args(argv)
    given = {name: getattr(arguments, name) for name in [*OPTIONS, "prefetch"]}
    given.update(ak=pairs(arguments.ak), env=pairs(arguments.env))
    with state.locked(arguments.state):
        driver = MapDriver.open(arguments.state, given)
        for name in arguments.close:
            driver.close(name)
        driver.add_sources(arguments.base_job, arguments.calibration)
        driver.refresh()
        driver.advance()
        status = finish(driver, arguments.launcher, arguments.max_trials,
                        f"recovery map {arguments.state}: {len(driver.new)} new jobs, {state.now()}")
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
