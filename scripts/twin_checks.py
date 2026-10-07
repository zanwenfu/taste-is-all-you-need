"""Check again what a recovery study's checker checked, with another model, and score both.

    sudo python3 scripts/twin_checks.py emit --state /root/study/recoveries.json \\
        --model gpt-6-luna [--effort medium] [--prefix twin] --launcher /root/study/twin-launch.sh
    sudo python3 scripts/twin_checks.py report --state /root/study/recoveries.json [--prefix twin] [--json out.json]

emit: for every check job of the recovery driver's state that has finished,
a twin job: the same checker task on the same final state, given to --model
(and --effort); written to a launcher that keeps the host's running trials
within --max-trials, as the drivers' launchers do. Run it as root, again after
more checks have finished: a twin already started is skipped.

report: for every check with a twin that has a verdict, both verdicts beside
the hidden tests' grade of the trial checked; agreement, each model's
accuracy (a "done" for a trial the tests pass, "not_done" for one they fail),
its false "done"s and false "not_done"s, and its dollars and seconds per check.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from taste.recovery_study import jobs, records
from taste.recovery_study.driver import MAX_TRIALS


def checked(state):
    """{check job: the hidden tests' grade of the trial it checked}, over every check the state holds."""
    settings = state["settings"]
    found = {}

    def grade(job, trial):
        return records.trial(Path(settings["jobs_dir"]) / job / trial, settings["trials_root"])["reward"]

    for run in (state.get("runs") or {}).values():
        if run.get("skipped"):
            continue
        for job in (run.get("check") or {}).get("jobs") or ():
            found[job] = (run.get("base") or {}).get("reward")
        for arm in (run.get("arms") or {}).values():
            for episode in arm.get("episodes") or ():
                for step in episode.get("rounds") or ():
                    if step.get("agent"):
                        for job in step["check"].get("jobs") or ():
                            found[job] = grade(*step["agent"])
    return found


def _verdict(state, job):
    """(verdict, dollars, seconds) of a check job's trial with a verdict, or None."""
    directory = Path(state["settings"]["jobs_dir"]) / job
    for trial in sorted(directory.glob("*/result.json")):
        summary = records.trial(trial.parent, state["settings"]["trials_root"])
        reply = records.verdict(summary)
        if reply is not None:
            return reply["verdict"], summary["cost_usd"], summary["seconds"]
    return None


def _twin_name(state, job, prefix):
    original = state["settings"]["prefix"] + "-"
    if not job.startswith(original):
        raise ValueError(f"check job {job!r} does not carry the study's prefix")
    return prefix + "-" + job[len(original):]


def twins(state, *, model, effort, prefix):
    """A twin job for every check job that has a verdict: the same task and files, another model."""
    specs = []
    for job in checked(state):
        if job not in state["jobs"] or _verdict(state, job) is None:
            continue
        spec = jobs.JobSpec.from_dict(state["jobs"][job]["spec"])
        name = _twin_name(state, job, prefix)
        argv = list(spec.argv)
        argv[argv.index(job)] = name
        effort_at = [i for i in range(len(argv) - 1) if argv[i] == "--ak" and argv[i + 1].startswith("worker_effort=")]
        for i in effort_at:
            argv[i + 1] = f"worker_effort={effort}"
        if not effort_at:
            argv += ["--ak", f"worker_effort={effort}"]
        env = {**dict(spec.env), "MODEL": jobs.model_flag(model)}
        specs.append(jobs.JobSpec(name, spec.attempts, tuple(argv), tuple(sorted(env.items())), spec.prepare))
    return specs


def _side(rows, key):
    graded = [row for row in rows if row["reward"] is not None]
    right = sum(1 for row in graded if (row[key] == "done") == (row["reward"] == 1.0))
    return {"accuracy": right / len(graded) if graded else None,
            "false_done": sum(1 for row in graded if row[key] == "done" and row["reward"] != 1.0),
            "false_not_done": sum(1 for row in graded if row[key] == "not_done" and row["reward"] == 1.0),
            "usd_per_check": round(statistics.fmean(row[key + "_usd"] or 0.0 for row in rows), 6) if rows else None,
            "seconds_per_check": round(statistics.fmean(row[key + "_seconds"] or 0.0 for row in rows), 1)
            if rows else None}


def score(state, *, prefix):
    """Both verdicts per check, beside the grade of the trial checked; agreement and each side's record."""
    rows = []
    for job, reward in checked(state).items():
        original, twin = _verdict(state, job), _verdict(state, _twin_name(state, job, prefix))
        if original is None or twin is None:
            continue
        rows.append({"check": job, "reward": reward, "original": original[0], "original_usd": original[1],
                     "original_seconds": original[2], "twin": twin[0], "twin_usd": twin[1], "twin_seconds": twin[2]})
    return {"checks": len(rows),
            "agreement": sum(1 for row in rows if row["original"] == row["twin"]) / len(rows) if rows else None,
            "original": _side(rows, "original"), "twin": _side(rows, "twin"), "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("emit", "report"))
    parser.add_argument("--state", required=True, type=Path, help="the recovery driver's state file")
    parser.add_argument("--prefix", default="twin", help="the twin jobs' name prefix")
    parser.add_argument("--model", default="gpt-6-luna", help="the twin checker's model")
    parser.add_argument("--effort", default="medium", choices=jobs.EFFORTS)
    parser.add_argument("--launcher", type=Path, help="emit: write the twin jobs' launcher here")
    parser.add_argument("--max-trials", type=int, default=MAX_TRIALS, help="the host's limit on running trials")
    parser.add_argument("--json", type=Path, help="report: also write the result here")
    arguments = parser.parse_args(argv)
    state = json.loads(arguments.state.read_text())
    if arguments.action == "emit":
        if arguments.launcher is None:
            parser.error("emit needs --launcher")
        specs = twins(state, model=arguments.model, effort=arguments.effort, prefix=arguments.prefix)
        arguments.launcher.write_text(jobs.launcher(specs, prefix=arguments.prefix, max_trials=arguments.max_trials,
                                                    title=f"twin checks with {arguments.model} ({arguments.effort})"))
        arguments.launcher.chmod(0o755)
        print(f"{len(specs)} twin checks in {arguments.launcher}")
        return 0
    found = score(state, prefix=arguments.prefix)
    if arguments.json:
        arguments.json.write_text(json.dumps(found, indent=1) + "\n")

    def line(side):
        record = found[side]
        return (f"{side}: accuracy {record['accuracy'] if record['accuracy'] is None else round(record['accuracy'], 2)}, "
                f"false done {record['false_done']}, false not_done {record['false_not_done']}, "
                f"${record['usd_per_check']} and {record['seconds_per_check']} s per check")

    agreement = found["agreement"]
    print(f"{found['checks']} checks with both verdicts; agreement "
          f"{agreement if agreement is None else round(agreement, 2)}")
    print(line("original"))
    print(line("twin"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
