"""Start the jobs the host's drivers queued, oldest first, within the host's trial limit.

    sudo python3 scripts/spool_scheduler.py --spool /data/study/spool [--spool DIR ...] \\
        [--max-trials 24] [--interval 15]

Drivers queue jobs (``--spool`` of scripts/recovery_map.py and recovery_runs.py,
``taste.recovery_study.jobs.spool``): one file per job, named by when it was
first queued. This one process starts them in that order across every spool
given, so a large job is never passed by later small ones, and only while the
host's running Harbor trials (each running job's -n, the study's and any
other's) leave room for the job's own; when nothing runs, the oldest starts
whatever its size. A job already started is dropped from the queue. Each job
starts holding the host's launch lock, which the drivers' launchers also take.
Run it as root, as a systemd unit; it runs until stopped.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from taste.recovery_study.driver import MAX_TRIALS
from taste.recovery_study.jobs import UNIT_PREFIX

LOCK = Path("/run/lock/taste-harbor-launch.lock")


@dataclass
class Host:
    trials: Callable[[], int]           # the host's running Harbor trials
    started: Callable[[dict], bool]     # whether a queued job has started
    run: Callable[[dict], bool]         # start a queued job


def entries(spools):
    """The queued jobs of every spool, oldest first."""
    return sorted((path for directory in spools for path in Path(directory).glob("*.job")), key=lambda path: path.name)


def step(spools, max_trials, host):
    """Start the oldest queued job if it fits, dropping those already started; its name, or None."""
    for path in entries(spools):
        try:
            entry = json.loads(path.read_text())
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            continue
        if host.started(entry):
            path.unlink(missing_ok=True)
            continue
        now = host.trials()
        if now and now + int(entry["trials"]) > max_trials:
            return None
        # Taken before it starts: a start that fails is queued again by its driver, not retried here.
        path.unlink(missing_ok=True)
        host.run(entry)
        return entry["name"]
    return None


def systemctl(*arguments):
    return subprocess.run(["systemctl", *arguments], capture_output=True, text=True, timeout=60).stdout


def host_trials(systemctl=systemctl):
    """Every running Harbor job on the host, each counted as its -n (1 if it names none)."""
    total = 0
    for line in systemctl("list-units", "--plain", "--no-legend", "--state=active,activating",
                          UNIT_PREFIX + "*").splitlines():
        if not line.split():
            continue
        found = re.search(r" -n (\d+)", systemctl("show", "-p", "ExecStart", "--value", line.split()[0]))
        total += int(found.group(1)) if found else 1
    return total


def real_host(log):
    def started(entry):
        return ((Path(entry["jobs_dir"]) / entry["name"]).exists()
                or subprocess.run(["systemctl", "is-active", "--quiet", UNIT_PREFIX + entry["name"]]).returncode == 0)

    def run(entry):
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        with open(LOCK, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            done = subprocess.run(["sh", "-c", entry["command"]], capture_output=True, text=True)
        log(f"{'started' if done.returncode == 0 else 'could not start'} {entry['name']} "
            f"({entry['trials']} trials): {(done.stdout + done.stderr).strip().splitlines()[-1:] or ''}")
        return done.returncode == 0

    return Host(trials=host_trials, started=started, run=run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spool", action="append", type=Path, required=True, help="a spool the drivers queue into")
    parser.add_argument("--max-trials", type=int, default=MAX_TRIALS, help="the host's limit on running trials")
    parser.add_argument("--interval", type=float, default=15.0, help="seconds between looks when nothing can start")
    arguments = parser.parse_args(argv)

    def log(message):
        print(time.strftime("%H:%M:%S"), message, flush=True)

    host = real_host(log)
    while True:
        if step(arguments.spool, arguments.max_trials, host) is None:
            time.sleep(arguments.interval)


if __name__ == "__main__":
    sys.exit(main())
