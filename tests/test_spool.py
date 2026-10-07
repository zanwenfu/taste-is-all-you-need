"""The host's job spool: drivers queue jobs, one scheduler starts them in order within the trial limit."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from taste.recovery_study import jobs

_SPEC = importlib.util.spec_from_file_location(
    "spool_scheduler", Path(__file__).resolve().parents[1] / "scripts" / "spool_scheduler.py")
scheduler = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scheduler  # its dataclass looks its module up there
_SPEC.loader.exec_module(scheduler)


def spec(name, trials, jobs_dir):
    return jobs.harbor_job(name, tasks_dir="/t", task="a", attempts=trials, settings={"agent": "x"},
                           model="gpt-6-luna", jobs_dir=str(jobs_dir), run_harbor="/r.sh", concurrent=trials)


def test_a_job_is_queued_once_keeps_its_place_and_is_not_queued_once_started(tmp_path):
    spool, jobs_dir = tmp_path / "spool", tmp_path / "jobs"
    first, second = spec("map-a-s3-b1", 8, jobs_dir), spec("rec-a-rt-f1", 3, jobs_dir)
    assert jobs.spool([first, second], spool) == 2
    names = sorted(path.name for path in spool.glob("*.job"))
    assert [name.split("-", 1)[1] for name in names] == ["map-a-s3-b1.job", "rec-a-rt-f1.job"]
    entry = json.loads((spool / names[0]).read_text())
    assert entry == {"name": "map-a-s3-b1", "trials": 8, "jobs_dir": str(jobs_dir), "command": first.shell()}
    # Queued again by the driver's next pass: nothing changes, the order stays.
    assert jobs.spool([second, first], spool) == 0
    assert sorted(path.name for path in spool.glob("*.job")) == names
    # A job whose directory exists has started: it is not queued again once the scheduler took it.
    (spool / names[1]).unlink()
    (jobs_dir / "rec-a-rt-f1").mkdir(parents=True)
    assert jobs.spool([second], spool) == 0 and not list(spool.glob("*rec-a-rt-f1.job"))


def test_the_scheduler_starts_the_oldest_job_when_it_fits_and_never_lets_a_later_one_pass(tmp_path):
    spool, jobs_dir = tmp_path / "spool", tmp_path / "jobs"
    jobs.spool([spec("map-a-s3-b1", 8, jobs_dir)], spool)
    jobs.spool([spec("rec-a-rt-f1", 1, jobs_dir)], spool)
    started, running = [], {"n": 20}

    def run(entry):
        started.append(entry["name"])
        running["n"] += entry["trials"]
        return True

    host = scheduler.Host(trials=lambda: running["n"], started=lambda entry: entry["name"] in started, run=run)
    # 20 running: the probe of 8 does not fit, and the later job of 1 that would fit waits behind it.
    assert scheduler.step([spool], 24, host) is None and started == []
    running["n"] = 15
    assert scheduler.step([spool], 24, host) == "map-a-s3-b1"
    # 23 running now: the job of 1 still fits.
    assert scheduler.step([spool], 24, host) == "rec-a-rt-f1"
    assert started == ["map-a-s3-b1", "rec-a-rt-f1"] and not list(spool.glob("*.job"))
    # Nothing runs: the oldest starts whatever its size.
    jobs.spool([spec("map-b-s5-b1", 30, jobs_dir)], spool)
    running["n"] = 0
    assert scheduler.step([spool], 24, host) == "map-b-s5-b1"


def test_the_scheduler_drops_a_job_that_already_started_and_reads_every_spool_in_order(tmp_path):
    first, second, jobs_dir = tmp_path / "map", tmp_path / "rec", tmp_path / "jobs"
    jobs.spool([spec("map-a-s3-b1", 8, jobs_dir)], first)
    jobs.spool([spec("rec-a-rt-f1", 1, jobs_dir)], second)
    started = []
    host = scheduler.Host(trials=lambda: 0, started=lambda entry: entry["name"] == "map-a-s3-b1",
                          run=lambda entry: started.append(entry["name"]) or True)
    # The map's job was queued first but has started elsewhere: dropped, and the next one runs.
    assert scheduler.step([second, first], 24, host) == "rec-a-rt-f1"
    assert started == ["rec-a-rt-f1"] and not list(first.glob("*.job")) and not list(second.glob("*.job"))


def test_host_trials_count_every_running_harbor_job_as_its_n():
    units = "taste-harbor-map-a.service loaded active running x\ntaste-harbor-cal.service loaded active running x\n"
    shown = {"taste-harbor-map-a.service": "{ argv[]=/h run -k 8 -n 8 ; }", "taste-harbor-cal.service": "{ argv[]=/h run -k 2 ; }"}

    def systemctl(*arguments):
        return units if arguments[0] == "list-units" else shown[arguments[-1]]

    assert scheduler.host_trials(systemctl) == 9
