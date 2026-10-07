"""The recovery study's jobs are run-harbor.sh command lines, built in one place."""

from __future__ import annotations

import contextlib
import os
import shlex
import subprocess

import pytest

from taste.recovery_study import jobs


def test_a_branch_probe_is_one_run_harbor_call_with_its_settings_in_order():
    settings = {"agent": "mini-swe-agent", "services": "none", "reply_reserve_seconds": "10",
                **jobs.branch("/w/scripts/tok1.json", 12, mode="rebuild", live=True)}
    spec = jobs.harbor_job("map-run-abc123-s12-b1", tasks_dir="/study/tasks", task="task-a", attempts=8,
                           settings=settings, model="gpt-6-luna", jobs_dir="/study/jobs",
                           run_harbor="/opt/taste/infra/azure/run-harbor.sh", concurrent=8,
                           prepare=[jobs.export_script("/var/lib/taste-trials/tok1", "/w/scripts/tok1.json")])
    assert list(spec.argv) == [
        "/opt/taste/infra/azure/run-harbor.sh", "map-run-abc123-s12-b1", "/study/tasks", "-i", "task-a",
        "-k", "8", "-n", "8", "--max-retries", "2", "--ak", "agent=mini-swe-agent", "--ak", "services=none",
        "--ak", "reply_reserve_seconds=10", "--ak", "branch=/w/scripts/tok1.json", "--ak", "branch_step=12",
        "--ak", "branch_mode=rebuild", "--ak", "branch_live=on"]
    assert dict(spec.env) == {"MODEL": "azure/gpt-6-luna", "JOBS": "/study/jobs"}
    assert spec.settings()["branch_step"] == "12"
    # The replay script is exported once, before the first job that reads it.
    assert spec.prepare == (("sh", "-c", "[ -e /w/scripts/tok1.json ] || python3 -m taste.benchmarks.replay_export "
                             "--record /var/lib/taste-trials/tok1 --out /w/scripts/tok1.json"),)
    line = spec.shell()
    assert line.startswith("sh -c ") and " && env JOBS=/study/jobs MODEL=azure/gpt-6-luna " in line
    assert shlex.split(line)[-2:] == ["--ak", "branch_live=on"]
    assert jobs.JobSpec.from_dict(spec.to_dict()) == spec


def test_the_branch_settings_cover_rebuild_restore_and_both_overrides():
    assert jobs.branch("s.json", 4, mode="rebuild", checkpoint="cp-1", live=False) == {
        "branch": "s.json", "branch_step": 4, "branch_mode": "rebuild", "branch_checkpoint": "cp-1",
        "branch_live": False}
    restore = jobs.branch("s.json", 4, mode="restore", checkpoint="cp-1", override="append", note="/n.txt")
    assert restore["branch_mode"] == "restore" and restore["branch_override"] == "append"
    assert restore["branch_note"] == "/n.txt" and jobs.value(restore["branch_live"]) == "on"
    with pytest.raises(ValueError, match="names the checkpoint"):
        jobs.branch("s.json", 4, mode="restore")
    with pytest.raises(ValueError, match="come together"):
        jobs.branch("s.json", 4, override="reject")
    with pytest.raises(ValueError, match="append or reject"):
        jobs.branch("s.json", 4, override="replace", note="/n.txt")
    with pytest.raises(ValueError, match="rebuild or restore"):
        jobs.branch("s.json", 4, mode="snapshot")
    with pytest.raises(ValueError, match="whole number"):
        jobs.branch("s.json", -1)


def test_caps_are_tastes_equal_cap_settings_and_must_be_positive():
    assert jobs.caps(0.4499999999, 400.9) == {"spend_cap_usd": 0.45, "agent_timeout_sec": 400}
    assert jobs.value(0.45) == "0.45" and jobs.value(2.0) == "2" and jobs.value(True) == "on"
    with pytest.raises(ValueError):
        jobs.caps(0.0, 600)
    with pytest.raises(ValueError):
        jobs.caps(1.0, 0.5)


def test_inherited_settings_drop_what_a_job_sets_itself_and_keep_a_task_suffix():
    trial = {"agent": "mini-swe-agent", "worker_python": "/venv/bin/python", "branch": "s.json", "branch_step": "3",
             "branch_mode": "restore", "branch_checkpoint": "c", "branch_live": "on", "branch_override": "append",
             "branch_note": "n", "task_text": "x.md", "task_suffix": "/notes/f.txt", "spend_cap_usd": "1.5"}
    assert jobs.inherited(trial) == {"agent": "mini-swe-agent", "task_suffix": "/notes/f.txt", "spend_cap_usd": "1.5"}
    assert jobs.inherited(trial, drop=jobs.CAPS) == {"agent": "mini-swe-agent", "task_suffix": "/notes/f.txt"}


def test_a_checker_trial_is_the_checker_alone_with_backstops_on_the_files_brought_back():
    files = jobs.branch("s.json", 9, mode="restore", checkpoint="cp-final", live=False)
    settings = jobs.checker_trial("/w/checker/c1.md", files, effort="high")
    assert settings == {"agent": "checker", "services": "none", "worker_max_calls": 35, "spend_cap_usd": 2.5,
                        "agent_timeout_sec": 1500, "worker_effort": "high", "branch": "s.json", "branch_step": 9,
                        "branch_mode": "restore", "branch_checkpoint": "cp-final", "branch_live": False,
                        "task_text": "/w/checker/c1.md"}
    spec = jobs.harbor_job("rec-x-check-c1", tasks_dir="/t", task="a", attempts=1, settings=settings,
                           model=jobs.CHECKER_MODEL, jobs_dir="/j")
    assert dict(spec.env)["MODEL"] == "azure/gpt-6-sol" and spec.settings()["branch_live"] == "off"
    with pytest.raises(ValueError, match="effort"):
        jobs.checker_trial("/x.md", files, effort="extreme")


def test_job_names_are_safe_for_run_harbor_and_bounded():
    key = jobs.run_key("django__django-12345__aB3x.Yz")
    assert key == "django__django-12345__aB3x_Yz-" + jobs.digest("django__django-12345__aB3x.Yz")
    name = jobs.job_name("map", key, "s12", "b1")
    assert name == f"map-{key}-s12-b1" and jobs.safe(name, len(name)) == name
    long = jobs.job_name("p" * 60, "q" * 60, "r" * 60)
    assert len(long) == jobs.NAME_LIMIT and jobs.safe(long, len(long)) == long
    with pytest.raises(ValueError, match="letters, digits"):
        jobs.harbor_job("bad name", tasks_dir="/t", task="a", attempts=1, settings={}, model="gpt-6-luna",
                        jobs_dir="/j")
    assert jobs.model_flag("azure/gpt-6-sol") == "azure/gpt-6-sol" and jobs.model_flag("gpt-6-sol") == "azure/gpt-6-sol"


def test_the_launcher_starts_each_job_once_within_the_trial_limit_and_then_waits():
    spec = jobs.harbor_job("map-a-s3-b1", tasks_dir="/t", task="a", attempts=2, settings={"agent": "x y"},
                           model="gpt-6-luna", jobs_dir="/study/jobs", run_harbor="/r.sh")
    text = jobs.launcher([spec], prefix="map", max_trials=24, title="round 1")
    assert text.startswith("#!/bin/sh\n# round 1\n")
    assert "MAX_TRIALS=${MAX_TRIALS:-24}" in text and "'taste-harbor-map-*'" in text
    # The limit is the host's: every Harbor job counts, this study's or not.
    assert "$(units 'taste-harbor-*')" in text
    start = next(line for line in text.splitlines() if line.startswith("start map-a-s3-b1 "))
    _, name, directory, trials, command = shlex.split(start)
    assert (name, directory, trials) == ("map-a-s3-b1", "/study/jobs", "2") and command == spec.shell()
    assert spec.concurrent == 2 and "'agent=x y'" in command
    assert text.rstrip().endswith("run the driver again\"")
    assert not any(line.startswith("start ") for line in jobs.launcher([], prefix="map", max_trials=24).splitlines())


def test_the_launcher_counts_a_running_job_as_the_trials_it_runs_at_once(tmp_path):
    # A probe of eight and a checkpoint of one weigh what they run, whichever round started them.
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "systemctl").write_text("""#!/bin/sh
case "$1" in
  list-units) printf 'taste-harbor-map-a-s3-b1.service loaded active running x\\n'
              case "$*" in *taste-harbor-\\**) printf 'taste-harbor-cal-w2.service loaded active running x\\n';; esac
              printf 'taste-harbor-map-b-s5-c1.service loaded active running x\\n';;
  show) case "$*" in
          *s3-b1*) echo '{ path=/h ; argv[]=/h run -p /t -i a -k 8 -n 8 --max-retries 2 --ak b=1 ; }';;
          *cal-w2*) echo '{ path=/h ; argv[]=/h run -p /t -k 2 -n 6 --max-retries 2 ; }';;
          *) echo '{ path=/h ; argv[]=/h run -p /t -i b -k 1 -n 1 --max-retries 2 ; }';;
        esac;;
esac
""")
    (fake / "systemctl").chmod(0o755)
    text = jobs.launcher([], prefix="map", max_trials=24)
    functions = text[:text.rindex("while [")]
    done = subprocess.run(["sh", "-c", functions + "trials\nrunning\n"], capture_output=True, text=True,
                          env={**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"}, check=True)
    # A calibration job of 6 counts toward the host's trials, not toward this study's jobs.
    assert done.stdout.split() == ["15", "2"]
    # WAIT=0 returns at once, with the study's jobs still running.
    returned = subprocess.run(["sh", "-c", text], capture_output=True, text=True, timeout=10,
                              env={**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "WAIT": "0"})
    assert returned.returncode == 0 and "run the driver again" not in returned.stdout


def test_a_job_starts_under_the_hosts_lock_only_when_it_fits(tmp_path):
    # Two launchers must not both see room and both start: the check and the start hold one lock.
    fake, marker, lock = tmp_path / "bin", tmp_path / "started", tmp_path / "launch.lock"
    fake.mkdir()
    (fake / "systemctl").write_text("""#!/bin/sh
case "$1" in
  list-units) [ -n "${BUSY:-}" ] && printf 'taste-harbor-cal-w2.service loaded active running x\\n';;
  show) echo '{ path=/h ; argv[]=/h run -p /t -k 2 -n 24 ; }';;
  is-active) exit 1;;
esac
""")
    (fake / "systemctl").chmod(0o755)
    # A job of one trial whose launch only leaves a mark (sh ignores the arguments after its script).
    spec = jobs.JobSpec("map-a-s3-b1", 1, ("sh", "-c", f"touch {marker}", "-n", "1"), (("JOBS", str(tmp_path / "jobs")),))
    text = jobs.launcher([spec], prefix="map", max_trials=24)
    assert 'flock 9' in text and "LOCK=${LOCK:-/run/lock/taste-harbor-launch.lock}" in text
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "WAIT": "0", "LOCK": str(lock),
           "QUEUE": str(tmp_path / "queue")}
    # The host is full: the job waits (here, until the test gives up on it).
    with contextlib.suppress(subprocess.TimeoutExpired):
        subprocess.run(["sh", "-c", text], env={**env, "BUSY": "1"}, timeout=3, capture_output=True)
    assert not marker.exists()
    # Room again: it starts, and the lock is free afterwards.
    done = subprocess.run(["sh", "-c", text], env=env, timeout=10, capture_output=True, text=True)
    assert done.returncode == 0 and marker.exists()
    assert subprocess.run(["flock", "-n", str(lock), "true"]).returncode == 0


def test_jobs_start_in_the_order_they_began_waiting_across_launchers(tmp_path):
    # A probe of eight waiting for room must not be overtaken by later one-trial jobs that fit.
    fake, marker, lock, queue = tmp_path / "bin", tmp_path / "started", tmp_path / "launch.lock", tmp_path / "queue"
    fake.mkdir()
    queue.mkdir()
    (fake / "systemctl").write_text("""#!/bin/sh
case "$1" in
  list-units) printf 'taste-harbor-cal-w2.service loaded active running x\\n';;
  show) echo '{ path=/h ; argv[]=/h run -p /t -k 2 -n 20 ; }';;
  is-active) exit 1;;
esac
""")
    (fake / "systemctl").chmod(0o755)
    spec = jobs.JobSpec("map-a-s3-b1", 1, ("sh", "-c", f"touch {marker}", "-n", "1"), (("JOBS", str(tmp_path / "jobs")),))
    text = jobs.launcher([spec], prefix="map", max_trials=24)
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "WAIT": "0", "LOCK": str(lock), "QUEUE": str(queue)}
    # An older ticket of a live launcher (this test) is first in line: the job fits but waits.
    older = queue / f"{1:019d}-{os.getpid()}-map-b-s5-b1"
    older.write_text("")
    with contextlib.suppress(subprocess.TimeoutExpired):
        subprocess.run(["sh", "-c", text], env=env, timeout=3, capture_output=True)
    assert not marker.exists()
    # A ticket whose launcher has died is cleared, and the job then starts and gives up its own.
    older.rename(queue / f"{1:019d}-999999999-map-b-s5-b1")
    done = subprocess.run(["sh", "-c", text], env=env, timeout=30, capture_output=True, text=True)
    assert done.returncode == 0 and marker.exists() and list(queue.iterdir()) == []
