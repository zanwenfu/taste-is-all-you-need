"""The recovery study's jobs are run-harbor.sh command lines, built in one place."""

from __future__ import annotations

import shlex

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
        "-k", "8", "-n", "8", "--ak", "agent=mini-swe-agent", "--ak", "services=none",
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


def test_the_launcher_starts_each_job_once_within_the_limit_and_then_waits():
    spec = jobs.harbor_job("map-a-s3-b1", tasks_dir="/t", task="a", attempts=2, settings={"agent": "x y"},
                           model="gpt-6-luna", jobs_dir="/study/jobs", run_harbor="/r.sh")
    text = jobs.launcher([spec], prefix="map", max_active=3, title="round 1")
    assert text.startswith("#!/bin/sh\n# round 1\n")
    assert "MAX_ACTIVE=${MAX_ACTIVE:-3}" in text and "'taste-harbor-map-*'" in text
    start = next(line for line in text.splitlines() if line.startswith("start map-a-s3-b1 "))
    _, name, directory, command = shlex.split(start)
    assert (name, directory) == ("map-a-s3-b1", "/study/jobs") and command == spec.shell()
    assert "'agent=x y'" in command
    assert text.rstrip().endswith("run the driver again\"")
    assert not any(line.startswith("start ") for line in jobs.launcher([], prefix="map", max_active=3).splitlines())
