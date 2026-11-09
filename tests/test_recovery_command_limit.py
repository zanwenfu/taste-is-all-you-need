"""A longer limit on each of the agent's commands, for the tasks of chosen repository languages (ISSTA plan, item 6).

On Luna, keeping going's margin sits on tasks without pass-to-pass tests, mostly Go, where mini-swe-agent's 30-second
command limit ends most of Luna's builds. With ``agent_command_seconds`` and ``agent_command_languages`` every agent
trial of a task in those languages is given that limit (``agent_command_seconds`` in its settings, which its record
keeps), and every other trial, the checker's among them, keeps its own; so one state at the longer limit and one at
30 s, on the same runs and the same day, differ in that alone. A task's language is SWE-bench Multilingual's tag or
SWE-Bench Pro's repository (``task_language``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from taste.benchmarks.harbor_settings import TrialSettings
from taste.recovery_study import recovery_driver
from taste.recovery_study.jobs import JobSpec
from taste.recovery_study.recovery_driver import RecoveryDriver, task_language
from tests.recovery_fakes import FakeHarbor, agent_steps, base_jobs, recoveries
from tests.test_recovery_runs import drive

PRO_GO = "instance_flipt-io__flipt-0b119520afca1cf25c470ff4288c464d4510b944-v6c0e9e5e8d2e4b3a"
PRO_PYTHON = "instance_qutebrowser__qutebrowser-1a9e74bfaf9a9db2a510dc14572d33ded6040a57-v2ef375ac784985212b1805e1d0431dc8f1b3c171"
PRO_NODEBB = "instance_NodeBB__NodeBB-00c70ce7b0541cfc94afe567921d7668cdc8f4ac-vnan"

_SPEC = importlib.util.spec_from_file_location(
    "recovery_runs_cli_limit", Path(__file__).resolve().parents[1] / "scripts" / "recovery_runs.py")
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)


def tagged(tasks, name, *tags):
    """A task directory whose task.toml has these metadata tags (SWE-bench Multilingual's last is its language)."""
    (tasks / name).mkdir(parents=True)
    listed = ", ".join(f'"{tag}"' for tag in tags)
    (tasks / name / "task.toml").write_text(f'schema_version = "1.4"\n\n[metadata]\ntags = [{listed}]\n\n'
                                            '[agent]\ntimeout_sec = 3000.0\n')
    return tasks / name


def study(tmp_path, language, **settings):
    """One failed run of task-a, tagged with ``language``."""
    jobs_dir, trials, tasks = tmp_path / "jobs", tmp_path / "trials", tmp_path / "tasks"
    tagged(tasks, "task-a", "debugging", "swe-bench", "swe-bench-multilingual", language)
    [run_id] = base_jobs(jobs_dir, trials, [agent_steps(6, tests_pass_at=(3,))],
                         calibration=[(1.0, 0.5, 300.0), (0.0, 0.7, 600.0), (0.0, 0.9, 900.0)], tasks_dir=str(tasks))
    given = {"jobs_dir": str(jobs_dir), "trials_root": str(trials), "run_harbor": "/opt/taste/run-harbor.sh",
             "repeats": 2, "rewind_sources": ["rules"], "budget_runs": 1.0, "recoveries": ["retry", "continue"],
             **settings}
    return jobs_dir, trials, given, run_id


def settings_by_purpose(driver):
    found = {}
    for job in driver.jobs.values():
        found.setdefault(job["purpose"], []).append(JobSpec.from_dict(job["spec"]).settings())
    return found


def test_a_tasks_language_is_multilinguals_tag_or_its_swe_bench_pro_repositorys(tmp_path):
    assert task_language(tagged(tmp_path, "rust-task", "debugging", "swe-bench-multilingual", "rust")) == "rust"
    # SWE-Bench Pro's tags name no language: its repository does, as the analysis splits the runs.
    assert task_language(tagged(tmp_path, PRO_GO, "debugging", "swe-bench-pro")) == "go"
    assert task_language(tmp_path / PRO_PYTHON) == "python"  # no task.toml here at all
    assert task_language(tmp_path / PRO_NODEBB) == "javascript"
    for name in ("instance_future-architect__vuls-" + "a" * 40 + "-v1", "instance_gravitational__teleport-" + "b" * 40,
                 "instance_navidrome__navidrome-" + "c" * 40 + "-vnan"):
        assert task_language(tmp_path / name) == "go", name
    # Neither says: no language, and no limit is given.
    assert task_language(tagged(tmp_path, "task-b", "debugging", "swe-bench")) is None
    assert task_language(tmp_path / ("instance_someone__else-" + "d" * 40)) is None
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "task.toml").write_text("not = [toml")
    assert task_language(tmp_path / "broken") is None


def test_every_agent_trial_of_a_task_in_the_languages_asked_for_gets_the_longer_limit(tmp_path):
    jobs_dir, trials, given, run_id = study(tmp_path, "go", agent_command_seconds=120,
                                            agent_command_languages=["go"])
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, recoveries()))
    found = settings_by_purpose(driver)
    agent = found["first"] + found.get("round", [])
    assert found.get("round") and all(settings["agent_command_seconds"] == "120" for settings in agent)
    # The checker keeps its own limit: the setting is mini-swe-agent's, and a checker trial refuses it.
    assert found["check"] and not any("agent_command_seconds" in settings for settings in found["check"])
    assert driver.summary()[run_id]["language"] == "go"
    # Each trial's record keeps the limit it ran with, so the analysis tells the states apart.
    rounds = [driver.trial(episode["rounds"][0]["agent"]) for arm in driver.runs[run_id]["arms"].values()
              for episode in arm["episodes"]]
    assert rounds and all(trial["settings"]["agent_command_seconds"] == "120" for trial in rounds)


def test_tasks_in_other_languages_and_states_without_the_limit_keep_the_agents_own(tmp_path):
    jobs_dir, trials, given, run_id = study(tmp_path / "rs", "rust", agent_command_seconds=120,
                                            agent_command_languages=["go"])
    driver, _ = drive(tmp_path / "rs" / "rec.json", given, jobs_dir, FakeHarbor(trials, recoveries()))
    assert not any("agent_command_seconds" in settings for group in settings_by_purpose(driver).values()
                   for settings in group)
    assert driver.summary()[run_id]["language"] == "rust"
    jobs_dir, trials, given, run_id = study(tmp_path / "off", "go")
    driver, _ = drive(tmp_path / "off" / "rec.json", given, jobs_dir, FakeHarbor(trials, recoveries()))
    assert not any("agent_command_seconds" in settings for group in settings_by_purpose(driver).values()
                   for settings in group)
    # An earlier study's state and summary are as they were: no language is read.
    assert "language" not in driver.runs[run_id] and "language" not in driver.summary()[run_id]


@pytest.mark.parametrize("settings, message", [
    ({"agent_command_seconds": 120}, "go together"),
    ({"agent_command_languages": ["go"]}, "go together"),
    ({"agent_command_seconds": -1, "agent_command_languages": ["go"]}, "agent_command_seconds"),
    ({"agent_command_seconds": True, "agent_command_languages": ["go"]}, "agent_command_seconds"),
    ({"agent_command_seconds": 601, "agent_command_languages": ["go"]}, "agent_command_seconds"),
    ({"agent_command_seconds": 120, "agent_command_languages": "go"}, "agent_command_languages"),
    ({"agent_command_seconds": 120, "agent_command_languages": [""]}, "agent_command_languages"),
    ({"agent_command_seconds": 120, "agent_command_languages": ["go"], "ak": {"agent_command_seconds": "60"}},
     "ak"),
])
def test_settings_no_job_could_be_built_from_are_refused(tmp_path, settings, message):
    _jobs_dir, _trials, given, _run_id = study(tmp_path, "go")
    with pytest.raises(ValueError, match=message):
        RecoveryDriver.open(tmp_path / "rec.json", {**given, **settings})


def test_the_drivers_ceiling_is_the_terminals_own_limit_on_any_command():
    assert TrialSettings().command_seconds == recovery_driver.COMMAND_CEILING


def test_the_command_line_takes_the_limit_its_languages_and_the_runs_named():
    assert cli.OPTIONS["agent_command_seconds"]("120") == 120
    assert cli.OPTIONS["agent_command_languages"]("go,rust") == ["go", "rust"]
    assert cli.OPTIONS["only_runs"]("run-1,run-2") == ["run-1", "run-2"]
