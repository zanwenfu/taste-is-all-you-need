"""What study S (Claude Sonnet 5.5 as the agent) needs of the recovery driver beyond Phase 1's.

- Arms whose episodes are the first attempt after the message only (``first_only``, the pre-registration's LEAN
  arms): the episode ends after round one; a reviewer arm's round one is still checked, so every arm has verdicts.
- A failed base run's budget is the medians of all its task's base runs, not of those read before it.
- The reviewer's trials may run on another model route than the agent's (``checker_env``).
- The command line takes these, and an empty list of recoveries (a state that only checks the base runs).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from taste.recovery_study.driver import STARTED
from taste.recovery_study.jobs import JobSpec
from taste.recovery_study.recovery_driver import DEFAULTS, WIDE_SENTENCE, RecoveryDriver
from tests.recovery_fakes import FakeHarbor, agent_steps, finish, recoveries, trial
from tests.test_recovery_messages import graded_recoveries
from tests.test_recovery_messages import study as messages_study
from tests.test_recovery_runs import drive, step
from tests.test_recovery_runs import study as runs_study

_SPEC = importlib.util.spec_from_file_location(
    "recovery_runs_cli_s", Path(__file__).resolve().parents[1] / "scripts" / "recovery_runs.py")
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)

LEAN = ["retry_bare", "continue_bare", "continue_tests", "continue_both"]


def episodes(driver, run_id, arm):
    return driver.summary()[run_id]["arms"][arm]["episodes"]


def checked_arms(driver):
    return {job.get("arm") for job in driver.jobs.values() if job["purpose"] == "check"}


# First attempt only ----------------------------------------------------------------------


def test_first_only_arms_end_after_round_one_and_the_reviewer_still_checks_their_round_one(tmp_path):
    arms = ["retry", *LEAN]
    jobs_dir, trials, given, run_id = messages_study(tmp_path, recoveries=arms, first_only=LEAN, budget_runs=2.0)
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, graded_recoveries()))
    # The full arm goes on round after round until its budget ends.
    assert all(e["end"] == "budget" and e["rounds"] >= 2 for e in episodes(driver, run_id, "retry"))
    for arm in ("retry_bare", "continue_bare", "continue_both"):
        found = episodes(driver, run_id, arm)
        assert [(e["end"], e["rounds"], [v["verdict"] for v in e["verdicts"]]) for e in found] == \
            [("first_only", 1, ["not_done"])] * 2, arm
    # The tests arm has no reviewer: one round, decided by nothing more.
    assert [(e["end"], e["rounds"], e["verdicts"]) for e in episodes(driver, run_id, "continue_tests")] == \
        [("first_only", 1, [])] * 2
    assert checked_arms(driver) == {None, "retry", "retry_bare", "continue_bare", "continue_both"}
    assert not [job for job in driver.jobs.values() if job["purpose"] == "round" and job["arm"] in LEAN]


def test_a_first_attempt_that_passes_ends_done(tmp_path):
    jobs_dir, trials, given, run_id = messages_study(tmp_path, recoveries=LEAN, first_only=LEAN, budget_runs=2.0)
    # The reviewer rejects the base run (its own check) and accepts every first attempt.
    accepts = graded_recoveries(solved_from_round=1,
                                verdict=lambda spec: "not_done" if "-check-" in spec.name else "done")
    harbor = FakeHarbor(trials, accepts)
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, harbor)
    for arm in LEAN:
        assert [(e["end"], e["rounds"], e["solved"]) for e in episodes(driver, run_id, arm)] == [("done", 1, True)] * 2


def test_a_first_only_reviewer_arm_is_checked_even_with_no_round_left(tmp_path):
    # Round one uses 500 of the 600 agent seconds: no round is left. A full arm then ends on its budget unchecked,
    # as it always has; a first-only arm's round one is checked all the same (the verdicts are compared across arms).
    jobs_dir, trials, given, run_id = runs_study(tmp_path, recoveries=["continue", "continue_bare"],
                                                 first_only=["continue_bare"])
    base = recoveries()

    def slow(spec, task, settings, harbor):
        fields = base(spec, task, settings, harbor)
        if settings.get("agent") != "checker" and settings.get("branch_live") != "off":
            fields["seconds"] = 500.0
        return fields

    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, slow))
    assert [(e["end"], e["verdicts"]) for e in episodes(driver, run_id, "continue")] == [("budget", [])] * 2
    assert [(e["end"], len(e["verdicts"])) for e in episodes(driver, run_id, "continue_bare")] == \
        [("first_only", 1)] * 2


def test_a_first_only_arm_can_go_unchecked_when_the_hidden_tests_alone_are_read(tmp_path):
    # One-sentence test: only the first attempt's hidden tests are measured, so no verdict is paid for after it.
    arms = ["continue", "continue_wide"]
    jobs_dir, trials, given, run_id = messages_study(tmp_path, recoveries=arms, first_only=arms,
                                                     first_only_checked=False, budget_runs=2.0)
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, graded_recoveries()))
    for arm in arms:
        assert [(e["end"], e["rounds"], e["verdicts"]) for e in episodes(driver, run_id, arm)] == \
            [("first_only", 1, [])] * 2, arm
    # The base run's own check is the only one left.
    assert checked_arms(driver) == {None}
    assert not [job for job in driver.jobs.values() if job["purpose"] == "round"]


def test_an_unchecked_first_only_arm_still_ends_no_submission_and_arms_outside_it_are_still_checked(tmp_path):
    jobs_dir, trials, given, run_id = messages_study(tmp_path, recoveries=["continue", "continue_wide"],
                                                     first_only=["continue_wide"], first_only_checked=False,
                                                     budget_runs=2.0)
    base = graded_recoveries()

    def stopped(spec, task, settings, harbor):
        fields = base(spec, task, settings, harbor)
        if settings.get("agent") != "checker" and "-cw-" in spec.name:
            fields["exit_status"] = "Stopped"
        return fields

    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, stopped))
    assert [e["end"] for e in episodes(driver, run_id, "continue_wide")] == ["no_submission"] * 2
    assert "continue" in checked_arms(driver)
    assert "continue_wide" not in checked_arms(driver)


def test_a_first_attempt_the_model_refused_is_tried_again_in_that_arm_and_never_counted(tmp_path):
    arms = ["continue", "continue_wide"]
    jobs_dir, trials, given, run_id = messages_study(tmp_path, recoveries=arms, first_only=arms,
                                                     first_only_checked=False, budget_runs=2.0)
    base = graded_recoveries()

    def refused_once(spec, task, settings, harbor):
        fields = base(spec, task, settings, harbor)
        if settings.get("agent") != "checker" and "-cw-" in spec.name and spec.name.endswith("-f1"):
            fields.update(stopped_by="model_unavailable", exit_status="Stopped")
        return fields

    harbor = FakeHarbor(trials, refused_once)
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, harbor)
    assert [e["end"] for e in episodes(driver, run_id, "continue_wide")] == ["first_only"] * 2
    wide = driver.runs[run_id]["arms"]["continue_wide"]
    assert len(wide["first"]) == 2
    assert all(episode["rounds"][0]["agent"][0] == wide["first"][1] for episode in wide["episodes"])


def test_the_wide_arm_keeps_its_sentence_in_a_later_round(tmp_path):
    arms = ["continue_wide"]
    jobs_dir, trials, given, run_id = messages_study(tmp_path, recoveries=arms, budget_runs=2.0)
    driver, harbor = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, graded_recoveries()))
    rounds = [job for job in driver.jobs.values() if job.get("purpose") == "round" and job.get("arm") == "continue_wide"]
    assert rounds
    for job in rounds:
        spec = JobSpec.from_dict(job["spec"])
        assert Path(spec.settings()["branch_note"]).read_text().endswith(WIDE_SENTENCE)


def test_first_only_rounds_are_checked_unless_told_otherwise_and_the_setting_is_a_true_or_false(tmp_path):
    assert DEFAULTS["first_only_checked"] is True
    _, _, given, _ = runs_study(tmp_path)
    with pytest.raises(ValueError, match="first_only_checked"):
        RecoveryDriver.open(tmp_path / "rec.json", {**given, "first_only_checked": "no"})
    assert cli.OPTIONS["first_only_checked"]("false") is False


def test_first_only_names_known_arms_and_not_the_rewind_tools_arm(tmp_path):
    _, _, given, _ = runs_study(tmp_path)
    for wrong in (["retry_bare", "no_such_arm"], ["agentrewind"]):
        with pytest.raises(ValueError, match="first_only"):
            RecoveryDriver.open(tmp_path / "rec.json", {**given, "first_only": wrong})
    assert DEFAULTS["first_only"] == []


# Budgets from all of a task's base runs ----------------------------------------------------


def test_a_failed_runs_budget_is_the_medians_of_all_its_tasks_base_runs_whatever_the_reading_order(tmp_path):
    jobs_dir, trials = tmp_path / "jobs", tmp_path / "trials"
    # Read in name order: the first failed run comes before the task's other runs, the last after them all.
    for name, reward, cost, seconds in (("a-run", 0.0, 0.1, 100.0), ("b-run", 1.0, 0.3, 300.0),
                                        ("c-run", 1.0, 0.5, 500.0), ("d-run", 0.0, 0.7, 700.0)):
        trial(jobs_dir, "base", "task-a", reward=reward, trials_root=trials, steps=agent_steps(8), cost=cost,
              seconds=seconds, name=name, settings={"agent": "mini-swe-agent", "services": "none"})
    finish(jobs_dir, "base")
    given = {"jobs_dir": str(jobs_dir), "trials_root": str(trials), "run_harbor": "/opt/taste/run-harbor.sh",
             "recoveries": ["retry"], "budget_runs": 2.0}
    driver = RecoveryDriver.open(tmp_path / "rec.json", given)
    driver.add_sources([jobs_dir / "base"], [])
    budgets = {run_id: run["budget"] for run_id, run in driver.runs.items()}
    assert budgets == {run_id: {"usd": 0.8, "seconds": 800.0, "runs": 4, "base_runs": 2.0, "source": "base runs"}
                       for run_id in ("a-run", "d-run")}


# The reviewer on its own route -------------------------------------------------------------


def test_the_reviewers_trials_take_their_own_environment_and_the_agents_keep_theirs(tmp_path):
    jobs_dir, trials, given, _ = runs_study(
        tmp_path, recoveries=["continue"], env={"SECRETS": "/root/taste-secrets/anthropic.env", "EXTRA": "1"},
        checker_env={"SECRETS": "/root/taste-secrets/azure.env"})
    path, harbor = tmp_path / "rec.json", FakeHarbor(trials, recoveries())
    _, [saved] = step(path, given, jobs_dir)
    harbor.run([saved])
    _, [check] = step(path, given, jobs_dir)
    harbor.run([check])
    _, first = step(path, given, jobs_dir)
    # The final-state save calls no model: it runs as the agent's trials do.
    assert dict(saved.env)["SECRETS"] == "/root/taste-secrets/anthropic.env"
    assert dict(check.env)["SECRETS"] == "/root/taste-secrets/azure.env" and dict(check.env)["EXTRA"] == "1"
    assert dict(check.env)["MODEL"] == "azure/gpt-6-sol"
    assert first and all(dict(spec.env)["SECRETS"] == "/root/taste-secrets/anthropic.env" for spec in first)


def test_checker_env_is_names_and_values(tmp_path):
    _, _, given, _ = runs_study(tmp_path)
    with pytest.raises(ValueError, match="checker_env"):
        RecoveryDriver.open(tmp_path / "rec.json", {**given, "checker_env": {"SECRETS": 3}})


# The command line ------------------------------------------------------------------------


def test_the_command_line_takes_first_only_checker_env_and_an_empty_list_of_recoveries(tmp_path):
    jobs_dir, trials, _, _ = runs_study(tmp_path)
    common = ["--base-job", str(jobs_dir / "base"), "--jobs-dir", str(jobs_dir), "--trials", str(trials),
              "--rewind-sources", "rules", "--spool", str(tmp_path / "spool")]
    # Checks only: no recovery is started for a rejected run.
    checks = tmp_path / "checks.json"
    assert cli.main(["--state", str(checks), "--prefix", "chk", "--recoveries", "", *common]) == STARTED
    stored = json.loads(checks.read_text())
    assert stored["settings"]["recoveries"] == [] and stored["settings"]["first_only"] == []
    grid = tmp_path / "grid.json"
    assert cli.main(["--state", str(grid), "--prefix", "grid", "--recoveries", "retry,continue_bare",
                     "--first-only", "continue_bare", "--checker-env", "SECRETS=/r.env",
                     "--env", "SECRETS=/a.env", *common]) == STARTED
    settings = json.loads(grid.read_text())["settings"]
    assert settings["first_only"] == ["continue_bare"] and settings["checker_env"] == {"SECRETS": "/r.env"}
    assert settings["env"] == {"SECRETS": "/a.env"}


def test_a_state_with_no_recoveries_checks_the_base_runs_and_starts_nothing_else(tmp_path):
    jobs_dir, trials, given, run_id = runs_study(tmp_path, recoveries=[])
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, recoveries()))
    run = driver.runs[run_id]
    assert run["status"] == "rejected" and run["arms"] == {}
    assert {job["purpose"] for job in driver.jobs.values()} == {"final_state", "check"}
    assert all(JobSpec.from_dict(job["spec"]).name.startswith("rec-") for job in driver.jobs.values())
