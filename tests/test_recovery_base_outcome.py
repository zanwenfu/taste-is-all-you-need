"""Which base runs a recovery state takes: failed ones (the study), or those that passed the hidden tests (what the
reviewer and the recoveries do to work that was already right)."""

from __future__ import annotations

import pytest

from taste.recovery_study.recovery_driver import RecoveryDriver
from tests.recovery_fakes import BASE_SETTINGS, FakeHarbor, agent_steps, finish, recoveries, trial
from tests.test_recovery_runs import drive, step

ARMS = ["continue", "retry_feedback", "retry"]


def both(tmp_path, **settings):
    """A base job of one failed and one passed run of the same task, and its calibration runs."""
    jobs_dir, trials = tmp_path / "jobs", tmp_path / "trials"
    failed, passed = (trial(jobs_dir, "base", "task-a", reward=reward, trials_root=trials,
                            steps=agent_steps(6, tests_pass_at=(3,)), settings=BASE_SETTINGS)
                      for reward in (0.0, 1.0))
    for reward, cost, seconds in ((1.0, 0.5, 300.0), (0.0, 0.7, 600.0), (0.0, 0.9, 900.0)):
        trial(jobs_dir, "calibration", "task-a", reward=reward, trials_root=trials, steps=agent_steps(3),
              settings=BASE_SETTINGS, cost=cost, seconds=seconds)
    finish(jobs_dir, "base")
    finish(jobs_dir, "calibration")
    given = {"jobs_dir": str(jobs_dir), "trials_root": str(trials), "run_harbor": "/opt/taste/run-harbor.sh",
             "repeats": 2, "rewind_sources": ["rules"], "budget_runs": 1.0, "recoveries": ARMS, **settings}
    return jobs_dir, trials, given, failed, passed


@pytest.mark.parametrize("outcome, wanted", [(None, "failed"), ("failed", "failed"), ("passed", "passed"),
                                             ("any", "both")])
def test_a_state_takes_the_base_runs_of_the_outcome_asked_for(tmp_path, outcome, wanted):
    jobs_dir, _, given, failed, passed = both(tmp_path, **({"base_outcome": outcome} if outcome else {}))
    driver, _ = step(tmp_path / "rec.json", given, jobs_dir)
    taken = {"failed": {failed}, "passed": {passed}, "both": {failed, passed}}[wanted]
    assert set(driver.runs) == taken
    if outcome in ("passed", "any"):
        # Why each run is in the state, where that is no longer only that it failed.
        assert driver.runs[passed]["taken"] == f"passed the hidden tests (base_outcome: {outcome})"
    else:
        assert all("taken" not in run for run in driver.runs.values())


def test_a_passed_run_the_reviewer_accepts_ends_there_and_one_it_rejects_gets_the_arms(tmp_path):
    for verdict, status in (("done", "accepted"), ("not_done", "rejected")):
        root = tmp_path / verdict
        jobs_dir, trials, given, _, passed = both(root, base_outcome="passed")
        harbor = FakeHarbor(trials, recoveries(verdict=lambda spec, verdict=verdict: verdict))
        driver, _ = drive(root / "rec.json", given, jobs_dir, harbor)
        run = driver.runs[passed]
        assert run["status"] == status
        if verdict == "done":
            assert not run["arms"]
            assert not [job for job in driver.jobs.values() if job["purpose"] in ("first", "round")]
        else:
            assert sorted(arm["recovery"] for arm in run["arms"].values()) == sorted(ARMS)
            assert all(episode["end"] for arm in run["arms"].values() for episode in arm["episodes"])


def test_the_checks_alone_first_then_a_second_state_recovers_from_their_verdicts(tmp_path):
    # The net-effect run in two stages: the reviewer checks the passed runs (no arms), then the runs it rejected are
    # recovered from those verdicts and saved final states, with no check made again.
    jobs_dir, trials, given, _, passed = both(tmp_path, base_outcome="passed")
    harbor = FakeHarbor(trials, recoveries(verdict=lambda spec: "not_done"))
    first, _ = drive(tmp_path / "ne1.json", {**given, "recoveries": [], "prefix": "ne1"}, jobs_dir, harbor)
    assert first.runs[passed]["status"] == "rejected" and not first.runs[passed]["arms"]
    checks = {name for name, job in first.jobs.items() if job["purpose"] == "check"}
    second, _ = drive(tmp_path / "ne2.json", {**given, "prefix": "ne2", "checks_from": str(tmp_path / "ne1.json"),
                                              "prior_checkpoints": "restore"}, jobs_dir, harbor)
    run = second.runs[passed]
    assert run["check"] == first.runs[passed]["check"] and run["status"] == "rejected"
    assert sorted(arm["recovery"] for arm in run["arms"].values()) == sorted(ARMS)
    made = {name for name, job in second.jobs.items() if not job.get("taken_from")}
    assert not made & checks and all(name.startswith("ne2") for name in made)


def test_only_runs_takes_the_named_base_runs_though_the_budget_still_reads_them_all(tmp_path):
    jobs_dir, _, given, failed, passed = both(tmp_path, base_outcome="any")
    open_all, _ = step(tmp_path / "all.json", given, jobs_dir)
    driver, _ = step(tmp_path / "rec.json", {**given, "only_runs": [passed]}, jobs_dir)
    assert set(driver.runs) == {passed}
    # The run left out is still read, so the task's budget does not change with the list.
    assert len(driver.data["fresh"]) == len(open_all.data["fresh"])
    assert driver.runs[passed]["budget"] == open_all.runs[passed]["budget"]


def test_only_runs_still_holds_to_the_outcome_asked_for(tmp_path):
    jobs_dir, _, given, failed, passed = both(tmp_path, base_outcome="passed")
    driver, _ = step(tmp_path / "rec.json", {**given, "only_runs": [failed, passed]}, jobs_dir)
    assert set(driver.runs) == {passed}


@pytest.mark.parametrize("given, message", [
    ({"base_outcome": "sometimes"}, "base_outcome"),
    # The tests arms tell the agent what the hidden tests failed: nothing, for a run that passed them.
    ({"base_outcome": "passed", "recoveries": ["continue_tests"]}, "hidden tests"),
    ({"base_outcome": "any", "recoveries": ["continue_both"]}, "hidden tests"),
    ({"only_runs": "a-run"}, "only_runs"),
    ({"only_runs": [1]}, "only_runs"),
])
def test_settings_that_cannot_hold_for_passed_runs_are_refused(tmp_path, given, message):
    _, _, settings, _, _ = both(tmp_path)
    with pytest.raises(ValueError, match=message):
        RecoveryDriver.open(tmp_path / "rec.json", {**settings, **given})
