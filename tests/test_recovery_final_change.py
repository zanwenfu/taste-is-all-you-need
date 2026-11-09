"""Each episode's final state, kept: its last trial's final change in its outcome, and its saves kept for a later check.

The restarts' final code was never kept (ISSTA plan, item 13). With ``final_change_chars`` every agent trial reads its
final change when it ends (``taste.benchmarks.final_change``) and an episode's outcome holds the one of the trial that
ended it, with that trial's own save and the cap; with ``keep_final_saves`` those saves are never released, so that
the requirement check (``taste.recovery_study.requirement_driver``) restores that very state.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from taste.recovery_study.recovery_driver import RecoveryDriver
from tests.recovery_fakes import FakeHarbor
from tests.test_recovery_final_saves import base_saved, saving, specs, two_runs
from tests.test_recovery_messages import graded_recoveries
from tests.test_recovery_messages import study as messages_study
from tests.test_recovery_runs import drive, step

_SPEC = importlib.util.spec_from_file_location(
    "recovery_runs_cli_change", Path(__file__).resolve().parents[1] / "scripts" / "recovery_runs.py")
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)


def changing(outcome):
    """Outcomes whose agent trials read their final change when the settings ask (as harbor_agent does)."""
    def wrapped(spec, task, settings, harbor):
        fields = outcome(spec, task, settings, harbor)
        chars = settings.get("final_change_chars")
        if chars and settings.get("agent") != "checker" and settings.get("branch_live") != "off":
            fields["change"] = {"read": True, "diff": f"diff --git a/src/app.py b/src/app.py\n+{spec.name}\n",
                                "paths": ["M\tsrc/app.py"], "cap_chars": int(chars), "cut": False}
        return fields
    return wrapped


def outcomes(driver, run_id):
    """(the episode as the state holds it, its outcome) for every episode of the run."""
    arms = driver.summary()[run_id]["arms"]
    return [(episode, outcome) for name, arm in arms.items()
            for episode, outcome in zip(driver.runs[run_id]["arms"][name]["episodes"], arm["episodes"], strict=True)]


def test_every_agent_trial_reads_its_final_change_and_each_outcome_holds_its_last_trials(tmp_path):
    jobs_dir, trials, given, run_id = messages_study(tmp_path, recoveries=["retry_feedback", "continue"],
                                                     save_final=True, final_change_chars=60000)
    base_saved(jobs_dir, run_id)
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir,
                      FakeHarbor(trials, changing(saving(graded_recoveries()))))
    live = specs(driver, "first") + specs(driver, "round")
    assert specs(driver, "round") and all(settings["final_change_chars"] == "60000" for _, settings in live)
    # The checker's trials read none: their containers are copies, and their change is the checker's own tests.
    assert specs(driver, "check") and not any("final_change_chars" in settings for _, settings in specs(driver, "check"))
    found = outcomes(driver, run_id)
    assert len(found) == 4
    for episode, outcome in found:
        last = [step for step in episode["rounds"] if step.get("agent")][-1]
        job, trial = last["agent"]
        final = outcome["final"]
        assert (final["trial"], final["cap_chars"]) == (trial, 60000)
        # The change of the trial that ended the episode, not of an earlier round's.
        assert final["change"]["diff"].endswith(f"+{job}\n") and final["change"]["read"]
        assert final["save"] == f"{job}-final-{driver.trial(last['agent'])['token']}"
        assert final["save_changed_paths"] == ["src/app.py"]


def test_without_it_no_trial_reads_one_and_every_outcome_is_as_before(tmp_path):
    jobs_dir, trials, given, run_id = messages_study(tmp_path, recoveries=["continue"], save_final=True)
    base_saved(jobs_dir, run_id)
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir,
                      FakeHarbor(trials, changing(saving(graded_recoveries()))))
    assert not any("final_change_chars" in settings for purpose in ("first", "round", "check")
                   for _, settings in specs(driver, purpose))
    found = outcomes(driver, run_id)
    assert found and all("final" not in outcome for _, outcome in found)


def test_with_keep_final_saves_each_episodes_last_saves_are_never_released(tmp_path):
    jobs_dir, trials, given, _ = two_runs(tmp_path, keep_final_saves=True, final_change_chars=60000)
    harbor, path = FakeHarbor(trials, changing(saving(graded_recoveries()))), tmp_path / "rec.json"
    for _ in range(60):
        driver, emitted = step(path, given, jobs_dir)
        if not emitted:
            break
        harbor.run(emitted)
    released, last, earlier = set(driver.released_saves()), set(), set()
    for run in driver.runs.values():
        for arm in run["arms"].values():
            for episode in arm["episodes"]:
                assert episode["end"]
                kept = set(driver._round_saves({"rounds": driver._last_round(episode)}))
                last |= kept
                earlier |= set(driver._round_saves(episode)) - kept
    assert last and earlier and not (last & released) and earlier <= released
    # Each kept save is the one its episode's outcome names.
    named = {outcome["final"]["save"] for run in driver.summary().values() for arm in run["arms"].values()
             for outcome in arm["episodes"]}
    assert named == last


def test_without_keep_final_saves_ended_episodes_release_every_save_as_before(tmp_path):
    jobs_dir, trials, given, _ = two_runs(tmp_path, final_change_chars=60000)
    harbor, path = FakeHarbor(trials, changing(saving(graded_recoveries()))), tmp_path / "rec.json"
    for _ in range(60):
        driver, emitted = step(path, given, jobs_dir)
        if not emitted:
            break
        harbor.run(emitted)
    every = {name for run in driver.runs.values() for arm in run["arms"].values() for episode in arm["episodes"]
             for name in driver._round_saves(episode)}
    assert every and every <= set(driver.released_saves())


@pytest.mark.parametrize("settings, message", [
    ({"final_change_chars": 999}, "final_change_chars"),
    ({"final_change_chars": -1}, "final_change_chars"),
    ({"final_change_chars": True}, "final_change_chars"),
    ({"final_change_chars": 1_000_001}, "final_change_chars"),
    ({"keep_final_saves": "yes"}, "keep_final_saves"),
])
def test_settings_no_job_could_be_built_from_are_refused(tmp_path, settings, message):
    _jobs_dir, _trials, given, _run_id = messages_study(tmp_path, recoveries=["continue"])
    with pytest.raises(ValueError, match=message):
        RecoveryDriver.open(tmp_path / "rec.json", {**given, **settings})


def test_the_command_line_takes_both():
    assert cli.OPTIONS["final_change_chars"]("60000") == 60000
    assert cli.OPTIONS["keep_final_saves"]("true") is True and cli.OPTIONS["keep_final_saves"]("0") is False
