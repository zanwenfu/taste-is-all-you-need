"""A checked trial's final state from its own save, not a replay (study S pilot, Nov 1 2026).

A check (and a later round that goes on from a submission) needs the checked trial's final files. Phase 1 rebuilt them
by replaying the trial's commands with no model call. Claude Sonnet 5.5 starts builds in the background and reads
their logs after a sleep: a replay's log differs (similarity 0 at that command), so the rebuild is unfaithful every
time, six saves and six checks were tried, and the attempt went unchecked. With ``save_final`` every live trial saves
its own final files when it ends (``final_checkpoint``), and its checks and the round after it restore that save. A
trial whose own save failed falls back to the replay, as before.
"""

from __future__ import annotations

import json

from taste.benchmarks import branch_trial
from taste.recovery_study import jobs
from taste.recovery_study.jobs import JobSpec
from tests.recovery_fakes import FakeHarbor
from tests.test_recovery_messages import graded_recoveries
from tests.test_recovery_messages import study as messages_study
from tests.test_recovery_runs import drive

CHANGED = ["src/app.py"]


def saving(outcome, *, saved=lambda spec: True):
    """Outcomes whose live trials save their own final files when the settings ask (as harbor_agent does)."""
    def wrapped(spec, task, settings, harbor):
        fields = outcome(spec, task, settings, harbor)
        if settings.get("final_checkpoint") and settings.get("agent") != "checker":
            fields["final"] = ({"saved": True, "partial": False, "changed_paths": CHANGED} if saved(spec)
                               else {"saved": False, "failed": "OSError"})
        return fields
    return wrapped


def base_saved(jobs_dir, run_id, name="base-final"):
    """The failed base run saved its own final files under ``name``."""
    path = jobs_dir / "base" / run_id / "result.json"
    result = json.loads(path.read_text())
    result["config"]["agent"]["kwargs"]["final_checkpoint"] = name
    result["agent_result"]["metadata"]["taste"]["final_checkpoint"] = {"saved": True, "partial": False,
                                                                       "changed_paths": CHANGED}
    path.write_text(json.dumps(result))
    return branch_trial.final_checkpoint_name(name, result["agent_result"]["metadata"]["taste"]["trial"])


def specs(driver, purpose):
    return [(name, JobSpec.from_dict(job["spec"]).settings()) for name, job in driver.jobs.items()
            if job["purpose"] == purpose]


def test_with_save_final_checks_and_later_rounds_restore_each_trials_own_save(tmp_path):
    jobs_dir, trials, given, run_id = messages_study(tmp_path, recoveries=["retry", "continue", "continue_bare"],
                                                     save_final=True)
    own = base_saved(jobs_dir, run_id)
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, saving(graded_recoveries())))
    # Nothing is replayed to save a final state.
    assert specs(driver, "final_state") == []
    # Every live trial saves its own final files, under its own job's name.
    live = specs(driver, "first") + specs(driver, "round")
    assert live and all(settings["final_checkpoint"] == f"{name}-final" for name, settings in live)
    # The base run's check restores the base run's own save, and so does round one of a carry-on arm.
    [base_check] = [settings for name, settings in specs(driver, "check") if "-check-" in name]
    assert (base_check[jobs.SETTING["mode"]], base_check[jobs.SETTING["checkpoint"]]) == ("restore", own)
    first = dict(specs(driver, "first"))
    carried = [settings for name, settings in first.items() if "-ct-" in name or "-cb-" in name]
    assert carried and all(s[jobs.SETTING["mode"]] == "restore" and s[jobs.SETTING["checkpoint"]] == own
                           for s in carried)
    # A check of a recovery trial, and the round after it, restore that trial's own save.
    checked = {job["spec"]["name"]: job for job in driver.jobs.values() if job["purpose"] == "check"}
    for name, settings in specs(driver, "check"):
        if "-check-" in name:
            continue
        assert settings[jobs.SETTING["mode"]] == "restore"
        assert "-final-" in settings[jobs.SETTING["checkpoint"]], name
    rounds = specs(driver, "round")
    assert any(s[jobs.SETTING["mode"]] == "restore" and "-final-" in s[jobs.SETTING["checkpoint"]]
               for name, s in rounds if "-ct-" in name)
    assert checked


def test_a_trial_whose_own_save_failed_is_rebuilt_for_its_check_as_before(tmp_path):
    jobs_dir, trials, given, run_id = messages_study(tmp_path, recoveries=["continue"], save_final=True)
    base_saved(jobs_dir, run_id)
    harbor = FakeHarbor(trials, saving(graded_recoveries(), saved=lambda spec: False))
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, harbor)
    # The recovery trials' saves failed: their checks rebuild the final state from the record first.
    rebuilt = [name for name, _ in specs(driver, "final_state")]
    assert rebuilt and all("-check-" not in name for name in rebuilt)


def test_without_save_final_nothing_changes(tmp_path):
    jobs_dir, trials, given, _ = messages_study(tmp_path, recoveries=["retry", "continue"])
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, graded_recoveries()))
    assert specs(driver, "final_state")
    assert not any("final_checkpoint" in settings for _, settings in specs(driver, "first") + specs(driver, "round"))


def test_a_trial_never_inherits_the_final_save_name_of_the_trial_it_repeats_or_branches_from():
    assert "final_checkpoint" not in jobs.inherited({"final_checkpoint": "base-final", "worker_effort": "low"})


def test_the_command_line_takes_save_final(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "recovery_runs_cli_final", Path(__file__).resolve().parents[1] / "scripts" / "recovery_runs.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.OPTIONS["save_final"]("true") is True and cli.OPTIONS["save_final"]("False") is False
