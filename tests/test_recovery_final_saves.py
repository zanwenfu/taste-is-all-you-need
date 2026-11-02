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


# Bounded disk: runs in waves, saves released once nothing will restore them -------------------------------------------

def two_runs(tmp_path, **settings):
    from tests.recovery_fakes import agent_steps, base_jobs

    jobs_dir, trials = tmp_path / "jobs", tmp_path / "trials"
    runs = base_jobs(jobs_dir, trials, [agent_steps(6, tests_pass_at=(3,)), agent_steps(7, tests_pass_at=(3,))],
                     calibration=[(1.0, 0.5, 300.0), (0.0, 0.7, 600.0), (0.0, 0.9, 900.0)])
    for run_id in runs:
        base_saved(jobs_dir, run_id, name=f"base-final-{run_id[-6:]}")
    given = {"jobs_dir": str(jobs_dir), "trials_root": str(trials), "run_harbor": "/opt/taste/run-harbor.sh",
             "repeats": 2, "rewind_sources": ["rules"], "budget_runs": 1.0, "recoveries": ["retry", "continue"],
             "save_final": True, **settings}
    return jobs_dir, trials, given, runs


def test_at_most_max_active_runs_runs_have_arms_in_progress(tmp_path):
    from tests.test_recovery_runs import step

    jobs_dir, trials, given, _ = two_runs(tmp_path, max_active_runs=1)
    harbor, path = FakeHarbor(trials, saving(graded_recoveries())), tmp_path / "rec.json"
    seen = []
    for _ in range(60):
        driver, emitted = step(path, given, jobs_dir)
        active = [run_id for run_id, run in driver.runs.items()
                  if run.get("arms") and not driver._run_done(run)]
        seen.append(len(active))
        if not emitted:
            break
        harbor.run(emitted)
    assert max(seen) == 1
    # Both runs were recovered in the end, one after the other.
    assert all(driver._run_done(run) for run in driver.runs.values())


def test_saves_are_released_when_their_episode_ends_and_a_base_save_when_its_run_ends(tmp_path):
    from tests.test_recovery_runs import step

    jobs_dir, trials, given, _ = two_runs(tmp_path, max_active_runs=1)
    harbor, path = FakeHarbor(trials, saving(graded_recoveries())), tmp_path / "rec.json"
    released_while = []
    for _ in range(60):
        driver, emitted = step(path, given, jobs_dir)
        released = set(driver.released_saves())
        ended = {name for run in driver.runs.values() for arm in (run.get("arms") or {}).values()
                 for episode in arm["episodes"] if episode["end"] for name in driver._round_saves(episode)}
        in_progress = {name for run in driver.runs.values() for arm in (run.get("arms") or {}).values()
                       for episode in arm["episodes"] if not episode["end"] for name in driver._round_saves(episode)}
        assert ended <= released and not (in_progress & released)
        for run in driver.runs.values():
            base = driver._base_save(run)
            if base and run.get("arms") and not driver._run_done(run):
                assert base not in released
        released_while.append(released)
        if not emitted:
            break
        harbor.run(emitted)
    final = released_while[-1]
    assert {driver._base_save(run) for run in driver.runs.values()} <= final
    assert any("-final-" in name for name in final)


def test_the_command_line_deletes_released_saves_under_the_checkpoints_root_once(tmp_path):
    import importlib.util
    from pathlib import Path

    jobs_dir, trials, given, _ = two_runs(tmp_path)
    root = tmp_path / "checkpoints"
    given = {**given, "env": {"TASTE_CHECKPOINTS_ROOT": str(root)}}
    harbor, path = FakeHarbor(trials, saving(graded_recoveries())), tmp_path / "rec.json"
    from tests.test_recovery_runs import step
    for _ in range(60):
        driver, emitted = step(path, given, jobs_dir)
        for name in driver.released_saves() + [n for r in driver.runs.values() for n in [driver._base_save(r)] if n]:
            (root / name).mkdir(parents=True, exist_ok=True)
        if not emitted:
            break
        harbor.run(emitted)
    spec = importlib.util.spec_from_file_location(
        "recovery_runs_cli_prune", Path(__file__).resolve().parents[1] / "scripts" / "recovery_runs.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    (root / "other-study-checkpoint").mkdir()
    cli.main(["--state", str(path), "--prune-saves"])
    assert sorted(p.name for p in root.iterdir()) == ["other-study-checkpoint"]
    # A second pass finds nothing left to delete and fails on nothing.
    cli.main(["--state", str(path), "--prune-saves"])


# Equal time allowance (study S grid, Nov 2): a branch trial's deadline adds prefix_allowance (300 s) to bring its
# prefix back, but a restore takes seconds, so carry-on arms got ~5 extra minutes, and on tasks Sonnet solves in under
# a minute a fresh trial's deadline was below the harness's least (no fresh trial could start). With equal_allowance
# every trial's deadline adds the allowance.

def test_with_equal_allowance_fresh_and_branch_trials_get_the_same_extra_time(tmp_path):
    from taste.recovery_study.jobs import SETTING

    jobs_dir, trials, given, _ = messages_study(tmp_path, recoveries=["retry", "continue"], equal_allowance=True)
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, graded_recoveries()))
    first = {job["recovery"]: JobSpec.from_dict(job["spec"]).settings() for job in driver.jobs.values()
             if job["purpose"] == "first"}
    deadline = {arm: int(settings[SETTING["deadline"]]) for arm, settings in first.items()}
    assert deadline["retry"] == deadline["continue"]
    run = next(iter(driver.runs.values()))
    assert deadline["retry"] >= run["budget"]["seconds"] + driver.settings["prefix_allowance"]


def test_without_equal_allowance_only_branch_trials_get_the_extra_time(tmp_path):
    from taste.recovery_study.jobs import SETTING

    jobs_dir, trials, given, _ = messages_study(tmp_path, recoveries=["retry", "continue"])
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, graded_recoveries()))
    first = {job["recovery"]: JobSpec.from_dict(job["spec"]).settings() for job in driver.jobs.values()
             if job["purpose"] == "first"}
    deadline = {arm: int(settings[SETTING["deadline"]]) for arm, settings in first.items()}
    assert deadline["continue"] - deadline["retry"] == int(driver.settings["prefix_allowance"])
