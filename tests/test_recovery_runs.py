"""The recovery driver: a checker's rejection, four recoveries at one budget, round after round."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from taste.agents.checker import BARE_FEEDBACK, parse_submission
from taste.recovery_study import recovery_driver
from taste.recovery_study.driver import FINISHED, STARTED
from taste.recovery_study.jobs import JobSpec
from taste.recovery_study.recovery_driver import RecoveryDriver, note, readings, round_seconds
from tests.recovery_fakes import FakeHarbor, agent_steps, base_jobs, recoveries, submission

_SPEC = importlib.util.spec_from_file_location(
    "recovery_runs", Path(__file__).resolve().parents[1] / "scripts" / "recovery_runs.py")
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)

FEEDBACK = ("Empty input must parse to an empty list; it raises.\n\n"
            "The reviewer's check: Parsed an empty string, as the task's first example does.\n"
            "$ python -c 'import app; print(app.parse(\"\"))'\n"
            "Traceback (most recent call last):\nValueError: empty\n"
            "Expected: [] on standard output\nObserved: a ValueError traceback")


def study(tmp_path, **settings):
    jobs_dir, trials = tmp_path / "jobs", tmp_path / "trials"
    [run_id] = base_jobs(jobs_dir, trials, [agent_steps(6, tests_pass_at=(3,))],
                         calibration=[(1.0, 0.5, 300.0), (0.0, 0.7, 600.0), (0.0, 0.9, 900.0)])
    given = {"jobs_dir": str(jobs_dir), "trials_root": str(trials), "run_harbor": "/opt/taste/run-harbor.sh",
             "repeats": 2, "rewind_sources": ["rules"], **settings}
    return jobs_dir, trials, given, run_id


def step(path, given, jobs_dir, *, oracle=None, reader=None):
    driver = RecoveryDriver.open(path, given)
    driver.use(oracle=oracle, reader=reader)
    driver.add_sources([jobs_dir / "base"], [jobs_dir / "calibration"])
    driver.refresh()
    specs = driver.advance()
    driver.save()
    return driver, specs


def drive(path, given, jobs_dir, harbor, *, oracle=None, reader=None, limit=60):
    batches = []
    for _ in range(limit):
        driver, specs = step(path, given, jobs_dir, oracle=oracle, reader=reader)
        if not specs:
            return driver, batches
        batches.append(specs)
        harbor.run(specs)
    raise AssertionError("the recoveries did not finish")


def specs_of(driver, purpose):
    return [JobSpec.from_dict(job["spec"]) for job in driver.jobs.values() if job["purpose"] == purpose]


def test_the_checker_checks_a_saved_copy_of_the_final_state_then_each_recovery_starts(tmp_path):
    jobs_dir, trials, given, run_id = study(
        tmp_path, recoveries=["retry", "retry_feedback", "rewind", "continue", "continue_bare"])
    path, harbor = tmp_path / "rec.json", FakeHarbor(trials, recoveries())
    driver, [saved] = step(path, given, jobs_dir)
    run = driver.runs[run_id]
    assert run["budget"] == {"usd": 0.7, "seconds": 600.0, "runs": 3, "source": "calibration runs"}
    assert run["rules"]["rules"] == 3
    # First the final state is rebuilt from the run's record and saved, the agent not run on.
    assert saved.settings() == {"agent": "mini-swe-agent", "services": "none", "reply_reserve_seconds": "10",
                                "branch": run["script"], "branch_step": "6", "branch_mode": "rebuild",
                                "branch_checkpoint": saved.name, "branch_live": "off"}
    harbor.run([saved])
    driver, [check] = step(path, given, jobs_dir)
    settings = check.settings()
    assert dict(check.env)["MODEL"] == "azure/gpt-6-sol" and check.attempts == 1
    assert {key: settings[key] for key in ("agent", "services", "worker_max_calls", "spend_cap_usd",
                                           "agent_timeout_sec", "worker_effort")} == {
        "agent": "checker", "services": "none", "worker_max_calls": "35", "spend_cap_usd": "2.5",
        "agent_timeout_sec": "1500", "worker_effort": "medium"}
    assert (settings["branch_mode"], settings["branch_checkpoint"], settings["branch_live"]) == (
        "restore", saved.name, "off")
    text = Path(settings["task_text"]).read_text()
    assert "<task>\nFix the bug.\n</task>" in text and "I fixed the bug and ran the tests." in text
    assert "diff --git a/src/app.py" in text and "<changed_paths>\nsrc/app.py\n</changed_paths>" in text
    harbor.run([check])
    driver, first = step(path, given, jobs_dir)
    first = {spec.name.split("-")[-2]: spec for spec in first}
    assert sorted(first) == ["cb", "ct", "rf", "rt", "rw3"] and all(s.attempts == 2 for s in first.values())
    assert driver.runs[run_id]["status"] == "rejected"
    assert driver.runs[run_id]["check"]["reply"]["verdict"] == "not_done"
    assert first["rt"].settings() == {"agent": "mini-swe-agent", "services": "none", "reply_reserve_seconds": "10",
                                      "spend_cap_usd": "0.7", "agent_timeout_sec": "750"}
    told = first["rf"].settings()
    assert Path(told["task_suffix"]).read_text() == "A previous attempt was rejected by a reviewer: " + FEEDBACK
    rewind = first["rw3"].settings()
    assert (rewind["branch_step"], rewind["branch_mode"], rewind["branch_override"]) == ("3", "rebuild", "append")
    assert Path(rewind["branch_note"]).read_text() == (
        "Note: an attempt that continued from here was rejected by a reviewer: " + FEEDBACK)
    assert rewind["agent_timeout_sec"] == "1050" and rewind["branch_live"] == "on"
    onward = first["ct"].settings()
    # Continue restores the saved final state, its submission's output replaced by the rejection.
    assert (onward["branch_step"], onward["branch_mode"], onward["branch_checkpoint"]) == ("6", "restore", saved.name)
    assert onward["branch_override"] == "reject"
    assert Path(onward["branch_note"]).read_text() == "Submission rejected by a reviewer: " + FEEDBACK
    assert Path(first["cb"].settings()["branch_note"]).read_text() == BARE_FEEDBACK
    assert first["ct"].prepare and not first["rt"].prepare


def test_rounds_continue_until_the_budget_cannot_pay_for_another_and_spending_adds_up(tmp_path):
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["retry", "continue"])
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, recoveries()))
    summary = driver.summary()[run_id]
    for arm in ("retry", "continue"):
        for episode in summary["arms"][arm]["episodes"]:
            assert (episode["end"], episode["rounds"]) == ("budget", 3)
            # Three trials at $0.20 and 150 s, two checks at $0.05 and 50 s; rebuilding a
            # final state for a check is not charged.
            assert episode["usd"] == pytest.approx(0.7) and episode["seconds"] == pytest.approx(550.0)
            assert episode["agent_usd"] == pytest.approx(0.6) and episode["check_usd"] == pytest.approx(0.1)
            assert [item["verdict"] for item in episode["verdicts"]] == ["not_done", "not_done"]
            assert episode["tokens"] == 3 * 5 * 1100
    rounds = [spec.settings() for spec in specs_of(driver, "round")]
    caps = sorted((s["spend_cap_usd"], s["agent_timeout_sec"]) for s in rounds)
    # Rounds two and three: what is left, plus the handoff and the prefix's allowance.
    assert caps == [("0.2", "650")] * 4 + [("0.45", "850")] * 4
    saved = {spec.name for spec in specs_of(driver, "final_state")}
    assert all(s["branch_override"] == "reject" and s["branch_mode"] == "restore" and s["branch_checkpoint"] in saved
               for s in rounds)
    assert all(Path(s["branch_note"]).read_text().startswith("Submission rejected by a reviewer:") for s in rounds)
    # Later rounds continue from the trial before, not from the base run.
    assert all(s["branch"] != driver.runs[run_id]["script"] for s in rounds)
    assert summary["check_usd"] == pytest.approx(0.05)


def test_the_checkers_done_ends_an_episode_whose_last_trial_is_its_outcome(tmp_path):
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["continue"])
    harbor = FakeHarbor(trials, recoveries(verdict=lambda spec: "done" if "-e1r2-" in spec.name else "not_done",
                                           reward=lambda spec: 1.0 if "-e1r2-a" in spec.name else 0.0))
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, harbor)
    first, second = driver.summary()[run_id]["arms"]["continue"]["episodes"]
    assert (first["end"], first["rounds"], first["solved"]) == ("done", 2, True)
    assert first["verdicts"] == [{"verdict": "not_done", "confidence": 0.8, "solved": False},
                                 {"verdict": "done", "confidence": 0.8, "solved": True}]
    assert first["usd"] == pytest.approx(0.5) and (second["end"], second["solved"]) == ("budget", False)


def test_an_agent_that_ends_without_submitting_ends_its_episode_unchecked(tmp_path):
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["retry"])
    harbor = FakeHarbor(trials, recoveries(status=lambda spec: "LimitsExceeded"))
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, harbor)
    episodes = driver.summary()[run_id]["arms"]["retry"]["episodes"]
    assert [(e["end"], e["rounds"], e["usd"]) for e in episodes] == [("no_submission", 1, 0.2)] * 2
    assert all(job.get("initial") for job in driver.jobs.values() if job["purpose"] in ("check", "final_state"))


def test_a_run_the_checker_accepts_is_not_recovered(tmp_path):
    jobs_dir, trials, given, run_id = study(tmp_path)
    driver, batches = drive(tmp_path / "rec.json", given, jobs_dir,
                            FakeHarbor(trials, recoveries(verdict=lambda spec: "done")))
    assert len(batches) == 2 and driver.runs[run_id]["status"] == "accepted" and driver.runs[run_id]["arms"] == {}


def test_a_verdict_on_an_unfaithful_copy_is_not_taken_and_the_check_is_made_again(tmp_path):
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["retry"])
    seen = []

    def faithful(spec):
        seen.append(spec.name)
        return len(seen) > 1

    driver, batches = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, recoveries(faithful=faithful)))
    assert [spec.name.rsplit("-", 1)[-1] for batch in batches[:3] for spec in batch] == ["s1", "c1", "c2"]
    assert driver.runs[run_id]["check"]["verdict"][0].endswith("-check-c2")
    assert specs_of(driver, "check")[1].settings()["branch_checkpoint"] == batches[0][0].name


def test_sources_that_choose_one_step_share_its_trials_and_the_map_lends_its_checkpoint(tmp_path):
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["rewind"], rewind_sources=["oracle", "reader", "rules"])
    mapped = {run_id: {"status": "done", "rewind": {"highest_v": 3, "latest": 4}, "checkpoints": {"3": "map-cp-3"}}}
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, recoveries()), oracle=mapped,
                      reader={run_id: {"step": 4, "reason": "wrong fix", "confidence": 0.6}})
    run = driver.runs[run_id]
    assert run["sources"] == {"oracle": 3, "reader": 3, "rules": 3} and list(run["arms"]) == ["rewind@3"]
    [first] = specs_of(driver, "first")
    assert first.settings()["branch_mode"] == "restore" and first.settings()["branch_checkpoint"] == "map-cp-3"
    # The oracle's other r*, and a map still searching: the oracle waits; the others go on.
    other = tmp_path / "other"
    jobs_dir, trials, given, run_id = study(other, recoveries=["rewind"], rewind_sources=["oracle", "rules"],
                                            oracle="latest")
    searching = {run_id: {"status": "searching", "rewind": {"highest_v": 3, "latest": 4}}}
    driver, _ = drive(other / "rec.json", given, jobs_dir, FakeHarbor(trials, recoveries()), oracle=searching)
    assert driver.runs[run_id]["sources"] == {"rules": 3}
    mapped = {run_id: {"status": "done", "rewind": {"highest_v": 3, "latest": 4}}}
    driver, _ = drive(other / "rec.json", given, jobs_dir, FakeHarbor(trials, recoveries()), oracle=mapped)
    assert driver.runs[run_id]["sources"] == {"rules": 3, "oracle": 4}
    assert sorted(driver.runs[run_id]["arms"]) == ["rewind@3", "rewind@4"]


def test_in_rebuild_mode_the_checker_trial_rebuilds_the_final_state_itself(tmp_path):
    jobs_dir, trials, given, _ = study(tmp_path, recoveries=["continue"], mode="rebuild", repeats=1)
    driver, batches = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, recoveries()))
    assert not specs_of(driver, "final_state")
    check = batches[0][0].settings()
    assert check["agent"] == "checker" and check["branch_mode"] == "rebuild" and "branch_checkpoint" not in check
    assert "<changed_paths>" not in Path(check["task_text"]).read_text()
    assert all(spec.settings()["branch_mode"] == "rebuild" for spec in specs_of(driver, "first") + specs_of(driver, "round"))


def test_plain_retry_can_keep_retrying_without_feedback(tmp_path):
    jobs_dir, trials, given, _ = study(tmp_path, recoveries=["retry"], retry_rounds="retry")
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, recoveries()))
    rounds = [spec.settings() for spec in specs_of(driver, "round")]
    assert rounds and all("branch" not in s and "task_suffix" not in s for s in rounds)


def test_budgets_and_kept_tasks_come_from_the_calibration_report(tmp_path):
    jobs_dir, _, given, run_id = study(tmp_path)
    report = tmp_path / "calibration.json"
    for kept, expected in ((True, None), (False, "the calibration did not keep the task")):
        report.write_text(json.dumps({"per_task": [{"task": "task-a", "kept": kept, "median_usd": 0.4,
                                                    "median_agent_seconds": 420.0, "attempts": 2}]}))
        driver = RecoveryDriver.open(tmp_path / f"rec-{kept}.json", {**given, "calibration_report": str(report)})
        driver.add_sources([jobs_dir / "base"], [jobs_dir / "calibration"])
        run = driver.runs[run_id]
        assert run.get("skipped") == expected
        if kept:
            assert run["budget"] == {"usd": 0.4, "seconds": 420.0, "runs": 2, "source": "calibration report"}
    report.write_text(json.dumps({"per_task": []}))
    driver = RecoveryDriver.open(tmp_path / "rec-none.json", {**given, "calibration_report": str(report)})
    driver.add_sources([jobs_dir / "base"])
    assert driver.runs[run_id]["skipped"] == "the task is not in the calibration report"


def test_settings_that_could_build_no_job_are_refused(tmp_path):
    for bad in ({"mode": "snapshot"}, {"checker_effort": "max"}, {"recoveries": ["undo"]}, {"oracle": "best"},
                {"prefix": "a b"}, {"repeats": 0}):
        with pytest.raises(ValueError):
            RecoveryDriver.open(tmp_path / "rec.json", bad)


def test_round_time_and_notes_follow_the_trials_settings_and_the_checkers_words():
    assert round_seconds({"reply_reserve_seconds": "10"}) == 160.0
    assert round_seconds({}) == 360.0 and round_seconds({}, 500) == 500.0
    reply = parse_submission(submission("not_done"))
    assert note("retry_feedback", reply) == "A previous attempt was rejected by a reviewer: " + FEEDBACK
    assert note("continue", reply).startswith("Submission rejected by a reviewer: Empty input")
    assert note("continue_bare", reply) == BARE_FEEDBACK


def test_the_time_constants_are_tastes_trial_defaults():
    from taste.benchmarks.harbor_settings import TrialSettings
    defaults = TrialSettings()
    assert (defaults.handoff_seconds, defaults.reply_reserve_seconds, defaults.plan_seconds) == (
        recovery_driver.HANDOFF_SECONDS, recovery_driver.REPLY_RESERVE_SECONDS, recovery_driver.PLAN_SECONDS)


def test_the_readers_file_holds_readings_by_run(tmp_path):
    path = tmp_path / "reader.json"
    path.write_text(json.dumps({"a": {"step": 4, "reason": "r", "confidence": 0.5}, "b": 7, "c": {"step": None}}))
    assert readings(path) == {"a": {"step": 4, "reason": "r", "confidence": 0.5}, "b": {"step": 7}}
    path.write_text(json.dumps([{"run": "a", "step": 2, "reason": "r", "confidence": 0.1}]))
    assert readings(path)["a"]["step"] == 2 and readings(tmp_path / "missing.json") == {}


def test_a_base_run_that_did_not_submit_is_left_out(tmp_path):
    jobs_dir, trials = tmp_path / "jobs", tmp_path / "trials"
    base_jobs(jobs_dir, trials, [(agent_steps(6, submit=False), "LimitsExceeded")],
              calibration=[(1.0, 0.5, 300.0)])
    driver = RecoveryDriver.open(tmp_path / "rec.json", {"jobs_dir": str(jobs_dir), "trials_root": str(trials)})
    driver.add_sources([jobs_dir / "base"], [jobs_dir / "calibration"])
    [run] = driver.runs.values()
    assert run["skipped"] == "the base run did not submit (LimitsExceeded)" and driver.advance() == []


def test_the_command_line_runs_the_driver_with_the_map_and_the_reader(tmp_path, capsys):
    jobs_dir, trials, _, run_id = study(tmp_path)
    state, launcher = tmp_path / "rec.json", tmp_path / "next.sh"
    reader = tmp_path / "reader.json"
    reader.write_text(json.dumps({run_id: {"step": 5, "reason": "a wrong fix", "confidence": 0.7}}))
    options = ["--state", str(state), "--base-job", str(jobs_dir / "base"), "--calibration",
               str(jobs_dir / "calibration"), "--jobs-dir", str(jobs_dir), "--trials", str(trials), "--repeats", "2",
               "--recoveries", "retry,rewind", "--rewind-sources", "reader,rules", "--reader", str(reader),
               "--launcher", str(launcher)]
    assert cli.main(options) == STARTED and "branch_live=off" in launcher.read_text()
    harbor = FakeHarbor(trials, recoveries())
    for _ in range(60):
        stored = json.loads(state.read_text())
        harbor.run([JobSpec.from_dict(job["spec"]) for job in stored["jobs"].values()
                    if not job["complete"] and job["spec"]["name"] not in harbor.started])
        status = cli.main(options)
        if status == FINISHED:
            break
    assert status == FINISHED
    stored = json.loads(state.read_text())
    assert stored["runs"][run_id]["sources"] == {"reader": 4, "rules": 3}
    assert "rewind@4" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main([*options, "--recoveries", "retry,undo"])
