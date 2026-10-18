"""What a recovery tells the agent: the reviewer's feedback, only that it failed, or the tests it failed."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from taste.agents.checker import BARE_FEEDBACK, parse_submission
from taste.recovery_study.recovery_driver import RecoveryDriver, note
from tests.recovery_fakes import FakeHarbor, agent_steps, base_jobs, recoveries, submission
from tests.test_recovery_runs import FEEDBACK, drive, rounds_of, step

_SPEC = importlib.util.spec_from_file_location(
    "recovery_report", Path(__file__).resolve().parents[1] / "scripts" / "recovery_report.py")
report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report)

FAILING = "--- FAIL: TestParse (0.00s)\n    app_test.go:12: expected [] got error\nFAIL\n"
TOLD = ("These required tests did not pass: TestParse.\n\nTest output:\n"
        "--- FAIL: TestParse (0.00s)\n    app_test.go:12: expected [] got error\nFAIL")


def graded(passed, output=FAILING):
    """What SWE-bench Pro's adapter leaves in a trial's verifier folder."""
    return {"output.json": json.dumps({"tests": [{"name": name, "status": "PASSED"} for name in passed]}),
            "run-script-stdout.txt": output, "run-script-stderr.txt": "", "reward.txt": "0"}


def study(tmp_path, **settings):
    """One failed run of a task whose tests are TestLoad and TestParse; the base run failed TestParse."""
    jobs_dir, trials, tasks = tmp_path / "jobs", tmp_path / "trials", tmp_path / "tasks"
    (tasks / "task-a" / "tests").mkdir(parents=True)
    (tasks / "task-a" / "tests" / "config.json").write_text(json.dumps(
        {"fail_to_pass": "['TestParse']", "pass_to_pass": "['TestLoad']"}))
    [run_id] = base_jobs(jobs_dir, trials, [agent_steps(6, tests_pass_at=(3,))],
                         calibration=[(1.0, 0.5, 300.0), (0.0, 0.7, 600.0), (0.0, 0.9, 900.0)], tasks_dir=str(tasks))
    for file, text in graded(["TestLoad"]).items():
        (jobs_dir / "base" / run_id / "verifier").mkdir(exist_ok=True)
        (jobs_dir / "base" / run_id / "verifier" / file).write_text(text)
    given = {"jobs_dir": str(jobs_dir), "trials_root": str(trials), "run_harbor": "/opt/taste/run-harbor.sh",
             "repeats": 2, "rewind_sources": ["rules"], "budget_runs": 1.0, **settings}
    return jobs_dir, trials, given, run_id


def graded_recoveries(solved_from_round=None, **kwargs):
    """Recovery outcomes whose agent trials leave grader output: solved from that round on, else failing."""
    base = recoveries(reward=lambda spec: 1.0 if solved(spec) else 0.0, **kwargs)

    def solved(spec):
        if solved_from_round is None:
            return False
        part = next((p for p in spec.name.split("-") if p.startswith("e") and "r" in p), "r1")
        return int(part.split("r")[1]) >= solved_from_round

    def outcome(spec, task, settings, harbor):
        fields = base(spec, task, settings, harbor)
        if settings.get("agent") != "checker" and settings.get("branch_live") != "off":
            fields["verifier"] = graded(["TestLoad", "TestParse"] if fields["reward"] else ["TestLoad"])
        return fields
    return outcome


def first_round(tmp_path, recoveries_asked, output=None):
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=recoveries_asked)
    if output is not None:
        (jobs_dir / "base" / run_id / "verifier" / "run-script-stdout.txt").write_text(output)
    path, harbor = tmp_path / "rec.json", FakeHarbor(trials, graded_recoveries())
    driver, [saved] = step(path, given, jobs_dir)
    harbor.run([saved])
    driver, [check] = step(path, given, jobs_dir)
    harbor.run([check])
    driver, first = step(path, given, jobs_dir)
    return driver, run_id, {spec.name.split("-")[-2]: spec.settings() for spec in first}, saved


def test_round_one_tells_the_agent_only_that_it_failed_or_which_tests_failed(tmp_path):
    driver, run_id, first, saved = first_round(
        tmp_path, ["retry_bare", "rewind_bare", "continue_bare", "retry_tests", "rewind_tests", "continue_tests"])
    assert sorted(first) == ["bw3", "cb", "cx", "rb", "rx", "xw3"]
    assert sorted(driver.runs[run_id]["arms"]) == ["continue_bare", "continue_tests", "retry_bare", "retry_tests",
                                                   "rewind_bare@3", "rewind_tests@3"]
    # From scratch: the task text, then the message.
    assert Path(first["rb"]["task_suffix"]).read_text() == "A previous attempt was rejected by a reviewer."
    assert Path(first["rx"]["task_suffix"]).read_text() == "A previous attempt failed the project's tests.\n\n" + TOLD
    assert "branch" not in first["rb"] and "branch" not in first["rx"]
    # From the restart step: the step's output, then the message.
    for slug, text in (("bw3", "Note: an attempt that continued from here was rejected by a reviewer."),
                       ("xw3", "Note: an attempt that continued from here failed the project's tests.\n\n" + TOLD)):
        settings = first[slug]
        assert (settings["branch_step"], settings["branch_override"]) == ("3", "append")
        assert Path(settings["branch_note"]).read_text() == text
    # Where it stopped: the submission's output replaced by the message, from the saved final state.
    assert Path(first["cb"]["branch_note"]).read_text() == BARE_FEEDBACK
    tests = first["cx"]
    assert (tests["branch_step"], tests["branch_override"], tests["branch_checkpoint"]) == ("6", "reject", saved.name)
    assert Path(tests["branch_note"]).read_text() == "Submission failed the project's tests.\n\n" + TOLD


def test_a_nul_in_the_tests_output_is_written_as_an_escape_where_it_goes_in_as_the_task(tmp_path):
    # A grader's log can hold NUL (a Redis assertion did); a task's text cannot, so a task suffix shows it as \x00.
    # A branch note delivers it as it is.
    output = FAILING.replace("got error", "got '0.00\x001'")
    driver, run_id, first, saved = first_round(tmp_path, ["retry_tests", "rewind_tests"], output=output)
    told = TOLD.replace("got error", "got '0.00\x001'")
    task = Path(first["rx"]["task_suffix"]).read_text()
    assert "\x00" not in task
    assert task == "A previous attempt failed the project's tests.\n\n" + told.replace("\x00", "\\x00")
    assert Path(first["xw3"]["branch_note"]).read_text() == (
        "Note: an attempt that continued from here failed the project's tests.\n\n" + told)


def test_a_tests_recovery_is_decided_by_the_hidden_tests_with_no_checker(tmp_path):
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["retry_tests", "continue_tests", "retry_bare"],
                                            budget_runs=2.0)
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, graded_recoveries(solved_from_round=2)))
    checks = [job for job in driver.jobs.values() if job["purpose"] == "check"]
    # The run's own check (which chose it), and checks of the bare arm's rounds only.
    assert {job.get("arm") for job in checks} == {None, "retry_bare"}
    summary = driver.summary()[run_id]["arms"]
    for arm in ("retry_tests", "continue_tests"):
        episodes = summary[arm]["episodes"]
        assert [(e["end"], e["rounds"], e["solved"], e["verdicts"]) for e in episodes] == [("done", 2, True, [])] * 2
        later = rounds_of(driver, arm)
        assert later and all(s["branch_override"] == "reject" and s["branch_mode"] == "rebuild" for s in later)
        assert all(Path(s["branch_note"]).read_text() == "Submission failed the project's tests.\n\n" + TOLD
                   for s in later)
    bare = rounds_of(driver, "retry_bare")
    assert bare and all(Path(s["branch_note"]).read_text() == BARE_FEEDBACK for s in bare)


def test_a_tests_recovery_whose_trials_keep_failing_ends_on_the_budget(tmp_path):
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["continue_tests"], budget_runs=2.0)
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, graded_recoveries()))
    episodes = driver.summary()[run_id]["arms"]["continue_tests"]["episodes"]
    assert all(e["end"] == "budget" and not e["solved"] and e["rounds"] >= 2 for e in episodes)


def test_notes_for_every_message():
    reply = parse_submission(submission("not_done"))
    assert note("retry_bare", reply) == "A previous attempt was rejected by a reviewer."
    assert note("rewind_bare", reply) == "Note: an attempt that continued from here was rejected by a reviewer."
    assert note("continue_bare", reply) == BARE_FEEDBACK
    assert note("retry_feedback", reply) == "A previous attempt was rejected by a reviewer: " + FEEDBACK
    assert note("continue_tests", TOLD) == "Submission failed the project's tests.\n\n" + TOLD
    assert note("retry_tests", "") == "A previous attempt failed the project's tests."


def test_the_tests_message_cap_is_a_setting_with_a_floor(tmp_path):
    assert RecoveryDriver.open(tmp_path / "b.json", {}).settings["tests_feedback_chars"] == 1200
    with pytest.raises(ValueError, match="tests_feedback_chars"):
        RecoveryDriver.open(tmp_path / "c.json", {"tests_feedback_chars": 50})


def test_the_report_names_each_message_and_restart_point():
    run = {"sources": {"oracle": 4, "rules": 3, "reader": 3},
           "arms": {"rewind_bare@3": {"recovery": "rewind_bare", "step": 3, "episodes": [{"solved": True}]},
                    "rewind_tests@4": {"recovery": "rewind_tests", "step": 4, "episodes": [{"solved": False}]},
                    "rewind@3": {"recovery": "rewind", "step": 3, "episodes": [{"solved": True}]},
                    "retry_tests": {"recovery": "retry_tests", "step": 0, "episodes": [{"solved": True}]}}}
    assert sorted(report.views(run)) == ["retry_tests", "rewind_bare_reader", "rewind_bare_rules", "rewind_reader",
                                         "rewind_rules", "rewind_tests_oracle"]
    for name in ("retry_bare", "retry_tests", "continue_tests", "rewind_bare_oracle", "rewind_tests_oracle"):
        assert name in report.VIEWS
