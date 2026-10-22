"""The AgentRewind arm decided by the hidden tests (agentrewind_tests): the same tools, sentence and fork into a rewind
and a twin, with Phase 1b's code message where the reviewer's feedback was, and no reviewer."""

from __future__ import annotations

import json
from pathlib import Path

from taste.recovery_study.recovery_driver import NOTES, note
from taste.recovery_study.test_failures import CODE_HEAD
from tests.recovery_fakes import FakeHarbor
from tests.test_recovery_agentrewind import NOTE, SENTENCE, committing, settings_of
from tests.test_recovery_messages import CODE_SHOWN, TOLD, graded, graded_recoveries, study
from tests.test_recovery_runs import drive

CODE_TOLD = "Submission failed the project's tests.\n\n" + TOLD + "\n\n" + CODE_HEAD + CODE_SHOWN


def by_tests(commits=lambda spec: "-ax-f" in spec.name, solved_from_round=1):
    """The hidden tests grade every agent trial (solved from that round on); a round ``commits`` names commits to
    llm-2 and saves its final files; a grading with no model call is graded as the base run was."""
    base = graded_recoveries(solved_from_round=solved_from_round)

    def outcome(spec, task, settings, harbor):
        if settings.get("branch_live") == "off" and "branch_checkpoint" not in settings:
            return {"reward": 0.0, "cost": 0.0, "seconds": 20.0, "branch": {"faithful": True},
                    "verifier": graded(["TestLoad"])}
        if settings.get("agent") != "checker" and settings.get("branch_live") != "off" and commits(spec):
            return {"reward": 0.0, "steps": committing(), "cost": 0.2, "seconds": 150.0, "exit_status": "Stopped",
                    "stopped_by": "backtrack_commit", "branch": {"faithful": True},
                    "final": {"saved": True, "directory": "/var/lib/taste-checkpoints/x"},
                    "verifier": graded(["TestLoad"])}
        return base(spec, task, settings, harbor)
    return outcome


def reviewed(driver, arm):
    return [job for job in driver.jobs.values() if job["purpose"] == "check" and job.get("arm") == arm]


def test_the_tests_message_then_the_one_sentence():
    told = "These required tests did not pass: TestParse."
    for arm in ("agentrewind_tests", "agentrewind_tests_norestore"):
        assert note(arm, told) == "Submission failed the project's tests.\n\n" + told + "\n\n" + SENTENCE
    # The reviewer's arm is told as before, word for word.
    assert NOTES["agentrewind"] == "Submission rejected by a reviewer: {feedback}\n\n" + SENTENCE


def test_round_one_carries_on_with_the_code_message_the_sentence_and_the_tools_and_the_tests_decide(tmp_path):
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["agentrewind_tests"])
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, by_tests(commits=lambda s: False)))
    [first] = settings_of(driver, "agentrewind_tests", "first")
    assert (first["branch_step"], first["branch_override"]) == ("6", "reject")
    assert Path(first["branch_note"]).read_text() == CODE_TOLD + "\n\n" + SENTENCE
    assert json.loads(Path(first["branch_rewind"]).read_text())["notes"] == []
    assert first["final_checkpoint"].endswith("-ax-f1-final")
    # No reviewer: the hidden tests passed, so each episode is done after its one round.
    assert reviewed(driver, "agentrewind_tests") == []
    episodes = driver.summary()[run_id]["arms"]["agentrewind_tests"]["episodes"]
    assert [(e["end"], e["rounds"], e["solved"]) for e in episodes] == [("done", 1, True)] * 2
    assert all(e["commits"] == 0 and e["fork"] is None for e in episodes)
    assert "agentrewind_tests_norestore" not in driver.runs[run_id]["arms"]


def test_the_first_commit_forks_a_rewind_and_its_own_twin_both_decided_by_the_tests(tmp_path):
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["agentrewind_tests"], repeats=1)
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, by_tests(solved_from_round=2)))
    assert sorted(driver.runs[run_id]["arms"]) == ["agentrewind_tests", "agentrewind_tests_norestore"]
    [rewound] = settings_of(driver, "agentrewind_tests")
    [stayed] = settings_of(driver, "agentrewind_tests_norestore")
    assert (rewound["branch_step"], rewound["branch_override"], rewound["branch_mode"]) == ("1", "rewind", "rebuild")
    assert (stayed["branch_step"], stayed["branch_override"], stayed["branch_mode"]) == ("6", "rewind", "restore")
    token = driver.trial(driver.runs[run_id]["arms"]["agentrewind_tests"]["episodes"][0]["rounds"][0]["agent"])["token"]
    assert stayed["branch_checkpoint"].endswith(f"-ax-f1-final-{token}") and "-nx-" in stayed["branch_note"]
    notes = json.loads(Path(rewound["branch_rewind"]).read_text())["notes"]
    assert notes == [{"memory_summary": NOTE, "target_record_uid": "llm-2", "reason": None}]
    assert reviewed(driver, "agentrewind_tests") == reviewed(driver, "agentrewind_tests_norestore") == []
    summary = driver.summary()[run_id]["arms"]
    [restored], [twin] = summary["agentrewind_tests"]["episodes"], summary["agentrewind_tests_norestore"]["episodes"]
    assert (restored["end"], restored["rounds"], restored["solved"]) == ("done", 2, True)
    assert (twin["end"], twin["rounds"], twin["solved"]) == ("done", 2, True)
    assert restored["commits"] == twin["commits"] == 1 and restored["targets"] == [2] and twin["targets"] == [7]
    assert restored["fork"] == 0 and twin["forked_from"] == 0


def test_a_round_after_a_failed_rewind_is_told_the_tests_in_code_form_with_the_sentence_and_the_notebook(tmp_path):
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["agentrewind_tests"], repeats=1, budget_runs=3.0)
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, by_tests(solved_from_round=3)))
    rounds = settings_of(driver, "agentrewind_tests")
    assert [settings["branch_override"] for settings in rounds] == ["rewind", "reject"]
    assert Path(rounds[1]["branch_note"]).read_text() == CODE_TOLD + "\n\n" + SENTENCE
    rewind = json.loads(Path(rounds[1]["branch_rewind"]).read_text())
    assert rewind["at_call"] == 2 and rewind["notes"][0]["target_record_uid"] == "llm-2"
    [episode] = driver.summary()[run_id]["arms"]["agentrewind_tests"]["episodes"]
    assert (episode["end"], episode["rounds"], episode["solved"], episode["commits"]) == ("done", 3, True, 1)


def test_a_commit_with_no_round_left_grades_the_state_it_chose(tmp_path):
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["agentrewind_tests"], repeats=1,
                                            min_round_share=0.8)
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, by_tests(solved_from_round=9)))
    [graded_state] = settings_of(driver, "agentrewind_tests")
    assert graded_state["branch_step"] == "1" and graded_state["branch_live"] == "off"
    summary = driver.summary()[run_id]["arms"]
    assert summary["agentrewind_tests"]["episodes"][0]["end"] == "budget"
    assert summary["agentrewind_tests_norestore"]["episodes"][0]["end"] == "budget"


def test_the_two_rewind_arms_run_side_by_side_each_with_its_own_twin(tmp_path):
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["agentrewind", "agentrewind_tests"], repeats=1)
    commits = lambda spec: "-ar-f" in spec.name or "-ax-f" in spec.name  # noqa: E731
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, by_tests(commits, 2)))
    assert sorted(driver.runs[run_id]["arms"]) == ["agentrewind", "agentrewind_norestore", "agentrewind_tests",
                                                   "agentrewind_tests_norestore"]
    # The reviewer decides its own pair only.
    assert reviewed(driver, "agentrewind") and not reviewed(driver, "agentrewind_tests")
    [first] = settings_of(driver, "agentrewind", "first")
    assert Path(first["branch_note"]).read_text().startswith("Submission rejected by a reviewer: ")
