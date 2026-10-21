"""The AgentRewind arm of the recovery study: carry on with its two tools, and at its first commit fork into a
rewind (the chosen call made again) and a twin that stays where it is, with the same note and budget left."""

from __future__ import annotations

import json
from pathlib import Path

from taste.agents import agentrewind
from taste.recovery_study.jobs import JobSpec
from tests.recovery_fakes import FakeHarbor, agent_steps, recoveries
from tests.test_recovery_runs import drive, study

NOTE = ("failed_check: test_parse. root_cause: the parser was edited before it was read. "
        "corrected_strategy: read src/app.py first.")
SENTENCE = ("You may also return to an earlier point of this run with backtrack_candidates and backtrack_commit, "
            "or repair the current state.")


def committing(target="llm-2"):
    """A round that, after the base run's six steps, lists the candidates and commits to ``target``."""
    listing = agentrewind.tool_command("backtrack_candidates", {"reason": "rejected"})
    commit = agentrewind.tool_command("backtrack_commit", {"record_uid": target, "memory_summary": NOTE})
    return [*agent_steps(6), (listing, 0, "[]", "Where to go back to?"), (commit, 0, "accepted", "Back to step 2.")]


def rewinding(commits=lambda spec: "-ar-f" in spec.name, faithful=lambda spec: True):
    """The base run is rejected; a round that commits does so as ``committing``; every submission is accepted."""
    base = recoveries(verdict=lambda spec: "not_done" if "-check-" in spec.name else "done")

    def outcome(spec, task, settings, harbor):
        if settings.get("branch_live") == "off" and "branch_checkpoint" not in settings:
            return {"reward": 0.0, "cost": 0.0, "seconds": 20.0, "branch": {"faithful": True}}  # graded, no call
        if settings.get("agent") == "checker" or settings.get("branch_live") == "off":
            return base(spec, task, settings, harbor)
        if not faithful(spec):
            return {"reward": 0.0, "cost": 0.0, "seconds": 20.0, "branch": {"faithful": False}}
        if commits(spec):
            return {"reward": 0.0, "steps": committing(), "cost": 0.2, "seconds": 150.0, "exit_status": "Stopped",
                    "stopped_by": "backtrack_commit", "branch": {"faithful": True}}
        return base(spec, task, settings, harbor)
    return outcome


def settings_of(driver, arm, purpose="round"):
    return [JobSpec.from_dict(job["spec"]).settings() for job in driver.jobs.values()
            if job["purpose"] == purpose and job.get("arm") == arm]


def test_round_one_carries_on_after_the_rejection_with_the_two_tools_and_one_sentence(tmp_path):
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["agentrewind"])
    path, harbor = tmp_path / "rec.json", FakeHarbor(trials, rewinding(commits=lambda spec: False))
    driver, _ = drive(path, given, jobs_dir, harbor)
    [first] = settings_of(driver, "agentrewind", "first")
    assert (first["branch_step"], first["branch_override"]) == ("6", "reject")
    told = Path(first["branch_note"]).read_text()
    assert told.startswith("Submission rejected by a reviewer: ") and told.endswith("\n\n" + SENTENCE)
    # The rewind arm's tools, no notes yet, and the floor a commit needs: a round's least share of the budget.
    assert json.loads(Path(first["branch_rewind"]).read_text()) == {"notes": [], "restart_usd": 0.07}
    episodes = driver.summary()[run_id]["arms"]["agentrewind"]["episodes"]
    assert [episode["end"] for episode in episodes] == ["done", "done"]
    assert all(episode["commits"] == 0 and episode["fork"] is None for episode in episodes)
    assert "agentrewind_norestore" not in driver.runs[run_id]["arms"]


def test_the_first_commit_forks_a_rewind_and_a_twin_that_stays_where_it_is(tmp_path):
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["agentrewind"], repeats=1)
    path, harbor = tmp_path / "rec.json", FakeHarbor(trials, rewinding())
    driver, _ = drive(path, given, jobs_dir, harbor)
    arms = driver.runs[run_id]["arms"]
    assert sorted(arms) == ["agentrewind", "agentrewind_norestore"]
    notes = [{"memory_summary": NOTE, "target_record_uid": "llm-2", "reason": None}]
    [rewound] = settings_of(driver, "agentrewind")
    [stayed] = settings_of(driver, "agentrewind_norestore")
    # Back to llm-2: step 1's files and messages, step 2's call made again with the notebook.
    assert (rewound["branch_step"], rewound["branch_override"], rewound["branch_mode"]) == ("1", "rewind", "rebuild")
    # The twin: where the agent was before it began to rewind (llm-7, the listing), with the same note.
    assert (stayed["branch_step"], stayed["branch_override"]) == ("6", "rewind")
    for settings in (rewound, stayed):
        assert Path(settings["branch_note"]).read_text() == agentrewind.notebook(notes)
        assert json.loads(Path(settings["branch_rewind"]).read_text()) == {"notes": notes, "restart_usd": 0.07}
        assert settings["branch"].endswith(".json")
    summary = driver.summary()[run_id]["arms"]
    [restored], [twin] = summary["agentrewind"]["episodes"], summary["agentrewind_norestore"]["episodes"]
    assert restored["end"] == twin["end"] == "done" and restored["rounds"] == twin["rounds"] == 2
    assert restored["commits"] == twin["commits"] == 1 and restored["targets"] == [2] and twin["targets"] == [7]
    assert restored["fork"] == 0 and twin["forked_from"] == 0
    # Both spent the same on the shared first round; each its own on the second.
    assert restored["agent_usd"] == twin["agent_usd"]


def test_a_rewind_that_cannot_be_brought_back_faithfully_is_scored_a_failure(tmp_path):
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["agentrewind"], repeats=1)
    harbor = FakeHarbor(trials, rewinding(faithful=lambda spec: "-ar-e1r2" not in spec.name))
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, harbor)
    summary = driver.summary()[run_id]["arms"]
    [restored] = summary["agentrewind"]["episodes"]
    assert restored["end"] == "rewind_unfaithful" and restored["solved"] is False and restored["reward"] == 0.0
    assert summary["agentrewind_norestore"]["episodes"][0]["end"] == "done"


def test_a_commit_with_no_round_left_grades_the_state_it_chose(tmp_path):
    """With too little budget left for another round, the rewind's state is graded with no model call; the twin's
    state is the committing round's own."""
    jobs_dir, trials, given, run_id = study(tmp_path, recoveries=["agentrewind"], repeats=1, min_round_share=0.8)
    driver, _ = drive(tmp_path / "rec.json", given, jobs_dir, FakeHarbor(trials, rewinding()))
    [graded] = settings_of(driver, "agentrewind")
    assert graded["branch_step"] == "1" and graded["branch_live"] == "off" and "branch_override" not in graded
    assert "branch_rewind" not in graded
    summary = driver.summary()[run_id]["arms"]
    assert summary["agentrewind"]["episodes"][0]["end"] == "budget"
    assert summary["agentrewind_norestore"]["episodes"][0]["end"] == "budget"
    assert settings_of(driver, "agentrewind_norestore") == []
