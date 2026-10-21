"""Rewards come from Harbor; cost, time and steps from Taste's settled record; rules from the steps."""

from __future__ import annotations

import json

import pytest

from taste.recovery_study import commands, records, rules
from tests import recovery_fakes as fakes
from tests.recovery_fakes import SENTINEL, agent_steps, finish, submission, trial


@pytest.mark.parametrize("command,written", [
    ("sed -i 's/a/b/' src/app.py", ["src/app.py"]),
    ("sed -i.bak -e 's/a/b/' -e 's/c/d/' src/a.py src/b.py", ["src/a.py", "src/b.py"]),
    ("perl -pi -e 's/x/y/' lib/x.pm", ["lib/x.pm"]),
    ("cat > tests/test_app.py <<'EOF'\nimport app\nx = a > b\nEOF", ["tests/test_app.py"]),
    ("python -m pytest -q > /tmp/log.txt 2>&1", []),
    ("echo hi >> notes.md && cp a.py b.py; mv c.py d/", ["b.py", "d/", "notes.md"]),
    ("python - <<'EOF'\nwith open('setup.cfg', 'w') as f:\n    f.write('x')\nEOF", ["setup.cfg"]),
    ("git apply <<'EOF'\n--- a/src/m.py\n+++ b/src/m.py\n@@ -1 +1 @@\n-a\n+b\nEOF", ["src/m.py"]),
    ("timeout 60 grep -rn pytest . | head", []),
    # SWE-agent's editor writes the path it names, unless it only views it.
    ("str_replace_editor str_replace /app/src/app.py --old_str 'a' --new_str 'b'", ["/app/src/app.py"]),
    ("str_replace_editor create /app/tests/test_new.py --file_text 'def test_x():\n    pass\n'",
     ["/app/tests/test_new.py"]),
    ("str_replace_editor view /app/src/app.py", []),
    ("str_replace_editor create /tmp/reproduce.py --file_text 'print(1)'", []),
])
def test_written_paths_follow_the_shell(command, written):
    assert commands.written_paths(command) == written


def test_gaming_flags_name_test_files_checks_and_skip_markers():
    diff = "diff --git a/tests/test_a.py b/tests/test_a.py\n+++ b/tests/test_a.py\n+@pytest.mark.skip\n"
    paths = ["src/app.py", "tests/test_a.py", "pytest.ini", ".github/workflows/ci.yml", "conftest.py"]
    assert commands.gaming_flags(paths, diff) == [
        "test_file:conftest.py", "test_file:tests/test_a.py", "check_file:.github/workflows/ci.yml",
        "check_file:pytest.ini", "skip_marker"]
    assert commands.gaming_flags(["src/app.py"]) == []


def test_a_trial_reads_its_reward_from_harbor_and_its_spend_and_steps_from_tastes_record(tmp_path):
    jobs_dir, trials = tmp_path / "jobs", tmp_path / "trials"
    steps = agent_steps(4, tests_pass_at=(2,), large_edit_at=(1,))
    name = trial(jobs_dir, "j", "task-a", reward=0.0, trials_root=trials, steps=steps, cost=0.08, seconds=301.5,
                 settings={"agent": "mini-swe-agent"},
                 branch={"faithful": True, "prefix_seconds": 41.5, "checkpoint": "cp-9", "changed_paths": ["a.txt"]})
    summary = records.trial(jobs_dir / "j" / name, trials)
    assert (summary["reward"], summary["solved"], summary["task"], summary["tasks_dir"]) == (0.0, False, "task-a", "/study/tasks")
    assert summary["cost_usd"] == pytest.approx(0.08) and summary["prompt_tokens"] == 4000
    assert summary["completion_tokens"] == 400 and summary["settled"] is True
    # One clock: the agent's execution by Harbor (301.5 s), not Taste's own count (318.5 s) nor the
    # whole trial's hour; charged without the 41.5 s branching spent bringing the prefix back.
    result = json.loads((jobs_dir / "j" / name / "result.json").read_text())
    assert result["agent_result"]["metadata"]["taste"]["seconds"] == 318.5
    assert summary["seconds"] == 301.5 and summary["charged_seconds"] == 260.0 and summary["prefix_seconds"] == 41.5
    assert (summary["faithful"], summary["checkpoint"], summary["changed_paths"]) == (True, "cp-9", ["a.txt"])
    assert summary["exit_status"] == "Submitted" and summary["steps"] == 4 and summary["checker"] is None
    assert summary["written_paths"] == ["src/app.py"] and summary["submission_paths"] == ["src/app.py"]
    assert not summary["skip_marker"] and summary["agent_started"] is True and records.usable(summary)
    view = records.agent_view(records.settled(summary["token"], trials))
    assert view["task"] == "Fix the bug." and view["final_message"] == "I fixed the bug and ran the tests."
    assert view["submission"].startswith("diff --git") and SENTINEL not in view["submission"]
    assert [step.number for step in view["steps"]] == [1, 2, 3, 4] and view["steps"][1].output == "3 passed in 0.12s"


def test_a_trial_says_what_stopped_it_and_counts_the_model_stops_that_end_or_cut_a_reply(tmp_path):
    jobs_dir, trials = tmp_path / "jobs", tmp_path / "trials"
    name = trial(jobs_dir, "j", "task-a", reward=0.0, trials_root=trials, steps=agent_steps(3), exit_status="Stopped")
    token = records.trial(jobs_dir / "j" / name, trials)["token"]
    fakes.record(trials, token, agent_steps(3), exit_status="Stopped", stopped_by="refusal",
                 stops={2: "max_tokens", 3: "refusal"})
    summary = records.trial(jobs_dir / "j" / name, trials)
    assert summary["stopped_by"] == "refusal"
    assert summary["model_stops"] == {"max_tokens": 1, "refusal": 1}
    plain = records.trial(jobs_dir / "j" / trial(jobs_dir, "j", "task-a", reward=1.0, trials_root=trials,
                                                 steps=agent_steps(2)), trials)
    assert plain["stopped_by"] == "" and plain["model_stops"] == {}
    unsettled = records.trial(jobs_dir / "j" / trial(jobs_dir, "j", "task-a", reward=1.0, trials_root=trials), trials)
    assert unsettled["stopped_by"] is None and unsettled["model_stops"] is None


def test_each_trial_keeps_the_image_it_ran_in_and_a_task_that_changed_image_is_named(tmp_path):
    """A task's image pulled again by its tag may not be the one the earlier study ran: each trial keeps its
    container's image ID, and a task whose trials ran in another image than expected, or in two, is named."""
    jobs_dir, trials = tmp_path / "jobs", tmp_path / "trials"
    old, new = "sha256:" + "a" * 64, "sha256:" + "b" * 64
    names = [trial(jobs_dir, "j", task, reward=0.0, trials_root=trials, steps=agent_steps(2), image=image)
             for task, image in (("task-a", old), ("task-a", old), ("task-b", old), ("task-b", new), ("task-c", None))]
    summaries = [records.trial(jobs_dir / "j" / name, trials) for name in names]
    assert [summary["image_id"] for summary in summaries] == [old, old, old, new, None]
    assert records.image_changes(summaries) == {"task-b": [old, new]}
    # Against the images the earlier study ran: task-a's moved too; a task with no expected image is checked alone.
    assert records.image_changes(summaries, expected={"task-a": new, "task-b": old}) == {
        "task-a": [old, new], "task-b": [old, new]}


def test_without_a_record_harbors_figures_stand_and_an_unfaithful_branch_is_not_a_sample(tmp_path):
    jobs_dir, trials = tmp_path / "jobs", tmp_path / "trials"
    name = trial(jobs_dir, "j", "task-a", reward=1.0, trials_root=trials, cost=0.5, seconds=99.0,
                 audit_flags=["branch_unfaithful:request_differs"])
    summary = records.trial(jobs_dir / "j" / name, trials)
    assert summary["settled"] is False and summary["cost_usd"] == 0.5 and summary["seconds"] == 99.0
    assert summary["faithful"] is False and not records.usable(summary)
    ungraded = trial(jobs_dir, "j", "task-a", reward=None, trials_root=trials, exception="AgentSetupError")
    assert not records.usable(records.trial(jobs_dir / "j" / ungraded, trials))
    # Graded, but its record holds no step of the agent: Taste never started it.
    untouched = records.trial(jobs_dir / "j" / trial(jobs_dir, "j", "task-a", reward=0.0, trials_root=trials,
                                                     steps=[]), trials)
    assert untouched["agent_started"] is False and not records.usable(untouched)
    assert summary["agent_started"] is None  # no record: Harbor's figures stand


def test_a_trial_keeps_its_cache_reads_cache_writes_and_reasoning_tokens(tmp_path):
    """Per-step metrics carry the prompt's cache split; the state keeps the trial's sums, so a pilot can
    check caching and cost without the machines' call logs. Harbor alone has no split: None."""
    jobs_dir, trials = tmp_path / "jobs", tmp_path / "trials"
    name = trial(jobs_dir, "j", "task-a", reward=0.0, trials_root=trials, steps=agent_steps(3), cache=(900, 80, 40))
    summary = records.trial(jobs_dir / "j" / name, trials)
    assert (summary["cache_read_tokens"], summary["cache_write_tokens"], summary["reasoning_tokens"]) == (2700, 240, 120)
    bare = records.trial(jobs_dir / "j" / trial(jobs_dir, "j", "task-a", reward=0.0, trials_root=trials), trials)
    assert (bare["cache_read_tokens"], bare["cache_write_tokens"], bare["reasoning_tokens"]) == (None, None, None)


def test_a_taste_trial_whose_record_was_never_settled_is_not_a_sample(tmp_path):
    """Graded and faithful, but Taste's record was never read (the settlement race of Phase 1, 43 of
    16,767 trials): its exit status and submission are unknown, so the driver must retry it rather
    than end the episode as "no submission"."""
    jobs_dir, trials = tmp_path / "jobs", tmp_path / "trials"
    lost = records.trial(jobs_dir / "j" / trial(jobs_dir, "j", "task-a", reward=0.0, trials_root=trials,
                                                audit_flags=["settlement_failed:ResourceCleanupError"]), trials)
    assert lost["settled"] is False and lost["faithful"] is None and not records.usable(lost)
    plain = records.trial(jobs_dir / "j" / trial(jobs_dir, "j", "task-a", reward=0.0, trials_root=trials), trials)
    assert records.usable(plain)  # no record and no flag: Harbor's figures stand, as before
    flagged = records.trial(jobs_dir / "j" / trial(jobs_dir, "j", "task-a", reward=0.0, trials_root=trials,
                                                   steps=agent_steps(2), audit_flags=["settlement_failed:RuntimeError"]),
                            trials)
    assert flagged["settled"] is True and not records.usable(flagged)


def test_time_is_the_agents_execution_by_harbors_clock_less_the_prefix_for_a_branch(tmp_path):
    jobs_dir, trials = tmp_path / "jobs", tmp_path / "trials"
    never = trial(jobs_dir, "j", "task-a", reward=0.0, trials_root=trials, seconds=None, exception="AgentSetupError")
    summary = records.trial(jobs_dir / "j" / never, trials)
    assert (summary["seconds"], summary["prefix_seconds"], summary["charged_seconds"]) == (None, None, None)
    plain = records.trial(jobs_dir / "j" / trial(jobs_dir, "j", "task-a", reward=0.0, trials_root=trials,
                                                  seconds=200.0), trials)
    assert (plain["seconds"], plain["prefix_seconds"], plain["charged_seconds"]) == (200.0, None, 200.0)
    assert records.charged(100.0, 30.0) == 70.0 and records.charged(20.0, 30.0) == 0.0
    assert records.charged(50.0, None) == 50.0 and records.charged(None, 5.0) is None


def test_a_checker_trials_verdict_is_the_submission_it_handed_back_on_a_faithful_copy(tmp_path):
    jobs_dir, trials = tmp_path / "jobs", tmp_path / "trials"
    name = trial(jobs_dir, "j", "task-a", reward=1.0, trials_root=trials, handed_back=submission("not_done"))
    summary = records.trial(jobs_dir / "j" / name, trials)
    reply = records.verdict(summary)
    assert reply["verdict"] == "not_done" and reply["confidence"] == 0.8 and reply["evidence"]["expected"]
    # A copy of the final state that was not faithful was not what the agent left: no verdict.
    unfaithful = trial(jobs_dir, "j", "task-a", reward=1.0, trials_root=trials, handed_back=submission("done"),
                       branch={"faithful": False})
    assert records.verdict(records.trial(jobs_dir / "j" / unfaithful, trials)) is None
    # A review that ended at a limit, a reply that is not a submission, no flat record: no verdict.
    limit = json.loads(submission("done"))
    limit.update(verdict=None, confidence=None, ended="steps")
    for handed_back in (json.dumps(limit), "I could not decide.", None):
        other = trial(jobs_dir, "j", "task-a", reward=0.0, trials_root=trials, handed_back=handed_back)
        assert records.verdict(records.trial(jobs_dir / "j" / other, trials)) is None


def test_a_job_is_read_once_its_trials_are_written_or_harbor_closes_it(tmp_path):
    jobs_dir, trials = tmp_path / "jobs", tmp_path / "trials"
    trial(jobs_dir, "j", "task-a", reward=0.0, trials_root=trials)
    (jobs_dir / "j" / "running").mkdir()
    (jobs_dir / "j" / "running" / "result.json").write_text(json.dumps({"trial_name": "running"}))
    assert len(records.job_trials(jobs_dir / "j", trials)) == 1 and not records.job_finished(jobs_dir / "j")
    finish(jobs_dir, "j")
    assert records.job_finished(jobs_dir / "j")


def test_budgets_are_medians_of_runs_from_scratch_or_the_calibration_reports(tmp_path):
    runs = [{"cost_usd": c, "seconds": s} for c, s in ((0.5, 300.0), (0.9, 900.0), (0.7, 600.0))]
    assert records.budget(runs) == {"usd": 0.7, "seconds": 600.0, "runs": 3}
    assert records.budget([{"cost_usd": None, "seconds": 1.0}]) is None
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"per_task": [
        {"task": "a", "kept": True, "median_usd": 0.4, "median_agent_seconds": 420.0, "median_trial_seconds": 500.0,
         "attempts": 2},
        {"task": "b", "kept": False, "median_usd": 0.1, "median_agent_seconds": None, "median_trial_seconds": 90.0,
         "attempts": 2}]}))
    # The agent's time only: a whole trial's time is another clock, so "b" has none.
    assert records.calibration_budgets(path) == {"a": {"kept": True, "usd": 0.4, "seconds": 420.0, "runs": 2},
                                                 "b": {"kept": False, "usd": 0.1, "seconds": None, "runs": 2}}


def test_rules_rewind_to_the_last_passing_tests_or_before_the_last_large_edit_or_the_start():
    steps = records.agent_view(_nested(agent_steps(10, tests_pass_at=(3, 6), large_edit_at=(2, 8))))["steps"]
    assert rules.rewind_points(steps) == {"visible_tests": 6, "before_large_edit": 7, "start": 0,
                                          "rules": 6, "rule": "visible_tests"}
    steps = records.agent_view(_nested(agent_steps(10, large_edit_at=(2, 8))))["steps"]
    chosen = rules.rewind_points(steps)
    assert (chosen["visible_tests"], chosen["rules"], chosen["rule"]) == (None, 7, "before_large_edit")
    chosen = rules.rewind_points(records.agent_view(_nested(agent_steps(5)))["steps"])
    assert (chosen["rules"], chosen["rule"]) == (0, "start")
    assert rules.features(steps) == {"run_length": 10, "visible_tests_passed": False, "edits": 2}
    assert rules.rewind_points([])["rules"] == 0


def _nested(steps):
    run = [{"source": "user", "message": "task"}] + [
        {"source": "agent", "message": message, "observation": {"results": [
            {"content": output, "extra": {"command": command, "returncode": code}}]}}
        for command, code, output, message in steps]
    return {"subagent_trajectories": [{"agent": {"name": "mini-swe-agent"}, "steps": run}]}


def test_a_swe_agent_runs_submission_is_the_patch_between_its_markers():
    patch = "diff --git a/src/app.py b/src/app.py\n+fix\n"
    marker = "<<SWE_AGENT_SUBMISSION>>"
    run = [{"source": "user", "message": "task"},
           {"source": "agent", "message": "Submitting.", "observation": {"results": [
               {"content": f"{marker}\n{patch}{marker}", "extra": {"command": "submit", "returncode": 0}},
               {"content": "", "extra": {"command": "_state_anthropic", "returncode": 0}},
               {"content": patch, "extra": {"command": "# swe-rex: read_file /root/model.patch",
                                            "returncode": 0}}]}}]
    nested = {"subagent_trajectories": [{"agent": {"name": "swe-agent"}, "steps": run,
                                         "extra": {"exit": {"exit_status": "Submitted", "stopped_by": ""}}}]}
    assert records.agent_view(nested)["submission"] == patch


def test_a_record_says_whether_the_checkpoint_its_trial_saved_is_partial(tmp_path):
    # Over the size cap a checkpoint lacks paths a restore would need: Taste refuses to restore it.
    jobs_dir, trials = tmp_path / "jobs", tmp_path / "trials"
    saved = {"saved": True, "partial": True, "bytes": 536870912}
    over = trial(jobs_dir, "j", "task-a", reward=0.0, trials_root=trials,
                 branch={"faithful": True, "checkpoint": "cp-1", "saved_checkpoint": saved})
    whole = trial(jobs_dir, "j", "task-a", reward=0.0, trials_root=trials,
                  branch={"faithful": True, "checkpoint": "cp-2", "saved_checkpoint": {**saved, "partial": False}})
    unsaved = trial(jobs_dir, "j", "task-a", reward=0.0, trials_root=trials, branch={"faithful": True})
    found = [records.trial(jobs_dir / "j" / name, trials)["checkpoint_partial"] for name in (over, whole, unsaved)]
    assert found == [True, False, None]
