"""The calibration report keeps mid-difficulty tasks with long runs, from Harbor results and Taste records."""

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "calibration_report", Path(__file__).resolve().parents[1] / "scripts" / "calibration_report.py")
report_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report_module)


def trial(jobs, trials, job, task, run, *, reward, steps=(), cost=0.01, exception=None, record=True,
          harbor_cost=None, seconds=(0, 600), benchmark="swebench-pro"):
    """One Harbor trial of mini-swe-agent alone; `steps` is the model calls of each agent run."""
    name = f"{task}__{job}{run}"
    (jobs / job / name).mkdir(parents=True)
    taste = {"configuration": {"services": "none"}, "audit_flags": [], "stop_reason": "generation_bound"}
    if record:
        taste["trial"] = name
        subs = [{"agent": {"name": "taste-coordinator"}, "steps": [
            {"source": "agent", "llm_call_count": 1, "metrics": {"cost_usd": 0.0}}], "extra": {}}]
        for index, calls in enumerate(steps):
            subs.append({"agent": {"name": "mini-swe-agent"},
                         "steps": [{"source": "user"}] + [
                             {"source": "agent", "llm_call_count": 1, "metrics": {"cost_usd": cost / calls / len(steps)}}
                             for _ in range(calls)],
                         "extra": {"run_id": f"r{index}", "phase": "report_accepted",
                                   "exit": {"exit_status": "Submitted", "stopped_by": ""}}})
        subs.append({"agent": {"name": "taste-monitor"}, "steps": [{"source": "system"}], "extra": {}})
        (trials / name / "controller").mkdir(parents=True)
        (trials / name / "controller" / "trajectory.json").write_text(json.dumps({"subagent_trajectories": subs}))
    start, end = seconds
    result = {"task_name": f"{benchmark}/{task}", "trial_name": name,
              "verifier_result": None if reward is None else {"rewards": {"reward": reward}},
              "exception_info": {"exception_type": exception} if exception else None,
              "agent_result": {"cost_usd": harbor_cost, "metadata": {"taste": taste}},
              "started_at": f"2026-10-07T10:{start // 60:02d}:{start % 60:02d}Z",
              "finished_at": f"2026-10-07T10:{end // 60:02d}:{end % 60:02d}Z",
              "agent_execution": {"started_at": f"2026-10-07T10:{start // 60:02d}:{start % 60:02d}Z",
                                  "finished_at": f"2026-10-07T10:{end // 60 - 1:02d}:{end % 60:02d}Z"}}
    (jobs / job / name / "result.json").write_text(json.dumps(result))


@pytest.fixture
def scene(tmp_path):
    jobs, trials = tmp_path / "jobs", tmp_path / "trials"
    trial(jobs, trials, "cal-a", "middle", 1, reward=1.0, steps=[20])
    trial(jobs, trials, "cal-a", "middle", 2, reward=0.0, steps=[30])
    trial(jobs, trials, "cal-a", "easy", 1, reward=1.0, steps=[25])
    trial(jobs, trials, "cal-a", "easy", 2, reward=1.0, steps=[25])
    trial(jobs, trials, "cal-a", "short", 1, reward=1.0, steps=[5])
    trial(jobs, trials, "cal-a", "short", 2, reward=0.0, steps=[8])
    # A second job holds the other attempts of two tasks.
    trial(jobs, trials, "cal-b", "hard", 1, reward=0.0, steps=[40], exception="AgentTimeoutError",
          seconds=(0, 3000))
    trial(jobs, trials, "cal-b", "hard", 2, reward=0.0, steps=[35])
    trial(jobs, trials, "cal-b", "flaky", 1, reward=1.0, steps=[18])
    trial(jobs, trials, "cal-b", "flaky", 2, reward=None, exception="OSError", record=False)
    trial(jobs, trials, "cal-b", "unrecorded", 1, reward=1.0, record=False, harbor_cost=0.5)
    trial(jobs, trials, "cal-b", "unrecorded", 2, reward=0.0, record=False, harbor_cost=0.25)
    return jobs, trials


def per_task(result):
    return {task["task"]: task for task in result["per_task"]}


def test_mid_difficulty_tasks_with_long_runs_are_kept(scene):
    jobs, trials = scene
    result = report_module.report([jobs / "cal-a", jobs / "cal-b"], trials)
    tasks = per_task(result)
    assert [task for task, value in tasks.items() if value["kept"]] == ["middle"]
    assert tasks["middle"]["steps_per_run"] == [20, 30] and tasks["middle"]["median_steps"] == 25
    assert tasks["easy"]["reason"] == "solved 2/2" and tasks["hard"]["reason"] == "solved 0/2"
    assert tasks["short"]["reason"] == "median 6.5 steps"
    assert tasks["unrecorded"]["reason"] == "steps unknown"
    assert result["kept"] == 1 and result["tasks"] == 6


def test_an_ungraded_trial_is_not_an_attempt_and_is_listed(scene):
    jobs, trials = scene
    result = report_module.report([jobs / "cal-a", jobs / "cal-b"], trials)
    flaky = per_task(result)["flaky"]
    # One graded success out of one graded attempt: 100%, so not kept, and the
    # OSError trial is named for a rerun rather than counted as a failure.
    assert (flaky["attempts"], flaky["solved"], flaky["reason"]) == (1, 1, "solved 1/1")
    assert flaky["ungraded"] == [{"trial": "flaky__cal-b2", "exception": "OSError"}]
    assert result["ungraded"] == [{"task": "flaky", "trial": "flaky__cal-b2", "job": "cal-b", "exception": "OSError"}]
    assert (result["trials"], result["graded"]) == (12, 11)


def test_a_timed_out_run_is_graded_and_counted(scene):
    jobs, trials = scene
    hard = per_task(report_module.report([jobs / "cal-a", jobs / "cal-b"], trials))["hard"]
    assert (hard["attempts"], hard["timeouts"], hard["max_trial_seconds"]) == (2, 1, 3000)


def test_dollars_come_from_the_record_or_else_from_harbor(scene):
    jobs, trials = scene
    result = report_module.report([jobs / "cal-a", jobs / "cal-b"], trials)
    tasks = per_task(result)
    assert tasks["middle"]["usd_total"] == pytest.approx(0.02) and tasks["middle"]["usd_per_trial"] == pytest.approx(0.01)
    assert tasks["unrecorded"]["usd_total"] == pytest.approx(0.75)
    # The recovery budget's unit: the median graded run.
    assert tasks["unrecorded"]["median_usd"] == pytest.approx(0.375)
    assert tasks["flaky"]["median_usd"] == pytest.approx(0.01)
    # The OSError trial has neither a record nor a Harbor cost.
    assert tasks["flaky"]["usd_unknown_trials"] == 1 and result["usd_unknown_trials"] == 1
    assert result["usd_total"] == pytest.approx(9 * 0.01 + 0.75)


def test_only_the_agents_model_calls_are_steps(tmp_path):
    jobs, trials = tmp_path / "jobs", tmp_path / "trials"
    trial(jobs, trials, "cal", "two-runs", 1, reward=1.0, steps=[12, 9])
    row = report_module.load_jobs([jobs / "cal"], trials)[0]
    # The coordinator's fixed plan and the monitor are not the agent's calls.
    assert row["steps"] == [12, 9]
    assert (row["trial_seconds"], row["agent_seconds"]) == (600, 540)


def test_the_selection_rule_can_be_changed(scene):
    jobs, trials = scene
    result = report_module.report([jobs / "cal-a", jobs / "cal-b"], trials, low=0.0, high=1.0, min_steps=20)
    assert sorted(task["task"] for task in result["per_task"] if task["kept"]) == ["easy", "hard", "middle"]
    assert result["rule"] == {"low": 0.0, "high": 1.0, "min_steps": 20}


def test_the_command_line_writes_the_kept_tasks_and_the_full_result(scene, tmp_path, capsys):
    jobs, trials = scene
    kept, full = tmp_path / "kept.txt", tmp_path / "full.json"
    report_module.main([str(jobs / "cal-a"), str(jobs / "cal-b"), "--trials", str(trials),
                        "--keep", str(kept), "--json", str(full)])
    out = capsys.readouterr().out
    assert kept.read_text() == "middle\n"
    assert json.loads(full.read_text())["kept"] == 1
    assert "| middle | 1/2 | 20, 30 | 0.0100 | 600 | yes |  |" in out
    assert "| unrecorded | 1/2 | ? |" in out
    assert "flaky (cal-b/flaky__cal-b2): OSError" in out
    assert "Outcomes (solved/attempts: tasks): 0/2: 1, 1/1: 1, 1/2: 3, 2/2: 1" in out
