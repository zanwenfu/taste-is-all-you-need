"""The go/no-go report reads Harbor results and Taste's settled records, nothing else."""

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "go_no_go_report", Path(__file__).resolve().parents[1] / "scripts" / "go_no_go_report.py")
report_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report_module)


def trial(jobs, trials, arm, task, run, *, reward, services="all", subs=(), token=True):
    name = f"{task}__{arm}{run}"
    (jobs / arm / name).mkdir(parents=True)
    taste = {"configuration": {"services": services}, "audit_flags": [], "stop_reason": "complete"}
    if token:
        taste["trial"] = name
        nested = {"subagent_trajectories": [
            {"agent": {"name": agent}, "steps": [{"metrics": {"cost_usd": cost}}],
             "extra": {"run_id": f"r{i}", "phase": phase, "exit": {"exit_status": status, "stopped_by": stop}}}
            for i, (agent, cost, phase, status, stop) in enumerate(subs)]}
        (trials / name / "controller").mkdir(parents=True)
        (trials / name / "controller" / "trajectory.json").write_text(json.dumps(nested))
    result = {"task_name": "terminal-bench/" + task, "trial_name": name,
              "verifier_result": {"rewards": {"reward": reward}},
              "agent_result": {"metadata": {"taste": taste}}}
    (jobs / arm / name / "result.json").write_text(json.dumps(result))


@pytest.fixture
def scene(tmp_path):
    jobs, trials = tmp_path / "jobs", tmp_path / "trials"
    for run in (1, 2):
        trial(jobs, trials, "alone", "easy", run, reward=1.0, services="none",
              subs=[("mini-swe-agent", 0.01, "failed", "Submitted", "")])
        trial(jobs, trials, "alone", "hard", run, reward=0.0, services="none",
              subs=[("mini-swe-agent", 0.02, "failed", "LimitsExceeded", "")])
    trial(jobs, trials, "taste", "easy", 1, reward=1.0, subs=[
        ("taste-coordinator", 0.03, None, None, ""), ("mini-swe-agent", 0.01, "delivered", "Submitted", ""),
        ("taste-monitor", 0.02, "delivered", None, "")])
    trial(jobs, trials, "taste", "easy", 2, reward=1.0, subs=[
        ("taste-coordinator", 0.03, None, None, ""), ("mini-swe-agent", 0.01, "failed", "Submitted", ""),
        ("mini-swe-agent", 0.01, "delivered", "Submitted", ""), ("taste-monitor", 0.02, "delivered", None, "")])
    trial(jobs, trials, "taste", "hard", 1, reward=1.0, subs=[
        ("taste-coordinator", 0.05, None, None, ""), ("mini-swe-agent", 0.02, "failed", "Stopped", "monitor_wrong"),
        ("mini-swe-agent", 0.03, "delivered", "Submitted", "")])
    trial(jobs, trials, "taste", "hard", 2, reward=0.0, token=False)
    return jobs, trials


def test_each_arm_is_summarized_from_its_own_records(scene):
    jobs, trials = scene
    result = report_module.report([("alone", jobs / "alone"), ("taste", jobs / "taste")], trials)
    alone, taste = result["arms"]["alone"], result["arms"]["taste"]
    assert (alone["solved"], alone["trials"], alone["settled"]) == (2, 4, 4)
    assert alone["spend_usd"]["agent"] == pytest.approx(0.06) and alone["usd_per_solved_task"] == pytest.approx(0.03)
    # Alone nothing certifies, so nothing counts as refused.
    assert alone["refused_submissions_in_solved"] == 0 and alone["stopped_by_monitor"] == 0
    assert (taste["solved"], taste["trials"], taste["settled"]) == (3, 4, 3)
    assert taste["spend_usd"]["coordinator"] == pytest.approx(0.11)
    assert taste["stopped_by_monitor"] == 1 and taste["refused_submissions_in_solved"] == 1
    assert taste["stops"] == {"monitor_wrong": 1} and taste["exits"] == {"Stopped": 1, "Submitted": 4}
    assert taste["claimed_complete"] == 4 and taste["claimed_complete_unsolved"] == 1
    assert alone["exits"] == {"LimitsExceeded": 2, "Submitted": 2} and alone["stops"] == {}


def test_arms_are_compared_task_by_task(scene):
    jobs, trials = scene
    paired = report_module.report([("alone", jobs / "alone"), ("taste", jobs / "taste")], trials)["paired"]
    assert (paired["tasks"], paired["other_better"], paired["base_better"], paired["same"]) == (2, 1, 0, 1)
    assert paired["per_task"]["hard"] == {"base": 0.0, "other": 0.5}
    assert paired["sign_test_p"] == 1.0


@pytest.mark.parametrize("wins,losses,p", [(0, 0, 1.0), (5, 0, 0.0625), (6, 1, 0.125), (3, 3, 1.0)])
def test_the_sign_test_is_exact(wins, losses, p):
    assert report_module.sign_test(wins, losses) == pytest.approx(p)


def test_the_markdown_names_every_arm(scene, capsys):
    jobs, trials = scene
    report_module.main(["--arm", f"alone={jobs / 'alone'}", "--arm", f"taste={jobs / 'taste'}",
                        "--trials", str(trials)])
    out = capsys.readouterr().out
    assert "| alone | 2 | 4 |" in out and "| taste | 3 | 4 |" in out and "sign test" in out
