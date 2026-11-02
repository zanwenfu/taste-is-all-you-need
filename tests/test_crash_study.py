"""When each task's run is killed, and how a killed run is counted."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "crash_study", Path(__file__).resolve().parents[1] / "scripts" / "crash_study.py")
crash_study = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(crash_study)


def test_a_task_is_killed_at_a_repeatable_point_in_the_middle_of_its_usual_run():
    first = crash_study.kill_second("build-pmars", 1000.0)
    assert first == crash_study.kill_second("build-pmars", 1000.0)
    assert 200.0 <= first <= 800.0
    assert first != crash_study.kill_second("build-pov-ray", 1000.0)


def test_a_very_short_task_still_gets_time_to_start():
    assert crash_study.kill_second("quick", 10.0) == crash_study.FLOOR_SECONDS


def row(task, reward, before, **crash):
    return {"task": task, "reward": reward, "before": before, "exception": None, "stop": "complete", "flags": [],
            "crash": crash}


def test_only_a_kill_that_landed_counts_as_a_recovery():
    rows = [row("a", 1.0, 1.0, landed=True, resumed=True, commands_repeated=0, paid_calls_repeated=0,
                paid_calls_cut_off=1, recovery_seconds=2.0),
            row("b", 1.0, 1.0, landed=False, resumed=False)]
    summary = crash_study.summarize(rows)
    assert summary["kills_landed"] == 1 and summary["ended_before_the_kill"] == 1
    assert summary["resumed"] == 1 and summary["solved_after_a_kill"] == 1
    assert summary["paid_calls_cut_off"] == 1 and summary["paid_calls_repeated"] == 0


def test_a_killed_run_that_could_not_be_graded_is_a_failed_recovery_not_a_gap():
    broken = row("a", None, 0.5, landed=True, resumed=True, commands_repeated=0, paid_calls_repeated=0,
                 paid_calls_cut_off=0, recovery_seconds=2.0)
    broken["exception"] = "VerifierTimeoutError"
    never = row("b", 0.0, 0.0, landed=True, resumed=False)
    summary = crash_study.summarize([broken, never])
    assert summary["failed_recoveries"] == 2 and summary["not_resumed"] == 1 and summary["ungraded"] == 1


def test_missing_evidence_stops_the_counts_from_reading_as_zero():
    summary = crash_study.summarize([row("a", 1.0, 1.0, landed=True, resumed=True)])
    assert summary["evidence_missing"] == 1


def test_the_comparison_is_per_task_against_the_runs_that_were_not_killed():
    rows = [row("a", 1.0, 0.5, landed=True, resumed=True), row("b", 0.0, 1.0, landed=True, resumed=True),
            row("c", 1.0, 1.0, landed=False, resumed=False)]
    summary = crash_study.summarize(rows, resamples=200)
    assert summary["landed_tasks"]["mean_difference"] == -0.25
    assert summary["landed_tasks"]["better"] == 1 and summary["landed_tasks"]["worse"] == 1
    assert summary["all_tasks"]["tasks"] == 3


def test_a_pass_from_work_done_before_the_kill_is_not_a_recovery():
    died = row("a", 1.0, 1.0, landed=True, resumed=True, commands_repeated=0, paid_calls_repeated=0,
               paid_calls_cut_off=1, recovery_seconds=0.1)
    died["stop"] = None
    kept = row("b", 1.0, 1.0, landed=True, resumed=True, commands_repeated=0, paid_calls_repeated=0,
               paid_calls_cut_off=0, recovery_seconds=0.1)
    kept["stop"] = "complete"
    summary = crash_study.summarize([died, kept], resamples=200)
    assert summary["failed_recoveries"] == 1 and summary["passed_without_recovering"] == 1
    assert summary["recovered"] == 1
    assert summary["solved_after_a_kill"] == 1


def test_every_run_lands_in_exactly_one_class():
    clean = {"landed": True, "resumed": True, "paid_calls_repeated": 0}
    assert crash_study.classify(row("a", 1.0, 1.0, **clean)) == "recovered"
    assert crash_study.classify(row("a", 0.0, 1.0, **clean)) == "recovered", "recovering is not solving"
    assert crash_study.classify(row("a", 1.0, 1.0, landed=False, resumed=False)) == "ended_first"
    assert crash_study.classify(row("a", 1.0, 1.0, landed=True, resumed=False)) == "failed_recovery"


@pytest.mark.parametrize("stop", ["runtime_error", "cancelled", "planner_failed", None])
def test_a_goal_that_did_not_reach_an_end_of_its_own_is_never_a_recovery(stop):
    killed = row("a", 1.0, 1.0, landed=True, resumed=True)
    killed["stop"] = stop
    assert crash_study.classify(killed) == "failed_recovery"
    missed = row("a", 1.0, 1.0, landed=False, resumed=False)
    missed["stop"] = stop
    assert crash_study.classify(missed) == "broken"


@pytest.mark.parametrize("change", [{"exception": "ExceptionGroup"}, {"flags": ["goal_service_failed"]},
                                    {"flags": ["settlement_failed:FileNotFoundError"]}, {"reward": None}])
def test_any_sign_of_trouble_keeps_a_run_out_of_the_clean_classes(change):
    missed = {**row("a", 1.0, 1.0, landed=False, resumed=False), **change}
    assert crash_study.classify(missed) == "broken", "a broken run did not end before the kill"
    killed = {**row("a", 1.0, 1.0, landed=True, resumed=True), **change}
    assert crash_study.classify(killed) == "failed_recovery"


def test_broken_runs_are_counted_apart_and_left_out_of_the_comparison():
    rows = [row("a", 1.0, 1.0, landed=True, resumed=True), row("b", 1.0, 1.0, landed=False, resumed=False),
            {**row("c", None, 1.0, landed=False, resumed=False), "exception": "RuntimeError"}]
    summary = crash_study.summarize(rows, resamples=200)
    assert summary["runs"] == summary["recovered"] + summary["failed_recoveries"] + \
        summary["ended_before_the_kill"] + summary["broken"] == 3
    assert summary["broken_tasks"] == {"c": "RuntimeError"}
    assert summary["all_tasks"]["tasks"] == 2 and summary["all_tasks"]["mean_difference"] == 0


def test_the_crash_record_is_read_from_the_run_when_its_metadata_lost_it(tmp_path):
    (tmp_path / "token" / "controller").mkdir(parents=True)
    (tmp_path / "token" / "controller" / "crash.json").write_text('{"landed": true, "resumed": true}')
    result = {"agent_result": {"metadata": {"taste": {"trial": "token"}}}}
    assert crash_study.crash_record(result, tmp_path)["landed"] is True
    assert crash_study.crash_record({"agent_result": None}, tmp_path) == {}


def test_starts_are_spaced_and_wait_for_a_service_manager_that_has_fallen_behind():
    now = [0.0]
    answers = [False, False, True, True]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    gate = crash_study.LaunchGate(spacing=10.0, quick=lambda: answers.pop(0), clock=lambda: now[0], sleep=sleep)
    gate.wait()
    assert slept == [10.0, 10.0], "a slow manager holds the next start back"
    slept.clear()
    gate.wait()
    assert slept == [10.0], "starts stay spaced even when it is quick"

