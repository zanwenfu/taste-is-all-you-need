"""When each task's run is killed, and how a killed run is counted."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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
    return {"task": task, "reward": reward, "before": before, "exception": None, "crash": crash}


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
