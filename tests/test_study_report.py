"""The study's analysis, checked on synthetic trial records with known answers."""

import importlib.util
import json
import random
from pathlib import Path

import pytest

from tests.test_go_no_go_report import trial

_SPEC = importlib.util.spec_from_file_location(
    "study_report", Path(__file__).resolve().parents[1] / "scripts" / "study_report.py")
study = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(study)


@pytest.fixture
def records(tmp_path):
    """Ten tasks, three runs each: the base arm never solves; one arm solves six tasks every
    time; another solves exactly what the base solves."""
    jobs, trials = tmp_path / "jobs", tmp_path / "trials"
    for index in range(10):
        task = f"task-{index}"
        for run in (1, 2, 3):
            spend = [("mini-swe-agent", 0.01, "delivered", "Submitted", "")]
            trial(jobs, trials, "base", task, run, reward=0.0, services="none", subs=spend)
            trial(jobs, trials, "better", task, run, reward=1.0 if index < 6 else 0.0, subs=spend)
            trial(jobs, trials, "same", task, run, reward=0.0, subs=spend)
    return jobs, trials


def test_paired_differences_intervals_and_exact_tests(records):
    jobs, trials = records
    result = study.report([("base", jobs / "base"), ("better", jobs / "better"), ("same", jobs / "same")],
                          [("better", "base"), ("same", "base")], trials, resamples=2000, seed=7)
    better, same = result["comparisons"]
    assert better["tasks"] == 10 and better["mean_difference"] == pytest.approx(0.6)
    assert (better["other_better"], better["base_better"], better["same"]) == (6, 0, 4)
    # Six tasks differ, all one way: exactly 2 of the 2**6 sign assignments are as extreme.
    assert better["permutation_p"] == pytest.approx(2 / 64) and better["sign_test_p"] == pytest.approx(2 / 64)
    low, high = better["bootstrap_95"]
    assert 0 < low <= 0.6 <= high <= 1
    assert same["mean_difference"] == 0 and same["permutation_p"] == 1.0 and same["bootstrap_95"] == [0, 0]
    # Holm over the two registered comparisons.
    assert better["holm_p"] == pytest.approx(2 * 2 / 64) and same["holm_p"] == 1.0
    assert result["arms"]["better"]["solved"] == 18 and result["arms"]["base"]["solved"] == 0


def test_the_same_records_and_seed_give_the_same_report(records):
    jobs, trials = records
    arms = [("base", jobs / "base"), ("better", jobs / "better")]
    first = study.report(arms, [("better", "base")], trials, resamples=500, seed=3)
    again = study.report(arms, [("better", "base")], trials, resamples=500, seed=3)
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_holm_steps_down_and_never_decreases():
    assert study.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert study.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_many_differing_tasks_use_random_sign_flips():
    differences = [0.5] * 30
    p = study.permutation_p(differences, 999, random.Random(0))
    # Thirty tasks all one way: no random flip of signs is as extreme, so (0 + 1) / (999 + 1).
    assert p == pytest.approx(1 / 1000)


def test_a_comparison_must_name_given_arms(records):
    jobs, trials = records
    with pytest.raises(ValueError, match="names an arm"):
        study.report([("base", jobs / "base")], [("missing", "base")], trials)
