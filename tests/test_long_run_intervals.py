"""Tests for studies.long_run.intervals: pooled obedience on shared tasks and intervals resampled over sessions."""

import json

from studies.long_run import intervals


def _summary(rule: str, by_task: dict, valid: bool = True, ended_by=None) -> dict:
    return {"valid": valid, "ended_by": ended_by, "outcomes": [{"task": t} for t in by_task],
            "scores": {rule: {"by_task": {t: {"obeyed": o, "applies": a} for t, (o, a) in by_task.items()}}}}


def test_shared_counts_use_only_tasks_both_reached_where_the_rule_applied():
    a = _summary("r", {"t1": (3, 4), "t2": (1, 2), "t3": (5, 5)})
    b = _summary("r", {"t1": (1, 4), "t2": (0, 0), "t4": (2, 2)})
    assert intervals.shared_counts(a, b, "r") == (3, 4, 1, 4)


def test_pooled_difference_sums_over_sessions_before_dividing():
    rows = [(3, 4, 1, 4), (1, 6, 0, 2)]
    assert abs(intervals.pooled_difference(rows) - ((4 / 10) - (1 / 6))) < 1e-12


def test_interval_is_fixed_by_its_seed_and_holds_the_point():
    rows = [(9, 10, 1, 10), (5, 10, 5, 10), (8, 10, 0, 10), (2, 10, 3, 10)]
    first = intervals.resampled_interval(rows, draws=2000, seed=7)
    again = intervals.resampled_interval(rows, draws=2000, seed=7)
    assert first == again
    low, high = first
    assert low <= intervals.pooled_difference(rows) <= high


def test_one_session_gives_no_spread():
    rows = [(3, 4, 1, 4)]
    low, high = intervals.resampled_interval(rows, draws=500, seed=1)
    assert low == high == intervals.pooled_difference(rows)


def test_invalid_summaries_are_left_out_and_counted(tmp_path):
    for name, arms in {"s1": {"x": _summary("r", {"t": (1, 1)}), "y": _summary("r", {"t": (0, 1)})},
                       "s2": {"x": _summary("r", {"t": (1, 1)}, valid=False), "y": _summary("r", {"t": (1, 1)})}}.items():
        for arm, summary in arms.items():
            (tmp_path / name / arm).mkdir(parents=True)
            (tmp_path / name / arm / "summary.json").write_text(json.dumps(summary))
    runs = intervals.load([tmp_path / "s1", tmp_path / "s2"])
    result = intervals.compare(runs, "x", "y", "r", draws=200, seed=0)
    assert result["sessions"] == 1
    assert result["left_out_invalid"] == 1
    assert result["obeyed_a"] == 1 and result["applies_a"] == 1 and result["obeyed_b"] == 0
