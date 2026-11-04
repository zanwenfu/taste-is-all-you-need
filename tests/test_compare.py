"""Comparisons for the rule-loss study (studies/context_rot/compare.py). Synthetic rows only."""

from __future__ import annotations

import pytest

from studies.context_rot import compare


def test_wilson_interval_matches_a_known_value():
    low, high = compare.wilson(45, 75)
    assert (low, high) == (48.7, 70.3)
    assert compare.wilson(0, 0) == (None, None)


def test_effective_retention_is_the_share_of_the_gap_recovered():
    assert compare.effective_retention(60.0, 40.0, 100.0) == pytest.approx(100 * 20 / 60)
    assert compare.effective_retention(50.0, 50.0, 50.0) is None


def test_a_rate_keeps_its_exact_value_for_later_arithmetic():
    rows = [{"x": True}] * 2 + [{"x": False}] + [{"x": None}]
    result = compare.rate(rows, "x")
    assert result["pct"] == 66.7 and result["exact"] == pytest.approx(200 / 3)
    assert compare.rate([], "x")["exact"] is None


def test_effective_retention_in_the_tables_uses_exact_rates_not_rounded_ones():
    # Arm 4/6, floor 1/6, ceiling 5/6: exactly 75.0; from rates rounded to 66.7, 16.7 and 83.3 it would be 75.1.
    baseline = {f"c{i}": {"id": f"c{i}", "full_without_sc_compliant": i < 1} for i in range(6)}
    rows = [{"id": f"c{i}", "haystack_id": "h", "prompt": "p", "k": 2, "retention": [True, True],
             "compaction_compliant": i < 4, "compaction_upper_compliant": i < 5} for i in range(6)]
    entry = compare.repeated_table(rows, baseline)["p"][2]
    exact = 100 * (4 / 6 - 1 / 6) / (5 / 6 - 1 / 6)
    assert entry["compaction_effective_retention"] == round(exact, 1) == 75.0


def test_the_median_of_an_even_count_is_the_mean_of_the_middle_two():
    assert compare.median([4, 1, 3, 2]) == 2.5
    assert compare.median([5, 1, 3]) == 3
    assert compare.median([]) is None


def _row(i, hay, base, arm):
    return ({"id": i, "haystack_id": hay, "compaction_compliant": base},
            {"id": i, "haystack_id": hay, "compaction_compliant": arm})


def test_paired_comparison_counts_changes_and_gives_an_exact_sign_test():
    pairs = [_row(f"h{n % 3}:{n}", f"h{n % 3}", False, True) for n in range(10)] + \
            [_row("h0:x", "h0", True, True), _row("h1:y", "h1", True, None)]
    base = {b["id"]: b for b, _ in pairs}
    arm = {a["id"]: a for _, a in pairs}
    result = compare.paired(base, arm)
    assert (result["better"], result["worse"], result["same"], result["unreadable"]) == (10, 0, 1, 1)
    assert result["sign_test_p"] == pytest.approx(2 / 2 ** 10)


def test_cluster_bootstrap_resamples_whole_histories():
    rows = [{"haystack_id": "h1", "x": True}] * 15 + [{"haystack_id": "h2", "x": False}] * 15
    low, high = compare.cluster_interval(rows, "x", draws=400, seed=1)
    assert low == 0.0 and high == 100.0  # with two histories, a resample can hold only one of them
    same = [{"haystack_id": f"h{i}", "x": i % 2 == 0} for i in range(40) for _ in range(2)]
    low, high = compare.cluster_interval(same, "x", draws=400, seed=1)
    assert low < 50 < high


def test_repeated_rounds_are_summarized_per_k_with_the_floor_from_the_baseline():
    baseline = {"a": {"id": "a", "full_without_sc_compliant": False},
                "b": {"id": "b", "full_without_sc_compliant": False}}
    rows = []
    for k, kept in ((2, [True, True]), (4, [True, False, False, False])):
        for case in ("a", "b"):
            rows.append({"id": case, "haystack_id": "h1", "prompt": "pi_mono", "k": k, "retention": kept,
                         "compaction_compliant": k == 2, "compaction_upper_compliant": True,
                         "user_record_compliant": True, "user_record_upper_compliant": True,
                         "recent_control_compliant": False, "recent_control_upper_compliant": True})
    table = compare.repeated_table(rows, baseline)
    two, four = table["pi_mono"][2], table["pi_mono"][4]
    assert two["kept_after_last_round_pct"] == 100.0 and four["kept_after_last_round_pct"] == 0.0
    assert four["kept_by_round_pct"] == [100.0, 0.0, 0.0, 0.0]
    assert two["compaction_pct"] == 100.0 and four["compaction_pct"] == 0.0
    assert four["user_record_pct"] == 100.0 and four["recent_control_pct"] == 0.0
    assert four["floor_pct"] == 0.0 and four["user_record_effective_retention"] == 100.0
    assert four["cases"] == 2


def test_two_judges_are_compared_case_by_case():
    a = {"1": True, "2": True, "3": False, "4": None}
    b = {"1": True, "2": False, "3": False, "4": True}
    result = compare.judge_agreement(a, b)
    assert result["both_clear"] == 3 and result["agree_pct"] == pytest.approx(200 / 3)
    assert -1 <= result["kappa"] <= 1
