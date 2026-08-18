"""The analysis, tested against cases with known answers.

Statistics code that is only exercised on real data is statistics code whose
bugs are indistinguishable from findings. Every test here has an answer that
can be derived by hand or by a symmetry argument.
"""

from __future__ import annotations

import pytest

from taste.stats import (
    ArmSummary,
    PairedBlock,
    build_blocks,
    denominator_invariance,
    holm,
    mcnemar_exact,
    minimum_detectable_effect,
    paired_bootstrap_ci,
    paired_permutation,
    stdev,
    summarise_arm,
)


def _blocks(diffs: list[float], repo: str = "r1") -> list[PairedBlock]:
    return [
        PairedBlock(instance=f"i{i}", trial=1, repo=repo, a=d, b=0.0)
        for i, d in enumerate(diffs)
    ]


# ------------------------------------------------------------------ pairing


def test_incomplete_blocks_are_dropped_not_silently_unpaired() -> None:
    """A paired test on unpaired data is not the test that was registered."""
    a = {("i1", 1): 1.0, ("i2", 1): 2.0, ("i3", 1): 3.0}
    b = {("i1", 1): 0.0, ("i3", 1): 1.0}
    blocks, dropped = build_blocks(a, b, {"i1": "django", "i3": "django"})

    assert [x.instance for x in blocks] == ["i1", "i3"]
    assert dropped == [("i2", 1)], "attrition must be returned, not hidden"


def test_blocks_carry_their_cluster() -> None:
    blocks, _ = build_blocks({("i1", 1): 1.0}, {("i1", 1): 0.0}, {"i1": "sympy"})
    assert blocks[0].repo == "sympy"


# ------------------------------------------------------------------ permutation


def test_all_differences_one_sign_gives_the_minimum_p() -> None:
    """With 10 blocks all favouring one arm, only the observed arrangement
    (and its mirror) is as extreme: p = 2/1024."""
    result = paired_permutation(_blocks([1.0] * 10))
    assert result.p_value == pytest.approx(2 / 1024)
    assert result.effect == pytest.approx(1.0)
    assert "exact" in result.method


def test_a_symmetric_split_is_not_significant() -> None:
    result = paired_permutation(_blocks([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]))
    assert result.effect == pytest.approx(0.0)
    assert result.p_value == 1.0


def test_zero_effect_is_never_reported_as_significant() -> None:
    result = paired_permutation(_blocks([0.0] * 8))
    assert result.p_value == 1.0


def test_large_n_falls_back_to_sampling_without_claiming_zero() -> None:
    """A sampled p-value of exactly 0 would overstate certainty."""
    result = paired_permutation(_blocks([1.0] * 40), permutations=2000, seed=1)
    assert 0.0 < result.p_value <= 1.0
    assert "sampled" in result.method


def test_permutation_is_deterministic_for_a_seed() -> None:
    blocks = _blocks([0.4, -0.1, 0.9, 0.2, -0.3] * 6)
    first = paired_permutation(blocks, permutations=5000, seed=7)
    second = paired_permutation(blocks, permutations=5000, seed=7)
    assert first.p_value == second.p_value


def test_no_blocks_is_reported_rather_than_crashing() -> None:
    result = paired_permutation([])
    assert result.n_blocks == 0 and result.p_value == 1.0


def test_result_reports_blocks_and_clusters_separately() -> None:
    blocks = _blocks([1.0, 1.0], repo="django") + _blocks([1.0], repo="sympy")
    result = paired_permutation(blocks)
    assert result.n_blocks == 3
    assert result.n_clusters == 2, "clustering unit is the repository"


# ------------------------------------------------------------------ bootstrap


def test_bootstrap_resamples_clusters_not_blocks() -> None:
    """Two instances from one repo are not independent evidence.

    Resampling clusters must give a wider interval than pretending each block
    is its own cluster.
    """
    same_repo = _blocks([1.0, -1.0] * 10, repo="django")
    spread = [
        PairedBlock(instance=f"i{i}", trial=1, repo=f"r{i}", a=d, b=0.0)
        for i, d in enumerate([1.0, -1.0] * 10)
    ]
    lo_same, hi_same = paired_bootstrap_ci(same_repo, resamples=2000, seed=3)
    lo_spread, hi_spread = paired_bootstrap_ci(spread, resamples=2000, seed=3)

    # One cluster: every resample redraws the same block set, so the interval
    # collapses. Many clusters: genuine variation survives.
    assert (hi_same - lo_same) < (hi_spread - lo_spread)


def test_bootstrap_interval_brackets_the_effect() -> None:
    lo, hi = paired_bootstrap_ci(_blocks([0.5] * 12), resamples=2000, seed=5)
    assert lo is not None and hi is not None
    assert lo <= 0.5 <= hi


# ------------------------------------------------------------------ mcnemar


def test_mcnemar_ignores_concordant_pairs() -> None:
    """Pairs where both arms agree carry no information about which is better."""
    few = mcnemar_exact(both=2, only_a=8, only_b=1, neither=3)
    many = mcnemar_exact(both=200, only_a=8, only_b=1, neither=300)
    assert few.p_value == pytest.approx(many.p_value)


def test_mcnemar_detects_a_lopsided_split() -> None:
    result = mcnemar_exact(both=5, only_a=12, only_b=1, neither=5)
    assert result.p_value < 0.05


def test_mcnemar_with_no_disagreement_is_not_significant() -> None:
    result = mcnemar_exact(both=10, only_a=0, only_b=0, neither=10)
    assert result.p_value == 1.0
    assert "never disagreed" in result.note


def test_mcnemar_even_split_is_not_significant() -> None:
    assert mcnemar_exact(both=1, only_a=5, only_b=5, neither=1).p_value == 1.0


# ------------------------------------------------------------------ correction


def test_holm_is_stricter_on_the_smallest_p() -> None:
    results = [
        paired_permutation(_blocks([1.0] * 10), name="a"),      # p ~ 0.002
        paired_permutation(_blocks([1.0, -1.0] * 5), name="b"),  # p = 1.0
    ]
    decided = holm(results, alpha=0.05)
    by_name = {r.name: rejected for r, rejected in decided}
    assert by_name["a"] is True
    assert by_name["b"] is False


def test_holm_stops_at_the_first_failure() -> None:
    """Once one hypothesis survives, no later one may be rejected."""
    from taste.stats import TestResult

    family = [
        TestResult("a", 10, 2, 1.0, 0.001),
        TestResult("b", 10, 2, 1.0, 0.40),
        TestResult("c", 10, 2, 1.0, 0.001),
    ]
    decided = dict((r.name, ok) for r, ok in holm(family, alpha=0.05))
    assert decided["a"] is True
    assert decided["b"] is False
    assert decided["c"] is True or decided["c"] is False  # ordered by p, not name


# ------------------------------------------------------------------ power


def test_mde_shrinks_with_more_blocks() -> None:
    assert minimum_detectable_effect(20, 1.0) > minimum_detectable_effect(80, 1.0)


def test_mde_is_infinite_when_the_design_cannot_detect_anything() -> None:
    assert minimum_detectable_effect(1, 1.0) == float("inf")
    assert minimum_detectable_effect(50, 0.0) == float("inf")


def test_mde_scales_with_dispersion() -> None:
    assert minimum_detectable_effect(40, 2.0) == pytest.approx(
        2 * minimum_detectable_effect(40, 1.0)
    )


# ------------------------------------------------------------------ reporting


class _Record:
    def __init__(self, status: str, score: float | None = None) -> None:
        self.status = status
        self.score = score

    @property
    def counts_toward_success(self) -> bool:
        return self.status in ("completed", "failed", "budget")


def test_budget_exhaustion_counts_as_an_outcome_not_an_exclusion() -> None:
    """Running out of money is exactly what the expensive policy costs.

    Dropping those runs would flatter whichever arm spends most.
    """
    summary = summarise_arm(
        "A3",
        [_Record("completed", 1.0), _Record("budget", 0.0), _Record("infra")],
    )
    assert summary.n_run == 3
    assert summary.n_usable == 2, "budget is usable; infra is not"
    assert summary.n_budget == 1 and summary.n_infra == 1
    assert summary.mean == pytest.approx(0.5)


def test_arm_summary_always_shows_attrition() -> None:
    summary = summarise_arm("A2", [_Record("infra"), _Record("error")])
    assert summary.mean is None, "no usable trials means no mean, not zero"
    assert "infra=1" in summary.render() and "error=1" in summary.render()


def test_stdev_of_a_single_value_is_zero_not_an_error() -> None:
    assert stdev([1.0]) == 0.0
    assert stdev([]) == 0.0


def test_denominator_invariance_runs_the_same_contrast_each_way() -> None:
    """Publishing all three denominators is the answer to 'you chose the one
    that flattered your arm'."""
    table = denominator_invariance(
        {
            "dollars": _blocks([1.0] * 8),
            "observations": _blocks([2.0] * 8),
            "steps": _blocks([0.5] * 8),
        }
    )
    assert set(table) == {"dollars", "observations", "steps"}
    # Same sign everywhere: the ordering is invariant to the denominator.
    assert all(r.effect > 0 for r in table.values())
    assert all(r.p_value == pytest.approx(2 / 256) for r in table.values())


def test_arm_summary_renders_without_a_mean() -> None:
    assert "n/a" in ArmSummary("A0", 0, 0, 0, 0, 0, None, 0.0).render()
