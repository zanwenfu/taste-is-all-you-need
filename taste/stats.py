"""The analysis, pre-specified so it cannot be chosen after seeing the data.

Small-N agent studies fail review in predictable ways: a bare cross-arm mean
that hides listwise deletion, asymptotic tests on eight clusters, a p-value
per contrast with no correction, and no statement of what effect the design
could have detected. Every function here exists to make one of those
impossible.

**The unit of pairing is the (instance, trial) block; the unit of clustering
is the repository.** Not the test, which would treat 200 tests in one repo as
200 independent observations, and not the trial, which would pretend two
samples of the same instance are independent evidence about different tasks.

**The primary test is an exact stratified paired permutation.** With a sample
spanning eight to ten repositories, cluster-robust standard errors do not
behave — their asymptotics need far more clusters than a solo-author budget
buys. A permutation test needs no asymptotics: under the null that the arm
label carries no information, flipping labels within a block is exactly as
likely as the arrangement observed, so enumerating flips gives an exact
p-value at any N.

**A null result is a result**, provided the design can say what it would have
detected. :func:`minimum_detectable_effect` is therefore reported alongside
every non-significant contrast, and pre-registered before the sweep.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

# Fixed so an analysis rerun reproduces exactly. Published in the manifest.
DEFAULT_SEED = 20260815
DEFAULT_PERMUTATIONS = 100_000
DEFAULT_BOOTSTRAP = 10_000


@dataclass(frozen=True)
class PairedBlock:
    """One (instance, trial) pair of arm outcomes, with its cluster."""

    instance: str
    trial: int
    repo: str
    a: float
    b: float

    @property
    def difference(self) -> float:
        return self.a - self.b


@dataclass
class TestResult:
    name: str
    n_blocks: int
    n_clusters: int
    effect: float
    """Mean paired difference, a minus b."""
    p_value: float
    ci_low: float | None = None
    ci_high: float | None = None
    method: str = ""
    note: str = ""

    @property
    def significant_at(self) -> float | None:
        return self.p_value

    def render(self) -> str:
        ci = (
            f" [{self.ci_low:+.4f}, {self.ci_high:+.4f}]"
            if self.ci_low is not None
            else ""
        )
        return (
            f"{self.name}: effect={self.effect:+.4f}{ci} p={self.p_value:.4f} "
            f"(n={self.n_blocks} blocks, {self.n_clusters} clusters, {self.method})"
        )


# ------------------------------------------------------------------ pairing


def build_blocks(
    rows_a: dict[tuple[str, int], float],
    rows_b: dict[tuple[str, int], float],
    repos: dict[str, str],
) -> tuple[list[PairedBlock], list[tuple[str, int]]]:
    """Pair two arms by (instance, trial), dropping incomplete blocks.

    Returns ``(blocks, dropped)``. Block-wise deletion is deliberate: keeping
    a block where one arm is missing silently unpairs the design, and a paired
    test on unpaired data is not the test that was pre-registered. The dropped
    list is returned rather than logged so attrition can be reported per arm
    instead of disappearing.
    """
    blocks: list[PairedBlock] = []
    dropped: list[tuple[str, int]] = []
    for key in sorted(set(rows_a) | set(rows_b)):
        if key not in rows_a or key not in rows_b:
            dropped.append(key)
            continue
        instance, trial = key
        blocks.append(
            PairedBlock(
                instance=instance,
                trial=trial,
                repo=repos.get(instance, "unknown"),
                a=rows_a[key],
                b=rows_b[key],
            )
        )
    return blocks, dropped


# ------------------------------------------------------------------ the tests


def paired_permutation(
    blocks: Sequence[PairedBlock],
    *,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    name: str = "primary",
) -> TestResult:
    """Exact stratified paired permutation test on the mean difference.

    Under the null, the arm label within a block is exchangeable, so each
    block's difference is equally likely to carry either sign. Sampling sign
    flips (stratification is implicit — flips happen *within* blocks, and
    blocks nest inside repositories) gives a p-value with no distributional
    assumption and no cluster asymptotics.

    When the number of blocks is small enough to enumerate exhaustively
    (2^n ≤ permutations), every arrangement is evaluated and the p-value is
    exact rather than estimated.
    """
    differences = [b.difference for b in blocks]
    n = len(differences)
    clusters = len({b.repo for b in blocks})
    if n == 0:
        return TestResult(name, 0, 0, 0.0, 1.0, method="no blocks", note="nothing to test")

    observed = sum(differences) / n

    if n <= 20 and 2**n <= permutations:
        total = 2**n
        extreme = 0
        for mask in range(total):
            flipped = sum(
                -d if (mask >> i) & 1 else d for i, d in enumerate(differences)
            ) / n
            if abs(flipped) >= abs(observed) - 1e-12:
                extreme += 1
        p = extreme / total
        method = f"exact permutation ({total} arrangements)"
    else:
        rng = random.Random(seed)
        extreme = 0
        for _ in range(permutations):
            flipped = sum(d if rng.random() < 0.5 else -d for d in differences) / n
            if abs(flipped) >= abs(observed) - 1e-12:
                extreme += 1
        # Add-one smoothing: a sampled p-value of exactly 0 is not credible
        # and would overstate certainty.
        p = (extreme + 1) / (permutations + 1)
        method = f"sampled permutation ({permutations:,})"

    low, high = paired_bootstrap_ci(blocks, seed=seed)
    return TestResult(name, n, clusters, observed, p, low, high, method)


def paired_bootstrap_ci(
    blocks: Sequence[PairedBlock],
    *,
    resamples: int = DEFAULT_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> tuple[float | None, float | None]:
    """Percentile CI on the mean paired difference, resampling CLUSTERS.

    Resampling blocks would treat two instances from the same repository as
    independent; they are not, since they share code, fixtures and flakiness.
    Clusters are the resampling unit, which widens the interval honestly.
    """
    if not blocks:
        return None, None
    by_repo: dict[str, list[float]] = {}
    for block in blocks:
        by_repo.setdefault(block.repo, []).append(block.difference)
    repos = list(by_repo)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        drawn: list[float] = []
        for _ in repos:
            drawn.extend(by_repo[repos[rng.randrange(len(repos))]])
        if drawn:
            means.append(sum(drawn) / len(drawn))
    if not means:
        return None, None
    means.sort()
    lo = means[max(0, int((alpha / 2) * len(means)) - 1)]
    hi = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    return lo, hi


def mcnemar_exact(both: int, only_a: int, only_b: int, neither: int) -> TestResult:
    """Exact McNemar on discordant pairs, for binary outcomes.

    Concordant pairs carry no information about which arm is better, so the
    test conditions on the discordant ones and asks whether they split evenly.
    Exact binomial rather than chi-square, because the discordant count will
    be small.
    """
    discordant = only_a + only_b
    n = both + only_a + only_b + neither
    if discordant == 0:
        return TestResult(
            "mcnemar", n, 0, 0.0, 1.0, method="exact McNemar",
            note="no discordant pairs — the arms never disagreed",
        )

    # Two-sided exact binomial at p=0.5.
    k = min(only_a, only_b)
    tail = sum(math.comb(discordant, i) for i in range(k + 1)) / (2**discordant)
    p = min(1.0, 2 * tail)
    effect = (only_a - only_b) / n
    return TestResult(
        "mcnemar", n, 0, effect, p, method="exact McNemar",
        note=f"discordant {only_a}/{only_b}",
    )


def holm(results: Sequence[TestResult], alpha: float = 0.05) -> list[tuple[TestResult, bool]]:
    """Holm-Bonferroni over a pre-registered secondary family.

    Uniformly more powerful than Bonferroni and makes no independence
    assumption, which matters because these contrasts are computed from
    overlapping data.
    """
    ordered = sorted(results, key=lambda r: r.p_value)
    m = len(ordered)
    out: list[tuple[TestResult, bool]] = []
    rejected_so_far = True
    for index, result in enumerate(ordered):
        threshold = alpha / (m - index)
        rejected = rejected_so_far and result.p_value <= threshold
        rejected_so_far = rejected
        out.append((result, rejected))
    return out


# ------------------------------------------------------------------ power


def minimum_detectable_effect(
    n_blocks: int,
    sd_difference: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Smallest mean difference this design could detect, in outcome units.

    Reported with every non-significant result. Without it, "no effect" and
    "no power" are indistinguishable, and a reviewer is right to assume the
    second.
    """
    if n_blocks <= 1 or sd_difference <= 0:
        return float("inf")
    z_alpha = 1.959963985  # two-sided 0.05
    z_power = 0.8416212336  # 80%
    return (z_alpha + z_power) * sd_difference / math.sqrt(n_blocks)


def stdev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


# ------------------------------------------------------------------ reporting


@dataclass
class ArmSummary:
    """Per-arm description. Never a bare mean — attrition is always visible."""

    arm: str
    n_run: int
    n_usable: int
    n_infra: int
    n_budget: int
    n_error: int
    mean: float | None
    sd: float
    values: list[float] = field(default_factory=list)

    def render(self) -> str:
        mean = "n/a" if self.mean is None else f"{self.mean:.4f}"
        return (
            f"{self.arm:<18} run={self.n_run:>3} usable={self.n_usable:>3} "
            f"mean={mean:>8} sd={self.sd:>7.4f} "
            f"[infra={self.n_infra} budget={self.n_budget} error={self.n_error}]"
        )


def summarise_arm(arm: str, records: Sequence) -> ArmSummary:
    """Describe one arm's outcomes with attrition made explicit.

    Budget exhaustion counts as a genuine outcome, not an exclusion: running
    out of money is precisely what an expensive policy costs, and dropping
    those runs would flatter whichever arm spends most (intention-to-treat).
    Only infrastructure faults and harness errors are excluded.
    """
    usable = [r for r in records if getattr(r, "counts_toward_success", True)]
    values = [r.score for r in usable if getattr(r, "score", None) is not None]
    return ArmSummary(
        arm=arm,
        n_run=len(records),
        n_usable=len(usable),
        n_infra=sum(1 for r in records if getattr(r, "status", "") == "infra"),
        n_budget=sum(1 for r in records if getattr(r, "status", "") == "budget"),
        n_error=sum(1 for r in records if getattr(r, "status", "") == "error"),
        mean=(sum(values) / len(values)) if values else None,
        sd=stdev(values),
        values=list(values),
    )


def denominator_invariance(
    blocks_by_unit: dict[str, Sequence[PairedBlock]],
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, TestResult]:
    """The same contrast under every denominator, published together.

    Latency and wasted work can be denominated in dollars, in observations,
    or in harness-native steps. Publishing all three is the only answer to
    "you picked the denominator that flattered your arm" that does not rely
    on being believed.
    """
    return {
        unit: paired_permutation(blocks, seed=seed, name=f"by-{unit}")
        for unit, blocks in blocks_by_unit.items()
    }
