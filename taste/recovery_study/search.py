"""The point of no return, found by a binary search over probes of a failed run.

A probe of step k is a set of branches from the state at step k (K = 8 at
first), each graded by the task's hidden tests. It is winnable if at least one
branch succeeded, lost otherwise. Step 0, a start from scratch, counts as
winnable: the study keeps tasks solved from scratch, and v(0) comes from
calibration. The run's own last step T, its submission, failed and counts as
lost.

``plan`` reads the probes and says what is wanted next:

- ``low`` is the last winnable probe and ``high`` the first lost probe above
  it (T when there is none), over every settled probe wherever it lies;
- while they are more than one step apart, the step nearest their middle is
  probed;
- once adjacent, each side gets K + extra branches (16 each, so that "lost"
  means v <= 0.17 at 95%, one-sided); a lost side that turns winnable moves
  the search on;
- a step from which no faithful branch could be made is skipped, and when no
  settled probe lies beyond it, it ends the search there: the run is
  censored at that step.

A full scan (a probe every N steps) is read the same way. Because ``low`` is
the last winnable probe, a curve that recovers after a drop moves the search
past the drop; the earlier drop is reported. The decisive step d is ``high``
once both sides are settled: the one step between the last winnable state and
the first lost one.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

Z95 = 1.959963984540054
# A run's search has ended: its boundary is confirmed, or it cannot be reached.
FINAL = ("done", "censored", "unresolved")


def wilson(successes, trials, z=Z95):
    """Wilson score interval for a success rate; (0, 1) without trials."""
    if trials <= 0:
        return 0.0, 1.0
    rate, spread = successes / trials, z * z / trials
    centre = (rate + spread / 2) / (1 + spread)
    half = z * math.sqrt(rate * (1 - rate) / trials + spread / (4 * trials)) / (1 + spread)
    return max(0.0, centre - half), min(1.0, centre + half)


def lost_bound(trials, level=0.95):
    """The exact one-sided upper bound on v when no branch of ``trials`` succeeded."""
    return 1.0 - (1.0 - level) ** (1.0 / trials) if trials > 0 else 1.0


@dataclass(frozen=True)
class Probe:
    """What is known about branches from one step."""

    successes: int = 0
    trials: int = 0             # graded, faithful branches from finished jobs
    pending: bool = False       # some of its branches are not finished
    exhausted: bool = False     # no more branches will be started from it
    unusable: bool = False      # no faithful branch could be made from it

    @property
    def settled(self):
        return not self.pending and not self.unusable and self.trials > 0

    @property
    def winnable(self):
        return self.successes > 0


@dataclass(frozen=True)
class Plan:
    targets: dict               # step -> faithful branches wanted from it in all
    status: str                 # searching, waiting, confirming, done, censored or unresolved
    low: int
    high: int
    limit: int                  # no faithful branch from this step on (T when none)
    next_steps: tuple = field(default=())   # the steps probed next, either way the current probe goes


def plan(steps, probes, *, k=8, extra=8, scan=()):
    """What to probe next in a run of ``steps`` steps, given what its probes found."""
    settled = {step: probe for step, probe in probes.items() if probe.settled}
    top = max(settled, default=0)
    limit = min((step for step, probe in probes.items() if probe.unusable and step > top), default=steps)
    blocked = {step for step, probe in probes.items() if probe.unusable}
    targets = {step: k for step in scan if 0 < step < limit and step not in blocked}
    low = max((step for step, probe in settled.items() if probe.winnable), default=0)
    high = min((step for step, probe in settled.items() if not probe.winnable and step > low), default=steps)
    ceiling = min(high, limit)

    def middle(start, end):
        open_steps = [step for step in range(start + 1, end) if step not in blocked and step not in settled]
        return min(open_steps, key=lambda step: (abs(step - (start + end) / 2), step)) if open_steps else None

    if any(probe.pending for probe in probes.values()):
        return Plan(targets, "waiting", low, high, limit)
    if ceiling - low > 1:
        step = middle(low, ceiling)
        if step is None:
            return Plan(targets, "unresolved", low, high, limit)
        targets[step] = max(targets.get(step, 0), k)
        following = tuple(s for s in (middle(low, step), middle(step, ceiling)) if s is not None)
        return Plan(targets, "searching", low, high, limit, following)
    if ceiling < high:
        return Plan(targets, "censored", low, high, limit)
    for side in (low, high):
        if 0 < side < steps:
            targets[side] = max(targets.get(side, 0), k + extra)
    short = any(probes.get(step, Probe()).trials < wanted and not probes.get(step, Probe()).exhausted
                for step, wanted in targets.items())
    return Plan(targets, "confirming" if short else "done", low, high, limit)


def point(step, successes, trials, **extra):
    low, high = wilson(successes, trials)
    value = {"step": step, "successes": successes, "trials": trials,
             "v": successes / trials if trials else None, "low": round(low, 6), "high": round(high, 6), **extra}
    if trials and not successes:
        value["lost_upper_95"] = round(lost_bound(trials), 6)
    return value


def result(steps, probes, *, v0=None, k=8, extra=8, scan=()):
    """A run's curve, decisive step, earlier drops and best rewind points."""
    found = plan(steps, probes, k=k, extra=extra, scan=scan)
    settled = {step: probe for step, probe in sorted(probes.items()) if probe.settled}
    curve = [point(0, *v0, source="calibration")] if v0 is not None else []
    curve += [point(step, probe.successes, probe.trials) for step, probe in settled.items()]
    # Step 0 is winnable by the study's choice of tasks; the drops are the
    # winnable-to-lost changes between settled probes in step order.
    sequence = [(0, True)] + [(step, probe.winnable) for step, probe in settled.items() if step > 0]
    changes = list(itertools.pairwise(sequence))
    drops = [{"after": a, "by": b} for (a, won_a), (b, won_b) in changes if won_a and not won_b]
    decisive = found.high if found.status == "done" else None
    winnable = [(step, probe) for step, probe in settled.items() if probe.winnable and step >= 1]
    highest = max(winnable, key=lambda item: (item[1].successes / item[1].trials, item[0]))[0] if winnable else 0
    return {
        "status": found.status, "steps": steps, "low": found.low, "high": found.high, "limit": found.limit,
        "decisive_step": decisive, "decisive_fraction": None if decisive is None else round(decisive / steps, 6),
        "earlier_drops": [drop for drop in drops if drop["by"] <= found.low],
        "monotone": not any(not won_a and won_b for (_, won_a), (_, won_b) in changes),
        "recoverable": bool(winnable), "rewind": {"highest_v": highest, "latest": found.low},
        "v0_winnable": None if v0 is None else v0[0] > 0, "curve": curve,
    }
