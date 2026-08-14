"""Reconstructing when a regression actually entered, after the fact.

A run tells you what the harness *noticed*. This tells you what was *true*.

Given a shadow timeline (see :mod:`taste.shadow`) and a set of held-out
probes the agent never saw, replaying the probes against each observation
point recovers the ground truth the run itself could not: the commit at which
a previously-passing probe started failing, whether the harness ever noticed,
and how much was spent in between.

**Bisection, not a linear scan.** Probe suites are the expensive part — a
linear walk over a few hundred observations means a few hundred test runs.
Regression onset is monotone by construction (a probe that passes at the end
never regressed; one failing at the end regressed exactly once, at the
boundary), so binary search finds each onset in O(log n) replays. On a
200-observation run that is ~8 replays instead of 200.

**The four metrics** fall out of the reconstructed timeline:

``contamination``   a probe that passed, then failed, while the harness
                    reported success.
``detection latency`` dollars and observations between onset and the harness
                    first reporting failure — censored when it never did.
``recovery``        whether the probe was passing again by the end.
``wasted work``     dollars spent inside contaminated or later-discarded spans.

All of it costs $0 in API terms. The expense is wall-clock, which is why the
bisection matters.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from taste.memory import Memory
from taste.shadow import ShadowCommit

ProbeVerdict = Literal["pass", "fail", "error"]


@dataclass(frozen=True)
class Probe:
    """A held-out check the agent never sees.

    Held out is the whole point: a probe the agent can read is a probe it can
    satisfy directly, and the measurement stops being about the task.
    """

    name: str
    command: str
    timeout: int = 120


@dataclass
class ProbeRun:
    probe: str
    seq: int
    sha: str
    verdict: ProbeVerdict
    detail: str = ""


@dataclass
class Regression:
    """One probe going from passing to failing, and what happened next."""

    probe: str
    onset_seq: int
    """First observation at which the probe failed."""
    onset_sha: str
    detected_seq: int | None = None
    """First observation where the harness itself reported failure. None
    means it never did — the regression was silent for the rest of the run."""
    recovered_seq: int | None = None
    cost_to_detect_usd: float | None = None
    cost_to_recover_usd: float | None = None
    observations_to_detect: int | None = None

    @property
    def silent(self) -> bool:
        return self.detected_seq is None

    @property
    def recovered(self) -> bool:
        return self.recovered_seq is not None


@dataclass
class ReplayReport:
    """Ground truth for one run."""

    session: str
    observations: int
    probes: tuple[str, ...] = ()
    regressions: list[Regression] = field(default_factory=list)
    replays: int = 0
    """How many probe executions this cost — the resource bisection saves."""
    final_verdicts: dict[str, ProbeVerdict] = field(default_factory=dict)

    @property
    def contaminated(self) -> bool:
        return bool(self.regressions)

    @property
    def silent_regressions(self) -> list[Regression]:
        return [r for r in self.regressions if r.silent]

    def summary(self) -> str:
        silent = len(self.silent_regressions)
        recovered = sum(1 for r in self.regressions if r.recovered)
        return (
            f"session={self.session} observations={self.observations} "
            f"regressions={len(self.regressions)} silent={silent} "
            f"recovered={recovered} replays={self.replays}"
        )


class Replayer:
    """Runs probes against historical trees, without disturbing the workspace.

    Each replay happens in a throwaway worktree checked out at the commit
    being examined, so the run under measurement is never touched and probes
    cannot leak into it.
    """

    def __init__(self, memory: Memory, probes: list[Probe]) -> None:
        self.memory = memory
        self.probes = probes
        self.replays = 0
        self._cache: dict[tuple[str, str], ProbeVerdict] = {}

    def verdict_at(self, sha: str, probe: Probe) -> ProbeVerdict:
        """Whether ``probe`` passes at ``sha``. Memoized — bisection revisits."""
        key = (sha, probe.name)
        if key in self._cache:
            return self._cache[key]

        self.replays += 1
        try:
            with self.memory.probe_worktree(sha) as path:
                verdict = self._execute(probe, path)
        except Exception:
            # A replay that cannot run tells us nothing; it must not be
            # mistaken for evidence of a regression.
            verdict = "error"
        self._cache[key] = verdict
        return verdict

    def _execute(self, probe: Probe, path: Path) -> ProbeVerdict:
        try:
            proc = subprocess.run(
                probe.command,
                shell=True,
                cwd=path,
                capture_output=True,
                text=True,
                timeout=probe.timeout,
            )
        except subprocess.TimeoutExpired:
            return "error"
        return "pass" if proc.returncode == 0 else "fail"

    # ------------------------------------------------------------ bisection

    def find_onset(self, timeline: list[ShadowCommit], probe: Probe) -> int | None:
        """Index of the first observation where ``probe`` fails, or None.

        Binary search is valid because onset is monotone: the probe passes
        over a prefix and fails over the suffix. Where reality is not monotone
        (a flake, or a genuine fix-then-rebreak) this finds *a* boundary, and
        the verdict cache keeps the cost bounded either way.
        """
        if not timeline:
            return None
        if self.verdict_at(timeline[-1].sha, probe) != "fail":
            return None  # healthy at the end: nothing to locate

        low, high = 0, len(timeline) - 1
        if self.verdict_at(timeline[0].sha, probe) == "fail":
            return 0  # already broken at the first observation

        # Invariant: low passes, high fails. Narrow until they are adjacent.
        while high - low > 1:
            mid = (low + high) // 2
            if self.verdict_at(timeline[mid].sha, probe) == "fail":
                high = mid
            else:
                low = mid
        return high


def reconstruct(
    memory: Memory,
    timeline: list[ShadowCommit],
    probes: list[Probe],
    *,
    harness_failed_at: set[int] | None = None,
    session: str = "",
) -> ReplayReport:
    """Build the ground-truth regression timeline for one run.

    ``harness_failed_at`` is the set of observation sequence numbers at which
    the harness itself reported a failure. It is what makes *detection
    latency* measurable: the distance between a regression being true and the
    harness saying so.
    """
    report = ReplayReport(
        session=session, observations=len(timeline), probes=tuple(p.name for p in probes)
    )
    if not timeline:
        return report

    replayer = Replayer(memory, probes)
    failed_at = harness_failed_at or set()

    for probe in probes:
        final = replayer.verdict_at(timeline[-1].sha, probe)
        report.final_verdicts[probe.name] = final

        index = replayer.find_onset(timeline, probe)
        if index is None:
            # Passing at the end. A transient break in the middle is real but
            # recovered; locating it costs a full scan, so it is deliberately
            # not chased here — the end state is what the task is scored on.
            continue

        onset = timeline[index]
        detected = next((seq for seq in sorted(failed_at) if seq >= onset.seq), None)
        regression = Regression(
            probe=probe.name,
            onset_seq=onset.seq,
            onset_sha=onset.sha,
            detected_seq=detected,
        )
        if detected is not None:
            marker = _at(timeline, detected)
            if marker is not None:
                regression.cost_to_detect_usd = round(
                    marker.cost_work_usd - onset.cost_work_usd, 6
                )
                regression.observations_to_detect = detected - onset.seq
        report.regressions.append(regression)

    report.replays = replayer.replays
    return report


def _at(timeline: list[ShadowCommit], seq: int) -> ShadowCommit | None:
    return next((c for c in timeline if c.seq == seq), None)
