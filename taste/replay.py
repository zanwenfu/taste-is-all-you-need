"""Reconstructing when a regression entered — and whether it was repaired.

A run tells you what the harness *noticed*. This tells you what was *true*.

Given a shadow timeline (see :mod:`taste.shadow`) and probes the agent never
saw, replaying the probes across observation points recovers the ground truth
the run itself could not: where a passing check began failing, whether it came
back, whether anything ever detected it, and how much was spent in between.

**Why this scans exhaustively instead of bisecting.** An earlier version binary
-searched for the point where a probe "started failing", on the premise that
the verdict sequence is monotone — passing over a prefix, failing over a
suffix. That premise is not merely an approximation here; it is violated
*systematically and differentially by arm*, because non-monotonicity is the
treatment under study. An arm that repairs forward and an arm that resets to
a passing tree both produce pass→fail→pass sequences by design. Bisection
therefore recorded only regressions that survived to the end of a run — which
are exactly the ones the recovering arm *failed* to fix. The arm whose entire
claim is that it recovers would have looked like the arm with the fewest
regressions and no measurable recovery rate at all.

The optimization was never needed. A probe against a warm container costs
seconds of CPU and nothing in API budget, so an exhaustive scan is affordable
and assumption-free. Bisection survives only behind an explicit flag, is
documented as an approximation, and is used in nothing reported.

**Events, not states.** A regression is a PASS→FAIL *transition*, so a check
that never passed cannot be a regression — it is a task that was never done.
A probe may regress and recover more than once, so each probe yields a list of
episodes rather than a single verdict.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from taste.memory import Memory
from taste.shadow import ShadowCommit

ProbeVerdict = Literal["pass", "fail", "skip", "error"]


@dataclass(frozen=True)
class Probe:
    """A check the agent never sees, evaluated out of band."""

    name: str
    command: str
    timeout: int = 120


@dataclass
class RegressionEpisode:
    """One pass→fail transition, and everything that followed it."""

    probe: str
    onset_seq: int
    onset_sha: str
    recovered_seq: int | None = None
    """First observation at which the probe passed again. None means it was
    still broken when the run ended."""
    detected_seq: int | None = None
    """First observation where the harness itself reported a failure while
    this episode was open. See ``attributed`` — proximity in time is not
    proof the harness was reacting to *this* regression."""
    attributed: bool = False
    """Whether the detecting failure could be linked to this probe by shared
    coverage, rather than merely co-occurring."""
    cost_to_detect_usd: float | None = None
    cost_to_recover_usd: float | None = None
    observations_to_detect: int | None = None
    observations_to_recover: int | None = None
    wasted_work_usd: float = 0.0
    """Spend between onset and recovery-or-end: work done while the tree was
    known-broken."""

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
    episodes: list[RegressionEpisode] = field(default_factory=list)
    replays: int = 0
    final_verdicts: dict[str, ProbeVerdict] = field(default_factory=dict)
    unknown_transitions: int = 0
    """PASS→SKIP occurrences. Never counted as "maintained" — a skipped test
    is not a passing one, and pretending otherwise understates contamination."""
    never_passed: tuple[str, ...] = ()
    """Probes that never passed anywhere in the run. Not regressions: there
    was nothing to break."""

    # ---- the four metrics, per run

    @property
    def contamination_events(self) -> int:
        return len(self.episodes)

    @property
    def contaminated(self) -> bool:
        return bool(self.episodes)

    @property
    def silent_episodes(self) -> list[RegressionEpisode]:
        return [e for e in self.episodes if e.silent]

    @property
    def recovery_rate(self) -> float | None:
        """Fraction of regressions the run repaired. None when there were
        none — an empty run is not a 0% recovery rate."""
        if not self.episodes:
            return None
        return sum(1 for e in self.episodes if e.recovered) / len(self.episodes)

    @property
    def wasted_work_usd(self) -> float:
        return round(sum(e.wasted_work_usd for e in self.episodes), 6)

    def summary(self) -> str:
        rate = self.recovery_rate
        rate_text = "n/a" if rate is None else f"{rate:.0%}"
        return (
            f"session={self.session} observations={self.observations} "
            f"events={self.contamination_events} silent={len(self.silent_episodes)} "
            f"recovery={rate_text} wasted=${self.wasted_work_usd:.4f} "
            f"replays={self.replays}"
        )


class Replayer:
    """Evaluates probes against historical trees, in throwaway worktrees."""

    def __init__(self, memory: Memory, probes: list[Probe]) -> None:
        self.memory = memory
        self.probes = probes
        self.replays = 0
        self._cache: dict[tuple[str, str], ProbeVerdict] = {}

    def verdict_at(self, sha: str, probe: Probe) -> ProbeVerdict:
        key = (sha, probe.name)
        if key in self._cache:
            return self._cache[key]

        self.replays += 1
        try:
            with self.memory.probe_worktree(sha) as path:
                verdict = self._execute(probe, path)
        except Exception:
            # A probe that could not run is not evidence of a regression.
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

    def verdicts_across(
        self, timeline: list[ShadowCommit], probe: Probe
    ) -> list[ProbeVerdict]:
        """Every observation, in order. No monotonicity assumed."""
        return [self.verdict_at(c.sha, probe) for c in timeline]


def episodes_from(
    verdicts: list[ProbeVerdict], timeline: list[ShadowCommit], probe_name: str
) -> tuple[list[RegressionEpisode], int, bool]:
    """Turn a verdict sequence into regression episodes.

    Returns ``(episodes, unknown_transitions, ever_passed)``. An episode opens
    on PASS→FAIL and closes on the next PASS. ``error`` verdicts are treated
    as missing observations — they neither open nor close an episode, because
    a probe that could not run says nothing about the tree.
    """
    episodes: list[RegressionEpisode] = []
    unknown = 0
    ever_passed = False
    last_known: ProbeVerdict | None = None
    open_episode: RegressionEpisode | None = None

    for index, verdict in enumerate(verdicts):
        if verdict == "error":
            continue
        if verdict == "pass":
            ever_passed = True

        if last_known == "pass" and verdict == "skip":
            # A skipped test is not a passing one; the grader's "maintained"
            # convention would hide this, so it is counted separately.
            unknown += 1

        if last_known == "pass" and verdict == "fail" and open_episode is None:
            open_episode = RegressionEpisode(
                probe=probe_name,
                onset_seq=timeline[index].seq,
                onset_sha=timeline[index].sha,
            )
        elif open_episode is not None and verdict == "pass":
            open_episode.recovered_seq = timeline[index].seq
            episodes.append(open_episode)
            open_episode = None

        if verdict in ("pass", "fail"):
            last_known = verdict

    if open_episode is not None:
        episodes.append(open_episode)  # still broken at the end
    return episodes, unknown, ever_passed


def reconstruct(
    memory: Memory,
    timeline: list[ShadowCommit],
    probes: list[Probe],
    *,
    harness_failed_at: set[int] | None = None,
    attribution: dict[int, set[str]] | None = None,
    session: str = "",
) -> ReplayReport:
    """Build the ground-truth regression timeline for one run.

    ``harness_failed_at`` is the set of observation sequence numbers at which
    the harness reported a failure — what makes detection latency measurable.

    ``attribution`` optionally maps such an observation to the probe names its
    failure can be linked to (by shared coverage of agent-modified files).
    Without it, detection is recorded but flagged unattributed, and must be
    read as an *upper bound* on the harness's detection ability: an arm that
    fails often would otherwise be credited with noticing regressions it was
    never reacting to.
    """
    report = ReplayReport(
        session=session, observations=len(timeline), probes=tuple(p.name for p in probes)
    )
    if not timeline:
        return report

    replayer = Replayer(memory, probes)
    failed_at = sorted(harness_failed_at or set())
    never_passed: list[str] = []

    for probe in probes:
        verdicts = replayer.verdicts_across(timeline, probe)
        report.final_verdicts[probe.name] = verdicts[-1] if verdicts else "error"

        episodes, unknown, ever_passed = episodes_from(verdicts, timeline, probe.name)
        report.unknown_transitions += unknown
        if not ever_passed:
            # Never worked, so nothing regressed. Recording it as a regression
            # is what made a from-scratch benchmark report 100% contamination
            # at the first observation.
            never_passed.append(probe.name)
            continue

        for episode in episodes:
            _annotate(episode, timeline, failed_at, attribution)
            report.episodes.append(episode)

    report.never_passed = tuple(never_passed)
    report.replays = replayer.replays
    return report


def _annotate(
    episode: RegressionEpisode,
    timeline: list[ShadowCommit],
    failed_at: list[int],
    attribution: dict[int, set[str]] | None,
) -> None:
    """Fill in detection, recovery and wasted work for one episode."""
    onset = _at(timeline, episode.onset_seq)
    if onset is None:
        return
    close_seq = episode.recovered_seq if episode.recovered else timeline[-1].seq
    close = _at(timeline, close_seq)

    # Detection: the harness reporting failure while this episode was open.
    for seq in failed_at:
        if seq < episode.onset_seq or seq > close_seq:
            continue
        if attribution is not None and episode.probe not in attribution.get(seq, set()):
            continue  # co-occurrence is not attribution
        episode.detected_seq = seq
        episode.attributed = attribution is not None
        marker = _at(timeline, seq)
        if marker is not None:
            episode.cost_to_detect_usd = round(marker.cost_work_usd - onset.cost_work_usd, 6)
            episode.observations_to_detect = seq - episode.onset_seq
        break

    if close is not None:
        # Everything spent while the tree was known-broken.
        episode.wasted_work_usd = round(close.cost_work_usd - onset.cost_work_usd, 6)
        if episode.recovered:
            episode.cost_to_recover_usd = episode.wasted_work_usd
            episode.observations_to_recover = close_seq - episode.onset_seq


def _at(timeline: list[ShadowCommit], seq: int) -> ShadowCommit | None:
    return next((c for c in timeline if c.seq == seq), None)


# ---------------------------------------------------------------- opt-in only


def find_onset_by_bisection(
    replayer: Replayer, timeline: list[ShadowCommit], probe: Probe
) -> int | None:
    """First failing observation, assuming monotonicity. NOT used in results.

    Retained for the case where a probe is known to be monotone and replay is
    genuinely expensive. It cannot see a regression that was later repaired,
    which is why nothing in the reported pipeline calls it: the arms under
    study produce non-monotone sequences *by design*.
    """
    if not timeline or replayer.verdict_at(timeline[-1].sha, probe) != "fail":
        return None
    if replayer.verdict_at(timeline[0].sha, probe) == "fail":
        return 0

    low, high = 0, len(timeline) - 1
    while high - low > 1:
        mid = (low + high) // 2
        if replayer.verdict_at(timeline[mid].sha, probe) == "fail":
            high = mid
        else:
            low = mid
    return high
