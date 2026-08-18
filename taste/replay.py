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

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from taste.execution import ExecResult, LocalSandbox, Sandbox
from taste.memory import Memory
from taste.shadow import ShadowCommit

ProbeVerdict = Literal["pass", "fail", "skip", "error"]

# How a runner's own vocabulary maps onto ours. XFAIL is an expected failure,
# which is a passing outcome for the suite; treating it as a regression would
# invent one at every observation on repositories that use it heavily.
_STATUS_TO_VERDICT: dict[str, ProbeVerdict] = {
    "PASSED": "pass",
    "XFAIL": "pass",
    "FAILED": "fail",
    "ERROR": "fail",
    "SKIPPED": "skip",
}


@dataclass(frozen=True)
class Probe:
    """A check the agent never sees, evaluated out of band.

    One command, one verdict, read from the exit code. Correct when the check
    *is* the question — Gate 0's synthetic trajectories, a baseline probe on a
    single assertion. For a real test suite see :class:`SuiteProbe`.
    """

    name: str
    command: str
    timeout: int = 120

    def as_suite(self) -> SuiteProbe:
        return SuiteProbe(
            name=self.name, command=self.command, members=(self.name,), timeout=self.timeout
        )


@dataclass(frozen=True)
class SuiteProbe:
    """A set of named tests evaluated by ONE command, parsed per test.

    Two reasons this exists rather than one probe per test. The cheap one is
    cost: one execution yields ``len(members)`` verdicts, which turns an
    exhaustive scan from observations x tests executions into observations.

    The one that matters is that a regression is defined per *test*. An
    aggregate verdict says only that something in the suite broke — not which
    thing, so it cannot be matched against the Monitor's coverage, and the
    silent/attributed distinction the study rests on becomes uncomputable.
    """

    name: str
    command: str
    members: tuple[str, ...] = ()
    parse: Callable[[str], Mapping[str, str]] | None = None
    """Raw log to ``{test_id: STATUS}``. ``None`` means exit-code semantics:
    every member takes the verdict of the process as a whole. Runner-specific
    parsers live with the benchmark adapter, not here."""
    timeout: int = 600


@dataclass(frozen=True)
class SuiteRun:
    """One execution of one suite against one tree."""

    statuses: Mapping[str, ProbeVerdict]
    exit_code: int = 0
    infra_error: str | None = None
    """Set when the suite could not be evaluated at all. Every member is then
    ``error`` — never ``fail``. A container that would not start, a patch that
    would not apply, an interpreter that was not there: rendering any of these
    as a test failure fabricates a regression, which is the worst thing this
    pipeline can do."""


class ProbeExecutor(Protocol):
    """Evaluates a suite against one historical tree, wherever that runs."""

    def run(self, sha: str, suite: SuiteProbe) -> SuiteRun: ...

    def close(self) -> None: ...


def verdicts_from(suite: SuiteProbe, result: ExecResult) -> SuiteRun:
    """Turn a raw execution into per-member verdicts.

    The rule that carries the weight: a member the parser did not mention is
    ``error``, not ``fail``. A test can go missing because the suite crashed
    on import, because a collection error stopped it running, or because the
    runner renamed it — none of which is evidence that the test broke.
    """
    if result.timed_out:
        return SuiteRun(
            statuses=dict.fromkeys(suite.members, "error"),
            exit_code=result.exit_code,
            infra_error="timed out",
        )

    if suite.parse is None:
        verdict: ProbeVerdict = "pass" if result.ok else "fail"
        return SuiteRun(
            statuses=dict.fromkeys(suite.members, verdict), exit_code=result.exit_code
        )

    try:
        raw = suite.parse(result.output)
    except Exception as exc:
        return SuiteRun(
            statuses=dict.fromkeys(suite.members, "error"),
            exit_code=result.exit_code,
            infra_error=f"parse failed: {exc!r}",
        )

    # Nothing we recognise, or nothing about any test we grade. Both mean the
    # suite did not run for our purposes, and calling that "everything failed"
    # would report a contamination event at every observation of the instance
    # — which is precisely what the wrong-runner defect did on django.
    #
    # The second half of the condition is the load-bearing one: a wrong runner
    # usually emits *something* parseable (a usage message, a collection
    # error), so testing only for an empty map lets the failure through.
    if not raw or (suite.members and not any(m in raw for m in suite.members)):
        return SuiteRun(
            statuses=dict.fromkeys(suite.members, "error"),
            exit_code=result.exit_code,
            infra_error="no results for any graded test",
        )

    statuses: dict[str, ProbeVerdict] = {
        member: _STATUS_TO_VERDICT.get(raw.get(member, ""), "error")
        for member in suite.members
    }
    return SuiteRun(statuses=statuses, exit_code=result.exit_code)


@dataclass
class RegressionEpisode:
    """One pass→fail transition, and everything that followed it."""

    probe: str
    onset_seq: int
    onset_sha: str
    recovered_seq: int | None = None
    """First observation at which the probe passed again. None means it was
    still broken when the run ended."""
    detected_seq_attributed: int | None = None
    """First observation where the harness reported a failure that could be
    *linked to this test* by shared coverage of agent-modified files. This is
    the primary detection measure."""
    detected_seq_any: int | None = None
    """First observation where the harness reported any failure at all while
    this episode was open, linked or not. Reported alongside as a bound, and
    it is a bound in a specific direction: it over-counts detection and
    therefore *under-counts silence*. An arm that simply fails more often
    accumulates more co-occurrences and would otherwise be credited with
    noticing regressions it was never reacting to."""
    cost_to_detect_usd: float | None = None
    cost_to_recover_usd: float | None = None
    observations_to_detect: int | None = None
    observations_to_recover: int | None = None
    wasted_work_usd: float = 0.0
    """Spend between onset and recovery-or-end: work done while the tree was
    known-broken."""

    @property
    def silent(self) -> bool:
        """Nothing the harness reported could be tied to this regression."""
        return self.detected_seq_attributed is None

    @property
    def silent_unattributed(self) -> bool:
        """The companion bound: nothing failed at all while it was open."""
        return self.detected_seq_any is None

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


class LocalWorktreeExecutor:
    """Materialises the historical tree on this machine and runs it here.

    What the hermetic suite and Gate 0 use. It is *not* the measurement path
    for a real benchmark instance: a worktree contains tracked content only,
    so everything ``.gitignore`` hides — compiled extensions, build output,
    installed metadata — is absent, and the suite errors on import. See
    :class:`SandboxProbeExecutor`.
    """

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    def run(self, sha: str, suite: SuiteProbe) -> SuiteRun:
        try:
            with self.memory.probe_worktree(sha) as path:
                result = LocalSandbox(path).exec(suite.command, timeout=suite.timeout)
        except Exception as exc:
            return SuiteRun(
                statuses=dict.fromkeys(suite.members, "error"),
                infra_error=f"worktree failed: {exc!r}",
            )
        return verdicts_from(suite, result)

    def close(self) -> None:
        return None


class SandboxProbeExecutor:
    """Runs the suite inside a prepared environment, by patching into it.

    The tree is delivered as ``git diff base..sha`` applied inside the image,
    rather than by mounting a materialised checkout over it. Mounting would
    *replace* a prepared environment with an unprepared one; patching keeps
    everything the image installed and changes only what the agent changed.

    A patch that will not apply is infrastructure, not evidence, and is
    reported as such.
    """

    def __init__(self, sandbox: Sandbox, memory: Memory, base_commit: str) -> None:
        self.sandbox = sandbox
        self.memory = memory
        self.base_commit = base_commit

    def run(self, sha: str, suite: SuiteProbe) -> SuiteRun:
        def hole(reason: str) -> SuiteRun:
            return SuiteRun(
                statuses=dict.fromkeys(suite.members, "error"), infra_error=reason
            )

        try:
            patch = self.memory.repo.git.diff(self.base_commit, sha)
        except Exception as exc:
            return hole(f"diff failed: {exc!r}")

        workdir = self.sandbox.workdir
        try:
            self.sandbox.put_text("/tmp/taste.diff", (patch or "") + "\n")
            reset = self.sandbox.exec(
                f"cd {workdir} && git checkout -q {self.base_commit} -- . "
                f"&& git clean -qfd",
                timeout=180,
            )
            if not reset.ok:
                return hole(f"reset failed: {reset.output[-400:]}")
            if patch.strip():
                applied = self.sandbox.exec(
                    f"cd {workdir} && git apply -v /tmp/taste.diff", timeout=180
                )
                if not applied.ok:
                    return hole(f"patch failed: {applied.output[-400:]}")
            result = self.sandbox.exec(suite.command, timeout=suite.timeout)
        except Exception as exc:
            return hole(f"sandbox failed: {exc!r}")

        return verdicts_from(suite, result)

    def close(self) -> None:
        self.sandbox.close()


class Replayer:
    """Evaluates suites against historical trees, once per tree.

    Iteration is tree-major rather than probe-major: the expensive act is
    preparing a tree, and every member verdict for that tree comes from the
    same execution.
    """

    def __init__(
        self,
        memory: Memory,
        probes: list[Probe] | list[SuiteProbe] | None = None,
        *,
        executor: ProbeExecutor | None = None,
    ) -> None:
        self.memory = memory
        self.suites = [_as_suite(p) for p in (probes or [])]
        self.executor: ProbeExecutor = executor or LocalWorktreeExecutor(memory)
        self.replays = 0
        self._cache: dict[tuple[str, str], SuiteRun] = {}

    def run_at(self, sha: str, suite: SuiteProbe) -> SuiteRun:
        key = (sha, suite.name)
        if key not in self._cache:
            self.replays += 1
            self._cache[key] = self.executor.run(sha, suite)
        return self._cache[key]

    def verdict_at(self, sha: str, probe: Probe | SuiteProbe) -> ProbeVerdict:
        """The verdict for a single-member probe. Retained for Gate 0."""
        suite = _as_suite(probe)
        run = self.run_at(sha, suite)
        member = suite.members[0] if suite.members else suite.name
        return run.statuses.get(member, "error")

    def verdicts_across(
        self, timeline: list[ShadowCommit], probe: Probe | SuiteProbe
    ) -> list[ProbeVerdict]:
        """Every observation, in order. No monotonicity assumed."""
        return [self.verdict_at(c.sha, probe) for c in timeline]

    def matrix(
        self, timeline: list[ShadowCommit], suite: SuiteProbe
    ) -> dict[str, list[ProbeVerdict]]:
        """``{member: verdict per observation}``, one execution per tree."""
        series: dict[str, list[ProbeVerdict]] = {m: [] for m in suite.members}
        for commit in timeline:
            run = self.run_at(commit.sha, suite)
            for member in suite.members:
                series[member].append(run.statuses.get(member, "error"))
        return series


def _as_suite(probe: Probe | SuiteProbe) -> SuiteProbe:
    return probe.as_suite() if isinstance(probe, Probe) else probe


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
    probes: list[Probe] | list[SuiteProbe],
    *,
    harness_failed_at: set[int] | None = None,
    attribution: dict[int, set[str]] | None = None,
    session: str = "",
    executor: ProbeExecutor | None = None,
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

    replayer = Replayer(memory, probes, executor=executor)
    failed_at = sorted(harness_failed_at or set())
    never_passed: list[str] = []

    # Tree-major: every member's verdict at an observation comes from the one
    # execution that prepared it.
    for suite in replayer.suites:
        series = replayer.matrix(timeline, suite)
        for member, verdicts in series.items():
            report.final_verdicts[member] = verdicts[-1] if verdicts else "error"

            episodes, unknown, ever_passed = episodes_from(verdicts, timeline, member)
            report.unknown_transitions += unknown
            if not ever_passed:
                # Never worked, so nothing regressed. Recording it as one is
                # what made a from-scratch benchmark report 100% contamination
                # at the first observation.
                never_passed.append(member)
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

    # Detection, both variants, from one pass. Computing them separately
    # would double the cost of a scan, and the protocol requires both to be
    # reported side by side for every arm — an arm's advantage on the
    # attributed measure is only readable against how often it failed at all.
    for seq in failed_at:
        if seq < episode.onset_seq or seq > close_seq:
            continue
        if episode.detected_seq_any is None:
            episode.detected_seq_any = seq
        linked = attribution is not None and episode.probe in attribution.get(seq, set())
        if linked and episode.detected_seq_attributed is None:
            episode.detected_seq_attributed = seq
            marker = _at(timeline, seq)
            if marker is not None:
                episode.cost_to_detect_usd = round(
                    marker.cost_work_usd - onset.cost_work_usd, 6
                )
                episode.observations_to_detect = seq - episode.onset_seq
        if episode.detected_seq_any is not None and episode.detected_seq_attributed is not None:
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
