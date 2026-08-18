"""Was the harness reacting to *this* regression, or just failing nearby?

The study's headline quantity is how often a regression goes **silent** — how
often previously-working behaviour breaks and nothing in the harness notices.
Deciding that requires linking a Monitor failure to a specific broken test,
and the only two ways to do it are both traps:

*By name.* Impossible here, and deliberately so. The Monitor's scope is
constructed disjoint from the graded set at file level, precisely so it cannot
run the tests that score us. Names can therefore never intersect.

*By co-occurrence.* "The Monitor failed at some point while the regression was
open." This is the trap. It is not a neutral approximation — it systematically
credits whichever arm fails most often with the best detection, because more
failures means more chances to overlap. An arm that flails constantly would
appear the most vigilant.

So attribution goes through **coverage**: a Monitor failure is linked to a
regression when some failing Monitor test and the broken graded test both
exercise a file the agent actually changed at that observation. All three
terms are needed. Without the third, any two tests touching a shared utility
module link forever.

Both variants are computed and reported side by side, always. The
co-occurrence number is not deleted, because a reader is entitled to see how
much of the difference rests on the coverage machinery — but it is labelled
for what it is: an over-count of detection, and therefore an **under**-count
of silence.

**What is deliberately absent.** No static import-graph approximation. It is
wrong in both directions on exactly the repositories that dominate the frame:
Django resolves models, apps and URLs through settings *strings* no import
graph can see, which under-attributes and inflates the headline silence
number; and a test importing a package ``__init__`` inherits its entire
subtree, which over-attributes. Being wrong in both directions with no
measurable error rate is not a fallback, it is a guess with a citation.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from taste.replay import RegressionEpisode
from taste.shadow import ShadowCommit

CoverageMethod = Literal["pytest_cov_context", "pytest_cov_per_test", "declared", "none"]

AttributionVerdict = Literal["attributed", "unattributed", "unknown"]


# ------------------------------------------------------------------ failures


@dataclass(frozen=True)
class MonitorFailure:
    """One Monitor FAIL, placed on the observation timeline."""

    session: str
    step_id: str
    attempt: int
    seq: int | None
    """The observation whose tree the Monitor graded. ``None`` means the
    worker left the tree byte-identical, so no shadow commit was written —
    a real detection at a point the timeline does not contain. It is kept
    rather than dropped: discarding it would silently convert a detected
    regression into a silent one."""
    reason: str = ""
    failing_tests: tuple[str, ...] = ()
    ts: float = 0.0


def harness_failures(
    events: Iterable[Mapping[str, object]],
    timeline: Sequence[ShadowCommit],
    *,
    session: str = "",
) -> list[MonitorFailure]:
    """Join the Monitor's failures onto the observation timeline.

    The join is exact rather than temporal. ``_observe`` fires immediately
    after the worker finishes and immediately before the Monitor evaluates,
    so the shadow commit carrying ``(step_id, attempt)`` *is* the tree that
    Monitor verdict graded. Matching by timestamp or by position in the event
    stream would be guesswork; matching by key is not.

    Note that the shadow sha and the session commit sha are different objects
    and never join — shadow commits form a chain of their own.
    """
    index: dict[tuple[str, int], ShadowCommit] = {}
    for commit in timeline:
        if session and commit.session != session:
            continue
        # A step may be observed more than once; the worker observation is the
        # one the Monitor graded.
        key = (commit.step_id, commit.attempt)
        if key not in index or commit.trigger == "worker":
            index[key] = commit

    failures: list[MonitorFailure] = []
    for event in events:
        if event.get("kind") != "monitor.verdict":
            continue
        payload = event.get("payload") or {}
        if not isinstance(payload, Mapping) or payload.get("passed") is not False:
            continue
        step_id = str(payload.get("id", ""))
        attempt = int(payload.get("attempt", 0) or 0)
        commit = index.get((step_id, attempt))
        failures.append(
            MonitorFailure(
                session=session or (commit.session if commit else ""),
                step_id=step_id,
                attempt=attempt,
                seq=commit.seq if commit else None,
                reason=str(payload.get("reason", "")),
                failing_tests=tuple(payload.get("failing_tests", ()) or ()),
                ts=float(event.get("ts", 0.0) or 0.0),
            )
        )
    return failures


def read_events(path: Path) -> list[dict[str, object]]:
    """Load an events.jsonl, tolerating a truncated final line.

    A killed run leaves a partial write; refusing to read the whole file
    because of it would discard an otherwise complete cell.
    """
    events: list[dict[str, object]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def failed_at(failures: Iterable[MonitorFailure]) -> set[int]:
    """Observation sequence numbers at which the harness reported failure."""
    return {f.seq for f in failures if f.seq is not None}


# ------------------------------------------------------------------ coverage


@dataclass(frozen=True)
class CoverageMap:
    """Which source files each test exercises.

    Built **once per instance at base_commit, before any arm runs**, so it is
    identical across arms by construction and cannot be a function of an
    outcome. That is what makes it admissible: there is no way to tune it to
    favour a treatment, because it does not know which treatment ran.

    It is measured at the base tree while the agent edits the tree, so a
    test's covered set at a later observation may differ. That is the
    conservative direction — a stale map can only *fail* to link, pushing
    toward more reported silence, symmetrically across arms. It cannot
    manufacture the effect.
    """

    instance_id: str
    built_at_commit: str
    method: CoverageMethod
    covers: Mapping[str, frozenset[str]] = field(default_factory=dict)
    uninstrumented: frozenset[str] = frozenset()
    """Tests that ran but produced no coverage data. Distinct from tests that
    ran and covered nothing."""

    def files_for(self, test_id: str) -> frozenset[str] | None:
        """``None`` means UNKNOWN; an empty set means measured-and-covers-nothing.

        Collapsing the first into the second is how a pipeline fabricates
        silence: an unmeasurable test would be reported as "linked to
        nothing", which reads identically to "the harness missed it".
        """
        if test_id in self.uninstrumented:
            return None
        found = self.covers.get(test_id)
        return None if found is None else frozenset(found)

    @property
    def measured(self) -> int:
        return len(self.covers)


def read_coverage_sqlite(
    path: Path, *, instance_id: str, built_at_commit: str, root: str = ""
) -> CoverageMap:
    """Read a coverage.py database written with ``--cov-context=test``.

    Contexts come back as ``"<test_id>|<phase>"``; the phase suffix is dropped
    because a regression is about the test, not about which of its setup,
    call or teardown touched a file. Reduced to file granularity: line-level
    precision would be spurious, since the agent's edit shifts line numbers.
    """
    covers: dict[str, set[str]] = {}
    with sqlite3.connect(str(path)) as conn:
        rows = conn.execute(
            "SELECT c.context, f.path "
            "FROM line_bits lb "
            "JOIN context c ON c.id = lb.context_id "
            "JOIN file f ON f.id = lb.file_id"
        ).fetchall()
    for context, file_path in rows:
        if not context:
            continue  # the empty context is "no test running"
        test_id = str(context).split("|", 1)[0]
        relative = str(file_path)
        if root and relative.startswith(root):
            relative = relative[len(root) :].lstrip("/")
        covers.setdefault(test_id, set()).add(relative)
    return CoverageMap(
        instance_id=instance_id,
        built_at_commit=built_at_commit,
        method="pytest_cov_context",
        covers={k: frozenset(v) for k, v in covers.items()},
    )


# ------------------------------------------------------------------ classify


@dataclass(frozen=True)
class AttributionLink:
    monitor_test: str
    probe_test: str
    shared_files: frozenset[str]


@dataclass(frozen=True)
class AttributionDecision:
    seq: int
    probe_test: str
    verdict: AttributionVerdict
    links: tuple[AttributionLink, ...] = ()
    reason: str = ""


@dataclass
class AttributionResult:
    """What ``reconstruct`` needs, plus the accounting a reviewer will want."""

    by_seq: dict[int, set[str]] = field(default_factory=dict)
    """``{observation: {probe tests its failure can be linked to}}`` — the
    exact shape :func:`taste.replay.reconstruct` already consumes."""
    decisions: tuple[AttributionDecision, ...] = ()
    unknown: int = 0
    considered: int = 0

    @property
    def unknown_rate(self) -> float:
        """How much of the classification rested on missing coverage data.

        Bounds how small a gap between the attributed and co-occurrence
        numbers is worth believing.
        """
        return self.unknown / self.considered if self.considered else 0.0


def attribution_map(
    *,
    failures: Sequence[MonitorFailure],
    probe_tests: Sequence[str],
    monitor_coverage: CoverageMap,
    probe_coverage: CoverageMap,
    modified_files_at: Mapping[int, frozenset[str]],
) -> AttributionResult:
    """Link each Monitor failure to the graded tests it could have been about.

    The rule, in full::

        linked(failure, probe_test) iff
            exists m in failure.failing_tests:
                monitor_covers(m) & probe_covers(probe_test) & modified_at(seq)

    Every term earns its place. Drop the third and two tests that both import
    a shared utility are linked in every run forever, which would attribute
    essentially everything and erase the phenomenon. Drop the first two and
    this is co-occurrence again.
    """
    result = AttributionResult()
    decisions: list[AttributionDecision] = []

    for failure in failures:
        if failure.seq is None:
            continue  # a real detection, but at no observation we can index
        changed = modified_files_at.get(failure.seq, frozenset())

        for probe_test in probe_tests:
            result.considered += 1
            probe_files = probe_coverage.files_for(probe_test)
            if probe_files is None:
                result.unknown += 1
                decisions.append(
                    AttributionDecision(
                        failure.seq, probe_test, "unknown", reason="probe coverage unknown"
                    )
                )
                continue

            links: list[AttributionLink] = []
            saw_unknown_monitor = False
            for monitor_test in failure.failing_tests:
                monitor_files = monitor_coverage.files_for(monitor_test)
                if monitor_files is None:
                    saw_unknown_monitor = True
                    continue
                shared = monitor_files & probe_files & changed
                if shared:
                    links.append(AttributionLink(monitor_test, probe_test, shared))

            if links:
                result.by_seq.setdefault(failure.seq, set()).add(probe_test)
                decisions.append(
                    AttributionDecision(
                        failure.seq, probe_test, "attributed", tuple(links)
                    )
                )
            elif saw_unknown_monitor or not failure.failing_tests:
                # Never "unattributed": we did not measure enough to say so,
                # and calling it unattributed would count as evidence of
                # silence something that is merely absence of data.
                result.unknown += 1
                reason = (
                    "monitor coverage unknown"
                    if saw_unknown_monitor
                    else "failure reported no test identities"
                )
                decisions.append(
                    AttributionDecision(failure.seq, probe_test, "unknown", reason=reason)
                )
            else:
                decisions.append(
                    AttributionDecision(
                        failure.seq, probe_test, "unattributed", reason="no shared changed file"
                    )
                )

    result.decisions = tuple(decisions)
    return result


# ------------------------------------------------------------------ reporting


@dataclass
class SilenceReport:
    """Both detection variants for one run, side by side."""

    episodes: int = 0
    silent_attributed: int = 0
    """Primary: no Monitor failure could be linked to the regression."""
    silent_unattributed: int = 0
    """Bound: nothing failed at all while it was open. Necessarily <= the
    primary count, because linking is strictly more demanding than
    co-occurring."""
    unknown_attribution_rate: float = 0.0
    method: CoverageMethod = "none"

    @property
    def silence_rate(self) -> float | None:
        return self.silent_attributed / self.episodes if self.episodes else None

    @property
    def silence_rate_bound(self) -> float | None:
        return self.silent_unattributed / self.episodes if self.episodes else None

    def summary(self) -> str:
        def pct(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.0%}"

        return (
            f"episodes={self.episodes} silent={pct(self.silence_rate)} "
            f"(bound {pct(self.silence_rate_bound)}) "
            f"unknown={self.unknown_attribution_rate:.1%} method={self.method}"
        )


def summarise_silence(
    episodes: Sequence[RegressionEpisode],
    result: AttributionResult,
    *,
    method: CoverageMethod = "none",
) -> SilenceReport:
    return SilenceReport(
        episodes=len(episodes),
        silent_attributed=sum(1 for e in episodes if e.silent),
        silent_unattributed=sum(1 for e in episodes if e.silent_unattributed),
        unknown_attribution_rate=result.unknown_rate,
        method=method,
    )
