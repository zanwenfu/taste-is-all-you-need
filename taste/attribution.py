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
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from taste.replay import RegressionEpisode
from taste.shadow import ShadowCommit

CoverageMethod = Literal[
    "coverage_dynamic_context", "pytest_cov_context", "pytest_cov_per_test",
    "declared", "none",
]

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
        # The Monitor evaluates *after* the worker finishes, so the tree it
        # graded is the LAST observation of that (step_id, attempt) —
        # whatever produced it.
        #
        # This used to prefer the `worker`-triggered commit and otherwise keep
        # the first. Under the per-tool grid that is wrong in a way that
        # matters: the final tool observation already captures the tree the
        # worker left, so ShadowLog dedupes the boundary commit away, no
        # `worker` observation exists, and the fallback pinned the failure to
        # the *earliest* tree of the attempt — understating detection latency
        # and attributing the failure to a tree that did not yet contain the
        # break.
        index[(commit.step_id, commit.attempt)] = commit

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


# ------------------------------------------------------------------ id shapes


@dataclass(frozen=True)
class TestKey:
    """A test identity reduced to the parts two naming schemes can share."""

    func: str
    cls: str | None = None
    module: str = ""


_UNITTEST_LABEL = re.compile(r"^(?P<func>[\w]+)\s+\((?P<path>[\w.]+)\)$")
_IDENTIFIER = re.compile(r"^[\w.]+$")


def parse_member_id(test_id: str) -> TestKey | None:
    """Reduce a PASS_TO_PASS identifier to a comparable key.

    The published set uses four different grammars, and a fifth thing that is
    not an identifier at all. Measured across all 60,142 ids in Verified:
    61.5% pytest node ids, 24.7% unittest labels, 6.5% bare function names
    (sympy), 1.3% unittest labels with the method re-appended (django 4.2+),
    and **6.1% that are not test names** — django's runner prints a test's
    docstring instead of its name when it has one, and one instance captured
    a pytest progress marker as an id.

    ``None`` means unmappable, and callers must treat that as UNKNOWN rather
    than as a test that covers nothing.
    """
    raw = test_id.strip()
    if not raw:
        return None

    if "::" in raw:
        parts = raw.split("::")
        path, tail = parts[0], parts[1:]
        module = Path(path).stem
        if len(tail) >= 2:
            return TestKey(func=tail[-1], cls=tail[-2], module=module)
        return TestKey(func=tail[-1], cls=None, module=module) if tail else None

    label = _UNITTEST_LABEL.match(raw)
    if label:
        func, path = label.group("func"), label.group("path")
        segments = path.split(".")
        # django 4.2+ repeats the method name at the end of the dotted path.
        if segments and segments[-1] == func:
            segments = segments[:-1]
        cls = segments[-1] if segments and segments[-1][:1].isupper() else None
        module = ".".join(segments[:-1] if cls else segments)
        return TestKey(func=func, cls=cls, module=module)

    if _IDENTIFIER.match(raw) and raw.startswith("test"):
        return TestKey(func=raw.split(".")[-1])

    return None  # a docstring, a progress marker, or something else entirely


def parse_context(context: str) -> TestKey | None:
    """Reduce a coverage.py dynamic context to the same key.

    ``dynamic_context = test_function`` is what makes coverage buildable at
    all here. pytest-cov's ``--cov-context`` only works under pytest, and
    django and sympy — 61% of the frame between them — do not run under
    pytest. coverage.py's own setting is runner-agnostic: it names the context
    after whatever test function is executing, whoever invoked it.
    """
    raw = context.split("|", 1)[0].strip()
    if not raw:
        return None
    segments = raw.split(".")
    func = segments[-1]
    if not func.startswith("test"):
        return None
    rest = segments[:-1]
    cls = rest[-1] if rest and rest[-1][:1].isupper() else None
    module = ".".join(rest[:-1] if cls else rest)
    return TestKey(func=func, cls=cls, module=module)


@dataclass
class Reconciliation:
    """Which graded tests were matched to a coverage context, and which were not."""

    matched: dict[str, str] = field(default_factory=dict)
    """graded id -> coverage context"""
    unmappable: tuple[str, ...] = ()
    """Ids that are not test identifiers — docstrings and the like."""
    unmatched: tuple[str, ...] = ()
    """Parsed fine, but no context ran under that name."""
    ambiguous: tuple[str, ...] = ()
    """Several contexts share the name and nothing disambiguates them."""

    @property
    def unresolved(self) -> tuple[str, ...]:
        return tuple(sorted({*self.unmappable, *self.unmatched, *self.ambiguous}))


def reconcile_contexts(contexts: Iterable[str], members: Sequence[str]) -> Reconciliation:
    """Match graded test ids to coverage contexts across naming schemes.

    Narrowed in three passes — function name, then class, then module tail.
    Anything still ambiguous is reported as ambiguous, never guessed: a wrong
    match would attribute a Monitor failure to the wrong test, which is
    exactly the error the coverage rule exists to prevent.
    """
    by_func: dict[str, list[tuple[str, TestKey]]] = {}
    for context in contexts:
        key = parse_context(context)
        if key is not None:
            by_func.setdefault(key.func, []).append((context, key))

    result = Reconciliation()
    unmappable: list[str] = []
    unmatched: list[str] = []
    ambiguous: list[str] = []

    for member in members:
        key = parse_member_id(member)
        if key is None:
            unmappable.append(member)
            continue
        candidates = by_func.get(key.func, [])
        if not candidates:
            unmatched.append(member)
            continue
        if len(candidates) > 1 and key.cls:
            candidates = [c for c in candidates if c[1].cls == key.cls] or candidates
        if len(candidates) > 1 and key.module:
            tail = key.module.split(".")[-1]
            narrowed = [c for c in candidates if tail and tail in c[1].module.split(".")]
            candidates = narrowed or candidates
        if len(candidates) == 1:
            result.matched[member] = candidates[0][0]
        else:
            ambiguous.append(member)

    result.unmappable = tuple(unmappable)
    result.unmatched = tuple(unmatched)
    result.ambiguous = tuple(ambiguous)
    return result


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
