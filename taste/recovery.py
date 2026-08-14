"""The trap handler: a step failure is a fault, not a command to retry.

The kernel used to answer every failure the same way — reset, retry, and
after N tries give up. A flaky test, a verification that was already broken
before the step ran, an under-specified step, a missing dependency, and a
genuine bug all got that one response. Nothing ever asked *why*.

This module reads the fault frame, names the fault, and dispatches to a typed
handler. The design constraint that shapes everything here: **diagnosis is
free**. Exit codes, whether the diff was empty, whether the identical test
failed identically last time, whether the failing tests live in files the
step actually touched — all of it is already in hand or one ``git diff
--name-only`` away. No model call is made, or needed, to separate most of the
taxonomy. Paying a model to guess what the exit code already states would be
slower, dearer, and less reliable.

Three types carry the design:

``Signals``   the fault frame — everything observable about a failure.
``Diagnosis`` a named failure class, from a first-match-wins rule table.
``Action``    one of seven verbs the kernel knows how to execute.

Two policies map diagnosis to action. :class:`FixedPolicy` ignores the
diagnosis entirely and always returns the same verb — that is how the
harness reproduces its own historical behavior and how the paper's arms are
expressed as configuration rather than as separate code paths.
:class:`TieredPolicy` routes on the diagnosis, with one invariant that is the
whole safety argument: **when the diagnosis is unknown or low-confidence it
returns exactly what the baseline policy returns.** A wrong diagnosis can
fail to help; it can never do worse than always rolling back.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Literal

# ================================================================= taxonomy


class FailureClass(StrEnum):
    """Why a step failed. Named so the response can differ."""

    FLAKY = "flaky"
    ENV_BROKEN = "env_broken"
    MISSING_DEPENDENCY = "missing_dependency"
    BAD_VERIFICATION = "bad_verification"
    """The check was already failing before the step ran — nothing the worker
    does can make it pass, so retrying is pure waste."""
    IMPLEMENTATION_BUG = "implementation_bug"
    OUT_OF_SCOPE_REGRESSION = "out_of_scope_regression"
    """It broke something it never touched. The edit is net-negative, so the
    tree is better off without it."""
    UNDER_SPECIFIED_STEP = "under_specified_step"
    STEP_TOO_LARGE = "step_too_large"
    CAPABILITY_GAP = "capability_gap"
    HARNESS_FAULT = "harness_fault"
    """Our bug, never the agent's. Never answered with guidance — telling a
    model to work around the harness's own defect teaches it nothing."""
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


class ActionKind(StrEnum):
    """What to do about it. Closed set — seven verbs, no more."""

    ACCEPT = "accept"
    REVERIFY = "reverify"
    RETRY_SAME = "retry_same"
    RETRY_WITH_GUIDANCE = "retry_with_guidance"
    REPAIR_IN_PLACE = "repair_in_place"
    ROLLBACK_AND_RETRY = "rollback_and_retry"
    HALT = "halt"


RETRY_FAMILY = frozenset(
    {
        ActionKind.RETRY_SAME,
        ActionKind.RETRY_WITH_GUIDANCE,
        ActionKind.REPAIR_IN_PLACE,
        ActionKind.ROLLBACK_AND_RETRY,
    }
)

# Exceptions raised by our own tooling, not by the model's work.
_HARNESS_EXC = frozenset({"PermissionError", "KeyError", "AttributeError", "TypeError"})


# ================================================================= fault frame


@dataclass(frozen=True)
class Signals:
    """The fault frame. Every field is free: already in hand, one git call,
    or a regex over output the harness already captured."""

    # from the verdict
    exit_code: int | None = None
    duration_s: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    # from git
    changed_files: tuple[str, ...] = ()
    diff_lines: int = 0
    empty_diff: bool = False
    # from the worker
    tool_calls: int = 0
    tool_errors: int = 0
    tool_error_kinds: tuple[str, ...] = ()
    stopped_reason: str = ""
    turns_exhausted: bool = False
    interrupt_kind: str | None = None
    # parsed out of the output
    failing_tests: tuple[str, ...] = ()
    failed_count: int | None = None
    passed_count: int | None = None
    env_marker: str | None = None
    missing_module: str | None = None
    # derived across attempts
    fingerprint: str = ""
    repeat_of_previous: bool = False
    progress_delta: int | None = None
    in_blast_radius: bool = True
    # probe result ($0 API; M3 fills this in, "skipped" until then)
    baseline_probe: Literal["pass", "fail_same", "fail_other", "skipped"] = "skipped"
    probe_seconds: float = 0.0


EMPTY_SIGNALS = Signals()

_ENV_MARKERS = (
    ("command not found", "command_not_found"),
    ("permission denied", "permission_denied"),
    ("no space left on device", "disk_full"),
    ("killed", "killed"),
    ("connection refused", "connection_refused"),
    ("address already in use", "port_in_use"),
    ("segmentation fault", "segfault"),
)

_MISSING_MODULE = re.compile(
    r"(?:ModuleNotFoundError|ImportError):\s*No module named ['\"]([\w.]+)['\"]"
)
_PYTEST_NODE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+?)(?:\s|$)", re.MULTILINE)
_PYTEST_TALLY = re.compile(r"(\d+)\s+(passed|failed|error|errors)")

# Volatile substrings that differ between two identical failures. Left in,
# they would make every fingerprint unique and the no-progress detector dead.
_NOISE = (
    re.compile(r"0x[0-9a-fA-F]+"),
    re.compile(r"/tmp/[\w./-]+"),
    re.compile(r"/private/var/folders/[\w./-]+"),
    re.compile(r"\b\d+\.\d+s\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\S*"),
    re.compile(r"line \d+"),
)


def parse_output(stdout: str, stderr: str) -> dict[str, Any]:
    """Pull the cheap structured facts out of a verification's output."""
    blob = f"{stdout}\n{stderr}"
    lowered = blob.lower()

    env_marker = next((tag for needle, tag in _ENV_MARKERS if needle in lowered), None)
    module_match = _MISSING_MODULE.search(blob)

    tallies = {kind: int(n) for n, kind in _PYTEST_TALLY.findall(lowered)}
    failed = tallies.get("failed", 0) + tallies.get("error", 0) + tallies.get("errors", 0)

    return {
        "failing_tests": tuple(dict.fromkeys(_PYTEST_NODE.findall(blob))),
        "failed_count": failed if tallies else None,
        "passed_count": tallies.get("passed") if tallies else None,
        "env_marker": env_marker,
        "missing_module": module_match.group(1) if module_match else None,
    }


def fingerprint(
    *,
    exit_code: int | None,
    failing_tests: tuple[str, ...],
    stdout: str,
    stderr: str,
) -> str:
    """A stable hash of *how* a step failed.

    Two attempts that fail the same way must collide, or the no-progress
    detector never fires. Test node ids dominate when present because they
    are already normalized; otherwise free text is stripped of the parts that
    vary run to run (temp paths, addresses, durations, timestamps, line
    numbers).
    """
    if failing_tests:
        material = "|".join(sorted(failing_tests))
    else:
        material = f"{stdout}\n{stderr}"
        for pattern in _NOISE:
            material = pattern.sub("", material)
        material = " ".join(material.split())[-2000:]
    return hashlib.sha256(f"{exit_code}:{material}".encode()).hexdigest()[:16]


def observe(
    *,
    verdict: Any,
    worker: Any,
    changed_files: tuple[str, ...] = (),
    diff_lines: int = 0,
    history: StepHistory | None = None,
    interrupt_kind: str | None = None,
    baseline_probe: Literal["pass", "fail_same", "fail_other", "skipped"] = "skipped",
    probe_seconds: float = 0.0,
) -> Signals:
    """Build the fault frame. Pure, deterministic, and free of API calls."""
    evidence = getattr(verdict, "evidence", "") or ""
    reason = getattr(verdict, "reason", "") or ""
    stdout, _, stderr = evidence.partition("\n[stderr]\n")
    parsed = parse_output(stdout, stderr)

    exit_code = None
    match = re.search(r"exited (\d+)", reason)
    if match:
        exit_code = int(match.group(1))

    fp = fingerprint(
        exit_code=exit_code,
        failing_tests=parsed["failing_tests"],
        stdout=stdout,
        stderr=stderr,
    )
    previous = history.last() if history else None
    progress_delta = None
    if previous is not None and previous.failed_count is not None and parsed["failed_count"] is not None:
        progress_delta = previous.failed_count - parsed["failed_count"]

    return Signals(
        exit_code=exit_code,
        stdout_tail=stdout[-4000:],
        stderr_tail=stderr[-4000:],
        changed_files=tuple(changed_files),
        diff_lines=diff_lines,
        empty_diff=not changed_files,
        tool_calls=getattr(worker, "tool_calls", 0) or 0,
        tool_errors=getattr(worker, "tool_errors", 0) or 0,
        tool_error_kinds=tuple(getattr(worker, "tool_error_kinds", ()) or ()),
        stopped_reason=getattr(worker, "stopped_reason", "") or "",
        turns_exhausted=getattr(worker, "stopped_reason", "") == "tool_use",
        interrupt_kind=interrupt_kind,
        fingerprint=fp,
        repeat_of_previous=previous is not None and previous.fingerprint == fp,
        progress_delta=progress_delta,
        in_blast_radius=_in_blast_radius(parsed["failing_tests"], changed_files),
        baseline_probe=baseline_probe,
        probe_seconds=probe_seconds,
        **parsed,
    )


def run_baseline_probe(
    *,
    memory: Any,
    before: Any,
    command: str | None,
    failure_fingerprint: str,
    max_seconds: float = 30.0,
) -> tuple[Literal["pass", "fail_same", "fail_other", "skipped"], float]:
    """Was this verification already failing *before* the step ran?

    The cheapest useful question in the whole system. Re-running the check
    against the pre-step commit answers it with a logically sufficient
    condition and no false positives: if the command already failed there,
    nothing the worker did caused it, and no amount of retrying can fix it.
    An entire failure class is eliminated for the price of one command
    execution and zero API tokens.

    Returns ``fail_same`` when the pre-existing failure is *identical* (the
    check is simply broken), ``fail_other`` when it failed differently (the
    environment is unwell), ``pass`` when the check was healthy before — which
    is what licenses blaming the step for the failure.
    """
    import subprocess
    import time as _time

    if not command:
        return "skipped", 0.0

    started = _time.monotonic()
    try:
        with memory.probe_worktree(before.sha) as probe_path:
            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    cwd=probe_path,
                    capture_output=True,
                    text=True,
                    timeout=max_seconds,
                )
            except subprocess.TimeoutExpired:
                return "skipped", _time.monotonic() - started
            elapsed = _time.monotonic() - started
            if proc.returncode == 0:
                return "pass", elapsed
            parsed = parse_output(proc.stdout, proc.stderr)
            probe_fp = fingerprint(
                exit_code=proc.returncode,
                failing_tests=parsed["failing_tests"],
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
            return ("fail_same" if probe_fp == failure_fingerprint else "fail_other"), elapsed
    except Exception:
        # A probe that cannot run tells us nothing; it must never be the
        # reason a step fails.
        return "skipped", _time.monotonic() - started


def _in_blast_radius(failing_tests: tuple[str, ...], changed_files: tuple[str, ...]) -> bool:
    """Do the failing tests live in files this step actually touched?

    Unknown means True: without evidence that a failure is out of scope, the
    conservative reading is that the step caused it.
    """
    if not failing_tests or not changed_files:
        return True
    changed = {f.rsplit("/", 1)[-1] for f in changed_files}
    changed |= {f.rsplit("/", 1)[-1].removesuffix(".py") for f in changed_files}
    for node in failing_tests:
        path = node.split("::", 1)[0]
        name = path.rsplit("/", 1)[-1]
        if name in changed or name.removesuffix(".py") in changed:
            return True
        stem = name.removeprefix("test_").removesuffix(".py")
        if stem and any(stem in c for c in changed):
            return True
    return False


# ================================================================= diagnosis


@dataclass(frozen=True)
class Hypothesis:
    failure_class: FailureClass
    confidence: float
    rationale: str
    rule_id: str


@dataclass(frozen=True)
class Diagnosis:
    top: Hypothesis
    tier: Literal["off", "deterministic"] = "deterministic"
    cost_usd: float = 0.0

    @property
    def failure_class(self) -> FailureClass:
        return self.top.failure_class

    @property
    def confidence(self) -> float:
        return self.top.confidence


@dataclass(frozen=True)
class Rule:
    rule_id: str
    failure_class: FailureClass
    confidence: float
    predicate: Callable[[Signals, StepHistory], bool]
    rationale: str = ""


# First match wins. Ordered by "how nearly always right this is when it
# fires", so a high-certainty structural signal is never shadowed by a
# heuristic further down.
RULES: tuple[Rule, ...] = (
    Rule("R0.interrupt", FailureClass.INTERRUPTED, 1.00,
         lambda s, h: s.interrupt_kind is not None,
         "a guardrail stopped the attempt"),
    Rule("R1.tool_errors", FailureClass.HARNESS_FAULT, 1.00,
         lambda s, h: bool(set(s.tool_error_kinds) & _HARNESS_EXC),
         "the tooling itself raised"),
    Rule("R2.baseline_fails_same", FailureClass.BAD_VERIFICATION, 1.00,
         lambda s, h: s.baseline_probe == "fail_same",
         "the check already failed identically before the step ran"),
    Rule("R2b.baseline_fails_other", FailureClass.ENV_BROKEN, 0.95,
         lambda s, h: s.baseline_probe == "fail_other",
         "the check failed before the step ran, differently"),
    Rule("R4.env_marker", FailureClass.ENV_BROKEN, 0.95,
         lambda s, h: s.env_marker is not None,
         "the environment reported a failure of its own"),
    Rule("R5.missing_module", FailureClass.MISSING_DEPENDENCY, 0.90,
         lambda s, h: s.missing_module is not None,
         "an import failed for a module that is not installed"),
    Rule("R6.empty_diff", FailureClass.UNDER_SPECIFIED_STEP, 0.90,
         lambda s, h: s.empty_diff and s.tool_calls == 0,
         "the worker changed nothing and called no tools"),
    Rule("R7.exhaustion", FailureClass.STEP_TOO_LARGE, 0.85,
         lambda s, h: s.stopped_reason == "max_tokens" or s.turns_exhausted,
         "the worker ran out of room before finishing"),
    Rule("R8.out_of_scope", FailureClass.OUT_OF_SCOPE_REGRESSION, 0.85,
         lambda s, h: s.baseline_probe == "pass"
         and bool(s.failing_tests)
         and not s.in_blast_radius,
         "it broke tests in files it never touched"),
    Rule("R9.progress", FailureClass.IMPLEMENTATION_BUG, 0.80,
         lambda s, h: (s.progress_delta or 0) > 0,
         "fewer failures than last attempt — it is converging"),
    Rule("R9b.in_scope", FailureClass.IMPLEMENTATION_BUG, 0.75,
         lambda s, h: s.baseline_probe == "pass" and s.in_blast_radius,
         "the failure is in code this step wrote"),
    Rule("R10.no_progress", FailureClass.CAPABILITY_GAP, 0.70,
         lambda s, h: h.no_progress_streak() >= 2,
         "repeated identical failures — it is not getting closer"),
)

RESIDUAL = Hypothesis(
    failure_class=FailureClass.IMPLEMENTATION_BUG,
    confidence=0.30,
    rationale="no rule matched; treated as an ordinary bug",
    rule_id="residual",
)


def diagnose(signals: Signals, history: StepHistory) -> Diagnosis:
    """Name the fault. Deterministic, first match wins, zero API cost."""
    for rule in RULES:
        try:
            matched = rule.predicate(signals, history)
        except Exception:
            continue  # a broken rule must never break the run
        if matched:
            return Diagnosis(
                top=Hypothesis(
                    failure_class=rule.failure_class,
                    confidence=rule.confidence,
                    rationale=rule.rationale,
                    rule_id=rule.rule_id,
                )
            )
    return Diagnosis(top=RESIDUAL)


# ================================================================= actions


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    reason: str
    guidance: str | None = None
    resets: bool = False
    """Stated explicitly rather than derived from the verb, so wasted-work
    accounting never has to interpret an enum."""


@dataclass(frozen=True)
class Budget:
    attempts_used: int = 0
    max_attempts: int = 3
    actions_used: int = 0
    max_actions: int = 8
    usd_spent_on_step: float = 0.0
    step_budget_usd: float | None = None

    @property
    def exhausted(self) -> bool:
        if self.attempts_used >= self.max_attempts:
            return True
        if self.actions_used >= self.max_actions:
            return True
        return (
            self.step_budget_usd is not None
            and self.usd_spent_on_step >= self.step_budget_usd
        )


# ================================================================= history


@dataclass
class GuidanceBook:
    """Itemized, append-only, deduplicated, capped.

    Guidance accumulates as discrete bullets keyed by failure fingerprint.
    Nothing is ever rewritten or model-summarized, so there is no rewrite
    step for context collapse to collapse — the failure mode that makes
    naive "feed the last error back" schemes degrade over a long step.
    """

    items: list[tuple[str, str]] = field(default_factory=list)
    cap: int = 6

    def add(self, fp: str, bullet: str) -> None:
        if any(existing_fp == fp for existing_fp, _ in self.items):
            return
        self.items.append((fp, bullet))
        if len(self.items) > self.cap:
            del self.items[0]

    def render(self, max_chars: int = 1200) -> str:
        if not self.items:
            return ""
        lines = [f"- {bullet}" for _fp, bullet in self.items]
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


@dataclass
class AttemptRecord:
    attempt: int
    fingerprint: str
    failure_class: FailureClass
    action: ActionKind
    failed_count: int | None = None
    passed: bool = False


@dataclass
class StepHistory:
    """What has already been tried for one step, and how it went."""

    step_id: str
    records: list[AttemptRecord] = field(default_factory=list)
    action_counts: dict[ActionKind, int] = field(default_factory=dict)
    book: GuidanceBook = field(default_factory=GuidanceBook)
    reset_count: int = 0
    baseline_probe: Literal["pass", "fail_same", "fail_other", "skipped"] = "skipped"
    """Memoized across attempts: the pre-step tree never changes, so the
    probe's answer cannot either. Paying for it twice would be waste."""

    def last(self) -> AttemptRecord | None:
        return self.records[-1] if self.records else None

    def record(self, record: AttemptRecord) -> None:
        self.records.append(record)
        self.action_counts[record.action] = self.action_counts.get(record.action, 0) + 1
        if record.action is ActionKind.ROLLBACK_AND_RETRY:
            self.reset_count += 1

    def no_progress_streak(self) -> int:
        """Consecutive most-recent attempts that failed identically."""
        if len(self.records) < 2:
            return 0
        streak = 0
        newest = self.records[-1].fingerprint
        for record in reversed(self.records):
            if record.fingerprint == newest and not record.passed:
                streak += 1
            else:
                break
        return streak


# ================================================================= policies


@dataclass(frozen=True)
class RecoveryConfig:
    """How the trap handler behaves. Defaults reproduce the historical kernel."""

    enabled: bool = False
    policy: Literal["fixed", "tiered"] = "fixed"
    fixed_action: ActionKind = ActionKind.ROLLBACK_AND_RETRY
    max_actions: int = 8
    guidance_max_chars: int = 1200
    baseline_probe: bool = False
    probe_max_seconds: float = 30.0

    @classmethod
    def arm(cls, name: str) -> RecoveryConfig:
        """The paper's arms, as configuration rather than separate code paths.

        A1  self-verification: take the monitor's FAIL as final, never retry.
        A2  repair in place: keep the work, fix forward.
        A3  monitor-gated rollback: reset and retry — the historical kernel.
        A3' attempt-matched control: retry with the same guidance, no reset.
            The only arm that separates "reset helps" from "more tries help".
        """
        arms = {
            "A1": cls(enabled=True, policy="fixed", fixed_action=ActionKind.ACCEPT),
            "A2": cls(enabled=True, policy="fixed", fixed_action=ActionKind.REPAIR_IN_PLACE),
            "A3": cls(enabled=True, policy="fixed", fixed_action=ActionKind.ROLLBACK_AND_RETRY),
            "A3prime": cls(
                enabled=True, policy="fixed", fixed_action=ActionKind.RETRY_WITH_GUIDANCE
            ),
            "tiered": cls(enabled=True, policy="tiered", baseline_probe=True),
        }
        if name not in arms:
            raise ValueError(f"unknown arm {name!r}; known: {sorted(arms)}")
        return arms[name]


class FixedPolicy:
    """Always the same verb, whatever the diagnosis says.

    Not a degenerate case — this is how the harness reproduces its own prior
    behavior exactly, and how each experimental arm is expressed without a
    second code path to keep in sync.
    """

    name = "fixed"

    def __init__(self, action: ActionKind = ActionKind.ROLLBACK_AND_RETRY) -> None:
        self.action = action

    def decide(
        self,
        *,
        diagnosis: Diagnosis,
        history: StepHistory,
        budget: Budget,
        config: RecoveryConfig,
        signals: Signals = EMPTY_SIGNALS,
    ) -> Action:
        if self.action is ActionKind.ACCEPT:
            return Action(ActionKind.ACCEPT, reason="policy=accept (self-verification arm)")
        if budget.exhausted:
            # Halting does not suspend the arm's discipline. A resetting arm
            # discards its last failed attempt too, or the step would leave
            # unverified work on the branch precisely when it gave up.
            return Action(
                ActionKind.HALT,
                reason="budget exhausted",
                resets=self.action is ActionKind.ROLLBACK_AND_RETRY,
            )
        return Action(
            kind=self.action,
            reason=f"policy=fixed({self.action.value})",
            guidance=history.book.render(config.guidance_max_chars) or None,
            resets=self.action is ActionKind.ROLLBACK_AND_RETRY,
        )


class TieredPolicy:
    """Route on the diagnosis.

    The invariant that makes this safe to enable: at ``UNKNOWN`` or
    confidence below 0.5 it returns exactly what the baseline returns. A
    wrong diagnosis can only forfeit an improvement; it cannot do worse than
    always rolling back.
    """

    name = "tiered"

    #: How many times one verb may be chosen for a single step before it is
    #: considered unproductive and the next option is taken.
    CAPS: ClassVar[dict[ActionKind, int]] = {
        ActionKind.REVERIFY: 1,
        ActionKind.RETRY_SAME: 1,
        ActionKind.RETRY_WITH_GUIDANCE: 2,
        ActionKind.REPAIR_IN_PLACE: 2,
        ActionKind.ROLLBACK_AND_RETRY: 3,
    }

    def decide(
        self,
        *,
        diagnosis: Diagnosis,
        history: StepHistory,
        budget: Budget,
        config: RecoveryConfig,
        signals: Signals = EMPTY_SIGNALS,
    ) -> Action:
        action = self._route(diagnosis, history, budget, config, signals)
        return self._bound(action, history, budget)

    # -------------------------------------------------------- routing

    def _route(
        self,
        diagnosis: Diagnosis,
        history: StepHistory,
        budget: Budget,
        config: RecoveryConfig,
        signals: Signals,
    ) -> Action:
        cls = diagnosis.failure_class
        guidance = history.book.render(config.guidance_max_chars) or None

        if budget.exhausted:
            return Action(ActionKind.HALT, reason="budget exhausted", resets=True)

        # Never hand the model guidance about our own bug, or about a
        # guardrail that stopped it. Neither is something it can act on.
        # These halts preserve the tree: the work was never fairly judged, so
        # discarding it would destroy the evidence needed to diagnose us.
        if cls in (FailureClass.INTERRUPTED, FailureClass.HARNESS_FAULT):
            return Action(ActionKind.HALT, reason=f"{cls.value}: not the agent's failure")

        if cls is FailureClass.BAD_VERIFICATION:
            return Action(
                ActionKind.HALT,
                reason="the verification was already failing before this step ran",
            )

        if cls is FailureClass.ENV_BROKEN:
            if history.action_counts.get(ActionKind.REVERIFY, 0) == 0:
                return Action(ActionKind.REVERIFY, reason="environment fault; re-checking once")
            return Action(ActionKind.HALT, reason="environment fault persisted")

        if cls is FailureClass.MISSING_DEPENDENCY:
            package = signals.missing_module or "the missing module"
            return Action(
                ActionKind.RETRY_WITH_GUIDANCE,
                reason=f"missing dependency: {package}",
                guidance=_join(guidance, f"`{package}` is not importable — install or vendor it."),
            )

        if cls is FailureClass.UNDER_SPECIFIED_STEP:
            # Nothing was written, so there is nothing to reset.
            return Action(
                ActionKind.RETRY_WITH_GUIDANCE,
                reason="the worker produced no changes",
                guidance=_join(guidance, "The previous attempt changed nothing. Make the edit."),
            )

        if cls is FailureClass.STEP_TOO_LARGE:
            return Action(
                ActionKind.RETRY_WITH_GUIDANCE,
                reason="the step did not fit in one attempt",
                guidance=_join(guidance, "Do the smallest verifiable subset of this step first."),
            )

        if cls is FailureClass.OUT_OF_SCOPE_REGRESSION:
            # The edit broke things it never touched: net-negative, discard it.
            return Action(
                ActionKind.ROLLBACK_AND_RETRY,
                reason="broke tests outside the files it changed",
                guidance=_join(guidance, "The last attempt broke unrelated tests. Keep changes local."),
                resets=True,
            )

        if cls is FailureClass.IMPLEMENTATION_BUG and (signals.progress_delta or 0) > 0:
            # Fewer failures than last time: it is converging, so keep the work.
            return Action(
                ActionKind.REPAIR_IN_PLACE,
                reason=f"converging ({signals.progress_delta} fewer failures)",
                guidance=guidance,
            )

        if cls is FailureClass.CAPABILITY_GAP:
            if history.reset_count == 0:
                return Action(
                    ActionKind.ROLLBACK_AND_RETRY,
                    reason="no progress across attempts; one clean retry",
                    guidance=guidance,
                    resets=True,
                )
            return Action(ActionKind.HALT, reason="repeated identical failures; not converging")

        # IMPLEMENTATION_BUG without progress evidence, UNKNOWN, and anything
        # low-confidence all land here — identical to the baseline arm.
        return Action(
            ActionKind.ROLLBACK_AND_RETRY,
            reason=f"{cls.value} (conf {diagnosis.confidence:.2f}); baseline response",
            guidance=guidance,
            resets=True,
        )

    # -------------------------------------------------------- loop prevention

    def _bound(self, action: Action, history: StepHistory, budget: Budget) -> Action:
        """Stop a policy from choosing the same unproductive verb forever.

        A halt produced here inherits the reset discipline of the action it
        replaced: giving up is not a reason to leave unverified work behind.
        """
        if action.kind is ActionKind.HALT:
            return action

        # L1: identical failures twice running means retrying is not working.
        if action.kind in RETRY_FAMILY and history.no_progress_streak() >= 2:
            return Action(
                ActionKind.HALT,
                reason=f"no progress across {history.no_progress_streak()} identical failures",
                resets=action.resets,
            )

        # L2: each verb has a per-step ceiling.
        cap = self.CAPS.get(action.kind)
        if cap is not None and history.action_counts.get(action.kind, 0) >= cap:
            return Action(
                ActionKind.HALT,
                reason=f"{action.kind.value} exhausted its cap of {cap} for this step",
                resets=action.resets,
            )

        # L3: a hard ceiling on total decisions, whatever they were.
        if budget.actions_used + 1 >= budget.max_actions:
            return Action(
                ActionKind.HALT, reason="action ceiling reached", resets=action.resets
            )

        return action


def build_policy(config: RecoveryConfig) -> FixedPolicy | TieredPolicy:
    if config.policy == "tiered":
        return TieredPolicy()
    return FixedPolicy(config.fixed_action)


def guidance_bullet(signals: Signals, diagnosis: Diagnosis) -> str:
    """One concrete sentence about this failure, for the guidance book."""
    if signals.failing_tests:
        names = ", ".join(signals.failing_tests[:3])
        more = f" (+{len(signals.failing_tests) - 3} more)" if len(signals.failing_tests) > 3 else ""
        return f"{diagnosis.failure_class.value}: failing — {names}{more}"
    tail = " ".join((signals.stderr_tail or signals.stdout_tail).split())[-200:]
    return f"{diagnosis.failure_class.value}: {tail}" if tail else diagnosis.failure_class.value


def _join(existing: str | None, addition: str) -> str:
    return f"{existing}\n- {addition}" if existing else f"- {addition}"
