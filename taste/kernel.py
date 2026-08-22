"""Kernel: the orchestration loop.

This is where the thesis stops being metaphor and starts executing:

    plan  ->  for each step:
                checkpoint = memory.head()
                worker.execute(step)              # mutates workspace
                verdict = monitor.evaluate(step)  # reads workspace
                memory.checkpoint(step.id, ...)   # commit worker + verdict
                if verdict.failed:
                    memory.rollback_to(checkpoint)
                    retry  (up to budget)  or  halt

Every step is a commit. Every failure is a `git reset --hard`. The kernel
doesn't "know" anything about LLMs beyond calling the three core functions —
swap the cores' implementations and the orchestration is unchanged.
"""

from __future__ import annotations

import contextlib
import functools
import json
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from taste import cores, integrate, recovery
from taste.agent import AgentSpec
from taste.config import HarnessConfig
from taste.cores import MonitorResult, Plan, PlannerError, Step, WorkerResult
from taste.guardrails import GuardConfig, Guardrails
from taste.journal import FileChange, Journal, card_from_step, parse_numstat
from taste.llm import (
    DEFAULT_TEMPERATURE,
    LLM,
    MODEL_MONITOR,
    MODEL_PLANNER,
    MODEL_WORKER,
    BudgetExceeded,
    InfraFailure,
    PricingError,
    RunStats,
    ensure_priced,
    prompt_sha,
)
from taste.memory import Checkpoint, Memory, MergeConflict
from taste.shadow import ShadowLog
from taste.tools import ToolRegistry, make_builtin_tools

# Typed harness failures that must be classified (never crash unclassified) and
# must leave the session branch clean when they interrupt a step.
_HARNESS_FAILURES = (BudgetExceeded, InfraFailure, PricingError)


def _failure_kind_of(exc: Exception) -> Literal["task", "infra", "budget"]:
    """Single source of truth for the failure taxonomy (see RunResult)."""
    kind = getattr(exc, "failure_kind", None)
    if kind in ("infra", "budget"):
        return kind
    return "task" if isinstance(exc, PlannerError) else "infra"

# ================================================================= events


@dataclass
class Event:
    kind: str
    payload: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def __str__(self) -> str:
        body = " ".join(f"{k}={v}" for k, v in self.payload.items() if k != "evidence")
        return f"[{self.kind}] {body}"

    def to_json(self) -> dict[str, Any]:
        return {"ts": self.ts, "kind": self.kind, "payload": self.payload}


# ================================================================= results


@dataclass(frozen=True)
class StepContext:
    """Where the step running on *this thread* lives.

    A step may execute on the session branch or on its own worktree, and only
    the kernel knows which. Publishing it here lets a worker — or a test's
    ``worker_override`` — locate its own working tree without reaching into
    the kernel's call stack, which silently breaks whenever the kernel is
    refactored. Set by :meth:`Kernel._run_step` in the executing thread, so
    it is correct under ``ThreadPoolExecutor`` without any context copying.
    """

    step: Step
    workspace: Path
    branch: str


CURRENT_STEP: ContextVar[StepContext | None] = ContextVar("taste_current_step", default=None)


def current_step() -> StepContext:
    """The step executing on this thread. Raises outside a step."""
    ctx = CURRENT_STEP.get()
    if ctx is None:
        raise RuntimeError("current_step() called outside a running step")
    return ctx


@dataclass
class StepOutcome:
    step: Step
    worker: WorkerResult
    verdict: MonitorResult
    checkpoint: Checkpoint
    attempts: int
    rolled_back: bool


@dataclass
class RunResult:
    task: str
    session_id: str
    branch: str
    status: Literal["completed", "failed"]
    plan: Plan
    outcomes: list[StepOutcome]
    final_sha: str
    elapsed_seconds: float
    failure_reason: str | None = None
    stats: RunStats | None = None
    # Failure taxonomy for analysis: "task" (the agent genuinely failed),
    # "infra" (API errors persisted past retries), "budget" (dollar cap hit).
    # Infra/budget runs must be excluded from task-success rates, never pooled.
    failure_kind: Literal["task", "infra", "budget"] | None = None

    def summary(self) -> str:
        passed = sum(1 for o in self.outcomes if o.verdict.passed)
        total = len(self.plan.steps) if self.plan.steps else 0
        rolled = sum(1 for o in self.outcomes if o.rolled_back)
        base = (
            f"status={self.status} steps={passed}/{total} "
            f"rollbacks={rolled} branch={self.branch} head={self.final_sha[:7]}"
        )
        if self.stats:
            base += f" cost=${self.stats.total_cost_usd:.4f} cache={self.stats.cache_hit_rate:.0%}"
        return base


# ================================================================= kernel


@dataclass
class _ObservingHook:
    """Adds a shadow observation after every tool call, wrapping any inner hook.

    Composition rather than replacement: guardrails are an arm-defining
    subsystem, and an observer bolted onto them would only fire in arms that
    guard — making observation density follow the treatment, which is the
    exact bias a finer grid is meant to remove.

    Observation runs **after** the inner hook, so a veto is already recorded
    before the tree is sampled, and the sample reflects what actually
    happened rather than what was attempted.

    Fail-open, per the TurnHook contract. A hook must never be the reason a
    run halts, and an instrument that can halt the thing it measures is worse
    than no instrument.
    """

    observe: Callable[[str], None]
    inner: Any = None

    def before_turn(self, turn: int) -> Any:
        return self.inner.before_turn(turn) if self.inner is not None else None

    def before_tool(self, turn: int, name: str, payload: dict[str, Any]) -> Any:
        if self.inner is not None:
            return self.inner.before_tool(turn, name, payload)
        return cores.ALLOW

    def after_tool(
        self, turn: int, name: str, payload: dict[str, Any],
        output: str, elapsed_s: float, decision: Any,
    ) -> None:
        if self.inner is not None:
            with contextlib.suppress(Exception):
                self.inner.after_tool(turn, name, payload, output, elapsed_s, decision)
        # A read-only tool leaves the tree byte-identical, and ShadowLog
        # already declines to write a commit for an unchanged tree — so no
        # tool classification is needed here, and none is done.
        with contextlib.suppress(Exception):
            self.observe(name)

    def after_turn(self, turn: int, message: Any, stop_reason: str) -> None:
        if self.inner is not None:
            with contextlib.suppress(Exception):
                self.inner.after_turn(turn, message, stop_reason)


class Kernel:
    """The orchestrator. Owns the loop; owns nothing else.

    Parameters
    ----------
    workspace:
        Path to a git repository that the agent is allowed to modify.
    llm:
        Optional LLM client. Required for real runs; tests may pass ``None``
        and wire their own cores.
    max_retries:
        How many times a failed step is retried (with monitor feedback fed back
        into the worker) before the kernel halts the run.
    on_event:
        Optional callback invoked on every state transition. Used by the CLI
        for rich printing and by tests for assertions. Events are also
        persisted to ``<workspace>/.taste/events.jsonl`` unconditionally — the
        dashboard reads that file.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        llm: LLM | None = None,
        max_retries: int = 2,
        on_event: Callable[[Event], None] | None = None,
        max_parallel: int = 4,
        planner_model: str | None = None,
        journal: bool = False,
        recovery_config: recovery.RecoveryConfig | None = None,
        guard_config: GuardConfig | None = None,
        two_phase_merge: bool = False,
        union_gate: bool = True,
        shadow: bool = False,
        observe_tools: bool = False,
        config: HarnessConfig | None = None,
    ) -> None:
        # A HarnessConfig names an entire arm in one object and one hash. When
        # given it wins outright, so a run's identity cannot be half-specified
        # by a config and half by loose keyword arguments.
        if config is not None:
            from taste.config import kernel_kwargs

            resolved = kernel_kwargs(config)
            max_retries = resolved["max_retries"]
            max_parallel = resolved["max_parallel"]
            planner_model = resolved["planner_model"]
            journal = resolved["journal"]
            recovery_config = resolved["recovery_config"]
            guard_config = resolved["guard_config"]
            two_phase_merge = resolved["two_phase_merge"]
            union_gate = resolved["union_gate"]
            shadow = resolved["shadow"]
            observe_tools = resolved["observe_tools"]
        self.config = config
        self.workspace = Path(workspace).resolve()
        self.llm = llm
        self.max_retries = max_retries
        self.on_event = on_event or (lambda _e: None)
        self.max_parallel = max_parallel
        # Off by default: with journalling disabled the kernel must produce a
        # byte-identical event stream and commit chain to the pre-journal
        # harness. tests/test_journal.py asserts exactly that.
        self.journal_enabled = journal
        # Disabled by default. When off, the fault path answers every failure
        # with rollback-and-retry — exactly the pre-recovery kernel — and no
        # diagnosis runs at all.
        self.recovery_config = recovery_config or recovery.RecoveryConfig()
        self._policy = recovery.build_policy(self.recovery_config)
        # Also off by default; when disabled no hook is installed at all, so
        # the worker loop is byte-identical to the unguarded version.
        self.guard_config = guard_config or GuardConfig()
        # Off by default: the sequential merge loop stays the default path so
        # single-step waves and existing behavior are untouched.
        self.two_phase_merge = two_phase_merge
        self.union_gate = union_gate
        # Observational checkpointing. Off by default; when on it must be
        # invisible to the agent, the Monitor and the session branch — it
        # exists to measure the run, not to participate in it.
        self.shadow_enabled = shadow
        # A finer observation grid. See HarnessConfig.observe_tools for why
        # this is a pre-registered protocol knob and not a default.
        self.observe_tools = observe_tools
        self._shadow: ShadowLog | None = None
        # Planner model is a kernel knob, NOT spec.model: the agent spec's
        # model field configures the Worker only (role-split invariant).
        self.planner_model = planner_model
        self._events_path: Path | None = None
        self._runtime_dir_cache: Path | None = None
        # The session currently executing. Set by run(); read by anchor
        # naming, which must not infer it from a branch name.
        self._session_id: str | None = None
        # Guards _emit + LLM stats when multiple workers run on worktrees concurrently.
        self._lock = threading.Lock()

    @property
    def _runtime_dir(self) -> Path:
        """Rollback-surviving runtime dir (events, manifests) inside git's own
        metadata. Resolved via the repo's actual git dir so worktree checkouts
        (where ``.git`` is a file, not a directory) work too.
        """
        if self._runtime_dir_cache is None:
            from git import Repo

            self._runtime_dir_cache = Path(Repo(self.workspace).git_dir) / "taste"
        return self._runtime_dir_cache

    # ------------------------------------------------------------ public

    def run(
        self,
        *,
        task: str,
        spec: AgentSpec,
        session_id: str | None = None,
        base_ref: str = "HEAD",
        plan_override: Plan | None = None,
        worker_override: Callable[[Step, Plan], WorkerResult] | None = None,
    ) -> RunResult:
        """Execute one task end-to-end on a fresh session branch.

        ``plan_override`` and ``worker_override`` exist so tests can swap in
        deterministic behavior without mocking the LLM client; they are the
        only hooks the kernel exposes for hermetic testing.
        """
        started = time.time()
        session_id = session_id or uuid.uuid4().hex[:8]
        self._session_id = session_id
        memory = Memory.open_session(self.workspace, session_id, base_ref=base_ref)
        self._open_event_log()
        self._emit(
            "run.start",
            task=task,
            session=session_id,
            branch=memory.branch,
            agent=spec.name,
        )
        self._write_manifest(task=task, session_id=session_id, spec=spec, memory=memory)
        self._shadow = (
            ShadowLog(
                memory,
                gitdir=self._runtime_dir,
                session=session_id,
                cost_pair_reader=(
                    (lambda: (self.llm.stats.total_cost_usd, self.llm.stats.total_work_usd))
                    if self.llm
                    else None
                ),
            )
            if self.shadow_enabled
            else None
        )
        self._observe(step_id="run", attempt=0, trigger="run")

        # Fail fast on unpriced models (before any money is spent), then plan.
        # PlannerError = the model refused or emitted a malformed plan.
        try:
            if self.llm is not None:
                self._validate_models(spec)
            plan = plan_override or self._plan(task, spec, memory)
        except (PlannerError, *_HARNESS_FAILURES) as exc:
            kind = _failure_kind_of(exc)
            self._emit("plan.error", reason=str(exc), failure_kind=kind)
            return self._halted_result(
                task=task,
                session_id=session_id,
                memory=memory,
                plan=Plan(task=task, steps=[]),
                outcomes=[],
                started=started,
                reason=f"planner: {exc}",
                failure_kind=kind,
            )

        self._persist_plan(memory, plan)

        try:
            waves = plan.waves()
        except ValueError as exc:
            self._emit("plan.error", reason=str(exc))
            return self._halted_result(
                task=task,
                session_id=session_id,
                memory=memory,
                plan=plan,
                outcomes=[],
                started=started,
                reason=f"plan graph: {exc}",
            )

        parallel_waves = sum(1 for w in waves if len(w) > 1)
        self._emit("plan.ready", steps=len(plan.steps), waves=len(waves), parallel_waves=parallel_waves)

        outcomes: list[StepOutcome] = []
        status: Literal["completed", "failed"] = "completed"
        failure_reason: str | None = None
        failure_kind: Literal["task", "infra", "budget"] | None = None

        try:
            for wave in waves:
                wave_outcomes, ok, fail_reason, wave_exc = self._run_wave(
                    wave=wave,
                    plan=plan,
                    spec=spec,
                    memory=memory,
                    worker_override=worker_override,
                )
                # Extend BEFORE re-raising a typed failure so completed sibling
                # steps of an interrupted parallel wave stay in the record.
                outcomes.extend(wave_outcomes)
                if wave_exc is not None:
                    raise wave_exc
                if not ok:
                    status = "failed"
                    failure_reason = fail_reason
                    failure_kind = "task"
                    failing = next(
                        (o for o in wave_outcomes if not o.verdict.passed), wave_outcomes[0]
                    )
                    self._emit(
                        "run.halt",
                        step=failing.step.id,
                        reason=fail_reason or failing.verdict.reason,
                    )
                    break
        except _HARNESS_FAILURES as exc:
            status = "failed"
            failure_reason = str(exc)
            failure_kind = _failure_kind_of(exc)
            self._emit("run.halt", reason=failure_reason, failure_kind=failure_kind)

        head = memory.head()
        result = RunResult(
            task=task,
            session_id=session_id,
            branch=memory.branch,
            status=status,
            plan=plan,
            outcomes=outcomes,
            final_sha=head.sha,
            elapsed_seconds=round(time.time() - started, 2),
            failure_reason=failure_reason,
            stats=self.llm.stats if self.llm else None,
            failure_kind=failure_kind,
        )
        self._emit_run_done(result)
        return result

    # ------------------------------------------------------------ waves

    def _run_wave(
        self,
        *,
        wave: list[Step],
        plan: Plan,
        spec: AgentSpec,
        memory: Memory,
        worker_override: Callable[[Step, Plan], WorkerResult] | None,
    ) -> tuple[list[StepOutcome], bool, str | None, Exception | None]:
        """Run a wave of parallel-eligible steps, then merge them into the session.

        Single-step waves run in-place on the session branch (identical
        semantics to the old sequential path). Multi-step waves spawn a
        worktree per step, run the workers concurrently, and — only if all
        succeed — merge each worker's branch back into the session.

        Returns ``(outcomes, ok, reason, typed_failure)``. When a typed
        harness failure interrupts a parallel wave, the completed sibling
        outcomes are still returned so the caller can record them before
        re-raising the failure.
        """
        if len(wave) == 1:
            step = wave[0]
            outcome = self._run_step(
                step=step,
                plan=plan,
                spec=spec,
                memory=memory,
                workspace=self.workspace,
                worker_override=worker_override,
            )
            ok = outcome.verdict.passed
            reason = None if ok else f"{step.id}: {outcome.verdict.reason}"
            return [outcome], ok, reason, None

        self._emit("wave.begin", steps=[s.id for s in wave], size=len(wave))

        # Spawn one worktree per step, branched off the current session head.
        # Use a non-nested branch prefix so git's ref-as-directory rule doesn't
        # collide with the session branch itself.
        worktrees: dict[str, Memory] = {}
        base_sha = memory.head().sha
        session_slug = memory.branch.replace("/", "-")
        try:
            for step in wave:
                wt_branch = f"taste-wt/{session_slug}-{step.id}"
                wt = memory.add_worktree(wt_branch, base=base_sha)
                worktrees[step.id] = wt
                self._emit(
                    "worktree.open",
                    step=step.id,
                    branch=wt_branch,
                    path=str(wt.repo_path),
                )

            # Run workers in parallel — one thread per worktree, capped.
            pool_size = min(len(wave), self.max_parallel)
            outcomes: dict[str, StepOutcome] = {}
            typed_failure: Exception | None = None
            with ThreadPoolExecutor(max_workers=pool_size) as pool:
                futures = {
                    pool.submit(
                        self._run_step,
                        step=step,
                        plan=plan,
                        spec=spec,
                        memory=worktrees[step.id],
                        workspace=worktrees[step.id].repo_path,
                        worker_override=worker_override,
                    ): step.id
                    for step in wave
                }
                for fut in as_completed(futures):
                    sid = futures[fut]
                    try:
                        outcomes[sid] = fut.result()
                    except _HARNESS_FAILURES as exc:
                        # Keep the first typed failure; still drain siblings so
                        # their outcomes (and spend) stay attributable.
                        if typed_failure is None:
                            typed_failure = exc

            ordered_outcomes = [outcomes[s.id] for s in wave if s.id in outcomes]

            if typed_failure is not None:
                aborted = [s.id for s in wave if s.id not in outcomes]
                self._emit("wave.halt", aborted=aborted, reason=str(typed_failure))
                return ordered_outcomes, False, str(typed_failure), typed_failure

            # If any worker failed, don't merge — halt with a clean session branch.
            failed = [o for o in ordered_outcomes if not o.verdict.passed]
            if failed:
                f = failed[0]
                reason = f"{f.step.id}: {f.verdict.reason}"
                self._emit("wave.halt", failed=[o.step.id for o in failed])
                return ordered_outcomes, False, reason, None

            if self.two_phase_merge:
                integration = integrate.integrate(
                    memory,
                    integrate.proposals_from_outcomes(ordered_outcomes, worktrees),
                    gate=self.union_gate,
                    emit=lambda kind, **payload: self._emit(kind, **payload),
                )
                if not integration.ok:
                    reason = (
                        f"{integration.conflicted[0][0]}: merge conflict"
                        if integration.conflicted
                        else f"combined tree failed verification: {integration.gate_failure}"
                    )
                    return ordered_outcomes, False, reason, None
                for step in wave:
                    self._emit(
                        "worktree.merge",
                        step=step.id,
                        source=worktrees[step.id].branch,
                        sha=memory.head().short_sha,
                    )
            else:
                # Merge each worktree's branch into the session, preserving topology.
                for step in wave:
                    wt = worktrees[step.id]
                    try:
                        merged = memory.merge_branch(
                            wt.branch,
                            message=f"merge: {step.id} from {wt.branch}",
                        )
                    except MergeConflict as exc:
                        self._emit("worktree.conflict", step=step.id, detail=exc.detail)
                        return (
                            ordered_outcomes,
                            False,
                            f"{step.id}: merge conflict — {exc.detail}",
                            None,
                        )
                    self._emit(
                        "worktree.merge",
                        step=step.id,
                        source=wt.branch,
                        sha=merged.short_sha,
                    )
            self._emit("wave.done", steps=[s.id for s in wave], size=len(wave))
            return ordered_outcomes, True, None, None
        finally:
            for wt in worktrees.values():
                with contextlib.suppress(Exception):
                    memory.remove_worktree(wt)

    def _run_step(
        self,
        *,
        step: Step,
        plan: Plan,
        spec: AgentSpec,
        memory: Memory,
        workspace: Path,
        worker_override: Callable[[Step, Plan], WorkerResult] | None,
    ) -> StepOutcome:
        """Execute one step's attempt-loop against ``memory``/``workspace``.

        When called with the session memory + main workspace, this is the
        classic sequential path. When called with a worktree memory + its
        path, the same loop drives a parallel worker in isolation.
        """
        tools = ToolRegistry()
        tools.extend(make_builtin_tools(workspace))

        before = memory.head()

        # Published for the duration of the step, in this thread. Parallel
        # steps run in their own threads, so each sees only its own context.
        ctx_token = CURRENT_STEP.set(
            StepContext(step=step, workspace=Path(workspace), branch=memory.branch)
        )
        try:
            return self._attempt_loop(
                step=step,
                plan=plan,
                spec=spec,
                memory=memory,
                workspace=workspace,
                worker_override=worker_override,
                tools=tools,
                before=before,
            )
        except _HARNESS_FAILURES:
            # A typed failure mid-step leaves half-finished, unverified edits
            # in the working tree. Left dirty, the NEXT run's first `git add
            # --all` checkpoint would silently absorb them — cross-run
            # contamination. Commit the debris (making untracked files
            # tracked), then hard-reset back to the pre-step checkpoint so the
            # branch is clean and the abort leaves no trace.
            memory.checkpoint(step.id, f"{step.id}: aborted mid-step", allow_empty=True)
            memory.rollback_to(before)
            self._emit("step.abort", id=step.id, to=before.short_sha)
            raise
        finally:
            CURRENT_STEP.reset(ctx_token)

    def _attempt_loop(
        self,
        *,
        step: Step,
        plan: Plan,
        spec: AgentSpec,
        memory: Memory,
        workspace: Path,
        worker_override: Callable[[Step, Plan], WorkerResult] | None,
        tools: ToolRegistry,
        before: Checkpoint,
    ) -> StepOutcome:
        attempts = 0
        rolled_back = False
        last_feedback: str | None = None
        journal = self._journal_for(memory)
        cost_before = self.llm.stats.total_cost_usd if self.llm else None
        # None means "recovery off": the fault path skips diagnosis entirely
        # and answers every failure the way the kernel always has.
        history = (
            recovery.StepHistory(step_id=step.id) if self.recovery_config.enabled else None
        )
        actions_used = 0
        # REPAIR_IN_PLACE reframes the next attempt as fixing what is on disk
        # rather than starting over; every other verb starts fresh.
        worker_mode: Literal["fresh", "repair"] = "fresh"
        # Set by REVERIFY: re-run the check against the unchanged tree without
        # invoking the worker again.
        reverifying = False
        worker = WorkerResult("", 0, "unknown")
        attempt_started = time.time()
        # One hook per STEP, not per attempt: a guard rebuilt each attempt
        # would re-baseline its spend and turn a step ceiling into a
        # per-attempt one. The observer inside it reads the attempt number
        # lazily for the same reason — the hook outlives every attempt.
        guard = self._hook_for(workspace, step, lambda: attempts)

        while attempts <= self.max_retries:
            if reverifying:
                reverifying = False
            else:
                attempts += 1
                attempt_started = time.time()
                self._emit("step.begin", id=step.id, attempt=attempts)

                if worker_override is not None:
                    worker = worker_override(step, plan)
                else:
                    worker = self._worker(
                        step, plan, spec, tools, last_feedback, hook=guard, mode=worker_mode
                    )
                self._emit(
                    "worker.done", id=step.id, tools=worker.tool_calls, stop=worker.stopped_reason
                )
                # Observe what the worker left behind, before the Monitor
                # judges it and before any checkpoint or reset moves it.
                self._observe(step_id=step.id, attempt=attempts, trigger="worker")

            verdict = cores.evaluate(
                step=step,
                memory=memory,
                workspace=workspace,
                llm=self.llm,
                before=before,
            )
            # Read the change set BEFORE committing: after the checkpoint the
            # working tree is clean and there is nothing pending to measure.
            # Needed by the journal AND by diagnosis — gating it on the
            # journal alone left recovery seeing an empty diff always, which
            # silently disabled the blast-radius and empty-diff rules.
            changes = (
                self._changes_since(memory, before)
                if (journal or history is not None)
                else ((), 0)
            )
            self._write_verdict(workspace, step, verdict)
            message = f"{step.id}: {step.description} [Monitor: {'PASS' if verdict.passed else 'FAIL'}]"
            checkpoint = memory.checkpoint(step.id, message, allow_empty=True)
            self._emit(
                "monitor.verdict",
                id=step.id,
                attempt=attempts,
                passed=verdict.passed,
                reason=verdict.reason,
                sha=checkpoint.short_sha,
            )
            if journal:
                cost_now = self.llm.stats.total_cost_usd if self.llm else None
                self._write_card(
                    journal,
                    memory=memory,
                    step=step,
                    checkpoint=checkpoint,
                    attempt=attempts,
                    verdict=verdict,
                    worker=worker,
                    changes=changes,
                    cost=(cost_now - cost_before) if cost_now is not None else None,
                    elapsed=time.time() - attempt_started,
                )
                cost_before = cost_now

            if verdict.passed:
                if history is not None:
                    history.record(
                        recovery.AttemptRecord(
                            attempt=attempts,
                            fingerprint="",
                            failure_class=recovery.FailureClass.UNKNOWN,
                            action=recovery.ActionKind.ACCEPT,
                            passed=True,
                        )
                    )
                return StepOutcome(
                    step=step,
                    worker=worker,
                    verdict=verdict,
                    checkpoint=checkpoint,
                    attempts=attempts,
                    rolled_back=rolled_back,
                )

            # ---- the fault path: diagnose, decide, dispatch ----
            action = self._handle_fault(
                step=step,
                verdict=verdict,
                worker=worker,
                changes=changes,
                history=history,
                attempts=attempts,
                actions_used=actions_used,
                memory=memory,
                before=before,
            )
            actions_used += 1

            if action.kind is recovery.ActionKind.ACCEPT:
                # The monitor's FAIL stands as the step's outcome, commit kept.
                return StepOutcome(
                    step=step,
                    worker=worker,
                    verdict=verdict,
                    checkpoint=checkpoint,
                    attempts=attempts,
                    rolled_back=rolled_back,
                )

            if action.kind is recovery.ActionKind.HALT:
                # Fall through to the anchor/reset block so a resetting arm
                # discards its final failed attempt too — giving up is not a
                # licence to leave unverified work on the branch.
                self._finish_attempt(
                    journal=journal,
                    memory=memory,
                    step=step,
                    attempt=attempts,
                    checkpoint=checkpoint,
                    action=action,
                    before=before,
                )
                rolled_back = rolled_back or action.resets
                break

            if action.kind is recovery.ActionKind.REVERIFY:
                # Re-run the check only: no worker, no new edits, no API
                # spend. Costs an action, never an attempt — which is what
                # keeps the reset and no-reset arms attempt-matched.
                reverifying = True
                continue

            self._finish_attempt(
                journal=journal,
                memory=memory,
                step=step,
                attempt=attempts,
                checkpoint=checkpoint,
                action=action,
                before=before,
            )
            rolled_back = rolled_back or action.resets
            last_feedback = self._feedback_for(step, verdict, action)
            worker_mode = (
                "repair" if action.kind is recovery.ActionKind.REPAIR_IN_PLACE else "fresh"
            )

        return StepOutcome(
            step=step,
            worker=worker,
            verdict=verdict,
            checkpoint=memory.head(),
            attempts=attempts,
            rolled_back=rolled_back,
        )

    def _finish_attempt(
        self,
        *,
        journal: Journal | None,
        memory: Memory,
        step: Step,
        attempt: int,
        checkpoint: Checkpoint,
        action: recovery.Action,
        before: Checkpoint,
    ) -> None:
        """Close out a failed attempt: preserve it, then discard it if asked.

        Anchoring must precede the reset — once the branch moves, the commit
        is unreachable and the attempt becomes unreadable.
        """
        if journal:
            # The run's own session id, never derived from the branch name:
            # a worktree branch (taste-wt/…-step-02) would yield "02", so
            # anchors from different sessions would collide and a targeted
            # prune would never match.
            ref = journal.anchor(
                session=self._session_id or "unknown",
                step_id=step.id,
                attempt=attempt,
                sha=checkpoint.sha,
            )
            if ref:
                self._emit("journal.anchor", id=step.id, attempt=attempt, ref=ref)

        if action.resets:
            memory.rollback_to(before)
            self._emit(
                "step.rollback",
                id=step.id,
                to=before.short_sha,
                remaining_retries=self.max_retries - attempt + 1,
            )

    def _handle_fault(
        self,
        *,
        step: Step,
        verdict: MonitorResult,
        worker: WorkerResult,
        changes: tuple[tuple[FileChange, ...], int],
        history: recovery.StepHistory | None,
        attempts: int,
        actions_used: int,
        memory: Memory,
        before: Checkpoint,
    ) -> recovery.Action:
        """Read the fault frame, name the fault, choose the response.

        With recovery disabled this collapses to the historical behavior —
        always roll back and retry — without consulting anything.
        """
        if history is None:
            return recovery.Action(
                recovery.ActionKind.ROLLBACK_AND_RETRY,
                reason="recovery disabled",
                resets=True,
            )

        files, diff_lines = changes
        # Without threading the interrupt through, a guardrail that kills an
        # attempt for overspending produces an unclassifiable FAIL, which
        # falls to the baseline action and re-runs the very worker that was
        # just stopped — turning the budget guard into a retry trigger.
        signals = recovery.observe(
            verdict=verdict,
            worker=worker,
            changed_files=tuple(f.path for f in files),
            diff_lines=diff_lines,
            history=history,
            interrupt_kind=worker.interrupt.kind if worker.interrupt else None,
        )

        # The baseline probe: re-run the same check against the pre-step
        # commit. If it was already failing there, the step cannot be at
        # fault and retrying is pure waste. $0 of API spend, and only worth
        # paying once per step — the answer cannot change.
        if (
            self.recovery_config.baseline_probe
            and step.verification.kind == "shell"
            and history.last() is None
        ):
            probe, probe_seconds = recovery.run_baseline_probe(
                memory=memory,
                before=before,
                command=step.verification.command,
                failure_fingerprint=signals.fingerprint,
                max_seconds=self.recovery_config.probe_max_seconds,
            )
            signals = replace(signals, baseline_probe=probe, probe_seconds=probe_seconds)
            self._emit(
                "recovery.probe",
                id=step.id,
                result=probe,
                seconds=round(probe_seconds, 3),
            )
        elif self.recovery_config.baseline_probe and history.last() is not None:
            # Reuse the first probe's answer: the pre-step tree never changes.
            signals = replace(signals, baseline_probe=history.baseline_probe)

        if history.baseline_probe == "skipped":
            history.baseline_probe = signals.baseline_probe

        # Record BEFORE diagnosing, so no_progress_streak counts the attempt
        # being diagnosed. Reading the streak first meant "two identical
        # failures running" needed a third attempt to notice the second.
        history.record(
            recovery.AttemptRecord(
                attempt=attempts,
                fingerprint=signals.fingerprint,
                failure_class=recovery.FailureClass.UNKNOWN,
                action=recovery.ActionKind.HALT,
                failed_count=signals.failed_count,
            )
        )
        diagnosis = recovery.diagnose(signals, history)
        self._emit(
            "recovery.diagnosis",
            id=step.id,
            attempt=attempts,
            failure_class=diagnosis.failure_class.value,
            rule=diagnosis.top.rule_id,
            confidence=diagnosis.confidence,
        )

        budget = recovery.Budget(
            attempts_used=attempts,
            max_attempts=self.max_retries + 1,
            actions_used=actions_used,
            max_actions=self.recovery_config.max_actions,
        )
        action = self._policy.decide(
            diagnosis=diagnosis,
            history=history,
            budget=budget,
            config=self.recovery_config,
            signals=signals,
        )
        history.book.add(signals.fingerprint, recovery.guidance_bullet(signals, diagnosis))
        # Fill in what the placeholder record could not know until now.
        history.amend_last(failure_class=diagnosis.failure_class, action=action.kind)
        self._emit(
            "recovery.action",
            id=step.id,
            attempt=attempts,
            action=action.kind.value,
            resets=action.resets,
            reason=action.reason,
        )
        return action

    def _feedback_for(
        self, step: Step, verdict: MonitorResult, action: recovery.Action
    ) -> str:
        """What the next attempt is told. Guidance when the policy supplied
        it, the raw verdict otherwise."""
        if action.guidance:
            return action.guidance
        return _build_feedback(step, verdict)

    # ------------------------------------------------------------ shadow

    def _observe(self, *, step_id: str, attempt: int, trigger: str, tool: str | None = None) -> None:
        """Record an observation point. Never fatal, never agent-visible."""
        if self._shadow is None:
            return
        commit = self._shadow.observe(
            step_id=step_id, attempt=attempt, trigger=trigger, tool=tool
        )
        if commit is not None:
            self._emit(
                "shadow.observe",
                id=step_id,
                seq=commit.seq,
                sha=commit.sha[:7],
                trigger=trigger,
                # Carried on the event as well as the shadow record. Under the
                # per-tool grid the event stream is what the console renders,
                # and without this every tool observation reads as anonymous
                # there while the record on disk knows exactly which call
                # produced it.
                tool=commit.tool,
                files=len(commit.files),
            )

    # ------------------------------------------------------------ journal

    def _journal_for(self, memory: Memory) -> Journal | None:
        """A Journal bound to this step's memory, or None when disabled.

        Bound per step because a parallel step's memory is its own worktree,
        and a card must be written against the repo that owns the commit.
        """
        if not self.journal_enabled:
            return None
        return Journal(memory, gitdir=self._runtime_dir, enabled=True)

    def _changes_since(self, memory: Memory, before: Checkpoint) -> tuple[tuple[FileChange, ...], int]:
        """File-level change summary for the pending work. Never fatal.

        Called before ``_write_verdict``, so the card describes what the
        *worker* changed, not the harness's own bookkeeping file.
        """
        try:
            return parse_numstat(memory.numstat_pending(before.sha))
        except Exception:
            return ((), 0)

    def _write_card(
        self,
        journal: Journal,
        *,
        memory: Memory,
        step: Step,
        checkpoint: Checkpoint,
        attempt: int,
        verdict: MonitorResult,
        worker: WorkerResult,
        changes: tuple[tuple[FileChange, ...], int],
        cost: float | None,
        elapsed: float,
    ) -> None:
        files, diff_lines = changes
        card = card_from_step(
            session=memory.branch,
            branch=memory.branch,
            sha=checkpoint.sha,
            parent_sha=checkpoint.parent_sha,
            step=step,
            attempt=attempt,
            verdict_passed=verdict.passed,
            verdict_reason=verdict.reason,
            files=files,
            diff_lines=diff_lines,
            worker=worker,
            cost_usd=cost,
            elapsed_s=round(elapsed, 3),
        )
        if journal.card(card):
            self._emit(
                "journal.card",
                id=step.id,
                attempt=attempt,
                sha=checkpoint.short_sha,
                files=len(files),
            )

    def _validate_models(self, spec: AgentSpec) -> None:
        """Fail fast (PricingError) if any role's model lacks a pricing entry."""
        for model in {
            self.planner_model or MODEL_PLANNER,
            spec.model or MODEL_WORKER,
            MODEL_MONITOR,
        }:
            ensure_priced(model)

    def _plan(self, task: str, spec: AgentSpec, memory: Memory) -> Plan:
        if self.llm is None:
            raise RuntimeError("LLM not configured; pass plan_override or construct with llm=...")
        summary = _summarize_workspace(self.workspace)
        return cores.plan(self.llm, task, spec, summary, model=self.planner_model)

    def _guard_for(self, workspace: Path) -> Guardrails | None:
        """A guard bound to this step's workspace, or None when disabled."""
        if not self.guard_config.enabled:
            return None
        return Guardrails(
            workspace=workspace,
            config=self.guard_config,
            cost_reader=(lambda: self.llm.stats.total_cost_usd) if self.llm else None,
            on_event=lambda kind, **payload: self._emit(kind, **payload),
        )

    def _hook_for(
        self, workspace: Path, step: Step, attempt_of: Callable[[], int]
    ) -> cores.TurnHook | None:
        """The worker's callback surface: guard, observer, both, or neither.

        The observer must not be attached to the guard. Guardrails are an
        *arm-defining* subsystem that some arms switch off, so hanging the
        observation grid off it would make the grid follow the treatment —
        precisely the confound a finer grid exists to remove.

        ``attempt_of`` is read at call time rather than captured: one hook is
        built per step and lives across every attempt, so a bound integer
        would stamp every observation with the first attempt's number.
        """
        guard = self._guard_for(workspace)
        if not (self.observe_tools and self._shadow is not None):
            return guard
        return _ObservingHook(
            inner=guard,
            observe=lambda tool: self._observe(
                step_id=step.id, attempt=attempt_of(), trigger="tool", tool=tool
            ),
        )

    def _worker(
        self,
        step: Step,
        plan: Plan,
        spec: AgentSpec,
        tools: ToolRegistry,
        last_feedback: str | None,
        *,
        hook: Guardrails | None = None,
        mode: Literal["fresh", "repair"] = "fresh",
    ) -> WorkerResult:
        if self.llm is None:
            raise RuntimeError("LLM not configured; pass worker_override or construct with llm=...")
        return cores.execute(
            self.llm,
            spec=spec,
            step=step,
            plan_context=plan.to_summary(),
            tools=tools,
            hook=hook,
            guidance=last_feedback,
            mode=mode,
        )

    # ------------------------------------------------------------ artifacts

    def _persist_plan(self, memory: Memory, plan: Plan) -> None:
        out = self.workspace / ".taste" / "plan.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "task": plan.task,
                    "steps": [
                        {
                            "id": s.id,
                            "description": s.description,
                            "verification": {
                                "kind": s.verification.kind,
                                "command": s.verification.command,
                                "criteria": s.verification.criteria,
                            },
                            # Without this the committed plan cannot reproduce
                            # its own wave structure — the DAG would be lost.
                            "depends_on": list(s.depends_on),
                        }
                        for s in plan.steps
                    ],
                },
                indent=2,
            )
        )
        memory.checkpoint("plan", "plan: commit decomposition", allow_empty=True)

    def _write_verdict(self, workspace: Path, step: Step, verdict: MonitorResult) -> None:
        # Explicit field list, never **asdict: this file is COMMITTED, so a new
        # MonitorResult field would silently change the content of every
        # checkpoint — and with it the diff the Monitor itself grades.
        path = Path(workspace) / ".taste" / "monitor" / f"{step.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "step_id": step.id,
                    "passed": verdict.passed,
                    "reason": verdict.reason,
                    "evidence": verdict.evidence,
                },
                indent=2,
            )
        )

    def _halted_result(
        self,
        *,
        task: str,
        session_id: str,
        memory: Memory,
        plan: Plan,
        outcomes: list[StepOutcome],
        started: float,
        reason: str,
        failure_kind: Literal["task", "infra", "budget"] = "task",
    ) -> RunResult:
        head = memory.head()
        result = RunResult(
            task=task,
            session_id=session_id,
            branch=memory.branch,
            status="failed",
            plan=plan,
            outcomes=outcomes,
            final_sha=head.sha,
            elapsed_seconds=round(time.time() - started, 2),
            failure_reason=reason,
            stats=self.llm.stats if self.llm else None,
            failure_kind=failure_kind,
        )
        self._emit_run_done(result)
        return result

    def _emit_run_done(self, result: RunResult) -> None:
        """One shape for run.done, whatever path reached it.

        Two emit sites used to carry different keys (cost on success, reason
        on halt), so any consumer parsing the event had to branch on how the
        run ended. A single key set keeps the stream mechanically diffable —
        which is what makes ablation-equivalence testable.
        """
        self._emit(
            "run.done",
            status=result.status,
            elapsed=result.elapsed_seconds,
            reason=result.failure_reason,
            failure_kind=result.failure_kind,
            cost_usd=round(result.stats.total_cost_usd, 4) if result.stats else 0.0,
            cache_hit_rate=round(result.stats.cache_hit_rate, 3) if result.stats else 0.0,
        )

    def _write_manifest(
        self,
        *,
        task: str,
        session_id: str,
        spec: AgentSpec,
        memory: Memory,
    ) -> None:
        """Persist run provenance next to the event log (survives rollback).

        Records everything needed to attribute a result months later: harness
        commit, package version, per-role model assignments, sampling params,
        and prompt hashes. Reproducibility on a seedless API means controlling
        and *recording* every controllable knob.
        """
        from taste import __version__

        manifest = {
            "session": session_id,
            "task": task,
            "agent": spec.name,
            "agent_spec_path": str(spec.source_path) if spec.source_path else None,
            "branch": memory.branch,
            "created_at": time.time(),
            "harness_git_sha": _harness_git_sha(),
            "taste_version": __version__,
            "models": {
                "planner": self.planner_model or MODEL_PLANNER,
                "worker": spec.model or MODEL_WORKER,
                "monitor": MODEL_MONITOR,
            },
            # Per-role: the Monitor deliberately judges at its own temperature.
            "temperature": {
                "planner": DEFAULT_TEMPERATURE,
                "worker": DEFAULT_TEMPERATURE,
                "monitor": cores.MONITOR_TEMPERATURE,
            },
            "max_retries": self.max_retries,
            "max_parallel": self.max_parallel,
            # The arm's identity. A run whose harness cannot be recovered from
            # its own manifest is a datapoint that cannot be grouped later.
            "harness": (
                self.config.to_manifest()
                if self.config is not None
                else {
                    "label": "adhoc",
                    "config_hash": None,
                    "journal": self.journal_enabled,
                    "recovery": {
                        "enabled": self.recovery_config.enabled,
                        "policy": self.recovery_config.policy,
                        "fixed_action": self.recovery_config.fixed_action.value,
                        "baseline_probe": self.recovery_config.baseline_probe,
                    },
                    "guardrails": {"enabled": self.guard_config.enabled},
                    "two_phase_merge": self.two_phase_merge,
                }
            ),
            "prompt_sha": {
                "planner_system": prompt_sha(cores.PLANNER_SYSTEM),
                "worker_system": prompt_sha(cores.WORKER_SYSTEM),
                "monitor_system": prompt_sha(cores.MONITOR_SYSTEM),
                "agent_system_prompt": prompt_sha(spec.system_prompt),
            },
        }
        # Keyed by session: a workspace accumulates one manifest per run, so
        # provenance for older sessions survives later runs (their branches do).
        path = self._runtime_dir / f"manifest-{session_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2))
        self._emit(
            "run.manifest",
            harness_git_sha=manifest["harness_git_sha"],
            models=manifest["models"],
            temperature=manifest["temperature"],
        )

    # ------------------------------------------------------------ event sink

    def _open_event_log(self) -> None:
        """Truncate and prepare the per-session JSONL event stream.

        The event log lives under ``.git/taste/`` so it survives ``git reset``:
        the committed artifacts (plan.json, monitor/*.json) are the audit
        trail; the event stream is runtime-only trace and must NOT roll back
        with the working tree.
        """
        path = self._runtime_dir / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        self._events_path = path

    def _emit(self, kind: str, /, **payload: object) -> None:
        # Positional-only so a payload key named "kind" cannot collide with
        # this parameter — a TypeError that would surface far from its cause.
        event = Event(kind=kind, payload=payload)
        # Serialize file writes + user callback so parallel workers don't
        # interleave partial JSON lines.
        with self._lock:
            if self._events_path is not None:
                with self._events_path.open("a") as f:
                    f.write(json.dumps(event.to_json(), default=str) + "\n")
            self.on_event(event)


# ================================================================= helpers


@functools.lru_cache(maxsize=1)
def _harness_git_sha() -> str:
    """Commit SHA of the taste package's own checkout, or 'unknown'.

    Uses GitPython WITHOUT parent-directory search on purpose: for a
    pip-installed package, walking upward from site-packages would find
    whatever unrelated repo encloses the venv and confidently record the
    wrong SHA — worse than 'unknown'. Only an exact editable-checkout root
    counts. Cached: the SHA cannot change within a process.
    """
    import taste

    pkg_root = Path(taste.__file__).resolve().parent.parent
    try:
        from git import Repo

        return Repo(pkg_root).head.commit.hexsha
    except Exception:
        return "unknown"


def _summarize_workspace(workspace: Path) -> str:
    """Cheap, deterministic summary for the Planner.

    Lists every tracked file; includes the first 80 lines of each small Python
    file (< 10 KB) so the Planner can see the shapes it's decomposing against
    without spending a tool-use turn on `read_file`. Non-Python files and large
    files appear only by name — demand-page them with `read_file` if needed.
    """
    from subprocess import run

    proc = run(
        ["git", "ls-files"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    files = proc.stdout.strip().splitlines()
    if not files:
        return "(empty workspace)"

    parts = ["Files:\n  " + "\n  ".join(files[:60])]
    for rel in files[:12]:
        p = workspace / rel
        try:
            if p.suffix == ".py" and p.stat().st_size < 10_000:
                head = "\n".join(p.read_text().splitlines()[:80])
                parts.append(f"\n--- {rel} ---\n{head}")
        except OSError:
            continue
    return "\n".join(parts)


def _build_feedback(step: Step, verdict: MonitorResult) -> str:
    lines = [f"Monitor failed step {step.id}: {verdict.reason}"]
    if verdict.evidence:
        lines.append("Evidence:\n" + verdict.evidence)
    return "\n".join(lines)
