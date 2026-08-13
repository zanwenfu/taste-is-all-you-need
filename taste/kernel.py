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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from taste import cores
from taste.agent import AgentSpec
from taste.cores import MonitorResult, Plan, PlannerError, Step, WorkerResult
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
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.llm = llm
        self.max_retries = max_retries
        self.on_event = on_event or (lambda _e: None)
        self.max_parallel = max_parallel
        # Planner model is a kernel knob, NOT spec.model: the agent spec's
        # model field configures the Worker only (role-split invariant).
        self.planner_model = planner_model
        self._events_path: Path | None = None
        self._runtime_dir_cache: Path | None = None
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
        self._emit(
            "run.done",
            status=status,
            elapsed=result.elapsed_seconds,
            cost_usd=round(result.stats.total_cost_usd, 4) if result.stats else 0.0,
            cache_hit_rate=round(result.stats.cache_hit_rate, 3) if result.stats else 0.0,
        )
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

        while attempts <= self.max_retries:
            attempts += 1
            self._emit("step.begin", id=step.id, attempt=attempts)

            if worker_override is not None:
                worker = worker_override(step, plan)
            else:
                worker = self._worker(step, plan, spec, tools, last_feedback)
            self._emit("worker.done", id=step.id, tools=worker.tool_calls, stop=worker.stopped_reason)

            verdict = cores.evaluate(
                step=step,
                memory=memory,
                workspace=workspace,
                llm=self.llm,
            )
            self._write_verdict(workspace, step, verdict)
            message = f"{step.id}: {step.description} [Monitor: {'PASS' if verdict.passed else 'FAIL'}]"
            checkpoint = memory.checkpoint(step.id, message, allow_empty=True)
            self._emit(
                "monitor.verdict",
                id=step.id,
                passed=verdict.passed,
                reason=verdict.reason,
                sha=checkpoint.short_sha,
            )

            if verdict.passed:
                return StepOutcome(
                    step=step,
                    worker=worker,
                    verdict=verdict,
                    checkpoint=checkpoint,
                    attempts=attempts,
                    rolled_back=rolled_back,
                )

            memory.rollback_to(before)
            rolled_back = True
            last_feedback = _build_feedback(step, verdict)
            self._emit(
                "step.rollback",
                id=step.id,
                to=before.short_sha,
                remaining_retries=self.max_retries - attempts + 1,
            )

        return StepOutcome(
            step=step,
            worker=worker,
            verdict=verdict,
            checkpoint=memory.head(),
            attempts=attempts,
            rolled_back=rolled_back,
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

    def _worker(
        self,
        step: Step,
        plan: Plan,
        spec: AgentSpec,
        tools: ToolRegistry,
        last_feedback: str | None,
    ) -> WorkerResult:
        if self.llm is None:
            raise RuntimeError("LLM not configured; pass worker_override or construct with llm=...")
        context = plan.to_summary()
        if last_feedback:
            context = f"{context}\n\nLast attempt failed: {last_feedback}\nAddress the failure on this retry."
        return cores.execute(
            self.llm,
            spec=spec,
            step=step,
            plan_context=context,
            tools=tools,
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
                        }
                        for s in plan.steps
                    ],
                },
                indent=2,
            )
        )
        memory.checkpoint("plan", "plan: commit decomposition", allow_empty=True)

    def _write_verdict(self, workspace: Path, step: Step, verdict: MonitorResult) -> None:
        path = Path(workspace) / ".taste" / "monitor" / f"{step.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"step_id": step.id, **asdict(verdict)}, indent=2)
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
        self._emit("run.done", status="failed", elapsed=result.elapsed_seconds, reason=reason)
        return result

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

    def _emit(self, kind: str, **payload: object) -> None:
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
