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

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from taste import cores
from taste.agent import AgentSpec
from taste.cores import MonitorResult, Plan, Step, WorkerResult
from taste.llm import LLM
from taste.memory import Checkpoint, Memory
from taste.tools import ToolRegistry, make_builtin_tools

# ================================================================= events


@dataclass
class Event:
    kind: str
    payload: dict = field(default_factory=dict)

    def __str__(self) -> str:
        body = " ".join(f"{k}={v}" for k, v in self.payload.items() if k != "evidence")
        return f"[{self.kind}] {body}"


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

    def summary(self) -> str:
        passed = sum(1 for o in self.outcomes if o.verdict.passed)
        total = len(self.plan.steps)
        rolled = sum(1 for o in self.outcomes if o.rolled_back)
        return (
            f"status={self.status} steps={passed}/{total} "
            f"rollbacks={rolled} branch={self.branch} head={self.final_sha[:7]}"
        )


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
        for rich printing and by tests for assertions.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        llm: LLM | None = None,
        max_retries: int = 2,
        on_event: Callable[[Event], None] | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.llm = llm
        self.max_retries = max_retries
        self.on_event = on_event or (lambda _e: None)
        self.tools = ToolRegistry()
        self.tools.extend(make_builtin_tools(self.workspace))

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
        self._emit("run.start", task=task, session=session_id, branch=memory.branch)

        plan = plan_override or self._plan(task, spec, memory)
        self._persist_plan(memory, plan)
        self._emit("plan.ready", steps=len(plan.steps))

        outcomes: list[StepOutcome] = []
        status: Literal["completed", "failed"] = "completed"

        for step in plan.steps:
            outcome = self._run_step(
                step=step,
                plan=plan,
                spec=spec,
                memory=memory,
                worker_override=worker_override,
            )
            outcomes.append(outcome)
            if not outcome.verdict.passed:
                status = "failed"
                self._emit(
                    "run.halt",
                    step=step.id,
                    reason=outcome.verdict.reason,
                )
                break

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
        )
        self._emit("run.done", status=status, elapsed=result.elapsed_seconds)
        return result

    # ------------------------------------------------------------ internals

    def _run_step(
        self,
        *,
        step: Step,
        plan: Plan,
        spec: AgentSpec,
        memory: Memory,
        worker_override: Callable[[Step, Plan], WorkerResult] | None,
    ) -> StepOutcome:
        before = memory.head()
        attempts = 0
        rolled_back = False
        last_feedback: str | None = None

        while attempts <= self.max_retries:
            attempts += 1
            self._emit("step.begin", id=step.id, attempt=attempts)

            if worker_override is not None:
                worker = worker_override(step, plan)
            else:
                worker = self._worker(step, plan, spec, last_feedback)
            self._emit("worker.done", tools=worker.tool_calls, stop=worker.stopped_reason)

            verdict = cores.evaluate(
                step=step,
                memory=memory,
                workspace=self.workspace,
                llm=self.llm,
            )
            self._write_verdict(step, verdict)
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

            # Failure: roll back and (maybe) retry.
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

    def _plan(self, task: str, spec: AgentSpec, memory: Memory) -> Plan:
        if self.llm is None:
            raise RuntimeError("LLM not configured; pass plan_override or construct with llm=...")
        summary = _summarize_workspace(self.workspace)
        return cores.plan(self.llm, task, spec, summary)

    def _worker(
        self,
        step: Step,
        plan: Plan,
        spec: AgentSpec,
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
            tools=self.tools,
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

    def _write_verdict(self, step: Step, verdict: MonitorResult) -> None:
        path = self.workspace / ".taste" / "monitor" / f"{step.id}.json"
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

    def _emit(self, kind: str, **payload: object) -> None:
        self.on_event(Event(kind=kind, payload=payload))


# ================================================================= helpers


def _summarize_workspace(workspace: Path) -> str:
    """Cheap, deterministic summary: ls-files output, truncated."""
    from subprocess import run

    proc = run(
        ["git", "ls-files"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    files = proc.stdout.strip().splitlines()[:60]
    return "files:\n  " + "\n  ".join(files) if files else "(empty workspace)"


def _build_feedback(step: Step, verdict: MonitorResult) -> str:
    lines = [f"Monitor failed step {step.id}: {verdict.reason}"]
    if verdict.evidence:
        lines.append("Evidence:\n" + verdict.evidence)
    return "\n".join(lines)
