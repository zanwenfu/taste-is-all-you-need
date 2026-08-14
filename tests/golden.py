"""Run signatures — the instrument that makes "build to delete" testable.

Every subsystem added to the kernel must be switchable OFF and, when off,
must leave behavior *exactly* as it was. Asserting that by eye does not
scale. :func:`run_signature` reduces a run to a stable fingerprint —
event-kind sequence, payload key sets, commit-chain shape, outcome fields —
with the volatile parts (SHAs, timings, dollar amounts, absolute paths)
deliberately excluded.

A new subsystem's OFF path is then one assertion:

    assert run_signature(...) == BASELINE_SIGNATURE

If a disabled subsystem emits one extra event or widens one payload, the
diff says exactly which.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from taste.agent import AgentSpec
from taste.cores import Plan, Step, Verification, WorkerResult
from taste.kernel import Event, Kernel, RunResult
from taste.memory import Memory


@dataclass(frozen=True)
class RunSignature:
    """A stable fingerprint of one kernel run."""

    events: tuple[tuple[str, tuple[str, ...]], ...]
    """(event kind, sorted payload keys) in order. Values excluded — they
    carry SHAs and timings that differ between identical runs."""

    commits: tuple[tuple[str, str], ...]
    """(checkpoint step_id, commit subject) from oldest to newest.

    Commits with no ``Taste-Checkpoint`` trailer — the repo's own pre-run
    history — have no step id, and :class:`Checkpoint` falls back to the raw
    short SHA. A SHA depends on the commit timestamp, so two identical runs
    bootstrapped a second apart would fingerprint differently. Those ids are
    normalized to ``(untrailed)``: the instrument must be sensitive to
    harness behavior and to nothing else.
    """

    status: str
    failure_kind: str | None
    step_results: tuple[tuple[str, bool, int, bool], ...]
    """(step id, passed, attempts, rolled_back) per outcome."""

    def diff(self, other: RunSignature) -> str:
        """Human-readable first divergence, for assertion messages."""
        lines: list[str] = []
        if self.events != other.events:
            for i, (a, b) in enumerate(zip(self.events, other.events, strict=False)):
                if a != b:
                    lines.append(f"events[{i}]: {a!r} != {b!r}")
                    break
            else:
                lines.append(
                    f"event count: {len(self.events)} != {len(other.events)}; "
                    f"extra={set(self.events) ^ set(other.events)}"
                )
        for field_name in ("commits", "status", "failure_kind", "step_results"):
            mine, theirs = getattr(self, field_name), getattr(other, field_name)
            if mine != theirs:
                lines.append(f"{field_name}: {mine!r} != {theirs!r}")
        return "\n".join(lines) or "(identical)"


def _stable_step_id(checkpoint) -> str:
    """A step id that does not vary with commit timestamps."""
    return "(untrailed)" if checkpoint.step_id == checkpoint.sha[:7] else checkpoint.step_id


def signature_of(events: list[Event], result: RunResult, memory: Memory) -> RunSignature:
    return RunSignature(
        events=tuple((e.kind, tuple(sorted(e.payload))) for e in events),
        commits=tuple(
            (_stable_step_id(c), c.message.splitlines()[0] if c.message else "")
            for c in reversed(memory.log())
        ),
        status=result.status,
        failure_kind=result.failure_kind,
        step_results=tuple(
            (o.step.id, o.verdict.passed, o.attempts, o.rolled_back) for o in result.outcomes
        ),
    )


# ------------------------------------------------------------------ scenario


@dataclass
class Scenario:
    """A deterministic kernel run: no model, no network, no clock dependence."""

    name: str
    plan: Plan
    worker: object
    max_retries: int = 2
    events: list[Event] = field(default_factory=list)

    def run(self, workspace: Path, **kernel_kwargs) -> RunSignature:
        self.events.clear()
        kernel = Kernel(
            workspace=workspace,
            max_retries=self.max_retries,
            on_event=self.events.append,
            **kernel_kwargs,
        )
        result = kernel.run(
            task=self.name,
            spec=AgentSpec(name="scripted", description="", system_prompt="p"),
            session_id=self.name,
            plan_override=self.plan,
            worker_override=self.worker,
        )
        memory = Memory(workspace, f"taste/session-{self.name}")
        return signature_of(self.events, result, memory)


def rollback_scenario(workspace: Path, name: str = "golden") -> Scenario:
    """Two steps; the first fails once, is rolled back, then succeeds.

    Chosen because it exercises every branch the recovery subsystem will
    later intercept: a pass, a fail, a rollback, a retry with feedback, and
    a clean completion.
    """
    plan = Plan(
        task="golden",
        steps=[
            Step(
                id="step-01",
                description="create the module",
                verification=Verification(kind="shell", command="test -f made.py"),
            ),
            Step(
                id="step-02",
                description="create the second module",
                verification=Verification(kind="shell", command="test -f also.py"),
            ),
        ],
    )

    state = {"attempts": 0}

    def worker(step: Step, plan_: Plan) -> WorkerResult:
        if step.id == "step-01":
            state["attempts"] += 1
            if state["attempts"] == 1:
                # First attempt does the wrong thing: verification will fail.
                (workspace / "wrong.py").write_text("# not what was asked\n")
                return WorkerResult("wrote the wrong file", 1, "end_turn")
            (workspace / "made.py").write_text("# correct\n")
            return WorkerResult("wrote the right file", 1, "end_turn")
        (workspace / "also.py").write_text("# second\n")
        return WorkerResult("wrote the second file", 1, "end_turn")

    return Scenario(name=name, plan=plan, worker=worker)
