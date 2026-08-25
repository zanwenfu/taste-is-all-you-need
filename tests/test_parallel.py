"""Parallel execution — worktrees, waves, merge orchestration.

Exercises the Milestone B primitives without an LLM: scripted workers run
concurrently on separate worktrees, the Kernel merges their branches back
into the session.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from taste.agent import AgentSpec
from taste.cores import Plan, Step, Verification, WorkerResult
from taste.kernel import Event, Kernel, current_step
from tests.conftest import PYTEST_CMD

# -------------------------------------------------------------- fixtures


def _parallel_plan() -> Plan:
    """Wave-1 (shared bootstrap) → Wave-2 (3 parallel per-module steps)."""
    ck = Verification(kind="shell", command=PYTEST_CMD)
    return Plan(
        task="add type hints to three independent modules",
        steps=[
            Step(id="step-01", description="smoke: tests pass unchanged", verification=ck, depends_on=[]),
            Step(id="step-02", description="type-hint math_utils", verification=ck, depends_on=["step-01"]),
            Step(id="step-03", description="type-hint string_utils", verification=ck, depends_on=["step-01"]),
            Step(id="step-04", description="type-hint list_utils", verification=ck, depends_on=["step-01"]),
        ],
    )


def _module_for(step_id: str) -> str:
    return {"step-02": "math_utils.py", "step-03": "string_utils.py", "step-04": "list_utils.py"}[step_id]


def _type_hint_module(path: Path) -> None:
    """Deterministic worker edit — add trivial type hints, keep tests green."""
    name = path.name
    if name == "math_utils.py":
        path.write_text("def add(a: int, b: int) -> int:\n    return a + b\n")
    elif name == "string_utils.py":
        path.write_text("def upper(s: str) -> str:\n    return s.upper()\n")
    elif name == "list_utils.py":
        path.write_text("def head(xs: list) -> object:\n    return xs[0] if xs else None\n")


# -------------------------------------------------------------- Plan.waves()


def test_waves_respect_dag() -> None:
    plan = _parallel_plan()
    waves = plan.waves()
    assert len(waves) == 2
    assert [s.id for s in waves[0]] == ["step-01"]
    assert sorted(s.id for s in waves[1]) == ["step-02", "step-03", "step-04"]


def test_waves_detects_cycle() -> None:
    ck = Verification(kind="shell", command="true")
    plan = Plan(
        task="cycle",
        steps=[
            Step(id="step-01", description="", verification=ck, depends_on=["step-02"]),
            Step(id="step-02", description="", verification=ck, depends_on=["step-01"]),
        ],
    )
    with pytest.raises(ValueError, match="cycle"):
        plan.waves()


def test_waves_linear_when_no_deps() -> None:
    """No explicit deps → linear chain (back-compat with Milestone A plans)."""
    ck = Verification(kind="shell", command="true")
    plan = Plan(
        task="linear",
        steps=[Step(id=f"step-0{i}", description="", verification=ck) for i in range(1, 4)],
    )
    waves = plan.waves()
    assert [[s.id for s in w] for w in waves] == [["step-01"], ["step-02"], ["step-03"]]


# -------------------------------------------------------------- parallel exec


def test_parallel_wave_runs_on_worktrees_and_merges_back(parallel_workspace: Path) -> None:
    ws = parallel_workspace
    start_latches: dict[str, threading.Event] = {sid: threading.Event() for sid in ["step-02", "step-03", "step-04"]}
    witnesses: dict[str, Path] = {}

    def worker(step: Step, plan: Plan) -> WorkerResult:
        if step.id == "step-01":
            (ws / "SMOKE.md").write_text("ok\n")
            return WorkerResult(summary="smoke", tool_calls=0, stopped_reason="end_turn")
        # Parallel steps: worker edits a module inside its own worktree. The
        # kernel publishes which one via CURRENT_STEP.
        ctx = current_step()
        assert ctx.step.id == step.id
        wt_path = ctx.workspace

        witnesses[step.id] = wt_path
        start_latches[step.id].set()
        # Ensure all three parallel steps are actually running concurrently
        for sid, latch in start_latches.items():
            if sid != step.id:
                assert latch.wait(timeout=5), f"{sid} didn't start concurrently"
        _type_hint_module(wt_path / _module_for(step.id))
        return WorkerResult(summary=f"hinted {step.id}", tool_calls=0, stopped_reason="end_turn")

    events: list[Event] = []
    kernel = Kernel(workspace=ws, max_retries=0, on_event=events.append)
    result = kernel.run(
        task="parallel",
        spec=AgentSpec(name="scripted_parallel", description=""),
        session_id="par",
        plan_override=_parallel_plan(),
        worker_override=worker,
    )

    assert result.status == "completed", result.failure_reason
    # All three parallel workers ran on distinct filesystem paths.
    assert len(set(witnesses.values())) == 3
    # None of them was the main workspace.
    assert ws not in witnesses.values()

    # Session branch has all three module changes merged in.
    from taste.memory import Memory
    main = Memory(ws, "taste/session-par")
    assert "int" in main.show("HEAD", "math_utils.py")
    assert "str" in main.show("HEAD", "string_utils.py")
    assert "list" in main.show("HEAD", "list_utils.py")

    # Event stream tells the parallel story.
    kinds = [e.kind for e in events]
    assert kinds.count("wave.begin") == 1       # one parallel wave
    assert kinds.count("worktree.open") == 3
    assert kinds.count("worktree.merge") == 3
    assert kinds.count("wave.done") == 1


def test_parallel_wave_halt_when_any_worker_fails(parallel_workspace: Path) -> None:
    """If one parallel worker fails, nothing merges — session branch stays clean."""
    ws = parallel_workspace

    def worker(step: Step, plan: Plan) -> WorkerResult:
        if step.id == "step-01":
            (ws / "SMOKE.md").write_text("ok\n")
        elif step.id == "step-03":
            # Corrupt string_utils so pytest fails on this worker's worktree.
            (current_step().workspace / "string_utils.py").write_text("def broken(\n")
        else:
            _type_hint_module(current_step().workspace / _module_for(step.id))
        return WorkerResult(summary="", tool_calls=0, stopped_reason="end_turn")

    events: list[Event] = []
    result = Kernel(workspace=ws, max_retries=0, on_event=events.append).run(
        task="parallel-fail",
        spec=AgentSpec(name="s", description=""),
        session_id="parfail",
        plan_override=_parallel_plan(),
        worker_override=worker,
    )

    assert result.status == "failed"
    assert any(e.kind == "wave.halt" for e in events)

    # Session branch must NOT contain any of the parallel steps' changes,
    # because the halt happened before the merges.
    from taste.memory import Memory
    main = Memory(ws, "taste/session-parfail")
    # String utils unchanged on session (worker-03's corruption never merged)
    assert "broken" not in main.show("HEAD", "string_utils.py")
    # math / list hints were made by passing workers — but they also weren't
    # merged because the wave halts atomically.
    assert "int" not in main.show("HEAD", "math_utils.py")
    assert "list" not in main.show("HEAD", "list_utils.py")


def test_session_events_record_every_parallel_step(parallel_workspace: Path) -> None:
    """The JSONL event log survives parallel execution and captures each step."""
    ws = parallel_workspace

    def worker(step: Step, plan: Plan) -> WorkerResult:
        if step.id == "step-01":
            (ws / "SMOKE.md").write_text("ok\n")
            return WorkerResult(summary="", tool_calls=0, stopped_reason="end_turn")
        _type_hint_module(current_step().workspace / _module_for(step.id))
        return WorkerResult(summary="", tool_calls=0, stopped_reason="end_turn")

    Kernel(workspace=ws, max_retries=0).run(
        task="parallel-events",
        spec=AgentSpec(name="s", description=""),
        session_id="parevents",
        plan_override=_parallel_plan(),
        worker_override=worker,
    )

    log = ws / ".git" / "taste" / "events.jsonl"
    lines = [json.loads(line) for line in log.read_text().splitlines() if line]
    verdicts = {e["payload"]["id"]: e["payload"]["passed"] for e in lines if e["kind"] == "monitor.verdict"}
    assert verdicts == {"step-01": True, "step-02": True, "step-03": True, "step-04": True}
