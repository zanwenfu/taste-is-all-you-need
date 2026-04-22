"""End-to-end rollback tests — the ``step-87 problem`` in miniature.

These tests use the Kernel's ``plan_override`` and ``worker_override`` hooks
so we can reproduce the exact failure mode that breaks agent runs in
production — a mid-run regression — without spending API credits.

If this file goes green in CI, the thesis of the repo is empirically true:
the harness catches a mid-run regression, rolls back, and recovers.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from taste.agent import AgentSpec
from taste.cores import Plan, Step, Verification, WorkerResult
from taste.kernel import Event, Kernel

LEGACY_MATH = "legacy_math.py"

BROKEN_STEP2 = """\
def run(items):
    total = 0
    for x in items:
        if x > 0:
            total += x  # REGRESSION: should be x * 2
        else:
            total -= x
    return total


def fmt(total):
    return f"total is {total}"


def main(items):
    return fmt(run(items))
"""

CORRECT_STEP2 = """\
def run(items: list[int]) -> int:
    total = 0
    for x in items:
        if x > 0:
            total += x * 2
        else:
            total -= x
    return total


def fmt(total: int) -> str:
    return f"total is {total}"


def main(items: list[int]) -> str:
    return fmt(run(items))
"""


def _spec() -> AgentSpec:
    return AgentSpec(
        name="scripted_refactor",
        description="deterministic harness test — no LLM calls",
        model=None,
        system_prompt="",
    )


def _plan() -> Plan:
    check = Verification(kind="shell", command="pytest -q")
    return Plan(
        task="refactor legacy_math.py while keeping tests green",
        steps=[
            Step(id="step-01", description="append module marker", verification=check),
            Step(id="step-02", description="add type hints to run/fmt/main", verification=check),
            Step(id="step-03", description="append trailing newline", verification=check),
        ],
    )


# ================================================================= the story


def test_monitor_catches_step2_regression_and_recovers(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    math = ws / LEGACY_MATH
    attempts: dict[str, int] = {}

    def worker(step: Step, plan: Plan) -> WorkerResult:
        attempts[step.id] = attempts.get(step.id, 0) + 1
        n = attempts[step.id]

        if step.id == "step-01":
            math.write_text(math.read_text() + "\n# refactor: legacy_math\n")
        elif step.id == "step-02":
            math.write_text(BROKEN_STEP2 if n == 1 else CORRECT_STEP2)
        elif step.id == "step-03":
            math.write_text(math.read_text().rstrip() + "\n")

        return WorkerResult(summary=f"{step.id} attempt {n}", tool_calls=0, stopped_reason="end_turn")

    events: list[Event] = []
    kernel = Kernel(workspace=ws, max_retries=2, on_event=events.append)
    result = kernel.run(task=_plan().task, spec=_spec(), plan_override=_plan(), worker_override=worker)

    # Status + attempt bookkeeping.
    assert result.status == "completed"
    ids = [o.step.id for o in result.outcomes]
    assert ids == ["step-01", "step-02", "step-03"]

    step2 = next(o for o in result.outcomes if o.step.id == "step-02")
    assert step2.attempts == 2, "step-02 should have been retried exactly once"
    assert step2.rolled_back is True, "rollback flag must persist across retries"
    assert step2.verdict.passed

    # The rollback event must have fired for step-02 (and only step-02).
    rollbacks = [e for e in events if e.kind == "step.rollback"]
    assert len(rollbacks) == 1
    assert rollbacks[0].payload["id"] == "step-02"

    # The final working tree passes pytest — ground truth.
    proc = subprocess.run(["pytest", "-q"], cwd=ws, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # The git log tells a clean story: one commit per step (+ plan commit), no
    # trace of the failed attempt. This is the ``audit trail`` promise from
    # the blog — rollback leaves no zombie commits behind.
    shas_by_step = {o.step.id: o.checkpoint.short_sha for o in result.outcomes}
    assert len(set(shas_by_step.values())) == 3

    # The monitor artifact for step-02 reflects the *successful* retry.
    verdict_json = json.loads((ws / ".taste" / "monitor" / "step-02.json").read_text())
    assert verdict_json["passed"] is True
    assert verdict_json["step_id"] == "step-02"


def test_halts_when_retries_exhausted(refactor_workspace: Path) -> None:
    """If the worker stays broken, the kernel gives up cleanly."""
    ws = refactor_workspace
    math = ws / LEGACY_MATH

    def always_broken(step: Step, plan: Plan) -> WorkerResult:
        if step.id == "step-02":
            math.write_text(BROKEN_STEP2)
        else:
            math.write_text(math.read_text() + "\n")
        return WorkerResult(summary=f"{step.id} broken", tool_calls=0, stopped_reason="end_turn")

    events: list[Event] = []
    kernel = Kernel(workspace=ws, max_retries=1, on_event=events.append)
    result = kernel.run(task="broken", spec=_spec(), plan_override=_plan(), worker_override=always_broken)

    assert result.status == "failed"
    # Execution halted after step-02 failed — step-03 never ran.
    assert [o.step.id for o in result.outcomes] == ["step-01", "step-02"]
    step2 = result.outcomes[-1]
    assert step2.attempts == 2  # initial attempt + 1 retry
    assert step2.rolled_back is True
    assert step2.verdict.passed is False

    # The halt event carries the monitor's reason so an operator can triage.
    halt = next(e for e in events if e.kind == "run.halt")
    assert halt.payload["step"] == "step-02"
    assert "pytest" in halt.payload["reason"].lower() or "exited" in halt.payload["reason"].lower()

    # The workspace was left at the last-good state (end of step-01), so the
    # tests pass — garbage from the broken attempts didn't survive.
    proc = subprocess.run(["pytest", "-q"], cwd=ws, capture_output=True, text=True)
    assert proc.returncode == 0


def test_rich_event_stream_is_complete(refactor_workspace: Path) -> None:
    """Smoke test the observability surface the dashboard will consume."""
    ws = refactor_workspace

    def noop_worker(step: Step, plan: Plan) -> WorkerResult:
        (ws / LEGACY_MATH).write_text((ws / LEGACY_MATH).read_text() + f"\n# {step.id}\n")
        return WorkerResult(summary="", tool_calls=0, stopped_reason="end_turn")

    events: list[Event] = []
    Kernel(workspace=ws, on_event=events.append).run(
        task="noop",
        spec=_spec(),
        plan_override=_plan(),
        worker_override=noop_worker,
    )

    kinds = [e.kind for e in events]
    assert kinds[0] == "run.start"
    assert kinds[-1] == "run.done"
    assert kinds.count("plan.ready") == 1
    assert kinds.count("step.begin") == 3
    assert kinds.count("monitor.verdict") == 3
    assert all(e.payload for e in events), "every event carries a payload"
