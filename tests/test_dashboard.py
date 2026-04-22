"""Dashboard rendering — the ``htop for agents`` artifact.

Verifies that the HTML output faithfully reflects a run's artifacts: plan
steps, per-step verdicts, rollbacks, timeline events, git topology.
"""

from __future__ import annotations

from pathlib import Path

from taste.agent import AgentSpec
from taste.cores import Plan, Step, Verification, WorkerResult
from taste.dashboard import RunArtifacts, render, write
from taste.kernel import Kernel

BROKEN = """\
def run(items):
    total = 0
    for x in items:
        if x > 0:
            total += x  # BUG
        else:
            total -= x
    return total


def fmt(total):
    return f"total is {total}"


def main(items):
    return fmt(run(items))
"""

CORRECT = """\
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


def _run_with_rollback(ws: Path) -> None:
    math = ws / "legacy_math.py"
    attempts: dict[str, int] = {}

    def worker(step: Step, plan: Plan) -> WorkerResult:
        attempts[step.id] = attempts.get(step.id, 0) + 1
        n = attempts[step.id]
        if step.id == "step-01":
            math.write_text(math.read_text() + "\n# hdr\n")
        elif step.id == "step-02":
            math.write_text(BROKEN if n == 1 else CORRECT)
        return WorkerResult(summary="", tool_calls=0, stopped_reason="end_turn")

    check = Verification(kind="shell", command="pytest -q")
    plan = Plan(
        task="dashboard fixture",
        steps=[
            Step(id="step-01", description="header", verification=check),
            Step(id="step-02", description="add hints", verification=check),
        ],
    )
    Kernel(workspace=ws, max_retries=2).run(
        task=plan.task,
        spec=AgentSpec(name="t", description=""),
        session_id="dash",
        plan_override=plan,
        worker_override=worker,
    )


def test_dashboard_captures_rollback_story(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    _run_with_rollback(ws)

    artifacts = RunArtifacts.load(ws)
    assert artifacts.branch == "taste/session-dash"
    assert len(artifacts.plan["steps"]) == 2
    assert any(e["kind"] == "step.rollback" for e in artifacts.events)

    html_out = render(artifacts)
    # Header cards.
    assert "session <code>dash</code>" in html_out
    assert ">completed<" in html_out
    # Per-step: step-01 shows PASS with attempts=1, step-02 shows PASS with
    # rollback="yes" and attempts=2.
    assert "badge-pass" in html_out
    assert "rollback-cell'>yes" in html_out
    # Timeline has the rollback event painted red.
    assert "evt-rollback" in html_out


def test_dashboard_file_is_self_contained(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    _run_with_rollback(ws)

    out = write(ws)
    assert out.exists()
    text = out.read_text()
    # No external CSS/JS/fonts — opens offline.
    for forbidden in ("<script", "http://", "cdn.", "googleapis", "fonts.google"):
        assert forbidden not in text


def test_dashboard_handles_missing_artifacts(tmp_path: Path) -> None:
    """Rendering an empty workspace doesn't crash — degrades gracefully."""
    workspace = tmp_path / "empty"
    workspace.mkdir()
    (workspace / ".git").mkdir()  # fake; git_log falls through to empty
    artifacts = RunArtifacts.load(workspace)
    html_out = render(artifacts)
    assert "session <code>(unknown)</code>" in html_out
    assert "No plan yet" in html_out
