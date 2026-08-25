"""Wave-0 kernel provenance + failure taxonomy.

Every run must leave a session-keyed manifest (models, temperatures, prompt
hashes, harness SHA) so results remain attributable months later; infra and
budget failures must be classified distinctly from genuine task failures; and
a typed failure mid-step must leave the session branch clean.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.conftest import PYTEST_CMD
from taste.agent import AgentSpec
from taste.cores import (
    MODEL_PLANNER,
    MONITOR_TEMPERATURE,
    Plan,
    Step,
    Verification,
    WorkerResult,
    plan,
)
from taste.kernel import Kernel
from taste.llm import BudgetExceeded, InfraFailure
from tests.fakes import FakeLLM, plan_turn


def _spec(model: str | None = None) -> AgentSpec:
    return AgentSpec(
        name="scripted",
        description="deterministic harness test",
        model=model,
        system_prompt="be careful",
    )


def _plan_one_step() -> Plan:
    return Plan(
        task="one step",
        steps=[
            Step(
                id="step-01",
                description="noop",
                verification=Verification(kind="shell", command=PYTEST_CMD),
            )
        ],
    )


def _noop_worker(step: Step, plan_: Plan) -> WorkerResult:
    return WorkerResult(summary="", tool_calls=0, stopped_reason="end_turn")


def _fake_planner_llm() -> FakeLLM:
    """A Planner that answers with a minimal one-step plan."""
    return FakeLLM(
        [
            plan_turn(
                [
                    {
                        "id": "step-01",
                        "description": "d",
                        "verification": {"kind": "shell", "command": "true"},
                    }
                ]
            )
        ],
        model=MODEL_PLANNER,
    )


def _manifest(ws: Path, session_id: str) -> dict:
    return json.loads((ws / ".git" / "taste" / f"manifest-{session_id}.json").read_text())


# ------------------------------------------------------------------ manifest


def test_manifest_written_with_provenance_fields(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    result = Kernel(workspace=ws).run(
        task="manifest",
        spec=_spec(model="claude-sonnet-4-6"),
        plan_override=_plan_one_step(),
        worker_override=_noop_worker,
    )
    assert result.status == "completed"

    manifest = _manifest(ws, result.session_id)
    assert manifest["session"] == result.session_id
    assert manifest["models"]["worker"] == "claude-sonnet-4-6"
    assert manifest["models"]["planner"] == MODEL_PLANNER  # spec.model must NOT leak here
    assert manifest["temperature"]["monitor"] == MONITOR_TEMPERATURE
    assert set(manifest["prompt_sha"]) == {
        "planner_system",
        "worker_system",
        "monitor_system",
        "agent_system_prompt",
    }
    assert all(len(v) == 16 for v in manifest["prompt_sha"].values())
    # This repo IS a git checkout, so the harness SHA must resolve.
    assert len(manifest["harness_git_sha"]) == 40


def test_manifests_accumulate_per_session(refactor_workspace: Path) -> None:
    """A second run must not clobber the first session's provenance."""
    ws = refactor_workspace
    kernel = Kernel(workspace=ws)
    r1 = kernel.run(
        task="a",
        spec=_spec(),
        session_id="s-one",
        plan_override=_plan_one_step(),
        worker_override=_noop_worker,
    )
    r2 = kernel.run(
        task="b",
        spec=_spec(),
        session_id="s-two",
        base_ref="taste/session-s-one",
        plan_override=_plan_one_step(),
        worker_override=_noop_worker,
    )
    assert _manifest(ws, r1.session_id)["task"] == "a"
    assert _manifest(ws, r2.session_id)["task"] == "b"


def test_planner_model_knob_reaches_manifest(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    result = Kernel(workspace=ws, planner_model="claude-haiku-4-5-20251001").run(
        task="knob",
        spec=_spec(),
        plan_override=_plan_one_step(),
        worker_override=_noop_worker,
    )
    assert _manifest(ws, result.session_id)["models"]["planner"] == "claude-haiku-4-5-20251001"


# ------------------------------------------------------------------ failure taxonomy


def test_infra_failure_classified_not_task_failure(refactor_workspace: Path) -> None:
    def infra_worker(step: Step, plan_: Plan) -> WorkerResult:
        raise InfraFailure("api down", attempts=5, last_error=None)

    result = Kernel(workspace=refactor_workspace).run(
        task="infra",
        spec=_spec(),
        plan_override=_plan_one_step(),
        worker_override=infra_worker,
    )
    assert result.status == "failed"
    assert result.failure_kind == "infra"
    assert "api down" in (result.failure_reason or "")


def test_budget_exceeded_classified(refactor_workspace: Path) -> None:
    def budget_worker(step: Step, plan_: Plan) -> WorkerResult:
        raise BudgetExceeded(spent_usd=5.0, budget_usd=4.0)

    result = Kernel(workspace=refactor_workspace).run(
        task="budget",
        spec=_spec(),
        plan_override=_plan_one_step(),
        worker_override=budget_worker,
    )
    assert result.status == "failed"
    assert result.failure_kind == "budget"


def test_task_failure_still_classified_task(refactor_workspace: Path) -> None:
    ws = refactor_workspace

    def breaking_worker(step: Step, plan_: Plan) -> WorkerResult:
        (ws / "legacy_math.py").write_text("def broken(\n")
        return WorkerResult(summary="", tool_calls=0, stopped_reason="end_turn")

    result = Kernel(workspace=ws, max_retries=0).run(
        task="task-fail",
        spec=_spec(),
        plan_override=_plan_one_step(),
        worker_override=breaking_worker,
    )
    assert result.status == "failed"
    assert result.failure_kind == "task"


def test_completed_run_has_no_failure_kind(refactor_workspace: Path) -> None:
    result = Kernel(workspace=refactor_workspace).run(
        task="ok",
        spec=_spec(),
        plan_override=_plan_one_step(),
        worker_override=_noop_worker,
    )
    assert result.status == "completed"
    assert result.failure_kind is None


def test_unpriced_spec_model_halts_cleanly_before_spend(refactor_workspace: Path) -> None:
    """A spec naming an unpriced model must yield a classified result, not a crash."""
    kernel = Kernel(workspace=refactor_workspace)
    # Any LLM at all makes this a "real run" as far as model validation cares.
    kernel.llm = FakeLLM([])
    result = kernel.run(
        task="unpriced",
        spec=_spec(model="claude-sonnet-99-imaginary"),
        plan_override=_plan_one_step(),
        worker_override=_noop_worker,
    )
    assert result.status == "failed"
    assert result.failure_kind == "infra"
    assert "no verified price" in (result.failure_reason or "")


def test_mid_step_infra_failure_leaves_branch_clean(refactor_workspace: Path) -> None:
    """Typed failure mid-step: in-flight junk must not survive to contaminate
    the next run's first checkpoint."""
    ws = refactor_workspace

    def dirty_then_die(step: Step, plan_: Plan) -> WorkerResult:
        (ws / "junk_in_flight.txt").write_text("half-finished\n")
        raise InfraFailure("api died mid-step", attempts=5, last_error=None)

    result = Kernel(workspace=ws).run(
        task="dirty-abort",
        spec=_spec(),
        plan_override=_plan_one_step(),
        worker_override=dirty_then_die,
    )
    assert result.failure_kind == "infra"
    assert not (ws / "junk_in_flight.txt").exists(), "abort must clean the working tree"
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ws, capture_output=True, text=True
    )
    assert porcelain.stdout.strip() == "", "session branch must be left clean"


# ------------------------------------------------------------------ role-collapse fix


def test_spec_model_does_not_override_planner_model() -> None:
    """The audited validity threat: spec.model used to hijack the Planner too."""
    llm = _fake_planner_llm()
    result = plan(llm, "task", _spec(model="claude-sonnet-4-6"), "(empty)")
    assert llm.calls[0]["model"] == MODEL_PLANNER, "spec.model configures the Worker only"
    assert llm.calls[0]["role"] == "planner"
    assert len(result.steps) == 1


def test_explicit_planner_model_override_wins() -> None:
    llm = _fake_planner_llm()
    plan(llm, "task", _spec(), "(empty)", model="claude-haiku-4-5-20251001")
    assert llm.calls[0]["model"] == "claude-haiku-4-5-20251001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
