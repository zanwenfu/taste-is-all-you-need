"""The frozen baseline every future subsystem must reproduce when disabled.

This test does two jobs. It pins today's kernel behavior as an explicit,
readable expectation, and it proves the signature instrument itself is
deterministic — a fingerprint that varied run to run would be useless as an
ablation check.
"""

from __future__ import annotations

from pathlib import Path

from tests.golden import rollback_scenario

# The event stream of a run with one failure, one rollback, one retry.
# Payload keys are part of the contract: existing kinds are frozen, so new
# information goes in NEW event kinds, never a widened payload. That is what
# keeps ablation-equivalence a mechanical diff.
EXPECTED_EVENTS = (
    ("run.start", ("agent", "branch", "session", "task")),
    ("run.manifest", ("harness_git_sha", "models", "temperature")),
    ("plan.ready", ("parallel_waves", "steps", "waves")),
    ("step.begin", ("attempt", "id")),
    ("worker.done", ("id", "stop", "tools")),
    ("monitor.verdict", ("id", "passed", "reason", "sha")),
    ("step.rollback", ("id", "remaining_retries", "to")),
    ("step.begin", ("attempt", "id")),
    ("worker.done", ("id", "stop", "tools")),
    ("monitor.verdict", ("id", "passed", "reason", "sha")),
    ("step.begin", ("attempt", "id")),
    ("worker.done", ("id", "stop", "tools")),
    ("monitor.verdict", ("id", "passed", "reason", "sha")),
    (
        "run.done",
        ("cache_hit_rate", "cost_usd", "elapsed", "failure_kind", "reason", "status"),
    ),
)


def test_baseline_signature_is_exactly_as_expected(refactor_workspace: Path) -> None:
    sig = rollback_scenario(refactor_workspace).run(refactor_workspace)

    assert sig.events == EXPECTED_EVENTS
    assert sig.status == "completed"
    assert sig.failure_kind is None
    # step-01 needed two attempts and was rolled back; step-02 was clean.
    assert sig.step_results == (
        ("step-01", True, 2, True),
        ("step-02", True, 1, False),
    )


def test_signature_is_deterministic_across_identical_runs(
    refactor_workspace: Path, tmp_path_factory
) -> None:
    """Two identical runs in two fresh workspaces must fingerprint identically."""
    from examples.refactor_demo.bootstrap import bootstrap

    second = tmp_path_factory.mktemp("golden-second")
    bootstrap(second)

    a = rollback_scenario(refactor_workspace).run(refactor_workspace)
    b = rollback_scenario(second).run(second)

    assert a == b, a.diff(b)


def test_rollback_actually_discarded_the_failed_attempt(refactor_workspace: Path) -> None:
    """The wrong file from attempt 1 must not survive into the final tree."""
    ws = refactor_workspace
    rollback_scenario(ws).run(ws)

    assert (ws / "made.py").exists()
    assert (ws / "also.py").exists()
    assert not (ws / "wrong.py").exists(), "rolled-back work leaked into the final tree"


def test_diff_reports_the_first_divergence(refactor_workspace: Path) -> None:
    """The instrument must explain a mismatch, not just report one."""
    sig = rollback_scenario(refactor_workspace).run(refactor_workspace)
    mutated = type(sig)(
        events=(*sig.events[:-1], ("run.done", ("elapsed", "extra", "status"))),
        commits=sig.commits,
        status=sig.status,
        failure_kind=sig.failure_kind,
        step_results=sig.step_results,
    )
    report = sig.diff(mutated)
    assert "run.done" in report and "extra" in report
    assert sig.diff(sig) == "(identical)"
