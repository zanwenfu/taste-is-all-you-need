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
#
# Payload keys are part of the contract: information produced by an optional
# subsystem goes in a NEW event kind, never a widened payload on an existing
# one. That is what keeps ablation-equivalence a mechanical diff — a widened
# payload carrying subsystem output would appear in one arm and not another
# and the diff would stop meaning "the subsystem is off".
#
# `attempt` and `failing_tests` on monitor.verdict are deliberate exceptions,
# and both are sound for the reason the rule exists: they are core kernel
# identity, emitted in every configuration and derived from the verdict
# itself. Nothing optional produces them, so they shift every arm's baseline
# identically and no ablation diff moves.
#
# `attempt` is here because attribution has to join a Monitor failure to the
# shadow observation it graded, and the key for that is (step_id, attempt).
# Recovering it by walking back to the preceding step.begin would work only
# while steps run one at a time; under a parallel wave the events interleave
# and the walk-back silently picks up another step's attempt.
#
# `failing_tests` is here because attribution's coverage rule starts from the
# test ids a failure named, and the event stream is the only artifact of a
# run that survives to scoring. Without it every real failure classified as
# "reported no test identities" — UNKNOWN, never attributed — and the
# silent-vs-detected split was structurally dead on every production run.
EXPECTED_EVENTS = (
    ("run.start", ("agent", "branch", "session", "task")),
    ("run.manifest", ("harness_git_sha", "models", "temperature")),
    ("plan.ready", ("parallel_waves", "steps", "waves")),
    ("step.begin", ("attempt", "id")),
    ("worker.done", ("id", "stop", "tools")),
    ("monitor.verdict", ("attempt", "failing_tests", "id", "passed", "reason", "sha")),
    ("step.rollback", ("id", "remaining_retries", "to")),
    ("step.begin", ("attempt", "id")),
    ("worker.done", ("id", "stop", "tools")),
    ("monitor.verdict", ("attempt", "failing_tests", "id", "passed", "reason", "sha")),
    ("step.begin", ("attempt", "id")),
    ("worker.done", ("id", "stop", "tools")),
    ("monitor.verdict", ("attempt", "failing_tests", "id", "passed", "reason", "sha")),
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
