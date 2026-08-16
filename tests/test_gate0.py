"""Gate 0 — and whether it could fail.

A validation gate that cannot fail validates nothing. Half of these tests
check that the gate passes on a working instrument; the other half break the
instrument deliberately and assert the gate notices.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taste import gate0
from taste.gate0 import (
    FLAKE_MAX,
    NEGATIVE_CONTROL_MIN,
    POSITIVE_CONTROL_MIN,
    UNKNOWN_MAX,
    CheckResult,
    Gate0Report,
    clean_trajectory,
    recovered_trajectory,
    regressed_trajectory,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "gate0"


def _workspace_factory(root: Path):
    import shutil

    def make(name: str) -> Path:
        path = root / name
        if path.exists():
            shutil.rmtree(path)
        return path

    return make


# ------------------------------------------------------------------ scenarios


def test_a_clean_trajectory_never_breaks_the_probe() -> None:
    for state in clean_trajectory().states:
        assert "return 1" in state


def test_a_regressed_trajectory_breaks_at_the_declared_observation() -> None:
    trajectory = regressed_trajectory(length=8, onset=5)
    # Observations are 1-based; states are 0-based.
    assert "return 1" in trajectory.states[3]
    assert "return 2" in trajectory.states[4]
    assert trajectory.regression_at == 5


def test_a_recovered_trajectory_returns_to_health() -> None:
    """The shape the previous instrument could not see at all."""
    trajectory = recovered_trajectory(length=9, onset=4, recovery=7)
    assert "return 1" in trajectory.states[2]   # before onset
    assert "return 2" in trajectory.states[3]   # observation 4: broken
    assert "return 2" in trajectory.states[5]   # observation 6: still broken
    assert "return 1" in trajectory.states[6]   # observation 7: repaired
    assert trajectory.regression_at == 4


# ------------------------------------------------------------------ the checks


def test_negative_control_passes_on_clean_runs(root: Path) -> None:
    result = gate0.negative_control(_workspace_factory(root), samples=3)
    assert result.passed, result.render()
    assert result.value == 1.0


def test_positive_control_locates_every_onset_including_recovered(root: Path) -> None:
    result = gate0.positive_control(_workspace_factory(root))
    assert result.passed, result.render()
    assert result.value == 1.0, "a recovered regression must still be located"


def test_flake_screen_finds_no_instability_on_deterministic_probes(root: Path) -> None:
    result = gate0.flake_screen(_workspace_factory(root), k=2)
    assert result.passed and result.value == 0.0


def test_unknown_rate_is_zero_when_probes_run(root: Path) -> None:
    result = gate0.unknown_rate(_workspace_factory(root))
    assert result.passed and result.value == 0.0


def test_the_whole_gate_passes_on_a_working_instrument(root: Path) -> None:
    report = gate0.run(root)
    assert report.passed, report.render()
    assert len(report.checks) == 4


# ------------------------------------------------------------------ can it fail?


def test_the_gate_fails_when_the_probe_cannot_run(root: Path, monkeypatch) -> None:
    """A gate that cannot fail validates nothing."""
    from taste.replay import LocalWorktreeExecutor, SuiteRun

    monkeypatch.setattr(
        LocalWorktreeExecutor,
        "run",
        lambda self, sha, suite: SuiteRun(
            statuses=dict.fromkeys(suite.members, "error"), infra_error="broken"
        ),
    )
    result = gate0.unknown_rate(_workspace_factory(root))
    assert not result.passed
    assert result.value == 1.0


def test_the_gate_fails_when_onsets_are_mislocated(root: Path, monkeypatch) -> None:
    """Exactly the earlier defect: an instrument blind to repaired regressions."""
    import taste.gate0 as module
    from taste.replay import ReplayReport

    def blind(memory, timeline, probes, **kwargs):
        # Mimics the old bisection: only a probe still failing at the end
        # produces an episode, so a recovered regression vanishes.
        return ReplayReport(session="x", observations=len(timeline))

    monkeypatch.setattr(module, "reconstruct", blind)
    result = gate0.positive_control(_workspace_factory(root))
    assert not result.passed
    assert result.value == 0.0


def test_the_gate_fails_on_a_manufactured_regression(root: Path, monkeypatch) -> None:
    """A false positive is the worst failure the instrument can have."""
    from taste.replay import LocalWorktreeExecutor, SuiteRun

    calls = {"n": 0}

    def flapping(self, sha, suite):
        calls["n"] += 1
        verdict = "fail" if calls["n"] % 2 == 0 else "pass"
        return SuiteRun(statuses=dict.fromkeys(suite.members, verdict))

    monkeypatch.setattr(LocalWorktreeExecutor, "run", flapping)
    result = gate0.negative_control(_workspace_factory(root), samples=2)
    assert not result.passed


# ------------------------------------------------------------------ reporting


def test_thresholds_are_declared_before_results() -> None:
    """Stated as module constants so the gate cannot be moved afterwards."""
    assert NEGATIVE_CONTROL_MIN == 0.95
    assert POSITIVE_CONTROL_MIN == 0.90
    assert FLAKE_MAX == 0.02
    assert UNKNOWN_MAX == 0.05


def test_a_report_with_one_failure_does_not_pass() -> None:
    report = Gate0Report(
        checks=[
            CheckResult("a", True, 1.0, 0.9),
            CheckResult("b", False, 0.1, 0.9),
        ]
    )
    assert not report.passed
    assert "GATE 0 FAILED" in report.render()
    assert "paid runs would be uninterpretable" in report.render()


def test_an_empty_report_does_not_pass() -> None:
    """No checks run is not the same as every check passing."""
    assert not Gate0Report().passed
