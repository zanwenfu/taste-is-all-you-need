"""Absolute trial admission survives a delayed driver and closes cycle bypass."""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone

import pytest

from taste.brains.central_host import compose_central_runtime
from taste.brains.central_runtime import CoordinatorError, _digest, _goal_root
from tests.test_brains_central_runtime import (
    FakeLauncher,
    ScriptedTransport,
    assignment_for,
    simple_goal,
)
from tests.test_brains_supervisor import FakeClock


def assignments(request, _prompt):
    return (assignment_for(request, "build", "worker-build", "product.txt"),)


@pytest.fixture
def host(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    value = compose_central_runtime(
        repo, "admission", simple_goal(), transport=ScriptedTransport(assignments),
        launcher=FakeLauncher(),
    )
    value.runtime.clock = value.supervisor.clock = FakeClock()
    try:
        yield value
    finally:
        if not value.closed:
            for run in value.supervisor.runs():
                value.supervisor.stop(run.run_id, "test_cleanup")
            value.close()


def bounds():
    return {"max_generations": 1, "wall_clock_seconds": 30.0, "max_planner_failures": 1}


def path(host):
    return f"{_goal_root(host.goal.goal_id)}/run-limits.json"


def stored_limits(host, *, expired=False):
    return {"schema": "taste.brains/GoalRunLimits/1", "goal_digest": _digest(host.goal.to_json()),
            **bounds(), "deadline_at": (host.runtime.clock() + timedelta(
                seconds=-1 if expired else 10)).isoformat()}


def test_direct_cycle_cannot_bypass_an_existing_expired_trial_deadline(host):
    host.control.checkpoint("previous trial admission", records={path(host): stored_limits(host, expired=True)})
    before = host.control.head.id
    with pytest.raises(CoordinatorError, match="run"):
        host.cycle()
    assert host.transport.calls == [] and host.launcher.launch_calls == []
    assert host.control.head.id == before


def test_preparation_is_idempotent_and_makes_no_provider_call_or_worker(host):
    deadline = host.runtime.clock() + timedelta(seconds=10)
    limits = host.prepare_run(**bounds(), deadline_at=deadline)
    assert datetime.fromisoformat(limits["deadline_at"].replace("Z", "+00:00")) == deadline
    before = host.control.head.id
    assert host.prepare_run(**bounds(), deadline_at=deadline.astimezone(timezone(timedelta(hours=3)))) == limits
    assert host.control.head.id == before
    assert host.transport.calls == [] and host.launcher.launch_calls == []
    assert not host.supervisor.runs()
    with pytest.raises(CoordinatorError, match="run"):
        host.cycle()


def test_expired_prepared_deadline_survives_host_close_and_reconstruction(host):
    clock = host.runtime.clock
    host.prepare_run(**bounds(), deadline_at=clock() + timedelta(seconds=5))
    root, goal = host.store.root, host.goal
    host.close()
    clock.advance(6)
    restored = compose_central_runtime(root, "admission", goal,
                                       transport=ScriptedTransport(assignments), launcher=FakeLauncher())
    restored.runtime.clock = restored.supervisor.clock = clock
    try:
        ending = restored.run(**bounds())
        assert ending.stop_reason == "wall_clock" and not ending.complete
        assert restored.transport.calls == [] and restored.launcher.launch_calls == []
        assert not ending.budget.unknown_planner_attempt_ids
    finally:
        restored.close()


def test_separate_driver_process_cannot_reset_an_expired_admission(host):
    host.runtime.clock = lambda: datetime.now(UTC)
    host.prepare_run(**bounds(), deadline_at=datetime.now(UTC) - timedelta(seconds=1))
    root = host.store.root
    host.close()
    code = """
import json, sys
from taste.brains.central_host import compose_central_runtime
from tests.test_brains_central_runtime import FakeLauncher, ScriptedTransport, simple_goal
def forbidden(*args):
    raise AssertionError('expired admission reached the provider')
host = compose_central_runtime(sys.argv[1], 'admission', simple_goal(),
    transport=ScriptedTransport(forbidden), launcher=FakeLauncher())
try:
    ending = host.run(max_generations=1, wall_clock_seconds=30, max_planner_failures=1)
    assert ending.stop_reason == 'wall_clock'
    assert not host.transport.calls and not host.launcher.launch_calls
    print(json.dumps(ending.to_dict()))
finally:
    host.close()
"""
    result = subprocess.run([sys.executable, "-c", code, str(root)], text=True,
                            capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"stop_reason": "wall_clock"' in result.stdout


@pytest.mark.parametrize("changes", [
    {"max_generations": 2}, {"wall_clock_seconds": 60.0}, {"max_planner_failures": 2},
    {"deadline_delta": 1}, {"deadline_delta": -1},
])
def test_prepared_identity_cannot_be_replaced_by_later_bounds(host, changes):
    deadline = host.runtime.clock() + timedelta(seconds=10)
    original = host.prepare_run(**bounds(), deadline_at=deadline)
    values = bounds() | {key: value for key, value in changes.items() if key != "deadline_delta"}
    with pytest.raises(CoordinatorError, match="differ"):
        host.prepare_run(**values, deadline_at=deadline + timedelta(seconds=changes.get("deadline_delta", 0)))
    assert host.control.head.record(path(host)) == original
    assert not host.transport.calls and not host.launcher.launch_calls


@pytest.mark.parametrize("invalid", [
    {"max_generations": True}, {"max_generations": 0}, {"max_planner_failures": False},
    {"wall_clock_seconds": True}, {"wall_clock_seconds": float("inf")},
    {"deadline_at": None}, {"deadline_at": "2026-09-22T00:00:00Z"},
    {"deadline_at": datetime(2026, 1, 1)},
])
def test_invalid_admission_is_rejected_before_any_control_write(host, invalid):
    before = host.control.head.id
    values = bounds() | {"deadline_at": host.runtime.clock() + timedelta(seconds=10)} | invalid
    with pytest.raises((ValueError, TypeError)):
        host.prepare_run(**values)
    assert host.control.head.id == before
    assert not host.transport.calls and not host.launcher.launch_calls


def test_admission_cannot_choose_a_deadline_beyond_its_declared_allowance(host):
    before = host.control.head.id
    with pytest.raises(ValueError, match="exceeds"):
        host.prepare_run(**bounds(), deadline_at=host.runtime.clock() + timedelta(seconds=31))
    assert host.control.head.id == before and not host.transport.calls


def test_delayed_driver_caps_worker_admission_at_the_prepared_remaining_time(host):
    clock = host.runtime.clock
    host.prepare_run(**bounds(), deadline_at=clock() + timedelta(seconds=10))
    clock.advance(6)
    observed = []

    def finish():
        runs = host.supervisor.runs()
        if runs:
            observed.extend(run.wall_timeout_seconds for run in runs)
            clock.advance(10)

    ending = host.run(**bounds(), between_cycles=finish)
    assert ending.stop_reason == "wall_clock"
    assert observed and all(0 < remaining <= 4 for remaining in observed)
    assert all(run.terminal for run in host.supervisor.runs())


def test_preparation_rejects_an_already_active_driver(host):
    clock = host.runtime.clock
    deadline = clock() + timedelta(seconds=10)
    host.prepare_run(**bounds(), deadline_at=deadline)

    def nested_prepare():
        with pytest.raises(CoordinatorError, match="active run"):
            host.prepare_run(**bounds(), deadline_at=deadline)
        clock.advance(11)

    assert host.run(**bounds(), between_cycles=nested_prepare).stop_reason == "wall_clock"


def test_preparation_cannot_reopen_a_cancelled_goal(host):
    ending = host.stop_and_drain()
    with pytest.raises(CoordinatorError, match="shutdown"):
        host.prepare_run(**bounds(), deadline_at=host.runtime.clock() + timedelta(seconds=10))
    assert host.outcome() == ending and host.control.head.record(path(host)) is None


def test_preparation_refuses_goal_execution_that_started_without_trial_admission(host):
    host.cycle()
    calls = len(host.transport.calls)
    with pytest.raises(CoordinatorError, match="execution has begun"):
        host.prepare_run(**bounds(), deadline_at=host.runtime.clock() + timedelta(seconds=10))
    assert len(host.transport.calls) == calls and host.control.head.record(path(host)) is None


@pytest.mark.parametrize("field", ["max_generations", "max_planner_failures", "wall_clock_seconds"])
def test_boolean_durable_bounds_are_not_accepted_as_numeric_limits(host, field):
    limits = stored_limits(host, expired=True)
    limits[field] = True
    if field == "wall_clock_seconds":
        requested = bounds() | {"wall_clock_seconds": 1.0}
    else:
        requested = bounds()
    host.control.checkpoint("corrupt durable type", records={path(host): limits})
    with pytest.raises(CoordinatorError, match="run limits differ"):
        host.run(**requested)
    assert not host.transport.calls and not host.launcher.launch_calls
