"""Prepared goal inputs cross file, process, memory, provider and cleanup boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from taste.brains.central_host import compose_central_runtime
from taste.brains.central_planner import PlannerTransportError, _goal_path
from taste.brains.central_runtime import _goal_root
from taste.brains.goal_entrypoint import (
    GoalInputError,
    GoalProcessInput,
    execute_goal,
    goal_command,
    load_goal_input,
    prepare_goal_process,
)
from taste.memstore import NoSuchState, Store
from tests.test_brains_central_runtime import (
    FakeLauncher,
    ScriptedTransport,
    complete_response,
    simple_goal,
)
from tests.test_goal_cancellation import _leaves, wait_event
from tests.test_thread_shutdown_ownership import cancel_loop_tasks


@pytest.fixture
def prepared(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    launcher = FakeLauncher()
    transport = ScriptedTransport(complete_response)
    hosts = []

    def factory(*args, **kwargs):
        host = compose_central_runtime(*args, **kwargs, transport=transport, launcher=launcher)
        hosts.append(host)
        return host

    config = prepare_goal_process(
        repo, "entrypoint", simple_goal(budget_usd=5),
        max_generations=2, wall_clock_seconds=30, max_planner_failures=2,
        deadline_at=datetime.now(UTC) + timedelta(seconds=30), host_factory=factory,
    )
    assert not transport.calls and not launcher.launch_calls
    assert hosts[-1].closed
    yield config, factory, transport, launcher, hosts
    for host in hosts:
        if not host.closed:
            host.stop_and_drain("test cleanup")
            host.close()


def mutate_control(config, records):
    store = Store.open(Path(config.repo_root), config.session)
    try:
        store.branch("central-control").checkpoint("test changes after admission", records=records)
    finally:
        store.close()


def test_prepared_input_roundtrip_retains_exact_deadline_and_budget(prepared, tmp_path):
    config, *_ = prepared
    raw = config.to_bytes()
    path = tmp_path / "input.json"
    path.write_bytes(raw)
    assert load_goal_input(path, hashlib.sha256(raw).hexdigest()) == config
    assert config.limits["wall_clock_seconds"] == 30
    assert config.goal.budget_usd == 5
    # A returned dictionary cannot mutate an admitted configuration in place.
    config.limits["deadline_at"] = "2099-01-01T00:00:00Z"
    assert config.to_bytes() == raw


def test_real_composition_completes_then_replays_without_new_calls(prepared):
    config, factory, transport, launcher, hosts = prepared
    outcome = asyncio.run(execute_goal(config, host_factory=factory))
    assert outcome.complete and outcome.budget.enforceable
    assert len(transport.calls) == 1 and not launcher.launch_calls
    assert hosts[-1].closed
    assert asyncio.run(execute_goal(config, host_factory=factory)) == outcome
    assert asyncio.run(execute_goal(config, mode="settle", host_factory=factory)) == outcome
    assert len(transport.calls) == 1 and all(host.closed for host in hosts)


@pytest.mark.parametrize("changed", ["missing_limits", "new_deadline", "different_goal"])
def test_changed_control_input_never_dispatches_or_recreates_limits(prepared, changed):
    config, factory, transport, launcher, hosts = prepared
    path = f"{_goal_root(config.goal.goal_id)}/run-limits.json"
    records = {path: None if changed == "missing_limits" else config.limits | {
        "deadline_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    }}
    if changed == "different_goal":
        records = {_goal_path(config.goal.goal_id): config.goal.to_dict() | {"task": "other task"}}
    mutate_control(config, records)
    count = len(hosts)
    with pytest.raises(GoalInputError, match="differs"):
        asyncio.run(execute_goal(config, host_factory=factory))
    assert len(hosts) == count and not transport.calls and not launcher.launch_calls
    store = Store.open(Path(config.repo_root), config.session)
    try:
        for path, value in records.items():
            assert store.view("central-control").head.record(path) == value
    finally:
        store.close()


def test_control_change_between_preflight_and_lease_is_rejected(prepared):
    config, factory, transport, launcher, hosts = prepared

    def racing_factory(*args, **kwargs):
        host = factory(*args, **kwargs)
        host.control.checkpoint("changed while taking lease", records={
            f"{_goal_root(config.goal.goal_id)}/run-limits.json": None,
        })
        return host

    with pytest.raises(GoalInputError, match="differs"):
        asyncio.run(execute_goal(config, host_factory=racing_factory))
    assert hosts[-1].closed and not transport.calls and not launcher.launch_calls


def test_missing_session_cannot_create_a_replacement_control_branch(prepared):
    config, factory, transport, launcher, hosts = prepared
    other = replace(config, session="absent-session")
    count = len(hosts)
    with pytest.raises(NoSuchState, match="no head"):
        asyncio.run(execute_goal(other, host_factory=factory))
    store = Store.open(Path(config.repo_root), other.session)
    try:
        assert store.branches() == []
    finally:
        store.close()
    assert len(hosts) == count and not transport.calls and not launcher.launch_calls


def test_foreign_prepared_state_is_rejected_even_when_limits_match(prepared):
    config, factory, transport, launcher, _hosts = prepared
    host = factory(config.repo_root, config.session, config.goal)
    try:
        foreign = host.store.branch("other-control", from_state=host.control.head)
        altered = replace(config, prepared_state_id=foreign.head.id)
    finally:
        host.close()
    with pytest.raises(GoalInputError, match="outside"):
        asyncio.run(execute_goal(altered, host_factory=factory))
    assert not transport.calls and not launcher.launch_calls


def test_source_change_is_rejected_before_opening_a_host(prepared):
    config, factory, transport, launcher, hosts = prepared
    count = len(hosts)
    with pytest.raises(GoalInputError, match="Python source"):
        asyncio.run(execute_goal(replace(config, python_source_sha256="0" * 64), host_factory=factory))
    assert len(hosts) == count and not transport.calls and not launcher.launch_calls


def test_settlement_keeps_unknown_provider_spending_and_never_plans(prepared):
    config, factory, transport, launcher, _hosts = prepared

    def lost_reply(*_args):
        raise PlannerTransportError("reply lost after dispatch")

    transport.responder = lost_reply
    host = factory(config.repo_root, config.session, config.goal)
    try:
        with pytest.raises(PlannerTransportError):
            host.planner.plan(config.goal)
        assert host.outcome() is None
    finally:
        host.close()
    outcome = asyncio.run(execute_goal(config, mode="settle", host_factory=factory))
    assert outcome.stop_reason == "cancelled" and not outcome.complete
    assert not outcome.budget.enforceable and outcome.budget.unknown_planner_attempt_ids
    assert len(transport.calls) == 1 and not launcher.launch_calls
    assert asyncio.run(execute_goal(config, host_factory=factory)) == outcome
    assert len(transport.calls) == 1


@pytest.mark.parametrize("mode", ["run", "settle"])
@pytest.mark.parametrize("whole_loop", [False, True])
def test_repeated_cancellation_waits_for_owned_operation_and_closes_afterward(prepared, mode, whole_loop):
    config, factory, transport, launcher, hosts = prepared
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    def delayed_response(request, prompt):
        entered.set()
        assert release.wait(10)
        return complete_response(request, prompt)

    transport.responder = delayed_response

    def delayed_factory(*args, **kwargs):
        host = factory(*args, **kwargs)
        original_close = host.close

        def close():
            original_close()
            closed.set()

        host.close = close
        if mode == "settle":
            original = host.stop_and_drain

            def stop(detail):
                entered.set()
                assert release.wait(10)
                return original(detail)

            host.stop_and_drain = stop
        return host

    async def scenario():
        task = asyncio.create_task(execute_goal(config, mode=mode, host_factory=delayed_factory))
        try:
            await wait_event(entered)
            for _ in range(3):
                cancel_loop_tasks() if whole_loop else task.cancel()
                await asyncio.sleep(0.02)
                # host.closed takes the lifecycle lock held by the driver;
                # querying it here would block this event loop, not observe it.
                assert not task.done() and not closed.is_set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert closed.is_set() and hosts[-1].closed
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert not launcher.launch_calls
    assert len(transport.calls) == (1 if mode == "run" else 0)
    # The process boundary also releases the externally opened Store's leases.
    host = factory(config.repo_root, config.session, config.goal)
    try:
        assert host.outcome().stop_reason == "cancelled"
    finally:
        host.close()


def test_driver_and_close_failures_are_both_retained(prepared):
    config, factory, *_ = prepared

    def broken_factory(*args, **kwargs):
        host = factory(*args, **kwargs)
        original_close = host.close

        async def fail(**_kwargs):
            raise RuntimeError("driver failed")

        def close():
            original_close()
            raise OSError("close failed")

        host.run_async = fail
        host.close = close
        return host

    with pytest.raises(BaseExceptionGroup) as caught:
        asyncio.run(execute_goal(config, host_factory=broken_factory))
    messages = [str(error) for error in _leaves(caught.value)]
    assert "driver failed" in messages and "close failed" in messages


@pytest.mark.parametrize("mutation", [
    {"max_generations": True}, {"wall_clock_seconds": True},
    {"wall_clock_seconds": float("inf")}, {"deadline_at": "2026-09-23T00:00:00"},
])
def test_malformed_admission_input_is_rejected(prepared, mutation):
    config, *_ = prepared
    with pytest.raises((ValueError, TypeError)):
        replace(config, run_limits_json=json.dumps(config.limits | mutation))


def test_input_rejects_duplicate_fields_extra_configuration_and_file_replacement(prepared, tmp_path):
    config, *_ = prepared
    raw = config.to_bytes()
    with pytest.raises(GoalInputError, match="duplicate"):
        GoalProcessInput.from_bytes(raw.replace(b'{"goal":', b'{"session":"replacement","goal":', 1))
    with pytest.raises(GoalInputError, match="fields"):
        GoalProcessInput.from_bytes(json.dumps(json.loads(raw) | {"environment": {}}).encode())
    path = tmp_path / "input.json"
    path.write_bytes(raw + b" ")
    with pytest.raises(GoalInputError, match="digest"):
        load_goal_input(path, hashlib.sha256(raw).hexdigest())
    path.write_bytes(b" " * (512 * 1024 + 1))
    with pytest.raises(GoalInputError, match="size"):
        load_goal_input(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_nonregular_input_files_do_not_follow_links_or_block(prepared, tmp_path):
    config, *_ = prepared
    target = tmp_path / "actual.json"
    target.write_bytes(config.to_bytes())
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(OSError):
        load_goal_input(link, hashlib.sha256(config.to_bytes()).hexdigest())
    fifo = tmp_path / "input.fifo"
    os.mkfifo(fifo)
    with pytest.raises(GoalInputError, match="regular"):
        load_goal_input(fifo, "0" * 64)


def test_real_expired_cli_ignores_task_imports_and_needs_no_provider_credentials(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = prepare_goal_process(
        repo, "expired-cli", simple_goal(budget_usd=5), max_generations=1,
        wall_clock_seconds=30, deadline_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    poisoned = "from pathlib import Path; Path('task-imported').write_text('bad'); raise RuntimeError('task import')\n"
    (repo / "sitecustomize.py").write_text(poisoned)
    (repo / "taste.py").write_text(poisoned)
    (repo / ".env").write_text("ANTHROPIC_BASE_URL=https://task-provider.invalid\n")
    path = tmp_path / "input.json"
    path.write_bytes(config.to_bytes())
    env = {key: value for key, value in os.environ.items()
           if key not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TASTE_LIVE_E2E"}}
    env["PYTHONPATH"] = str(repo)
    command = goal_command(path, hashlib.sha256(path.read_bytes()).hexdigest())
    result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 10, result.stdout + result.stderr
    assert not (repo / "task-imported").exists()
    host = compose_central_runtime(repo, config.session, config.goal)
    try:
        outcome = host.outcome()
        assert outcome.stop_reason == "wall_clock" and outcome.budget.enforceable
        assert not host.planner.planner_attempts(config.goal.goal_id) and not host.supervisor.runs()
        assert host.control.head.record(f"{_goal_root(config.goal.goal_id)}/run-limits.json") == config.limits
    finally:
        host.close()


def test_real_cli_rejects_changed_file_without_echoing_its_contents(prepared, tmp_path):
    config, *_ = prepared
    raw = config.to_bytes()
    path = tmp_path / "input.json"
    path.write_bytes(raw + b"do-not-echo-input")
    result = subprocess.run(goal_command(path, hashlib.sha256(raw).hexdigest()),
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 65
    assert result.stderr.strip() == "goal input rejected" and not result.stdout
