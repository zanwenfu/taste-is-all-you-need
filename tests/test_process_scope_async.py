"""An async scope owner must retain launch and cleanup across cancellation."""

from __future__ import annotations

import asyncio
import threading

import pytest

from taste.brains.process_scope import OwnedProcessScope, ScopeSpec
from taste.resources import ResourceCleanupError, resource_failures
from tests.test_goal_cancellation import _leaves, wait_event
from tests.test_process_scope import Manager, state


@pytest.fixture
def scope(tmp_path):
    return OwnedProcessScope.create(
        tmp_path / "owner", ScopeSpec(("/usr/bin/python3", "-V"), str(tmp_path), 1000, 6, 0.5),
        manager=Manager(),
    )


def test_normal_async_execution_returns_only_after_drain_and_release(scope):
    start = scope.manager.start

    def complete(*args):
        start(*args)
        scope.manager.units[scope.unit]["ActiveState"] = "inactive"
        scope.manager.populated.clear()

    scope.manager.start = complete
    receipt = asyncio.run(scope.run_async(timeout_seconds=10))
    assert receipt["processes_stopped"] and receipt["goal_settlement_required"]
    assert not scope.manager.units and not scope.manager.populated
    assert len(scope.manager.starts) == len(scope.manager.stops) == len(scope.manager.releases) == 1


@pytest.mark.parametrize("boundary", ["launch", "stop"])
def test_repeated_cancellation_waits_for_inflight_manager_operation(scope, boundary):
    entered, release, operation_done = threading.Event(), threading.Event(), threading.Event()

    def delayed():
        entered.set()
        try:
            assert release.wait(10)
        finally:
            operation_done.set()

    if boundary == "launch":
        scope.manager.before_start = delayed
    else:
        scope.manager.before_stop = delayed
        original = scope.manager.start

        def start(*args):
            original(*args)
            scope.manager.units[scope.unit]["ActiveState"] = "inactive"

        scope.manager.start = start

    async def scenario():
        loop_errors = []
        asyncio.get_running_loop().set_exception_handler(lambda _loop, error: loop_errors.append(error))
        task = asyncio.create_task(scope.run_async(timeout_seconds=600))
        try:
            await wait_event(entered)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0.02)
                assert not task.done() and not operation_done.is_set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0)
            assert not loop_errors, "owned cancellation leaked an event-loop error"
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert operation_done.is_set() and state(scope)["phase"] == "stopped"
    assert not scope.manager.units and not scope.manager.populated
    assert len(scope.manager.starts) == 1


def test_failed_cleanup_does_not_leave_a_background_polling_thread(scope):
    entered, poll_done = threading.Event(), threading.Event()
    original_start, original_wait = scope.manager.start, scope.wait
    inspections = []
    original_inspect = scope.manager.inspect

    def start(*args):
        original_start(*args)
        entered.set()

    def wait(**kwargs):
        try:
            return original_wait(**kwargs)
        finally:
            poll_done.set()

    def inspect(unit):
        inspections.append(unit)
        return original_inspect(unit)

    scope.manager.start, scope.manager.inspect, scope.wait = start, inspect, wait
    scope.manager.stop_error = OSError("manager cannot confirm stop")

    async def scenario():
        loop_errors = []
        asyncio.get_running_loop().set_exception_handler(lambda _loop, error: loop_errors.append(error))
        task = asyncio.create_task(scope.run_async(timeout_seconds=600))
        await wait_event(entered)
        task.cancel()
        with pytest.raises(BaseExceptionGroup) as caught:
            await asyncio.wait_for(task, timeout=2)
        assert any(isinstance(error, asyncio.CancelledError) for error in _leaves(caught.value))
        assert resource_failures(caught.value)[0].operation == "stop"
        assert poll_done.is_set()
        count = len(inspections)
        await asyncio.sleep(0.1)
        assert len(inspections) == count, "abandoned scope poller kept running after return"
        assert not loop_errors, "owned failures escaped through the event-loop handler"

    try:
        asyncio.run(scenario())
        assert state(scope)["phase"] == "stop_pending" and scope.manager.populated
    finally:
        scope.manager.stop_error = None
        scope.stop("test recovery")


def test_lost_start_reply_is_drained_but_original_failure_is_preserved(scope):
    scope.manager.start_error = ConnectionError("launch acknowledgement lost")
    with pytest.raises(ResourceCleanupError, match="acknowledgement lost"):
        asyncio.run(scope.run_async(timeout_seconds=10))
    assert state(scope)["phase"] == "stopped" and not scope.manager.units
    assert len(scope.manager.starts) == 1


def test_launch_and_cleanup_errors_both_reach_the_owner(scope):
    scope.manager.start_error = ConnectionError("launch reply lost")
    scope.manager.stop_error = OSError("stop reply lost")
    try:
        with pytest.raises(BaseExceptionGroup) as caught:
            asyncio.run(scope.run_async(timeout_seconds=10))
        assert {failure.operation for failure in resource_failures(caught.value)} == {"start", "stop"}
        assert state(scope)["phase"] == "stop_pending" and len(scope.manager.starts) == 1
    finally:
        scope.manager.stop_error = None
        scope.stop("test recovery")


@pytest.mark.parametrize("failure", [KeyboardInterrupt, SystemExit])
def test_thread_interrupt_cannot_halt_the_loop_before_cleanup(scope, failure):
    original = scope.start

    def interrupted():
        original()
        raise failure("controller interrupted")

    scope.start = interrupted

    async def scenario():
        with pytest.raises(BaseExceptionGroup) as caught:
            await scope.run_async(timeout_seconds=10)
        assert any(isinstance(error, failure) for error in _leaves(caught.value))
        await asyncio.sleep(0)
        assert state(scope)["phase"] == "stopped"

    asyncio.run(scenario())
    assert not scope.manager.units and not scope.manager.populated


def test_async_recovery_never_launches_and_waits_through_cancellation(scope):
    scope.start()
    entered, release = threading.Event(), threading.Event()

    def delayed():
        entered.set()
        assert release.wait(10)

    scope.manager.before_stop = delayed

    async def scenario():
        task = asyncio.create_task(scope.stop_async("recovery after controller restart"))
        try:
            await wait_event(entered)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0.02)
                assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert len(scope.manager.starts) == 1 and not scope.manager.units
    assert state(scope)["stop_reason"] == "recovery after controller restart"


@pytest.mark.parametrize("invalid", [True, 0, float("inf")])
def test_invalid_async_wait_bound_makes_no_launch(scope, invalid):
    with pytest.raises(ValueError):
        asyncio.run(scope.run_async(timeout_seconds=invalid))
    assert not scope.manager.starts and state(scope)["phase"] == "ready"
