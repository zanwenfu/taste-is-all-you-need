"""Whole-loop cancellation must not discard live threads or their receipts."""

from __future__ import annotations

import asyncio
import threading

import pytest

from taste.brains.contract import Contract
from taste.brains.monitor import Judgement, MonitorBrain, Severity
from taste.brains.terminal_broker import TerminalFenced
from taste.memstore import Store
from tests.test_goal_cancellation import _leaves, bounds, make_host, run_async, wait_event
from tests.test_terminal_broker import Environment, broker, request

__all__ = ["make_host"]


def cancel_loop_tasks():
    # asyncio.run() uses this same snapshot-and-cancel operation at shutdown.
    # Keep only this test observer alive so it can inspect the ownership gap.
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            task.cancel()


def test_monitor_loop_shutdown_keeps_judge_and_persistence_owned(tmp_path):
    entered, release = threading.Event(), threading.Event()
    store = Store.open(tmp_path / "repo", "shutdown")
    contract = Contract("worker", "produce evidence", success_criteria=("correct",))
    branch = store.branch("worker")
    branch.turn(kind="tool_result", tool_use_id="one", summary="evidence")

    def judge(*_args):
        entered.set()
        assert release.wait(10)
        return Judgement(Severity.WRONG, "late persisted finding", cost_usd=0.01)

    monitor = MonitorBrain(store, contract, judge, batch_size=1)

    async def scenario():
        caller = asyncio.create_task(monitor.cycle(None))
        try:
            await wait_event(entered)
            cancel_loop_tasks()
            for _ in range(3):
                await asyncio.sleep(0.01)
                assert not caller.done(), "monitor abandoned the live judge thread"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await caller
            recovered = MonitorBrain(store, contract, judge, batch_size=1)
            assert recovered.report()["model_calls"] == 1
            assert recovered.report()["cost_usd"] == pytest.approx(0.01)
            assert recovered.report()["worst"] == "wrong"
        finally:
            release.set()
            await asyncio.gather(caller, return_exceptions=True)

    try:
        asyncio.run(scenario())
    finally:
        store.close()


@pytest.mark.parametrize("stage", ["execute", "stop"])
def test_terminal_loop_shutdown_waits_for_exec_and_stop_receipts(tmp_path, stage):
    env = Environment()
    env.release.clear()
    env.stop_release.clear()
    owner = broker(tmp_path, env)

    async def scenario():
        caller = asyncio.create_task(owner.execute(request()))
        try:
            await wait_event(env.entered)
            if stage == "stop":
                caller.cancel()
                await wait_event(env.stop_entered)
            cancel_loop_tasks()
            await wait_event(env.stop_entered)
            for _ in range(3):
                await asyncio.sleep(0.01)
                assert not caller.done(), "terminal owner abandoned a live transport"
                with pytest.raises(TerminalFenced):
                    owner.close()
            env.stop_release.set()
            with pytest.raises(BaseException) as error:
                await caller
            assert any(isinstance(item, asyncio.CancelledError) for item in _leaves(error.value))
            assert owner.phase == "stopped"
            assert ("late_result", {"request_id": "effect_1"}) in owner.events()
            assert len(env.calls) == env.stop_calls == 1
        finally:
            env.release.set()
            env.stop_release.set()
            await asyncio.gather(caller, return_exceptions=True)
            await owner.abort()

    try:
        asyncio.run(scenario())
    finally:
        owner.close()


def test_goal_loop_shutdown_preserves_late_driver_failure(make_host):
    entered, release = threading.Event(), threading.Event()
    host = make_host()

    def boundary():
        entered.set()
        assert release.wait(10)
        raise OSError("late driver failure must survive shutdown")

    async def scenario():
        caller = asyncio.create_task(run_async(host, **bounds(), between_cycles=boundary))
        try:
            await wait_event(entered)
            cancel_loop_tasks()
            await asyncio.sleep(0.02)
            assert not caller.done()
            release.set()
            with pytest.raises(BaseException) as error:
                await caller
            errors = _leaves(error.value)
            assert any(isinstance(item, OSError) and "late driver failure" in str(item)
                       for item in errors), "the cancelled driver Task lost the thread's late failure"
            assert all(run.terminal for run in host.supervisor.runs())
        finally:
            release.set()
            await asyncio.gather(caller, return_exceptions=True)

    asyncio.run(scenario())


def test_cancellation_before_terminal_owner_starts_still_stops_environment(tmp_path, monkeypatch):
    owner = broker(tmp_path)
    original = owner._event

    async def scenario():
        observer = asyncio.current_task()

        def cancel_new_tasks():
            for task in asyncio.all_tasks():
                if task is not observer:
                    task.cancel()

        def at_intent(kind, payload):
            original(kind, payload)
            if kind == "intent":
                # Queue shutdown before create_task schedules the effect owner.
                asyncio.get_running_loop().call_soon(cancel_new_tasks)

        monkeypatch.setattr(owner, "_event", at_intent)
        caller = asyncio.create_task(owner.execute(request()))
        try:
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert owner.phase == "stopped", "pre-start cancellation skipped environment cleanup"
            assert owner.backend.stop_calls == 1
            assert owner.backend.calls == []
        finally:
            await owner.abort()

    try:
        asyncio.run(scenario())
    finally:
        owner.close()


def test_cancellation_before_idle_abort_owner_starts_still_drains(tmp_path, monkeypatch):
    owner = broker(tmp_path)
    original = owner._abort_owned

    async def scenario():
        observer = asyncio.current_task()

        def cancel_new_tasks():
            for task in asyncio.all_tasks():
                if task is not observer:
                    task.cancel()

        first = True

        def abort_coroutine():
            nonlocal first
            if first:
                first = False
                asyncio.get_running_loop().call_soon(cancel_new_tasks)
            return original()

        monkeypatch.setattr(owner, "_abort_owned", abort_coroutine)
        caller = asyncio.create_task(owner.abort())
        try:
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert owner.phase == "stopped", "cancelled abort never reached environment stop"
            assert owner.backend.stop_calls == 1
        finally:
            await owner.abort()

    try:
        asyncio.run(scenario())
    finally:
        owner.close()
