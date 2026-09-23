"""Cancellation and SDK failure cannot release a still-owned worker boundary."""

from __future__ import annotations

import asyncio
import threading

import pytest
from claude_agent_sdk import MirrorErrorMessage

from taste.brains.monitor import Judgement, MonitorBrain, Severity
from taste.brains.subbrain import SubBrain
from taste.brains.worker_entrypoint import (
    _QUARANTINED_RESOURCES,
    WorkerExitCode,
    execute_worker,
)
from taste.brains.worker_runtime import WORKER_REPORT_PATH, ShutdownUnconfirmed
from taste.memstore import BranchBusy, Store
from tests.test_brains_monitor import FakeClient, ScriptedTerminalJudge, a_contract
from tests.test_brains_worker_entrypoint import (
    FakeMonitorLLM,
    _assignment,
    _client_factory,
    _config,
    _environment,
)
from tests.test_brains_worker_runtime import (
    QuietMonitor,
    ScriptedClient,
    result_message,
    runtime_for,
    scaffold,
)
from tests.test_goal_cancellation import _leaves, wait_event


@pytest.mark.parametrize("boundary", ["judge", "save", "save_failure"])
def test_terminal_certification_retains_thread_and_persistence_failure(tmp_path, boundary):
    store = Store.open(tmp_path / "repo", "monitor-ownership")
    brain = SubBrain(store, a_contract())
    brain.install_contract()
    work = brain.checkpoint("exact terminal work")
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    judge = ScriptedTerminalJudge()
    monitor = MonitorBrain(store, brain.contract, judge)
    original_sync = monitor._certify_terminal_sync
    original_save = monitor._save_state
    original_judge = judge.judge_terminal

    def blocked(call):
        entered.set()
        assert release.wait(10)
        return call()

    def save():
        if boundary == "save_failure":
            raise OSError("test-only assessment persistence failed")
        return original_save()

    if boundary == "judge":
        judge.judge_terminal = lambda *args: blocked(lambda: original_judge(*args))
    else:
        monitor._save_state = lambda: blocked(save)

    def sync(*args):
        try:
            return original_sync(*args)
        finally:
            finished.set()

    monitor._certify_terminal_sync = sync

    async def scenario():
        loop_errors = []
        asyncio.get_running_loop().set_exception_handler(lambda _loop, context: loop_errors.append(context))
        task = asyncio.create_task(monitor.certify_terminal(work, context={"tests_passed": True}))
        try:
            await wait_event(entered)
            # One cancellation on the failure path isolates the old loss of
            # the persistence error from the separate repeated-cancel race.
            for _ in range(1 if boundary == "save_failure" else 3):
                task.cancel()
                await asyncio.sleep(0.02)
                assert not task.done() and not finished.is_set()
            release.set()
            if boundary == "save_failure":
                with pytest.raises(BaseExceptionGroup) as caught:
                    await task
                leaves = _leaves(caught.value)
                assert any(isinstance(error, asyncio.CancelledError) for error in leaves)
                assert any(isinstance(error, OSError) for error in leaves)
                assert not monitor.state.terminal_assessments
            else:
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert finished.is_set()
                restarted_judge = ScriptedTerminalJudge()
                restarted = MonitorBrain(store, brain.contract, restarted_judge)
                assessment = await restarted.certify_terminal(work, context={"tests_passed": True})
                assert assessment.acceptable and not restarted_judge.terminal_calls
            await asyncio.sleep(0)
            assert not loop_errors
        finally:
            release.set()
            await wait_event(finished)
            await asyncio.gather(task, return_exceptions=True)

    try:
        asyncio.run(scenario())
    finally:
        brain.close()
        store.close()


def test_cancelled_incremental_judgement_persists_once_before_releasing_its_owner(tmp_path):
    store = Store.open(tmp_path / "repo", "incremental-ownership")
    brain = SubBrain(store, a_contract())
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def judge(*_args):
        calls.append(True)
        entered.set()
        assert release.wait(10)
        return Judgement(Severity.FINE, "test-only incremental assessment", cost_usd=0)

    brain.wal.intent("Read", "observed-read", {"file_path": "parser.py"})
    monitor = MonitorBrain(store, brain.contract, judge, batch_size=1)
    original_tick = monitor.tick

    def tick(**kwargs):
        try:
            return original_tick(**kwargs)
        finally:
            finished.set()

    monitor.tick = tick

    async def scenario():
        task = asyncio.create_task(monitor.cycle(FakeClient()))
        try:
            await wait_event(entered)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0.02)
                assert not task.done() and not finished.is_set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(calls) == 1 and finished.is_set()
            restarted = MonitorBrain(store, brain.contract, judge, batch_size=1)
            assert len(restarted.state.judgements) == 1 and restarted.pending_actions
            await restarted.cycle(FakeClient())
            assert len(calls) == 1 and not restarted.pending_actions
        finally:
            release.set()
            await wait_event(finished)
            await asyncio.gather(task, return_exceptions=True)

    try:
        asyncio.run(scenario())
    finally:
        brain.close()
        store.close()


@pytest.mark.parametrize("cancel", [False, True])
def test_runtime_disconnect_failure_is_typed_and_keeps_the_writer_lease(tmp_path, cancel):
    store = Store.open(tmp_path / "repo", "disconnect-ownership")
    contract = scaffold(store)

    class FailedDisconnect(ScriptedClient):
        async def disconnect(self):
            entered.set()
            await release.wait()
            self.disconnected.set()
            raise OSError("test-only SDK reaping is unconfirmed")

    async def scenario():
        nonlocal entered, release
        entered, release = asyncio.Event(), asyncio.Event()
        client = FailedDisconnect(None)
        runtime = runtime_for(store, contract, client)
        task = asyncio.create_task(runtime.run())
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            if cancel:
                for _ in range(3):
                    task.cancel()
                    await asyncio.sleep(0.02)
                    assert not task.done()
            release.set()
            with pytest.raises(ShutdownUnconfirmed):
                await task
            assert not runtime._shutdown_confirmed
            assert store.view(contract.identity).head.read(WORKER_REPORT_PATH) is None
            other = Store.open(store.root, store.session)
            try:
                with pytest.raises(BranchBusy):
                    other.branch(contract.identity)
            finally:
                other.close()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            runtime.brain.close()  # Fixture has no real SDK child.

    entered = release = None
    try:
        asyncio.run(scenario())
    finally:
        store.close()


def test_entrypoint_quarantines_disconnect_failure_instead_of_releasing_the_branch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    store = Store.open(repo, "session-1")

    class FailedDisconnect(ScriptedClient):
        async def disconnect(self):
            self.disconnected.set()
            raise OSError("test-only SDK reap failure")

    retained_before = len(_QUARANTINED_RESOURCES)
    try:
        assignment, prepared = _assignment(store)
        code = asyncio.run(execute_worker(
            _config(repo, prepared), store=store, environ=_environment(store, assignment),
            llm_factory=FakeMonitorLLM, client_factory=_client_factory(FailedDisconnect(None)),
            ready_callback=lambda: None,
        ))
        assert code is WorkerExitCode.SHUTDOWN_UNCONFIRMED
        assert len(_QUARANTINED_RESOURCES) == retained_before + 1
        assert store.view(assignment.worker).head.read(WORKER_REPORT_PATH) is None
        other = Store.open(repo, store.session)
        try:
            with pytest.raises(BranchBusy):
                other.branch(assignment.worker)
        finally:
            other.close()
    finally:
        for _opened, brain in _QUARANTINED_RESOURCES[retained_before:]:
            if brain is not None:
                brain.close()
        del _QUARANTINED_RESOURCES[retained_before:]
        store.close()


def test_cancellation_after_disconnect_waits_for_reader_and_persists_final_tail(tmp_path):
    store = Store.open(tmp_path / "repo", "tail-ownership")
    contract = scaffold(store)

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        class DelayedTail(ScriptedClient):
            async def receive_messages(self):
                await self.queried.wait()
                yield result_message()
                await self.disconnected.wait()
                entered.set()
                await release.wait()
                yield MirrorErrorMessage(subtype="mirror_error", data={"session_id": "session-1"},
                                         error="test-only final mirror flush failure")

        client = DelayedTail(None)
        runtime = runtime_for(store, contract, client)
        runtime.shutdown_timeout = 10
        task = asyncio.create_task(runtime.run())
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0.02)
                assert not task.done(), "branch owner returned while its final reader was active"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert runtime._reader_task.done() and runtime._shutdown_confirmed
            assert any(turn.get("message_type") == "MirrorErrorMessage"
                       for turn in store.view(contract.identity).pending_turns())
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            if runtime._reader_task is not None:
                await asyncio.gather(runtime._reader_task, return_exceptions=True)
            runtime.brain.close()

    try:
        asyncio.run(scenario())
    finally:
        store.close()


def test_background_stop_failure_still_attempts_sdk_cleanup_and_keeps_quarantine(tmp_path):
    store = Store.open(tmp_path / "repo", "background-ownership")
    contract = scaffold(store)
    client = ScriptedClient(None)
    runtime = runtime_for(store, contract, client)

    async def fail_stop():
        raise OSError("test-only producer shutdown failed")

    runtime._stop_background_loops = fail_stop
    try:
        with pytest.raises(ShutdownUnconfirmed):
            asyncio.run(runtime.run())
        assert ("disconnect", None) in client.calls
        assert not runtime._shutdown_confirmed
        other = Store.open(store.root, store.session)
        try:
            with pytest.raises(BranchBusy):
                other.branch(contract.identity)
        finally:
            other.close()
    finally:
        runtime.brain.close()
        store.close()


def test_self_cancelled_monitor_cannot_be_treated_as_successful_shutdown(tmp_path):
    store = Store.open(tmp_path / "repo", "cancelled-monitor")
    contract = scaffold(store)

    class CancelledMonitor(QuietMonitor):
        async def cycle(self, *_args, **_kwargs):
            raise asyncio.CancelledError

    client = ScriptedClient(None)
    runtime = runtime_for(store, contract, client, monitor=CancelledMonitor())
    try:
        with pytest.raises(ShutdownUnconfirmed):
            asyncio.run(runtime.run())
        assert ("disconnect", None) in client.calls and not runtime._shutdown_confirmed
        assert store.view(contract.identity).head.read(WORKER_REPORT_PATH) is None
    finally:
        runtime.brain.close()
        store.close()
