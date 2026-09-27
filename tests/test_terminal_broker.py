"""Terminal effects survive brain rollback, cancellation and controller death."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace

import pytest

from taste.brains.terminal_broker import (
    MAX_TERMINAL_OUTPUT_BYTES,
    TerminalBinding,
    TerminalBroker,
    TerminalConflict,
    TerminalFenced,
    TerminalRequest,
    TerminalResult,
)
from taste.memstore import Store
from tests.test_goal_cancellation import _leaves, wait_event


class Environment:
    environment_id = "container_instance_123"

    def __init__(self):
        self.calls, self.stop_calls = [], 0
        self.entered, self.release = threading.Event(), threading.Event()
        self.stop_entered, self.stop_release = threading.Event(), threading.Event()
        self.release.set()
        self.stop_release.set()
        self.error = self.stop_error = None
        self.receipt = self.environment_id
        self.result = TerminalResult(0, b"stdout\x00\xff", b"stderr\n")
        self.stopped = False

    def execute(self, request):
        if self.stopped:
            raise RuntimeError("environment already stopped")
        self.calls.append(request)
        self.entered.set()
        assert self.release.wait(5)
        if self.error:
            raise self.error
        return self.result

    def stop_and_confirm(self):
        self.stop_calls += 1
        self.stop_entered.set()
        assert self.stop_release.wait(5)
        self.stopped = True
        self.release.set()
        if self.stop_error:
            raise self.stop_error
        return self.receipt


def request(identifier="effect_1", **changes):
    return replace(TerminalRequest(identifier, "worker_A", "write persistent effect", "/task", 3), **changes)


def broker(tmp_path, environment=None, **limits):
    environment = environment or Environment()
    binding = TerminalBinding("trial_1", environment.environment_id, time.time() + 60, 100)
    binding = replace(binding, **limits)
    return TerminalBroker.create(tmp_path / "terminal", binding, environment)


@pytest.mark.parametrize("code", [0, 1, 124, 137])
def test_result_is_lossless_and_reopen_never_reexecutes_completed_id(tmp_path, code):
    env = Environment()
    env.result = TerminalResult(code, b"\xff\x00\n", b"failure is data\n")
    owner = broker(tmp_path, env)
    assert asyncio.run(owner.execute(request())) == env.result
    binding = owner.binding
    owner.close()
    recovered = TerminalBroker.open(tmp_path / "terminal", binding, env)
    try:
        assert asyncio.run(recovered.execute(request())) == env.result
        assert len(env.calls) == 1 and not env.stopped
        with pytest.raises(TerminalConflict):
            recovered.lookup(request(command="different effect"))
        assert [kind for kind, _ in recovered.events()] == ["created", "intent", "completed"]
    finally:
        recovered.close()


def test_brain_rollback_preserves_external_effect_and_terminal_receipt(tmp_path):
    owner = broker(tmp_path)
    store = Store.open(tmp_path / "memory", "trial_1")
    brain = store.branch("worker")
    before = brain.checkpoint("before package installation")
    try:
        result = asyncio.run(owner.execute(request()))
        brain.write("observed.txt", repr(result))
        after = brain.checkpoint("record terminal observation")
        brain.rollback(before, reason="reconsider approach")
        assert "observed.txt" not in brain.head.files()
        assert after.read("observed.txt") == repr(result)
        assert owner.lookup(request()) == result and len(owner.backend.calls) == 1
    finally:
        store.close()
        owner.close()


def test_only_one_controller_can_admit_for_the_environment(tmp_path):
    owner = broker(tmp_path)
    try:
        with pytest.raises(BlockingIOError):
            TerminalBroker.open(tmp_path / "terminal", owner.binding, owner.backend)
        assert asyncio.run(owner.execute(request())).return_code == 0
    finally:
        owner.close()


@pytest.mark.parametrize("failed", [False, True])
def test_command_completion_or_failure_does_not_wait_for_unused_interrupt(tmp_path, failed):
    owner = broker(tmp_path)
    if failed:
        owner.backend.error = ConnectionError("command reply failed immediately")
    started = time.monotonic()
    try:
        if failed:
            with pytest.raises(ConnectionError):
                asyncio.run(owner.execute(request(timeout_seconds=3)))
            assert owner.phase == "stopped"
        else:
            assert asyncio.run(owner.execute(request(timeout_seconds=3))) == owner.backend.result
            assert not owner.backend.stopped
        assert time.monotonic() - started < 1.5, "broker waited for a signal that was never needed"
    finally:
        owner.close()


def test_serial_commands_and_cancelled_queue_cannot_overlap_effects(tmp_path):
    owner = broker(tmp_path)
    env = owner.backend
    env.release.clear()

    async def scenario():
        first = asyncio.create_task(owner.execute(request()))
        await wait_event(env.entered)
        queued = asyncio.create_task(owner.execute(request("queued")))
        following = asyncio.create_task(owner.execute(request("following")))
        await asyncio.sleep(0.03)
        assert len(env.calls) == 1
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        env.release.set()
        await asyncio.gather(first, following)
        assert [call.request_id for call in env.calls] == ["effect_1", "following"]
        assert owner.lookup(request("queued")) is None
        assert not env.stopped

    try:
        asyncio.run(scenario())
    finally:
        env.release.set()
        owner.close()


@pytest.mark.parametrize("ending", ["cancel", "abort", "timeout"])
def test_active_ending_drains_before_return_and_retains_late_output(tmp_path, ending):
    owner = broker(tmp_path)
    env = owner.backend
    env.release.clear()
    env.stop_release.clear()
    req = request(timeout_seconds=0.05 if ending == "timeout" else 3)

    async def scenario():
        errors = []
        asyncio.get_running_loop().set_exception_handler(lambda _loop, context: errors.append(context))
        active = asyncio.create_task(owner.execute(req))
        await wait_event(env.entered)
        stopper = None
        if ending == "cancel":
            active.cancel()
        elif ending == "abort":
            stopper = asyncio.create_task(owner.abort())
        await wait_event(env.stop_entered)
        for _ in range(3):
            if ending == "cancel":
                active.cancel()
            elif stopper:
                stopper.cancel()
            await asyncio.sleep(0.01)
            assert not active.done()
            with pytest.raises(TerminalFenced):
                owner.close()
        env.stop_release.set()
        with pytest.raises(TimeoutError if ending == "timeout" else asyncio.CancelledError):
            await active
        if stopper:
            with pytest.raises(asyncio.CancelledError):
                await stopper
        assert owner.phase == "stopped" and env.stopped
        assert len(env.calls) == env.stop_calls == 1
        assert ("late_result", {"request_id": req.request_id}) in owner.events()
        with pytest.raises(TerminalFenced):
            owner.lookup(req)
        with pytest.raises(TerminalFenced):
            await owner.execute(request("cannot_retry"))
        await asyncio.sleep(0)
        assert not errors

    try:
        asyncio.run(scenario())
    finally:
        env.release.set()
        env.stop_release.set()
        owner.close()


def test_cancel_between_durable_intent_and_async_owner_start_still_stops(tmp_path, monkeypatch):
    owner = broker(tmp_path)
    original = owner._event

    def cancel_after_intent(kind, payload):
        original(kind, payload)
        if kind == "intent":
            asyncio.current_task().cancel()

    monkeypatch.setattr(owner, "_event", cancel_after_intent)
    try:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(owner.execute(request()))
        assert owner.phase == "stopped" and owner.backend.stop_calls == 1
        assert not owner.backend.calls
    finally:
        owner.close()


def test_exec_and_stop_failures_are_retained_and_close_is_fenced(tmp_path):
    owner = broker(tmp_path)
    owner.backend.error = ConnectionError("exec reply lost")
    owner.backend.stop_error = OSError("stop reply lost")

    async def scenario():
        with pytest.raises(BaseExceptionGroup) as caught:
            await owner.execute(request())
        assert {str(error) for error in _leaves(caught.value)} == {"exec reply lost", "stop reply lost"}
        assert owner.phase == "fenced"
        with pytest.raises(TerminalFenced):
            owner.close()
        with pytest.raises(TerminalFenced):
            await owner.execute(request("new"))
        owner.backend.stop_error = None
        await owner.abort()
        assert owner.phase == "stopped"

    asyncio.run(scenario())
    owner.close()


def test_wrong_environment_stop_receipt_cannot_release_lease(tmp_path):
    owner = broker(tmp_path)
    owner.backend.receipt = "another_container"

    async def scenario():
        with pytest.raises(TerminalConflict):
            await owner.abort()
        with pytest.raises(TerminalFenced):
            owner.close()
        owner.backend.receipt = owner.binding.environment_id
        await owner.abort()

    asyncio.run(scenario())
    owner.close()


def test_failed_result_publication_stops_and_preserves_late_observation(tmp_path, monkeypatch):
    owner = broker(tmp_path)
    original = owner._record_result

    def fail_completed(req, result, *, completed):
        if completed:
            raise OSError("receipt persistence failed")
        return original(req, result, completed=False)

    monkeypatch.setattr(owner, "_record_result", fail_completed)
    try:
        with pytest.raises(OSError, match="receipt persistence"):
            asyncio.run(owner.execute(request()))
        assert owner.phase == "stopped" and owner.backend.stop_calls == 1
        assert ("late_result", {"request_id": "effect_1"}) in owner.events()
        with pytest.raises(TerminalFenced):
            owner.lookup(request())
    finally:
        owner.close()


def test_failed_abort_intent_still_attempts_environment_stop(tmp_path, monkeypatch):
    owner = broker(tmp_path)

    def fail_fence(_reason):
        raise OSError("fence persistence failed")

    monkeypatch.setattr(owner, "_fence", fail_fence)
    try:
        with pytest.raises(OSError, match="fence persistence"):
            asyncio.run(owner.abort())
        assert owner.phase == "stopped" and owner.backend.stop_calls == 1
    finally:
        owner.close()


def test_explicit_abort_ends_admission_for_queued_commands(tmp_path):
    owner = broker(tmp_path)
    owner.backend.release.clear()

    async def scenario():
        active = asyncio.create_task(owner.execute(request()))
        await wait_event(owner.backend.entered)
        queued = asyncio.create_task(owner.execute(request("queued")))
        aborting = asyncio.create_task(owner.abort())
        with pytest.raises(asyncio.CancelledError):
            await active
        with pytest.raises(TerminalFenced):
            await queued
        await aborting
        assert len(owner.backend.calls) == owner.backend.stop_calls == 1

    asyncio.run(scenario())
    owner.close()


@pytest.mark.parametrize("boundary", ["expired", "count", "queued_deadline"])
def test_limits_are_checked_before_effect_including_after_waiting(tmp_path, boundary, monkeypatch):
    owner = broker(tmp_path, max_commands=1 if boundary == "count" else 100)
    clock = [owner.binding.deadline_unix - 1]
    monkeypatch.setattr("taste.brains.terminal_broker.time.time", lambda: clock[0])

    async def scenario():
        if boundary == "expired":
            clock[0] += 2
        elif boundary == "count":
            await owner.execute(request())
        else:
            owner.backend.release.clear()
            active = asyncio.create_task(owner.execute(request()))
            await wait_event(owner.backend.entered)
            following = asyncio.create_task(owner.execute(request("following")))
            await asyncio.sleep(0)
            clock[0] += 2
            owner.backend.release.set()
            await active
            with pytest.raises(TerminalFenced, match="deadline"):
                await following
            return
        with pytest.raises(TerminalFenced):
            await owner.execute(request("refused"))

    try:
        asyncio.run(scenario())
        assert len(owner.backend.calls) == (0 if boundary == "expired" else 1)
    finally:
        owner.close()


def test_process_death_after_effect_never_allows_replay(tmp_path):
    directory = tmp_path / "terminal"
    marker = tmp_path / "external_effect"
    deadline = time.time() + 60
    script = r'''
import asyncio, os, sys
from pathlib import Path
from taste.brains.terminal_broker import TerminalBinding, TerminalBroker, TerminalRequest
class Environment:
    environment_id = "container_instance_123"
    def execute(self, request):
        with open(sys.argv[2], "ab", buffering=0) as f:
            f.write(b"one external effect\n")
            os.fsync(f.fileno())
        os._exit(73)
owner = TerminalBroker.create(Path(sys.argv[1]), TerminalBinding("trial_1", Environment.environment_id, float(sys.argv[3]), 100), Environment())
asyncio.run(owner.execute(TerminalRequest("effect_1", "worker_A", "write persistent effect", "/task", 3)))
'''
    child = subprocess.run([sys.executable, "-c", script, str(directory), str(marker), str(deadline)],
                           capture_output=True, timeout=10, check=False)
    assert child.returncode == 73, child.stderr.decode()
    env = Environment()
    owner = TerminalBroker.open(directory, TerminalBinding("trial_1", env.environment_id, deadline, 100), env)

    async def scenario():
        assert owner.phase == "fenced"
        with pytest.raises(TerminalFenced):
            await owner.execute(request())
        with pytest.raises(TerminalFenced):
            await owner.execute(request("new_id"))
        await owner.abort()

    asyncio.run(scenario())
    owner.close()
    assert marker.read_bytes() == b"one external effect\n" and not env.calls
    # A stopped recovered ledger remains stopped when opened yet again.
    again = TerminalBroker.open(directory, owner.binding, env)
    try:
        assert again.phase == "stopped"
    finally:
        again.close()


@pytest.mark.parametrize("changes", [{"timeout_seconds": True}, {"timeout_seconds": float("nan")},
                                    {"cwd": "../host"}, {"command": "bad\x00command"},
                                    {"request_id": "../../escape"}])
def test_invalid_requests_cannot_reach_the_transport(changes):
    with pytest.raises(ValueError):
        request(**changes)


def test_reopen_requires_exact_binding_and_private_database(tmp_path):
    owner = broker(tmp_path)
    binding, env = owner.binding, owner.backend
    owner.close()
    with pytest.raises(TerminalConflict):
        TerminalBroker.open(tmp_path / "terminal", replace(binding, max_commands=2), env)
    database = tmp_path / "terminal" / "terminal.sqlite3"
    database.chmod(0o644)
    with pytest.raises(TerminalFenced, match="private regular file"):
        TerminalBroker.open(tmp_path / "terminal", binding, env)
    database.chmod(0o600)
    database.rename(database.with_suffix(".saved"))
    database.symlink_to(database.with_suffix(".saved"))
    with pytest.raises(TerminalFenced, match="private regular file"):
        TerminalBroker.open(tmp_path / "terminal", binding, env)


def test_truncated_binary_receipt_survives_stop_and_reopen_without_reexecution(tmp_path):
    env = Environment()
    env.result = TerminalResult(7, b"\xff\x00" * 1024, b"\x80", 3_000_000, 123)
    owner = broker(tmp_path, env)
    binding = owner.binding

    async def scenario():
        assert await owner.execute(request()) == env.result
        await owner.abort()

    asyncio.run(scenario())
    owner.close()
    reopened = TerminalBroker.open(tmp_path / "terminal", binding, env)
    try:
        assert asyncio.run(reopened.execute(request())) == env.result
        assert len(env.calls) == 1 and env.stopped
    finally:
        reopened.close()


@pytest.mark.parametrize("field", ["stdout_dropped_bytes", "stderr_dropped_bytes"])
@pytest.mark.parametrize("value", [True, -1, 1.5, 2**63, None])
def test_invalid_truncation_cannot_be_mistaken_for_a_complete_receipt(field, value):
    with pytest.raises(ValueError, match="dropped byte count"):
        TerminalResult(0, **{field: value})


@pytest.mark.parametrize("field", ["stdout", "stderr"])
def test_receipts_enforce_stream_limit_before_persistence(field):
    TerminalResult(0, **{field: b"x" * MAX_TERMINAL_OUTPUT_BYTES})
    with pytest.raises(ValueError, match="receipt limit"):
        TerminalResult(0, **{field: b"x" * (MAX_TERMINAL_OUTPUT_BYTES + 1)})


@pytest.mark.parametrize("status", ["completed", "pending"])
def test_legacy_database_upgrade_preserves_receipt_or_fences_incomplete_effect(tmp_path, status):
    env = Environment()
    binding = TerminalBinding("trial_1", env.environment_id, time.time() + 60, 100)
    directory = tmp_path / "terminal"
    directory.mkdir(mode=0o700)
    database = directory / "terminal.sqlite3"
    def encode(value):
        return json.dumps(asdict(value), sort_keys=True, separators=(",", ":"))

    # Original on-disk format, independent of the current broker constructor.
    with sqlite3.connect(database) as db:
        db.executescript("""
            CREATE TABLE binding (payload TEXT NOT NULL);
            CREATE TABLE state (phase TEXT NOT NULL);
            CREATE TABLE events (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE requests (id TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL,
                                   code INTEGER, stdout BLOB, stderr BLOB);
            INSERT INTO state VALUES ('ready');
        """)
        db.execute("INSERT INTO binding VALUES (?)", (encode(binding),))
        db.execute("INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?)",
                   (request().request_id, encode(request()), status, 1, b"\xff", b"original error"))
    database.chmod(0o600)
    owner = TerminalBroker.open(directory, binding, env)

    async def scenario():
        if status == "completed":
            assert await owner.execute(request()) == TerminalResult(1, b"\xff", b"original error")
        else:
            assert owner.phase == "fenced"
            with pytest.raises(TerminalFenced):
                await owner.execute(request())
        await owner.abort()

    try:
        asyncio.run(scenario())
        assert not env.calls and env.stopped
        with sqlite3.connect(database) as db:
            assert db.execute("PRAGMA user_version").fetchone() == (1,)
    finally:
        owner.close()


def test_unknown_receipt_version_refuses_open_and_releases_controller_lease(tmp_path):
    owner = broker(tmp_path)
    binding, env = owner.binding, owner.backend
    owner.close()
    database = tmp_path / "terminal" / "terminal.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("PRAGMA user_version=2")
    with pytest.raises(TerminalFenced, match="unsupported terminal receipt schema"):
        TerminalBroker.open(tmp_path / "terminal", binding, env)
    with sqlite3.connect(database) as db:
        db.execute("PRAGMA user_version=1")
    recovered = TerminalBroker.open(tmp_path / "terminal", binding, env)
    recovered.close()
