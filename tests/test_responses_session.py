"""Real SDK calls through a mock transport, durable storage, and process death."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from taste.brains.responses_session import (
    ResponsesBinding,
    ResponsesConflict,
    ResponsesFenced,
    ResponsesSession,
)
from taste.llm import BudgetExceeded, InfraFailure
from taste.memstore import Store
from taste.pricing import max_call_cost_usd
from taste.providers.azure_openai import AZURE_WORKER_MODEL
from tests.test_azure_openai import config, httpx, success
from tests.test_azure_openai import sdk_transport as _sdk_transport

sdk_transport = _sdk_transport


def binding(**overrides):
    return ResponsesBinding(**{
        "run_id": "worker-run.test", "model": AZURE_WORKER_MODEL,
        "endpoint": config().base_url, "deployment": "gpt-6-sol",
        "budget_usd": 30, "max_calls": 4, "max_output_tokens": 128,
        "deadline_unix": time.time() + 60, **overrides,
    })


async def call(session, request_id="turn-1", **overrides):
    return await session.complete(request_id, **{
        "system": "Read precisely.", "messages": [{"role": "user", "content": "hi"}],
        **overrides,
    })


def test_completed_receipt_replays_across_restart_without_another_paid_call(tmp_path, sdk_transport):
    sent, _ = sdk_transport(success)
    directory = tmp_path / "session"
    limits = binding()
    first = ResponsesSession.create(directory, limits, config())
    result = asyncio.run(call(first))
    first.close()
    resumed = ResponsesSession.open(directory, limits, config())
    try:
        replayed = asyncio.run(call(resumed))
        assert replayed == result
        assert len(sent) == 1
        assert resumed.known_cost_usd == pytest.approx(0.000366)
        assert not resumed.unsettled
    finally:
        resumed.close()


def test_request_identity_and_serial_admission(tmp_path, sdk_transport):
    sent, _ = sdk_transport(success)
    session = ResponsesSession.create(tmp_path / "session", binding(max_calls=2), config())

    async def scenario():
        first, replay = await asyncio.gather(call(session), call(session))
        assert first == replay
        assert len(sent) == 1
        with pytest.raises(ResponsesConflict, match="different input"):
            await call(session, messages=[{"role": "user", "content": "changed"}])
        await call(session, "turn-2")
        with pytest.raises(ResponsesFenced, match="call limit"):
            await call(session, "turn-3")

    try:
        asyncio.run(scenario())
        assert len(sent) == 2
    finally:
        session.close()


def test_budget_cannot_reset_when_the_process_local_facade_restarts(tmp_path, sdk_transport):
    sent, _ = sdk_transport(success)
    exposure = max_call_cost_usd(AZURE_WORKER_MODEL, max_output_tokens=128)
    limits = binding(budget_usd=exposure + 0.000183)
    directory = tmp_path / "session"
    session = ResponsesSession.create(directory, limits, config())
    asyncio.run(call(session))
    session.close()
    reopened = ResponsesSession.open(directory, limits, config())
    try:
        with pytest.raises(BudgetExceeded):
            asyncio.run(call(reopened, "turn-2"))
        assert len(sent) == 1
    finally:
        reopened.close()


def test_memory_rollback_does_not_erase_spending(tmp_path, sdk_transport):
    sdk_transport(success)
    store = Store.open(tmp_path / "repo", "rollback-test")
    branch = store.branch("worker")
    before = branch.checkpoint("before provider call")
    session = ResponsesSession.create(store.backend.common_dir / "responses-journal", binding(), config())
    try:
        asyncio.run(call(session))
        branch.write("artifact.txt", "new work")
        branch.checkpoint("after provider call")
        branch.rollback(before, "restore earlier reasoning and files")
        assert not branch.path("artifact.txt").exists()
        assert session.lookup("turn-1") is not None
        assert session.known_cost_usd == pytest.approx(0.000366)
    finally:
        session.close()
        store.close()


@pytest.mark.parametrize("mode", ["lost_reply", "before_commit", "after_commit"])
def test_failures_keep_the_correct_receipt_and_fence_restart(tmp_path, sdk_transport, monkeypatch, mode):
    def handle(wire):
        if mode == "lost_reply":
            raise httpx.ReadError("private-failure-marker", request=wire)
        return success(wire)

    sent, _ = sdk_transport(handle)
    directory = tmp_path / "session"
    limits = binding()
    session = ResponsesSession.create(directory, limits, config())
    original = session._save_receipt

    def fail(request_id, payload):
        if mode == "after_commit":
            original(request_id, payload)
        raise OSError("private-failure-marker")

    if mode != "lost_reply":
        monkeypatch.setattr(session, "_save_receipt", fail)
    with pytest.raises((InfraFailure, OSError)):
        asyncio.run(call(session))
    session.close()
    reopened = ResponsesSession.open(directory, limits, config())
    try:
        assert reopened.unsettled is (mode != "after_commit")
        if mode == "after_commit":
            assert reopened.lookup("turn-1") is not None
        else:
            with pytest.raises(ResponsesFenced, match="unknown"):
                reopened.lookup("turn-1")
        with pytest.raises(ResponsesFenced):
            asyncio.run(call(reopened, "turn-2"))
        assert len(sent) == 1
        assert b"private-failure-marker" not in (directory / "calls.sqlite3").read_bytes()
    finally:
        reopened.close()


@pytest.mark.parametrize("fail", [None, "read_error", "system_exit"])
def test_repeated_cancellation_owns_the_call_and_its_late_outcome(tmp_path, sdk_transport, fail):
    entered, release = threading.Event(), threading.Event()

    def handle(wire):
        entered.set()
        assert release.wait(10)
        if fail == "system_exit":
            raise SystemExit(17)
        if fail:
            raise httpx.ReadError("late-private-failure", request=wire)
        return success(wire)

    sent, _ = sdk_transport(handle)
    session = ResponsesSession.create(tmp_path / "session", binding(), config())

    async def scenario():
        loop = asyncio.get_running_loop()
        diagnostics = []
        loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))
        task = asyncio.create_task(call(session))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            with pytest.raises(ResponsesFenced, match="active"):
                session.close()
            release.set()
            expected = BaseExceptionGroup if fail else asyncio.CancelledError
            with pytest.raises(expected):
                await task
            assert session.unsettled is bool(fail)
            if not fail:
                assert session.lookup("turn-1") is not None
            with pytest.raises(ResponsesFenced):
                await call(session, "turn-2")
            assert diagnostics == []
        finally:
            release.set()
            if not task.done():
                await asyncio.wait((task,))

    try:
        asyncio.run(scenario())
        assert len(sent) == 1
    finally:
        session.close()


def test_intent_is_durable_when_the_provider_process_dies(tmp_path):
    directory = tmp_path / "session"
    limits = binding()
    code = r'''
import json, os, sys, httpx
from pathlib import Path
from taste.brains.responses_session import ResponsesBinding, ResponsesSession
from tests.test_azure_openai import config
import asyncio
original = httpx.Client
def die(wire):
    os._exit(37)
class Client(original):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, transport=httpx.MockTransport(die))
httpx.Client = Client
session = ResponsesSession.create(Path(sys.argv[1]), ResponsesBinding(**json.loads(sys.argv[2])), config())
asyncio.run(session.complete('turn-1', system='s', messages=[]))
'''
    result = subprocess.run([sys.executable, "-c", code, str(directory), json.dumps(asdict(limits))],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 37, result.stderr
    reopened = ResponsesSession.open(directory, limits, config())
    try:
        assert reopened.unsettled
        with pytest.raises(ResponsesFenced, match="unknown"):
            reopened.lookup("turn-1")
        with pytest.raises(ResponsesFenced):
            asyncio.run(call(reopened, "turn-2"))
    finally:
        reopened.close()


def test_limits_route_and_lease_are_checked_before_dispatch(tmp_path, sdk_transport):
    sent, _ = sdk_transport(success)
    limits = binding(deadline_unix=time.time() - 1)
    directory = tmp_path / "session"
    session = ResponsesSession.create(directory, limits, config())
    try:
        with pytest.raises(BlockingIOError):
            ResponsesSession.open(directory, limits, config())
        with pytest.raises(ResponsesFenced, match="deadline"):
            asyncio.run(call(session))
    finally:
        session.close()
    with pytest.raises(ResponsesConflict, match="binding"):
        ResponsesSession.open(directory, replace(limits, budget_usd=31), config())
    with pytest.raises(ResponsesConflict, match="route"):
        ResponsesSession.create(tmp_path / "bad", replace(limits, endpoint="https://api.openai.com/v1"), config())
    assert sent == []


def test_corrupt_receipt_is_rejected_on_reopen(tmp_path, sdk_transport):
    sdk_transport(success)
    directory = tmp_path / "session"
    limits = binding()
    session = ResponsesSession.create(directory, limits, config())
    asyncio.run(call(session))
    session.close()
    with sqlite3.connect(directory / "calls.sqlite3") as db:
        db.execute("UPDATE calls SET result='{}' WHERE id='turn-1'")
    with pytest.raises(ResponsesConflict, match="digest"):
        ResponsesSession.open(directory, limits, config())


def test_queued_cancellation_does_not_add_an_intent_or_fence_active_work(tmp_path, sdk_transport):
    entered, release = threading.Event(), threading.Event()

    def handle(wire):
        entered.set()
        assert release.wait(10)
        return success(wire)

    sent, _ = sdk_transport(handle)
    session = ResponsesSession.create(tmp_path / "session", binding(), config())

    async def scenario():
        first = asyncio.create_task(call(session))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            queued = asyncio.create_task(call(session, "queued"))
            await asyncio.sleep(0)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            assert session.lookup("queued") is None
            release.set()
            await first
            await call(session, "turn-2")
        finally:
            release.set()
            if not first.done():
                await asyncio.wait((first,))

    try:
        asyncio.run(scenario())
        assert len(sent) == 2
    finally:
        session.close()


def test_expiry_while_the_call_thread_is_queued_does_not_spend(tmp_path, sdk_transport, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    sent, _ = sdk_transport(success)
    limits = binding()
    session = ResponsesSession.create(tmp_path / "session", limits, config())
    now = [time.time()]
    monkeypatch.setattr("taste.brains.responses_session.time", SimpleNamespace(time=lambda: now[0]))
    original = session._dispatch

    def delayed(request):
        entered.set()
        assert release.wait(10)
        return original(request)

    monkeypatch.setattr(session, "_dispatch", delayed)

    async def scenario():
        task = asyncio.create_task(call(session))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            now[0] = limits.deadline_unix + 1
            release.set()
            with pytest.raises(ResponsesFenced, match="deadline"):
                await task
        finally:
            release.set()
            if not task.done():
                await asyncio.wait((task,))

    try:
        asyncio.run(scenario())
        assert sent == []
        assert not session.unsettled
        assert session.known_cost_usd == 0
    finally:
        session.close()


def test_cancelling_the_internal_owner_still_drains_the_http_thread(tmp_path, sdk_transport):
    entered, release = threading.Event(), threading.Event()

    def handle(wire):
        entered.set()
        assert release.wait(10)
        return success(wire)

    sdk_transport(handle)
    session = ResponsesSession.create(tmp_path / "session", binding(), config())

    async def scenario():
        caller = asyncio.create_task(call(session))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            # Event-loop shutdown can cancel this Task independently of the
            # public caller. The HTTP thread must remain owned nevertheless.
            session._active.cancel()
            await asyncio.sleep(0.03)
            assert not caller.done(), "owner cancellation abandoned the running HTTP call"
            with pytest.raises(ResponsesFenced, match="active"):
                session.close()
            release.set()
            with pytest.raises(BaseExceptionGroup):
                await caller
            assert session.lookup("turn-1") is not None
            assert not session.unsettled
        finally:
            release.set()
            await asyncio.gather(caller, return_exceptions=True)

    try:
        asyncio.run(scenario())
    finally:
        session.close()


def test_another_thread_cannot_release_the_owner_lease(tmp_path):
    limits = binding()
    directory = tmp_path / "session"
    session = ResponsesSession.create(directory, limits, config())

    async def scenario():
        with pytest.raises(ResponsesFenced, match="thread"):
            await asyncio.to_thread(session.close)
        with pytest.raises(BlockingIOError):
            ResponsesSession.open(directory, limits, config())

    try:
        asyncio.run(scenario())
    finally:
        session.close()
