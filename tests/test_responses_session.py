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
from tests.test_openai_responses import response

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


def test_a_request_ends_at_its_own_ceiling_and_is_charged_by_what_it_sent(tmp_path, sdk_transport):
    # The service accepts the request and never answers. Bounded only by the
    # run's deadline, this call would have waited the ten minutes out.
    def never_answers(wire):
        raise httpx.ReadTimeout("no answer", request=wire)

    sent, _ = sdk_transport(never_answers)
    limits = binding(request_seconds=20, deadline_unix=time.time() + 600)
    session = ResponsesSession.create(tmp_path / "session", limits, config())
    try:
        with pytest.raises(InfraFailure):
            asyncio.run(call(session))
        assert len(sent) == 1 and sent[0].extensions["timeout"]["read"] == 20
        accounting = session.call_accounting()
        assert accounting.unknown_calls == 1 and accounting.cost_usd is None
        # The request was 100 bytes, so it held at most 100 tokens and the
        # framing allowance of 4,096, at the dearest rates such a request can
        # meet on this model: 2.50 a million in, 10.00 a million out for the
        # 128 output tokens allowed. Bounded by the model's whole window
        # instead, the figure is $5.25.
        assert accounting.unknown_exposure_usd == pytest.approx(0.01177)
        assert accounting.cost_ceiling_usd == pytest.approx(0.01177)
    finally:
        session.close()


def test_a_settled_journal_has_nothing_unknown_to_charge(tmp_path, sdk_transport):
    sdk_transport(success)
    session = ResponsesSession.create(tmp_path / "session", binding(), config())
    try:
        asyncio.run(call(session))
        accounting = session.call_accounting()
        assert accounting.unknown_calls == 0 and accounting.unknown_exposure_usd == 0
        assert accounting.cost_usd == accounting.cost_ceiling_usd == pytest.approx(0.000366)
    finally:
        session.close()


def test_a_time_ceiling_is_bound_to_the_journal_like_its_other_limits(tmp_path, sdk_transport):
    sdk_transport(success)
    deadline = time.time() + 60
    plain, bounded = tmp_path / "plain", tmp_path / "bounded"
    ResponsesSession.create(plain, binding(deadline_unix=deadline), config()).close()
    # A journal that names no ceiling keeps the identity it had before
    # ceilings existed, so one made by earlier code still reopens.
    identity = json.loads(sqlite3.connect(plain / "calls.sqlite3").execute(
        "SELECT value FROM meta WHERE key='identity'").fetchone()[0])
    assert "request_seconds" not in identity["binding"]
    ResponsesSession.open(plain, binding(deadline_unix=deadline), config()).close()
    with pytest.raises(ResponsesConflict, match="binding changed"):
        ResponsesSession.open(plain, binding(deadline_unix=deadline, request_seconds=20), config())

    ResponsesSession.create(bounded, binding(deadline_unix=deadline, request_seconds=20), config()).close()
    ResponsesSession.open(bounded, binding(deadline_unix=deadline, request_seconds=20), config()).close()
    with pytest.raises(ResponsesConflict, match="binding changed"):
        ResponsesSession.open(bounded, binding(deadline_unix=deadline, request_seconds=30), config())


@pytest.mark.parametrize("ceiling", [0, -5, float("inf"), True, "20", 3601])
def test_a_time_ceiling_must_bound_a_request(ceiling):
    with pytest.raises(ValueError, match="request_seconds"):
        binding(request_seconds=ceiling)


LOST = [{"role": "user", "content": "lost"}]


def never_answers_lost(reply=success):
    """A service that accepts any request containing the word and never answers it."""
    def handle(wire):
        if b"lost" in wire.content:
            raise httpx.ReadTimeout("no answer", request=wire)
        return reply(wire)
    return handle


def test_a_lost_reply_can_be_given_up_and_the_journal_goes_on(tmp_path, sdk_transport):
    # One unanswered request closed the journal for good: every later call was
    # refused, so the worker that owned it ended and another started over.
    sent, _ = sdk_transport(never_answers_lost())
    directory, limits = tmp_path / "session", binding()

    async def lose_one_and_go_on(session):
        with pytest.raises(InfraFailure):
            await call(session, "turn-1", messages=LOST)
        assert session.fenced and session.unsettled
        # The request was 102 bytes: at most 102 tokens and the framing
        # allowance of 4,096, at 2.50 a million, and 128 output tokens at 10.00.
        charged = session.forfeit("turn-1")
        assert charged == pytest.approx(0.011775)
        assert not session.fenced and not session.unsettled
        assert (await call(session, "turn-2")).text_blocks == ("done",)
        accounting = session.call_accounting()
        assert (accounting.completed_calls, accounting.unknown_calls, accounting.lost_calls) == (1, 0, 1)
        assert accounting.settled and accounting.lost_exposure_usd == pytest.approx(0.011775)
        assert accounting.cost_usd is None, "what the lost call cost is still not known"
        assert accounting.cost_ceiling_usd == pytest.approx(0.000366 + 0.011775)
        assert session.forfeit("turn-1") == charged, "giving it up again changes nothing"
        with pytest.raises(ResponsesFenced, match="lost"):
            await call(session, "turn-1", messages=LOST)

    session = ResponsesSession.create(directory, limits, config())
    try:
        asyncio.run(lose_one_and_go_on(session))
    finally:
        session.close()
    # The charge is durable, and so is the journal's being open for calls.
    reopened = ResponsesSession.open(directory, limits, config())
    try:
        assert not reopened.fenced
        assert reopened.outcome("turn-1") == "lost" and reopened.outcome("turn-2") == "completed"
        assert reopened.outcome("never-asked") is None
        assert reopened.call_accounting().lost_exposure_usd == pytest.approx(0.011775)
        asyncio.run(call(reopened, "turn-3"))
        assert len(sent) == 3
    finally:
        reopened.close()


@pytest.mark.parametrize("room,admitted", [(0.01, False), (0.02, True)])
def test_a_lost_reply_is_charged_against_the_cap(tmp_path, sdk_transport, room, admitted):
    # The cap leaves this much beyond one whole call. The lost reply is
    # charged $0.011775 of it, which fits in two cents and not in one.
    sdk_transport(never_answers_lost())
    whole = max_call_cost_usd(AZURE_WORKER_MODEL, max_output_tokens=128)

    async def lose_one_then_ask(session):
        with pytest.raises(InfraFailure):
            await call(session, "turn-1", messages=LOST)
        session.forfeit("turn-1")
        if admitted:
            await call(session, "turn-2")
        else:
            with pytest.raises(BudgetExceeded):
                await call(session, "turn-2")

    session = ResponsesSession.create(tmp_path / "session", binding(budget_usd=whole + room), config())
    try:
        asyncio.run(lose_one_then_ask(session))
    finally:
        session.close()


def test_only_a_call_with_an_unknown_outcome_can_be_given_up(tmp_path, sdk_transport):
    sdk_transport(success)
    session = ResponsesSession.create(tmp_path / "session", binding(), config())
    try:
        asyncio.run(call(session, "answered"))
        for request_id in ("answered", "never-asked"):
            with pytest.raises(ResponsesConflict, match="unknown outcome"):
                session.forfeit(request_id)
        assert session.call_accounting().lost_calls == 0
    finally:
        session.close()


def test_a_lost_request_is_bounded_by_what_the_service_measured_before_it(tmp_path, sdk_transport):
    # A worker's request is its whole conversation again and a little more.
    # The service has already counted the earlier part's tokens, so only what
    # was added needs the byte bound.
    history = [{"role": "user", "content": "x" * 40_000}]

    def measured(wire):
        return httpx.Response(200, json=response(model="gpt-6-sol", usage={
            "input_tokens": 9_000, "output_tokens": 20, "total_tokens": 9_020,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }), headers={"x-ms-served-model": AZURE_WORKER_MODEL})

    sdk_transport(never_answers_lost(measured))

    async def lose_a_continuation_then_a_fresh_request(session):
        await call(session, "turn-1", messages=history)
        with pytest.raises(InfraFailure):
            await call(session, "turn-2", messages=[*history, *LOST])
        # By its 40,131 bytes alone the request could hold 44,227 tokens. The
        # part sent before was measured at 9,000, and the 34 bytes added hold
        # at most 34 more: 13,130 tokens with the allowance, at 2.50 a
        # million, and 128 output tokens at 10.00.
        assert session.call_accounting().unknown_exposure_usd == pytest.approx(0.034105)
        assert session.forfeit("turn-2") == pytest.approx(0.034105)
        # A request that does not continue the measured one has only its bytes.
        with pytest.raises(InfraFailure):
            await call(session, "turn-3", messages=LOST)
        assert session.forfeit("turn-3") == pytest.approx(0.011775)

    session = ResponsesSession.create(tmp_path / "session", binding(), config())
    try:
        asyncio.run(lose_a_continuation_then_a_fresh_request(session))
    finally:
        session.close()


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
