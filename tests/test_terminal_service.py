"""Real Unix sockets and worker processes across the terminal ownership boundary."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from taste.brains import terminal_service as rpc
from taste.brains.terminal_broker import (
    TerminalBinding,
    TerminalBroker,
    TerminalConflict,
    TerminalFenced,
    TerminalRequest,
    TerminalResult,
)
from taste.brains.terminal_service import (
    TerminalAccessDenied,
    TerminalClient,
    TerminalCredential,
    TerminalGrant,
    TerminalService,
    TerminalUnavailable,
)
from tests.test_goal_cancellation import _leaves, wait_event
from tests.test_terminal_broker import Environment


@pytest.fixture
def rig(tmp_path):
    @asynccontextmanager
    async def start(*, max_connections=16, worker_uid=None):
        env = Environment()
        binding = TerminalBinding("trial_1", env.environment_id, time.time() + 30, 10)
        owner = TerminalBroker.create(tmp_path / "ledger", binding, env)
        with tempfile.TemporaryDirectory(prefix="taste-rpc-", dir="/tmp") as temporary:
            path = str(Path(temporary) / "service" / "terminal.sock")
            credentials = [TerminalCredential(path, os.geteuid(), os.geteuid() if worker_uid is None else worker_uid,
                TerminalGrant(binding, actor, 5), token * 64) for actor, token in [("worker_A", "a"), ("worker_B", "b")]]
            service = TerminalService(owner, credentials, max_connections=max_connections)
            await service.start()
            value = SimpleNamespace(owner=owner, env=env, service=service, credentials=credentials,
                                    a=TerminalClient(credentials[0]), b=TerminalClient(credentials[1]))
            try:
                yield value
            finally:
                env.release.set()
                env.stop_release.set()
                env.stop_error = None
                await service.close()
                owner.close()

    return start


def req(identifier="effect_1", actor="worker_A", **changes):
    return replace(TerminalRequest(identifier, actor, "do task", "/tmp", 3), **changes)


def message(credential, *, operation="execute", arguments=None):
    return {"version": 1, "token": credential.token, "grant": credential.grant.to_dict(),
            "operation": operation, "arguments": asdict(req()) if arguments is None else arguments}


async def connect(credential, payload):
    reader, writer = await asyncio.open_unix_connection(credential.socket_path)
    await rpc._write(writer, payload, rpc.REQUEST_BYTES)
    return reader, writer


async def raw_reply(credential, payload):
    reader, writer = await connect(credential, payload)
    try:
        return await rpc._read(reader, rpc.RESPONSE_BYTES)
    finally:
        writer.close()
        await writer.wait_closed()


def test_binary_receipts_replay_and_actor_scoped_lookup(rig):
    async def scenario():
        async with rig() as r:
            r.env.result = TerminalResult(127, b"\xff\0captured", b"\x80error", 7_000_000, 100)
            assert await r.a.ping() == "ready"
            assert await r.a.execute(req()) == r.env.result
            assert await r.a.execute(req()) == r.env.result
            assert await r.a.lookup(req().request_id) == r.env.result
            with pytest.raises(TerminalConflict):
                await r.b.lookup(req().request_id)
            assert await r.b.lookup("absent") is None
            with pytest.raises(TerminalConflict):
                await r.a.execute(req(command="changed effect"))
            assert len(r.env.calls) == 1
            assert r.owner.phase == "ready"
            assert r.credentials[0].token not in repr(r.credentials[0])
            assert TerminalCredential.from_dict(r.credentials[0].to_dict()) == r.credentials[0]
    asyncio.run(scenario())


def test_grading_handoff_fences_incomplete_rpc_and_new_grants_but_keeps_receipts(rig):
    async def scenario():
        async with rig() as r:
            await r.a.execute(req())
            reader, writer = await asyncio.open_unix_connection(r.credentials[0].socket_path)
            payload = rpc._json(message(r.credentials[0], arguments=asdict(req("late"))))
            writer.write(len(payload).to_bytes(4, "big")[:2])
            await writer.drain()
            assert r.service.seal_for_grading() == r.owner.binding
            assert r.service.seal_for_grading() == r.owner.binding
            writer.write(len(payload).to_bytes(4, "big")[2:] + payload)
            await writer.drain()
            try:
                assert (await rpc._read(reader, rpc.RESPONSE_BYTES))["status"] == "fenced"
            finally:
                writer.close()
                await writer.wait_closed()
            assert await r.a.ping() == "sealed"
            assert await r.a.lookup(req().request_id) == r.env.result
            assert await r.a.execute(req()) == r.env.result
            with pytest.raises(TerminalFenced):
                await r.b.execute(req("new", actor="worker_B"))
            with pytest.raises(TerminalFenced):
                r.service.authorize(r.credentials[0])
            assert not r.env.stopped and len(r.env.calls) == 1
            assert [kind for kind, _ in r.owner.events()].count("sealed") == 1
    asyncio.run(scenario())


def test_only_idle_trusted_owner_can_hand_environment_to_grading(rig):
    async def scenario():
        async with rig() as r:
            reply = await raw_reply(r.credentials[0], message(r.credentials[0], operation="seal_for_grading", arguments={}))
            assert reply["status"] == "denied" and r.owner.phase == "ready"
            r.env.release.clear()
            active = asyncio.create_task(r.a.execute(req()))
            await wait_event(r.env.entered)
            with pytest.raises(TerminalFenced):
                r.service.seal_for_grading()
            r.env.release.set()
            assert await active == r.env.result
            r.service.seal_for_grading()
            assert not r.env.stopped
            await r.service.close()
            assert r.env.stopped
            with pytest.raises(TerminalFenced):
                r.service.seal_for_grading()
    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["token", "actor", "trial", "deadline", "boolean_scope", "timeout", "version", "extra"])
def test_forged_or_excess_authority_never_reaches_broker(rig, fault):
    async def scenario():
        async with rig() as r:
            payload = message(r.credentials[0])
            if fault == "token":
                payload["token"] = "f" * 64
            elif fault == "actor":
                payload["arguments"]["actor_id"] = "worker_B"
            elif fault == "trial":
                payload["grant"]["binding"]["trial_id"] = "another_trial"
            elif fault == "deadline":
                payload["grant"]["binding"]["deadline_unix"] += 100
            elif fault == "boolean_scope":
                payload["grant"]["max_timeout_seconds"] = True
            elif fault == "timeout":
                payload["arguments"]["timeout_seconds"] = 6
            elif fault == "version":
                payload["version"] = True
            else:
                payload["host_shell"] = True
            reply = await raw_reply(r.credentials[0], payload)
            assert reply["status"] == "denied"
            assert r.credentials[0].token not in json.dumps(reply)
            assert not r.env.calls and not r.env.stopped
            assert r.owner.lookup(req()) is None
    asyncio.run(scenario())


def test_worker_uid_is_verified_at_the_socket_even_when_token_is_valid(rig):
    async def scenario():
        async with rig(worker_uid=os.geteuid() + 1) as r:
            assert (await raw_reply(r.credentials[0], message(r.credentials[0])))["status"] == "denied"
            assert not r.env.calls
    asyncio.run(scenario())


def test_client_verifies_controller_uid_before_sending_a_bearer_token(rig):
    async def scenario():
        async with rig() as r:
            client = TerminalClient(replace(r.credentials[0], server_uid=os.geteuid() + 1))
            with pytest.raises(TerminalAccessDenied, match="server UID"):
                await client.execute(req())
            assert not r.env.calls
    asyncio.run(scenario())


@pytest.mark.parametrize("raw", [b"", b"{", b'{"version":1,"version":1}', b"[1]", b"\xff"])
def test_invalid_framing_or_json_is_bounded_and_has_no_effect(rig, raw):
    async def scenario():
        async with rig() as r:
            reader, writer = await asyncio.open_unix_connection(r.credentials[0].socket_path)
            try:
                writer.write(len(raw).to_bytes(4, "big") + raw)
                await writer.drain()
                assert (await rpc._read(reader, rpc.RESPONSE_BYTES))["status"] == "denied"
                assert not r.env.calls
            finally:
                writer.close()
                await writer.wait_closed()
    asyncio.run(scenario())


def test_oversized_header_is_rejected_without_reading_the_payload(rig):
    async def scenario():
        async with rig() as r:
            reader, writer = await asyncio.open_unix_connection(r.credentials[0].socket_path)
            writer.write((rpc.REQUEST_BYTES + 1).to_bytes(4, "big"))
            await writer.drain()
            try:
                assert (await asyncio.wait_for(rpc._read(reader, rpc.RESPONSE_BYTES), 1))["status"] == "denied"
                assert not r.env.calls
            finally:
                writer.close()
                await writer.wait_closed()
    asyncio.run(scenario())


@pytest.mark.parametrize("ending", ["cancel", "disconnect", "trailing_bytes"])
def test_lost_worker_connection_drains_active_effect_before_settlement(rig, ending):
    async def scenario():
        async with rig() as r:
            r.env.release.clear()
            r.env.stop_release.clear()
            active = writer = None
            if ending == "cancel":
                active = asyncio.create_task(r.a.execute(req()))
            else:
                _reader, writer = await connect(r.credentials[0], message(r.credentials[0]))
            await wait_event(r.env.entered)
            if active is not None:
                active.cancel()
            elif ending == "disconnect":
                writer.close()
                await writer.wait_closed()
            else:
                writer.write(b"unexpected second request")
                await writer.drain()
            await wait_event(r.env.stop_entered)
            if active is not None:
                for _ in range(3):
                    active.cancel()
                    await asyncio.sleep(0.01)
                    assert not active.done()
            assert r.owner.phase == "fenced" and not r.env.stopped
            r.env.stop_release.set()
            if active is not None:
                with pytest.raises(BaseException) as caught:
                    await active
                assert any(isinstance(e, asyncio.CancelledError) for e in _leaves(caught.value))
            else:
                deadline = time.monotonic() + 2
                while r.owner.phase != "stopped":
                    assert time.monotonic() < deadline
                    await asyncio.sleep(0.01)
                writer.close()
                await writer.wait_closed()
            assert r.env.stopped and r.owner.phase == "stopped"
            assert len(r.env.calls) == r.env.stop_calls == 1
            with pytest.raises(TerminalFenced):
                await r.a.lookup(req().request_id)
    asyncio.run(scenario())


def test_cancelled_queued_actor_has_no_effect_and_does_not_stop_active_actor(rig):
    async def scenario():
        async with rig() as r:
            r.env.release.clear()
            first = asyncio.create_task(r.a.execute(req()))
            await wait_event(r.env.entered)
            queued = asyncio.create_task(r.b.execute(req("queued", "worker_B")))
            # Ping ensures the event loop has processed both connections; the
            # operation remains queued behind the first broker effect.
            assert await r.b.ping() == "ready"
            queued.cancel()
            with pytest.raises(BaseException) as caught:
                await queued
            assert any(isinstance(e, asyncio.CancelledError) for e in _leaves(caught.value))
            assert r.owner.lookup(req("queued", "worker_B")) is None
            assert not r.env.stopped and not r.env.stop_calls
            r.env.release.set()
            assert await first == r.env.result
            assert len(r.env.calls) == 1
    asyncio.run(scenario())


def test_concurrent_duplicate_rpc_is_a_single_persistent_effect(rig):
    async def scenario():
        async with rig() as r:
            results = await asyncio.gather(r.a.execute(req()), r.a.execute(req()))
            assert results == [r.env.result, r.env.result]
            assert len(r.env.calls) == 1
    asyncio.run(scenario())


def test_service_close_owns_active_effect_and_partial_handshakes_through_repeated_cancel(rig):
    async def scenario():
        async with rig() as r:
            r.env.release.clear()
            r.env.stop_release.clear()
            active = asyncio.create_task(r.a.execute(req()))
            await wait_event(r.env.entered)
            _reader, idle = await asyncio.open_unix_connection(r.credentials[0].socket_path)
            idle.write(b"\0")
            await idle.drain()
            closing = asyncio.create_task(r.service.close())
            await wait_event(r.env.stop_entered)
            for _ in range(3):
                closing.cancel()
                await asyncio.sleep(0.01)
                assert not closing.done()
            r.env.stop_release.set()
            with pytest.raises(asyncio.CancelledError):
                await closing
            with pytest.raises(TerminalUnavailable):
                await active
            idle.close()
            with suppress(ConnectionResetError):
                await idle.wait_closed()
            assert r.owner.phase == "stopped" and not r.service.path.exists()
            assert not r.service._handlers
    asyncio.run(scenario())


def test_failed_stop_retains_service_cleanup_obligation_until_retry(rig):
    async def scenario():
        async with rig() as r:
            r.env.stop_error = ConnectionError("stop acknowledgement lost")
            with pytest.raises(ConnectionError):
                await r.service.close()
            assert r.owner.phase == "fenced" and r.service.path.exists()
            with pytest.raises(TerminalFenced):
                r.owner.close()
            r.env.stop_error = None
            await r.service.close()
            assert r.owner.phase == "stopped" and not r.service.path.exists()
    asyncio.run(scenario())


def test_backend_value_error_is_uncertain_after_admission_not_a_validation_denial(rig):
    async def scenario():
        async with rig() as r:
            r.env.error = ValueError("secret-token-must-not-cross-the-wire")
            with pytest.raises(TerminalUnavailable) as caught:
                await r.a.execute(req())
            assert "secret-token" not in str(caught.value)
            assert len(r.env.calls) == 1 and r.env.stopped
            with pytest.raises(TerminalFenced):
                await r.a.lookup(req().request_id)
    asyncio.run(scenario())


def test_connection_limit_and_incomplete_handshake_timeout_do_not_admit_effects(rig, monkeypatch):
    monkeypatch.setattr(rpc, "HANDSHAKE_SECONDS", 0.15)

    async def scenario():
        async with rig(max_connections=1) as r:
            reader, writer = await asyncio.open_unix_connection(r.credentials[0].socket_path)
            writer.write(b"\0")
            await writer.drain()
            await asyncio.sleep(0.01)
            with pytest.raises(TerminalUnavailable):
                await r.a.execute(req())
            assert (await rpc._read(reader, rpc.RESPONSE_BYTES))["status"] == "uncertain"
            writer.close()
            await writer.wait_closed()
            await asyncio.sleep(0.01)
            assert await r.a.ping() == "ready"
            assert not r.env.calls
    asyncio.run(scenario())


def test_maximum_binary_receipt_crosses_framing_without_truncation_or_reexecution(rig):
    async def scenario():
        async with rig() as r:
            r.env.result = TerminalResult(0, bytes(range(256)) * 4096, b"\xff" * 1048576, 77, 88)
            assert await r.a.execute(req()) == r.env.result
            assert await r.a.lookup(req().request_id) == r.env.result
            assert len(r.env.calls) == 1
    asyncio.run(scenario())


def test_killed_worker_process_is_drained_by_surviving_service(rig, tmp_path):
    async def scenario():
        async with rig() as r:
            credential_path = tmp_path / "private.json"
            credential_path.write_text(json.dumps(r.credentials[0].to_dict()))
            credential_path.chmod(0o600)
            r.env.release.clear()
            script = """
import asyncio,json,sys
from pathlib import Path
from taste.brains.terminal_service import TerminalClient,TerminalCredential
from taste.brains.terminal_broker import TerminalRequest
client=TerminalClient(TerminalCredential.from_dict(json.loads(Path(sys.argv[1]).read_text())))
asyncio.run(client.execute(TerminalRequest('effect_1','worker_A','do task','/tmp',3)))
"""
            process = await asyncio.create_subprocess_exec(sys.executable, "-c", script, str(credential_path),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                await wait_event(r.env.entered)
                process.kill()
                await asyncio.wait_for(process.communicate(), 3)
                assert process.returncode == -9
                await wait_event(r.env.stop_entered)
                await r.service.close()
                assert r.env.stopped and r.owner.phase == "stopped"
                assert len(r.env.calls) == 1
            finally:
                if process.returncode is None:
                    with suppress(ProcessLookupError):
                        process.kill()
                    await process.communicate()
    asyncio.run(scenario())


def test_shutdown_owner_cancelled_before_start_is_replaced_and_drained(rig, monkeypatch):
    async def scenario():
        async with rig() as r:
            create = asyncio.create_task
            cancelled = []

            def cancel_first_owner(coroutine, **kwargs):
                task = create(coroutine, **kwargs)
                if coroutine.cr_code.co_name == "_shutdown" and not cancelled:
                    cancelled.append(task)
                    task.cancel()
                return task

            with monkeypatch.context() as patch:
                patch.setattr(asyncio, "create_task", cancel_first_owner)
                with pytest.raises(asyncio.CancelledError):
                    await r.service.close()
            assert r.env.stopped and r.owner.phase == "stopped" and r.env.stop_calls == 1
            assert not r.service.path.exists()
    asyncio.run(scenario())


def test_socket_replacement_is_not_removed_by_service_cleanup(rig):
    async def scenario():
        async with rig() as r:
            r.service.path.unlink()
            r.service.path.write_text("replacement owned by another operation")
            try:
                with pytest.raises(TerminalConflict, match="replaced"):
                    await r.service.close()
                assert r.env.stopped and r.owner.phase == "stopped"
                assert r.service.path.read_text() == "replacement owned by another operation"
            finally:
                r.service.path.unlink()
    asyncio.run(scenario())


def test_trusted_registration_is_exact_and_not_an_rpc_operation(rig):
    async def scenario():
        async with rig() as r:
            credential = replace(r.credentials[0], token="c" * 64,
                                 grant=replace(r.credentials[0].grant, actor_id="worker_C"))
            r.service.authorize(credential)
            r.service.authorize(credential)
            with pytest.raises(TerminalConflict):
                r.service.authorize(replace(credential, token="d" * 64))
            client = TerminalClient(credential)
            assert await client.execute(req(actor="worker_C")) == r.env.result
            response = await raw_reply(r.credentials[0], message(r.credentials[0], operation="authorize", arguments=credential.to_dict()))
            assert response["status"] == "denied"
            assert len(r.env.calls) == 1
    asyncio.run(scenario())
