"""Coordinator-only grant issuance over real bounded Unix sockets."""

from __future__ import annotations

import asyncio
import os
import secrets
import tempfile
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from taste.brains import terminal_issuer as issuer_rpc
from taste.brains import terminal_service as rpc
from taste.brains.terminal_broker import (
    TerminalBinding,
    TerminalBroker,
    TerminalConflict,
    TerminalFenced,
)
from taste.brains.terminal_issuer import TerminalIssuerClient, TerminalIssuerCredential
from taste.brains.terminal_service import (
    TerminalAccessDenied,
    TerminalClient,
    TerminalCredential,
    TerminalGrant,
    TerminalService,
    TerminalUnavailable,
)
from taste.brains.terminal_worker_policy import TERMINAL_POLICY_KEY, TerminalWorkerPolicy
from taste.brains.worker_protocol import assignment_run_id
from tests.test_azure_worker_policy import assignment
from tests.test_terminal_broker import Environment
from tests.test_terminal_service import raw_reply, req


@pytest.fixture
def rig(tmp_path):
    @asynccontextmanager
    async def start(*, coordinator_uid=None):
        source = assignment()
        env = Environment()
        binding = TerminalBinding("issuer_trial", env.environment_id, source.resources["azure_openai"]["deadline_unix"], 20)
        policy = TerminalWorkerPolicy(binding, 5)
        source = replace(source, resources={**source.resources, TERMINAL_POLICY_KEY: policy.to_dict()})
        owner = TerminalBroker.create(tmp_path / "ledger", binding, env)
        with tempfile.TemporaryDirectory(prefix="taste-issuer-", dir="/tmp") as directory:
            credential = TerminalIssuerCredential(str(Path(directory) / "service/terminal.sock"),
                os.geteuid(), os.geteuid() if coordinator_uid is None else coordinator_uid,
                "1" * 64, policy, secrets.token_hex(32))
            service = TerminalService(owner, issuer=credential)
            await service.start()
            try:
                yield SimpleNamespace(credential=credential, assignment=source, prepared="a" * 40,
                    owner=owner, service=service, env=env, client=TerminalIssuerClient(credential))
            finally:
                await service.close()
                owner.close()
    return start


def message(r):
    return {"version": 1, "operation": "issue", "token": r.credential.token,
            "scope": r.credential.public_scope(), "arguments": {
                "assignment": r.assignment.to_dict(), "prepared_state_id": r.prepared}}


def test_issuance_is_stable_and_worker_grants_cannot_cross_roles(rig):
    async def scenario():
        async with rig() as r:
            spec = SimpleNamespace(assignment=r.assignment, prepared_state_id=r.prepared,
                                   run_id=assignment_run_id(r.assignment))
            grant = await asyncio.to_thread(r.client, spec)
            assert await asyncio.to_thread(r.client, spec) == grant
            assert len(r.service._credentials) == len(r.service._issued) == 1
            assert not r.env.calls
            client = TerminalClient(grant)
            request = req(actor=grant.grant.actor_id)
            assert await client.execute(request) == r.env.result
            assert await client.lookup(request.request_id) == r.env.result
            with pytest.raises(TerminalAccessDenied):
                await asyncio.to_thread(TerminalIssuerClient(replace(r.credential, token=grant.token)), spec)
            forged = TerminalClient(replace(grant, token=r.credential.token))
            with pytest.raises(TerminalAccessDenied):
                await forged.execute(req("forbidden", actor=grant.grant.actor_id))
            assert len(r.env.calls) == 1
            assert grant.token not in repr(grant) and r.credential.token not in repr(r.credential)
    asyncio.run(scenario())


def test_checkpoint_conflict_and_attempt_identity_cannot_rotate_an_existing_grant(rig):
    async def scenario():
        async with rig() as r:
            first = await asyncio.to_thread(r.client.issue, r.assignment, r.prepared)
            with pytest.raises(TerminalConflict):
                await asyncio.to_thread(r.client.issue, r.assignment, "b" * 40)
            second = await asyncio.to_thread(r.client.issue, replace(r.assignment, attempt=1), "b" * 40)
            assert second.grant.actor_id != first.grant.actor_id and second.token != first.token
            assert await asyncio.to_thread(r.client.issue, r.assignment, r.prepared) == first
            assert not r.env.calls
    asyncio.run(scenario())


@pytest.mark.parametrize("damage", ["token", "scope", "version", "extra", "prepared", "terminal", "deadline", "timeout"])
def test_forged_or_drifted_issuance_never_registers_a_worker(rig, damage):
    async def scenario():
        async with rig() as r:
            payload = message(r)
            if damage == "token":
                payload["token"] = "f" * 64
            elif damage == "scope":
                payload["scope"]["input_sha256"] = "2" * 64
            elif damage == "version":
                payload["version"] = True
            elif damage == "extra":
                payload["worker_uid"] = 0
            elif damage == "prepared":
                payload["arguments"]["prepared_state_id"] = "../not-a-checkpoint"
            else:
                policy = payload["arguments"]["assignment"]["resources"][TERMINAL_POLICY_KEY]
                if damage == "terminal":
                    policy["binding"]["environment_id"] = "foreign"
                elif damage == "deadline":
                    policy["binding"]["deadline_unix"] += 10
                else:
                    policy["max_timeout_seconds"] += 1
            assert await raw_reply(r.credential, payload) == {"version": 1, "status": "denied"}
            assert not r.service._credentials and not r.service._issued and not r.env.calls
    asyncio.run(scenario())


def test_peer_uid_is_checked_on_both_ends(rig):
    async def scenario():
        async with rig(coordinator_uid=os.geteuid() + 1) as r:
            assert await raw_reply(r.credential, message(r)) == {"version": 1, "status": "denied"}
            client = TerminalIssuerClient(replace(r.credential, coordinator_uid=os.geteuid(),
                                                 server_uid=os.geteuid() + 1))
            with pytest.raises(TerminalAccessDenied, match="server UID"):
                await asyncio.to_thread(client.issue, r.assignment, r.prepared)
            assert not r.service._credentials
    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["sealed", "expired", "full"])
def test_closed_admission_cannot_issue_more_authority(rig, monkeypatch, phase):
    async def scenario():
        async with rig() as r:
            if phase == "sealed":
                r.service.seal_for_grading()
            elif phase == "expired":
                monkeypatch.setattr(issuer_rpc.time, "time", lambda: r.owner.binding.deadline_unix + 1)
            else:
                for i in range(256):
                    r.service.authorize(TerminalCredential(r.credential.socket_path, os.geteuid(), os.geteuid(),
                        TerminalGrant(r.owner.binding, f"actor_{i}", 5), secrets.token_hex(32)))
            assert await raw_reply(r.credential, message(r)) == {"version": 1, "status": "fenced"}
            assert not r.service._issued and not r.env.calls
    asyncio.run(scenario())


def test_lost_reply_has_no_automatic_retry_and_explicit_retrieval_keeps_the_same_token(rig, monkeypatch):
    async def scenario():
        async with rig() as r:
            original = rpc._write
            entered, release = asyncio.Event(), asyncio.Event()

            async def delay_reply(writer, value, limit):
                if "credential" in value:
                    entered.set()
                    await release.wait()
                await original(writer, value, limit)

            monkeypatch.setattr(rpc, "_write", delay_reply)
            monkeypatch.setattr(issuer_rpc, "HANDSHAKE_SECONDS", 0.1)
            try:
                with pytest.raises(TerminalUnavailable):
                    await asyncio.to_thread(r.client.issue, r.assignment, r.prepared)
                assert entered.is_set() and len(r.service._issued) == 1 and not r.env.calls
                saved = next(iter(r.service._issued.values()))[1]
            finally:
                release.set()
            monkeypatch.setattr(rpc, "_write", original)
            monkeypatch.setattr(issuer_rpc, "HANDSHAKE_SECONDS", 5)
            assert await asyncio.to_thread(r.client.issue, r.assignment, r.prepared) == saved
    asyncio.run(scenario())


def test_issuer_token_cannot_be_registered_as_a_worker(rig):
    async def scenario():
        async with rig() as r:
            credential = TerminalCredential(r.credential.socket_path, os.geteuid(), os.geteuid(),
                r.credential.policy.grant(r.assignment), r.credential.token)
            with pytest.raises(TerminalAccessDenied):
                r.service.authorize(credential)
            with pytest.raises(ValueError):
                TerminalService(r.owner, [credential], issuer=r.credential)
            assert not r.service._credentials
    asyncio.run(scenario())


def test_private_issuer_roundtrip_and_expired_client_are_checked(rig, monkeypatch):
    async def scenario():
        async with rig() as r:
            assert TerminalIssuerCredential.from_dict(r.credential.to_dict()) == r.credential
            with pytest.raises(ValueError):
                TerminalIssuerCredential.from_dict({**r.credential.to_dict(), "extra": True})
            monkeypatch.setattr(issuer_rpc.time, "time", lambda: r.owner.binding.deadline_unix)
            with pytest.raises(TerminalFenced):
                await asyncio.to_thread(r.client.issue, r.assignment, r.prepared)
            assert not r.service._credentials
    asyncio.run(scenario())
