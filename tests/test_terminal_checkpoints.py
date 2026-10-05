"""Checkpoints of the task's files: taken and restored by the controller, at the coordinator's request."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import sqlite3
import stat
import tempfile
import threading
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from taste.brains import terminal_broker
from taste.brains.terminal_broker import (
    TerminalBinding,
    TerminalBroker,
    TerminalConflict,
    TerminalFenced,
)
from taste.brains.terminal_issuer import TerminalIssuerClient, TerminalIssuerCredential
from taste.brains.terminal_service import (
    TerminalClient,
    TerminalService,
    TerminalUnavailable,
)
from taste.brains.terminal_worker_policy import TERMINAL_POLICY_KEY, TerminalWorkerPolicy
from taste.brains.worker_protocol import assignment_run_id
from tests.test_azure_worker_policy import assignment
from tests.test_terminal_broker import Environment, broker, request
from tests.test_terminal_service import raw_reply


class Manifest:
    def __init__(self, files, tar, cap_bytes):
        self.files, self.tar, self.cap_bytes = files, tar, cap_bytes

    def to_dict(self):
        return {"files": self.files, "tar": self.tar, "cap_bytes": self.cap_bytes}

    def summary(self):
        return {"files": sorted(self.files), "bytes": 7}


class Receipt:
    def __init__(self, manifest):
        self.manifest = manifest

    def summary(self):
        return {"files": sorted(self.manifest.files), "exact": True}


class Checkpointing(Environment):
    """The task's files are a dict: a checkpoint copies it, a restore puts the copy back."""

    def __init__(self):
        super().__init__()
        self.files, self.taken, self.restored = {}, [], []
        self.checkpoint_error = self.restore_error = None
        self.tar_bytes = 0
        self.inside, self.hold = threading.Event(), threading.Event()
        self.hold.set()

    def execute(self, request):
        result = super().execute(request)
        self.files[request.request_id] = request.command
        return result

    def checkpoint(self, directory, *, cap_bytes):
        self.inside.set()
        assert self.hold.wait(5)
        if self.checkpoint_error:
            raise self.checkpoint_error
        body = json.dumps(self.files, sort_keys=True).encode()
        tar = hashlib.sha256(body).hexdigest() + ".tar"
        (Path(directory) / tar).write_bytes(b"x" * min(self.tar_bytes, cap_bytes))
        manifest = Manifest(dict(self.files), tar, cap_bytes)
        self.taken.append(manifest)
        return manifest

    def restore(self, manifest, directory):
        self.inside.set()
        assert self.hold.wait(5)
        if self.restore_error:
            raise self.restore_error
        assert (Path(directory) / manifest.tar).exists()
        self.files = dict(manifest.files)
        self.restored.append(manifest)
        return Receipt(manifest)


def run(owner, scenario):
    try:
        asyncio.run(scenario())
    finally:
        owner.close()


def test_a_checkpoint_is_kept_privately_and_read_again_by_its_id(tmp_path):
    env = Checkpointing()
    owner = broker(tmp_path, env)

    async def scenario():
        await owner.execute(request("first", command="one"))
        record = await owner.checkpoint("cp1")
        assert record == {"checkpoint_id": "cp1", "files": ["first"], "bytes": 7}
        assert await owner.checkpoint("cp1") == record and len(env.taken) == 1
        store = tmp_path / "terminal" / "checkpoints"
        assert stat.S_IMODE(store.stat().st_mode) == 0o700
        manifest = store / "cp1.json"
        assert stat.S_IMODE(manifest.stat().st_mode) == 0o600
        assert json.loads(manifest.read_text()) == env.taken[0].to_dict()
        assert owner.checkpoints() == [record]
        assert [kind for kind, _ in owner.events()][-2:] == ["checkpoint_intent", "checkpoint"]

    run(owner, scenario)


def test_restore_puts_back_a_checkpoint_this_controller_took_at_most_once(tmp_path):
    env = Checkpointing()
    owner = broker(tmp_path, env)

    async def scenario():
        await owner.execute(request("first", command="one"))
        await owner.checkpoint("cp1")
        await owner.execute(request("second", command="two"))
        record = await owner.restore("undo_1", "cp1")
        assert record == {"operation_id": "undo_1", "checkpoint_id": "cp1", "files": ["first"], "exact": True}
        assert env.files == {"first": "one"}
        assert await owner.restore("undo_1", "cp1") == record and len(env.restored) == 1
        with pytest.raises(TerminalConflict):
            await owner.restore("undo_1", "cp2")
        with pytest.raises(TerminalConflict):
            await owner.restore("undo_2", "cp9")
        # Commands go on after a restore; checkpoints never count as commands.
        await owner.execute(request("third", command="three"))
        assert owner._db.execute("SELECT count(*) FROM requests").fetchone()[0] == 3
        assert env.files == {"first": "one", "third": "three"}

    run(owner, scenario)


def test_a_failed_checkpoint_or_restore_is_recorded_without_its_text_and_commands_go_on(tmp_path):
    env = Checkpointing()
    owner = broker(tmp_path, env)

    async def scenario():
        env.checkpoint_error = OSError("daemon said no to /secret/path")
        failed = await owner.checkpoint("cp1")
        assert failed == {"checkpoint_id": "cp1", "failed": True, "error": "OSError"}
        env.checkpoint_error = None
        assert await owner.checkpoint("cp1") == failed and not env.taken
        await owner.checkpoint("cp2")
        env.restore_error = RuntimeError("copy failed halfway")
        failed = await owner.restore("undo_1", "cp2")
        assert failed == {"operation_id": "undo_1", "checkpoint_id": "cp2", "failed": True, "error": "RuntimeError"}
        assert owner.phase == "ready"
        assert await owner.execute(request()) == env.result
        assert [record["checkpoint_id"] for record in owner.checkpoints()] == ["cp2"]
        assert "/secret/path" not in json.dumps(owner.events()) and "halfway" not in json.dumps(owner.events())

    run(owner, scenario)


def test_no_command_runs_and_nothing_is_sealed_while_a_checkpoint_is_taken(tmp_path):
    env = Checkpointing()
    owner = broker(tmp_path, env)

    async def scenario():
        env.hold.clear()
        taking = asyncio.create_task(owner.checkpoint("cp1"))
        assert await asyncio.to_thread(env.inside.wait, 5)
        command = asyncio.create_task(owner.execute(request()))
        await asyncio.sleep(0.05)
        assert env.calls == []
        with pytest.raises(TerminalFenced):
            owner.seal_for_grading()
        env.hold.set()
        assert (await taking)["checkpoint_id"] == "cp1"
        assert await command == env.result
        assert [kind for kind, _ in owner.events()] == ["created", "checkpoint_intent", "checkpoint",
                                                       "intent", "completed"]

    run(owner, scenario)


def test_a_departed_caller_never_leaves_a_restore_half_done(tmp_path):
    env = Checkpointing()
    owner = broker(tmp_path, env)

    async def scenario():
        await owner.execute(request("first", command="one"))
        await owner.checkpoint("cp1")
        await owner.execute(request("second", command="two"))
        env.hold.clear()
        env.inside.clear()
        caller = asyncio.create_task(owner.restore("undo_1", "cp1"))
        assert await asyncio.to_thread(env.inside.wait, 5)
        caller.cancel()
        await asyncio.sleep(0.05)
        assert not caller.done()
        env.hold.set()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert env.files == {"first": "one"}
        again = await owner.restore("undo_1", "cp1")
        assert again["exact"] is True and len(env.restored) == 1
        assert [kind for kind, _ in owner.events()][-2:] == ["restore_intent", "restored"]

    run(owner, scenario)


def test_only_a_ready_environment_that_can_checkpoint_takes_one(tmp_path):
    (tmp_path / "plain").mkdir()
    (tmp_path / "sealed").mkdir()
    plain = broker(tmp_path / "plain")

    async def cannot():
        with pytest.raises(TerminalFenced, match="cannot take or restore"):
            await plain.checkpoint("cp1")
        with pytest.raises(TerminalFenced, match="cannot take or restore"):
            await plain.restore("undo_1", "cp1")
        with pytest.raises(ValueError):
            await plain.checkpoint("../escape")

    run(plain, cannot)
    env = Checkpointing()
    sealed = broker(tmp_path / "sealed", env)

    async def after_sealing():
        await sealed.checkpoint("cp1")
        sealed.seal_for_grading()
        with pytest.raises(TerminalFenced, match="no longer admitting"):
            await sealed.checkpoint("cp2")
        with pytest.raises(TerminalFenced, match="no longer admitting"):
            await sealed.restore("undo_1", "cp1")
        assert not env.restored

    run(sealed, after_sealing)


def test_the_store_is_bounded_per_checkpoint_and_in_total(tmp_path, monkeypatch):
    monkeypatch.setattr(terminal_broker, "CHECKPOINT_STORE_BYTES", 2000)
    monkeypatch.setattr(terminal_broker, "CHECKPOINT_CAP_BYTES", 1500)
    env = Checkpointing()
    env.tar_bytes = 1000
    owner = broker(tmp_path, env)

    async def scenario():
        for number in (1, 2, 3):
            await owner.execute(request(f"step_{number}", command=str(number)))
            await owner.checkpoint(f"cp{number}")
        assert [manifest.cap_bytes for manifest in env.taken] == [1500, 1000]
        assert (await owner.checkpoint("cp3")) == {"checkpoint_id": "cp3", "failed": True,
                                                    "error": "CheckpointStoreFull"}

    run(owner, scenario)


def test_restart_records_an_interrupted_checkpoint_and_fences_an_interrupted_restore(tmp_path):
    env = Checkpointing()
    owner = broker(tmp_path, env)
    binding = owner.binding
    owner.close()
    database = tmp_path / "terminal" / "terminal.sqlite3"

    def add_event(kind, payload):
        db = sqlite3.connect(database)
        try:
            with db:
                db.execute("INSERT INTO events(kind,payload) VALUES (?,?)", (kind, json.dumps(payload)))
        finally:
            db.close()

    add_event("checkpoint_intent", {"checkpoint_id": "cp1"})
    reopened = TerminalBroker.open(tmp_path / "terminal", binding, env)
    try:
        assert reopened.phase == "ready"
        assert reopened.events()[-1] == ("checkpoint_failed", {"checkpoint_id": "cp1", "failed": True,
                                                               "error": "interrupted"})
    finally:
        reopened.close()
    add_event("restore_intent", {"operation_id": "undo_1", "checkpoint_id": "cp1"})
    fenced = TerminalBroker.open(tmp_path / "terminal", binding, env)
    assert fenced.phase == "fenced"
    assert fenced.events()[-1] == ("fenced", {"reason": "recovered incomplete restore"})
    asyncio.run(fenced.abort())
    fenced.close()


# -- over the coordinator's socket ------------------------------------------------------


@pytest.fixture
def rig(tmp_path):
    @asynccontextmanager
    async def start():
        source = assignment()
        env = Checkpointing()
        binding = TerminalBinding("issuer_trial", env.environment_id,
                                  source.resources["azure_openai"]["deadline_unix"], 20)
        policy = TerminalWorkerPolicy(binding, 5)
        source = replace(source, resources={**source.resources, TERMINAL_POLICY_KEY: policy.to_dict()})
        owner = TerminalBroker.create(tmp_path / "ledger", binding, env)
        with tempfile.TemporaryDirectory(prefix="taste-checkpoint-", dir="/tmp") as directory:
            credential = TerminalIssuerCredential(str(Path(directory) / "service/terminal.sock"),
                                                  os.geteuid(), os.geteuid(), "1" * 64, policy,
                                                  secrets.token_hex(32))
            service = TerminalService(owner, issuer=credential)
            await service.start()
            try:
                yield SimpleNamespace(credential=credential, assignment=source, owner=owner,
                                      service=service, env=env, client=TerminalIssuerClient(credential))
            finally:
                await service.close()
                owner.close()
    return start


def envelope(r, operation, arguments, *, token=None):
    return {"version": 1, "operation": operation, "token": token or r.credential.token,
            "scope": r.credential.public_scope(), "arguments": arguments}


def test_the_coordinator_checkpoints_and_restores_over_its_socket(rig):
    async def scenario():
        async with rig() as r:
            r.env.files = {"made_by_run_1": "x"}
            record = await asyncio.to_thread(r.client.checkpoint, "cp1", timeout_seconds=30)
            assert record == {"checkpoint_id": "cp1", "files": ["made_by_run_1"], "bytes": 7}
            assert await asyncio.to_thread(r.client.checkpoint, "cp1", timeout_seconds=30) == record
            r.env.files = {"made_by_run_1": "x", "broken_by_run_2": "y"}
            restored = await asyncio.to_thread(r.client.restore, "undo_1", "cp1", timeout_seconds=30)
            assert restored == {"operation_id": "undo_1", "checkpoint_id": "cp1", "files": ["made_by_run_1"],
                                "exact": True}
            assert r.env.files == {"made_by_run_1": "x"} and len(r.env.taken) == len(r.env.restored) == 1
            with pytest.raises(TerminalConflict):
                await asyncio.to_thread(r.client.restore, "undo_2", "cp404", timeout_seconds=30)

    asyncio.run(scenario())


def test_a_checkpoint_longer_than_the_handshake_is_still_answered(rig, monkeypatch):
    monkeypatch.setattr("taste.brains.terminal_service.HANDSHAKE_SECONDS", 0.2)

    async def scenario():
        async with rig() as r:
            r.env.hold.clear()
            asking = asyncio.create_task(asyncio.to_thread(r.client.checkpoint, "cp1", timeout_seconds=30))
            assert await asyncio.to_thread(r.env.inside.wait, 5)
            await asyncio.sleep(0.5)
            r.env.hold.set()
            assert (await asking)["checkpoint_id"] == "cp1"

    asyncio.run(scenario())


def test_worker_credentials_and_forged_requests_cannot_checkpoint_or_restore(rig):
    async def scenario():
        async with rig() as r:
            spec = SimpleNamespace(assignment=r.assignment, prepared_state_id="a" * 40,
                                   run_id=assignment_run_id(r.assignment))
            worker = await asyncio.to_thread(r.client, spec)
            for operation, arguments in (("checkpoint", {"checkpoint_id": "cp1"}),
                                         ("restore", {"operation_id": "undo_1", "checkpoint_id": "cp1"})):
                as_worker = {"version": 1, "token": worker.token, "grant": worker.grant.to_dict(),
                             "operation": operation, "arguments": arguments}
                assert await raw_reply(r.credential, as_worker) == {"version": 1, "status": "denied"}
                forged = envelope(r, operation, arguments, token=secrets.token_hex(32))
                assert await raw_reply(r.credential, forged) == {"version": 1, "status": "denied"}
            for arguments in ({"checkpoint_id": "../x"}, {"checkpoint_id": 7}, {"checkpoint_id": "cp1", "more": "x"}, []):
                reply = await raw_reply(r.credential, envelope(r, "checkpoint", arguments))
                assert reply == {"version": 1, "status": "denied"}
            assert not r.env.taken and not r.env.restored
            assert not [kind for kind, _ in r.owner.events() if kind.startswith(("checkpoint", "restore"))]
            # The worker's own grant still runs commands.
            assert await TerminalClient(worker).ping() == "ready"

    asyncio.run(scenario())


def test_a_lost_reply_is_read_again_by_its_id_and_never_restores_twice(rig, monkeypatch):
    async def scenario():
        async with rig() as r:
            r.env.files = {"good": "x"}
            await asyncio.to_thread(r.client.checkpoint, "cp1", timeout_seconds=30)
            r.env.files = {"good": "x", "bad": "y"}
            r.env.hold.clear()
            r.env.inside.clear()
            with pytest.raises(TerminalUnavailable):
                await asyncio.to_thread(r.client.restore, "undo_1", "cp1", timeout_seconds=0.3)
            assert await asyncio.to_thread(r.env.inside.wait, 5)
            r.env.hold.set()
            for _ in range(100):
                if r.env.restored:
                    break
                await asyncio.sleep(0.05)
            again = await asyncio.to_thread(r.client.restore, "undo_1", "cp1", timeout_seconds=30)
            assert again["exact"] is True and len(r.env.restored) == 1 and r.env.files == {"good": "x"}

    asyncio.run(scenario())


def test_the_client_refuses_bad_identifiers_and_allowances_before_connecting(rig):
    async def scenario():
        async with rig() as r:
            with pytest.raises(ValueError):
                await asyncio.to_thread(r.client.checkpoint, "a b", timeout_seconds=30)
            with pytest.raises(ValueError):
                await asyncio.to_thread(r.client.restore, "undo_1", "cp1", timeout_seconds=float("inf"))
            with pytest.raises(ValueError):
                await asyncio.to_thread(r.client.checkpoint, "cp1", timeout_seconds=0)
            assert not [kind for kind, _ in r.owner.events() if kind != "created"]

    asyncio.run(scenario())
