"""Prepared goal -> private coordinator credential -> broker -> actual worker."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import struct
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from taste.brains import azure_worker_launch
from taste.brains.azure_central_host import compose_azure_central_runtime
from taste.brains.azure_goal_credentials import (
    GOAL_CREDENTIAL_NAME,
    encode_azure_goal_credentials,
    load_azure_goal_credentials,
)
from taste.brains.azure_goal_entrypoint import execute_azure_goal, prepare_azure_goal_process
from taste.brains.goal_entrypoint import GoalInputError
from taste.brains.terminal_broker import TerminalBroker
from taste.brains.terminal_issuer import TerminalIssuerClient, TerminalIssuerCredential
from taste.brains.terminal_service import TerminalService, TerminalUnavailable
from tests.test_azure_central_host import goal as _goal
from tests.test_azure_central_host import install_planner
from tests.test_azure_central_host import policy as _policy
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_azure_terminal_policy import bound
from tests.test_azure_worker_process import BOOTSTRAP
from tests.test_goal_cancellation import wait_event
from tests.test_terminal_broker import Environment
from tests.test_thread_shutdown_ownership import cancel_loop_tasks

goal, policy, sdk_transport = _goal, _policy, _sdk_transport


@pytest.fixture
def prepared(tmp_path, policy, goal):
    admitted = bound(replace(policy, deadline_unix=time.time() + 45))
    root = tmp_path / "repo"
    root.mkdir()
    config = prepare_azure_goal_process(root, "private-goal", goal, policy=admitted, environment={},
                                        max_generations=3, wall_clock_seconds=60)
    return config, admitted


def credential(config, policy, socket_path):
    return TerminalIssuerCredential(socket_path, os.geteuid(), os.geteuid(),
        hashlib.sha256(config.to_bytes()).hexdigest(), policy.terminal, secrets.token_hex(32))


def install(directory, config, issuer):
    path = directory / GOAL_CREDENTIAL_NAME
    path.write_bytes(encode_azure_goal_credentials(config, "azure-test-only", terminal_issuer=issuer))
    path.chmod(0o400)
    return {"CREDENTIALS_DIRECTORY": str(directory)}


def test_prepared_goal_loads_private_issuer_and_delivers_real_worker_product(prepared, tmp_path, sdk_transport, monkeypatch):
    config, policy = prepared
    sent, payloads = install_planner(sdk_transport)
    original = azure_worker_launch.worker_command
    bootstrap = BOOTSTRAP.replace("install(network)",
        "from tests.terminal_worker_wire import replies\ninstall(network, worker_reply=replies)")
    bootstrap += "\nsys.modules.pop('taste.brains.azure_worker_entrypoint', None)\n"
    bootstrap += "\nimport os; assert 'CREDENTIALS_DIRECTORY' not in os.environ\n"

    def command(*args, **kwargs):
        argv = list(original(*args, **kwargs))
        argv[3] = argv[3].replace("import runpy;", bootstrap + "\nimport runpy;", 1)
        return tuple(argv)

    monkeypatch.setattr(azure_worker_launch, "worker_command", command)

    async def scenario():
        env = Environment()
        owner = TerminalBroker.create(tmp_path / "ledger", policy.terminal.binding, env)
        with tempfile.TemporaryDirectory(prefix="taste-goal-creds-", dir="/tmp") as temporary:
            directory = Path(temporary)
            issuer = credential(config, policy, str(directory / "service/terminal.sock"))
            service = TerminalService(owner, issuer=issuer)
            await service.start()
            source = install(directory, config, issuer)
            monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(directory))
            try:
                outcome = await execute_azure_goal(config, systemd_credentials=True, environment=source)
                assert outcome.complete and outcome.budget.enforceable, outcome.to_dict()
                assert len(service._issued) == len(env.calls) == 1 and len(sent) == 2
                public = json.dumps(payloads) + config.to_bytes().decode() + json.dumps(outcome.to_dict())
                assert issuer.token not in public and "azure-test-only" not in public
                for _prepared, issued in service._issued.values():
                    assert issued.token not in public and issued.grant.actor_id == env.calls[0].actor_id
                # Delete the service copy: settlement remains credential-free,
                # and neither execution nor paid accounting is repeated.
                (directory / GOAL_CREDENTIAL_NAME).unlink()
                assert await execute_azure_goal(config, mode="settle", systemd_credentials=True, environment={}) == outcome
                with compose_azure_central_runtime(config.repo_root, config.session, config.goal, policy=policy,
                                                   environment={}, settlement_only=True) as host:
                    assert host.integration.head.read("output.txt") == "correct"
                    assert all(run.reaped for run in host.supervisor.runs())
                assert len(env.calls) == 1 and len(sent) == 2
            finally:
                await service.close()
                owner.close()
    asyncio.run(scenario())


def test_unreachable_terminal_issuer_prevents_even_the_first_paid_plan(prepared, tmp_path, sdk_transport):
    config, policy = prepared
    sent, _ = install_planner(sdk_transport)
    source = install(tmp_path, config, credential(config, policy, "/tmp/taste-absent-goal-issuer.sock"))
    with pytest.raises(TerminalUnavailable):
        asyncio.run(execute_azure_goal(config, environment=source, systemd_credentials=True))
    assert not sent


@pytest.mark.parametrize("damage", ["goal", "uid", "policy", "missing"])
def test_private_issuer_must_bind_exact_goal_uid_and_terminal_scope(prepared, tmp_path, damage):
    config, policy = prepared
    source = install(tmp_path, config, credential(config, policy, "/tmp/taste-unused-issuer.sock"))
    path = tmp_path / GOAL_CREDENTIAL_NAME
    raw = json.loads(path.read_bytes())
    if damage == "goal":
        raw["terminal_issuer"]["input_sha256"] = "f" * 64
    elif damage == "uid":
        raw["terminal_issuer"]["coordinator_uid"] += 1
    elif damage == "policy":
        raw["terminal_issuer"]["policy"]["binding"]["max_commands"] += 1
    else:
        raw["terminal_issuer"] = None
    path.chmod(0o600)
    path.write_text(json.dumps(raw))
    path.chmod(0o400)
    with pytest.raises(GoalInputError, match="private Azure goal credentials were rejected"):
        load_azure_goal_credentials(config, source["CREDENTIALS_DIRECTORY"])


@pytest.mark.parametrize("whole_loop", [False, True])
def test_cancelled_preflight_retains_probe_thread_and_never_composes_a_host(prepared, tmp_path, monkeypatch, whole_loop):
    config, policy = prepared
    source = install(tmp_path, config, credential(config, policy, "/tmp/taste-unused-issuer.sock"))
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def ping(_self):
        entered.set()
        assert release.wait(5)
        finished.set()
        return "ready"

    async def cannot_compose(*_args, **_kwargs):
        pytest.fail("a cancelled readiness probe cannot dispatch a goal")

    monkeypatch.setattr(TerminalIssuerClient, "ping", ping)
    monkeypatch.setattr("taste.brains.azure_goal_entrypoint.execute_goal", cannot_compose)

    async def scenario():
        task = asyncio.create_task(execute_azure_goal(config, environment=source, systemd_credentials=True))
        try:
            await wait_event(entered)
            for _ in range(3):
                cancel_loop_tasks() if whole_loop else task.cancel()
                await asyncio.sleep(0.01)
                assert not task.done() and not finished.is_set()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    asyncio.run(scenario())


@pytest.mark.parametrize("damage", [None, "foreign_user", "extra_user", "group", "other", "write", "version", "truncated", "absent"])
def test_root_owned_systemd_copy_requires_exact_service_reader_acl(prepared, tmp_path, monkeypatch, damage):
    config, policy = prepared
    source = install(tmp_path, config, credential(config, policy, "/tmp/taste-unused-issuer.sock"))
    path = tmp_path / GOAL_CREDENTIAL_NAME
    path.chmod(0o440)
    original_stat = os.fstat
    file_info = path.stat()

    def root_owned(fd):
        info = original_stat(fd)
        if (info.st_dev, info.st_ino) == (file_info.st_dev, file_info.st_ino):
            values = list(info)
            values[4] = 0
            return os.stat_result(values)
        return info

    undefined = 2**32 - 1
    entries = [(1, 4, undefined), (2, 4, os.geteuid()), (4, 0, undefined),
               (16, 4, undefined), (32, 0, undefined)]
    if damage == "foreign_user":
        entries[1] = (2, 4, os.geteuid() + 1)
    elif damage == "extra_user":
        entries.insert(2, (2, 4, os.geteuid() + 1))
    elif damage == "group":
        entries[2] = (4, 4, undefined)
    elif damage == "other":
        entries[-1] = (32, 4, undefined)
    elif damage == "write":
        entries[1] = (2, 6, os.geteuid())
        entries[-2] = (16, 6, undefined)
    wire = struct.pack("<I", 3 if damage == "version" else 2)
    wire += b"".join(struct.pack("<HHI", *entry) for entry in entries)
    if damage == "truncated":
        wire = wire[:-1]

    def acl(fd, name):
        assert name == "system.posix_acl_access"
        assert original_stat(fd).st_ino == file_info.st_ino
        if damage == "absent":
            raise OSError("no ACL")
        return wire

    monkeypatch.setattr(os, "fstat", root_owned)
    monkeypatch.setattr(os, "getxattr", acl)
    if damage is None:
        environment, issuer = load_azure_goal_credentials(config, source["CREDENTIALS_DIRECTORY"])
        assert environment["AZURE_OPENAI_API_KEY"] == "azure-test-only" and issuer is not None
    else:
        with pytest.raises(GoalInputError, match="private Azure goal credentials were rejected"):
            load_azure_goal_credentials(config, source["CREDENTIALS_DIRECTORY"])
