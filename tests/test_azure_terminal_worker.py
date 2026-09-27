"""Prepared Azure workers use their private terminal grant through real RPC."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from taste.brains.azure_worker_entrypoint import execute_worker, run_directory
from taste.brains.azure_worker_launch import worker_command
from taste.brains.records import contract_digest
from taste.brains.supervisor import CentralSupervisor, SubprocessLauncher
from taste.brains.terminal_broker import TerminalBinding, TerminalBroker, TerminalResult
from taste.brains.terminal_service import TerminalCredential, TerminalService
from taste.brains.terminal_worker_policy import (
    TERMINAL_POLICY_KEY,
    TerminalWorkerPolicy,
    credential_path,
    install_terminal_credential,
    load_terminal_client,
)
from taste.brains.worker_admission import EntrypointInputError, WorkerExitCode
from taste.brains.worker_protocol import assignment_run_id
from tests.terminal_worker_wire import replies
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_azure_worker_policy import environment
from tests.test_azure_worker_process import BOOTSTRAP
from tests.test_azure_worker_runtime import install, report
from tests.test_azure_worker_runtime import worker as _worker
from tests.test_goal_cancellation import wait_event
from tests.test_terminal_broker import Environment
from tests.test_worker_admission import install_assignment
from tests.test_worker_admission import launch as _launch

sdk_transport = _sdk_transport
launch = _launch
worker = _worker


@asynccontextmanager
async def terminal(worker, tmp_path, *, assignment=None):
    env = Environment()
    source = worker.assignment if assignment is None else assignment
    binding = TerminalBinding("terminal_trial", env.environment_id, source.resources["azure_openai"]["deadline_unix"], 10)
    policy = TerminalWorkerPolicy(binding, 5)
    updated = replace(source, resources={**source.resources, TERMINAL_POLICY_KEY: policy.to_dict()})
    if assignment is None:
        worker.assignment = updated
        install_assignment(worker, updated)
    owner = TerminalBroker.create(tmp_path / "terminal-ledger", binding, env)
    with tempfile.TemporaryDirectory(prefix="taste-azure-rpc-", dir="/tmp") as directory:
        credential = TerminalCredential(str(Path(directory) / "service" / "terminal.sock"),
            os.geteuid(), os.geteuid(), policy.grant(updated), "e" * 64)
        service = TerminalService(owner, [credential])
        await service.start()
        if assignment is None:
            install_terminal_credential(worker.store, updated, worker.config.prepared_state_id, credential)
        try:
            yield SimpleNamespace(env=env, owner=owner, service=service, credential=credential,
                                  assignment=updated, policy=policy)
        finally:
            env.release.set()
            env.stop_release.set()
            env.stop_error = None
            await service.close()
            owner.close()


def test_worker_executes_task_terminal_then_certifies_artifact_without_leaking_credentials(worker, sdk_transport, tmp_path):
    sent, calls, _ = install(sdk_transport, worker_reply=replies)

    async def scenario():
        async with terminal(worker, tmp_path) as t:
            t.env.result = TerminalResult(0, b"verified task evidence\xff", b"", 9000, 0)
            assert await execute_worker(worker.config, store=worker.store, environ=worker.environ) == WorkerExitCode.COMPLETED
            result = report(worker)
            assert result.completed and not result.uncertain
            assert len(t.env.calls) == 1 and not t.env.stopped
            assert t.env.calls[0].actor_id == t.credential.grant.actor_id
            assert t.owner.lookup(t.env.calls[0]) == t.env.result
            model_wire = "\n".join(w.content.decode() for w in sent)
            assert "terminal_exec" in model_wire and "9000" in model_wire
            assert t.credential.token not in model_wire + result.to_json()
            assert t.credential.socket_path not in model_wire + result.to_json()
            transcript = worker.store.state(result.final_state_id).transcript.turns
            assert t.credential.token not in json.dumps(transcript)
            assert worker.store.state(result.final_state_id).read("output.txt") == "correct"
            assert calls["worker"] == 3
    asyncio.run(scenario())


@pytest.mark.parametrize("damage", ["missing", "mode", "symlink", "hardlink", "partial", "prepared", "actor", "token", "oversize"])
def test_bad_terminal_launch_material_blocks_before_readiness_or_paid_call(worker, sdk_transport, tmp_path, damage):
    sent, _, _ = install(sdk_transport, worker_reply=replies)

    async def scenario():
        async with terminal(worker, tmp_path) as t:
            path = credential_path(worker.store, worker.assignment)
            original = path.read_bytes()
            if damage == "missing":
                path.unlink()
            elif damage == "mode":
                path.chmod(0o644)
            elif damage == "symlink":
                saved = path.with_suffix(".saved")
                path.rename(saved)
                path.symlink_to(saved)
            elif damage == "hardlink":
                os.link(path, path.with_suffix(".extra"))
            elif damage == "partial":
                path.write_bytes(b"{")
            elif damage == "oversize":
                path.write_bytes(b"x" * 8193)
            else:
                raw = json.loads(original)
                if damage == "prepared":
                    raw["prepared_state_id"] = "f" * 40
                elif damage == "actor":
                    raw["credential"]["grant"]["actor_id"] = "another_worker"
                else:
                    raw["credential"]["token"] = "f" * 64
                path.write_text(json.dumps(raw))
            assert await execute_worker(worker.config, store=worker.store, environ=worker.environ) == WorkerExitCode.INPUT_REJECTED
            assert not sent and not t.env.calls
            assert not run_directory(worker.store, assignment_run_id(worker.assignment)).exists()
            assert not Path(worker.environ["TASTE_WORKER_READY_PATH"]).exists()
    asyncio.run(scenario())


def test_terminal_credentials_cannot_rotate_or_move_to_a_new_attempt(worker, tmp_path):
    async def scenario():
        async with terminal(worker, tmp_path) as t:
            prepared = worker.config.prepared_state_id
            path = install_terminal_credential(worker.store, worker.assignment, prepared, t.credential)
            original = path.read_bytes()
            with pytest.raises(EntrypointInputError):
                install_terminal_credential(worker.store, worker.assignment, prepared,
                                            replace(t.credential, token="f" * 64))
            assert path.read_bytes() == original
            different = replace(worker.assignment, attempt=worker.assignment.attempt + 1)
            assert t.policy.grant(different).actor_id != t.credential.grant.actor_id
            with pytest.raises(EntrypointInputError):
                load_terminal_client(worker.store, different, prepared)
    asyncio.run(scenario())


def test_cancelled_worker_waits_for_remote_terminal_settlement_and_reports_uncertainty(worker, sdk_transport, tmp_path):
    sent, _, _ = install(sdk_transport, worker_reply=replies)

    async def scenario():
        async with terminal(worker, tmp_path) as t:
            t.env.release.clear()
            t.env.stop_release.clear()
            active = asyncio.create_task(execute_worker(worker.config, store=worker.store, environ=worker.environ))
            await wait_event(t.env.entered)
            active.cancel()
            await wait_event(t.env.stop_entered)
            for _ in range(3):
                active.cancel()
                await asyncio.sleep(0.01)
                assert not active.done()
            t.env.stop_release.set()
            assert await active == WorkerExitCode.INTERRUPTED
            assert t.env.stopped and t.owner.phase == "stopped"
            result = report(worker)
            assert not result.completed and result.uncertain
            assert len(t.env.calls) == len(sent) == 1
            assert worker.store.view(worker.assignment.worker).holder is None
    asyncio.run(scenario())


def test_real_azure_subprocess_uses_private_grant_and_delivers_after_terminal_receipt(worker, tmp_path):
    async def scenario():
        bootstrap = BOOTSTRAP.replace("install(network)",
            "from tests.terminal_worker_wire import replies\ninstall(network, worker_reply=replies)")
        bootstrap += "\nsys.modules.pop('taste.brains.azure_worker_entrypoint', None)\n"

        def command(spec):
            argv = list(worker_command(spec, repo_root=worker.store.root, session=worker.store.session,
                                       terminal_credential_provider=lambda _spec: t.credential))
            argv[3] = argv[3].replace("import runpy;", bootstrap + "\nimport runpy;", 1)
            assert t.credential.token not in " ".join(argv)
            return argv

        launcher = SubprocessLauncher(command, env=environment())
        supervisor = CentralSupervisor(worker.store, launcher=launcher)
        source = worker.assignment
        contract = replace(source.contract, identity="terminal-process-worker", inputs=())
        assignment = replace(source, contract=contract, contract_digest=contract_digest(contract),
                             base_state_id=supervisor.integration.head.id, inputs=())
        async with terminal(worker, tmp_path, assignment=assignment) as t:
            prepared = supervisor.prepare(t.assignment, wall_timeout_seconds=30)
            try:
                await asyncio.to_thread(supervisor.start, prepared.run_id, active_generation=1)
                finished = await asyncio.to_thread(supervisor.wait, prepared.run_id, active_generation=1)
                assert finished.reaped and finished.exit_code == 0, finished.to_dict()
                result = supervisor.collect(prepared.run_id, active_generation=1)
                assert result.completed and not result.uncertain
                assert supervisor.deliver(prepared.run_id, active_generation=1).ok
                assert supervisor.integration.head.read("output.txt") == "correct"
                assert len(t.env.calls) == 1 and not t.env.stopped
                assert t.owner.lookup(t.env.calls[0]) == t.env.result
            finally:
                if not supervisor.get(prepared.run_id).terminal:
                    await asyncio.to_thread(supervisor.stop, prepared.run_id, "test_cleanup")
    asyncio.run(scenario())
