"""Subprocess launch, process deadlines and fail-closed Azure startup/shutdown."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from taste.brains import azure_worker_entrypoint as entrypoint
from taste.brains.azure_worker_launch import worker_command
from taste.brains.communication import Communicator, Message
from taste.brains.monitor import MonitorBrain, monitor_state_suffix
from taste.brains.records import contract_digest
from taste.brains.responses_session import ResponsesSession
from taste.brains.supervisor import CentralSupervisor, DeliveryRejected, SubprocessLauncher
from taste.brains.worker_admission import WorkerExitCode
from taste.brains.worker_protocol import assignment_run_id
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_azure_worker_policy import environment
from tests.test_azure_worker_runtime import install, report, run
from tests.test_azure_worker_runtime import worker as _worker
from tests.test_worker_admission import install_assignment
from tests.test_worker_admission import launch as _launch

sdk_transport = _sdk_transport
launch = _launch
worker = _worker

# Only the network transport is substituted in the actual child. The launcher,
# isolated module command, SDK, worker and monitor all run normally. Block any
# accidental import of the historical Claude harness's SDK in that process.
BOOTSTRAP = """
import importlib.abc, httpx
class NoClaude(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('claude_agent_sdk'):
            raise AssertionError('Azure process imported the Claude SDK')
sys.meta_path.insert(0, NoClaude())
from tests.test_azure_worker_runtime import install
OriginalClient = httpx.Client
def network(handler):
    class Client(OriginalClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, transport=httpx.MockTransport(handler))
    httpx.Client = Client
    return [], []
install(network)
"""


def assigned(supervisor, worker):
    source = worker.assignment
    contract = replace(source.contract, identity="process-worker", inputs=())
    return replace(source, contract=contract, contract_digest=contract_digest(contract),
                   base_state_id=supervisor.integration.head.id, inputs=())


@pytest.mark.parametrize("late_feedback", [False, True])
def test_real_subprocess_has_no_claude_dependency_and_delivers_certified_product(worker, late_feedback):
    def command(spec):
        argv = list(worker_command(spec, repo_root=worker.store.root, session=worker.store.session))
        argv[3] = argv[3].replace("import runpy;", BOOTSTRAP + "\nimport runpy;", 1)
        return argv

    launcher = SubprocessLauncher(command, env=environment())
    supervisor = CentralSupervisor(worker.store, launcher=launcher)
    assignment = assigned(supervisor, worker)
    prepared = supervisor.prepare(assignment, wall_timeout_seconds=30)
    try:
        supervisor.start(prepared.run_id, active_generation=1)
        terminal = supervisor.wait(prepared.run_id, active_generation=1)
        assert terminal.reaped and terminal.exit_code == 0, terminal.to_dict()
        result = supervisor.collect(prepared.run_id, active_generation=1)
        assert result.completed and result.cost_usd == pytest.approx(0.001464)
        assert result.metadata["harness"] == "azure-responses/1"
        if late_feedback:
            Communicator(worker.store).send(Message.create(
                idempotency_key="after-report", kind="feedback", sender="central",
                recipient=assignment.worker, generation=assignment.generation,
                payload={"instruction": "additional requirement after worker stopped"}))
            with pytest.raises(DeliveryRejected, match="unaccepted feedback"):
                supervisor.deliver(prepared.run_id, active_generation=1)
            assert supervisor.integration.head.read("output.txt") is None
            return
        # A forged notification is insufficient: delivery checks the separate
        # exact-run monitor sidecar, not a legacy contract-only certificate.
        scoped = worker.store.sidecar("monitor", assignment.worker,
                                       monitor_state_suffix(assignment.contract_digest, prepared.run_id))
        saved = scoped.read_bytes()
        scoped.unlink()
        legacy = worker.store.sidecar("monitor", assignment.worker,
                                       monitor_state_suffix(assignment.contract_digest))
        legacy.write_bytes(saved)
        with pytest.raises(DeliveryRejected, match="unavailable"):
            supervisor.deliver(prepared.run_id, active_generation=1)
        scoped.write_bytes(saved)
        assert supervisor.deliver(prepared.run_id, active_generation=1).ok
        assert supervisor.integration.head.read("output.txt") == "correct"
        assert worker.store.view(assignment.worker).holder is None
    finally:
        if not supervisor.get(prepared.run_id).terminal:
            supervisor.stop(prepared.run_id, "test_cleanup")


@pytest.mark.parametrize("outer_seconds", [10, 300])
def test_supervisor_persists_earliest_goal_or_assignment_deadline(worker, outer_seconds):
    from tests.test_brains_supervisor import FakeLauncher

    supervisor = CentralSupervisor(worker.store, launcher=FakeLauncher())
    assignment = assigned(supervisor, worker)
    prepared = supervisor.prepare(assignment, wall_timeout_seconds=600)
    outer = datetime.now(UTC) + timedelta(seconds=outer_seconds)
    spawned = supervisor.start(prepared.run_id, active_generation=1, deadline_at=outer)
    expected = min(outer.timestamp(), assignment.resources["azure_openai"]["deadline_unix"])
    assert datetime.fromisoformat(spawned.deadline_at).timestamp() == pytest.approx(expected, rel=0, abs=0.002)
    # A later caller cannot extend the durable process deadline.
    again = supervisor.poll(prepared.run_id, active_generation=1, deadline_at=outer + timedelta(hours=1))
    assert again.deadline_at == spawned.deadline_at


def test_expired_assignment_does_not_initialize_journals_or_publish_readiness(worker, sdk_transport):
    raw = dict(worker.assignment.resources["azure_openai"])
    raw["deadline_unix"] = 1
    worker.assignment = replace(worker.assignment, resources={**worker.assignment.resources, "azure_openai": raw})
    install_assignment(worker, worker.assignment)
    sent, _, _ = install(sdk_transport)
    assert run(worker) == WorkerExitCode.INPUT_REJECTED
    assert not entrypoint.run_directory(worker.store, assignment_run_id(worker.assignment)).exists()
    assert not Path(worker.environ["TASTE_WORKER_READY_PATH"]).exists()
    assert sent == []


@pytest.mark.parametrize("role", ["worker", "monitor"])
def test_sdk_preflight_failure_never_announces_readiness_or_dispatches(worker, sdk_transport, monkeypatch, role):
    original = ResponsesSession.ensure_ready

    def unavailable(session):
        if session.binding.role == role:
            raise RuntimeError("unavailable SDK, no dispatch")
        return original(session)
    monkeypatch.setattr(ResponsesSession, "ensure_ready", unavailable)
    sent, _, _ = install(sdk_transport)
    assert run(worker) == WorkerExitCode.RUNTIME_FAILURE
    assert sent == []
    assert not Path(worker.environ["TASTE_WORKER_READY_PATH"]).exists()
    assert worker.store.view("worker").holder is None


def test_cleanup_failure_retains_branch_until_process_quarantine_ends(worker, sdk_transport, monkeypatch):
    original = ResponsesSession.close

    def unavailable(session):
        if session.binding.role == "worker":
            raise RuntimeError("unconfirmed session close")
        return original(session)
    sent, _, _ = install(sdk_transport)
    previous = len(entrypoint._QUARANTINED_RESOURCES)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(ResponsesSession, "close", unavailable)
            assert run(worker) == WorkerExitCode.SHUTDOWN_UNCONFIRMED
        assert report(worker).completed  # report alone cannot authorize delivery
        assert len(sent) == 4
        assert len(entrypoint._QUARANTINED_RESOURCES) == previous + 1
        assert worker.store.view("worker").holder is not None
    finally:
        for session, branch, _store in entrypoint._QUARANTINED_RESOURCES[previous:]:
            session.close()
            branch.close()
        del entrypoint._QUARANTINED_RESOURCES[previous:]
    assert worker.store.view("worker").holder is None


def test_monitor_rejects_state_with_another_run_inside_the_scoped_file(worker):
    monitor = MonitorBrain(worker.store, worker.assignment.contract, None, run_id="original")
    monitor._save_state()
    raw = json.loads(monitor._state_path().read_text())
    raw["run_id"] = "another"
    monitor._state_path().write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="different run"):
        MonitorBrain(worker.store, worker.assignment.contract, None, run_id="original")
