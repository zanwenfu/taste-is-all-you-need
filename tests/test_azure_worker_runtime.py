"""Actual entrypoint -> SDK wire -> effects -> feedback -> certification/report."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from taste.brains.azure_worker_entrypoint import execute_worker, run_directory
from taste.brains.azure_worker_launch import worker_command
from taste.brains.communication import Communicator, Message
from taste.brains.monitor import MonitorBrain
from taste.brains.monitor_judge import TERMINAL_JUDGEMENT_SCHEMA
from taste.brains.records import WorkerReport, contract_digest
from taste.brains.responses_monitor import ResponsesMonitorJudge
from taste.brains.worker_admission import WorkerExitCode
from taste.brains.worker_protocol import WORKER_REPORT_PATH, assignment_run_id
from tests.test_azure_openai import httpx
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_azure_worker_policy import assignment as policy_assignment
from tests.test_azure_worker_policy import environment
from tests.test_openai_responses import function_call, message, response
from tests.test_responses_feedback import claim
from tests.test_responses_monitor import verdict
from tests.test_worker_admission import install_assignment
from tests.test_worker_admission import launch as _launch

sdk_transport = _sdk_transport
launch = _launch


@pytest.fixture
def worker(launch):
    template = policy_assignment(monitor_max_calls=20)
    launch.assignment = replace(launch.assignment, resources=template.resources)
    install_assignment(launch, launch.assignment)
    launch.config = replace(launch.config, monitor_max_tokens=256, monitor_budget_usd=20)
    launch.environ.update(environment())
    return launch


def report(worker):
    return WorkerReport.from_json(worker.store.view(worker.assignment.worker).head.read(WORKER_REPORT_PATH))


def run(worker, **kwargs):
    return asyncio.run(execute_worker(worker.config, store=worker.store, environ=worker.environ, **kwargs))


def send(worker, key):
    return Communicator(worker.store).send(Message.create(
        idempotency_key=key, kind="feedback", sender="central", recipient=worker.assignment.worker,
        generation=1, payload={"instruction": "handle " + key}))


def accepted(payload):
    inbox, verdicts = [], {}
    for item in payload["input"]:
        if item.get("role") != "user":
            continue
        content = item["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content)
        try:
            value = json.loads(content)
        except ValueError:
            continue
        if value.get("kind") == "inbox":
            inbox.append(value["inbox_id"])
        if value.get("kind") == "verdicts":
            for row in value["states"]:
                verdicts[row["state_id"]] = max(verdicts.get(row["state_id"], 0), row["through"])
    return claim(*inbox, verdicts=verdicts)


def install(sdk_transport, *, worker_reply=None, monitor_reply=None, hook=None):
    calls = {"worker": 0, "monitor": 0, "terminal": 0}

    def handle(wire):
        payload = json.loads(wire.content)
        role = ("worker" if "tools" in payload else
                "terminal" if TERMINAL_JUDGEMENT_SCHEMA in wire.content.decode() else "monitor")
        calls[role] += 1
        if hook is not None:
            hook(role, calls[role], payload)
        if role == "worker":
            if worker_reply is not None:
                output = worker_reply(calls[role], payload)
            elif calls[role] == 1:
                output = [function_call(json.dumps({"path": "output.txt", "content": "correct", "executable": False}),
                                        name="write_artifact")]
            else:
                output = [message(json.dumps(accepted(payload)))]
        else:
            text = monitor_reply(role, payload) if monitor_reply else verdict(terminal=role == "terminal")
            output = [message(text)]
        return httpx.Response(200, json=response(model="gpt-6-sol", output=output),
                              headers={"x-ms-served-model": policy_assignment().model})
    sent, options = sdk_transport(handle)
    return sent, calls, options


def test_complete_worker_uses_azure_and_certifies_exact_artifact_after_readiness(worker, sdk_transport):
    def before(role, number, payload):
        if role == "worker" and number == 1:
            ready = json.loads(Path(worker.environ["TASTE_WORKER_READY_PATH"]).read_text())
            assert ready["run_id"] == assignment_run_id(worker.assignment)
            assert ready["pid"] == os.getpid()
            directory = run_directory(worker.store, ready["run_id"])
            assert (directory / "worker" / "calls.sqlite3").is_file()
            assert (directory / "monitor" / "calls.sqlite3").is_file()
    sent, calls, _ = install(sdk_transport, hook=before)
    assert run(worker) == WorkerExitCode.COMPLETED
    result = report(worker)
    assert result.completed and not result.uncertain
    assert result.metadata["harness"] == "azure-responses/1"
    assert calls == {"worker": 2, "monitor": 1, "terminal": 1}
    assert result.cost_usd == pytest.approx(4 * 0.000366)
    work = worker.store.state(result.final_state_id)
    assert work.read("output.txt") == "correct"
    assert work.read(WORKER_REPORT_PATH) is None
    assert result.outputs[0].blob_id == work.blob("output.txt")
    assert result.metadata["monitor"]["current_state"] == work.id
    assert result.metadata["monitor"]["terminal_assessment"]["state_id"] == work.id
    assert all(wire.headers["authorization"] == "Bearer azure-secret" for wire in sent)
    assert all(str(wire.url).startswith(environment()["AZURE_OPENAI_BASE_URL"]) for wire in sent)
    assert worker.store.view("worker").holder is None


def test_feedback_arriving_during_paid_turn_is_seen_and_explicitly_accepted(worker, sdk_transport):
    first = send(worker, "first")
    later = []

    def before(role, number, payload):
        if role == "worker" and number == 2:
            later.append(send(worker, "during-completion"))
    sent, calls, _ = install(sdk_transport, hook=before)
    assert run(worker) == WorkerExitCode.COMPLETED
    assert calls["worker"] == 3
    assert Communicator(worker.store).pending("worker") == ()
    turns = worker.store.state(report(worker).final_state_id).transcript.turns
    ids = {turn["message_id"] for turn in turns if turn.get("kind") == "inbox_accepted"}
    assert ids == {first.inbox_id, later[0].inbox_id}
    assert len(sent) < 10


@pytest.mark.parametrize("bad", ["invalid_claim", "forged_ack", "missing_output", "blocked", "continue_limit"])
def test_incomplete_workers_cannot_report_success(worker, sdk_transport, bad):
    def replies(number, payload):
        value = accepted(payload)
        if bad == "invalid_claim":
            return [message("not JSON")]
        if bad == "forged_ack":
            value["accepted_inbox_ids"] = ["a" * 40]
        if bad == "blocked":
            value["status"] = "blocked"
        if bad == "continue_limit":
            value["status"] = "continue"
        return [message(json.dumps(value))]
    sent, _, _ = install(sdk_transport, worker_reply=replies)
    assert run(worker) == WorkerExitCode.INCOMPLETE
    result = report(worker)
    assert not result.completed
    assert result.cost_usd is not None
    assert len(sent) <= 10
    assert worker.store.view("worker").holder is None


@pytest.mark.parametrize("phase", ["monitor", "terminal"])
def test_paid_invalid_monitor_reply_remains_known_cost_and_blocks_completion(worker, sdk_transport, phase):
    sent, _, _ = install(sdk_transport, monitor_reply=lambda role, _: (
        "bad paid reply" if role == phase else verdict(terminal=role == "terminal")))
    assert run(worker) == WorkerExitCode.INCOMPLETE
    result = report(worker)
    assert not result.completed and result.uncertain
    assert result.cost_usd == pytest.approx(len(sent) * 0.000366)
    assert result.metadata["monitor"]["cost_known"]


def test_lost_provider_reply_is_unknown_cost_and_never_retried(worker, sdk_transport):
    def fail(wire):
        raise httpx.ReadError("private Azure token must not appear in report", request=wire)
    sent, _ = sdk_transport(fail)
    assert run(worker) == WorkerExitCode.INCOMPLETE
    result = report(worker)
    assert result.cost_usd is None and not result.completed
    assert result.metadata["worker_accounting"]["unknown_calls"] == 1
    assert len(sent) == 1
    assert "private Azure token" not in result.to_json()


@pytest.mark.parametrize("phase", ["worker", "monitor", "terminal"])
@pytest.mark.parametrize("late_failure", [False, True])
def test_repeated_cancellation_waits_for_paid_thread_then_reports_without_more_calls(
        worker, sdk_transport, phase, late_failure):
    entered, release = threading.Event(), threading.Event()

    def block(role, number, payload):
        if role == phase:
            entered.set()
            assert release.wait(10)
            if late_failure:
                raise httpx.ReadError("late provider failure")
    sent, _, _ = install(sdk_transport, hook=block)

    async def scenario():
        task = asyncio.create_task(execute_worker(worker.config, store=worker.store, environ=worker.environ))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            count = len(sent)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert worker.store.view("worker").holder is not None
            release.set()
            assert await task == WorkerExitCode.INTERRUPTED
            assert len(sent) == count
        finally:
            release.set()
            if not task.done():
                await asyncio.wait((task,))
    asyncio.run(scenario())
    result = report(worker)
    assert not result.completed and result.uncertain
    assert (result.cost_usd is None) == late_failure
    assert worker.store.view("worker").holder is None


def test_inbox_race_during_terminal_certification_invalidates_completion(worker, sdk_transport):
    def race(role, number, payload):
        if role == "terminal":
            send(worker, "after-worker-stopped")
    install(sdk_transport, hook=race)
    assert run(worker) == WorkerExitCode.INCOMPLETE
    result = report(worker)
    assert result.metadata["monitor"]["current"] == "fine"
    assert "feedback_arrived_during_finalization" in result.uncertainty_reasons
    assert len(Communicator(worker.store).pending("worker")) == 1


@pytest.mark.parametrize("damage", ["existing", "partial", "missing_azure_key", "wrong_monitor", "pending_context"])
def test_bad_admission_never_dispatches_or_announces_readiness(worker, sdk_transport, damage):
    directory = run_directory(worker.store, assignment_run_id(worker.assignment))
    if damage in {"existing", "partial"}:
        directory.mkdir(mode=0o700)
        if damage == "partial":
            (directory / "worker").mkdir(mode=0o700)
    elif damage == "missing_azure_key":
        worker.environ.pop("AZURE_OPENAI_API_KEY")
    elif damage == "wrong_monitor":
        worker.config = replace(worker.config, monitor_max_tokens=257)
    else:
        branch = worker.store.branch("worker")
        branch.turn(kind="unknown_previous_work")
        branch.close()
    sent, _, _ = install(sdk_transport)
    assert run(worker) == WorkerExitCode.INPUT_REJECTED
    assert sent == []
    assert not Path(worker.environ["TASTE_WORKER_READY_PATH"]).exists()


def test_partial_journal_initialization_cannot_refresh_allowances(worker, sdk_transport, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("crash between worker and monitor journals")
    sent, _, _ = install(sdk_transport)
    with monkeypatch.context() as patch:
        patch.setattr(ResponsesMonitorJudge, "create", fail)
        assert run(worker) == WorkerExitCode.RUNTIME_FAILURE
    assert run(worker) == WorkerExitCode.INPUT_REJECTED
    assert not Path(worker.environ["TASTE_WORKER_READY_PATH"]).exists()
    assert sent == []


def test_azure_entrypoint_import_and_cli_need_no_claude_sdk(worker):
    code = '''
import importlib.abc, sys
class NoClaude(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('claude_agent_sdk'):
            raise AssertionError('Claude SDK imported by Azure harness')
sys.meta_path.insert(0, NoClaude())
from taste.brains.azure_worker_entrypoint import main
raise SystemExit(main(['--help']))
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "Azure Responses worker" in result.stdout


def test_run_scoped_monitor_does_not_inherit_previous_attempt_or_legacy_findings(worker):
    first = MonitorBrain(worker.store, worker.assignment.contract, None, run_id="first")
    first.state.fingerprints.append("previous-work")
    first._save_state()
    second = MonitorBrain(worker.store, worker.assignment.contract, None, run_id="second")
    legacy = MonitorBrain(worker.store, worker.assignment.contract, None)
    assert second.state.fingerprints == legacy.state.fingerprints == []
    assert MonitorBrain(worker.store, worker.assignment.contract, None, run_id="first").state.fingerprints == ["previous-work"]


def test_supervisor_collects_and_delivers_actual_azure_report(worker, sdk_transport):
    from taste.brains.supervisor import CentralSupervisor, ProcessExit
    from tests.test_brains_supervisor import FakeLauncher

    launcher = FakeLauncher()
    supervisor = CentralSupervisor(worker.store, launcher=launcher)
    source = worker.assignment
    contract = replace(source.contract, identity="integrated-worker", inputs=())
    assignment = replace(source, contract=contract, contract_digest=contract_digest(contract),
                         base_state_id=supervisor.integration.head.id, inputs=())
    prepared = supervisor.prepare(assignment, wall_timeout_seconds=30)
    spawned = supervisor.start(prepared.run_id, active_generation=1)
    spec = supervisor._spec(spawned)
    command = worker_command(spec, repo_root=worker.store.root, session=worker.store.session)
    assert "azure_worker_entrypoint" in command[3]
    worker.assignment = assignment
    worker.config = replace(worker.config, worker=assignment.worker, prepared_state_id=spec.prepared_state_id)
    worker.environ.update(TASTE_WORKER_RUN_ID=prepared.run_id, TASTE_WORKER_READY_PATH=str(spec.readiness_path))
    install(sdk_transport)
    assert run(worker) == WorkerExitCode.COMPLETED
    launcher.handle.is_ready = True
    launcher.handle.exit = ProcessExit(exit_code=0, reaped=True)
    supervisor.poll(prepared.run_id, active_generation=1)
    result = supervisor.collect(prepared.run_id, active_generation=1)
    assert result.completed
    delivered = supervisor.deliver(prepared.run_id, active_generation=1)
    assert not delivered.conflicts
    assert supervisor.integration.head.read("output.txt") == "correct"
    assert supervisor.integration.head.read(WORKER_REPORT_PATH) is None
