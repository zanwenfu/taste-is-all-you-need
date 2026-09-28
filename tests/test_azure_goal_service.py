"""The outside observer cannot use a report before whole-scope drainage."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from dataclasses import replace

import pytest

from taste.brains.azure_goal_credentials import encode_azure_goal_credentials
from taste.brains.azure_goal_handoff import grading_ready, perform
from taste.brains.azure_goal_service import AzureGoalService
from taste.brains.goal_entrypoint import GoalInputError, GoalProcessInput
from taste.brains.process_scope import OwnedProcessScope
from taste.resources import ResourceCleanupError
from tests.test_azure_central_host import goal as _goal
from tests.test_azure_central_host import policy as _policy
from tests.test_azure_goal_handoff import preparation, valid_outcome, write_input
from tests.test_process_scope import Manager
from tests.test_thread_shutdown_ownership import cancel_loop_tasks

goal = _goal
policy = _policy


class CredentialManager(Manager):
    def start(self, unit, description, spec, *, credential_directory=None):
        self.credential_directory = credential_directory
        super().start(unit, description, spec)


@pytest.fixture
def service(tmp_path, goal, policy):
    path, digest = preparation(tmp_path, goal, policy)
    value = perform(path, digest, tmp_path / "prepared", operation="prepare")
    config = GoalProcessInput.from_bytes(json.dumps(value).encode())
    path = tmp_path / "goal.json"
    digest = write_input(path, config.to_bytes())
    instance = AzureGoalService.create(tmp_path / "owner", path, digest, tmp_path / "run",
        operation="run", uid=os.geteuid(), python_executable=sys.executable,
        runtime_seconds=60, credential=encode_azure_goal_credentials(config, "private-test-value"),
        manager=CredentialManager())
    return instance, config


def publish(service, result):
    from pathlib import Path

    output = Path(service.output_dir)
    output.mkdir(mode=0o700)
    binding = {"schema": "taste.brains/AzureGoalHandoff/1", "input_sha256": service.input_sha256,
               "operation": service.operation}
    for name, value in (("intent.json", binding), ("result.json", {**binding, "result": result})):
        path = output / name
        path.write_text(json.dumps(value))
        path.chmod(0o600)


def observe_only_after_drain(service, monkeypatch):
    from taste.brains import azure_goal_service as module

    calls = []
    reader = module.read_handoff

    def guarded(*args, **kwargs):
        assert not service.scope.manager.populated
        state = json.loads((service.scope.directory / "state.json").read_text())
        assert state["phase"] == "stopped" and state["termination"]["processes_stopped"]
        calls.append(args)
        return reader(*args, **kwargs)

    monkeypatch.setattr(module, "read_handoff", guarded)
    return calls


def reopen(service):
    return AzureGoalService(OwnedProcessScope(service.scope.directory, manager=service.scope.manager))


def test_run_and_reopen_observe_only_after_drain_and_never_launch_twice(service, monkeypatch):
    instance, config = service
    publish(instance, valid_outcome(config.goal))
    calls = observe_only_after_drain(instance, monkeypatch)
    result = asyncio.run(instance.run(timeout_seconds=0.01))
    assert result.termination["processes_stopped"] and grading_ready(result.value)
    assert asyncio.run(reopen(instance).recover()) == result
    assert len(calls) == 2 and len(instance.scope.manager.starts) == 1
    assert "private-test-value" not in (instance.scope.directory / "state.json").read_text()
    with pytest.raises(RuntimeError, match="already admitted"):
        asyncio.run(reopen(instance).run(timeout_seconds=0.01))
    assert len(calls) == 2 and len(instance.scope.manager.starts) == 1


def test_missing_result_does_not_convert_drain_receipt_to_success_or_replay(service, monkeypatch):
    instance, _ = service
    calls = observe_only_after_drain(instance, monkeypatch)
    with pytest.raises(FileNotFoundError):
        asyncio.run(instance.run(timeout_seconds=0.01))
    with pytest.raises(FileNotFoundError):
        asyncio.run(reopen(instance).recover())
    assert len(calls) == 2 and len(instance.scope.manager.starts) == 1


def test_unlaunched_scope_cannot_accept_planted_report(service, monkeypatch):
    instance, config = service
    publish(instance, valid_outcome(config.goal))
    calls = observe_only_after_drain(instance, monkeypatch)
    with pytest.raises(GoalInputError, match="unlaunched"):
        asyncio.run(reopen(instance).recover())
    assert not calls and not instance.scope.manager.starts


def test_ambiguous_launch_stays_fenced_even_when_result_file_exists(service, monkeypatch):
    instance, config = service
    publish(instance, valid_outcome(config.goal))
    calls = observe_only_after_drain(instance, monkeypatch)
    manager = instance.scope.manager
    manager.start_error = TimeoutError("lost launch acknowledgement")
    with pytest.raises(ResourceCleanupError):
        instance.scope.start()
    delayed = manager.units.pop(instance.scope.unit)
    manager.populated.clear()
    with pytest.raises(ResourceCleanupError, match="ambiguous"):
        asyncio.run(reopen(instance).recover())
    assert not calls
    manager.units[instance.scope.unit] = delayed
    manager.populated.add(instance.scope.unit)
    result = asyncio.run(reopen(instance).recover())
    assert grading_ready(result.value) and len(calls) == len(manager.starts) == 1


@pytest.mark.parametrize("whole_loop", [False, True])
def test_cancellation_cannot_read_report_before_stop_thread_finishes(service, monkeypatch, whole_loop):
    instance, config = service
    publish(instance, valid_outcome(config.goal))
    calls = observe_only_after_drain(instance, monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def block():
        entered.set()
        assert release.wait(5)

    instance.scope.manager.before_stop = block

    async def check():
        task = asyncio.create_task(instance.run(timeout_seconds=0.01))
        while not entered.is_set():
            await asyncio.sleep(0.001)
        if whole_loop:
            cancel_loop_tasks()
        else:
            task.cancel()
        try:
            await asyncio.sleep(0.03)
            assert not task.done() and not calls
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not calls and not instance.scope.manager.populated
        recovered = await reopen(instance).recover()
        assert grading_ready(recovered.value) and len(calls) == 1

    asyncio.run(check())


@pytest.mark.parametrize("damage", ["entrypoint", "output", "pythonpath", "cwd", "credential", "uid"])
def test_scope_binding_cannot_be_replaced_with_another_command_or_exchange(service, tmp_path, damage):
    instance, _ = service
    spec = instance.scope.spec
    argv = list(spec.argv)
    if damage == "entrypoint":
        argv[3] += "; print('foreign code')"
        spec = replace(spec, argv=tuple(argv))
    elif damage == "output":
        argv[9] = "--unrecognized-option"
        spec = replace(spec, argv=tuple(argv))
    elif damage == "pythonpath":
        spec = replace(spec, python_path=str(tmp_path))
    elif damage == "cwd":
        spec = replace(spec, cwd=str(tmp_path / "other"))
    elif damage == "credential":
        spec = replace(spec, credentials=())
    else:
        # Persisted scope digests independently reject identity changes.
        state_path = instance.scope.directory / "state.json"
        state = json.loads(state_path.read_text())
        state["spec"]["uid"] += 1
        state_path.write_text(json.dumps(state))
        with pytest.raises(ValueError, match="digest"):
            reopen(instance)
        return
    instance.scope.spec = spec
    with pytest.raises(GoalInputError):
        AzureGoalService(instance.scope)


def test_preparation_result_is_bound_to_original_goal_and_limits(tmp_path, goal, policy):
    path, digest = preparation(tmp_path, goal, policy)
    operation = AzureGoalService.create(tmp_path / "owner", path, digest, tmp_path / "exchange",
        operation="prepare", uid=os.geteuid(), python_executable=sys.executable,
        runtime_seconds=10, manager=CredentialManager())
    result = perform(path, digest, tmp_path / "exchange", operation="prepare")
    observed = asyncio.run(operation.run(timeout_seconds=0.01))
    assert observed.value.goal.goal_id == goal.goal_id
    report_path = tmp_path / "exchange/result.json"
    result["session"] = "foreign-session"
    report = json.loads(report_path.read_text())
    report["result"] = result
    report_path.write_text(json.dumps(report))
    with pytest.raises(GoalInputError, match="admission"):
        asyncio.run(reopen(operation).recover())
