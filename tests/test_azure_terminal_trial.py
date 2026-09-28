"""Ownership across goal services, a real issuer socket and container drainage."""
from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from taste.benchmarks.azure_terminal_trial import AzureTerminalTrial, cleanup_trial
from taste.brains.azure_goal_handoff import perform
from taste.brains.central_planner import Goal
from taste.brains.central_runtime import BudgetState, GoalOutcome
from taste.brains.docker_terminal import DockerTerminalBackend, DockerTerminalBinding
from taste.brains.goal_entrypoint import GoalInputError
from taste.brains.terminal_broker import TerminalBinding
from taste.brains.terminal_worker_policy import TerminalWorkerPolicy
from tests.test_azure_central_host import goal as _goal
from tests.test_azure_central_host import policy as _policy
from tests.test_process_scope import Manager
from tests.test_thread_shutdown_ownership import cancel_loop_tasks

goal = _goal
policy = _policy


def run_scenario(trial, coroutine):
    async def guarded():
        try:
            return await coroutine
        finally:
            if not trial.closed:
                await trial.close()
    return asyncio.run(guarded())


class InProcessManager(Manager):
    """Replace only systemd and goal work; use the actual handoff/file/lease path."""

    def __init__(self):
        super().__init__()
        self.operations = []
        self.missing_run = False

    def start(self, unit, description, spec, *, credential_directory=None):
        super().start(unit, description, spec)
        self.units[unit].update(RuntimeMaxUSec=f"{math.ceil(spec.runtime_seconds * 1e6)}us",
                                TimeoutStopUSec=f"{math.ceil(spec.grace_seconds * 1e6)}us")
        argv = spec.argv
        operation = argv[12]
        self.operations.append(operation)
        if operation != "run" or not self.missing_run:
            perform(argv[6], argv[8], argv[10], operation=operation)
        self.units[unit].update(ActiveState="inactive", ControlGroup="")
        self.populated.discard(unit)


@pytest.fixture
def trial(goal, policy, monkeypatch):
    with tempfile.TemporaryDirectory(prefix="tt-", dir="/tmp") as directory:
        # CI runner group membership is irrelevant to the simulated manager;
        # real server checks use a separate account with no extra groups.
        monkeypatch.setattr(os, "getgrouplist", lambda _name, gid: [gid])
        agent_deadline = time.time() + 90
        binding = DockerTerminalBinding("/var/run/docker.sock", "a" * 64, "b" * 32,
            "2026-09-28T00:00:00Z", "/system.slice/docker-" + "a" * 64 + ".scope", agent_deadline + 30, 8192)
        backend = DockerTerminalBackend(binding)
        stops = []

        def stop(self):
            stops.append(self.environment_id)
            return self.environment_id

        monkeypatch.setattr(DockerTerminalBackend, "stop_and_confirm", stop)
        policy = replace(policy, deadline_unix=agent_deadline,
            terminal=TerminalWorkerPolicy(TerminalBinding(binding.owner_token, binding.container_id, agent_deadline, 10), 5))
        manager = InProcessManager()
        instance = AzureTerminalTrial.create(Path(directory) / "trial", backend, goal, policy,
            service_uid=os.geteuid(), python_executable=sys.executable, max_generations=2,
            wall_clock_seconds=100, manager=manager)
        modes = []

        async def goal_work(config, mode, **_kwargs):
            modes.append(mode)
            return GoalOutcome(config.goal.goal_id, "complete", True, 1, 1,
                budget=BudgetState(config.goal.budget_usd, 1, 0, planner_spent_usd=1))

        monkeypatch.setattr("taste.brains.azure_goal_handoff._run", goal_work)
        instance.test_modes, instance.test_stops = modes, stops
        try:
            yield instance
        finally:
            if not instance.closed:
                asyncio.run(instance.close())


def test_goal_settles_before_sealing_and_container_lives_until_after_grading(trial):
    async def scenario():
        outcome = await trial.run(api_key="private-test-key")
        assert outcome.complete and trial.broker.phase == "sealed"
        assert trial.broker.binding.deadline_unix < trial.backend.binding.deadline_unix
        assert trial.manager.operations == ["prepare", "run", "settle"]
        assert trial.test_modes == ["run", "settle"] and not trial.manager.populated
        assert not trial.test_stops
        assert (trial.root / "controller/grading.json").is_file()
        for path in (trial.root / "controller").glob("*/state.json"):
            assert "private-test-key" not in path.read_text()
        with pytest.raises(GoalInputError, match="already admitted"):
            await trial.run(api_key="private-test-key")
        await trial.close()
        assert trial.test_stops == [trial.backend.environment_id]
        assert not (trial.root / "controller/run/credentials").exists()
        assert not (trial.root / "rpc").exists()
        # An outside watchdog can reuse proved drainage after Harbor removes
        # the container. It must not contact/re-admit a replacement container.
        record = cleanup_trial(trial.root, manager=trial.manager)
        assert record["container_stopped"] and len(record["scopes"]) == 3
        assert len(trial.test_stops) == 1
    run_scenario(trial, scenario())


def test_missing_run_report_uses_separate_settlement_after_original_drain(trial):
    trial.manager.missing_run = True

    async def scenario():
        result = await trial.run(api_key="private-test-key")
        assert result.complete and trial.broker.phase == "sealed"
        assert trial.test_modes == ["settle"]
        assert trial.manager.operations == ["prepare", "run", "settle"]
        assert not trial.manager.populated
        await trial.close()
    run_scenario(trial, scenario())


def test_unknown_cost_is_retained_and_stops_task_before_grading(trial, monkeypatch):
    async def unknown(config, mode, **_kwargs):
        return GoalOutcome(config.goal.goal_id, "budget_blocked", False, 1, 1,
            budget=BudgetState(config.goal.budget_usd, 0, 0, unknown_planner_attempt_ids=("lost-call",)))
    monkeypatch.setattr("taste.brains.azure_goal_handoff._run", unknown)
    with pytest.raises(GoalInputError, match="not settled"):
        asyncio.run(trial.run(api_key="private-test-key"))
    assert trial.closed and trial.test_stops == [trial.backend.environment_id]
    assert not (trial.root / "controller/grading.json").exists()
    saved = json.loads((trial.root / "controller/outcome.json").read_text())
    assert saved["outcome"]["budget"]["unknown_planner_attempt_ids"] == ["lost-call"]


@pytest.mark.parametrize("whole_loop", [False, True])
def test_cancellation_waits_for_goal_scope_and_then_stops_terminal(trial, monkeypatch, whole_loop):
    entered, release = threading.Event(), threading.Event()

    async def blocked(config, mode, **_kwargs):
        entered.set()
        assert release.wait(5)
        return GoalOutcome(config.goal.goal_id, "complete", True, 1, 1,
            budget=BudgetState(config.goal.budget_usd, 0, 0))
    monkeypatch.setattr("taste.brains.azure_goal_handoff._run", blocked)

    async def scenario():
        task = asyncio.create_task(trial.run(api_key="private-test-key"))
        while not entered.is_set():
            await asyncio.sleep(0.001)
        if whole_loop:
            cancel_loop_tasks()
        else:
            task.cancel()
        try:
            await asyncio.sleep(0.03)
            assert not task.done() and not trial.test_stops
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert trial.closed and not trial.manager.populated
        assert trial.manager.operations == ["prepare", "run"]
        assert trial.test_stops == [trial.backend.environment_id]
        assert not (trial.root / "controller/grading.json").exists()
    run_scenario(trial, scenario())


def test_watchdog_rejects_live_owner_and_recovers_after_owner_lease_is_released(trial):
    with pytest.raises(BlockingIOError):
        cleanup_trial(trial.root, manager=trial.manager)
    # Simulate death before any service or issuer was started. Recovery must
    # still stop the bound container, never launch preparation or a model.
    os.close(trial.fd)
    trial.closed = True
    report = cleanup_trial(trial.root, manager=trial.manager)
    assert report["container_stopped"] and report["scopes"] == []
    assert not report["goal_settlement_required"]
    assert not trial.manager.starts and not trial.test_modes


def test_changed_drain_record_cannot_authorize_cleanup_of_missing_container(trial):
    asyncio.run(trial.close())
    path = trial.root / "controller/drain.json"
    value = json.loads(path.read_text())
    value["container_id"] = "f" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(GoalInputError, match="evidence differs"):
        cleanup_trial(trial.root, manager=trial.manager)
    assert len(trial.test_stops) == 1


@pytest.mark.parametrize("damage", ["container", "deadline", "lifetime", "limit", "credential_parent"])
def test_invalid_admission_creates_no_controller_or_service(trial, damage):
    backend, policy = trial.backend, trial.policy
    root = trial.root.parent / "rejected"
    allowance = 100
    if damage == "container":
        backend = DockerTerminalBackend(replace(backend.binding, container_id="f" * 64,
            cgroup_path="/system.slice/docker-" + "f" * 64 + ".scope"))
    elif damage == "deadline":
        allowance = 1
    elif damage == "lifetime":
        backend = DockerTerminalBackend(replace(backend.binding, deadline_unix=policy.deadline_unix - 1))
    elif damage == "limit":
        allowance = True
    else:
        trial.root.parent.chmod(0o777)
    try:
        with pytest.raises(GoalInputError):
            AzureTerminalTrial.create(root, backend, Goal.from_dict(trial.config["goal"]), policy,
                service_uid=os.geteuid(), python_executable=sys.executable, max_generations=2,
                wall_clock_seconds=allowance, manager=trial.manager)
        assert not root.exists() and not trial.manager.starts
    finally:
        trial.root.parent.chmod(0o700)


def test_corrupt_scope_still_stops_container_and_retains_cleanup_failure(trial):
    operation = trial._operation("prepare", "prepare.json", trial.preparation_sha)
    path = operation.scope.directory / "state.json"
    original = path.read_bytes()
    path.write_text("{}")
    try:
        with pytest.raises(ExceptionGroup, match="cleanup is incomplete"):
            asyncio.run(trial.close())
        assert trial.test_stops == [trial.backend.environment_id]
        assert not (trial.root / "controller/drain.json").exists()
    finally:
        path.write_bytes(original)
