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


def test_unsettled_accounting_is_flagged_and_the_task_is_still_graded(trial, monkeypatch):
    async def unknown(config, mode, **_kwargs):
        return GoalOutcome(config.goal.goal_id, "budget_blocked", False, 1, 1,
            budget=BudgetState(config.goal.budget_usd, 0, 0, unknown_planner_attempt_ids=("lost-call",)))
    monkeypatch.setattr("taste.brains.azure_goal_handoff._run", unknown)

    async def scenario():
        outcome = await trial.run(api_key="private-test-key")
        # Our audit lacks an exact cost. The task's container is unaffected
        # and its grader still receives it, sealed, with the gap on record.
        assert not outcome.complete and trial.audit_flags == ("accounting_unsettled",)
        assert trial.sealed and trial.broker.phase == "sealed" and not trial.test_stops
        grading = json.loads((trial.root / "controller/grading.json").read_text())
        assert grading["audit_flags"] == ["accounting_unsettled"]
        saved = json.loads((trial.root / "controller/outcome.json").read_text())
        assert saved["outcome"]["budget"]["unknown_planner_attempt_ids"] == ["lost-call"]
    run_scenario(trial, scenario())


@pytest.mark.parametrize("stop_reason", ["planner_failed", "runtime_error"])
def test_a_goal_that_ended_badly_still_leaves_its_container_for_grading(trial, monkeypatch, stop_reason):
    async def failed(config, mode, **_kwargs):
        return GoalOutcome(config.goal.goal_id, stop_reason, False, 1, 3,
            budget=BudgetState(config.goal.budget_usd, 1, 0, planner_spent_usd=1))
    monkeypatch.setattr("taste.brains.azure_goal_handoff._run", failed)

    async def scenario():
        outcome = await trial.run(api_key="private-test-key")
        assert outcome.stop_reason == stop_reason and trial.sealed and not trial.test_stops
        # Its spending was settled: the flag says how it stopped, not that it was unsettled.
        assert trial.audit_flags == ("stopped:" + stop_reason,)
    run_scenario(trial, scenario())


def test_failed_settlement_is_flagged_and_does_not_withhold_the_container(trial, monkeypatch):
    async def work(config, mode, **_kwargs):
        if mode == "settle":
            raise RuntimeError("the goal's own record cannot be reopened")
        return GoalOutcome(config.goal.goal_id, "complete", True, 1, 1,
            budget=BudgetState(config.goal.budget_usd, 1, 0, planner_spent_usd=1))
    monkeypatch.setattr("taste.brains.azure_goal_handoff._run", work)

    async def scenario():
        assert await trial.run(api_key="private-test-key") is None
        assert len(trial.audit_flags) == 1 and trial.audit_flags[0].startswith("settlement_failed:")
        assert trial.sealed and trial.broker.phase == "sealed" and not trial.test_stops
        assert not (trial.root / "controller/outcome.json").exists()
        assert json.loads((trial.root / "controller/grading.json").read_text())["result_sha256"] is None
    run_scenario(trial, scenario())


def test_release_hands_the_sealed_container_to_its_grader_and_frees_everything_else(trial):
    async def scenario():
        await trial.run(api_key="private-test-key")
        socket = trial.root / "rpc/terminal.sock"
        assert socket.exists()
        await trial.release()
        assert trial.closed and not trial.test_stops, "release must not stop the container"
        handoff = json.loads((trial.root / "controller/handoff.json").read_text())
        assert handoff["container_stopped"] is False and len(handoff["scopes"]) == 3
        assert not (trial.root / "rpc").exists()
        assert not (trial.root / "controller/run/credentials").exists()
        assert not (trial.root / "controller/drain.json").exists()
        await trial.release()  # Idempotent once closed.
        # The outside watchdog still drains the original container afterwards.
        record = cleanup_trial(trial.root, manager=trial.manager)
        assert record["container_stopped"] and trial.test_stops == [trial.backend.environment_id]
    run_scenario(trial, scenario())


def test_release_of_an_unsealed_trial_stops_its_container(trial):
    async def scenario():
        await trial.release()
        assert trial.closed and trial.test_stops == [trial.backend.environment_id]
    run_scenario(trial, scenario())


@pytest.mark.parametrize("whole_loop", [False, True])
def test_cancellation_waits_for_goal_scope_then_still_hands_over_for_grading(trial, monkeypatch, whole_loop):
    entered, release = threading.Event(), threading.Event()

    async def blocked(config, mode, **_kwargs):
        entered.set()
        assert release.wait(5)
        return GoalOutcome(config.goal.goal_id, "wall_clock", False, 1, 1,
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
        # The benchmark's own time limit cancelled the agent. It is reported
        # as that cancellation, after the environment was sealed for grading.
        with pytest.raises(asyncio.CancelledError):
            await task
        assert trial.sealed and not trial.closed and not trial.manager.populated
        assert trial.manager.operations == ["prepare", "run", "settle"]
        assert trial.broker.phase == "sealed" and not trial.test_stops
        assert trial.audit_flags == ("owner_cancelled",)
        grading = json.loads((trial.root / "controller/grading.json").read_text())
        assert grading["audit_flags"] == ["owner_cancelled"]
        await trial.release()
        assert trial.closed and not trial.test_stops
    run_scenario(trial, scenario())


def test_a_cancellation_during_settlement_waits_for_the_record_and_the_seal(trial, monkeypatch):
    # The goal has finished and its reply exists. Stopping here would throw
    # both away and stop the container the benchmark is about to grade.
    entered, release = threading.Event(), threading.Event()

    async def settling(config, mode, **_kwargs):
        if mode == "settle":
            entered.set()
            assert release.wait(5)
        return GoalOutcome(config.goal.goal_id, "complete", True, 1, 1,
            budget=BudgetState(config.goal.budget_usd, 1, 0, planner_spent_usd=1))
    monkeypatch.setattr("taste.brains.azure_goal_handoff._run", settling)

    async def scenario():
        task = asyncio.create_task(trial.run(api_key="private-test-key"))
        while not entered.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        try:
            await asyncio.sleep(0.03)
            assert not task.done() and not trial.test_stops
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert trial.sealed and not trial.closed and not trial.test_stops
        assert trial.outcome is not None and trial.outcome.complete
        assert trial.manager.operations == ["prepare", "run", "settle"]
        assert trial.audit_flags == ("owner_cancelled",)
        outcome = json.loads((trial.root / "controller/outcome.json").read_text())
        assert outcome["outcome"]["complete"] is True
        await trial.release()
        assert trial.closed and not trial.test_stops
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
