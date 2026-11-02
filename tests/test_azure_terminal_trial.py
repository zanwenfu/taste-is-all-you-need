"""Ownership across goal services, a real issuer socket and container drainage."""
from __future__ import annotations

import asyncio
import hashlib
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


def test_release_discards_checkpoint_tars_and_keeps_their_manifests(trial):
    async def scenario():
        await trial.run(api_key="private-test-key")
        store = trial.root / "controller/terminal/checkpoints"
        store.mkdir(mode=0o700)
        (store / ("a" * 64 + ".tar")).write_bytes(b"x" * 100)
        (store / "initial.json").write_text("{}")
        await trial.release()
        assert sorted(item.name for item in store.iterdir()) == ["initial.json"]
    run_scenario(trial, scenario())


def test_a_study_grading_each_saved_state_keeps_its_tars(trial):
    async def scenario():
        await trial.run(api_key="private-test-key")
        store = trial.root / "controller/terminal/checkpoints"
        store.mkdir(mode=0o700)
        tar = "a" * 64 + ".tar"
        (store / tar).write_bytes(b"x" * 100)
        (store / "initial.json").write_text("{}")
        trial.policy = replace(trial.policy, rollback=True, keep_checkpoints=True)
        await trial.release()
        assert sorted(item.name for item in store.iterdir()) == [tar, "initial.json"]
    run_scenario(trial, scenario())


def test_saves_a_study_will_grade_get_room_to_be_whole():
    from taste.benchmarks.azure_terminal_trial import KEPT_CHECKPOINT_BYTES, KEPT_STORE_BYTES
    assert KEPT_CHECKPOINT_BYTES == 4 * 1024 ** 3 and KEPT_STORE_BYTES == 32 * 1024 ** 3


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


class CrashingManager(InProcessManager):
    """The first run works until it is killed, as a real goal process would; any later one finishes."""

    def __init__(self):
        super().__init__()
        self.kills = []
        self.kill_is_lost = False

    def start(self, unit, description, spec, *, credential_directory=None):
        if spec.argv[12] != "run" or "run" in self.operations:
            return super().start(unit, description, spec, credential_directory=credential_directory)
        Manager.start(self, unit, description, spec)
        self.units[unit].update(RuntimeMaxUSec=f"{math.ceil(spec.runtime_seconds * 1e6)}us",
                                TimeoutStopUSec=f"{math.ceil(spec.grace_seconds * 1e6)}us")
        self.operations.append("run")

    def kill(self, unit):
        self.kills.append(unit)
        if self.kill_is_lost:
            return
        self.units[unit].update(ActiveState="failed", Result="signal", ControlGroup="")
        self.populated.discard(unit)


def crashing(trial, after=0.05):
    trial.manager = CrashingManager()
    trial.policy = replace(trial.policy, crash_after_seconds=after)
    return trial.manager


def test_a_killed_goal_runs_again_from_its_records_and_its_task_is_still_graded(trial):
    manager = crashing(trial)

    async def scenario():
        before = time.time()
        outcome = await trial.run(api_key="private-test-key")
        assert outcome.complete and trial.sealed and trial.broker.phase == "sealed"
        assert manager.operations == ["prepare", "run", "run", "settle"] and len(manager.kills) == 1
        assert trial.test_modes == ["run", "settle"], "only the second run reached the goal's work"
        assert not trial.test_stops, "the task's container outlives the goal process"
        assert not manager.populated and "goal_service_failed" not in trial.audit_flags
        crash = json.loads((trial.root / "controller/crash.json").read_text())
        assert crash == trial.crash and crash["landed"] is True and crash["resumed"] is True
        assert crash["after_seconds"] == 0.05 and before < crash["killed_unix"] <= crash["resumed_unix"]
        assert crash["recovery_seconds"] == round(crash["resumed_unix"] - crash["killed_unix"], 3)
        assert crash["commands_repeated"] == 0 and crash["paid_calls_repeated"] == 0
        for name in ("run", "resume"):
            assert (trial.root / "controller" / name / "credentials").exists()
            assert "private-test-key" not in (trial.root / "controller" / name / "state.json").read_text()
        await trial.release()
        assert not (trial.root / "controller/run/credentials").exists()
        assert not (trial.root / "controller/resume/credentials").exists()
        handoff = json.loads((trial.root / "controller/handoff.json").read_text())
        assert len(handoff["scopes"]) == 4
        record = cleanup_trial(trial.root, manager=trial.manager)
        assert len(record["scopes"]) == 4, "the outside watchdog also proves the resumed scope drained"
    run_scenario(trial, scenario())


def test_a_goal_that_ends_before_its_kill_is_never_killed_and_says_so(trial):
    trial.policy = replace(trial.policy, crash_after_seconds=30.0)

    async def scenario():
        outcome = await trial.run(api_key="private-test-key")
        assert outcome.complete and trial.manager.operations == ["prepare", "run", "settle"]
        assert trial.crash["landed"] is False and trial.crash["resumed"] is False
        assert json.loads((trial.root / "controller/crash.json").read_text()) == trial.crash
    run_scenario(trial, scenario())


def test_a_kill_that_did_not_land_leaves_the_goal_running_and_says_so(trial):
    manager = crashing(trial)
    manager.kill_is_lost = True

    async def scenario():
        task = asyncio.create_task(trial.run(api_key="private-test-key"))
        while not manager.kills:
            await asyncio.sleep(0.005)
        unit = manager.kills[0]
        await asyncio.sleep(0.02)
        assert not task.done() and manager.operations == ["prepare", "run"]
        (trial.root / "exchange/run").mkdir(mode=0o700)
        from taste.brains.azure_goal_handoff import _write

        digest = hashlib.sha256((trial.root / "goal.json").read_bytes()).hexdigest()
        binding = {"schema": "taste.brains/AzureGoalHandoff/1", "input_sha256": digest, "operation": "run"}
        goal_id = trial.config["goal"]["goal_id"]
        budget = trial.config["goal"]["budget_usd"]
        result = GoalOutcome(goal_id, "complete", True, 1, 1,
                             budget=BudgetState(budget, 1, 0, planner_spent_usd=1)).to_dict()
        fd = os.open(trial.root / "exchange/run", os.O_RDONLY | os.O_DIRECTORY)
        try:
            _write(fd, "intent.json", (json.dumps(binding) + "\n").encode())
            _write(fd, "result.json", (json.dumps({**binding, "result": result}) + "\n").encode())
        finally:
            os.close(fd)
        manager.units[unit].update(ActiveState="inactive", ControlGroup="")
        manager.populated.discard(unit)
        outcome = await task
        assert outcome.complete and trial.crash["landed"] is False and trial.crash["resumed"] is False
        assert manager.operations == ["prepare", "run", "settle"]
    run_scenario(trial, scenario())


def test_the_benchmark_time_limit_before_the_kill_drains_and_hands_over_as_usual(trial):
    manager = crashing(trial, after=30.0)

    async def scenario():
        task = asyncio.create_task(trial.run(api_key="private-test-key"))
        while "run" not in manager.operations:
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert trial.sealed and trial.audit_flags == ("owner_cancelled",) and not manager.kills
        assert manager.operations == ["prepare", "run", "settle"] and not manager.populated
        assert trial.crash["landed"] is False
    run_scenario(trial, scenario())


def test_the_benchmark_time_limit_during_the_resumed_run_drains_both_and_hands_over(trial, monkeypatch):
    manager = crashing(trial)
    entered, release = threading.Event(), threading.Event()

    async def blocked(config, mode, **_kwargs):
        if mode == "run":
            entered.set()
            assert release.wait(5)
        return GoalOutcome(config.goal.goal_id, "wall_clock", False, 1, 1,
                           budget=BudgetState(config.goal.budget_usd, 0, 0))
    monkeypatch.setattr("taste.brains.azure_goal_handoff._run", blocked)

    async def scenario():
        task = asyncio.create_task(trial.run(api_key="private-test-key"))
        while not entered.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        await asyncio.sleep(0.02)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert trial.sealed and "owner_cancelled" in trial.audit_flags and not manager.populated
        assert manager.operations == ["prepare", "run", "run", "settle"] and trial.crash["resumed"] is True
    run_scenario(trial, scenario())


def test_a_resumed_goal_that_cannot_start_is_settled_and_still_graded(trial, monkeypatch):
    crashing(trial)
    original = trial._operation

    def refused(name, *args, **kwargs):
        if name == "resume":
            raise GoalInputError("goal service cannot renew an expired deadline")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(trial, "_operation", refused)

    async def scenario():
        outcome = await trial.run(api_key="private-test-key")
        assert outcome.complete and trial.sealed and not trial.test_stops
        assert "goal_service_failed" in trial.audit_flags
        assert trial.crash["landed"] is True and trial.crash["resumed"] is False
        assert trial.test_modes == ["settle"], "nothing ran the goal after the kill"
    run_scenario(trial, scenario())


def failing_to_resume(trial, monkeypatch, *, stops_that_fail):
    """Launching the new goal timed out, and stopping it times out too, while it is alive."""
    crashing(trial)
    monkeypatch.setattr("taste.benchmarks.azure_terminal_trial.DRAIN_RETRY_SECONDS", 0)
    original = trial._operation
    failing = {"left": stops_that_fail}
    order = []

    def operation(name, *args, **kwargs):
        service = original(name, *args, **kwargs)
        order.append((name, failing["left"]))
        if name == "resume":
            async def run(*, timeout_seconds):
                raise RuntimeError("the manager never acknowledged the launch")

            async def stop_async(reason):
                if failing["left"]:
                    failing["left"] -= 1
                    raise RuntimeError("stopping the service timed out")
                return {"processes_stopped": True}
            service.run = run
            service.scope.stop_async = stop_async
        return service
    monkeypatch.setattr(trial, "_operation", operation)
    return order


def test_settlement_waits_until_a_resumed_goal_is_proved_drained(trial, monkeypatch):
    """Settlement opens the same branches. On a real task it started beside the living goal,
    found the branch held, and the run's record was lost."""
    order = failing_to_resume(trial, monkeypatch, stops_that_fail=2)

    async def scenario():
        outcome = await trial.run(api_key="private-test-key")
        assert outcome.complete and trial.sealed and "goal_service_failed" in trial.audit_flags
        assert dict(order)["settle"] == 0, "settlement began only once the resumed scope had drained"
    run_scenario(trial, scenario())


def test_a_resumed_goal_that_cannot_be_proved_drained_is_never_settled_beside(trial, monkeypatch):
    order = failing_to_resume(trial, monkeypatch, stops_that_fail=99)

    async def scenario():
        with pytest.raises(RuntimeError, match="timed out"):
            await trial.run(api_key="private-test-key")
        assert "settle" not in dict(order)
    run_scenario(trial, scenario())


def test_a_crash_record_that_cannot_be_written_does_not_cost_the_handoff(trial, monkeypatch):
    crashing(trial)
    monkeypatch.setattr("taste.benchmarks.crash_evidence.read_evidence",
                        lambda *_: (_ for _ in ()).throw(OSError("unreadable")))

    async def scenario():
        outcome = await trial.run(api_key="private-test-key")
        assert outcome.complete and trial.sealed and trial.crash["evidence_failed"] == "OSError"
    run_scenario(trial, scenario())


def test_no_crash_is_recorded_unless_a_study_asked_for_one(trial):
    async def scenario():
        await trial.run(api_key="private-test-key")
        assert trial.crash is None and not (trial.root / "controller/crash.json").exists()
        assert "crash_after_seconds" not in trial.policy.to_dict()
    run_scenario(trial, scenario())

