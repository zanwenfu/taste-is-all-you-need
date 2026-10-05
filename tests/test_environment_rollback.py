"""The coordinator's checkpoints of the task's files, and its rollback to one."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from typing import Any

import pytest

from taste.brains.central_planner import CentralPlanner, InvalidPlannerOutput, PlanningRequest
from taste.brains.central_runtime import CentralRuntime
from taste.brains.environment_records import (
    RESTORE_SCHEMA,
    ROLLBACK_RULE,
    EnvironmentHistory,
    checkpoint_id,
    read_history,
    restore_operation_id,
    restore_path,
    validate_rollback,
)
from taste.brains.supervisor import CentralSupervisor
from taste.brains.terminal_issuer import IssuerEnvironment, TerminalIssuerClient
from taste.brains.terminal_service import TerminalUnavailable
from taste.memstore import Store
from tests.test_brains_central_runtime import (
    FakeLauncher,
    ScriptedTransport,
    assignment_for,
    proposal,
    settle,
    simple_goal,
)


class FakeEnvironment:
    """The controller as the runtime sees it: records for checkpoints and restores."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.failing: set[str] = set()
        # The size and partiality of checkpoints taken after runs.
        self.bytes, self.partial = 2048, False

    def checkpoint(self, identity, *, timeout_seconds):
        assert 0 < timeout_seconds <= 300
        self.calls.append(("checkpoint", identity))
        if "checkpoint" in self.failing:
            raise OSError("daemon unavailable at /var/run/docker.sock")
        after = identity != "initial"
        return {"checkpoint_id": identity, "tar_sha256": "f" * 64, "taken_at": 1.0,
                "bytes": self.bytes if after else 10240,
                "added": 1, "changed": 1, "deleted": 0, "partial": self.partial and after, "left_out": 0,
                "paths": ["/app/server.py"], "more_paths": 0, "deleted_paths": [], "more_deleted_paths": 0,
                "over_cap": [], "more_over_cap": 0}

    def restore(self, operation_id, identity, *, timeout_seconds):
        assert 0 < timeout_seconds <= 300
        self.calls.append(("restore", operation_id, identity))
        if "restore" in self.failing:
            return {"operation_id": operation_id, "checkpoint_id": identity, "failed": True, "error": "OSError"}
        return {"operation_id": operation_id, "checkpoint_id": identity, "tar_sha256": "f" * 64, "exact": True,
                "seconds": 0.5, "removed": 2, "from_image": 1, "deleted": 0, "mismatches": [],
                "more_mismatches": 0}


def stack(store, transport, launcher, environment, *, fault=None, shared=None):
    if shared is None:
        lock = threading.RLock()
        control = store.branch("central-control", producer="central-runtime")
        integration = store.branch("integration", producer="central-integration")
    else:
        control, integration, lock = shared
    planner = CentralPlanner(store, transport=transport, control=control, control_branch=control.name,
                             integration_branch=integration.name, mutation_lock=lock)
    supervisor = CentralSupervisor(store, launcher=launcher, control_branch=control,
                                   integration_branch=integration, control_lock=lock)
    runtime = CentralRuntime(store, simple_goal(), planner=planner, supervisor=supervisor, control=control,
                             integration=integration, control_lock=lock, fault_injector=fault,
                             task_environment=environment)
    return runtime, (control, integration, lock)


def rolling_back(prompt: str, *assignments, to, reason, complete=False) -> str:
    response = json.loads(proposal(prompt, *assignments, complete=complete))
    response.update(rollback_to=to, rollback_reason=reason)
    return json.dumps(response, sort_keys=True)


class Planner:
    """Generation 1 builds; generation 2 rolls back to ``to`` and builds again, or completes."""

    def __init__(self, *, to="cp1", reason="the check passed before run 1 and fails after it",
                 complete=False):
        self.to, self.reason, self.complete = to, reason, complete
        self.prompts: dict[int, dict[str, Any]] = {}

    def __call__(self, request: PlanningRequest, prompt: str):
        self.prompts[request.generation] = json.loads(prompt)
        if request.generation == 1:
            return (assignment_for(request, "build", "worker-build", "product.txt"),)
        if request.generation == 2 and self.to is not None:
            work = () if self.complete else (assignment_for(request, "rebuild", "worker-rebuild", "again.txt"),)
            return rolling_back(prompt, *work, to=self.to, reason=self.reason, complete=self.complete)
        return proposal(prompt, complete=True)


def run_to_second_plan(store, environment, planner, **kwargs):
    transport = ScriptedTransport(planner)
    launcher = FakeLauncher()
    runtime, shared = stack(store, transport, launcher, environment, **kwargs)
    assert runtime.cycle().status == "planned"
    assert runtime.cycle().status == "running"
    settle(runtime, launcher, "build")
    second = runtime.cycle()
    return runtime, launcher, transport, shared, second


@pytest.fixture
def store(tmp_path):
    opened = Store.open(tmp_path / "repo", "rollback-test")
    yield opened
    opened.close()


def history(runtime) -> EnvironmentHistory:
    return read_history(runtime.control.head, runtime.goal.goal_id)


def test_a_checkpoint_before_the_first_worker_and_after_its_run_reach_the_planner(store):
    environment, planner = FakeEnvironment(), Planner(to=None)
    runtime, _, _, _, second = run_to_second_plan(store, environment, planner)
    assert second.status in {"replanned", "complete"}
    run = next(item.run_id for item in runtime.supervisor.runs())
    assert environment.calls == [("checkpoint", "initial"), ("checkpoint", checkpoint_id({run}))]
    taken = history(runtime).checkpoints
    assert [(item["label"], item["runs_ended"]) for item in taken] == [("cp1", []), ("cp2", [run])]
    # The first plan saw no checkpoint; the second saw both, the rule and the fields.
    assert "task_environment_checkpoints" not in planner.prompts[1]
    assert "rollback_to" not in planner.prompts[1]["required_output_shape"]
    shown = planner.prompts[2]["task_environment_checkpoints"]
    assert [item["checkpoint"] for item in shown] == ["cp1", "cp2"]
    assert shown[0]["taken"] == "before any worker ran" and shown[1]["runs_ended"] == [run]
    assert shown[1]["files"]["paths"] == ["/app/server.py"] and "tar_sha256" not in shown[1]["files"]
    assert planner.prompts[2]["rules"]["task_environment_rollback"] == ROLLBACK_RULE
    assert planner.prompts[2]["required_output_shape"]["rollback_to"] is None


def test_a_plan_that_names_a_checkpoint_restores_it_once_before_its_worker_starts(store):
    environment, planner = FakeEnvironment(), Planner(to="cp1")
    runtime, launcher, _, _, second = run_to_second_plan(store, environment, planner)
    assert second.status == "replanned"
    plan = runtime.planner.current_plan(runtime.goal.goal_id)
    assert dict(plan.metadata["rollback"]) == {"checkpoint": "cp1", "checkpoint_id": "initial",
                                               "reason": planner.reason}
    assert len(launcher.launch_calls) == 1
    assert runtime.cycle().status == "running"
    operation = restore_operation_id(plan.plan_id)
    assert environment.calls[-1] == ("restore", operation, "initial")
    assert len(launcher.launch_calls) == 2
    [restored] = history(runtime).restores
    assert (restored["label"], restored["reason"], restored["result"]["exact"]) == ("cp1", planner.reason, True)
    runtime.cycle()
    assert [call[0] for call in environment.calls].count("restore") == 1


def test_a_failed_restore_starts_no_worker_and_asks_for_a_new_plan(store):
    environment, planner = FakeEnvironment(), Planner(to="cp1")
    environment.failing.add("restore")
    runtime, launcher, _, _, _ = run_to_second_plan(store, environment, planner)
    third = runtime.cycle()
    assert len(launcher.launch_calls) == 1
    assert third.status in {"replanned", "complete"}
    assert any(item.kind == "rollback_failed" for item in third.triggers)
    [restored] = history(runtime).restores
    assert restored["result"] == {"operation_id": restored["operation_id"], "checkpoint_id": "initial",
                                  "failed": True, "error": "OSError"}
    assert planner.prompts[3]["task_environment_rollbacks"][0]["outcome"].startswith("failed")


def test_a_failed_restore_recorded_before_a_restart_still_starts_no_worker(store):
    environment, planner = FakeEnvironment(), Planner(to="cp1")
    runtime, launcher, _, _, _ = run_to_second_plan(store, environment, planner)
    plan = runtime.planner.current_plan(runtime.goal.goal_id)
    # The previous coordinator recorded the failed restore and died before
    # recording the trigger that plans again.
    operation = restore_operation_id(plan.plan_id)
    runtime._immutable(restore_path(runtime.goal.goal_id, plan.plan_id), {
        "schema": RESTORE_SCHEMA, "goal_id": runtime.goal.goal_id, "plan_id": plan.plan_id,
        "generation": plan.generation, "operation_id": operation, "label": "cp1", "checkpoint_id": "initial",
        "reason": planner.reason, "at": "2026-10-05T00:00:00Z",
        "result": {"operation_id": operation, "checkpoint_id": "initial", "failed": True,
                   "error": "TerminalUnavailable"}}, "a restore that failed before a restart")
    third = runtime.cycle()
    assert len(launcher.launch_calls) == 1
    assert any(item.kind == "rollback_failed" for item in third.triggers)
    assert not any(call[0] == "restore" for call in environment.calls)


def test_a_complete_plan_restores_before_the_goal_ends(store):
    environment, planner = FakeEnvironment(), Planner(to="cp1", complete=True)
    runtime, launcher, _, _, second = run_to_second_plan(store, environment, planner)
    assert second.status == "complete"
    assert environment.calls[-1][0] == "restore" and len(history(runtime).restores) == 1
    assert len(launcher.launch_calls) == 1


@pytest.mark.parametrize(("to", "reason"), [("cp9", "evidence"), ("initial", "evidence"), ("cp1", "  ")])
def test_an_unlisted_checkpoint_or_a_missing_reason_is_refused(store, to, reason):
    environment, planner = FakeEnvironment(), Planner(to=to, reason=reason)
    with pytest.raises(InvalidPlannerOutput, match="rollback"):
        run_to_second_plan(store, environment, planner)
    assert not any(call[0] == "restore" for call in environment.calls)


def test_without_a_task_environment_nothing_changes_and_a_rollback_is_ignored(store):
    planner = Planner(to="cp1")
    runtime, _, _, _, second = run_to_second_plan(store, None, planner)
    assert second.status == "replanned"
    assert "task_environment_checkpoints" not in planner.prompts[2]
    assert "rollback_to" not in planner.prompts[2]["required_output_shape"]
    plan = runtime.planner.current_plan(runtime.goal.goal_id)
    assert "rollback" not in plan.metadata
    assert "ignored:rollback_to" in plan.metadata["filled_by_harness"]
    assert history(runtime) == EnvironmentHistory()


def test_a_failed_checkpoint_is_recorded_and_never_offered(store):
    environment, planner = FakeEnvironment(), Planner(to=None)
    environment.failing.add("checkpoint")
    runtime, launcher, _, _, _ = run_to_second_plan(store, environment, planner)
    taken = history(runtime).checkpoints
    assert [item["result"] for item in taken] == [{"failed": True, "error": "OSError"}] * 2
    assert "daemon unavailable" not in json.dumps(taken)
    assert "task_environment_checkpoints" not in planner.prompts[2]
    assert len(launcher.launch_calls) == 1


def test_a_coordinator_that_restarts_after_a_restore_reads_it_back(store):
    environment, planner = FakeEnvironment(), Planner(to="cp1")
    fired = []

    def fault(boundary, _payload):
        if boundary == "decision_effect:environment_restore" and not fired:
            fired.append(boundary)
            raise RuntimeError("coordinator died after the restore")

    runtime, launcher, transport, shared, _ = run_to_second_plan(store, environment, planner, fault=fault)
    with pytest.raises(RuntimeError, match="died after the restore"):
        runtime.cycle()
    assert len(launcher.launch_calls) == 1
    replacement, _ = stack(store, transport, launcher, environment, shared=shared)
    assert replacement.cycle().status == "running"
    restores = [call for call in environment.calls if call[0] == "restore"]
    # Asked again with the same operation ID: the controller's record, not a second restore.
    assert len(restores) == 2 and restores[0] == restores[1]
    assert len(history(replacement).restores) == 1 and len(launcher.launch_calls) == 2


def _history_with(*states):
    """Checkpoints cp1, cp2, ... each "ok", "failed" or "partial"."""
    results = {"ok": lambda number: {"checkpoint_id": f"id{number}"}, "failed": lambda number: {"failed": True},
               "partial": lambda number: {"checkpoint_id": f"id{number}", "partial": True}}
    checkpoints = tuple(
        {"label": f"cp{number}", "checkpoint_id": f"id{number}", "runs_ended": [], "generation": 1,
         "result": results[state](number)}
        for number, state in enumerate(states, start=1))
    return EnvironmentHistory(checkpoints=checkpoints)


def test_rollback_validation_names_only_checkpoints_that_can_be_restored():
    listed = _history_with("ok", "failed", "ok", "partial")
    for nothing in (None, "", "none", "NULL", "  "):
        assert validate_rollback(nothing, "reason given anyway", listed) is None
    for name in ("cp3", " CP3 ", "Cp3"):
        assert validate_rollback(name, "fails after run 2", listed) == {
            "checkpoint": "cp3", "checkpoint_id": "id3", "reason": "fails after run 2"}
    for target in ("cp2", "cp4", "cp5", "id1", 1):
        with pytest.raises(ValueError, match="cp1, cp3"):
            validate_rollback(target, "evidence", listed)
    for reason in ("", "   ", None, "x" * 4001):
        with pytest.raises(ValueError, match="rollback_reason"):
            validate_rollback("cp1", reason, listed)


def test_the_planner_is_told_plainly_what_a_failed_restore_did_to_the_files():
    def restore(error):
        return {"generation": 2, "label": "cp1", "reason": "r", "at": "t",
                "result": {"failed": True, "error": error}}

    history = EnvironmentHistory(restores=(restore("no_time"), restore("TerminalConflict"),
                                           restore("TerminalUnavailable"), restore("DockerTransportError")))
    outcomes = [item["outcome"] for item in history.for_planner()["rollbacks"]]
    assert outcomes[0] == outcomes[1] == "not done: the files were left as they were"
    assert outcomes[2].startswith("not confirmed") and outcomes[3].startswith("failed partway")


def test_a_partial_checkpoint_is_never_offered(store):
    environment, planner = FakeEnvironment(), Planner(to="cp2")
    environment.partial = True
    with pytest.raises(InvalidPlannerOutput, match="cp1"):
        run_to_second_plan(store, environment, planner)
    assert [item["checkpoint"] for item in planner.prompts[2]["task_environment_checkpoints"]] == ["cp1"]


def test_a_restore_is_not_started_without_the_time_its_size_needs(store):
    environment, planner = FakeEnvironment(), Planner(to="cp2")
    environment.bytes = 10**12  # 50,000 s at the assumed copy rate
    runtime, launcher, _, _, _ = run_to_second_plan(store, environment, planner)
    third = runtime.cycle()
    assert not any(call[0] == "restore" for call in environment.calls)
    [restored] = history(runtime).restores
    assert restored["result"] == {"failed": True, "error": "no_time"}
    assert any(item.kind == "rollback_failed" for item in third.triggers) and len(launcher.launch_calls) == 1


def test_a_rollback_left_pending_when_the_run_ends_is_made_before_the_closing_reply(store):
    from datetime import UTC, datetime, timedelta

    from taste.brains import benchmark_reply

    goal = replace(simple_goal(), metadata={benchmark_reply.KEY: benchmark_reply.SCHEMA,
                                            benchmark_reply.RESERVE_KEY: 10})
    seen = []

    def respond(request, prompt):
        seen.append(json.loads(prompt))
        raw = json.loads(proposal(prompt, *(() if benchmark_reply.is_closing(request.operation_id) else
                                           (assignment_for(request, f"build-{request.generation}",
                                                           f"worker-build-{request.generation}", "product.txt"),))))
        raw["metadata"] = {"final_reply": "Rolled back; nothing else was finished."
                           if benchmark_reply.is_closing(request.operation_id) else ""}
        if request.generation == 2 and not benchmark_reply.is_closing(request.operation_id):
            raw.update(rollback_to="cp1", rollback_reason="run 1 broke the build")
        return json.dumps(raw, sort_keys=True)

    environment, transport, launcher = FakeEnvironment(), ScriptedTransport(respond), FakeLauncher()
    lock = threading.RLock()
    control = store.branch("central-control", producer="central-runtime")
    integration = store.branch("integration", producer="central-integration")
    planner = CentralPlanner(store, transport=transport, control=control, control_branch=control.name,
                             integration_branch=integration.name, mutation_lock=lock)
    supervisor = CentralSupervisor(store, launcher=launcher, control_branch=control,
                                   integration_branch=integration, control_lock=lock)
    runtime = CentralRuntime(store, goal, planner=planner, supervisor=supervisor, control=control,
                             integration=integration, control_lock=lock, task_environment=environment)
    runtime.cycle()
    runtime.cycle()
    settle(runtime, launcher, "build-1")
    assert runtime.cycle().status == "replanned"
    plan = runtime.planner.current_plan(goal.goal_id)
    assert plan.metadata["rollback"]["checkpoint"] == "cp1" and not history(runtime).restores
    # The run ends before another cycle could make the rollback.
    runtime._plan_minimum = 0.0
    runtime._closing_reply("wall_clock", "the run reached its bound", datetime.now(UTC) + timedelta(seconds=600))
    assert runtime.closing_failure is None, runtime.closing_failure
    assert ("restore", restore_operation_id(plan.plan_id), "initial") in environment.calls
    closing = seen[-1]
    assert closing["task_environment_rollbacks"][0]["outcome"] == "exact"


def test_checkpoint_ids_follow_the_runs_not_their_order():
    assert checkpoint_id(set()) == "initial"
    assert checkpoint_id({"b", "a"}) == checkpoint_id(["a", "b"]) != checkpoint_id({"a"})
    assert len(checkpoint_id({"a"})) <= 128 and checkpoint_id({"a"}).startswith("after_")


# -- the coordinator's way to the controller -----------------------------------------


class FlakyClient(TerminalIssuerClient):
    """A coordinator client whose first replies never arrive."""

    def __init__(self, lost):
        self.lost, self.calls = lost, []

    def checkpoint(self, identity, *, timeout_seconds):
        self.calls.append(("checkpoint", identity))
        if len(self.calls) <= self.lost:
            raise TerminalUnavailable("reply lost")
        return {"checkpoint_id": identity}

    def restore(self, operation_id, identity, *, timeout_seconds):
        self.calls.append(("restore", operation_id, identity))
        if len(self.calls) <= self.lost:
            raise TerminalUnavailable("reply lost")
        return {"operation_id": operation_id, "checkpoint_id": identity, "exact": True}


def test_a_lost_reply_is_asked_for_again_by_the_same_id_a_bounded_number_of_times():
    recovered = FlakyClient(lost=2)
    assert IssuerEnvironment(recovered).restore("undo_1", "initial", timeout_seconds=30)["exact"] is True
    assert recovered.calls == [("restore", "undo_1", "initial")] * 3
    gone = FlakyClient(lost=10)
    with pytest.raises(TerminalUnavailable):
        IssuerEnvironment(gone).checkpoint("initial", timeout_seconds=30)
    assert gone.calls == [("checkpoint", "initial")] * 3
    with pytest.raises(TypeError):
        IssuerEnvironment(object())


def test_only_a_rollback_policy_gives_the_runtime_a_task_environment(tmp_path):
    from taste.brains.azure_central_host import compose_azure_central_runtime
    from taste.brains.azure_execution_policy import AzureExecutionPolicy
    from taste.brains.central_planner import Goal
    from taste.brains.terminal_broker import TerminalBinding
    from taste.brains.terminal_worker_policy import TerminalWorkerPolicy
    from taste.pricing import table_sha
    from tests.test_azure_openai import config
    from tests.test_brains_central_host import NoLaunchLauncher

    deadline = time.time() + 120
    policy = AzureExecutionPolicy(
        endpoint=config().base_url, planner_deployment="gpt-6-astra", worker_deployment="gpt-6-sol",
        deadline_unix=deadline, worker_budget_usd=20, monitor_budget_usd=20, worker_max_calls=8,
        monitor_max_calls=16, worker_max_output_tokens=256, monitor_max_output_tokens=512,
        planner_max_output_tokens=512, monitor_batch_size=1, pricing_sha=table_sha(),
        terminal=TerminalWorkerPolicy(TerminalBinding("trial_1", "container_instance_123", deadline, 100), 5),
        rollback=True)
    goal = Goal(goal_id="azure-goal", task="Write output.txt with the exact text correct",
                success_criteria=("output.txt contains correct",), budget_usd=100)
    environment = {"AZURE_OPENAI_BASE_URL": config().base_url, "AZURE_OPENAI_API_KEY": "azure-test-only"}
    controller = FakeEnvironment()

    def compose(name, chosen, **kwargs):
        root = tmp_path / name
        root.mkdir()
        return compose_azure_central_runtime(root, "azure-central", goal, policy=chosen,
                                             environment=environment, **kwargs)

    for name, chosen, kwargs, expected in (
            ("on", policy, {"launcher": NoLaunchLauncher(), "task_environment": controller}, controller),
            ("off", replace(policy, rollback=False), {"launcher": NoLaunchLauncher(),
                                                      "task_environment": controller}, None),
            ("settling", policy, {"settlement_only": True, "task_environment": controller}, None)):
        host = compose(name, chosen, **kwargs)
        try:
            assert host.runtime.task_environment is expected, name
        finally:
            host.close()
    with pytest.raises(TypeError, match="issuer client"):
        compose("no-issuer", policy, launcher=NoLaunchLauncher())
    with pytest.raises(ValueError, match="rollback needs a task terminal"):
        replace(policy, terminal=None)
