from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from taste.brains.central_planner import (
    CentralPlanner,
    Goal,
    InvalidPlannerOutput,
    PlannerTransportError,
    PlanningRequest,
    StalePlanningWorld,
)
from taste.brains.central_runtime import (
    BudgetBlocked,
    CentralRuntime,
    CoordinatorCorruption,
    ExternalSignal,
    GoalOutcome,
    RuntimeTrigger,
    SharedRuntimeStateError,
    _outcome_path,
)
from taste.brains.contract import Contract
from taste.brains.monitor import Judgement, MonitorBrain, Severity, TerminalDecision
from taste.brains.planner_transport import (
    PlannerCompletion,
    PlannerTelemetry,
    PlannerUsage,
)
from taste.brains.records import (
    ArtifactRef,
    ArtifactSpec,
    Assignment,
    WorkerReport,
    contract_digest,
)
from taste.brains.supervisor import CentralSupervisor, ProcessExit
from taste.brains.worker_runtime import WORKER_REPORT_PATH
from taste.memstore import Store


class FakeHandle:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.process_group_id = pid
        self.launch_token = f"launch-{pid}"
        self.process_identity = f"process-{pid}"
        self.exit: ProcessExit | None = None
        self.termination_calls = 0

    def poll(self) -> ProcessExit | None:
        return self.exit

    def ready(self) -> bool:
        return True

    def terminate_tree(self, grace_seconds: float) -> ProcessExit:
        self.termination_calls += 1
        if self.exit is None:
            self.exit = ProcessExit(signal=9, reaped=True)
        return self.exit


class FakeLauncher:
    def __init__(self) -> None:
        self.handles: dict[str, FakeHandle] = {}
        self.launch_calls: list[str] = []
        self.fail_once_assignment_ids: set[str] = set()

    def launch(self, spec):
        self.launch_calls.append(spec.run_id)
        if spec.assignment.assignment_id in self.fail_once_assignment_ids:
            self.fail_once_assignment_ids.remove(spec.assignment.assignment_id)
            raise RuntimeError("injected launch failure")
        handle = self.handles.setdefault(spec.run_id, FakeHandle(7000 + len(self.handles)))
        return handle

    def recover(self, spec):
        return self.handles.get(spec.run_id)

    def cancel(self, spec, grace_seconds):
        handle = self.handles.get(spec.run_id)
        return handle.terminate_tree(grace_seconds) if handle else ProcessExit(reaped=True)


class ScriptedTransport:
    def __init__(
        self,
        responder: Callable[[PlanningRequest, str], tuple[Assignment, ...] | str],
        *,
        telemetry: PlannerTelemetry | None = None,
        call_ceiling_usd: float = 0.0,
    ) -> None:
        self.responder = responder
        self.telemetry = telemetry
        self.call_ceiling_usd = call_ceiling_usd
        self.calls: list[PlanningRequest] = []
        self.bound_remaining_usd: list[float] = []

    def max_billed_call_usd(self) -> float:
        return self.call_ceiling_usd

    def bind_budget(self, *, remaining_usd: float) -> None:
        self.bound_remaining_usd.append(remaining_usd)

    def complete(
        self, *, request_id: str, system: str, prompt: str
    ) -> str | PlannerCompletion:
        payload = json.loads(prompt)
        request = PlanningRequest.from_dict(payload["request"])
        self.calls.append(request)
        response = self.responder(request, prompt)
        text = response if isinstance(response, str) else proposal(prompt, *response)
        if self.telemetry is None and request.goal.budget_usd is None:
            return text
        return PlannerCompletion(
            text,
            self.telemetry or PlannerTelemetry.synthetic_zero(),
        )


class CountingSupervisor(CentralSupervisor):
    delivery_calls = 0

    def deliver(self, run_id: str, *, active_generation: int):
        type(self).delivery_calls += 1
        return super().deliver(run_id, active_generation=active_generation)


def assignment_for(
    request: PlanningRequest,
    assignment_id: str,
    worker: str,
    path: str,
    *,
    depends_on: tuple[str, ...] = (),
    budget_usd: float | None = None,
    monitor_budget_usd: float | None = None,
    inputs: tuple[ArtifactRef, ...] = (),
) -> Assignment:
    contract = Contract(
        identity=worker,
        task=f"produce {path}",
        inputs=tuple(item.path for item in inputs),
        outputs=(path,),
        success_criteria=(f"{path} is correct",),
        budget_usd=budget_usd,
    )
    resources: dict[str, Any] = {"wall_timeout_seconds": 60}
    if budget_usd is not None:
        resources["monitor_budget_usd"] = (
            budget_usd if monitor_budget_usd is None else monitor_budget_usd
        )
    return Assignment(
        assignment_id=assignment_id,
        generation=request.generation,
        attempt=0,
        contract=contract,
        contract_digest=contract_digest(contract),
        base_state_id=request.world.integration_state_id,
        depends_on=depends_on,
        inputs=inputs,
        outputs=(ArtifactSpec(artifact_id=f"artifact-{assignment_id}", path=path),),
        resources=resources,
    )


def proposal(
    prompt: str,
    *assignments: Assignment,
    complete: bool = False,
) -> str:
    payload = json.loads(prompt)
    response = dict(payload["required_output_shape"])
    # Completion has to account for every standing obligation, so the default
    # mirrors what a real planner would have to state rather than letting a
    # scripted transport declare completion for free.
    verdict = "met" if complete else "not_met"
    evidence = "satisfied by the delivered work" if complete else "work is in flight"
    response.update(
        assignments=[item.to_dict() for item in assignments],
        rationale="mechanically testable plan",
        complete=complete,
        completion_reason="all goal criteria are durably satisfied" if complete else "",
        assessment=[
            {"criterion_id": item["criterion_id"], "verdict": verdict, "evidence": evidence}
            for item in payload.get("standing_criteria", ())
        ],
        metadata={},
    )
    return json.dumps(response, sort_keys=True)


def one_assignment_response_runtime(request: PlanningRequest, prompt: str):
    return (assignment_for(request, "build", "worker-build", "product.txt"),)


def complete_response(_request: PlanningRequest, prompt: str) -> str:
    return proposal(prompt, complete=True)


class FineJudge:
    def judge_terminal(self, _contract, _state, _context, findings):
        return TerminalDecision(
            judgement=Judgement(Severity.FINE, "exact state satisfies the assignment"),
            resolved_finding_ids=tuple(item.id for item in findings),
        )


def install_report(
    runtime: CentralRuntime,
    assignment_id: str,
    *,
    content: str = "certified product\n",
    completed: bool = True,
    cost_usd: float | None = 0.25,
) -> WorkerReport:
    run = next(
        item
        for item in runtime.supervisor.runs()
        if item.assignment.assignment_id == assignment_id
        and item.assignment.generation
        == runtime.planner.current_plan(runtime.goal.goal_id).generation
    )
    assignment = run.assignment
    worker = runtime.store.branch(assignment.worker)
    try:
        outputs: list[ArtifactRef] = []
        for spec in assignment.outputs:
            if spec.disposition == "absent":
                continue
            worker.write(spec.path, content)
        work = worker.checkpoint(f"terminal product for {assignment_id}")
        for spec in assignment.outputs:
            if spec.disposition == "absent":
                continue
            outputs.append(
                ArtifactRef(
                    artifact_id=spec.artifact_id,
                    branch=assignment.worker,
                    state_id=work.id,
                    path=spec.path,
                    blob_id=work.blob(spec.path) or "",
                )
            )
        monitor = MonitorBrain(runtime.store, assignment.contract, FineJudge(), batch_size=100)
        assessment = asyncio.run(monitor.certify_terminal(work, context={"test": True}))
        report = WorkerReport(
            report_id=f"report.{assignment.generation}.{assignment.assignment_id}",
            run_id=run.run_id,
            assignment_id=assignment.assignment_id,
            worker=assignment.worker,
            generation=assignment.generation,
            attempt=assignment.attempt,
            contract_digest=assignment.contract_digest,
            base_state_id=assignment.base_state_id,
            final_state_id=work.id,
            at=work.meta.created_at,
            completed=completed,
            terminal_reason="completed" if completed else "incomplete",
            outputs=tuple(outputs),
            cost_usd=cost_usd,
            monitor_severity="fine",
            metadata={
                "durability_ok": True,
                "monitor": {
                    "worst": "fine",
                    "current": "fine",
                    "current_state": work.id,
                    "pending_actions": [],
                    "terminal_assessment": assessment.to_dict(),
                },
            },
        )
        worker.write(WORKER_REPORT_PATH, report.to_json())
        worker.checkpoint(f"terminal report for {assignment_id}")
    finally:
        worker.close()
    return report


@pytest.fixture
def store(tmp_path: Path):
    opened = Store.open(tmp_path / "repo", "runtime-test")
    yield opened
    opened.close()


def stack(
    store: Store,
    goal: Goal,
    transport: ScriptedTransport,
    launcher: FakeLauncher,
    *,
    fault=None,
    supervisor_type=CentralSupervisor,
    shared=None,
    communication=None,
):
    if shared is None:
        lock = threading.RLock()
        control = store.branch("central-control", producer="central-runtime")
        integration = store.branch("integration", producer="central-integration")
    else:
        control, integration, lock = shared
    planner = CentralPlanner(
        store,
        transport=transport,
        control=control,
        control_branch=control.name,
        integration_branch=integration.name,
        mutation_lock=lock,
    )
    supervisor = supervisor_type(
        store,
        launcher=launcher,
        control_branch=control,
        integration_branch=integration,
        control_lock=lock,
    )
    runtime = CentralRuntime(
        store,
        goal,
        planner=planner,
        supervisor=supervisor,
        control=control,
        integration=integration,
        control_lock=lock,
        fault_injector=fault,
        communication=communication,
    )
    return runtime, (control, integration, lock)


def simple_goal(*, budget_usd: float | None = None) -> Goal:
    return Goal(
        goal_id="runtime-goal",
        task="produce the requested product files",
        success_criteria=("all product files are integrated",),
        budget_usd=budget_usd,
    )


def test_restart_recovers_current_plan_and_live_run_without_planner_or_spawn(store: Store) -> None:
    def respond(request: PlanningRequest, prompt: str):
        if request.generation == 1:
            return (assignment_for(request, "build", "worker-build", "product.txt"),)
        return complete_response(request, prompt)

    transport = ScriptedTransport(respond)
    launcher = FakeLauncher()
    runtime, shared = stack(store, simple_goal(), transport, launcher)
    assert runtime.cycle().status == "planned"
    assert runtime.cycle().status == "running"
    assert len(transport.calls) == 1
    assert len(launcher.launch_calls) == 1

    replacement, _ = stack(store, simple_goal(), transport, launcher, shared=shared)
    recovered = replacement.cycle()
    assert recovered.status == "running"
    assert len(transport.calls) == 1
    assert len(launcher.launch_calls) == 1


def test_crash_after_start_does_not_duplicate_spawn(store: Store) -> None:
    transport = ScriptedTransport(
        lambda request, _prompt: (assignment_for(request, "build", "worker-build", "product.txt"),)
    )
    launcher = FakeLauncher()
    fired = False

    def fault(boundary: str, _payload: Any) -> None:
        nonlocal fired
        if boundary == "decision_effect:start" and not fired:
            fired = True
            raise RuntimeError("central process died after durable spawn")

    runtime, shared = stack(store, simple_goal(), transport, launcher, fault=fault)
    runtime.cycle()
    with pytest.raises(RuntimeError, match="died after durable spawn"):
        runtime.cycle()
    assert len(launcher.launch_calls) == 1

    replacement, _ = stack(store, simple_goal(), transport, launcher, shared=shared)
    assert replacement.cycle().status == "running"
    assert len(launcher.launch_calls) == 1
    assert len(transport.calls) == 1


def test_dependencies_release_only_after_exact_delivery(store: Store) -> None:
    def respond(request: PlanningRequest, _prompt: str):
        if request.generation == 1:
            first = assignment_for(request, "first", "worker-first", "first.txt")
            downstream_template = assignment_for(
                request,
                "second-template",
                "worker-second-template",
                "second.txt",
                depends_on=("first",),
            )
            return first, downstream_template
        integration = store.state(request.world.integration_state_id)
        source = ArtifactRef(
            artifact_id="artifact-first",
            branch="integration",
            state_id=integration.id,
            path="first.txt",
            blob_id=integration.blob("first.txt") or "",
        )
        return (
            assignment_for(
                request,
                "second",
                "worker-second",
                "second.txt",
                inputs=(source,),
            ),
        )

    transport = ScriptedTransport(respond)
    launcher = FakeLauncher()
    runtime, _ = stack(store, simple_goal(), transport, launcher)
    runtime.cycle()
    first_cycle = runtime.cycle()
    assert len(launcher.launch_calls) == 1
    assert first_cycle.waiting_assignment_ids == ("second-template",)

    install_report(runtime, "first")
    first_run = next(
        run for run in runtime.supervisor.runs() if run.assignment.assignment_id == "first"
    )
    launcher.handles[first_run.run_id].exit = ProcessExit(exit_code=0, reaped=True)
    wave = runtime.cycle()
    assert wave.status == "replanned"
    assert any(item.kind == "dependency_wave_delivered" for item in wave.triggers)
    assert len(launcher.launch_calls) == 1
    second_cycle = runtime.cycle()
    assert second_cycle.status == "running"
    assert len(launcher.launch_calls) == 2
    assert runtime.integration.head.read("first.txt") == "certified product\n"
    second_run = next(
        run for run in runtime.supervisor.runs() if run.assignment.assignment_id == "second"
    )
    prepared = store.state(second_run.prepared_state_id)
    assert prepared.read("first.txt") == "certified product\n"


def test_partial_success_is_delivered_before_failure_replan(store: Store) -> None:
    seen_integration: list[str] = []

    def respond(request: PlanningRequest, prompt: str):
        seen_integration.append(request.world.integration_state_id)
        if request.generation == 1:
            return (
                assignment_for(request, "good", "worker-good", "good.txt"),
                assignment_for(request, "bad", "worker-bad", "bad.txt"),
            )
        return complete_response(request, prompt)

    transport = ScriptedTransport(respond)
    launcher = FakeLauncher()
    runtime, _ = stack(store, simple_goal(), transport, launcher)
    runtime.cycle()
    runtime.cycle()
    install_report(runtime, "good")
    runs = {run.assignment.assignment_id: run for run in runtime.supervisor.runs()}
    launcher.handles[runs["good"].run_id].exit = ProcessExit(exit_code=0, reaped=True)
    launcher.handles[runs["bad"].run_id].exit = ProcessExit(exit_code=2, reaped=True)

    outcome = runtime.cycle()
    assert outcome.complete
    assert runtime.integration.head.read("good.txt") == "certified product\n"
    assert "good" in outcome.delivered_assignment_ids
    assert "bad" in outcome.failed_assignment_ids
    assert len(transport.calls) == 2
    assert seen_integration[1] == runtime.integration.head.id


def test_crash_after_delivery_and_replan_reuses_both_effects(store: Store) -> None:
    def respond(request: PlanningRequest, prompt: str):
        if request.generation == 1:
            return (assignment_for(request, "build", "worker-build", "product.txt"),)
        return complete_response(request, prompt)

    transport = ScriptedTransport(respond)
    launcher = FakeLauncher()
    CountingSupervisor.delivery_calls = 0
    boundary_to_fail = {"decision_effect:deliver"}

    def fault(boundary: str, _payload: Any) -> None:
        if boundary in boundary_to_fail:
            boundary_to_fail.remove(boundary)
            raise RuntimeError("injected central crash")

    runtime, shared = stack(
        store,
        simple_goal(),
        transport,
        launcher,
        fault=fault,
        supervisor_type=CountingSupervisor,
    )
    runtime.cycle()
    runtime.cycle()
    install_report(runtime, "build")
    run = runtime.supervisor.runs()[0]
    launcher.handles[run.run_id].exit = ProcessExit(exit_code=0, reaped=True)
    with pytest.raises(RuntimeError, match="injected central crash"):
        runtime.cycle()
    assert CountingSupervisor.delivery_calls == 1

    # Fail after the planner has promoted the completion revision too.  A
    # second replacement must still make no third model call.
    boundary_to_fail.add("decision_effect:replan")
    replacement, _ = stack(
        store,
        simple_goal(),
        transport,
        launcher,
        fault=fault,
        supervisor_type=CountingSupervisor,
        shared=shared,
    )
    with pytest.raises(RuntimeError, match="injected central crash"):
        replacement.cycle()
    assert CountingSupervisor.delivery_calls == 1
    assert len(transport.calls) == 2

    final, _ = stack(
        store,
        simple_goal(),
        transport,
        launcher,
        supervisor_type=CountingSupervisor,
        shared=shared,
    )
    assert final.cycle().complete
    assert CountingSupervisor.delivery_calls == 1
    assert len(transport.calls) == 2


def test_stale_generation_report_is_fenced_and_old_live_worker_is_stopped(store: Store) -> None:
    def respond(request: PlanningRequest, _prompt: str):
        path = "old.txt" if request.generation == 1 else "new.txt"
        return (
            assignment_for(
                request,
                f"build-{request.generation}",
                f"worker-{request.generation}",
                path,
            ),
        )

    transport = ScriptedTransport(respond)
    launcher = FakeLauncher()
    runtime, _ = stack(store, simple_goal(), transport, launcher)
    runtime.cycle()
    runtime.cycle()
    install_report(runtime, "build-1", content="stale bytes\n")
    old_run = runtime.supervisor.runs()[0]

    promoted = runtime.planner.revise(runtime.goal, operation_id="external-revision")
    assert promoted.generation == 2
    outcome = runtime.cycle()
    assert outcome.status == "running"
    assert runtime.integration.head.read("old.txt") is None
    assert launcher.handles[old_run.run_id].termination_calls >= 1
    assert len(launcher.launch_calls) == 2


def test_delivery_conflict_is_durable_and_causes_replan(store: Store) -> None:
    def respond(request: PlanningRequest, prompt: str):
        if request.generation == 1:
            return (assignment_for(request, "build", "worker-build", "shared.txt"),)
        return complete_response(request, prompt)

    transport = ScriptedTransport(respond)
    launcher = FakeLauncher()
    runtime, _ = stack(store, simple_goal(), transport, launcher)
    runtime.cycle()
    runtime.cycle()
    install_report(runtime, "build", content="worker version\n")
    runtime.integration.write("shared.txt", "central version\n")
    runtime.integration.checkpoint("concurrent central product edit")
    run = runtime.supervisor.runs()[0]
    launcher.handles[run.run_id].exit = ProcessExit(exit_code=0, reaped=True)

    outcome = runtime.cycle()
    assert outcome.complete
    assert any(item.kind == "delivery_conflict" for item in outcome.triggers)
    durable = runtime.supervisor.get(run.run_id)
    assert durable.phase == "conflict"
    assert durable.conflict_paths == ("shared.txt",)


def test_delivered_run_is_not_satisfied_after_integration_rollback(store: Store) -> None:
    transport = ScriptedTransport(
        lambda request, _prompt: (
            assignment_for(request, "build", "worker-build", "product.txt"),
        )
    )
    launcher = FakeLauncher()
    runtime, _ = stack(store, simple_goal(), transport, launcher)
    runtime.cycle()
    runtime.cycle()
    install_report(runtime, "build")
    run = runtime.supervisor.runs()[0]
    launcher.handles[run.run_id].exit = ProcessExit(exit_code=0, reaped=True)
    runtime.supervisor.poll(run.run_id, active_generation=1)
    delivered = runtime.supervisor.deliver(run.run_id, active_generation=1)
    durable = runtime.supervisor.get(run.run_id)
    assert delivered.ok
    assert runtime._is_delivered(durable)

    runtime.integration.rollback(
        store.state(run.assignment.base_state_id),
        "operator rolls back integrated product",
    )

    assert not runtime._is_delivered(durable)


def test_unknown_cost_is_not_zero_and_prevents_new_budgeted_spawn(store: Store) -> None:
    def respond(request: PlanningRequest, prompt: str):
        if request.generation == 1:
            return (assignment_for(request, "first", "worker-first", "first.txt", budget_usd=1.0),)
        if request.generation == 2:
            return (
                assignment_for(request, "second", "worker-second", "second.txt", budget_usd=1.0),
            )
        return complete_response(request, prompt)

    transport = ScriptedTransport(respond)
    launcher = FakeLauncher()
    runtime, _ = stack(store, simple_goal(budget_usd=5.0), transport, launcher)
    runtime.cycle()
    runtime.cycle()
    install_report(runtime, "first", cost_usd=None)
    run = runtime.supervisor.runs()[0]
    launcher.handles[run.run_id].exit = ProcessExit(exit_code=0, reaped=True)
    blocked = runtime.cycle()
    assert blocked.status == "budget_blocked"
    assert blocked.plan.generation == 1
    assert blocked.budget.known_spent_usd == 0.0
    assert blocked.budget.unknown_run_ids == (run.run_id,)
    assert len(transport.calls) == 1

    blocked_again = runtime.cycle()
    assert blocked_again.status == "budget_blocked"
    assert len(launcher.launch_calls) == 1
    assert len(transport.calls) == 1
    assert any(item.kind == "budget_unknown" for item in blocked_again.triggers)


def test_global_budget_counts_planner_cost_and_both_worker_reservations(
    store: Store,
) -> None:
    telemetry = PlannerTelemetry(
        source="provider_completion",
        cost_known=True,
        requested_model="claude-sonnet-5",
        model="claude-sonnet-5",
        provider="anthropic",
        usage=PlannerUsage(100, 20, 0, 0, 0),
        billed_usd=0.25,
        work_usd=0.5,
        pricing_table_sha="test-pricing-table",
        pricing_as_of="2026-09-10",
    )

    def respond(request: PlanningRequest, _prompt: str):
        return (
            assignment_for(
                request,
                "build",
                "worker-build",
                "product.txt",
                budget_usd=1.0,
                monitor_budget_usd=0.5,
            ),
        )

    transport = ScriptedTransport(respond, telemetry=telemetry)
    launcher = FakeLauncher()
    runtime, _ = stack(store, simple_goal(budget_usd=10.0), transport, launcher)

    planned = runtime.cycle()
    assert planned.budget.planner_spent_usd == pytest.approx(0.25)
    assert planned.budget.worker_spent_usd == 0.0
    assert planned.budget.known_spent_usd == pytest.approx(0.25)
    assert planned.budget.reserved_usd == 0.0

    running = runtime.cycle()
    assert running.status == "running"
    assert running.budget.planner_spent_usd == pytest.approx(0.25)
    assert running.budget.worker_spent_usd == 0.0
    assert running.budget.known_spent_usd == pytest.approx(0.25)
    assert running.budget.reserved_usd == pytest.approx(1.5)
    assert running.budget.remaining_usd == pytest.approx(8.25)


def test_unknown_bootstrap_planner_cost_cannot_be_retried_under_goal_budget(
    store: Store,
) -> None:
    def fail_after_possible_charge(_request: PlanningRequest, _prompt: str):
        raise RuntimeError("provider outcome is ambiguous")

    transport = ScriptedTransport(fail_after_possible_charge)
    runtime, _ = stack(
        store,
        simple_goal(budget_usd=10.0),
        transport,
        FakeLauncher(),
    )

    with pytest.raises(PlannerTransportError, match="ambiguous"):
        runtime.cycle()
    assert len(transport.calls) == 1

    with pytest.raises(BudgetBlocked, match="durable cost is unknown"):
        runtime.cycle()
    assert len(transport.calls) == 1


def test_planner_max_call_exposure_cannot_overspend_small_remainder(
    store: Store,
) -> None:
    def respond(request: PlanningRequest, prompt: str):
        if request.generation == 1:
            return (
                assignment_for(
                    request,
                    "build",
                    "worker-build",
                    "product.txt",
                    budget_usd=0.2,
                    monitor_budget_usd=0.2,
                ),
            )
        return complete_response(request, prompt)

    transport = ScriptedTransport(respond, call_ceiling_usd=0.5)
    launcher = FakeLauncher()
    runtime, _ = stack(store, simple_goal(budget_usd=1.0), transport, launcher)
    runtime.cycle()
    runtime.cycle()
    install_report(runtime, "build", cost_usd=0.99)
    run = runtime.supervisor.runs()[0]
    launcher.handles[run.run_id].exit = ProcessExit(exit_code=0, reaped=True)

    blocked = runtime.cycle()

    assert blocked.status == "budget_blocked"
    assert runtime.integration.head.read("product.txt") == "certified product\n"
    assert blocked.budget.remaining_usd == pytest.approx(0.01)
    assert len(transport.calls) == 1
    assert transport.bound_remaining_usd == [pytest.approx(1.0)]


def test_completed_reconcile_decision_can_replay_after_process_state_changes(store: Store) -> None:
    def respond(request: PlanningRequest, prompt: str):
        if request.generation == 1:
            return (assignment_for(request, "build", "worker-build", "product.txt"),)
        return complete_response(request, prompt)

    transport = ScriptedTransport(respond)
    launcher = FakeLauncher()
    armed = False

    def fault(boundary: str, _payload: Any) -> None:
        if armed and boundary == "decision_result:reconcile":
            raise RuntimeError("crash after reconcile result")

    runtime, shared = stack(store, simple_goal(), transport, launcher, fault=fault)
    runtime.cycle()
    runtime.cycle()
    run = runtime.supervisor.runs()[0]
    armed = True
    with pytest.raises(RuntimeError, match="crash after reconcile result"):
        runtime.cycle()

    launcher.handles[run.run_id].exit = ProcessExit(exit_code=2, reaped=True)
    replacement, _ = stack(store, simple_goal(), transport, launcher, shared=shared)
    recovered = replacement.cycle()
    assert recovered.complete
    assert len(transport.calls) == 2


def test_one_shot_replan_signal_remains_pending_while_worker_is_live(store: Store) -> None:
    class OneShotHook:
        armed = False
        emitted = False

        def signals(self, *, goal, plan, runs):
            if not self.armed or self.emitted:
                return ()
            self.emitted = True
            return (
                ExternalSignal(
                    signal_id="central-message-1",
                    kind="new_constraint",
                    detail="the accepted user constraint requires replanning",
                    requires_replan=True,
                ),
            )

    observed_trigger_context: list[str] = []

    def respond(request: PlanningRequest, prompt: str):
        if request.generation == 1:
            return (
                assignment_for(request, "first", "worker-first", "first.txt"),
                assignment_for(
                    request,
                    "second",
                    "worker-second",
                    "second.txt",
                    depends_on=("first",),
                ),
            )
        control = next(
            item
            for item in request.world.observations
            if item.branch == request.world.control_branch
        )
        observed_trigger_context.extend(
            entry["description"]
            for entry in control.manifest["entries"].values()
            if entry["name"].startswith("central-runtime-trigger.")
        )
        return complete_response(request, prompt)

    hook = OneShotHook()
    transport = ScriptedTransport(respond)
    launcher = FakeLauncher()
    runtime, _ = stack(
        store, simple_goal(), transport, launcher, communication=hook
    )
    runtime.cycle()
    runtime.cycle()
    assert len(launcher.launch_calls) == 1

    hook.armed = True
    signalled = runtime.cycle()
    assert any(item.kind == "external_new_constraint" for item in signalled.triggers)
    still_pending = runtime.cycle()
    assert any(item.kind == "external_new_constraint" for item in still_pending.triggers)
    assert len(launcher.launch_calls) == 1

    install_report(runtime, "first")
    first = runtime.supervisor.runs()[0]
    launcher.handles[first.run_id].exit = ProcessExit(exit_code=0, reaped=True)
    outcome = runtime.cycle()
    assert outcome.complete
    assert runtime.integration.head.read("first.txt") == "certified product\n"
    assert len(launcher.launch_calls) == 1
    assert any(
        "the accepted user constraint requires replanning" in item
        for item in observed_trigger_context
    )


def test_start_failure_trigger_stops_unstarted_retry_and_survives_live_peer(store: Store) -> None:
    def respond(request: PlanningRequest, prompt: str):
        if request.generation == 1:
            return (
                assignment_for(request, "first", "worker-first", "first.txt"),
                assignment_for(request, "second", "worker-second", "second.txt"),
            )
        return complete_response(request, prompt)

    transport = ScriptedTransport(respond)
    launcher = FakeLauncher()
    launcher.fail_once_assignment_ids.add("second")
    runtime, _ = stack(store, simple_goal(), transport, launcher)
    runtime.cycle()
    failed_start = runtime.cycle()
    assert failed_start.status == "running"
    assert any(item.kind == "worker_start_failed" for item in failed_start.triggers)
    assert len(launcher.launch_calls) == 2

    pending = runtime.cycle()
    second = next(
        run for run in runtime.supervisor.runs() if run.assignment.assignment_id == "second"
    )
    assert second.phase == "terminal"
    assert any(item.kind == "worker_start_failed" for item in pending.triggers)
    assert len(launcher.launch_calls) == 2

    install_report(runtime, "first")
    first = next(
        run for run in runtime.supervisor.runs() if run.assignment.assignment_id == "first"
    )
    launcher.handles[first.run_id].exit = ProcessExit(exit_code=0, reaped=True)
    assert runtime.cycle().complete
    assert len(launcher.launch_calls) == 2


def test_cycle_index_detects_deleted_historical_cycle_subtree(store: Store) -> None:
    transport = ScriptedTransport(
        lambda request, _prompt: (
            assignment_for(request, "build", "worker-build", "product.txt"),
        )
    )
    launcher = FakeLauncher()
    runtime, _ = stack(store, simple_goal(), transport, launcher)
    runtime.cycle()
    intent_path = next(
        path
        for path in runtime.control.head.files()
        if "/cycles/" in path
        and path.endswith("/intent.json")
        and "/decisions/" not in path
    )
    cycle_root = intent_path.rsplit("/", 1)[0]
    for path in runtime.control.head.files():
        if path == cycle_root or path.startswith(f"{cycle_root}/"):
            runtime.control.path(path).unlink()
    runtime.control.checkpoint("inject deletion of a complete coordinator cycle")

    with pytest.raises(CoordinatorCorruption, match="indexed cycle intent"):
        runtime.cycle()


def test_trigger_index_detects_deleted_pending_trigger(store: Store) -> None:
    transport = ScriptedTransport(
        lambda request, _prompt: (
            assignment_for(request, "build", "worker-build", "product.txt"),
        )
    )
    runtime, _ = stack(store, simple_goal(), transport, FakeLauncher())
    runtime.cycle()
    plan = runtime.planner.current_plan(runtime.goal.goal_id)
    assert plan is not None
    cycle = runtime._begin_cycle(plan)
    trigger = RuntimeTrigger(
        kind="external_constraint",
        subject_id="message-1",
        detail="accepted once and must remain pending",
    )
    runtime._record_trigger(cycle, trigger)
    trigger_path = next(
        path
        for path in runtime.control.head.files()
        if "/triggers/" in path and path.endswith(".json")
    )
    runtime.control.path(trigger_path).unlink()
    runtime.control.checkpoint("inject deletion of pending trigger")

    with pytest.raises(CoordinatorCorruption, match="trigger files"):
        runtime._pending_triggers(plan)


def test_terminally_rejected_bootstrap_advances_to_fresh_durable_epoch(store: Store) -> None:
    attempts = 0

    def respond(request: PlanningRequest, _prompt: str):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return "{}"
        return (assignment_for(request, "build", "worker-build", "product.txt"),)

    transport = ScriptedTransport(respond)
    launcher = FakeLauncher()
    runtime, shared = stack(store, simple_goal(), transport, launcher)
    with pytest.raises(InvalidPlannerOutput):
        runtime.cycle()

    replacement, _ = stack(store, simple_goal(), transport, launcher, shared=shared)
    assert replacement.cycle().status == "planned"
    assert len(transport.calls) == 2
    assert transport.calls[0].operation_id != transport.calls[1].operation_id


def test_rejected_planner_epoch_index_deletion_fails_closed(store: Store) -> None:
    transport = ScriptedTransport(lambda _request, _prompt: "{}")
    runtime, _ = stack(store, simple_goal(), transport, FakeLauncher())
    with pytest.raises(InvalidPlannerOutput):
        runtime.cycle()
    index_path = next(
        path
        for path in runtime.control.head.files()
        if "/planner-operations/" in path and path.endswith("/index.json")
    )
    runtime.control.path(index_path).unlink()
    runtime.control.checkpoint("inject deletion of planner retry epoch index")

    with pytest.raises(CoordinatorCorruption, match="index disappeared"):
        runtime.cycle()


def test_stale_replan_advances_epoch_and_uses_fresh_world_after_restart(store: Store) -> None:
    moved = False

    def respond(request: PlanningRequest, prompt: str):
        nonlocal moved
        if request.generation == 1:
            return (assignment_for(request, "build", "worker-build", "product.txt"),)
        if not moved:
            moved = True
            runtime.integration.write("external.txt", "new exact world\n")
            runtime.integration.checkpoint("move world during planning")
        return complete_response(request, prompt)

    transport = ScriptedTransport(respond)
    runtime, shared = stack(store, simple_goal(), transport, FakeLauncher())
    runtime.cycle()
    plan = runtime.planner.current_plan(runtime.goal.goal_id)
    assert plan is not None
    cycle = runtime._begin_cycle(plan)
    trigger = RuntimeTrigger(
        kind="manual_replan",
        subject_id=plan.plan_id,
        detail="exercise stale-world epoch recovery",
    )
    with pytest.raises(StalePlanningWorld):
        runtime._replan(cycle, plan, (trigger,))

    replacement, _ = stack(
        store, simple_goal(), transport, runtime.supervisor.launcher, shared=shared
    )
    resumed_cycle = replacement._begin_cycle(plan)
    revised = replacement._replan(resumed_cycle, plan, (trigger,))
    assert revised.complete
    assert len(transport.calls) == 3
    assert transport.calls[1].operation_id != transport.calls[2].operation_id
    assert transport.calls[2].world.integration_state_id == runtime.integration.head.id


def test_constructor_rejects_lookalike_but_different_lock(store: Store) -> None:
    transport = ScriptedTransport(lambda request, _prompt: ())
    launcher = FakeLauncher()
    goal = simple_goal()
    lock = threading.RLock()
    control = store.branch("central-control", producer="central-runtime")
    integration = store.branch("integration", producer="central-integration")
    planner = CentralPlanner(
        store,
        transport=transport,
        control=control,
        mutation_lock=lock,
    )
    supervisor = CentralSupervisor(
        store,
        launcher=launcher,
        control_branch=control,
        integration_branch=integration,
        control_lock=lock,
    )
    with pytest.raises(SharedRuntimeStateError, match="planner must use"):
        CentralRuntime(
            store,
            goal,
            planner=planner,
            supervisor=supervisor,
            control=control,
            integration=integration,
            control_lock=threading.RLock(),
        )


def test_three_worker_restart_failure_conflict_and_completion_acceptance(store: Store) -> None:
    generations: list[int] = []
    generation_two_base: list[str] = []

    def exact_input(branch: str, artifact_id: str, path: str) -> ArtifactRef:
        state = store.view(branch).head
        return ArtifactRef(
            artifact_id=artifact_id,
            branch=branch,
            state_id=state.id,
            path=path,
            blob_id=state.blob(path) or "",
        )

    def respond(request: PlanningRequest, prompt: str):
        generations.append(request.generation)
        if request.generation == 1:
            return (
                assignment_for(request, "build-a", "worker-a", "a.txt"),
                assignment_for(request, "build-b", "worker-b", "b.txt"),
                assignment_for(request, "build-c", "worker-c", "c.txt"),
            )
        if request.generation == 2:
            generation_two_base.append(request.world.integration_state_id)
            partial_b = exact_input("worker-b", "partial-b", "b.txt")
            proposed_c = exact_input("worker-c", "proposed-c", "c.txt")
            return (
                assignment_for(
                    request,
                    "retry-b",
                    "worker-b-retry",
                    "b.txt",
                    inputs=(partial_b,),
                ),
                assignment_for(
                    request,
                    "resolve-c",
                    "worker-c-resolver",
                    "c.txt",
                    inputs=(proposed_c,),
                ),
            )
        return complete_response(request, prompt)

    transport = ScriptedTransport(respond)
    launcher = FakeLauncher()
    CountingSupervisor.delivery_calls = 0
    runtime, shared = stack(
        store,
        simple_goal(),
        transport,
        launcher,
        supervisor_type=CountingSupervisor,
    )

    assert runtime.cycle().status == "planned"
    assert runtime.cycle().status == "running"
    assert len(launcher.launch_calls) == 3

    install_report(runtime, "build-a", content="final a\n")
    install_report(runtime, "build-c", content="first c proposal\n")
    dirty_b = store.branch("worker-b")
    try:
        dirty_b.write("b.txt", "recoverable partial b\n")
    finally:
        dirty_b.close()

    runtime.integration.write("c.txt", "concurrent central c\n")
    runtime.integration.checkpoint("create exact C delivery conflict")
    first_runs = {
        run.assignment.assignment_id: run for run in runtime.supervisor.runs()
    }
    launcher.handles[first_runs["build-a"].run_id].exit = ProcessExit(
        exit_code=0, reaped=True
    )
    launcher.handles[first_runs["build-b"].run_id].exit = ProcessExit(
        exit_code=2, reaped=True
    )
    launcher.handles[first_runs["build-c"].run_id].exit = ProcessExit(
        exit_code=0, reaped=True
    )

    first_outcome = runtime.cycle()
    assert first_outcome.status == "replanned"
    assert first_outcome.plan.generation == 2
    assert any(item.kind == "invalid_report" for item in first_outcome.triggers)
    assert any(item.kind == "delivery_conflict" for item in first_outcome.triggers)
    conflict_state = runtime.integration.head
    assert conflict_state.meta.kind == "conflict"
    assert tuple(item.path for item in conflict_state.conflicts) == ("c.txt",)
    assert runtime.integration.head.read("a.txt") == "final a\n"
    assert runtime.integration.head.read("c.txt") == "concurrent central c\n"

    failed_b = runtime.supervisor.get(first_runs["build-b"].run_id)
    assert failed_b.recovery_state_id is not None
    failed_b_recovery = store.state(failed_b.recovery_state_id)
    assert failed_b_recovery.read("b.txt") == "recoverable partial b\n"
    conflicted_c = runtime.supervisor.get(first_runs["build-c"].run_id)
    assert conflicted_c.phase == "conflict"
    assert conflicted_c.conflict_paths == ("c.txt",)
    assert generation_two_base == [conflict_state.id]
    assert all(
        assignment.base_state_id == conflict_state.id
        for assignment in first_outcome.plan.assignments
    )

    replacement, _ = stack(
        store,
        simple_goal(),
        transport,
        launcher,
        supervisor_type=CountingSupervisor,
        shared=shared,
    )
    second_outcome = replacement.cycle()
    assert second_outcome.status == "running"
    assert len(launcher.launch_calls) == 5
    assert len(set(launcher.launch_calls)) == 5
    retry_run = next(
        run
        for run in replacement.supervisor.runs()
        if run.assignment.assignment_id == "retry-b"
    )
    resolver_run = next(
        run
        for run in replacement.supervisor.runs()
        if run.assignment.assignment_id == "resolve-c"
    )
    assert store.state(retry_run.prepared_state_id).read("b.txt") == "recoverable partial b\n"
    assert (
        store.state(resolver_run.prepared_state_id).read("c.txt")
        == "first c proposal\n"
    )

    install_report(replacement, "retry-b", content="final b\n")
    install_report(replacement, "resolve-c", content="resolved final c\n")
    launcher.handles[retry_run.run_id].exit = ProcessExit(exit_code=0, reaped=True)
    launcher.handles[resolver_run.run_id].exit = ProcessExit(exit_code=0, reaped=True)

    completed = replacement.cycle()
    assert completed.complete
    assert completed.plan.generation == 3
    assert generations == [1, 2, 3]
    assert len(transport.calls) == 3
    assert len(launcher.launch_calls) == 5
    assert CountingSupervisor.delivery_calls == 4
    assert replacement.integration.head.read("a.txt") == "final a\n"
    assert replacement.integration.head.read("b.txt") == "final b\n"
    assert replacement.integration.head.read("c.txt") == "resolved final c\n"

    loaded_conflict = store.state(conflict_state.id)
    assert loaded_conflict.meta.kind == "conflict"
    assert tuple(item.path for item in loaded_conflict.conflicts) == ("c.txt",)
    all_runs = replacement.supervisor.runs()
    assert len(all_runs) == 5
    for run in all_runs:
        state_ids = (
            run.prepared_state_id,
            run.recovery_state_id,
            run.report_state_id,
            run.integration_state_id,
        )
        for state_id in state_ids:
            if state_id is None:
                continue
            reachable = store.state(state_id)
            assert reachable.id == state_id
            assert store.backend.tree_of(reachable.id)


def settle(runtime: CentralRuntime, launcher: FakeLauncher, assignment_id: str) -> None:
    """Report and reap ``assignment_id`` if, and only if, it is live now."""
    plan = runtime.planner.current_plan(runtime.goal.goal_id)
    if plan is None:
        return
    live = [
        item
        for item in runtime.supervisor.runs()
        if item.assignment.assignment_id == assignment_id
        and item.assignment.generation == plan.generation
    ]
    if not live:
        return
    run = live[0]
    handle = launcher.handles.get(run.run_id)
    if handle is None or handle.exit is not None:
        return
    install_report(runtime, assignment_id)
    handle.exit = ProcessExit(exit_code=0, reaped=True)



# ------------------------------------------------------------ run() and exit
#
# cycle() advances one step and returns; nothing in the repository drove it to
# a conclusion, so a goal had no ending and no record of one.  run() is that
# driver, and GoalOutcome is the durable answer to "what was asked, what was
# delivered, and on what evidence did it stop".
#
# Being stuck is not an ending.  A blocked dependency or an invalid report is
# a reason to replan -- that is what the central brain is for -- so the only
# stops are completion, a hard bound, and a budget that can no longer be
# proven.


def test_run_drives_cycles_until_the_plan_is_complete(store: Store) -> None:
    def respond(request: PlanningRequest, prompt: str):
        if request.generation == 1:
            return (assignment_for(request, "build", "worker-build", "product.txt"),)
        return complete_response(request, prompt)

    launcher = FakeLauncher()
    runtime, _ = stack(store, simple_goal(), ScriptedTransport(respond), launcher)

    def advance() -> None:
        settle(runtime, launcher, "build")

    outcome = runtime.run(max_generations=5, wall_clock_seconds=60.0, between_cycles=advance)
    assert outcome.stop_reason == "complete"
    assert outcome.complete
    assert outcome.generations >= 2


def test_the_outcome_is_durable_and_names_every_criterion(store: Store) -> None:
    def respond(request: PlanningRequest, prompt: str):
        if request.generation == 1:
            return (assignment_for(request, "build", "worker-build", "product.txt"),)
        return complete_response(request, prompt)

    launcher = FakeLauncher()
    goal = simple_goal()
    runtime, _ = stack(store, goal, ScriptedTransport(respond), launcher)

    def advance() -> None:
        settle(runtime, launcher, "build")

    outcome = runtime.run(max_generations=5, wall_clock_seconds=60.0, between_cycles=advance)

    # Read it back through the accessor rather than a hand-built path: the
    # goal key is a digest the layer owns, and the claim under test is that
    # the outcome survives the round trip, not where it happens to live.
    restored = runtime.outcome()
    assert restored is not None
    assert restored == outcome
    reopened, _ = stack(store, goal, ScriptedTransport(respond), launcher, shared=None)
    assert GoalOutcome.from_json(
        reopened.control.head.read(_outcome_path(goal.goal_id))
    ) == outcome
    standing = runtime.planner.criteria(goal.goal_id)
    assert {item["criterion_id"] for item in restored.assessment} == {
        item.criterion_id for item in standing.criteria
    }
    assert all(item["verdict"] == "met" for item in restored.assessment)
    assert restored.criteria == standing


def test_run_stops_at_the_generation_bound_without_claiming_completion(store: Store) -> None:
    def respond(request: PlanningRequest, prompt: str):
        return (
            assignment_for(
                request, f"build-{request.generation}", f"worker-{request.generation}", "product.txt"
            ),
        )

    launcher = FakeLauncher()
    runtime, _ = stack(store, simple_goal(), ScriptedTransport(respond), launcher)

    def advance() -> None:
        for run in runtime.supervisor.runs():
            handle = launcher.handles.get(run.run_id)
            if handle is not None:
                handle.exit = ProcessExit(exit_code=1, reaped=True)

    outcome = runtime.run(max_generations=3, wall_clock_seconds=60.0, between_cycles=advance)
    assert outcome.stop_reason == "generation_bound"
    assert not outcome.complete
    assert outcome.generations == 3


def test_run_stops_on_the_wall_clock(store: Store) -> None:
    def respond(request: PlanningRequest, prompt: str):
        return (
            assignment_for(
                request, f"build-{request.generation}", f"worker-{request.generation}", "product.txt"
            ),
        )

    launcher = FakeLauncher()
    runtime, _ = stack(store, simple_goal(), ScriptedTransport(respond), launcher)
    ticks = iter([0.0, 0.0, 1.0, 99.0, 99.0, 99.0, 99.0, 99.0])

    outcome = runtime.run(
        max_generations=50,
        wall_clock_seconds=10.0,
        monotonic=lambda: next(ticks, 99.0),
    )
    assert outcome.stop_reason == "wall_clock"
    assert not outcome.complete


def test_a_budget_that_cannot_be_proven_stops_the_run(store: Store) -> None:
    def respond(request: PlanningRequest, prompt: str):
        if request.generation == 1:
            return (
                assignment_for(
                    request, "build", "worker-build", "product.txt", budget_usd=1.0,
                ),
            )
        return complete_response(request, prompt)

    launcher = FakeLauncher()
    runtime, _ = stack(store, simple_goal(budget_usd=5.0), ScriptedTransport(respond), launcher)

    def advance() -> None:
        plan = runtime.planner.current_plan(runtime.goal.goal_id)
        if plan is None:
            return
        live = [
            item
            for item in runtime.supervisor.runs()
            if item.assignment.assignment_id == "build"
            and item.assignment.generation == plan.generation
        ]
        if not live:
            return
        handle = launcher.handles.get(live[0].run_id)
        if handle is None or handle.exit is not None:
            return
        install_report(runtime, "build", cost_usd=None)
        handle.exit = ProcessExit(exit_code=0, reaped=True)

    outcome = runtime.run(max_generations=5, wall_clock_seconds=60.0, between_cycles=advance)
    assert outcome.stop_reason == "budget_blocked"
    assert not outcome.complete


def test_being_stuck_replans_rather_than_stopping(store: Store) -> None:
    seen: list[int] = []

    def respond(request: PlanningRequest, prompt: str):
        seen.append(request.generation)
        if request.generation >= 3:
            return complete_response(request, prompt)
        return (
            assignment_for(
                request, f"build-{request.generation}", f"worker-{request.generation}", "product.txt"
            ),
        )

    launcher = FakeLauncher()
    runtime, _ = stack(store, simple_goal(), ScriptedTransport(respond), launcher)

    def advance() -> None:
        for run in runtime.supervisor.runs():
            handle = launcher.handles.get(run.run_id)
            if handle is not None and handle.exit is None:
                handle.exit = ProcessExit(exit_code=1, reaped=True)

    outcome = runtime.run(max_generations=6, wall_clock_seconds=60.0, between_cycles=advance)
    # A failed worker is a reason to replan, not a reason to give up.
    assert len(seen) >= 3
    assert outcome.stop_reason in {"complete", "generation_bound"}


def test_bounds_are_required(store: Store) -> None:
    runtime, _ = stack(
        store, simple_goal(), ScriptedTransport(one_assignment_response_runtime), FakeLauncher()
    )
    with pytest.raises(TypeError):
        runtime.run()  # type: ignore[call-arg]


def test_a_second_run_replays_the_recorded_outcome(store: Store) -> None:
    def respond(request: PlanningRequest, prompt: str):
        if request.generation == 1:
            return (assignment_for(request, "build", "worker-build", "product.txt"),)
        return complete_response(request, prompt)

    launcher = FakeLauncher()
    runtime, _ = stack(store, simple_goal(), ScriptedTransport(respond), launcher)

    def advance() -> None:
        settle(runtime, launcher, "build")

    first = runtime.run(max_generations=5, wall_clock_seconds=60.0, between_cycles=advance)
    again = runtime.run(max_generations=5, wall_clock_seconds=60.0)
    assert again == first
