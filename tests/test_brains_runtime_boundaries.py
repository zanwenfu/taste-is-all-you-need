"""Whole planner/supervisor/runtime boundary regressions, without a provider."""
from dataclasses import replace

import pytest

from taste.brains.central_planner import PlannerIdentityConflict
from taste.brains.supervisor import ProcessExit, SupervisorError
from taste.memstore import Store
from tests.test_brains_central_runtime import (
    FakeLauncher,
    ScriptedTransport,
    assignment_for,
    complete_response,
    settle,
    simple_goal,
    stack,
)


@pytest.fixture
def store(tmp_path):
    opened = Store.open(tmp_path / "repo", "bounds")
    yield opened
    opened.close()


def one(request, prompt):
    return (assignment_for(request, "build", "worker-build", "product.txt"),)


def test_one_generation_executes_and_collects_without_buying_another_plan(store):
    launcher, transport = FakeLauncher(), ScriptedTransport(one)
    runtime, _ = stack(store, simple_goal(), transport, launcher)
    outcome = runtime.run(max_generations=1, wall_clock_seconds=30,
                          between_cycles=lambda: settle(runtime, launcher, "build"))
    assert len(launcher.launch_calls) == 1
    assert len(transport.calls) == 1
    assert outcome.stop_reason == "generation_bound" and not outcome.complete
    assert outcome.delivered_assignment_ids == ("build",)
    assert runtime.integration.head.read("product.txt") == "certified product\n"
    assert all(run.terminal for run in runtime.supervisor.runs())


def test_null_assignment_timeout_survives_plan_persistence_and_reopen(store):
    def respond(request, prompt):
        return tuple(replace(a, resources={"wall_timeout_seconds": None}) for a in one(request, prompt))

    launcher, transport = FakeLauncher(), ScriptedTransport(respond)
    runtime, shared = stack(store, simple_goal(), transport, launcher)
    runtime.cycle()
    reopened, _ = stack(store, simple_goal(), transport, launcher, shared=shared)
    reopened.cycle()
    (run,) = reopened.supervisor.runs()
    assert run.wall_timeout_seconds == reopened.default_wall_timeout_seconds
    reopened.supervisor.stop(run.run_id)


@pytest.mark.parametrize("entrypoint", ["cycle", "outcome", "run"])
def test_reused_goal_id_with_changed_objective_is_rejected_before_effect(store, entrypoint):
    launcher, transport = FakeLauncher(), ScriptedTransport(one)
    runtime, shared = stack(store, simple_goal(), transport, launcher)
    runtime.cycle()
    changed = replace(simple_goal(), task="a different requested product")
    other, _ = stack(store, changed, transport, launcher, shared=shared)
    before = runtime.control.head.id
    kwargs = {"max_generations": 1, "wall_clock_seconds": 30} if entrypoint == "run" else {}
    with pytest.raises(PlannerIdentityConflict):
        getattr(other, entrypoint)(**kwargs)
    assert runtime.control.head.id == before
    assert not launcher.launch_calls and len(transport.calls) == 1


def test_wall_stop_drains_workers_and_accounts_after_stop(store):
    launcher = FakeLauncher()
    runtime, _ = stack(store, simple_goal(), ScriptedTransport(one), launcher)
    runtime.cycle()
    runtime.cycle()
    ticks = iter([0, 10])
    outcome = runtime.run(max_generations=9, wall_clock_seconds=1,
                          monotonic=lambda: next(ticks, 10))
    (run,) = runtime.supervisor.runs()
    assert run.terminal and run.recovery_status == "complete"
    assert launcher.handles[run.run_id].termination_calls == 1
    assert not store.worktree_path_for(run.assignment.worker).exists()
    assert outcome.budget.reserved_usd == 0
    assert outcome.budget.unknown_run_ids == (run.run_id,)
    assert outcome.generations == 1 and outcome.cycles >= 2


def test_shutdown_failure_retries_finalization_without_resuming_work(store):
    class FailsOnce(FakeLauncher):
        fail = True

        def cancel(self, spec, grace_seconds):
            if self.fail:
                self.fail = False
                raise SupervisorError("injected drain failure")
            return super().cancel(spec, grace_seconds)

    launcher, transport = FailsOnce(), ScriptedTransport(one)
    runtime, shared = stack(store, simple_goal(), transport, launcher)
    runtime.cycle()
    runtime.cycle()
    ticks = iter([0, 10])
    with pytest.raises(SupervisorError, match="drain failure"):
        runtime.run(max_generations=2, wall_clock_seconds=1, monotonic=lambda: next(ticks, 10))
    assert runtime.outcome() is None
    reopened, _ = stack(store, simple_goal(), transport, launcher, shared=shared)
    outcome = reopened.run(max_generations=2, wall_clock_seconds=1)
    assert outcome.stop_reason == "wall_clock"
    assert all(run.terminal for run in reopened.supervisor.runs())
    assert len(transport.calls) == len(launcher.launch_calls) == 1


def test_outcome_includes_deliveries_and_cycles_before_restart(store):
    def respond(request, prompt):
        if request.generation < 3:
            name = f"build-{request.generation}"
            return (assignment_for(request, name, f"worker-{name}", f"{name}.txt"),)
        return complete_response(request, prompt)

    launcher, transport = FakeLauncher(), ScriptedTransport(respond)
    runtime, shared = stack(store, simple_goal(), transport, launcher)
    runtime.cycle()
    runtime.cycle()
    settle(runtime, launcher, "build-1")
    runtime.cycle()
    reopened, _ = stack(store, simple_goal(), transport, launcher, shared=shared)
    outcome = reopened.run(max_generations=3, wall_clock_seconds=30,
                           between_cycles=lambda: settle(reopened, launcher, "build-2"))
    assert outcome.complete
    assert outcome.delivered_assignment_ids == ("build-1", "build-2")
    assert outcome.cycles == len(reopened._cycle_records())
    assert outcome.budget.worker_spent_usd == 0.5


def test_initial_budget_refusal_has_durable_outcome_without_model_call(store):
    transport = ScriptedTransport(one, call_ceiling_usd=10)
    runtime, _ = stack(store, simple_goal(budget_usd=1), transport, FakeLauncher())
    outcome = runtime.run(max_generations=1, wall_clock_seconds=30)
    assert outcome.stop_reason == "budget_blocked"
    assert not outcome.complete and outcome.generations == 0
    assert runtime.outcome() == outcome
    assert transport.calls == []


def test_exception_during_run_drains_live_workers(store):
    launcher = FakeLauncher()
    runtime, _ = stack(store, simple_goal(), ScriptedTransport(one), launcher)
    runtime.cycle()
    runtime.cycle()

    def interrupted():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        runtime.run(max_generations=2, wall_clock_seconds=30, between_cycles=interrupted)
    (run,) = runtime.supervisor.runs()
    assert run.terminal and launcher.handles[run.run_id].exit == ProcessExit(signal=9, reaped=True)
