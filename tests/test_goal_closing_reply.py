"""A benchmark goal that stops short still gives the developer its reply."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from taste.brains import benchmark_reply
from taste.brains.central_host import compose_central_runtime
from taste.brains.central_planner import PlanningRequest
from tests.test_brains_central_runtime import (
    FakeLauncher,
    ScriptedTransport,
    WorkingHandle,
    assignment_for,
    proposal,
    simple_goal,
)

REPLY = "I started the product file. It is not finished and nothing was verified."


def replying_goal(*, reserve=10, budget=None):
    metadata = {benchmark_reply.KEY: benchmark_reply.SCHEMA}
    if reserve is not None:
        metadata[benchmark_reply.RESERVE_KEY] = reserve
    return replace(simple_goal(budget_usd=budget), metadata=metadata)


def with_reply(prompt, reply, *assignments, complete=False):
    raw = json.loads(proposal(prompt, *assignments, complete=complete))
    raw["metadata"] = {"final_reply": reply}
    return json.dumps(raw, sort_keys=True)


def responder(closing_reply=REPLY, *, budget=None, seen=None):
    def respond(request: PlanningRequest, prompt: str):
        if seen is not None:
            seen.append(request)
        if benchmark_reply.is_closing(request.operation_id):
            if isinstance(closing_reply, Exception):
                raise closing_reply
            return with_reply(prompt, closing_reply)
        return with_reply(prompt, "", assignment_for(
            request, f"build-{request.generation}", f"worker-build-{request.generation}",
            "product.txt", budget_usd=budget))
    return respond


@pytest.fixture
def make_host(tmp_path):
    hosts = []

    def make(goal, respond, *, launcher=None, call_ceiling_usd=0.0):
        root = tmp_path / f"repo-{len(hosts)}"
        root.mkdir()
        host = compose_central_runtime(
            root, "closing-test", goal,
            transport=ScriptedTransport(respond, call_ceiling_usd=call_ceiling_usd),
            launcher=launcher or FakeLauncher(), supervisor_termination_grace=0.05,
        )
        hosts.append(host)
        return host

    try:
        yield make
    finally:
        for host in hosts:
            for run in host.supervisor.runs():
                host.supervisor.stop(run.run_id, "test_cleanup")
            host.close()


def final_reply(host):
    plan = host.planner.current_plan(host.goal.goal_id)
    return None if plan is None else plan.metadata["proposal"].get("final_reply")


def test_run_out_of_time_drains_workers_then_asks_once_for_the_reply(make_host):
    seen: list[PlanningRequest] = []
    launcher = FakeLauncher()
    host = make_host(replying_goal(reserve=10), responder(seen=seen), launcher=launcher)
    # 15 s of trial, 10 s of it reserved: the worker gets 5 s and never finishes.
    outcome = host.run(max_generations=5, wall_clock_seconds=15)

    assert host.runtime.closing_failure is None, host.runtime.closing_failure
    assert outcome.stop_reason == "wall_clock" and not outcome.complete
    assert final_reply(host) == REPLY
    plan = host.planner.current_plan(host.goal.goal_id)
    assert not plan.complete and plan.assignments == ()
    # Every worker was stopped before the coordinator was asked, so the reply
    # describes the final state rather than work still changing under it.
    assert launcher.handles
    assert all(handle.termination_calls >= 1 for handle in launcher.handles.values())
    closing = [item for item in seen if benchmark_reply.is_closing(item.operation_id)]
    assert len(closing) == 1 and seen[-1] is closing[0]
    observed = closing[0].world.outcomes
    assert observed and all(item.run.terminal for item in observed)
    # The worker's own allowance ends at the same instant as the goal's working
    # time; either can be the one that is recorded as stopping it.
    assert observed[-1].run.terminal_reason in {"wall_clock", "wall_timeout"}
    assert outcome.generations == plan.generation == closing[0].generation
    # The recorded ending is immutable and a later driver returns it as is.
    calls = len(seen)
    assert host.run(max_generations=5, wall_clock_seconds=15) == outcome
    assert len(seen) == calls


def test_no_plan_is_started_in_the_last_seconds_of_working_time(make_host):
    # Measured on a real trial: a revision was requested ten seconds before
    # working time ended, was cut off, and barred the closing reply with it.
    seen: list[PlanningRequest] = []
    launcher = FakeLauncher()

    def launch(spec):
        launcher.launch_calls.append(spec.run_id)
        # The worker ends without a report two seconds in: a reason to replan.
        return launcher.handles.setdefault(spec.run_id, WorkingHandle(7000, 2.0, lambda: None))

    launcher.launch = launch
    goal = replying_goal(reserve=10)
    goal = replace(goal, metadata={**goal.metadata, benchmark_reply.PLAN_KEY: 4})
    host = make_host(goal, responder(seen=seen), launcher=launcher)
    # 15 s of trial, 10 reserved: 5 s of work. At about 2 s the worker is gone
    # and under 4 s remain, so no revision is asked for.
    outcome = host.run(max_generations=5, wall_clock_seconds=15)

    assert host.runtime.closing_failure is None, host.runtime.closing_failure
    assert outcome.stop_reason == "wall_clock" and "too little working time" in outcome.detail
    kinds = [item.operation_id.split(".", 1)[0] for item in seen]
    assert kinds == ["runtime-initial", "runtime-closing"], kinds
    assert final_reply(host) == REPLY


def test_generation_bound_also_ends_with_a_reply(make_host):
    host = make_host(replying_goal(reserve=10), responder())
    # Generation 1 is planned; its worker never reports, so the only way on is
    # a replan, and the bound of one generation is reached by the trigger of
    # that worker's stop at the end. Force it by a tiny working window.
    outcome = host.run(max_generations=1, wall_clock_seconds=15)
    assert outcome.stop_reason in {"wall_clock", "generation_bound"}
    assert final_reply(host) == REPLY


def test_goal_without_a_reserve_keeps_the_old_ending(make_host):
    seen: list[PlanningRequest] = []
    host = make_host(replying_goal(reserve=None), responder(seen=seen))
    outcome = host.run(max_generations=5, wall_clock_seconds=3)
    assert outcome.stop_reason == "wall_clock"
    assert final_reply(host) == "" and seen
    assert not any(benchmark_reply.is_closing(item.operation_id) for item in seen)


def test_completed_goal_asks_for_no_closing_reply(make_host):
    seen: list[PlanningRequest] = []

    def respond(request, prompt):
        seen.append(request)
        return with_reply(prompt, "Nothing needed doing.", complete=True)

    host = make_host(replying_goal(reserve=10), respond)
    outcome = host.run(max_generations=5, wall_clock_seconds=30)
    assert outcome.complete and final_reply(host) == "Nothing needed doing." and len(seen) == 1


@pytest.mark.parametrize("fault", ["invalid", "transport"])
def test_failed_closing_reply_never_changes_or_delays_the_recorded_stop(make_host, fault):
    from taste.brains.central_planner import PlannerTransportError

    reply = " " if fault == "invalid" else PlannerTransportError("provider unavailable")
    host = make_host(replying_goal(reserve=10), responder(reply))
    outcome = host.run(max_generations=5, wall_clock_seconds=15)
    assert outcome.stop_reason == "wall_clock" and not outcome.complete
    assert final_reply(host) == ""  # Still the last promoted, working plan.
    assert host.runtime.closing_failure
    assert host.outcome() == outcome


def test_lost_cost_of_a_capped_worker_is_charged_at_its_ceiling(make_host):
    """A lost exact cost used to make the whole budget unprovable and block every later call."""
    host = make_host(replying_goal(reserve=10, budget=10.0), responder(budget=1.5), call_ceiling_usd=1.0)
    host.run(max_generations=5, wall_clock_seconds=15)
    run = next(item for item in host.supervisor.runs() if item.pid is not None)
    assert run.terminal and run.report_id is None

    plain, capped = run.assignment, replace(run.assignment, resources={
        **run.assignment.resources, "azure_openai": {"schema": "caps enforced before each dispatch"}})
    reservations, unknown = [], []
    host.runtime._charge_lost_cost(replace(run, assignment=capped), reservations, unknown)
    # 1.5 for the worker and 1.5 for its monitor: it cannot have spent more.
    assert reservations == [3.0] and unknown == []
    host.runtime._charge_lost_cost(replace(run, assignment=plain), reservations, unknown)
    assert reservations == [3.0] and unknown == [run.run_id]


def test_lost_cost_of_a_worker_without_proven_caps_stays_unknown(make_host):
    goal = replying_goal(reserve=10, budget=10.0)
    host = make_host(goal, responder(budget=1.5), call_ceiling_usd=1.0)
    outcome = host.run(max_generations=5, wall_clock_seconds=15)
    # No pre-dispatch bound exists for this harness, so nothing is assumed:
    # the cost is unknown, and the closing call that would spend more is refused.
    assert outcome.budget.unknown_run_ids and not outcome.budget.enforceable
    assert final_reply(host) == "" and "BudgetBlocked" in host.runtime.closing_failure
