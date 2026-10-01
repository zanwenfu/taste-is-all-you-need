"""The real planner admits explicit replies without changing historical goals."""

import json
from dataclasses import replace

import pytest

from taste.brains import benchmark_reply
from taste.brains.central_planner import InvalidPlannerOutput
from tests.test_brains_central_planner import (
    assignment_for,
    goal,
    planner,
    proposal,
    request_from_prompt,
    store,
)

__all__ = ["goal", "store"]


@pytest.mark.parametrize("complete,reply", [
    (True, None), (True, ""), (True, " \n"), (True, 42), (True, "bad\x00reply"),
    (True, "☃" * (benchmark_reply.MAX_REPLY_BYTES // 3 + 1)), (False, "premature claim"),
], ids=["missing", "empty", "whitespace", "number", "nul", "oversize", "premature"])
def test_invalid_reply_is_retained_and_rejected_by_planner(store, goal, complete, reply):
    goal = replace(goal, metadata={benchmark_reply.KEY: benchmark_reply.SCHEMA})

    def response(_id, _system, prompt):
        payload = json.loads(prompt)
        assert payload["required_output_shape"]["metadata"] == {"final_reply": ""}
        assert "benchmark_final_reply" in payload["rules"]
        assignments = () if complete else (assignment_for(request_from_prompt(prompt)),)
        return proposal(prompt, *assignments, complete=complete,
                        completion_reason="internal" if complete else "",
                        changes={"metadata": {} if reply is None else {"final_reply": reply}})

    owner, transport = planner(store, response)
    try:
        with pytest.raises(InvalidPlannerOutput):
            owner.plan(goal)
        assert owner.current_plan(goal.goal_id) is None
        assert owner.planner_attempts(goal.goal_id)[0].status == "rejected"
        assert len(transport.calls) == 1
    finally:
        owner.control.close()


def test_exact_model_reply_survives_promotion_and_reopen(store, goal):
    goal = replace(goal, metadata={benchmark_reply.KEY: benchmark_reply.SCHEMA})
    reply = '  Updated "parser.py".\r\nI could not run the integration test. ☃\n'

    def response(_id, _system, prompt):
        return proposal(prompt, complete=True, completion_reason="internal criterion witness",
                        changes={"metadata": {"final_reply": reply}})

    owner, transport = planner(store, response)
    try:
        plan = owner.plan(goal)
        assert plan.metadata["proposal"]["final_reply"] == reply
        assert plan.completion_reason != reply
    finally:
        owner.control.close()
    owner, _ = planner(store, lambda *_: pytest.fail("reopen cannot call the model"))
    try:
        assert owner.current_plan(goal.goal_id) == plan
        assert owner.plan(goal) == plan
        assert len(transport.calls) == 1
    finally:
        owner.control.close()


def test_invalid_contract_fails_at_goal_admission(goal):
    with pytest.raises(ValueError, match="reply contract"):
        replace(goal, metadata={benchmark_reply.KEY: None})


CLOSING = benchmark_reply.CLOSING_OPERATION_PREFIX + "0" * 64


def _replying(goal, **extra):
    return replace(goal, metadata={benchmark_reply.KEY: benchmark_reply.SCHEMA, **extra})


def test_closing_proposal_carries_the_reply_of_an_unfinished_goal(store, goal):
    goal = _replying(goal)
    reply = "I changed parser.py. The integration test was not run."
    prompts = []

    def response(_id, _system, prompt):
        prompts.append(json.loads(prompt))
        if len(prompts) == 1:
            return proposal(prompt, assignment_for(request_from_prompt(prompt)),
                            changes={"metadata": {"final_reply": ""}})
        return proposal(prompt, changes={"metadata": {"final_reply": reply}})

    owner, transport = planner(store, response)
    try:
        first = owner.plan(goal)
        closing = owner.revise(goal, operation_id=CLOSING)
        # The stop stands: nothing is assigned and completion is not claimed.
        assert not closing.complete and closing.assignments == ()
        assert closing.generation == first.generation + 1
        assert closing.metadata["proposal"]["final_reply"] == reply
        assert owner.current_plan(goal.goal_id) == closing
        rules = prompts[1]["rules"]
        assert prompts[1]["required_output_shape"]["assignments"] == []
        assert rules["non_complete_requires_assignments"] is False and "closing_proposal" in rules
        assert "must not be empty" in rules["benchmark_final_reply"]
        # An ordinary request is unchanged by the closing rules.
        assert "closing_proposal" not in prompts[0]["rules"]
        assert prompts[0]["rules"]["non_complete_requires_assignments"] is True
        assert len(transport.calls) == 2
    finally:
        owner.control.close()


@pytest.mark.parametrize("fault", ["no_reply", "assigns_work"])
def test_closing_proposal_must_reply_and_cannot_assign_work(store, goal, fault):
    goal = _replying(goal)

    def response(_id, _system, prompt):
        request = request_from_prompt(prompt)
        if not benchmark_reply.is_closing(request.operation_id):
            return proposal(prompt, assignment_for(request), changes={"metadata": {"final_reply": ""}})
        if fault == "no_reply":
            return proposal(prompt, changes={"metadata": {"final_reply": " "}})
        return proposal(prompt, assignment_for(request, assignment_id="more", worker="worker-more"),
                        changes={"metadata": {"final_reply": "still working"}})

    owner, _ = planner(store, response)
    try:
        first = owner.plan(goal)
        with pytest.raises(InvalidPlannerOutput):
            owner.revise(goal, operation_id=CLOSING)
        assert owner.current_plan(goal.goal_id) == first
    finally:
        owner.control.close()


def test_closing_may_declare_completion_only_with_every_criterion_met(store, goal):
    goal = _replying(goal)

    def response(_id, _system, prompt):
        request = request_from_prompt(prompt)
        if not benchmark_reply.is_closing(request.operation_id):
            return proposal(prompt, assignment_for(request), changes={"metadata": {"final_reply": ""}})
        return proposal(prompt, complete=True, completion_reason="all criteria were already met",
                        changes={"metadata": {"final_reply": "Everything asked for is done."}})

    owner, _ = planner(store, response)
    try:
        owner.plan(goal)
        assert owner.revise(goal, operation_id=CLOSING).complete
    finally:
        owner.control.close()


def test_a_non_closing_operation_cannot_borrow_the_closing_shape(store, goal):
    goal = _replying(goal)

    def response(_id, _system, prompt):
        return proposal(prompt, changes={"metadata": {"final_reply": "nothing was assigned"}})

    owner, _ = planner(store, response)
    try:
        with pytest.raises(InvalidPlannerOutput):
            owner.plan(goal)
    finally:
        owner.control.close()


@pytest.mark.parametrize("value", [9, 3601, True, "120", None])
def test_reply_reserve_is_bounded_and_needs_the_reply_contract(goal, value):
    with pytest.raises(ValueError, match="reserve"):
        _replying(goal, **{benchmark_reply.RESERVE_KEY: value})
    with pytest.raises(ValueError, match="reserve"):
        replace(goal, metadata={benchmark_reply.RESERVE_KEY: 120})


def test_reply_reserve_defaults_to_none(goal):
    assert benchmark_reply.closing_reserve(_replying(goal).metadata) == 0.0
    assert benchmark_reply.closing_reserve(goal.metadata) == 0.0
    assert benchmark_reply.closing_reserve(
        _replying(goal, **{benchmark_reply.RESERVE_KEY: 150}).metadata) == 150.0
