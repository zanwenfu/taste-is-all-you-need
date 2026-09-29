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
