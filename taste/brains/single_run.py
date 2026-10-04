"""The planner of an agent run alone: one assignment, the task as given; then its report.

With every service off (``services="none"``), a goal is the agent and nothing
else, run through the same goal machinery as a supervised one: the same
worker process, journals, terminal broker, settlement and record. Only the
planner, the monitors and the certifier are absent. The planner is this fixed
rule, not a model: its first plan is one assignment whose task is the goal's
task verbatim, for the hosted agent; its closing reply is that agent's own
final words. Its calls go through the ordinary planner transport, so they are
recorded like any plan, at no cost, under the model name ``taste-fixed-plan``.
"""

from __future__ import annotations

import json

from taste.brains import benchmark_reply
from taste.brains.contract import Contract
from taste.brains.records import ArtifactSpec, Assignment, contract_digest

FIXED_PLAN_MODEL = "taste-fixed-plan"
REPORT_PATH = "report.md"
_RATIONALE = "One run of the agent on the task as given, with planning, monitoring and certification off."


def _assessment(payload, evidence):
    return [{"criterion_id": item["criterion_id"], "verdict": "not_met", "evidence": evidence}
            for item in payload.get("standing_criteria", ())]


def _closing_reply(request):
    """The agent's own final words, from the report of its run."""
    for outcome in reversed(request["world"].get("outcomes") or ()):
        report = outcome.get("report")
        if isinstance(report, dict) and str(report.get("summary", "")).strip():
            return report["summary"]
    return "The agent ran once and ended without a report."


def single_run_proposal(prompt: str) -> str:
    payload = json.loads(prompt)
    request = payload["request"]
    shape = dict(payload["required_output_shape"])
    shape.update(rationale=_RATIONALE, complete=False, completion_reason="")
    if benchmark_reply.is_closing(request["operation_id"]):
        shape.update(assignments=[],
                     assessment=_assessment(payload, "No one judged the agent's work in this run."),
                     metadata={"final_reply": _closing_reply(request)})
        return json.dumps(shape, sort_keys=True)
    if request["generation"] != 1 or request.get("parent_plan") is not None:
        raise ValueError("an agent run alone has one generation")
    exemplar = payload["required_output_shape"]["assignments"][0]
    goal = request["goal"]
    contract = Contract(
        identity="agent", task=goal["task"], inputs=(), outputs=(REPORT_PATH,),
        success_criteria=tuple(goal.get("success_criteria") or ("the task is done",)),
        budget_usd=exemplar["contract"]["budget_usd"], max_turns=exemplar["contract"]["max_turns"],
    )
    assignment = Assignment(
        assignment_id="agent-run", generation=1, attempt=0, contract=contract,
        contract_digest=contract_digest(contract), base_state_id=request["world"]["integration_state_id"],
        outputs=(ArtifactSpec("report", REPORT_PATH, kind="report",
                              description="the agent's report, written by the harness"),),
        model=exemplar["model"], resources=exemplar["resources"],
    )
    shape.update(assignments=[assignment.to_dict()],
                 assessment=_assessment(payload, "The agent has not run yet."),
                 metadata={"final_reply": ""})
    return json.dumps(shape, sort_keys=True)
