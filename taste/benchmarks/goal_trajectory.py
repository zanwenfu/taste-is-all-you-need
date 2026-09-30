"""Project a settled Azure goal into ATIF without dispatching a model call.

The root contains only the original task and the accepted coordinator reply.
Internal planning, every worker (including abandoned runs), and monitor calls
are embedded trajectories. No global ordering between parallel workers is
invented. Missing evidence remains a gap and prevents grading admission.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict

from taste.benchmarks.worker_trajectory import worker_trajectory
from taste.brains import benchmark_reply
from taste.brains.azure_worker_entrypoint import run_directory
from taste.brains.azure_worker_policy import AzureWorkerPolicy
from taste.brains.central_planner import (
    PLANNER_SYSTEM,
    PlanningRequest,
    _canonical,
    _digest,
    _load_json,
    _operation_root,
)
from taste.brains.goal_entrypoint import GoalInputError
from taste.brains.responses_session import ResponsesSession

MAX_TRAJECTORY_BYTES = 128 * 1024 * 1024


def encode_trajectory(value):
    result = bytearray()
    for part in json.JSONEncoder(sort_keys=True, ensure_ascii=True, allow_nan=False,
                                 separators=(",", ":")).iterencode(value):
        raw = part.encode()
        if len(result) + len(raw) + 1 > MAX_TRAJECTORY_BYTES:
            raise GoalInputError("goal trajectory exceeds 128 MiB; no truncated report is admissible")
        result.extend(raw)
    result.extend(b"\n")
    return bytes(result)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _metrics(usage, cost):
    return {"prompt_tokens": usage["input_tokens"] + usage["cache_read_tokens"] + usage["cache_write_tokens"],
            "completion_tokens": usage["output_tokens"], "cached_tokens": usage["cache_read_tokens"],
            "cost_usd": cost, "extra": {"reasoning_tokens": usage["reasoning_tokens"],
                                       "cache_write_tokens": usage["cache_write_tokens"]}}


def _step(steps, source, message, **fields):
    extra = fields.pop("extra", {})
    if source == "agent":
        extra["is_sidechain"] = True
    steps.append({"step_id": len(steps) + 1, "source": source, "message": message,
                  "extra": extra, **fields})
    return steps[-1]


def _trajectory(identifier, name, steps, **extra):
    return {"schema_version": "ATIF-v1.7", "session_id": identifier, "trajectory_id": identifier,
            "agent": {"name": name, "version": "1"}, "steps": steps, "extra": extra}


def _planner(host, plan, gaps):
    steps, final, costs = [], [], []
    audits = host.planner.planner_attempts(host.goal.goal_id)
    receipts = {item.request_id: item for item in host.planner._paired_transport_evidences()}
    _step(steps, "system", PLANNER_SYSTEM)
    for audit in audits:
        receipt = receipts.get(audit.attempt_id)
        root = _operation_root(host.goal.goal_id, audit.operation_id)
        request = PlanningRequest.from_json(host.control.head.read(root + "/request.json"))
        prompt = host.planner._prompt(request)
        extra = {"attempt_id": audit.attempt_id, "request_id": audit.request_id,
                 "status": audit.status, "category": audit.category}
        # Prompts are reproducible from the immutable request and pinned source.
        # Never label a reconstruction exact unless its original receipt agrees.
        if receipt is not None:
            if (receipt.binding["prompt_sha256"] != _sha(prompt.encode())
                    or receipt.binding["system_sha256"] != _sha(PLANNER_SYSTEM.encode())):
                raise GoalInputError("planner prompt differs from its original transport receipt")
            response = receipt.response
        else:
            raw = host.control.head.record(f"{root}/attempts/{audit.attempt:06d}/outcome.json")
            response = None if raw is None else raw["response"]
            if audit.telemetry.source != "not_dispatched_zero":
                gaps.append("planner_receipt_missing:" + audit.attempt_id)
            extra["transport_receipt_missing"] = True
        _step(steps, "user", prompt, extra={**extra, "prompt_sha256": _sha(prompt.encode())})
        telemetry = audit.telemetry
        if telemetry.cost_known:
            costs.append(telemetry.billed_usd)
        else:
            gaps.append("planner_cost_unknown:" + audit.attempt_id)
        if audit.status in {"pending", "orphaned"}:
            gaps.append("planner_attempt_unsettled:" + audit.attempt_id)
        if response is None:
            continue
        if _digest(response) != audit.response_digest:
            raise GoalInputError("planner response differs from its audited control outcome")
        value = _step(steps, "agent", response, timestamp=audit.at,
                      extra={**extra, "telemetry": telemetry.to_dict()}, llm_call_count=1)
        if telemetry.model is not None:
            value["model_name"] = telemetry.model
        if telemetry.usage is not None:
            value["metrics"] = _metrics(telemetry.usage.to_dict(), telemetry.billed_usd)
        if (plan is not None and plan.complete and audit.status == "accepted"
                and audit.request_id == plan.metadata["request_id"]):
            proposal = _load_json(response, "final planner response")
            if (_digest(_canonical(proposal)) != plan.metadata["proposal_digest"]
                    or proposal["metadata"] != plan.to_dict()["metadata"]["proposal"]):
                raise GoalInputError("final reply is not bound to its accepted model proposal")
            final.append((benchmark_reply.validate(proposal["metadata"], complete=True), audit))
    if plan is not None and plan.complete and len(final) != 1:
        raise GoalInputError("complete benchmark goal has no unique accepted model reply")
    return _trajectory(host.goal.goal_id + ".planner", "taste-coordinator", steps), final, math.fsum(costs)


def _worker_calls(trace, rows, calls, gaps, run_id):
    completions = {r["event"]["id"]: r["event"]["completion"] for r in rows
                   if r["event"]["kind"] == "responses_completion"}
    requested = {r["event"]["id"] for r in rows if r["event"]["kind"] == "responses_request"}
    by_id = {row["request_id"]: row for row in calls}
    if requested != set(by_id):
        gaps.append("worker_request_audit_mismatch:" + run_id)
    for step in trace["steps"]:
        identifier = step.get("extra", {}).get("request_id")
        if identifier is None:
            continue
        call = by_id.get(identifier)
        reply = None if call is None else call["completion"]
        if reply is None or any(completions[identifier][key] != value for key, value in reply.items()):
            raise GoalInputError("worker conversation differs from its provider receipt")
        step["metrics"] = _metrics(reply["usage"], call["cost_usd"])
    for call in calls:
        reply = call["completion"]
        if reply is None or call["request_id"] in completions:
            continue
        # A reply can be durable before memory/audit publication. Preserve its
        # actual text and requested tools without claiming they were executed.
        gaps.append("worker_reply_unpublished:" + run_id + ":" + call["request_id"])
        step = _step(trace["steps"], "agent", "\n".join(reply["text_blocks"]),
                     model_name=reply["model"], llm_call_count=1,
                     metrics=_metrics(reply["usage"], call["cost_usd"]),
                     extra={"request_id": call["request_id"], "memory_published": False})
        for tool in reply["tool_calls"]:
            step.setdefault("tool_calls", []).append({"tool_call_id": "call_" + _sha(
                (call["request_id"] + "\0" + tool["id"]).encode()),
                "function_name": tool["name"], "arguments": tool["arguments"]})


def _workers(host, runs, gaps):
    traces, costs, retained_bytes = [], [], 0
    for run in runs:
        policy = AzureWorkerPolicy.from_assignment(run.assignment)
        azure = policy.azure_config({"AZURE_OPENAI_BASE_URL": policy.worker.endpoint,
                                     "AZURE_OPENAI_API_KEY": "settlement-no-dispatch"})
        directory = run_directory(host.store, run.run_id)
        # The supervisor writes a deadline in its durable spawn intent BEFORE
        # launching. A prepared run stopped before that intent owes no model
        # journal. Preserve that attempt explicitly without inventing activity
        # or misclassifying a legitimate bounded stop as missing evidence.
        if run.pid is None and run.deadline_at is None and run.launch_token is None:
            if not run.terminal or run.recovery_status != "complete" or os.path.lexists(directory):
                raise GoalInputError("unlaunched worker has inconsistent settlement evidence")
            traces.append(_trajectory(run.run_id, "taste-azure-worker", [
                {"step_id": 1, "source": "system",
                 "message": "Worker settled before any durable spawn intent; no model or tool call was launched."}],
                run_id=run.run_id, assignment_id=run.assignment.assignment_id,
                phase=run.phase, not_launched=True, call_statuses=[]))
            continue
        for role, binding in (("worker", policy.worker), ("monitor", policy.monitor)):
            path = directory / role
            if not os.path.lexists(path):
                gaps.append("missing_" + role + "_journal:" + run.run_id)
                continue
            session = ResponsesSession.open(path, binding, azure)
            try:
                calls = session.call_evidence()
                rows = session.conversation_audit() if role == "worker" else []
            finally:
                session.close()
            for call in calls:
                if call["cost_usd"] is not None:
                    costs.append(call["cost_usd"])
                elif call["status"] != "not_dispatched":
                    gaps.append(role + "_cost_unknown:" + run.run_id + ":" + call["request_id"])
            if role == "worker":
                trace = (worker_trajectory(rows, run_id=run.run_id) if rows else
                         _trajectory(run.run_id, "taste-azure-worker", [
                             {"step_id": 1, "source": "system", "message": "No conversation audit was published."}]))
                if not rows or trace["extra"]["incomplete_worker_trace"]:
                    gaps.append("worker_conversation_incomplete:" + run.run_id)
                _worker_calls(trace, rows, calls, gaps, run.run_id)
            else:
                steps = []
                _step(steps, "system", "Internal monitor evidence; call timestamps were not recorded.")
                for call in calls:
                    extra = {"request_id": call["request_id"], "status": call["status"],
                             "error_type": call["error_type"], "timestamp_unavailable": True}
                    _step(steps, "system", call["request"]["system"], extra=extra)
                    _step(steps, "user", json.dumps(call["request"]["messages"], ensure_ascii=True), extra=extra)
                    reply = call["completion"]
                    if reply is not None:
                        if reply["tool_calls"]:
                            raise GoalInputError("tool-free monitor receipt contains tool calls")
                        _step(steps, "agent", "\n".join(reply["text_blocks"]), model_name=reply["model"],
                              llm_call_count=1, metrics=_metrics(reply["usage"], call["cost_usd"]), extra=extra)
                trace = _trajectory(binding.run_id, "taste-monitor", steps)
            trace["extra"].update({"run_id": run.run_id, "assignment_id": run.assignment.assignment_id,
                                   "phase": run.phase, "call_statuses": [
                                       {k: c[k] for k in ("request_id", "status", "error_type")} for c in calls]})
            # Bound each component before retaining it with all the others.
            retained_bytes += len(encode_trajectory(trace))
            if retained_bytes > MAX_TRAJECTORY_BYTES:
                raise GoalInputError("combined worker evidence exceeds the trajectory byte limit")
            traces.append(trace)
    return traces, math.fsum(costs)


def goal_trajectory(host, config, outcome):
    """Called under the settled host's ownership; no worker may still be live."""
    if (host.goal != config.goal or host.runtime.outcome() != outcome
            or not benchmark_reply.required(config.goal.metadata)):
        raise GoalInputError("goal trajectory requires its exact settled benchmark admission")
    runs = host.supervisor.runs()
    host.runtime._validate_supervisor_scope(runs)
    if any(not run.terminal or run.recovery_status != "complete" for run in runs):
        raise GoalInputError("cannot export a goal while worker settlement is incomplete")
    plan = host.planner.current_plan(config.goal.goal_id)
    gaps = []
    planner, final, planner_cost = _planner(host, plan, gaps)
    workers, worker_cost = _workers(host, runs, gaps)
    for label, actual, expected in (("planner", planner_cost, outcome.budget.planner_spent_usd),
                                    ("worker", worker_cost, outcome.budget.worker_spent_usd)):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
            gaps.append(label + "_cost_mismatch")
    if not outcome.budget.enforceable or outcome.budget.reserved_usd != 0:
        gaps.append("goal_accounting_unsettled")
    result = _trajectory(config.goal.goal_id, "taste", [
        {"step_id": 1, "source": "user", "message": config.goal.task}], trace_scope="goal",
        input_sha256=_sha(config.to_bytes()), python_source_sha256=config.python_source_sha256,
        control_state_id=host.control.head.id, outcome=outcome.to_dict(),
        runs=[{"run_id": r.run_id, "assignment_id": r.assignment.assignment_id,
               "phase": r.phase, "terminal_reason": r.terminal_reason,
               "recovery_status": r.recovery_status} for r in runs],
        evidence_complete=not gaps, complete_attempt=not gaps, gaps=sorted(set(gaps)),
        final_reply_present=bool(final), goal_complete=outcome.complete)
    if final:
        reply, audit = final[0]
        result["steps"].append({"step_id": 2, "source": "agent", "message": reply,
            "timestamp": audit.at, "llm_call_count": 0,
            "extra": {"accepted_planner_attempt_id": audit.attempt_id, "is_sidechain": False}})
    result["subagent_trajectories"] = [planner, *workers]
    metrics = [step["metrics"] for sub in result["subagent_trajectories"] for step in sub["steps"]
               if step.get("metrics") is not None]
    result["final_metrics"] = {"extra": {"known_cost_usd": outcome.budget.known_spent_usd,
                                           "budget": asdict(outcome.budget)}}
    if not gaps:
        result["final_metrics"].update(total_cost_usd=outcome.budget.known_spent_usd,
            **{"total_" + name: sum(row[name] for row in metrics)
               for name in ("prompt_tokens", "completion_tokens", "cached_tokens")})
    encode_trajectory(result)
    return result
