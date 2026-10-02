"""ATIF evidence for an internal Azure worker, including discarded branches.

This is a subagent record, not a benchmark answer. Every agent step is marked
as a sidechain so a worker claim cannot be mistaken for the coordinator's final
reply. A future trial exporter must include every worker and the coordinator,
and publish the actual final reply separately. No Harbor/OpenAI dependency is
needed to project the private audit; provider reasoning ciphertext is omitted.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy

from taste.brains.responses_audit import ROOT, ResponsesAuditError, event_id


def worker_trajectory(rows, *, run_id):
    """Project an audited worker snapshot without inventing missing observations.

    Call IDs are scoped by request, since a memory rollback can permit a model
    to reuse its earlier call ID. Results stay byte-for-byte as the tool exposed
    them, including binary encodings, paging and explicit truncation metadata.
    """
    if not rows or not isinstance(run_id, str) or not run_id:
        raise ResponsesAuditError("a worker trajectory requires its run and durable evidence")
    steps, history, request_at, completions, calls = [], {}, {}, {}, {}
    pending, unseen_results, lost = set(), set(), []
    incomplete = False
    binding = None

    def step(row, source, message, **fields):
        value = {"step_id": len(steps) + 1, "timestamp": row["at"], "source": source,
                 "message": message, "extra": {"audit_event_id": row["id"],
                    "context_parent": row["parent"], "memory_published": row["published"],
                    **({"is_sidechain": True} if source == "agent" else {})}, **fields}
        steps.append(value)
        return value

    for row in rows:
        identifier, parent, event = row["id"], row["parent"], row["event"]
        if (identifier in history or event_id(parent, event) != identifier
                or (parent != ROOT and parent not in history) or type(row["published"]) is not bool):
            raise ResponsesAuditError("worker trajectory audit chain is malformed")
        incomplete |= not row["published"]
        request = request_at.get(parent)
        kind = event["kind"]
        if kind == "responses_binding":
            if binding is not None or parent != ROOT or event["binding"]["run_id"] != run_id:
                raise ResponsesAuditError("worker trajectory binding changed")
            binding = event["binding"]
            step(row, "system", binding["request_config"]["system"])
        elif binding is None:
            raise ResponsesAuditError("worker trajectory has no original binding")
        elif kind == "responses_input":
            step(row, "user", event["content"])
        elif kind == "responses_request":
            request = event["id"]
            if request in pending or request in completions:
                raise ResponsesAuditError("worker trajectory repeats a model request")
            pending.add(request)
        elif kind == "responses_completion":
            if request != event["id"] or request not in pending:
                raise ResponsesAuditError("worker reply has no original model request")
            pending.remove(request)
            reply = event["completion"]
            value = step(row, "agent", "\n".join(reply["text_blocks"]), model_name=reply["model"], llm_call_count=1)
            value["extra"].update({"request_id": request, "stop_reason": reply["stop_reason"],
                                    "tool_execution": {}})
            completions[request] = value
            for call in reply["tool_calls"]:
                key = (request, call["id"])
                if key in calls:
                    raise ResponsesAuditError("worker reply repeats a tool call")
                scoped = "call_" + hashlib.sha256((request + "\0" + call["id"]).encode()).hexdigest()
                # Copy arguments so editing the projection cannot edit source evidence.
                value.setdefault("tool_calls", []).append({"tool_call_id": scoped,
                    "function_name": call["name"], "arguments": deepcopy(call["arguments"])})
                calls[key] = (scoped, call, "requested")
                unseen_results.add(key)
                value["extra"]["tool_execution"][scoped] = "requested"
        elif kind == "responses_lost":
            # The reply never came and the call was given up. It is named, and
            # no reply, tool call or result is shown for it.
            if request != event["id"] or request not in pending:
                raise ResponsesAuditError("worker lost reply has no original model request")
            pending.remove(request)
            lost.append(request)
        elif kind in {"responses_tool_intent", "responses_tool_result"}:
            call = event["call"]
            key = (request, call["id"])
            if key not in calls:
                raise ResponsesAuditError("worker tool event has no original model call")
            scoped, original, phase = calls[key]
            if any(call[field] != original[field] for field in ("id", "name", "arguments")):
                raise ResponsesAuditError("worker tool arguments changed")
            value = completions[request]
            if kind == "responses_tool_intent":
                if phase != "requested":
                    raise ResponsesAuditError("worker tool intent was repeated")
                phase = "intent"
            else:
                if phase != "intent":
                    raise ResponsesAuditError("worker tool result has no unique intent")
                result = event["result"]
                value.setdefault("observation", {"results": []})["results"].append({
                    "source_call_id": scoped, "content": result["content"], "extra": {
                        "is_error": result["is_error"], "effect_id": event["effect_id"],
                        "memory_published": row["published"], "timestamp": row["at"]}})
                phase = "result"
                unseen_results.remove(key)
            value["extra"]["tool_execution"][scoped] = phase
            calls[key] = (scoped, original, phase)
        else:
            raise ResponsesAuditError("worker trajectory contains an unsupported audit event")
        history[identifier] = row
        request_at[identifier] = request

    return {"schema_version": "ATIF-v1.7", "session_id": run_id, "trajectory_id": run_id,
            "agent": {"name": "taste-azure-worker", "version": "1", "model_name": binding["model"]},
            "steps": steps, "extra": {"trace_scope": "internal_worker", "complete_attempt": False,
                "incomplete_worker_trace": incomplete or bool(pending) or bool(unseen_results),
                "pending_model_requests": sorted(pending), "lost_model_requests": lost,
                "unobserved_tool_calls": len(unseen_results), "includes_discarded_context": True}}
