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


def _branch_account(branch, rebuilt, rebuilds, replayed, prefix, live, unfaithful):
    """What serving a branch of an earlier run came to (taste.brains.branch_replay)."""
    rebuild_ok = branch["mode"] != "rebuild" or branch["step"] == 0 or bool(rebuilt and rebuilt["faithful"])
    return {**branch, "replayed_steps": replayed, "prefix_complete": prefix,
            "rebuild": None if rebuilt is None else {
                **rebuilt, "divergences": [row for row in rebuilds if row.get("divergent")][:20]},
            "went_live": bool(live and live["live"]), "context": None if live is None else live["context"],
            "unfaithful": unfaithful, "faithful": prefix and unfaithful is None and rebuild_ok}


def hosted_trajectory(rows, *, run_id):
    """ATIF evidence for a hosted agent's run: its model replies, commands and outputs.

    The agent's own messages stay its own; this is Taste's record of what
    crossed the host: each reply's text and commands, each command's output as
    recorded in memory, a call given up as lost, and how the run ended. In a
    branch of an earlier run, the steps answered from that run's record are
    shown as replayed (no model call, no cost), and the account of the branch
    (what was rebuilt, whether each step matched the record, whether the
    agent went on live) is in ``extra.branch``.
    """
    if not rows or not isinstance(run_id, str) or not run_id:
        raise ResponsesAuditError("a hosted trajectory requires its run and durable evidence")
    steps, history, pending, lost, results = [], set(), set(), [], {}
    binding, current, incomplete, exit_event = None, None, False, None
    branch, rebuilt, rebuilds, rebuilding, replayed, prefix, live, unfaithful = (
        None, None, [], {}, 0, False, None, None)

    def step(row, source, message, **fields):
        value = {"step_id": len(steps) + 1, "timestamp": row["at"], "source": source, "message": message,
                 "extra": {"audit_event_id": row["id"], "memory_published": row["published"],
                           **({"is_sidechain": True} if source == "agent" else {})}, **fields}
        steps.append(value)
        return value

    for row in rows:
        identifier, parent, event = row["id"], row["parent"], row["event"]
        if (identifier in history or event_id(parent, event) != identifier
                or (parent != ROOT and parent not in history) or type(row["published"]) is not bool):
            raise ResponsesAuditError("hosted trajectory audit chain is malformed")
        history.add(identifier)
        incomplete |= not row["published"]
        kind = event["kind"]
        if kind == "hosted_binding":
            if binding is not None or parent != ROOT or event["binding"]["run_id"] != run_id:
                raise ResponsesAuditError("hosted trajectory binding changed")
            binding = event["binding"]
        elif binding is None:
            raise ResponsesAuditError("hosted trajectory has no original binding")
        elif kind == "hosted_task":
            step(row, "user", event["content"])
        elif kind == "hosted_request":
            pending.add(event["id"])
        elif kind == "hosted_lost":
            pending.discard(event["id"])
            lost.append(event["id"])
        elif kind == "hosted_completion":
            if event["id"] not in pending:
                raise ResponsesAuditError("hosted reply has no original model request")
            pending.discard(event["id"])
            current = step(row, "agent", "\n".join(event["text"]), model_name=event["model"], llm_call_count=1)
            current["extra"].update(request_id=event["id"], stop_reason=event["stop_reason"],
                                    cost_usd=event["cost_usd"],
                                    # Paid for, but the run ended before the agent received it.
                                    **({"cut_off": True} if event.get("cut_off") else {}))
            calls = [{"tool_call_id": "call_" + hashlib.sha256((event["id"] + "\0" + call["id"]).encode()).hexdigest(),
                      "function_name": call["name"], "arguments": deepcopy(call["arguments"])}
                     for call in event["calls"]]
            if calls:
                current["tool_calls"] = calls
        elif kind == "hosted_command":
            results[event["effect_id"]] = (current, event)
        elif kind == "hosted_output":
            owner, command = results.pop(event["effect_id"], (None, None))
            if command is None:
                raise ResponsesAuditError("hosted output has no recorded command")
            observed = {"content": event["output"], "extra": {
                "command": command["command"], "returncode": event["returncode"],
                "terminated": event["terminated"], "output_chars": event["output_chars"],
                "effect_id": event["effect_id"], "timestamp": row["at"]}}
            if owner is None:
                step(row, "system", "command run before any model reply", observation={"results": [observed]})
            else:
                owner.setdefault("observation", {"results": []})["results"].append(observed)
        elif kind == "hosted_exit":
            exit_event = event
        elif kind == "hosted_branch":
            if branch is not None:
                raise ResponsesAuditError("hosted trajectory names two branches")
            branch = {key: event[key] for key in ("script_sha256", "step", "mode", "live", "tolerance",
                                                  "source", "task_matches", "agent_matches")}
            branch["override"] = event.get("override", "")
        elif kind == "hosted_rebuild_command":
            rebuilding[event["effect_id"]] = event
        elif kind == "hosted_rebuild_output":
            if rebuilding.pop(event["effect_id"], None) is None:
                raise ResponsesAuditError("rebuilt output has no recorded command")
            rebuilds.append({key: event[key] for key in (
                "step", "returncode", "recorded_returncode", "timed_out", "recorded_timed_out", "similarity",
                "divergent", "terminated") if key in event})
        elif kind == "hosted_rebuilt":
            rebuilt = {key: event[key] for key in ("commands", "divergent", "faithful", "seconds")}
        elif kind == "hosted_replay":
            # Answered from the earlier run's record: no model was asked.
            replayed += 1
            current = step(row, "agent", "\n".join(event["text"]), model_name=event["model"], llm_call_count=0)
            current["extra"].update(replayed=True, branch_step=event["step"], request_sha=event["request_sha"],
                                    recorded_cost_usd=event["recorded_cost_usd"])
            calls = [{"tool_call_id": "call_" + hashlib.sha256(
                         f"replay\0{event['step']}\0{call['id']}".encode()).hexdigest(),
                      "function_name": call["name"], "arguments": deepcopy(call["arguments"])}
                     for call in event["calls"]]
            if calls:
                current["tool_calls"] = calls
        elif kind == "hosted_replay_output":
            if current is None or not current["extra"].get("replayed"):
                raise ResponsesAuditError("replayed output has no replayed reply")
            current.setdefault("observation", {"results": []})["results"].append({
                "content": event["output"], "extra": {
                    "command": event["command"], "returncode": event["returncode"],
                    "timed_out": event["timed_out"], "output_chars": event["output_chars"],
                    "replayed": True, "override": event["override"], "timestamp": row["at"]}})
        elif kind == "hosted_prefix":
            prefix = True
        elif kind == "hosted_live":
            live = live or {"live": event["live"], "context": event["context"]}
        elif kind == "hosted_unfaithful":
            unfaithful = unfaithful or {key: event[key] for key in ("step", "reason", "detail")}
        else:
            raise ResponsesAuditError("hosted trajectory contains an unsupported audit event")

    agent = binding["agent"]
    return {"schema_version": "ATIF-v1.7", "session_id": run_id, "trajectory_id": run_id,
            "agent": {"name": agent["name"], "version": agent["version"], "model_name": binding["model"]},
            "steps": steps, "extra": {
                "trace_scope": "internal_worker", "hosted_agent": agent, "complete_attempt": False,
                "incomplete_worker_trace": incomplete or bool(pending) or bool(results) or bool(rebuilding),
                "pending_model_requests": sorted(pending), "lost_model_requests": lost,
                "unfinished_commands": len(results) + len(rebuilding),
                "exit": None if exit_event is None else {
                    "exit_status": exit_event["exit_status"], "stopped_by": exit_event["stopped_by"]},
                **({} if branch is None else {"branch": _branch_account(
                    branch, rebuilt, rebuilds, replayed, prefix, live, unfaithful)})}}
