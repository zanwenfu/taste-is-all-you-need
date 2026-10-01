"""One chronological record of a goal, for a benchmark that reads steps in order.

The settled goal trajectory keeps each worker as its own embedded trajectory.
That is faithful to how the work was organised, but a reader that walks the
root and then each embedded trajectory in turn sees one worker's calls after
another's, not the order in which the shared task environment saw them. A
record read to check an agent's account of its own work has to be in the
order things happened.

This projection puts every worker step in the root, in time order, one tool
call per step, marked as a sidechain so that only the coordinator's reply can
be read as the answer. Nothing a worker did is invented or dropped: a call
without a recorded result has no observation, and the audit's own findings
travel in ``extra``.

The planner's and monitors' calls use no tools, and their prompts are this
system's internal messages (they restate the task many times and name the
provider route). They are left out of this record unless asked for; the
complete settled trajectory stays in the trial owner's private directory and
is bound here by its digest.

When no settled evidence exists at all, the terminal ledger alone still
yields a truthful record of every command the task environment ran.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from taste.brains.terminal_broker import TerminalResult
from taste.brains.terminal_tools import render

SCHEMA = "ATIF-v1.7"
WORKER_AGENT = "taste-azure-worker"
_EPOCH = datetime.fromtimestamp(0, UTC)


def _moment(text):
    if not isinstance(text, str):
        return _EPOCH
    try:
        value = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError:
        return _EPOCH
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _worker_entries(index, sub):
    """(time, worker, position, call) keys with one flat step per call or message."""
    entries = []
    for position, step in enumerate(sub.get("steps") or []):
        if step.get("source") != "agent":
            continue
        source = step.get("extra") or {}
        shared = {"is_sidechain": True, "worker_run": sub.get("session_id"),
                  **({"request_id": source["request_id"]} if "request_id" in source else {})}
        results = {item.get("source_call_id"): item
                   for item in (step.get("observation") or {}).get("results") or []}
        calls = step.get("tool_calls") or []
        common = {"source": "agent", **({"model_name": step["model_name"]} if step.get("model_name") else {})}
        if not calls:
            entry = {**common, "timestamp": step.get("timestamp"), "message": step.get("message", ""),
                     "llm_call_count": 1, "extra": dict(shared)}
            if step.get("metrics") is not None:
                entry["metrics"] = step["metrics"]
            entries.append(((_moment(step.get("timestamp")), index, position, 0), entry))
            continue
        when = step.get("timestamp")
        for number, call in enumerate(calls):
            result = results.get(call["tool_call_id"])
            # A reply's calls run one after another. One with no recorded
            # result still came after the call before it.
            when = ((result or {}).get("extra") or {}).get("timestamp") or when
            entry = {**common, "timestamp": when,
                     # The reply's own words belong to its first call.
                     "message": step.get("message", "") if number == 0 else "",
                     "tool_calls": [{"tool_call_id": call["tool_call_id"],
                                     "function_name": call["function_name"],
                                     "arguments": call["arguments"]}],
                     "llm_call_count": 1 if number == 0 else 0,
                     "extra": {**shared, "executed": result is not None}}
            if number == 0 and step.get("metrics") is not None:
                entry["metrics"] = step["metrics"]
            if result is not None:
                entry["observation"] = {"results": [{"source_call_id": call["tool_call_id"],
                                                     "content": result.get("content")}]}
                entry["extra"]["tool_error"] = bool((result.get("extra") or {}).get("is_error"))
            entries.append(((_moment(when), index, position, number), entry))
    return entries


def flat_trajectory(nested, *, audit_flags=(), include_internal=False):
    """Project a settled goal trajectory; raises ValueError on a shape it does not know."""
    steps = nested.get("steps") if isinstance(nested, dict) else None
    if (not isinstance(steps, list) or not steps or steps[0].get("source") != "user"
            or nested.get("schema_version") != SCHEMA):
        raise ValueError("a settled goal trajectory is required")
    replies = [step for step in steps[1:] if step.get("source") == "agent"]
    if len(replies) > 1 or len(steps) != 1 + len(replies):
        raise ValueError("a goal trajectory holds its task and at most one final reply")
    embedded = nested.get("subagent_trajectories") or []
    entries = []
    for index, sub in enumerate(embedded):
        if (sub.get("agent") or {}).get("name") == WORKER_AGENT:
            entries.extend(_worker_entries(index, sub))
    entries.sort(key=lambda item: item[0])
    root = [{"step_id": 1, "source": "user", "message": steps[0]["message"]}]
    for _key, entry in entries:
        root.append({"step_id": len(root) + 1, **entry})
    if replies:
        final = {key: value for key, value in replies[0].items() if key != "step_id"}
        final["extra"] = {**(final.get("extra") or {}), "is_sidechain": False}
        root.append({"step_id": len(root) + 1, **final})
    raw = json.dumps(nested, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    models = [step["model_name"] for sub in embedded for step in sub.get("steps") or []
              if (sub.get("agent") or {}).get("name") == "taste-coordinator" and step.get("model_name")]
    result = {
        "schema_version": SCHEMA, "session_id": nested["session_id"],
        "trajectory_id": nested.get("trajectory_id", nested["session_id"]),
        "agent": {**nested["agent"], **({"model_name": models[-1]} if models else {})},
        "steps": root,
        "extra": {**(nested.get("extra") or {}), "record": "chronological",
                  "nested_sha256": hashlib.sha256(raw).hexdigest(),
                  "worker_runs": [sub.get("session_id") for sub in embedded
                                  if (sub.get("agent") or {}).get("name") == WORKER_AGENT],
                  "audit_flags": list(audit_flags)},
    }
    if nested.get("final_metrics") is not None:
        result["final_metrics"] = nested["final_metrics"]
    internal = [sub for sub in embedded if (sub.get("agent") or {}).get("name") != WORKER_AGENT]
    if include_internal and internal:
        # No tools are called there, so nothing is a second copy of a root call.
        result["subagent_trajectories"] = internal
    return result


def ledger_trajectory(session_id, task, commands, *, audit_flags=()):
    """The commands the task environment ran, when no other evidence survived.

    ``commands`` are ledger rows in execution order: request_id, command, cwd,
    timeout_seconds, status and, for a completed row, a TerminalResult. There
    is no reply: an agent whose own record was lost said nothing that can be
    attributed to it.
    """
    steps = [{"step_id": 1, "source": "user", "message": task}]
    for row in commands:
        identifier = "call_" + hashlib.sha256(row["request_id"].encode()).hexdigest()
        step = {"step_id": len(steps) + 1, "source": "agent", "message": "",
                "tool_calls": [{"tool_call_id": identifier, "function_name": "terminal_exec",
                                "arguments": {"command": row["command"], "cwd": row["cwd"],
                                              "timeout_seconds": row["timeout_seconds"]}}],
                "llm_call_count": 0,
                "extra": {"is_sidechain": True, "request_id": row["request_id"],
                          "ledger_status": row["status"]}}
        result = row.get("result")
        if isinstance(result, TerminalResult) and row["status"] == "completed":
            step["observation"] = {"results": [{"source_call_id": identifier, "content": render(
                row["request_id"], row["timeout_seconds"], result)}]}
        steps.append(step)
    return {"schema_version": SCHEMA, "session_id": session_id, "trajectory_id": session_id,
            "agent": {"name": "taste", "version": "1"}, "steps": steps,
            "extra": {"record": "terminal_ledger_only", "evidence_complete": False,
                      "final_reply_present": False, "audit_flags": list(audit_flags)}}


def encode(trajectory):
    return (json.dumps(trajectory, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
