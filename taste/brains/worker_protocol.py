"""Provider-independent worker records, identities and failure categories.

Importing coordination or launch code must not require a particular model SDK.
The legacy runtime re-exports these names for existing callers.
"""

from __future__ import annotations

import hashlib

from taste.brains.records import Assignment

ASSIGNMENT_PATH = "assignment.json"


WORKER_REPORT_PATH = "worker-report.json"


WORKER_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["completed", "blocked", "continue"]},
        "summary": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "accepted_inbox_ids": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
        },
        "accepted_verdicts": {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 1},
        },
    },
    "required": [
        "status",
        "summary",
        "evidence",
        "accepted_inbox_ids",
        "accepted_verdicts",
    ],
    "additionalProperties": False,
}


class ContractMismatch(RuntimeError):
    """The runtime was asked to execute something other than durable truth."""


class ShutdownUnconfirmed(RuntimeError):
    """A worker operation may still be active, so branch handoff is unsafe."""


def assignment_run_id(assignment: Assignment) -> str:
    """The same content-bound run identity minted by CentralSupervisor."""
    token = hashlib.sha256(assignment.to_json().encode("utf-8")).hexdigest()
    return f"worker-run.{token}"


def _assignment_monitor_budget_usd(assignment: Assignment) -> float | None:
    if "monitor_budget_usd" not in assignment.resources:
        return None
    value = assignment.resources["monitor_budget_usd"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < float(value) < float("inf")
    ):
        raise ValueError("Assignment.resources.monitor_budget_usd must be finite and positive")
    return float(value)

