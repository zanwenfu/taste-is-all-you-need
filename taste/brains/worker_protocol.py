"""Provider-independent worker records, identities and failure categories.

Importing coordination or launch code must not require a particular model SDK.
The legacy runtime re-exports these names for existing callers.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from taste.brains.records import Assignment


@dataclass(frozen=True)
class ModelCallAccounting:
    """Provider receipts, including dispatches whose final cost is unknown."""

    known_cost_usd: float
    completed_calls: int
    unknown_calls: int

    def __post_init__(self) -> None:
        value = self.known_cost_usd
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError("known model cost must be finite and nonnegative")
        for count in (self.completed_calls, self.unknown_calls):
            if type(count) is not int or count < 0:
                raise ValueError("model call counts must be nonnegative integers")

    @property
    def cost_usd(self) -> float | None:
        return None if self.unknown_calls else self.known_cost_usd

    @property
    def model_calls(self) -> int:
        return self.completed_calls + self.unknown_calls

ASSIGNMENT_PATH = "assignment.json"


WORKER_REPORT_PATH = "worker-report.json"

# The goal's original task, verbatim, placed in the integration branch before
# any plan exists. Every worker branch inherits it and is shown it whole: an
# assignment is the coordinator's summary, and a summary of a long developer
# conversation loses the details the work depends on.
GOAL_TASK_PATH = "goal-task.md"


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
