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
    # Calls whose outcome is not settled: the journal admits no call after one.
    unknown_calls: int
    # The most the unknown calls can have cost, each bounded by the request it
    # sent. Zero when no call is unknown.
    unknown_exposure_usd: float = 0.0
    # Calls whose reply was lost and given up: charged at their worst case,
    # after which the journal admitted calls again.
    lost_calls: int = 0
    lost_exposure_usd: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (("known model cost", self.known_cost_usd),
                            ("unknown model call exposure", self.unknown_exposure_usd),
                            ("lost model call exposure", self.lost_exposure_usd)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        for count in (self.completed_calls, self.unknown_calls, self.lost_calls):
            if type(count) is not int or count < 0:
                raise ValueError("model call counts must be nonnegative integers")
        if self.unknown_exposure_usd and not self.unknown_calls:
            raise ValueError("only an unknown model call has an exposure")
        if self.lost_exposure_usd and not self.lost_calls:
            raise ValueError("only a lost model call has an exposure")

    @property
    def cost_usd(self) -> float | None:
        """The exact cost, when every call has a receipt."""
        return None if self.unknown_calls or self.lost_calls else self.known_cost_usd

    @property
    def settled(self) -> bool:
        """Whether every call either has a receipt or was given up and charged."""
        return not self.unknown_calls

    @property
    def exposure_usd(self) -> float:
        """The most the calls without a receipt can have cost."""
        return math.fsum((self.unknown_exposure_usd, self.lost_exposure_usd))

    @property
    def cost_ceiling_usd(self) -> float:
        """What these calls cost at most: the receipts, and each other call's worst case."""
        return math.fsum((self.known_cost_usd, self.exposure_usd))

    @property
    def model_calls(self) -> int:
        return self.completed_calls + self.unknown_calls + self.lost_calls

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
