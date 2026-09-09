"""What a sub-brain is told when it is created.

A sub-brain never receives a bare goal. The central brain's planner lobe issues
it an identity, a task, the inputs it can expect, the outputs it owes, and the
criteria by which it will be judged -- and its monitor is created from the same
contract, so the thing doing the work and the thing grading it are reading the
same document. A success criterion that lives only in the monitor's prompt is a
criterion the worker was never told about.

The contract is a record in memstore, so it survives a kill, moves with a
rollback, and is readable by any brain that can read the branch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Contract", "CONTRACT_PATH"]

CONTRACT_PATH = "contract.json"
"""Where a sub-brain's contract lives in its branch."""


@dataclass(frozen=True)
class Contract:
    """The brief a sub-brain is created with.

    ``identity`` is who this brain is, in one line, and it is also its branch
    name -- one address space, one identity, one contract.
    """

    identity: str
    task: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    issued_by: str = "central"
    notes: str = ""
    budget_usd: float | None = None
    max_turns: int | None = None

    def __post_init__(self) -> None:
        if not self.identity.strip():
            raise ValueError("a sub-brain must have an identity")
        if not self.task.strip():
            raise ValueError(f"{self.identity}: a sub-brain must have a task")
        if not self.success_criteria:
            # Without this a monitor has nothing to judge against and falls
            # back to taste, which is exactly the failure the architecture is
            # trying to remove.
            raise ValueError(f"{self.identity}: a sub-brain must have success criteria")

    # ------------------------------------------------------------------ text

    def brief(self) -> str:
        """The contract as the worker is told it.

        Deliberately plain prose rather than JSON: this goes into a model's
        context, where structure costs tokens and buys nothing.
        """
        parts = [f"You are {self.identity}.", "", f"Your task: {self.task}"]
        if self.inputs:
            parts += ["", "You have been given:"]
            parts += [f"  - {i}" for i in self.inputs]
        if self.outputs:
            parts += ["", "You must produce:"]
            parts += [f"  - {o}" for o in self.outputs]
        parts += ["", "You are done when all of these are true:"]
        parts += [f"  - {c}" for c in self.success_criteria]
        if self.notes:
            parts += ["", self.notes]
        return "\n".join(parts)

    def judging_brief(self) -> str:
        """The same contract as the monitor is told it."""
        parts = [
            f"You are watching a worker called {self.identity}.",
            "",
            f"Its task: {self.task}",
            "",
            "It succeeds only when all of these are true:",
        ]
        parts += [f"  - {c}" for c in self.success_criteria]
        if self.outputs:
            parts += ["", "It owes these outputs:"]
            parts += [f"  - {o}" for o in self.outputs]
        return "\n".join(parts)

    # ------------------------------------------------------------------ json

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "task": self.task,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "success_criteria": list(self.success_criteria),
            "issued_by": self.issued_by,
            "notes": self.notes,
            "budget_usd": self.budget_usd,
            "max_turns": self.max_turns,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Contract:
        return cls(
            identity=raw["identity"],
            task=raw["task"],
            inputs=tuple(raw.get("inputs", ())),
            outputs=tuple(raw.get("outputs", ())),
            success_criteria=tuple(raw.get("success_criteria", ())),
            issued_by=raw.get("issued_by", "central"),
            notes=raw.get("notes", ""),
            budget_usd=raw.get("budget_usd"),
            max_turns=raw.get("max_turns"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Contract:
        return cls.from_dict(json.loads(text))
