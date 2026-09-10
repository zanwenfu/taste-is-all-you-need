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
from dataclasses import dataclass
from typing import Any

__all__ = ["CONTRACT_PATH", "Contract"]

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
    # Declared as tuples and normalised in __post_init__: a caller passing a
    # list produces a Contract that is not equal to the same contract read
    # back from JSON, which turns a round-trip test into a false negative.
    issued_by: str = "central"
    notes: str = ""
    budget_usd: float | None = None
    max_turns: int | None = None

    def __post_init__(self) -> None:
        if not self.identity.strip():
            raise ValueError("a sub-brain must have an identity")
        # The identity IS the branch name, so it has to satisfy memstore's own
        # rule. Checked here rather than at spawn because the planner is the
        # one that can still choose a different name: a contract that only
        # fails when the spawner opens the branch fails after the planner has
        # already committed to a decomposition around it.
        from taste.memstore.store import _check_name

        _check_name(self.identity, "branch")
        if not self.task.strip():
            raise ValueError(f"{self.identity}: a sub-brain must have a task")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "success_criteria", tuple(self.success_criteria))
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

    @staticmethod
    def _lines(value: Any) -> tuple[str, ...]:
        """Coerce a field that should be a list of lines.

        A bare string is the likely mistake -- a planner writing
        ``"success_criteria": "the tests pass"`` -- and ``tuple()`` would
        silently shred it into single characters, leaving a contract with
        ten one-character criteria that no monitor can judge.
        """
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return tuple(str(v) for v in value)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Contract:
        return cls(
            identity=raw["identity"],
            task=raw["task"],
            inputs=cls._lines(raw.get("inputs")),
            outputs=cls._lines(raw.get("outputs")),
            success_criteria=cls._lines(raw.get("success_criteria")),
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
