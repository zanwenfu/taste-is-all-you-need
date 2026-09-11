"""Durable value records shared by the central brain and worker runtimes.

This module deliberately contains no planning or routing policy.  It defines
the records that those mechanisms exchange and persist: assignments name exact
inputs and outputs, plans name the world-state they were made against, process
events distinguish observations from claims, and worker reports never use a
zero value to mean "unknown".

Every record has an explicit schema and caller-supplied stable identifier.
Wire decoding is strict: missing and unknown fields are rejected, with future
extensions carried in ``metadata`` instead of being silently ignored.  The
metadata and resource maps are recursively frozen on construction so a frozen
record cannot be mutated through one of its values.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar

from taste.brains.contract import Contract

__all__ = [
    "ArtifactRef",
    "ArtifactSpec",
    "Assignment",
    "LifecycleEvent",
    "PlanRevision",
    "WorkerReport",
    "contract_digest",
]

_RECORD_FAMILY = "taste.brains"
_DIGEST_RE = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
_OBJECT_ID_RE = re.compile(r"\A(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_TOKEN_RE = re.compile(r"\A[a-z][a-z0-9_.-]*\Z")
_BRANCH_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MONITOR_SEVERITIES = frozenset({"fine", "drifting", "wrong", "lost", "unknown"})
_ARTIFACT_DISPOSITIONS = frozenset({"present", "absent"})


def _schema(name: str) -> str:
    return f"{_RECORD_FAMILY}/{name}/1"


def _expect_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a JSON object")
    if not all(isinstance(k, str) for k in value):
        raise ValueError(f"{where} keys must be strings")
    return value


def _check_fields(
    raw: dict[str, Any],
    *,
    schema: str,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    where = schema.rsplit("/", 2)[-2]
    got_schema = raw.get("schema")
    if got_schema != schema:
        raise ValueError(f"{where}.schema must be {schema!r}, got {got_schema!r}")
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"{where} is missing required fields: {sorted(missing)}")
    unknown = raw.keys() - required - optional - {"schema"}
    if unknown:
        raise ValueError(
            f"{where} has unknown fields {sorted(unknown)}; put extensions in metadata"
        )


def _expect_str(value: Any, where: str, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string")
    if not empty and not value.strip():
        raise ValueError(f"{where} must not be empty")
    if "\x00" in value:
        raise ValueError(f"{where} must not contain NUL")
    return value


def _stable_id(value: Any, where: str) -> str:
    text = _expect_str(value, where)
    if text != text.strip() or len(text) > 256 or any(ord(c) < 32 for c in text):
        raise ValueError(f"{where} is not a stable identifier")
    return text


def _branch(value: Any, where: str) -> str:
    text = _expect_str(value, where)
    if not _BRANCH_RE.fullmatch(text) or text in {".", ".."} or text.endswith(".lock"):
        raise ValueError(f"{where} is not a valid branch identity")
    return text


def _object_id(value: Any, where: str) -> str:
    text = _expect_str(value, where)
    if not _OBJECT_ID_RE.fullmatch(text):
        raise ValueError(f"{where} must be a full lowercase 40- or 64-hex object ID")
    return text


def _state_id(value: Any, where: str) -> str:
    return _object_id(value, where)


def _token(value: Any, where: str) -> str:
    text = _expect_str(value, where)
    if not _TOKEN_RE.fullmatch(text):
        raise ValueError(f"{where} must be a lowercase token")
    return text


def _relative_path(value: Any, where: str) -> str:
    text = _expect_str(value, where)
    if "\\" in text:
        raise ValueError(f"{where} must use '/' separators")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in text.split("/")):
        raise ValueError(f"{where} must be a normalized relative path")
    return text


def _expect_int(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an integer")
    if value < minimum:
        raise ValueError(f"{where} must be >= {minimum}")
    return value


def _expect_float(value: Any, where: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{where} must be finite and >= {minimum}")
    return number


def _expect_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{where} must be a boolean")
    return value


def _timestamp(value: Any, where: str) -> str:
    text = _expect_str(value, where)
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError as exc:
        raise ValueError(f"{where} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{where} must include a timezone")
    return text


def _expect_array(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a JSON array")
    return value


def _string_tuple(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{where} must be a tuple of strings")
    out = tuple(_stable_id(item, f"{where}[{i}]") for i, item in enumerate(value))
    if len(set(out)) != len(out):
        raise ValueError(f"{where} must not contain duplicates")
    return out


def _strings_from_wire(value: Any, where: str) -> tuple[str, ...]:
    return tuple(
        _stable_id(item, f"{where}[{i}]")
        for i, item in enumerate(_expect_array(value, where))
    )


def _freeze_json(value: Any, where: str) -> Any:
    """Validate and recursively freeze a JSON-compatible value."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where} keys must be strings")
            frozen[key] = _freeze_json(item, f"{where}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{where}[{i}]") for i, item in enumerate(value))
    raise ValueError(f"{where} contains non-JSON value {type(value).__name__}")


def _freeze_map(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a JSON object")
    frozen = _freeze_json(value, where)
    assert isinstance(frozen, Mapping)
    return frozen


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _loads(text: str, where: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"{where} contains duplicate key {key!r}")
            out[key] = value
        return out

    def no_constant(value: str) -> Any:
        raise ValueError(f"{where} contains non-JSON number {value}")

    try:
        raw = json.loads(text, object_pairs_hook=no_duplicates, parse_constant=no_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where} is not valid JSON") from exc
    return _expect_object(raw, where)


class _JsonRecord:
    SCHEMA: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - abstract convention
        raise NotImplementedError

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, allow_nan=False, indent=1, sort_keys=True
        ) + "\n"

    @classmethod
    def from_json(cls, text: str):
        return cls.from_dict(_loads(text, cls.__name__))


def _validate_contract(contract: Any, where: str = "contract") -> Contract:
    if not isinstance(contract, Contract):
        raise ValueError(f"{where} must be a Contract")
    _branch(contract.identity, f"{where}.identity")
    _expect_str(contract.task, f"{where}.task")
    if not contract.success_criteria:
        raise ValueError(f"{where}.success_criteria must not be empty")
    for field_name in ("inputs", "outputs", "success_criteria"):
        values = getattr(contract, field_name)
        if not isinstance(values, tuple) or not all(
            isinstance(item, str) and item.strip() for item in values
        ):
            raise ValueError(f"{where}.{field_name} must be a tuple of non-empty strings")
    _expect_str(contract.issued_by, f"{where}.issued_by")
    _expect_str(contract.notes, f"{where}.notes", empty=True)
    if contract.budget_usd is not None:
        _expect_float(contract.budget_usd, f"{where}.budget_usd", minimum=0.0)
    if contract.max_turns is not None:
        _expect_int(contract.max_turns, f"{where}.max_turns", minimum=1)
    return contract


def _contract_from_dict(value: Any, where: str) -> Contract:
    raw = _expect_object(value, where)
    allowed = {
        "identity",
        "task",
        "inputs",
        "outputs",
        "success_criteria",
        "issued_by",
        "notes",
        "budget_usd",
        "max_turns",
    }
    missing = {"identity", "task", "success_criteria"} - raw.keys()
    if missing:
        raise ValueError(f"{where} is missing required fields: {sorted(missing)}")
    unknown = raw.keys() - allowed
    if unknown:
        raise ValueError(f"{where} has unknown fields: {sorted(unknown)}")
    identity = _branch(raw["identity"], f"{where}.identity")
    task = _expect_str(raw["task"], f"{where}.task")

    def lines(name: str, *, required: bool = False) -> tuple[str, ...]:
        if name not in raw:
            if required:
                raise ValueError(f"{where}.{name} is required")
            return ()
        return tuple(
            _expect_str(item, f"{where}.{name}[{i}]")
            for i, item in enumerate(_expect_array(raw[name], f"{where}.{name}"))
        )

    budget = raw.get("budget_usd")
    if budget is not None:
        budget = _expect_float(budget, f"{where}.budget_usd")
    turns = raw.get("max_turns")
    if turns is not None:
        turns = _expect_int(turns, f"{where}.max_turns", minimum=1)
    contract = Contract(
        identity=identity,
        task=task,
        inputs=lines("inputs"),
        outputs=lines("outputs"),
        success_criteria=lines("success_criteria", required=True),
        issued_by=_expect_str(raw.get("issued_by", "central"), f"{where}.issued_by"),
        notes=_expect_str(raw.get("notes", ""), f"{where}.notes", empty=True),
        budget_usd=budget,
        max_turns=turns,
    )
    return _validate_contract(contract, where)


def _contract_wire_dict(contract: Contract) -> dict[str, Any]:
    """The canonical Contract representation used on the wire and in digests."""
    checked = _validate_contract(contract)
    canonical = checked.to_dict()
    # ``budget_usd`` is semantically a float, but Python permits callers to
    # construct ``Contract(budget_usd=2)``.  JSON distinguishes ``2`` from
    # ``2.0`` textually while the wire decoder correctly materialises the field
    # as a float.  Canonicalise it here so a round trip cannot change the digest.
    if canonical["budget_usd"] is not None:
        canonical["budget_usd"] = float(canonical["budget_usd"])
    return canonical


def contract_digest(contract: Contract) -> str:
    """A stable, self-describing digest of the exact contract fields."""
    payload = json.dumps(
        _contract_wire_dict(contract),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _checked_digest(value: Any, where: str) -> str:
    text = _expect_str(value, where)
    if not _DIGEST_RE.fullmatch(text):
        raise ValueError(f"{where} must be a sha256 digest")
    return text


@dataclass(frozen=True, slots=True)
class ArtifactSpec(_JsonRecord):
    """One artifact an assignment is required to produce or remove."""

    SCHEMA: ClassVar[str] = _schema("ArtifactSpec")

    artifact_id: str
    path: str
    kind: str = "file"
    description: str = ""
    required: bool = True
    disposition: str = "present"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _stable_id(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "path", _relative_path(self.path, "path"))
        object.__setattr__(self, "kind", _token(self.kind, "kind"))
        object.__setattr__(self, "description", _expect_str(self.description, "description", empty=True))
        object.__setattr__(self, "required", _expect_bool(self.required, "required"))
        disposition = _expect_str(self.disposition, "disposition")
        if disposition not in _ARTIFACT_DISPOSITIONS:
            raise ValueError(f"disposition must be one of {sorted(_ARTIFACT_DISPOSITIONS)}")
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "metadata", _freeze_map(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "artifact_id": self.artifact_id,
            "path": self.path,
            "kind": self.kind,
            "description": self.description,
            "required": self.required,
            "disposition": self.disposition,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> ArtifactSpec:
        raw = _expect_object(value, "ArtifactSpec")
        _check_fields(
            raw,
            schema=cls.SCHEMA,
            required={"artifact_id", "path", "kind", "description", "required", "disposition", "metadata"},
        )
        return cls(
            artifact_id=raw["artifact_id"],
            path=raw["path"],
            kind=raw["kind"],
            description=raw["description"],
            required=raw["required"],
            disposition=raw["disposition"],
            metadata=raw["metadata"],
        )


@dataclass(frozen=True, slots=True)
class ArtifactRef(_JsonRecord):
    """An immutable pointer to bytes in one exact branch state."""

    SCHEMA: ClassVar[str] = _schema("ArtifactRef")

    artifact_id: str
    branch: str
    state_id: str
    path: str
    blob_id: str
    kind: str = "file"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _stable_id(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "branch", _branch(self.branch, "branch"))
        object.__setattr__(self, "state_id", _state_id(self.state_id, "state_id"))
        object.__setattr__(self, "path", _relative_path(self.path, "path"))
        object.__setattr__(self, "blob_id", _object_id(self.blob_id, "blob_id"))
        object.__setattr__(self, "kind", _token(self.kind, "kind"))
        object.__setattr__(self, "metadata", _freeze_map(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "artifact_id": self.artifact_id,
            "branch": self.branch,
            "state_id": self.state_id,
            "path": self.path,
            "blob_id": self.blob_id,
            "kind": self.kind,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> ArtifactRef:
        raw = _expect_object(value, "ArtifactRef")
        _check_fields(
            raw,
            schema=cls.SCHEMA,
            required={"artifact_id", "branch", "state_id", "path", "blob_id", "kind", "metadata"},
        )
        return cls(
            artifact_id=raw["artifact_id"],
            branch=raw["branch"],
            state_id=raw["state_id"],
            path=raw["path"],
            blob_id=raw["blob_id"],
            kind=raw["kind"],
            metadata=raw["metadata"],
        )


@dataclass(frozen=True, slots=True)
class Assignment(_JsonRecord):
    """One versioned worker assignment accepted from a plan revision."""

    SCHEMA: ClassVar[str] = _schema("Assignment")

    assignment_id: str
    generation: int
    attempt: int
    contract: Contract
    contract_digest: str
    base_state_id: str
    depends_on: tuple[str, ...] = ()
    inputs: tuple[ArtifactRef, ...] = ()
    outputs: tuple[ArtifactSpec, ...] = ()
    model: str = "claude-sonnet-5"
    resources: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignment_id", _stable_id(self.assignment_id, "assignment_id"))
        object.__setattr__(self, "generation", _expect_int(self.generation, "generation", minimum=1))
        object.__setattr__(self, "attempt", _expect_int(self.attempt, "attempt"))
        checked = _validate_contract(self.contract)
        supplied_digest = _checked_digest(self.contract_digest, "contract_digest")
        wanted_digest = contract_digest(checked)
        if supplied_digest != wanted_digest:
            raise ValueError("contract_digest does not match contract")
        object.__setattr__(self, "contract_digest", supplied_digest)
        object.__setattr__(self, "base_state_id", _state_id(self.base_state_id, "base_state_id"))
        depends = _string_tuple(self.depends_on, "depends_on")
        if self.assignment_id in depends:
            raise ValueError("an assignment cannot depend on itself")
        object.__setattr__(self, "depends_on", depends)
        if not isinstance(self.inputs, tuple) or not all(isinstance(v, ArtifactRef) for v in self.inputs):
            raise ValueError("inputs must be a tuple of ArtifactRef")
        if not isinstance(self.outputs, tuple) or not all(isinstance(v, ArtifactSpec) for v in self.outputs):
            raise ValueError("outputs must be a tuple of ArtifactSpec")
        input_ids = [item.artifact_id for item in self.inputs]
        output_ids = [item.artifact_id for item in self.outputs]
        output_paths = [item.path for item in self.outputs]
        if len(set(input_ids)) != len(input_ids):
            raise ValueError("inputs contain duplicate artifact_id values")
        if len(set(output_ids)) != len(output_ids):
            raise ValueError("outputs contain duplicate artifact_id values")
        if len(set(output_paths)) != len(output_paths):
            raise ValueError("outputs contain duplicate paths")
        object.__setattr__(self, "model", _expect_str(self.model, "model"))
        object.__setattr__(self, "resources", _freeze_map(self.resources, "resources"))
        object.__setattr__(self, "metadata", _freeze_map(self.metadata, "metadata"))

    @property
    def worker(self) -> str:
        return self.contract.identity

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "assignment_id": self.assignment_id,
            "generation": self.generation,
            "attempt": self.attempt,
            "contract": _contract_wire_dict(self.contract),
            "contract_digest": self.contract_digest,
            "base_state_id": self.base_state_id,
            "depends_on": list(self.depends_on),
            "inputs": [item.to_dict() for item in self.inputs],
            "outputs": [item.to_dict() for item in self.outputs],
            "model": self.model,
            "resources": _thaw_json(self.resources),
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> Assignment:
        raw = _expect_object(value, "Assignment")
        _check_fields(
            raw,
            schema=cls.SCHEMA,
            required={
                "assignment_id",
                "generation",
                "attempt",
                "contract",
                "contract_digest",
                "base_state_id",
                "depends_on",
                "inputs",
                "outputs",
                "model",
                "resources",
                "metadata",
            },
        )
        return cls(
            assignment_id=raw["assignment_id"],
            generation=raw["generation"],
            attempt=raw["attempt"],
            contract=_contract_from_dict(raw["contract"], "Assignment.contract"),
            contract_digest=raw["contract_digest"],
            base_state_id=raw["base_state_id"],
            depends_on=_strings_from_wire(raw["depends_on"], "Assignment.depends_on"),
            inputs=tuple(
                ArtifactRef.from_dict(item)
                for item in _expect_array(raw["inputs"], "Assignment.inputs")
            ),
            outputs=tuple(
                ArtifactSpec.from_dict(item)
                for item in _expect_array(raw["outputs"], "Assignment.outputs")
            ),
            model=raw["model"],
            resources=raw["resources"],
            metadata=raw["metadata"],
        )


def _assert_acyclic(assignments: tuple[Assignment, ...]) -> None:
    graph = {assignment.assignment_id: assignment.depends_on for assignment in assignments}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("assignment dependencies contain a cycle")
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for assignment_id in graph:
        visit(assignment_id)


@dataclass(frozen=True, slots=True)
class PlanRevision(_JsonRecord):
    """One immutable plan generation, fenced to an observed world state."""

    SCHEMA: ClassVar[str] = _schema("PlanRevision")

    plan_id: str
    generation: int
    goal_id: str
    based_on_state_id: str
    observed_heads: Mapping[str, str]
    assignments: tuple[Assignment, ...]
    created_at: str
    parent_plan_id: str | None = None
    rationale: str = ""
    complete: bool = False
    completion_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _stable_id(self.plan_id, "plan_id"))
        object.__setattr__(self, "generation", _expect_int(self.generation, "generation", minimum=1))
        object.__setattr__(self, "goal_id", _stable_id(self.goal_id, "goal_id"))
        object.__setattr__(self, "based_on_state_id", _state_id(self.based_on_state_id, "based_on_state_id"))
        if not isinstance(self.observed_heads, Mapping):
            raise ValueError("observed_heads must be an object")
        heads: dict[str, str] = {}
        for branch, state_id in self.observed_heads.items():
            checked_branch = _branch(branch, "observed_heads branch")
            heads[checked_branch] = _state_id(state_id, f"observed_heads.{checked_branch}")
        object.__setattr__(self, "observed_heads", MappingProxyType(heads))
        if not isinstance(self.assignments, tuple) or not all(
            isinstance(item, Assignment) for item in self.assignments
        ):
            raise ValueError("assignments must be a tuple of Assignment")
        ids = [item.assignment_id for item in self.assignments]
        workers = [item.worker for item in self.assignments]
        if len(ids) != len(set(ids)):
            raise ValueError("assignments contain duplicate assignment_id values")
        if len(workers) != len(set(workers)):
            raise ValueError("assignments contain duplicate worker identities")
        known = set(ids)
        for assignment in self.assignments:
            if assignment.generation != self.generation:
                raise ValueError(
                    "every assignment must belong to the exact plan generation"
                )
            unknown = set(assignment.depends_on) - known
            if unknown:
                raise ValueError(
                    f"{assignment.assignment_id} has unknown dependencies: {sorted(unknown)}"
                )
        _assert_acyclic(self.assignments)
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))
        if self.parent_plan_id is not None:
            parent = _stable_id(self.parent_plan_id, "parent_plan_id")
            if parent == self.plan_id:
                raise ValueError("a plan cannot be its own parent")
            object.__setattr__(self, "parent_plan_id", parent)
        object.__setattr__(self, "rationale", _expect_str(self.rationale, "rationale", empty=True))
        object.__setattr__(self, "complete", _expect_bool(self.complete, "complete"))
        object.__setattr__(
            self,
            "completion_reason",
            _expect_str(self.completion_reason, "completion_reason", empty=True),
        )
        if self.complete != bool(self.completion_reason.strip()):
            raise ValueError("complete and completion_reason must be set together")
        object.__setattr__(self, "metadata", _freeze_map(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "plan_id": self.plan_id,
            "generation": self.generation,
            "goal_id": self.goal_id,
            "based_on_state_id": self.based_on_state_id,
            "observed_heads": dict(self.observed_heads),
            "assignments": [item.to_dict() for item in self.assignments],
            "created_at": self.created_at,
            "parent_plan_id": self.parent_plan_id,
            "rationale": self.rationale,
            "complete": self.complete,
            "completion_reason": self.completion_reason,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> PlanRevision:
        raw = _expect_object(value, "PlanRevision")
        _check_fields(
            raw,
            schema=cls.SCHEMA,
            required={
                "plan_id",
                "generation",
                "goal_id",
                "based_on_state_id",
                "observed_heads",
                "assignments",
                "created_at",
                "parent_plan_id",
                "rationale",
                "complete",
                "completion_reason",
                "metadata",
            },
        )
        heads = _expect_object(raw["observed_heads"], "PlanRevision.observed_heads")
        return cls(
            plan_id=raw["plan_id"],
            generation=raw["generation"],
            goal_id=raw["goal_id"],
            based_on_state_id=raw["based_on_state_id"],
            observed_heads=heads,
            assignments=tuple(
                Assignment.from_dict(item)
                for item in _expect_array(raw["assignments"], "PlanRevision.assignments")
            ),
            created_at=raw["created_at"],
            parent_plan_id=raw["parent_plan_id"],
            rationale=raw["rationale"],
            complete=raw["complete"],
            completion_reason=raw["completion_reason"],
            metadata=raw["metadata"],
        )


@dataclass(frozen=True, slots=True)
class LifecycleEvent(_JsonRecord):
    """One durable observation made by the mechanical supervisor."""

    SCHEMA: ClassVar[str] = _schema("LifecycleEvent")

    event_id: str
    run_id: str
    assignment_id: str
    worker: str
    generation: int
    attempt: int
    kind: str
    at: str
    observed_state_id: str | None = None
    pid: int | None = None
    process_group_id: int | None = None
    exit_code: int | None = None
    signal: int | None = None
    terminal: bool = False
    terminal_reason: str = ""
    uncertain: bool = False
    uncertainty_reasons: tuple[str, ...] = ()
    detail: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("event_id", "run_id", "assignment_id"):
            object.__setattr__(self, name, _stable_id(getattr(self, name), name))
        object.__setattr__(self, "worker", _branch(self.worker, "worker"))
        object.__setattr__(self, "generation", _expect_int(self.generation, "generation", minimum=1))
        object.__setattr__(self, "attempt", _expect_int(self.attempt, "attempt"))
        object.__setattr__(self, "kind", _token(self.kind, "kind"))
        object.__setattr__(self, "at", _timestamp(self.at, "at"))
        if self.observed_state_id is not None:
            object.__setattr__(
                self,
                "observed_state_id",
                _state_id(self.observed_state_id, "observed_state_id"),
            )
        for name in ("pid", "process_group_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _expect_int(value, name, minimum=1))
        if self.exit_code is not None:
            object.__setattr__(self, "exit_code", _expect_int(self.exit_code, "exit_code"))
        if self.signal is not None:
            object.__setattr__(self, "signal", _expect_int(self.signal, "signal", minimum=1))
        if self.exit_code is not None and self.signal is not None:
            raise ValueError("exit_code and signal are mutually exclusive")
        object.__setattr__(self, "terminal", _expect_bool(self.terminal, "terminal"))
        object.__setattr__(
            self,
            "terminal_reason",
            _expect_str(self.terminal_reason, "terminal_reason", empty=True),
        )
        if self.terminal != bool(self.terminal_reason.strip()):
            raise ValueError("terminal and terminal_reason must be set together")
        object.__setattr__(self, "uncertain", _expect_bool(self.uncertain, "uncertain"))
        reasons = _string_tuple(self.uncertainty_reasons, "uncertainty_reasons")
        if self.uncertain != bool(reasons):
            raise ValueError("uncertain and uncertainty_reasons must be set together")
        object.__setattr__(self, "uncertainty_reasons", reasons)
        object.__setattr__(self, "detail", _expect_str(self.detail, "detail", empty=True))
        object.__setattr__(self, "metadata", _freeze_map(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "assignment_id": self.assignment_id,
            "worker": self.worker,
            "generation": self.generation,
            "attempt": self.attempt,
            "kind": self.kind,
            "at": self.at,
            "observed_state_id": self.observed_state_id,
            "pid": self.pid,
            "process_group_id": self.process_group_id,
            "exit_code": self.exit_code,
            "signal": self.signal,
            "terminal": self.terminal,
            "terminal_reason": self.terminal_reason,
            "uncertain": self.uncertain,
            "uncertainty_reasons": list(self.uncertainty_reasons),
            "detail": self.detail,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> LifecycleEvent:
        raw = _expect_object(value, "LifecycleEvent")
        fields = {
            "event_id",
            "run_id",
            "assignment_id",
            "worker",
            "generation",
            "attempt",
            "kind",
            "at",
            "observed_state_id",
            "pid",
            "process_group_id",
            "exit_code",
            "signal",
            "terminal",
            "terminal_reason",
            "uncertain",
            "uncertainty_reasons",
            "detail",
            "metadata",
        }
        _check_fields(raw, schema=cls.SCHEMA, required=fields)
        return cls(
            event_id=raw["event_id"],
            run_id=raw["run_id"],
            assignment_id=raw["assignment_id"],
            worker=raw["worker"],
            generation=raw["generation"],
            attempt=raw["attempt"],
            kind=raw["kind"],
            at=raw["at"],
            observed_state_id=raw["observed_state_id"],
            pid=raw["pid"],
            process_group_id=raw["process_group_id"],
            exit_code=raw["exit_code"],
            signal=raw["signal"],
            terminal=raw["terminal"],
            terminal_reason=raw["terminal_reason"],
            uncertain=raw["uncertain"],
            uncertainty_reasons=_strings_from_wire(
                raw["uncertainty_reasons"], "LifecycleEvent.uncertainty_reasons"
            ),
            detail=raw["detail"],
            metadata=raw["metadata"],
        )


@dataclass(frozen=True, slots=True)
class WorkerReport(_JsonRecord):
    """A worker's terminal report, pinned to exact base and final states."""

    SCHEMA: ClassVar[str] = _schema("WorkerReport")

    report_id: str
    run_id: str
    assignment_id: str
    worker: str
    generation: int
    attempt: int
    contract_digest: str
    base_state_id: str
    final_state_id: str
    at: str
    completed: bool
    terminal_reason: str
    outputs: tuple[ArtifactRef, ...] = ()
    turns: int | None = None
    cost_usd: float | None = None
    monitor_severity: str = "unknown"
    uncertain: bool = False
    uncertainty_reasons: tuple[str, ...] = ()
    summary: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("report_id", "run_id", "assignment_id"):
            object.__setattr__(self, name, _stable_id(getattr(self, name), name))
        object.__setattr__(self, "worker", _branch(self.worker, "worker"))
        object.__setattr__(self, "generation", _expect_int(self.generation, "generation", minimum=1))
        object.__setattr__(self, "attempt", _expect_int(self.attempt, "attempt"))
        object.__setattr__(
            self, "contract_digest", _checked_digest(self.contract_digest, "contract_digest")
        )
        object.__setattr__(self, "base_state_id", _state_id(self.base_state_id, "base_state_id"))
        object.__setattr__(self, "final_state_id", _state_id(self.final_state_id, "final_state_id"))
        object.__setattr__(self, "at", _timestamp(self.at, "at"))
        object.__setattr__(self, "completed", _expect_bool(self.completed, "completed"))
        object.__setattr__(self, "terminal_reason", _token(self.terminal_reason, "terminal_reason"))
        if not isinstance(self.outputs, tuple) or not all(isinstance(v, ArtifactRef) for v in self.outputs):
            raise ValueError("outputs must be a tuple of ArtifactRef")
        output_ids = [item.artifact_id for item in self.outputs]
        if len(set(output_ids)) != len(output_ids):
            raise ValueError("outputs contain duplicate artifact_id values")
        for output in self.outputs:
            if output.branch != self.worker or output.state_id != self.final_state_id:
                raise ValueError("every output must point to this worker's exact final state")
        if self.turns is not None:
            object.__setattr__(self, "turns", _expect_int(self.turns, "turns"))
        if self.cost_usd is not None:
            object.__setattr__(self, "cost_usd", _expect_float(self.cost_usd, "cost_usd"))
        severity = _expect_str(self.monitor_severity, "monitor_severity")
        if severity not in _MONITOR_SEVERITIES:
            raise ValueError(f"monitor_severity must be one of {sorted(_MONITOR_SEVERITIES)}")
        object.__setattr__(self, "monitor_severity", severity)
        object.__setattr__(self, "uncertain", _expect_bool(self.uncertain, "uncertain"))
        reasons = _string_tuple(self.uncertainty_reasons, "uncertainty_reasons")
        if self.uncertain != bool(reasons):
            raise ValueError("uncertain and uncertainty_reasons must be set together")
        object.__setattr__(self, "uncertainty_reasons", reasons)
        object.__setattr__(self, "summary", _expect_str(self.summary, "summary", empty=True))
        object.__setattr__(self, "metadata", _freeze_map(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "report_id": self.report_id,
            "run_id": self.run_id,
            "assignment_id": self.assignment_id,
            "worker": self.worker,
            "generation": self.generation,
            "attempt": self.attempt,
            "contract_digest": self.contract_digest,
            "base_state_id": self.base_state_id,
            "final_state_id": self.final_state_id,
            "at": self.at,
            "completed": self.completed,
            "terminal_reason": self.terminal_reason,
            "outputs": [item.to_dict() for item in self.outputs],
            "turns": self.turns,
            "cost_usd": self.cost_usd,
            "monitor_severity": self.monitor_severity,
            "uncertain": self.uncertain,
            "uncertainty_reasons": list(self.uncertainty_reasons),
            "summary": self.summary,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> WorkerReport:
        raw = _expect_object(value, "WorkerReport")
        fields = {
            "report_id",
            "run_id",
            "assignment_id",
            "worker",
            "generation",
            "attempt",
            "contract_digest",
            "base_state_id",
            "final_state_id",
            "at",
            "completed",
            "terminal_reason",
            "outputs",
            "turns",
            "cost_usd",
            "monitor_severity",
            "uncertain",
            "uncertainty_reasons",
            "summary",
            "metadata",
        }
        _check_fields(raw, schema=cls.SCHEMA, required=fields)
        return cls(
            report_id=raw["report_id"],
            run_id=raw["run_id"],
            assignment_id=raw["assignment_id"],
            worker=raw["worker"],
            generation=raw["generation"],
            attempt=raw["attempt"],
            contract_digest=raw["contract_digest"],
            base_state_id=raw["base_state_id"],
            final_state_id=raw["final_state_id"],
            at=raw["at"],
            completed=raw["completed"],
            terminal_reason=raw["terminal_reason"],
            outputs=tuple(
                ArtifactRef.from_dict(item)
                for item in _expect_array(raw["outputs"], "WorkerReport.outputs")
            ),
            turns=raw["turns"],
            cost_usd=raw["cost_usd"],
            monitor_severity=raw["monitor_severity"],
            uncertain=raw["uncertain"],
            uncertainty_reasons=_strings_from_wire(
                raw["uncertainty_reasons"], "WorkerReport.uncertainty_reasons"
            ),
            summary=raw["summary"],
            metadata=raw["metadata"],
        )
