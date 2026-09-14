"""Durable, exact-state planning for the central brain.

The planner is the LLM-facing half of central orchestration.  It does not
spawn workers or mutate the integration branch.  Instead it freezes a world
snapshot, records a planning-call intent, asks an injected transport for one
strict proposal, and promotes that proposal to an immutable
:class:`~taste.brains.records.PlanRevision` only after mechanical validation.

Planning operations are named by a caller-supplied idempotency key.  Retrying
an accepted operation reads its durable result without another model call.  A
transport failure leaves the operation pending and records the failure; a
malformed or stale response is terminally rejected and retained verbatim.
Using a new operation key is therefore an explicit request for a new decision,
not an accidental overwrite of an old one.

The central supervisor and this class may share one writable control
:class:`~taste.memstore.Branch` in a single process.  Pass that Branch and the
same mutation lock to both components.  This class never closes the Branch and
never leases the integration branch.  Distinct locks around one shared
writable Branch are unsupported; with separate processes the memstore lease
correctly rejects a second writer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from taste.brains.contract import Contract
from taste.brains.delivery import validate_artifact_path
from taste.brains.planner_transport import (
    PLANNER_TRANSPORT_ROOT,
    LLMPlannerTransport,
    PlannerCompletion,
    PlannerCompletionError,
    PlannerReceiptError,
    PlannerTelemetry,
    PlannerTransportEvidence,
    planner_transport_intent_path,
    planner_transport_outcome_path,
)
from taste.brains.records import (
    Assignment,
    CriteriaRevision,
    Criterion,
    PlanRevision,
    WorkerReport,
    contract_digest,
)
from taste.brains.supervisor import RUN_ROOT, SupervisorRun
from taste.brains.worker_runtime import WORKER_REPORT_PATH
from taste.memstore import Branch, State, Store

__all__ = [
    "BranchObservation",
    "CentralPlanner",
    "Goal",
    "InvalidPlannerOutput",
    "PlannerAttemptAudit",
    "PlannerError",
    "PlannerIdentityConflict",
    "PlannerStateError",
    "PlannerTransport",
    "PlannerTransportError",
    "PlanningRequest",
    "RejectedPlannerOperation",
    "StalePlanningWorld",
    "WorldSnapshot",
]

PLANNER_ROOT = ".taste/planner"
GOAL_SCHEMA = "taste.brains/Goal/1"
WORLD_SCHEMA = "taste.brains/WorldSnapshot/1"
BRANCH_OBSERVATION_SCHEMA = "taste.brains/BranchObservation/1"
OBSERVED_RUN_SCHEMA = "taste.brains/ObservedRun/1"
REQUEST_SCHEMA = "taste.brains/PlanningRequest/1"
PROPOSAL_SCHEMA = "taste.brains/PlannerProposal/1"
ATTEMPT_SCHEMA = "taste.brains/PlanningAttempt/1"
OUTCOME_SCHEMA = "taste.brains/PlanningOutcome/2"
_OUTCOME_SCHEMA_V1 = "taste.brains/PlanningOutcome/1"
RESULT_SCHEMA = "taste.brains/PlanningResult/1"
CURRENT_SCHEMA = "taste.brains/CurrentPlan/1"

_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


class PlannerError(RuntimeError):
    """Base class for central planning failures."""


class PlannerIdentityConflict(PlannerError):
    """A stable goal, operation, or plan identity denotes different bytes."""


class PlannerStateError(PlannerError):
    """Durable control state is malformed or internally inconsistent."""


class InvalidPlannerOutput(PlannerError):
    """The model response is not one valid, mechanically safe proposal."""


class StalePlanningWorld(PlannerError):
    """The observed world moved while the planner was deliberating."""


class RejectedPlannerOperation(PlannerError):
    """A prior attempt for this operation was durably rejected."""


class PlannerTransportError(PlannerError):
    """The injected planning transport failed before returning a response."""


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must include a timezone")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _timestamp(value: Any, where: str) -> str:
    text = _text(value, where)
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError as exc:
        raise ValueError(f"{where} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{where} must include a timezone")
    return text


def _text(value: Any, where: str, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string")
    if "\x00" in value:
        raise ValueError(f"{where} must not contain NUL")
    if not empty and not value.strip():
        raise ValueError(f"{where} must not be empty")
    return value


def _stable_id(value: Any, where: str) -> str:
    text = _text(value, where)
    if text != text.strip() or len(text) > 256 or any(ord(char) < 32 for char in text):
        raise ValueError(f"{where} is not a stable identifier")
    return text


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{where} must be an integer >= {minimum}")
    return value


def _number(value: Any, where: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{where} must be finite and >= {minimum}")
    return result


def _positive_number(value: Any, where: str) -> float:
    result = _number(value, where)
    if result <= 0.0:
        raise ValueError(f"{where} must be finite and positive")
    return result


def _object_id(value: Any, where: str) -> str:
    text = _text(value, where)
    if _OBJECT_ID.fullmatch(text) is None:
        raise ValueError(f"{where} must be a full lowercase object ID")
    return text


def _relative_path(value: Any, where: str) -> str:
    text = _text(value, where)
    path = PurePosixPath(text)
    if (
        "\\" in text
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in text.split("/"))
    ):
        raise ValueError(f"{where} must be a normalized relative path")
    return text


def _digest(value: bytes | str) -> str:
    raw = value.encode("utf-8", "surrogatepass") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _checked_digest(value: Any, where: str) -> str:
    text = _text(value, where)
    if _DIGEST.fullmatch(text) is None:
        raise ValueError(f"{where} must be a sha256 digest")
    return text


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        # ``pending_diff`` is read with surrogateescape so a repository with
        # non-UTF-8 bytes still has a lossless, digestible world snapshot.
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, indent=1, sort_keys=True) + "\n"


def _stored(value: Any) -> str:
    """Exact JSON spelling used by ``Branch.checkpoint(records=...)``."""
    return json.dumps(value, indent=1, sort_keys=True) + "\n"


def _load_json(text: str, where: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{where} contains duplicate key {key!r}")
            result[key] = value
        return result

    def no_constant(value: str) -> Any:
        raise ValueError(f"{where} contains non-JSON number {value}")

    try:
        raw = json.loads(text, object_pairs_hook=no_duplicates, parse_constant=no_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where} is not valid JSON") from exc
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise ValueError(f"{where} must be a JSON object")
    return raw


def _fields(raw: Mapping[str, Any], schema: str, required: set[str]) -> None:
    if raw.get("schema") != schema:
        raise ValueError(f"schema must be {schema!r}")
    missing = required - raw.keys()
    unknown = raw.keys() - required - {"schema"}
    if missing or unknown:
        raise ValueError(
            f"record fields differ; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _freeze(value: Any, where: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                _text(key, f"{where} key"): _freeze(item, f"{where}.{key}")
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, f"{where}[{index}]") for index, item in enumerate(value))
    raise ValueError(f"{where} contains non-JSON value {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{where} must be a JSON object")
    return value


def _array(value: Any, where: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a JSON array")
    return value


def _key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


def _goal_path(goal_id: str) -> str:
    return f"{PLANNER_ROOT}/goals/{_key(goal_id)}.json"


def _operation_root(goal_id: str, operation_id: str) -> str:
    return f"{PLANNER_ROOT}/operations/{_key(goal_id + chr(0) + operation_id)}"


def _criteria_path(goal_id: str) -> str:
    return f"{PLANNER_ROOT}/criteria/{_key(goal_id)}.json"


def _plan_path(plan_id: str) -> str:
    return f"{PLANNER_ROOT}/plans/{_key(plan_id)}.json"


def _current_path(goal_id: str) -> str:
    return f"{PLANNER_ROOT}/current/{_key(goal_id)}.json"


@dataclass(frozen=True, slots=True)
class Goal:
    """The strict, durable objective from which all plan generations descend."""

    goal_id: str
    task: str
    success_criteria: tuple[str, ...]
    budget_usd: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _stable_id(self.goal_id, "goal_id"))
        object.__setattr__(self, "task", _text(self.task, "task"))
        if not isinstance(self.success_criteria, tuple) or not self.success_criteria:
            raise ValueError("success_criteria must be a non-empty tuple")
        criteria = tuple(
            _text(item, f"success_criteria[{index}]")
            for index, item in enumerate(self.success_criteria)
        )
        object.__setattr__(self, "success_criteria", criteria)
        if self.budget_usd is not None:
            object.__setattr__(self, "budget_usd", _number(self.budget_usd, "budget_usd"))
        object.__setattr__(
            self, "metadata", _freeze(_mapping(self.metadata, "metadata"), "metadata")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GOAL_SCHEMA,
            "goal_id": self.goal_id,
            "task": self.task,
            "success_criteria": list(self.success_criteria),
            "budget_usd": self.budget_usd,
            "metadata": _thaw(self.metadata),
        }

    def to_json(self) -> str:
        return _pretty(self.to_dict())

    @classmethod
    def from_dict(cls, value: Any) -> Goal:
        raw = _mapping(value, "Goal")
        _fields(raw, GOAL_SCHEMA, {"goal_id", "task", "success_criteria", "budget_usd", "metadata"})
        return cls(
            goal_id=raw["goal_id"],
            task=raw["task"],
            success_criteria=tuple(_array(raw["success_criteria"], "success_criteria")),
            budget_usd=raw["budget_usd"],
            metadata=raw["metadata"],
        )

    @classmethod
    def from_json(cls, text: str) -> Goal:
        return cls.from_dict(_load_json(text, "Goal"))


@dataclass(frozen=True, slots=True)
class ObservedRun:
    """One exact supervisor run and, when accepted, its exact worker report."""

    record_path: str
    record_blob_id: str
    run: SupervisorRun
    report: WorkerReport | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_path", _relative_path(self.record_path, "record_path"))
        object.__setattr__(
            self, "record_blob_id", _object_id(self.record_blob_id, "record_blob_id")
        )
        if not isinstance(self.run, SupervisorRun):
            raise ValueError("run must be a SupervisorRun")
        if self.report is not None:
            if not isinstance(self.report, WorkerReport):
                raise ValueError("report must be a WorkerReport")
            if (
                self.report.report_id != self.run.report_id
                or self.report.run_id != self.run.run_id
                or self.report.assignment_id != self.run.assignment.assignment_id
                or self.report.generation != self.run.assignment.generation
                or self.report.attempt != self.run.assignment.attempt
            ):
                raise ValueError("report identity does not match observed supervisor run")
        if (self.run.report_id is None) != (self.report is None):
            raise ValueError(
                "accepted report identity and observed report must be present together"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OBSERVED_RUN_SCHEMA,
            "record_path": self.record_path,
            "record_blob_id": self.record_blob_id,
            "run": self.run.to_dict(),
            "report": None if self.report is None else self.report.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> ObservedRun:
        raw = _mapping(value, "ObservedRun")
        _fields(raw, OBSERVED_RUN_SCHEMA, {"record_path", "record_blob_id", "run", "report"})
        return cls(
            record_path=raw["record_path"],
            record_blob_id=raw["record_blob_id"],
            run=SupervisorRun.from_dict(raw["run"]),
            report=None if raw["report"] is None else WorkerReport.from_dict(raw["report"]),
        )


@dataclass(frozen=True, slots=True)
class BranchObservation:
    """One content-bound, lease-free branch observation.

    Immutable state data is pinned to ``head_state_id``.  Work-in-flight data
    (holder, dirty tree, intent, pending turn digest, and inbox cursors) is
    sampled beside it and included in ``observation_digest``.  Rechecking an
    observation therefore catches mutations that do not advance a branch ref,
    including a new verdict note or an uncheckpointed worker edit.
    """

    observation_digest: str
    branch: str
    head_state_id: str
    head_tree_id: str
    live: bool
    holder: Mapping[str, Any] | None
    dirty_paths: tuple[str, ...]
    pending_diff: str
    intent: str | None
    pending_turn_count: int
    pending_turns_digest: str
    inbox_head_id: str | None
    inbox_seen_id: str | None
    inbox: tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]
    conflicts: tuple[Mapping[str, Any], ...]
    verdicts: tuple[Mapping[str, Any], ...]
    sources: tuple[Mapping[str, Any], ...]
    origins: tuple[Mapping[str, Any], ...]
    abstract: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_digest",
            _checked_digest(self.observation_digest, "observation_digest"),
        )
        object.__setattr__(self, "branch", _stable_id(self.branch, "branch"))
        object.__setattr__(self, "head_state_id", _object_id(self.head_state_id, "head_state_id"))
        object.__setattr__(self, "head_tree_id", _object_id(self.head_tree_id, "head_tree_id"))
        if not isinstance(self.live, bool):
            raise ValueError("live must be a boolean")
        if self.holder is not None:
            object.__setattr__(self, "holder", _freeze(_mapping(self.holder, "holder"), "holder"))
        if self.live != (self.holder is not None):
            raise ValueError("live and holder presence disagree")
        if not isinstance(self.dirty_paths, tuple):
            raise ValueError("dirty_paths must be a tuple")
        dirty = tuple(_relative_path(item, "dirty path") for item in self.dirty_paths)
        if dirty != tuple(sorted(set(dirty))):
            raise ValueError("dirty_paths must be sorted and unique")
        object.__setattr__(self, "dirty_paths", dirty)
        object.__setattr__(
            self, "pending_diff", _text(self.pending_diff, "pending_diff", empty=True)
        )
        if self.intent is not None:
            object.__setattr__(self, "intent", _text(self.intent, "intent", empty=True))
        object.__setattr__(
            self,
            "pending_turn_count",
            _integer(self.pending_turn_count, "pending_turn_count"),
        )
        object.__setattr__(
            self,
            "pending_turns_digest",
            _checked_digest(self.pending_turns_digest, "pending_turns_digest"),
        )
        for name in ("inbox_head_id", "inbox_seen_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _object_id(value, name))
        for name in ("inbox", "conflicts", "verdicts", "sources", "origins"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                raise ValueError(f"{name} must be a tuple")
            object.__setattr__(
                self,
                name,
                tuple(
                    _freeze(_mapping(item, f"{name}[{index}]"), f"{name}[{index}]")
                    for index, item in enumerate(value)
                ),
            )
        object.__setattr__(
            self,
            "manifest",
            _freeze(_mapping(self.manifest, "manifest"), "manifest"),
        )
        object.__setattr__(self, "abstract", _text(self.abstract, "abstract"))
        wanted = _digest(_canonical(self._identity_dict()))
        if wanted != self.observation_digest:
            raise ValueError("observation_digest does not match exact branch observation")

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "head_state_id": self.head_state_id,
            "head_tree_id": self.head_tree_id,
            "live": self.live,
            "holder": _thaw(self.holder),
            "dirty_paths": list(self.dirty_paths),
            "pending_diff": self.pending_diff,
            "intent": self.intent,
            "pending_turn_count": self.pending_turn_count,
            "pending_turns_digest": self.pending_turns_digest,
            "inbox_head_id": self.inbox_head_id,
            "inbox_seen_id": self.inbox_seen_id,
            "inbox": [_thaw(item) for item in self.inbox],
            "manifest": _thaw(self.manifest),
            "conflicts": [_thaw(item) for item in self.conflicts],
            "verdicts": [_thaw(item) for item in self.verdicts],
            "sources": [_thaw(item) for item in self.sources],
            "origins": [_thaw(item) for item in self.origins],
            "abstract": self.abstract,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BRANCH_OBSERVATION_SCHEMA,
            "observation_digest": self.observation_digest,
            **self._identity_dict(),
        }

    @classmethod
    def create(cls, **identity: Any) -> BranchObservation:
        return cls(observation_digest=_digest(_canonical(identity)), **identity)

    @classmethod
    def from_dict(cls, value: Any) -> BranchObservation:
        raw = _mapping(value, "BranchObservation")
        required = {
            "observation_digest",
            "branch",
            "head_state_id",
            "head_tree_id",
            "live",
            "holder",
            "dirty_paths",
            "pending_diff",
            "intent",
            "pending_turn_count",
            "pending_turns_digest",
            "inbox_head_id",
            "inbox_seen_id",
            "inbox",
            "manifest",
            "conflicts",
            "verdicts",
            "sources",
            "origins",
            "abstract",
        }
        _fields(raw, BRANCH_OBSERVATION_SCHEMA, required)
        return cls(
            observation_digest=raw["observation_digest"],
            branch=raw["branch"],
            head_state_id=raw["head_state_id"],
            head_tree_id=raw["head_tree_id"],
            live=raw["live"],
            holder=raw["holder"],
            dirty_paths=tuple(_array(raw["dirty_paths"], "dirty_paths")),
            pending_diff=raw["pending_diff"],
            intent=raw["intent"],
            pending_turn_count=raw["pending_turn_count"],
            pending_turns_digest=raw["pending_turns_digest"],
            inbox_head_id=raw["inbox_head_id"],
            inbox_seen_id=raw["inbox_seen_id"],
            inbox=tuple(_array(raw["inbox"], "inbox")),
            manifest=raw["manifest"],
            conflicts=tuple(_array(raw["conflicts"], "conflicts")),
            verdicts=tuple(_array(raw["verdicts"], "verdicts")),
            sources=tuple(_array(raw["sources"], "sources")),
            origins=tuple(_array(raw["origins"], "origins")),
            abstract=raw["abstract"],
        )


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    """The exact branch and durable outcome view given to one planning call."""

    snapshot_id: str
    captured_at: str
    control_branch: str
    control_state_id: str
    integration_branch: str
    integration_state_id: str
    observed_heads: Mapping[str, str]
    observations: tuple[BranchObservation, ...]
    orientation: str
    outcomes: tuple[ObservedRun, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_id", _checked_digest(self.snapshot_id, "snapshot_id"))
        object.__setattr__(self, "captured_at", _timestamp(self.captured_at, "captured_at"))
        object.__setattr__(
            self, "control_branch", _stable_id(self.control_branch, "control_branch")
        )
        object.__setattr__(
            self, "control_state_id", _object_id(self.control_state_id, "control_state_id")
        )
        object.__setattr__(
            self, "integration_branch", _stable_id(self.integration_branch, "integration_branch")
        )
        object.__setattr__(
            self,
            "integration_state_id",
            _object_id(self.integration_state_id, "integration_state_id"),
        )
        heads: dict[str, str] = {}
        for branch, state_id in _mapping(self.observed_heads, "observed_heads").items():
            heads[_stable_id(branch, "observed branch")] = _object_id(
                state_id, f"observed_heads.{branch}"
            )
        if heads.get(self.control_branch) != self.control_state_id:
            raise ValueError("observed_heads does not contain the exact control state")
        if heads.get(self.integration_branch) != self.integration_state_id:
            raise ValueError("observed_heads does not contain the exact integration state")
        object.__setattr__(self, "observed_heads", MappingProxyType(heads))
        if not isinstance(self.observations, tuple) or not all(
            isinstance(item, BranchObservation) for item in self.observations
        ):
            raise ValueError("observations must be a tuple of BranchObservation")
        observed_branches = [item.branch for item in self.observations]
        if observed_branches != sorted(heads) or set(observed_branches) != set(heads):
            raise ValueError("observations must cover every observed branch in sorted order")
        if any(item.head_state_id != heads[item.branch] for item in self.observations):
            raise ValueError("branch observations do not match observed_heads")
        object.__setattr__(self, "orientation", _text(self.orientation, "orientation"))
        expected_orientation = "\n".join(item.abstract for item in self.observations)
        if self.orientation != expected_orientation:
            raise ValueError("orientation does not match branch observation abstracts")
        if not isinstance(self.outcomes, tuple) or not all(
            isinstance(item, ObservedRun) for item in self.outcomes
        ):
            raise ValueError("outcomes must be a tuple of ObservedRun")
        run_ids = [item.run.run_id for item in self.outcomes]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("outcomes contain duplicate run identities")
        wanted = _digest(_canonical(self._identity_dict()))
        if self.snapshot_id != wanted:
            raise ValueError("snapshot_id does not match the exact observed world")

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "captured_at": self.captured_at,
            "control_branch": self.control_branch,
            "control_state_id": self.control_state_id,
            "integration_branch": self.integration_branch,
            "integration_state_id": self.integration_state_id,
            "observed_heads": dict(self.observed_heads),
            "observations": [item.to_dict() for item in self.observations],
            "orientation": self.orientation,
            "outcomes": [item.to_dict() for item in self.outcomes],
        }

    def to_dict(self) -> dict[str, Any]:
        return {"schema": WORLD_SCHEMA, "snapshot_id": self.snapshot_id, **self._identity_dict()}

    @classmethod
    def create(
        cls,
        *,
        captured_at: str,
        control_branch: str,
        control_state_id: str,
        integration_branch: str,
        integration_state_id: str,
        observed_heads: Mapping[str, str],
        observations: tuple[BranchObservation, ...],
        orientation: str,
        outcomes: tuple[ObservedRun, ...],
    ) -> WorldSnapshot:
        identity = {
            "captured_at": captured_at,
            "control_branch": control_branch,
            "control_state_id": control_state_id,
            "integration_branch": integration_branch,
            "integration_state_id": integration_state_id,
            "observed_heads": dict(observed_heads),
            "observations": [item.to_dict() for item in observations],
            "orientation": orientation,
            "outcomes": [item.to_dict() for item in outcomes],
        }
        return cls(
            snapshot_id=_digest(_canonical(identity)),
            captured_at=captured_at,
            control_branch=control_branch,
            control_state_id=control_state_id,
            integration_branch=integration_branch,
            integration_state_id=integration_state_id,
            observed_heads=observed_heads,
            observations=observations,
            orientation=orientation,
            outcomes=outcomes,
        )

    @classmethod
    def from_dict(cls, value: Any) -> WorldSnapshot:
        raw = _mapping(value, "WorldSnapshot")
        _fields(
            raw,
            WORLD_SCHEMA,
            {
                "snapshot_id",
                "captured_at",
                "control_branch",
                "control_state_id",
                "integration_branch",
                "integration_state_id",
                "observed_heads",
                "observations",
                "orientation",
                "outcomes",
            },
        )
        return cls(
            snapshot_id=raw["snapshot_id"],
            captured_at=raw["captured_at"],
            control_branch=raw["control_branch"],
            control_state_id=raw["control_state_id"],
            integration_branch=raw["integration_branch"],
            integration_state_id=raw["integration_state_id"],
            observed_heads=raw["observed_heads"],
            observations=tuple(
                BranchObservation.from_dict(item)
                for item in _array(raw["observations"], "observations")
            ),
            orientation=raw["orientation"],
            outcomes=tuple(
                ObservedRun.from_dict(item) for item in _array(raw["outcomes"], "outcomes")
            ),
        )


@dataclass(frozen=True, slots=True)
class PlanningRequest:
    """Immutable input to an idempotent planning operation."""

    request_id: str
    operation_id: str
    generation: int
    created_at: str
    goal: Goal
    world: WorldSnapshot
    parent_plan: PlanRevision | None = None
    criteria: CriteriaRevision | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _checked_digest(self.request_id, "request_id"))
        object.__setattr__(self, "operation_id", _stable_id(self.operation_id, "operation_id"))
        object.__setattr__(self, "generation", _integer(self.generation, "generation", minimum=1))
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))
        if not isinstance(self.goal, Goal) or not isinstance(self.world, WorldSnapshot):
            raise ValueError("request goal and world records are required")
        if self.parent_plan is None:
            if self.generation != 1:
                raise ValueError("an initial request must be generation 1")
        elif (
            not isinstance(self.parent_plan, PlanRevision)
            or self.parent_plan.goal_id != self.goal.goal_id
            or self.generation != self.parent_plan.generation + 1
        ):
            raise ValueError("request generation does not immediately follow its parent plan")
        wanted = _digest(_canonical(self._identity_dict()))
        if self.request_id != wanted:
            raise ValueError("request_id does not match the exact planning request")

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "generation": self.generation,
            "created_at": self.created_at,
            "goal": self.goal.to_dict(),
            "world": self.world.to_dict(),
            "parent_plan": None if self.parent_plan is None else self.parent_plan.to_dict(),
            "criteria": None if self.criteria is None else self.criteria.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"schema": REQUEST_SCHEMA, "request_id": self.request_id, **self._identity_dict()}

    def to_json(self) -> str:
        return _pretty(self.to_dict())

    @classmethod
    def create(
        cls,
        *,
        operation_id: str,
        generation: int,
        created_at: str,
        goal: Goal,
        world: WorldSnapshot,
        parent_plan: PlanRevision | None,
        criteria: CriteriaRevision | None = None,
    ) -> PlanningRequest:
        identity = {
            "operation_id": operation_id,
            "generation": generation,
            "created_at": created_at,
            "goal": goal.to_dict(),
            "world": world.to_dict(),
            "parent_plan": None if parent_plan is None else parent_plan.to_dict(),
            "criteria": None if criteria is None else criteria.to_dict(),
        }
        return cls(
            request_id=_digest(_canonical(identity)),
            operation_id=operation_id,
            generation=generation,
            created_at=created_at,
            goal=goal,
            world=world,
            parent_plan=parent_plan,
            criteria=criteria,
        )

    @classmethod
    def from_dict(cls, value: Any) -> PlanningRequest:
        raw = _mapping(value, "PlanningRequest")
        _fields(
            raw,
            REQUEST_SCHEMA,
            {
                "request_id",
                "operation_id",
                "generation",
                "created_at",
                "goal",
                "world",
                "parent_plan",
                "criteria",
            },
        )
        return cls(
            request_id=raw["request_id"],
            operation_id=raw["operation_id"],
            generation=raw["generation"],
            created_at=raw["created_at"],
            goal=Goal.from_dict(raw["goal"]),
            world=WorldSnapshot.from_dict(raw["world"]),
            parent_plan=(
                None if raw["parent_plan"] is None else PlanRevision.from_dict(raw["parent_plan"])
            ),
            criteria=(
                None if raw["criteria"] is None else CriteriaRevision.from_dict(raw["criteria"])
            ),
        )

    @classmethod
    def from_json(cls, text: str) -> PlanningRequest:
        return cls.from_dict(_load_json(text, "PlanningRequest"))


@runtime_checkable
class PlannerTransport(Protocol):
    """Injectable LLM boundary; ``request_id`` is its idempotency key."""

    def complete(self, *, request_id: str, system: str, prompt: str) -> str | PlannerCompletion: ...


@dataclass(frozen=True)
class PlannerAttemptAudit:
    """Read-only view of one durable planner intent and its terminal telemetry."""

    goal_id: str
    operation_id: str
    request_id: str
    attempt_id: str
    attempt: int
    status: str
    at: str
    category: str
    response_digest: str | None
    telemetry: PlannerTelemetry


PLANNER_SYSTEM = """You are the central planner for a durable multi-agent system.
Return exactly one JSON object and no markdown. It must use schema
taste.brains/PlannerProposal/1 and echo every request/world binding exactly.
Assignments must use the complete taste.brains/Assignment/1 wire schema.
Every assignment must declare explicit dependencies, immutable ArtifactRef
inputs, and unique ArtifactSpec outputs. Never invent a state, blob, artifact,
failure, or completed result. A completion is only a claim over the supplied
durable world; malformed or stale output will be retained and rejected.
The Assignment.worker field is a physical execution-branch identity. It must
be fresh: never reuse any branch already present in observed_heads. A same-plan
dependency delays launch but does not make its producer's future bytes visible;
work that consumes a newly produced artifact must be reissued in a later plan
revision against the new integration state and an exact ArtifactRef.
"""


class CentralPlanner:
    """Create and recover exact, immutable plan revisions on a control branch."""

    def __init__(
        self,
        store: Store,
        *,
        transport: PlannerTransport,
        control: Branch | None = None,
        control_branch: str = "central-control",
        integration_branch: str = "integration",
        mutation_lock: threading.RLock | None = None,
        clock: Any = _now,
    ) -> None:
        if control is not None and (control.store is not store or control.name != control_branch):
            raise ValueError("injected control branch has the wrong store or identity")
        self.store = store
        self.transport = transport
        self.control = control or store.branch(control_branch, producer="central-planner")
        self.control_branch = control_branch
        self.integration_branch = integration_branch
        self.mutation_lock = mutation_lock or threading.RLock()
        self.clock = clock
        # A successful immutable-history audit is a monotonic checkpoint, not
        # a best-effort memo.  Subsequent calls at the same authoritative head
        # can reuse it exactly; a forward head audits only its new first-parent
        # suffix.  Keeping the last audited head also makes an in-process raw
        # ref rewind or divergence observable instead of accidentally serving
        # an older cached result.
        self._planner_history_audits: dict[tuple[str, str], tuple[str, dict[str, str]]] = {}
        self._planner_paired_evidence_cache: (
            tuple[
                tuple[LLMPlannerTransport, str, str, str, str],
                tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]],
                tuple[PlannerTransportEvidence, ...],
            ]
            | None
        ) = None
        self._bind_llm_transport()

    def _bind_llm_transport(self) -> None:
        transport = self.transport
        if not isinstance(transport, LLMPlannerTransport):
            return
        if (
            transport.store is not self.store
            or transport.control is not self.control
            or transport.journal.store is not self.store
            or transport.journal is self.control
            or transport.journal.name == self.integration_branch
            or transport.mutation_lock is not self.mutation_lock
        ):
            raise ValueError(
                "LLMPlannerTransport must share CentralPlanner's exact Store, "
                "distinct journal Branch, control Branch, and mutation RLock"
            )
        transport.bind_planner()

    def _immutable_record(
        self, path: str, payload: dict[str, Any], records: dict[str, Any]
    ) -> None:
        existing = self.control.head.read(path)
        wanted = _stored(payload)
        if existing is not None and existing != wanted:
            raise PlannerIdentityConflict(f"immutable planner record {path!r} changed identity")
        if existing is None:
            historical = {
                raw for state in self.control.history() if (raw := state.read(path)) is not None
            }
            if historical:
                if historical != {wanted}:
                    raise PlannerIdentityConflict(
                        f"immutable planner record {path!r} had another historical identity"
                    )
                raise PlannerStateError(
                    f"immutable planner record {path!r} disappeared from current control state"
                )
            records[path] = payload

    def _checkpoint(self, reason: str, records: dict[str, Any]) -> State:
        if not records:
            return self.control.head
        return self.control.checkpoint(reason, records=records)

    def _ensure_goal(self, goal: Goal) -> None:
        path = _goal_path(goal.goal_id)
        raw = self.control.head.read(path)
        if raw is not None:
            try:
                recorded = Goal.from_json(raw)
            except ValueError as exc:
                raise PlannerStateError("durable goal record is malformed") from exc
            if recorded != goal:
                raise PlannerIdentityConflict(
                    f"goal id {goal.goal_id!r} is already bound to different objective bytes"
                )
            return
        records: dict[str, Any] = {}
        self._immutable_record(path, goal.to_dict(), records)
        # Sequence 0 is derived verbatim from the goal, so the goal record is
        # never rewritten when criteria are later refined.
        genesis = CriteriaRevision.genesis(goal, at=_iso(self.clock()))
        self._immutable_record(_criteria_path(goal.goal_id), genesis.to_dict(), records)
        self._checkpoint(f"planner goal: {goal.goal_id}", records)

    def _load_plan_path(self, path: str, digest: str | None = None) -> PlanRevision:
        raw = self.control.head.read(path)
        if raw is None:
            raise PlannerStateError(f"referenced plan record {path!r} is missing")
        if digest is not None and _digest(raw) != digest:
            raise PlannerStateError("referenced plan bytes do not match their durable digest")
        try:
            return PlanRevision.from_json(raw)
        except ValueError as exc:
            raise PlannerStateError("durable plan record is malformed") from exc

    def _plan_from_pointer(self, state: State, goal_id: str) -> tuple[str, PlanRevision] | None:
        pointer_path = _current_path(goal_id)
        raw = state.read(pointer_path)
        if raw is None:
            return None
        try:
            pointer = _load_json(raw, "CurrentPlan")
            _fields(
                pointer,
                CURRENT_SCHEMA,
                {"goal_id", "plan_id", "generation", "plan_path", "plan_digest"},
            )
            if pointer["goal_id"] != goal_id:
                raise ValueError("current plan goal identity is wrong")
            _integer(pointer["generation"], "generation", minimum=1)
            _checked_digest(pointer["plan_digest"], "plan_digest")
            plan_raw = state.read(pointer["plan_path"])
            if plan_raw is None or _digest(plan_raw) != pointer["plan_digest"]:
                raise ValueError("referenced plan bytes do not match their durable digest")
            plan = PlanRevision.from_json(plan_raw)
            if (
                plan.plan_id != pointer["plan_id"]
                or plan.goal_id != goal_id
                or plan.generation != pointer["generation"]
                or pointer["plan_path"] != _plan_path(plan.plan_id)
            ):
                raise ValueError("current plan pointer does not match plan record")
            return raw, plan
        except (ValueError, TypeError) as exc:
            raise PlannerStateError("current plan pointer is malformed") from exc

    def _current_plan(self, goal_id: str) -> PlanRevision | None:
        goal_id = _stable_id(goal_id, "goal_id")
        previous_raw: str | None = None
        previous_plan: PlanRevision | None = None
        try:
            history = tuple(reversed(self.control.history()))
            for state in history:
                current = self._plan_from_pointer(state, goal_id)
                if previous_plan is None:
                    if current is None:
                        continue
                    raw, plan = current
                    if plan.generation != 1 or plan.parent_plan_id is not None:
                        raise PlannerStateError("current plan history must begin at generation one")
                    previous_raw, previous_plan = raw, plan
                    continue
                if current is None:
                    raise PlannerStateError("current plan history deleted its pointer")
                raw, plan = current
                if raw == previous_raw:
                    continue
                if (
                    plan.generation != previous_plan.generation + 1
                    or plan.parent_plan_id != previous_plan.plan_id
                    or plan.goal_id != goal_id
                ):
                    raise PlannerStateError(
                        "current plan history broke generation or parent linkage"
                    )
                previous_raw, previous_plan = raw, plan

            terminal = self._plan_from_pointer(self.control.head, goal_id)
            if previous_plan is None:
                if terminal is not None:
                    raise PlannerStateError("current plan terminal pointer is unaccounted")
                return None
            if terminal is None or terminal[0] != previous_raw or terminal[1] != previous_plan:
                raise PlannerStateError("current plan does not equal terminal pointer history")
            return previous_plan
        except PlannerStateError:
            raise
        except Exception as exc:
            raise PlannerStateError("current plan history is unreadable") from exc

    def current_plan(self, goal_id: str) -> PlanRevision | None:
        """Return the promoted plan for ``goal_id`` without invoking the model."""
        with self.mutation_lock:
            return self._current_plan(goal_id)

    def _audit_immutable_history(
        self,
        branch: Branch,
        *,
        authority: str,
        protected: Callable[[str], bool],
    ) -> None:
        """Audit one authority once per head, incrementally after a success.

        The cache is deliberately process-local.  A replacement process first
        proves the entire reachable history, while a live process remembers
        the newest head it proved and rejects any move that is not a strict
        first-parent continuation of that authority.
        """
        key = (branch.ref, authority)
        try:
            head_id = branch.head.id
            cached = self._planner_history_audits.get(key)
            if cached is not None and cached[0] == head_id:
                return

            lineage = branch.store.backend.rev_list_first_parent(head_id)
            if cached is None:
                established: dict[str, str] = {}
                states: list[State] = []
                for state_id in lineage:
                    state = branch.store.state(state_id)
                    if state.meta.branch != branch.name:
                        break
                    states.append(state)
                states.reverse()
            else:
                audited_head, audited_records = cached
                if not branch.store.backend.is_ancestor(audited_head, head_id):
                    raise PlannerStateError(
                        f"{authority} head rewound or diverged from its last successful audit"
                    )
                try:
                    boundary = lineage.index(audited_head)
                except ValueError as exc:
                    raise PlannerStateError(
                        f"{authority} head left the audited first-parent lineage"
                    ) from exc
                established = dict(audited_records)
                states = [branch.store.state(state_id) for state_id in lineage[:boundary]]
                if any(state.meta.branch != branch.name for state in states):
                    raise PlannerStateError(
                        f"{authority} history crossed a foreign branch boundary"
                    )
                states.reverse()

            for state in states:
                present = {
                    path: _digest(raw)
                    for path in state.files()
                    if protected(path) and (raw := state.read(path)) is not None
                }
                missing = established.keys() - present.keys()
                if missing:
                    raise PlannerStateError(
                        f"{authority} deleted immutable record {sorted(missing)[0]!r}"
                    )
                for path, digest in present.items():
                    previous = established.setdefault(path, digest)
                    if previous != digest:
                        raise PlannerStateError(f"{authority} rewrote immutable record {path!r}")

            if branch.head.id != head_id:
                raise PlannerStateError(f"{authority} head moved during its history audit")
            # Publish only after every state and the terminal ref were proved.
            self._planner_history_audits[key] = (head_id, established)
        except PlannerStateError:
            raise
        except Exception as exc:
            raise PlannerStateError(f"{authority} history is unreadable") from exc

    def _assert_planner_attempt_history(self) -> None:
        """Prove cost-bearing operation records were never removed or rewritten."""

        def protected_transport(path: str) -> bool:
            return path.startswith(f"{PLANNER_TRANSPORT_ROOT}/") and path.endswith(
                ("/intent.json", "/outcome.json")
            )

        def protected_control(path: str) -> bool:
            return protected_transport(path) or (
                path.startswith(f"{PLANNER_ROOT}/operations/")
                and path.endswith(("/request.json", "/intent.json", "/outcome.json"))
            )

        self._audit_immutable_history(
            self.control,
            authority="planner control",
            protected=protected_control,
        )
        if isinstance(self.transport, LLMPlannerTransport):
            self._audit_immutable_history(
                self.transport.journal,
                authority="planner transport authority",
                protected=protected_transport,
            )

    def _paired_transport_evidences(self) -> tuple[PlannerTransportEvidence, ...]:
        """Return exact paired receipts without re-auditing unchanged ledgers."""
        transport = self.transport
        if not isinstance(transport, LLMPlannerTransport):
            return ()
        self._assert_planner_attempt_history()

        control_key = (self.control.ref, "planner control")
        journal_key = (transport.journal.ref, "planner transport authority")
        control_audit = self._planner_history_audits.get(control_key)
        journal_audit = self._planner_history_audits.get(journal_key)
        if control_audit is None or journal_audit is None:
            raise PlannerStateError("planner transport history audit cache is incomplete")
        control_head, _control_records = control_audit
        journal_head, _journal_records = journal_audit
        cache_key = (
            transport,
            self.control.ref,
            control_head,
            transport.journal.ref,
            journal_head,
        )
        cached = self._planner_paired_evidence_cache
        if cached is not None and cached[0][0] is transport and cached[0][1:] == cache_key[1:]:
            if self.control.head.id != control_head or transport.journal.head.id != journal_head:
                raise PlannerStateError("planner transport heads moved during receipt pairing")
            return cached[2]

        def transport_records(state: State) -> tuple[tuple[str, str], ...]:
            return tuple(
                sorted(
                    (path, _digest(raw))
                    for path in state.files()
                    if path.startswith(f"{PLANNER_TRANSPORT_ROOT}/")
                    and (raw := state.read(path)) is not None
                )
            )

        fingerprint = (
            transport_records(self.store.state(control_head)),
            transport_records(self.store.state(journal_head)),
        )
        if self.control.head.id != control_head or transport.journal.head.id != journal_head:
            raise PlannerStateError("planner transport heads moved during receipt pairing")
        if (
            cached is not None
            and cached[0][0] is transport
            and cached[0][1] == self.control.ref
            and cached[0][3] == transport.journal.ref
            and cached[1] == fingerprint
        ):
            # Only unrelated commits moved one or both authorities.  Their
            # immutable receipt bytes were incrementally audited above, so the
            # exact pairing result is unchanged.
            self._planner_paired_evidence_cache = (cache_key, fingerprint, cached[2])
            return cached[2]

        evidences = transport._paired_evidences()
        if self.control.head.id != control_head or transport.journal.head.id != journal_head:
            raise PlannerStateError("planner transport heads moved during receipt pairing")
        self._planner_paired_evidence_cache = (cache_key, fingerprint, evidences)
        return evidences

    def planner_attempts(self, goal_id: str) -> tuple[PlannerAttemptAudit, ...]:
        """Audit every durable attempt for one goal without invoking a transport."""
        goal_id = _stable_id(goal_id, "goal_id")
        with self.mutation_lock:
            self._bind_llm_transport()
            authoritative_by_attempt: dict[str, PlannerTransportEvidence] = {}
            if isinstance(self.transport, LLMPlannerTransport):
                try:
                    evidences = self._paired_transport_evidences()
                except PlannerReceiptError as exc:
                    raise PlannerStateError(
                        "planner transport authority or control mirror is corrupt"
                    ) from exc
                for evidence in evidences:
                    if evidence.scope is None:
                        raise PlannerStateError(
                            "planner transport authority contains unscoped evidence"
                        )
                    authoritative_by_attempt[evidence.request_id] = evidence
            else:
                self._assert_planner_attempt_history()
            state = self.control.head
            request_paths = sorted(
                path
                for path in state.files()
                if path.startswith(f"{PLANNER_ROOT}/operations/") and path.endswith("/request.json")
            )
            audits: list[PlannerAttemptAudit] = []
            for request_path in request_paths:
                raw_request = state.read(request_path)
                if raw_request is None:
                    raise PlannerStateError("planner request disappeared during audit")
                try:
                    request = PlanningRequest.from_json(raw_request)
                except ValueError as exc:
                    raise PlannerStateError("durable planning request is malformed") from exc
                if request.goal.goal_id != goal_id:
                    continue
                root = request_path[: -len("/request.json")]
                intent_prefix = f"{root}/attempts/"
                intent_paths = sorted(
                    path
                    for path in state.files()
                    if path.startswith(intent_prefix) and path.endswith("/intent.json")
                )
                for intent_path in intent_paths:
                    token = intent_path[len(intent_prefix) :].split("/", 1)[0]
                    if len(token) != 6 or not token.isdigit():
                        raise PlannerStateError("planning attempt path is malformed")
                    attempt = int(token)
                    raw_intent = state.read(intent_path)
                    if raw_intent is None:
                        raise PlannerStateError("planner attempt intent disappeared")
                    try:
                        intent = _load_json(raw_intent, "PlanningAttempt")
                        _fields(
                            intent,
                            ATTEMPT_SCHEMA,
                            {"request_id", "attempt_id", "attempt", "at"},
                        )
                        expected_id = _digest(f"{request.request_id}\0{attempt}")
                        if (
                            intent["request_id"] != request.request_id
                            or intent["attempt_id"] != expected_id
                            or _integer(intent["attempt"], "attempt", minimum=1) != attempt
                        ):
                            raise ValueError("planning attempt identity is invalid")
                        intent_at = _timestamp(intent["at"], "attempt at")
                    except (TypeError, ValueError) as exc:
                        raise PlannerStateError("durable planning attempt is malformed") from exc

                    outcome_path = f"{intent_prefix}{attempt:06d}/outcome.json"
                    raw_outcome = state.read(outcome_path)
                    authoritative = authoritative_by_attempt.pop(expected_id, None)
                    if authoritative is not None:
                        scope = authoritative.scope
                        assert scope is not None
                        if (
                            scope["goal_id"] != goal_id
                            or scope["operation_id"] != request.operation_id
                            or scope["planning_request_id"] != request.request_id
                            or scope["attempt"] != attempt
                            or not self.transport._scope_is_current(scope)
                        ):
                            raise PlannerStateError(
                                "planner transport evidence is not bound to its central attempt"
                            )
                    if raw_outcome is None:
                        if authoritative is None:
                            pending_category = ""
                            pending_digest = None
                            pending_telemetry = PlannerTelemetry.unknown(source="pending_attempt")
                        else:
                            pending_category = f"transport_{authoritative.status}"
                            pending_digest = (
                                None
                                if authoritative.response is None
                                else _digest(authoritative.response)
                            )
                            pending_telemetry = authoritative.telemetry
                        audits.append(
                            PlannerAttemptAudit(
                                goal_id=goal_id,
                                operation_id=request.operation_id,
                                request_id=request.request_id,
                                attempt_id=expected_id,
                                attempt=attempt,
                                status="pending",
                                at=intent_at,
                                category=pending_category,
                                response_digest=pending_digest,
                                telemetry=pending_telemetry,
                            )
                        )
                        continue
                    try:
                        outcome = _load_json(raw_outcome, "PlanningOutcome")
                        base_fields = {
                            "request_id",
                            "attempt_id",
                            "attempt",
                            "status",
                            "at",
                            "response",
                            "response_digest",
                            "category",
                            "detail",
                        }
                        schema = outcome.get("schema")
                        if schema == OUTCOME_SCHEMA:
                            _fields(outcome, OUTCOME_SCHEMA, base_fields | {"telemetry"})
                            telemetry = PlannerTelemetry.from_dict(outcome["telemetry"])
                        elif schema == _OUTCOME_SCHEMA_V1:
                            _fields(outcome, _OUTCOME_SCHEMA_V1, base_fields)
                            telemetry = PlannerTelemetry.unknown(source="legacy_outcome")
                        else:
                            raise ValueError("planning outcome schema is invalid")
                        if (
                            outcome["request_id"] != request.request_id
                            or outcome["attempt_id"] != expected_id
                            or _integer(outcome["attempt"], "attempt", minimum=1) != attempt
                            or outcome["status"] not in {"accepted", "rejected", "transport_error"}
                        ):
                            raise ValueError("planning outcome identity or status is invalid")
                        response = outcome["response"]
                        response_digest = outcome["response_digest"]
                        if response is None:
                            if response_digest is not None:
                                raise ValueError("absent response has a digest")
                        elif not isinstance(response, str) or response_digest != _digest(response):
                            raise ValueError("planning outcome response digest is invalid")
                        at = _timestamp(outcome["at"], "outcome at")
                        category = _text(outcome["category"], "category", empty=True)
                        _text(outcome["detail"], "detail", empty=True)
                    except (TypeError, ValueError) as exc:
                        raise PlannerStateError("durable planning outcome is malformed") from exc
                    if authoritative is not None and telemetry != authoritative.telemetry:
                        raise PlannerStateError(
                            "planner outcome telemetry disagrees with transport authority"
                        )
                    audits.append(
                        PlannerAttemptAudit(
                            goal_id=goal_id,
                            operation_id=request.operation_id,
                            request_id=request.request_id,
                            attempt_id=expected_id,
                            attempt=attempt,
                            status=outcome["status"],
                            at=at,
                            category=category,
                            response_digest=response_digest,
                            telemetry=telemetry,
                        )
                    )
            for evidence in authoritative_by_attempt.values():
                scope = evidence.scope
                assert scope is not None
                if scope["goal_id"] != goal_id:
                    continue
                audits.append(
                    PlannerAttemptAudit(
                        goal_id=goal_id,
                        operation_id=scope["operation_id"],
                        request_id=scope["planning_request_id"],
                        attempt_id=evidence.request_id,
                        attempt=scope["attempt"],
                        status="orphaned",
                        at=scope["at"],
                        category=f"orphaned_transport_{evidence.status}",
                        response_digest=(
                            None if evidence.response is None else _digest(evidence.response)
                        ),
                        telemetry=evidence.telemetry,
                    )
                )
            return tuple(sorted(audits, key=lambda item: (item.at, item.request_id, item.attempt)))

    def planner_cost(self, goal_id: str, *, currency: str = "work") -> float | None:
        """Sum durable planner cost, or ``None`` if any attempt is unknown."""
        if currency not in {"work", "billed"}:
            raise ValueError("planner cost currency must be 'work' or 'billed'")
        total = 0.0
        for attempt in self.planner_attempts(goal_id):
            telemetry = attempt.telemetry
            if not telemetry.cost_known:
                return None
            amount = telemetry.work_usd if currency == "work" else telemetry.billed_usd
            if amount is None:  # guarded by PlannerTelemetry; fail closed if corrupted in memory
                return None
            total += amount
        return total

    def _authoritative_operation_attempts(
        self,
        goal_id: str,
        operation_id: str,
    ) -> tuple[PlannerTransportEvidence, ...]:
        """Return receipt authority scoped to one logical planning operation."""
        if not isinstance(self.transport, LLMPlannerTransport):
            return ()
        try:
            matches = []
            for evidence in self._paired_transport_evidences():
                scope = evidence.scope
                if scope is None:
                    raise PlannerStateError(
                        "planner transport authority contains unscoped evidence"
                    )
                if scope["goal_id"] == goal_id and scope["operation_id"] == operation_id:
                    matches.append(evidence)
            return tuple(matches)
        except PlannerStateError:
            raise
        except PlannerReceiptError as exc:
            raise PlannerStateError("planner transport authority is corrupt") from exc

    def _observed_runs(self, control_state: State) -> tuple[ObservedRun, ...]:
        paths = sorted(
            path
            for path in control_state.files()
            if path.startswith(f"{RUN_ROOT}/") and path.endswith("/run.json")
        )
        outcomes: list[ObservedRun] = []
        for path in paths:
            raw = control_state.read(path)
            blob = control_state.blob(path)
            if raw is None or blob is None:
                raise PlannerStateError(f"supervisor run {path!r} is not an exact file")
            try:
                run = SupervisorRun.from_json(raw)
            except ValueError as exc:
                raise PlannerStateError(f"supervisor run {path!r} is malformed") from exc
            report: WorkerReport | None = None
            if run.report_id is not None:
                if run.report_state_id is None:
                    raise PlannerStateError("accepted supervisor report has no exact report state")
                report_state = self.store.state(run.report_state_id)
                try:
                    _ = report_state.meta
                except Exception as exc:
                    raise PlannerStateError(
                        "accepted supervisor report state does not exist"
                    ) from exc
                if (
                    report_state.meta.session != self.store.session
                    or report_state.meta.branch != run.assignment.worker
                ):
                    raise PlannerStateError("accepted supervisor report state has wrong identity")
                report_raw = report_state.read(WORKER_REPORT_PATH)
                if report_raw is None:
                    raise PlannerStateError("accepted supervisor report bytes are missing")
                try:
                    report = WorkerReport.from_json(report_raw)
                except ValueError as exc:
                    raise PlannerStateError("accepted supervisor report is malformed") from exc
                if (
                    report.worker != run.assignment.worker
                    or report.contract_digest != run.assignment.contract_digest
                    or report.base_state_id != run.assignment.base_state_id
                ):
                    raise PlannerStateError("accepted supervisor report is not assignment-bound")
                final_state = self.store.state(report.final_state_id)
                try:
                    _ = final_state.meta
                except Exception as exc:
                    raise PlannerStateError("accepted worker final state does not exist") from exc
                if (
                    final_state.meta.branch != run.assignment.worker
                    or not self.store.backend.is_ancestor(final_state.id, report_state.id)
                ):
                    raise PlannerStateError("accepted worker final state has wrong lineage")
            try:
                outcomes.append(
                    ObservedRun(record_path=path, record_blob_id=blob, run=run, report=report)
                )
            except ValueError as exc:
                raise PlannerStateError(f"supervisor outcome {path!r} is inconsistent") from exc
        return tuple(outcomes)

    def _observe_branch(self, branch: str, state: State) -> BranchObservation:
        view = self.store.view(branch)
        holder = view.holder
        dirty_paths = tuple(sorted(set(view.dirty_paths())))
        pending_diff = view.pending_diff()
        pending_turns = view.pending_turns()
        inbox_ref = f"{self.store.INBOX_REF}/{self.store.session}/{branch}"
        seen_ref = f"{self.store.SEEN_REF}/{self.store.session}/{branch}"
        inbox_head = self.store.backend.ref_sha(inbox_ref)
        inbox_seen = self.store.backend.ref_sha(seen_ref)
        inbox = tuple(self.store.inbox(branch))
        manifest = json.loads(state.manifest.to_json())
        conflicts = tuple(conflict.to_dict() for conflict in state.conflicts)
        verdicts = tuple(verdict.to_dict() for verdict in state.verdicts)
        sources = tuple(source.to_dict() for source in state.meta.sources)
        origins: list[dict[str, Any]] = []
        for artifact_id, entry in sorted(state.manifest.entries.items()):
            origin = self.store.origin(state, entry.path)
            origins.append(
                {
                    "artifact_id": artifact_id,
                    "path": entry.path,
                    "blob_id": entry.blob,
                    "origin_state_id": None if origin is None else origin.id,
                    "origin_branch": None if origin is None else origin.meta.branch,
                }
            )
        statuses = ",".join(str(item["status"]) for item in verdicts) or "none"
        abstract = (
            f"{branch}: head={state.id[:12]} "
            f"{'live' if holder is not None else 'idle'}; dirty={len(dirty_paths)}; "
            f"artifacts={len(state.manifest.entries)}; conflicts={len(conflicts)}; "
            f"inbox={len(inbox)}; verdicts={statuses}"
        )
        identity = {
            "branch": branch,
            "head_state_id": state.id,
            "head_tree_id": self.store.backend.tree_of(state.id),
            "live": holder is not None,
            "holder": holder,
            "dirty_paths": dirty_paths,
            "pending_diff": pending_diff,
            "intent": view.intent,
            "pending_turn_count": len(pending_turns),
            "pending_turns_digest": _digest(_canonical(pending_turns)),
            "inbox_head_id": inbox_head,
            "inbox_seen_id": inbox_seen,
            "inbox": inbox,
            "manifest": manifest,
            "conflicts": conflicts,
            "verdicts": verdicts,
            "sources": sources,
            "origins": tuple(origins),
            "abstract": abstract,
        }
        return BranchObservation.create(**identity)

    def _snapshot(self) -> WorldSnapshot:
        heads = {branch: state.id for branch, state in self.store.heads().items()}
        if self.control_branch not in heads:
            raise PlannerStateError("control branch disappeared while taking world snapshot")
        if self.integration_branch not in heads:
            raise PlannerStateError("integration branch must exist before planning")
        control_state = self.store.state(heads[self.control_branch])
        observations = tuple(
            self._observe_branch(branch, self.store.state(heads[branch]))
            for branch in sorted(heads)
        )
        # Heads are independent refs.  Refuse a visibly torn snapshot rather
        # than labeling a mixture of two worlds with one identity.
        if {branch: state.id for branch, state in self.store.heads().items()} != heads:
            raise StalePlanningWorld("branch heads moved while taking world snapshot")
        return WorldSnapshot.create(
            captured_at=_iso(self.clock()),
            control_branch=self.control_branch,
            control_state_id=control_state.id,
            integration_branch=self.integration_branch,
            integration_state_id=heads[self.integration_branch],
            observed_heads=heads,
            observations=observations,
            orientation="\n".join(item.abstract for item in observations),
            outcomes=self._observed_runs(control_state),
        )

    def _world_movement(
        self,
        request: PlanningRequest,
        root: str,
        expected_control: str,
        attempt_id: str,
    ) -> list[str]:
        current = {branch: state.id for branch, state in self.store.heads().items()}
        movement: list[str] = []
        expected = dict(request.world.observed_heads)
        bookkeeping_branches = {self.control_branch}
        if isinstance(self.transport, LLMPlannerTransport):
            bookkeeping_branches.add(self.transport.journal.name)
        for branch in sorted((set(expected) | set(current)) - bookkeeping_branches):
            if expected.get(branch) != current.get(branch):
                movement.append(
                    f"{branch}:{expected.get(branch, 'missing')}->{current.get(branch, 'missing')}"
                )
        expected_observations = {item.branch: item for item in request.world.observations}
        for branch in sorted(expected_observations):
            if branch in bookkeeping_branches - {self.control_branch}:
                # The call itself moves this dedicated immutable receipt
                # branch; its independent history and control mirror are
                # audited by the transport boundary.
                continue
            expected_observation = expected_observations[branch]
            state_id = (
                expected_observation.head_state_id
                if branch == self.control_branch
                else current.get(branch)
            )
            if state_id is None:
                continue
            try:
                observed = self._observe_branch(branch, self.store.state(state_id))
            except Exception as exc:
                movement.append(f"{branch}:observation-failed:{type(exc).__name__}")
                continue
            expected_identity = expected_observation._identity_dict()
            observed_identity = observed._identity_dict()
            # The coordinator necessarily reacquires its own writable branch
            # leases after a process restart.  Their PID/opened-at labels are
            # not a change in the planning world: an overlapping old central
            # process could not have opened the same Branch in the first
            # place because the kernel lease excludes it.  Worker lease
            # identity remains part of freshness.
            for identity in (expected_identity, observed_identity):
                identity.pop("abstract", None)  # derived from the exact fields
                if branch in {self.control_branch, self.integration_branch}:
                    identity.pop("live", None)
                    identity.pop("holder", None)
            if observed_identity != expected_identity:
                movement.append(
                    f"{branch}:observation:{expected_observation.observation_digest}"
                    f"->{observed.observation_digest}"
                )
        control_id = current.get(self.control_branch)
        if control_id is None:
            movement.append(f"{self.control_branch}:missing")
        else:
            state = self.store.state(control_id)
            reached_expected = state.id == expected_control
            reached_snapshot = state.id == request.world.control_state_id
            while not reached_snapshot:
                parents = state.meta.parents
                if not parents:
                    break
                parent = self.store.state(parents[0])
                changed = set(state.diff(parent).paths())
                if not reached_expected:
                    permitted = (
                        state.meta.reason == f"planner transport intent: {attempt_id}"
                        and changed == {planner_transport_intent_path(attempt_id)}
                    ) or (
                        state.meta.reason == f"planner transport outcome: {attempt_id}"
                        and changed == {planner_transport_outcome_path(attempt_id)}
                    )
                else:
                    permitted = state.meta.reason in {
                        f"planner request: {request.request_id}",
                        f"planner attempt: {request.request_id}",
                        f"planner transport failure: {request.request_id}",
                    } and all(path == root or path.startswith(f"{root}/") for path in changed)
                if not permitted:
                    movement.append(f"{self.control_branch}:foreign-state:{state.id}")
                    break
                state = parent
                reached_expected = reached_expected or state.id == expected_control
                reached_snapshot = state.id == request.world.control_state_id
            if not reached_expected and not any("foreign-state" in item for item in movement):
                movement.append(f"{self.control_branch}:attempt-not-in-first-parent-lineage")
            if not reached_snapshot and not any("foreign-state" in item for item in movement):
                movement.append(f"{self.control_branch}:snapshot-not-in-first-parent-lineage")
        return movement

    def _create_request(
        self, goal: Goal, operation_id: str, parent: PlanRevision | None
    ) -> tuple[PlanningRequest, str, str]:
        world = self._snapshot()
        request = PlanningRequest.create(
            operation_id=operation_id,
            generation=1 if parent is None else parent.generation + 1,
            created_at=_iso(self.clock()),
            goal=goal,
            world=world,
            parent_plan=parent,
            criteria=self.criteria(goal.goal_id),
        )
        root = _operation_root(goal.goal_id, operation_id)
        request_path = f"{root}/request.json"
        records: dict[str, Any] = {}
        self._immutable_record(request_path, request.to_dict(), records)
        state = self._checkpoint(f"planner request: {request.request_id}", records)
        return request, root, state.id

    def _load_request(self, goal: Goal, operation_id: str) -> tuple[PlanningRequest, str] | None:
        root = _operation_root(goal.goal_id, operation_id)
        raw = self.control.head.read(f"{root}/request.json")
        if raw is None:
            return None
        try:
            request = PlanningRequest.from_json(raw)
        except ValueError as exc:
            raise PlannerStateError("durable planning request is malformed") from exc
        if request.goal != goal or request.operation_id != operation_id:
            raise PlannerIdentityConflict("planning operation identity was reused")
        return request, root

    def _result(self, request: PlanningRequest, root: str) -> PlanRevision | None:
        raw = self.control.head.read(f"{root}/result.json")
        if raw is None:
            return None
        try:
            result = _load_json(raw, "PlanningResult")
            _fields(
                result,
                RESULT_SCHEMA,
                {"request_id", "status", "category", "detail", "plan_path", "plan_digest"},
            )
            if result["request_id"] != request.request_id:
                raise ValueError("planning result request identity is wrong")
            if result["status"] == "accepted":
                if result["category"] or result["detail"]:
                    raise ValueError("accepted result contains rejection fields")
                plan = self._load_plan_path(result["plan_path"], result["plan_digest"])
                if (
                    result["plan_path"] != _plan_path(plan.plan_id)
                    or plan.goal_id != request.goal.goal_id
                    or plan.generation != request.generation
                    or plan.parent_plan_id
                    != (None if request.parent_plan is None else request.parent_plan.plan_id)
                    or plan.based_on_state_id != request.world.control_state_id
                    or plan.metadata.get("request_id") != request.request_id
                ):
                    raise ValueError("accepted plan is not bound to its exact request")
                return plan
            if (
                result["status"] != "rejected"
                or result["plan_path"] is not None
                or result["plan_digest"] is not None
            ):
                raise ValueError("planning result status is invalid")
            detail = _text(result["detail"], "detail")
            category = _text(result["category"], "category")
        except (ValueError, TypeError) as exc:
            raise PlannerStateError("durable planning result is malformed") from exc
        if category == "stale_world":
            raise StalePlanningWorld(detail)
        raise RejectedPlannerOperation(f"{category}: {detail}")

    def _attempt(self, request: PlanningRequest, root: str) -> tuple[int, str, str]:
        # Attempt files are the cost ledger.  Refuse to infer a retry from the
        # current tree until first-parent history proves that no prior ledger
        # entry was deleted or rewritten.
        self._assert_planner_attempt_history()
        prefix = f"{root}/attempts/"
        files = set(self.control.head.files())
        numbers: set[int] = set()
        for path in files:
            if path.startswith(prefix) and path.endswith("/intent.json"):
                token = path[len(prefix) :].split("/", 1)[0]
                if len(token) != 6 or not token.isdigit():
                    raise PlannerStateError("planning attempt path is malformed")
                numbers.add(int(token))
        if numbers:
            latest = max(numbers)
            outcome_path = f"{prefix}{latest:06d}/outcome.json"
            if outcome_path not in files:
                attempt_id = _digest(f"{request.request_id}\0{latest}")
                intent_path = f"{prefix}{latest:06d}/intent.json"
                introduction: State | None = None
                for state in reversed(self.control.history()):
                    if state.read(intent_path) is not None:
                        introduction = state
                        break
                if introduction is None:
                    raise PlannerStateError("planning attempt intent has no durable origin state")
                return latest, attempt_id, introduction.id
        number = max(numbers, default=0) + 1
        attempt_id = _digest(f"{request.request_id}\0{number}")
        intent_path = f"{prefix}{number:06d}/intent.json"
        intent = {
            "schema": ATTEMPT_SCHEMA,
            "request_id": request.request_id,
            "attempt_id": attempt_id,
            "attempt": number,
            "at": _iso(self.clock()),
        }
        state = self._checkpoint(f"planner attempt: {request.request_id}", {intent_path: intent})
        return number, attempt_id, state.id

    @staticmethod
    def _proposal_template(request: PlanningRequest) -> dict[str, Any]:
        return {
            "schema": PROPOSAL_SCHEMA,
            "request_id": request.request_id,
            "goal_id": request.goal.goal_id,
            "generation": request.generation,
            "parent_plan_id": None if request.parent_plan is None else request.parent_plan.plan_id,
            "based_on_state_id": request.world.control_state_id,
            "observed_heads": dict(request.world.observed_heads),
            "assignments": [],
            "rationale": "",
            "complete": False,
            "completion_reason": "",
            "assessment": [],
            "metadata": {},
        }

    @staticmethod
    def _assignment_exemplar(request: PlanningRequest) -> dict[str, Any]:
        """One filled Assignment, with this request's own exact values in it.

        The first real planner call returned a well-formed proposal that was
        rejected for five missing fields -- ``attempt``, ``contract_digest``,
        ``depends_on``, ``model``, ``resources``. The prompt had told it to use
        "the complete taste.brains/Assignment/1 wire schema" and then shown it
        ``"assignments": []``, so it was guessing a strict schema from its name.
        It guessed six of eleven.

        ``generation`` and ``base_state_id`` are the request's real values so
        the model copies rather than invents them; everything else is shaped.
        """
        return {
            "schema": Assignment.SCHEMA,
            "assignment_id": "<unique id for this assignment>",
            "generation": request.generation,
            "attempt": 0,
            "contract": {
                "identity": "<fresh execution-branch name, not in observed_heads>",
                "task": "<what this worker must do, in its own words>",
                "inputs": [],
                "outputs": ["<path it must produce>"],
                "success_criteria": ["<how the monitor will know it is done>"],
                "issued_by": "central",
                "notes": "",
                "budget_usd": None,
                "max_turns": None,
            },
            "base_state_id": request.world.integration_state_id,
            "depends_on": [],
            "inputs": [],
            "outputs": [
                {
                    "schema": "taste.brains/ArtifactSpec/1",
                    "artifact_id": "<unique id for this artifact>",
                    "path": "<same path as contract.outputs>",
                    "kind": "file",
                    "description": "",
                    "required": True,
                    "disposition": "present",
                    "metadata": {},
                }
            ],
            "model": "claude-sonnet-5",
            "resources": {"wall_timeout_seconds": 600},
            "metadata": {},
        }

    @staticmethod
    def _assignment_schema(request: PlanningRequest) -> dict[str, Any]:
        """The wire contract, stated rather than named.

        ``contract_digest`` is deliberately absent from ``required``: it is a
        sha256 over the canonical contract JSON, which no model can compute by
        hand, so the parser derives it. Every other key must be present --
        ``_check_fields`` rejects both missing and unknown keys.

        ``worker`` is not a field. It is a property equal to
        ``contract.identity``, and naming it as one is how a plan acquires a
        key that parsing then refuses.
        """
        return {
            "type": "object",
            "required": [
                "schema",
                "assignment_id",
                "generation",
                "attempt",
                "contract",
                "base_state_id",
                "depends_on",
                "inputs",
                "outputs",
                "model",
                "resources",
                "metadata",
            ],
            "additionalProperties": False,
            "properties": {
                "schema": {"const": Assignment.SCHEMA},
                "assignment_id": {"type": "string", "minLength": 1},
                "generation": {"const": request.generation},
                "attempt": {"type": "integer", "minimum": 0},
                "contract": {
                    "type": "object",
                    "required": [
                        "identity",
                        "task",
                        "inputs",
                        "outputs",
                        "success_criteria",
                        "issued_by",
                        "notes",
                        "budget_usd",
                        "max_turns",
                    ],
                    "additionalProperties": False,
                },
                "contract_digest": {
                    "type": "string",
                    "description": "omit this; the parser derives it from contract",
                },
                "base_state_id": {"type": "string"},
                "depends_on": {"type": "array", "items": {"type": "string"}},
                "inputs": {
                    "type": "array",
                    "description": (
                        "immutable ArtifactRefs; every branch must appear in "
                        "observed_heads, and bytes that do not exist yet cannot "
                        "be referenced"
                    ),
                },
                "outputs": {"type": "array", "minItems": 1},
                "model": {"type": "string"},
                "resources": {"type": "object"},
                "metadata": {"type": "object"},
            },
        }

    def _prompt(self, request: PlanningRequest) -> str:
        template = self._proposal_template(request)
        template["assignments"] = [self._assignment_exemplar(request)]
        standing = () if request.criteria is None else request.criteria.criteria
        payload = {
            "request": request.to_dict(),
            "standing_criteria": [
                {
                    "criterion_id": item.criterion_id,
                    "text": item.text,
                    "parent_id": item.parent_id,
                }
                for item in standing
            ],
            "required_output_shape": template,
            "assignment_schema": self._assignment_schema(request),
            "rules": {
                # A refinement may only make satisfaction harder, so a parent
                # is never retired.  Declaring completion therefore means
                # accounting for every standing obligation, including the ones
                # a later, sharper criterion was built on top of.
                "criteria_are_append_only": True,
                "complete_requires_every_standing_criterion_met": True,
                "assessment_verdicts": ["met", "not_met"],
                "non_complete_requires_assignments": True,
                "complete_requires_no_assignments": True,
                "assignment_generation": request.generation,
                "assignment_base_state_id": request.world.integration_state_id,
                "fresh_worker_execution_branch": True,
                "same_plan_dependencies_do_not_transfer_future_artifacts": True,
                "unique_output_artifact_ids_and_paths": True,
                "contract_io_must_equal_structured_artifact_paths": True,
                "goal_has_global_budget": request.goal.budget_usd is not None,
                "budgeted_assignment_requires_positive_finite_contract_budget_usd": (
                    request.goal.budget_usd is not None
                ),
                "budgeted_assignment_requires_separate_positive_finite_monitor_budget_usd": (
                    request.goal.budget_usd is not None
                ),
                "monitor_budget_usd_if_present_must_be_positive_finite": True,
            },
        }
        return _pretty(payload)

    @staticmethod
    def _validate_assessment(
        value: Any, request: PlanningRequest, complete: bool
    ) -> tuple[Mapping[str, Any], ...]:
        """Account for the standing obligations, before completion is allowed.

        The planner both proposes revisions to the criteria and is judged
        against them, so ``complete`` cannot be a bare assertion.  A complete
        plan must name every standing criterion and find it met; an active
        plan may report progress without being held to that.
        """
        standing = {} if request.criteria is None else {
            item.criterion_id: item for item in request.criteria.criteria
        }
        items: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        for entry in _array(value, "PlannerProposal.assessment"):
            raw = _mapping(entry, "assessment entry")
            # An assessment entry is an inline sub-object, not a schema'd
            # record, so the strict field check is spelled out here rather
            # than borrowed from ``_fields``, which requires a schema key.
            wanted = {"criterion_id", "verdict", "evidence"}
            if raw.keys() != wanted:
                raise ValueError(
                    "assessment entry fields differ; "
                    f"missing={sorted(wanted - raw.keys())}, unknown={sorted(raw.keys() - wanted)}"
                )
            criterion_id = _text(raw["criterion_id"], "assessment.criterion_id")
            if criterion_id not in standing:
                raise ValueError(
                    f"assessment names criterion {criterion_id!r}, which is not standing"
                )
            if criterion_id in seen:
                raise ValueError(f"criterion {criterion_id!r} is assessed more than once")
            seen.add(criterion_id)
            verdict = _text(raw["verdict"], "assessment.verdict")
            if verdict not in {"met", "not_met"}:
                raise ValueError("assessment verdict must be 'met' or 'not_met'")
            evidence = _text(raw["evidence"], "assessment.evidence")
            items.append({"criterion_id": criterion_id, "verdict": verdict, "evidence": evidence})
        if complete:
            missing = sorted(set(standing) - seen)
            if missing:
                raise ValueError(
                    f"a complete plan must assess every standing criterion; {len(missing)} unassessed"
                )
            unmet = [item["criterion_id"] for item in items if item["verdict"] != "met"]
            if unmet:
                raise ValueError(
                    f"a plan cannot be complete while {len(unmet)} standing criterion(s) are not met"
                )
        return tuple(items)

    @staticmethod
    def _validate_assignment_budget_caps(assignment: Assignment, request: PlanningRequest) -> None:
        budgeted = request.goal.budget_usd is not None
        target_budget = assignment.contract.budget_usd
        monitor_present = "monitor_budget_usd" in assignment.resources

        if budgeted and target_budget is None:
            raise ValueError(
                f"assignment {assignment.assignment_id!r} must declare a positive finite "
                "contract.budget_usd worker cap for a budgeted goal"
            )
        if target_budget is not None and budgeted:
            _positive_number(
                target_budget,
                f"assignment {assignment.assignment_id!r} contract.budget_usd",
            )

        if budgeted and not monitor_present:
            raise ValueError(
                f"assignment {assignment.assignment_id!r} must declare a separate positive "
                "finite resources.monitor_budget_usd cap for a budgeted goal"
            )
        if monitor_present:
            _positive_number(
                assignment.resources["monitor_budget_usd"],
                f"assignment {assignment.assignment_id!r} resources.monitor_budget_usd",
            )

    def _validate_artifact_ref(
        self, assignment: Assignment, index: int, request: PlanningRequest
    ) -> None:
        artifact = assignment.inputs[index]
        source_head = request.world.observed_heads.get(artifact.branch)
        if source_head is None:
            raise InvalidPlannerOutput(
                f"input {artifact.artifact_id!r} comes from an unobserved branch"
            )
        try:
            source = self.store.state(artifact.state_id)
            _ = source.meta
        except Exception as exc:
            raise InvalidPlannerOutput(
                f"input {artifact.artifact_id!r} state does not exist"
            ) from exc
        if source.meta.session != self.store.session or source.meta.branch != artifact.branch:
            raise InvalidPlannerOutput(f"input {artifact.artifact_id!r} has wrong source identity")
        if not self.store.backend.is_ancestor(source.id, source_head):
            raise InvalidPlannerOutput(
                f"input {artifact.artifact_id!r} was not reachable in the observed world"
            )
        entry = self.store.backend.entry_at(source.id, artifact.path)
        if entry is None or entry.mode == "040000" or entry.sha != artifact.blob_id:
            raise InvalidPlannerOutput(
                f"input {artifact.artifact_id!r} does not bind the declared bytes"
            )

    @staticmethod
    def _with_derived_digest(item: Any) -> Any:
        """Fill ``contract_digest`` from the contract, when the model omitted it.

        ``contract_digest`` is a sha256 over the canonical contract JSON, and
        ``Assignment.__post_init__`` recomputes it and rejects a mismatch. That
        check is worth keeping -- it proves the contract was not altered between
        proposal and execution -- but *asking a language model to compute a hash
        by hand* is not. Measured: the first real planner call returned a
        well-formed proposal and was rejected for five missing fields, this one
        among them.

        So a digest the planner omits is derived here, and a digest it supplies
        is left exactly as sent. A wrong digest must still fail: that is the
        tamper signal, and silently overwriting it would turn evidence into a
        shrug.
        """
        if not isinstance(item, Mapping) or "contract_digest" in item:
            return item
        contract = item.get("contract")
        if not isinstance(contract, Mapping):
            return item
        try:
            derived = contract_digest(Contract.from_dict(dict(contract)))
        except Exception:
            # Let Assignment.from_dict report the real problem with the
            # contract, rather than masking it as a digest failure.
            return item
        return {**item, "contract_digest": derived}

    def _parse_proposal(self, response: str, request: PlanningRequest) -> PlanRevision:
        if not isinstance(response, str):
            raise InvalidPlannerOutput("planner transport must return JSON text")
        try:
            raw = _load_json(response, "PlannerProposal")
            required = {
                "request_id",
                "goal_id",
                "generation",
                "parent_plan_id",
                "based_on_state_id",
                "observed_heads",
                "assignments",
                "rationale",
                "complete",
                "completion_reason",
                "assessment",
                "metadata",
            }
            _fields(raw, PROPOSAL_SCHEMA, required)
            expected = self._proposal_template(request)
            for name in (
                "request_id",
                "goal_id",
                "generation",
                "parent_plan_id",
                "based_on_state_id",
                "observed_heads",
            ):
                if raw[name] != expected[name]:
                    raise ValueError(f"proposal {name} does not echo the exact request")
            assignments = tuple(
                Assignment.from_dict(self._with_derived_digest(item))
                for item in _array(raw["assignments"], "PlannerProposal.assignments")
            )
            if not isinstance(raw["complete"], bool):
                raise ValueError("proposal complete must be a boolean")
            if raw["complete"] == bool(assignments):
                raise ValueError(
                    "complete plans must be empty and active plans must have assignments"
                )
            assessment = self._validate_assessment(raw["assessment"], request, raw["complete"])
            for assignment in assignments:
                self._validate_assignment_budget_caps(assignment, request)
            metadata = _mapping(raw["metadata"], "PlannerProposal.metadata")
            proposal_digest = _digest(_canonical(raw))
            plan = PlanRevision(
                plan_id=f"plan.{proposal_digest.removeprefix('sha256:')}",
                generation=request.generation,
                goal_id=request.goal.goal_id,
                based_on_state_id=request.world.control_state_id,
                observed_heads=request.world.observed_heads,
                assignments=assignments,
                created_at=request.created_at,
                parent_plan_id=(
                    None if request.parent_plan is None else request.parent_plan.plan_id
                ),
                rationale=_text(raw["rationale"], "rationale", empty=True),
                complete=raw["complete"],
                completion_reason=_text(raw["completion_reason"], "completion_reason", empty=True),
                assessment=assessment,
                metadata={
                    "operation_id": request.operation_id,
                    "request_id": request.request_id,
                    "snapshot_id": request.world.snapshot_id,
                    "proposal_digest": proposal_digest,
                    "proposal": _thaw(metadata),
                },
            )
        except InvalidPlannerOutput:
            raise
        except (ValueError, TypeError, KeyError) as exc:
            raise InvalidPlannerOutput(str(exc)) from exc

        output_ids: dict[str, str] = {}
        output_paths: dict[str, str] = {}
        input_ids: dict[str, tuple[str, str, str, str]] = {}
        for assignment in plan.assignments:
            if assignment.generation != request.generation:
                raise InvalidPlannerOutput(
                    f"assignment {assignment.assignment_id!r} has a stale generation"
                )
            if assignment.base_state_id != request.world.integration_state_id:
                raise InvalidPlannerOutput(
                    f"assignment {assignment.assignment_id!r} has the wrong integration base"
                )
            if assignment.worker in {self.control_branch, self.integration_branch}:
                raise InvalidPlannerOutput("a worker cannot own a central branch identity")
            if assignment.worker in request.world.observed_heads:
                raise InvalidPlannerOutput(
                    f"worker execution branch {assignment.worker!r} already exists; "
                    "each assignment attempt requires a fresh branch identity"
                )
            if not assignment.outputs:
                raise InvalidPlannerOutput(
                    f"assignment {assignment.assignment_id!r} declares no product artifact"
                )
            if assignment.contract.inputs != tuple(item.path for item in assignment.inputs):
                raise InvalidPlannerOutput(
                    f"assignment {assignment.assignment_id!r} contract inputs are not exact"
                )
            if assignment.contract.outputs != tuple(item.path for item in assignment.outputs):
                raise InvalidPlannerOutput(
                    f"assignment {assignment.assignment_id!r} contract outputs are not exact"
                )
            timeout = assignment.resources.get("wall_timeout_seconds")
            if timeout is not None:
                try:
                    _number(timeout, "wall_timeout_seconds", minimum=0.000001)
                except ValueError as exc:
                    raise InvalidPlannerOutput(str(exc)) from exc
            for index, artifact in enumerate(assignment.inputs):
                try:
                    validate_artifact_path(artifact.path)
                except ValueError as exc:
                    raise InvalidPlannerOutput(str(exc)) from exc
                self._validate_artifact_ref(assignment, index, request)
                identity = (artifact.branch, artifact.state_id, artifact.path, artifact.blob_id)
                previous = input_ids.setdefault(artifact.artifact_id, identity)
                if previous != identity:
                    raise InvalidPlannerOutput(
                        f"input artifact {artifact.artifact_id!r} has two immutable identities"
                    )
            for artifact in assignment.outputs:
                try:
                    validate_artifact_path(artifact.path)
                except ValueError as exc:
                    raise InvalidPlannerOutput(str(exc)) from exc
                prior_owner = output_ids.setdefault(artifact.artifact_id, assignment.assignment_id)
                if prior_owner != assignment.assignment_id:
                    raise InvalidPlannerOutput(
                        f"artifact {artifact.artifact_id!r} has multiple owners"
                    )
                prior_path_owner = output_paths.setdefault(artifact.path, assignment.assignment_id)
                if prior_path_owner != assignment.assignment_id:
                    raise InvalidPlannerOutput(f"path {artifact.path!r} has multiple owners")
        return plan

    def _record_transport_failure(
        self,
        request: PlanningRequest,
        root: str,
        attempt: int,
        attempt_id: str,
        error: Exception,
        telemetry: PlannerTelemetry,
        response: str | None,
    ) -> None:
        outcome = {
            "schema": OUTCOME_SCHEMA,
            "request_id": request.request_id,
            "attempt_id": attempt_id,
            "attempt": attempt,
            "status": "transport_error",
            "at": _iso(self.clock()),
            "response": response,
            "response_digest": None if response is None else _digest(response),
            "category": type(error).__name__,
            "detail": str(error),
            "telemetry": telemetry.to_dict(),
        }
        path = f"{root}/attempts/{attempt:06d}/outcome.json"
        records: dict[str, Any] = {}
        self._immutable_record(path, outcome, records)
        self._checkpoint(f"planner transport failure: {request.request_id}", records)

    def _reject(
        self,
        request: PlanningRequest,
        root: str,
        attempt: int,
        attempt_id: str,
        response: str,
        category: str,
        detail: str,
        telemetry: PlannerTelemetry,
    ) -> None:
        outcome = {
            "schema": OUTCOME_SCHEMA,
            "request_id": request.request_id,
            "attempt_id": attempt_id,
            "attempt": attempt,
            "status": "rejected",
            "at": _iso(self.clock()),
            "response": response,
            "response_digest": _digest(response),
            "category": category,
            "detail": detail,
            "telemetry": telemetry.to_dict(),
        }
        result = {
            "schema": RESULT_SCHEMA,
            "request_id": request.request_id,
            "status": "rejected",
            "category": category,
            "detail": detail,
            "plan_path": None,
            "plan_digest": None,
        }
        records: dict[str, Any] = {}
        self._immutable_record(f"{root}/attempts/{attempt:06d}/outcome.json", outcome, records)
        self._immutable_record(f"{root}/result.json", result, records)
        self._checkpoint(f"planner rejected: {request.request_id}", records)

    def _accept(
        self,
        request: PlanningRequest,
        root: str,
        attempt: int,
        attempt_id: str,
        response: str,
        plan: PlanRevision,
        telemetry: PlannerTelemetry,
    ) -> PlanRevision:
        path = _plan_path(plan.plan_id)
        plan_digest = _digest(_stored(plan.to_dict()))
        outcome = {
            "schema": OUTCOME_SCHEMA,
            "request_id": request.request_id,
            "attempt_id": attempt_id,
            "attempt": attempt,
            "status": "accepted",
            "at": _iso(self.clock()),
            "response": response,
            "response_digest": _digest(response),
            "category": "",
            "detail": "",
            "telemetry": telemetry.to_dict(),
        }
        result = {
            "schema": RESULT_SCHEMA,
            "request_id": request.request_id,
            "status": "accepted",
            "category": "",
            "detail": "",
            "plan_path": path,
            "plan_digest": plan_digest,
        }
        current = {
            "schema": CURRENT_SCHEMA,
            "goal_id": request.goal.goal_id,
            "plan_id": plan.plan_id,
            "generation": plan.generation,
            "plan_path": path,
            "plan_digest": plan_digest,
        }
        current_path = _current_path(request.goal.goal_id)
        existing_current = self._current_plan(request.goal.goal_id)
        if existing_current != request.parent_plan:
            raise StalePlanningWorld("current plan changed before proposal promotion")
        records: dict[str, Any] = {}
        self._immutable_record(path, plan.to_dict(), records)
        self._immutable_record(f"{root}/attempts/{attempt:06d}/outcome.json", outcome, records)
        self._immutable_record(f"{root}/result.json", result, records)
        records[current_path] = current
        self._checkpoint(f"planner accepted: {request.request_id}", records)
        return plan

    def criteria(self, goal_id: str) -> CriteriaRevision | None:
        """The obligations currently standing for this goal."""
        raw = self.control.head.read(_criteria_path(_stable_id(goal_id, "goal_id")))
        if raw is None:
            return None
        try:
            return CriteriaRevision.from_json(raw)
        except ValueError as exc:
            raise PlannerStateError("durable criteria record is malformed") from exc

    def refine_criteria(
        self, goal_id: str, additions: tuple[Criterion, ...], *, reason: str
    ) -> CriteriaRevision:
        """Append obligations.  Nothing already standing can be released.

        A refinement names the criterion it sharpens and that parent stays
        live, so a revision can only ever make satisfaction harder.  There is
        no operation here that retires an obligation, which is what makes a
        regression unrepresentable rather than merely detected.
        """
        goal_id = _stable_id(goal_id, "goal_id")
        with self.mutation_lock:
            current = self.criteria(goal_id)
            if current is None:
                raise PlannerStateError("no goal criteria are recorded for this goal")
            revised = current.extend(additions, reason=reason, at=_iso(self.clock()))
            records: dict[str, Any] = {}
            # The chain is append-only, so a revision overwrites the pointer
            # while every superseded revision stays reachable in history.
            records[_criteria_path(goal_id)] = revised.to_dict()
            self._checkpoint(f"planner criteria: {revised.revision_id}", records)
            return revised

    def plan(self, goal: Goal, *, operation_id: str = "initial") -> PlanRevision:
        """Create the first plan, or replay the already-promoted first plan."""
        if not isinstance(goal, Goal):
            raise ValueError("goal must be a Goal record")
        with self.mutation_lock:
            self._ensure_goal(goal)
            current = self._current_plan(goal.goal_id)
            if current is not None:
                return current
        return self.revise(goal, operation_id=operation_id)

    def revise(self, goal: Goal, *, operation_id: str) -> PlanRevision:
        """Make or recover one named plan revision.

        ``operation_id`` is mandatory so a caller crash after acceptance cannot
        accidentally trigger another LLM decision on retry.
        """
        if not isinstance(goal, Goal):
            raise ValueError("goal must be a Goal record")
        operation_id = _stable_id(operation_id, "operation_id")
        with self.mutation_lock:
            self._bind_llm_transport()
            self._ensure_goal(goal)
            existing = self._load_request(goal, operation_id)
            if existing is None:
                if self._authoritative_operation_attempts(goal.goal_id, operation_id):
                    raise PlannerStateError(
                        "planner receipt authority records this operation, but its exact "
                        "control request is absent"
                    )
                parent = self._current_plan(goal.goal_id)
                request, root, _request_state = self._create_request(goal, operation_id, parent)
            else:
                request, root = existing
                for evidence in self._authoritative_operation_attempts(
                    goal.goal_id,
                    operation_id,
                ):
                    scope = evidence.scope
                    assert scope is not None
                    if scope["planning_request_id"] != request.request_id:
                        raise PlannerStateError(
                            "planner receipt authority disagrees with the operation request"
                        )
            result = self._result(request, root)
            if result is not None:
                return result
            attempt, attempt_id, expected_control = self._attempt(request, root)
            movement = self._world_movement(request, root, expected_control, attempt_id)
            if movement:
                detail = "world changed before planning call: " + "; ".join(movement)
                self._reject(
                    request,
                    root,
                    attempt,
                    attempt_id,
                    "",
                    "stale_world",
                    detail,
                    PlannerTelemetry.synthetic_zero("not_called_zero"),
                )
                raise StalePlanningWorld(detail)
            prompt = self._prompt(request)

        try:
            transport_result = self.transport.complete(
                request_id=attempt_id,
                system=PLANNER_SYSTEM,
                prompt=prompt,
            )
        except Exception as exc:
            telemetry = getattr(exc, "telemetry", None)
            if not isinstance(telemetry, PlannerTelemetry):
                telemetry = PlannerTelemetry.unknown(source="transport_exception")
            error_response = getattr(exc, "response", None)
            if not isinstance(error_response, str):
                error_response = None
            with self.mutation_lock:
                if isinstance(exc, PlannerCompletionError) or getattr(
                    exc, "planner_terminal", False
                ):
                    self._reject(
                        request,
                        root,
                        attempt,
                        attempt_id,
                        error_response or "",
                        "transport_terminal",
                        str(exc),
                        telemetry,
                    )
                else:
                    self._record_transport_failure(
                        request,
                        root,
                        attempt,
                        attempt_id,
                        exc,
                        telemetry,
                        error_response,
                    )
            raise PlannerTransportError(f"{type(exc).__name__}: {exc}") from exc

        invalid_transport = ""
        if isinstance(transport_result, PlannerCompletion):
            response = transport_result.text
            telemetry = transport_result.telemetry
        elif isinstance(transport_result, str):
            response = transport_result
            if request.goal.budget_usd is None:
                telemetry = PlannerTelemetry.synthetic_zero()
            else:
                # A legacy text-only transport has no evidence that the call
                # was free.  Treating it as synthetic $0 would make the global
                # goal ledger an undercount exactly when it must fail closed.
                telemetry = PlannerTelemetry.unknown(
                    source="legacy_string_transport",
                )
                invalid_transport = (
                    "budgeted planner transports must return PlannerCompletion "
                    "with exact cost telemetry"
                )
        else:
            response = repr(transport_result)
            telemetry = PlannerTelemetry.unknown(source="invalid_transport_return")
            invalid_transport = (
                "planner transport must return str or PlannerCompletion, got "
                f"{type(transport_result).__name__}"
            )

        with self.mutation_lock:
            movement = self._world_movement(request, root, expected_control, attempt_id)
            if movement:
                detail = "world changed during planning call: " + "; ".join(movement)
                self._reject(
                    request,
                    root,
                    attempt,
                    attempt_id,
                    response if isinstance(response, str) else repr(response),
                    "stale_world",
                    detail,
                    telemetry,
                )
                raise StalePlanningWorld(detail)
            try:
                if invalid_transport:
                    raise InvalidPlannerOutput(invalid_transport)
                plan = self._parse_proposal(response, request)
            except InvalidPlannerOutput as exc:
                self._reject(
                    request,
                    root,
                    attempt,
                    attempt_id,
                    response,
                    "invalid_output",
                    str(exc),
                    telemetry,
                )
                raise
            return self._accept(
                request,
                root,
                attempt,
                attempt_id,
                response,
                plan,
                telemetry,
            )
