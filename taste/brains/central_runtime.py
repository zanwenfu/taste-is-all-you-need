"""Crash-replayable coordination over the central planner and supervisor.

The planner decides what the current generation means and the supervisor owns
process and delivery mechanics.  This module is the deliberately small state
machine between them.  It never treats an in-memory future, a process result,
or a worker claim as truth: every cycle is recovered from the promoted
``PlanRevision`` and the supervisor records on the shared control branch.

All coordinator actions have an append-only intent/result pair.  The intent is
checkpointed before the mechanical effect and the underlying effect is itself
idempotent.  Consequently a crash after an effect but before its coordinator
result is repaired by observing the planner/supervisor durable state, without
another model decision, process spawn, or product projection.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, Protocol, TypeVar, runtime_checkable

from taste.brains import benchmark_reply
from taste.brains.central_planner import (
    PLANNER_ROOT,
    CentralPlanner,
    Goal,
    InvalidPlannerOutput,
    PlannerTransportError,
    RejectedPlannerOperation,
    StalePlanningWorld,
)
from taste.brains.delivery import DeliveryIdentityConflict, DeliveryRecoveryRequired
from taste.brains.records import Assignment, CriteriaRevision, PlanRevision, WorkerReport
from taste.brains.supervisor import (
    CentralSupervisor,
    DeliveryRejected,
    InvalidWorkerReport,
    SupervisorError,
    SupervisorRun,
    SupervisorStateConflict,
)
from taste.brains.worker_protocol import WORKER_REPORT_PATH
from taste.memstore import Branch, Store

__all__ = [
    "BudgetBlocked",
    "BudgetState",
    "CentralRuntime",
    "CommunicationHook",
    "CoordinatorCorruption",
    "CoordinatorError",
    "CycleOutcome",
    "ExternalSignal",
    "GoalOutcome",
    "RuntimeTrigger",
    "SharedRuntimeStateError",
]

RUNTIME_ROOT = ".taste/central-runtime"
CYCLE_SCHEMA = "taste.brains/CentralCycle/1"
DECISION_SCHEMA = "taste.brains/CentralDecision/1"
DECISION_RESULT_SCHEMA = "taste.brains/CentralDecisionResult/1"
CYCLE_RESULT_SCHEMA = "taste.brains/CentralCycleResult/1"
TRIGGER_SCHEMA = "taste.brains/CentralTrigger/1"
TRIGGER_INDEX_SCHEMA = "taste.brains/CentralTriggerIndex/1"
INDEX_SCHEMA = "taste.brains/CentralCycleIndex/1"
OPERATION_INDEX_SCHEMA = "taste.brains/CentralPlannerOperationIndex/1"
OPERATION_INTENT_SCHEMA = "taste.brains/CentralPlannerOperation/1"
OPERATION_RESULT_SCHEMA = "taste.brains/CentralPlannerOperationResult/1"

_STABLE_ID = re.compile(r"[^\x00-\x1f\x7f]{1,256}\Z")
_TERMINAL_CURRENT_PHASES = frozenset({"terminal", "report_accepted", "delivered", "conflict"})


# Stops after which a benchmark goal still owes the developer a reply. A
# cancelled or crashed run does not: its owner has withdrawn the time for one.
_CLOSING_STOPS = frozenset({"wall_clock", "generation_bound", "budget_blocked", "planner_failed",
                            "spend_cap"})
_CLOSING_MINIMUM_SECONDS = 5.0
# Between cycles that changed nothing: how often the live workers are looked
# at, and the longest the coordinator goes without a full durable cycle.
IDLE_POLL_SECONDS = 0.25
IDLE_HEARTBEAT_SECONDS = 30.0


class CoordinatorError(RuntimeError):
    """Base class for central-runtime failures."""


class SharedRuntimeStateError(CoordinatorError):
    """Planner and supervisor do not share the required owned objects."""


class CoordinatorCorruption(CoordinatorError):
    """Durable coordinator bytes are malformed or changed identity."""


class BudgetBlocked(CoordinatorError):
    """A budgeted planner call is unsafe because durable cost is unprovable."""


class _WallDeadlineReached(CoordinatorError):
    """The goal deadline has elapsed before another external effect."""


class _SpendCapReached(CoordinatorError):
    """The goal's known spending has reached its cap; it takes on no more work."""


class _StopRequested(CoordinatorError):
    """An external caller requested orderly goal cancellation."""


class _InjectedCoordinatorFault(CoordinatorError):
    """Testing/host fault boundary; never reinterpret it as worker failure."""


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, indent=1, sort_keys=True) + "\n"


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


def _key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


def _stable_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or _STABLE_ID.fullmatch(value) is None or value != value.strip():
        raise ValueError(f"{where} must be a stable identifier")
    return value


def _freeze(value: Any, where: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{where} keys must be strings")
        return MappingProxyType(
            {key: _freeze(item, f"{where}.{key}") for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, f"{where}[]") for item in value)
    raise ValueError(f"{where} contains non-JSON value {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _goal_root(goal_id: str) -> str:
    return f"{RUNTIME_ROOT}/goals/{_key(goal_id)}"


def _cycle_root(goal_id: str, sequence: int, cycle_id: str) -> str:
    return f"{_goal_root(goal_id)}/cycles/{sequence:08d}-{_key(cycle_id)}"


def _outcome_path(goal_id: str) -> str:
    return f"{_goal_root(goal_id)}/outcome.json"


def _index_path(goal_id: str) -> str:
    return f"{_goal_root(goal_id)}/cycle-index.json"


def _trigger_path(goal_id: str, plan_id: str, trigger_id: str) -> str:
    return f"{_goal_root(goal_id)}/plans/{_key(plan_id)}/triggers/{_key(trigger_id)}.json"


def _trigger_index_path(goal_id: str, plan_id: str) -> str:
    return f"{_goal_root(goal_id)}/plans/{_key(plan_id)}/trigger-index.json"


def _trigger_artifact_name(plan_id: str, trigger_id: str) -> str:
    return f"central-runtime-trigger.{_key(plan_id)[:16]}.{_key(trigger_id)}"


def _planner_operation_root(goal_id: str, base_id: str) -> str:
    return f"{_goal_root(goal_id)}/planner-operations/{_key(base_id)}"


def _decision_root(cycle_root: str, decision_id: str) -> str:
    return f"{cycle_root}/decisions/{_key(decision_id)}"


@dataclass(frozen=True, slots=True)
class RuntimeTrigger:
    """One exact, content-identified reason to request a new plan."""

    kind: str
    subject_id: str
    detail: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _stable_id(self.kind, "trigger kind"))
        object.__setattr__(self, "subject_id", _stable_id(self.subject_id, "trigger subject_id"))
        if not isinstance(self.detail, str) or "\x00" in self.detail:
            raise ValueError("trigger detail must be text without NUL")
        if not isinstance(self.evidence, Mapping):
            raise ValueError("trigger evidence must be an object")
        object.__setattr__(self, "evidence", _freeze(self.evidence, "trigger evidence"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "subject_id": self.subject_id,
            "detail": self.detail,
            "evidence": _thaw(self.evidence),
        }

    @classmethod
    def from_dict(cls, value: Any) -> RuntimeTrigger:
        if not isinstance(value, dict) or set(value) != {
            "kind",
            "subject_id",
            "detail",
            "evidence",
        }:
            raise ValueError("invalid runtime trigger record")
        return cls(
            kind=value["kind"],
            subject_id=value["subject_id"],
            detail=value["detail"],
            evidence=value["evidence"],
        )

    @property
    def trigger_id(self) -> str:
        return "trigger." + hashlib.sha256(_canonical(self.to_dict()).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ExternalSignal:
    """A communication adapter's already-durable signal to the coordinator.

    The hook, rather than this runtime, owns message acknowledgement semantics.
    Repeated calls must return the same ``signal_id`` and bytes until handled.
    """

    signal_id: str
    kind: str
    detail: str
    requires_replan: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "signal_id", _stable_id(self.signal_id, "signal_id"))
        object.__setattr__(self, "kind", _stable_id(self.kind, "signal kind"))
        if not isinstance(self.detail, str) or "\x00" in self.detail:
            raise ValueError("signal detail must be text without NUL")
        if not isinstance(self.requires_replan, bool):
            raise ValueError("requires_replan must be boolean")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("signal metadata must be an object")
        object.__setattr__(self, "metadata", _freeze(self.metadata, "signal metadata"))

    def to_trigger(self) -> RuntimeTrigger:
        return RuntimeTrigger(
            kind=f"external_{self.kind}",
            subject_id=self.signal_id,
            detail=self.detail,
            evidence=self.metadata,
        )


@runtime_checkable
class CommunicationHook(Protocol):
    """Narrow seam for a typed communication adapter.

    ``signals`` is an observation boundary, not an acknowledgement boundary.
    An adapter backed by :class:`taste.brains.communication.Communicator` must
    first persist acceptance and then replay the resulting signal for the
    entire current generation.  It may be called at least once after a crash
    and therefore must be idempotent.
    """

    def signals(
        self,
        *,
        goal: Goal,
        plan: PlanRevision,
        runs: tuple[SupervisorRun, ...],
    ) -> Sequence[ExternalSignal]: ...


@dataclass(frozen=True, slots=True)
class BudgetState:
    """Conservative billed-cost accounting from exact durable evidence."""

    limit_usd: float | None
    known_spent_usd: float
    reserved_usd: float
    unknown_run_ids: tuple[str, ...] = ()
    unbounded_live_run_ids: tuple[str, ...] = ()
    unknown_planner_attempt_ids: tuple[str, ...] = ()
    worker_spent_usd: float = 0.0
    planner_spent_usd: float = 0.0

    @property
    def enforceable(self) -> bool:
        return not (
            self.unknown_run_ids
            or self.unbounded_live_run_ids
            or self.unknown_planner_attempt_ids
        )

    @property
    def remaining_usd(self) -> float | None:
        if self.limit_usd is None or not self.enforceable:
            return None
        return max(0.0, self.limit_usd - self.known_spent_usd - self.reserved_usd)

    def to_dict(self) -> dict[str, Any]:
        return {
            "limit_usd": self.limit_usd,
            "known_spent_usd": self.known_spent_usd,
            "reserved_usd": self.reserved_usd,
            "remaining_usd": self.remaining_usd,
            "unknown_run_ids": list(self.unknown_run_ids),
            "unbounded_live_run_ids": list(self.unbounded_live_run_ids),
            "unknown_planner_attempt_ids": list(self.unknown_planner_attempt_ids),
            "worker_spent_usd": self.worker_spent_usd,
            "planner_spent_usd": self.planner_spent_usd,
        }


GOAL_OUTCOME_SCHEMA = "taste.brains/GoalOutcome/1"


@dataclass(frozen=True, slots=True)
class GoalOutcome:
    """Why a goal stopped, and what it owes its answer to.

    ``cycle()`` advances one step; nothing said when a goal was over.  This is
    that answer, written once and durably: the obligations that were standing,
    the verdict on each with its evidence, and the exact reason the loop
    ended.  A run that stopped at a bound says so rather than implying success
    by omission.
    """

    goal_id: str
    stop_reason: str
    complete: bool
    generations: int
    cycles: int
    completion_reason: str = ""
    criteria: CriteriaRevision | None = None
    assessment: tuple[Mapping[str, Any], ...] = ()
    delivered_assignment_ids: tuple[str, ...] = ()
    budget: BudgetState = BudgetState(None, 0.0, 0.0)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GOAL_OUTCOME_SCHEMA,
            "goal_id": self.goal_id,
            "stop_reason": self.stop_reason,
            "complete": self.complete,
            "generations": self.generations,
            "cycles": self.cycles,
            "completion_reason": self.completion_reason,
            "criteria": None if self.criteria is None else self.criteria.to_dict(),
            "assessment": [_thaw(dict(item)) for item in self.assessment],
            "delivered_assignment_ids": list(self.delivered_assignment_ids),
            "budget": {
                "limit_usd": self.budget.limit_usd,
                "known_spent_usd": self.budget.known_spent_usd,
                "reserved_usd": self.budget.reserved_usd,
                "unknown_run_ids": list(self.budget.unknown_run_ids),
                "unbounded_live_run_ids": list(self.budget.unbounded_live_run_ids),
                "unknown_planner_attempt_ids": list(self.budget.unknown_planner_attempt_ids),
                "worker_spent_usd": self.budget.worker_spent_usd,
                "planner_spent_usd": self.budget.planner_spent_usd,
            },
            "detail": self.detail,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=1, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, value: Any) -> GoalOutcome:
        raw = dict(value)
        if raw.get("schema") != GOAL_OUTCOME_SCHEMA:
            raise ValueError("GoalOutcome.schema is wrong")
        budget = raw["budget"]
        return cls(
            goal_id=raw["goal_id"],
            stop_reason=raw["stop_reason"],
            complete=raw["complete"],
            generations=raw["generations"],
            cycles=raw["cycles"],
            completion_reason=raw["completion_reason"],
            criteria=(
                None if raw["criteria"] is None else CriteriaRevision.from_dict(raw["criteria"])
            ),
            assessment=tuple(MappingProxyType(dict(item)) for item in raw["assessment"]),
            delivered_assignment_ids=tuple(raw["delivered_assignment_ids"]),
            budget=BudgetState(
                budget["limit_usd"],
                budget["known_spent_usd"],
                budget["reserved_usd"],
                tuple(budget["unknown_run_ids"]),
                tuple(budget["unbounded_live_run_ids"]),
                tuple(budget["unknown_planner_attempt_ids"]),
                budget["worker_spent_usd"],
                budget["planner_spent_usd"],
            ),
            detail=raw["detail"],
        )

    @classmethod
    def from_json(cls, text: str) -> GoalOutcome:
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True, slots=True)
class CycleOutcome:
    """One durable coordinator-cycle result returned to the host."""

    cycle_id: str
    status: str
    plan: PlanRevision
    runs: tuple[SupervisorRun, ...]
    delivered_assignment_ids: tuple[str, ...] = ()
    waiting_assignment_ids: tuple[str, ...] = ()
    failed_assignment_ids: tuple[str, ...] = ()
    triggers: tuple[RuntimeTrigger, ...] = ()
    budget: BudgetState = BudgetState(None, 0.0, 0.0)

    @property
    def complete(self) -> bool:
        return self.status == "complete" and self.plan.complete

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "status": self.status,
            "plan_id": self.plan.plan_id,
            "generation": self.plan.generation,
            "run_ids": [run.run_id for run in self.runs],
            "delivered_assignment_ids": list(self.delivered_assignment_ids),
            "waiting_assignment_ids": list(self.waiting_assignment_ids),
            "failed_assignment_ids": list(self.failed_assignment_ids),
            "trigger_ids": [trigger.trigger_id for trigger in self.triggers],
            "budget": self.budget.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _Cycle:
    cycle_id: str
    sequence: int
    root: str
    plan_id: str | None
    generation: int
    intent: Mapping[str, Any]


T = TypeVar("T")


class CentralRuntime:
    """Reconcile one promoted plan generation at a time.

    ``control``, ``integration`` and ``control_lock`` are externally owned.
    The runtime validates object identity so separately opened writable branch
    leases or look-alike locks cannot accidentally split central truth.
    """

    # Why the last closing reply could not be produced, when it could not.
    closing_failure: str | None = None

    def __init__(
        self,
        store: Store,
        goal: Goal,
        *,
        planner: CentralPlanner,
        supervisor: CentralSupervisor,
        control: Branch,
        integration: Branch,
        control_lock: threading.RLock,
        default_wall_timeout_seconds: float = 900.0,
        communication: CommunicationHook | None = None,
        clock: Callable[[], datetime] = _now,
        fault_injector: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        if not isinstance(goal, Goal):
            raise ValueError("goal must be a Goal")
        if default_wall_timeout_seconds <= 0 or not math.isfinite(default_wall_timeout_seconds):
            raise ValueError("default wall timeout must be finite and positive")
        if control.store is not store or integration.store is not store:
            raise SharedRuntimeStateError("shared branches must belong to this Store")
        if control is integration or control.name == integration.name:
            raise SharedRuntimeStateError("control and integration must be distinct branches")
        if (
            planner.store is not store
            or planner.control is not control
            or planner.control_branch != control.name
            or planner.integration_branch != integration.name
            or planner.mutation_lock is not control_lock
        ):
            raise SharedRuntimeStateError(
                "planner must use the exact shared control, integration identity, and RLock"
            )
        if (
            supervisor.store is not store
            or supervisor.control is not control
            or supervisor.integration is not integration
            or supervisor.control_lock is not control_lock
        ):
            raise SharedRuntimeStateError("supervisor must use the exact shared branches and RLock")
        if communication is not None:
            assert_shared = getattr(communication, "assert_shared", None)
            if assert_shared is not None:
                assert_shared(store=store, control=control, control_lock=control_lock)
        self.store = store
        self.goal = goal
        self.planner = planner
        self.supervisor = supervisor
        self.control = control
        self.integration = integration
        self.control_lock = control_lock
        self.default_wall_timeout_seconds = float(default_wall_timeout_seconds)
        self.communication = communication
        # While its workers work and nothing changes, run() watches them this
        # often and makes a durable cycle only this often. See _await_change.
        self.idle_poll_seconds = IDLE_POLL_SECONDS
        self.idle_heartbeat_seconds = IDLE_HEARTBEAT_SECONDS
        self._plan_minimum = 0.0
        self.clock = clock
        self.fault_injector = fault_injector
        self._lock = threading.RLock()
        self._remaining_wall: Callable[[], float] | None = None
        self._deadline_at: datetime | None = None
        self._running = False
        self._stop_signal = threading.Event()
        self._stop_signal_lock = threading.Lock()
        self._requested_stop_detail: str | None = None
        self._cycle_audit_head: str | None = None
        self._cycle_audit_entries: list[Any] = []

    def _fault(self, boundary: str, payload: Mapping[str, Any]) -> None:
        if self.fault_injector is not None:
            try:
                self.fault_injector(boundary, payload)
            except Exception as exc:
                raise _InjectedCoordinatorFault(str(exc)) from exc

    def _immutable(self, path: str, value: Mapping[str, Any], reason: str) -> None:
        wanted = _pretty(value)
        with self.control_lock:
            existing = self.control.head.read(path)
            if existing is not None:
                if existing != wanted:
                    raise CoordinatorCorruption(
                        f"durable coordinator record {path!r} changed identity"
                    )
                return
            # Ask git for commits which touched this exact path.  Walking every
            # central state and demand-paging the absent path is quadratic as
            # the append-only ledger grows.
            changed = self.store.backend.repo.git.rev_list(
                "--first-parent", self.control.head.id, "--", path
            )
            historical = {
                raw
                for commit in changed.splitlines()
                if (raw := self.store.backend.show(commit, path)) is not None
            }
            if historical:
                if historical != {wanted}:
                    raise CoordinatorCorruption(
                        f"durable coordinator record {path!r} had another identity"
                    )
                raise CoordinatorCorruption(
                    f"durable coordinator record {path!r} disappeared from current state"
                )
            self.control.checkpoint(reason, records={path: dict(value)})

    def _cycle_records(self) -> list[tuple[int, str, Mapping[str, Any], str]]:
        prefix = f"{_goal_root(self.goal.goal_id)}/cycles/"
        index_path = _index_path(self.goal.goal_id)
        with self.control_lock:
            head = self.control.head
            index = head.record(index_path)
            # Reuse only history already audited on this exact first-parent
            # chain. Ordinary ancestry is insufficient: a merge can contain
            # the audited head only through its second parent.
            revision = head.id
            if self._cycle_audit_head is not None:
                revision = f"{self._cycle_audit_head}..{head.id}"
                if head.id != self._cycle_audit_head:
                    appended = self.store.backend.repo.git.rev_list("--first-parent", revision).splitlines()
                    parents = self.store.backend.repo.commit(appended[-1]).parents if appended else ()
                    if not parents or parents[0].hexsha != self._cycle_audit_head:
                        raise CoordinatorCorruption(
                            "cycle index control history is not a first-parent extension of the audited head"
                        )
            changed = self.store.backend.repo.git.rev_list(
                "--first-parent", revision, "--", index_path
            )
            historical_indexes: list[dict[str, Any]] = []
            for commit in changed.splitlines():
                text = self.store.backend.show(commit, index_path)
                if text is None:
                    raise CoordinatorCorruption("durable cycle index was deleted from its history")
                try:
                    item = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise CoordinatorCorruption("cycle index history is malformed") from exc
                if not isinstance(item, dict):
                    raise CoordinatorCorruption("cycle index history is malformed")
                historical_indexes.append(item)

            if index is None:
                if historical_indexes:
                    raise CoordinatorCorruption("durable cycle index disappeared from current state")
                expected_entries: list[Any] = []
            else:
                if (
                    not isinstance(index, dict)
                    or set(index) != {"schema", "goal_id", "cycles"}
                    or index.get("schema") != INDEX_SCHEMA
                    or index.get("goal_id") != self.goal.goal_id
                    or not isinstance(index.get("cycles"), list)
                ):
                    raise CoordinatorCorruption("durable cycle index is malformed")
                expected_entries = index["cycles"]
                newer = expected_entries
                for historical in historical_indexes:
                    old = historical.get("cycles")
                    if (
                        set(historical) != {"schema", "goal_id", "cycles"}
                        or historical.get("schema") != INDEX_SCHEMA
                        or historical.get("goal_id") != self.goal.goal_id
                        or not isinstance(old, list)
                        or newer[: len(old)] != old
                    ):
                        raise CoordinatorCorruption(
                            "durable cycle index was rolled back or rewritten"
                        )
                    newer = old

            # A transient shrink followed by restoration is also a rollback.
            # Anchor the oldest newly inspected prefix to our previous audit.
            oldest = historical_indexes[-1]["cycles"] if historical_indexes else expected_entries
            if oldest[: len(self._cycle_audit_entries)] != self._cycle_audit_entries:
                raise CoordinatorCorruption("durable cycle index was rolled back or rewritten")

            indexed_paths: set[str] = set()
            records: list[tuple[int, str, Mapping[str, Any], str]] = []
            for position, entry in enumerate(expected_entries, 1):
                if not isinstance(entry, dict) or set(entry) != {
                    "sequence",
                    "cycle_id",
                    "root",
                }:
                    raise CoordinatorCorruption("durable cycle index entry is malformed")
                sequence = entry["sequence"]
                cycle_id = entry["cycle_id"]
                root = entry["root"]
                if (
                    isinstance(sequence, bool)
                    or not isinstance(sequence, int)
                    or sequence != position
                    or not isinstance(cycle_id, str)
                    or not isinstance(root, str)
                    or root != _cycle_root(self.goal.goal_id, sequence, cycle_id)
                ):
                    raise CoordinatorCorruption("durable cycle index entry has wrong identity")
                path = f"{root}/intent.json"
                raw = head.record(path)
                if not isinstance(raw, dict) or raw.get("schema") != CYCLE_SCHEMA:
                    raise CoordinatorCorruption(
                        f"indexed cycle intent {path!r} is missing or malformed"
                    )
                if raw.get("sequence") != sequence or raw.get("cycle_id") != cycle_id:
                    raise CoordinatorCorruption(f"cycle intent {path!r} has the wrong identity")
                indexed_paths.add(path)
                records.append((sequence, cycle_id, raw, root))

            actual_paths = {
                path
                for path in head.files()
                if path.startswith(prefix)
                and path.endswith("/intent.json")
                and path[len(prefix) :].count("/") == 1
            }
            if actual_paths != indexed_paths:
                raise CoordinatorCorruption("cycle intents and durable cycle index disagree")
            # Publish cache state only after every index/intent check passed.
            # JSON copying prevents caller-owned record mutations reaching it.
            self._cycle_audit_entries = json.loads(json.dumps(expected_entries))
            self._cycle_audit_head = head.id
        return records

    def _append_cycle_intent(
        self,
        records: Sequence[tuple[int, str, Mapping[str, Any], str]],
        root: str,
        intent: Mapping[str, Any],
    ) -> None:
        entries = [
            {"sequence": item[0], "cycle_id": item[1], "root": item[3]}
            for item in records
        ]
        entries.append(
            {
                "sequence": intent["sequence"],
                "cycle_id": intent["cycle_id"],
                "root": root,
            }
        )
        index = {
            "schema": INDEX_SCHEMA,
            "goal_id": self.goal.goal_id,
            "cycles": entries,
        }
        prior_index = (
            None
            if not records
            else {
                "schema": INDEX_SCHEMA,
                "goal_id": self.goal.goal_id,
                "cycles": entries[:-1],
            }
        )
        with self.control_lock:
            if self.control.head.record(_index_path(self.goal.goal_id)) != prior_index:
                raise CoordinatorCorruption("cycle index moved during cycle creation")
            if self.control.head.read(f"{root}/intent.json") is not None:
                raise CoordinatorCorruption("new cycle identity already exists")
            self.control.checkpoint(
                f"central cycle intent: {intent['cycle_id']}",
                records={
                    f"{root}/intent.json": dict(intent),
                    _index_path(self.goal.goal_id): index,
                },
            )

    def _finish_raw(self, cycle: _Cycle, status: str, payload: Mapping[str, Any]) -> None:
        record = {
            "schema": CYCLE_RESULT_SCHEMA,
            "cycle_id": cycle.cycle_id,
            "goal_id": self.goal.goal_id,
            "plan_id": cycle.plan_id,
            "generation": cycle.generation,
            "status": status,
            "at": _iso(self.clock()),
            "payload": _thaw(payload),
        }
        self._immutable(
            f"{cycle.root}/result.json",
            record,
            f"central cycle {status}: {cycle.cycle_id}",
        )
        self._fault("cycle_result", record)

    def _begin_cycle(self, plan: PlanRevision | None) -> _Cycle:
        records = self._cycle_records()
        if records:
            sequence, cycle_id, intent, root = records[-1]
            with self.control_lock:
                complete = self.control.head.read(f"{root}/result.json") is not None
            if not complete:
                same_plan = intent.get("plan_id") == (None if plan is None else plan.plan_id)
                if same_plan:
                    return _Cycle(
                        cycle_id=cycle_id,
                        sequence=sequence,
                        root=root,
                        plan_id=intent.get("plan_id"),
                        generation=intent.get("generation", 0),
                        intent=intent,
                    )
                stale = _Cycle(
                    cycle_id=cycle_id,
                    sequence=sequence,
                    root=root,
                    plan_id=intent.get("plan_id"),
                    generation=intent.get("generation", 0),
                    intent=intent,
                )
                self._finish_raw(
                    stale,
                    "superseded",
                    {
                        "promoted_plan_id": None if plan is None else plan.plan_id,
                        "promoted_generation": 0 if plan is None else plan.generation,
                    },
                )
        sequence = (records[-1][0] if records else 0) + 1
        plan_id = None if plan is None else plan.plan_id
        generation = 0 if plan is None else plan.generation
        identity = {
            "goal_id": self.goal.goal_id,
            "sequence": sequence,
            "plan_id": plan_id,
            "generation": generation,
        }
        cycle_id = "cycle." + hashlib.sha256(_canonical(identity).encode()).hexdigest()
        root = _cycle_root(self.goal.goal_id, sequence, cycle_id)
        intent = {
            "schema": CYCLE_SCHEMA,
            "cycle_id": cycle_id,
            **identity,
            "at": _iso(self.clock()),
        }
        self._append_cycle_intent(records, root, intent)
        self._fault("cycle_intent", intent)
        return _Cycle(cycle_id, sequence, root, plan_id, generation, intent)

    def _decision_intent(
        self,
        cycle: _Cycle,
        kind: str,
        subject_id: str,
        payload: Mapping[str, Any],
    ) -> tuple[str, str, Mapping[str, Any]]:
        base_identity = {
            "cycle_id": cycle.cycle_id,
            "kind": _stable_id(kind, "decision kind"),
            "subject_id": _stable_id(subject_id, "decision subject_id"),
            "payload": _thaw(payload),
        }
        # An effect failure is evidence, not a permanent poison pill.  Reuse
        # an unfinished attempt after a crash, but allocate a new append-only
        # attempt after any completed observation.
        attempt = 1
        prefix = f"{cycle.root}/decisions/"
        with self.control_lock:
            head = self.control.head
            matching: list[tuple[int, str]] = []
            for path in head.files():
                if not path.startswith(prefix) or not path.endswith("/intent.json"):
                    continue
                raw = head.record(path)
                if not isinstance(raw, dict):
                    raise CoordinatorCorruption(f"decision intent {path!r} is malformed")
                compared = {
                    "cycle_id": raw.get("cycle_id"),
                    "kind": raw.get("kind"),
                    "subject_id": raw.get("subject_id"),
                    "payload": raw.get("payload"),
                }
                if compared == base_identity:
                    number = raw.get("attempt")
                    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
                        raise CoordinatorCorruption(f"decision intent {path!r} has bad attempt")
                    matching.append((number, path.rsplit("/", 1)[0]))
            if matching:
                attempt, latest_root = max(matching)
                result_raw = head.record(f"{latest_root}/result.json")
                # A completed attempt is historical evidence.  A replay may
                # validly observe a newer process phase even when its input
                # ledger sequence had not changed yet (the OS can exit between
                # polls), so never overwrite or compare against that result.
                if isinstance(result_raw, dict):
                    attempt += 1
        identity = {**base_identity, "attempt": attempt}
        decision_id = "decision." + hashlib.sha256(_canonical(identity).encode()).hexdigest()
        root = _decision_root(cycle.root, decision_id)
        record = {
            "schema": DECISION_SCHEMA,
            "decision_id": decision_id,
            "goal_id": self.goal.goal_id,
            "plan_id": cycle.plan_id,
            "generation": cycle.generation,
            "at": _iso(self.clock()),
            **identity,
        }
        path = f"{root}/intent.json"
        with self.control_lock:
            existing = self.control.head.record(path)
        if existing is not None:
            semantic = dict(record)
            semantic.pop("at")
            observed = dict(existing) if isinstance(existing, dict) else {}
            observed.pop("at", None)
            if observed != semantic:
                raise CoordinatorCorruption(
                    f"decision {decision_id!r} has two intent identities"
                )
            return decision_id, root, existing
        self._immutable(
            path,
            record,
            f"central decision intent: {kind} {subject_id}",
        )
        self._fault(f"decision_intent:{kind}", record)
        return decision_id, root, record

    def _decision_result(
        self,
        cycle: _Cycle,
        decision_id: str,
        root: str,
        kind: str,
        subject_id: str,
        status: str,
        payload: Mapping[str, Any],
    ) -> None:
        record = {
            "schema": DECISION_RESULT_SCHEMA,
            "decision_id": decision_id,
            "cycle_id": cycle.cycle_id,
            "goal_id": self.goal.goal_id,
            "plan_id": cycle.plan_id,
            "generation": cycle.generation,
            "kind": kind,
            "subject_id": subject_id,
            "status": status,
            "at": _iso(self.clock()),
            "payload": _thaw(payload),
        }
        path = f"{root}/result.json"
        with self.control_lock:
            existing = self.control.head.record(path)
        if existing is not None:
            semantic = dict(record)
            semantic.pop("at")
            observed = dict(existing) if isinstance(existing, dict) else {}
            observed.pop("at", None)
            if observed != semantic:
                raise CoordinatorCorruption(
                    f"decision {decision_id!r} produced two different results"
                )
            return
        self._immutable(
            path,
            record,
            f"central decision {status}: {kind} {subject_id}",
        )
        self._fault(f"decision_result:{kind}", record)

    def _perform(
        self,
        cycle: _Cycle,
        kind: str,
        subject_id: str,
        intent: Mapping[str, Any],
        effect: Callable[[], T],
        result: Callable[[T], Mapping[str, Any]],
    ) -> T:
        self._raise_if_stop_requested()
        decision_id, root, record = self._decision_intent(cycle, kind, subject_id, intent)
        self._raise_if_stop_requested()
        try:
            value = effect()
        except Exception as exc:
            self._decision_result(
                cycle,
                decision_id,
                root,
                kind,
                subject_id,
                "error",
                {"error_type": type(exc).__name__, "detail": str(exc)[:1024]},
            )
            raise
        self._fault(f"decision_effect:{kind}", record)
        self._decision_result(cycle, decision_id, root, kind, subject_id, "applied", result(value))
        return value

    @staticmethod
    def _run_exact(run: SupervisorRun, assignment: Assignment, generation: int) -> bool:
        return run.assignment.generation == generation and run.assignment == assignment

    def _exact_current_runs(
        self, plan: PlanRevision, runs: Sequence[SupervisorRun]
    ) -> dict[str, SupervisorRun]:
        assignments = {item.assignment_id: item for item in plan.assignments}
        owners = self._assignment_owners()
        exact: dict[str, SupervisorRun] = {}
        for run in runs:
            assignment = assignments.get(run.assignment.assignment_id)
            if assignment is None or not self._run_exact(run, assignment, plan.generation):
                continue
            run_owners = owners.get(_digest(run.assignment.to_json()))
            if run_owners != frozenset({self.goal.goal_id}):
                raise CoordinatorCorruption(
                    "supervisor run is not uniquely attributable to this goal"
                )
            if assignment.assignment_id in exact:
                raise CoordinatorCorruption(
                    f"current assignment {assignment.assignment_id!r} has two exact runs"
                )
            exact[assignment.assignment_id] = run
        return exact

    def _load_report_for_accounting(self, run: SupervisorRun) -> WorkerReport | None:
        state = None
        if run.report_state_id is not None:
            try:
                state = self.store.state(run.report_state_id)
            except Exception:
                return None
        elif run.terminal:
            view = self.store.view(run.assignment.worker)
            if not view.exists():
                return None
            state = view.head
        if state is None:
            return None
        raw = state.read(WORKER_REPORT_PATH)
        if raw is None:
            return None
        try:
            report = WorkerReport.from_json(raw)
        except Exception:
            return None
        if not (
            report.run_id == run.run_id
            and report.assignment_id == run.assignment.assignment_id
            and report.generation == run.assignment.generation
            and report.attempt == run.assignment.attempt
            and report.worker == run.assignment.worker
            and report.contract_digest == run.assignment.contract_digest
            and report.base_state_id == run.assignment.base_state_id
        ):
            return None
        return report

    def _assignment_owners(self) -> Mapping[str, frozenset[str]]:
        """Map exact promoted assignment bytes to their durable goal owners."""
        prefix = f"{PLANNER_ROOT}/plans/"
        owners: dict[str, set[str]] = {}
        with self.control_lock:
            head = self.control.head
            for path in head.files():
                if not path.startswith(prefix) or not path.endswith(".json"):
                    continue
                raw = head.read(path)
                try:
                    plan = PlanRevision.from_json(raw or "")
                except ValueError as exc:
                    raise CoordinatorCorruption(
                        f"promoted plan record {path!r} is malformed"
                    ) from exc
                for assignment in plan.assignments:
                    owners.setdefault(_digest(assignment.to_json()), set()).add(plan.goal_id)
        return {key: frozenset(value) for key, value in owners.items()}

    def _validate_supervisor_scope(self, runs: Sequence[SupervisorRun]) -> None:
        """Fail closed until supervisor records carry first-class goal IDs."""
        owners = self._assignment_owners()
        for run in runs:
            run_owners = owners.get(_digest(run.assignment.to_json()))
            if run_owners != frozenset({self.goal.goal_id}):
                raise SharedRuntimeStateError(
                    "shared supervisor contains a run not uniquely owned by this goal"
                )

    def _budget(self, runs: Sequence[SupervisorRun]) -> BudgetState:
        worker_costs: list[float] = []
        planner_costs: list[float] = []
        reservations: list[float] = []
        unknown: list[str] = []
        unknown_planner: list[str] = []
        unbounded: list[str] = []

        planner_ceiling = self._planner_call_ceiling()
        for attempt in self.planner.planner_attempts(self.goal.goal_id):
            telemetry = attempt.telemetry
            if not telemetry.cost_known or telemetry.billed_usd is None:
                if planner_ceiling is None:
                    unknown_planner.append(attempt.attempt_id)
                else:
                    reservations.append(planner_ceiling)
            else:
                planner_costs.append(telemetry.billed_usd)

        owners = self._assignment_owners()
        for run in runs:
            run_owners = owners.get(_digest(run.assignment.to_json()))
            if run_owners is not None and len(run_owners) > 1:
                raise CoordinatorCorruption(
                    "one exact supervisor assignment is ambiguously owned by multiple goals"
                )
            if run_owners is not None and self.goal.goal_id not in run_owners:
                continue
            if run_owners is None:
                # An unowned supervisor run is corrupt or predates exact goal
                # attribution.  For a budgeted goal it is unsafe to guess that
                # its cost belongs elsewhere.
                if self.goal.budget_usd is not None:
                    unknown.append(run.run_id)
                continue
            report = self._load_report_for_accounting(run)
            if report is not None:
                bounds = None if report.cost_usd is not None else self._reported_cost_bounds(run, report)
                if report.cost_usd is not None:
                    worker_costs.append(report.cost_usd)
                elif bounds is None:
                    self._charge_lost_cost(run, reservations, unknown)
                else:
                    worker_costs.append(bounds[0])
                    reservations.append(bounds[1])
                continue
            if run.phase in _TERMINAL_CURRENT_PHASES:
                # A process may have spent money even if its exact report is
                # missing or corrupt.  Treating that as $0 would violate the
                # global goal budget.
                if run.pid is not None or run.deadline_at is not None:
                    self._charge_lost_cost(run, reservations, unknown)
                continue
            ceiling = self._assignment_cost_ceiling(run.assignment)
            if ceiling is None:
                unbounded.append(run.run_id)
            else:
                reservations.append(ceiling)

        try:
            worker_spent = math.fsum(worker_costs)
            planner_spent = math.fsum(planner_costs)
            spent = math.fsum((worker_spent, planner_spent))
            reserved = math.fsum(reservations)
        except OverflowError as exc:
            raise CoordinatorCorruption("durable cost accounting overflowed") from exc
        if not all(math.isfinite(value) for value in (worker_spent, planner_spent, spent, reserved)):
            raise CoordinatorCorruption("durable cost accounting is not finite")
        return BudgetState(
            self.goal.budget_usd,
            spent,
            reserved,
            tuple(sorted(set(unknown))),
            tuple(sorted(set(unbounded))),
            tuple(sorted(set(unknown_planner))),
            worker_spent,
            planner_spent,
        )

    def _planner_call_ceiling(self) -> float | None:
        """The most one planner call can have cost, where that can be proved.

        A planner call whose reply was lost has an unknown cost. Calling the
        whole budget unprovable for it ended the goal: no further plan and no
        closing reply. An Azure planner call is admitted against a fixed worst
        case, one attempt and no resend, so it cannot have cost more than
        that. It is charged at that ceiling, as a lost worker is charged at
        its caps; the unknown stays visible as that reservation. Other
        transports keep the original rule.
        """
        if "azure_execution" not in self.goal.metadata:
            return None
        ceiling = getattr(self.planner.transport, "max_billed_call_usd", None)
        try:
            value = float(ceiling()) if callable(ceiling) else None
        except Exception:
            return None
        return value if value is not None and math.isfinite(value) and value > 0 else None

    def _require_planning_time(self) -> None:
        """Refuse to start a plan that cannot be answered and acted on in time.

        Measured on a real trial: a revision was requested ten seconds before
        working time ended. Its call was cut off at the deadline, its cost
        became unknown, the closing reply was barred with it, and the trial
        ended with no answer. With too little working time left, the run ends
        on its clock and the reserve is spent on the reply instead.
        """
        remaining = self._remaining_wall
        if self._plan_minimum and remaining is not None and remaining() < self._plan_minimum:
            raise _WallDeadlineReached("too little working time remains to start another plan")

    def _spend_cap_reached(self) -> str | None:
        """Why the goal may take on no more work, once its known spending is at its cap.

        The goal's budget is a bound on the worst case of every call it
        admits, and a trial's was about fifteen times what the trial was meant
        to spend. Nothing held a goal to that smaller amount: a goal whose
        workers kept failing could go on until the bound itself ran out. The
        cap counts what is known to be spent, and it is checked before each
        new plan, which every new worker needs. A worker already running is
        not interrupted, so the goal can pass its cap by that worker's
        allowance, and by the one call that writes its closing reply.
        """
        cap = self.goal.spend_cap_usd
        if cap is None:
            return None
        spent = self._budget(self.supervisor.runs()).known_spent_usd
        if spent < cap:
            return None
        return f"known spending of ${spent:.2f} reached the goal's spend cap of ${cap:.2f}"

    def _require_spend_room(self) -> None:
        reached = self._spend_cap_reached()
        if reached is not None:
            raise _SpendCapReached(reached)

    def _charge_lost_cost(self, run: SupervisorRun, reservations: list[float], unknown: list[str]) -> None:
        """Account for an ended run whose exact spending cannot be read.

        A worker killed during a model call leaves no usage for that call: its
        exact cost is lost for good. Calling the whole goal's budget unprovable
        then stopped every later worker and planner call, including the reply
        owed at the end, which is how one timeout used to end a trial. An
        Azure worker cannot outspend its caps: each journal admits a call only
        while its known spend plus that call's worst case fits. So its ended
        run stays charged at the assignment's full ceiling, as a reservation
        that is never released. The budget remains a proof; the outcome still
        shows unsettled accounting through that nonzero reservation. Harnesses
        without such a pre-dispatch bound keep the unknown classification.
        """
        ceiling = self._assignment_cost_ceiling(run.assignment)
        if ceiling is None or "azure_openai" not in run.assignment.resources:
            unknown.append(run.run_id)
        else:
            reservations.append(ceiling)

    def _reported_cost_bounds(self, run: SupervisorRun, report: WorkerReport) -> tuple[float, float] | None:
        """What an ended run spent for certain, and the most its unknown calls can add.

        A run that lost one reply still wrote its report, from journals that
        hold every receipt and the exact request of each call whose outcome is
        unknown. Charged its whole ceiling, as a killed run is, one unanswered
        request cost a trial about $60 of budget for a run that had spent
        under a dollar. With the report's own account, what it paid is counted
        as spent and only the unknown calls' worst case stays reserved. An
        account that is missing, malformed or beyond the run's own caps is not
        relied on, and the run stays at its ceiling.
        """
        ceiling = self._assignment_cost_ceiling(run.assignment)
        account = report.metadata.get("model_cost")
        if (ceiling is None or "azure_openai" not in run.assignment.resources
                or not isinstance(account, Mapping)
                or set(account) != {"known_usd", "unknown_exposure_usd"}):
            return None
        known, exposure = account["known_usd"], account["unknown_exposure_usd"]
        for value in (known, exposure):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                return None
        if known + exposure > ceiling:
            return None
        return float(known), float(exposure)

    @staticmethod
    def _assignment_cost_ceiling(assignment: Assignment) -> float | None:
        target = assignment.contract.budget_usd
        monitor = assignment.resources.get("monitor_budget_usd")
        for value in (target, monitor):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                return None
        try:
            total = math.fsum((float(target), float(monitor)))
        except OverflowError:
            return None
        return total if math.isfinite(total) and total > 0 else None

    def _can_start(
        self, assignment: Assignment, budget: BudgetState
    ) -> tuple[bool, RuntimeTrigger | None]:
        if budget.limit_usd is None:
            return True, None
        if (
            budget.unknown_run_ids
            or budget.unbounded_live_run_ids
            or budget.unknown_planner_attempt_ids
        ):
            trigger = RuntimeTrigger(
                kind="budget_unknown",
                subject_id=assignment.assignment_id,
                detail="global budget cannot be proven while prior cost is unknown",
                evidence=budget.to_dict(),
            )
            return False, trigger
        ceiling = self._assignment_cost_ceiling(assignment)
        if ceiling is None:
            return False, RuntimeTrigger(
                kind="unbounded_assignment",
                subject_id=assignment.assignment_id,
                detail=(
                    "budgeted goals require positive worker and monitor cost ceilings"
                ),
                evidence=budget.to_dict(),
            )
        remaining = budget.remaining_usd
        assert remaining is not None
        if ceiling > remaining:
            return False, RuntimeTrigger(
                kind="budget_exhausted",
                subject_id=assignment.assignment_id,
                detail="assignment cost ceiling exceeds the provable remaining goal budget",
                evidence={**budget.to_dict(), "assignment_ceiling_usd": ceiling},
            )
        return True, None

    def _authorize_planner_call(
        self,
        budget: BudgetState,
        *,
        operation_id: str,
    ) -> float:
        """Reserve a provable upper bound before a budgeted planner effect.

        A completed transport receipt with no promoted planner outcome is a
        free replay, so it may finish even after the budget was exhausted.
        Every genuinely new provider call must expose a conservative billed
        ceiling and bind its process-local guard to the durable remainder.
        """
        self._raise_if_stop_requested()
        if budget.limit_usd is None:
            return 0.0

        ceiling_fn = getattr(self.planner.transport, "max_billed_call_usd", None)
        if not callable(ceiling_fn):
            raise BudgetBlocked(
                "budgeted planner transport has no provable per-call billed ceiling"
            )
        try:
            raw_ceiling = ceiling_fn()
        except Exception as exc:
            raise BudgetBlocked("planner call ceiling could not be established") from exc
        if (
            isinstance(raw_ceiling, bool)
            or not isinstance(raw_ceiling, (int, float))
            or not math.isfinite(float(raw_ceiling))
            or float(raw_ceiling) < 0
        ):
            raise BudgetBlocked("planner call ceiling is not a finite non-negative number")
        ceiling = float(raw_ceiling)

        matching = tuple(
            attempt
            for attempt in self.planner.planner_attempts(self.goal.goal_id)
            if attempt.operation_id == operation_id and attempt.status == "pending"
        )
        replay_only = bool(matching) and all(
            attempt.telemetry.cost_known for attempt in matching
        )
        if replay_only:
            return ceiling
        if not budget.enforceable:
            raise BudgetBlocked("planner call is blocked because durable cost is unknown")
        remaining = budget.remaining_usd
        if remaining is None or ceiling > remaining:
            raise BudgetBlocked(
                "planner call ceiling exceeds the provable remaining goal budget"
            )
        bind = getattr(self.planner.transport, "bind_budget", None)
        if ceiling > 0 and not callable(bind):
            raise BudgetBlocked(
                "budgeted planner transport cannot bind the durable remaining budget"
            )
        if callable(bind):
            try:
                bind(remaining_usd=remaining)
            except Exception as exc:
                raise BudgetBlocked("planner budget binding failed") from exc
        return ceiling

    def _is_delivered(self, run: SupervisorRun) -> bool:
        if run.phase != "delivered" or run.integration_state_id is None:
            return False
        try:
            if not self.store.backend.is_ancestor(
                run.integration_state_id, self.integration.head.id
            ):
                return False
            report = self._load_report_for_accounting(run)
            if report is None:
                return False
            final = self.store.state(report.final_state_id)
            reported_ids = {item.artifact_id for item in report.outputs}
            selected = tuple(
                spec
                for spec in run.assignment.outputs
                if spec.disposition == "absent" or spec.artifact_id in reported_ids
            )
            if not selected:
                return False
            current = self.integration.head
            for spec in selected:
                expected = self.store.backend.entry_at(final.id, spec.path)
                actual = self.store.backend.entry_at(current.id, spec.path)
                if expected is None:
                    if actual is not None:
                        return False
                elif (
                    actual is None
                    or actual.sha != expected.sha
                    or actual.mode != expected.mode
                ):
                    return False
            return True
        except Exception as exc:
            raise CoordinatorCorruption("delivered integration state is unavailable") from exc

    @staticmethod
    def _wall_timeout(assignment: Assignment, fallback: float) -> float:
        value = assignment.resources.get("wall_timeout_seconds", fallback)
        if value is None:
            value = fallback  # Null and absence both mean an unspecified timeout.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("wall_timeout_seconds must be numeric")
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError("wall_timeout_seconds must be finite and positive")
        return result

    def _pending_trigger_records(
        self, plan_id: str, generation: int
    ) -> tuple[RuntimeTrigger, ...]:
        prefix = f"{_goal_root(self.goal.goal_id)}/plans/{_key(plan_id)}/triggers/"
        index_path = _trigger_index_path(self.goal.goal_id, plan_id)
        with self.control_lock:
            head = self.control.head
            index = head.record(index_path)
            changed = self.store.backend.repo.git.rev_list(
                "--first-parent", head.id, "--", index_path
            )
            historical_indexes: list[dict[str, Any]] = []
            for commit in changed.splitlines():
                text = self.store.backend.show(commit, index_path)
                if text is None:
                    continue
                try:
                    item = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise CoordinatorCorruption("trigger index history is malformed") from exc
                if not isinstance(item, dict):
                    raise CoordinatorCorruption("trigger index history is malformed")
                historical_indexes.append(item)
            if index is None:
                if historical_indexes:
                    raise CoordinatorCorruption(
                        "durable trigger index disappeared from current state"
                    )
                trigger_ids: list[Any] = []
            else:
                if (
                    not isinstance(index, dict)
                    or set(index)
                    != {"schema", "goal_id", "plan_id", "generation", "trigger_ids"}
                    or index.get("schema") != TRIGGER_INDEX_SCHEMA
                    or index.get("goal_id") != self.goal.goal_id
                    or index.get("plan_id") != plan_id
                    or index.get("generation") != generation
                    or not isinstance(index.get("trigger_ids"), list)
                ):
                    raise CoordinatorCorruption("durable trigger index is malformed")
                trigger_ids = index["trigger_ids"]
                for historical in historical_indexes:
                    old = historical.get("trigger_ids")
                    if (
                        historical.get("schema") != TRIGGER_INDEX_SCHEMA
                        or historical.get("goal_id") != self.goal.goal_id
                        or historical.get("plan_id") != plan_id
                        or historical.get("generation") != generation
                        or not isinstance(old, list)
                        or trigger_ids[: len(old)] != old
                    ):
                        raise CoordinatorCorruption(
                            "durable trigger index was rolled back or rewritten"
                        )
            if not all(isinstance(item, str) for item in trigger_ids) or len(
                set(trigger_ids)
            ) != len(trigger_ids):
                raise CoordinatorCorruption("durable trigger index IDs are malformed")

            expected_paths = {
                _trigger_path(self.goal.goal_id, plan_id, trigger_id)
                for trigger_id in trigger_ids
            }
            actual_paths = {
                path
                for path in head.files()
                if path.startswith(prefix) and path.endswith(".json")
            }
            if actual_paths != expected_paths:
                raise CoordinatorCorruption("trigger files and durable trigger index disagree")
            triggers: list[RuntimeTrigger] = []
            for trigger_id in trigger_ids:
                path = _trigger_path(self.goal.goal_id, plan_id, trigger_id)
                raw = head.record(path)
                if (
                    not isinstance(raw, dict)
                    or set(raw)
                    != {"schema", "goal_id", "plan_id", "generation", "trigger"}
                    or raw.get("schema") != TRIGGER_SCHEMA
                    or raw.get("goal_id") != self.goal.goal_id
                    or raw.get("plan_id") != plan_id
                    or raw.get("generation") != generation
                ):
                    raise CoordinatorCorruption(f"pending trigger {path!r} is malformed")
                try:
                    trigger = RuntimeTrigger.from_dict(raw["trigger"])
                except ValueError as exc:
                    raise CoordinatorCorruption(f"pending trigger {path!r} is malformed") from exc
                if trigger.trigger_id != trigger_id:
                    raise CoordinatorCorruption(f"pending trigger {path!r} has wrong identity")
                artifact_name = _trigger_artifact_name(plan_id, trigger_id)
                entry = head.manifest.entries.get(artifact_name)
                if (
                    entry is None
                    or entry.path != path
                    or entry.description != _canonical(trigger.to_dict())
                    or entry.blob != head.blob(path)
                ):
                    raise CoordinatorCorruption(
                        f"pending trigger {path!r} is not exactly published to the planner"
                    )
                triggers.append(trigger)
        return tuple(triggers)

    def _pending_triggers(self, plan: PlanRevision) -> tuple[RuntimeTrigger, ...]:
        return self._pending_trigger_records(plan.plan_id, plan.generation)

    def _record_trigger(self, cycle: _Cycle, trigger: RuntimeTrigger) -> None:
        if cycle.plan_id is None:
            raise CoordinatorCorruption("a bootstrap cycle cannot own a replan trigger")
        durable = {
            "schema": TRIGGER_SCHEMA,
            "goal_id": self.goal.goal_id,
            "plan_id": cycle.plan_id,
            "generation": cycle.generation,
            "trigger": trigger.to_dict(),
        }
        existing = self._pending_trigger_records(cycle.plan_id, cycle.generation)
        identities = {item.trigger_id: item for item in existing}
        prior = identities.get(trigger.trigger_id)
        if prior is not None and prior != trigger:
            raise CoordinatorCorruption("trigger ID is bound to different evidence")
        if prior is None:
            trigger_ids = [item.trigger_id for item in existing]
            prior_index = (
                None
                if not trigger_ids
                else {
                    "schema": TRIGGER_INDEX_SCHEMA,
                    "goal_id": self.goal.goal_id,
                    "plan_id": cycle.plan_id,
                    "generation": cycle.generation,
                    "trigger_ids": list(trigger_ids),
                }
            )
            trigger_ids.append(trigger.trigger_id)
            index = {
                "schema": TRIGGER_INDEX_SCHEMA,
                "goal_id": self.goal.goal_id,
                "plan_id": cycle.plan_id,
                "generation": cycle.generation,
                "trigger_ids": trigger_ids,
            }
            with self.control_lock:
                index_path = _trigger_index_path(self.goal.goal_id, cycle.plan_id)
                if self.control.head.record(index_path) != prior_index:
                    raise CoordinatorCorruption("trigger index moved during trigger creation")
                trigger_path = _trigger_path(
                    self.goal.goal_id, cycle.plan_id, trigger.trigger_id
                )
                self.control.publish(
                    _trigger_artifact_name(cycle.plan_id, trigger.trigger_id),
                    trigger_path,
                    description=_canonical(trigger.to_dict()),
                )
                self.control.checkpoint(
                    f"central trigger: {trigger.trigger_id}",
                    records={
                        trigger_path: durable,
                        index_path: index,
                    },
                )
        decision_id, root, _record = self._decision_intent(
            cycle,
            "trigger",
            trigger.trigger_id,
            trigger.to_dict(),
        )
        self._decision_result(
            cycle,
            decision_id,
            root,
            "trigger",
            trigger.trigger_id,
            "observed",
            trigger.to_dict(),
        )

    def _operation_id(self, plan: PlanRevision, triggers: Sequence[RuntimeTrigger], *,
                      closing: bool = False) -> str:
        identity = {
            "goal_id": self.goal.goal_id,
            "plan_id": plan.plan_id,
            "generation": plan.generation,
            "triggers": [
                trigger.to_dict() for trigger in sorted(triggers, key=lambda item: item.trigger_id)
            ],
        }
        # The prefix is part of the immutable request, so the planner's prompt
        # and parser know a closing proposal without any other channel.
        prefix = benchmark_reply.CLOSING_OPERATION_PREFIX if closing else "runtime-revise."
        return prefix + hashlib.sha256(_canonical(identity).encode()).hexdigest()

    def _planner_operation_epochs(self, base_id: str) -> tuple[dict[str, Any], ...]:
        root = _planner_operation_root(self.goal.goal_id, base_id)
        index_path = f"{root}/index.json"
        with self.control_lock:
            head = self.control.head
            index = head.record(index_path)
            changed = self.store.backend.repo.git.rev_list(
                "--first-parent", head.id, "--", index_path
            )
            history: list[dict[str, Any]] = []
            for commit in changed.splitlines():
                text = self.store.backend.show(commit, index_path)
                if text is None:
                    continue
                try:
                    historical = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise CoordinatorCorruption(
                        "planner-operation index history is malformed"
                    ) from exc
                if not isinstance(historical, dict):
                    raise CoordinatorCorruption(
                        "planner-operation index history is malformed"
                    )
                history.append(historical)
            if index is None:
                if history:
                    raise CoordinatorCorruption(
                        "planner-operation index disappeared from current state"
                    )
                return ()
            if (
                not isinstance(index, dict)
                or set(index) != {"schema", "goal_id", "base_id", "epochs"}
                or index.get("schema") != OPERATION_INDEX_SCHEMA
                or index.get("goal_id") != self.goal.goal_id
                or index.get("base_id") != base_id
                or not isinstance(index.get("epochs"), list)
            ):
                raise CoordinatorCorruption("planner-operation index is malformed")
            epochs = index["epochs"]
            ranks = {"pending": 0, "terminal": 1, "applied": 1}
            for position, epoch in enumerate(epochs, 1):
                if (
                    not isinstance(epoch, dict)
                    or set(epoch) != {"epoch", "operation_id", "status"}
                    or epoch.get("epoch") != position
                    or epoch.get("status") not in ranks
                    or not isinstance(epoch.get("operation_id"), str)
                ):
                    raise CoordinatorCorruption("planner-operation epoch is malformed")
                intent_path = f"{root}/epochs/{position:06d}/intent.json"
                intent = head.record(intent_path)
                if (
                    not isinstance(intent, dict)
                    or intent.get("schema") != OPERATION_INTENT_SCHEMA
                    or intent.get("goal_id") != self.goal.goal_id
                    or intent.get("base_id") != base_id
                    or intent.get("epoch") != position
                    or intent.get("operation_id") != epoch["operation_id"]
                ):
                    raise CoordinatorCorruption(
                        "planner-operation intent is missing or malformed"
                    )
                result_path = f"{root}/epochs/{position:06d}/result.json"
                result = head.record(result_path)
                if epoch["status"] == "pending" and result is not None:
                    raise CoordinatorCorruption(
                        "pending planner operation already has a terminal result"
                    )
                if epoch["status"] != "pending" and (
                    not isinstance(result, dict)
                    or result.get("schema") != OPERATION_RESULT_SCHEMA
                    or result.get("operation_id") != epoch["operation_id"]
                    or result.get("status") != epoch["status"]
                ):
                    raise CoordinatorCorruption(
                        "completed planner-operation result is missing or malformed"
                    )
            if any(item["status"] == "pending" for item in epochs[:-1]):
                raise CoordinatorCorruption("only the latest planner epoch may be pending")
            if any(item["status"] == "applied" for item in epochs[:-1]):
                raise CoordinatorCorruption("an applied planner epoch cannot have a successor")

            for historical in history:
                old = historical.get("epochs")
                if (
                    historical.get("schema") != OPERATION_INDEX_SCHEMA
                    or historical.get("goal_id") != self.goal.goal_id
                    or historical.get("base_id") != base_id
                    or not isinstance(old, list)
                    or len(old) > len(epochs)
                ):
                    raise CoordinatorCorruption(
                        "planner-operation index was rolled back or rewritten"
                    )
                for position, old_epoch in enumerate(old):
                    current = epochs[position]
                    if (
                        not isinstance(old_epoch, dict)
                        or old_epoch.get("epoch") != current["epoch"]
                        or old_epoch.get("operation_id") != current["operation_id"]
                        or old_epoch.get("status") not in ranks
                        or ranks[current["status"]] < ranks[old_epoch["status"]]
                        or (
                            ranks[current["status"]] == ranks[old_epoch["status"]] == 1
                            and current["status"] != old_epoch["status"]
                        )
                    ):
                        raise CoordinatorCorruption(
                            "planner-operation index was rolled back or rewritten"
                        )
        return tuple(dict(item) for item in epochs)

    def _select_planner_operation(self, base_id: str) -> str:
        epochs = list(self._planner_operation_epochs(base_id))
        if epochs and epochs[-1]["status"] == "pending":
            return epochs[-1]["operation_id"]
        if epochs and epochs[-1]["status"] == "applied":
            raise CoordinatorCorruption("an applied planner operation was unexpectedly replayed")
        epoch = len(epochs) + 1
        operation_id = f"{base_id}.epoch.{epoch}"
        _stable_id(operation_id, "planner operation_id")
        entry = {"epoch": epoch, "operation_id": operation_id, "status": "pending"}
        new_epochs = [*epochs, entry]
        root = _planner_operation_root(self.goal.goal_id, base_id)
        old_index = (
            None
            if not epochs
            else {
                "schema": OPERATION_INDEX_SCHEMA,
                "goal_id": self.goal.goal_id,
                "base_id": base_id,
                "epochs": epochs,
            }
        )
        index = {
            "schema": OPERATION_INDEX_SCHEMA,
            "goal_id": self.goal.goal_id,
            "base_id": base_id,
            "epochs": new_epochs,
        }
        intent = {
            "schema": OPERATION_INTENT_SCHEMA,
            "goal_id": self.goal.goal_id,
            "base_id": base_id,
            "epoch": epoch,
            "operation_id": operation_id,
            "at": _iso(self.clock()),
        }
        with self.control_lock:
            index_path = f"{root}/index.json"
            if self.control.head.record(index_path) != old_index:
                raise CoordinatorCorruption(
                    "planner-operation index moved while allocating an epoch"
                )
            self.control.checkpoint(
                f"central planner operation: {operation_id}",
                records={
                    index_path: index,
                    f"{root}/epochs/{epoch:06d}/intent.json": intent,
                },
            )
        return operation_id

    def _finish_planner_operation(
        self,
        base_id: str,
        operation_id: str,
        status: str,
        *,
        detail: str = "",
    ) -> None:
        if status not in {"terminal", "applied"}:
            raise ValueError("planner operation status must be terminal or applied")
        epochs = list(self._planner_operation_epochs(base_id))
        if not epochs or epochs[-1]["operation_id"] != operation_id:
            raise CoordinatorCorruption("planner-operation completion has no matching epoch")
        current = epochs[-1]
        if current["status"] == status:
            return
        if current["status"] != "pending":
            raise CoordinatorCorruption("planner operation completed two different ways")
        updated = {**current, "status": status}
        new_epochs = [*epochs[:-1], updated]
        root = _planner_operation_root(self.goal.goal_id, base_id)
        old_index = {
            "schema": OPERATION_INDEX_SCHEMA,
            "goal_id": self.goal.goal_id,
            "base_id": base_id,
            "epochs": epochs,
        }
        index = {**old_index, "epochs": new_epochs}
        result = {
            "schema": OPERATION_RESULT_SCHEMA,
            "goal_id": self.goal.goal_id,
            "base_id": base_id,
            "epoch": current["epoch"],
            "operation_id": operation_id,
            "status": status,
            "detail": detail[:2048],
            "at": _iso(self.clock()),
        }
        with self.control_lock:
            index_path = f"{root}/index.json"
            if self.control.head.record(index_path) != old_index:
                raise CoordinatorCorruption(
                    "planner-operation index moved while recording completion"
                )
            self.control.checkpoint(
                f"central planner operation {status}: {operation_id}",
                records={
                    index_path: index,
                    f"{root}/epochs/{current['epoch']:06d}/result.json": result,
                },
            )

    def _retire_stale_trigger_context(
        self, cycle: _Cycle, current_plan: PlanRevision
    ) -> None:
        current_prefix = (
            f"{_goal_root(self.goal.goal_id)}/plans/{_key(current_plan.plan_id)}/triggers/"
        )
        with self.control_lock:
            stale_names = tuple(
                sorted(
                    name
                    for name, entry in self.control.head.manifest.entries.items()
                    if name.startswith("central-runtime-trigger.")
                    and not entry.path.startswith(current_prefix)
                )
            )
        if not stale_names:
            return

        def retire() -> Any:
            with self.control_lock:
                for name in stale_names:
                    self.control.unpublish(name)
                return self.control.checkpoint(
                    f"retire trigger context after plan {current_plan.plan_id} promotion"
                )

        self._perform(
            cycle,
            "retire_stale_trigger_context",
            current_plan.plan_id,
            {"artifact_names": list(stale_names)},
            retire,
            lambda state: {"control_state_id": state.id},
        )

    def _replan(
        self,
        cycle: _Cycle,
        plan: PlanRevision,
        triggers: tuple[RuntimeTrigger, ...],
        *,
        closing: bool = False,
    ) -> PlanRevision:
        if not closing:
            self._require_planning_time()
            self._require_spend_room()
        for trigger in triggers:
            self._record_trigger(cycle, trigger)
        base_id = self._operation_id(plan, triggers, closing=closing)
        operation_id = self._select_planner_operation(base_id)
        call_ceiling = self._authorize_planner_call(
            self._budget(self.supervisor.runs()),
            operation_id=operation_id,
        )
        try:
            revised = self._perform(
                cycle,
                "replan",
                operation_id,
                {
                    "operation_id": operation_id,
                    "operation_base_id": base_id,
                    "parent_plan_id": plan.plan_id,
                    "trigger_ids": [item.trigger_id for item in triggers],
                    "planner_call_ceiling_usd": call_ceiling,
                },
                lambda: self.planner.revise(self.goal, operation_id=operation_id),
                lambda value: {
                    "plan_id": value.plan_id,
                    "generation": value.generation,
                    "complete": value.complete,
                },
            )
        except (InvalidPlannerOutput, RejectedPlannerOperation, StalePlanningWorld, PlannerTransportError) as exc:
            self._finish_planner_operation(
                base_id,
                operation_id,
                "terminal",
                detail=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._finish_planner_operation(base_id, operation_id, "applied")
        if revised.plan_id == plan.plan_id or revised.generation <= plan.generation:
            raise CoordinatorCorruption("planner revision did not promote a newer plan")
        self._retire_stale_trigger_context(cycle, revised)
        # Promotion is the generation fence.  Only after it is durable do we
        # stop old live workers, preserving their dirty tree via supervisor.
        self._validate_supervisor_scope(self.supervisor.runs())
        self._perform(
            cycle,
            "fence_old_generation",
            revised.plan_id,
            {"active_generation": revised.generation},
            lambda: self.supervisor.reconcile(active_generation=revised.generation),
            lambda values: {
                "run_ids": [item.run_id for item in values],
                "phases": {item.run_id: item.phase for item in values},
            },
        )
        return revised

    def _complete_cycle(self, cycle: _Cycle, outcome: CycleOutcome) -> CycleOutcome:
        self._finish_raw(cycle, outcome.status, outcome.to_dict())
        return outcome

    def _bootstrap(self, cycle: _Cycle, budget: BudgetState) -> PlanRevision:
        base_id = "runtime-initial." + hashlib.sha256(self.goal.to_json().encode()).hexdigest()
        operation_id = self._select_planner_operation(base_id)
        call_ceiling = self._authorize_planner_call(budget, operation_id=operation_id)
        try:
            plan = self._perform(
                cycle,
                "initial_plan",
                operation_id,
                {
                    "operation_id": operation_id,
                    "operation_base_id": base_id,
                    "goal_id": self.goal.goal_id,
                    "planner_call_ceiling_usd": call_ceiling,
                },
                lambda: self.planner.plan(self.goal, operation_id=operation_id),
                lambda value: {
                    "plan_id": value.plan_id,
                    "generation": value.generation,
                    "complete": value.complete,
                },
            )
        except (InvalidPlannerOutput, RejectedPlannerOperation, StalePlanningWorld, PlannerTransportError) as exc:
            self._finish_planner_operation(
                base_id,
                operation_id,
                "terminal",
                detail=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._finish_planner_operation(base_id, operation_id, "applied")
        return plan

    def request_stop(self, detail: str = "external cancellation") -> None:
        """Signal the driver without waiting for a planner or coordinator lock.

        This is an in-process request, not a receipt proving termination. The
        driver persists its first stop reason at the next effect boundary and
        drains owned workers. A hung external call still needs an outer process
        boundary; callers must await run/stop before treating cleanup as done.
        """
        if not isinstance(detail, str) or "\x00" in detail or len(detail) > 2048:
            raise ValueError("stop detail must be text without NUL, at most 2048 characters")
        with self._stop_signal_lock:
            if not self._stop_signal.is_set():
                self._requested_stop_detail = detail
                self._stop_signal.set()

    def _raise_if_stop_requested(self) -> None:
        if self._stop_signal.is_set():
            raise _StopRequested(self._requested_stop_detail)

    def stop(self, detail: str = "external cancellation") -> GoalOutcome:
        """Stop further admission, wait for the driver, and drain every worker.

        A prior ending is immutable. Incomplete cleanup retains the durable
        stop request and raises, allowing this operation to be retried without
        another planner call or worker launch.
        """
        self.request_stop(detail)
        with self._lock:
            recorded = self.outcome()
            if recorded is not None:
                return recorded
            return self._finish_run("cancelled", self._requested_stop_detail or "")

    def _validated_run_limits(self, max_generations, wall_clock_seconds, max_planner_failures):
        if isinstance(max_generations, bool) or not isinstance(max_generations, int) or max_generations < 1:
            raise ValueError("max_generations must be a positive integer")
        if isinstance(max_planner_failures, bool) or not isinstance(max_planner_failures, int) or max_planner_failures < 1:
            raise ValueError("max_planner_failures must be a positive integer")
        if isinstance(wall_clock_seconds, bool):
            raise ValueError("wall_clock_seconds must be a positive finite number")
        wall_clock_seconds = float(wall_clock_seconds)
        if not math.isfinite(wall_clock_seconds) or wall_clock_seconds <= 0:
            raise ValueError("wall_clock_seconds must be a positive finite number")
        return {"schema": "taste.brains/GoalRunLimits/1", "goal_digest": _digest(self.goal.to_json()),
                "max_generations": max_generations, "wall_clock_seconds": wall_clock_seconds,
                "max_planner_failures": max_planner_failures}

    def _bind_run_limits(self, expected, *, deadline_at: datetime | None = None):
        limits_path = f"{_goal_root(self.goal.goal_id)}/run-limits.json"
        limits = self.control.head.record(limits_path)
        if limits is None:
            deadline = deadline_at or self.clock() + timedelta(seconds=expected["wall_clock_seconds"])
            limits = {**expected, "deadline_at": _iso(deadline)}
            self._immutable(limits_path, limits, f"central run limits: {self.goal.goal_id}")
        if (not isinstance(limits, dict) or set(limits) != {*expected, "deadline_at"}
                or type(limits["max_generations"]) is not int
                or type(limits["max_planner_failures"]) is not int
                or isinstance(limits["wall_clock_seconds"], bool)
                or any(limits[key] != value for key, value in expected.items())):
            raise CoordinatorError("run limits differ from this goal's durable trial limits")
        try:
            deadline = datetime.fromisoformat(limits["deadline_at"].replace("Z", "+00:00"))
            if deadline.tzinfo is None or deadline.utcoffset() is None:
                raise ValueError("deadline has no timezone")
        except (TypeError, ValueError, AttributeError) as exc:
            raise CoordinatorCorruption("durable run deadline is malformed") from exc
        if deadline_at is not None and deadline != deadline_at:
            raise CoordinatorError("deadline differs from this goal's durable trial deadline")
        return limits, deadline

    def prepare_run(
        self, *, max_generations: int, wall_clock_seconds: float,
        deadline_at: datetime, max_planner_failures: int = 3,
    ) -> dict[str, Any]:
        """Persist the containing trial's deadline before launching its driver.

        This admits no planner call or worker. A delayed/restarted driver must
        use these same limits, and an expired deadline permits only drainage.
        Prepared goals must be advanced through run(), which enforces the
        bounds; the unbounded cycle() API is then unavailable to outside callers.
        """
        expected = self._validated_run_limits(max_generations, wall_clock_seconds, max_planner_failures)
        if not isinstance(deadline_at, datetime):
            raise ValueError("deadline_at must be a timezone-aware datetime")
        _iso(deadline_at)
        with self._lock:
            if self._running:
                raise CoordinatorError("cannot prepare limits during an active run")
            if deadline_at > self.clock() + timedelta(seconds=expected["wall_clock_seconds"]):
                raise ValueError("deadline_at exceeds the wall-clock allowance")
            if self.outcome() is not None or self._stop_request() is not None or self._stop_signal.is_set():
                raise CoordinatorError("goal shutdown is pending or complete; cannot prepare execution")
            if (self.control.head.record(f"{_goal_root(self.goal.goal_id)}/run-limits.json") is None
                    and (self.supervisor.runs() or self.planner.planner_attempts(self.goal.goal_id))):
                raise CoordinatorError("cannot prepare a trial after goal execution has begun")
            limits, _ = self._bind_run_limits(expected, deadline_at=deadline_at)
            return dict(limits)

    def _await_change(self, runs: Sequence[SupervisorRun], limit: float) -> None:
        """After a cycle that changed nothing, wait for something a cycle could act on.

        A cycle is durable: it records its intent, its decisions and its
        result. Run back to back while a worker was simply working, cycles
        were the coordinator's main cost. Measured on a real nine-minute
        trial: 184 cycles, 852 control commits and seven CPU-minutes, each
        cycle slower than the one before because the history it audits had
        grown.

        What a cycle can newly find while its workers work is a process that
        ended or announced readiness, a deadline, or a stop request. Those
        are watched here without recording anything. Whatever else may have
        arrived, such as a message in the coordinator's inbox, is found at
        the next heartbeat.
        """
        waited, pause = 0.0, self.supervisor.poll_interval
        heartbeat = min(self.idle_heartbeat_seconds, limit)
        while waited < heartbeat:
            if self._stop_signal.is_set() or self._remaining_wall() <= 0:
                return
            if not self.supervisor.quiet(runs, deadline_at=self._deadline_at):
                return
            step = min(pause, heartbeat - waited)
            time.sleep(step)
            waited += step
            pause = min(pause * 2, max(self.idle_poll_seconds, self.supervisor.poll_interval))

    def run(
        self,
        *,
        max_generations: int,
        wall_clock_seconds: float,
        between_cycles: Any = None,
        monotonic: Any = None,
        max_planner_failures: int = 3,
    ) -> GoalOutcome:
        """Run one durable trial, serializing concurrent drivers of this runtime."""
        with self._lock:
            if self._running:
                raise CoordinatorError("this runtime already has an active run() driver")
            self._running = True
            try:
                return self._run(
                    max_generations=max_generations, wall_clock_seconds=wall_clock_seconds,
                    between_cycles=between_cycles, monotonic=monotonic,
                    max_planner_failures=max_planner_failures,
                )
            finally:
                self._running = False

    def _run(
        self,
        *,
        max_generations: int,
        wall_clock_seconds: float,
        between_cycles: Any = None,
        monotonic: Any = None,
        max_planner_failures: int = 3,
    ) -> GoalOutcome:
        """Drive cycles until the goal is answered, then record why it stopped.

        The bounds are required.  An unbudgeted goal has no other limit, and a
        default here would quietly choose a policy on the caller's behalf.

        Being stuck is not a stop.  A blocked dependency or a rejected report
        is a reason to replan, which is what the central brain exists to do,
        so the loop keeps going until the plan is complete, a hard bound is
        reached, or the budget can no longer be proven.
        """
        expected = self._validated_run_limits(max_generations, wall_clock_seconds, max_planner_failures)
        wall_clock_seconds = expected["wall_clock_seconds"]
        clock = monotonic or time.monotonic

        with self._lock:
            recorded = self.outcome()
            if recorded is not None:
                return recorded
            pending_stop = self._stop_request()
            if pending_stop is not None:
                return self._finish_run(pending_stop["stop_reason"], pending_stop["detail"])
            if self._stop_signal.is_set():
                return self._finish_run("cancelled", self._requested_stop_detail or "")

        started = clock()
        unchanged = None
        _, deadline = self._bind_run_limits(expected)
        # Work, including every worker's own allowance, ends this long before
        # the trial's deadline. The reserve is spent on the closing reply.
        reserve = benchmark_reply.closing_reserve(self.goal.metadata)
        self._plan_minimum = benchmark_reply.planning_minimum(self.goal.metadata) if reserve else 0.0
        self._remaining_wall = lambda: min(wall_clock_seconds - (clock() - started),
                                            (deadline - self.clock()).total_seconds()) - reserve
        self._deadline_at = deadline - timedelta(seconds=reserve)
        stop_reason = "generation_bound"
        detail = ""
        try:
            planner_failures = sum(attempt.status in {"rejected", "transport_error", "orphaned"}
                                   for attempt in self.planner.planner_attempts(self.goal.goal_id))
            while True:
                self._raise_if_stop_requested()
                remaining = self._remaining_wall()
                if remaining <= 0:
                    stop_reason = "wall_clock"
                    detail = f"the run reached its {wall_clock_seconds:g}s bound"
                    break
                if planner_failures >= max_planner_failures:
                    stop_reason, detail = "planner_failed", "durable planner failure limit reached"
                    break
                reached = self._spend_cap_reached()
                if reached is not None:
                    stop_reason, detail = "spend_cap", reached
                    break
                try:
                    bind_deadline = getattr(self.planner.transport, "bind_deadline", None)
                    if callable(bind_deadline):
                        # A plan already being written when working time ends
                        # may take half the reserve to arrive. Cut off at the
                        # deadline, it was paid for, unusable and of unknown
                        # cost; the closing reply keeps the other half.
                        bind_deadline(remaining_seconds=remaining + reserve / 2)
                    outcome = self.cycle(max_generations=max_generations)
                except (InvalidPlannerOutput, RejectedPlannerOperation, StalePlanningWorld,
                        PlannerTransportError) as exc:
                    planner_failures += 1
                    self._raise_if_stop_requested()
                    if self._remaining_wall() <= 0:
                        stop_reason, detail = "wall_clock", "goal deadline elapsed during planner call"
                        break
                    budget = self._budget(self.supervisor.runs())
                    if budget.limit_usd is not None and not budget.enforceable:
                        stop_reason, detail = "budget_blocked", "planner failure left durable cost unknown"
                        break
                    if planner_failures >= max_planner_failures:
                        stop_reason = "planner_failed"
                        detail = f"planner failed {planner_failures} times: {type(exc).__name__}: {exc}"[:2048]
                        break
                    # Each failed operation is terminal. The next operation
                    # snapshots the current world, including our decision
                    # records, and preserves all earlier billed receipts.
                    time.sleep(min(self.supervisor.poll_interval,
                                   max(0.0, wall_clock_seconds - (clock() - started))))
                    continue
                self._raise_if_stop_requested()
                if self._remaining_wall() <= 0:
                    stop_reason, detail = "wall_clock", "goal deadline elapsed during a cycle"
                    break
                if outcome.status == "budget_blocked":
                    stop_reason = "budget_blocked"
                    detail = "the budget could no longer be proven"
                    break
                if outcome.complete:
                    stop_reason = "complete"
                    detail = outcome.plan.completion_reason
                    break
                if outcome.status == "generation_bound":
                    stop_reason = "generation_bound"
                    detail = f"the run reached its {max_generations} generation bound"
                    break
                if between_cycles is not None:
                    between_cycles()
                    continue
                left = max(0.0, wall_clock_seconds - (clock() - started))
                seen = (outcome.plan.plan_id, outcome.status,
                        tuple((run.run_id, run.sequence) for run in outcome.runs))
                if seen == unchanged:
                    self._await_change(outcome.runs, left)
                else:
                    time.sleep(min(self.supervisor.poll_interval, left))
                unchanged = seen
        except _StopRequested as exc:
            stop_reason, detail = "cancelled", str(exc)
        except _WallDeadlineReached as exc:
            stop_reason, detail = "wall_clock", str(exc)
        except _SpendCapReached as exc:
            stop_reason, detail = "spend_cap", str(exc)
        except BudgetBlocked as exc:
            stop_reason, detail = "budget_blocked", str(exc)
        except BaseException as exc:
            self._finish_run("interrupted" if isinstance(exc, KeyboardInterrupt) else "runtime_error",
                             f"{type(exc).__name__}: {exc}"[:2048])
            raise
        finally:
            self._remaining_wall = None
            self._deadline_at = None
        if reserve and stop_reason in _CLOSING_STOPS:
            self._closing_reply(stop_reason, detail, deadline)
        return self._finish_run(stop_reason, detail)

    def _closing_reply(self, stop_reason: str, detail: str, deadline: datetime) -> None:
        """Drain the workers, then ask the coordinator once for its final reply.

        A run stopped by its clock, its generation bound, its budget or its
        planner used to end with no reply at all, because a reply existed only
        inside a complete plan. The stop stands; this adds what the coordinator
        can honestly say about the work so far. It is one ordinary, audited
        planning operation whose proposal assigns nothing. Any failure here
        leaves the goal without a reply and is recorded; it never replaces the
        original stop reason or delays its durable outcome.
        """
        self.closing_failure = None
        try:
            with self._lock:
                self.planner.bind_goal(self.goal)
                plan = self.planner.current_plan(self.goal.goal_id)
                if plan is None or plan.complete:
                    return
                runs = self.supervisor.runs()
                self._validate_supervisor_scope(runs)
                for run in runs:
                    stopped = self.supervisor.stop(run.run_id, stop_reason)
                    if stopped.recovery_status != "complete":
                        raise SupervisorError(f"worker {run.run_id} recovery is {stopped.recovery_status}")
                    if stopped.phase == "terminal" and stopped.assignment.generation == plan.generation:
                        # A worker stopped in good order has settled its call
                        # and written its report. No ordinary cycle follows to
                        # collect it, and without it the coordinator closes
                        # knowing nothing of what that worker ran or changed.
                        # A killed worker, or an unusable report, adds nothing.
                        with contextlib.suppress(SupervisorError):
                            self.supervisor.collect(run.run_id, active_generation=plan.generation)
                remaining = (deadline - self.clock()).total_seconds()
                if remaining < _CLOSING_MINIMUM_SECONDS:
                    raise CoordinatorError("too little time remains for a closing reply")
                bind_deadline = getattr(self.planner.transport, "bind_deadline", None)
                if callable(bind_deadline):
                    bind_deadline(remaining_seconds=remaining)
                cycle = self._begin_cycle(plan)
                trigger = RuntimeTrigger(
                    kind="closing", subject_id=self.goal.goal_id,
                    detail=f"the run ended ({stop_reason}); no further work can be assigned",
                    evidence={"stop_reason": stop_reason, "detail": detail[:1024]},
                )
                revised = self._replan(cycle, plan, (trigger,), closing=True)
                runs = self.supervisor.runs()
                self._complete_cycle(cycle, CycleOutcome(
                    cycle_id=cycle.cycle_id, status="closed", plan=revised,
                    runs=runs, budget=self._budget(runs),
                ))
        except CoordinatorCorruption:
            raise
        except (CoordinatorError, SupervisorError, InvalidPlannerOutput, RejectedPlannerOperation,
                StalePlanningWorld, PlannerTransportError) as exc:
            self.closing_failure = f"{type(exc).__name__}: {exc}"[:1024]

    def outcome(self) -> GoalOutcome | None:
        """The recorded answer for this goal, if the run already ended."""
        self.planner.bind_goal(self.goal)
        raw = self.control.head.read(_outcome_path(self.goal.goal_id))
        if raw is None:
            return None
        try:
            return GoalOutcome.from_json(raw)
        except (ValueError, KeyError, TypeError) as exc:
            raise CoordinatorCorruption("durable goal outcome is malformed") from exc

    def _stop_request(self) -> dict[str, Any] | None:
        raw = self.control.head.record(f"{_goal_root(self.goal.goal_id)}/stop.json")
        if raw is None:
            return None
        if (not isinstance(raw, dict)
                or set(raw) != {"schema", "goal_digest", "stop_reason", "detail"}
                or raw["schema"] != "taste.brains/GoalStop/1"
                or raw["goal_digest"] != _digest(self.goal.to_json())
                or not isinstance(raw["stop_reason"], str) or not raw["stop_reason"]
                or not isinstance(raw["detail"], str)):
            raise CoordinatorCorruption("durable goal stop request is malformed")
        return raw

    def _finish_run(self, stop_reason: str, detail: str) -> GoalOutcome:
        """Durably stop, drain every owned run, then publish the final answer.

        A cleanup failure leaves the stop request pending. Restart retries
        cleanup without planning or launching, and never publishes a result
        that could be mistaken for a safely finished trial.
        """
        with self._lock:
            self.planner.bind_goal(self.goal)
            runs = self.supervisor.runs()
            self._validate_supervisor_scope(runs)
            pending = self._stop_request()
            if pending is None:
                pending = {"schema": "taste.brains/GoalStop/1", "goal_digest": _digest(self.goal.to_json()),
                           "stop_reason": stop_reason, "detail": detail}
                self._immutable(f"{_goal_root(self.goal.goal_id)}/stop.json", pending,
                                f"central stop requested: {self.goal.goal_id}")
            errors: list[BaseException] = []
            for run in runs:
                try:
                    stopped = self.supervisor.stop(run.run_id, pending["stop_reason"])
                    if stopped.recovery_status != "complete":
                        raise SupervisorError(f"worker {run.run_id} recovery is {stopped.recovery_status}")
                except BaseException as exc:
                    errors.append(exc)
            if len(errors) == 1:
                raise errors[0]
            if errors:
                raise BaseExceptionGroup("goal shutdown could not drain every worker", errors)
            return self._record_outcome(pending["stop_reason"], pending["detail"])

    def _record_outcome(self, stop_reason: str, detail: str) -> GoalOutcome:
        plan = self.planner.current_plan(self.goal.goal_id)
        runs = self.supervisor.runs()
        criteria = self.planner.criteria(self.goal.goal_id)
        outcome = GoalOutcome(
            goal_id=self.goal.goal_id,
            stop_reason=stop_reason,
            complete=bool(stop_reason == "complete" and plan is not None and plan.complete),
            generations=0 if plan is None else plan.generation,
            cycles=len(self._cycle_records()),
            completion_reason="" if plan is None else plan.completion_reason,
            criteria=criteria,
            assessment=() if plan is None else tuple(plan.assessment),
            delivered_assignment_ids=tuple(sorted({run.assignment.assignment_id for run in runs
                                                   if run.phase == "delivered"})),
            budget=self._budget(runs),
            detail=detail,
        )
        with self._lock:
            self.control.checkpoint(
                f"central outcome: {self.goal.goal_id}",
                records={_outcome_path(self.goal.goal_id): outcome.to_dict()},
            )
        return outcome

    def cycle(self, *, max_generations: int | None = None) -> CycleOutcome:
        """Advance or recover one coordinator cycle without waiting for workers.

        Hosts call this repeatedly.  A returned ``complete`` is possible only
        when the planner has promoted an empty, complete ``PlanRevision``.
        Exceptions leave an unfinished cycle whose exact intents are replayed
        on the next call or after constructing a replacement runtime.
        """
        with self._lock:
            self._raise_if_stop_requested()
            if (not self._running and self.control.head.record(
                    f"{_goal_root(self.goal.goal_id)}/run-limits.json") is not None
                    and self._stop_request() is None):
                raise CoordinatorError("a bounded goal must be advanced through run(), not cycle()")
            self.planner.bind_goal(self.goal)
            if self._stop_request() is not None:
                raise CoordinatorError("goal shutdown is pending or complete; use run() to finalize")
            plan = self.planner.current_plan(self.goal.goal_id)
            cycle = self._begin_cycle(plan)
            if plan is not None and max_generations is not None and plan.generation > max_generations:
                return self._complete_cycle(cycle, CycleOutcome(
                    cycle_id=cycle.cycle_id, status="generation_bound", plan=plan,
                    runs=self.supervisor.runs(), budget=self._budget(self.supervisor.runs()),
                ))
            if plan is None:
                before_plan = self._budget(self.supervisor.runs())
                plan = self._bootstrap(cycle, before_plan)
                bootstrap_budget = self._budget(self.supervisor.runs())
                outcome = CycleOutcome(
                    cycle_id=cycle.cycle_id,
                    status="complete" if plan.complete else "planned",
                    plan=plan,
                    runs=self.supervisor.runs(),
                    budget=bootstrap_budget,
                )
                return self._complete_cycle(cycle, outcome)

            self._retire_stale_trigger_context(cycle, plan)

            # A complete plan is the sole completion authority.  No worker
            # status, all-delivered heuristic, or stale completion claim can
            # substitute for this promoted record.
            if plan.complete:
                before_reconcile = self.supervisor.runs()
                self._validate_supervisor_scope(before_reconcile)
                runs = self._perform(
                    cycle,
                    "reconcile_complete",
                    plan.plan_id,
                    {
                        "active_generation": plan.generation,
                        "observed_run_sequences": {
                            item.run_id: item.sequence for item in before_reconcile
                        },
                    },
                    lambda: self.supervisor.reconcile(
                        active_generation=plan.generation, deadline_at=self._deadline_at,
                    ),
                    lambda values: {
                        "run_ids": [item.run_id for item in values],
                        "sequences": {item.run_id: item.sequence for item in values},
                    },
                )
                outcome = CycleOutcome(
                    cycle_id=cycle.cycle_id,
                    status="complete",
                    plan=plan,
                    runs=runs,
                    budget=self._budget(runs),
                )
                return self._complete_cycle(cycle, outcome)

            triggers: list[RuntimeTrigger] = list(self._pending_triggers(plan))
            before_reconcile = self.supervisor.runs()
            self._validate_supervisor_scope(before_reconcile)
            if triggers:
                exact_before = self._exact_current_runs(plan, before_reconcile)
                for run in exact_before.values():
                    if run.phase not in {"prepared", "spawn_intent"}:
                        continue
                    self._perform(
                        cycle,
                        "stop_unstarted_after_trigger",
                        run.run_id,
                        {"run_sequence": run.sequence},
                        lambda run_id=run.run_id: self.supervisor.stop(
                            run_id, "pending_replan_trigger"
                        ),
                        lambda value: {
                            "run_id": value.run_id,
                            "phase": value.phase,
                            "recovery_state_id": value.recovery_state_id,
                        },
                    )
                before_reconcile = self.supervisor.runs()
            runs = self._perform(
                cycle,
                "reconcile",
                plan.plan_id,
                {
                    "active_generation": plan.generation,
                    "observed_run_sequences": {
                        item.run_id: item.sequence for item in before_reconcile
                    },
                },
                lambda: self.supervisor.reconcile(
                    active_generation=plan.generation, deadline_at=self._deadline_at,
                ),
                lambda values: {
                    "run_ids": [item.run_id for item in values],
                    "sequences": {item.run_id: item.sequence for item in values},
                },
            )
            exact = self._exact_current_runs(plan, runs)
            failed: set[str] = set()

            # First collect and deliver every exact terminal success.  This is
            # deliberately before replanning so the next world snapshot reuses
            # all certified partial progress already available.
            for assignment in plan.assignments:
                run = exact.get(assignment.assignment_id)
                if run is None or run.phase not in {"terminal", "report_accepted"}:
                    continue
                try:
                    if run.phase == "report_accepted":
                        report = self._load_report_for_accounting(run)
                        if report is None:
                            raise InvalidWorkerReport(
                                "accepted report accounting bytes are unavailable"
                            )
                    else:
                        report = self._perform(
                            cycle,
                            "collect",
                            run.run_id,
                            {"run_sequence": run.sequence},
                            lambda run_id=run.run_id: self.supervisor.collect(
                                run_id, active_generation=plan.generation
                            ),
                            lambda value: {
                                "report_id": value.report_id,
                                "completed": value.completed,
                                "cost_usd": value.cost_usd,
                            },
                        )
                    run = self.supervisor.get(run.run_id)
                    exact[assignment.assignment_id] = run
                except InvalidWorkerReport as exc:
                    trigger = RuntimeTrigger(
                        kind="invalid_report",
                        subject_id=assignment.assignment_id,
                        detail=str(exc),
                        evidence={
                            "run_id": run.run_id,
                            "run_sequence": run.sequence,
                            "terminal_reason": run.terminal_reason,
                        },
                    )
                    triggers.append(trigger)
                    failed.add(assignment.assignment_id)
                    continue
                if not report.completed:
                    triggers.append(
                        RuntimeTrigger(
                            kind="worker_incomplete",
                            subject_id=assignment.assignment_id,
                            detail=report.terminal_reason,
                            evidence={"run_id": run.run_id, "report_id": report.report_id},
                        )
                    )
                    failed.add(assignment.assignment_id)
                    continue
                try:
                    delivery = self._perform(
                        cycle,
                        "deliver",
                        report.report_id,
                        {"run_id": run.run_id, "final_state_id": report.final_state_id},
                        lambda run_id=run.run_id: self.supervisor.deliver(
                            run_id, active_generation=plan.generation
                        ),
                        lambda value: {
                            "projection_state_id": value.projection.id,
                            "integration_state_id": (
                                None
                                if value.integration_state is None
                                else value.integration_state.id
                            ),
                            "conflict_paths": [item.path for item in value.conflicts],
                        },
                    )
                except DeliveryRecoveryRequired as exc:
                    triggers.append(
                        RuntimeTrigger(
                            kind="delivery_uncertain",
                            subject_id=assignment.assignment_id,
                            detail=str(exc),
                            evidence={
                                "run_id": run.run_id,
                                "report_id": report.report_id,
                                "requires_operator_recovery": True,
                            },
                        )
                    )
                    failed.add(assignment.assignment_id)
                    continue
                except DeliveryIdentityConflict as exc:
                    raise CoordinatorCorruption(
                        "durable delivery identity changed for an exact report"
                    ) from exc
                except DeliveryRejected as exc:
                    triggers.append(
                        RuntimeTrigger(
                            kind="delivery_rejected",
                            subject_id=assignment.assignment_id,
                            detail=str(exc),
                            evidence={"run_id": run.run_id, "report_id": report.report_id},
                        )
                    )
                    failed.add(assignment.assignment_id)
                    continue
                run = self.supervisor.get(run.run_id)
                exact[assignment.assignment_id] = run
                if delivery.conflicts:
                    paths = tuple(item.path for item in delivery.conflicts)
                    triggers.append(
                        RuntimeTrigger(
                            kind="delivery_conflict",
                            subject_id=assignment.assignment_id,
                            detail="product projection conflicts with integration",
                            evidence={
                                "run_id": run.run_id,
                                "report_id": report.report_id,
                                "projection_state_id": delivery.projection.id,
                                "paths": list(paths),
                            },
                        )
                    )
                    failed.add(assignment.assignment_id)

            # Conflict/delivered phases survive coordinator restart and must be
            # converted back into the same trigger without calling deliver.
            for assignment in plan.assignments:
                run = exact.get(assignment.assignment_id)
                if (
                    run is not None
                    and run.phase == "conflict"
                    and assignment.assignment_id not in failed
                ):
                    triggers.append(
                        RuntimeTrigger(
                            kind="delivery_conflict",
                            subject_id=assignment.assignment_id,
                            detail="product projection conflicts with integration",
                            evidence={
                                "run_id": run.run_id,
                                "report_id": run.report_id,
                                "paths": list(run.conflict_paths),
                            },
                        )
                    )
                    failed.add(assignment.assignment_id)

            # Exact terminal runs without an accepted report are failures even
            # if the process itself exited zero.
            for assignment in plan.assignments:
                run = exact.get(assignment.assignment_id)
                if (
                    run is not None
                    and run.phase == "terminal"
                    and assignment.assignment_id not in failed
                ):
                    triggers.append(
                        RuntimeTrigger(
                            kind="worker_terminal",
                            subject_id=assignment.assignment_id,
                            detail=run.terminal_reason or "terminal worker has no accepted report",
                            evidence={"run_id": run.run_id, "run_sequence": run.sequence},
                        )
                    )
                    failed.add(assignment.assignment_id)

            current_runs = tuple(self.supervisor.runs())
            budget = self._budget(current_runs)
            if self.goal.budget_usd is not None and (
                budget.unknown_run_ids or budget.unknown_planner_attempt_ids
            ):
                triggers.append(
                    RuntimeTrigger(
                        kind="budget_unknown",
                        subject_id=plan.plan_id,
                        detail="one or more incurred planner or worker costs are unknown",
                        evidence=budget.to_dict(),
                    )
                )
            if (
                budget.limit_usd is not None
                and budget.known_spent_usd > budget.limit_usd
            ):
                triggers.append(
                    RuntimeTrigger(
                        kind="budget_exceeded",
                        subject_id=plan.plan_id,
                        detail="known planner and worker cost exceeds the durable goal budget",
                        evidence=budget.to_dict(),
                    )
                )
            exact = self._exact_current_runs(plan, current_runs)
            delivered = {
                assignment_id for assignment_id, run in exact.items() if self._is_delivered(run)
            }
            live = {
                assignment_id
                for assignment_id, run in exact.items()
                if run.phase not in _TERMINAL_CURRENT_PHASES
            }

            signals: tuple[ExternalSignal, ...] = ()
            if self.communication is not None:
                raw_signals = tuple(
                    self.communication.signals(goal=self.goal, plan=plan, runs=current_runs)
                )
                if not all(isinstance(item, ExternalSignal) for item in raw_signals):
                    raise CoordinatorCorruption("communication hook returned an invalid signal")
                if len({item.signal_id for item in raw_signals}) != len(raw_signals):
                    raise CoordinatorCorruption("communication hook returned duplicate signal IDs")
                signals = raw_signals
                for signal in signals:
                    decision_id, root, _ = self._decision_intent(
                        cycle,
                        "external_signal",
                        signal.signal_id,
                        {
                            "kind": signal.kind,
                            "detail": signal.detail,
                            "requires_replan": signal.requires_replan,
                            "metadata": _thaw(signal.metadata),
                        },
                    )
                    self._decision_result(
                        cycle,
                        decision_id,
                        root,
                        "external_signal",
                        signal.signal_id,
                        "observed",
                        {"requires_replan": signal.requires_replan},
                    )
                    if signal.requires_replan:
                        signal_trigger = signal.to_trigger()
                        triggers.append(signal_trigger)
                        self._record_trigger(cycle, signal_trigger)

            # Once a failure/conflict/replan signal exists, no additional
            # spending is authorized.  Already-live independent workers may
            # finish so their certified products can be delivered first.
            started: set[str] = set()
            waiting: set[str] = set()
            if not triggers:
                for assignment in plan.assignments:
                    if assignment.assignment_id in delivered or assignment.assignment_id in exact:
                        continue
                    missing = set(assignment.depends_on) - delivered
                    if missing:
                        waiting.add(assignment.assignment_id)
                        continue
                    if assignment.depends_on:
                        # Assignments are exact records, not templates: their
                        # base and ArtifactRefs were frozen before dependency
                        # products existed.  Mint the downstream assignment in
                        # a fresh world snapshot so it can name those bytes.
                        triggers.append(
                            RuntimeTrigger(
                                kind="dependency_wave_delivered",
                                subject_id=assignment.assignment_id,
                                detail=(
                                    "dependencies are delivered; replan downstream work "
                                    "against the new exact integration state"
                                ),
                                evidence={
                                    "depends_on": list(assignment.depends_on),
                                    "integration_state_id": self.integration.head.id,
                                },
                            )
                        )
                        waiting.add(assignment.assignment_id)
                        break
                    allowed, budget_trigger = self._can_start(assignment, budget)
                    if not allowed:
                        assert budget_trigger is not None
                        triggers.append(budget_trigger)
                        failed.add(assignment.assignment_id)
                        waiting.add(assignment.assignment_id)
                        break
                    try:
                        timeout = self._wall_timeout(assignment, self.default_wall_timeout_seconds)
                        if self._remaining_wall is not None:
                            remaining = self._remaining_wall()
                            if remaining <= 0:
                                raise _WallDeadlineReached("goal deadline elapsed before worker launch")
                            timeout = min(timeout, remaining)
                        prepared = self._perform(
                            cycle,
                            "prepare",
                            assignment.assignment_id,
                            {
                                "assignment_digest": _digest(assignment.to_json()),
                                "wall_timeout_seconds": timeout,
                            },
                            lambda assignment=assignment, timeout=timeout: self.supervisor.prepare(
                                assignment,
                                wall_timeout_seconds=timeout,
                                active_generation=plan.generation,
                            ),
                            lambda value: {
                                "run_id": value.run_id,
                                "prepared_state_id": value.prepared_state_id,
                            },
                        )
                        started_run = self._perform(
                            cycle,
                            "start",
                            prepared.run_id,
                            {"prepared_state_id": prepared.prepared_state_id},
                            lambda run_id=prepared.run_id: self.supervisor.start(
                                run_id, active_generation=plan.generation, deadline_at=self._deadline_at,
                            ),
                            lambda value: {
                                "run_id": value.run_id,
                                "phase": value.phase,
                                "pid": value.pid,
                            },
                        )
                        exact[assignment.assignment_id] = started_run
                        started.add(assignment.assignment_id)
                        if started_run.phase not in _TERMINAL_CURRENT_PHASES:
                            live.add(assignment.assignment_id)
                        # A newly authorized bounded run consumes budget before
                        # considering another assignment in this same cycle.
                        budget = self._budget(self.supervisor.runs())
                    except (ValueError, CoordinatorError, SupervisorStateConflict):
                        raise
                    except Exception as exc:
                        triggers.append(
                            RuntimeTrigger(
                                kind="worker_start_failed",
                                subject_id=assignment.assignment_id,
                                detail=f"{type(exc).__name__}: {exc}"[:1024],
                                evidence={"generation": plan.generation},
                            )
                        )
                        failed.add(assignment.assignment_id)
                        break

            current_runs = tuple(self.supervisor.runs())
            exact = self._exact_current_runs(plan, current_runs)
            delivered = {
                assignment_id for assignment_id, run in exact.items() if self._is_delivered(run)
            }
            live = {
                assignment_id
                for assignment_id, run in exact.items()
                if run.phase not in _TERMINAL_CURRENT_PHASES
            }
            budget = self._budget(current_runs)

            # Preserve exact triggers before either returning to poll live work
            # or invoking the model.  This makes every failure/conflict visible
            # even when the central process dies on the following boundary.
            unique_triggers = tuple(
                sorted(
                    {trigger.trigger_id: trigger for trigger in triggers}.values(),
                    key=lambda item: item.trigger_id,
                )
            )
            for trigger in unique_triggers:
                self._record_trigger(cycle, trigger)

            if live or started:
                outcome = CycleOutcome(
                    cycle_id=cycle.cycle_id,
                    status="running",
                    plan=plan,
                    runs=current_runs,
                    delivered_assignment_ids=tuple(sorted(delivered)),
                    waiting_assignment_ids=tuple(sorted(waiting)),
                    failed_assignment_ids=tuple(sorted(failed)),
                    triggers=unique_triggers,
                    budget=budget,
                )
                return self._complete_cycle(cycle, outcome)

            # A planner call is itself billable.  Once accounting is unknown,
            # a declared ceiling is absent, or the known/reserved total has
            # exhausted the goal limit, asking the model how to recover would
            # authorize more spend while the budget cannot be proven.  Leave
            # the exact current plan and triggers durable for operator action.
            if budget.limit_usd is not None and not budget.enforceable:
                outcome = CycleOutcome(
                    cycle_id=cycle.cycle_id,
                    status="budget_blocked",
                    plan=plan,
                    runs=current_runs,
                    delivered_assignment_ids=tuple(sorted(delivered)),
                    waiting_assignment_ids=tuple(sorted(waiting)),
                    failed_assignment_ids=tuple(sorted(failed)),
                    triggers=unique_triggers,
                    budget=budget,
                )
                return self._complete_cycle(cycle, outcome)

            if not unique_triggers and delivered == {
                item.assignment_id for item in plan.assignments
            }:
                unique_triggers = (
                    RuntimeTrigger(
                        kind="plan_satisfied",
                        subject_id=plan.plan_id,
                        detail="all exact current-generation assignments were delivered",
                        evidence={
                            "integration_state_id": self.integration.head.id,
                            "run_ids": sorted(run.run_id for run in exact.values()),
                        },
                    ),
                )
            elif not unique_triggers:
                blocked = sorted(
                    item.assignment_id
                    for item in plan.assignments
                    if item.assignment_id not in delivered
                )
                unique_triggers = (
                    RuntimeTrigger(
                        kind="dependency_blocked",
                        subject_id=plan.plan_id,
                        detail="no live or runnable assignment can advance the current plan",
                        evidence={"blocked_assignment_ids": blocked},
                    ),
                )

            if max_generations is not None and plan.generation >= max_generations:
                for trigger in unique_triggers:
                    self._record_trigger(cycle, trigger)
                return self._complete_cycle(cycle, CycleOutcome(
                    cycle_id=cycle.cycle_id, status="generation_bound", plan=plan, runs=current_runs,
                    delivered_assignment_ids=tuple(sorted(delivered)),
                    waiting_assignment_ids=tuple(sorted(waiting)),
                    failed_assignment_ids=tuple(sorted(failed)), triggers=unique_triggers, budget=budget,
                ))
            try:
                revised = self._replan(cycle, plan, unique_triggers)
            except BudgetBlocked:
                outcome = CycleOutcome(
                    cycle_id=cycle.cycle_id,
                    status="budget_blocked",
                    plan=plan,
                    runs=current_runs,
                    delivered_assignment_ids=tuple(sorted(delivered)),
                    waiting_assignment_ids=tuple(sorted(waiting)),
                    failed_assignment_ids=tuple(sorted(failed)),
                    triggers=unique_triggers,
                    budget=budget,
                )
                return self._complete_cycle(cycle, outcome)
            revised_runs = tuple(self.supervisor.runs())
            outcome = CycleOutcome(
                cycle_id=cycle.cycle_id,
                status="complete" if revised.complete else "replanned",
                plan=revised,
                runs=revised_runs,
                delivered_assignment_ids=tuple(sorted(delivered)),
                failed_assignment_ids=tuple(sorted(failed)),
                triggers=unique_triggers,
                budget=self._budget(revised_runs),
            )
            return self._complete_cycle(cycle, outcome)
