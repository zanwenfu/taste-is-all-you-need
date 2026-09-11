"""Crash-reconciling process supervision for typed worker assignments.

The central model decides *what* to run.  This module owns the mechanical
boundary around that decision: it makes an exact assignment durable before a
fork, starts it idempotently, enforces a wall-clock deadline outside the worker,
and accepts only a byte- and generation-bound terminal report.

There are deliberately two persistence layers.  Immutable
:class:`~taste.brains.records.LifecycleEvent` objects and a small current-run
snapshot live on the central control branch.  Launcher sidecars contain only
ephemeral process evidence needed to close the fork/record crash window.  A
sidecar can cause a process to be recovered; it can never make work eligible
for delivery without the branch records and exact worker report.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from taste.brains.contract import CONTRACT_PATH
from taste.brains.delivery import DeliveryResult, deliver_product, validate_artifact_path
from taste.brains.monitor import TerminalAssessment
from taste.brains.records import Assignment, LifecycleEvent, WorkerReport, contract_digest
from taste.brains.worker_runtime import ASSIGNMENT_PATH, WORKER_REPORT_PATH
from taste.memstore import Branch, BranchBusy, State, Store

__all__ = [
    "AssignmentIdentityConflict",
    "CentralSupervisor",
    "DeliveryRejected",
    "InvalidWorkerReport",
    "LaunchSpec",
    "ProcessExit",
    "ProcessHandle",
    "ProcessLauncher",
    "StaleGeneration",
    "SubprocessLauncher",
    "SupervisorError",
    "SupervisorLedgerCorruption",
    "SupervisorRun",
    "SupervisorStateConflict",
    "mark_worker_ready",
]

RUN_SCHEMA = "taste.brains/SupervisorRun/1"
RUN_INDEX_SCHEMA = "taste.brains/SupervisorRunIndex/1"
RUN_ROOT = ".taste/supervisor/runs"
RUN_INDEX_PATH = f"{RUN_ROOT}/index.json"
_EXACT_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_STABLE_ID = re.compile(r"[^\x00-\x1f\x7f]{1,256}\Z")
_SAFE_TOKEN = re.compile(r"[^a-z0-9_.-]+")
_TERMINAL_PHASES = frozenset({"terminal", "report_accepted", "delivered", "conflict"})
_PROCESS_MEMBERS_SCHEMA = "taste.brains/SubprocessMembers/1"
_RUN_ID_ENV = "TASTE_WORKER_RUN_ID"
_LAUNCH_TOKEN_ENV = "TASTE_WORKER_LAUNCH_TOKEN"
_READY_PATH_ENV = "TASTE_WORKER_READY_PATH"


class SupervisorError(RuntimeError):
    """Base class for deterministic supervision failures."""


class SupervisorStateConflict(SupervisorError):
    """A transition was based on a run snapshot that is no longer current."""


class SupervisorLedgerCorruption(SupervisorError):
    """The append-only supervisor run index was deleted or rewritten."""


class AssignmentIdentityConflict(SupervisorError):
    """A stable run or worker identity already denotes different work."""


class StaleGeneration(SupervisorError):
    """An old plan generation attempted to start, report, or deliver work."""


class InvalidWorkerReport(SupervisorError):
    """A terminal report is missing, malformed, stale, or not exact-state bound."""


class DeliveryRejected(SupervisorError):
    """A report does not meet the fail-closed product delivery gate."""


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must include a timezone")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp must be non-empty text")
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except ValueError as exc:
        raise ValueError("timestamp is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp lacks timezone")
    return parsed.astimezone(UTC)


def _digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_id(assignment: Assignment) -> str:
    token = hashlib.sha256(assignment.to_json().encode("utf-8")).hexdigest()
    return f"worker-run.{token}"


def _run_key(run_id: str) -> str:
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest()


def _run_path(run_id: str) -> str:
    return f"{RUN_ROOT}/{_run_key(run_id)}/run.json"


def _event_path(run_id: str, sequence: int, kind: str) -> str:
    return f"{RUN_ROOT}/{_run_key(run_id)}/events/{sequence:06d}-{kind}.json"


def _reason_token(value: str, fallback: str = "stopped") -> str:
    token = _SAFE_TOKEN.sub("_", value.strip().lower()).strip("_.-")
    return token or fallback


def _json_value(value: Any) -> Any:
    """Thaw records' recursively frozen extension maps."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _assert_safe_write_destination(root: Path, relative: str) -> None:
    """Refuse every symlink/special node on a host-owned write path.

    Preparation owns the branch lease, so after this preflight there is no
    legitimate concurrent writer.  Checking all paths before the first write
    also means a bad second input cannot leave a partial projection behind.
    """
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or ".." in parts:
        raise AssignmentIdentityConflict(f"unsafe worker destination {relative!r}")
    current = root
    for index, part in enumerate(parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            # No deeper component can exist without this ancestor.
            return
        if stat.S_ISLNK(mode):
            raise AssignmentIdentityConflict(
                f"worker destination {relative!r} traverses symlink {current.relative_to(root)!s}"
            )
        if index < len(parts) - 1 and not stat.S_ISDIR(mode):
            raise AssignmentIdentityConflict(
                f"worker destination {relative!r} has a non-directory ancestor"
            )
        if index == len(parts) - 1 and not stat.S_ISREG(mode):
            raise AssignmentIdentityConflict(
                f"worker destination {relative!r} is not a regular file"
            )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(temporary, "xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


@contextlib.contextmanager
def _file_lock(path: Path):
    """Cross-process lock used only around the short fork handshake."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _members_path(launch_path: Path) -> Path:
    return launch_path.with_name(f"{launch_path.name}.members.json")


def _read_member_record(
    path: Path,
    *,
    run_id: str,
    launch_token: str,
    root_pid: int,
    root_identity: str,
) -> tuple[tuple[int, str], ...]:
    """Load the exact, durable set of process births observed for one run."""

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ()
    except (OSError, UnicodeError) as exc:
        raise SupervisorError("durable process-member evidence is unreadable") from exc

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SupervisorError(
                    f"durable process-member evidence repeats field {key!r}"
                )
            result[key] = value
        return result

    try:
        raw = json.loads(text, object_pairs_hook=no_duplicates)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise SupervisorError("durable process-member evidence is malformed") from exc
    expected_fields = {
        "schema",
        "run_id",
        "launch_token",
        "root_pid",
        "root_process_identity",
        "members",
    }
    if not isinstance(raw, dict) or set(raw) != expected_fields:
        raise SupervisorError("durable process-member evidence has the wrong fields")
    if (
        raw["schema"] != _PROCESS_MEMBERS_SCHEMA
        or raw["run_id"] != run_id
        or raw["launch_token"] != launch_token
        or raw["root_pid"] != root_pid
        or raw["root_process_identity"] != root_identity
        or not isinstance(raw["members"], list)
    ):
        raise SupervisorError("durable process-member evidence belongs to another launch")

    members: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for item in raw["members"]:
        if not isinstance(item, dict) or set(item) != {"pid", "process_identity"}:
            raise SupervisorError("durable process-member entry is malformed")
        pid = item["pid"]
        identity = item["process_identity"]
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or not isinstance(identity, str)
            or not identity
        ):
            raise SupervisorError("durable process-member identity is malformed")
        pair = (pid, identity)
        if pair in seen:
            raise SupervisorError("durable process-member evidence contains a duplicate")
        seen.add(pair)
        members.append(pair)
    if (root_pid, root_identity) not in seen:
        raise SupervisorError("durable process-member evidence omitted its launch root")
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    if text != canonical:
        raise SupervisorError("durable process-member evidence is not canonical")
    return tuple(members)


def _remember_member_births(
    path: Path,
    *,
    run_id: str,
    launch_token: str,
    root_pid: int,
    root_identity: str,
    members: Mapping[int, str],
) -> tuple[tuple[int, str], ...]:
    """Union newly observed births into a crash-safe per-run sidecar."""

    lock_path = path.with_name(f"{path.name}.lock")
    with _file_lock(lock_path):
        existing = _read_member_record(
            path,
            run_id=run_id,
            launch_token=launch_token,
            root_pid=root_pid,
            root_identity=root_identity,
        )
        combined = set(existing)
        combined.add((root_pid, root_identity))
        combined.update(members.items())
        ordered = tuple(sorted(combined))
        if ordered != existing:
            _atomic_json(
                path,
                {
                    "schema": _PROCESS_MEMBERS_SCHEMA,
                    "run_id": run_id,
                    "launch_token": launch_token,
                    "root_pid": root_pid,
                    "root_process_identity": root_identity,
                    "members": [
                        {"pid": pid, "process_identity": identity}
                        for pid, identity in ordered
                    ],
                },
            )
        return ordered


@dataclass(frozen=True, slots=True)
class ProcessExit:
    """Observed process termination; ``None`` means evidence is unavailable."""

    exit_code: int | None = None
    signal: int | None = None
    reaped: bool = False

    def __post_init__(self) -> None:
        if self.exit_code is not None and self.signal is not None:
            raise ValueError("exit_code and signal are mutually exclusive")


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    """Exact process-boundary input given to an idempotent launcher."""

    run_id: str
    assignment: Assignment
    prepared_state_id: str
    worktree: Path
    deadline_at: str
    readiness_path: Path
    launch_path: Path


@runtime_checkable
class ProcessHandle(Protocol):
    """Live or recovered process evidence used by the supervisor."""

    @property
    def pid(self) -> int: ...

    @property
    def process_group_id(self) -> int: ...

    @property
    def launch_token(self) -> str: ...

    @property
    def process_identity(self) -> str: ...

    def poll(self) -> ProcessExit | None: ...

    def ready(self) -> bool: ...

    def terminate_tree(self, grace_seconds: float) -> ProcessExit: ...


@runtime_checkable
class ProcessLauncher(Protocol):
    """A launcher must make ``launch`` idempotent for one ``run_id``."""

    def launch(self, spec: LaunchSpec) -> ProcessHandle: ...

    def recover(self, spec: LaunchSpec) -> ProcessHandle | None: ...


@dataclass(frozen=True, slots=True)
class SupervisorRun:
    """Durable current view of one exact assignment attempt."""

    run_id: str
    assignment: Assignment
    assignment_digest: str
    prepared_state_id: str
    wall_timeout_seconds: float
    phase: str = "prepared"
    sequence: int = 0
    created_at: str = ""
    deadline_at: str | None = None
    pid: int | None = None
    process_group_id: int | None = None
    launch_token: str | None = None
    process_identity: str | None = None
    ready_at: str | None = None
    terminal_reason: str = ""
    exit_code: int | None = None
    signal: int | None = None
    reaped: bool = False
    recovery_state_id: str | None = None
    report_id: str | None = None
    report_state_id: str | None = None
    integration_state_id: str | None = None
    conflict_paths: tuple[str, ...] = ()
    uncertain: bool = False
    uncertainty_reasons: tuple[str, ...] = ()

    @property
    def terminal(self) -> bool:
        return self.phase in _TERMINAL_PHASES

    @property
    def ready(self) -> bool:
        """Readiness is current only while the process phase is live."""
        return self.phase == "ready" and self.ready_at is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RUN_SCHEMA,
            "run_id": self.run_id,
            "assignment": self.assignment.to_dict(),
            "assignment_digest": self.assignment_digest,
            "prepared_state_id": self.prepared_state_id,
            "wall_timeout_seconds": self.wall_timeout_seconds,
            "phase": self.phase,
            "sequence": self.sequence,
            "created_at": self.created_at,
            "deadline_at": self.deadline_at,
            "pid": self.pid,
            "process_group_id": self.process_group_id,
            "launch_token": self.launch_token,
            "process_identity": self.process_identity,
            "ready_at": self.ready_at,
            "terminal_reason": self.terminal_reason,
            "exit_code": self.exit_code,
            "signal": self.signal,
            "reaped": self.reaped,
            "recovery_state_id": self.recovery_state_id,
            "report_id": self.report_id,
            "report_state_id": self.report_state_id,
            "integration_state_id": self.integration_state_id,
            "conflict_paths": list(self.conflict_paths),
            "uncertain": self.uncertain,
            "uncertainty_reasons": list(self.uncertainty_reasons),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> SupervisorRun:
        if not isinstance(raw, dict) or raw.get("schema") != RUN_SCHEMA:
            raise ValueError("invalid supervisor run record")
        required = {
            "schema",
            "run_id",
            "assignment",
            "assignment_digest",
            "prepared_state_id",
            "wall_timeout_seconds",
            "phase",
            "sequence",
            "created_at",
            "deadline_at",
            "pid",
            "process_group_id",
            "launch_token",
            "process_identity",
            "ready_at",
            "terminal_reason",
            "exit_code",
            "signal",
            "reaped",
            "recovery_state_id",
            "report_id",
            "report_state_id",
            "integration_state_id",
            "conflict_paths",
            "uncertain",
            "uncertainty_reasons",
        }
        if set(raw) != required:
            raise ValueError("supervisor run record has missing or unknown fields")
        assignment = Assignment.from_dict(raw["assignment"])
        if (
            isinstance(raw["wall_timeout_seconds"], bool)
            or not isinstance(raw["wall_timeout_seconds"], (int, float))
        ):
            raise ValueError("wall_timeout_seconds must be numeric")
        if not isinstance(raw["conflict_paths"], list) or not isinstance(
            raw["uncertainty_reasons"], list
        ):
            raise ValueError("supervisor run tuple fields must be JSON arrays")
        record = cls(
            run_id=raw["run_id"],
            assignment=assignment,
            assignment_digest=raw["assignment_digest"],
            prepared_state_id=raw["prepared_state_id"],
            wall_timeout_seconds=raw["wall_timeout_seconds"],
            phase=raw["phase"],
            sequence=raw["sequence"],
            created_at=raw["created_at"],
            deadline_at=raw["deadline_at"],
            pid=raw["pid"],
            process_group_id=raw["process_group_id"],
            launch_token=raw["launch_token"],
            process_identity=raw["process_identity"],
            ready_at=raw["ready_at"],
            terminal_reason=raw["terminal_reason"],
            exit_code=raw["exit_code"],
            signal=raw["signal"],
            reaped=raw["reaped"],
            recovery_state_id=raw["recovery_state_id"],
            report_id=raw["report_id"],
            report_state_id=raw["report_state_id"],
            integration_state_id=raw["integration_state_id"],
            conflict_paths=tuple(raw["conflict_paths"]),
            uncertain=raw["uncertain"],
            uncertainty_reasons=tuple(raw["uncertainty_reasons"]),
        )
        record._validate()
        if record.sequence < 1:
            raise ValueError("durable supervisor sequence must be positive")
        return record

    @classmethod
    def from_json(cls, text: str) -> SupervisorRun:
        def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError(f"supervisor run record has duplicate key {key!r}")
                value[key] = item
            return value

        def no_constant(value: str) -> Any:
            raise ValueError(f"supervisor run record contains non-JSON number {value}")

        try:
            return cls.from_dict(
                json.loads(
                    text,
                    object_pairs_hook=no_duplicates,
                    parse_constant=no_constant,
                )
            )
        except json.JSONDecodeError as exc:
            raise ValueError("supervisor run record is not JSON") from exc

    def _validate(self) -> None:
        if self.run_id != _run_id(self.assignment):
            raise ValueError("run_id does not match the exact assignment")
        if self.assignment_digest != _digest(self.assignment.to_json()):
            raise ValueError("assignment_digest does not match assignment")
        if _EXACT_OBJECT_ID.fullmatch(self.prepared_state_id) is None:
            raise ValueError("prepared_state_id is not exact")
        if (
            isinstance(self.wall_timeout_seconds, bool)
            or not isinstance(self.wall_timeout_seconds, (int, float))
            or not 0 < float(self.wall_timeout_seconds) < float("inf")
        ):
            raise ValueError("wall timeout must be finite and positive")
        if self.phase not in {
            "prepared",
            "spawn_intent",
            "spawned",
            "ready",
            "terminal",
            "report_accepted",
            "delivered",
            "conflict",
        }:
            raise ValueError("unknown supervisor phase")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        _parse_time(self.created_at)
        if self.deadline_at is not None:
            _parse_time(self.deadline_at)
        for name in ("pid", "process_group_id"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or null")
        process_values = (
            self.pid,
            self.process_group_id,
            self.launch_token,
            self.process_identity,
        )
        if any(value is not None for value in process_values) and not all(
            value is not None for value in process_values
        ):
            raise ValueError("process identity fields must be present together")
        for name in ("launch_token", "process_identity", "report_id"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str)
                or _STABLE_ID.fullmatch(value) is None
                or value != value.strip()
            ):
                raise ValueError(f"{name} must be a stable identifier or null")
        for name in (
            "recovery_state_id",
            "report_state_id",
            "integration_state_id",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or _EXACT_OBJECT_ID.fullmatch(value) is None
            ):
                raise ValueError(f"{name} must be an exact object ID or null")
        if not isinstance(self.terminal_reason, str) or (
            self.terminal_reason
            and (
                _STABLE_ID.fullmatch(self.terminal_reason) is None
                or self.terminal_reason != self.terminal_reason.strip()
            )
        ):
            raise ValueError("terminal_reason must be a stable token or empty")
        for name in ("exit_code", "signal"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or null")
        if self.signal == 0:
            raise ValueError("signal must be positive when present")
        if self.exit_code is not None and self.signal is not None:
            raise ValueError("exit code and signal are mutually exclusive")
        if not isinstance(self.reaped, bool) or not isinstance(self.uncertain, bool):
            raise ValueError("reaped and uncertain must be booleans")
        if not isinstance(self.conflict_paths, tuple) or not all(
            isinstance(path, str) for path in self.conflict_paths
        ):
            raise ValueError("conflict_paths must be a tuple of paths")
        for path in self.conflict_paths:
            validate_artifact_path(path)
        if len(set(self.conflict_paths)) != len(self.conflict_paths):
            raise ValueError("conflict_paths must be unique")
        if not isinstance(self.uncertainty_reasons, tuple) or not all(
            isinstance(reason, str)
            and bool(reason)
            and _STABLE_ID.fullmatch(reason) is not None
            for reason in self.uncertainty_reasons
        ):
            raise ValueError("uncertainty_reasons must be stable non-empty strings")
        if len(set(self.uncertainty_reasons)) != len(self.uncertainty_reasons):
            raise ValueError("uncertainty_reasons must be unique")
        if self.uncertain != bool(self.uncertainty_reasons):
            raise ValueError("uncertainty flag and reasons disagree")

        terminal = self.phase in _TERMINAL_PHASES
        if terminal != bool(self.terminal_reason):
            raise ValueError("terminal phase and terminal_reason must be set together")
        if self.phase in {"spawn_intent", "spawned", "ready"} and self.deadline_at is None:
            raise ValueError("a launched phase requires a durable deadline")
        if self.phase in {"spawned", "ready"} and self.pid is None:
            raise ValueError("a live process phase requires exact process identity")
        if self.phase == "ready" and self.ready_at is None:
            raise ValueError("ready phase requires ready_at")
        if self.ready_at is not None:
            _parse_time(self.ready_at)
            if self.pid is None:
                raise ValueError("ready_at requires exact process identity")
        if not terminal and any(
            value is not None for value in (self.exit_code, self.signal, self.recovery_state_id)
        ):
            raise ValueError("non-terminal phase contains terminal process evidence")
        if not terminal and self.reaped:
            raise ValueError("a non-terminal process cannot be reaped")
        if terminal and self.recovery_state_id is None:
            raise ValueError("terminal phase requires an exact recovery state")

        reported = self.phase in {"report_accepted", "delivered", "conflict"}
        if reported != (self.report_id is not None and self.report_state_id is not None):
            raise ValueError("report phase and exact report identity disagree")
        if (self.report_id is None) != (self.report_state_id is None):
            raise ValueError("report_id and report_state_id must be present together")
        if (self.phase == "delivered") != (self.integration_state_id is not None):
            raise ValueError("delivered phase and integration_state_id disagree")
        if (self.phase == "conflict") != bool(self.conflict_paths):
            raise ValueError("conflict phase and conflict_paths disagree")


class _SubprocessHandle:
    def __init__(
        self,
        *,
        spec: LaunchSpec,
        pid: int,
        process_group_id: int,
        launch_token: str,
        process_identity: str,
        popen: subprocess.Popen[bytes] | None,
    ) -> None:
        self.spec = spec
        self._pid = pid
        self._pgid = process_group_id
        self._launch_token = launch_token
        self._identity = process_identity
        self._popen = popen
        # Every signal remains birth-fenced.  Once a descendant has been
        # observed while connected to the exact launch root/group, retain its
        # identity so it cannot escape by reparenting or changing groups.  The
        # complete observed set is also persisted: reconstructing a supervisor
        # must not turn a previously fenced daemon back into an unknown PID.
        self._member_path = _members_path(spec.launch_path)
        member_record_existed = self._member_path.exists()
        self._cleanup_uncertain = popen is None and not member_record_existed
        self._member_births = set(
            _remember_member_births(
                self._member_path,
                run_id=spec.run_id,
                launch_token=launch_token,
                root_pid=pid,
                root_identity=process_identity,
                members={pid: process_identity},
            )
        )
        self._known_members: dict[int, str] = {pid: process_identity}
        self._observed_root_live = False
        self._root_exit_observed = False

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def process_group_id(self) -> int:
        return self._pgid

    @property
    def launch_token(self) -> str:
        return self._launch_token

    @property
    def process_identity(self) -> str:
        return self._identity

    def _same_process(self) -> bool:
        # Prefer the independently observable birth identity.  The inherited
        # flock is run-liveness evidence, but a child can retain it after the
        # root exits; treating that as root liveness would postpone descendant
        # cleanup until the wall deadline.  On known local platforms, failure
        # to observe the exact root is terminal and therefore fail-closed.
        if sys.platform == "darwin":
            info = _darwin_process_info(self.pid)
            return info is not None and info[2] == self.process_identity
        if sys.platform.startswith("linux"):
            info = _linux_process_info(self.pid)
            return info is not None and info[2] == self.process_identity

        # The bootstrap holds this exact-run flock across exec.  It is the
        # conservative fallback on hosts where a birth identity is not
        # available.
        lock_path = self.spec.launch_path.with_name(
            f"{self.spec.launch_path.name}.bootstrap.lock"
        )
        observed = _process_identity(self.pid)
        return (
            observed == self.process_identity
            if observed
            else _lock_is_held(lock_path)
        )

    def poll(self) -> ProcessExit | None:
        if self._popen is not None:
            code = self._popen.poll()
            if code is not None:
                self._root_exit_observed = True
                self._observe_members(root_live=False, discover_markers=True)
                return _exit_from_returncode(code, reaped=True)
            # This Popen observation is an exact live-root boundary.  Discover
            # ordinary descendants while their ancestry is still available.
            self._observe_members(root_live=True)
            code = self._popen.poll()
            if code is None:
                return None
            self._root_exit_observed = True
            self._observe_members(root_live=False, discover_markers=True)
            return _exit_from_returncode(code, reaped=True)
        live = self._same_process()
        self._observe_members(root_live=live, discover_markers=not live)
        if live:
            return None
        self._root_exit_observed = True
        return ProcessExit(reaped=False)

    def _refresh_persisted_members(
        self, table: Mapping[int, tuple[int, int, str]]
    ) -> None:
        for pid, identity in self._member_births:
            row = table.get(pid)
            if row is not None and row[2] == identity:
                self._known_members[pid] = identity

    def _remember_members(self, members: Mapping[int, str]) -> None:
        if not self._member_path.exists():
            # Recreate best-effort evidence from this handle, but never treat a
            # vanished durable record as proof that no earlier births were lost.
            self._cleanup_uncertain = True
        combined = dict(self._known_members)
        combined.update(members)
        self._member_births = set(
            _remember_member_births(
                self._member_path,
                run_id=self.spec.run_id,
                launch_token=self.launch_token,
                root_pid=self.pid,
                root_identity=self.process_identity,
                members=combined,
            )
        )
        self._known_members.update(members)

    def _observe_members(
        self,
        *,
        root_live: bool | None = None,
        discover_markers: bool = False,
    ) -> dict[int, str]:
        if root_live is None:
            root_live = not self._root_exit_observed and self._same_process()
        if root_live:
            self._observed_root_live = True
        elif not self._observed_root_live:
            # The root disappeared before any live-root/tree observation.  An
            # inherited marker can still find and kill ordinary detached
            # children, but it cannot prove that no child deliberately erased
            # the marker.  Keep the terminal result uncertain.
            self._cleanup_uncertain = True
        table = _process_table()
        self._refresh_persisted_members(table)
        observed, _safe_group = _tree_members(
            self.pid,
            self.process_group_id,
            self.process_identity,
            allow_orphaned_group=True,
            known=self._known_members,
            table=table,
        )
        if discover_markers:
            marked, discovery_available = _marked_run_members(
                table,
                run_id=self.spec.run_id,
                launch_token=self.launch_token,
            )
            observed.update(marked)
            if not discovery_available:
                self._cleanup_uncertain = True
        self._remember_members(observed)
        return dict(self._known_members)

    def _cleanup_result(self, result: ProcessExit) -> ProcessExit:
        return replace(result, reaped=False) if self._cleanup_uncertain else result

    def ready(self) -> bool:
        try:
            raw = json.loads(self.spec.readiness_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return False
        return bool(
            isinstance(raw, dict)
            and set(raw) == {"schema", "run_id", "launch_token", "pid"}
            and raw.get("schema") == "taste.brains/WorkerReady/1"
            and raw.get("run_id") == self.spec.run_id
            and raw.get("launch_token") == self.launch_token
            and raw.get("pid") == self.pid
        )

    def terminate_tree(self, grace_seconds: float) -> ProcessExit:
        return _terminate_tree(self, grace_seconds)


def _exit_from_returncode(returncode: int, *, reaped: bool) -> ProcessExit:
    if returncode < 0:
        return ProcessExit(signal=-returncode, reaped=reaped)
    return ProcessExit(exit_code=returncode, reaped=reaped)


def _lock_is_held(path: Path) -> bool:
    import fcntl

    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def _process_identity(pid: int) -> str:
    """Return a PID-reuse fence from the OS process table."""
    darwin = _darwin_process_info(pid)
    if darwin is not None:
        return darwin[2]
    linux = _linux_process_info(pid)
    if linux is not None:
        return linux[2]
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    stamp = result.stdout.strip()
    return _digest(f"{pid}\0{stamp}") if result.returncode == 0 and stamp else ""


def _darwin_process_info(pid: int) -> tuple[int, int, str] | None:
    if sys.platform != "darwin":
        return None
    try:
        import ctypes

        class ProcBsdInfo(ctypes.Structure):
            _fields_ = [
                ("flags", ctypes.c_uint32),
                ("status", ctypes.c_uint32),
                ("xstatus", ctypes.c_uint32),
                ("pid", ctypes.c_uint32),
                ("ppid", ctypes.c_uint32),
                ("uid", ctypes.c_uint32),
                ("gid", ctypes.c_uint32),
                ("ruid", ctypes.c_uint32),
                ("rgid", ctypes.c_uint32),
                ("svuid", ctypes.c_uint32),
                ("svgid", ctypes.c_uint32),
                ("rfu", ctypes.c_uint32),
                ("comm", ctypes.c_char * 16),
                ("name", ctypes.c_char * 32),
                ("nfiles", ctypes.c_uint32),
                ("pgid", ctypes.c_uint32),
                ("pjobc", ctypes.c_uint32),
                ("tdev", ctypes.c_uint32),
                ("tpgid", ctypes.c_uint32),
                ("nice", ctypes.c_int32),
                ("start_sec", ctypes.c_uint64),
                ("start_usec", ctypes.c_uint64),
            ]

        library = ctypes.CDLL("/usr/lib/libproc.dylib")
        library.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        library.proc_pidinfo.restype = ctypes.c_int
        info = ProcBsdInfo()
        copied = library.proc_pidinfo(
            pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info)
        )
        if copied != ctypes.sizeof(info) or info.pid != pid:
            return None
        identity = _digest(f"{pid}\0{info.start_sec}\0{info.start_usec}")
        return int(info.ppid), int(info.pgid), identity
    except (OSError, TypeError, ValueError):
        return None


def _linux_process_info(pid: int) -> tuple[int, int, str] | None:
    if not sys.platform.startswith("linux"):
        return None
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = raw[raw.rindex(")") + 2 :].split()
        parent = int(fields[1])
        group = int(fields[2])
        start_ticks = fields[19]
        boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except (FileNotFoundError, OSError, UnicodeError, ValueError, IndexError):
        return None
    return parent, group, _digest(f"{pid}\0{boot}\0{start_ticks}")


def _process_table() -> dict[int, tuple[int, int, str]]:
    if sys.platform == "darwin":
        try:
            import ctypes

            library = ctypes.CDLL("/usr/lib/libproc.dylib")
            library.proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]
            library.proc_listallpids.restype = ctypes.c_int
            capacity = max(1024, library.proc_listallpids(None, 0) * 2)
            pids = (ctypes.c_int * capacity)()
            count = library.proc_listallpids(pids, ctypes.sizeof(pids))
            table = {}
            for pid in pids[: max(0, count)]:
                if pid <= 0:
                    continue
                info = _darwin_process_info(pid)
                if info is not None:
                    table[pid] = info
            return table
        except (OSError, TypeError, ValueError):
            pass
    if sys.platform.startswith("linux"):
        table = {}
        with contextlib.suppress(OSError):
            for path in Path("/proc").iterdir():
                if not path.name.isdigit():
                    continue
                pid = int(path.name)
                info = _linux_process_info(pid)
                if info is not None:
                    table[pid] = info
        if table:
            return table
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pgid=,lstart="],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    table: dict[int, tuple[int, int, str]] = {}
    for line in result.stdout.splitlines():
        fields = line.split(None, 3)
        if len(fields) != 4:
            continue
        with contextlib.suppress(ValueError):
            pid, ppid, pgid = (int(fields[index]) for index in range(3))
            table[pid] = (ppid, pgid, _digest(f"{pid}\0{fields[3]}"))
    return table


def _marker_entries_match(
    entries: Sequence[bytes], *, run_id: str, launch_token: str
) -> bool:
    expected = {
        f"{_RUN_ID_ENV}={run_id}".encode("utf-8", "surrogatepass"),
        f"{_LAUNCH_TOKEN_ENV}={launch_token}".encode("utf-8", "surrogatepass"),
    }
    return expected.issubset(set(entries))


def _darwin_process_has_marker(
    pid: int, *, run_id: str, launch_token: str
) -> bool | None:
    """Read one process's initial argv/environment through KERN_PROCARGS2."""

    if sys.platform != "darwin":
        return None
    try:
        import ctypes

        library = ctypes.CDLL(None, use_errno=True)
        library.sysctl.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        library.sysctl.restype = ctypes.c_int
        mib = (ctypes.c_int * 3)(1, 49, pid)  # CTL_KERN, KERN_PROCARGS2, pid
        size = ctypes.c_size_t()
        if library.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0 or size.value < 4:
            return None
        buffer = ctypes.create_string_buffer(size.value)
        if (
            library.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0) != 0
            or size.value < 4
        ):
            return None
        raw = bytes(buffer.raw[: size.value])
        integer_size = ctypes.sizeof(ctypes.c_int)
        argc = int.from_bytes(raw[:integer_size], sys.byteorder, signed=True)
        if argc < 0:
            return None
        cursor = integer_size
        executable_end = raw.find(b"\0", cursor)
        if executable_end < 0:
            return None
        cursor = executable_end + 1
        while cursor < len(raw) and raw[cursor] == 0:
            cursor += 1
        for _ in range(argc):
            argument_end = raw.find(b"\0", cursor)
            if argument_end < 0:
                return None
            cursor = argument_end + 1
        return _marker_entries_match(
            tuple(part for part in raw[cursor:].split(b"\0") if part),
            run_id=run_id,
            launch_token=launch_token,
        )
    except (OSError, OverflowError, TypeError, ValueError):
        return None


def _process_has_marker(pid: int, *, run_id: str, launch_token: str) -> bool | None:
    """Whether a process inherited this exact run's unguessable launch marker."""

    if sys.platform.startswith("linux"):
        try:
            raw = Path(f"/proc/{pid}/environ").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            return False
        except OSError:
            return None
        return _marker_entries_match(
            tuple(part for part in raw.split(b"\0") if part),
            run_id=run_id,
            launch_token=launch_token,
        )
    if sys.platform == "darwin":
        return _darwin_process_has_marker(
            pid,
            run_id=run_id,
            launch_token=launch_token,
        )
    return None


def _marked_run_members(
    table: Mapping[int, tuple[int, int, str]],
    *,
    run_id: str,
    launch_token: str,
) -> tuple[dict[int, str], bool]:
    """Discover detached descendants which retained the exact launch marker.

    The marker is a discovery seam, not authority to signal a numeric PID:
    callers persist and later re-check the independently observed birth value.
    A process can intentionally clear its environment, so a scan cannot prove
    containment; the unobserved-root path remains explicitly uncertain.
    """

    supported = sys.platform == "darwin" or sys.platform.startswith("linux")
    if not supported or not table:
        return {}, False
    found: dict[int, str] = {}
    for pid, (_parent, _group, identity) in table.items():
        if pid == os.getpid():
            continue
        if _process_has_marker(pid, run_id=run_id, launch_token=launch_token) is True:
            found[pid] = identity
    return found, True


def _tree_members(
    root_pid: int,
    process_group_id: int,
    root_identity: str,
    *,
    allow_orphaned_group: bool,
    known: Mapping[int, str] | None = None,
    table: Mapping[int, tuple[int, int, str]] | None = None,
) -> tuple[dict[int, str], bool]:
    table = dict(table) if table is not None else _process_table()
    known = known or {}
    live_known = {
        pid
        for pid, identity in known.items()
        if pid in table and table[pid][2] == identity
    }
    root = table.get(root_pid)
    if root is not None and root[2] != root_identity:
        # The PID has been reused.  Neither it nor a group carrying its numeric
        # id is evidence of this run.  Already birth-fenced descendants remain
        # owned, however, and must not escape just because the root was reused.
        descendants = set(live_known)
        changed = True
        while changed:
            changed = False
            for pid, (parent, _group, _identity) in table.items():
                if parent in descendants and pid not in descendants:
                    descendants.add(pid)
                    changed = True
        return (
            {
                pid: table[pid][2]
                for pid in descendants
                if pid in table and pid != os.getpid()
            },
            False,
        )
    if root is None and not allow_orphaned_group:
        return {pid: table[pid][2] for pid in live_known}, False
    descendants = {root_pid} if root is not None else set(live_known)
    # A numeric process-group ID is not durable ownership evidence after its
    # leader disappears: it can eventually be reused.  We may follow that
    # group only while the exact root, or an already birth-fenced member of
    # the original group, is still observable.
    safe_group = root is not None or any(
        table[pid][1] == process_group_id for pid in live_known
    )
    if root is None and not descendants:
        return {}, False
    changed = True
    while changed:
        changed = False
        for pid, (parent, _group, _identity) in table.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    # A child may have outlived and been reparented after the root exited.  Its
    # inherited process group is still exact launch evidence.
    if safe_group:
        descendants.update(
            pid
            for pid, (_parent, group, _identity) in table.items()
            if group == process_group_id
        )
    return (
        {
            pid: table[pid][2]
            for pid in descendants
            if pid in table and pid != os.getpid()
        },
        safe_group,
    )


def _signal_exact(members: Mapping[int, str], sig: int) -> None:
    current = _process_table()
    for pid in sorted(members, reverse=True):
        row = current.get(pid)
        if row is None or row[2] != members[pid]:
            continue
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, sig)


def _terminate_tree(handle: _SubprocessHandle, grace_seconds: float) -> ProcessExit:
    if grace_seconds < 0:
        raise ValueError("termination grace must not be negative")
    allow_orphaned = handle._popen is not None or _lock_is_held(
        handle.spec.launch_path.with_name(
            f"{handle.spec.launch_path.name}.bootstrap.lock"
        )
    )
    members = dict(handle._known_members)
    # Scan the inherited run marker even while the root is live: a child may
    # already have changed both its session and parent before the first ordinary
    # ancestry observation.  Every discovered birth is persisted before the
    # first signal.
    members.update(handle._observe_members(discover_markers=True))
    _signal_exact(members, signal.SIGTERM)

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        observed = handle.poll()
        members.update(handle._known_members)
        remaining, _safe = _tree_members(
            handle.pid,
            handle.process_group_id,
            handle.process_identity,
            allow_orphaned_group=allow_orphaned,
            known=members,
        )
        handle._remember_members(remaining)
        members.update(remaining)
        _signal_exact(remaining, signal.SIGTERM)
        if observed is not None and not remaining:
            return handle._cleanup_result(observed)
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))

    # Keep the original identities so a daemon that detached from the launch
    # group cannot escape merely because its parent died during the grace.
    latest, _latest_safe = _tree_members(
        handle.pid,
        handle.process_group_id,
        handle.process_identity,
        allow_orphaned_group=allow_orphaned,
        known=members,
    )
    handle._remember_members(latest)
    members = {**members, **latest}
    _signal_exact(members, signal.SIGKILL)

    if handle._popen is not None:
        try:
            result = _exit_from_returncode(handle._popen.wait(timeout=5), reaped=True)
        except subprocess.TimeoutExpired:
            return handle._cleanup_result(ProcessExit(reaped=False))
        end = time.monotonic() + 5
        while time.monotonic() < end:
            # The root or another child can create and detach a final process
            # while handling SIGTERM.  Re-scan before declaring the marked set
            # empty, then persist its birth before sending SIGKILL.
            members.update(
                handle._observe_members(root_live=False, discover_markers=True)
            )
            remaining, _safe = _tree_members(
                handle.pid,
                handle.process_group_id,
                handle.process_identity,
                allow_orphaned_group=allow_orphaned,
                known=members,
            )
            handle._remember_members(remaining)
            members.update(remaining)
            if not remaining:
                return handle._cleanup_result(result)
            _signal_exact(remaining, signal.SIGKILL)
            time.sleep(0.02)
        return handle._cleanup_result(replace(result, reaped=False))
    # A restarted supervisor is not the orphan's parent and cannot waitpid it.
    # Still verify and repeatedly signal every persisted birth; after the root
    # vanishes, repeat marker discovery so a final detached child is not
    # forgotten merely because the supervising Python process restarted.
    end = time.monotonic() + 5
    while time.monotonic() < end:
        root_live = handle._same_process()
        if not root_live:
            members.update(
                handle._observe_members(root_live=False, discover_markers=True)
            )
        remaining, _safe = _tree_members(
            handle.pid,
            handle.process_group_id,
            handle.process_identity,
            allow_orphaned_group=allow_orphaned,
            known=members,
        )
        handle._remember_members(remaining)
        members.update(remaining)
        if not root_live and not remaining:
            break
        _signal_exact(remaining, signal.SIGKILL)
        time.sleep(0.02)
    return handle._cleanup_result(ProcessExit(reaped=False))


class SubprocessLauncher:
    """POSIX process-group launcher with a crash-safe fork handshake.

    ``command`` may be a fixed argv or a function of :class:`LaunchSpec`.
    Worker entrypoints should call :func:`mark_worker_ready` from the runtime's
    ready callback.  Stdout and stderr are inherited so this mechanism never
    creates an unflushed private log that is mistaken for durable evidence.

    A prelaunch token is fsync'd before :class:`subprocess.Popen`.  Popen starts
    a tiny bootstrap, not the worker directly.  The bootstrap wins a
    cross-process lock, publishes its own PID/birth evidence, and only then
    ``exec``s the worker.  Thus killing the parent in the fork/record window
    leaves either a reusable prelaunch token or child-owned process evidence;
    two recovery bootstraps with the same token cannot both run the worker.

    The exact run/token pair is also inherited in the worker environment.  On
    Darwin and Linux this lets cleanup rediscover an ordinary descendant after
    it changes session and loses its ancestry.  Every such PID is paired with an
    OS birth identity and persisted before it can be signalled.  This is a
    recovery seam, not platform containment: a child can replace its environment,
    so a root lost before the first live observation remains explicitly uncertain.
    """

    def __init__(
        self,
        command: Sequence[str] | Callable[[LaunchSpec], Sequence[str]],
        *,
        env: Mapping[str, str] | None = None,
        handshake_timeout: float = 5.0,
    ) -> None:
        if handshake_timeout <= 0:
            raise ValueError("handshake_timeout must be positive")
        self.command = command
        self.env = dict(env or {})
        self.handshake_timeout = handshake_timeout
        self._owned: dict[str, _SubprocessHandle] = {}
        self._bootstraps: dict[str, subprocess.Popen[bytes]] = {}
        self._lock = threading.RLock()

    def _command(self, spec: LaunchSpec) -> list[str]:
        command = self.command(spec) if callable(self.command) else self.command
        argv = [str(value) for value in command]
        if not argv:
            raise ValueError("worker command must not be empty")
        return argv

    def launch(self, spec: LaunchSpec) -> ProcessHandle:
        guard = spec.launch_path.with_name(f"{spec.launch_path.name}.guard.lock")
        with self._lock, _file_lock(guard):
            recovered = self.recover(spec)
            if recovered is not None:
                return recovered
            claim = _read_launch_record(spec.launch_path, spec.run_id)
            if claim is None:
                token = uuid.uuid4().hex
                _atomic_json(
                    spec.launch_path,
                    {
                        "schema": "taste.brains/SubprocessPrelaunch/1",
                        "run_id": spec.run_id,
                        "launch_token": token,
                    },
                )
            else:
                # ``recover`` returned None, so the only valid record here is
                # a parent that died before its bootstrap published a PID.
                token = claim["launch_token"]
            with contextlib.suppress(FileNotFoundError):
                spec.readiness_path.unlink()
            environment = dict(os.environ)
            environment.update(self.env)
            environment.update(
                {
                    _RUN_ID_ENV: spec.run_id,
                    _LAUNCH_TOKEN_ENV: token,
                    _READY_PATH_ENV: str(spec.readiness_path),
                }
            )
            worker_command = self._command(spec)
            bootstrap = (
                "import sys; from taste.brains.supervisor import _subprocess_bootstrap; "
                "_subprocess_bootstrap(sys.argv[1:])"
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    bootstrap,
                    str(spec.launch_path),
                    spec.run_id,
                    token,
                    *worker_command,
                ],
                cwd=spec.worktree,
                env=environment,
                start_new_session=True,
            )
            self._bootstraps[spec.run_id] = process
            raw = _await_launch_evidence(
                spec.launch_path,
                spec.run_id,
                token,
                process,
                self.handshake_timeout,
            )
            owned_process = process if raw["pid"] == process.pid else None
            if owned_process is None:
                # This bootstrap lost the token lock to the child of a parent
                # that died.  It performs no worker work and is safe to reap.
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=2)
            handle = _SubprocessHandle(
                spec=spec,
                pid=raw["pid"],
                process_group_id=raw["process_group_id"],
                launch_token=token,
                process_identity=raw["process_identity"],
                popen=owned_process,
            )
            self._owned[spec.run_id] = handle
            return handle

    def recover(self, spec: LaunchSpec) -> ProcessHandle | None:
        with self._lock:
            owned = self._owned.get(spec.run_id)
            if owned is not None:
                return owned
            raw = _read_launch_record(spec.launch_path, spec.run_id)
            if raw is None or raw["schema"] == "taste.brains/SubprocessPrelaunch/1":
                return None
            bootstrap = self._bootstraps.get(spec.run_id)
            owned_process = bootstrap if bootstrap is not None and bootstrap.pid == raw["pid"] else None
            return _SubprocessHandle(
                spec=spec,
                pid=raw["pid"],
                process_group_id=raw["process_group_id"],
                launch_token=raw["launch_token"],
                process_identity=raw["process_identity"],
                popen=owned_process,
            )


def _read_launch_record(path: Path, run_id: str) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SupervisorError("launcher evidence is unreadable") from exc
    prelaunch = {"schema", "run_id", "launch_token"}
    launched = {
        "schema",
        "run_id",
        "pid",
        "process_group_id",
        "launch_token",
        "process_identity",
    }
    if not isinstance(raw, dict) or raw.get("run_id") != run_id:
        raise SupervisorError("launcher evidence belongs to another run")
    if raw.get("schema") == "taste.brains/SubprocessPrelaunch/1" and set(raw) == prelaunch:
        if not isinstance(raw.get("launch_token"), str) or not raw["launch_token"]:
            raise SupervisorError("prelaunch token is malformed")
        return raw
    if raw.get("schema") != "taste.brains/SubprocessLaunch/1" or set(raw) != launched:
        raise SupervisorError("launcher evidence is malformed")
    if not all(
        isinstance(raw.get(name), int) and not isinstance(raw.get(name), bool) and raw[name] > 0
        for name in ("pid", "process_group_id")
    ):
        raise SupervisorError("launcher PID evidence is malformed")
    if not all(
        isinstance(raw.get(name), str) and raw[name]
        for name in ("launch_token", "process_identity")
    ):
        raise SupervisorError("launcher identity evidence is malformed")
    return raw


def _await_launch_evidence(
    path: Path,
    run_id: str,
    token: str,
    process: subprocess.Popen[bytes],
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raw = _read_launch_record(path, run_id)
        if raw is not None and raw["schema"] == "taste.brains/SubprocessLaunch/1":
            if raw["launch_token"] != token:
                raise SupervisorError("launcher token changed during fork handshake")
            return raw
        returncode = process.poll()
        if returncode is not None:
            raise SupervisorError(f"worker bootstrap exited before PID evidence ({returncode})")
        time.sleep(0.01)
    raise SupervisorError("worker bootstrap PID handshake is still pending")


def _subprocess_bootstrap(arguments: Sequence[str]) -> None:
    """Child half of the prelaunch/PID handshake; execs at most one worker."""
    if len(arguments) < 4:
        raise SystemExit(70)
    launch_path = Path(arguments[0])
    run_id = arguments[1]
    token = arguments[2]
    command = list(arguments[3:])
    lock_path = launch_path.with_name(f"{launch_path.name}.bootstrap.lock")
    import fcntl

    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(75) from None
        raw = _read_launch_record(launch_path, run_id)
        if (
            raw is None
            or raw["launch_token"] != token
            or raw["schema"] == "taste.brains/SubprocessLaunch/1"
        ):
            raise SystemExit(75)
        # Keep the lock through exec.  It is the birth/liveness fence used by
        # a restarted supervisor.  The independently observed OS birth value
        # below is the PID-reuse fence used before any signal is sent.
        os.set_inheritable(descriptor, True)
        identity = _process_identity(os.getpid())
        if not identity:
            raise SystemExit(71)
        _atomic_json(
            launch_path,
            {
                "schema": "taste.brains/SubprocessLaunch/1",
                "run_id": run_id,
                "pid": os.getpid(),
                "process_group_id": os.getpgrp(),
                "launch_token": token,
                "process_identity": identity,
            },
        )
        os.execvpe(command[0], command, os.environ)
    finally:
        os.close(descriptor)


def mark_worker_ready() -> None:
    """Publish readiness bound to the exact launched process.

    This is suitable as the body of ``WorkerRuntime(..., ready=...)``.  It is
    intentionally a no-op error outside a supervised worker: a missing launch
    identity means there is no process boundary to certify.
    """
    run_id = os.environ.get(_RUN_ID_ENV, "")
    token = os.environ.get(_LAUNCH_TOKEN_ENV, "")
    raw_path = os.environ.get(_READY_PATH_ENV, "")
    if not run_id or not token or not raw_path:
        raise SupervisorError("worker readiness environment is incomplete")
    _atomic_json(
        Path(raw_path),
        {
            "schema": "taste.brains/WorkerReady/1",
            "run_id": run_id,
            "launch_token": token,
            "pid": os.getpid(),
        },
    )


class CentralSupervisor:
    """Mechanical supervisor for exact typed assignments.

    The class is intentionally synchronous.  A central brain can call it from
    an async thread, while lifecycle mutations remain serialized under the
    control branch's write lease.
    """

    def __init__(
        self,
        store: Store,
        *,
        launcher: ProcessLauncher,
        control_branch: str | Branch = "central-control",
        integration_branch: str | Branch = "integration",
        control_lock: Any | None = None,
        clock: Callable[[], datetime] = _now,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval: float = 0.05,
        termination_grace: float = 2.0,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if termination_grace < 0:
            raise ValueError("termination_grace must not be negative")
        self.store = store
        self.launcher = launcher
        self._owns_control = isinstance(control_branch, str)
        self._owns_integration = isinstance(integration_branch, str)
        self.control = (
            store.branch(control_branch, producer="central-supervisor")
            if isinstance(control_branch, str)
            else control_branch
        )
        self.integration = (
            store.branch(integration_branch, producer="central-integration")
            if isinstance(integration_branch, str)
            else integration_branch
        )
        if self.control.store is not store or self.integration.store is not store:
            raise ValueError("shared branches must belong to the supervisor Store")
        if self.control.name == self.integration.name:
            raise ValueError("control and integration branches must be distinct")
        # A coordinator which also checkpoints plans on ``control`` passes the
        # same re-entrant lock to both components.  It is held only around a
        # branch read/checkpoint, never while launching, polling, sleeping, or
        # reaping a process.
        self.control_lock = control_lock or threading.RLock()
        self.clock = clock
        self.sleep = sleep
        self.poll_interval = poll_interval
        self.termination_grace = termination_grace
        self._handles: dict[str, ProcessHandle] = {}
        self._lock = threading.RLock()
        self._run_index_audit_cache: dict[str, tuple[dict[str, Any], ...]] = {}
        self._run_history_audit_cache: set[tuple[str, str]] = set()

    def close(self) -> None:
        """Release only branches opened by this supervisor.

        Injected Branch objects remain owned by the coordinator which passed
        them.  This lets planner and supervisor share one central-control
        writer and one ``control_lock`` without either closing it underneath
        the other.
        """
        if self._owns_integration:
            self.integration.close()
        if self._owns_control:
            self.control.close()

    def __enter__(self) -> CentralSupervisor:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _sidecars(self, run_id: str) -> tuple[Path, Path]:
        root = self.store.backend.common_dir
        key = _run_key(run_id)
        return (
            root / f"taste-supervisor.{self.store.session}.{key}.ready.json",
            root / f"taste-supervisor.{self.store.session}.{key}.launch.json",
        )

    def _spec(self, run: SupervisorRun) -> LaunchSpec:
        if run.deadline_at is None:
            raise SupervisorError("run has no durable process deadline")
        ready, launch = self._sidecars(run.run_id)
        return LaunchSpec(
            run_id=run.run_id,
            assignment=run.assignment,
            prepared_state_id=run.prepared_state_id,
            worktree=self.store.worktree_path_for(run.assignment.worker),
            deadline_at=run.deadline_at,
            readiness_path=ready,
            launch_path=launch,
        )

    def _parse_run_index(self, text: str, where: str) -> tuple[dict[str, Any], ...]:
        def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise SupervisorLedgerCorruption(
                        f"{where} contains duplicate key {key!r}"
                    )
                value[key] = item
            return value

        def no_constant(value: str) -> Any:
            raise SupervisorLedgerCorruption(
                f"{where} contains non-JSON number {value}"
            )

        try:
            raw = json.loads(
                text,
                object_pairs_hook=no_duplicates,
                parse_constant=no_constant,
            )
        except json.JSONDecodeError as exc:
            raise SupervisorLedgerCorruption(f"{where} is not JSON") from exc
        if (
            not isinstance(raw, dict)
            or set(raw) != {"schema", "session", "control_branch", "runs"}
            or raw.get("schema") != RUN_INDEX_SCHEMA
            or raw.get("session") != self.store.session
            or raw.get("control_branch") != self.control.name
            or not isinstance(raw.get("runs"), list)
        ):
            raise SupervisorLedgerCorruption(f"{where} has the wrong identity or fields")
        entries: list[dict[str, Any]] = []
        seen_run_ids: set[str] = set()
        seen_paths: set[str] = set()
        for sequence, entry in enumerate(raw["runs"], 1):
            if not isinstance(entry, dict) or set(entry) != {
                "sequence",
                "run_id",
                "path",
                "assignment_digest",
            }:
                raise SupervisorLedgerCorruption(f"{where} has a malformed run entry")
            run_id = entry["run_id"]
            path = entry["path"]
            assignment_digest = entry["assignment_digest"]
            if (
                entry["sequence"] != sequence
                or not isinstance(run_id, str)
                or _STABLE_ID.fullmatch(run_id) is None
                or not isinstance(path, str)
                or path != _run_path(run_id)
                or not isinstance(assignment_digest, str)
                or _SHA256_DIGEST.fullmatch(assignment_digest) is None
                or run_id in seen_run_ids
                or path in seen_paths
            ):
                raise SupervisorLedgerCorruption(f"{where} run entry identity is invalid")
            seen_run_ids.add(run_id)
            seen_paths.add(path)
            entries.append(dict(entry))
        return tuple(entries)

    def _current_run_index(
        self,
        head: State,
        *,
        check_files: bool = True,
    ) -> tuple[dict[str, Any], ...]:
        text = head.read(RUN_INDEX_PATH)
        indexed = () if text is None else self._parse_run_index(text, "supervisor run index")
        actual = {
            path
            for path in head.files()
            if path.startswith(f"{RUN_ROOT}/") and path.endswith("/run.json")
        }
        if check_files and actual != {entry["path"] for entry in indexed}:
            raise SupervisorLedgerCorruption(
                "supervisor run index and durable run files disagree"
            )
        return indexed

    def _audited_run_index(self, head: State) -> tuple[dict[str, Any], ...]:
        cached = self._run_index_audit_cache.get(head.id)
        if cached is not None:
            return cached
        current = self._current_run_index(head)
        try:
            changed = self.store.backend.repo.git.rev_list(
                "--reverse",
                "--first-parent",
                head.id,
                "--",
                RUN_INDEX_PATH,
            )
        except Exception as exc:
            raise SupervisorLedgerCorruption(
                "supervisor run index history is unreadable"
            ) from exc
        commits = tuple(item for item in changed.splitlines() if item)
        if not commits:
            if current:
                raise SupervisorLedgerCorruption(
                    "supervisor run index has no durable introduction"
                )
            self._run_index_audit_cache[head.id] = ()
            return ()
        previous: tuple[dict[str, Any], ...] = ()
        latest_text = ""
        for version, commit in enumerate(commits, 1):
            text = self.store.backend.show(commit, RUN_INDEX_PATH)
            if text is None:
                raise SupervisorLedgerCorruption(
                    "supervisor run index was deleted from durable history"
                )
            entries = self._parse_run_index(text, f"supervisor run index version {version}")
            if len(entries) != len(previous) + 1 or entries[:-1] != previous:
                raise SupervisorLedgerCorruption(
                    "supervisor run index is not append-only"
                )
            previous = entries
            latest_text = text
        if head.read(RUN_INDEX_PATH) != latest_text or current != previous:
            raise SupervisorLedgerCorruption(
                "current supervisor run index differs from its audited history"
            )
        self._run_index_audit_cache[head.id] = current
        return current

    def _audit_run_history(self, head: State, entry: Mapping[str, Any]) -> None:
        cache_key = (head.id, entry["run_id"])
        if cache_key in self._run_history_audit_cache:
            return
        path = entry["path"]
        try:
            changed = self.store.backend.repo.git.rev_list(
                "--reverse",
                "--first-parent",
                head.id,
                "--",
                path,
            )
        except Exception as exc:
            raise SupervisorLedgerCorruption(
                f"supervisor run history for {entry['run_id']!r} is unreadable"
            ) from exc
        commits = tuple(item for item in changed.splitlines() if item)
        if not commits:
            raise SupervisorLedgerCorruption("indexed supervisor run has no durable history")
        prior_sequence = 0
        for commit in commits:
            text = self.store.backend.show(commit, path)
            if text is None:
                raise SupervisorLedgerCorruption(
                    "supervisor run record was deleted from durable history"
                )
            try:
                run = SupervisorRun.from_json(text)
            except ValueError as exc:
                raise SupervisorLedgerCorruption(
                    "supervisor run history contains a malformed record"
                ) from exc
            if (
                run.run_id != entry["run_id"]
                or run.assignment_digest != entry["assignment_digest"]
            ):
                raise SupervisorLedgerCorruption(
                    "supervisor run changed its indexed identity in durable history"
                )
            if run.sequence != prior_sequence + 1:
                raise SupervisorLedgerCorruption(
                    "supervisor run history is not an append-only transition sequence"
                )
            prior_sequence = run.sequence
        self._run_history_audit_cache.add(cache_key)

    def _load(self, run_id: str, *, audit_history: bool = True) -> SupervisorRun:
        with self.control_lock:
            head = self.control.head
            indexed = (
                self._audited_run_index(head)
                if audit_history
                else self._current_run_index(head)
            )
            raw = head.read(_run_path(run_id))
        if raw is None:
            raise KeyError(run_id)
        run = SupervisorRun.from_json(raw)
        matching = [entry for entry in indexed if entry["run_id"] == run_id]
        if len(matching) != 1 or matching[0]["assignment_digest"] != run.assignment_digest:
            raise SupervisorLedgerCorruption(
                "durable run does not match its supervisor index entry"
            )
        if audit_history:
            self._audit_run_history(head, matching[0])
        return run

    def get(self, run_id: str) -> SupervisorRun:
        with self._lock:
            return self._load(run_id, audit_history=True)

    def runs(self) -> tuple[SupervisorRun, ...]:
        with self._lock, self.control_lock:
            head = self.control.head
            indexed = self._audited_run_index(head)
            runs = tuple(
                SupervisorRun.from_json(head.read(entry["path"]) or "")
                for entry in indexed
            )
            for entry in indexed:
                self._audit_run_history(head, entry)
            if any(
                run.run_id != entry["run_id"]
                or run.assignment_digest != entry["assignment_digest"]
                for run, entry in zip(runs, indexed, strict=True)
            ):
                raise SupervisorLedgerCorruption(
                    "durable supervisor run changed its indexed identity"
                )
            return runs

    def _transition(
        self,
        run: SupervisorRun,
        *,
        kind: str,
        phase: str | None = None,
        observed_state_id: str | None = None,
        terminal: bool = False,
        terminal_reason: str = "",
        uncertain: bool | None = None,
        uncertainty_reasons: tuple[str, ...] | None = None,
        detail: str = "",
        metadata: Mapping[str, Any] | None = None,
        **changes: Any,
    ) -> SupervisorRun:
        sequence = run.sequence + 1
        at = _iso(self.clock())
        reasons = run.uncertainty_reasons if uncertainty_reasons is None else uncertainty_reasons
        is_uncertain = run.uncertain if uncertain is None else uncertain
        if terminal:
            changes.setdefault("terminal_reason", terminal_reason)
        updated = replace(
            run,
            phase=phase or run.phase,
            sequence=sequence,
            uncertain=is_uncertain,
            uncertainty_reasons=reasons,
            **changes,
        )
        updated._validate()
        event_payload = {
            "sequence": sequence,
            "kind": kind,
            "at": at,
            "observed_state_id": observed_state_id,
            "terminal": terminal,
            "terminal_reason": terminal_reason,
            "uncertain": is_uncertain,
            "uncertainty_reasons": list(reasons),
            "detail": detail,
            "metadata": _json_value(metadata or {}),
        }
        event_token = hashlib.sha256(
            json.dumps(event_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        event = LifecycleEvent(
            event_id=f"supervisor-event.{event_token}",
            run_id=run.run_id,
            assignment_id=run.assignment.assignment_id,
            worker=run.assignment.worker,
            generation=run.assignment.generation,
            attempt=run.assignment.attempt,
            kind=kind,
            at=at,
            observed_state_id=observed_state_id,
            pid=updated.pid,
            process_group_id=updated.process_group_id,
            exit_code=updated.exit_code,
            signal=updated.signal,
            terminal=terminal,
            terminal_reason=terminal_reason,
            uncertain=is_uncertain,
            uncertainty_reasons=reasons,
            detail=detail,
            metadata=metadata or {},
        )
        with self.control_lock:
            head = self.control.head
            raw = head.read(_run_path(run.run_id))
            records: dict[str, Any] = {
                _run_path(run.run_id): updated.to_dict(),
                _event_path(run.run_id, sequence, kind): event.to_dict(),
            }
            if run.sequence == 0:
                if raw is not None:
                    raise SupervisorStateConflict(
                        "initial transition found an existing durable run"
                    )
                indexed = self._audited_run_index(head)
                if any(entry["run_id"] == run.run_id for entry in indexed):
                    raise SupervisorLedgerCorruption(
                        "new run identity already exists in the supervisor index"
                    )
                index_entries = [dict(entry) for entry in indexed]
                index_entries.append(
                    {
                        "sequence": len(index_entries) + 1,
                        "run_id": run.run_id,
                        "path": _run_path(run.run_id),
                        "assignment_digest": run.assignment_digest,
                    }
                )
                records[RUN_INDEX_PATH] = {
                    "schema": RUN_INDEX_SCHEMA,
                    "session": self.store.session,
                    "control_branch": self.control.name,
                    "runs": index_entries,
                }
            else:
                if raw is None:
                    raise SupervisorStateConflict(
                        "transition source run is missing from durable state"
                    )
                indexed = self._current_run_index(head)
                matching = [entry for entry in indexed if entry["run_id"] == run.run_id]
                if (
                    len(matching) != 1
                    or matching[0]["assignment_digest"] != run.assignment_digest
                ):
                    raise SupervisorLedgerCorruption(
                        "transition source does not match the supervisor index"
                    )
                try:
                    current = SupervisorRun.from_json(raw)
                except ValueError as exc:
                    raise SupervisorStateConflict(
                        "durable transition source is malformed"
                    ) from exc
                if current != run:
                    raise SupervisorStateConflict(
                        "transition source is stale relative to durable state"
                    )
            self.control.checkpoint(
                f"supervisor {kind}: {run.run_id}",
                records=records,
            )
        return updated

    def _validate_generation(self, run: SupervisorRun, active_generation: int) -> None:
        if run.assignment.generation != active_generation:
            raise StaleGeneration(
                f"run generation {run.assignment.generation} is stale; active is {active_generation}"
            )

    def _validate_assignment(self, assignment: Assignment) -> State:
        if assignment.contract_digest != contract_digest(assignment.contract):
            raise AssignmentIdentityConflict("contract digest does not match contract")
        if assignment.contract.inputs != tuple(item.path for item in assignment.inputs):
            raise AssignmentIdentityConflict(
                "contract inputs do not match the exact structured input paths"
            )
        if assignment.contract.outputs != tuple(item.path for item in assignment.outputs):
            raise AssignmentIdentityConflict(
                "contract outputs do not match the exact structured output paths"
            )
        if _EXACT_OBJECT_ID.fullmatch(assignment.base_state_id) is None:
            raise AssignmentIdentityConflict("base_state_id must be a full object id")
        try:
            base = self.store.state(assignment.base_state_id)
            _ = base.meta
        except Exception as exc:
            raise AssignmentIdentityConflict("assignment base state does not exist") from exc
        if base.meta.session != self.store.session:
            raise AssignmentIdentityConflict("assignment base belongs to another session")
        if base.meta.branch != self.integration.name or not self.store.backend.is_ancestor(
            base.id, self.integration.head.id
        ):
            raise AssignmentIdentityConflict(
                "assignment base must be an exact state in the integration lineage"
            )
        paths: dict[str, str] = {}
        for artifact in assignment.inputs:
            validate_artifact_path(artifact.path)
            prior = paths.setdefault(artifact.path, artifact.blob_id)
            if prior != artifact.blob_id:
                raise AssignmentIdentityConflict(
                    f"two inputs project different bytes to {artifact.path!r}"
                )
            if _EXACT_OBJECT_ID.fullmatch(artifact.state_id) is None:
                raise AssignmentIdentityConflict(
                    f"input {artifact.artifact_id!r} does not name an exact state"
                )
            try:
                state = self.store.state(artifact.state_id)
                _ = state.meta
            except Exception as exc:
                raise AssignmentIdentityConflict(
                    f"input {artifact.artifact_id!r} state does not exist"
                ) from exc
            if state.meta.session != self.store.session or state.meta.branch != artifact.branch:
                raise AssignmentIdentityConflict(
                    f"input {artifact.artifact_id!r} source identity is wrong"
                )
            entry = self.store.backend.entry_at(state.id, artifact.path)
            if entry is None or entry.mode == "040000" or entry.sha != artifact.blob_id:
                raise AssignmentIdentityConflict(
                    f"input {artifact.artifact_id!r} does not point to the declared file bytes"
                )
        for output in assignment.outputs:
            validate_artifact_path(output.path)
        return base

    def _worker_runs(self, worker: str) -> tuple[SupervisorRun, ...]:
        return tuple(run for run in self.runs() if run.assignment.worker == worker)

    def _verify_prepared(self, run: SupervisorRun) -> None:
        state = self.store.state(run.prepared_state_id)
        if state.meta.branch != run.assignment.worker:
            raise AssignmentIdentityConflict("prepared state belongs to the wrong worker")
        if state.read(CONTRACT_PATH) != run.assignment.contract.to_json():
            raise AssignmentIdentityConflict("prepared contract is not exact")
        if state.read(ASSIGNMENT_PATH) != run.assignment.to_json():
            raise AssignmentIdentityConflict("prepared assignment is not exact")
        if not self.store.backend.is_ancestor(run.assignment.base_state_id, state.id):
            raise AssignmentIdentityConflict("prepared state does not descend from exact base")
        for artifact in run.assignment.inputs:
            if state.blob(artifact.path) != artifact.blob_id:
                raise AssignmentIdentityConflict(
                    f"prepared input {artifact.artifact_id!r} bytes do not match"
                )

    def prepare(
        self,
        assignment: Assignment,
        *,
        wall_timeout_seconds: float,
        active_generation: int | None = None,
    ) -> SupervisorRun:
        """Install exact worker inputs and persist a launchable run.

        No process is created here.  Retrying an identical call returns the
        original prepared record after re-verifying its immutable state.
        """
        if wall_timeout_seconds <= 0:
            raise ValueError("wall_timeout_seconds must be positive")
        if active_generation is not None and assignment.generation != active_generation:
            raise StaleGeneration("cannot prepare an assignment from a stale generation")
        with self._lock:
            run_id = _run_id(assignment)
            try:
                existing = self._load(run_id)
            except KeyError:
                existing = None
            if existing is not None:
                if existing.assignment != assignment:
                    raise AssignmentIdentityConflict("run id is already bound to another assignment")
                if existing.wall_timeout_seconds != float(wall_timeout_seconds):
                    raise AssignmentIdentityConflict("run timeout is immutable")
                self._verify_prepared(existing)
                return existing

            base = self._validate_assignment(assignment)
            previous = self._worker_runs(assignment.worker)
            if assignment.worker in {self.control.name, self.integration.name}:
                raise AssignmentIdentityConflict(
                    "a worker identity cannot alias a central branch"
                )
            for prior in previous:
                if not prior.terminal:
                    raise AssignmentIdentityConflict(
                        f"worker {assignment.worker!r} already has a live assignment"
                    )
                if (
                    prior.assignment.assignment_id == assignment.assignment_id
                    and prior.assignment.generation == assignment.generation
                    and prior.assignment.attempt >= assignment.attempt
                ):
                    raise AssignmentIdentityConflict(
                        "assignment attempts must increase monotonically"
                    )

            existed = self.store.view(assignment.worker).exists()
            branch = self.store.branch(
                assignment.worker,
                from_state=base,
                producer=assignment.worker,
            )
            try:
                if existed and not previous:
                    live_assignment = branch.read(ASSIGNMENT_PATH)
                    dirty = set(branch.dirty_paths())
                    permitted = {
                        CONTRACT_PATH,
                        ASSIGNMENT_PATH,
                        *(artifact.path for artifact in assignment.inputs),
                    }
                    if live_assignment not in {None, assignment.to_json()} or dirty - permitted:
                        raise AssignmentIdentityConflict(
                            "unclaimed worker branch contains ambiguous work"
                        )
                    # A clean pre-existing branch with no durable identity is
                    # not silently repurposed.  Dirty exact projection paths
                    # are the recognisable crash-during-prepare exception.
                    if (
                        live_assignment is None
                        and branch.head.read(ASSIGNMENT_PATH) is None
                        and not dirty
                        and not (
                            branch.head.meta.kind == "branch"
                            and branch.head.meta.parents == (base.id,)
                            and self.store.backend.tree_of(branch.head.id)
                            == self.store.backend.tree_of(base.id)
                        )
                    ):
                        raise AssignmentIdentityConflict(
                            "pre-existing worker branch has no supervisor identity"
                        )
                if not self.store.backend.is_ancestor(base.id, branch.head.id):
                    raise AssignmentIdentityConflict("worker branch does not descend from base")
                for destination in (
                    *(artifact.path for artifact in assignment.inputs),
                    CONTRACT_PATH,
                    ASSIGNMENT_PATH,
                ):
                    _assert_safe_write_destination(branch.worktree, destination)
                for artifact in assignment.inputs:
                    source = self.store.state(artifact.state_id)
                    branch.adopt(source, artifact.path, as_=artifact.path)
                    if branch.path(artifact.path).read_bytes() != source.read_bytes(artifact.path):
                        raise AssignmentIdentityConflict(
                            f"input projection changed bytes for {artifact.artifact_id!r}"
                        )
                branch.write(CONTRACT_PATH, assignment.contract.to_json())
                branch.write(ASSIGNMENT_PATH, assignment.to_json())
                prepared = branch.checkpoint(
                    f"prepare {assignment.assignment_id} generation "
                    f"{assignment.generation} attempt {assignment.attempt}"
                )
            finally:
                branch.close()

            created_at = _iso(self.clock())
            run = SupervisorRun(
                run_id=run_id,
                assignment=assignment,
                assignment_digest=_digest(assignment.to_json()),
                prepared_state_id=prepared.id,
                wall_timeout_seconds=float(wall_timeout_seconds),
                created_at=created_at,
            )
            run._validate()
            run = self._transition(
                run,
                kind="prepared",
                phase="prepared",
                observed_state_id=prepared.id,
                detail="exact contract, assignment, base, and input projection checkpointed",
            )
            self._verify_prepared(run)
            return run

    def _recover_handle(self, run: SupervisorRun) -> ProcessHandle | None:
        handle = self._handles.get(run.run_id)
        if handle is None:
            handle = self.launcher.recover(self._spec(run))
        if handle is not None:
            if run.pid is not None and (
                handle.pid != run.pid
                or handle.process_group_id != run.process_group_id
                or handle.launch_token != run.launch_token
                or handle.process_identity != run.process_identity
            ):
                raise SupervisorError(
                    "recovered process evidence differs from the durable spawned event"
                )
            self._handles[run.run_id] = handle
        return handle

    def start(self, run_id: str, *, active_generation: int) -> SupervisorRun:
        """Launch once, always after a durable spawn intent and deadline."""
        with self._lock:
            run = self._load(run_id)
            self._validate_generation(run, active_generation)
            if run.terminal or run.phase in {"spawned", "ready"}:
                return run
            self._verify_prepared(run)
            if run.phase == "prepared":
                deadline = self.clock() + timedelta(seconds=run.wall_timeout_seconds)
                run = self._transition(
                    run,
                    kind="spawn_intent",
                    phase="spawn_intent",
                    observed_state_id=run.prepared_state_id,
                    deadline_at=_iso(deadline),
                    detail="durable launch intent recorded before process creation",
                )

            handle = self._recover_handle(run)
            if handle is None and self.clock() >= _parse_time(run.deadline_at or ""):
                recovery_state = self._capture_worker(run)
                return self._transition(
                    run,
                    kind="deadline_before_launch",
                    phase="terminal",
                    observed_state_id=recovery_state,
                    terminal=True,
                    terminal_reason="wall_timeout",
                    recovery_state_id=recovery_state,
                    reaped=True,
                    detail="durable wall deadline elapsed before a process was launched",
                )
            if handle is None:
                try:
                    handle = self.launcher.launch(self._spec(run))
                except Exception as exc:
                    reasons = tuple(
                        dict.fromkeys((*run.uncertainty_reasons, "launch_outcome_unknown"))
                    )
                    self._transition(
                        run,
                        kind="launch_failed",
                        phase="spawn_intent",
                        uncertain=True,
                        uncertainty_reasons=reasons,
                        detail=f"{type(exc).__name__}: {exc}"[:512],
                    )
                    raise
                self._handles[run.run_id] = handle

            run = self._transition(
                run,
                kind="spawned",
                phase="spawned",
                observed_state_id=run.prepared_state_id,
                pid=handle.pid,
                process_group_id=handle.process_group_id,
                launch_token=handle.launch_token,
                process_identity=handle.process_identity,
                uncertain=False,
                uncertainty_reasons=(),
                metadata={"deadline_at": run.deadline_at},
            )
            return self.poll(run.run_id, active_generation=active_generation)

    def _capture_worker(self, run: SupervisorRun) -> str:
        """Acquire the dead worker lease, checkpoint dirty work, remove worktree."""
        before = self.store.view(run.assignment.worker).head.id
        try:
            self.store.remove_branch(run.assignment.worker)
        except BranchBusy as exc:
            raise SupervisorError("worker lease remained live after process termination") from exc
        return self.store.view(run.assignment.worker).head.id if self.store.view(
            run.assignment.worker
        ).exists() else before

    def _terminalize(
        self,
        run: SupervisorRun,
        handle: ProcessHandle | None,
        observed: ProcessExit,
        *,
        reason: str,
    ) -> SupervisorRun:
        # Even when the root exited naturally, descendants may remain.  Asking
        # the launch adapter to terminate/reap the exact tree is mandatory.
        final_exit = observed
        if handle is not None:
            cleaned = handle.terminate_tree(self.termination_grace)
            if cleaned.exit_code is not None or cleaned.signal is not None or cleaned.reaped:
                final_exit = cleaned
        recovery_state = self._capture_worker(run)
        uncertainty: list[str] = []
        if not final_exit.reaped:
            uncertainty.append("process_not_reapable_after_supervisor_restart")
        if final_exit.exit_code is None and final_exit.signal is None:
            uncertainty.append("exit_status_unknown")
        return self._transition(
            run,
            kind="process_terminal",
            phase="terminal",
            observed_state_id=recovery_state,
            terminal=True,
            terminal_reason=reason,
            exit_code=final_exit.exit_code,
            signal=final_exit.signal,
            reaped=final_exit.reaped,
            recovery_state_id=recovery_state,
            uncertain=bool(uncertainty),
            uncertainty_reasons=tuple(uncertainty),
        )

    def poll(self, run_id: str, *, active_generation: int) -> SupervisorRun:
        """Observe readiness, deadline, or process death once."""
        with self._lock:
            run = self._load(run_id)
            self._validate_generation(run, active_generation)
            if run.terminal:
                return run
            if run.phase in {"prepared", "spawn_intent"}:
                return self.start(run_id, active_generation=active_generation)
            handle = self._recover_handle(run)
            if handle is None:
                recovery_state = self._capture_worker(run)
                reasons = ("process_evidence_missing", "exit_status_unknown")
                return self._transition(
                    run,
                    kind="process_lost",
                    phase="terminal",
                    observed_state_id=recovery_state,
                    terminal=True,
                    terminal_reason="process_lost",
                    recovery_state_id=recovery_state,
                    uncertain=True,
                    uncertainty_reasons=reasons,
                )

            observed = handle.poll()
            # A readiness marker is accepted only between two live-process
            # observations.  It remains historical after a later terminal
            # event, while SupervisorRun.ready immediately becomes false.
            if observed is None and run.ready_at is None and handle.ready():
                observed = handle.poll()
                if observed is None:
                    run = self._transition(
                        run,
                        kind="ready",
                        phase="ready",
                        observed_state_id=self.store.view(run.assignment.worker).head.id,
                        ready_at=_iso(self.clock()),
                    )

            if observed is not None:
                reason = "process_exited"
                if observed.signal is not None:
                    reason = "process_signalled"
                elif observed.exit_code not in {None, 0}:
                    reason = "process_failed"
                return self._terminalize(run, handle, observed, reason=reason)

            assert run.deadline_at is not None
            if self.clock() >= _parse_time(run.deadline_at):
                killed = handle.terminate_tree(self.termination_grace)
                return self._terminalize(run, None, killed, reason="wall_timeout")
            return run

    def wait(self, run_id: str, *, active_generation: int) -> SupervisorRun:
        """Wait through the persisted hard wall deadline."""
        while True:
            run = self.poll(run_id, active_generation=active_generation)
            if run.terminal:
                return run
            assert run.deadline_at is not None
            remaining = (_parse_time(run.deadline_at) - self.clock()).total_seconds()
            self.sleep(min(self.poll_interval, max(0.0, remaining)))

    def stop(self, run_id: str, reason: str = "cancelled") -> SupervisorRun:
        """Stop and reap the complete process tree, then capture dirty work."""
        with self._lock:
            run = self._load(run_id)
            if run.terminal:
                return run
            if run.phase == "prepared":
                recovery_state = self._capture_worker(run)
                token = _reason_token(reason)
                return self._transition(
                    run,
                    kind="stopped",
                    phase="terminal",
                    observed_state_id=recovery_state,
                    terminal=True,
                    terminal_reason=token,
                    recovery_state_id=recovery_state,
                )
            handle = self._recover_handle(run)
            if handle is None:
                exit_status = ProcessExit(reaped=False)
            else:
                exit_status = handle.terminate_tree(self.termination_grace)
            return self._terminalize(
                run,
                None,
                exit_status,
                reason=_reason_token(reason),
            )

    def reconcile(self, *, active_generation: int) -> tuple[SupervisorRun, ...]:
        """Resume every current-generation run without another planner call."""
        reconciled: list[SupervisorRun] = []
        for run in self.runs():
            if run.terminal:
                reconciled.append(run)
                continue
            if run.assignment.generation != active_generation:
                reconciled.append(self.stop(run.run_id, "stale_generation"))
                continue
            reconciled.append(self.poll(run.run_id, active_generation=active_generation))
        return tuple(reconciled)

    def _report(self, run: SupervisorRun) -> tuple[WorkerReport, State]:
        head = self.store.view(run.assignment.worker).head
        raw = head.read(WORKER_REPORT_PATH)
        if raw is None:
            raise InvalidWorkerReport("worker branch has no durable terminal report")
        try:
            report = WorkerReport.from_json(raw)
        except Exception as exc:
            raise InvalidWorkerReport("worker report is malformed") from exc
        expected = (
            report.run_id == run.run_id
            and report.assignment_id == run.assignment.assignment_id
            and report.worker == run.assignment.worker
            and report.generation == run.assignment.generation
            and report.attempt == run.assignment.attempt
            and report.contract_digest == run.assignment.contract_digest
            and report.base_state_id == run.assignment.base_state_id
        )
        if not expected:
            raise InvalidWorkerReport("worker report identity does not match this exact run")
        if _EXACT_OBJECT_ID.fullmatch(report.final_state_id) is None:
            raise InvalidWorkerReport("worker report final state is not exact")
        try:
            final = self.store.state(report.final_state_id)
            _ = final.meta
        except Exception as exc:
            raise InvalidWorkerReport("worker report final state does not exist") from exc
        if (
            final.meta.branch != run.assignment.worker
            or not self.store.backend.is_ancestor(run.prepared_state_id, final.id)
            or not self.store.backend.is_ancestor(final.id, head.id)
        ):
            raise InvalidWorkerReport("worker report final state has the wrong lineage")
        if final.read(CONTRACT_PATH) != run.assignment.contract.to_json() or final.read(
            ASSIGNMENT_PATH
        ) != run.assignment.to_json():
            raise InvalidWorkerReport("worker changed its durable control records")

        specs = {spec.artifact_id: spec for spec in run.assignment.outputs}
        reported = {output.artifact_id: output for output in report.outputs}
        if set(reported) - set(specs):
            raise InvalidWorkerReport("worker report contains undeclared outputs")
        for artifact_id, output in reported.items():
            spec = specs[artifact_id]
            if (
                spec.disposition != "present"
                or output.path != spec.path
                or output.kind != spec.kind
                or output.branch != run.assignment.worker
                or output.state_id != final.id
                or final.blob(output.path) != output.blob_id
            ):
                raise InvalidWorkerReport(f"output {artifact_id!r} is not exact-state bound")
        for spec in run.assignment.outputs:
            if spec.disposition == "absent":
                if final.blob(spec.path) is not None or spec.artifact_id in reported:
                    raise InvalidWorkerReport(f"output {spec.artifact_id!r} was not removed")
            elif spec.required and spec.artifact_id not in reported:
                raise InvalidWorkerReport(f"required output {spec.artifact_id!r} is missing")
        return report, head

    def _terminal_assessment(self, report: WorkerReport) -> TerminalAssessment:
        metadata = report.metadata
        monitor = metadata.get("monitor") if isinstance(metadata, Mapping) else None
        assessment = monitor.get("terminal_assessment") if isinstance(monitor, Mapping) else None
        if not isinstance(assessment, Mapping):
            raise DeliveryRejected("report has no exact terminal monitor assessment")
        try:
            typed = TerminalAssessment.from_dict(_json_value(assessment))
        except Exception as exc:
            raise DeliveryRejected("terminal monitor assessment is corrupt") from exc
        if not (
            typed.acceptable
            and typed.state_id == report.final_state_id
            and typed.contract_digest == report.contract_digest
        ):
            raise DeliveryRejected("terminal monitor assessment is not acceptable and exact")
        if not isinstance(monitor, Mapping) or monitor.get("current_state") != report.final_state_id:
            raise DeliveryRejected("monitor current state is not the report final state")
        if monitor.get("current") != "fine":
            raise DeliveryRejected("monitor current terminal severity is not fine")
        if monitor.get("pending_actions"):
            raise DeliveryRejected("monitor actions remain pending")

        # The worker report is a notification, not the source of the monitor's
        # certificate.  Require the same content-identified assessment in the
        # contract-scoped durable monitor sidecar.
        digest_hex = report.contract_digest.removeprefix("sha256:")
        sidecar = self.store.sidecar(
            "monitor", report.worker, f".contract-{digest_hex}"
        )
        try:
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
            if raw.get("contract_digest") != report.contract_digest:
                raise ValueError("monitor sidecar contract digest differs")
            durable = [
                TerminalAssessment.from_dict(item)
                for item in raw.get("terminal_assessments", ())
            ]
        except Exception as exc:
            raise DeliveryRejected("durable terminal monitor assessment is unavailable") from exc
        matching = [item for item in durable if item.id == typed.id]
        if len(matching) != 1 or matching[0] != typed:
            raise DeliveryRejected("report assessment is absent from the durable monitor sidecar")
        return typed

    def collect(self, run_id: str, *, active_generation: int) -> WorkerReport:
        """Accept one exact report only after the process boundary is terminal."""
        with self._lock:
            run = self._load(run_id)
            self._validate_generation(run, active_generation)
            if not run.terminal:
                raise InvalidWorkerReport("worker process is not terminal")
            report, report_state = self._report(run)
            if run.report_id is not None:
                if run.report_id != report.report_id or run.report_state_id != report_state.id:
                    raise InvalidWorkerReport("accepted report identity changed")
                return report
            self._transition(
                run,
                kind="report_accepted",
                phase="report_accepted",
                observed_state_id=report.final_state_id,
                report_id=report.report_id,
                report_state_id=report_state.id,
                metadata={"completed": report.completed, "terminal_reason": report.terminal_reason},
            )
            return report

    def _delivery_gate(self, run: SupervisorRun, report: WorkerReport) -> tuple[str, ...]:
        if run.terminal_reason != "process_exited":
            raise DeliveryRejected(f"process ended as {run.terminal_reason}")
        if run.uncertain or run.uncertainty_reasons:
            raise DeliveryRejected("supervisor process outcome remains uncertain")
        if not run.reaped or run.exit_code != 0 or run.signal is not None:
            raise DeliveryRejected("worker process did not have a proven reaped zero exit")
        if not report.completed or report.terminal_reason != "completed":
            raise DeliveryRejected("worker did not make a validated completion claim")
        if report.uncertain:
            raise DeliveryRejected("worker report contains unresolved uncertainty")
        if report.monitor_severity != "fine":
            raise DeliveryRejected("terminal monitor severity is not fine")
        if report.metadata.get("durability_ok") is not True:
            raise DeliveryRejected("worker transcript durability is not certified")
        self._terminal_assessment(report)

        reported_ids = {output.artifact_id for output in report.outputs}
        paths = tuple(
            spec.path
            for spec in run.assignment.outputs
            if spec.disposition == "absent" or spec.artifact_id in reported_ids
        )
        if not paths:
            raise DeliveryRejected("assignment has no completed product paths to deliver")
        return paths

    def deliver(self, run_id: str, *, active_generation: int) -> DeliveryResult:
        """Project only certified product paths into the integration branch."""
        with self._lock:
            run = self._load(run_id)
            self._validate_generation(run, active_generation)
            report = self.collect(run_id, active_generation=active_generation)
            run = self._load(run_id)
            paths = self._delivery_gate(run, report)
            result = deliver_product(
                self.store,
                self.integration,
                delivery_id=f"{run.run_id}:{report.report_id}",
                final_state_id=report.final_state_id,
                base_state_id=run.assignment.base_state_id,
                artifact_paths=paths,
            )
            run = self._load(run_id)
            if result.conflicts:
                paths = tuple(conflict.path for conflict in result.conflicts)
                if run.phase != "conflict" or run.conflict_paths != paths:
                    self._transition(
                        run,
                        kind="delivery_conflict",
                        phase="conflict",
                        observed_state_id=result.projection.id,
                        conflict_paths=paths,
                        metadata={"projection_state_id": result.projection.id},
                    )
            else:
                assert result.integration_state is not None
                if (
                    run.phase != "delivered"
                    or run.integration_state_id != result.integration_state.id
                ):
                    self._transition(
                        run,
                        kind="delivered",
                        phase="delivered",
                        observed_state_id=result.integration_state.id,
                        integration_state_id=result.integration_state.id,
                        conflict_paths=(),
                        metadata={"projection_state_id": result.projection.id},
                    )
            return result
