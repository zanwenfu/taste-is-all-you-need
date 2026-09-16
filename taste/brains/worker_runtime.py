"""The process-local host for one worker and its colocated monitor.

The Claude Agent SDK exposes a persistent stream, not an RPC protocol:
several calls to ``query()`` may share one result, one query may produce
several results while delegated tasks continue, and a result has no request
identifier. This runtime therefore never counts queries against results. It
records submitted prompt epochs, tracks SDK task lifecycles, waits for a quiet
stream, and requires an explicit typed completion claim before a worker can be
reported complete.

Every SDK message is journaled as it arrives. Terminalization uses two
checkpoints: an immutable work state first, then a report state which points
back to that exact work state. A git commit cannot contain its own id, so the
second checkpoint is a correctness boundary rather than bookkeeping.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import contextvars
import fcntl
import hashlib
import inspect
import json
import math
import os
import re
import stat
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    TERMINAL_TASK_STATUSES,
    AssistantMessage,
    ConversationResetMessage,
    HookEventMessage,
    MirrorErrorMessage,
    ResultMessage,
    ServerToolUseBlock,
    StreamEvent,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    UserMessage,
)

from taste.brains.communication import (
    AcceptanceReceipt,
    Communicator,
    GenerationStatus,
    InboxMessage,
)
from taste.brains.contract import CONTRACT_PATH, Contract
from taste.brains.delivery import is_control_path, validate_artifact_path
from taste.brains.records import ArtifactRef, Assignment, WorkerReport, contract_digest
from taste.brains.subbrain import SubBrain, SubBrainResult, Waking
from taste.pricing import PRICES, PricingError, call_cost, ensure_priced, table_sha

__all__ = [
    "ASSIGNMENT_PATH",
    "WORKER_REPORT_PATH",
    "WORKER_RESULT_SCHEMA",
    "ContractMismatch",
    "MirrorDurabilityError",
    "ShutdownUnconfirmed",
    "WorkerRuntime",
]

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

_TRACKED_TASK_TYPES = frozenset({"local_agent", "local_workflow"})
_ABORT_REASONS = frozenset({"aborted_streaming", "aborted_tools"})
_SUCCESS_REASONS = frozenset({"completed", "end_turn"})
_RELEVANT_ORIGINS = frozenset({None, "human", "task-notification", "auto-continuation"})
_TOKEN_CLEANER = re.compile(r"[^a-z0-9_.-]+")
_EXACT_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_BUDGET_JOURNAL_SCHEMA = "taste.brains/WorkerBudgetJournal/1"
_BUDGET_JOURNAL_EVENTS = frozenset(
    {
        "connection_intent",
        "process_bound",
        "session_bound",
        "result_accounting",
        "conversation_reset",
        "connection_outcome",
        "identity_violation",
    }
)
_AUDITED_CLAUDE_AGENT_SDK_VERSION = "0.2.152"
_AUDITED_CLAUDE_CODE_VERSION = "2.1.259"
# Only executable artifacts whose exact bytes have been exercised belong in
# this allowlist.  A wheel for another platform/version fails closed until its
# bundled CLI has been measured and added explicitly.
_AUDITED_CLAUDE_CODE_SHA256 = frozenset(
    {"884baa38fe1a624be25c4a91568bf5a08b5cf4e7d7acf29b7760e3525d964898"}
)


class ContractMismatch(RuntimeError):
    """The runtime was asked to execute something other than durable truth."""


class MirrorDurabilityError(RuntimeError):
    """The SDK continued after failing to mirror part of its transcript."""


class ShutdownUnconfirmed(RuntimeError):
    """The SDK client may still own a live child, so branch handoff is unsafe."""


@dataclass(frozen=True)
class _BackgroundFailure:
    source: str
    error: Exception


@dataclass(frozen=True)
class _StreamClosed:
    error: Exception | None = None


@dataclass(frozen=True)
class _Received:
    message: Any
    submitted_epoch: int
    recorded: bool = False


@dataclass(frozen=True)
class _QuerySubmitted:
    epoch: int
    purpose: str


@dataclass(frozen=True)
class _QueryFailed:
    epoch: int
    purpose: str
    error: str


@dataclass(frozen=True)
class _PendingInbox:
    message: dict[str, Any]
    epoch: int


@dataclass(frozen=True)
class _PendingVerdicts:
    through: dict[str, int]
    epoch: int


@dataclass(frozen=True)
class _AccountingSnapshot:
    total_cost_usd: float | None
    num_turns: int | None


@dataclass(frozen=True)
class _BudgetJournalRecord:
    sequence: int
    event: str
    payload: dict[str, Any]
    digest: str
    raw: str


def _validate_budget_journal_lifecycle(records: list[_BudgetJournalRecord]) -> None:
    """Require one ordered, fail-closed provider-process lifecycle."""
    if not records:
        return
    first = records[0]
    if first.event != "connection_intent" or not _valid_cli_evidence(
        first.payload.get("cli_identity")
    ):
        raise ContractMismatch("worker budget journal has no valid connection intent")

    process_bound = False
    outcome_seen = False
    for index, record in enumerate(records[1:], start=1):
        event = record.event
        if outcome_seen:
            raise ContractMismatch("worker budget journal continued after connection outcome")
        if event == "connection_intent":
            raise ContractMismatch("worker budget journal contains another connection intent")
        if event == "process_bound":
            if process_bound or index != 1:
                raise ContractMismatch("worker budget journal process binding is out of order")
            if not _valid_cli_evidence(record.payload) or record.payload != first.payload.get(
                "cli_identity"
            ):
                raise ContractMismatch("worker budget journal process identity changed")
            process_bound = True
            continue
        if event == "connection_outcome":
            if index != len(records) - 1:
                raise ContractMismatch("worker budget journal outcome is not terminal")
            if not process_bound and record.payload.get("accounting_unknown") is not True:
                raise ContractMismatch(
                    "unbound budgeted process outcome must have unknown accounting"
                )
            outcome_seen = True
            continue
        if not process_bound:
            raise ContractMismatch("worker budget activity preceded exact process binding")


def _valid_billed_cost(value: Any) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def _valid_cli_evidence(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value)
        == {
            "claude_agent_sdk_version",
            "claude_code_version",
            "cli_sha256",
        }
        and value.get("claude_agent_sdk_version") == _AUDITED_CLAUDE_AGENT_SDK_VERSION
        and value.get("claude_code_version") == _AUDITED_CLAUDE_CODE_VERSION
        and value.get("cli_sha256") in _AUDITED_CLAUDE_CODE_SHA256
    )


def _monitor_billed_cost(report: Mapping[str, Any]) -> tuple[bool, float | None]:
    """Strictly decode aggregate monitor accounting from its durable report."""
    calls = report.get("model_calls")
    known = report.get("cost_known")
    cost = report.get("cost_usd")
    if (
        isinstance(calls, bool)
        or not isinstance(calls, int)
        or calls < 0
        or not isinstance(known, bool)
    ):
        return False, None
    if not known:
        return False, None
    if not _valid_billed_cost(cost):
        return False, None
    numeric = float(cost)
    if calls == 0 and numeric != 0:
        return False, None
    return True, numeric


def _jsonable(value: Any) -> Any:
    """Turn an SDK value into stable JSON without depending on SDK internals."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return {"python_type": type(value).__name__, "repr": repr(value)}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Replace a small sidecar durably; readers see the old or new record."""
    payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode()
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory = -1
        if directory >= 0:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def _read_regular_text_no_follow(path: Path) -> tuple[str | None, str | None]:
    """Read one control file without accepting or following a special node."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None, "missing"
    except OSError as exc:
        return None, f"not a readable regular file ({type(exc).__name__})"
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            return None, "not a regular file"
        try:
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                descriptor = -1
                return handle.read(), None
        except (OSError, UnicodeDecodeError) as exc:
            return None, f"not readable UTF-8 ({type(exc).__name__})"
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _atomic_report_write(path: Path, content: str) -> None:
    """Publish a host-owned report atomically without following its target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(tmp, flags, 0o600)
    try:
        payload = content.encode("utf-8")
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # os.replace replaces a final symlink rather than following it.
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def _reason_token(value: str | None, fallback: str) -> str:
    text = _TOKEN_CLEANER.sub("_", str(value or fallback).strip().lower()).strip("_.-")
    return text or fallback


def _record_digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_run_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > 256
        or any(ord(character) < 32 for character in value)
    ):
        raise ContractMismatch("run_id is not a stable identifier")
    return value


class WorkerRuntime:
    """Drive one :class:`SubBrain` and monitor on one live SDK client.

    ``ready`` fires only after the monitor's first cycle returns successfully.
    The opening prompt may already be in flight then: this avoids deadlocking
    a monitor whose first useful observation is the beginning of that stream.
    """

    def __init__(
        self,
        brain: SubBrain,
        monitor: Any,
        *,
        assignment: Assignment | None = None,
        run_id: str | None = None,
        communicator: Communicator | None = None,
        client_factory: Callable[[Any], Any] | None = None,
        ready: Callable[[str], Awaitable[None] | None] | None = None,
        poll_interval: float = 0.05,
        terminal_quiet_period: float = 0.25,
        shutdown_timeout: float = 10.0,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if terminal_quiet_period <= 0:
            raise ValueError("terminal_quiet_period must be positive")
        if shutdown_timeout <= 0:
            raise ValueError("shutdown_timeout must be positive")
        self.brain = brain
        self.monitor = monitor
        self.assignment = assignment
        self.run_id = run_id or ""
        self._run_id_supplied = run_id is not None
        if communicator is not None:
            if not isinstance(communicator, Communicator):
                raise TypeError("communicator must be a Communicator")
            if communicator.store is not brain.store:
                raise ContractMismatch("communicator must use the worker's exact Store")
            if assignment is None:
                raise ContractMismatch("typed communication requires an exact Assignment")
        self.communicator = communicator
        # ``client_factory`` is a useful simulation seam for ordinary workers,
        # but it is not part of the audited budget boundary.  Remember how the
        # client was selected rather than trusting capabilities that an
        # arbitrary injected object can claim later (for example
        # ``abort_now()``/``wait_reaped()``).
        self._client_factory_injected = client_factory is not None
        self.client_factory = client_factory or self._make_client
        self.ready = ready
        self.poll_interval = poll_interval
        self.terminal_quiet_period = terminal_quiet_period
        self.shutdown_timeout = shutdown_timeout

        self._client: Any = None
        self._connected = False
        self._connect_attempted = False
        self._shutdown_confirmed = False
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._background_stop = asyncio.Event()
        self._monitor_entered = asyncio.Event()
        self._monitor_lock = asyncio.Lock()
        self._query_lock = asyncio.Lock()
        self._delivery_lock = asyncio.Lock()
        self._query_context: contextvars.ContextVar[tuple[str, dict[str, int]] | None] = (
            contextvars.ContextVar(f"taste_worker_query_{id(self)}", default=None)
        )
        self._monitor_task: asyncio.Task[Any] | None = None
        self._inbox_task: asyncio.Task[Any] | None = None
        self._reader_task: asyncio.Task[Any] | None = None
        self._backgrounds_stopped = False

        self._query_epoch = 0
        self._submitted_query_epoch = 0
        self._query_writes = 0
        self._covered_query_epoch = 0
        self._pending_inbox: list[_PendingInbox] = []
        self._typed_inbox_items: dict[str, InboxMessage] = {}
        self._pending_verdicts: list[_PendingVerdicts] = []
        self._accepted_inbox_evidence: set[str] = set()
        self._accepted_verdict_evidence: dict[str, int] = {}
        self._delivered_inbox_epochs: dict[str, int] = {}
        self._delivered_verdict_history: list[_PendingVerdicts] = []
        self._active_tasks: set[str] = set()
        self._results: list[ResultMessage] = []
        self._latest_relevant: ResultMessage | None = None
        self._sticky_result_errors: list[str] = []
        self._protocol_uncertainty: list[str] = []
        self._completion_claim: dict[str, Any] | None = None
        self._completion_claim_epoch = 0
        self._candidate: ResultMessage | None = None
        self._candidate_epoch = 0
        self._monitor_actions: list[tuple[Any, str]] = []
        self._expected_abort_tokens = 0
        self._delegated_task_seen = False
        self._final_drained = False
        self._tail_uncertainty: list[str] = []
        self._allow_session_change = False
        self._accounting_latest: ResultMessage | _AccountingSnapshot | None = None
        self._accounting_cost_before_reset = 0.0
        self._accounting_turns_before_reset = 0
        self._accounting_unknown = False
        self._budget_reset_seen = False
        self._budget_connection_admitted = False
        self._budget_connection_closed = False
        self._budget_process_bound = False
        self._budget_abort_process: Any = None
        self._budget_abort_transport: Any = None
        self._budget_abort_invoked = False
        self._budget_cli_path: Path | None = None
        self._budget_cli_evidence: dict[str, Any] | None = None
        self._latest_budget_receipt: _BudgetJournalRecord | None = None
        self._budget_result_digests: dict[str, str] = {}

        self._accepted_contract_text = ""
        self._accepted_assignment_text: str | None = None
        self._accepted_control_entries: dict[str, tuple[str, str]] = {}
        self._session_binding: dict[str, Any] | None = None

    @staticmethod
    def _make_client(options: Any) -> Any:
        from claude_agent_sdk import ClaudeSDKClient

        return ClaudeSDKClient(options=options)

    # --------------------------------------------------------------- durable input

    def _validate_durable_input(self) -> Assignment | None:
        """Require exact checkpointed control records and exact input blobs."""
        accepted_head = self.brain.branch.head
        raw = self.brain.branch.read(CONTRACT_PATH, at=accepted_head)
        if raw is None:
            raise ContractMismatch(
                f"{self.brain.contract.identity}: no checkpointed contract at {CONTRACT_PATH}"
            )
        try:
            accepted = Contract.from_json(raw)
        except Exception as exc:
            raise ContractMismatch("the checkpointed contract is not readable") from exc
        if accepted != self.brain.contract:
            raise ContractMismatch(
                "the runtime contract differs from the checkpointed contract; "
                "refusing a split worker/monitor brief"
            )
        contract_entry = self.brain.branch.backend.entry_at(accepted_head.id, CONTRACT_PATH)
        if contract_entry is None:
            raise ContractMismatch("the checkpointed contract has no Git tree entry")
        self._accepted_control_entries[CONTRACT_PATH] = (
            contract_entry.mode,
            contract_entry.sha,
        )
        live_contract_text, live_contract_error = _read_regular_text_no_follow(
            self.brain.branch.path(CONTRACT_PATH)
        )
        if live_contract_error is not None or live_contract_text != raw:
            raise ContractMismatch("the live contract differs from the checkpointed contract")
        self._accepted_contract_text = raw

        assignment = self.assignment
        assignment_raw = self.brain.branch.read(ASSIGNMENT_PATH)
        if assignment is not None and assignment_raw is None:
            raise ContractMismatch(
                f"the supplied assignment was not checkpointed at {ASSIGNMENT_PATH}"
            )
        if assignment_raw is not None:
            try:
                durable = Assignment.from_json(assignment_raw)
            except Exception as exc:
                raise ContractMismatch("the checkpointed assignment is not readable") from exc
            if assignment is not None and durable != assignment:
                raise ContractMismatch("the supplied assignment differs from the checkpointed one")
            assignment = durable
            assignment_entry = self.brain.branch.backend.entry_at(accepted_head.id, ASSIGNMENT_PATH)
            if assignment_entry is None:
                raise ContractMismatch("the checkpointed assignment has no Git tree entry")
            self._accepted_control_entries[ASSIGNMENT_PATH] = (
                assignment_entry.mode,
                assignment_entry.sha,
            )
            live_assignment_text, live_assignment_error = _read_regular_text_no_follow(
                self.brain.branch.path(ASSIGNMENT_PATH)
            )
            if live_assignment_error is not None or live_assignment_text != assignment_raw:
                raise ContractMismatch(
                    "the live assignment differs from the checkpointed assignment"
                )
            self._accepted_assignment_text = assignment_raw

        if assignment is not None:
            if assignment.contract != accepted:
                raise ContractMismatch("the assignment and checkpointed contract disagree")
            if assignment.contract_digest != contract_digest(accepted):
                raise ContractMismatch("the assignment contract digest is wrong")
            if accepted.inputs != tuple(item.path for item in assignment.inputs):
                raise ContractMismatch(
                    "contract inputs do not match the exact structured input paths"
                )
            if accepted.outputs != tuple(item.path for item in assignment.outputs):
                raise ContractMismatch(
                    "contract outputs do not match the exact structured output paths"
                )
            if assignment.model != self.brain.model:
                raise ContractMismatch(
                    f"assignment model {assignment.model!r} != runtime model {self.brain.model!r}"
                )
            for artifact in (*assignment.inputs, *assignment.outputs):
                try:
                    validate_artifact_path(artifact.path)
                except ValueError as exc:
                    raise ContractMismatch(
                        f"assignment artifact {artifact.artifact_id!r} has an invalid "
                        f"product path: {exc}"
                    ) from exc
            if _EXACT_OBJECT_ID.fullmatch(assignment.base_state_id) is None:
                raise ContractMismatch("assignment base_state_id must be a full immutable state id")
            try:
                base_state = self.brain.store.state(assignment.base_state_id)
            except Exception as exc:
                raise ContractMismatch("the assignment base state does not exist") from exc
            if base_state.id != assignment.base_state_id:
                raise ContractMismatch("assignment base_state_id must be a full immutable state id")
            if not self.brain.branch.backend.is_ancestor(
                assignment.base_state_id, self.brain.branch.head.id
            ):
                raise ContractMismatch(
                    "the assignment base state is not an ancestor of the worker branch"
                )
            for artifact in assignment.inputs:
                if _EXACT_OBJECT_ID.fullmatch(artifact.state_id) is None:
                    raise ContractMismatch(
                        f"input {artifact.artifact_id!r} must name a full immutable state id"
                    )
                try:
                    state = self.brain.store.state(artifact.state_id)
                except Exception as exc:
                    raise ContractMismatch(
                        f"input {artifact.artifact_id!r} points to an unknown state"
                    ) from exc
                if state.id != artifact.state_id:
                    raise ContractMismatch(
                        f"input {artifact.artifact_id!r} must name a full immutable state id"
                    )
                if state.meta.session != self.brain.store.session:
                    raise ContractMismatch(
                        f"input {artifact.artifact_id!r} comes from another session"
                    )
                if state.meta.branch != artifact.branch:
                    raise ContractMismatch(
                        f"input {artifact.artifact_id!r} has the wrong source branch"
                    )
                entry = self.brain.store.backend.entry_at(state.id, artifact.path)
                if entry is not None and entry.mode == "040000":
                    raise ContractMismatch(
                        f"input {artifact.artifact_id!r} names a directory; "
                        "select product files explicitly"
                    )
                if state.blob(artifact.path) != artifact.blob_id:
                    raise ContractMismatch(
                        f"input {artifact.artifact_id!r} no longer names the declared bytes"
                    )

        exact_record = self._accepted_assignment_text or self._accepted_contract_text
        derived_run_id = f"worker-run.{_record_digest(exact_record).removeprefix('sha256:')}"
        if assignment is not None and accepted.budget_usd is not None:
            if self._run_id_supplied and self.run_id != derived_run_id:
                raise ContractMismatch(
                    "budgeted run_id must be derived from the exact durable assignment"
                )
            self.run_id = derived_run_id
        elif not self._run_id_supplied:
            self.run_id = derived_run_id
        _validate_run_id(self.run_id)
        return assignment

    def _binding_identity(self) -> dict[str, Any]:
        identity: dict[str, Any] = {
            "schema": "taste.brains/RuntimeSessionBinding/1",
            "worker": self.brain.contract.identity,
            "run_id": self.run_id,
            "contract_digest": contract_digest(self.brain.contract),
        }
        if self.assignment is not None:
            identity.update(
                {
                    "assignment_id": self.assignment.assignment_id,
                    "assignment_digest": _record_digest(
                        self._accepted_assignment_text or self.assignment.to_json()
                    ),
                    "generation": self.assignment.generation,
                    "attempt": self.assignment.attempt,
                }
            )
        return identity

    def _binding_path(self) -> Path:
        encoded = json.dumps(
            self._binding_identity(), sort_keys=True, separators=(",", ":")
        ).encode()
        suffix = hashlib.sha256(encoded).hexdigest()[:24]
        return self.brain.store.sidecar(
            "runtime-session", self.brain.contract.identity, f".{suffix}"
        )

    def _budget_journal_path(self) -> Path:
        """Independent append-only authority for one exact budgeted run."""
        encoded = json.dumps(
            self._binding_identity(), sort_keys=True, separators=(",", ":")
        ).encode()
        suffix = hashlib.sha256(encoded).hexdigest()
        return self.brain.store.sidecar(
            "worker-budget", self.brain.contract.identity, f".{suffix}.jsonl"
        )

    def _decode_budget_journal(self, data: bytes) -> list[_BudgetJournalRecord]:
        if not data:
            return []
        if not data.endswith(b"\n"):
            raise ContractMismatch("the worker budget journal has a torn final record")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ContractMismatch("the worker budget journal is not UTF-8") from exc

        identity = self._binding_identity()
        records: list[_BudgetJournalRecord] = []
        previous_digest: str | None = None
        for sequence, raw in enumerate(text.splitlines(keepends=True), start=1):
            if raw == "\n":
                raise ContractMismatch("the worker budget journal contains an empty record")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ContractMismatch("the worker budget journal contains invalid JSON") from exc
            if not isinstance(value, dict) or set(value) != {
                "schema",
                "sequence",
                "previous_sha256",
                "identity",
                "event",
                "payload",
            }:
                raise ContractMismatch("the worker budget journal record shape is invalid")
            if (
                value.get("schema") != _BUDGET_JOURNAL_SCHEMA
                or value.get("sequence") != sequence
                or value.get("previous_sha256") != previous_digest
                or value.get("identity") != identity
                or value.get("event") not in _BUDGET_JOURNAL_EVENTS
                or not isinstance(value.get("payload"), dict)
            ):
                raise ContractMismatch("the worker budget journal chain is invalid")
            canonical = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            if canonical != raw:
                raise ContractMismatch("the worker budget journal is not canonical")
            digest = _record_digest(raw)
            records.append(
                _BudgetJournalRecord(
                    sequence=sequence,
                    event=value["event"],
                    payload=value["payload"],
                    digest=digest,
                    raw=raw,
                )
            )
            previous_digest = digest
        _validate_budget_journal_lifecycle(records)
        return records

    @staticmethod
    def _read_locked_file(descriptor: int) -> bytes:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)

    def _budget_journal_records(self) -> list[_BudgetJournalRecord]:
        path = self._budget_journal_path()
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise ContractMismatch("the worker budget journal is unreadable") from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ContractMismatch("the worker budget journal is not a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            return self._decode_budget_journal(self._read_locked_file(descriptor))
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _append_budget_journal(
        self, event: str, payload: Mapping[str, Any]
    ) -> _BudgetJournalRecord:
        if event not in _BUDGET_JOURNAL_EVENTS:
            raise ContractMismatch("unknown worker budget journal event")
        path = self._budget_journal_path()
        flags = (
            os.O_RDWR
            | os.O_APPEND
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise ContractMismatch("the worker budget journal cannot be opened") from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ContractMismatch("the worker budget journal is not a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            records = self._decode_budget_journal(self._read_locked_file(descriptor))
            if event == "connection_intent" and records:
                raise ContractMismatch(
                    "budgeted worker run already has a durable connection intent"
                )
            if event != "connection_intent" and (
                not records or records[0].event != "connection_intent"
            ):
                raise ContractMismatch("worker budget journal has no connection intent")
            previous = records[-1].digest if records else None
            value = {
                "schema": _BUDGET_JOURNAL_SCHEMA,
                "sequence": len(records) + 1,
                "previous_sha256": previous,
                "identity": self._binding_identity(),
                "event": event,
                "payload": dict(payload),
            }
            raw = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            record = _BudgetJournalRecord(
                sequence=len(records) + 1,
                event=event,
                payload=dict(payload),
                digest=_record_digest(raw),
                raw=raw,
            )
            _validate_budget_journal_lifecycle([*records, record])
            encoded = raw.encode("utf-8")
            written = 0
            while written < len(encoded):
                count = os.write(descriptor, encoded[written:])
                if count <= 0:  # pragma: no cover - os.write contract
                    raise OSError("short append to worker budget journal")
                written += count
            os.fsync(descriptor)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            self._latest_budget_receipt = record
            return record
        except (TypeError, ValueError) as exc:
            raise ContractMismatch("worker budget journal payload is not canonical JSON") from exc
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _prior_runtime_evidence(self) -> list[dict[str, Any]]:
        risky = {
            "runtime_budget_connection_intent",
            "runtime_query_intent",
            "runtime_query_submitted",
            "sdk_message",
        }
        return [
            event
            for event in self.brain._recorded_turns()
            if event.get("run_id") == self.run_id and event.get("kind") in risky
        ]

    def _admit_budgeted_connection(self, options: Any) -> None:
        """Spend-once fence, durably written before a provider process exists.

        A dead CLI can leave an in-flight provider request whose exact charge
        is unknowable.  Reusing the same assignment allowance on a replacement
        process would therefore not be a hard cap.  One exact budgeted run gets
        one connection intent; any surviving intent blocks automatic replay.
        """
        self._assert_budgeted_connection_fresh()
        if not _valid_cli_evidence(self._budget_cli_evidence):
            raise ContractMismatch("budgeted connection has no audited CLI identity")
        receipt = self._append_budget_journal(
            "connection_intent",
            {
                "model": self.brain.model,
                "contract_budget_usd": self.brain.contract.budget_usd,
                "provider_threshold_usd": getattr(options, "max_budget_usd", None),
                "pricing_table_sha": table_sha(),
                "cli_identity": dict(self._budget_cli_evidence),
                "tools": list(getattr(options, "tools", None) or ()),
                "disallowed_tools": list(getattr(options, "disallowed_tools", None) or ()),
                "strict_mcp_config": getattr(options, "strict_mcp_config", None),
                "fallback_model": getattr(options, "fallback_model", None),
            },
        )
        self.brain.branch.turn(
            kind="runtime_budget_connection_intent",
            run_id=self.run_id,
            receipt_sha256=receipt.digest,
            sequence=receipt.sequence,
        )
        admission_state = self.brain.checkpoint(
            "durable budget connection intent before provider process"
        )
        if not any(
            event.get("kind") == "runtime_budget_connection_intent"
            and event.get("run_id") == self.run_id
            and event.get("receipt_sha256") == receipt.digest
            for event in admission_state.transcript.turns
        ):
            raise ContractMismatch("budget connection intent was not checkpointed exactly")
        self._budget_connection_admitted = True

    def _assert_budgeted_connection_fresh(self) -> None:
        records = self._budget_journal_records()
        if records:
            raise ContractMismatch(
                "budgeted worker run already has a durable connection intent; "
                "automatic provider replay is forbidden"
            )
        if os.path.lexists(self._binding_path()):
            raise ContractMismatch(
                "budgeted runtime session binding exists without its spend authority"
            )
        if self._prior_runtime_evidence():
            raise ContractMismatch(
                "budgeted worker has prior provider evidence without a spend authority"
            )

    def _record_budget_event(self, event: str, payload: Mapping[str, Any]) -> None:
        if self.brain.contract.budget_usd is None:
            return
        if not self._budget_connection_admitted:
            raise ContractMismatch("budgeted provider activity preceded durable admission")
        receipt = self._append_budget_journal(event, payload)
        self.brain.branch.turn(
            kind="runtime_budget_receipt",
            run_id=self.run_id,
            event=event,
            receipt_sha256=receipt.digest,
            sequence=receipt.sequence,
        )

    def _binding_accounting(self) -> dict[str, Any]:
        accounting: dict[str, Any] = {
            "cost_usd_before_reset": self._accounting_cost_before_reset,
            "turns_before_reset": self._accounting_turns_before_reset,
            "accounting_unknown": self._accounting_unknown,
        }
        if self._accounting_latest is not None:
            accounting["current_accounting"] = {
                "cost_usd": self._accounting_latest.total_cost_usd,
                "turns": self._accounting_latest.num_turns,
            }
        return accounting

    def _restore_binding_accounting(self, raw: Mapping[str, Any]) -> None:
        cost = raw.get("cost_usd_before_reset", 0.0)
        turns = raw.get("turns_before_reset", 0)
        unknown = raw.get("accounting_unknown", False)
        if (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(cost)
            or cost < 0
            or isinstance(turns, bool)
            or not isinstance(turns, int)
            or turns < 0
            or not isinstance(unknown, bool)
        ):
            raise ContractMismatch("the runtime session accounting is unreadable")
        self._accounting_cost_before_reset = float(cost)
        self._accounting_turns_before_reset = turns
        self._accounting_unknown = unknown
        current = raw.get("current_accounting")
        if current is None:
            self._accounting_latest = None
            return
        if not isinstance(current, Mapping):
            raise ContractMismatch("the current runtime session accounting is unreadable")
        current_cost = current.get("cost_usd")
        current_turns = current.get("turns")
        if (
            current_cost is not None
            and (
                isinstance(current_cost, bool)
                or not isinstance(current_cost, (int, float))
                or not math.isfinite(current_cost)
                or current_cost < 0
            )
        ) or (
            current_turns is not None
            and (
                isinstance(current_turns, bool)
                or not isinstance(current_turns, int)
                or current_turns < 0
            )
        ):
            raise ContractMismatch("the current runtime session accounting is unreadable")
        self._accounting_latest = _AccountingSnapshot(
            total_cost_usd=(float(current_cost) if current_cost is not None else None),
            num_turns=current_turns,
        )

    def _resume_session_id(self) -> str | None:
        path = self._binding_path()
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ContractMismatch("the runtime session binding is unreadable") from exc
        expected = self._binding_identity()
        if not isinstance(raw, dict) or any(
            raw.get(key) != value for key, value in expected.items()
        ):
            raise ContractMismatch("the runtime session binding belongs to another assignment")
        self._restore_binding_accounting(raw)
        reset_pending = raw.get("reset_pending", False)
        if not isinstance(reset_pending, bool):
            raise ContractMismatch("the runtime session reset marker is unreadable")
        self._session_binding = raw
        if reset_pending:
            previous = raw.get("previous_session_id")
            if not isinstance(previous, str) or not previous or raw.get("session_id") is not None:
                raise ContractMismatch("the runtime session reset marker is incomplete")
            # A reset was durable before the process stopped. Starting without
            # ``resume`` is essential: the outgoing session is no longer a
            # valid continuation target even though no replacement was bound.
            self._allow_session_change = True
            self.brain._session_id = None
            return None
        session_id = raw.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ContractMismatch("the runtime session binding has no session id")
        self.brain._session_id = session_id
        return session_id

    def _bind_session(self, session_id: str) -> None:
        if not session_id:
            return
        current = self._session_binding or self._binding_identity()
        if current.get("session_id") == session_id:
            self.brain._session_id = session_id
            return
        reset_pending = current.get("reset_pending") is True
        prior = current.get("previous_session_id") if reset_pending else current.get("session_id")
        if reset_pending and prior == session_id:
            raise ContractMismatch(
                f"SDK tried to resume outgoing session {session_id!r} after a "
                "ConversationResetMessage"
            )
        if isinstance(prior, str) and prior and not self._allow_session_change:
            raise ContractMismatch(
                f"SDK session changed from {prior!r} to {session_id!r} without a "
                "ConversationResetMessage"
            )
        bound = {
            **self._binding_identity(),
            **self._binding_accounting(),
            "session_id": session_id,
        }
        if isinstance(prior, str) and prior:
            bound["previous_session_id"] = prior
        # Replacing the whole record binds the first post-reset session and
        # clears ``reset_pending`` in the same durable operation.
        self._record_budget_event("session_bound", {"binding": bound})
        _atomic_json(self._binding_path(), bound)
        self._session_binding = bound
        self._allow_session_change = False
        self.brain._session_id = session_id
        self.brain.branch.turn(
            kind="runtime_session_bound",
            run_id=self.run_id,
            session_id=session_id,
            previous_session_id=prior,
        )

    def _persist_conversation_reset(self, message: ConversationResetMessage) -> None:
        current = self._session_binding or self._binding_identity()
        if current.get("reset_pending") is True:
            raise ContractMismatch(
                "received another ConversationResetMessage before binding its replacement"
            )
        outgoing = getattr(message, "session_id", None) or current.get("session_id")
        bound_session = current.get("session_id")
        if not isinstance(outgoing, str) or not outgoing:
            raise ContractMismatch("ConversationResetMessage has no outgoing session id")
        if isinstance(bound_session, str) and bound_session and bound_session != outgoing:
            raise ContractMismatch(
                f"ConversationResetMessage names {outgoing!r}, not bound session {bound_session!r}"
            )

        latest = self._accounting_latest
        unknown = self._accounting_unknown
        cost = self._accounting_cost_before_reset
        turns = self._accounting_turns_before_reset
        if (
            latest is None
            or not _valid_billed_cost(latest.total_cost_usd)
            or isinstance(latest.num_turns, bool)
            or not isinstance(latest.num_turns, int)
            or latest.num_turns < 0
        ):
            unknown = True
        else:
            try:
                combined = math.fsum((cost, float(latest.total_cost_usd)))
            except OverflowError:
                combined = math.inf
            if math.isfinite(combined):
                cost = combined
            else:
                unknown = True
            turns += latest.num_turns
        pending = {
            **self._binding_identity(),
            "reset_pending": True,
            "previous_session_id": outgoing,
            "cost_usd_before_reset": cost,
            "turns_before_reset": turns,
            "accounting_unknown": unknown,
            "reset_uuid": getattr(message, "uuid", None),
            "new_conversation_id": getattr(message, "new_conversation_id", None),
        }
        # This precedes transcript journaling so even a crash at the next
        # instruction cannot make a restart resume the outgoing conversation.
        self._record_budget_event("conversation_reset", {"binding": pending})
        _atomic_json(self._binding_path(), pending)
        self._session_binding = pending
        self._accounting_cost_before_reset = cost
        self._accounting_turns_before_reset = turns
        self._accounting_unknown = unknown
        self._accounting_latest = None
        self._allow_session_change = True
        self.brain._session_id = None

    def _persist_current_accounting(self) -> None:
        current = self._session_binding
        if current is None or current.get("reset_pending") is True:
            raise ContractMismatch(
                "cannot persist Result accounting without a bound runtime session"
            )
        updated = {**current, **self._binding_accounting()}
        latest = self._accounting_latest
        result_evidence: dict[str, Any] | None = None
        if isinstance(latest, ResultMessage):
            encoded = json.dumps(_jsonable(latest), sort_keys=True, separators=(",", ":"))
            result_evidence = {
                "uuid": latest.uuid,
                "session_id": latest.session_id,
                "message_sha256": _record_digest(encoded),
                "total_cost_usd": latest.total_cost_usd,
                "num_turns": latest.num_turns,
                "model_usage": _jsonable(latest.model_usage),
            }
        self._record_budget_event(
            "result_accounting",
            {"binding": updated, "result": result_evidence},
        )
        _atomic_json(self._binding_path(), updated)
        self._session_binding = updated

    # --------------------------------------------------------------- recording/querying

    def _record_message(self, message: Any) -> None:
        self._final_drained = False
        if isinstance(message, ConversationResetMessage):
            self._persist_conversation_reset(message)
        session_id = getattr(message, "session_id", None)
        if not session_id:
            data = getattr(message, "data", None)
            if isinstance(data, Mapping):
                session_id = data.get("session_id")
        if session_id and not isinstance(message, ConversationResetMessage):
            self._bind_session(str(session_id))
        if isinstance(message, ResultMessage):
            # Persist the running total in stream order, before publishing the
            # transcript event that says this billed result was observed.
            self._accounting_latest = message
            self._persist_current_accounting()
        self.brain.branch.turn(
            kind="sdk_message",
            run_id=self.run_id,
            budget_receipt_sha256=(
                None if self._latest_budget_receipt is None else self._latest_budget_receipt.digest
            ),
            message_type=type(message).__name__,
            message=_jsonable(message),
        )

    def _abort_budgeted_cli_now(self) -> None:
        """Synchronously stop the local CLI after its cost counter resets.

        Public ``interrupt()`` is a request/response control message and can
        wait for a full provider turn.  At a reset that is too late: the CLI's
        fresh ledger can admit queued input while this process is awaiting the
        acknowledgement.  The audited production SDK exposes the local anyio
        process through its exact subprocess transport; ``kill()`` sends the
        signal synchronously.
        """
        self._budget_abort_invoked = True
        transport = self._budget_abort_transport
        process = self._budget_abort_process
        if process is None or not callable(getattr(process, "kill", None)):
            raise RuntimeError("budgeted local CLI kill boundary was not armed")
        if getattr(process, "returncode", None) is not None:
            return
        # Prevent another transport write even before the OS updates
        # ``returncode``.  This is deliberately paired with the runtime-level
        # reset flag checked under the query lock.
        if hasattr(transport, "_ready"):
            transport._ready = False
        try:
            process.kill()
        except ProcessLookupError:
            # The process crossed its exit boundary between the return-code
            # observation and kill.  Either way it cannot admit more input.
            return

    @staticmethod
    def _audited_sdk_client_class() -> type[Any]:
        """Return the one SDK client class admitted for budgeted execution."""
        try:
            from importlib.metadata import version

            from claude_agent_sdk import ClaudeSDKClient
        except Exception as exc:  # pragma: no cover - installed SDK is required
            raise ContractMismatch("cannot inspect the budgeted SDK client boundary") from exc
        if version("claude-agent-sdk") != _AUDITED_CLAUDE_AGENT_SDK_VERSION:
            raise ContractMismatch("budgeted worker requires audited claude-agent-sdk 0.2.152")
        return ClaudeSDKClient

    @classmethod
    def _preflight_budgeted_cli(cls) -> tuple[Path, dict[str, Any]]:
        """Verify exact bundled CLI bytes before any provider-capable process starts."""
        cls._audited_sdk_client_class()
        try:
            from importlib.metadata import distribution

            import claude_agent_sdk

            installed = distribution("claude-agent-sdk")
            cli_name = "claude.exe" if os.name == "nt" else "claude"
            relative = f"claude_agent_sdk/_bundled/{cli_name}"
            matches = [item for item in (installed.files or ()) if item.as_posix() == relative]
            if len(matches) != 1:
                raise ContractMismatch(
                    "audited SDK distribution has no unique bundled CLI artifact"
                )
            packaged = matches[0]
            packaged_hash = packaged.hash
            if packaged_hash is None or packaged_hash.mode != "sha256":
                raise ContractMismatch("bundled CLI has no SHA-256 distribution record")
            encoded = packaged_hash.value
            recorded_digest = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).hex()
            cli_path = Path(packaged.locate())
            package_root = Path(claude_agent_sdk.__file__ or "").resolve().parent
            expected_path = (package_root / "_bundled" / cli_name).resolve(strict=True)
            metadata = cli_path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or cli_path.resolve(strict=True) != expected_path
                or (packaged.size is not None and metadata.st_size != packaged.size)
            ):
                raise ContractMismatch("bundled CLI distribution entry is not an exact file")
        except ContractMismatch:
            raise
        except Exception as exc:
            raise ContractMismatch("budgeted bundled CLI metadata is unreadable") from exc

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(cli_path, flags)
        except OSError as exc:
            raise ContractMismatch("budgeted bundled CLI cannot be opened exactly") from exc
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != metadata.st_dev
                or opened.st_ino != metadata.st_ino
                or opened.st_size != metadata.st_size
            ):
                raise ContractMismatch("bundled CLI changed while its identity was read")
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
        finally:
            os.close(descriptor)
        cli_digest = digest.hexdigest()
        if cli_digest != recorded_digest:
            raise ContractMismatch("bundled CLI bytes disagree with their distribution record")
        if cli_digest not in _AUDITED_CLAUDE_CODE_SHA256:
            raise ContractMismatch("bundled CLI bytes are not in the audited artifact allowlist")
        return expected_path, {
            "claude_agent_sdk_version": _AUDITED_CLAUDE_AGENT_SDK_VERSION,
            "claude_code_version": _AUDITED_CLAUDE_CODE_VERSION,
            "cli_sha256": cli_digest,
        }

    def _assert_budgeted_client_identity(self) -> None:
        if self.brain.contract.budget_usd is None:
            return
        client_class = self._audited_sdk_client_class()
        if type(self._client) is not client_class:
            raise ContractMismatch("budgeted worker requires the exact built-in SDK client")
        if getattr(self._client, "_custom_transport", None) is not None:
            raise ContractMismatch("budgeted worker forbids a custom SDK transport")

    def _arm_budgeted_process_boundary(self) -> dict[str, Any]:
        """Capture a proven synchronous kill handle before the first query."""
        if self.brain.contract.budget_usd is None:
            raise ContractMismatch("an unbudgeted worker has no budgeted process boundary")
        self._assert_budgeted_client_identity()
        try:
            from claude_agent_sdk._internal.transport.subprocess_cli import (
                SubprocessCLITransport,
            )
        except Exception as exc:  # pragma: no cover - installed SDK is required
            raise ContractMismatch("cannot inspect the budgeted SDK process boundary") from exc
        transport = getattr(self._client, "_transport", None)
        if type(transport) is not SubprocessCLITransport:
            raise ContractMismatch(
                "budgeted worker requires the audited local subprocess transport"
            )
        process = getattr(transport, "_process", None)
        if (
            process is None
            or not callable(getattr(process, "kill", None))
            or getattr(process, "returncode", None) is not None
        ):
            raise ContractMismatch("budgeted worker has no live local CLI process handle")
        self._budget_abort_transport = transport
        self._budget_abort_process = process
        raw_cli_path = getattr(transport, "_cli_path", None)
        if not isinstance(raw_cli_path, str) or not raw_cli_path:
            self._abort_budgeted_cli_now()
            raise ContractMismatch("budgeted worker has no exact local CLI identity")
        try:
            cli_path = Path(raw_cli_path).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            self._abort_budgeted_cli_now()
            raise ContractMismatch("budgeted worker local CLI identity is unreadable") from exc
        try:
            observed_path, observed_evidence = self._preflight_budgeted_cli()
        except Exception:
            self._abort_budgeted_cli_now()
            raise
        if (
            self._budget_cli_path is None
            or cli_path != self._budget_cli_path
            or observed_path != self._budget_cli_path
            or observed_evidence != self._budget_cli_evidence
        ):
            self._abort_budgeted_cli_now()
            raise ContractMismatch("budgeted CLI process differs from its preflight identity")
        return observed_evidence

    async def _confirm_budget_abort_reaped(self) -> bool:
        if not self._budget_abort_invoked:
            return True
        if self._budget_abort_process is not None:
            try:
                await asyncio.wait_for(
                    self._budget_abort_process.wait(), timeout=self.shutdown_timeout
                )
            except Exception:
                return False
            return getattr(self._budget_abort_process, "returncode", None) is not None
        return False

    def _budgeted_message_violation(self, message: Any) -> str | None:
        """Validate cost identity before a budgeted frame becomes accounting."""
        if isinstance(message, AssistantMessage):
            if message.model != self.brain.model:
                return (
                    f"assistant model {message.model!r} differs from exact "
                    f"assignment model {self.brain.model!r}"
                )
            if any(isinstance(block, ServerToolUseBlock) for block in message.content):
                return "assistant invoked an unpriced server-side tool"
            return None
        if not isinstance(message, ResultMessage):
            return None
        if not _valid_billed_cost(message.total_cost_usd):
            return "budgeted Result has no finite billed total"
        if (
            isinstance(message.num_turns, bool)
            or not isinstance(message.num_turns, int)
            or message.num_turns < 0
        ):
            return "budgeted Result has invalid cumulative turns"
        if not isinstance(message.session_id, str) or not message.session_id:
            return "budgeted Result has no exact session id"
        bound = self._session_binding
        if (
            isinstance(bound, Mapping)
            and bound.get("reset_pending") is not True
            and isinstance(bound.get("session_id"), str)
            and bound.get("session_id") != message.session_id
        ):
            return "budgeted Result changed session without a reset"

        previous = self._accounting_latest
        if previous is not None:
            previous_cost = previous.total_cost_usd
            previous_turns = previous.num_turns
            if not _valid_billed_cost(previous_cost) or (
                isinstance(previous_turns, bool)
                or not isinstance(previous_turns, int)
                or previous_turns < 0
            ):
                return "prior budget accounting is invalid"
            if float(message.total_cost_usd) < float(previous_cost):
                return "budgeted Result cumulative cost decreased"
            if message.num_turns < previous_turns:
                return "budgeted Result cumulative turns decreased"

        message_uuid = message.uuid
        if not isinstance(message_uuid, str) or not message_uuid:
            return "budgeted Result has no stable UUID"
        message_digest = _record_digest(
            json.dumps(_jsonable(message), sort_keys=True, separators=(",", ":"))
        )
        prior_digest = self._budget_result_digests.get(message_uuid)
        if prior_digest is not None and prior_digest != message_digest:
            return "budgeted Result UUID was replayed with different bytes"

        usage_by_model = message.model_usage
        if not isinstance(usage_by_model, Mapping) or not usage_by_model:
            return "budgeted Result has no complete per-model usage"
        summed_cost = 0.0
        saw_worker_model = False
        required_counts = {
            "inputTokens",
            "outputTokens",
            "cacheReadInputTokens",
            "cacheCreationInputTokens",
            "webSearchRequests",
            "contextWindow",
            "maxOutputTokens",
        }
        # ``model_usage`` reports every model the CLI billed, not the one this
        # contract asked for: Claude Code bills a little Haiku for a session
        # title on every session. Measured live, twice -- a worker wrote its
        # artifact correctly, spent $0.034 of $8.25, and the run was failed at
        # the reporting boundary because 13 tokens of title were read as an
        # identity violation.
        #
        # So the claim narrows to what it should always have been: our model
        # did our work, every entry is priced by *its own* rates, and every
        # dollar lands against this contract. An unknown model still fails
        # closed, because an unpriced entry cannot be accounted for at all.
        for name, usage in usage_by_model.items():
            if not isinstance(usage, Mapping):
                return "budgeted Result model usage is malformed"
            canonical = usage.get("canonicalModel")
            if not isinstance(canonical, str) or not canonical:
                return "budgeted Result model usage has no canonical model"
            if usage.get("provider") != "firstParty":
                return (
                    "budgeted Result used a different provider: "
                    f"{usage.get('provider')!r}"
                )
            # The key names the exact snapshot billed; the canonical name may
            # be the family it belongs to. Price by whichever one is priced,
            # and refuse when neither is.
            billed_model = name if name in PRICES else canonical
            try:
                entry_price = ensure_priced(billed_model)
            except PricingError:
                return (
                    "budgeted Result billed an unpriced model: "
                    f"{name!r} (canonical {canonical!r})"
                )
            if canonical == self.brain.model or name == self.brain.model:
                saw_worker_model = True
            for field in required_counts:
                value = usage.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    return f"budgeted Result has invalid {field}"
            if usage["webSearchRequests"] != 0:
                return "budgeted Result contains separately billed web searches"
            if usage["contextWindow"] > entry_price.context_window:
                return (
                    f"budgeted Result exceeded the priced context window for "
                    f"{billed_model!r}: {usage['contextWindow']} > "
                    f"{entry_price.context_window}"
                )
            if usage["maxOutputTokens"] > entry_price.context_window:
                return (
                    f"budgeted Result exceeded the priced output bound for "
                    f"{billed_model!r}: {usage['maxOutputTokens']} > "
                    f"{entry_price.context_window}"
                )
            cost = usage.get("costUSD")
            if not _valid_billed_cost(cost):
                return "budgeted Result has invalid per-model cost"
            priced_cost, _work_cost = call_cost(
                billed_model,
                input_tokens=usage["inputTokens"],
                output_tokens=usage["outputTokens"],
                cache_read_tokens=usage["cacheReadInputTokens"],
                cache_write_tokens=usage["cacheCreationInputTokens"],
            )
            if not math.isclose(
                priced_cost,
                float(cost),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                return (
                    f"budgeted Result per-model cost disagrees with priced token "
                    f"usage for {billed_model!r}"
                )
            summed_cost = math.fsum((summed_cost, float(cost)))
        if not saw_worker_model:
            return (
                "budgeted Result never billed this worker's own model "
                f"{self.brain.model!r}; billed {sorted(usage_by_model)}"
            )
        if not math.isclose(
            summed_cost,
            float(message.total_cost_usd),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            return "budgeted Result total cost disagrees with per-model usage"
        self._budget_result_digests[message_uuid] = message_digest
        return None

    def _record_budget_identity_violation(self, message: Any, detail: str) -> None:
        self._accounting_unknown = True
        encoded = json.dumps(_jsonable(message), sort_keys=True, separators=(",", ":"))
        # The digest proves which bytes were rejected; it cannot say what was
        # wrong with them. Measured: a live worker died here and the durable
        # record held only "used a different canonical model" and a sha256, so
        # the offending value could not be recovered from the evidence at all.
        # A rejection that destroys its own subject is not evidence.
        self._record_budget_event(
            "identity_violation",
            {
                "detail": detail,
                "message_type": type(message).__name__,
                "message_sha256": _record_digest(encoded),
                "session_id": getattr(message, "session_id", None),
                "uuid": getattr(message, "uuid", None),
                "model_usage": _jsonable(getattr(message, "model_usage", None)),
                "requested_model": self.brain.model,
            },
        )
        self.brain.branch.turn(
            kind="sdk_message_rejected",
            run_id=self.run_id,
            message_type=type(message).__name__,
            detail=detail,
            message=_jsonable(message),
        )

    @staticmethod
    def _infer_query_purpose(prompt: Any) -> str:
        if isinstance(prompt, str):
            if prompt.startswith("[monitor]"):
                return "monitor"
            if prompt.startswith("[inbox "):
                return "inbox"
        return "sdk-host"

    def _install_query_tracker(self) -> None:
        """Instrument the exact client object shared with the monitor."""
        original = self._client.query

        async def tracked_query(prompt: Any, *args: Any, **kwargs: Any) -> Any:
            context = self._query_context.get()
            purpose = context[0] if context is not None else self._infer_query_purpose(prompt)
            box = context[1] if context is not None else None
            async with self._query_lock:
                if self._budget_reset_seen:
                    raise ContractMismatch(
                        "a budgeted worker cannot submit after its conversation cost counter reset"
                    )
                monitor_watermark = (
                    self.brain.branch.verdict_watermark() if purpose == "monitor" else {}
                )
                delivered_prompt = prompt
                if monitor_watermark and isinstance(prompt, str):
                    token = json.dumps(monitor_watermark, sort_keys=True)
                    delivered_prompt = (
                        f"{prompt}\n\n[host acknowledgement token]\n"
                        "After processing this monitor feedback, copy this exact "
                        f"mapping into accepted_verdicts: {token}"
                    )
                self._query_epoch += 1
                epoch = self._query_epoch
                if box is not None:
                    box["epoch"] = epoch
                self._query_writes += 1
                self._final_drained = False
                self.brain.branch.turn(
                    kind="runtime_query_intent",
                    run_id=self.run_id,
                    epoch=epoch,
                    purpose=purpose,
                    budget_receipt_sha256=(
                        None
                        if self._latest_budget_receipt is None
                        else self._latest_budget_receipt.digest
                    ),
                )
                try:
                    result = await original(delivered_prompt, *args, **kwargs)
                except BaseException as exc:
                    detail = f"{type(exc).__name__}: {exc}"
                    self.brain.branch.turn(
                        kind="runtime_query_failed",
                        run_id=self.run_id,
                        epoch=epoch,
                        purpose=purpose,
                        error=detail,
                    )
                    self._queue.put_nowait(_QueryFailed(epoch, purpose, detail))
                    raise
                else:
                    self._submitted_query_epoch = max(self._submitted_query_epoch, epoch)
                    self.brain.branch.turn(
                        kind="runtime_query_submitted",
                        run_id=self.run_id,
                        epoch=epoch,
                        purpose=purpose,
                    )
                    self._queue.put_nowait(_QuerySubmitted(epoch, purpose))
                    if monitor_watermark:
                        await self._register_verdicts(monitor_watermark, epoch)
                    return result
                finally:
                    self._query_writes -= 1

        self._client.query = tracked_query
        original_interrupt = getattr(self._client, "interrupt", None)
        if original_interrupt is not None:

            async def tracked_interrupt(*args: Any, **kwargs: Any) -> Any:
                # Authorize exactly one abort before sending the interrupt so a
                # fast result cannot race ahead of the wrapper's return.
                self._expected_abort_tokens += 1
                self.brain.branch.turn(
                    kind="runtime_interrupt_requested",
                    run_id=self.run_id,
                    authorized_aborts=self._expected_abort_tokens,
                )
                try:
                    return await original_interrupt(*args, **kwargs)
                except BaseException:
                    self._expected_abort_tokens = max(0, self._expected_abort_tokens - 1)
                    raise

            self._client.interrupt = tracked_interrupt

    async def _submit_query(self, prompt: str, purpose: str) -> int:
        box: dict[str, int] = {}
        token = self._query_context.set((purpose, box))
        try:
            await self._client.query(prompt)
        finally:
            self._query_context.reset(token)
        return box["epoch"]

    @staticmethod
    def _inbox_prompt(message: dict[str, Any]) -> str:
        message_id = str(message.get("id") or "unknown")
        sender = str(message.get("sender") or "unknown sender")
        body = json.dumps(message.get("body"), sort_keys=True)
        return (
            f"[inbox {message_id[:10]}] From {sender}: {body}\n"
            f"Treat {message_id} as the idempotency key. If you have already handled "
            "it, do not repeat its side effects; report the existing result instead. "
            "Only after processing it, copy the full id into accepted_inbox_ids in "
            f"your next structured result: {message_id}"
        )

    def _worker_prompt(self, waking: Waking) -> str:
        prompt = self.brain.opening_prompt(waking)
        parts = [prompt]
        if self.assignment is not None:
            parts.append(
                "The following is the exact accepted assignment record. Its artifact paths, "
                "input state/blob references, generation, and attempt are authoritative:\n"
                f"{self._accepted_assignment_text}\n"
                "Finish with the required structured status. completed is a claim, not a "
                "shortcut: cite concrete evidence, and the host will independently verify "
                "every required output against the final immutable state."
            )
        inbox_ids = [str(message.get("id")) for message in waking.inbox if message.get("id")]
        verdicts = self.brain.branch.verdict_watermark() if waking.unacked else {}
        parts.append(
            "Every structured result must include accepted_inbox_ids and "
            "accepted_verdicts. Echo only items you actually processed. Current exact "
            f"inbox ids: {json.dumps(inbox_ids, sort_keys=True)}. Current exact verdict "
            f"watermark: {json.dumps(verdicts, sort_keys=True)}."
        )
        return "\n\n".join(parts)

    # --------------------------------------------------------------- acknowledgements

    @staticmethod
    def _typed_envelope(item: InboxMessage) -> dict[str, Any]:
        """Present a typed item through the existing model prompt shape."""
        return {
            "id": item.inbox_id,
            "sender": item.message.sender,
            "at": item.received_at,
            "body": item.message.to_dict(),
        }

    def _typed_acceptance_boundary(
        self,
        item: InboxMessage,
        disposition: str,
        current_generation: int,
    ) -> AcceptanceReceipt:
        """Persist one typed disposition before Communicator moves its cursor."""
        kind = "inbox_accepted" if disposition == "accepted" else "inbox_retired_stale"
        evidence = "structured_result_echo" if disposition == "accepted" else "generation_fence"
        record = {
            "kind": kind,
            "message_id": item.inbox_id,
            "semantic_message_id": item.message.message_id,
            "message_digest": _record_digest(item.message.to_json()),
            "sender": item.message.sender,
            "recipient": item.message.recipient,
            "message_generation": item.message.generation,
            "current_generation": current_generation,
            "disposition": disposition,
            "acceptance_evidence": evidence,
        }
        matching = [
            event
            for event in self._recorded_events()
            if event.get("kind") in {"inbox_accepted", "inbox_retired_stale"}
            and event.get("message_id") == item.inbox_id
        ]
        if matching and any(event != record for event in matching):
            raise ContractMismatch("durable typed inbox disposition changed identity")
        if not matching:
            self.brain.branch.turn(**record)
        durable_ref = (
            "worker-turn."
            + hashlib.sha256(
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )
        return AcceptanceReceipt.for_message(
            item,
            disposition=disposition,
            current_generation=current_generation,
            durable_ref=durable_ref,
        )

    def _typed_pending_envelopes(self) -> list[dict[str, Any]]:
        """Fence generations, retiring only an exact stale oldest prefix."""
        if self.communicator is None:
            return self.brain.store.inbox(self.brain.contract.identity)
        assert self.assignment is not None
        envelopes: list[dict[str, Any]] = []
        for item in self.communicator.pending(self.brain.contract.identity):
            status = self.communicator.generation_status(item.message, self.assignment.generation)
            if status is GenerationStatus.STALE:
                # A stale item behind an unaccepted current item is not yet
                # the oldest cursor entry and therefore cannot be retired.
                if envelopes:
                    break
                self.communicator.retire_stale(
                    item,
                    current_generation=self.assignment.generation,
                    boundary=self._typed_acceptance_boundary,
                )
                self._typed_inbox_items.pop(item.inbox_id, None)
                continue
            if status is GenerationStatus.FUTURE:
                # The assignment's generation is immutable.  Leave this item
                # for the future worker rather than exposing it to this model.
                break
            self._typed_inbox_items[item.inbox_id] = item
            envelopes.append(self._typed_envelope(item))
        return envelopes

    def _recorded_events(self) -> list[dict[str, Any]]:
        return self.brain._recorded_turns()

    def _reconcile_delivery_markers(self) -> None:
        """Finish cursor writes whose durable acceptance event survived a kill."""
        events = self._recorded_events()
        accepted_ids = {
            str(event.get("message_id"))
            for event in events
            if event.get("kind") == "inbox_accepted" and event.get("message_id")
        }
        self._accepted_inbox_evidence.update(accepted_ids)
        if self.communicator is not None:
            assert self.assignment is not None
            retired_ids = {
                str(event.get("message_id"))
                for event in events
                if event.get("kind") == "inbox_retired_stale" and event.get("message_id")
            }
            for item in self.communicator.pending(self.brain.contract.identity):
                status = self.communicator.generation_status(
                    item.message, self.assignment.generation
                )
                if status is GenerationStatus.CURRENT and item.inbox_id in accepted_ids:
                    self.communicator.accept(
                        item,
                        current_generation=self.assignment.generation,
                        boundary=self._typed_acceptance_boundary,
                    )
                    continue
                if status is GenerationStatus.STALE and item.inbox_id in retired_ids:
                    self.communicator.retire_stale(
                        item,
                        current_generation=self.assignment.generation,
                        boundary=self._typed_acceptance_boundary,
                    )
                    continue
                break
        else:
            # The store cursor is cumulative.  A durable acceptance for a later
            # message is useful evidence, but it cannot advance the cursor across
            # an older message for which no acceptance survived.  Walk the exact
            # unseen log oldest-first and stop at the first gap.
            for message in self.brain.store.inbox(self.brain.contract.identity):
                message_id = str(message.get("id") or "")
                if message_id not in accepted_ids:
                    break
                self.brain.store.mark_inbox_seen(self.brain.contract.identity, message_id)
                self.brain.branch.turn(kind="inbox_cursor_reconciled", message_id=message_id)

        for event in events:
            if event.get("kind") != "monitor_feedback_accepted":
                continue
            raw = event.get("through")
            if not isinstance(raw, dict):
                continue
            visible = self.brain.branch.verdict_watermark()
            through = {
                str(state_id): count
                for state_id, count in raw.items()
                if isinstance(count, int)
                and not isinstance(count, bool)
                and state_id in visible
                and count <= visible[state_id]
            }
            if through:
                self.brain.branch.acknowledge(through=through)

    async def _register_inbox(self, messages: list[dict[str, Any]], epoch: int) -> None:
        async with self._delivery_lock:
            known = {str(item.message.get("id") or "") for item in self._pending_inbox}
            for message in messages:
                message_id = str(message.get("id") or "")
                if not message_id or message_id in known:
                    continue
                self._pending_inbox.append(_PendingInbox(message, epoch))
                self._delivered_inbox_epochs[message_id] = min(
                    epoch, self._delivered_inbox_epochs.get(message_id, epoch)
                )
                known.add(message_id)
                self.brain.branch.turn(
                    kind="inbox_submitted",
                    message_id=message_id,
                    message=message,
                    epoch=epoch,
                )
            await self._accept_deliveries_locked()

    async def _register_verdicts(self, through: dict[str, int], epoch: int) -> None:
        if not through:
            return
        async with self._delivery_lock:
            delivered = _PendingVerdicts(dict(through), epoch)
            if delivered not in self._delivered_verdict_history:
                self._delivered_verdict_history.append(delivered)
            if not any(pending.through == through for pending in self._pending_verdicts):
                self._pending_verdicts.append(delivered)
                self.brain.branch.turn(
                    kind="monitor_feedback_submitted", through=through, epoch=epoch
                )
            await self._accept_deliveries_locked()

    async def _accept_deliveries(self, claim: Mapping[str, Any], *, submitted_epoch: int) -> None:
        async with self._delivery_lock:
            for message_id in claim["accepted_inbox_ids"]:
                delivered_epoch = self._delivered_inbox_epochs.get(message_id)
                if delivered_epoch is None or delivered_epoch > submitted_epoch:
                    self._protocol_uncertainty.append(
                        f"worker acknowledged undelivered inbox id {message_id}"
                    )
                    continue
                self._accepted_inbox_evidence.add(message_id)
            for state_id, count in claim["accepted_verdicts"].items():
                delivered_count = max(
                    (
                        pending.through.get(state_id, 0)
                        for pending in self._delivered_verdict_history
                        if pending.epoch <= submitted_epoch
                    ),
                    default=0,
                )
                if count > delivered_count:
                    self._protocol_uncertainty.append(
                        f"worker acknowledged undelivered monitor watermark {state_id}:{count}"
                    )
                    continue
                self._accepted_verdict_evidence[state_id] = max(
                    count, self._accepted_verdict_evidence.get(state_id, 0)
                )
            await self._accept_deliveries_locked()

    async def _accept_deliveries_locked(self) -> None:
        remaining_inbox: list[_PendingInbox] = []
        gap_seen = False
        for pending in self._pending_inbox:
            message_id = str(pending.message.get("id") or "")
            # A later echo may arrive before an earlier one, but
            # ``mark_inbox_seen`` acknowledges the entire prefix ending at
            # that message.  Retain all entries from the first unaccepted gap
            # onward; once the gap is accepted, already-recorded later
            # evidence lets the whole contiguous prefix settle in order.
            if gap_seen or message_id not in self._accepted_inbox_evidence:
                gap_seen = True
                remaining_inbox.append(pending)
                continue
            if self.communicator is None:
                self.brain.branch.turn(
                    kind="inbox_accepted",
                    message_id=message_id,
                    message=pending.message,
                    epoch=pending.epoch,
                    acceptance_evidence="structured_result_echo",
                )
                self.brain.store.mark_inbox_seen(self.brain.contract.identity, message_id)
            else:
                assert self.assignment is not None
                item = self._typed_inbox_items.get(message_id)
                if item is None:
                    raise ContractMismatch(
                        "typed pending inbox entry lost its exact protocol value"
                    )
                self.communicator.accept(
                    item,
                    current_generation=self.assignment.generation,
                    boundary=self._typed_acceptance_boundary,
                )
                self._typed_inbox_items.pop(message_id, None)
        self._pending_inbox = remaining_inbox

        remaining_verdicts: list[_PendingVerdicts] = []
        for pending in self._pending_verdicts:
            covered = all(
                self._accepted_verdict_evidence.get(state_id, 0) >= count
                for state_id, count in pending.through.items()
            )
            if not covered:
                remaining_verdicts.append(pending)
                continue
            self.brain.branch.turn(
                kind="monitor_feedback_accepted",
                through=pending.through,
                epoch=pending.epoch,
                acceptance_evidence="structured_result_echo",
            )
            self.brain.branch.acknowledge(through=pending.through)
        self._pending_verdicts = remaining_verdicts

    # --------------------------------------------------------------- concurrent loops

    async def _sleep_until_poll(self) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._background_stop.wait(), self.poll_interval)

    async def _call_ready(self) -> None:
        if self.ready is None:
            return
        result = self.ready(self.brain.contract.identity)
        if inspect.isawaitable(result):
            await result

    async def _note_monitor_action(self, judgement: Any, rung: str | None) -> None:
        if judgement is not None and rung is not None:
            self._monitor_actions.append((judgement, rung))

    async def _monitor_loop(self) -> None:
        first_cycle = True
        self._monitor_entered.set()
        try:
            while not self._background_stop.is_set():
                async with self._monitor_lock:
                    judgement, rung = await self.monitor.cycle(self._client)
                    await self._note_monitor_action(judgement, rung)
                if first_cycle:
                    if self._background_stop.is_set():
                        return
                    await self._call_ready()
                    first_cycle = False
                if judgement is None:
                    await self._sleep_until_poll()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._queue.put(_BackgroundFailure("monitor", exc))

    async def _inbox_loop(self) -> None:
        try:
            while not self._background_stop.is_set():
                pending_ids = {str(item.message.get("id") or "") for item in self._pending_inbox}
                messages = self._typed_pending_envelopes()
                for message in messages:
                    message_id = str(message.get("id") or "")
                    if self._background_stop.is_set():
                        return
                    if not message_id or message_id in pending_ids:
                        continue
                    self.brain.branch.turn(
                        kind="inbox_delivery_intent",
                        message_id=message_id,
                        message=message,
                    )
                    epoch = await self._submit_query(self._inbox_prompt(message), "inbox")
                    await self._register_inbox([message], epoch)
                    pending_ids.add(message_id)
                await self._sleep_until_poll()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._queue.put(_BackgroundFailure("inbox", exc))

    async def _reader_loop(self) -> None:
        try:
            async for message in self._client.receive_messages():
                submitted_epoch = self._submitted_query_epoch
                budgeted = self.brain.contract.budget_usd is not None
                violation = self._budgeted_message_violation(message) if budgeted else None
                if violation is not None:
                    self._budget_reset_seen = True
                    self._background_stop.set()
                    abort_error: Exception | None = None
                    try:
                        self._abort_budgeted_cli_now()
                    except Exception as exc:
                        abort_error = exc
                    try:
                        self._record_budget_identity_violation(message, violation)
                    except Exception as exc:
                        self._queue.put_nowait(
                            _BackgroundFailure("budget identity persistence", exc)
                        )
                    self._queue.put_nowait(
                        _BackgroundFailure(
                            "budget identity",
                            ContractMismatch(violation),
                        )
                    )
                    if abort_error is not None:
                        self._queue.put_nowait(
                            _BackgroundFailure("budget identity abort", abort_error)
                        )
                    continue
                if isinstance(message, ConversationResetMessage) and budgeted:
                    # The CLI zeroes the counter used by --max-budget-usd at
                    # this boundary. Stop every host producer before the reset
                    # can silently reuse the same contract allowance.
                    self._budget_reset_seen = True
                    self._background_stop.set()
                    abort_error: Exception | None = None
                    try:
                        self._abort_budgeted_cli_now()
                    except Exception as exc:
                        abort_error = exc
                    # Persist the reset only after the synchronous kill signal,
                    # but before the next await or stream read.  A crash cannot
                    # then resume the outgoing scope as though no reset happened.
                recorded = False
                if budgeted:
                    # Keep session and cumulative-cost durability in exact
                    # stream order.  In particular, a fast reset cannot race
                    # ahead of an earlier Result still waiting in the queue.
                    try:
                        self._record_message(message)
                        recorded = True
                    except Exception as exc:
                        self._queue.put_nowait(
                            _BackgroundFailure("budgeted stream persistence", exc)
                        )
                if isinstance(message, ConversationResetMessage) and budgeted:
                    self._queue.put_nowait(_Received(message, submitted_epoch, recorded=recorded))
                    if abort_error is not None:
                        self._queue.put_nowait(
                            _BackgroundFailure("budget-reset abort", abort_error)
                        )
                    continue
                # Snapshot causality at receipt. Processing can lag behind a
                # concurrent inbox/monitor query; using the later current
                # epoch would let an older result acknowledge that newer
                # prompt without any response to it.
                await self._queue.put(_Received(message, submitted_epoch, recorded=recorded))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._queue.put(_StreamClosed(exc))
        else:
            await self._queue.put(_StreamClosed())

    async def _deliver_opening(self, waking: Waking) -> None:
        verdict_watermark = self.brain.branch.verdict_watermark() if waking.unacked else {}
        epoch = await self._submit_query(self._worker_prompt(waking), "opening")
        await self._register_inbox(list(waking.inbox), epoch)
        await self._register_verdicts(verdict_watermark, epoch)

    async def _drain_monitor(self) -> list[tuple[Any, str]]:
        drained = await self.monitor.drain(self._client, final=True)
        for judgement, rung in drained:
            await self._note_monitor_action(judgement, rung)
        self._final_drained = True
        return drained

    async def _stop_background_loops(self) -> None:
        if self._backgrounds_stopped:
            return
        self._background_stop.set()
        # Do not cancel the monitor. MonitorBrain runs its judge through
        # asyncio.to_thread(); cancellation would release the async lock while
        # that thread continued mutating the same durable state.
        tasks = [task for task in (self._monitor_task, self._inbox_task) if task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._backgrounds_stopped = True

    # --------------------------------------------------------------- SDK semantics

    @staticmethod
    def _result_relevant(message: ResultMessage) -> bool:
        origin = message.origin
        kind = origin.get("kind") if isinstance(origin, Mapping) else None
        return kind in _RELEVANT_ORIGINS

    @staticmethod
    def _result_from_host(message: ResultMessage) -> bool:
        origin = message.origin
        kind = origin.get("kind") if isinstance(origin, Mapping) else None
        return kind in {None, "human"}

    def _consume_expected_abort(self) -> bool:
        if self._expected_abort_tokens <= 0:
            return False
        self._expected_abort_tokens -= 1
        return True

    @staticmethod
    def _effective_reason(message: ResultMessage) -> str | None:
        return message.terminal_reason or message.stop_reason

    @classmethod
    def _result_successful(cls, message: ResultMessage) -> bool:
        return not message.is_error and cls._effective_reason(message) in _SUCCESS_REASONS

    @classmethod
    def _result_accepts_queries(cls, message: ResultMessage) -> bool:
        return cls._result_from_host(message) and cls._result_successful(message)

    def _result_error(self, message: ResultMessage, *, expected_abort: bool = False) -> str | None:
        reason = self._effective_reason(message)
        if reason in _ABORT_REASONS:
            if expected_abort:
                return None
            return f"unexpected SDK abort: {reason}"
        if message.is_error:
            details = "; ".join(message.errors or ()) or message.result or message.subtype
            return f"SDK result error: {details}"
        if reason is None:
            # Slash-command/bypass results can legitimately omit a terminal
            # reason. They are neither proof of host-query completion nor an
            # SDK failure; wait for the real host turn result.
            return None
        if reason not in _SUCCESS_REASONS:
            return f"SDK terminated with {reason}"
        return None

    def _parse_completion_claim(self, message: ResultMessage, submitted_epoch: int) -> None:
        if not self._result_successful(message):
            return
        raw = message.structured_output
        if raw is None:
            return
        required = {
            "status",
            "summary",
            "evidence",
            "accepted_inbox_ids",
            "accepted_verdicts",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            self._protocol_uncertainty.append("worker returned malformed structured output")
            return
        status, summary, evidence = raw.get("status"), raw.get("summary"), raw.get("evidence")
        if status not in {"completed", "blocked", "continue"}:
            self._protocol_uncertainty.append("worker returned an unknown structured status")
            return
        if (
            not isinstance(summary, str)
            or not isinstance(evidence, list)
            or not all(isinstance(item, str) for item in evidence)
        ):
            self._protocol_uncertainty.append("worker structured output has invalid fields")
            return
        accepted_inbox = raw.get("accepted_inbox_ids")
        accepted_verdicts = raw.get("accepted_verdicts")
        if (
            not isinstance(accepted_inbox, list)
            or not all(isinstance(item, str) and item for item in accepted_inbox)
            or len(accepted_inbox) != len(set(accepted_inbox))
            or not isinstance(accepted_verdicts, dict)
            or not all(
                isinstance(state_id, str)
                and state_id
                and isinstance(count, int)
                and not isinstance(count, bool)
                and count >= 1
                for state_id, count in accepted_verdicts.items()
            )
        ):
            self._protocol_uncertainty.append(
                "worker structured acknowledgements have invalid fields"
            )
            return
        if status == "completed" and not any(item.strip() for item in evidence):
            self._protocol_uncertainty.append("worker claimed completion without evidence")
            return
        self._completion_claim = {
            "status": status,
            "summary": summary,
            "evidence": list(evidence),
            "accepted_inbox_ids": list(accepted_inbox),
            "accepted_verdicts": dict(accepted_verdicts),
        }
        self._completion_claim_epoch = submitted_epoch

    def _invalidate_terminal_claim(self) -> None:
        self._candidate = None
        self._completion_claim = None
        self._completion_claim_epoch = 0

    def _track_activity(self, message: Any) -> None:
        if isinstance(message, ConversationResetMessage):
            self._invalidate_terminal_claim()
            return
        if isinstance(
            message,
            (
                UserMessage,
                AssistantMessage,
                StreamEvent,
                HookEventMessage,
                TaskStartedMessage,
                TaskProgressMessage,
                TaskNotificationMessage,
                TaskUpdatedMessage,
            ),
        ):
            # An old Result is only an idle point. Any later turn/activity
            # frame requires a new terminal Result before disconnecting.
            self._invalidate_terminal_claim()

    def _track_task_lifecycle(self, message: Any) -> None:
        if isinstance(message, TaskStartedMessage):
            if message.task_type in _TRACKED_TASK_TYPES:
                self._delegated_task_seen = True
                self._active_tasks.add(message.task_id)
                self._candidate = None
            return
        if isinstance(message, TaskNotificationMessage):
            if message.status in TERMINAL_TASK_STATUSES:
                self._active_tasks.discard(message.task_id)
            return
        if isinstance(message, TaskUpdatedMessage):
            status = message.status or message.patch.get("status")
            if status in TERMINAL_TASK_STATUSES:
                self._active_tasks.discard(message.task_id)
            elif status in {"pending", "running", "paused"}:
                # A resumed stream can reveal a task through an update even if
                # its original task_started event predates this connection.
                self._active_tasks.add(message.task_id)
                self._delegated_task_seen = True
                self._candidate = None

    async def _handle_sdk_message(
        self, message: Any, *, submitted_epoch: int, recorded: bool = False
    ) -> None:
        if not recorded:
            self._record_message(message)
        if (
            isinstance(message, ConversationResetMessage)
            and self.brain.contract.budget_usd is not None
        ):
            detail = "budgeted SDK conversation reset zeroed the provider-side cost guard"
            if detail not in self._protocol_uncertainty:
                self._protocol_uncertainty.append(detail)
            raise ContractMismatch(detail)
        if isinstance(message, MirrorErrorMessage):
            detail = message.error or json.dumps(message.data, sort_keys=True)
            raise MirrorDurabilityError(detail)

        self._track_activity(message)
        self._track_task_lifecycle(message)
        if not isinstance(message, ResultMessage):
            return

        # Result totals are session-wide even for peer/channel turns that do
        # not answer a host prompt. Their accounting was persisted in exact
        # stream order by ``_record_message`` above; causality remains here.
        if not self._result_relevant(message):
            self._invalidate_terminal_claim()
            return

        # A completion claim belongs to this exact relevant Result, not merely
        # to a query epoch. The SDK can emit multiple Results for one query.
        self._completion_claim = None
        self._completion_claim_epoch = 0
        self._results.append(message)
        self._latest_relevant = message
        # There is no result/request id. Epoch coverage is deliberately a
        # conservative batch boundary: prompts initiated before this result
        # may have been coalesced, while later prompts still need a new result.
        if self._result_from_host(message):
            self._covered_query_epoch = max(self._covered_query_epoch, submitted_epoch)
        effective_reason = self._effective_reason(message)
        expected_abort = bool(effective_reason in _ABORT_REASONS and self._consume_expected_abort())
        failure = self._result_error(message, expected_abort=expected_abort)
        if failure and failure not in self._sticky_result_errors:
            self._sticky_result_errors.append(failure)
        self._parse_completion_claim(message, submitted_epoch)
        if self._result_accepts_queries(message) and self._completion_claim is not None:
            await self._accept_deliveries(self._completion_claim, submitted_epoch=submitted_epoch)

        if (effective_reason in _ABORT_REASONS and expected_abort) or effective_reason is None:
            self._candidate = None
            return
        if self._active_tasks:
            self._candidate = None
            return
        self._candidate = message
        self._candidate_epoch = self._covered_query_epoch

    def _candidate_ready(self) -> bool:
        return bool(
            self._candidate is not None
            and not self._active_tasks
            and not self._pending_inbox
            and not self._pending_verdicts
            and self._query_writes == 0
            and self._candidate_epoch >= self._submitted_query_epoch
        )

    async def _process_drive_item(self, item: Any) -> None:
        if isinstance(item, _BackgroundFailure):
            raise RuntimeError(f"{item.source} loop failed: {item.error}") from item.error
        if isinstance(item, _StreamClosed):
            if item.error is not None:
                raise RuntimeError(f"SDK stream failed: {item.error}") from item.error
            raise RuntimeError("SDK stream closed before the runtime disconnected it")
        if isinstance(item, _QuerySubmitted):
            self._final_drained = False
            if self._candidate is not None and item.epoch > self._candidate_epoch:
                self._candidate = None
            if self._completion_claim is not None and item.epoch > self._completion_claim_epoch:
                self._completion_claim = None
                self._completion_claim_epoch = 0
            return
        if isinstance(item, _QueryFailed):
            self._final_drained = False
            return
        if isinstance(item, _Received):
            await self._handle_sdk_message(
                item.message,
                submitted_epoch=item.submitted_epoch,
                recorded=item.recorded,
            )
            return
        raise TypeError(f"unexpected worker runtime queue item {type(item).__name__}")

    async def _drive(self) -> ResultMessage:
        """Run until a relevant result, task quiescence, and final monitor drain."""
        self._reader_task = asyncio.create_task(self._reader_loop(), name="worker-sdk-reader")
        self._monitor_task = asyncio.create_task(self._monitor_loop(), name="worker-monitor")
        await self._monitor_entered.wait()

        self._reconcile_delivery_markers()
        waking = self.brain.wake()
        if self.communicator is not None:
            inbox = tuple(self._typed_pending_envelopes())
            waking = replace(
                waking,
                fresh=not (
                    waking.intent
                    or waking.in_flight
                    or waking.dirty_paths
                    or waking.unacked
                    or inbox
                    or waking.open_conflicts
                ),
                inbox=inbox,
            )
        await self._deliver_opening(waking)
        self._inbox_task = asyncio.create_task(self._inbox_loop(), name="worker-inbox")

        stable_since = time.monotonic()
        while True:
            timeout: float | None = None
            if self._candidate_ready():
                timeout = max(
                    0.0,
                    self.terminal_quiet_period - (time.monotonic() - stable_since),
                )
            try:
                if timeout is None:
                    item = await self._queue.get()
                else:
                    item = await asyncio.wait_for(self._queue.get(), timeout)
            except TimeoutError:
                if not self._backgrounds_stopped:
                    await self._stop_background_loops()
                    stable_since = time.monotonic()
                    continue
                if not self._final_drained:
                    async with self._monitor_lock:
                        await self._drain_monitor()
                    stable_since = time.monotonic()
                    continue
                if self._candidate is None:  # pragma: no cover - guarded by timeout
                    continue
                return self._candidate

            await self._process_drive_item(item)
            stable_since = time.monotonic()

    # --------------------------------------------------------------- shutdown/reporting

    async def _disconnect_and_collect_tail(self) -> tuple[list[str], bool]:
        errors: list[str] = []
        durability_ok = True
        delayed_cancellation: asyncio.CancelledError | None = None
        if self._connect_attempted:
            disconnect_task = asyncio.create_task(
                self._client.disconnect(), name="worker-sdk-disconnect"
            )
            while not disconnect_task.done():
                try:
                    await asyncio.shield(disconnect_task)
                except asyncio.CancelledError as exc:
                    # Raw asyncio cancellation can penetrate the SDK's anyio
                    # shields if it reaches disconnect itself. Delay it until
                    # the SDK's bounded terminate/kill/reap sequence finishes.
                    delayed_cancellation = exc
            try:
                disconnect_task.result()
            except asyncio.CancelledError as exc:
                raise ShutdownUnconfirmed(
                    "SDK disconnect cancelled itself before child reaping was confirmed"
                ) from exc
            except Exception as exc:
                errors.append(f"disconnect failed: {type(exc).__name__}: {exc}")
            else:
                self._shutdown_confirmed = True
        else:
            self._shutdown_confirmed = True

        if self._reader_task is not None and not self._reader_task.done():
            done, _ = await asyncio.wait({self._reader_task}, timeout=self.shutdown_timeout)
            if not done:
                errors.append("SDK reader did not close after disconnect")
                self._reader_task.cancel()
                await asyncio.gather(self._reader_task, return_exceptions=True)

        while not self._queue.empty():
            item = self._queue.get_nowait()
            if isinstance(item, _StreamClosed):
                if item.error is not None:
                    errors.append(f"SDK stream failed: {item.error}")
                continue
            if isinstance(item, _BackgroundFailure):
                errors.append(f"{item.source} loop failed: {item.error}")
                continue
            if isinstance(item, (_QuerySubmitted, _QueryFailed)):
                continue
            if not isinstance(item, _Received):
                errors.append(f"unexpected shutdown queue item {type(item).__name__}")
                continue
            message = item.message
            try:
                await self._handle_sdk_message(
                    message,
                    submitted_epoch=item.submitted_epoch,
                    recorded=item.recorded,
                )
            except MirrorDurabilityError as exc:
                durability_ok = False
                errors.append(f"{type(exc).__name__}: {exc}")
                continue
            except Exception as exc:
                errors.append(f"shutdown message handling failed: {type(exc).__name__}: {exc}")
                continue
            if isinstance(message, (ResultMessage, StreamEvent)):
                self._tail_uncertainty.append(
                    f"{type(message).__name__} arrived only during SDK shutdown"
                )
        if delayed_cancellation is not None:
            raise delayed_cancellation
        return errors, durability_ok

    @staticmethod
    def _permission_denials(messages: list[ResultMessage]) -> tuple[tuple[str, str], ...]:
        out: list[tuple[str, str]] = []
        for message in messages:
            for denial in message.permission_denials or ():
                if isinstance(denial, Mapping):
                    tool = str(denial.get("tool_name") or denial.get("name") or "unknown")
                    reason = str(denial.get("reason") or denial.get("message") or denial)
                else:
                    tool, reason = "unknown", str(denial)
                pair = (tool, reason)
                if pair not in out:
                    out.append(pair)
        return tuple(out)

    def _control_integrity_failures(self) -> list[str]:
        failures: list[str] = []

        contract_text, contract_error = _read_regular_text_no_follow(
            self.brain.branch.path(CONTRACT_PATH)
        )
        if contract_error is not None or contract_text != self._accepted_contract_text:
            failures.append(f"control record {CONTRACT_PATH} was changed or deleted")

        assignment_path = self.brain.branch.path(ASSIGNMENT_PATH)
        assignment_text, assignment_error = _read_regular_text_no_follow(assignment_path)
        if self._accepted_assignment_text is None:
            if os.path.lexists(assignment_path):
                failures.append(f"reserved control record {ASSIGNMENT_PATH} was created")
        elif assignment_error is not None or assignment_text != self._accepted_assignment_text:
            failures.append(f"control record {ASSIGNMENT_PATH} was changed or deleted")

        if os.path.lexists(self.brain.branch.path(WORKER_REPORT_PATH)):
            failures.append(f"reserved control record {WORKER_REPORT_PATH} was created")
        return failures

    def _state_control_integrity_failures(self, state: Any) -> list[str]:
        """Verify the immutable work checkpoint preserved exact control entries."""
        failures: list[str] = []
        for path, expected in self._accepted_control_entries.items():
            entry = self.brain.branch.backend.entry_at(state.id, path)
            actual = (entry.mode, entry.sha) if entry is not None else None
            if actual != expected:
                failures.append(f"checkpointed control record {path} changed Git mode or bytes")
        if (
            self._accepted_assignment_text is None
            and self.brain.branch.backend.entry_at(state.id, ASSIGNMENT_PATH) is not None
        ):
            failures.append(f"checkpointed reserved control record {ASSIGNMENT_PATH} was created")
        if self.brain.branch.backend.entry_at(state.id, WORKER_REPORT_PATH) is not None:
            failures.append(
                f"checkpointed reserved control record {WORKER_REPORT_PATH} was created"
            )
        return failures

    def _advertise_outputs(self) -> None:
        """Publish this assignment's products, so the world can see them.

        ``publish`` records a pending manifest entry that the *next* checkpoint
        folds in, so this must run before the terminal work checkpoint -- that
        is the state delivery projects from (``final_state_id``).

        Why it matters beyond this branch: ``deliver_product`` builds the
        projection's manifest by filtering the *source* manifest, and
        ``Branch.merge`` unions manifests into the target. The whole chain
        worker -> projection -> integration already exists; it carried nothing
        because no worker ever published. Measured consequence: a central run
        delivered ``adder.py`` to integration, the planner was shown
        ``integration: ... artifacts=0`` with no file list anywhere in its
        observation, correctly concluded "the artifact has not been
        integrated", and reissued the same assignment three times.

        Three guards, each for a failure this would otherwise cause:

        * ``disposition == "absent"`` specs name something that must *not*
          exist, so there is nothing to advertise.
        * A path missing from the worktree is left unpublished. ``publish``
          raises ``PublishError`` at checkpoint for an absent path, which would
          turn a missing required output -- an ordinary, reportable failure
          that ``_artifact_outputs`` already describes -- into a crash that
          wedges the branch against ever committing again.
        * Control plumbing is skipped. ``_manifest`` filters it out of the
          projection anyway; advertising it would still pollute this worker's
          own catalog with its assignment and report files.
        """
        if self.assignment is None:
            return
        for spec in self.assignment.outputs:
            if spec.disposition != "present" or is_control_path(spec.path):
                continue
            if not self.brain.branch.path(spec.path).is_file():
                continue
            self.brain.branch.publish(
                spec.artifact_id,
                spec.path,
                description=spec.description,
            )

    def _artifact_outputs(self, work_state: Any) -> tuple[tuple[ArtifactRef, ...], list[str]]:
        if self.assignment is None:
            return (), []
        outputs: list[ArtifactRef] = []
        failures: list[str] = []
        for spec in self.assignment.outputs:
            entry = self.brain.store.backend.entry_at(work_state.id, spec.path)
            base_entry = self.brain.store.backend.entry_at(self.assignment.base_state_id, spec.path)
            if any(
                candidate is not None and candidate.mode == "040000"
                for candidate in (entry, base_entry)
            ):
                failures.append(
                    f"{spec.artifact_id}: {spec.path} names a directory; "
                    "select product files explicitly"
                )
                continue
            blob = entry.sha if entry is not None else None
            if spec.disposition == "absent":
                if blob is not None:
                    failures.append(f"{spec.artifact_id}: expected {spec.path} to be absent")
                continue
            if blob is None:
                if spec.required:
                    failures.append(f"{spec.artifact_id}: missing {spec.path}")
                continue
            outputs.append(
                ArtifactRef(
                    artifact_id=spec.artifact_id,
                    branch=self.brain.contract.identity,
                    state_id=work_state.id,
                    path=spec.path,
                    blob_id=blob,
                    kind=spec.kind,
                    metadata=spec.metadata,
                )
            )
        return tuple(outputs), failures

    async def run(self) -> SubBrainResult:
        """Execute, then checkpoint immutable work and its terminal report."""
        try:
            self.assignment = self._validate_durable_input()
            return await self._run_validated()
        finally:
            # A worker-process supervisor may kill an unresponsive process,
            # which releases this kernel lease. In-process handoff is forbidden
            # until SDK child cleanup is known complete.
            if not self._connect_attempted or self._shutdown_confirmed:
                self.brain.close()

    async def _run_validated(self) -> SubBrainResult:
        budgeted = self.brain.contract.budget_usd is not None
        if budgeted:
            if self._client_factory_injected:
                raise ContractMismatch(
                    "budgeted workers require the built-in audited SDK client; "
                    "an injected client factory cannot establish the production boundary"
                )
            # Check before touching an existing terminal report.  An exact run
            # with surviving spend evidence is immutable, not a report to
            # delete on the way to a replay that will then be refused.
            self._assert_budgeted_connection_fresh()
        old_report = self.brain.branch.path(WORKER_REPORT_PATH)
        if os.path.lexists(old_report):
            if stat.S_ISDIR(old_report.lstat().st_mode):
                raise ContractMismatch(
                    f"reserved control record {WORKER_REPORT_PATH} is a directory"
                )
            old_report.unlink()

        sdk_result: ResultMessage | None = None
        runtime_errors: list[str] = []
        durability_ok = True
        normal_terminal = False
        # A budgeted exact run is spend-once.  Its independent connection
        # intent is the crash fence, so it never resumes a prior provider
        # process under the same allowance. Unbudgeted interactive workers
        # retain the ordinary SDK session-resume path.
        resume = None if budgeted else self._resume_session_id()
        # The SDK explicitly has no proof-safe boundary after background
        # agent/workflow tasks complete: a continuation may appear after the
        # task ledger is empty and after an arbitrary quiet gap. Central mode
        # already delegates through isolated worker processes, so nested SDK
        # delegation is disabled rather than pretending a timer proves done.
        option_overrides: dict[str, Any] = {
            "output_format": {
                "type": "json_schema",
                "schema": WORKER_RESULT_SCHEMA,
            },
        }
        options = self.brain.options(
            resume=resume,
            budget_already_spent_usd=self._accounting_cost_before_reset,
            **option_overrides,
        )
        if budgeted:
            # A budgeted run never dispatches through the injectable factory.
            # Verify the packaged executable before the pinned client's inert
            # constructor or the spend-once connection intent.  Setting the
            # exact path here also prevents the SDK from falling back to a
            # system-wide executable.  The live transport is checked again
            # after connect and before the first provider query.
            cli_path, cli_evidence = self._preflight_budgeted_cli()
            options.cli_path = str(cli_path)
            self._budget_cli_path = cli_path
            self._budget_cli_evidence = cli_evidence
            self._client = self._make_client(options)
            self._assert_budgeted_client_identity()
            self._admit_budgeted_connection(options)
        else:
            self._client = self.client_factory(options)
        self._install_query_tracker()

        try:
            self._connect_attempted = True
            await self._client.connect()
            self._connected = True
            if budgeted:
                bound_evidence = self._arm_budgeted_process_boundary()
                if (
                    not _valid_cli_evidence(bound_evidence)
                    or bound_evidence != self._budget_cli_evidence
                ):
                    raise ContractMismatch("budgeted process has no exact CLI evidence")
                self._record_budget_event("process_bound", bound_evidence)
                self._budget_process_bound = True
            sdk_result = await self._drive()
            normal_terminal = True
        except MirrorDurabilityError as exc:
            durability_ok = False
            runtime_errors.append(f"{type(exc).__name__}: {exc}")
            self.brain.branch.turn(kind="durability_alarm", detail=runtime_errors[-1])
        except Exception as exc:
            runtime_errors.append(f"{type(exc).__name__}: {exc}")
            self.brain.branch.turn(kind="runtime_error", detail=runtime_errors[-1])
        finally:
            await self._stop_background_loops()
            tail_errors, tail_durable = await self._disconnect_and_collect_tail()
            runtime_errors.extend(tail_errors)
            durability_ok = durability_ok and tail_durable

        if not await self._confirm_budget_abort_reaped():
            self._shutdown_confirmed = False
            runtime_errors.append("budgeted CLI kill could not be proven reaped")

        if not self._shutdown_confirmed:
            detail = "; ".join(runtime_errors) or "SDK disconnect did not complete"
            raise ShutdownUnconfirmed(detail)

        if budgeted and self._budget_connection_admitted:
            if not self._budget_process_bound:
                # A provider-capable process may have existed, but its exact
                # identity never crossed the durable process-bound fence.
                self._accounting_unknown = True
            try:
                self._record_budget_event(
                    "connection_outcome",
                    {
                        "shutdown_confirmed": True,
                        "normal_terminal": normal_terminal,
                        "cost_usd_before_reset": self._accounting_cost_before_reset,
                        "turns_before_reset": self._accounting_turns_before_reset,
                        "current_accounting": (
                            None
                            if self._accounting_latest is None
                            else {
                                "cost_usd": self._accounting_latest.total_cost_usd,
                                "turns": self._accounting_latest.num_turns,
                            }
                        ),
                        "accounting_unknown": self._accounting_unknown,
                        "runtime_errors": list(runtime_errors),
                    },
                )
                self._budget_connection_closed = True
            except Exception as exc:
                durability_ok = False
                runtime_errors.append(
                    f"budget connection outcome failed: {type(exc).__name__}: {exc}"
                )

        try:
            unresolved_mirror = self.brain.sessions.unresolved_mirror_batches()
        except Exception as exc:
            unresolved_mirror = ()
            durability_ok = False
            runtime_errors.append(f"session mirror ledger failed: {type(exc).__name__}: {exc}")
        if unresolved_mirror:
            durability_ok = False
            fingerprints = ", ".join(
                str(item.get("fingerprint", "unknown"))[:12] for item in unresolved_mirror[:5]
            )
            runtime_errors.append(f"unresolved session mirror batches: {fingerprints}")

        control_failures = self._control_integrity_failures()
        latest = self._latest_relevant or sdk_result
        claim = self._completion_claim
        claim_status = str(claim.get("status")) if claim else ""

        # Advertise the products before the state that carries them exists:
        # publication is pending until the next checkpoint, and this is the
        # state delivery projects from.
        self._advertise_outputs()

        # Capture worker-created control corruption as evidence in the immutable
        # work state before replacing worker-report.json with the host report.
        work_state = self.brain.checkpoint(f"worker {self.brain.contract.identity} terminal work")
        control_failures.extend(self._state_control_integrity_failures(work_state))
        control_failures = list(dict.fromkeys(control_failures))

        outputs, output_failures = self._artifact_outputs(work_state)
        terminal_context = {
            "schema": "taste.brains/WorkerTerminalContext/1",
            "run_id": self.run_id,
            "assignment_id": (
                self.assignment.assignment_id if self.assignment is not None else None
            ),
            "generation": (self.assignment.generation if self.assignment is not None else None),
            "attempt": self.assignment.attempt if self.assignment is not None else None,
            "worker_claim": claim,
            "sdk": {
                "normal_terminal": normal_terminal,
                "terminal_reason": (self._effective_reason(latest) if latest is not None else None),
                "runtime_errors": list(runtime_errors),
                "durability_ok": durability_ok,
            },
            "mechanical_validation": {
                "control_failures": list(control_failures),
                "output_failures": list(output_failures),
                "conflicts": [conflict.to_dict() for conflict in work_state.conflicts],
                "outputs": [output.to_dict() for output in outputs],
                "active_tasks": sorted(self._active_tasks),
                "delegated_task_seen": self._delegated_task_seen,
                "protocol_uncertainty": list(self._protocol_uncertainty),
                "tail_uncertainty": list(self._tail_uncertainty),
                "pending_inbox_ids": [
                    str(item.message.get("id") or "") for item in self._pending_inbox
                ],
                "pending_verdicts": [dict(item.through) for item in self._pending_verdicts],
            },
        }
        terminal_assessment: Any | None = None
        assessment_error = ""
        try:
            certifier = getattr(self.monitor, "certify_terminal", None)
            if not callable(certifier):
                raise RuntimeError("monitor has no exact-state terminal certifier")
            terminal_assessment = await certifier(work_state, context=terminal_context)
            if getattr(terminal_assessment, "state_id", None) != work_state.id or getattr(
                terminal_assessment, "contract_digest", None
            ) != contract_digest(self.brain.contract):
                raise RuntimeError(
                    "monitor terminal assessment is not bound to this work state and contract"
                )
        except Exception as exc:
            assessment_error = f"terminal monitor assessment failed: {type(exc).__name__}: {exc}"

        try:
            monitor_report = {
                **self.monitor.report(),
                "worker_runtime_terminal": True,
            }
            # The report is embedded in a strict durable WorkerReport. Reject
            # non-JSON/non-finite telemetry here so a bad monitor cannot break
            # the terminal reporting boundary later.
            json.dumps(monitor_report, allow_nan=False)
        except Exception as exc:
            assessment_error = assessment_error or (
                f"terminal monitor report failed: {type(exc).__name__}: {exc}"
            )
            monitor_report = {
                "worker": self.brain.contract.identity,
                "contract_digest": contract_digest(self.brain.contract),
                "worst": "unknown",
                "current": "unknown",
                "current_state": None,
                "terminal_assessment": None,
                "model_calls": 0,
                "cost_known": False,
                "cost_usd": None,
                "pending_actions": [],
                "worker_runtime_terminal": True,
            }
        historical_monitor_severity = str(monitor_report.get("worst") or "unknown")
        monitor_severity = str(monitor_report.get("current") or "unknown")
        assessment_acceptable = bool(
            terminal_assessment is not None
            and getattr(terminal_assessment, "acceptable", False)
            and not assessment_error
        )

        if not durability_ok:
            terminal_reason = "mirror_error"
        elif runtime_errors:
            terminal_reason = "runtime_error"
        elif control_failures:
            terminal_reason = "control_integrity_error"
        elif self._sticky_result_errors:
            terminal_reason = "sdk_error"
        elif output_failures:
            terminal_reason = "output_validation_error"
        elif work_state.conflicts:
            terminal_reason = "work_conflict"
        elif assessment_error:
            terminal_reason = "monitor_assessment_error"
        elif not assessment_acceptable:
            terminal_reason = "monitor_rejected"
        elif self.assignment is None:
            terminal_reason = "turn_complete_without_assignment"
        elif self.assignment is not None and claim is None:
            terminal_reason = "missing_completion_claim"
        elif claim_status:
            terminal_reason = _reason_token(claim_status, "incomplete")
        elif latest is not None:
            terminal_reason = _reason_token(
                latest.terminal_reason or latest.stop_reason,
                "missing_completion_claim",
            )
        else:
            terminal_reason = "missing_result"

        uncertainty = list(self._protocol_uncertainty)
        uncertainty.extend(self._sticky_result_errors)
        uncertainty.extend(self._tail_uncertainty)
        uncertainty.extend(control_failures)
        if self.assignment is None:
            uncertainty.append("no typed assignment; SDK turn completion is not task completion")
        elif claim is None:
            uncertainty.append("worker emitted no valid structured completion claim")
        if self._active_tasks:
            uncertainty.append(
                "SDK delegated tasks remained active: " + ", ".join(sorted(self._active_tasks))
            )
        if self._delegated_task_seen:
            uncertainty.append(
                "delegated SDK task observed; the SDK exposes no proof-safe final "
                "continuation boundary"
            )
        if self._pending_inbox:
            uncertainty.append("submitted inbox messages remain unacknowledged")
        if self._pending_verdicts:
            uncertainty.append("submitted monitor feedback remains unacknowledged")
        if not durability_ok:
            uncertainty.append("the SDK transcript mirror is incomplete")
        uncertainty.extend(runtime_errors)
        if not assessment_acceptable:
            uncertainty.append(f"terminal monitor severity {monitor_severity}")
            if assessment_error:
                uncertainty.append(assessment_error)
            if terminal_assessment is not None and getattr(terminal_assessment, "failure", ""):
                uncertainty.append(str(terminal_assessment.failure))
        # ``worst`` is historical audit metadata.  A successful exact-state
        # TerminalAssessment has explicitly partitioned every prior finding
        # into resolved/unresolved and is acceptable only when none remain
        # unresolved.  Reintroducing its resolved history here made a
        # completed report uncertain and impossible for central delivery.
        if historical_monitor_severity != "fine" and not assessment_acceptable:
            uncertainty.append(f"historical monitor severity {historical_monitor_severity}")
        if monitor_report.get("pending_actions"):
            uncertainty.append("monitor actions remain pending")
        uncertainty = list(dict.fromkeys(reason[:512] for reason in uncertainty if reason))

        uncertainty.extend(output_failures)
        if work_state.conflicts:
            uncertainty.append("unresolved merge conflicts")
        uncertainty = list(dict.fromkeys(uncertainty))

        completed = bool(
            self.assignment is not None
            and claim_status == "completed"
            and normal_terminal
            and durability_ok
            and not runtime_errors
            and not self._sticky_result_errors
            and not self._protocol_uncertainty
            and not self._tail_uncertainty
            and not control_failures
            and not output_failures
            and not work_state.conflicts
            and not self._active_tasks
            and not self._delegated_task_seen
            and not self._pending_inbox
            and not self._pending_verdicts
            and assessment_acceptable
            and monitor_severity == "fine"
            and not monitor_report.get("pending_actions")
            and (not budgeted or (self._budget_process_bound and self._budget_connection_closed))
        )

        accounting = self._accounting_latest
        target_accounting_known = bool(
            accounting is not None
            and _valid_billed_cost(accounting.total_cost_usd)
            and accounting.num_turns is not None
            and not isinstance(accounting.num_turns, bool)
            and isinstance(accounting.num_turns, int)
            and accounting.num_turns >= 0
            and _valid_billed_cost(self._accounting_cost_before_reset)
            and not isinstance(self._accounting_turns_before_reset, bool)
            and isinstance(self._accounting_turns_before_reset, int)
            and self._accounting_turns_before_reset >= 0
            and not self._accounting_unknown
            and durability_ok
            and not runtime_errors
            and (not budgeted or (self._budget_process_bound and self._budget_connection_closed))
        )
        monitor_cost_known, monitor_cost = _monitor_billed_cost(monitor_report)
        monitor_assessment_failed = bool(
            assessment_error
            or terminal_assessment is None
            or getattr(terminal_assessment, "failure", "")
        )
        cost: float | None = None
        if (
            target_accounting_known
            and accounting is not None
            and monitor_cost_known
            and monitor_cost is not None
            and not monitor_assessment_failed
        ):
            try:
                combined = math.fsum(
                    (
                        self._accounting_cost_before_reset,
                        float(accounting.total_cost_usd),
                        monitor_cost,
                    )
                )
            except OverflowError:
                combined = math.inf
            if math.isfinite(combined):
                cost = combined
        turns = (
            self._accounting_turns_before_reset + accounting.num_turns
            if target_accounting_known and accounting is not None
            else None
        )
        error = "; ".join(runtime_errors)
        summary = str(claim.get("summary") or "") if claim else error
        evidence = list(claim.get("evidence") or ()) if claim else []

        generic_report: dict[str, Any] = {
            "schema": "taste.brains/RuntimeReport/1",
            "worker": self.brain.contract.identity,
            "run_id": self.run_id,
            "contract_digest": contract_digest(self.brain.contract),
            "final_state": work_state.id,
            "completed_claim": completed,
            "terminal_reason": terminal_reason,
            "session_id": self.brain._session_id,
            "turns": turns,
            "cost_usd": cost,
            "durability_ok": durability_ok,
            "monitor": monitor_report,
            "uncertainty": uncertainty,
            "error": error,
        }
        if self.assignment is not None:
            report_token = hashlib.sha256(f"{self.run_id}\0{work_state.id}".encode()).hexdigest()
            report = WorkerReport(
                report_id=f"worker-report.{report_token}",
                run_id=self.run_id,
                assignment_id=self.assignment.assignment_id,
                worker=self.brain.contract.identity,
                generation=self.assignment.generation,
                attempt=self.assignment.attempt,
                contract_digest=self.assignment.contract_digest,
                base_state_id=self.assignment.base_state_id,
                final_state_id=work_state.id,
                at=work_state.meta.created_at,
                completed=completed,
                terminal_reason=terminal_reason,
                outputs=outputs,
                turns=turns,
                cost_usd=cost,
                monitor_severity=monitor_severity,
                uncertain=bool(uncertainty),
                uncertainty_reasons=tuple(uncertainty),
                # The coordinator replans from this record. Without the
                # refusals in it, a planner cannot see that the last worker
                # spent its turns against a wall the next one will hit too.
                denials=self._permission_denials(self._results),
                summary=summary,
                metadata={
                    "durability_ok": durability_ok,
                    "session_id": self.brain._session_id,
                    "monitor": monitor_report,
                    "worker_evidence": evidence,
                    "structured_status": claim_status or None,
                    "assignment_digest": _record_digest(
                        self._accepted_assignment_text or self.assignment.to_json()
                    ),
                },
            )
            report_text = report.to_json()
        else:
            report_text = json.dumps(generic_report, indent=1, sort_keys=True) + "\n"

        _atomic_report_write(self.brain.branch.path(WORKER_REPORT_PATH), report_text)
        report_state = self.brain.checkpoint(
            f"worker {self.brain.contract.identity} terminal report: {terminal_reason}"
        )
        return SubBrainResult(
            identity=self.brain.contract.identity,
            completed=completed,
            terminal_reason=terminal_reason,
            turns=turns,
            cost_usd=cost,
            denials=self._permission_denials(self._results),
            error=error,
            states=[work_state.id, report_state.id],
        )
