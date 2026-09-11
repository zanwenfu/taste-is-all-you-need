"""Concrete typed-communication hook for the central coordinator.

``Communicator`` owns the exact inbox and its cumulative cursor.  This adapter
owns the other half of that protocol: an append-only acceptance ledger on the
same central control Branch used by planner, supervisor, and runtime.  Every
settlement is checkpointed under their exact shared RLock before Communicator
is allowed to advance the cursor.

Accepted current-generation messages remain signals in the ledger after inbox
acknowledgement.  They are returned on every call for that generation, so a
process death after ack but before CentralRuntime records its trigger cannot
turn a one-shot message invisible.  Once the promoted generation advances,
the old signal naturally retires.  Stale messages receive an audited ``stale``
receipt; a future message remains at the head of the inbox.

Each goal has a content-derived central recipient, so equal generation numbers
in two goals are not authentication and can never cross-consume one another's
messages.  The safe semantic default is deliberately conservative: every accepted
message requires replanning.  Unknown kinds therefore cost a planner turn
instead of disappearing into an observational log.  Artifact requests with
the same :func:`artifact_request_dedup_key` are returned as one grouped signal
containing every exact envelope and receipt, giving the planner one stable
producer identity for several consumers.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from taste.brains.central_planner import Goal
from taste.brains.communication import (
    ARTIFACT_REQUEST_KIND,
    AcceptanceReceipt,
    Communicator,
    GenerationStatus,
    InboxMessage,
    Message,
    artifact_request_dedup_key,
    group_artifact_requests,
)
from taste.brains.records import PlanRevision
from taste.brains.supervisor import SupervisorRun
from taste.memstore import Branch, Store
from taste.memstore.store import _check_name

if TYPE_CHECKING:
    from taste.brains.central_runtime import ExternalSignal

__all__ = [
    "ACCEPTANCE_INDEX_SCHEMA",
    "ACCEPTANCE_RECORD_SCHEMA",
    "CentralCommunication",
    "CentralCommunicationError",
    "CommunicationLedgerCorruption",
    "SharedCommunicationStateError",
    "central_recipient_for_goal",
]

ACCEPTANCE_INDEX_SCHEMA = "taste.brains/CentralAcceptanceIndex/1"
ACCEPTANCE_RECORD_SCHEMA = "taste.brains/CentralAcceptance/1"
SIGNAL_METADATA_SCHEMA = "taste.brains/CentralMessageSignal/1"
LEDGER_ROOT = ".taste/central-communication"

_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_STABLE_ID = re.compile(r"[^\x00-\x1f\x7f]{1,256}\Z")
_RLOCK_TYPE = type(threading.RLock())


class CentralCommunicationError(RuntimeError):
    """Base class for concrete central communication failures."""


class SharedCommunicationStateError(CentralCommunicationError):
    """The adapter was not given the coordinator's exact shared state."""


class CommunicationLedgerCorruption(CentralCommunicationError):
    """The append-only acceptance ledger was deleted, rewritten, or malformed."""


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


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8", "surrogatepass")).hexdigest()


def central_recipient_for_goal(goal_id: str) -> str:
    """Return the only central inbox identity authorized for ``goal_id``."""
    checked = _stable(goal_id, "goal_id")
    return f"central.{_hash({'goal_id': checked})}"


def _stable(value: Any, where: str) -> str:
    if not isinstance(value, str) or _STABLE_ID.fullmatch(value) is None or value != value.strip():
        raise CommunicationLedgerCorruption(f"{where} is not a stable identifier")
    return value


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CommunicationLedgerCorruption(f"{where} must be an integer >= 1")
    return value


def _exact_json(text: str, where: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CommunicationLedgerCorruption(
                    f"{where} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def no_constant(value: str) -> Any:
        raise CommunicationLedgerCorruption(f"{where} contains non-JSON number {value}")

    try:
        raw = json.loads(
            text,
            object_pairs_hook=no_duplicates,
            parse_constant=no_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise CommunicationLedgerCorruption(f"{where} is not exact JSON") from exc
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise CommunicationLedgerCorruption(f"{where} must be a JSON object")
    if _pretty(raw) != text:
        raise CommunicationLedgerCorruption(f"{where} is not canonical durable JSON")
    return raw


def _fields(raw: Mapping[str, Any], expected: set[str], where: str) -> None:
    if set(raw) != expected:
        raise CommunicationLedgerCorruption(f"{where} has missing or unknown fields")


def _operation_id(
    *,
    goal_id: str,
    recipient: str,
    item: InboxMessage,
    disposition: str,
    current_generation: int,
) -> str:
    identity = {
        "schema": "taste.brains/CentralAcceptanceIdentity/1",
        "goal_id": goal_id,
        "recipient": recipient,
        "inbox_id": item.inbox_id,
        "message_id": item.message.message_id,
        "disposition": disposition,
        "current_generation": current_generation,
    }
    return "acceptance." + _hash(identity)


@dataclass(frozen=True, slots=True)
class _LedgerEntry:
    sequence: int
    operation_id: str
    path: str
    goal_id: str
    accepted_plan_id: str
    item: InboxMessage
    receipt: AcceptanceReceipt


class CentralCommunication:
    """Production :class:`CommunicationHook` over one exact central inbox.

    ``control`` and ``control_lock`` are required objects, never names.  The
    caller should pass the very same instances given to ``CentralRuntime``;
    this adapter never opens a competing writable central branch.
    """

    def __init__(
        self,
        store: Store,
        *,
        goal_id: str,
        control: Branch,
        control_lock: threading.RLock,
        recipient: str | None = None,
        communicator: Communicator | None = None,
        fault_injector: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        if not isinstance(store, Store):
            raise TypeError("store must be a Store")
        if not isinstance(control, Branch) or control.store is not store:
            raise SharedCommunicationStateError(
                "control must be the exact writable Branch from this Store"
            )
        if not isinstance(control_lock, _RLOCK_TYPE):
            raise SharedCommunicationStateError("control_lock must be the shared RLock")
        if communicator is not None and (
            not isinstance(communicator, Communicator) or communicator.store is not store
        ):
            raise SharedCommunicationStateError(
                "communicator must use the exact shared Store"
            )
        self.store = store
        self.goal_id = _stable(goal_id, "goal_id")
        self.control = control
        self.control_lock = control_lock
        expected_recipient = central_recipient_for_goal(self.goal_id)
        supplied_recipient = expected_recipient if recipient is None else recipient
        self.recipient = _check_name(supplied_recipient, "central recipient")
        if self.recipient != expected_recipient:
            raise SharedCommunicationStateError(
                "central recipient must be the exact goal-derived inbox identity"
            )
        self.communicator = communicator or Communicator(store)
        self.fault_injector = fault_injector
        root_key = _hash(
            {
                "schema": "taste.brains/CentralAcceptanceLedgerIdentity/1",
                "session": store.session,
                "control_branch": control.name,
                "goal_id": self.goal_id,
                "recipient": self.recipient,
            }
        )
        self._root = f"{LEDGER_ROOT}/{root_key}"
        self._index_path = f"{self._root}/acceptance-index.json"
        self._receipts_root = f"{self._root}/receipts/"
        self._lock = threading.RLock()

    def assert_shared(
        self,
        *,
        store: Store,
        control: Branch,
        control_lock: threading.RLock,
    ) -> None:
        """Fail if coordinator wiring substituted a look-alike shared object."""
        if store is not self.store or control is not self.control or control_lock is not self.control_lock:
            raise SharedCommunicationStateError(
                "communication must share the exact coordinator Store, Branch, and RLock"
            )

    def _fault(self, boundary: str, payload: Mapping[str, Any]) -> None:
        if self.fault_injector is not None:
            self.fault_injector(boundary, payload)

    def _touches(self, path: str) -> tuple[str, ...]:
        try:
            output = self.store.backend.repo.git.rev_list(
                "--reverse",
                "--first-parent",
                self.control.head.id,
                "--",
                path,
            )
        except Exception as exc:
            raise CommunicationLedgerCorruption(
                f"cannot audit central communication path {path!r}"
            ) from exc
        return tuple(line for line in output.splitlines() if line)

    def _parse_index(self, text: str, where: str) -> list[dict[str, Any]]:
        raw = _exact_json(text, where)
        _fields(
            raw,
            {"schema", "session", "control_branch", "goal_id", "recipient", "entries"},
            where,
        )
        if (
            raw["schema"] != ACCEPTANCE_INDEX_SCHEMA
            or raw["session"] != self.store.session
            or raw["control_branch"] != self.control.name
            or raw["goal_id"] != self.goal_id
            or raw["recipient"] != self.recipient
            or not isinstance(raw["entries"], list)
        ):
            raise CommunicationLedgerCorruption(f"{where} has the wrong ledger identity")
        entries: list[dict[str, Any]] = []
        prior_generation = 0
        seen_operations: set[str] = set()
        seen_paths: set[str] = set()
        for offset, value in enumerate(raw["entries"], 1):
            if not isinstance(value, dict):
                raise CommunicationLedgerCorruption(f"{where} entry {offset} is not an object")
            _fields(
                value,
                {
                    "sequence",
                    "operation_id",
                    "path",
                    "inbox_id",
                    "message_id",
                    "disposition",
                    "message_generation",
                    "current_generation",
                },
                f"{where} entry {offset}",
            )
            if value["sequence"] != offset:
                raise CommunicationLedgerCorruption(f"{where} sequence is not contiguous")
            operation_id = _stable(value["operation_id"], "operation_id")
            path = value["path"]
            if (
                not isinstance(path, str)
                or path != f"{self._receipts_root}{operation_id}.json"
            ):
                raise CommunicationLedgerCorruption(f"{where} receipt path is not exact")
            if operation_id in seen_operations or path in seen_paths:
                raise CommunicationLedgerCorruption(f"{where} contains duplicate operations")
            seen_operations.add(operation_id)
            seen_paths.add(path)
            if not isinstance(value["inbox_id"], str) or _OBJECT_ID.fullmatch(
                value["inbox_id"]
            ) is None:
                raise CommunicationLedgerCorruption(f"{where} inbox id is not exact")
            if not isinstance(value["message_id"], str) or _DIGEST.fullmatch(
                value["message_id"]
            ) is None:
                raise CommunicationLedgerCorruption(f"{where} message id is not exact")
            if value["disposition"] not in {"accepted", "stale"}:
                raise CommunicationLedgerCorruption(f"{where} disposition is invalid")
            message_generation = _positive_int(
                value["message_generation"], "message_generation"
            )
            current_generation = _positive_int(
                value["current_generation"], "current_generation"
            )
            if current_generation < prior_generation:
                raise CommunicationLedgerCorruption(
                    f"{where} current generation regressed"
                )
            if value["disposition"] == "accepted" and message_generation != current_generation:
                raise CommunicationLedgerCorruption(
                    f"{where} accepted a non-current generation"
                )
            if value["disposition"] == "stale" and message_generation >= current_generation:
                raise CommunicationLedgerCorruption(f"{where} stale fence is invalid")
            prior_generation = current_generation
            entries.append(dict(value))
        return entries

    def _parse_receipt(
        self,
        text: str,
        index: Mapping[str, Any],
        durable_ref: str,
    ) -> _LedgerEntry:
        where = f"central acceptance {index['operation_id']}"
        raw = _exact_json(text, where)
        _fields(
            raw,
            {
                "schema",
                "sequence",
                "operation_id",
                "goal_id",
                "accepted_plan_id",
                "recipient",
                "inbox_id",
                "received_at",
                "message",
                "message_generation",
                "current_generation",
                "disposition",
            },
            where,
        )
        if raw["schema"] != ACCEPTANCE_RECORD_SCHEMA:
            raise CommunicationLedgerCorruption(f"{where} has the wrong schema")
        for name in (
            "sequence",
            "operation_id",
            "inbox_id",
            "message_generation",
            "current_generation",
            "disposition",
        ):
            if raw[name] != index[name]:
                raise CommunicationLedgerCorruption(f"{where} disagrees with its index")
        if (
            raw["goal_id"] != self.goal_id
            or raw["recipient"] != self.recipient
        ):
            raise CommunicationLedgerCorruption(f"{where} has the wrong owner")
        _stable(raw["accepted_plan_id"], "accepted_plan_id")
        if not isinstance(raw["message"], dict):
            raise CommunicationLedgerCorruption(f"{where} message is not an object")
        try:
            message = Message.from_dict(raw["message"])
            item = InboxMessage(
                inbox_id=raw["inbox_id"],
                received_at=raw["received_at"],
                message=message,
            )
            receipt = AcceptanceReceipt.for_message(
                item,
                disposition=raw["disposition"],
                current_generation=raw["current_generation"],
                durable_ref=durable_ref,
            )
        except Exception as exc:
            raise CommunicationLedgerCorruption(f"{where} has invalid typed content") from exc
        if (
            message.to_dict() != raw["message"]
            or message.message_id != index["message_id"]
            or message.generation != raw["message_generation"]
            or message.recipient != self.recipient
        ):
            raise CommunicationLedgerCorruption(f"{where} message identity changed")
        expected_operation = _operation_id(
            goal_id=self.goal_id,
            recipient=self.recipient,
            item=item,
            disposition=raw["disposition"],
            current_generation=raw["current_generation"],
        )
        if expected_operation != raw["operation_id"]:
            raise CommunicationLedgerCorruption(f"{where} operation identity changed")
        return _LedgerEntry(
            sequence=raw["sequence"],
            operation_id=raw["operation_id"],
            path=index["path"],
            goal_id=raw["goal_id"],
            accepted_plan_id=raw["accepted_plan_id"],
            item=item,
            receipt=receipt,
        )

    def _load_ledger(self) -> tuple[_LedgerEntry, ...]:
        """Audit every index version and immutable receipt before trusting any."""
        head = self.control.head
        index_commits = self._touches(self._index_path)
        current_index = head.read(self._index_path)
        files = {path for path in head.files() if path.startswith(f"{self._root}/")}
        if not index_commits:
            if current_index is not None or files:
                raise CommunicationLedgerCorruption(
                    "central acceptance files exist without an index history"
                )
            seen_ref = (
                f"{self.store.SEEN_REF}/{self.store.session}/{self.recipient}"
            )
            if self.store.backend.ref_sha(seen_ref) is not None:
                raise CommunicationLedgerCorruption(
                    "acknowledged inbox entry has no durable acceptance receipt"
                )
            return ()
        if current_index is None:
            raise CommunicationLedgerCorruption("central acceptance index was deleted")

        previous: list[dict[str, Any]] = []
        introduction_commits: list[str] = []
        latest_text = ""
        for version, commit in enumerate(index_commits, 1):
            text = self.store.backend.show(commit, self._index_path)
            if text is None:
                raise CommunicationLedgerCorruption("central acceptance index was deleted")
            entries = self._parse_index(text, f"central acceptance index version {version}")
            if len(entries) != len(previous) + 1 or entries[:-1] != previous:
                raise CommunicationLedgerCorruption(
                    "central acceptance index is not append-only"
                )
            previous = entries
            introduction_commits.append(commit)
            latest_text = text
        if current_index != latest_text:
            raise CommunicationLedgerCorruption(
                "current central acceptance index differs from its audited history"
            )

        expected_files = {self._index_path, *(entry["path"] for entry in previous)}
        if files != expected_files:
            raise CommunicationLedgerCorruption(
                "central acceptance index and receipt files disagree"
            )

        loaded: list[_LedgerEntry] = []
        for entry, introduction in zip(previous, introduction_commits, strict=True):
            path = entry["path"]
            touches = self._touches(path)
            if touches != (introduction,):
                raise CommunicationLedgerCorruption(
                    f"central acceptance receipt {path!r} was rewritten or deleted"
                )
            text = head.read(path)
            if text is None:
                raise CommunicationLedgerCorruption(
                    f"central acceptance receipt {path!r} disappeared"
                )
            loaded.append(self._parse_receipt(text, entry, introduction))

        # A syntactically valid forged ledger entry is still corruption unless
        # its exact immutable envelope exists in this recipient's inbox log.
        try:
            history = self.communicator.history(self.recipient)
        except Exception as exc:
            raise CommunicationLedgerCorruption(
                "cannot prove acceptance receipts against the exact inbox history"
            ) from exc
        history_by_id: dict[str, InboxMessage] = {}
        for item in history:
            if item.inbox_id in history_by_id:
                raise CommunicationLedgerCorruption(
                    "the exact inbox history contains a duplicate commit identity"
                )
            history_by_id[item.inbox_id] = item
        for entry in loaded:
            historical = history_by_id.get(entry.item.inbox_id)
            if historical != entry.item:
                raise CommunicationLedgerCorruption(
                    "acceptance receipt is absent from the exact inbox history"
                )

        # Prove the converse as well.  The inbox cursor is cumulative and only
        # moves after ``_persist_acceptance`` returns, so every item before it
        # must have exactly one receipt in this ledger.  Without this check an
        # external force-update of the control ref could truncate the ledger
        # while leaving the cursor advanced: the message would be neither
        # pending nor replayed, and the central brain would silently forget an
        # accepted signal.  Reading history and pending under the repository
        # lock gives one coherent cursor/inbox snapshot even if another process
        # is concurrently sending.
        try:
            with self.store.backend.lock():
                pending = self.communicator.pending(self.recipient)
                cursor_history = self.communicator.history(self.recipient)
        except Exception as exc:
            raise CommunicationLedgerCorruption(
                "cannot prove the acceptance ledger against the inbox cursor"
            ) from exc
        if cursor_history != history:
            raise CommunicationLedgerCorruption(
                "typed inbox history changed during acceptance audit"
            )
        acknowledged_count = len(history) - len(pending)
        if acknowledged_count < 0 or history[acknowledged_count:] != pending:
            raise CommunicationLedgerCorruption(
                "typed inbox cursor does not select an exact history suffix"
            )
        # Settlement is oldest-first.  Therefore the ledger must be the exact
        # oldest prefix through the cursor, with at most one additional entry:
        # the intentional crash window after its receipt checkpoint and before
        # cursor advancement.  Merely checking set inclusion would accept a
        # forged receipt for a later message or a gap hiding an acknowledged
        # predecessor.
        if len(loaded) not in {
            acknowledged_count,
            acknowledged_count + 1,
        } or len(loaded) > len(history):
            raise CommunicationLedgerCorruption(
                "acceptance ledger does not exactly cover the inbox cursor"
            )
        if tuple(entry.item for entry in loaded) != history[: len(loaded)]:
            raise CommunicationLedgerCorruption(
                "acceptance ledger is not the exact oldest inbox prefix"
            )
        return tuple(loaded)

    def _index_record(self, entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "schema": ACCEPTANCE_INDEX_SCHEMA,
            "session": self.store.session,
            "control_branch": self.control.name,
            "goal_id": self.goal_id,
            "recipient": self.recipient,
            "entries": [dict(entry) for entry in entries],
        }

    @staticmethod
    def _index_entry(entry: _LedgerEntry) -> dict[str, Any]:
        receipt = entry.receipt
        return {
            "sequence": entry.sequence,
            "operation_id": entry.operation_id,
            "path": entry.path,
            "inbox_id": receipt.inbox_id,
            "message_id": receipt.message_id,
            "disposition": receipt.disposition,
            "message_generation": receipt.message_generation,
            "current_generation": receipt.current_generation,
        }

    def _persist_acceptance(
        self,
        item: InboxMessage,
        disposition: str,
        current_generation: int,
        *,
        plan_id: str,
    ) -> AcceptanceReceipt:
        operation_id = _operation_id(
            goal_id=self.goal_id,
            recipient=self.recipient,
            item=item,
            disposition=disposition,
            current_generation=current_generation,
        )
        with self.control_lock:
            entries = self._load_ledger()
            matching = [entry for entry in entries if entry.operation_id == operation_id]
            if matching:
                if len(matching) != 1:
                    raise CommunicationLedgerCorruption(
                        "acceptance operation appears more than once"
                    )
                existing = matching[0]
                expected = AcceptanceReceipt.for_message(
                    item,
                    disposition=disposition,
                    current_generation=current_generation,
                    durable_ref=existing.receipt.durable_ref,
                )
                if existing.receipt != expected:
                    raise CommunicationLedgerCorruption(
                        "idempotent acceptance operation changed message identity"
                    )
                return existing.receipt

            if entries and current_generation < entries[-1].receipt.current_generation:
                raise CommunicationLedgerCorruption(
                    "central acceptance generation cannot regress"
                )
            dirty = self.control.dirty_paths()
            if dirty:
                raise CommunicationLedgerCorruption(
                    "central control worktree is dirty at acceptance boundary"
                )
            sequence = len(entries) + 1
            path = f"{self._receipts_root}{operation_id}.json"
            record = {
                "schema": ACCEPTANCE_RECORD_SCHEMA,
                "sequence": sequence,
                "operation_id": operation_id,
                "goal_id": self.goal_id,
                "accepted_plan_id": plan_id,
                "recipient": self.recipient,
                "inbox_id": item.inbox_id,
                "received_at": item.received_at,
                "message": item.message.to_dict(),
                "message_generation": item.message.generation,
                "current_generation": current_generation,
                "disposition": disposition,
            }
            provisional = _LedgerEntry(
                sequence=sequence,
                operation_id=operation_id,
                path=path,
                goal_id=self.goal_id,
                accepted_plan_id=plan_id,
                item=item,
                receipt=AcceptanceReceipt.for_message(
                    item,
                    disposition=disposition,
                    current_generation=current_generation,
                    durable_ref="pending-checkpoint",
                ),
            )
            index_entries = [self._index_entry(entry) for entry in entries]
            index_entries.append(self._index_entry(provisional))
            state = self.control.checkpoint(
                f"central communication {disposition}: {item.message.message_id}",
                records={
                    path: record,
                    self._index_path: self._index_record(index_entries),
                },
            )
            receipt = AcceptanceReceipt.for_message(
                item,
                disposition=disposition,
                current_generation=current_generation,
                durable_ref=state.id,
            )
            # Read back through the full append-only audit before cursor
            # movement. A checkpoint return alone is not enough if another
            # central component wrote unexpected bytes into the same tree.
            verified = self._load_ledger()
            if not verified or verified[-1].receipt != receipt:
                raise CommunicationLedgerCorruption(
                    "new central acceptance receipt failed durable read-back"
                )
        self._fault(
            "acceptance_checkpoint",
            {
                "operation_id": operation_id,
                "inbox_id": item.inbox_id,
                "message_id": item.message.message_id,
                "disposition": disposition,
                "current_generation": current_generation,
                "durable_ref": receipt.durable_ref,
            },
        )
        return receipt

    @staticmethod
    def _message_metadata(entry: _LedgerEntry) -> dict[str, Any]:
        receipt = entry.receipt
        return {
            "acceptance_sequence": entry.sequence,
            "acceptance_operation_id": entry.operation_id,
            "acceptance_state_id": receipt.durable_ref,
            "accepted_plan_id": entry.accepted_plan_id,
            "inbox_id": entry.item.inbox_id,
            "received_at": entry.item.received_at,
            "message": entry.item.message.to_dict(),
            "receipt": {
                "inbox_id": receipt.inbox_id,
                "message_id": receipt.message_id,
                "recipient": receipt.recipient,
                "message_generation": receipt.message_generation,
                "current_generation": receipt.current_generation,
                "disposition": receipt.disposition,
                "durable_ref": receipt.durable_ref,
            },
        }

    def _durable_signals(
        self,
        plan: PlanRevision,
        entries: Sequence[_LedgerEntry],
    ) -> tuple[ExternalSignal, ...]:
        # Local import keeps this concrete adapter importable from
        # central_runtime without making ExternalSignal a module-load cycle.
        from taste.brains.central_runtime import ExternalSignal

        accepted = [
            entry
            for entry in entries
            if entry.receipt.disposition == "accepted"
            and entry.receipt.current_generation == plan.generation
        ]
        artifact_entries = [
            entry for entry in accepted if entry.item.message.kind == ARTIFACT_REQUEST_KIND
        ]
        artifact_by_inbox = {entry.item.inbox_id: entry for entry in artifact_entries}
        grouped = group_artifact_requests(entry.item for entry in artifact_entries)

        ordered: list[tuple[int, ExternalSignal]] = []
        for dedup_key, items in grouped.items():
            members = [artifact_by_inbox[item.inbox_id] for item in items]
            members.sort(key=lambda entry: entry.sequence)
            metadata = {
                "schema": SIGNAL_METADATA_SCHEMA,
                "goal_id": self.goal_id,
                "current_plan_id": plan.plan_id,
                "generation": plan.generation,
                "replan_policy": "all_accepted_messages",
                "artifact_request_dedup_key": dedup_key,
                "artifact_request_group": [
                    self._message_metadata(entry) for entry in members
                ],
            }
            signal_id = "central-message-group." + dedup_key.removeprefix("sha256:")
            ordered.append(
                (
                    members[0].sequence,
                    ExternalSignal(
                        signal_id=signal_id,
                        kind=ARTIFACT_REQUEST_KIND,
                        detail=(
                            f"{len(members)} worker request(s) need one deduplicated "
                            "artifact producer"
                        ),
                        requires_replan=True,
                        metadata=metadata,
                    ),
                )
            )

        for entry in accepted:
            if entry.item.message.kind == ARTIFACT_REQUEST_KIND:
                continue
            message = entry.item.message
            metadata = {
                "schema": SIGNAL_METADATA_SCHEMA,
                "goal_id": self.goal_id,
                "current_plan_id": plan.plan_id,
                "generation": plan.generation,
                "replan_policy": "all_accepted_messages",
                **self._message_metadata(entry),
            }
            ordered.append(
                (
                    entry.sequence,
                    ExternalSignal(
                        signal_id=(
                            "central-message."
                            + message.message_id.removeprefix("sha256:")
                        ),
                        kind=message.kind,
                        detail=f"accepted {message.kind} from {message.sender}",
                        requires_replan=True,
                        metadata=metadata,
                    ),
                )
            )
        ordered.sort(key=lambda value: (value[0], value[1].signal_id))
        return tuple(signal for _sequence, signal in ordered)

    @staticmethod
    def _validate_current_message(item: InboxMessage) -> None:
        """Reject semantic messages that could not become an exact signal."""
        from taste.brains.central_runtime import ExternalSignal

        message = item.message
        if message.kind == ARTIFACT_REQUEST_KIND:
            dedup_key = artifact_request_dedup_key(message)
            signal_id = "central-message-group." + dedup_key.removeprefix("sha256:")
        else:
            signal_id = "central-message." + message.message_id.removeprefix("sha256:")
        ExternalSignal(
            signal_id=signal_id,
            kind=message.kind,
            detail=f"accepted {message.kind} from {message.sender}",
            requires_replan=True,
            metadata={"inbox_id": item.inbox_id, "message": message.to_dict()},
        )

    def signals(
        self,
        *,
        goal: Goal,
        plan: PlanRevision,
        runs: tuple[SupervisorRun, ...],
    ) -> Sequence[ExternalSignal]:
        """Settle the exact inbox oldest-first, then replay durable signals."""
        if not isinstance(goal, Goal) or goal.goal_id != self.goal_id:
            raise SharedCommunicationStateError("communication received the wrong Goal")
        if (
            not isinstance(plan, PlanRevision)
            or plan.goal_id != self.goal_id
            or plan.generation < 1
        ):
            raise SharedCommunicationStateError("communication received the wrong PlanRevision")
        if not isinstance(runs, tuple) or not all(isinstance(run, SupervisorRun) for run in runs):
            raise SharedCommunicationStateError("communication runs must be typed SupervisorRuns")

        with self._lock:
            with self.control_lock:
                initial_ledger = self._load_ledger()
            if initial_ledger and (
                initial_ledger[-1].receipt.current_generation > plan.generation
            ):
                raise CommunicationLedgerCorruption(
                    "promoted plan generation regressed behind the acceptance ledger"
                )
            while True:
                pending = self.communicator.pending(self.recipient)
                if not pending:
                    break
                item = pending[0]
                status = self.communicator.generation_status(
                    item.message, plan.generation
                )
                if status is GenerationStatus.FUTURE:
                    break

                if status is GenerationStatus.CURRENT:
                    self._validate_current_message(item)

                def boundary(
                    accepted: InboxMessage,
                    disposition: str,
                    current_generation: int,
                ) -> AcceptanceReceipt:
                    return self._persist_acceptance(
                        accepted,
                        disposition,
                        current_generation,
                        plan_id=plan.plan_id,
                    )

                if status is GenerationStatus.STALE:
                    self.communicator.retire_stale(
                        item,
                        current_generation=plan.generation,
                        boundary=boundary,
                    )
                else:
                    self.communicator.accept(
                        item,
                        current_generation=plan.generation,
                        boundary=boundary,
                    )
            with self.control_lock:
                ledger = self._load_ledger()
            return self._durable_signals(plan, ledger)
