"""Typed, durable communication between central and worker brains.

The memstore inbox is the transport and the durable log.  This module adds the
protocol properties that a central coordinator needs above it:

* a strict, versioned message value with a stable semantic identity;
* one durable inbox entry for one sender/idempotency-key pair;
* generation fencing without deleting stale history; and
* a two-phase receive boundary: persist acceptance first, then advance the
  exact memstore inbox cursor.

The outer inbox commit id and the inner ``Message.message_id`` intentionally
mean different things.  The former is the monotonically ordered delivery
cursor; the latter is stable across retries, including a retry after the first
process was killed immediately after :meth:`Store.send` returned.

Acceptance callbacks are part of the durability boundary.  They must persist
and idempotently recover the returned :class:`AcceptanceReceipt`; this module
validates the receipt's exact binding before acknowledging the inbox entry.
It cannot prove that an arbitrary callback really fsync'd its state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol

from git.exc import GitCommandError

from taste.brains.records import ArtifactRef
from taste.memstore import Store

__all__ = [
    "ARTIFACT_REQUEST_KIND",
    "AcceptanceBoundary",
    "AcceptanceReceipt",
    "CommunicationCorruption",
    "CommunicationError",
    "Communicator",
    "DuplicateDurableMessage",
    "GenerationFence",
    "GenerationStatus",
    "IdempotencyConflict",
    "InboxMessage",
    "InboxOrderError",
    "InvalidAcceptanceReceipt",
    "Message",
    "MessagePolicy",
    "MessageWireError",
    "artifact_request_dedup_key",
    "group_artifact_requests",
]

MESSAGE_SCHEMA = "taste.brains/Message/1"
ARTIFACT_REQUEST_KIND = "artifact_request"
_ARTIFACT_REQUEST_KEY_SCHEMA = "taste.brains/ArtifactRequestKey/1"

_BRANCH_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DIGEST_RE = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
_OBJECT_ID_RE = re.compile(r"\A(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_TOKEN_RE = re.compile(r"\A[a-z][a-z0-9_.-]*\Z")

# ``GitBackend.lock`` is process-reentrant and therefore deliberately does not
# serialize two threads in the same process.  Communication's check-then-send
# must do so, hence this narrower per-repository thread lock outside it.
_SEND_LOCKS_GUARD = threading.Lock()
_SEND_LOCKS: dict[tuple[int, str], threading.RLock] = {}


def _send_lock(store: Store) -> threading.RLock:
    # A fork inherits Python lock objects but not the owning thread.  Give the
    # child a fresh local lock; the backend flock still serializes processes.
    key = (os.getpid(), str(store.backend.common_dir))
    with _SEND_LOCKS_GUARD:
        return _SEND_LOCKS.setdefault(key, threading.RLock())


class CommunicationError(RuntimeError):
    """Base class for protocol failures that must not be ignored."""


class MessageWireError(CommunicationError, ValueError):
    """An inbox entry claims to be typed communication but is not exact wire."""


class IdempotencyConflict(CommunicationError):
    """One sender reused an idempotency key for different message bytes."""


class DuplicateDurableMessage(CommunicationError):
    """More than one inbox commit contains the same semantic message."""


class CommunicationCorruption(CommunicationError):
    """Durable transport state contradicts the typed protocol."""


class GenerationFence(CommunicationError):
    """A stale or future message was offered to the current generation."""


class InboxOrderError(CommunicationError):
    """A caller attempted to acknowledge something other than the oldest item."""


class InvalidAcceptanceReceipt(CommunicationError, ValueError):
    """The durable boundary returned a receipt for a different operation."""


def _expect_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise MessageWireError(f"{where} must be a JSON object with string keys")
    return value


def _check_fields(
    raw: Mapping[str, Any],
    *,
    where: str,
    required: set[str],
) -> None:
    missing = required - raw.keys()
    if missing:
        raise MessageWireError(f"{where} is missing required fields: {sorted(missing)}")
    unknown = raw.keys() - required
    if unknown:
        raise MessageWireError(f"{where} has unknown fields: {sorted(unknown)}")


def _string(value: Any, where: str, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise MessageWireError(f"{where} must be a string")
    if not empty and not value:
        raise MessageWireError(f"{where} must not be empty")
    if "\x00" in value:
        raise MessageWireError(f"{where} must not contain NUL")
    return value


def _stable_id(value: Any, where: str) -> str:
    text = _string(value, where)
    if (
        text != text.strip()
        or len(text) > 256
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
    ):
        raise MessageWireError(f"{where} is not a stable identifier")
    return text


def _branch(value: Any, where: str) -> str:
    text = _string(value, where)
    if not _BRANCH_RE.fullmatch(text) or text in {".", ".."} or text.endswith(".lock"):
        raise MessageWireError(f"{where} is not a valid branch identity")
    return text


def _generation(value: Any, where: str = "generation") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MessageWireError(f"{where} must be an integer >= 1")
    return value


def _digest(value: Any, where: str) -> str:
    text = _string(value, where)
    if not _DIGEST_RE.fullmatch(text):
        raise MessageWireError(f"{where} must be a sha256 digest")
    return text


def _object_id(value: Any, where: str) -> str:
    text = _string(value, where)
    if not _OBJECT_ID_RE.fullmatch(text):
        raise MessageWireError(f"{where} must be a full lowercase object ID")
    return text


def _token(value: Any, where: str) -> str:
    text = _string(value, where)
    if not _TOKEN_RE.fullmatch(text):
        raise MessageWireError(f"{where} must be a lowercase token")
    return text


def _timestamp(value: Any, where: str) -> str:
    text = _string(value, where)
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError as exc:
        raise MessageWireError(f"{where} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MessageWireError(f"{where} must include a timezone")
    return text


def _freeze_json(value: Any, where: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MessageWireError(f"{where} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise MessageWireError(f"{where} keys must be strings")
            frozen[key] = _freeze_json(item, f"{where}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{where}[{index}]") for index, item in enumerate(value))
    raise MessageWireError(f"{where} contains non-JSON value {type(value).__name__}")


def _freeze_map(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MessageWireError(f"{where} must be a JSON object")
    frozen = _freeze_json(value, where)
    assert isinstance(frozen, Mapping)
    return frozen


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _thaw_json(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _loads(text: str, where: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise MessageWireError(f"{where} contains duplicate key {key!r}")
            out[key] = value
        return out

    def no_constant(value: str) -> Any:
        raise MessageWireError(f"{where} contains non-JSON number {value}")

    try:
        value = json.loads(text, object_pairs_hook=no_duplicates, parse_constant=no_constant)
    except json.JSONDecodeError as exc:
        raise MessageWireError(f"{where} is not valid JSON") from exc
    return _expect_object(value, where)


def _message_id(sender: str, idempotency_key: str) -> str:
    return _sha256(
        {
            "schema": "taste.brains/MessageIdentity/1",
            "sender": sender,
            "idempotency_key": idempotency_key,
        }
    )


@dataclass(frozen=True, slots=True)
class Message:
    """One immutable semantic message, independent of delivery attempts."""

    message_id: str
    idempotency_key: str
    kind: str
    sender: str
    recipient: str
    generation: int
    request_id: str | None = None
    reply_to_id: str | None = None
    artifact_ref: ArtifactRef | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sender = _branch(self.sender, "sender")
        key = _stable_id(self.idempotency_key, "idempotency_key")
        supplied = _digest(self.message_id, "message_id")
        wanted = _message_id(sender, key)
        if supplied != wanted:
            raise MessageWireError("message_id does not match sender and idempotency_key")
        object.__setattr__(self, "message_id", supplied)
        object.__setattr__(self, "idempotency_key", key)
        object.__setattr__(self, "kind", _token(self.kind, "kind"))
        object.__setattr__(self, "sender", sender)
        object.__setattr__(self, "recipient", _branch(self.recipient, "recipient"))
        object.__setattr__(self, "generation", _generation(self.generation))
        if self.request_id is not None:
            object.__setattr__(self, "request_id", _stable_id(self.request_id, "request_id"))
        if self.reply_to_id is not None:
            object.__setattr__(self, "reply_to_id", _digest(self.reply_to_id, "reply_to_id"))
        if self.artifact_ref is not None and not isinstance(self.artifact_ref, ArtifactRef):
            raise MessageWireError("artifact_ref must be an ArtifactRef or null")
        object.__setattr__(self, "payload", _freeze_map(self.payload, "payload"))

    @classmethod
    def create(
        cls,
        *,
        idempotency_key: str,
        kind: str,
        sender: str,
        recipient: str,
        generation: int,
        request_id: str | None = None,
        reply_to_id: str | None = None,
        artifact_ref: ArtifactRef | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> Message:
        checked_sender = _branch(sender, "sender")
        checked_key = _stable_id(idempotency_key, "idempotency_key")
        return cls(
            message_id=_message_id(checked_sender, checked_key),
            idempotency_key=checked_key,
            kind=kind,
            sender=checked_sender,
            recipient=recipient,
            generation=generation,
            request_id=request_id,
            reply_to_id=reply_to_id,
            artifact_ref=artifact_ref,
            payload={} if payload is None else payload,
        )

    @classmethod
    def artifact_request(
        cls,
        *,
        idempotency_key: str,
        sender: str,
        recipient: str,
        generation: int,
        artifact_id: str,
        requirement: Mapping[str, Any] | None = None,
        request_id: str | None = None,
        reply_to_id: str | None = None,
    ) -> Message:
        """Build a request whose dedup identity excludes the requester."""
        return cls.create(
            idempotency_key=idempotency_key,
            kind=ARTIFACT_REQUEST_KIND,
            sender=sender,
            recipient=recipient,
            generation=generation,
            request_id=request_id,
            reply_to_id=reply_to_id,
            payload={
                "artifact_id": _stable_id(artifact_id, "artifact_id"),
                "requirement": {} if requirement is None else requirement,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MESSAGE_SCHEMA,
            "message_id": self.message_id,
            "idempotency_key": self.idempotency_key,
            "kind": self.kind,
            "sender": self.sender,
            "recipient": self.recipient,
            "generation": self.generation,
            "request_id": self.request_id,
            "reply_to_id": self.reply_to_id,
            "artifact_ref": None if self.artifact_ref is None else self.artifact_ref.to_dict(),
            "payload": _thaw_json(self.payload),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=1,
            sort_keys=True,
        ) + "\n"

    @classmethod
    def from_dict(cls, value: Any) -> Message:
        raw = _expect_object(value, "Message")
        required = {
            "schema",
            "message_id",
            "idempotency_key",
            "kind",
            "sender",
            "recipient",
            "generation",
            "request_id",
            "reply_to_id",
            "artifact_ref",
            "payload",
        }
        _check_fields(raw, where="Message", required=required)
        if raw["schema"] != MESSAGE_SCHEMA:
            raise MessageWireError(
                f"Message.schema must be {MESSAGE_SCHEMA!r}, got {raw['schema']!r}"
            )
        request_id = raw["request_id"]
        if request_id is not None and not isinstance(request_id, str):
            raise MessageWireError("request_id must be a string or null")
        reply_to_id = raw["reply_to_id"]
        if reply_to_id is not None and not isinstance(reply_to_id, str):
            raise MessageWireError("reply_to_id must be a string or null")
        artifact = raw["artifact_ref"]
        if artifact is not None:
            try:
                artifact = ArtifactRef.from_dict(_expect_object(artifact, "artifact_ref"))
            except ValueError as exc:
                raise MessageWireError(f"invalid artifact_ref: {exc}") from exc
        return cls(
            message_id=raw["message_id"],
            idempotency_key=raw["idempotency_key"],
            kind=raw["kind"],
            sender=raw["sender"],
            recipient=raw["recipient"],
            generation=raw["generation"],
            request_id=request_id,
            reply_to_id=reply_to_id,
            artifact_ref=artifact,
            payload=raw["payload"],
        )

    @classmethod
    def from_json(cls, text: str) -> Message:
        return cls.from_dict(_loads(text, "Message"))


@dataclass(frozen=True, slots=True)
class InboxMessage:
    """A typed message bound to one exact commit in a recipient's inbox."""

    inbox_id: str
    received_at: str
    message: Message

    def __post_init__(self) -> None:
        object.__setattr__(self, "inbox_id", _object_id(self.inbox_id, "inbox_id"))
        object.__setattr__(self, "received_at", _timestamp(self.received_at, "received_at"))
        if not isinstance(self.message, Message):
            raise MessageWireError("message must be a Message")


class GenerationStatus(StrEnum):
    STALE = "stale"
    CURRENT = "current"
    FUTURE = "future"


def _generation_status(message_generation: int, current_generation: int) -> GenerationStatus:
    current = _generation(current_generation, "current_generation")
    if message_generation < current:
        return GenerationStatus.STALE
    if message_generation > current:
        return GenerationStatus.FUTURE
    return GenerationStatus.CURRENT


@dataclass(frozen=True, slots=True)
class AcceptanceReceipt:
    """Proof returned by a caller's durable, idempotent acceptance boundary."""

    inbox_id: str
    message_id: str
    recipient: str
    message_generation: int
    current_generation: int
    disposition: str
    durable_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "inbox_id", _object_id(self.inbox_id, "inbox_id"))
        object.__setattr__(self, "message_id", _digest(self.message_id, "message_id"))
        object.__setattr__(self, "recipient", _branch(self.recipient, "recipient"))
        object.__setattr__(
            self,
            "message_generation",
            _generation(self.message_generation, "message_generation"),
        )
        object.__setattr__(
            self,
            "current_generation",
            _generation(self.current_generation, "current_generation"),
        )
        if self.disposition not in {"accepted", "stale"}:
            raise InvalidAcceptanceReceipt("disposition must be 'accepted' or 'stale'")
        status = _generation_status(self.message_generation, self.current_generation)
        expected = {
            "accepted": GenerationStatus.CURRENT,
            "stale": GenerationStatus.STALE,
        }[self.disposition]
        if status is not expected:
            raise InvalidAcceptanceReceipt(
                f"{self.disposition} receipt has {status.value} generation binding"
            )
        object.__setattr__(self, "durable_ref", _stable_id(self.durable_ref, "durable_ref"))

    @classmethod
    def for_message(
        cls,
        item: InboxMessage,
        *,
        disposition: str,
        current_generation: int,
        durable_ref: str,
    ) -> AcceptanceReceipt:
        return cls(
            inbox_id=item.inbox_id,
            message_id=item.message.message_id,
            recipient=item.message.recipient,
            message_generation=item.message.generation,
            current_generation=current_generation,
            disposition=disposition,
            durable_ref=durable_ref,
        )


class AcceptanceBoundary(Protocol):
    """Persist one disposition idempotently and return its exact receipt."""

    def __call__(
        self, item: InboxMessage, disposition: str, current_generation: int, /
    ) -> AcceptanceReceipt: ...


class MessagePolicy(Protocol):
    """Injectable semantic validation beyond the stable wire schema."""

    def __call__(self, message: Message, /) -> None: ...


def artifact_request_dedup_key(message: Message) -> str:
    """Return the producer-assignment identity for an exact artifact request.

    Sender, recipient, request id, reply id, and message id are deliberately
    excluded, so several consumers asking for the same exact requirement in
    one plan generation collapse to one key.  The generation and canonical
    requirement are included, so an old request or a changed requirement does
    not accidentally reuse a producer assignment.
    """
    if not isinstance(message, Message) or message.kind != ARTIFACT_REQUEST_KIND:
        raise MessageWireError(f"message kind must be {ARTIFACT_REQUEST_KIND!r}")
    if message.artifact_ref is not None:
        raise MessageWireError("an artifact request cannot already carry an artifact_ref")
    raw = _thaw_json(message.payload)
    assert isinstance(raw, dict)
    _check_fields(
        raw,
        where="artifact request payload",
        required={"artifact_id", "requirement"},
    )
    artifact_id = _stable_id(raw["artifact_id"], "artifact request artifact_id")
    requirement = _freeze_map(raw["requirement"], "artifact request requirement")
    return _sha256(
        {
            "schema": _ARTIFACT_REQUEST_KEY_SCHEMA,
            "generation": message.generation,
            "artifact_id": artifact_id,
            "requirement": requirement,
        }
    )


def group_artifact_requests(
    items: Iterable[InboxMessage],
) -> dict[str, tuple[InboxMessage, ...]]:
    """Group requests by the key a coordinator should use for one producer."""
    grouped: dict[str, list[InboxMessage]] = {}
    for item in items:
        key = artifact_request_dedup_key(item.message)
        grouped.setdefault(key, []).append(item)
    return {key: tuple(values) for key, values in grouped.items()}


class Communicator:
    """Strict typed protocol over one :class:`~taste.memstore.Store` session."""

    def __init__(
        self,
        store: Store,
        *,
        message_policy: MessagePolicy | Callable[[Message], None] | None = None,
    ) -> None:
        if not isinstance(store, Store):
            raise TypeError("store must be a Store")
        self.store = store
        self.message_policy = message_policy

    def _validate_policy(self, message: Message) -> None:
        if self.message_policy is not None:
            self.message_policy(message)

    @staticmethod
    def generation_status(message: Message, current_generation: int) -> GenerationStatus:
        return _generation_status(message.generation, current_generation)

    @staticmethod
    def _parse_envelope(raw: Mapping[str, Any], recipient: str) -> InboxMessage:
        envelope = dict(raw)
        _check_fields(
            envelope,
            where="inbox envelope",
            required={"sender", "at", "body", "id"},
        )
        body = _expect_object(envelope["body"], "inbox envelope.body")
        message = Message.from_dict(body)
        if message.recipient != recipient:
            raise MessageWireError(
                f"message recipient {message.recipient!r} does not match inbox {recipient!r}"
            )
        outer_sender = _branch(envelope["sender"], "inbox envelope.sender")
        if outer_sender != message.sender:
            raise MessageWireError("inbox envelope sender does not match Message.sender")
        return InboxMessage(
            inbox_id=envelope["id"],
            received_at=envelope["at"],
            message=message,
        )

    @staticmethod
    def _claims_typed_protocol(raw: Mapping[str, Any]) -> bool:
        body = raw.get("body")
        return isinstance(body, Mapping) and body.get("schema") == MESSAGE_SCHEMA

    def _raw_inbox(
        self,
        recipient: str,
        *,
        include_acknowledged: bool,
    ) -> tuple[dict[str, Any], ...]:
        """Read the exact linear inbox without memstore's permissive decoder.

        The typed layer decodes the log independently so its stricter wire
        schema, duplicate-key checks, and cursor proof stay local to this
        protocol.  Every relevant predecessor is validated and the seen ref
        must lie on the exact first-parent chain.
        """
        checked = _branch(recipient, "recipient")
        inbox_ref = f"{self.store.INBOX_REF}/{self.store.session}/{checked}"
        seen_ref = f"{self.store.SEEN_REF}/{self.store.session}/{checked}"
        head = self.store.backend.ref_sha(inbox_ref)
        seen = self.store.backend.ref_sha(seen_ref)
        if head is None:
            if seen is not None:
                raise CommunicationCorruption(
                    f"typed inbox {checked!r} has a seen marker but no inbox head"
                )
            return ()

        newest_first = self.store.backend.rev_list_first_parent(head)
        found_seen = seen is None
        selected: list[dict[str, Any]] = []
        for inbox_id in newest_first:
            if inbox_id == seen:
                found_seen = True
                if not include_acknowledged:
                    break
            parents = self.store.backend.parents_of(inbox_id)
            if len(parents) > 1:
                raise CommunicationCorruption(
                    f"inbox commit {inbox_id[:10]} has {len(parents)} parents; "
                    "the cursor log must be linear"
                )
            try:
                text = self.store.backend.repo.git.log("-1", "--format=%B", inbox_id)
            except GitCommandError as exc:
                raise CommunicationCorruption(
                    f"cannot read inbox commit {inbox_id[:10]}"
                ) from exc
            raw = _loads(text, f"inbox commit {inbox_id[:10]}")
            _check_fields(
                raw,
                where=f"inbox commit {inbox_id[:10]}",
                required={"sender", "at", "body"},
            )
            raw["id"] = inbox_id
            selected.append(raw)

        if not found_seen:
            raise CommunicationCorruption(
                f"seen marker {seen[:10] if seen else 'unknown'} is not on "
                f"the first-parent inbox chain for {checked!r}"
            )
        return tuple(reversed(selected))

    def _typed_history(self) -> tuple[InboxMessage, ...]:
        prefix = f"{self.store.INBOX_REF}/{self.store.session}/"
        items: list[InboxMessage] = []
        for ref, _head in sorted(self.store.backend.for_each_ref(prefix)):
            recipient = _branch(ref.removeprefix(prefix), "inbox ref recipient")
            for raw in self._raw_inbox(recipient, include_acknowledged=True):
                if not self._claims_typed_protocol(raw):
                    continue
                item = self._parse_envelope(raw, recipient)
                self._validate_policy(item.message)
                items.append(item)
        return tuple(items)

    def send(self, message: Message) -> InboxMessage:
        """Publish once, or recover the one exact entry published by a retry.

        The repo-wide memstore lock covers the history check and ``Store.send``
        together.  Consequently concurrent processes cannot both observe a
        missing idempotency key and append duplicates.
        """
        if not isinstance(message, Message):
            raise TypeError("message must be a Message")
        self._validate_policy(message)
        with _send_lock(self.store), self.store.backend.lock():
            matches = [
                item
                for item in self._typed_history()
                if (
                    item.message.message_id == message.message_id
                    or (
                        item.message.sender == message.sender
                        and item.message.idempotency_key == message.idempotency_key
                    )
                )
            ]
            conflicts = [item for item in matches if item.message.to_dict() != message.to_dict()]
            if conflicts:
                raise IdempotencyConflict(
                    f"{message.sender!r} reused idempotency key "
                    f"{message.idempotency_key!r} for different message bytes"
                )
            if len(matches) > 1:
                raise DuplicateDurableMessage(
                    f"message {message.message_id} has {len(matches)} durable inbox entries"
                )
            if matches:
                return matches[0]

            inbox_id = self.store.send(
                message.recipient,
                message.to_dict(),
                sender=message.sender,
            )
            written = [
                item
                for item in self._typed_history()
                if item.inbox_id == inbox_id and item.message.message_id == message.message_id
            ]
            if len(written) != 1 or written[0].message.to_dict() != message.to_dict():
                raise CommunicationCorruption(
                    "Store.send returned without one exact durable typed inbox entry"
                )
            return written[0]

    def pending(self, recipient: str) -> tuple[InboxMessage, ...]:
        """Return every unacknowledged typed message, oldest first.

        An untyped or malformed entry raises and remains at the cursor.  It is
        never skipped in order to expose a later valid message.
        """
        checked = _branch(recipient, "recipient")
        items: list[InboxMessage] = []
        for raw in self._raw_inbox(checked, include_acknowledged=False):
            if not self._claims_typed_protocol(raw):
                raise MessageWireError(
                    f"untyped message {str(raw.get('id', 'unknown'))[:10]} blocks "
                    f"typed inbox {checked!r}"
                )
            item = self._parse_envelope(raw, checked)
            self._validate_policy(item.message)
            items.append(item)
        return tuple(items)

    def history(self, recipient: str) -> tuple[InboxMessage, ...]:
        """Return typed history including acknowledged and stale entries."""
        checked = _branch(recipient, "recipient")
        items: list[InboxMessage] = []
        for raw in self._raw_inbox(checked, include_acknowledged=True):
            if not self._claims_typed_protocol(raw):
                raise MessageWireError(
                    f"untyped message {str(raw.get('id', 'unknown'))[:10]} exists in "
                    f"typed inbox history {checked!r}"
                )
            item = self._parse_envelope(raw, checked)
            self._validate_policy(item.message)
            items.append(item)
        return tuple(items)

    def _assert_oldest(self, item: InboxMessage) -> None:
        if not isinstance(item, InboxMessage):
            raise TypeError("item must be an InboxMessage")
        waiting = self._raw_inbox(
            item.message.recipient,
            include_acknowledged=False,
        )
        if not waiting:
            raise InboxOrderError(f"inbox entry {item.inbox_id[:10]} is no longer pending")
        raw = waiting[0]
        if not self._claims_typed_protocol(raw):
            raise MessageWireError(
                f"untyped message {str(raw.get('id', 'unknown'))[:10]} blocks the cursor"
            )
        oldest = self._parse_envelope(raw, item.message.recipient)
        self._validate_policy(oldest.message)
        if oldest.inbox_id != item.inbox_id or oldest.message.to_dict() != item.message.to_dict():
            raise InboxOrderError(
                f"inbox entry {item.inbox_id[:10]} is not the exact oldest pending message"
            )

    @staticmethod
    def _validate_receipt(
        receipt: AcceptanceReceipt,
        item: InboxMessage,
        disposition: str,
        current_generation: int,
    ) -> None:
        if not isinstance(receipt, AcceptanceReceipt):
            raise InvalidAcceptanceReceipt(
                "durable boundary must return an AcceptanceReceipt"
            )
        expected = AcceptanceReceipt.for_message(
            item,
            disposition=disposition,
            current_generation=current_generation,
            durable_ref=receipt.durable_ref,
        )
        if receipt != expected:
            raise InvalidAcceptanceReceipt(
                "durable acceptance receipt does not exactly bind this inbox message"
            )

    def _settle(
        self,
        item: InboxMessage,
        *,
        disposition: str,
        current_generation: int,
        boundary: AcceptanceBoundary,
    ) -> AcceptanceReceipt:
        self._assert_oldest(item)
        receipt = boundary(item, disposition, current_generation)
        self._validate_receipt(receipt, item, disposition, current_generation)
        # This exact, monotonic cursor movement is deliberately last.  If the
        # process dies anywhere above, the message is delivered again.
        self.store.mark_inbox_seen(item.message.recipient, item.inbox_id)
        return receipt

    def accept(
        self,
        item: InboxMessage,
        *,
        current_generation: int,
        boundary: AcceptanceBoundary,
    ) -> AcceptanceReceipt:
        """Accept the oldest item only when it belongs to the current plan."""
        status = self.generation_status(item.message, current_generation)
        if status is not GenerationStatus.CURRENT:
            raise GenerationFence(
                f"message generation {item.message.generation} is {status.value} "
                f"for current generation {current_generation}"
            )
        return self._settle(
            item,
            disposition="accepted",
            current_generation=current_generation,
            boundary=boundary,
        )

    def retire_stale(
        self,
        item: InboxMessage,
        *,
        current_generation: int,
        boundary: AcceptanceBoundary,
    ) -> AcceptanceReceipt:
        """Durably classify an old item as stale, then advance past it."""
        status = self.generation_status(item.message, current_generation)
        if status is not GenerationStatus.STALE:
            raise GenerationFence(
                f"only stale messages can be retired; message is {status.value}"
            )
        return self._settle(
            item,
            disposition="stale",
            current_generation=current_generation,
            boundary=boundary,
        )
