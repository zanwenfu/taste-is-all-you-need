"""Append-only conversation evidence, in the private Responses call database.

Memory chooses the next prompt; this journal retains discarded branches too.
An event is recorded before memory publication, then marked after publication.
Neither a missing publication nor a missing tool result implies a safe replay.
The ResponsesSession supplies its existing exclusive database lease.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

ROOT = "0" * 64
MAX_EVENT_BYTES = 4 * 1024 * 1024
MAX_AUDIT_BYTES = 64 * 1024 * 1024
MAX_EVENTS = 100_000


class ResponsesAuditError(RuntimeError):
    pass


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":"))


def event_id(parent, event):
    return hashlib.sha256(encoded([parent, event]).encode()).hexdigest()


def prefix_id(events):
    parent = ROOT
    for event in events:
        parent = event_id(parent, event)
    return parent


def initialize(db):
    with db:
        db.execute("CREATE TABLE IF NOT EXISTS conversation_events ("
                   "id TEXT PRIMARY KEY, parent TEXT NOT NULL, event TEXT NOT NULL, "
                   "at TEXT NOT NULL, published INTEGER NOT NULL CHECK(published IN (0,1)))")


def record(db, parent, event):
    raw = encoded(event)
    if len(raw.encode()) > MAX_EVENT_BYTES:
        raise ResponsesAuditError("conversation audit event exceeds its byte limit")
    if not isinstance(event, dict) or not str(event.get("kind", "")).startswith("responses_"):
        raise ResponsesAuditError("conversation audit requires a Responses event")
    identifier = event_id(parent, event)
    previous = db.execute("SELECT parent,event FROM conversation_events WHERE id=?", (identifier,)).fetchone()
    if previous is not None:
        if previous != (parent, raw):
            raise ResponsesAuditError("conversation audit event identity changed")
        return identifier
    if parent != ROOT and db.execute("SELECT 1 FROM conversation_events WHERE id=?", (parent,)).fetchone() is None:
        raise ResponsesAuditError("conversation history predates its audit; start a new run")
    count, size = db.execute("SELECT count(*),coalesce(sum(length(event)),0) FROM conversation_events").fetchone()
    if count >= MAX_EVENTS or size + len(raw) > MAX_AUDIT_BYTES:
        raise ResponsesAuditError("conversation audit capacity exhausted")
    with db:
        db.execute("INSERT INTO conversation_events VALUES (?,?,?,?,0)",
                   (identifier, parent, raw, datetime.now(UTC).isoformat()))
    return identifier


def published(db, identifier):
    with db:
        updated = db.execute("UPDATE conversation_events SET published=1 WHERE id=?", (identifier,))
        if updated.rowcount != 1:
            raise ResponsesAuditError("published conversation event has no audit intent")


def reconcile(db, events):
    """Memory publication may have succeeded before its acknowledgement was lost."""
    parent = ROOT
    for event in events:
        identifier = event_id(parent, event)
        row = db.execute("SELECT parent,event,published FROM conversation_events WHERE id=?", (identifier,)).fetchone()
        if row is None or row[:2] != (parent, encoded(event)):
            raise ResponsesAuditError("conversation memory has no matching durable audit")
        if not row[2]:
            published(db, identifier)
        parent = identifier


def snapshot(db):
    """Detached evidence in occurrence order, including abandoned context forks."""
    rows, seen, size = [], set(), 0
    for identifier, parent, raw, at, delivered in db.execute(
            "SELECT id,parent,event,at,published FROM conversation_events ORDER BY rowid"):
        size += len(raw)
        if len(rows) >= MAX_EVENTS or size > MAX_AUDIT_BYTES or len(raw) > MAX_EVENT_BYTES:
            raise ResponsesAuditError("conversation audit exceeds its admitted limits")
        event = json.loads(raw)
        if (event_id(parent, event) != identifier or identifier in seen
                or (parent != ROOT and parent not in seen) or delivered not in (0, 1)):
            raise ResponsesAuditError("conversation audit history is malformed")
        seen.add(identifier)
        rows.append({"id": identifier, "parent": parent, "event": event,
                     "at": at, "published": bool(delivered)})
    return rows
