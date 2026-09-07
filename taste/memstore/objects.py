"""The object model of the memory layer.

A *state* is a checkpoint of an agent: the artifacts it holds, the brain's
context at that moment, and why the state exists. Everything here is an
immutable value; the store (``store.py``) is what gives them identity.

Types decide two things and nothing else: how an object merges, and how it is
found. A file merges three-way. A record merges by key. A transcript never
merges. A manifest is a union. That is the whole reason the layer is typed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

SCHEMA = "taste.memstore/1"


class ObjectType(StrEnum):
    FILE = "file"
    RECORD = "record"
    TRANSCRIPT = "transcript"
    VERDICT = "verdict"
    NOTE = "note"
    MANIFEST = "manifest"


class StaleBranch(RuntimeError):
    """The branch moved underneath us: our parent is no longer its head.

    Raised by the compare-and-swap that finishes every checkpoint. The commit
    and notes we built are left dangling and harmless; nothing was published.
    """


class BranchBusy(RuntimeError):
    """Another live process holds this branch's working tree.

    A branch is one brain's address space and is never shared. The lease is
    a kernel file lock, so a holder that dies releases it without cleanup;
    compare-and-swap on the ref covers the moment between its death and the
    next holder noticing.
    """


class NoSuchState(KeyError):
    """No state with that id is reachable in this store."""


class NotAnAncestor(ValueError):
    """A rollback target must be in the branch's own history."""


class PublishError(ValueError):
    """A published path must exist in the checkpoint that publishes it."""


# ------------------------------------------------------------------ verdict


@dataclass(frozen=True)
class Verdict:
    """A monitor's judgment of a state. Attached to the state it judged."""

    status: Literal["pass", "fail", "unknown"]
    by: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "by": self.by, "detail": self.detail}

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> Verdict | None:
        if not raw:
            return None
        return cls(status=raw["status"], by=raw.get("by", ""), detail=raw.get("detail", ""))


# ------------------------------------------------------------------ manifest


@dataclass(frozen=True)
class ManifestEntry:
    """One thing a branch has published: where it is and what kind it is."""

    name: str
    path: str
    type: ObjectType
    description: str = ""
    blob: str | None = None
    published_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "type": str(self.type),
            "description": self.description,
            "blob": self.blob,
            "published_at": self.published_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ManifestEntry:
        return cls(
            name=raw["name"],
            path=raw["path"],
            type=ObjectType(raw.get("type", "file")),
            description=raw.get("description", ""),
            blob=raw.get("blob"),
            published_at=raw.get("published_at"),
        )


@dataclass(frozen=True)
class Manifest:
    """What a branch has: the index the communicator queries.

    Immutable; every operation returns a new manifest. Union keeps the entry
    with the later ``published_at`` when both sides publish the same name.
    """

    entries: dict[str, ManifestEntry] = field(default_factory=dict)

    def with_entry(self, entry: ManifestEntry) -> Manifest:
        merged = dict(self.entries)
        merged[entry.name] = entry
        return Manifest(merged)

    def without(self, name: str) -> Manifest:
        merged = dict(self.entries)
        merged.pop(name, None)
        return Manifest(merged)

    def union(self, other: Manifest) -> Manifest:
        merged = dict(self.entries)
        for name, entry in other.entries.items():
            mine = merged.get(name)
            if mine is None or (entry.published_at or "") >= (mine.published_at or ""):
                merged[name] = entry
        return Manifest(merged)

    def type_of(self, path: str) -> ObjectType:
        for entry in self.entries.values():
            if entry.path == path:
                return entry.type
        return ObjectType.FILE

    def to_json(self) -> str:
        return json.dumps(
            {"schema": SCHEMA, "entries": {k: v.to_dict() for k, v in sorted(self.entries.items())}},
            indent=1,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str | None) -> Manifest:
        if not text:
            return cls()
        raw = json.loads(text)
        return cls({k: ManifestEntry.from_dict(v) for k, v in raw.get("entries", {}).items()})

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, name: str) -> bool:
        return name in self.entries


# ------------------------------------------------------------------ transcript


@dataclass(frozen=True)
class Transcript:
    """A brain's context at a state: an append-only sequence of turns.

    Stored as JSONL so a partial write is detectable line by line. Never
    merged: two brains' contexts are two transcripts.
    """

    turns: tuple[dict[str, Any], ...] = ()

    def append(self, **turn: Any) -> Transcript:
        return Transcript((*self.turns, dict(turn)))

    def extend(self, turns: list[dict[str, Any]]) -> Transcript:
        return Transcript(self.turns + tuple(dict(t) for t in turns))

    def to_jsonl(self) -> str:
        return "".join(json.dumps(t, sort_keys=True) + "\n" for t in self.turns)

    @classmethod
    def from_jsonl(cls, text: str | None) -> Transcript:
        if not text:
            return cls()
        turns = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                turns.append(json.loads(line))
        return cls(tuple(turns))

    def __len__(self) -> int:
        return len(self.turns)


# ------------------------------------------------------------------ meta


StateKind = Literal["root", "checkpoint", "rollback", "merge", "branch"]


@dataclass(frozen=True)
class Meta:
    """Why a state exists: its provenance, in one record.

    ``parents`` is the git parentage (one for a checkpoint, two for a merge).
    ``restores`` is set on a rollback: the state whose artifacts and
    transcript this state reinstated. ``rolled_back_from`` is the state that
    was superseded, which remains the first parent and therefore reachable
    forever. Nothing is ever lost; a rollback is an append.
    """

    branch: str
    kind: StateKind
    reason: str
    producer: str
    parents: tuple[str, ...]
    created_at: str
    attempt: int = 0
    verdict: Verdict | None = None
    restores: str | None = None
    rolled_back_from: str | None = None
    schema: str = SCHEMA

    def to_json(self) -> str:
        raw = {
            "schema": self.schema,
            "branch": self.branch,
            "kind": self.kind,
            "reason": self.reason,
            "producer": self.producer,
            "parents": list(self.parents),
            "created_at": self.created_at,
            "attempt": self.attempt,
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "restores": self.restores,
            "rolled_back_from": self.rolled_back_from,
        }
        return json.dumps(raw, indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Meta:
        raw = json.loads(text)
        return cls(
            branch=raw["branch"],
            kind=raw["kind"],
            reason=raw["reason"],
            producer=raw.get("producer", ""),
            parents=tuple(raw.get("parents", [])),
            created_at=raw["created_at"],
            attempt=int(raw.get("attempt", 0)),
            verdict=Verdict.from_dict(raw.get("verdict")),
            restores=raw.get("restores"),
            rolled_back_from=raw.get("rolled_back_from"),
            schema=raw.get("schema", SCHEMA),
        )

    def with_verdict(self, verdict: Verdict) -> Meta:
        return replace(self, verdict=verdict)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


# ------------------------------------------------------------------ conflicts


@dataclass(frozen=True)
class Conflict:
    """A merge the store could not resolve by type rule.

    It is a value a brain resolves later, not an exception. ``base``, ``ours``
    and ``theirs`` are blob ids (``None`` where the side lacks the path).
    """

    path: str
    type: ObjectType
    base: str | None
    ours: str | None
    theirs: str | None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "type": str(self.type),
            "base": self.base,
            "ours": self.ours,
            "theirs": self.theirs,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Conflict:
        return cls(
            path=raw["path"],
            type=ObjectType(raw.get("type", "file")),
            base=raw.get("base"),
            ours=raw.get("ours"),
            theirs=raw.get("theirs"),
            detail=raw.get("detail", ""),
        )
