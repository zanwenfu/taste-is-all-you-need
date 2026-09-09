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
SCHEMA_FAMILY, _, SCHEMA_VERSION = SCHEMA.partition("/")


class ObjectType(StrEnum):
    FILE = "file"
    RECORD = "record"
    TRANSCRIPT = "transcript"
    VERDICT = "verdict"
    NOTE = "note"
    MANIFEST = "manifest"


class BranchBusy(RuntimeError):
    """Another live process holds this branch's working tree.

    A branch is one brain's address space and is never shared for writing.
    Reading needs no lease: see ``Store.view``.
    """


class StaleBranch(RuntimeError):
    """The branch moved underneath us: our parent is no longer its head.

    Raised by the compare-and-swap that finishes every checkpoint. The commit
    and notes we built are left dangling and harmless; nothing was published.
    """


class NoSuchState(KeyError):
    """No state with that id is reachable in this store."""


class NotAnAncestor(ValueError):
    """A rollback target must be in the branch's own history."""


class PublishError(ValueError):
    """A published path must exist in the checkpoint that publishes it."""


class BadName(ValueError):
    """A branch name that cannot be used verbatim.

    Names are rejected rather than mangled: the first version slugged, so
    ``worker/1`` and ``worker-1`` silently became one address space.
    """


class SchemaMismatch(RuntimeError):
    """A record was written by an incompatible version of this layer."""


def check_schema(raw: dict[str, Any], where: str) -> None:
    got = str(raw.get("schema", SCHEMA))
    family, _, version = got.partition("/")
    if family != SCHEMA_FAMILY:
        raise SchemaMismatch(f"{where}: foreign schema {got!r}")
    if version and version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        raise SchemaMismatch(f"{where}: schema {got!r} is not readable by {SCHEMA!r}")


# ------------------------------------------------------------------ verdict


@dataclass(frozen=True)
class Verdict:
    """A judgment of a state, by whoever judged it.

    ``failure_class`` carries the structure the harness already reasons about
    (``taste.recovery.FailureClass``) instead of flattening it into prose.
    """

    status: Literal["pass", "fail", "unknown"]
    by: str
    detail: str = ""
    failure_class: str | None = None
    at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "by": self.by,
            "detail": self.detail,
            "failure_class": self.failure_class,
            "at": self.at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> Verdict | None:
        if not raw:
            return None
        return cls(
            status=raw["status"],
            by=raw.get("by", ""),
            detail=raw.get("detail", ""),
            failure_class=raw.get("failure_class"),
            at=raw.get("at"),
        )


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
        check_schema(raw, "manifest")
        return cls({k: ManifestEntry.from_dict(v) for k, v in raw.get("entries", {}).items()})

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, name: str) -> bool:
        return name in self.entries


# ------------------------------------------------------------------ transcript


@dataclass(frozen=True)
class Transcript:
    """A brain's context: an append-only sequence of turns.

    Stored per state as the *delta* since the state it extends, so appending
    one turn writes one turn rather than rewriting the whole history. See
    ``Meta.transcript_from`` and ``State.transcript``.
    """

    turns: tuple[dict[str, Any], ...] = ()

    def append(self, **turn: Any) -> Transcript:
        return Transcript((*self.turns, dict(turn)))

    def extend(self, turns: list[dict[str, Any]]) -> Transcript:
        return Transcript(self.turns + tuple(dict(t) for t in turns))

    def extends(self, base: Transcript) -> bool:
        """True if this transcript is ``base`` plus zero or more new turns."""
        return len(self.turns) >= len(base.turns) and self.turns[: len(base.turns)] == base.turns

    def since(self, base: Transcript) -> Transcript:
        return Transcript(self.turns[len(base.turns) :])

    def slice(self, start: int | None = None, stop: int | None = None) -> Transcript:
        return Transcript(self.turns[start:stop])

    def to_jsonl(self) -> str:
        return "".join(json.dumps(t, sort_keys=True) + "\n" for t in self.turns)

    @classmethod
    def from_jsonl(cls, text: str | None) -> Transcript:
        if not text:
            return cls()
        turns = [json.loads(line) for line in text.splitlines() if line.strip()]
        return cls(tuple(turns))

    def __len__(self) -> int:
        return len(self.turns)

    def __add__(self, other: Transcript) -> Transcript:
        return Transcript(self.turns + other.turns)


# ------------------------------------------------------------------ meta


StateKind = Literal["root", "checkpoint", "rollback", "merge", "branch", "conflict"]


@dataclass(frozen=True)
class Source:
    """Where an adopted artifact came from, so provenance crosses branches."""

    branch: str
    state: str
    path: str
    as_path: str

    def to_dict(self) -> dict[str, Any]:
        return {"branch": self.branch, "state": self.state, "path": self.path, "as_path": self.as_path}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Source:
        return cls(branch=raw["branch"], state=raw["state"], path=raw["path"], as_path=raw["as_path"])


@dataclass(frozen=True)
class Meta:
    """Why a state exists: its provenance, in one record.

    ``parents`` is the git parentage (one for a checkpoint, two for a merge).
    ``restores`` is set on a rollback: the state whose artifacts and
    transcript this state reinstated. ``rolled_back_from`` is the state that
    was superseded, which remains the first parent and therefore reachable
    forever. Nothing is ever lost; a rollback is an append.

    ``transcript_from`` names the state this one's transcript extends, so the
    note holds only the new turns. ``session`` makes a state's identity
    unique to its session, which stops two sessions minting the same commit.
    """

    branch: str
    kind: StateKind
    reason: str
    producer: str
    parents: tuple[str, ...]
    created_at: str
    session: str = ""
    attempt: int = 0
    verdict: Verdict | None = None
    restores: str | None = None
    rolled_back_from: str | None = None
    transcript_from: str | None = None
    merged: str | None = None
    sources: tuple[Source, ...] = ()
    schema: str = SCHEMA

    def to_json(self) -> str:
        raw = {
            "schema": self.schema,
            "branch": self.branch,
            "session": self.session,
            "kind": self.kind,
            "reason": self.reason,
            "producer": self.producer,
            "parents": list(self.parents),
            "created_at": self.created_at,
            "attempt": self.attempt,
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "restores": self.restores,
            "rolled_back_from": self.rolled_back_from,
            "transcript_from": self.transcript_from,
            "merged": self.merged,
            "sources": [s.to_dict() for s in self.sources],
        }
        return json.dumps(raw, indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Meta:
        raw = json.loads(text)
        check_schema(raw, "meta")
        return cls(
            branch=raw["branch"],
            kind=raw["kind"],
            reason=raw["reason"],
            producer=raw.get("producer", ""),
            parents=tuple(raw.get("parents", [])),
            created_at=raw["created_at"],
            session=raw.get("session", ""),
            attempt=int(raw.get("attempt", 0)),
            verdict=Verdict.from_dict(raw.get("verdict")),
            restores=raw.get("restores"),
            rolled_back_from=raw.get("rolled_back_from"),
            transcript_from=raw.get("transcript_from"),
            merged=raw.get("merged"),
            sources=tuple(Source.from_dict(s) for s in raw.get("sources", [])),
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
    and ``theirs`` are blob ids (``None`` where the side lacks the path). The
    ``theirs_*`` fields say what was being merged, which is what an
    orchestrator needs in order to act on a conflict it finds later.
    """

    path: str
    type: ObjectType
    base: str | None
    ours: str | None
    theirs: str | None
    detail: str = ""
    theirs_branch: str = ""
    theirs_state: str = ""
    ours_state: str = ""
    at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "type": str(self.type),
            "base": self.base,
            "ours": self.ours,
            "theirs": self.theirs,
            "detail": self.detail,
            "theirs_branch": self.theirs_branch,
            "theirs_state": self.theirs_state,
            "ours_state": self.ours_state,
            "at": self.at,
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
            theirs_branch=raw.get("theirs_branch", ""),
            theirs_state=raw.get("theirs_state", ""),
            ours_state=raw.get("ours_state", ""),
            at=raw.get("at", ""),
        )


# ------------------------------------------------------------------ resume


@dataclass(frozen=True)
class Resume:
    """Everything a brain needs on waking, in one call.

    ``intent`` is what the branch said it was about to do; it survives a
    crash precisely because it is written before the work, not after.
    """

    head_id: str
    last_reason: str
    dirty_paths: tuple[str, ...]
    intent: str | None
    open_conflicts: tuple[Conflict, ...]
    inbox: tuple[dict[str, Any], ...]
    recovered_turns: tuple[dict[str, Any], ...] = ()
    verdicts: tuple[Verdict, ...] = ()
    unacked: tuple[Verdict, ...] = ()
    """Verdicts the brain has not yet acknowledged, newest state first.

    A monitor is a separate process, so its judgment arrives as a note on the
    state rather than as a return value. Without this, a brain woke to a clean
    resume and carried on building on a state the monitor had already failed;
    and because verdicts are keyed by state, one more checkpoint hid the
    failure behind an ancestor.
    """

    @property
    def crashed(self) -> bool:
        """True if work was in flight when the process ended."""
        return bool(self.dirty_paths) or self.intent is not None or bool(self.recovered_turns)

    @property
    def failed(self) -> bool:
        """True if an unacknowledged verdict says this branch went wrong."""
        return any(v.status == "fail" for v in self.unacked)
