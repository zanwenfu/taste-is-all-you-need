"""The memory layer's API: stores, branches, states.

No git in the signatures. A ``Store`` is a session's memory; a ``Branch`` is
one brain's execution context with its own working tree; a ``State`` is a
checkpoint of that brain: artifacts, transcript, and the reason it exists.

Four rules hold everywhere in this file:

1. **Nothing is lost.** A rollback is an append: the superseded state stays
   the first parent of the rollback state and is reachable forever, with its
   transcript. Every operation that would discard the working tree
   (rollback, merge, dropping a worktree) checkpoints it first.
2. **A checkpoint is atomic.** Build the commit, attach its notes, then move
   the branch ref with compare-and-swap. A crash before the move leaves the
   store exactly as it was; a crash after it leaves a complete state. A lost
   compare raises ``StaleBranch`` and publishes nothing.
3. **The store is the truth.** Every answer here comes from refs and notes.
   There is no sidecar file to fall out of sync.
4. **Reading never takes the write lease.** A branch is one brain's address
   space for *writing*; a monitor, an orchestrator or a dashboard reads it
   through a ``BranchView`` that holds nothing.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import socket
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from taste.memstore.backend import EMPTY_TREE, GitBackend
from taste.memstore.objects import (
    BadName,
    BranchBusy,
    Conflict,
    Manifest,
    ManifestEntry,
    Meta,
    NoSuchState,
    NotAnAncestor,
    ObjectType,
    PublishError,
    Resume,
    Source,
    StaleBranch,
    StateKind,
    Transcript,
    Verdict,
    now_iso,
)

NOTES = {
    "meta": "refs/notes/taste/meta",
    "manifest": "refs/notes/taste/manifest",
    "transcript": "refs/notes/taste/transcript",
    "conflicts": "refs/notes/taste/conflicts",
    "verdicts": "refs/notes/taste/verdicts",
}

#: Write a full transcript instead of a delta every this many states, so
#: reconstructing one is a bounded walk rather than a walk of the whole run.
TRANSCRIPT_SNAPSHOT_EVERY = 64

_NAME_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _check_name(text: str, what: str = "name") -> str:
    """Names are validated, never mangled.

    The first version slugged, so ``worker/1`` and ``worker-1`` collapsed to
    one address space with no error. A function that maps address spaces must
    be injective; the cheapest injective function is the identity plus a
    rejection.
    """
    if not _NAME_RE.match(text) or text in {".", ".."} or text.endswith(".lock"):
        raise BadName(
            f"{what} {text!r} is not usable: use letters, digits, dot, dash or "
            "underscore, starting with a letter or digit"
        )
    return text


# ------------------------------------------------------------------ diffs


@dataclass(frozen=True)
class DiffEntry:
    path: str
    type: ObjectType
    status: str
    detail: str = ""


@dataclass(frozen=True)
class TypedDiff:
    entries: tuple[DiffEntry, ...]

    def __len__(self) -> int:
        return len(self.entries)

    def paths(self) -> list[str]:
        return [e.path for e in self.entries]


@dataclass(frozen=True)
class MergeResult:
    state: State | None
    conflicts: tuple[Conflict, ...] = ()

    @property
    def ok(self) -> bool:
        return self.state is not None and not self.conflicts


@dataclass(frozen=True)
class Hit:
    branch: str
    state: State
    entry: ManifestEntry
    score: int


@dataclass(frozen=True)
class Built:
    """A state that exists as objects but is not published.

    Handed out by ``Branch.build`` so the window between building and
    publishing has a public form. It is invisible to every reader until
    ``Branch.publish_state`` moves the ref.
    """

    sha: str
    expected_head: str | None


# ------------------------------------------------------------------ state


class State:
    """One checkpoint. Identity is the commit id; everything else is lazy."""

    __slots__ = ("_manifest", "_meta", "_store", "_transcript", "id")

    def __init__(self, store: Store, sha: str) -> None:
        self.id = sha
        self._store = store
        self._meta: Meta | None = None
        self._manifest: Manifest | None = None
        self._transcript: Transcript | None = None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, State) and other.id == self.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f"State({self.id[:10]}, {self.meta.kind}, {self.meta.reason!r})"

    @property
    def meta(self) -> Meta:
        if self._meta is None:
            raw = self._store.backend.note_get(NOTES["meta"], self.id)
            if raw is None:
                raise NoSuchState(self.id)
            self._meta = Meta.from_json(raw)
        return self._meta

    @property
    def manifest(self) -> Manifest:
        if self._manifest is None:
            self._manifest = Manifest.from_json(
                self._store.backend.note_get(NOTES["manifest"], self.id)
            )
        return self._manifest

    @property
    def transcript(self) -> Transcript:
        """The brain's full context here, reassembled from deltas.

        Each state's note holds only the turns it added; ``transcript_from``
        names the state it extends. A full snapshot is written periodically
        so this walk is bounded.
        """
        if self._transcript is None:
            chain: list[Transcript] = []
            cur: State | None = self
            while cur is not None:
                chain.append(Transcript.from_jsonl(
                    self._store.backend.note_get(NOTES["transcript"], cur.id)
                ))
                parent = cur.meta.transcript_from
                cur = self._store.state(parent) if parent else None
            out = Transcript()
            for part in reversed(chain):
                out = out + part
            self._transcript = out
        return self._transcript

    def turns(self, start: int | None = None, stop: int | None = None) -> Transcript:
        """A slice of the context, for a brain that wants only its last turns."""
        return self.transcript.slice(start, stop)

    @property
    def conflicts(self) -> list[Conflict]:
        raw = self._store.backend.note_get(NOTES["conflicts"], self.id)
        return [Conflict.from_dict(c) for c in json.loads(raw)] if raw else []

    @property
    def verdicts(self) -> list[Verdict]:
        """Every judgment recorded on this state, by any process."""
        raw = self._store.backend.note_get(NOTES["verdicts"], self.id)
        recorded = [Verdict.from_dict(v) for v in json.loads(raw)] if raw else []
        own = self.meta.verdict
        return ([own] if own else []) + [v for v in recorded if v]

    @property
    def parents(self) -> list[State]:
        return [self._store.state(p) for p in self.meta.parents]

    @property
    def branch(self) -> str:
        return self.meta.branch

    def read(self, path: str) -> str | None:
        """Demand-page one artifact from this state."""
        return self._store.backend.show(self.id, path)

    def read_bytes(self, path: str) -> bytes | None:
        return self._store.backend.show_bytes(self.id, path)

    def blob(self, path: str) -> str | None:
        return self._store.backend.blob_at(self.id, path)

    def files(self) -> list[str]:
        return self._store.backend.ls_files(self.id)

    def record(self, path: str) -> Any:
        text = self.read(path)
        return None if text is None else json.loads(text)

    def diff(self, other: State) -> TypedDiff:
        """What changed from ``other`` to this state, by type."""
        entries: list[DiffEntry] = []
        for status, path in self._store.backend.diff_names(other.id, self.id):
            kind = (
                self.manifest.type_of(path)
                if path in self.manifest_paths()
                else other.manifest.type_of(path)
            )
            detail = ""
            if kind is ObjectType.RECORD and status == "M":
                a, b = other.record(path), self.record(path)
                if isinstance(a, dict) and isinstance(b, dict):
                    added = sorted(set(b) - set(a))
                    removed = sorted(set(a) - set(b))
                    changed = sorted(k for k in set(a) & set(b) if a[k] != b[k])
                    detail = json.dumps({"added": added, "removed": removed, "changed": changed})
            entries.append(DiffEntry(path=path, type=kind, status=status, detail=detail))
        return TypedDiff(tuple(entries))

    def manifest_paths(self) -> set[str]:
        return {e.path for e in self.manifest.entries.values()}


# ------------------------------------------------------------------ view


class BranchView:
    """A read-only window on a branch. Holds no lease and no working tree.

    This is what a monitor, an orchestrator or a dashboard uses. It answers
    from refs and notes only, so any number of them can watch a branch that
    a brain is actively writing.
    """

    def __init__(self, store: Store, name: str) -> None:
        self.store = store
        self.name = name
        self.ref = store.ref_for(name)

    def __repr__(self) -> str:
        return f"BranchView({self.name!r})"

    @property
    def head(self) -> State:
        sha = self.store.backend.ref_sha(self.ref)
        if sha is None:
            raise NoSuchState(f"branch {self.name} has no head")
        return self.store.state(sha)

    def exists(self) -> bool:
        return self.store.backend.ref_sha(self.ref) is not None

    def history(self, limit: int | None = None) -> list[State]:
        """This branch's own states, newest first, stopping where it began."""
        out: list[State] = []
        for sha in self.store.backend.rev_list_first_parent(self.head.id):
            st = self.store.state(sha)
            if st.meta.branch != self.name:
                break
            out.append(st)
            if limit is not None and len(out) >= limit:
                break
        return out

    def read(self, path: str, at: State | None = None) -> str | None:
        return (at or self.head).read(path)

    def manifest(self) -> Manifest:
        return self.head.manifest

    def conflicts(self) -> list[Conflict]:
        return self.head.conflicts

    def inbox(self, since: str | None = None) -> list[dict[str, Any]]:
        return self.store.inbox(self.name, since=since)


# ------------------------------------------------------------------ branch


class Branch:
    """One brain's execution context: a branch, its working tree, its lease."""

    def __init__(self, store: Store, name: str, *, producer: str = "") -> None:
        self.store = store
        self.name = name
        self.producer = producer or name
        self.ref = store.ref_for(name)
        self.short_ref = store.short_ref_for(name)
        self.worktree = store.worktree_path_for(name)
        self.backend = GitBackend(self.worktree)
        self._assert_same_repo()
        self._lease: IO[str] | None = self._acquire_lease()
        self._pending: dict[str, ManifestEntry] = {}
        self._unpublish: set[str] = set()
        self._pending_turns: list[dict[str, Any]] = []
        self._pending_sources: list[Source] = []

    def __repr__(self) -> str:
        return f"Branch({self.name!r})"

    def close(self) -> None:
        self.release()
        self.backend.close()

    def release(self) -> None:
        """Give up the write lease, keeping the object readable.

        This is the handoff the vision needs: a brain that is done with a
        branch releases it and another brain can take it, without either of
        them going through a restart. A holder that dies instead of
        releasing frees the lease anyway, because the lock is a kernel file
        lock rather than a record in the store.
        """
        if self._lease is not None:
            fcntl.flock(self._lease.fileno(), fcntl.LOCK_UN)
            self._lease.close()
            self._lease = None
            self.store._branches.pop(self.name, None)

    @property
    def holder(self) -> dict[str, Any] | None:
        """Who holds this branch, from the lease file, or None if it is free."""
        try:
            raw = self._lease_path().read_text()
        except OSError:
            return None
        try:
            return dict(json.loads(raw))
        except json.JSONDecodeError:
            return None

    # ---------------------------------------------------------- lease

    def _lease_path(self) -> Path:
        return self.store.backend.common_dir / f"memstore.lease.{self.store.session}.{self.name}"

    def _assert_same_repo(self) -> None:
        """A worktree must belong to this repository, not merely sit at that path."""
        if self.backend.common_dir != self.store.backend.common_dir:
            raise BadName(
                f"worktree at {self.worktree} belongs to {self.backend.common_dir}, "
                f"not to {self.store.backend.common_dir}"
            )

    def _acquire_lease(self) -> IO[str]:
        path = self._lease_path()
        fh = open(path, "a+")  # noqa: SIM115  held for the life of the Branch
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            held = ""
            with contextlib.suppress(OSError):
                held = f" (held by {path.read_text().strip()})"
            fh.close()
            raise BranchBusy(f"branch {self.name!r} is held by another live process{held}") from None
        fh.seek(0)
        fh.truncate()
        fh.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "producer": self.producer,
                    "opened_at": now_iso(),
                }
            )
        )
        fh.flush()
        return fh

    # ---------------------------------------------------------- reading

    @property
    def view(self) -> BranchView:
        return BranchView(self.store, self.name)

    @property
    def head(self) -> State:
        return self.view.head

    def history(self, limit: int | None = None) -> list[State]:
        return self.view.history(limit)

    def read(self, path: str, at: State | None = None) -> str | None:
        return (at or self.head).read(path)

    def path(self, rel: str) -> Path:
        return self.worktree / rel

    def write(self, rel: str, content: str | bytes) -> None:
        p = self.worktree / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content)

    def dirty_paths(self) -> list[str]:
        """What differs from the last state, without staging anything."""
        return self.backend.dirty_paths()

    def is_dirty(self) -> bool:
        return bool(self.backend.dirty_paths())

    # ---------------------------------------------------------- intent and resume

    def _intent_path(self) -> Path:
        return self.store.backend.common_dir / f"memstore.intent.{self.store.session}.{self.name}"

    def intend(self, reason: str) -> None:
        """Record what this brain is about to attempt.

        Written before the work, so a crash leaves it behind. ``resume``
        reports it, and the next checkpoint clears it. Without this, the one
        thing a failed turn loses is what it was trying to do.
        """
        self._intent_path().write_text(reason)

    def resume(self) -> Resume:
        """Everything a brain needs on waking, in one call."""
        head = self.head
        intent_path = self._intent_path()
        intent = intent_path.read_text() if intent_path.exists() else None
        return Resume(
            head_id=head.id,
            last_reason=head.meta.reason,
            dirty_paths=tuple(self.dirty_paths()),
            intent=intent,
            open_conflicts=tuple(head.conflicts),
            inbox=tuple(self.store.inbox(self.name)),
        )

    # ---------------------------------------------------------- publishing

    def publish(
        self,
        name: str,
        path: str,
        *,
        type: ObjectType = ObjectType.FILE,
        description: str = "",
    ) -> None:
        """Make an artifact discoverable. Takes effect at the next checkpoint."""
        self._unpublish.discard(name)
        self._pending[name] = ManifestEntry(name=name, path=path, type=type, description=description)

    def unpublish(self, name: str) -> None:
        self._pending.pop(name, None)
        self._unpublish.add(name)

    def adopt(self, source: State | Hit, path: str | None = None, *, as_: str | None = None) -> str:
        """Take another branch's artifact, recording where it came from.

        Copies the bytes into this working tree and records the source in the
        next state's provenance, so ``store.origin`` can cross branches. This
        is the read half of the communicator: A finds B's artifact and adopts
        it rather than re-deriving it.
        """
        if isinstance(source, Hit):
            state, path = source.state, path or source.entry.path
        else:
            state = source
        if path is None:
            raise ValueError("adopt needs a path when given a state")
        raw = state.read_bytes(path)
        if raw is None:
            raise PublishError(f"{path} is not in state {state.id[:10]}")
        target = as_ or path
        self.write(target, raw)
        self._pending_sources.append(
            Source(branch=state.meta.branch, state=state.id, path=path, as_path=target)
        )
        return target

    # ---------------------------------------------------------- context

    def turn(self, **turn: Any) -> None:
        """Append one turn to the brain's context, flushed at the next state."""
        self._pending_turns.append(dict(turn))

    # ---------------------------------------------------------- writing states

    def checkpoint(
        self,
        reason: str,
        *,
        transcript: Transcript | None = None,
        records: dict[str, Any] | None = None,
        verdict: Verdict | None = None,
        attempt: int = 0,
    ) -> State:
        """Commit the working tree, the brain's context, and the reason, atomically.

        ``records`` writes JSON files into the working tree before staging.
        A transcript of ``None`` carries the previous state's forward plus any
        turns appended with :meth:`turn`.
        """
        built = self.build(
            reason,
            transcript=transcript,
            records=records,
            verdict=verdict,
            attempt=attempt,
        )
        return self.publish_state(built)

    def build(
        self,
        reason: str,
        *,
        transcript: Transcript | None = None,
        records: dict[str, Any] | None = None,
        verdict: Verdict | None = None,
        attempt: int = 0,
        kind: StateKind = "checkpoint",
    ) -> Built:
        """Create the objects for a state without publishing it.

        Public so the window this layer's atomicity claim is about can be
        exercised and observed. Nothing sees the result until
        :meth:`publish_state`.
        """
        for rel, value in (records or {}).items():
            self.write(rel, json.dumps(value, indent=1, sort_keys=True) + "\n")
        head = self.head
        self.backend.stage_all()
        tree = self.backend.write_tree()
        manifest = head.manifest
        for name in self._unpublish:
            manifest = manifest.without(name)
        return self._build(
            tree=tree,
            parents=[head.id],
            kind=kind,
            reason=reason,
            manifest=manifest,
            transcript=transcript,
            transcript_base=head,
            verdict=verdict,
            attempt=attempt,
            expected_head=head.id,
        )

    def rollback(
        self,
        to: State,
        reason: str,
        *,
        transcript: Transcript | None = None,
    ) -> State:
        """Reinstate ``to`` as a new state. The superseded state stays reachable.

        If the working tree holds uncommitted work, it is checkpointed first
        so that it, too, survives. The new state's transcript is ``to``'s
        unless one is given, so the brain resumes with the context it had at
        the state it is returning to; the failed context is one parent away.
        """
        self._capture("rollback", reason)
        head = self.head
        if not self.backend.is_ancestor(to.id, head.id):
            raise NotAnAncestor(f"{to.id[:10]} is not in the history of {self.name}")
        self._pending.clear()
        self._unpublish.clear()
        built = self._build(
            tree=self.backend.tree_of(to.id),
            parents=[head.id],
            kind="rollback",
            reason=reason,
            manifest=to.manifest,
            transcript=transcript,
            transcript_base=to,
            verdict=None,
            attempt=0,
            expected_head=head.id,
            restores=to.id,
            rolled_back_from=head.id,
        )
        state = self.publish_state(built)
        self.backend.reset_hard_to_head()
        return state

    def merge(self, other: Branch | BranchView, *, reason: str, resolved: bool = False) -> MergeResult:
        """Merge ``other`` into this branch by type rule; see ``merge.py``.

        If the result carries conflicts, nothing was published. Resolve them
        by writing the files you want into this working tree and calling
        again with ``resolved=True``.
        """
        from taste.memstore.merge import merge_branches

        return merge_branches(self, other, reason=reason, resolved=resolved)

    def _capture(self, op: str, reason: str) -> None:
        """Checkpoint uncommitted work before an operation that would discard it.

        Applied to every destructive path, not just rollback: the first
        version protected rollback alone, so merge and worktree removal threw
        away work the layer promised to keep.
        """
        if self.is_dirty():
            self.checkpoint(f"capture before {op}: {reason}")

    # ---------------------------------------------------------- internals

    def _build(
        self,
        *,
        tree: str,
        parents: list[str],
        kind: StateKind,
        reason: str,
        manifest: Manifest,
        transcript: Transcript | None,
        transcript_base: State | None,
        verdict: Verdict | None,
        attempt: int,
        expected_head: str | None,
        restores: str | None = None,
        rolled_back_from: str | None = None,
        merged: str | None = None,
        conflicts: list[Conflict] | None = None,
    ) -> Built:
        """Create commit and notes. Nothing is visible until the ref moves."""
        now = now_iso()
        for name, entry in self._pending.items():
            blob = self.backend.blob_at(tree, entry.path)
            if blob is None:
                raise PublishError(f"{entry.path} is not in the checkpoint that publishes {name!r}")
            manifest = manifest.with_entry(
                ManifestEntry(
                    name=entry.name,
                    path=entry.path,
                    type=entry.type,
                    description=entry.description,
                    blob=blob,
                    published_at=now,
                )
            )

        base_full = transcript_base.transcript if transcript_base is not None else Transcript()
        wanted = transcript if transcript is not None else base_full
        if self._pending_turns:
            wanted = wanted.extend(self._pending_turns)
        depth = self._transcript_depth(transcript_base)
        if transcript_base is not None and wanted.extends(base_full) and depth < TRANSCRIPT_SNAPSHOT_EVERY:
            delta, transcript_from = wanted.since(base_full), transcript_base.id
        else:
            delta, transcript_from = wanted, None

        meta = Meta(
            branch=self.name,
            kind=kind,
            reason=reason,
            producer=self.producer,
            parents=tuple(parents),
            created_at=now,
            session=self.store.session,
            attempt=attempt,
            verdict=verdict,
            restores=restores,
            rolled_back_from=rolled_back_from,
            transcript_from=transcript_from,
            merged=merged,
            sources=tuple(self._pending_sources),
        )
        # The message carries session and timestamp so no two states can ever
        # share a commit id: git derives the id from tree, parents, message and
        # a one-second timestamp, and two sessions once minted the same id.
        message = (
            f"{reason}\n\n"
            f"Taste-State: {kind}\n"
            f"Taste-Branch: {self.name}\n"
            f"Taste-Session: {self.store.session}\n"
            f"Taste-Created: {now}\n"
        )
        sha = self.backend.commit_tree(tree, parents, message)
        # One lock acquisition for all of a state's notes: each ``git notes``
        # call is a read-modify-write of one tree under a repo-wide lock, and
        # taking it three or four times put most of a checkpoint inside a
        # mutex that every other branch also waits on.
        with self.backend.lock():
            self.backend.note_set(NOTES["meta"], sha, meta.to_json(), overwrite=False)
            self.backend.note_set(NOTES["manifest"], sha, manifest.to_json(), overwrite=False)
            self.backend.note_set(NOTES["transcript"], sha, delta.to_jsonl(), overwrite=False)
            if conflicts:
                self.backend.note_set(
                    NOTES["conflicts"], sha,
                    json.dumps([c.to_dict() for c in conflicts]), overwrite=False,
                )
        return Built(sha=sha, expected_head=expected_head)

    def _transcript_depth(self, base: State | None) -> int:
        depth = 0
        cur = base
        while cur is not None and cur.meta.transcript_from and depth < TRANSCRIPT_SNAPSHOT_EVERY:
            depth += 1
            cur = self.store.state(cur.meta.transcript_from)
        return depth

    def publish_state(self, built: Built) -> State:
        """Move the branch ref to a built state, or fail having published nothing."""
        if not self.backend.cas_update_ref(self.ref, built.sha, built.expected_head):
            raise StaleBranch(
                f"{self.name} moved: expected head {(built.expected_head or 'none')[:10]}; "
                "nothing published"
            )
        self._pending.clear()
        self._unpublish.clear()
        self._pending_turns.clear()
        self._pending_sources.clear()
        with contextlib.suppress(FileNotFoundError):
            self._intent_path().unlink()
        return self.store.state(built.sha)


# ------------------------------------------------------------------ store


class Store:
    """A session's memory. Opens or creates the repository under ``root``."""

    REF_ROOT = "refs/heads/mem"
    ROOT_REF = "refs/taste/memstore"
    INBOX_REF = "refs/taste/inbox"
    SEEN_REF = "refs/taste/inbox-seen"

    def __init__(self, root: Path, session: str) -> None:
        self.root = Path(root).resolve()
        self.session = _check_name(session, "session")
        self.backend = GitBackend.init(self.root)
        self._branches: dict[str, Branch] = {}

    @classmethod
    def open(cls, root: Path, session: str) -> Store:
        return cls(root, session)

    def close(self) -> None:
        for b in list(self._branches.values()):
            b.close()
        self._branches.clear()
        self.backend.close()

    # ---------------------------------------------------------- naming

    def short_ref_for(self, name: str) -> str:
        return f"mem/{self.session}/{_check_name(name, 'branch')}"

    def ref_for(self, name: str) -> str:
        return f"refs/heads/{self.short_ref_for(name)}"

    def worktree_path_for(self, name: str) -> Path:
        """Keyed by repository identity, not by the parent directory.

        Two sibling repositories once resolved to the same worktree path, so
        one session committed into the other's object store.
        """
        return (
            self.root.parent
            / ".taste-worktrees"
            / self.backend.identity
            / self.session
            / _check_name(name, "branch")
        )

    # ---------------------------------------------------------- session root

    def _session_root(self) -> str:
        """The commit every branch of this session descends from."""
        ref = f"{self.ROOT_REF}/{self.session}/root"
        sha = self.backend.ref_sha(ref)
        if sha:
            return sha
        base = self.backend.head_commit()
        if base is None:
            base = self.backend.commit_tree(EMPTY_TREE, [], f"memstore root: {self.session}")
        with self.backend.lock():
            sha = self.backend.ref_sha(ref)
            if sha:
                return sha
            if self.backend.note_get(NOTES["meta"], base) is None:
                meta = Meta(
                    branch="",
                    kind="root",
                    reason=f"session root: {self.session}",
                    producer="",
                    parents=(),
                    created_at=now_iso(),
                    session=self.session,
                )
                self.backend.note_set(NOTES["meta"], base, meta.to_json())
                self.backend.note_set(NOTES["manifest"], base, Manifest().to_json())
                self.backend.note_set(NOTES["transcript"], base, "")
            self.backend.cas_update_ref(ref, base, None)
        return base

    # ---------------------------------------------------------- branches

    def branch(self, name: str, *, from_state: State | None = None, producer: str = "") -> Branch:
        """Open ``name`` for writing, creating it if it does not exist."""
        name = _check_name(name, "branch")
        cached = self._branches.get(name)
        if cached is not None and cached._lease is not None:
            return cached
        ref = self.ref_for(name)
        if self.backend.ref_sha(ref) is None:
            base = from_state.id if from_state is not None else self._session_root()
            if not self.backend.cas_update_ref(ref, base, None):
                raise StaleBranch(f"branch {name} was created concurrently")
            self._seed(name, ref, base, from_state, producer)
        wt = self.worktree_path_for(name)
        if not wt.exists():
            self.backend.worktree_add(wt, self.short_ref_for(name))
        b = Branch(self, name, producer=producer)
        self._branches[name] = b
        return b

    def view(self, name: str) -> BranchView:
        """A lease-free read-only window: what a monitor or dashboard uses."""
        return BranchView(self, _check_name(name, "branch"))

    def _seed(self, name: str, ref: str, base: str, from_state: State | None, producer: str) -> None:
        tree = self.backend.tree_of(base)
        now = now_iso()
        meta = Meta(
            branch=name,
            kind="branch",
            reason=f"branch {name} from {(from_state.id[:10] if from_state else 'session root')}",
            producer=producer or name,
            parents=(base,),
            created_at=now,
            session=self.session,
            transcript_from=from_state.id if from_state else None,
        )
        manifest = from_state.manifest if from_state else Manifest()
        message = (
            f"{meta.reason}\n\nTaste-State: branch\nTaste-Branch: {name}\n"
            f"Taste-Session: {self.session}\nTaste-Created: {now}\n"
        )
        sha = self.backend.commit_tree(tree, [base], message)
        self.backend.note_set(NOTES["meta"], sha, meta.to_json(), overwrite=False)
        self.backend.note_set(NOTES["manifest"], sha, manifest.to_json(), overwrite=False)
        self.backend.note_set(NOTES["transcript"], sha, "", overwrite=False)
        if not self.backend.cas_update_ref(ref, sha, base):
            raise StaleBranch(f"branch {name} moved while being seeded")

    def branches(self) -> list[str]:
        prefix = f"{self.REF_ROOT}/{self.session}/"
        return sorted(name[len(prefix) :] for name, _ in self.backend.for_each_ref(prefix))

    def get(self, name: str) -> BranchView:
        """The read-only window. Use :meth:`branch` to write."""
        return self.view(name)

    def heads(self) -> dict[str, State]:
        """Every branch's head, without opening or leasing any of them."""
        return {name: self.view(name).head for name in self.branches()}

    def remove_branch(self, name: str) -> None:
        """Drop the working tree. The branch and its states stay."""
        name = _check_name(name, "branch")
        b = self._branches.pop(name, None)
        if b is not None:
            b._capture("worktree removal", name)
            b.close()
        self.backend.worktree_remove(self.worktree_path_for(name))

    # ---------------------------------------------------------- housekeeping

    def gc(self) -> None:
        """Drop notes whose commit no ref reaches.

        A checkpoint that loses its compare-and-swap, or a state that is
        built and never published, leaves a commit and its notes behind. No
        ref reaches them, so nothing can read them and no answer changes;
        they are simply still on disk. This is the explicit way to collect
        them. Nothing calls it automatically because pruning is not free and
        a store is allowed to be untidy.
        """
        self.backend.prune_unreachable()
        for ns in NOTES.values():
            self.backend.note_prune(ns)

    # ---------------------------------------------------------- states

    def state(self, sha: str) -> State:
        return State(self, sha)

    def judge(self, state: State, verdict: Verdict) -> None:
        """Record a judgment of ``state``. Callable without the branch lease.

        A monitor is a separate process from the brain it judges, so its
        verdict cannot go through the branch's writer. Verdicts accumulate;
        recording one never rewrites another.
        """
        stamped = verdict if verdict.at else Verdict(
            status=verdict.status,
            by=verdict.by,
            detail=verdict.detail,
            failure_class=verdict.failure_class,
            at=now_iso(),
        )
        with self.backend.lock():
            raw = self.backend.note_get(NOTES["verdicts"], state.id)
            current = json.loads(raw) if raw else []
            current.append(stamped.to_dict())
            self.backend.note_set(NOTES["verdicts"], state.id, json.dumps(current))

    # ---------------------------------------------------------- inbox

    def send(self, to_branch: str, message: dict[str, Any], *, sender: str = "") -> str:
        """Leave a message for a branch, without holding its lease.

        This is the ask half of the communicator: A cannot write in B's
        address space, so a request travels as shared state rather than over
        a channel the store cannot see.
        """
        ref = f"{self.INBOX_REF}/{self.session}/{_check_name(to_branch, 'branch')}"
        body = json.dumps({"sender": sender, "at": now_iso(), "body": message}, sort_keys=True)
        with self.backend.lock():
            head = self.backend.ref_sha(ref)
            sha = self.backend.commit_tree(EMPTY_TREE, [head] if head else [], body)
            if not self.backend.cas_update_ref(ref, sha, head):
                raise StaleBranch(f"inbox for {to_branch} moved while sending")
        return sha

    def inbox(self, branch: str, *, since: str | None = None) -> list[dict[str, Any]]:
        """Messages waiting for a branch, oldest first."""
        ref = f"{self.INBOX_REF}/{self.session}/{_check_name(branch, 'branch')}"
        head = self.backend.ref_sha(ref)
        if head is None:
            return []
        marker = since or self.backend.ref_sha(f"{self.SEEN_REF}/{self.session}/{branch}")
        out: list[dict[str, Any]] = []
        for sha in self.backend.rev_list_first_parent(head):
            if marker and sha == marker:
                break
            raw = self.backend.repo.git.log("-1", "--format=%B", sha)
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            msg["id"] = sha
            out.append(msg)
        return list(reversed(out))

    def mark_inbox_seen(self, branch: str, message_id: str) -> None:
        ref = f"{self.SEEN_REF}/{self.session}/{_check_name(branch, 'branch')}"
        self.backend.cas_update_ref(ref, message_id, self.backend.ref_sha(ref))

    # ---------------------------------------------------------- provenance

    def provenance(self, state: State) -> list[State]:
        """The chain of states that produced ``state``, newest first."""
        return [self.state(s) for s in self.backend.rev_list_first_parent(state.id)]

    def origin(self, state: State, path: str) -> State | None:
        """The state in which the artifact at ``path`` took its current form.

        Crosses a branch boundary when the artifact was adopted from another
        brain, so provenance follows the data rather than the branch.
        """
        target = state.blob(path)
        if target is None:
            return None
        origin = state
        for sha in self.backend.rev_list_first_parent(state.id):
            s = self.state(sha)
            if s.blob(path) != target:
                break
            origin = s
            for src in s.meta.sources:
                if src.as_path == path:
                    upstream = self.state(src.state)
                    return self.origin(upstream, src.path) or upstream
        return origin

    # ---------------------------------------------------------- discovery

    def catalog(self, *, branches: Iterable[str] | None = None) -> list[Hit]:
        """Everything published on every branch: a brain's first look around.

        Orientation before search: a brain that does not yet know the right
        word cannot ask for it, so the whole index is small enough to read.
        """
        out: list[Hit] = []
        for name in branches or self.branches():
            sha = self.backend.ref_sha(self.ref_for(name))
            if sha is None:
                continue
            head = self.state(sha)
            for entry in head.manifest.entries.values():
                out.append(Hit(branch=name, state=head, entry=entry, score=0))
        out.sort(key=lambda h: (h.branch, h.entry.name))
        return out

    def search(
        self,
        query: str,
        *,
        types: Iterable[ObjectType] | None = None,
        branches: Iterable[str] | None = None,
    ) -> list[Hit]:
        from taste.memstore.search import search as _search

        return _search(self, query, types=types, branches=branches)
