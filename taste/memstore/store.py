"""The memory layer's API: stores, branches, states.

No git in the signatures. A ``Store`` is a session's memory; a ``Branch`` is
one brain's execution context with its own working tree; a ``State`` is a
checkpoint of that brain: artifacts, transcript, and the reason it exists.

Three rules hold everywhere in this file:

1. **Nothing is lost.** A rollback is an append: the superseded state stays
   the first parent of the rollback state and is reachable forever, with its
   transcript. If the working tree is dirty when a rollback is asked for, that
   dirt is checkpointed first, so even unverified work survives.
2. **A checkpoint is atomic.** Build the commit, attach its notes, then move
   the branch ref with compare-and-swap. A crash before the move leaves the
   store exactly as it was; a crash after it leaves a complete state. A lost
   compare raises ``StaleBranch`` and publishes nothing.
3. **The store is the truth.** Every answer here comes from refs and notes.
   There is no sidecar file to fall out of sync.
"""

from __future__ import annotations

import fcntl
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from taste.memstore.backend import EMPTY_TREE, GitBackend
from taste.memstore.objects import (
    BranchBusy,
    Conflict,
    Manifest,
    ManifestEntry,
    Meta,
    NoSuchState,
    NotAnAncestor,
    ObjectType,
    PublishError,
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
}


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")
    if not s or s.startswith("."):
        raise ValueError(f"not a usable name: {text!r}")
    return s


# ------------------------------------------------------------------ state


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
            self._manifest = Manifest.from_json(self._store.backend.note_get(NOTES["manifest"], self.id))
        return self._manifest

    @property
    def transcript(self) -> Transcript:
        if self._transcript is None:
            self._transcript = Transcript.from_jsonl(
                self._store.backend.note_get(NOTES["transcript"], self.id)
            )
        return self._transcript

    @property
    def conflicts(self) -> list[Conflict]:
        raw = self._store.backend.note_get(NOTES["conflicts"], self.id)
        if not raw:
            return []
        return [Conflict.from_dict(c) for c in json.loads(raw)]

    @property
    def parents(self) -> list[State]:
        return [self._store.state(p) for p in self.meta.parents]

    @property
    def branch(self) -> str:
        return self.meta.branch

    def read(self, path: str) -> str | None:
        """Demand-page one artifact from this state."""
        return self._store.backend.show(self.id, path)

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
            kind = self.manifest.type_of(path) if path in self.manifest_paths() else other.manifest.type_of(path)
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


# ------------------------------------------------------------------ branch


class Branch:
    """One brain's execution context: a branch and its own working tree."""

    def __init__(self, store: Store, name: str, *, producer: str = "") -> None:
        self.store = store
        self.name = name
        self.producer = producer or name
        self.ref = store.ref_for(name)
        self.short_ref = store.short_ref_for(name)
        self.worktree = store.worktree_path_for(name)
        self.backend = GitBackend(self.worktree)
        self._lease = self._acquire_lease()
        self._pending: dict[str, ManifestEntry] = {}
        self._unpublish: set[str] = set()

    def __repr__(self) -> str:
        return f"Branch({self.name!r})"

    def close(self) -> None:
        self.backend.close()
        if self._lease is not None:
            fcntl.flock(self._lease.fileno(), fcntl.LOCK_UN)
            self._lease.close()
            self._lease = None

    def _acquire_lease(self) -> IO[str]:
        path = self.backend.common_dir / f"memstore.lease.{self.store.session}.{self.name}"
        fh = open(path, "a+")  # noqa: SIM115  held for the life of the Branch
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            raise BranchBusy(f"branch {self.name!r} is held by another live process") from None
        return fh

    # ---------------------------------------------------------- reading

    @property
    def head(self) -> State:
        sha = self.backend.ref_sha(self.ref)
        if sha is None:
            raise NoSuchState(f"branch {self.name} has no head")
        return self.store.state(sha)

    def history(self, limit: int | None = None) -> list[State]:
        """This branch's own states, newest first, stopping where the branch began.

        Follows first parents and stops at the first state that belongs to
        another branch (or to the session root), so a branch created from
        another branch's state does not report that branch's past as its own.
        ``store.provenance`` is the walk that crosses those boundaries.
        """
        out: list[State] = []
        for sha in self.backend.rev_list_first_parent(self.head.id):
            st = self.store.state(sha)
            if st.meta.branch != self.name:
                break
            out.append(st)
            if limit is not None and len(out) >= limit:
                break
        return out

    def read(self, path: str, at: State | None = None) -> str | None:
        return (at or self.head).read(path)

    def path(self, rel: str) -> Path:
        return self.worktree / rel

    def write(self, rel: str, content: str) -> None:
        p = self.worktree / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def is_dirty(self) -> bool:
        self.backend.stage_all()
        return self.backend.write_tree() != self.backend.tree_of(self.head.id)

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
        A transcript of ``None`` carries the previous state's forward: the
        brain's context is unchanged, not erased.
        """
        for rel, value in (records or {}).items():
            self.write(rel, json.dumps(value, indent=1, sort_keys=True) + "\n")

        head = self.head
        self.backend.stage_all()
        tree = self.backend.write_tree()

        manifest = head.manifest
        for name in self._unpublish:
            manifest = manifest.without(name)
        return self._commit(
            tree=tree,
            parents=[head.id],
            kind="checkpoint",
            reason=reason,
            manifest=manifest,
            transcript=transcript if transcript is not None else head.transcript,
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
        if self.is_dirty():
            self.checkpoint(f"capture before rollback: {reason}")
        head = self.head
        if not self.backend.is_ancestor(to.id, head.id):
            raise NotAnAncestor(f"{to.id[:10]} is not in the history of {self.name}")
        self._pending.clear()
        self._unpublish.clear()
        state = self._commit(
            tree=self.backend.tree_of(to.id),
            parents=[head.id],
            kind="rollback",
            reason=reason,
            manifest=to.manifest,
            transcript=transcript if transcript is not None else to.transcript,
            verdict=None,
            attempt=0,
            expected_head=head.id,
            restores=to.id,
            rolled_back_from=head.id,
        )
        self.backend.reset_hard_to_head()
        return state

    def merge(self, other: Branch, *, reason: str, resolved: bool = False) -> MergeResult:
        """Merge ``other`` into this branch by type rule; see ``merge.py``.

        If the result carries conflicts, nothing was published. Resolve them
        by writing the files you want into this working tree and calling
        again with ``resolved=True``.
        """
        from taste.memstore.merge import merge_branches

        return merge_branches(self, other, reason=reason, resolved=resolved)

    # ---------------------------------------------------------- internals

    def _commit(
        self,
        *,
        tree: str,
        parents: list[str],
        kind: StateKind,
        reason: str,
        manifest: Manifest,
        transcript: Transcript,
        verdict: Verdict | None,
        attempt: int,
        expected_head: str | None,
        restores: str | None = None,
        rolled_back_from: str | None = None,
        conflicts: list[Conflict] | None = None,
    ) -> State:
        """The atomic sequence every state goes through. See the module docstring."""
        now = now_iso()
        # Pending publishes are resolved against the tree we are about to commit.
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
        meta = Meta(
            branch=self.name,
            kind=kind,
            reason=reason,
            producer=self.producer,
            parents=tuple(parents),
            created_at=now,
            attempt=attempt,
            verdict=verdict,
            restores=restores,
            rolled_back_from=rolled_back_from,
        )
        message = f"{reason}\n\nTaste-State: {kind}\nTaste-Branch: {self.name}\n"
        sha = self.backend.commit_tree(tree, parents, message)
        self.backend.note_set(NOTES["meta"], sha, meta.to_json())
        self.backend.note_set(NOTES["manifest"], sha, manifest.to_json())
        self.backend.note_set(NOTES["transcript"], sha, transcript.to_jsonl())
        if conflicts:
            self.backend.note_set(NOTES["conflicts"], sha, json.dumps([c.to_dict() for c in conflicts]))
        if not self.backend.cas_update_ref(self.ref, sha, expected_head):
            raise StaleBranch(
                f"{self.name} moved: expected head {(expected_head or 'none')[:10]}; nothing published"
            )
        self._pending.clear()
        self._unpublish.clear()
        return self.store.state(sha)


# ------------------------------------------------------------------ store


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


class Store:
    """A session's memory. Opens or creates the repository under ``root``."""

    REF_ROOT = "refs/heads/mem"
    ROOT_REF = "refs/taste/memstore"

    def __init__(self, root: Path, session: str) -> None:
        self.root = Path(root).resolve()
        self.session = _slug(session)
        self.backend = GitBackend.init(self.root)
        self._branches: dict[str, Branch] = {}

    @classmethod
    def open(cls, root: Path, session: str) -> Store:
        return cls(root, session)

    def close(self) -> None:
        for b in self._branches.values():
            b.close()
        self.backend.close()

    # ---------------------------------------------------------- naming

    def short_ref_for(self, name: str) -> str:
        return f"mem/{self.session}/{_slug(name)}"

    def ref_for(self, name: str) -> str:
        return f"refs/heads/{self.short_ref_for(name)}"

    def worktree_path_for(self, name: str) -> Path:
        return self.root.parent / ".taste-worktrees" / "mem" / self.session / _slug(name)

    # ---------------------------------------------------------- session root

    def _session_root(self) -> str:
        """The commit every branch of this session descends from.

        Created once: the repository's HEAD if it has one, else a commit of
        the empty tree. Kept under a ref so every branch shares a merge base.
        """
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
            # The root is a state like any other, so every reachable commit
            # answers ``meta``. Notes sit outside the tree, so annotating a
            # pre-existing commit changes nothing the user can see.
            if self.backend.note_get(NOTES["meta"], base) is None:
                meta = Meta(
                    branch="",
                    kind="root",
                    reason=f"session root: {self.session}",
                    producer="",
                    parents=(),
                    created_at=now_iso(),
                )
                self.backend.note_set(NOTES["meta"], base, meta.to_json())
                self.backend.note_set(NOTES["manifest"], base, Manifest().to_json())
                self.backend.note_set(NOTES["transcript"], base, "")
            self.backend.cas_update_ref(ref, base, None)
        return base

    # ---------------------------------------------------------- branches

    def branch(self, name: str, *, from_state: State | None = None, producer: str = "") -> Branch:
        """Open ``name``, creating it (ref and working tree) if it does not exist."""
        name = _slug(name)
        cached = self._branches.get(name)
        if cached is not None and cached._lease is not None:
            return cached
        ref = self.ref_for(name)
        if self.backend.ref_sha(ref) is None:
            base = from_state.id if from_state is not None else self._session_root()
            if not self.backend.cas_update_ref(ref, base, None):
                raise StaleBranch(f"branch {name} was created concurrently")
            # The branch's first state is a "branch" record: it names where the
            # branch came from and marks where its own history begins.
            self._seed(name, ref, base, from_state, producer)
        wt = self.worktree_path_for(name)
        if not wt.exists():
            self.backend.worktree_add(wt, self.short_ref_for(name))
        b = Branch(self, name, producer=producer)
        self._branches[name] = b
        return b

    def _seed(self, name: str, ref: str, base: str, from_state: State | None, producer: str) -> None:
        tree = self.backend.tree_of(base)
        meta = Meta(
            branch=name,
            kind="branch",
            reason=f"branch {name} from {(from_state.id[:10] if from_state else 'session root')}",
            producer=producer or name,
            parents=(base,),
            created_at=now_iso(),
        )
        manifest = from_state.manifest if from_state else Manifest()
        transcript = from_state.transcript if from_state else Transcript()
        sha = self.backend.commit_tree(tree, [base], f"{meta.reason}\n\nTaste-State: branch\nTaste-Branch: {name}\n")
        self.backend.note_set(NOTES["meta"], sha, meta.to_json())
        self.backend.note_set(NOTES["manifest"], sha, manifest.to_json())
        self.backend.note_set(NOTES["transcript"], sha, transcript.to_jsonl())
        if not self.backend.cas_update_ref(ref, sha, base):
            raise StaleBranch(f"branch {name} moved while being seeded")

    def branches(self) -> list[str]:
        prefix = f"{self.REF_ROOT}/{self.session}/"
        return sorted(name[len(prefix) :] for name, _ in self.backend.for_each_ref(prefix))

    def get(self, name: str) -> Branch:
        return self.branch(name)

    def remove_branch(self, name: str) -> None:
        """Delete the working tree only. The branch and its states stay."""
        b = self._branches.pop(_slug(name), None)
        if b is not None:
            b.close()
        self.backend.worktree_remove(self.worktree_path_for(name))

    # ---------------------------------------------------------- states

    def state(self, sha: str) -> State:
        return State(self, sha)

    def heads(self) -> dict[str, State]:
        return {name: self.branch(name).head for name in self.branches()}

    # ---------------------------------------------------------- provenance

    def provenance(self, state: State) -> list[State]:
        """The chain of states that produced ``state``, newest first."""
        return [self.state(s) for s in self.backend.rev_list_first_parent(state.id)]

    def origin(self, state: State, path: str) -> State | None:
        """The state in which the artifact at ``path`` took its current form."""
        target = state.blob(path)
        if target is None:
            return None
        origin = state
        for sha in self.backend.rev_list_first_parent(state.id):
            s = self.state(sha)
            if s.blob(path) != target:
                break
            origin = s
        return origin

    # ---------------------------------------------------------- search

    def search(
        self,
        query: str,
        *,
        types: Iterable[ObjectType] | None = None,
        branches: Iterable[str] | None = None,
    ) -> list[Hit]:
        from taste.memstore.search import search as _search

        return _search(self, query, types=types, branches=branches)
