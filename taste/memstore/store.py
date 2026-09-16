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
import functools
import json
import os
import re
import socket
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from taste.memstore.backend import EMPTY_TREE, GitBackend
from taste.memstore.objects import (
    BadName,
    BranchBusy,
    Conflict,
    ForeignHead,
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
    conflict_digest,
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

VERDICT_LOOKBACK = 64

#: Written beside the repository on every publish; see ``Branch._record_tip``.
TIP_SCHEMA = "taste.memstore/BranchTip/1"

#: One line per accepted rewind; see ``Branch.accept_rewind``.
REWIND_SCHEMA = "taste.memstore/AcceptedRewind/1"


def _read_holder(path: Path) -> dict[str, Any] | None:
    """Who holds a lease, or None if it is free.

    The kernel file lock is the truth; this file is only the label on it. The
    label is cleared on release and discarded when the process that wrote it
    is gone, so a leftover record cannot make a free branch look busy. It is
    never used to decide whether the lease can be taken -- only to say who has
    it -- because a label can be stale in ways a lock cannot.
    """
    try:
        handle = open(path, "a+")  # noqa: SIM115 - held through label + lock check
    except OSError:
        return None
    try:
        handle.seek(0)
        raw = handle.read().strip()
        if not raw:
            return None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        except OSError:
            # An unreadable lock state is not evidence that a writer is live.
            return None
        else:
            # The label can name an alive process which has already released
            # this branch.  The kernel lock, not kill(pid, 0), is the lease.
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return None
    finally:
        handle.close()
    try:
        holder = dict(json.loads(raw))
    except json.JSONDecodeError:
        return None
    pid = holder.get("pid")
    if holder.get("host") == socket.gethostname() and isinstance(pid, int):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        except OSError:
            pass  # alive, just not ours to signal
    return holder


def _parse_journal(raw: bytes) -> tuple[list[dict[str, Any]], int]:
    """Turns in a journal, and how many bytes of it they account for.

    A process killed mid-write leaves one partial line, and it can only be the
    last. The turn it describes never finished being recorded, so there is
    nothing in it to recover; the walk stops there and those bytes are not
    counted as consumed.
    """
    turns: list[dict[str, Any]] = []
    offset = 0
    for line in raw.splitlines(keepends=True):
        text = line.decode("utf-8", "surrogateescape")
        if not text.strip():
            offset += len(line)
            continue
        try:
            turns.append(json.loads(text))
        except json.JSONDecodeError:
            break
        offset += len(line)
    return turns, offset
"""How far back ``resume`` looks for judgments a brain has not acknowledged.

Bounded rather than unbounded: a verdict the brain walked past many states ago
is history, not news, and reading every ancestor's notes on every wake would
make the cost of waking grow with the length of the run.
"""

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


def _serialized_branch_mutation(method):
    """Serialize one Branch object's shared working-tree mutations."""

    @functools.wraps(method)
    def locked(self, *args, **kwargs):
        with self._mutation_lock:
            return method(self, *args, **kwargs)

    return locked


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
    turns_offset: int = 0
    """How many bytes of the turn journal this state folded in.

    Recorded by position rather than by filename so that turns arriving while
    the state is being built -- a git commit and four notes writes, a fifth of
    a second in which the brain is still thinking -- are carried forward
    instead of deleted unread.
    """


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
        subject = self._store.backend.repo.git.log("-1", "--format=%s", self.id)
        binding = re.search(r"\[(sha256:[0-9a-f]{64})\]\Z", subject)
        if not raw:
            if binding is not None:
                raise ValueError("conflict state lost its immutable-bound conflict evidence")
            return []
        if binding is None:
            raise ValueError("conflict evidence has no immutable commit binding")
        try:
            decoded = json.loads(raw)
            if not isinstance(decoded, list):
                raise ValueError("conflict evidence must be an array")
            conflicts = [Conflict.from_dict(value) for value in decoded]
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError("conflict evidence is malformed") from exc
        if conflict_digest(conflicts) != binding.group(1):
            raise ValueError("conflict evidence differs from its immutable commit binding")
        return conflicts

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
        self._backend: GitBackend | None = None

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

    # ------------------------------------------------------ work in flight
    #
    # Everything below reports what a brain is doing *before* it checkpoints.
    # Judging only published states means a monitor cannot catch a wrong turn
    # until a minute of tool time has already been spent on it, and none of
    # this needs the write lease: the sidecars are ordinary files and reading
    # a working tree takes no lock.

    @property
    def intent(self) -> str | None:
        """What the brain said it was about to attempt, if it said."""
        path = self.store.sidecar("intent", self.name)
        return path.read_text(encoding="utf-8") if path.exists() else None

    @property
    def holder(self) -> dict[str, Any] | None:
        """Who holds the write lease, or None if the branch is free."""
        return _read_holder(self.store.sidecar("lease", self.name))

    def pending_turns(self) -> list[dict[str, Any]]:
        """Turns the brain has recorded since its last state."""
        if not self.exists():
            return []
        path = self.store.sidecar("turns", self.name, f".{self.head.id}")
        if not path.exists():
            return []
        return _parse_journal(path.read_bytes())[0]

    def _worktree_backend(self) -> GitBackend | None:
        worktree = self.store.worktree_path_for(self.name)
        if not worktree.exists():
            return None
        if self._backend is None:
            self._backend = GitBackend(worktree)
        return self._backend

    def dirty_paths(self) -> list[str]:
        """Files the brain has changed but not yet checkpointed."""
        backend = self._worktree_backend()
        return backend.dirty_paths() if backend is not None else []

    def pending_diff(self, *, numstat: bool = False) -> str:
        """Uncommitted work as patch text -- what a monitor actually reads."""
        backend = self._worktree_backend()
        return backend.diff_pending(numstat=numstat) if backend is not None else ""


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
        self._pending_sources: list[Source] = []
        self._journal_lock = threading.Lock()
        # A Store returns the same writable Branch object to in-process
        # callers.  Its lease excludes other processes, while this lock keeps
        # those callers from staging/resetting the shared worktree across one
        # another's checkpoint or merge publication window.
        self._mutation_lock = threading.RLock()

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
            # Clear the label but keep the file: unlinking it would let the
            # next process flock a different inode and believe it held the
            # same lease.
            with contextlib.suppress(OSError):
                self._lease.seek(0)
                self._lease.truncate()
                self._lease.flush()
            fcntl.flock(self._lease.fileno(), fcntl.LOCK_UN)
            self._lease.close()
            self._lease = None
            self.store._branches.pop(self.name, None)

    @property
    def holder(self) -> dict[str, Any] | None:
        """Who holds this branch, or None if it is free."""
        return _read_holder(self._lease_path())

    # ---------------------------------------------------------- lease

    def _lease_path(self) -> Path:
        return self.store.sidecar("lease", self.name)

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

    @_serialized_branch_mutation
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

    def pending_diff(self, *, numstat: bool = False) -> str:
        """Uncommitted work as patch text, untracked files included."""
        return self.backend.diff_pending(numstat=numstat)

    # ---------------------------------------------------------- intent and resume

    def _intent_path(self) -> Path:
        return self.store.sidecar("intent", self.name)

    def _turns_path(self, head_id: str | None = None) -> Path:
        """Where turns accumulate before they are folded into a state.

        Keyed on the state the turns extend, so a journal left behind by a
        crash *after* publication names a state the branch has already moved
        past and can never be mistaken for pending work.
        """
        head = head_id if head_id is not None else self.head.id
        return self.store.sidecar("turns", self.name, f".{head}")

    def _read_journal(self, head_id: str | None = None) -> tuple[list[dict[str, Any]], int]:
        """The pending turns, and how many bytes of journal they account for.

        The offset is what makes the fold safe: a state consumes a prefix of
        the journal, never the file, so a turn appended while the state was
        being built is still there afterwards.
        """
        path = self._turns_path(head_id)
        if not path.exists():
            return [], 0
        return _parse_journal(path.read_bytes())

    def _pending_turns(self) -> list[dict[str, Any]]:
        """Turns recorded since the head state and not yet folded into one."""
        with self._journal_lock:
            return self._read_journal()[0]

    def _acked_path(self) -> Path:
        return self.store.sidecar("acked", self.name)

    def _acked(self) -> dict[str, int]:
        path = self._acked_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _recent_verdicts(self) -> list[tuple[str, list[Verdict]]]:
        """Verdicts on the head and its recent ancestors, newest state first."""
        found: list[tuple[str, list[Verdict]]] = []
        for state in self.store.provenance(self.head)[:VERDICT_LOOKBACK]:
            verdicts = state.verdicts
            if verdicts:
                found.append((state.id, verdicts))
        return found

    def unacked_verdicts(self) -> list[Verdict]:
        """Judgments this brain has not yet seen, newest state first."""
        acked = self._acked()
        pending: list[Verdict] = []
        for state_id, verdicts in self._recent_verdicts():
            pending.extend(verdicts[acked.get(state_id, 0) :])
        return pending

    def verdict_watermark(self) -> dict[str, int]:
        """The exact verdict counts visible in one observation.

        A caller can put this watermark in a prompt and later pass the same
        mapping to :meth:`acknowledge`.  Verdicts which race the prompt have a
        larger count and therefore remain unread instead of being silently
        acknowledged despite never having reached the worker.
        """
        return {
            state_id: len(verdicts)
            for state_id, verdicts in self._recent_verdicts()
        }

    def acknowledge(self, through: Mapping[str, int] | None = None) -> None:
        """Mark an exact observed prefix of verdicts as read.

        Counts are kept per state, so a monitor that judges an older state
        after the brain has moved on is still delivered exactly once.  With
        no argument this retains the original convenience API and snapshots
        everything currently visible.  Passing ``through`` is the safe path
        for asynchronous delivery: only the counts actually put in that
        prompt are advanced.
        """
        visible = dict(self._recent_verdicts())
        wanted = self.verdict_watermark() if through is None else dict(through)
        acked = self._acked()
        for state_id, count in wanted.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("verdict acknowledgement counts must be non-negative integers")
            verdicts = visible.get(state_id)
            if verdicts is None:
                raise ValueError(
                    f"cannot acknowledge verdicts for invisible state {state_id[:10]}"
                )
            if count > len(verdicts):
                raise ValueError(
                    f"cannot acknowledge {count} verdicts for {state_id[:10]}; "
                    f"only {len(verdicts)} are visible"
                )
            acked[state_id] = max(acked.get(state_id, 0), count)

        path = self._acked_path()
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        payload = json.dumps(acked, sort_keys=True).encode("utf-8")
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
            recovered_turns=tuple(self._pending_turns()),
            verdicts=tuple(head.verdicts),
            unacked=tuple(self.unacked_verdicts()),
        )

    # ---------------------------------------------------------- publishing

    @_serialized_branch_mutation
    def publish(
        self,
        name: str,
        path: str,
        *,
        type: ObjectType = ObjectType.FILE,
        description: str = "",
    ) -> None:
        """Make a live artifact path discoverable at the next checkpoint.

        Publication follows the path, not one forever-frozen blob: later
        checkpoints refresh its advertised blob and publication time whenever
        its bytes change, and remove it from the manifest if the path is gone.
        A catalog hit is still an immutable snapshot because it carries the
        particular state and blob observed by the reader.
        """
        self._unpublish.discard(name)
        self._pending[name] = ManifestEntry(name=name, path=path, type=type, description=description)

    @_serialized_branch_mutation
    def unpublish(self, name: str) -> None:
        self._pending.pop(name, None)
        self._unpublish.add(name)

    @_serialized_branch_mutation
    def adopt(self, source: State | Hit, path: str | None = None, *, as_: str | None = None) -> str:
        """Take another branch's artifact, recording where it came from.

        Copies the bytes into this working tree and records the source in the
        next state's provenance, so ``store.origin`` can cross branches. This
        is the read half of the communicator: A finds B's artifact and adopts
        it rather than re-deriving it.
        """
        advertised_blob: str | None = None
        if isinstance(source, Hit):
            if path is not None and path != source.entry.path:
                raise PublishError(
                    f"catalog hit {source.entry.name!r} advertises {source.entry.path}, not {path}"
                )
            state, path = source.state, source.entry.path
            advertised_blob = source.entry.blob
        else:
            state = source
        if path is None:
            raise ValueError("adopt needs a path when given a state")
        blob = state.blob(path)
        if advertised_blob is not None and blob != advertised_blob:
            raise PublishError(
                f"{path} in state {state.id[:10]} does not match its advertised blob"
            )
        raw = state.read_bytes(path)
        if raw is None or blob is None:
            raise PublishError(f"{path} is not in state {state.id[:10]}")
        target = as_ or path
        self.write(target, raw)
        self._pending_sources.append(
            Source(
                branch=state.meta.branch,
                state=state.id,
                path=path,
                as_path=target,
                blob=blob,
                session=state.meta.session,
            )
        )
        return target

    # ---------------------------------------------------------- context

    def turn(self, **turn: Any) -> None:
        """Append one turn to the brain's context, durably, as it happens.

        Written to disk on the spot rather than held until the next state. A
        model's output is paid for the moment it arrives, and a brain killed
        before it can checkpoint has to wake up still holding its own
        reasoning and its tool results -- that is the material it improves
        from. The write is the durability point because a killed process runs
        no ``finally``, no ``atexit`` and no flush on the way out.
        """
        line = (json.dumps(dict(turn), sort_keys=True) + "\n").encode("utf-8", "surrogateescape")
        with self._journal_lock:
            self._append_journal(self.head.id, line)

    def _append_journal(self, head_id: str, payload: bytes) -> None:
        """Append to a journal and return only once the bytes are on disk."""
        with open(self._turns_path(head_id), "ab") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())

    # ---------------------------------------------------------- writing states

    @_serialized_branch_mutation
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
        # Held across build and publish so a gc() in another process cannot
        # land between the commit and the ref move: prune takes this same
        # lock, and an object pruned mid-checkpoint used to surface as a
        # spurious StaleBranch. The lock is re-entrant, so the notes writes
        # inside still take it as before. The public build/publish split is
        # deliberately left unlocked -- a caller may hold a Built for as long
        # as it likes, and that window is documented.
        with self.backend.lock():
            built = self.build(
                reason,
                transcript=transcript,
                records=records,
                verdict=verdict,
                attempt=attempt,
            )
            return self.publish_state(built)

    @_serialized_branch_mutation
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
        self._require_memstore_head()
        head = self.head
        self.backend.stage_all()
        tree = self.backend.write_tree()
        manifest = head.manifest
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

    @_serialized_branch_mutation
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
        self._require_memstore_head()
        head = self.head
        # Ancestry alone is too weak once this branch has merged another: the
        # other branch's states become git ancestors, so rolling "back" to one
        # of them would silently adopt its tree, manifest and transcript and
        # drop this branch's own files. Reinstating another branch's work is
        # what adopt and merge are for.
        if to.meta.branch != self.name and to.id not in {s.id for s in self.history()}:
            raise NotAnAncestor(f"{to.id[:10]} is not a state of {self.name}")
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

    @_serialized_branch_mutation
    def merge(self, other: Branch | BranchView, *, reason: str, resolved: bool = False) -> MergeResult:
        """Merge ``other`` into this branch by type rule; see ``merge.py``.

        If the result carries conflicts, the merge itself did not happen, but
        a one-parent ``conflict`` state *was* published on this branch so the
        failed attempt is durable. Resolve it by writing the files you want
        into this working tree and calling again with ``resolved=True``.
        """
        from taste.memstore.merge import merge_branches

        return merge_branches(self, other, reason=reason, resolved=resolved)

    def _require_memstore_head(self) -> None:
        """Refuse to build on a head this layer did not create.

        Every state carries a ``meta`` note; a commit without one was put on
        the ref by something else. Checked on the write paths only -- reads
        must keep working, because whoever repairs this has to look at it
        first.

        The note is read through the backend rather than through
        ``State.meta``, which raises ``NoSuchState`` for exactly this case:
        the point here is to name the condition, not to re-raise it.
        """
        sha = self.store.backend.ref_sha(self.ref)
        if sha is None:
            return  # No head at all is a different failure; let it surface as one.
        if self.backend.note_get(NOTES["meta"], sha) is None:
            raise ForeignHead(
                f"{self.name} points at {sha[:10]}, which memstore did not create: "
                "it carries no state metadata. A commit made inside the worktree "
                "(git commit, amend, revert) moves the branch out from under this "
                "layer. The last memstore state is still reachable in the reflog."
            )
        self._require_attached_worktree()
        self._require_no_silent_rewind()

    def _tip_path(self) -> Path:
        return self.store.sidecar("tip", self.name)

    def _record_tip(self, sha: str) -> None:
        """Write down how far this branch has got.

        Beside the repository, not inside the tree: ``git reset --hard``
        rewrites the tree and moves the ref, and cannot touch this. It is the
        only durable answer to "was this branch ever further along", because
        ``history()`` walks back from the current head -- so after a rewind the
        short history *is* the whole history and the loss is invisible.
        """
        path = self._tip_path()
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_text(
            json.dumps({"schema": TIP_SCHEMA, "state": sha, "at": now_iso()}) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _recorded_tip(self) -> str | None:
        """The furthest state this branch published, or None if unrecorded."""
        try:
            raw = self._tip_path().read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            record = json.loads(raw)
            state = record["state"]
        except (ValueError, KeyError, TypeError):
            # A torn or foreign tip record cannot prove anything was lost, and
            # refusing on it would strand a branch over its own bookkeeping.
            return None
        return state if isinstance(state, str) and state else None

    def _require_no_silent_rewind(self) -> None:
        """Refuse a head that is behind a state this branch published.

        The metadata guard cannot see this. ``git reset --hard`` lands on a
        commit this layer *did* create, carrying every note, so the branch
        looks perfectly healthy while states have silently left it. Measured:
        three checkpoints, a reset to the first, and the next checkpoint was
        accepted with two states gone -- after which delivery projected one of
        the orphans into the shared integration branch, advertising bytes the
        worker's own history no longer contained.

        A branch with no recorded tip proceeds. Failing closed there would
        make the layer unusable against any repository it did not create,
        which is a worse failure than the one being prevented.
        """
        tip = self._recorded_tip()
        if tip is None:
            return
        head = self.store.backend.ref_sha(self.ref)
        if head is None or head == tip:
            return
        if self.backend.is_ancestor(tip, head):
            return  # ordinary advance, or a rollback, which appends
        if not self.backend.is_ancestor(head, tip):
            return  # diverged rather than rewound; a different failure
        orphans = [
            sha
            for sha in self.backend.rev_list_first_parent(tip)
            if not self.backend.is_ancestor(sha, head)
        ]
        lost = ", ".join(sha[:10] for sha in orphans[:5])
        raise ForeignHead(
            f"{self.name} is at {head[:10]}, behind {tip[:10]}, which this branch "
            f"published. {len(orphans)} state(s) are no longer reachable: {lost}. "
            "A `git reset --hard` inside the worktree moves the branch back onto "
            "a commit this layer did create, so it carries every note and looks "
            "healthy while the states after it are gone. Nothing was deleted -- "
            "they are still in the object store -- but this branch no longer "
            "claims them, and building here would publish work as if they never "
            "existed. A writer that is repairing this deliberately should use "
            "`accept_rewind(evidence=..., reason=...)` instead."
        )

    def rewound_past(self) -> tuple[str, ...]:
        """States this branch published that its head no longer claims.

        Empty for a healthy branch, which is the ordinary case. A caller that
        owns a branch and may legitimately be repairing it asks this rather
        than reaching for the tip file: the question is the layer's to answer,
        and the answer is what ``accept_rewind`` is for.
        """
        tip = self._recorded_tip()
        if tip is None:
            return ()
        head = self.store.backend.ref_sha(self.ref)
        if head is None or head == tip or not self.backend.is_ancestor(head, tip):
            return ()
        return tuple(
            sha
            for sha in self.backend.rev_list_first_parent(tip)
            if not self.backend.is_ancestor(sha, head)
        )

    def adopt_rewind_if_any(self, *, evidence: str, reason: str) -> tuple[str, ...]:
        """Accept a rewind if there is one, and say nothing if there is not.

        The common shape for a writer that owns a branch and may be resuming
        over damage: it cannot know in advance whether the last process died
        mid-write. Returns the states that stopped being claimed, empty when
        the branch was healthy, so a caller can surface a real loss without
        having to ask twice.
        """
        orphaned = self.rewound_past()
        if orphaned:
            self.accept_rewind(evidence=evidence, reason=reason)
        return orphaned

    def accept_rewind(self, *, evidence: str, reason: str) -> str:
        """Accept a rewound head as this branch's truth, recording why.

        A rewound ref is ambiguous: a worker corrupting itself, or an owner
        repairing damage. From the ref alone the two are identical, which is
        why an earlier forward-only guard had to be backed out -- it refused
        the coordinator's own recovery from a rewound control branch.

        The difference is evidence. A worker that ran ``git reset --hard`` has
        none; a coordinator replaying from a journal on another branch names
        it. So the way through is explicit and it writes itself down.

        Deliberately not ``rollback(to)``. The repairing writer is already
        where it means to be -- it does not want to move the branch further,
        it wants to stop being refused and carry on. Measured: after the raw
        rewind the tests perform, the head is an *ancestor* of the tip, so
        there is no state to roll back to and a move-shaped primitive has
        nothing to do. What is stale is this layer's record of how far the
        branch got, and that is what this corrects.

        Returns the state ids that stop being claimed, so a caller can log or
        surface them rather than discovering the loss later.
        """
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("accept_rewind requires evidence naming what this repair replays")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("accept_rewind requires a reason")
        head = self.store.backend.ref_sha(self.ref)
        if head is None:
            raise ForeignHead(f"{self.name} has no head to accept")
        tip = self._recorded_tip()
        orphans: tuple[str, ...] = ()
        if tip is not None and tip != head and self.backend.is_ancestor(head, tip):
            orphans = tuple(
                sha
                for sha in self.backend.rev_list_first_parent(tip)
                if not self.backend.is_ancestor(sha, head)
            )
        self._record_tip(head)
        self._record_rewind_acceptance(head, orphans, evidence.strip(), reason.strip())
        return head

    def _record_rewind_acceptance(
        self, head: str, orphans: tuple[str, ...], evidence: str, reason: str
    ) -> None:
        """Append the acceptance beside the branch, so it is not only a gap.

        The tip file alone would say the branch is healthy again and nothing
        would say why it moved. A reader six months later should find the
        claim, its evidence, and what stopped being claimed -- in the store,
        not in git's reflog, which prunes.
        """
        path = self.store.sidecar("rewinds", self.name)
        entry = json.dumps(
            {
                "schema": REWIND_SCHEMA,
                "accepted_head": head,
                "orphaned": list(orphans),
                "evidence": evidence,
                "reason": reason,
                "at": now_iso(),
            }
        )
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(entry + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _require_attached_worktree(self) -> None:
        """Refuse to build from a tree that has left this branch.

        The two checks around this one both look at the ref, and a detached
        HEAD never touches it. Measured: after ``git checkout --detach`` the
        branch ref was still the newest state and still carried its notes --
        both guards saw a perfectly healthy branch -- while the working tree
        sat on an older commit. The next checkpoint staged *that* tree and
        compare-and-swapped the ref onto it, publishing content from a
        different point in history as this branch's next state. The file this
        layer had recorded as ``'2'`` came back as ``'1'``, with no error
        anywhere.

        ``git worktree add`` binds a tree to exactly one branch, so the
        invariant is simply that the binding still holds.
        """
        attached = self.backend.worktree_branch_ref()
        if attached == self.ref:
            return
        if attached is None:
            raise ForeignHead(
                f"{self.name}'s working tree is on a detached HEAD, not on {self.ref}. "
                "A checkpoint would stage this tree and move the branch onto it, "
                "publishing another commit's content as this branch's work. "
                "Reattach with `git checkout` of the branch before continuing."
            )
        raise ForeignHead(
            f"{self.name}'s working tree is checked out on {attached}, not on "
            f"{self.ref}. A checkpoint stages this tree and moves that ref, so the "
            "two must be the same branch."
        )

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
        # Manifest entries are live paths. Apply explicit removals on every
        # state-building path (including merge), then keep each remaining
        # entry pinned to the bytes in this exact tree. A deletion makes the
        # live artifact cease to exist; a content change is a new publication.
        for name in self._unpublish:
            manifest = manifest.without(name)
        for name, entry in list(manifest.entries.items()):
            blob = self.backend.blob_at(tree, entry.path)
            if blob is None:
                manifest = manifest.without(name)
            elif blob != entry.blob:
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
        for name, entry in list(self._pending.items()):
            blob = self.backend.blob_at(tree, entry.path)
            if blob is None:
                # Drop the request before raising. Left pending, it made every
                # later checkpoint re-raise the same error, and because
                # ``_capture`` checkpoints before rollback and merge, a single
                # bad publish() wedged the branch against ever committing
                # again -- turning a rejected bit of metadata into total loss.
                self._pending.pop(name, None)
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
        with self._journal_lock:
            pending, turns_offset = self._read_journal()
        if pending:
            wanted = wanted.extend(pending)
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
        return Built(sha=sha, expected_head=expected_head, turns_offset=turns_offset)

    def _transcript_depth(self, base: State | None) -> int:
        depth = 0
        cur = base
        while cur is not None and cur.meta.transcript_from and depth < TRANSCRIPT_SNAPSHOT_EVERY:
            depth += 1
            cur = self.store.state(cur.meta.transcript_from)
        return depth

    @_serialized_branch_mutation
    def publish_state(self, built: Built) -> State:
        """Move the branch ref to a built state, or fail having published nothing."""
        with self._journal_lock:
            carried = self._carry_forward(built)
            if not self.backend.cas_update_ref(self.ref, built.sha, built.expected_head):
                if carried:
                    # Nothing was published, so the turns belong where they
                    # already are: on the journal of the head we are still on.
                    with contextlib.suppress(FileNotFoundError):
                        self._turns_path(built.sha).unlink()
                raise StaleBranch(
                    f"{self.name} moved: expected head {(built.expected_head or 'none')[:10]}; "
                    "nothing published"
                )
            if built.expected_head is not None:
                with contextlib.suppress(FileNotFoundError):
                    self._turns_path(built.expected_head).unlink()
        self._record_tip(built.sha)
        self._pending.clear()
        self._unpublish.clear()
        self._pending_sources.clear()
        with contextlib.suppress(FileNotFoundError):
            self._intent_path().unlink()
        return self.store.state(built.sha)

    def _carry_forward(self, built: Built) -> bool:
        """Move turns recorded after the fold onto the state about to publish.

        Written before the compare-and-swap so a crash on either side of it is
        safe. If the swap succeeds the turns are already where the next state
        will look for them; if it fails they are still in the journal the
        branch is really on, and the speculative copy is simply dropped.
        Either way they appear exactly once, which an unlink by filename could
        not promise.
        """
        if built.expected_head is None:
            return False
        path = self._turns_path(built.expected_head)
        if not path.exists():
            return False
        leftover = path.read_bytes()[built.turns_offset :]
        if not leftover:
            return False
        self._append_journal(built.sha, leftover)
        return True


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

    def sidecar(self, kind: str, branch: str, suffix: str = "") -> Path:
        """A per-branch file beside the repository, not inside any state.

        Intent, lease, turn journal and verdict acknowledgements live here:
        they describe work in flight rather than work that happened, so they
        are not history and must not be committed as if they were.
        """
        return (
            self.backend.common_dir
            / f"memstore.{kind}.{self.session}.{_check_name(branch, 'branch')}{suffix}"
        )

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
        # One ref write, from nothing straight to the seeded state. Claiming
        # the name first and seeding second meant a crash in between left the
        # branch permanently pointing at another branch's state, with an empty
        # history and no way to repair it by reopening.
        if not self.backend.cas_update_ref(ref, sha, None):
            raise StaleBranch(f"branch {name} was created concurrently")

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
        """Drop the working tree after preserving anything dirty in it.

        A supervisor normally opens a fresh ``Store`` after its worker dies,
        so it does not have that worker's ``Branch`` object cached. It must
        still acquire the now-free lease and capture the tree before the
        backend's force-removal. If the worker is actually alive, acquiring
        the lease raises ``BranchBusy`` and nothing is removed.
        """
        name = _check_name(name, "branch")
        worktree = self.worktree_path_for(name)
        b = self._branches.get(name)
        opened_here = False
        if worktree.exists() and (b is None or b._lease is None):
            b = self.branch(name)
            opened_here = True
        if b is not None:
            try:
                b._capture("worktree removal", name)
            except Exception:
                if opened_here:
                    b.close()
                raise
            b.close()
        self.backend.worktree_remove(worktree)

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
        self._sweep_turn_journals()

    def _sweep_turn_journals(self) -> None:
        """Drop turn journals that no branch can still be extending.

        A journal is live only while it names its branch's current head; once
        the branch moves on, the turns in it were either folded into that move
        or belong to a state that was never published. Either way no brain can
        still be adding to it. Journals for a head that is still current are
        untouched, so collecting is never a way to lose pending reasoning.
        """
        live = set()
        for name in self.branches():
            sha = self.backend.ref_sha(self.ref_for(name))
            if sha is not None:
                live.add(f"memstore.turns.{self.session}.{name}.{sha}")
        for path in self.backend.common_dir.glob(f"memstore.turns.{self.session}.*"):
            if path.name not in live:
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()

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
        if not isinstance(message, dict) or not isinstance(sender, str):
            raise ValueError("inbox message must be an object and sender must be text")
        body = json.dumps(
            {"sender": sender, "at": now_iso(), "body": message},
            allow_nan=False,
            sort_keys=True,
        )
        with self.backend.lock():
            head = self.backend.ref_sha(ref)
            sha = self.backend.commit_tree(EMPTY_TREE, [head] if head else [], body)
            if not self.backend.cas_update_ref(ref, sha, head):
                raise StaleBranch(f"inbox for {to_branch} moved while sending")
        return sha

    def inbox(self, branch: str, *, since: str | None = None) -> list[dict[str, Any]]:
        """Messages waiting for a branch, oldest first.

        The inbox cursor is cumulative, so omitting one corrupt predecessor
        would let a caller acknowledge a later message *through* bytes it
        never saw.  Decode the exact first-parent log fail-closed instead of
        skipping malformed commits.
        """
        ref = f"{self.INBOX_REF}/{self.session}/{_check_name(branch, 'branch')}"
        head = self.backend.ref_sha(ref)
        marker = since or self.backend.ref_sha(f"{self.SEEN_REF}/{self.session}/{branch}")
        if head is None:
            if marker is not None:
                raise ValueError(f"inbox for {branch} has a cursor but no message log")
            return []
        out: list[dict[str, Any]] = []
        found_marker = marker is None

        def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            decoded: dict[str, Any] = {}
            for key, value in pairs:
                if key in decoded:
                    raise ValueError(f"duplicate key {key!r}")
                decoded[key] = value
            return decoded

        def no_constant(value: str) -> Any:
            raise ValueError(f"non-JSON number {value}")

        for sha in self.backend.rev_list_first_parent(head):
            if marker and sha == marker:
                found_marker = True
                break
            if len(self.backend.parents_of(sha)) > 1:
                raise ValueError(f"inbox commit {sha[:10]} is not a linear log entry")
            raw = self.backend.repo.git.log("-1", "--format=%B", sha)
            try:
                msg = json.loads(
                    raw,
                    object_pairs_hook=no_duplicates,
                    parse_constant=no_constant,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"inbox commit {sha[:10]} is corrupt: {exc}") from exc
            if not isinstance(msg, dict) or set(msg) != {"sender", "at", "body"}:
                raise ValueError(f"inbox commit {sha[:10]} has an invalid envelope")
            if not isinstance(msg["sender"], str) or not isinstance(msg["at"], str):
                raise ValueError(f"inbox commit {sha[:10]} has invalid envelope identity")
            if not isinstance(msg["body"], dict):
                raise ValueError(f"inbox commit {sha[:10]} body is not an object")
            msg["id"] = sha
            out.append(msg)
        if not found_marker:
            raise ValueError(
                f"inbox cursor {marker[:10] if marker else 'unknown'} is not in {branch} history"
            )
        return list(reversed(out))

    def mark_inbox_seen(self, branch: str, message_id: str) -> None:
        """Advance a branch's inbox cursor through ``message_id``.

        The inbox is a linear, oldest-to-newest log. Acknowledging one message
        acknowledges everything before it, so the cursor may only move forward
        along that exact chain. The update is compare-and-swap checked rather
        than best-effort: a caller must never believe a message was consumed
        when the durable marker says otherwise.
        """
        branch = _check_name(branch, "branch")
        inbox_ref = f"{self.INBOX_REF}/{self.session}/{branch}"
        seen_ref = f"{self.SEEN_REF}/{self.session}/{branch}"
        with self.backend.lock():
            head = self.backend.ref_sha(inbox_ref)
            if head is None or not self.backend.is_ancestor(message_id, head):
                raise ValueError(f"message {message_id[:10]} is not in the inbox for {branch}")
            current = self.backend.ref_sha(seen_ref)
            if current == message_id:
                return
            if current is not None:
                if not self.backend.is_ancestor(current, head):
                    raise ValueError(f"current seen marker for {branch} is not in its inbox")
                if self.backend.is_ancestor(message_id, current):
                    raise ValueError(
                        f"acknowledging {message_id[:10]} would rewind {branch}'s inbox"
                    )
                if not self.backend.is_ancestor(current, message_id):
                    raise ValueError(
                        f"message {message_id[:10]} does not follow {branch}'s seen marker"
                    )
            if not self.backend.cas_update_ref(seen_ref, message_id, current):
                raise StaleBranch(f"inbox seen marker for {branch} moved while acknowledging")

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
            # Most recent adoption wins when a path was adopted more than once.
            # Crossing branches is safe only when the source record pins these
            # exact bytes; old source records without a blob remain readable
            # but cannot establish the origin of a current value.
            for src in reversed(s.meta.sources):
                if src.as_path != path or src.blob is None or src.blob != target:
                    continue
                upstream = self.state(src.state)
                if src.session and upstream.meta.session != src.session:
                    continue
                if upstream.blob(src.path) != src.blob:
                    continue
                return self.origin(upstream, src.path) or upstream
        return origin

    # ---------------------------------------------------------- discovery

    def catalog(self, *, branches: Iterable[str] | None = None) -> list[Hit]:
        """Everything published on every branch: a brain's first look around.

        Orientation before search: a brain that does not yet know the right
        word cannot ask for it, so the whole index is small enough to read.
        Each hit pins the branch head and manifest blob observed during this
        call; a producer moving afterwards does not mutate an existing hit.
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
