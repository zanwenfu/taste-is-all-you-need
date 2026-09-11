"""Deliver one pinned worker state without merging the worker's private state.

A worker branch is not a product branch.  Besides the requested artifact it
contains the accepted assignment, SDK transcript mirrors, runtime reports and
possibly exploratory edits.  Merging that branch directly makes all of those
part of the integration tree and lets an irrelevant conflict in one of them
block useful work.

This module turns an exact worker state into a narrow delivery commit.  The
commit starts from the assignment's exact integration base and overlays only
the explicitly named artifact paths from the pinned state.  It is descended
from the worker state so the original evidence remains reachable, but its tree
and transcript contain no worker-private material.  The ordinary memstore
typed merge then owns all genuine product conflicts.

Delivery ids name durable, immutable operations.  A dedicated claim ref moves
only after its git note records the source, target and byte-exact projection;
only then may the deterministic delivery branch be seeded.  A second outcome
note closes the merge-ref/worktree-reset crash window.  Retrying the same
operation is a no-op; reusing an id for different work is rejected rather than
guessed at.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Any

from git.exc import GitCommandError

from taste.memstore import (
    Branch,
    Conflict,
    Manifest,
    MergeResult,
    NoSuchState,
    State,
    Store,
    Transcript,
)
from taste.memstore.backend import EMPTY_TREE
from taste.memstore.objects import bind_conflict_reason, now_iso

__all__ = [
    "DeliveryIdentityConflict",
    "DeliveryRecord",
    "DeliveryRecoveryRequired",
    "DeliveryResult",
    "InvalidArtifactPath",
    "ProjectionEntry",
    "ReservedArtifactPath",
    "deliver_product",
    "is_control_path",
    "validate_artifact_path",
]


DELIVERY_SCHEMA = "taste.delivery/1"
DELIVERY_NOTES = "refs/notes/taste/delivery"
DELIVERY_IDENTITY_SCHEMA = "taste.delivery-identity/1"
DELIVERY_IDENTITY_NOTES = "refs/notes/taste/delivery-identity"
DELIVERY_IDENTITY_REF = "refs/taste/delivery-identity"
DELIVERY_PROJECTION_REF = "refs/taste/delivery-projection"
DELIVERY_OUTCOME_SCHEMA = "taste.delivery-outcome/1"
DELIVERY_OUTCOME_NOTES = "refs/notes/taste/delivery-outcome"
_EXACT_STATE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")

# Root-owned records and private directories used by the worker host.  An
# explicit allow-list is still the primary boundary; this deny-list makes it
# impossible for a mistaken Assignment to promote known plumbing as product.
_CONTROL_FILES = frozenset(
    {
        ".claude",
        ".taste",
        "assignment.json",
        "contract.json",
        "monitor-state.json",
        "runtime-state.json",
        "sdk-sessions",
        "taste-runtime",
        "worker-report.json",
        "worker-state.json",
    }
)
class DeliveryIdentityConflict(ValueError):
    """A delivery id already names different source, target, or bytes."""


class InvalidArtifactPath(ValueError):
    """An artifact path is not one normalized repository-relative file."""


class ReservedArtifactPath(InvalidArtifactPath):
    """A requested artifact path belongs to worker/runtime control state."""


class DeliveryRecoveryRequired(RuntimeError):
    """A published merge cannot be acknowledged without risking dirty work."""


@dataclass(frozen=True, slots=True)
class ProjectionEntry:
    """The exact source entry selected for one artifact path.

    ``blob=None`` is an intentional deletion.  The git mode is part of the
    identity because turning an executable into a regular file (or a symlink
    into file contents) is a meaningful product change.
    """

    path: str
    blob: str | None
    mode: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "blob": self.blob, "mode": self.mode}

    @classmethod
    def from_dict(cls, raw: Any) -> ProjectionEntry:
        if not isinstance(raw, dict) or set(raw) != {"path", "blob", "mode"}:
            raise ValueError("invalid delivery projection entry")
        path = validate_artifact_path(raw["path"])
        blob, mode = raw["blob"], raw["mode"]
        if blob is not None and not isinstance(blob, str):
            raise ValueError("delivery projection blob must be a string or null")
        if mode is not None and not isinstance(mode, str):
            raise ValueError("delivery projection mode must be a string or null")
        if (blob is None) != (mode is None):
            raise ValueError("delivery projection blob and mode must both be present or absent")
        return cls(path=path, blob=blob, mode=mode)


@dataclass(frozen=True, slots=True)
class DeliveryRecord:
    """Durable identity of one product projection."""

    delivery_id: str
    session: str
    source_state_id: str
    base_state_id: str
    target_branch: str
    projection_state_id: str
    projection_tree_id: str
    entries: tuple[ProjectionEntry, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema": DELIVERY_SCHEMA,
                "delivery_id": self.delivery_id,
                "session": self.session,
                "source_state_id": self.source_state_id,
                "base_state_id": self.base_state_id,
                "target_branch": self.target_branch,
                "projection_state_id": self.projection_state_id,
                "projection_tree_id": self.projection_tree_id,
                "entries": [entry.to_dict() for entry in self.entries],
            },
            indent=1,
            sort_keys=True,
        ) + "\n"

    @classmethod
    def from_json(cls, text: str) -> DeliveryRecord:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("delivery record is not valid JSON") from exc
        required = {
            "schema",
            "delivery_id",
            "session",
            "source_state_id",
            "base_state_id",
            "target_branch",
            "projection_state_id",
            "projection_tree_id",
            "entries",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("delivery record has missing or unknown fields")
        if raw["schema"] != DELIVERY_SCHEMA:
            raise ValueError(f"unsupported delivery schema {raw['schema']!r}")
        strings = required - {"schema", "entries"}
        if any(not isinstance(raw[field], str) for field in strings):
            raise ValueError("delivery record identity fields must be strings")
        if not isinstance(raw["entries"], list):
            raise ValueError("delivery record entries must be an array")
        return cls(
            delivery_id=raw["delivery_id"],
            session=raw["session"],
            source_state_id=raw["source_state_id"],
            base_state_id=raw["base_state_id"],
            target_branch=raw["target_branch"],
            projection_state_id=raw["projection_state_id"],
            projection_tree_id=raw["projection_tree_id"],
            entries=tuple(ProjectionEntry.from_dict(entry) for entry in raw["entries"]),
        )


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    """The immutable projection and its integration outcome."""

    record: DeliveryRecord
    projection: State
    integration_state: State | None
    conflicts: tuple[Conflict, ...] = ()
    reused_projection: bool = False
    reused_outcome: bool = False

    @property
    def ok(self) -> bool:
        return self.integration_state is not None and not self.conflicts


@dataclass(frozen=True, slots=True)
class _PinnedView:
    """The two read-only fields merge_branches needs, fixed to one state."""

    name: str
    state: State

    @property
    def head(self) -> State:
        return self.state


def _pinned(state: State) -> _PinnedView:
    return _PinnedView(name=state.meta.branch, state=state)


def is_control_path(path: str) -> bool:
    """Whether ``path`` is worker plumbing that cannot be delivered."""
    return any(path == root or path.startswith(f"{root}/") for root in _CONTROL_FILES)


def validate_artifact_path(value: str) -> str:
    """Validate one product path shared by Assignment input/output checks.

    Cross-brain inputs and outputs use the same policy: both must be normalized
    repository-relative paths outside git metadata and worker control state.
    The returned string is unchanged, so validation never aliases two paths.
    """
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise InvalidArtifactPath("artifact paths must be non-empty POSIX paths")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or value.endswith("/")
        or value != parsed.as_posix()
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or any(part.casefold() == ".git" for part in parsed.parts)
    ):
        raise InvalidArtifactPath(f"artifact path {value!r} is not normalized and relative")
    if is_control_path(value):
        raise ReservedArtifactPath(f"worker control path {value!r} cannot be delivered")
    return value


def _delivery_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 512
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("delivery_id must be a non-empty string of at most 512 characters")
    return value


def _state(store: Store, state_id: str, what: str) -> State:
    if not isinstance(state_id, str) or _EXACT_STATE.fullmatch(state_id) is None:
        raise NoSuchState(f"{what} must be a full, lowercase state id")
    state = store.state(state_id)
    # Loading both forces the state and its annotated identity to exist.  A
    # bare git commit is not a memstore State and cannot be a delivery input.
    _ = state.meta
    _ = store.backend.tree_of(state.id)
    if state.meta.session != store.session:
        raise NoSuchState(f"{what} belongs to session {state.meta.session!r}, not {store.session!r}")
    return state


def _branch_name(delivery_id: str) -> str:
    digest = hashlib.sha256(delivery_id.encode("utf-8")).hexdigest()
    return f"delivery.{digest}"


def _projection_ref(store: Store, delivery_id: str) -> str:
    digest = hashlib.sha256(delivery_id.encode("utf-8")).hexdigest()
    return f"{DELIVERY_PROJECTION_REF}/{store.session}/{digest}"


def _merge_bases(store: Store, left: str, right: str) -> tuple[str, ...]:
    """All best common ancestors, not git's arbitrary single representative."""
    try:
        raw = store.backend.repo.git.merge_base("--all", left, right)
    except GitCommandError:
        return ()
    return tuple(sorted(line for line in raw.splitlines() if line))


def _entries(
    store: Store, base: State, source: State, paths: tuple[str, ...]
) -> tuple[ProjectionEntry, ...]:
    backend = store.backend
    out: list[ProjectionEntry] = []
    for path in paths:
        entry = backend.entry_at(source.id, path)
        base_entry = backend.entry_at(base.id, path)
        if any(
            candidate is not None and candidate.mode == "040000"
            for candidate in (entry, base_entry)
        ):
            raise InvalidArtifactPath(
                f"artifact path {path!r} names a directory; select its product files explicitly"
            )
        out.append(
            ProjectionEntry(
                path=path,
                blob=entry.sha if entry is not None else None,
                mode=entry.mode if entry is not None else None,
            )
        )
    return tuple(out)


def _projection_tree(store: Store, base: State, source: State, entries: tuple[ProjectionEntry, ...]) -> str:
    """Build a tree in a throwaway index, never a mutable branch index."""
    backend = store.backend
    index = backend.common_dir / (
        f"memstore.delivery-index.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
    )
    try:
        with backend.repo.git.custom_environment(GIT_INDEX_FILE=str(index)):
            backend.repo.git.read_tree(backend.tree_of(base.id))
            # A base created by an older runtime may itself contain control
            # files.  Strip those too; the integration side is sanitized to
            # the same product-only boundary before merging.
            for path in sorted(set(base.files()) | set(source.files())):
                if is_control_path(path):
                    backend.repo.git.update_index("--force-remove", "--", path)
            for entry in entries:
                if entry.blob is None:
                    backend.repo.git.update_index("--force-remove", "--", entry.path)
                else:
                    backend.repo.git.update_index(
                        "--add", "--cacheinfo", f"{entry.mode},{entry.blob},{entry.path}"
                    )
            return backend.repo.git.write_tree()
    finally:
        with contextlib.suppress(FileNotFoundError):
            index.unlink()


def _identity_json(
    *,
    delivery_id: str,
    store: Store,
    source: State,
    base: State,
    target: Branch,
    tree: str,
    entries: tuple[ProjectionEntry, ...],
) -> str:
    """Canonical full intent written before any delivery branch is public."""
    return json.dumps(
        {
            "schema": DELIVERY_IDENTITY_SCHEMA,
            "delivery_id": delivery_id,
            "session": store.session,
            "source_state_id": source.id,
            "base_state_id": base.id,
            "target_branch": target.name,
            "projection_tree_id": tree,
            "entries": [entry.to_dict() for entry in entries],
        },
        indent=1,
        sort_keys=True,
    ) + "\n"


def _identity_ref(store: Store, delivery_id: str) -> str:
    digest = hashlib.sha256(delivery_id.encode("utf-8")).hexdigest()
    return f"{DELIVERY_IDENTITY_REF}/{store.session}/{digest}"


def _claim_identity(store: Store, delivery_id: str, expected: str) -> None:
    """Publish a full immutable intent in one compare-and-swap.

    The note is attached before the claim ref moves.  Therefore a crash sees
    either no public claim or a claim whose complete source/target/projection
    identity is already readable; there is no source-only seed ambiguity.
    """
    backend = store.backend
    ref = _identity_ref(store, delivery_id)

    def check(commit: str) -> None:
        if backend.note_get(DELIVERY_IDENTITY_NOTES, commit) != expected:
            raise DeliveryIdentityConflict(
                f"delivery id {delivery_id!r} was already claimed for different work"
            )

    with backend.lock():
        current = backend.ref_sha(ref)
        if current is not None:
            check(current)
            return
        digest = hashlib.sha256(expected.encode("utf-8")).hexdigest()
        candidate = backend.commit_tree(
            EMPTY_TREE,
            [],
            f"delivery identity claim\n\nTaste-Delivery-Identity: {digest}\n",
        )
        backend.note_set(
            DELIVERY_IDENTITY_NOTES,
            candidate,
            expected,
            overwrite=False,
        )
        if backend.cas_update_ref(ref, candidate, None):
            return
        winner = backend.ref_sha(ref)
        if winner is None:
            raise DeliveryRecoveryRequired(
                f"delivery identity ref for {delivery_id!r} changed without a readable winner"
            )
        check(winner)


def _manifest(source: State, paths: frozenset[str]) -> Manifest:
    return Manifest(
        {
            name: entry
            for name, entry in source.manifest.entries.items()
            if entry.path in paths and not is_control_path(entry.path)
        }
    )


def _expected_matches(
    record: DeliveryRecord,
    *,
    delivery_id: str,
    store: Store,
    source: State,
    base: State,
    target: Branch,
    tree: str,
    entries: tuple[ProjectionEntry, ...],
) -> bool:
    return (
        record.delivery_id == delivery_id
        and record.session == store.session
        and record.source_state_id == source.id
        and record.base_state_id == base.id
        and record.target_branch == target.name
        and record.projection_tree_id == tree
        and record.entries == entries
    )


def _load_projection(
    store: Store,
    branch_name: str,
    *,
    delivery_id: str,
    source: State,
    base: State,
    target: Branch,
    tree: str,
    entries: tuple[ProjectionEntry, ...],
) -> tuple[DeliveryRecord, State] | None:
    def validate(state: State) -> tuple[DeliveryRecord, State]:
        raw = store.backend.note_get(DELIVERY_NOTES, state.id)
        if raw is None:
            raise DeliveryIdentityConflict(
                f"delivery id {delivery_id!r} has a projection ref without its durable record"
            )
        try:
            record = DeliveryRecord.from_json(raw)
        except ValueError as exc:
            raise DeliveryIdentityConflict(
                f"delivery id {delivery_id!r} has an unreadable durable record"
            ) from exc
        if (
            not _expected_matches(
                record,
                delivery_id=delivery_id,
                store=store,
                source=source,
                base=base,
                target=target,
                tree=tree,
                entries=entries,
            )
            or record.projection_state_id != state.id
            or store.backend.tree_of(state.id) != record.projection_tree_id
            or state.meta.branch != branch_name
        ):
            raise DeliveryIdentityConflict(
                f"delivery id {delivery_id!r} was already used for a different projection"
            )
        return record, state

    pinned_id = store.backend.ref_sha(_projection_ref(store, delivery_id))
    if pinned_id is not None:
        return validate(store.state(pinned_id))

    view = store.view(branch_name)
    if not view.exists():
        return None
    head = view.head
    raw = store.backend.note_get(DELIVERY_NOTES, head.id)
    if raw is None:
        # The one legitimate note-free state is the seed published before the
        # projection is built.  It is safe to resume only when it is still the
        # exact seed for this source.  Anything else is identity reuse.
        if (
            head.meta.kind == "branch"
            and head.meta.branch == branch_name
            and head.meta.parents == (source.id,)
            and store.backend.tree_of(head.id) == store.backend.tree_of(source.id)
        ):
            return None
        raise DeliveryIdentityConflict(
            f"delivery id {delivery_id!r} already names an unrecognized branch state"
        )
    return validate(head)


def _publish_projection(
    store: Store,
    branch_name: str,
    *,
    delivery_id: str,
    source: State,
    base: State,
    target: Branch,
    tree: str,
    entries: tuple[ProjectionEntry, ...],
) -> tuple[DeliveryRecord, State]:
    branch = store.branch(branch_name, from_state=source, producer="delivery")
    try:
        head = branch.head
        if not (
            head.meta.kind == "branch"
            and head.meta.parents == (source.id,)
            and store.backend.tree_of(head.id) == store.backend.tree_of(source.id)
        ):
            raise DeliveryIdentityConflict(
                f"delivery id {delivery_id!r} already names a different incomplete projection"
            )
        built = branch._build(
            tree=tree,
            parents=[head.id],
            kind="checkpoint",
            reason=f"product projection for delivery {delivery_id}",
            manifest=_manifest(source, frozenset(entry.path for entry in entries)),
            # The evidence remains reachable through the parent, but private
            # worker conversation is not copied into the delivery state.
            transcript=Transcript(),
            transcript_base=None,
            verdict=None,
            attempt=0,
            expected_head=head.id,
        )
        record = DeliveryRecord(
            delivery_id=delivery_id,
            session=store.session,
            source_state_id=source.id,
            base_state_id=base.id,
            target_branch=target.name,
            projection_state_id=built.sha,
            projection_tree_id=tree,
            entries=entries,
        )
        # The note is durable before the ref move.  A crash before publication
        # leaves only an unreachable candidate; a crash after it leaves a
        # complete, self-identifying operation.
        store.backend.note_set(DELIVERY_NOTES, built.sha, record.to_json(), overwrite=False)
        pinned_ref = _projection_ref(store, delivery_id)
        if not store.backend.cas_update_ref(pinned_ref, built.sha, None):
            winner = store.backend.ref_sha(pinned_ref)
            if winner != built.sha:
                raise DeliveryIdentityConflict(
                    f"delivery id {delivery_id!r} already pinned a different projection state"
                )
        projection = branch.publish_state(built)
        branch.backend.reset_hard_to_head()
        return record, projection
    finally:
        branch.close()


def _outcome_json(record: DeliveryRecord, merge_state: State) -> str:
    return json.dumps(
        {
            "schema": DELIVERY_OUTCOME_SCHEMA,
            "delivery_id": record.delivery_id,
            "session": record.session,
            "target_branch": record.target_branch,
            "projection_state_id": record.projection_state_id,
            "merge_state_id": merge_state.id,
            "status": "merged",
        },
        indent=1,
        sort_keys=True,
    ) + "\n"


def _mark_merged(store: Store, record: DeliveryRecord, merge_state: State) -> None:
    """Acknowledge success only after merge_branches reset the worktree."""
    store.backend.note_set(
        DELIVERY_OUTCOME_NOTES,
        merge_state.id,
        _outcome_json(record, merge_state),
        overwrite=False,
    )


def _recover_published_merge(target: Branch, merge_state: State) -> None:
    """Close merge_branches' ref-move -> worktree-reset crash window.

    A dirty tree equal to the first parent could be stale pre-reset bytes *or*
    a later writer deliberately undoing the merge.  Content equality cannot
    establish provenance across a process death.  Preserve it as a state and
    fail closed; only a clean tree proves that reset completed.
    """
    if not target.is_dirty():
        return
    dirty = tuple(target.dirty_paths())
    captured = target.checkpoint(
        f"capture ambiguous worktree after unacknowledged delivery merge {merge_state.id}"
    )
    raise DeliveryRecoveryRequired(
        f"delivery merge {merge_state.id[:10]} is published but reset completion is unknown; "
        f"preserved dirty paths {dirty!r} as state {captured.id[:10]} instead of resetting"
    )


def _selected_entries_hold(target: Branch, record: DeliveryRecord) -> bool:
    """Whether the current target still contains the exact delivered bytes."""
    head = target.head
    for expected in record.entries:
        actual = target.backend.entry_at(head.id, expected.path)
        if expected.blob is None:
            if actual is not None:
                return False
        elif (
            actual is None
            or actual.sha != expected.blob
            or actual.mode != expected.mode
        ):
            return False
    return True


def _delivery_conflict_reason(
    delivery_id: str,
    conflicts: tuple[Conflict, ...],
) -> str:
    # ``reason`` is the immutable git commit subject as well as a Meta field.
    # Binding the sidecar conflict bytes here makes a later note rewrite
    # detectable even if the mutable Meta note is rewritten with it.
    return bind_conflict_reason(
        f"merge conflict: deliver product {delivery_id}",
        conflicts,
    )


def _prior_outcome(
    store: Store,
    target: Branch,
    record: DeliveryRecord,
    projection: State,
) -> MergeResult | None:
    history = target.history()
    direct_merge: State | None = None
    prior_conflict: State | None = None
    for state in history:
        if state.meta.kind == "merge" and projection.id in state.meta.parents[1:]:
            raw = store.backend.note_get(DELIVERY_OUTCOME_NOTES, state.id)
            if raw is not None:
                if raw != _outcome_json(record, state):
                    raise DeliveryIdentityConflict(
                        f"merge state {state.id[:10]} has a different delivery outcome identity"
                    )
                if not _selected_entries_hold(target, record):
                    raise DeliveryRecoveryRequired(
                        f"previously delivered product {record.delivery_id!r} no longer "
                        "matches the current target bytes"
                    )
                return MergeResult(target.head)
            if direct_merge is None:
                direct_merge = state
        elif (
            prior_conflict is None
            and state.meta.kind == "conflict"
            and state.meta.merged == projection.id
        ):
            prior_conflict = state

    if direct_merge is not None:
        if direct_merge.id != target.head.id:
            raise DeliveryRecoveryRequired(
                f"delivery merge {direct_merge.id[:10]} was published without a completed "
                "worktree acknowledgement and target history has advanced"
            )
        _recover_published_merge(target, direct_merge)
        _mark_merged(store, record, direct_merge)
        return MergeResult(target.head)

    if target.backend.is_ancestor(projection.id, target.head.id):
        raise DeliveryRecoveryRequired(
            "projection is in target ancestry without a delivery merge/outcome record"
        )
    if prior_conflict is not None:
        try:
            conflicts = tuple(prior_conflict.conflicts)
        except Exception as exc:
            raise DeliveryRecoveryRequired(
                f"delivery conflict {prior_conflict.id[:10]} lost its durable conflict evidence"
            ) from exc
        if not conflicts:
            raise DeliveryRecoveryRequired(
                f"delivery conflict {prior_conflict.id[:10]} lost its durable conflict evidence"
            )
        expected_subject = _delivery_conflict_reason(record.delivery_id, conflicts)
        try:
            actual_subject = target.backend.repo.git.log(
                "-1", "--format=%s", prior_conflict.id
            )
        except GitCommandError as exc:
            raise DeliveryRecoveryRequired(
                f"delivery conflict {prior_conflict.id[:10]} has no readable immutable binding"
            ) from exc
        if actual_subject != expected_subject:
            raise DeliveryRecoveryRequired(
                f"delivery conflict {prior_conflict.id[:10]} does not match its "
                "immutable conflict binding"
            )
        return MergeResult(None, conflicts)
    return None


def _strip_target_controls(target: Branch, delivery_id: str) -> None:
    """Make the target obey the same product-only boundary without loss."""
    if target.is_dirty():
        target.checkpoint(f"capture target before delivery {delivery_id}")
    private = [path for path in target.head.files() if is_control_path(path)]
    # Let git remove its own tracked entries.  Unlike Path.unlink(), this is
    # correct for symlinks, submodules and a future control directory stored
    # as a single tree entry.  Chunking avoids an argument-limit failure for a
    # worker with a long mirrored SDK transcript.
    for start in range(0, len(private), 128):
        target.backend.repo.git.rm("-r", "-f", "--", *private[start : start + 128])
    if private:
        target.checkpoint(f"strip private control paths before delivery {delivery_id}")


def _absence_conflicts(
    target: Branch,
    base: State,
    projection: State,
    entries: tuple[ProjectionEntry, ...],
) -> tuple[Conflict, ...]:
    """Represent target-only additions against an explicit absence.

    Git has no deletion delta when a path was absent in both base and worker,
    so a concurrent target addition would otherwise win silently.  Selecting
    a missing path is an explicit final-state requirement; record the clash as
    the same durable conflict value used by the ordinary typed merge.
    """
    ours = target.head
    union = ours.manifest.union(projection.manifest)
    at = now_iso()
    return tuple(
        Conflict(
            path=entry.path,
            type=union.type_of(entry.path),
            base=None,
            ours=ours.blob(entry.path),
            theirs=None,
            detail="target added path after assignment base; delivery requires absence",
            theirs_branch=projection.meta.branch,
            theirs_state=projection.id,
            ours_state=ours.id,
            at=at,
        )
        for entry in entries
        if entry.blob is None
        and base.blob(entry.path) is None
        and ours.blob(entry.path) is not None
    )


def _probe_typed_conflicts(
    store: Store,
    target: Branch,
    projection: State,
    delivery_id: str,
) -> tuple[Conflict, ...]:
    """Run the ordinary typed merge on an isolated branch.

    An explicit-absence conflict must block the whole delivery, but it must
    not hide another product conflict.  The memstore merge rules are the sole
    authority for those conflicts, so probe them against an exact target
    snapshot instead of reimplementing the rules here.
    """
    ours = target.head
    probe_name = f"delivery-probe.{uuid.uuid4().hex}"
    probe = store.branch(probe_name, from_state=ours, producer="delivery-probe")
    try:
        result = probe.merge(_pinned(projection), reason=f"probe {delivery_id}")
        return tuple(
            replace(
                conflict,
                ours_state=ours.id,
                theirs_branch=projection.meta.branch,
                theirs_state=projection.id,
            )
            for conflict in result.conflicts
        )
    finally:
        probe.close()
        # The ref remains as durable evidence of what the typed rules saw; the
        # mechanical probe needs no persistent worktree or lease.
        store.remove_branch(probe_name)


def _publish_delivery_conflicts(
    target: Branch,
    projection: State,
    conflicts: tuple[Conflict, ...],
    delivery_id: str,
) -> MergeResult:
    ours = target.head
    built = target._build(
        tree=target.backend.tree_of(ours.id),
        parents=[ours.id],
        kind="conflict",
        reason=_delivery_conflict_reason(delivery_id, conflicts),
        manifest=ours.manifest,
        transcript=None,
        transcript_base=ours,
        verdict=None,
        attempt=0,
        expected_head=ours.id,
        merged=projection.id,
        conflicts=list(conflicts),
    )
    target.publish_state(built)
    return MergeResult(None, conflicts)


def deliver_product(
    store: Store,
    target: Branch,
    *,
    delivery_id: str,
    final_state_id: str,
    base_state_id: str,
    artifact_paths: Iterable[str],
) -> DeliveryResult:
    """Serialize one complete delivery decision against the target worktree.

    Individual Branch mutations already share this re-entrant lock.  Delivery
    must hold it across its reads as well: in particular, an explicit-absence
    check and the merge that follows are one compare-and-act boundary.
    """
    if not isinstance(target, Branch):
        raise TypeError("target must be a writable Branch")
    with target._mutation_lock:
        return _deliver_product_locked(
            store,
            target,
            delivery_id=delivery_id,
            final_state_id=final_state_id,
            base_state_id=base_state_id,
            artifact_paths=artifact_paths,
        )


def _deliver_product_locked(
    store: Store,
    target: Branch,
    *,
    delivery_id: str,
    final_state_id: str,
    base_state_id: str,
    artifact_paths: Iterable[str],
) -> DeliveryResult:
    """Project artifacts from one exact state and merge them into ``target``.

    ``base_state_id`` is the exact integration base recorded by the worker's
    Assignment.  Requiring it prevents the meaning of a projection from
    changing as the target branch advances.  ``artifact_paths`` is an
    allow-list, not a filter over the current worker head: bytes are read only
    from ``final_state_id`` and a missing selected path represents deletion.
    """
    delivery_id = _delivery_id(delivery_id)
    if target.store.session != store.session or target.store.backend.common_dir != store.backend.common_dir:
        raise ValueError("target branch belongs to a different memstore session")
    source = _state(store, final_state_id, "final_state_id")
    base = _state(store, base_state_id, "base_state_id")
    if not store.backend.is_ancestor(base.id, source.id):
        raise ValueError("base_state_id is not an ancestor of final_state_id")
    if not target.backend.is_ancestor(base.id, target.head.id):
        raise ValueError("target branch does not descend from base_state_id")

    paths = tuple(sorted(validate_artifact_path(path) for path in artifact_paths))
    if not paths:
        raise ValueError("artifact_paths must contain at least one product path")
    if len(paths) != len(set(paths)):
        raise InvalidArtifactPath("artifact_paths contains a duplicate path")
    entries = _entries(store, base, source, paths)
    tree = _projection_tree(store, base, source, entries)
    branch_name = _branch_name(delivery_id)
    if branch_name == target.name:
        raise DeliveryIdentityConflict("delivery branch collides with its target branch")
    identity = _identity_json(
        delivery_id=delivery_id,
        store=store,
        source=source,
        base=base,
        target=target,
        tree=tree,
        entries=entries,
    )
    claim_exists = store.backend.ref_sha(_identity_ref(store, delivery_id)) is not None
    if not claim_exists:
        # A branch without the earlier full-intent claim can only come from an
        # interrupted/older implementation.  Its seed records the source but
        # not the target or selected paths, so adopting it would permit silent
        # identity reuse.  Fail closed; the immutable worker state remains.
        if store.view(branch_name).exists():
            raise DeliveryIdentityConflict(
                f"delivery id {delivery_id!r} has an ambiguous unclaimed delivery branch"
            )
        if _merge_bases(store, source.id, target.head.id) != (base.id,):
            raise ValueError(
                "base_state_id is not the unique exact worker/target merge base; "
                "the worker topology is advanced or ambiguous relative to its assignment base"
            )
    _claim_identity(store, delivery_id, identity)

    loaded = _load_projection(
        store,
        branch_name,
        delivery_id=delivery_id,
        source=source,
        base=base,
        target=target,
        tree=tree,
        entries=entries,
    )
    reused_projection = loaded is not None
    if loaded is None:
        record, projection = _publish_projection(
            store,
            branch_name,
            delivery_id=delivery_id,
            source=source,
            base=base,
            target=target,
            tree=tree,
            entries=entries,
        )
    else:
        record, projection = loaded

    prior = _prior_outcome(store, target, record, projection)
    if prior is not None:
        return DeliveryResult(
            record=record,
            projection=projection,
            integration_state=prior.state,
            conflicts=prior.conflicts,
            reused_projection=reused_projection,
            reused_outcome=True,
        )

    # Parentage controls three-way merge semantics.  If somebody merged the
    # raw worker lineage into the target after the projection was prepared,
    # git would use that newer worker state as the base and interpret the
    # projection's removal of exploratory/private files as a new deletion.
    # Refuse that topology rather than broaden this delivery silently.
    if _merge_bases(store, projection.id, target.head.id) != (base.id,):
        raise ValueError(
            "target topology changed after projection; it no longer has base_state_id as its "
            "unique exact merge base"
        )

    _strip_target_controls(target, delivery_id)
    absence_conflicts = _absence_conflicts(target, base, projection, entries)
    if absence_conflicts:
        typed_conflicts = _probe_typed_conflicts(store, target, projection, delivery_id)
        all_conflicts = tuple(
            sorted((*typed_conflicts, *absence_conflicts), key=lambda conflict: conflict.path)
        )
        merged = _publish_delivery_conflicts(
            target,
            projection,
            all_conflicts,
            delivery_id,
        )
    else:
        merged = target.merge(_pinned(projection), reason=f"deliver product {delivery_id}")
    if merged.state is not None:
        if len(merged.state.meta.parents) < 2 or merged.state.meta.parents[1] != projection.id:
            raise DeliveryRecoveryRequired(
                "published delivery merge does not have the pinned projection as second parent"
            )
        _mark_merged(store, record, merged.state)
    return DeliveryResult(
        record=record,
        projection=projection,
        integration_state=merged.state,
        conflicts=merged.conflicts,
        reused_projection=reused_projection,
        reused_outcome=False,
    )
