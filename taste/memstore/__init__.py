"""The memory layer: version control designed for agent memory, on git.

Public surface:

    Store.open(root, session)     a session's memory
    store.branch(name)            one brain's execution context, own working tree
    branch.checkpoint(...)        artifacts + transcript + reason, atomically
    branch.rollback(state, ...)   an append; nothing is lost
    branch.publish(...)           make an artifact findable
    store.catalog()               everything published, for orientation
    store.search("...")           does anyone have X?
    store.search_contents("...")  where was X said or written, in any state?
    store.fetch(pointer)          the exact bytes a hit points at
    store.view(name)              read a branch without taking its lease
    store.judge(state, verdict)   a monitor records a judgment
    store.send(branch, message)   ask another brain for something
    branch.adopt(hit)             take an artifact, recording where it came from
    branch.resume()               what a brain needs on waking
    branch.merge(other, ...)      typed merge; conflicts are values
    store.provenance(state)       where did this come from
"""

from taste.memstore.objects import (
    SCHEMA,
    BadName,
    BadPointer,
    BranchBusy,
    CommitHeader,
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
    SchemaMismatch,
    Source,
    StaleBranch,
    Transcript,
    UnsafeCapture,
    Verdict,
    WorktreeUnavailable,
)
from taste.memstore.store import (
    Branch,
    BranchView,
    Built,
    ContentHit,
    DiffEntry,
    Fetched,
    Hit,
    MergeResult,
    Pointer,
    State,
    Store,
    TypedDiff,
)

__all__ = [
    "SCHEMA",
    "BadName",
    "BadPointer",
    "Branch",
    "BranchBusy",
    "BranchView",
    "Built",
    "CommitHeader",
    "Conflict",
    "ContentHit",
    "DiffEntry",
    "Fetched",
    "ForeignHead",
    "Hit",
    "Manifest",
    "ManifestEntry",
    "MergeResult",
    "Meta",
    "NoSuchState",
    "NotAnAncestor",
    "ObjectType",
    "Pointer",
    "PublishError",
    "Resume",
    "SchemaMismatch",
    "Source",
    "StaleBranch",
    "State",
    "Store",
    "Transcript",
    "TypedDiff",
    "UnsafeCapture",
    "Verdict",
    "WorktreeUnavailable",
]
