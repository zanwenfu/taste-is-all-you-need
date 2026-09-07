"""The memory layer: version control designed for agent memory, on git.

Public surface:

    Store.open(root, session)     a session's memory
    store.branch(name)            one brain's execution context, own working tree
    branch.checkpoint(...)        artifacts + transcript + reason, atomically
    branch.rollback(state, ...)   an append; nothing is lost
    branch.publish(...)           make an artifact findable
    store.search("...")           does anyone have X?
    branch.merge(other, ...)      typed merge; conflicts are values
    store.provenance(state)       where did this come from
"""

from taste.memstore.objects import (
    SCHEMA,
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
    Transcript,
    Verdict,
)
from taste.memstore.store import Branch, DiffEntry, Hit, MergeResult, State, Store, TypedDiff

__all__ = [
    "SCHEMA",
    "Branch",
    "BranchBusy",
    "Conflict",
    "DiffEntry",
    "Hit",
    "Manifest",
    "ManifestEntry",
    "MergeResult",
    "Meta",
    "NoSuchState",
    "NotAnAncestor",
    "ObjectType",
    "PublishError",
    "StaleBranch",
    "State",
    "Store",
    "Transcript",
    "TypedDiff",
    "Verdict",
]
