"""Typed merge: the type of a path decides how two branches' versions combine.

Files merge three-way, the way git does. Records merge by key: the same key
changed to the same value on both sides is agreement, a key changed on one
side is taken, a key changed differently on both sides is a conflict for
that key only. Transcripts never merge: ours is kept, theirs stays on its
branch. Manifests are a union.

A conflict is not an exception. It is a ``Conflict`` value, returned to the
caller and attached as a note to our head, so a brain can find it later
and resolve it by writing the file and checkpointing.
"""

from __future__ import annotations

import json
from typing import Any

from taste.memstore.objects import Conflict, ObjectType
from taste.memstore.store import NOTES, Branch, MergeResult


def merge_records(base: Any, ours: Any, theirs: Any) -> tuple[Any, list[str]]:
    """Three-way merge of JSON values. Returns (merged, clashing keys).

    For objects, each key is judged on its own: unchanged on both sides is
    kept, changed on one side is taken (including a deletion), the same
    change on both sides is agreement, different changes are a clash for
    that key. Anything that is not an object on all three sides merges as
    one value.
    """
    if isinstance(base, dict) and isinstance(ours, dict) and isinstance(theirs, dict):
        merged: dict[str, Any] = {}
        clashes: list[str] = []
        for key in sorted(set(base) | set(ours) | set(theirs)):
            in_b, in_o, in_t = key in base, key in ours, key in theirs
            b, o, t = base.get(key), ours.get(key), theirs.get(key)
            o_changed = (in_o != in_b) or (in_o and o != b)
            t_changed = (in_t != in_b) or (in_t and t != b)
            if not o_changed and not t_changed:
                if in_b:
                    merged[key] = b
            elif o_changed and not t_changed:
                if in_o:
                    merged[key] = o
            elif t_changed and not o_changed:
                if in_t:
                    merged[key] = t
            elif in_o == in_t and (not in_o or o == t):
                if in_o:
                    merged[key] = o
            else:
                clashes.append(key)
                if in_o:
                    merged[key] = o
                elif in_t:
                    merged[key] = t
        return merged, clashes
    if ours == theirs:
        return ours, []
    if ours == base:
        return theirs, []
    if theirs == base:
        return ours, []
    return ours, ["<value>"]


def _load(text: str | None) -> Any:
    if text is None:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def merge_branches(ours: Branch, theirs: Branch, *, reason: str, resolved: bool = False) -> MergeResult:
    """Merge ``theirs`` into ``ours`` as a new state on ``ours``.

    With ``resolved=True`` the working tree of ``ours`` is taken as the
    resolution: a brain that was handed conflicts writes the files it wants
    and records the merge with both parents. No type rule runs.
    """
    ours_head, theirs_head = ours.head, theirs.head
    backend = ours.backend
    if backend.is_ancestor(theirs_head.id, ours_head.id):
        return MergeResult(ours_head)

    union = ours_head.manifest.union(theirs_head.manifest)
    if resolved:
        backend.stage_all()
        state = ours._commit(
            tree=backend.write_tree(),
            parents=[ours_head.id, theirs_head.id],
            kind="merge",
            reason=reason,
            manifest=union,
            transcript=ours_head.transcript,
            verdict=None,
            attempt=0,
            expected_head=ours_head.id,
        )
        return MergeResult(state)

    result = backend.merge_tree(ours_head.id, theirs_head.id)
    base = backend.merge_base(ours_head.id, theirs_head.id)
    tree = result.tree
    conflicts: list[Conflict] = []

    for path in result.conflicted_paths:
        kind = union.type_of(path)
        b_blob = backend.blob_at(base, path) if base else None
        o_blob = backend.blob_at(ours_head.id, path)
        t_blob = backend.blob_at(theirs_head.id, path)

        if kind is ObjectType.RECORD:
            b = _load(backend.cat_blob(b_blob) if b_blob else None)
            o = _load(backend.cat_blob(o_blob) if o_blob else None)
            t = _load(backend.cat_blob(t_blob) if t_blob else None)
            if b is None or o is None or t is None:
                conflicts.append(Conflict(path, kind, b_blob, o_blob, t_blob, "record is not valid JSON"))
                continue
            merged, clashes = merge_records(b, o, t)
            if clashes:
                conflicts.append(Conflict(path, kind, b_blob, o_blob, t_blob, "keys: " + ", ".join(clashes)))
                continue
            blob = backend.hash_blob(json.dumps(merged, indent=1, sort_keys=True) + "\n")
            tree = backend.tree_with_blob(tree, path, blob)
        elif kind is ObjectType.TRANSCRIPT:
            if o_blob:
                tree = backend.tree_with_blob(tree, path, o_blob)
        elif kind is ObjectType.NOTE:
            o_text = backend.cat_blob(o_blob) if o_blob else ""
            t_text = backend.cat_blob(t_blob) if t_blob else ""
            joined = o_text if t_text in o_text else o_text.rstrip("\n") + "\n" + t_text
            tree = backend.tree_with_blob(tree, path, backend.hash_blob(joined))
        else:
            conflicts.append(Conflict(path, ObjectType.FILE, b_blob, o_blob, t_blob, "content conflict"))

    if conflicts:
        backend.note_set(NOTES["conflicts"], ours_head.id, json.dumps([c.to_dict() for c in conflicts]))
        return MergeResult(None, tuple(conflicts))

    state = ours._commit(
        tree=tree,
        parents=[ours_head.id, theirs_head.id],
        kind="merge",
        reason=reason,
        manifest=union,
        transcript=ours_head.transcript,
        verdict=None,
        attempt=0,
        expected_head=ours_head.id,
    )
    backend.reset_hard_to_head()
    return MergeResult(state)
