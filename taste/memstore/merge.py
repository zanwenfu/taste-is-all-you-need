"""Typed merge: the type of a path decides how two branches' versions combine.

Files merge three-way, the way git does. Records merge by key: the same key
changed to the same value on both sides is agreement, a key changed on one
side is taken, a key changed differently on both sides is a conflict for
that key only. Transcripts never merge: ours is kept, theirs stays on its
branch. Manifests are a union.

Two rules an audit forced, both about the promise that a conflict is a value
and not an exception:

* **Every type rule is total.** Any failure inside a rule (unparsable JSON,
  bytes that are not text, an unexpected mode) degrades to a ``Conflict``
  rather than propagating. The invariant is enforced structurally, not by
  hoping each rule handles every input.
* **A failed merge is recorded as a state, not as a note bolted onto a
  published one.** The first version rewrote the annotations of a state
  others had already read, and a second failed merge overwrote the first.
"""

from __future__ import annotations

import json
from typing import Any

from taste.memstore.backend import MODE_FILE, UNMERGEABLE_MODES
from taste.memstore.objects import Conflict, ObjectType, now_iso
from taste.memstore.store import Branch, BranchView, MergeResult


def _same(a: Any, b: Any) -> bool:
    """Type-aware equality for JSON values.

    Python's ``==`` makes ``1 == True == 1.0``, under which a real change
    from ``1`` to ``true`` looked like no change at all and was dropped with
    no conflict reported.
    """
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return type(a) is type(b) and a == b


def merge_records(base: Any, ours: Any, theirs: Any) -> tuple[Any, list[str]]:
    """Three-way merge of JSON values. Returns (merged, clashing keys).

    Objects are merged key by key, recursing into nested objects so two
    brains editing different sub-keys of one record do not collide. Anything
    that is not an object on all three sides merges as a single value, and a
    genuine clash keeps *ours* so the merging brain's view wins consistently.
    """
    if isinstance(base, dict) and isinstance(ours, dict) and isinstance(theirs, dict):
        merged: dict[str, Any] = {}
        clashes: list[str] = []
        for key in sorted(set(base) | set(ours) | set(theirs)):
            in_b, in_o, in_t = key in base, key in ours, key in theirs
            b, o, t = base.get(key), ours.get(key), theirs.get(key)
            o_changed = (in_o != in_b) or (in_o and not _same(o, b))
            t_changed = (in_t != in_b) or (in_t and not _same(t, b))
            if not o_changed and not t_changed:
                if in_b:
                    merged[key] = b
            elif o_changed and not t_changed:
                if in_o:
                    merged[key] = o
            elif t_changed and not o_changed:
                if in_t:
                    merged[key] = t
            elif in_o and in_t and isinstance(o, dict) and isinstance(t, dict):
                sub_base = b if isinstance(b, dict) else {}
                sub, sub_clashes = merge_records(sub_base, o, t)
                merged[key] = sub
                clashes.extend(f"{key}.{c}" for c in sub_clashes)
            elif in_o == in_t and (not in_o or _same(o, t)):
                if in_o:
                    merged[key] = o
            else:
                clashes.append(key)
                if in_o:
                    merged[key] = o
        return merged, clashes
    if _same(ours, theirs):
        return ours, []
    if _same(ours, base):
        return theirs, []
    if _same(theirs, base):
        return ours, []
    return ours, ["<value>"]


def _load(text: str | None) -> Any:
    if text is None:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def merge_branches(
    ours: Branch,
    theirs: Branch | BranchView,
    *,
    reason: str,
    resolved: bool = False,
) -> MergeResult:
    """Merge ``theirs`` into ``ours`` as a new state on ``ours``.

    ``theirs`` may be a read-only view: merging reads the other branch and
    never writes to it, so an orchestrator can integrate a brain's work
    while that brain is still running.

    With ``resolved=True`` the working tree of ``ours`` is taken as the
    resolution: a brain that was handed conflicts writes the files it wants
    and records the merge with both parents. No type rule runs.
    """
    ours._capture("merge", reason)
    ours_head, theirs_head = ours.head, theirs.head
    backend = ours.backend
    if backend.is_ancestor(theirs_head.id, ours_head.id):
        return MergeResult(ours_head)

    union = ours_head.manifest.union(theirs_head.manifest)
    if resolved:
        backend.stage_all()
        built = ours._build(
            tree=backend.write_tree(),
            parents=[ours_head.id, theirs_head.id],
            kind="merge",
            reason=reason,
            manifest=union,
            transcript=None,
            transcript_base=ours_head,
            verdict=None,
            attempt=0,
            expected_head=ours_head.id,
            merged=theirs_head.id,
        )
        state = ours.publish_state(built)
        backend.reset_hard_to_head()
        return MergeResult(state)

    result = backend.merge_tree(ours_head.id, theirs_head.id)
    base = backend.merge_base(ours_head.id, theirs_head.id)
    tree = result.tree
    conflicts: list[Conflict] = []
    at = now_iso()

    def conflict(path: str, kind: ObjectType, detail: str) -> Conflict:
        return Conflict(
            path=path,
            type=kind,
            base=backend.blob_at(base, path) if base else None,
            ours=backend.blob_at(ours_head.id, path),
            theirs=backend.blob_at(theirs_head.id, path),
            detail=detail,
            theirs_branch=theirs.name,
            theirs_state=theirs_head.id,
            ours_state=ours_head.id,
            at=at,
        )

    for path in result.conflicted_paths:
        kind = union.type_of(path)
        o_mode = backend.mode_at(ours_head.id, path)
        t_mode = backend.mode_at(theirs_head.id, path)
        try:
            if (o_mode in UNMERGEABLE_MODES) or (t_mode in UNMERGEABLE_MODES) or o_mode != t_mode:
                # A symlink, a submodule, or a mode change: content merging is
                # meaningless and the first version flattened these into files.
                conflicts.append(conflict(path, kind, f"mode {o_mode} against {t_mode}"))
                continue
            mode = o_mode or MODE_FILE
            o_blob = backend.blob_at(ours_head.id, path)
            t_blob = backend.blob_at(theirs_head.id, path)
            if kind is ObjectType.RECORD:
                b_blob = backend.blob_at(base, path) if base else None
                b = _load(backend.cat_blob(b_blob) if b_blob else None)
                o = _load(backend.cat_blob(o_blob) if o_blob else None)
                t = _load(backend.cat_blob(t_blob) if t_blob else None)
                if b is None or o is None or t is None:
                    conflicts.append(conflict(path, kind, "record is not valid JSON"))
                    continue
                merged, clashes = merge_records(b, o, t)
                if clashes:
                    conflicts.append(conflict(path, kind, "keys: " + ", ".join(clashes)))
                    continue
                blob = backend.hash_blob(json.dumps(merged, indent=1, sort_keys=True) + "\n")
                tree = backend.tree_with_blob(tree, path, blob, mode)
            elif kind is ObjectType.TRANSCRIPT:
                if o_blob:
                    tree = backend.tree_with_blob(tree, path, o_blob, mode)
            elif kind is ObjectType.NOTE:
                o_raw = backend.cat_blob_bytes(o_blob) if o_blob else b""
                t_raw = backend.cat_blob_bytes(t_blob) if t_blob else b""
                joined = o_raw if t_raw in o_raw else o_raw.rstrip(b"\n") + b"\n" + t_raw
                tree = backend.tree_with_blob(tree, path, backend.hash_blob(joined), mode)
            else:
                conflicts.append(conflict(path, ObjectType.FILE, "content conflict"))
        except Exception as exc:
            conflicts.append(conflict(path, kind, f"{type(exc).__name__}: {exc}"))

    if conflicts:
        # Recorded as a state of its own: our parent only, our tree unchanged,
        # so the merge has demonstrably not happened and a retry still works.
        built = ours._build(
            tree=backend.tree_of(ours_head.id),
            parents=[ours_head.id],
            kind="conflict",
            reason=f"merge conflict: {reason}",
            manifest=ours_head.manifest,
            transcript=None,
            transcript_base=ours_head,
            verdict=None,
            attempt=0,
            expected_head=ours_head.id,
            merged=theirs_head.id,
            conflicts=conflicts,
        )
        ours.publish_state(built)
        return MergeResult(None, tuple(conflicts))

    built = ours._build(
        tree=tree,
        parents=[ours_head.id, theirs_head.id],
        kind="merge",
        reason=reason,
        manifest=union,
        transcript=None,
        transcript_base=ours_head,
        verdict=None,
        attempt=0,
        expected_head=ours_head.id,
        merged=theirs_head.id,
    )
    state = ours.publish_state(built)
    backend.reset_hard_to_head()
    return MergeResult(state)
