"""A ``SessionStore`` backed by memstore, so a brain's memory is our memory.

The Claude Agent SDK keeps a session transcript on local disk and, given a
``session_store``, mirrors it to an adapter. This is that adapter, and it is
what makes a brain resumable from the memory layer rather than from a file that
happens to be on one machine: with the local transcript deleted, ``load``
serves the resume and the brain wakes with its reasoning intact.

Three things about the SDK contract shape everything here.

**Entries are opaque.** The concrete transcript format is an internal
discriminated union, and the only invariant the SDK promises is that entries
round-trip through ``json.dumps``/``json.loads``. So they are stored verbatim
and never parsed. A store that interprets them is a store that breaks on the
next CLI release.

**Order is call order.** Entries come back in the order they were appended,
never sorted, and duplicate uuids within a session are kept -- the SDK's own
conformance suite appends uuids ``z, a, m, b`` and requires exactly that back.
Deduplication belongs where the SDK asks for it (idempotent re-delivery of the
*same* batch), not as a blanket rule.

**The mirror is best-effort, and this is not.** The SDK retries a failed batch
three times and then drops it, surfacing a ``MirrorErrorMessage`` while the run
still reports success. Measured, mid-turn batches arrive ~100ms after the
parent sees the message. This adapter therefore commits synchronously: when
``append`` returns, the entries are in git, not queued for it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from taste.memstore import Store

try:
    from claude_agent_sdk._internal.session_summary import fold_session_summary
except ImportError as exc:  # pragma: no cover - exercised by the isolation test
    # Fail at import, not on the first append. The fold is on the hot path
    # (every main-transcript write maintains the summary sidecar), so a
    # missing SDK would otherwise surface mid-run as a confusing traceback
    # from inside a batch the caller believed was durable.
    raise ImportError(
        "taste.brains needs claude-agent-sdk: pip install 'taste[brains]'"
    ) from exc

__all__ = ["MAIN_SUBPATH", "TRANSCRIPT_DIR", "MemstoreSessionStore"]

TRANSCRIPT_DIR = "sdk-sessions"
"""Where session transcripts live inside a brain's branch."""

MAIN_SUBPATH = "_main"
"""Stands in for the absent ``subpath`` of a main transcript.

``SessionKey`` omits ``subpath`` for the main transcript and sets it for
subagents; an empty string is explicitly invalid. A path segment needs *some*
name, so the main transcript gets a reserved one that cannot collide with a
real subpath (the SDK's are ``subagents/...``).
"""


def _now_ms() -> int:
    """Storage write time, in epoch milliseconds.

    Deliberately the wall clock at persist time rather than anything derived
    from an entry's own ISO timestamp: the SDK compares this against the
    sidecar mtime to decide whether a summary is stale, and a value derived
    from entry timestamps is always older than the write that produced it, so
    every summary would look stale forever.
    """
    return int(time.time() * 1000)


class MemstoreSessionStore:
    """Mirrors SDK session transcripts into a memstore branch.

    Deliberately duck-typed rather than subclassing ``SessionStore``: the SDK
    never uses ``isinstance``, and it probes for the optional methods by
    checking whether the class overrides them. Inheriting would advertise
    every optional method as present.

    One store serves many sessions. Transcripts are written as records under
    ``sdk-sessions/<project>/<session>/<subpath>.jsonl`` on ``branch``, so a
    brain's conversation and its artifacts are the same history and move
    together on rollback.
    """

    def __init__(self, store: Store, branch: str, *, project_key: str | None = None) -> None:
        self.store = store
        self.branch_name = branch
        # A brain's memory follows its identity, not its filesystem location.
        # The SDK derives project_key from the realpath of cwd and offers no
        # way to set it, so a worktree recreated at a different path would
        # address a different transcript and the brain would wake amnesiac.
        # Pinning it to the branch makes worktrees disposable, which is what
        # the architecture assumes: the branch is the address space, the
        # worktree is scratch. Left None, the SDK's own scoping is honoured.
        self.project_key = project_key
        self._seen: dict[str, set[str]] = {}
        self._seen_head: str | None = None

    # ------------------------------------------------------------------ paths

    @staticmethod
    def _safe(segment: str) -> str:
        """Make one key segment safe as a path component.

        ``project_key`` is a sanitized cwd by default and can be an arbitrary
        caller-defined string; it must not be able to escape the transcript
        directory or collide after sanitising.
        """
        out = "".join(c if c.isalnum() or c in "-_." else "_" for c in segment)
        out = out.strip(".") or "_"
        if len(out) > 120:
            # Keep it addressable, keep it unique.
            # sha256, never hash(): Python randomizes string hashing per
            # process, so the same key would resolve to a different directory
            # on every run and a brain could never find its own transcript
            # again. A realistic worktree realpath is already ~127 characters,
            # so this is the ordinary path, not an edge case.
            digest = hashlib.sha256(segment.encode("utf-8", "surrogateescape")).hexdigest()[:16]
            out = f"{out[:100]}_{digest}"
        return out

    def _scope(self, project_key: str) -> str:
        """The directory a project's sessions live under.

        Digested rather than used verbatim: the raw key is a sanitised
        filesystem path, so it is long, ugly, and different on every machine.
        The digest keeps distinct projects isolated -- which the protocol
        requires -- while making the layout independent of path length and of
        where the worktree happens to sit.
        """
        key = self.project_key if self.project_key is not None else project_key
        digest = hashlib.sha256(key.encode("utf-8", "surrogateescape")).hexdigest()[:16]
        return f"{self._safe(key)[:60]}-{digest}"

    def _dir(self, project_key: str, session_id: str) -> str:
        return f"{TRANSCRIPT_DIR}/{self._scope(project_key)}/{self._safe(session_id)}"

    def _path(self, key: Any) -> str:
        subpath = key.get("subpath") or MAIN_SUBPATH
        return f"{self._dir(key['project_key'], key['session_id'])}/{self._safe_subpath(subpath)}.jsonl"

    @classmethod
    def _safe_subpath(cls, subpath: str) -> str:
        """Sanitise a subpath while keeping its shape.

        ``subpath`` mirrors an on-disk directory structure ("subagents/
        agent-1") and is opaque to us, so its separators are meaningful and
        must survive the round trip -- ``list_subkeys`` has to hand back
        exactly what was appended. Each segment is still sanitised
        individually, so no segment can be ``..`` and nothing escapes.
        """
        parts = [cls._safe(p) for p in subpath.split("/") if p not in ("", ".", "..")]
        return "/".join(parts) or "_"

    def _summary_path(self, project_key: str, session_id: str) -> str:
        return f"{self._dir(project_key, session_id)}/{MAIN_SUBPATH}.summary.json"

    def _meta_path(self, project_key: str, session_id: str) -> str:
        return f"{self._dir(project_key, session_id)}/{MAIN_SUBPATH}.meta.json"

    # ------------------------------------------------------------- required

    async def append(self, key: Any, entries: list[dict[str, Any]]) -> None:
        """Mirror a batch of transcript entries, durably.

        Returns only once the entries are committed, so the caller's
        "durability is already guaranteed locally" assumption is true of our
        store as well and not merely of the CLI's own file.
        """
        if not entries:
            return  # contract 4: append([]) is a no-op, not an empty state

        branch = self.store.branch(self.branch_name)
        path = self._path(key)
        # Append to the file; never rebuild it from the committed head.
        # ``branch.read`` returns the last *committed* state while
        # ``branch.write`` writes the working tree, so read-then-rewrite
        # rebased every batch onto a base that only advances at a checkpoint:
        # at checkpoint_every=5, eight of ten entries were destroyed before
        # any commit could preserve them, silently. Appending is also O(1)
        # rather than O(n), which is what made a long session quadratic.
        fresh = self._undelivered(path, entries)
        if not fresh:
            return
        added = "".join(json.dumps(e, sort_keys=True) + "\n" for e in fresh)
        target = branch.path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(added)
            fh.flush()
            os.fsync(fh.fileno())

        project_key, session_id = key["project_key"], key["session_id"]
        stamped = _now_ms()
        is_main = not key.get("subpath")
        if is_main:
            # Subagent appends must not touch the main session's summary or
            # its listing, so both sidecars are main-transcript only.
            self._fold_summary(branch, project_key, session_id, entries, stamped)
        branch.write(
            self._meta_path(project_key, session_id),
            json.dumps({"session_id": session_id, "mtime": stamped}) + "\n",
        )

        # Deliberately no checkpoint here. ``Branch.checkpoint`` stages the
        # whole worktree, so a mirror batch arriving while the brain was
        # mid-edit committed half-written code under a reason claiming to be a
        # transcript write -- a state no monitor could interpret, and one that
        # a rollback would then restore. Durability rests on the fsync above,
        # not on the commit; turning these entries into history is the
        # sub-brain's own checkpoint, where the tree is coherent and the
        # reason is true.

    def _undelivered(self, path: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop entries this path has already stored.

        The protocol says most entries carry a stable uuid "that adapters
        should treat as an idempotency key", because a batch can be
        re-delivered -- the SDK retries a failed append three times. Without
        this, a retry that partially succeeded appends every entry twice and
        a resumed brain reads its own turns doubled.

        Entries without a uuid are appended unconditionally, as the protocol
        requires: titles, tags and mode markers carry no identity and are not
        duplicates of one another.
        """
        # Keyed on the head as well as the path: a rollback removes turns
        # from the transcript, and a cache that outlived it swallowed the
        # SDK's re-mirror of exactly those entries -- the brain lost them for
        # good, silently, having been told they were already stored.
        head = self._head_id()
        if self._seen_head != head:
            self._seen.clear()
            self._seen_head = head
        seen = self._seen.get(path)
        if seen is None:
            seen = set()
            for line in (self._read_current(path) or "").splitlines():
                if line.strip():
                    try:
                        uuid = json.loads(line).get("uuid")
                    except json.JSONDecodeError:
                        continue
                    if uuid:
                        seen.add(uuid)
            self._seen[path] = seen
        out = []
        for entry in entries:
            uuid = entry.get("uuid")
            if uuid and uuid in seen:
                continue
            if uuid:
                seen.add(uuid)
            out.append(entry)
        return out

    async def load(self, key: Any) -> list[dict[str, Any]] | None:
        """Serve a resume from our store. ``None`` means never written."""
        raw = self._read_current(self._path(key))
        if raw is None:
            return None
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    # ------------------------------------------------------------- optional

    async def list_sessions(self, project_key: str) -> list[dict[str, Any]]:
        """Sessions in a project, newest-first ordering left to the SDK."""
        out: list[dict[str, Any]] = []
        for session_id, meta in self._each_session(project_key):
            out.append({"session_id": session_id, "mtime": int(meta.get("mtime", 0))})
        return out

    async def list_session_summaries(self, project_key: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for session_id, _meta in self._each_session(project_key):
            raw = self._read_current(self._summary_path(project_key, session_id))
            if raw:
                out.append(json.loads(raw))
        return out

    async def delete(self, key: Any) -> None:
        """Remove a transcript. A main key cascades to its subagents.

        Nothing is destroyed in the memory-layer sense: the state that held
        these files stays reachable, so a deleted transcript is still in the
        history. This only removes it from the working tree the SDK reads.
        """
        branch = self.store.branch(self.branch_name)
        if key.get("subpath"):
            targets = [self._path(key)]
        else:
            prefix = self._dir(key["project_key"], key["session_id"]) + "/"
            targets = [p for p in self._files() if p.startswith(prefix)]
        for path in targets:
            full = branch.path(path)
            if full.exists():
                full.unlink()
            self._seen.pop(path, None)
        # No checkpoint, for the same reason append takes none: committing
        # here would sweep the sub-brain's unrelated work into a state
        # labelled "transcript deleted". The removal is real on disk and
        # becomes history at the sub-brain's next checkpoint. Nothing is lost
        # in the memory-layer sense either -- the states that held these
        # files stay reachable forever.

    async def list_subkeys(self, key: Any) -> list[str]:
        """Subagent transcripts under a session.

        Required for correctness, not tidiness: without it, resume
        materializes only the main transcript and every subagent's reasoning
        is silently dropped.
        """
        prefix = self._dir(key["project_key"], key["session_id"]) + "/"
        out: list[str] = []
        for path in self._files():
            if not path.startswith(prefix) or not path.endswith(".jsonl"):
                continue
            name = path[len(prefix) : -len(".jsonl")]
            if name != MAIN_SUBPATH:
                out.append(name)
        return sorted(out)

    # ------------------------------------------------------------- internals

    def _head_id(self) -> str:
        """Which state the dedup cache was built against."""
        view = self.store.view(self.branch_name)
        return view.head.id if view.exists() else ""

    def _read_current(self, path: str) -> str | None:
        """The live file if the worktree has it, else the committed state.

        Between checkpoints the working tree is ahead of the head, and after a
        worktree is removed the head is all there is, so a correct read
        consults the worktree first and falls back to history. A resume that
        saw only committed state would hand the brain a transcript missing its
        most recent turns -- the reasoning it needs most.
        """
        try:
            live = self.store.worktree_path_for(self.branch_name) / path
            if live.exists():
                return live.read_text(encoding="utf-8", errors="surrogateescape")
        except (OSError, ValueError):
            pass
        return self._read(path)

    def _read(self, path: str) -> str | None:
        """Read from the branch head, or None if the branch has no head yet.

        A store is constructed before its brain has written anything, so
        "nothing here" must answer as absence rather than raising -- the SDK
        calls ``load`` on resume before it calls ``append``.
        """
        view = self.store.view(self.branch_name)
        if not view.exists():
            return None
        return view.read(path)

    def _files(self) -> list[str]:
        """Transcript files that exist now, committed or not.

        The adapter writes the working tree and lets the sub-brain's own
        checkpoint turn that into history, so committed state alone is always
        behind. Listing sessions from it would hide the session currently
        being written -- the only one anybody is asking about.
        """
        seen: list[str] = []
        root = self.store.worktree_path_for(self.branch_name) / TRANSCRIPT_DIR
        if root.exists():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    seen.append(str(path.relative_to(root.parent)))
        view = self.store.view(self.branch_name)
        if view.exists():
            for path in view.head.files():
                if path.startswith(TRANSCRIPT_DIR + "/") and path not in seen:
                    seen.append(path)
        return seen

    def _each_session(self, project_key: str):
        """(session_id, meta) for every session written under a project."""
        prefix = f"{TRANSCRIPT_DIR}/{self._scope(project_key)}/"
        suffix = f"/{MAIN_SUBPATH}.meta.json"
        for path in self._files():
            if path.startswith(prefix) and path.endswith(suffix):
                raw = self._read_current(path)
                if raw:
                    meta = json.loads(raw)
                    yield meta["session_id"], meta

    def _fold_summary(
        self, branch: Any, project_key: str, session_id: str, entries: list[dict[str, Any]], stamped: int
    ) -> None:
        """Maintain the SDK's incremental summary sidecar.

        ``fold_session_summary`` is pure and SDK-owned; ``data`` is opaque and
        must be persisted verbatim. The mtime is ours to stamp, after
        persisting, on the same clock as ``list_sessions``.
        """
        path = self._summary_path(project_key, session_id)
        raw = branch.read(path)
        prev = json.loads(raw) if raw else None
        folded = dict(
            fold_session_summary(
                prev, {"project_key": project_key, "session_id": session_id}, entries
            )
        )
        folded["mtime"] = stamped
        branch.write(path, json.dumps(folded, sort_keys=True) + "\n")
