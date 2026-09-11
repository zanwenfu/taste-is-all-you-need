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
parent sees the message. This adapter therefore persists synchronously: when
``append`` returns, the entries and recovery evidence have been fsynced into
the branch worktree rather than queued in memory. The worker's next checkpoint
commits that worktree without sweeping unrelated in-flight code into history.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
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
    raise ImportError("taste.brains needs claude-agent-sdk: pip install 'taste[brains]'") from exc

__all__ = ["MAIN_SUBPATH", "TRANSCRIPT_DIR", "MemstoreSessionStore"]

_MIRROR_LEDGER_SCHEMA = "taste.brains/SessionMirrorLedger/2"
_MIRROR_LEDGER_SCHEMA_V1 = "taste.brains/SessionMirrorLedger/1"
_MIRROR_RECORD_SCHEMA = "taste.brains/SessionMirrorRecord/2"

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
        self._mirror_lock = threading.RLock()
        # A hard kill can leave the write-ahead record after the transcript
        # bytes reached disk.  Reconcile at construction, before the SDK can
        # send another batch, so an exact completed write is finalized without
        # relying on the SDK to redeliver it.
        self._reconcile_mirror_ledger()

    # ------------------------------------------------------------- mirror ledger

    def _mirror_ledger_path(self):
        return self.store.sidecar("session-mirror", self.branch_name)

    @staticmethod
    def _canonical_key(key: Any) -> dict[str, str | None]:
        if not isinstance(key, dict):
            raise TypeError("a session key must be a mapping")
        project_key = key.get("project_key")
        session_id = key.get("session_id")
        subpath = key.get("subpath")
        if not isinstance(project_key, str) or not isinstance(session_id, str):
            raise TypeError("session project_key and session_id must be strings")
        if subpath is not None and not isinstance(subpath, str):
            raise TypeError("session subpath must be a string or None")
        return {
            "project_key": project_key,
            "session_id": session_id,
            "subpath": subpath,
        }

    @staticmethod
    def _entry_payload(entries: list[dict[str, Any]]) -> str:
        """The exact ordered bytes written to the JSONL transcript.

        Keeping the serialization itself in the write-ahead record matters
        for entries without UUIDs: content hashes alone cannot establish
        which of two identical marker entries was appended, while an exact
        byte range at a recorded pre-write offset can.
        """
        if not all(isinstance(entry, dict) for entry in entries):
            raise TypeError("session entries must be mappings")
        return "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries)

    @classmethod
    def _batch_fingerprint(cls, key: Any, entries: list[dict[str, Any]]) -> str:
        canonical_key = json.dumps(
            cls._canonical_key(key),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        payload = cls._entry_payload(entries).encode("utf-8")
        return hashlib.sha256(canonical_key + b"\0" + payload).hexdigest()

    @staticmethod
    def _blob_evidence(data: bytes, *, exists: bool = True) -> dict[str, Any]:
        return {
            "exists": exists,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    @staticmethod
    def _fsync_directory(path: Any) -> None:
        directory = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    @classmethod
    def _ensure_parent_durable(cls, path: Any) -> None:
        missing = []
        parent = path.parent
        cursor = parent
        while not cursor.exists() and cursor != cursor.parent:
            missing.append(cursor)
            cursor = cursor.parent
        parent.mkdir(parents=True, exist_ok=True)
        # Persist every newly-created directory entry, from the existing
        # ancestor down to the transcript directory.
        for created in reversed(missing):
            cls._fsync_directory(created.parent)

    def _atomic_replace(self, path: Any, payload: bytes) -> None:
        """Replace one recovery/summary file and fsync file plus directory."""
        self._ensure_parent_durable(path)
        # Keep plumbing outside the worktree. A brain checkpoint stages the
        # entire tree, so an in-directory temporary could otherwise become an
        # immutable artifact if a checkpoint overlaps this tiny write window.
        token = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
        tmp = self.store.backend.common_dir / (
            f"memstore.session-write.{os.getpid()}.{threading.get_ident()}.{token}.tmp"
        )
        try:
            with open(tmp, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            self._fsync_directory(path.parent)
        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()

    def _read_mirror_ledger(self) -> dict[str, Any]:
        path = self._mirror_ledger_path()
        if not path.exists():
            return {"schema": _MIRROR_LEDGER_SCHEMA, "unresolved": {}, "history": []}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("the session mirror ledger is unreadable") from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("unresolved"), dict):
            raise RuntimeError("the session mirror ledger has an unknown schema")
        schema = raw.get("schema")
        if schema == _MIRROR_LEDGER_SCHEMA_V1:
            # V1 deliberately did not retain the entries or their exact byte
            # boundary.  Preserve its alarms, but never pretend they can be
            # reconciled under the stronger protocol.
            converted: dict[str, Any] = {}
            for fingerprint, record in raw["unresolved"].items():
                if not isinstance(record, dict):
                    raise RuntimeError("the session mirror ledger contains an invalid record")
                legacy = dict(record)
                legacy["fingerprint"] = fingerprint
                legacy["status"] = "legacy_unverifiable"
                legacy["legacy_schema"] = _MIRROR_LEDGER_SCHEMA_V1
                converted[fingerprint] = legacy
            return {
                "schema": _MIRROR_LEDGER_SCHEMA,
                "unresolved": converted,
                "history": [],
            }
        if schema != _MIRROR_LEDGER_SCHEMA:
            raise RuntimeError("the session mirror ledger has an unknown schema")
        history = raw.setdefault("history", [])
        if not isinstance(history, list):
            raise RuntimeError("the session mirror ledger has invalid recovery history")
        return raw

    def _write_mirror_ledger(self, raw: dict[str, Any]) -> None:
        path = self._mirror_ledger_path()
        payload = json.dumps(
            raw,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self._atomic_replace(path, payload)

    @staticmethod
    def _planned_blob(path: str, before: bytes | None, payload: str) -> dict[str, Any]:
        expected = payload.encode("utf-8")
        return {
            "path": path,
            "before": MemstoreSessionStore._blob_evidence(before or b"", exists=before is not None),
            "payload": payload,
            "payload_sha256": hashlib.sha256(expected).hexdigest(),
            "payload_size": len(expected),
        }

    def _current_branch_bytes(self, branch: Any, path: str) -> bytes | None:
        target = branch.path(path)
        if target.exists():
            return target.read_bytes()
        raw = self._read(path)
        return raw.encode("utf-8", "surrogateescape") if raw is not None else None

    def _summary_payload(
        self,
        project_key: str,
        session_id: str,
        entries: list[dict[str, Any]],
        stamped: int,
    ) -> str:
        raw = self._read_current(self._summary_path(project_key, session_id))
        prev = json.loads(raw) if raw else None
        folded = dict(
            fold_session_summary(
                prev,
                {"project_key": project_key, "session_id": session_id},
                entries,
            )
        )
        folded["mtime"] = stamped
        return json.dumps(folded, sort_keys=True) + "\n"

    def _prepare_mirror_record(
        self, key: Any, entries: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        canonical_key = self._canonical_key(key)
        branch = self.store.branch(self.branch_name)
        path = self._path(canonical_key)
        fresh = self._undelivered(path, entries, remember=False)
        if not fresh:
            return None

        current_before = self._current_branch_bytes(branch, path)
        before = current_before or b""
        payload = self._entry_payload(fresh)
        payload_bytes = payload.encode("utf-8")
        stamped = _now_ms()
        sidecars: list[dict[str, Any]] = []
        if canonical_key["subpath"] is None:
            project_key = str(canonical_key["project_key"])
            session_id = str(canonical_key["session_id"])
            summary_path = self._summary_path(project_key, session_id)
            summary_payload = self._summary_payload(project_key, session_id, fresh, stamped)
            sidecars.append(
                self._planned_blob(
                    summary_path,
                    self._current_branch_bytes(branch, summary_path),
                    summary_payload,
                )
            )
            meta_path = self._meta_path(project_key, session_id)
            meta_payload = json.dumps({"session_id": session_id, "mtime": stamped}) + "\n"
            sidecars.append(
                self._planned_blob(
                    meta_path,
                    self._current_branch_bytes(branch, meta_path),
                    meta_payload,
                )
            )

        input_payload = self._entry_payload(entries)
        fingerprint = self._batch_fingerprint(canonical_key, entries)
        return {
            "record_schema": _MIRROR_RECORD_SCHEMA,
            "fingerprint": fingerprint,
            "status": "pending",
            "attempts": 1,
            "key": canonical_key,
            "transcript_path": path,
            "entry_count": len(entries),
            "fresh_entry_count": len(fresh),
            "input_payload": input_payload,
            "input_payload_sha256": hashlib.sha256(input_payload.encode("utf-8")).hexdigest(),
            "transcript": {
                "before": self._blob_evidence(before, exists=current_before is not None),
                "payload": payload,
                "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
                "payload_size": len(payload_bytes),
            },
            "sidecars": sidecars,
            "stamped_at_ms": stamped,
            "failures": [],
            "evidence": [],
            "updated_at_ms": stamped,
        }

    @staticmethod
    def _decode_payload(payload: str, expected_count: int) -> list[dict[str, Any]]:
        try:
            entries = [json.loads(line) for line in payload.splitlines()]
        except json.JSONDecodeError as exc:
            raise RuntimeError("the session mirror ledger contains invalid JSONL") from exc
        if len(entries) != expected_count or not all(isinstance(entry, dict) for entry in entries):
            raise RuntimeError("the session mirror ledger has an invalid entry count")
        return entries

    def _validate_record(self, fingerprint: str, record: dict[str, Any]) -> None:
        if record.get("legacy_schema"):
            return
        if record.get("record_schema") != _MIRROR_RECORD_SCHEMA:
            raise RuntimeError("the session mirror ledger has an unknown record schema")
        if (
            record.get("fingerprint") != fingerprint
            or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)
        ):
            raise RuntimeError("the session mirror ledger has an invalid fingerprint")
        key = self._canonical_key(record.get("key"))
        if record.get("transcript_path") != self._path(key):
            raise RuntimeError("the session mirror ledger targets the wrong transcript")
        for count_name in ("entry_count", "fresh_entry_count", "attempts", "stamped_at_ms"):
            value = record.get(count_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError("the session mirror ledger contains an invalid count")
        for payload_name, count_name in (
            ("input_payload", "entry_count"),
            ("transcript.payload", "fresh_entry_count"),
        ):
            if payload_name == "input_payload":
                payload = record.get(payload_name)
            else:
                transcript = record.get("transcript")
                payload = transcript.get("payload") if isinstance(transcript, dict) else None
            if not isinstance(payload, str):
                raise RuntimeError("the session mirror ledger is missing canonical entries")
            self._decode_payload(payload, record[count_name])

        input_payload = record["input_payload"]
        input_entries = self._decode_payload(input_payload, record["entry_count"])
        if (
            record.get("input_payload_sha256")
            != hashlib.sha256(input_payload.encode("utf-8")).hexdigest()
            or self._entry_payload(input_entries) != input_payload
            or self._batch_fingerprint(key, input_entries) != fingerprint
        ):
            raise RuntimeError("the session mirror ledger input evidence does not match")

        transcript = record.get("transcript")
        if not isinstance(transcript, dict) or not isinstance(transcript.get("before"), dict):
            raise RuntimeError("the session mirror ledger has invalid transcript evidence")
        payload_bytes = transcript["payload"].encode("utf-8")
        fresh_entries = self._decode_payload(transcript["payload"], record["fresh_entry_count"])
        input_lines = input_payload.splitlines(keepends=True)
        fresh_lines = transcript["payload"].splitlines(keepends=True)
        cursor = 0
        for fresh_line in fresh_lines:
            while cursor < len(input_lines) and input_lines[cursor] != fresh_line:
                cursor += 1
            if cursor == len(input_lines):
                raise RuntimeError(
                    "the session mirror ledger fresh entries are not an ordered subset"
                )
            cursor += 1
        if (
            transcript.get("payload_size") != len(payload_bytes)
            or transcript.get("payload_sha256") != hashlib.sha256(payload_bytes).hexdigest()
            or self._entry_payload(fresh_entries) != transcript["payload"]
        ):
            raise RuntimeError("the session mirror ledger payload evidence does not match")
        before = transcript["before"]
        if (
            not isinstance(before.get("exists"), bool)
            or isinstance(before.get("size"), bool)
            or not isinstance(before.get("size"), int)
            or before["size"] < 0
            or not isinstance(before.get("sha256"), str)
        ):
            raise RuntimeError("the session mirror ledger has invalid pre-write evidence")

        sidecars = record.get("sidecars")
        if not isinstance(sidecars, list):
            raise RuntimeError("the session mirror ledger has invalid sidecar evidence")
        project_key = str(key["project_key"])
        session_id = str(key["session_id"])
        expected_paths = (
            {self._summary_path(project_key, session_id), self._meta_path(project_key, session_id)}
            if key["subpath"] is None
            else set()
        )
        actual_paths: set[str] = set()
        for plan in sidecars:
            if not isinstance(plan, dict) or not isinstance(plan.get("path"), str):
                raise RuntimeError("the session mirror ledger has an invalid sidecar plan")
            actual_paths.add(plan["path"])
            before_blob = plan.get("before")
            planned_payload = plan.get("payload")
            if not isinstance(before_blob, dict) or not isinstance(planned_payload, str):
                raise RuntimeError("the session mirror ledger has incomplete sidecar evidence")
            expected = planned_payload.encode("utf-8")
            if (
                plan.get("payload_size") != len(expected)
                or plan.get("payload_sha256") != hashlib.sha256(expected).hexdigest()
            ):
                raise RuntimeError("the session mirror ledger sidecar payload does not match")
            if (
                not isinstance(before_blob.get("exists"), bool)
                or isinstance(before_blob.get("size"), bool)
                or not isinstance(before_blob.get("size"), int)
                or before_blob["size"] < 0
                or not isinstance(before_blob.get("sha256"), str)
            ):
                raise RuntimeError("the session mirror ledger has invalid sidecar evidence")
        if actual_paths != expected_paths:
            raise RuntimeError("the session mirror ledger has unexpected sidecar targets")

    @staticmethod
    def _matches_evidence(data: bytes | None, evidence: dict[str, Any]) -> bool:
        exists = data is not None
        payload = data or b""
        return (
            exists is evidence.get("exists")
            and len(payload) == evidence.get("size")
            and hashlib.sha256(payload).hexdigest() == evidence.get("sha256")
        )

    def _classify_transcript(self, record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        branch = self.store.branch(self.branch_name)
        current = self._current_branch_bytes(branch, record["transcript_path"]) or b""
        transcript = record["transcript"]
        before = transcript["before"]
        before_size = before["size"]
        payload = transcript["payload"].encode("utf-8")

        classification = "mismatch"
        prefix = current[:before_size]
        tail = current[before_size:] if len(current) >= before_size else b""
        if (
            len(current) >= before_size
            and len(prefix) == before_size
            and hashlib.sha256(prefix).hexdigest() == before["sha256"]
        ):
            if tail == payload:
                classification = "complete"
            elif not tail:
                classification = "absent"
            elif len(tail) < len(payload) and payload.startswith(tail):
                classification = "partial"

        evidence = {
            "classification": classification,
            "current_size": len(current),
            "current_sha256": hashlib.sha256(current).hexdigest(),
            "observed_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
            "tail_size": len(tail),
            "tail_sha256": hashlib.sha256(tail).hexdigest(),
            "observed_at_ms": _now_ms(),
        }
        return classification, evidence

    @staticmethod
    def _remember_evidence(record: dict[str, Any], evidence: dict[str, Any]) -> bool:
        history = record.setdefault("evidence", [])
        comparable = {key: value for key, value in evidence.items() if key != "observed_at_ms"}
        if history:
            previous = {key: value for key, value in history[-1].items() if key != "observed_at_ms"}
            if previous == comparable:
                return False
        history.append(evidence)
        record["updated_at_ms"] = evidence["observed_at_ms"]
        return True

    @staticmethod
    def _record_failure(record: dict[str, Any], exc: BaseException, stage: str) -> None:
        rendered = f"{type(exc).__name__}: {exc}"
        now = _now_ms()
        failures = record.setdefault("failures", [])
        failure = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
            "rendered": rendered,
            "at_ms": now,
        }
        if not failures or any(
            failures[-1].get(key) != failure[key]
            for key in ("stage", "type", "message", "rendered")
        ):
            failures.append(failure)
        record["error"] = rendered
        record["status"] = "failed"
        record["updated_at_ms"] = now

    def _apply_planned_blob(self, branch: Any, plan: dict[str, Any]) -> None:
        current = self._current_branch_bytes(branch, plan["path"])
        expected = plan["payload"].encode("utf-8")
        if current == expected:
            return
        if not self._matches_evidence(current, plan["before"]):
            raise RuntimeError(f"session mirror recovery found divergent sidecar {plan['path']!r}")
        self._atomic_replace(branch.path(plan["path"]), expected)

    def _apply_recovery_sidecars(self, record: dict[str, Any]) -> None:
        branch = self.store.branch(self.branch_name)
        for plan in record["sidecars"]:
            self._apply_planned_blob(branch, plan)

    def _apply_append_sidecars(self, record: dict[str, Any]) -> None:
        """Apply sidecars normally, retaining ``_fold_summary`` as a fault seam."""
        key = record["key"]
        if key["subpath"] is not None:
            return
        branch = self.store.branch(self.branch_name)
        fresh = self._decode_payload(record["transcript"]["payload"], record["fresh_entry_count"])
        self._fold_summary(
            branch,
            str(key["project_key"]),
            str(key["session_id"]),
            fresh,
            record["stamped_at_ms"],
        )
        # The first plan is the summary. Verify that the SDK fold produced the
        # exact transition journalled before the transcript append.
        summary = self._current_branch_bytes(branch, record["sidecars"][0]["path"])
        if summary != record["sidecars"][0]["payload"].encode("utf-8"):
            raise RuntimeError("the persisted session summary did not match its write-ahead plan")
        self._apply_planned_blob(branch, record["sidecars"][1])

    def _write_record_transcript(self, record: dict[str, Any]) -> None:
        classification, evidence = self._classify_transcript(record)
        self._remember_evidence(record, evidence)
        if classification not in {"absent", "partial", "complete"}:
            raise RuntimeError("session mirror transcript diverged from its write-ahead record")
        if classification == "complete":
            return

        transcript = record["transcript"]
        payload = transcript["payload"].encode("utf-8")
        already = evidence["tail_size"]
        suffix = payload[already:]
        branch = self.store.branch(self.branch_name)
        target = branch.path(record["transcript_path"])
        self._ensure_parent_durable(target)
        with open(target, "ab") as handle:
            written = handle.write(suffix)
            if written != len(suffix):
                raise OSError(f"short transcript write: wrote {written} of {len(suffix)} bytes")
            handle.flush()
            os.fsync(handle.fileno())
        self._fsync_directory(target.parent)

        after, after_evidence = self._classify_transcript(record)
        self._remember_evidence(record, after_evidence)
        if after != "complete":
            raise RuntimeError("session transcript was not exact after its fsynced append")
        fresh = self._decode_payload(transcript["payload"], record["fresh_entry_count"])
        self._remember_uuids(record["transcript_path"], fresh)

    def _reconcile_mirror_ledger(self) -> set[str]:
        resolved: set[str] = set()
        with self._mirror_lock:
            path = self._mirror_ledger_path()
            if not path.exists():
                return resolved
            raw = self._read_mirror_ledger()
            changed = raw.get("schema") != _MIRROR_LEDGER_SCHEMA
            unresolved = raw["unresolved"]
            for fingerprint in list(unresolved):
                record = unresolved[fingerprint]
                if not isinstance(record, dict):
                    raise RuntimeError("the session mirror ledger contains an invalid record")
                self._validate_record(fingerprint, record)
                if record.get("legacy_schema"):
                    changed = True
                    continue
                classification, evidence = self._classify_transcript(record)
                changed = self._remember_evidence(record, evidence) or changed
                if classification != "complete":
                    if record.get("status") != classification:
                        record["status"] = classification
                        changed = True
                    continue
                try:
                    self._apply_recovery_sidecars(record)
                except Exception as exc:
                    self._record_failure(record, exc, "reconcile_sidecars")
                    changed = True
                    continue
                unresolved.pop(fingerprint)
                recovered = dict(record)
                recovered["status"] = "recovered"
                recovered["resolved_at_ms"] = _now_ms()
                raw["history"].append(recovered)
                resolved.add(fingerprint)
                changed = True
            if changed:
                self._write_mirror_ledger(raw)
        return resolved

    def unresolved_mirror_batches(self) -> tuple[dict[str, Any], ...]:
        """Batches not proven present, including a process-killed attempt."""
        with self._mirror_lock:
            self._reconcile_mirror_ledger()
            raw = self._read_mirror_ledger()
            return tuple(
                dict(raw["unresolved"][fingerprint]) for fingerprint in sorted(raw["unresolved"])
            )

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

        Returns only once the entries and their recovery evidence have been
        fsynced into the branch worktree, so the caller's "durability is
        already guaranteed locally" assumption is true of our store as well
        and not merely of the CLI's own file. The worker checkpoint owns the
        later git commit.
        """
        if not entries:
            return  # contract 4: append([]) is a no-op, not an empty state
        canonical_key = self._canonical_key(key)
        fingerprint = self._batch_fingerprint(canonical_key, entries)
        input_payload = self._entry_payload(entries)

        with self._mirror_lock:
            recovered = self._reconcile_mirror_ledger()
            if fingerprint in recovered:
                # This invocation is the SDK retry of the journalled batch.
                # Recovery has already reconstructed its summary/meta and the
                # transcript range is exact, including for UUID-less entries.
                return
            raw = self._read_mirror_ledger()
            unresolved = raw["unresolved"]
            record = unresolved.get(fingerprint)

            if record is not None:
                if not isinstance(record, dict):
                    raise RuntimeError("the session mirror ledger contains an invalid record")
                self._validate_record(fingerprint, record)
                if record.get("legacy_schema"):
                    raise RuntimeError("a legacy mirror alarm cannot be safely retried")
                if record["key"] != canonical_key or record["input_payload"] != input_payload:
                    raise RuntimeError("a mirror fingerprint aliases a different SDK batch")
                if record.get("status") not in {"absent", "partial", "pending", "failed"}:
                    raise RuntimeError("the mirror batch is divergent and cannot be retried safely")
                record["attempts"] += 1
                record["status"] = "pending"
                record["updated_at_ms"] = _now_ms()
            else:
                transcript_path = self._path(canonical_key)
                for pending in unresolved.values():
                    if not isinstance(pending, dict):
                        raise RuntimeError("the session mirror ledger contains an invalid record")
                    pending_path = pending.get("transcript_path")
                    if pending.get("legacy_schema"):
                        same_key = (
                            pending.get("project_key") == canonical_key["project_key"]
                            and pending.get("session_id") == canonical_key["session_id"]
                            and pending.get("subpath") == canonical_key["subpath"]
                        )
                        if same_key:
                            raise RuntimeError("a legacy mirror alarm blocks this transcript")
                    elif pending_path == transcript_path:
                        raise RuntimeError(
                            "an earlier mirror batch on this transcript is unresolved"
                        )
                record = self._prepare_mirror_record(canonical_key, entries)
                if record is None:
                    return
                unresolved[fingerprint] = record

            # The full ordered batch, exact fresh JSONL bytes, byte offset,
            # and sidecar transitions are durable before transcript mutation.
            self._write_mirror_ledger(raw)
            try:
                self._write_record_transcript(record)
                self._apply_append_sidecars(record)
            except BaseException as exc:
                # Preserve the original failure and all evidence gathered so
                # far. If this second ledger write fails, the earlier durable
                # pending record still makes the gap detectable.
                self._record_failure(record, exc, "append")
                with contextlib.suppress(Exception):
                    self._write_mirror_ledger(raw)
                raise
            unresolved.pop(fingerprint)
            self._write_mirror_ledger(raw)

        # Deliberately no checkpoint here. ``Branch.checkpoint`` stages the
        # whole worktree, so a mirror batch landing mid-edit must not commit
        # unrelated half-written code. Transcript, summary, meta, and recovery
        # record durability all come from their own fsyncs.

    def _undelivered(
        self,
        path: str,
        entries: list[dict[str, Any]],
        *,
        remember: bool = True,
    ) -> list[dict[str, Any]]:
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
        planned_seen = set(seen)
        out = []
        for entry in entries:
            uuid = entry.get("uuid")
            if uuid and uuid in planned_seen:
                continue
            if uuid:
                planned_seen.add(uuid)
            out.append(entry)
        if remember:
            seen.update(planned_seen)
        return out

    def _remember_uuids(self, path: str, entries: list[dict[str, Any]]) -> None:
        # Build the cache against the live transcript if this is the first
        # append in the process, then add only UUIDs whose bytes were fsynced.
        self._undelivered(path, [], remember=False)
        seen = self._seen[path]
        seen.update(entry["uuid"] for entry in entries if entry.get("uuid"))

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
        self,
        branch: Any,
        project_key: str,
        session_id: str,
        entries: list[dict[str, Any]],
        stamped: int,
    ) -> None:
        """Maintain the SDK's incremental summary sidecar.

        ``fold_session_summary`` is pure and SDK-owned; ``data`` is opaque and
        must be persisted verbatim. The mtime is ours to stamp, after
        persisting, on the same clock as ``list_sessions``.
        """
        path = self._summary_path(project_key, session_id)
        payload = self._summary_payload(project_key, session_id, entries, stamped)
        self._atomic_replace(branch.path(path), payload.encode("utf-8"))
