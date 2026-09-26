"""Owned, durable Azure Responses calls for a worker run.

The controller stores this journal outside the worker worktree. Memory rollback
can change the next prompt, but cannot erase spending or make a lost reply safe
to repeat. An identical completed request is replayed from its receipt. Any
unsettled dispatch fences new calls, including after process death.

This layer owns model calls, not tool execution or memory publication. It never
releases an active call on cancellation: the call and receipt finish first.
The containing process scope must still bound a provider that never returns.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from taste.brains.owned_thread import start_owned_thread
from taste.brains.worker_protocol import ModelCallAccounting
from taste.llm import LLM, BudgetExceeded
from taste.pricing import call_cost, ensure_priced, max_call_cost_usd, table_sha
from taste.providers.azure_openai import AzureOpenAIConfig
from taste.providers.base import Completion, ProtocolFailure, ToolCall, Usage

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}\Z")
_VERSION = 1


class ResponsesFenced(ProtocolFailure):
    """No new model dispatch can be admitted in this run."""


class ResponsesConflict(ProtocolFailure):
    """A durable request or run identity was reused with different input."""


class _NotDispatched(RuntimeError):
    pass


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _positive(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ResponsesBinding:
    run_id: str
    model: str
    endpoint: str
    deployment: str
    budget_usd: float
    max_calls: int
    max_output_tokens: int
    deadline_unix: float
    max_request_bytes: int = 196_608
    role: str = "worker"

    def __post_init__(self):
        if self.role not in {"worker", "monitor", "planner"}:
            raise ValueError("Responses role must be worker, monitor or planner")
        if not isinstance(self.run_id, str) or _ID.fullmatch(self.run_id) is None:
            raise ValueError("run_id must be a stable identifier")
        for name in ("budget_usd", "deadline_unix"):
            _positive(getattr(self, name), name)
            object.__setattr__(self, name, float(getattr(self, name)))
        for name, ceiling in (("max_calls", 10_000), ("max_output_tokens", 128_000), ("max_request_bytes", 1_048_576)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be a positive integer at most {ceiling}")


def _completion_payload(completion: Completion) -> dict[str, Any]:
    return {
        "text_blocks": list(completion.text_blocks),
        "tool_calls": [asdict(call) for call in completion.tool_calls],
        "stop_reason": completion.stop_reason, "model": completion.model,
        "provider": completion.provider, "usage": asdict(completion.usage),
        "transcript_blocks": list(completion.transcript_blocks),
        "effective_sampling": dict(completion.effective_sampling),
        "provenance": dict(completion.provenance),
    }


def _completion(value: dict[str, Any]) -> Completion:
    return Completion(
        text_blocks=tuple(value["text_blocks"]),
        tool_calls=tuple(ToolCall(**call) for call in value["tool_calls"]),
        stop_reason=value["stop_reason"], model=value["model"], provider=value["provider"],
        usage=Usage(**value["usage"]), transcript_blocks=tuple(value["transcript_blocks"]),
        effective_sampling=value["effective_sampling"], provenance=value["provenance"],
    )


class ResponsesSession:
    """One private run journal, one Azure facade, one owned call at a time.

    The directory belongs to the trusted controller; task tools must not be
    able to write it. A fresh facade on reopen is bounded by this journal's
    cumulative costs, not by its own empty process-local statistics.
    """

    @classmethod
    def create(cls, directory: Path, binding: ResponsesBinding, azure: AzureOpenAIConfig):
        return cls(directory, binding, azure, fresh=True)

    @classmethod
    def open(cls, directory: Path, binding: ResponsesBinding, azure: AzureOpenAIConfig):
        return cls(directory, binding, azure, fresh=False)

    def __init__(self, directory, binding, azure, *, fresh):
        deployment = azure.deployment_for(binding.model)
        if binding.endpoint != azure.base_url or binding.deployment != deployment.deployment:
            raise ResponsesConflict("session binding differs from the Azure route")
        self.binding = binding
        self.directory = Path(directory).absolute()
        self._pid = os.getpid()
        self._thread = threading.get_ident()
        self._loop = None
        self._lock = asyncio.Lock()
        self._active = None
        self._fenced = False
        self._closed = False
        self._lease = None
        self._db = None
        self._llm = LLM(azure_openai=azure, budget_usd=binding.budget_usd,
                        cap_on="billed", max_attempts=1, run_id=binding.run_id)
        identity_binding = asdict(binding)
        # Existing /1 journals predate the role field and are worker-only.
        # Preserve their exact identity; every other role is explicitly bound.
        if binding.role == "worker":
            identity_binding.pop("role")
        identity = _json({"schema": _VERSION, "binding": identity_binding, "pricing_sha": table_sha()})
        try:
            if fresh:
                self.directory.mkdir(mode=0o700)
            info = self.directory.lstat()
            if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid():
                raise ResponsesConflict("session directory must be private and controller-owned")
            self._lease = os.open(self.directory / "owner.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            lease_info = os.fstat(self._lease)
            if not stat.S_ISREG(lease_info.st_mode) or lease_info.st_nlink != 1 or stat.S_IMODE(lease_info.st_mode) != 0o600 or lease_info.st_uid != os.getuid():
                raise ResponsesConflict("session lease must be a private regular file")
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            database = self.directory / "calls.sqlite3"
            if fresh:
                fd = os.open(database, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                os.close(fd)
            info = database.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
                raise ResponsesConflict("session database must be a private regular file")
            self._db = sqlite3.connect(f"file:{quote(str(database))}?mode=rw", uri=True, timeout=0)
            self._db.execute("PRAGMA synchronous=FULL")
            if fresh:
                with self._db:
                    self._db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                    self._db.execute("CREATE TABLE calls (id TEXT PRIMARY KEY, request TEXT NOT NULL, status TEXT NOT NULL, result TEXT, digest TEXT, error_type TEXT)")
                    self._db.execute("CREATE INDEX calls_status ON calls(status)")
                    self._db.execute("INSERT INTO meta VALUES ('identity', ?), ('phase', 'ready'), ('known_cost_usd', '0.0')", (identity,))
                    self._db.execute(f"PRAGMA user_version={_VERSION}")
                for directory_to_sync in (self.directory, self.directory.parent):
                    parent = os.open(directory_to_sync, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(parent)
                    finally:
                        os.close(parent)
            if self._db.execute("PRAGMA user_version").fetchone()[0] != _VERSION:
                raise ResponsesConflict("unsupported Responses journal version")
            if self._db.execute("SELECT value FROM meta WHERE key='identity'").fetchone() != (identity,):
                raise ResponsesConflict("durable Responses binding changed")
            if self._db.execute("SELECT value FROM meta WHERE key='phase'").fetchone() not in {("ready",), ("fenced",)}:
                raise ResponsesConflict("invalid Responses admission phase")
            self._fenced = self._db.execute("SELECT value FROM meta WHERE key='phase'").fetchone()[0] == "fenced"
            self._audit()
            if self._db.execute("SELECT 1 FROM calls WHERE status IN ('pending','unknown') LIMIT 1").fetchone():
                self._fence()
        except BaseException:
            self._release()
            raise

    def _check(self):
        if self._closed or os.getpid() != self._pid or threading.get_ident() != self._thread:
            raise ResponsesFenced("session is closed or belongs to another process/thread")

    def _on_loop(self):
        self._check()
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise ResponsesConflict("session operations require one event loop")
        self._loop = loop

    def _fence(self):
        self._fenced = True
        with self._db:
            self._db.execute("UPDATE meta SET value='fenced' WHERE key='phase'")

    def _receipt(self, result, digest):
        if not isinstance(result, str) or _digest(result) != digest:
            raise ResponsesConflict("Responses receipt digest is invalid")
        try:
            completion = _completion(json.loads(result))
            if completion.model != self.binding.model or completion.provider != "openai":
                raise ValueError("model identity changed")
            if completion.provenance.get("endpoint") != self.binding.endpoint or completion.provenance.get("deployment") != self.binding.deployment:
                raise ValueError("route identity changed")
            usage = completion.usage
            for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens"):
                if type(getattr(usage, name)) is not int or getattr(usage, name) < 0:
                    raise ValueError("usage is invalid")
            if usage.reasoning_tokens > usage.output_tokens:
                raise ValueError("reasoning exceeds output")
            if usage.output_tokens > self.binding.max_output_tokens or usage.prompt_total > ensure_priced(self.binding.model).context_window:
                raise ValueError("usage exceeds the admitted model limits")
            return completion
        except (TypeError, ValueError, KeyError) as exc:
            raise ResponsesConflict("Responses receipt is malformed") from exc

    def _audit(self):
        total = 0.0
        for request, status, result, digest in self._db.execute("SELECT request,status,result,digest FROM calls ORDER BY rowid"):
            if _json(json.loads(request)) != request or status not in {"pending", "unknown", "completed", "not_dispatched"}:
                raise ResponsesConflict("Responses request history is malformed")
            if status == "completed":
                total = math.fsum((total, self._receipt_cost(self._receipt(result, digest))))
            elif result is not None or digest is not None:
                raise ResponsesConflict("unsettled Responses call has a receipt")
        if total != self.known_cost_usd:
            raise ResponsesConflict("Responses cumulative cost differs from its receipts")

    def _receipt_cost(self, completion):
        usage = completion.usage
        billed, _ = call_cost(self.binding.model, input_tokens=usage.input_tokens,
                              output_tokens=usage.output_tokens, cache_read_tokens=usage.cache_read_tokens,
                              cache_write_tokens=usage.cache_write_tokens)
        return billed

    @property
    def known_cost_usd(self) -> float:
        self._check()
        row = self._db.execute("SELECT value FROM meta WHERE key='known_cost_usd'").fetchone()
        try:
            value = float(row[0])
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid cost")
        except (TypeError, ValueError) as exc:
            raise ResponsesConflict("Responses cumulative cost is malformed") from exc
        return value

    @property
    def fenced(self) -> bool:
        self._check()
        return self._fenced

    @property
    def unsettled(self) -> bool:
        self._check()
        return bool(self._db.execute("SELECT 1 FROM calls WHERE status IN ('pending','unknown') LIMIT 1").fetchone())

    def call_accounting(self) -> ModelCallAccounting:
        self._check()
        counts = dict(self._db.execute("SELECT status,count(*) FROM calls GROUP BY status"))
        return ModelCallAccounting(
            known_cost_usd=self.known_cost_usd,
            completed_calls=counts.get("completed", 0),
            unknown_calls=counts.get("pending", 0) + counts.get("unknown", 0),
        )

    def ensure_ready(self) -> None:
        """Validate SDK/credential configuration without dispatch or spending."""
        self._check()
        self._llm.ensure_ready(self.binding.model)

    def pin_context(self, context_id: str, value: dict[str, Any]) -> dict[str, Any]:
        """Persist the first observation before its potentially paid request.

        The caller binds context_id to all immutable evidence and verifies a
        recovered observation against it. Time-of-observation fields can then
        retain their original value when publication is retried much later.
        These snapshots do not count as provider calls or reset spending.
        """
        self._check()
        if not isinstance(context_id, str) or _ID.fullmatch(context_id) is None:
            raise ValueError("context_id must be a stable identifier")
        key = "context:" + context_id
        prior = self._db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if prior is not None:
            envelope = json.loads(prior[0])
            raw = envelope["value"]
            if (not isinstance(raw, dict) or _digest(_json(raw)) != envelope["digest"]
                    or len(_json(raw).encode()) > self.binding.max_request_bytes):
                raise ResponsesConflict("pinned Responses context is malformed")
            return raw
        if not isinstance(value, dict):
            raise ValueError("Responses context must be an object")
        raw = _json(value)
        if len(raw.encode()) > self.binding.max_request_bytes:
            raise ValueError("Responses context exceeds the admitted byte limit")
        if self._fenced or self.unsettled:
            raise ResponsesFenced("Responses session is fenced")
        if time.time() >= self.binding.deadline_unix:
            self._fence()
            raise ResponsesFenced("Responses session deadline elapsed")
        if self._db.execute("SELECT count(*) FROM meta WHERE key LIKE 'context:%'").fetchone()[0] >= self.binding.max_calls:
            raise ResponsesFenced("Responses context limit reached")
        with self._db:
            self._db.execute("INSERT INTO meta VALUES (?,?)", (key, _json({"value": value, "digest": _digest(raw)})))
        return json.loads(raw)

    def lookup(self, request_id: str) -> Completion | None:
        self._check()
        row = self._db.execute("SELECT status,result,digest FROM calls WHERE id=?", (request_id,)).fetchone()
        if row is None or row[0] == "not_dispatched":
            return None
        if row[0] != "completed":
            raise ResponsesFenced("Responses call has an unknown outcome")
        return self._receipt(row[1], row[2])

    async def complete(self, request_id: str, *, system, messages, tools=None, effort="low") -> Completion:
        self._on_loop()
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError()
        if not isinstance(request_id, str) or _ID.fullmatch(request_id) is None:
            raise ValueError("request_id must be a stable identifier")
        request = _json({"system": system, "messages": messages, "tools": tools, "effort": effort})
        if len(request.encode()) > self.binding.max_request_bytes:
            raise ValueError("Responses request exceeds the admitted byte limit")
        async with self._lock:
            self._check()
            prior = self._db.execute("SELECT request FROM calls WHERE id=?", (request_id,)).fetchone()
            if prior is not None:
                if prior[0] != request:
                    raise ResponsesConflict("request_id was reused with different input")
                result = self.lookup(request_id)
                if result is not None:
                    return result
                raise ResponsesFenced("the prior call was not dispatched; this run ended")
            if self._fenced or self.unsettled:
                raise ResponsesFenced("Responses session is fenced")
            if time.time() >= self.binding.deadline_unix:
                self._fence()
                raise ResponsesFenced("Responses session deadline elapsed")
            count = self._db.execute("SELECT count(*) FROM calls").fetchone()[0]
            if count >= self.binding.max_calls:
                raise ResponsesFenced("Responses session call limit reached")
            exposure = max_call_cost_usd(self.binding.model, max_output_tokens=self.binding.max_output_tokens,
                                         max_attempts=1, cap_on="billed")
            spent = self.known_cost_usd
            if spent + exposure > self.binding.budget_usd:
                raise BudgetExceeded(spent, self.binding.budget_usd, required_usd=exposure)
            self._llm.ensure_ready(self.binding.model)
            with self._db:
                self._db.execute("INSERT INTO calls (id,request,status) VALUES (?,?,'pending')", (request_id, request))
            task = asyncio.create_task(self._invoke(request_id, json.loads(request)), name="taste-responses-call")
            self._active = task
            cancelled = None
            errors = []
            try:
                while not task.done():
                    try:
                        await asyncio.wait((task,))
                    except asyncio.CancelledError as exc:
                        cancelled = exc
                        if not self._fenced:
                            try:
                                self._fence()
                            except BaseException as failure:
                                errors.append(failure)
                try:
                    result = task.result()
                except BaseException as exc:
                    errors.append(exc)
                if cancelled is not None:
                    if errors:
                        raise BaseExceptionGroup("cancelled Responses call failed while settling", [cancelled, *errors])
                    raise cancelled
                if errors:
                    if len(errors) == 1:
                        raise errors[0]
                    raise BaseExceptionGroup("Responses call settlement failed", errors)
                return result
            finally:
                self._active = None

    def _dispatch(self, request):
        try:
            remaining = self.binding.deadline_unix - time.time()
            if remaining <= 0:
                raise _NotDispatched("deadline elapsed before provider invocation")
            return self._llm.call(model=self.binding.model, max_tokens=self.binding.max_output_tokens,
                                  role=self.binding.role, temperature=None,
                                  timeout_seconds=remaining, **request)
        except (KeyboardInterrupt, SystemExit) as exc:
            raise BaseExceptionGroup("Responses provider was interrupted", [exc]) from None

    async def _invoke(self, request_id, request):
        if self._fenced or time.time() >= self.binding.deadline_unix:
            with self._db:
                self._db.execute("UPDATE calls SET status='not_dispatched' WHERE id=?", (request_id,))
            raise ResponsesFenced("call stopped before provider dispatch")
        try:
            # asyncio.run shutdown can cancel the owner Task itself, not only
            # its caller. Awaiting to_thread directly would abandon a running
            # HTTP operation in that case. This Future is not a Task, and wait
            # never propagates cancellation into it.
            operation = start_owned_thread(self._dispatch, request)
            interrupted = None
            wait_errors = []
            while not operation.done():
                try:
                    await asyncio.wait((operation,))
                except asyncio.CancelledError as exc:
                    interrupted = exc
                    try:
                        self._fence()
                    except BaseException as failure:
                        wait_errors.append(failure)
            try:
                completion = operation.result()
            except BaseException as failure:
                if interrupted is not None or wait_errors:
                    raise BaseExceptionGroup("interrupted provider failed while settling",
                                             [*([interrupted] if interrupted else []), *wait_errors, failure]) from None
                raise
            payload = _json(_completion_payload(completion))
            self._receipt(payload, _digest(payload))
            self._save_receipt(request_id, payload)
            if interrupted is not None or wait_errors:
                raise BaseExceptionGroup("Responses owner was interrupted while settling",
                                         [*([interrupted] if interrupted else []), *wait_errors])
            return _completion(json.loads(payload))
        except _NotDispatched:
            self._fenced = True
            with self._db:
                self._db.execute("UPDATE calls SET status='not_dispatched' WHERE id=?", (request_id,))
                self._db.execute("UPDATE meta SET value='fenced' WHERE key='phase'")
            raise ResponsesFenced("deadline elapsed before provider invocation") from None
        except BaseException as exc:
            self._fenced = True
            try:
                with self._db:
                    # A receipt write can commit and then raise. Preserve that
                    # recoverable response rather than relabel paid known work
                    # as unknown; the run still stops on the persistence fault.
                    self._db.execute("UPDATE calls SET status='unknown',error_type=? WHERE id=? AND status='pending'", (type(exc).__name__, request_id))
                    self._db.execute("UPDATE meta SET value='fenced' WHERE key='phase'")
            except BaseException as persistence_error:
                raise BaseExceptionGroup("Responses failure could not be persisted", [exc, persistence_error]) from None
            raise

    def _save_receipt(self, request_id, payload):
        cost = self._receipt_cost(self._receipt(payload, _digest(payload)))
        with self._db:
            total = math.fsum((self.known_cost_usd, cost))
            changed = self._db.execute("UPDATE calls SET status='completed',result=?,digest=? WHERE id=? AND status='pending'", (payload, _digest(payload), request_id))
            if changed.rowcount != 1:
                raise ResponsesConflict("Responses receipt lost its exact pending intent")
            self._db.execute("UPDATE meta SET value=? WHERE key='known_cost_usd'", (_json(total),))

    def close(self):
        if self._closed:
            return
        self._check()
        if self._active is not None:
            raise ResponsesFenced("cannot release an active Responses call")
        try:
            for provider in self._llm._providers.values():
                if provider._client is not None:
                    provider._client.close()
        finally:
            self._release()

    def _release(self):
        try:
            if self._db is not None:
                self._db.close()
        finally:
            if self._lease is not None:
                os.close(self._lease)
                self._lease = None
            self._closed = True
