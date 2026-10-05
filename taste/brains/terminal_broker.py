"""Own serial terminal effects in one authoritative task environment.

The ledger belongs to the trusted controller, OUTSIDE Git rollback and task
write access. A request ID identifies an effect, not an attempt: a completed
request can be read again, but an interrupted request can never be replayed.
Normal nonzero exits are command results. A backend that can end one command
and confirm its exit turns a timeout or a departed caller into a completed,
explicitly terminated receipt; the environment and later commands survive.
Without that confirmation, an uncertain exit, timeout or active cancellation
still ends admission and requires stopping the entire environment.
Package/service changes persist between successful requests. A backend that
can also take and restore checkpoints of the task's files lets the controller
record them between commands and put one back (``checkpoint``, ``restore``):
files only, never running processes, and only on the controller's request.

This is the ownership core, not a Harbor agent, credential boundary or sandbox.
The backend must bind an exact environment instance, bound its transport/output
resources, and confirm whole-environment termination independently of exec.
An outside owner must also stop that environment if this controller is killed.
Merely cancelling a Docker/Harbor exec client does not satisfy that contract.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import math
import os
import re
import sqlite3
import stat
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from taste.brains.owned_thread import start_owned_thread

MAX_TERMINAL_OUTPUT_BYTES = 1024 * 1024  # Per stream, before persistence or IPC.
# How long a backend may take to end one command and confirm its exit, beyond
# the command's own allowance. Past it the whole environment is stopped.
COMMAND_SETTLE_SECONDS = 20
TERMINATIONS = ("", "timeout", "cancelled")
# One checkpoint copies at most this much; the store of one trial holds at most
# the second. Past the first a checkpoint is partial (it lists what it left out);
# past the second none is taken.
CHECKPOINT_CAP_BYTES = 512 * 1024 * 1024
CHECKPOINT_STORE_BYTES = 4 * 1024 * 1024 * 1024
ENVIRONMENT_EVENTS = ("checkpoint_intent", "checkpoint", "checkpoint_failed",
                      "restore_intent", "restored", "restore_failed")


class TerminalFenced(RuntimeError):
    """This environment may not accept another terminal effect."""


class TerminalConflict(ValueError):
    """A request ID or environment identity was reused with different inputs."""


class CheckpointStoreFull(RuntimeError):
    """The controller's checkpoint store has no room for another checkpoint."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identifier(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) is None:
        raise ValueError("terminal identities must be 1-128 ASCII letters, digits, '_' or '-'")


def _positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("terminal time limits must be finite and positive")


@dataclass(frozen=True)
class TerminalBinding:
    trial_id: str
    environment_id: str
    deadline_unix: float
    max_commands: int

    def __post_init__(self):
        _identifier(self.trial_id)
        _identifier(self.environment_id)
        _positive(self.deadline_unix)
        if type(self.max_commands) is not int or not 1 <= self.max_commands <= 100000:
            raise ValueError("max_commands must be between 1 and 100000")


@dataclass(frozen=True)
class TerminalRequest:
    request_id: str
    actor_id: str
    command: str
    cwd: str
    timeout_seconds: float

    def __post_init__(self):
        _identifier(self.request_id)
        _identifier(self.actor_id)
        if (not isinstance(self.command, str) or not self.command or "\x00" in self.command
                or len(self.command.encode()) > 65536):
            raise ValueError("command must be nonempty text without NUL, at most 64 KiB")
        if (not isinstance(self.cwd, str) or not PurePosixPath(self.cwd).is_absolute()
                or "\x00" in self.cwd or len(self.cwd.encode()) > 4096):
            raise ValueError("cwd must be an absolute task-environment path, at most 4 KiB")
        _positive(self.timeout_seconds)


@dataclass(frozen=True)
class TerminalResult:
    """Captured stream prefixes and exact byte counts discarded while draining."""

    return_code: int
    stdout: bytes = b""
    stderr: bytes = b""
    stdout_dropped_bytes: int = 0
    stderr_dropped_bytes: int = 0
    # "timeout" or "cancelled": the backend ended this command and confirmed
    # its exit. The output is what was captured before that; effects may be partial.
    terminated: str = ""

    def __post_init__(self):
        if type(self.return_code) is not int:
            raise ValueError("terminal return code must be an integer")
        if self.terminated not in TERMINATIONS:
            raise ValueError("terminal termination cause must be empty, timeout or cancelled")
        if not isinstance(self.stdout, bytes) or not isinstance(self.stderr, bytes):
            raise ValueError("terminal output must preserve bytes")
        for output, dropped in ((self.stdout, self.stdout_dropped_bytes),
                                (self.stderr, self.stderr_dropped_bytes)):
            if len(output) > MAX_TERMINAL_OUTPUT_BYTES:
                raise ValueError("terminal stream exceeds the 1 MiB receipt limit")
            if type(dropped) is not int or not 0 <= dropped <= 2**63 - 1:
                raise ValueError("dropped byte count must be a nonnegative SQLite integer")


class TerminalBackend(Protocol):
    """Trusted finite transports, called in owned threads, never on the loop.

    execute and stop_and_confirm can overlap. Stop must prevent a delayed exec
    from launching after confirmation (e.g. stop/remove the exact container,
    never create/restart it). It must drain all task descendants and cause any
    in-flight execute call to return. A failed stop must raise, not return a
    receipt. Identity denotes the instance, not an image or human-readable name.

    A backend may also define ``interrupt()``. It then owns each command's
    allowance: at the request timeout, or when interrupt() is called, execute
    ends that command alone, confirms its exit and returns a receipt whose
    ``terminated`` names the cause. If the exit cannot be confirmed it raises.

    A backend may also define ``checkpoint(directory, *, cap_bytes)``, which
    returns a manifest with ``to_dict()`` and ``summary()``, and
    ``restore(manifest, directory)``, which returns a receipt with
    ``summary()``. Both are called only between commands.
    """

    @property
    def environment_id(self) -> str: ...

    def execute(self, request: TerminalRequest) -> TerminalResult: ...

    def stop_and_confirm(self) -> str: ...


def _call(call, *args):
    try:
        return call(*args)
    except (KeyboardInterrupt, SystemExit) as exc:
        raise BaseExceptionGroup("terminal transport interrupted", [exc]) from None


async def _settle(task):
    while not task.done():
        try:
            await asyncio.wait((task,))
        except asyncio.CancelledError:
            continue
    return task.result()


class TerminalBroker:
    """One controller lease, one active effect, durable at-most-once admission.

    Use create for a new trial and open for the same retained environment.
    close only releases the controller lease; abort stops the environment.
    seal_for_grading permanently ends command admission once commands are idle,
    leaving the environment alive for its outside verifier and cleanup owner.
    An incomplete recovered request fences the trial even if exec never began.
    Call abort to confirm drainage; do not invent a new request ID to retry it.
    All async methods of an instance belong to one event loop.
    """

    @classmethod
    def create(cls, directory: Path, binding: TerminalBinding, backend: TerminalBackend):
        directory = Path(directory)
        directory.mkdir(mode=0o700)
        return cls(directory, binding, backend, fresh=True)

    @classmethod
    def open(cls, directory: Path, binding: TerminalBinding, backend: TerminalBackend):
        return cls(Path(directory), binding, backend, fresh=False)

    def __init__(self, directory, binding, backend, *, fresh):
        directory = directory.absolute()
        self.binding, self.backend = binding, backend
        self._directory = directory
        # Checkpoints this controller took, by ID. A reopened controller can
        # list earlier ones from its ledger but not restore them.
        self._manifests = {}
        self._interruptible = callable(getattr(backend, "interrupt", None))
        self._pid, self._fd, self._db = os.getpid(), None, None
        self._operation, self._loop = None, None
        self._interrupt = None
        self._abort_task = None
        self._lock = asyncio.Lock()
        self._closing = False
        try:
            self._identity()
            self._fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            info = os.fstat(self._fd)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise TerminalFenced("terminal ledger must be private and controller-owned")
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            database = directory / "terminal.sqlite3"
            if fresh:
                fd = os.open(database, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                os.close(fd)
                os.fsync(self._fd)
            info = database.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                raise TerminalFenced("terminal database must be a private regular file")
            self._db = sqlite3.connect(database.as_uri() + "?mode=rw", uri=True)
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute("PRAGMA journal_mode=DELETE")
            if fresh:
                with self._db:
                    self._db.execute("CREATE TABLE binding (payload TEXT NOT NULL)")
                    self._db.execute("CREATE TABLE state (phase TEXT NOT NULL)")
                    self._db.execute("CREATE TABLE events (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL)")
                    self._db.execute("CREATE TABLE requests (id TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL, code INTEGER, stdout BLOB, stderr BLOB)")
                    self._db.execute("INSERT INTO binding VALUES (?)", (_json(asdict(binding)),))
                    self._db.execute("INSERT INTO state VALUES ('ready')")
                    self._event("created", asdict(binding))
                os.fsync(self._fd)
            if self._db.execute("SELECT payload FROM binding").fetchall() != [(_json(asdict(binding)),)]:
                raise TerminalConflict("terminal binding differs from the admitted trial")
            if self._db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise TerminalFenced("terminal database integrity check failed")
            self._upgrade_schema()
            if self._db.execute("SELECT 1 FROM requests WHERE status='pending' LIMIT 1").fetchone():
                self._fence("recovered incomplete request")
            self._recover_environment()
        except BaseException:
            self._release()
            raise

    def _upgrade_schema(self):
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        columns = [row[1] for row in self._db.execute("PRAGMA table_info(requests)")]
        legacy = ["id", "payload", "status", "code", "stdout", "stderr"]
        counted = [*legacy, "stdout_dropped_bytes", "stderr_dropped_bytes"]
        current = [*counted, "terminated"]
        if (version, columns) in ((0, legacy), (1, counted)):
            # Old transports retained all bytes and ended no command alone.
            # Upgrade atomically under the controller lease; pending effects
            # still fence below, never replay.
            with self._db:
                self._db.execute("BEGIN IMMEDIATE")
                for column in counted[len(columns):]:
                    self._db.execute(f"ALTER TABLE requests ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
                self._db.execute("ALTER TABLE requests ADD COLUMN terminated TEXT NOT NULL DEFAULT ''")
                self._db.execute("PRAGMA user_version=2")
        elif version != 2 or columns != current:
            raise TerminalFenced("unsupported terminal receipt schema")

    def _identity(self):
        if self.backend.environment_id != self.binding.environment_id:
            raise TerminalConflict("backend no longer identifies the admitted environment")

    def _check(self):
        if os.getpid() != self._pid or self._db is None:
            raise TerminalFenced("terminal controller is closed or belongs to another process")

    def _on_loop(self):
        self._check()
        loop = asyncio.get_running_loop()
        if self._loop is not None and loop is not self._loop:
            raise TerminalFenced("terminal controller belongs to another event loop")
        self._loop = loop

    def _event(self, kind, payload):
        self._db.execute("INSERT INTO events(kind,payload) VALUES (?,?)", (kind, _json(payload)))

    @property
    def phase(self):
        self._check()
        rows = self._db.execute("SELECT phase FROM state").fetchall()
        if len(rows) != 1 or rows[0][0] not in {"ready", "sealed", "fenced", "stopped"}:
            raise TerminalFenced("invalid terminal ledger state")
        return rows[0][0]

    def events(self):
        self._check()
        return [(kind, json.loads(payload)) for kind, payload in
                self._db.execute("SELECT kind,payload FROM events ORDER BY seq")]

    def lookup(self, request: TerminalRequest):
        self._check()
        row = self._db.execute("SELECT payload,status,code,stdout,stderr,stdout_dropped_bytes,stderr_dropped_bytes,terminated FROM requests WHERE id=?",
                               (request.request_id,)).fetchone()
        if row is None:
            return None
        if row[0] != _json(asdict(request)):
            raise TerminalConflict("terminal request ID was reused with different inputs")
        if row[1] != "completed":
            raise TerminalFenced("terminal request has no confirmed completed receipt")
        return TerminalResult(*row[2:])

    def lookup_actor(self, request_id: str, actor_id: str):
        """Read one actor's completed receipt without allowing an effect replay."""
        self._check()
        _identifier(request_id)
        _identifier(actor_id)
        row = self._db.execute("SELECT payload FROM requests WHERE id=?", (request_id,)).fetchone()
        if row is None:
            return None
        request = TerminalRequest(**json.loads(row[0]))
        if request.actor_id != actor_id:
            raise TerminalConflict("terminal receipt belongs to another actor")
        return self.lookup(request)

    def seal_for_grading(self) -> TerminalBinding:
        """Irreversibly hand an idle environment to the outside verifier.

        The lifecycle owner must first drain agent processes. This synchronous
        transition on the broker loop cannot race admission. A queued request
        will see the sealed state; an already active request makes sealing fail.
        Completed receipts remain readable, including after controller recovery.
        This proves no future broker effect, not that task background services
        have stopped. Those services belong to the live task being graded.
        """
        self._on_loop()
        self._identity()
        if self.phase == "sealed" and not self._closing:
            return self.binding
        if (self._closing or self.phase != "ready" or self._lock.locked()
                or (self._operation is not None and not self._operation.done())
                or (self._abort_task is not None and not self._abort_task.done())
                or self._db.execute("SELECT 1 FROM requests WHERE status='pending' LIMIT 1").fetchone()):
            raise TerminalFenced("cannot hand an active or uncertain terminal environment to grading")
        with self._db:
            self._db.execute("UPDATE state SET phase='sealed'")
            self._event("sealed", {"environment_id": self.binding.environment_id})
        return self.binding

    def _fence(self, reason):
        with self._db:
            self._db.execute("UPDATE state SET phase='fenced'")
            self._event("fenced", {"reason": reason})

    def _record_result(self, request, result, *, completed):
        if not isinstance(result, TerminalResult):
            raise TypeError("backend returned no TerminalResult")
        with self._db:
            self._db.execute("UPDATE requests SET status=?,code=?,stdout=?,stderr=?,stdout_dropped_bytes=?,stderr_dropped_bytes=?,terminated=? WHERE id=?",
                             ("completed" if completed else "uncertain", result.return_code,
                              result.stdout, result.stderr, result.stdout_dropped_bytes,
                              result.stderr_dropped_bytes, result.terminated, request.request_id))
            self._event("completed" if completed else "late_result",
                        {"request_id": request.request_id,
                         **({"terminated": result.terminated} if result.terminated else {})})

    async def execute(self, request: TerminalRequest) -> TerminalResult:
        self._on_loop()
        if not isinstance(request, TerminalRequest):
            raise TypeError("a validated TerminalRequest is required")
        async with self._lock:
            cached = self.lookup(request)
            if cached is not None:
                return cached
            if self._closing or self.phase != "ready":
                raise TerminalFenced("terminal environment is no longer admitting commands")
            self._identity()
            remaining = self.binding.deadline_unix - time.time()
            if remaining <= 0:
                raise TerminalFenced("terminal admission deadline has expired")
            count = self._db.execute("SELECT count(*) FROM requests").fetchone()[0]
            if count >= self.binding.max_commands:
                raise TerminalFenced("terminal command limit reached")
            # No await between the intent transaction and installing its owner.
            with self._db:
                self._db.execute("INSERT INTO requests(id,payload,status) VALUES (?,?,'pending')",
                                 (request.request_id, _json(asdict(request))))
                self._event("intent", asdict(request))
            self._interrupt = asyncio.get_running_loop().create_future()
            if asyncio.current_task().cancelling():
                self._interrupt.set_result(None)
            self._operation = asyncio.create_task(self._execute_owned(request, self._interrupt))
            try:
                await asyncio.wait((self._operation,))
                return self._operation.result()
            except asyncio.CancelledError as cancelled:
                if not self._interrupt.done():
                    self._interrupt.set_result(None)
                try:
                    # Whole-loop shutdown can cancel the owner before its
                    # coroutine enters the cleanup handler. Its intent is
                    # durable, but the task environment still needs a stop.
                    with suppress(asyncio.CancelledError):
                        await _settle(self._operation)
                    if self._ended_alone(request):
                        # One caller left. Its command has a durable receipt
                        # and a confirmed exit; other actors keep the environment.
                        raise cancelled
                    self._closing = True
                    # Cancellation can race a fully persisted receipt. The
                    # effect stays completed, but the caller still ended the trial.
                    if self.phase != "stopped":
                        self._fence("terminal caller cancelled after completion")
                        await self._stop_owned()
                except asyncio.CancelledError:
                    raise cancelled from None
                except BaseException as failure:
                    raise BaseExceptionGroup("terminal cancellation and settlement failed", [cancelled, failure]) from None
                raise

    def _ended_alone(self, request):
        if not self._interruptible or self._closing or self.phase != "ready":
            return False
        row = self._db.execute("SELECT status FROM requests WHERE id=?", (request.request_id,)).fetchone()
        return row is not None and row[0] == "completed"

    async def _stop_owned(self):
        self._identity()
        receipt = await _settle(start_owned_thread(_call, self.backend.stop_and_confirm))
        if receipt != self.binding.environment_id:
            raise TerminalConflict("termination receipt identifies another environment")
        with self._db:
            self._db.execute("UPDATE state SET phase='stopped'")
            self._db.execute("UPDATE requests SET status='uncertain' WHERE status='pending'")
            self._event("stopped", {"environment_id": receipt})

    def _ready_for_transport(self):
        # The thread pool can queue work after the event-loop admission check.
        if self.binding.deadline_unix <= time.time():
            raise TimeoutError("terminal deadline expired before transport invocation")
        self._identity()

    def _execute_backend(self, request):
        self._ready_for_transport()
        return self.backend.execute(request)

    async def _execute_owned(self, request, interrupt):
        effect = None
        ended_alone = False
        try:
            if interrupt.done():
                raise asyncio.CancelledError()
            timeout = min(request.timeout_seconds, self.binding.deadline_unix - time.time())
            if timeout <= 0:
                raise TimeoutError("terminal deadline expired before execution")
            effect = start_owned_thread(_call, self._execute_backend, request)
            # An interruptible backend ends an overdue command itself and
            # returns its receipt; allow it the time to confirm that exit.
            settle = COMMAND_SETTLE_SECONDS if self._interruptible else 0
            done, _ = await asyncio.wait((effect, interrupt), timeout=timeout + settle,
                                         return_when=asyncio.FIRST_COMPLETED)
            if interrupt in done and not self._interruptible:
                raise asyncio.CancelledError()
            if effect not in done and interrupt in done:
                await _settle(start_owned_thread(_call, self.backend.interrupt))
                done, _ = await asyncio.wait((effect,), timeout=settle)
                if effect not in done:
                    raise TimeoutError("terminal command could not be ended within its settlement bound")
                result = effect.result()
                self._identity()
                self._record_result(request, result, completed=True)
                ended_alone = True
                raise asyncio.CancelledError()
            if effect not in done:
                raise TimeoutError("terminal execution exceeded its admitted time")
            result = effect.result()
            self._identity()
            self._record_result(request, result, completed=True)
            return result
        except BaseException as original:
            if ended_alone:
                raise
            failures = [original]
            self._closing = True
            try:
                self._fence("terminal execution did not complete durably")
            except BaseException as failure:
                failures.append(failure)
            try:
                await self._stop_owned()
            except BaseException as failure:
                failures.append(failure)
            # Retain ownership even when stopping failed: a live transport may
            # still write. The outer controller must enforce a hard deadline.
            if effect is not None:
                try:
                    result = await _settle(effect)
                    self._record_result(request, result, completed=False)
                except BaseException as failure:
                    if failure is not original:
                        failures.append(failure)
            if len(failures) > 1:
                raise BaseExceptionGroup("terminal execution and settlement failed", failures) from None
            raise

    # -- checkpoints of the task's files -----------------------------------------

    def checkpoints(self):
        """Every checkpoint taken, in order, as its summary: never a tar or a path on disk."""
        self._check()
        return [payload for kind, payload in self._environment_events(("checkpoint",))]

    async def checkpoint(self, checkpoint_id: str) -> dict:
        """Record the task's files as they are, between commands; at most once per ID.

        Returns the checkpoint's summary under its ID. When the backend fails,
        returns a record with ``failed`` and the error's class instead: taking
        a checkpoint only reads, so the environment keeps admitting commands.
        """
        self._on_loop()
        _identifier(checkpoint_id)
        self._can_checkpoint()
        async with self._lock:
            prior = self._recorded("checkpoint_id", checkpoint_id, ("checkpoint", "checkpoint_failed"))
            if prior is not None:
                return prior
            self._admitting()
            store = self._store()

            def finished(manifest, failure):
                if failure is None:
                    self._manifests[checkpoint_id] = manifest
                    return "checkpoint", {"checkpoint_id": checkpoint_id, **manifest.summary()}
                return "checkpoint_failed", {"checkpoint_id": checkpoint_id, "failed": True,
                                             "error": type(failure).__name__}

            return await self._environment("checkpoint_intent", {"checkpoint_id": checkpoint_id}, finished,
                                           self._take, checkpoint_id, store)

    async def restore(self, operation_id: str, checkpoint_id: str) -> dict:
        """Return the task's files to a checkpoint this controller took; at most once per operation ID.

        Returns the receipt's summary (``exact`` when the files now match the
        checkpoint). When the backend fails partway the files may be mixed:
        the record says ``failed`` and the environment keeps admitting
        commands, so the trial can still be graded as it stands.
        """
        self._on_loop()
        _identifier(operation_id)
        _identifier(checkpoint_id)
        self._can_checkpoint()
        intent = {"operation_id": operation_id, "checkpoint_id": checkpoint_id}
        async with self._lock:
            prior = self._recorded("operation_id", operation_id, ("restored", "restore_failed"))
            if prior is not None:
                if prior["checkpoint_id"] != checkpoint_id:
                    raise TerminalConflict("restore operation ID was reused for another checkpoint")
                return prior
            if self._recorded("operation_id", operation_id, ("restore_intent",)) is not None:
                raise TerminalFenced("this restore was begun and never recorded")
            self._admitting()
            manifest = self._manifests.get(checkpoint_id)
            if manifest is None:
                raise TerminalConflict("this controller took no checkpoint with that ID")
            store = self._store()

            def finished(receipt, failure):
                if failure is None:
                    return "restored", {**intent, **receipt.summary()}
                return "restore_failed", {**intent, "failed": True, "error": type(failure).__name__}

            return await self._environment("restore_intent", intent, finished, self._put_back, manifest, store)

    def _can_checkpoint(self):
        if not (callable(getattr(self.backend, "checkpoint", None))
                and callable(getattr(self.backend, "restore", None))):
            raise TerminalFenced("this environment's transport cannot take or restore checkpoints")

    def _admitting(self):
        if self._closing or self.phase != "ready":
            raise TerminalFenced("terminal environment is no longer admitting effects")
        self._identity()
        if self.binding.deadline_unix - time.time() <= 0:
            raise TerminalFenced("terminal admission deadline has expired")

    def _store(self):
        """The private directory of checkpoint tars and manifests, beside the ledger."""
        store = self._directory / "checkpoints"
        with suppress(FileExistsError):
            store.mkdir(mode=0o700)
        info = store.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise TerminalFenced("checkpoint store must be private and controller-owned")
        return store

    def _environment_events(self, kinds):
        marks = ",".join("?" * len(kinds))
        return [(kind, json.loads(payload)) for kind, payload in self._db.execute(
            f"SELECT kind,payload FROM events WHERE kind IN ({marks}) ORDER BY seq", tuple(kinds))]

    def _recorded(self, key, value, kinds):
        found = [payload for _, payload in self._environment_events(kinds) if payload.get(key) == value]
        return found[-1] if found else None

    def _take(self, checkpoint_id, store):
        self._ready_for_transport()
        used = sum(entry.stat().st_size for entry in store.iterdir()
                   if re.fullmatch(r"[0-9a-f]{64}\.tar", entry.name))
        cap = min(CHECKPOINT_CAP_BYTES, CHECKPOINT_STORE_BYTES - used)
        if cap <= 0:
            raise CheckpointStoreFull("the controller's checkpoint store is full")
        manifest = self.backend.checkpoint(store, cap_bytes=cap)
        fd = os.open(store / f"{checkpoint_id}.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as written:
            written.write(_json(manifest.to_dict()))
            written.flush()
            os.fsync(written.fileno())
        return manifest

    def _put_back(self, manifest, store):
        self._ready_for_transport()
        return self.backend.restore(manifest, store)

    async def _environment(self, kind, intent, finished, call, *args):
        """Run one checkpoint or restore under the caller's lock; return its record."""
        with self._db:
            self._event(kind, intent)
        self._operation = asyncio.create_task(self._environment_owned(finished, call, *args))
        try:
            await asyncio.wait((self._operation,))
        except asyncio.CancelledError:
            # A departing caller never leaves the files mid-copy: the effect
            # finishes and is recorded, and its ID reads the record again.
            with suppress(BaseException):
                await _settle(self._operation)
            raise
        return self._operation.result()

    async def _environment_owned(self, finished, call, *args):
        # The record is written here, not by the caller, so that it is written
        # even when the caller has gone.
        try:
            value, failure = await _settle(start_owned_thread(_call, call, *args)), None
        except BaseException as exc:
            value, failure = None, exc
        kind, record = finished(value, failure)
        with self._db:
            self._event(kind, record)
        if failure is not None and not isinstance(failure, Exception):
            raise failure
        return record

    def _recover_environment(self):
        """After a restart: a checkpoint begun only read files and is recorded as failed;
        a restore begun and never recorded may have left them mixed, which fences."""
        events = self._environment_events(ENVIRONMENT_EVENTS)
        taken = {payload["checkpoint_id"] for kind, payload in events if kind in ("checkpoint", "checkpoint_failed")}
        restored = {payload["operation_id"] for kind, payload in events if kind in ("restored", "restore_failed")}
        for kind, payload in events:
            if kind == "checkpoint_intent" and payload["checkpoint_id"] not in taken:
                with self._db:
                    self._event("checkpoint_failed", {"checkpoint_id": payload["checkpoint_id"], "failed": True,
                                                      "error": "interrupted"})
                taken.add(payload["checkpoint_id"])
            elif (kind == "restore_intent" and payload["operation_id"] not in restored
                    and self.phase not in ("fenced", "stopped")):
                self._fence("recovered incomplete restore")

    async def abort(self):
        """End admission immediately, drain an active request or stop an idle environment."""
        self._on_loop()
        self._closing = True
        if self._interrupt is not None and not self._interrupt.done():
            self._interrupt.set_result(None)
        if self._abort_task is None or self._abort_task.done():
            self._abort_task = asyncio.create_task(self._abort_owned())
        try:
            await asyncio.wait((self._abort_task,))
            return self._abort_task.result()
        except asyncio.CancelledError as cancelled:
            try:
                while True:
                    try:
                        await _settle(self._abort_task)
                        break
                    except asyncio.CancelledError:
                        # A task cancelled before it starts owns no cleanup.
                        # Keep this caller responsible until a stop completes.
                        self._abort_task = asyncio.create_task(self._abort_owned())
            except BaseException as failure:
                raise BaseExceptionGroup("terminal abort cancellation and drainage failed", [cancelled, failure]) from None
            raise

    async def _abort_owned(self):
        operation = self._operation
        if operation is not None and not operation.done():
            with suppress(asyncio.CancelledError):
                await _settle(operation)
        async with self._lock:
            if self.phase == "stopped":
                return
            failure = None
            try:
                self._fence("terminal owner ended the trial")
            except BaseException as exc:
                failure = exc
            self._operation = asyncio.create_task(self._stop_owned())
            try:
                await _settle(self._operation)
            except BaseException as exc:
                if failure is not None:
                    raise BaseExceptionGroup("terminal fence and drainage failed", [failure, exc]) from None
                raise
            if failure is not None:
                raise failure

    def close(self):
        self._check()
        if ((self._operation is not None and not self._operation.done())
                or (self._abort_task is not None and not self._abort_task.done())
                or self.phase == "fenced"
                or (self.phase != "stopped" and self._db.execute(
                    "SELECT 1 FROM requests WHERE status='pending' LIMIT 1").fetchone())):
            raise TerminalFenced("cannot release an undrained terminal operation")
        self._release()

    def _release(self):
        if self._db is not None:
            self._db.close()
            self._db = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
