"""Own serial terminal effects in one authoritative task environment.

The ledger belongs to the trusted controller, OUTSIDE Git rollback and task
write access. A request ID identifies an effect, not an attempt: a completed
request can be read again, but an interrupted request can never be replayed.
Normal nonzero exits are command results. An uncertain exit, timeout or active
cancellation ends admission and requires stopping the entire environment.
Package/service changes persist between successful requests; this module does
not promise filesystem or operating-system rollback.

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


class TerminalFenced(RuntimeError):
    """This environment may not accept another terminal effect."""


class TerminalConflict(ValueError):
    """A request ID or environment identity was reused with different inputs."""


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
    return_code: int
    stdout: bytes = b""
    stderr: bytes = b""

    def __post_init__(self):
        if type(self.return_code) is not int:
            raise ValueError("terminal return code must be an integer")
        if not isinstance(self.stdout, bytes) or not isinstance(self.stderr, bytes):
            raise ValueError("terminal output must preserve bytes")


class TerminalBackend(Protocol):
    """Trusted finite transports, called in owned threads, never on the loop.

    execute and stop_and_confirm can overlap. Stop must prevent a delayed exec
    from launching after confirmation (e.g. stop/remove the exact container,
    never create/restart it). It must drain all task descendants and cause any
    in-flight execute call to return. A failed stop must raise, not return a
    receipt. Identity denotes the instance, not an image or human-readable name.
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
            if self._db.execute("SELECT 1 FROM requests WHERE status='pending' LIMIT 1").fetchone():
                self._fence("recovered incomplete request")
        except BaseException:
            self._release()
            raise

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
        if len(rows) != 1 or rows[0][0] not in {"ready", "fenced", "stopped"}:
            raise TerminalFenced("invalid terminal ledger state")
        return rows[0][0]

    def events(self):
        self._check()
        return [(kind, json.loads(payload)) for kind, payload in
                self._db.execute("SELECT kind,payload FROM events ORDER BY seq")]

    def lookup(self, request: TerminalRequest):
        self._check()
        row = self._db.execute("SELECT payload,status,code,stdout,stderr FROM requests WHERE id=?",
                               (request.request_id,)).fetchone()
        if row is None:
            return None
        if row[0] != _json(asdict(request)):
            raise TerminalConflict("terminal request ID was reused with different inputs")
        if row[1] != "completed":
            raise TerminalFenced("terminal request has no confirmed completed receipt")
        return TerminalResult(row[2], row[3], row[4])

    def _fence(self, reason):
        with self._db:
            self._db.execute("UPDATE state SET phase='fenced'")
            self._event("fenced", {"reason": reason})

    def _record_result(self, request, result, *, completed):
        if not isinstance(result, TerminalResult):
            raise TypeError("backend returned no TerminalResult")
        with self._db:
            self._db.execute("UPDATE requests SET status=?,code=?,stdout=?,stderr=? WHERE id=?",
                             ("completed" if completed else "uncertain", result.return_code,
                              result.stdout, result.stderr, request.request_id))
            self._event("completed" if completed else "late_result", {"request_id": request.request_id})

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
            except asyncio.CancelledError as cancelled:
                self._closing = True
                if not self._interrupt.done():
                    self._interrupt.set_result(None)
                try:
                    await _settle(self._operation)
                    # Cancellation can race a fully persisted receipt. The
                    # effect stays completed, but the caller still ended the trial.
                    if self.phase != "stopped":
                        self._fence("terminal caller cancelled after completion")
                        self._operation = asyncio.create_task(self._stop_owned())
                        await _settle(self._operation)
                except asyncio.CancelledError:
                    raise cancelled from None
                except BaseException as failure:
                    raise BaseExceptionGroup("terminal cancellation and settlement failed", [cancelled, failure]) from None
                raise
            return self._operation.result()

    async def _stop_owned(self):
        self._identity()
        receipt = await _settle(asyncio.create_task(asyncio.to_thread(_call, self.backend.stop_and_confirm)))
        if receipt != self.binding.environment_id:
            raise TerminalConflict("termination receipt identifies another environment")
        with self._db:
            self._db.execute("UPDATE state SET phase='stopped'")
            self._db.execute("UPDATE requests SET status='uncertain' WHERE status='pending'")
            self._event("stopped", {"environment_id": receipt})

    def _execute_backend(self, request):
        # The thread pool can queue work after the event-loop admission check.
        if self.binding.deadline_unix <= time.time():
            raise TimeoutError("terminal deadline expired before transport invocation")
        self._identity()
        return self.backend.execute(request)

    async def _execute_owned(self, request, interrupt):
        effect = None
        try:
            if interrupt.done():
                raise asyncio.CancelledError()
            timeout = min(request.timeout_seconds, self.binding.deadline_unix - time.time())
            if timeout <= 0:
                raise TimeoutError("terminal deadline expired before execution")
            effect = asyncio.create_task(asyncio.to_thread(_call, self._execute_backend, request))
            done, _ = await asyncio.wait((effect, interrupt), timeout=timeout)
            if interrupt in done:
                raise asyncio.CancelledError()
            if effect not in done:
                raise TimeoutError("terminal execution exceeded its admitted time")
            result = effect.result()
            self._identity()
            self._record_result(request, result, completed=True)
            return result
        except BaseException as original:
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
        except asyncio.CancelledError as cancelled:
            try:
                await _settle(self._abort_task)
            except BaseException as failure:
                raise BaseExceptionGroup("terminal abort cancellation and drainage failed", [cancelled, failure]) from None
            raise
        return self._abort_task.result()

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
