"""Durable ownership of a bounded, unprivileged Linux system service.

This controller lives OUTSIDE the central driver and its SDK descendants.
The system manager enforces the active-runtime limit plus termination grace
even if this controller disappears. A caller's absolute trial deadline still
needs durable admission checks in the runtime entrypoint and terminal broker.
Stopping a scope only proves process drainage. Callers must separately settle
the goal journal, retained work and spending before grading or admitting work.

Commands are trusted runtime entrypoints, never task-supplied host commands.
Task commands still belong in their separate benchmark sandbox. State lives
in a private controller-owned directory, outside the service's write access.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import math
import os
import pwd
import re
import stat
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from taste.brains.process_credentials import (
    MAX_CREDENTIALS,
    ScopeCredential,
    credential_values,
    load_credential_properties,
    write_credentials,
)
from taste.resources import resource_error


class _WaitCancelled(Exception):
    """Wake only the outside polling thread; never imply service drainage."""


def _owned_call(call):
    try:
        return call()
    except (KeyboardInterrupt, SystemExit) as exc:
        # These exceptions must not escape a Task directly and halt the loop
        # before its owner can wait for cleanup.
        raise BaseExceptionGroup("scope control operation interrupted", [exc]) from None


async def _settled_task(task):
    while not task.done():
        try:
            # wait() does not propagate caller cancellation into its Tasks.
            # Unlike a cancelled shield on Python 3.14, it also leaves their
            # eventual exceptions for this owner to retrieve exactly once.
            await asyncio.wait((task,))
        except asyncio.CancelledError:
            continue
        except BaseException:
            break
    return task.result()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(value) or not 0.001 <= value <= 604800:
        raise ValueError(f"{name} must be between 0.001 and 604800 seconds")
    return float(value)


def _duration_us(value: str) -> int:
    factors = {"us": 1, "ms": 1000, "s": 1000000, "min": 60000000,
               "h": 3600000000, "d": 86400000000, "w": 604800000000}
    parts = re.findall(r"([0-9]+(?:\.[0-9]+)?)(us|ms|min|s|h|d|w)", value)
    if not parts or "".join(number + unit for number, unit in parts) != value.replace(" ", ""):
        raise ValueError("service duration is not finite or recognized")
    total = sum(Decimal(number) * factors[unit] for number, unit in parts)
    if total != int(total):
        raise ValueError("service duration has sub-microsecond precision")
    return int(total)


@dataclass(frozen=True)
class ScopeSpec:
    argv: tuple[str, ...]
    cwd: str
    uid: int
    runtime_seconds: float
    grace_seconds: float = 2.0
    python_path: str | None = None
    credentials: tuple[ScopeCredential, ...] = ()

    def __post_init__(self):
        if not isinstance(self.argv, tuple) or not self.argv:
            raise ValueError("argv must be a nonempty tuple")
        paths = [self.cwd, self.argv[0]]
        if self.python_path is not None:
            paths.append(self.python_path)
        if any(not isinstance(p, str) or not Path(p).is_absolute() for p in paths):
            raise ValueError("executable, cwd and python_path must be absolute paths")
        # systemd specifiers are a separate expansion language from shell
        # quoting. Reject them in this deliberately narrow runtime API.
        if any(not isinstance(s, str) or not s or "\x00" in s or "%" in s
               for s in [*self.argv, *paths]):
            raise ValueError("scope arguments must be nonempty text without NUL or % specifiers")
        if isinstance(self.uid, bool) or not isinstance(self.uid, int) or self.uid <= 0:
            raise ValueError("scope uid must be a non-root numeric uid")
        object.__setattr__(self, "runtime_seconds", _positive(self.runtime_seconds, "runtime_seconds"))
        object.__setattr__(self, "grace_seconds", _positive(self.grace_seconds, "grace_seconds"))
        if (not isinstance(self.credentials, tuple) or len(self.credentials) > MAX_CREDENTIALS
                or not all(isinstance(item, ScopeCredential) for item in self.credentials)
                or len({item.name for item in self.credentials}) != len(self.credentials)):
            raise ValueError("scope requires at most eight uniquely named credential descriptors")
        if len(_canonical(self.to_dict()).encode()) > 65536:
            raise ValueError("scope configuration exceeds 64 KiB")

    def to_dict(self):
        value = asdict(self)
        # Preserve every existing credential-free scope's digest and wire form.
        if not self.credentials:
            del value["credentials"]
        return value

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical(self.to_dict()).encode()).hexdigest()


class SystemdManager:
    """Finite control commands; only systemd owns the execution deadline."""

    def _command(self, argv, *, timeout=15, check=True):
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        if check and result.returncode:
            # CalledProcessError.__str__ omits stderr, losing the manager's
            # reason for rejecting a launch. Retain bounded diagnostic text.
            raise RuntimeError(f"{argv[0]} failed ({result.returncode}): {result.stderr[:2048]}")
        return result

    def start(self, unit: str, description: str, spec: ScopeSpec, *, credential_directory=None) -> None:
        if os.geteuid() != 0 or not Path("/sys/fs/cgroup/cgroup.controllers").is_file():
            raise RuntimeError("system scope controller requires root and Linux cgroup v2")
        account = pwd.getpwuid(spec.uid)
        if set(os.getgrouplist(account.pw_name, account.pw_gid)) != {account.pw_gid}:
            raise ValueError("service user must have no supplementary groups")
        argv = [
            "systemd-run", "--quiet", f"--unit={unit}", f"--description={description}",
            "--service-type=exec", "--expand-environment=no",
            f"--uid={spec.uid}", f"--gid={account.pw_gid}", f"--working-directory={spec.cwd}",
            "--property=Slice=system.slice", "--property=KillMode=control-group",
            f"--property=RuntimeMaxSec={math.ceil(spec.runtime_seconds * 1e6)}us",
            f"--property=TimeoutStopSec={math.ceil(spec.grace_seconds * 1e6)}us",
            "--property=TimeoutStartSec=15s",
            "--property=SendSIGKILL=yes", "--property=FinalKillSignal=SIGKILL",
            "--property=Restart=no", "--property=Delegate=no",
            "--property=NoNewPrivileges=yes", "--property=CapabilityBoundingSet=",
            "--property=ProtectControlGroups=yes", "--property=TasksMax=256",
            "--property=MemoryMax=2G", "--property=CPUQuota=100%",
            "--property=InaccessiblePaths=-/run/user -/run/dbus -/run/systemd/private -/run/docker.sock",
            "--property=StandardOutput=journal", "--property=StandardError=journal",
        ]
        if spec.python_path is not None:
            argv.append(f"--setenv=PYTHONPATH={spec.python_path}")
        if spec.credentials:
            if credential_directory is None:
                raise ValueError("scope credential directory is missing")
            argv.extend(load_credential_properties(credential_directory, spec.credentials))
        elif credential_directory is not None:
            raise ValueError("credential-free scope cannot load private files")
        self._command([*argv, "--", *spec.argv])

    def inspect(self, unit: str) -> dict[str, str] | None:
        result = self._command([
            "systemctl", "show", unit, "--no-pager",
            "--property=LoadState,ActiveState,SubState,Description,ControlGroup,Result,"
            "InvocationID,KillMode,SendSIGKILL,User,Restart,RuntimeMaxUSec,TimeoutStopUSec,"
            "Delegate,NoNewPrivileges,ProtectControlGroups",
        ], check=False)
        values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        if values.get("LoadState") == "not-found":
            return None
        if result.returncode or values.get("LoadState") != "loaded":
            raise RuntimeError(f"cannot inspect service: {result.returncode}: {result.stderr[:1024]}")
        return values

    def stop(self, unit: str, grace_seconds: float) -> None:
        self._command(["systemctl", "stop", unit], timeout=grace_seconds + 15)

    def empty(self, unit: str) -> bool:
        path = Path("/sys/fs/cgroup/system.slice") / unit / "cgroup.events"
        try:
            values = dict(line.split() for line in path.read_text().splitlines())
        except FileNotFoundError:
            return True
        if "populated" not in values:
            raise RuntimeError("cgroup occupancy evidence is missing")
        return values["populated"] == "0"

    def release(self, unit: str) -> None:
        # Execution is already confirmed stopped. Resetting the failed state
        # allows systemd to collect this transient unit; it never starts work.
        self._command(["systemctl", "reset-failed", unit], check=False)
        deadline = time.monotonic() + 15
        while self.inspect(unit) is not None:
            if time.monotonic() >= deadline:
                raise RuntimeError("stopped transient service was not released")
            time.sleep(0.05)


class OwnedProcessScope:
    """One durable launch; reopening never authorizes a second execution.

    A lost start acknowledgement with no observable service remains ambiguous,
    even if one inspection sees nothing. A delayed manager request could still
    create it. Such state is deliberately fenced rather than declared drained.
    """

    _fields = frozenset({"schema", "owner", "spec", "spec_digest", "phase", "launch_attempted", "launch_acknowledged",
                        "invocation_id", "stop_reason", "termination", "last_error", "last_observation"})

    def __init__(self, directory: Path | str, *, manager: Any = None):
        self.directory = Path(directory).absolute()
        self.manager = manager if manager is not None else SystemdManager()
        with self._locked() as fd:
            state = self._read(fd)
        self.spec = self._spec(state)
        self.unit = f"taste-goal-{state['owner']}.service"
        self.description = f"taste-goal:{state['owner']}:{self.spec.digest}"

    @classmethod
    def create(cls, directory: Path | str, spec: ScopeSpec, *, manager: Any = None,
               credentials=None):
        if not isinstance(spec, ScopeSpec):
            raise TypeError("spec must be a ScopeSpec")
        values = credential_values(spec.credentials, credentials)
        path = Path(directory).absolute()
        path.mkdir(mode=0o700)  # Existing state is never overwritten or reused.
        owner = uuid.uuid4().hex
        state = {"schema": "taste.brains/ProcessScope/1", "owner": owner,
                 "spec": spec.to_dict(), "spec_digest": spec.digest, "phase": "ready",
                 "launch_attempted": False, "launch_acknowledged": False,
                 "invocation_id": None, "stop_reason": None,
                 "termination": None, "last_error": None, "last_observation": None}
        # Keep the same directory lock protocol for the initial durable intent.
        temporary = cls.__new__(cls)
        temporary.directory = path
        with temporary._locked() as fd:
            write_credentials(fd, values)
            temporary._write(fd, state)
        return cls(path, manager=manager)

    @contextmanager
    def _locked(self):
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            mode = os.fstat(fd)
            if mode.st_uid != os.geteuid() or stat.S_IMODE(mode.st_mode) & 0o077:
                raise ValueError("scope state must be private and owned by this controller uid")
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield fd
        finally:
            os.close(fd)

    @staticmethod
    def _spec(state):
        if not isinstance(state["spec"], dict) or not isinstance(state["spec"].get("argv"), list):
            raise ValueError("malformed scope specification")
        raw = dict(state["spec"])
        raw["argv"] = tuple(raw["argv"])
        if "credentials" in raw:
            if not isinstance(raw["credentials"], list) or not raw["credentials"]:
                raise ValueError("malformed scope credentials")
            raw["credentials"] = tuple(ScopeCredential(**item) for item in raw["credentials"])
        spec = ScopeSpec(**raw)
        if spec.digest != state["spec_digest"]:
            raise ValueError("scope configuration digest differs from durable intent")
        return spec

    def _read(self, fd):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate scope state key")
                result[key] = value
            return result

        source = os.open("state.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        with os.fdopen(source) as stream:
            raw = stream.read(131073)
        if len(raw) > 131072:
            raise ValueError("scope state exceeds 128 KiB")
        state = json.loads(raw, object_pairs_hook=unique)
        if (not isinstance(state, dict) or set(state) != self._fields
                or state["schema"] != "taste.brains/ProcessScope/1"
                or not isinstance(state["owner"], str)
                or re.fullmatch(r"[0-9a-f]{32}", state["owner"]) is None
                or not isinstance(state["phase"], str)
                or state["phase"] not in {"ready", "launch_pending", "running", "stop_pending", "stopped"}
                or type(state["launch_acknowledged"]) is not bool
                or type(state["launch_attempted"]) is not bool
                or (state["launch_acknowledged"] and not state["launch_attempted"])
                or (state["phase"] == "ready" and state["launch_attempted"])
                or (state["phase"] == "launch_pending" and not state["launch_attempted"])
                or (state["phase"] == "running" and not state["launch_acknowledged"])):
            raise ValueError("malformed scope state")
        if (state["invocation_id"] is not None and (
                not isinstance(state["invocation_id"], str)
                or re.fullmatch(r"[0-9a-f]{32}", state["invocation_id"]) is None)):
            raise ValueError("malformed durable service invocation identity")
        spec = self._spec(state)
        if hasattr(self, "unit") and (self.unit != f"taste-goal-{state['owner']}.service"
                                     or self.spec != spec):
            raise ValueError("scope identity changed under this controller")
        if state["phase"] == "stopped":
            receipt = state["termination"]
            if (not isinstance(receipt, dict)
                    or receipt.get("unit") != f"taste-goal-{state['owner']}.service"
                    or receipt.get("spec_digest") != spec.digest
                    or receipt.get("processes_stopped") is not True
                    or receipt.get("reason") != state["stop_reason"]
                    or receipt.get("invocation_id") != state["invocation_id"]
                    or receipt.get("goal_settlement_required") is not state["launch_attempted"]):
                raise ValueError("stopped scope has invalid termination evidence")
        return state

    @staticmethod
    def _write(fd, state):
        name = f".state-{uuid.uuid4().hex}"
        target = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=fd)
        with os.fdopen(target, "w") as stream:
            stream.write(_canonical(state) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(name, "state.json", src_dir_fd=fd, dst_dir_fd=fd)
        os.fsync(fd)

    def _verify(self, state, observed):
        if observed is None:
            return
        expected = {"Description": self.description, "User": str(self.spec.uid),
                    "KillMode": "control-group", "SendSIGKILL": "yes", "Restart": "no",
                    "Delegate": "no", "NoNewPrivileges": "yes", "ProtectControlGroups": "yes"}
        if any(observed.get(key) != value for key, value in expected.items()):
            raise RuntimeError("service ownership or execution configuration differs")
        if (_duration_us(observed.get("RuntimeMaxUSec", "")) != math.ceil(self.spec.runtime_seconds * 1e6)
                or _duration_us(observed.get("TimeoutStopUSec", "")) != math.ceil(self.spec.grace_seconds * 1e6)):
            raise RuntimeError("service deadline configuration differs")
        invocation = observed.get("InvocationID")
        if not invocation or (state["invocation_id"] is not None
                              and state["invocation_id"] != invocation):
            raise RuntimeError("service invocation identity differs")
        group = observed.get("ControlGroup")
        if group not in {"", f"/system.slice/{self.unit}"}:
            raise RuntimeError("service cgroup differs")
        state["invocation_id"] = invocation
        state["last_observation"] = observed

    def _failure(self, fd, state, operation, exc):
        error = resource_error("systemd_scope", self.unit, operation, exc)
        state["last_error"] = f"{type(exc).__name__}: {exc}"[:2048]
        try:
            self._write(fd, state)
        except BaseException as persistence_error:
            raise BaseExceptionGroup("scope failure and its evidence could not be persisted",
                                     [error, persistence_error]) from exc
        return error

    def start(self) -> None:
        with self._locked() as fd:
            state = self._read(fd)
            if state["phase"] != "ready":
                raise RuntimeError("scope was already admitted; recover it without another launch")
            try:
                if self.manager.inspect(self.unit) is not None or not self.manager.empty(self.unit):
                    raise RuntimeError("scope name or cgroup already exists")
                state.update(phase="launch_pending", launch_attempted=True)
                self._write(fd, state)
                options = {"credential_directory": self.directory / "credentials"} if self.spec.credentials else {}
                self.manager.start(self.unit, self.description, self.spec, **options)
                state["launch_acknowledged"] = True
                state["phase"] = "running"
                self._write(fd, state)
                self._verify(state, self.manager.inspect(self.unit))
                self._write(fd, state)
            except BaseException as exc:
                raise self._failure(fd, state, "start", exc) from exc

    def stop(self, reason: str = "external cancellation") -> dict[str, Any]:
        if not isinstance(reason, str) or not reason or "\x00" in reason or len(reason) > 2048:
            raise ValueError("stop reason must contain 1..2048 characters without NUL")
        with self._locked() as fd:
            state = self._read(fd)
            if state["phase"] == "stopped":
                return state["termination"]
            never_launched = not state["launch_attempted"]
            state["stop_reason"] = state["stop_reason"] or reason
            state["phase"] = "stop_pending"
            try:
                self._write(fd, state)  # Intent must precede the first signal.
                observed = None if never_launched else self.manager.inspect(self.unit)
                self._verify(state, observed)
                self._write(fd, state)  # Bind an observed incarnation before acting.
                if (not never_launched and observed is None and not state["launch_acknowledged"]
                        and state["invocation_id"] is None):
                    raise RuntimeError("launch acknowledgement is missing; an absent service is ambiguous")
                if observed is not None:
                    self.manager.stop(self.unit, self.spec.grace_seconds)
                    observed = self.manager.inspect(self.unit)
                    self._verify(state, observed)
                    self._write(fd, state)
                    if observed is not None and observed.get("ActiveState") not in {"inactive", "failed"}:
                        raise RuntimeError("service did not become inactive")
                if not never_launched and not self.manager.empty(self.unit):
                    raise RuntimeError("service cgroup still has live descendants")
                if observed is not None:
                    self.manager.release(self.unit)
                    if self.manager.inspect(self.unit) is not None or not self.manager.empty(self.unit):
                        raise RuntimeError("service release could not be confirmed")
                termination = {"unit": self.unit, "spec_digest": self.spec.digest,
                               "invocation_id": state["invocation_id"],
                               "reason": state["stop_reason"], "processes_stopped": True,
                               "goal_settlement_required": not never_launched,
                               "manager_result": (state["last_observation"] or {}).get("Result")}
                state.update(phase="stopped", termination=termination, last_error=None)
                self._write(fd, state)
                return termination
            except BaseException as exc:
                raise self._failure(fd, state, "stop", exc) from exc

    def wait(self, *, timeout_seconds: float, _cancel_wait: threading.Event | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + _positive(timeout_seconds, "timeout_seconds")
        while True:
            if _cancel_wait is not None and _cancel_wait.is_set():
                raise _WaitCancelled()
            with self._locked() as fd:
                state = self._read(fd)
                if state["phase"] == "ready":
                    raise RuntimeError("scope has not been admitted")
                if state["phase"] == "stopped":
                    return state["termination"]
                try:
                    observed = self.manager.inspect(self.unit)
                    previous_invocation = state["invocation_id"]
                    self._verify(state, observed)
                    if state["invocation_id"] != previous_invocation:
                        self._write(fd, state)
                except BaseException as exc:
                    raise self._failure(fd, state, "wait", exc) from exc
                ended = (state["phase"] == "stop_pending" or observed is None
                         or observed.get("ActiveState") in {"inactive", "failed"})
            if ended:
                return self.stop("service execution ended")
            if time.monotonic() >= deadline:
                return self.stop("outside controller wait deadline expired")
            time.sleep(0.05)

    async def run_async(self, *, timeout_seconds: float) -> dict[str, Any]:
        """Own launch, polling and drainage across repeated caller cancellation.

        An in-flight manager request is allowed to settle before the caller
        returns. Cancellation wakes the polling thread even if service cleanup
        fails, so no background controller is abandoned. Such failure retains
        the durable scope fence and raises; it never produces a drain receipt.
        The returned receipt still requires separate goal settlement.
        """
        _positive(timeout_seconds, "timeout_seconds")
        cancelled = threading.Event()

        def drive():
            if cancelled.is_set():
                raise _WaitCancelled()
            self.start()
            return self.wait(timeout_seconds=timeout_seconds, _cancel_wait=cancelled)

        driver = asyncio.create_task(asyncio.to_thread(_owned_call, drive))
        try:
            await asyncio.wait((driver,))
            return driver.result()
        except BaseException as original:
            cancelled.set()
            reason = ("asynchronous caller cancelled" if isinstance(original, asyncio.CancelledError)
                      else "asynchronous scope operation failed")
            cleanup = asyncio.create_task(asyncio.to_thread(_owned_call, lambda: self.stop(reason)))
            failures = []
            for task in (driver, cleanup):
                try:
                    await _settled_task(task)
                except _WaitCancelled:
                    if task is not driver:
                        raise
                except BaseException as exc:
                    if exc is not original:
                        failures.append(exc)
            if failures:
                raise BaseExceptionGroup("scope execution or drainage failed", [original, *failures]) from None
            raise

    async def stop_async(self, reason: str = "external cancellation") -> dict[str, Any]:
        """Recover/drain without launch, retaining ownership through cancellation."""
        cleanup = asyncio.create_task(asyncio.to_thread(_owned_call, lambda: self.stop(reason)))
        try:
            await asyncio.wait((cleanup,))
            return cleanup.result()
        except BaseException as original:
            try:
                await _settled_task(cleanup)
            except BaseException as exc:
                if exc is not original:
                    raise BaseExceptionGroup("scope recovery failed", [original, exc]) from None
            raise
