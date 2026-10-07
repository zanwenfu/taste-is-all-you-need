"""Bounded Linux Docker transport for the trusted, outside-container broker.

Uses only an explicitly named local daemon socket, pinned Docker API v1.51,
non-TTY exec and cgroup v2. Never creates, restarts or removes a container.
Persist DockerTerminalBinding outside task write access before admitting work;
recovery must use that original binding, not admit the container again.

This is not the worker credential boundary or the outside death watchdog.
The lifecycle owner must create an isolated container with restart disabled,
reserve its ownership label or Compose project, and stop/remove it even if this controller dies.
Neither task containers nor model workers should receive the Docker socket.
"""

from __future__ import annotations

import http.client
import json
import math
import os
import re
import signal
import socket
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from taste.brains.terminal_broker import (
    MAX_TERMINAL_OUTPUT_BYTES,
    TerminalConflict,
    TerminalFenced,
    TerminalRequest,
    TerminalResult,
)

API_VERSION = "v1.51"
OWNER_LABEL = "taste.terminal.owner"
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"
CONTROL_BYTES = 1024 * 1024
READ_BYTES = 65536
CONTROL_SECONDS = 10
# Bound for ending one overdue or abandoned command and confirming its exit.
# It must stay inside the broker's own settlement allowance for that command.
END_SECONDS = 10
CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC_ROOT = Path("/proc")


class DockerTransportError(RuntimeError):
    """An unconfirmed transport outcome, never an ordinary command exit."""


class _ContainerMissing(DockerTransportError):
    """The local daemon explicitly reported the exact container ID absent."""


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Docker transport deadline expired")
    return remaining


def _full_id(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class _DeadlineSocket(socket.socket):
    # SocketIO calls recv_into for each underlying read. Applying the remaining
    # *absolute* allowance here also bounds slowly trickled HTTP headers/chunks.
    def __init__(self, deadline):
        timeout = _remaining(deadline)
        super().__init__(socket.AF_UNIX, socket.SOCK_STREAM)
        self.deadline = deadline
        self.settimeout(timeout)

    def recv_into(self, buffer, nbytes=0, flags=0):
        self.settimeout(_remaining(self.deadline))
        return super().recv_into(buffer, nbytes, flags)

    def sendall(self, data, flags=0):
        self.settimeout(_remaining(self.deadline))
        return super().sendall(data, flags)


class _Wire:
    def __init__(self, socket_path):
        self.socket_path = socket_path
        self._lock = threading.Lock()
        self._sockets = set()
        self._closed = False

    def cancel(self):
        with self._lock:
            self._closed = True
            for sock in self._sockets:
                with suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)

    @contextmanager
    def request(self, method, path, deadline, body=None, *, upgrade=False, raw=None):
        connection = http.client.HTTPConnection("localhost")
        sock = _DeadlineSocket(deadline)
        response = None
        try:
            with self._lock:
                if self._closed:
                    raise TerminalFenced("Docker transport admission ended")
                self._sockets.add(sock)
            sock.connect(self.socket_path)
            with self._lock:
                if self._closed:
                    raise TerminalFenced("Docker transport admission ended")
            connection.sock = sock
            headers = {"Content-Type": "application/json" if raw is None else "application/x-tar"}
            if upgrade:
                headers.update(Connection="Upgrade", Upgrade="tcp")
            else:
                headers["Connection"] = "close"
            # A tar sent to the archive endpoint goes as it is, from bytes or a file.
            payload = raw if raw is not None else (None if body is None else json.dumps(body, allow_nan=False).encode())
            if raw is not None and not isinstance(raw, bytes | bytearray):
                headers["Content-Length"] = str(os.fstat(raw.fileno()).st_size)
            connection.request(method, f"/{API_VERSION}{path}", body=payload, headers=headers)
            response = connection.getresponse()
            _remaining(deadline)
            yield response
        finally:
            if response is not None:
                response.close()
            connection.close()
            with self._lock:
                self._sockets.discard(sock)
            sock.close()

    def control(self, method, path, deadline, body=None, *, statuses=(200,)):
        with self.request(method, path, deadline, body) as response:
            if response.status not in statuses:
                # Daemon error bodies can contain task data; don't echo them.
                if (response.status == 404 and method == "GET"
                        and re.fullmatch(r"/containers/[0-9a-f]{64}/json", path)):
                    raise _ContainerMissing("Docker reports the original container absent")
                raise DockerTransportError(f"Docker control request returned HTTP {response.status}")
            data = bytearray()
            while True:
                _remaining(deadline)
                chunk = response.read1(min(READ_BYTES, CONTROL_BYTES + 1 - len(data)))
                if not chunk:
                    if response.length not in (None, 0):
                        raise DockerTransportError("truncated Docker control reply")
                    break
                data.extend(chunk)
                if len(data) > CONTROL_BYTES:
                    raise DockerTransportError("Docker control reply exceeded its byte limit")
            if not data:
                return None
            try:
                result = json.loads(data)
            except (ValueError, RecursionError) as exc:
                raise DockerTransportError("invalid Docker control JSON") from exc
            if not isinstance(result, dict):
                raise DockerTransportError("Docker control reply must be an object")
            return result


class _Captured:
    """Stream prefixes and dropped counts, kept when the stream is cut short."""

    def __init__(self):
        self.streams = [bytearray(), bytearray()]
        self.dropped = [0, 0]


def _capture(response, deadline, limit, captured):
    if response.status == 101:
        if response.getheader("Upgrade", "").lower() != "tcp":
            raise DockerTransportError("invalid Docker stream upgrade")
        read = response.fp.read1  # HTTP framing ends at a successful upgrade.
    elif response.status == 200:
        read = response.read1  # Includes HTTP chunk decoding when present.
    else:
        raise DockerTransportError(f"Docker exec start returned HTTP {response.status}")
    if response.getheader("Content-Type", "").split(";", 1)[0].strip() not in {
        "application/vnd.docker.raw-stream", "application/vnd.docker.multiplexed-stream",
    }:
        raise DockerTransportError("Docker did not supply a framed non-TTY stream")

    streams, dropped = captured.streams, captured.dropped

    def take(count):
        _remaining(deadline)
        return read(count)

    while True:
        header = bytearray()
        while len(header) < 8:
            part = take(8 - len(header))
            if not part:
                if not header:
                    if response.status == 200 and response.length not in (None, 0):
                        raise DockerTransportError("truncated Docker HTTP stream")
                    return
                raise DockerTransportError("truncated Docker stream header")
            header.extend(part)
        if header[0] not in (1, 2) or header[1:4] != b"\0\0\0":
            raise DockerTransportError("invalid Docker stdout/stderr frame")
        index, remaining = header[0] - 1, int.from_bytes(header[4:], "big")
        while remaining:
            part = take(min(remaining, READ_BYTES))
            if not part:
                raise DockerTransportError("truncated Docker stream payload")
            remaining -= len(part)
            keep = min(len(part), limit - len(streams[index]))
            streams[index].extend(part[:keep])
            dropped[index] += len(part) - keep
            if dropped[index] > 2**63 - 1:
                raise DockerTransportError("Docker output byte count overflow")


@dataclass(frozen=True)
class DockerTerminalBinding:
    socket_path: str
    container_id: str
    owner_token: str
    started_at: str
    cgroup_path: str
    deadline_unix: float
    output_limit: int = 65536
    compose_project: str | None = None
    image_id: str | None = None
    exec_user: str | None = None

    def __post_init__(self):
        if (not isinstance(self.socket_path, str) or not self.socket_path.startswith("/")
                or "\0" in self.socket_path or len(self.socket_path.encode()) > 107):
            raise ValueError("an absolute local Docker Unix socket path is required")
        if not _full_id(self.container_id):
            raise ValueError("a full Docker container ID is required")
        if not isinstance(self.owner_token, str) or re.fullmatch(r"[0-9a-f]{32}", self.owner_token) is None:
            raise ValueError("a 32-character lowercase hex owner token is required")
        if (not isinstance(self.started_at, str) or not self.started_at
                or len(self.started_at) > 64):
            raise ValueError("the original Docker start timestamp is required")
        if (not isinstance(self.cgroup_path, str) or not self.cgroup_path.startswith("/")
                or "\0" in self.cgroup_path or len(self.cgroup_path) > 4096
                or ".." in PurePosixPath(self.cgroup_path).parts
                or self.container_id not in PurePosixPath(self.cgroup_path).name):
            raise ValueError("the original container-specific cgroup v2 path is required")
        if (type(self.deadline_unix) not in (int, float) or not math.isfinite(self.deadline_unix)
                or self.deadline_unix <= 0):
            raise ValueError("a finite positive task deadline is required")
        if type(self.output_limit) is not int or not 1 <= self.output_limit <= MAX_TERMINAL_OUTPUT_BYTES:
            raise ValueError("output limit must be between 1 byte and 1 MiB per stream")
        if self.compose_project is not None and (not isinstance(self.compose_project, str)
                or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", self.compose_project) is None
                or self.image_id is None):
            raise ValueError("native Compose admission requires a project and exact image ID")
        if self.image_id is not None and (not isinstance(self.image_id, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", self.image_id) is None):
            raise ValueError("an exact Docker image ID is required")
        if self.exec_user is not None and (not isinstance(self.exec_user, str)
                or re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_.\-$]{0,255}"
                                r"(?::[a-zA-Z0-9_][a-zA-Z0-9_.\-$]{0,255})?", self.exec_user) is None):
            raise ValueError("Docker exec user must be a fixed user or user:group")


class DockerTerminalBackend:
    def __init__(self, binding: DockerTerminalBinding):
        if not isinstance(binding, DockerTerminalBinding):
            raise TypeError("a validated DockerTerminalBinding is required")
        self.binding = binding
        self._lock, self._stop_lock = threading.Lock(), threading.Lock()
        self._closing = False
        self._active = None
        self._interrupted = False

    @property
    def environment_id(self):
        return self.binding.container_id

    @classmethod
    def admit(cls, socket_path, container_id, owner_token, deadline_unix, *, output_limit=65536,
              compose_project=None, image_id=None, exec_user=None):
        """Bind a running container without modifying its published settings.

        Native Harbor callers reserve a unique Compose project externally and
        supply its exact main-container ID and image ID. The owner token still
        scopes the broker and independent watchdog; it need not be a task label.
        Persist the returned binding before launch and reuse it for recovery.
        ``exec_user`` is the admitted Harbor agent user, never a model argument.
        """
        # Validate caller values before constructing any daemon URL. The real
        # start/cgroup binding is captured below, then checked a second time.
        provisional = DockerTerminalBinding(socket_path, container_id, owner_token, "pending",
                                            f"/{container_id}", deadline_unix, output_limit,
                                            compose_project, image_id, exec_user)
        if deadline_unix <= time.time():
            raise TerminalFenced("Docker task admission deadline expired")
        wire = _Wire(socket_path)
        deadline = time.monotonic() + min(CONTROL_SECONDS, deadline_unix - time.time())
        info = wire.control("GET", f"/containers/{container_id}/json", deadline)
        cls._check_container(info, provisional, original_start=False)
        if info["State"]["Running"] is not True:
            raise TerminalFenced("Docker admission requires an already running container")
        pid = info["State"].get("Pid")
        if type(pid) is not int or pid <= 0 or not (CGROUP_ROOT / "cgroup.controllers").is_file():
            raise TerminalFenced("Docker transport requires local Linux cgroup v2")
        paths = [line[3:] for line in (PROC_ROOT / str(pid) / "cgroup").read_text().splitlines()
                 if line.startswith("0::")]
        if len(paths) != 1:
            raise TerminalFenced("cannot identify the original container cgroup")
        backend = cls(DockerTerminalBinding(socket_path, container_id, owner_token,
                      info["State"].get("StartedAt"), paths[0], deadline_unix, output_limit,
                      compose_project, image_id, exec_user))
        confirmed = backend._inspect(wire, deadline)
        if confirmed["State"]["Running"] is not True or confirmed["State"].get("Pid") != pid:
            raise TerminalFenced("container changed during terminal admission")
        return backend

    @staticmethod
    def _check_container(info, binding, *, original_start=True):
        if (not isinstance(info, dict) or info.get("Id") != binding.container_id
                or not isinstance(info.get("Config"), dict)
                or not isinstance(info["Config"].get("Labels"), dict)):
            raise TerminalConflict("Docker container identity or ownership label changed")
        labels = info["Config"]["Labels"]
        if binding.compose_project is None:
            if labels.get(OWNER_LABEL) != binding.owner_token:
                raise TerminalConflict("Docker container identity or ownership label changed")
        elif any(labels.get(key) != expected for key, expected in {
                COMPOSE_PROJECT_LABEL: binding.compose_project,
                "com.docker.compose.service": "main", "com.docker.compose.oneoff": "False",
                "com.docker.compose.container-number": "1"}.items()):
            raise TerminalConflict("Docker Compose main-container ownership changed")
        if binding.image_id is not None and info.get("Image") != binding.image_id:
            raise TerminalConflict("Docker container image differs from admission")
        state, config = info.get("State"), info.get("HostConfig")
        if (not isinstance(state, dict) or type(state.get("Running")) is not bool
                or state.get("Paused") is not False or state.get("Restarting") is not False
                or state.get("Dead") is not False or not isinstance(config, dict)
                or config.get("AutoRemove") is not False
                or not isinstance(config.get("RestartPolicy"), dict)
                or config["RestartPolicy"].get("Name") != "no"):
            raise TerminalFenced("Docker container state or lifecycle policy is not admissible")
        if original_start and state.get("StartedAt") != binding.started_at:
            raise TerminalConflict("Docker container was restarted after admission")

    def _inspect(self, wire, deadline):
        info = wire.control("GET", f"/containers/{self.environment_id}/json", deadline)
        self._check_container(info, self.binding)
        return info

    def execute(self, request: TerminalRequest) -> TerminalResult:
        if not isinstance(request, TerminalRequest):
            raise TypeError("a validated TerminalRequest is required")
        wire = _Wire(self.binding.socket_path)
        with self._lock:
            if self._closing or self._active is not None:
                raise TerminalFenced("Docker terminal is closed or already executing")
            self._active, self._interrupted = wire, False
        try:
            allowance = min(request.timeout_seconds, self.binding.deadline_unix - time.time())
            deadline = time.monotonic() + allowance
            _remaining(deadline)
            info = self._inspect(wire, deadline)
            if info["State"]["Running"] is not True:
                raise TerminalFenced("Docker terminal never restarts a stopped container")
            command = {
                "AttachStdout": True, "AttachStderr": True, "AttachStdin": False,
                "Tty": False, "Cmd": ["/bin/sh", "-c", request.command], "WorkingDir": request.cwd,
            }
            if self.binding.exec_user is not None:
                command["User"] = self.binding.exec_user
            created = wire.control("POST", f"/containers/{self.environment_id}/exec", deadline,
                                   command, statuses=(201,))
            exec_id = created.get("Id") if isinstance(created, dict) else None
            if not _full_id(exec_id):
                raise DockerTransportError("Docker did not identify the created exec")
            captured, failure = _Captured(), None
            try:
                with wire.request("POST", f"/exec/{exec_id}/start", deadline,
                                  {"Detach": False, "Tty": False}, upgrade=True) as response:
                    _capture(response, deadline, self.binding.output_limit, captured)
            except (OSError, http.client.HTTPException, DockerTransportError, TerminalFenced) as exc:
                failure = exc  # TimeoutError is an OSError.
            with self._lock:
                stopping, interrupted = self._closing, self._interrupted
            if stopping:
                raise failure or TerminalFenced("Docker terminal is stopping its environment")
            # A departed caller or an overdue command ends that command only.
            # Any other stream failure stays an unconfirmed transport outcome.
            terminated = ("cancelled" if interrupted
                          else "timeout" if isinstance(failure, TimeoutError) else "")
            if failure is not None and not terminated:
                raise failure
            # The start connection may be cut; confirm the exit on a new one.
            control = _Wire(self.binding.socket_path)
            settle = time.monotonic() + END_SECONDS if terminated else deadline
            code = self._exec_exit(control, exec_id, settle, end=bool(terminated))
            self._inspect(control, settle)
            return TerminalResult(code, bytes(captured.streams[0]), bytes(captured.streams[1]),
                                  *captured.dropped, terminated)
        except BaseException:
            with self._lock:
                self._closing = True
            raise
        finally:
            wire.cancel()
            with self._lock:
                self._active = None

    def checkpoint(self, directory, *, cap_bytes=None):
        """Copy out what the task changed against its image, between commands.

        Holds the terminal as a command would, so no command can start while
        the files are read, and none is running when they are. See
        ``taste.brains.docker_checkpoint``.
        """
        from taste.brains.docker_checkpoint import CHECKPOINT_BYTES, take

        wire = _Wire(self.binding.socket_path)
        with self._lock:
            if self._closing or self._active is not None:
                raise TerminalFenced("a checkpoint is taken only between commands")
            self._active, self._interrupted = wire, False
        try:
            return take(self, wire, directory, cap_bytes=CHECKPOINT_BYTES if cap_bytes is None else cap_bytes)
        finally:
            wire.cancel()
            with self._lock:
                self._active = None

    def restore(self, manifest, directory, *, any_container=False):
        """Return the task's files to a checkpoint, between commands. See ``docker_checkpoint``."""
        from taste.brains.docker_checkpoint import restore

        wire = _Wire(self.binding.socket_path)
        with self._lock:
            if self._closing or self._active is not None:
                raise TerminalFenced("a checkpoint is restored only between commands")
            self._active, self._interrupted = wire, False
        try:
            return restore(self, wire, manifest, directory, any_container=any_container)
        finally:
            wire.cancel()
            with self._lock:
                self._active = None

    def interrupt(self):
        """End only the active command; its execute call returns a ``cancelled`` receipt.

        Used when the command's caller is gone. Unlike stop_and_confirm, the
        container and its other processes continue. No effect without a command.
        """
        with self._lock:
            if self._active is not None and not self._closing:
                self._interrupted = True
                self._active.cancel()

    def _exec_exit(self, wire, exec_id, deadline, *, end):
        while True:
            result = wire.control("GET", f"/exec/{exec_id}/json", deadline)
            if (not isinstance(result, dict) or result.get("ID") != exec_id
                    or result.get("ContainerID") != self.environment_id
                    or type(result.get("Running")) is not bool):
                raise DockerTransportError("Docker exec has no confirmed exit for this container")
            if result["Running"] is False:
                code = result.get("ExitCode")
                if type(code) is not int or not 0 <= code <= 255:
                    raise DockerTransportError("Docker exec has no confirmed exit for this container")
                return code
            if not end:
                raise DockerTransportError("Docker exec has no confirmed exit for this container")
            self._kill_exec(result.get("Pid"))
            time.sleep(min(0.05, _remaining(deadline)))

    def _container_pids(self):
        base = CGROUP_ROOT / self.binding.cgroup_path.lstrip("/")
        pids = set()
        for directory, _children, files in os.walk(base):
            if "cgroup.procs" in files:
                try:
                    text = (Path(directory) / "cgroup.procs").read_text()
                except OSError:
                    continue
                pids.update(int(item) for item in text.split() if item.isdigit())
        return pids

    def _kill_exec(self, pid):
        """SIGKILL one exec's process tree, never a process outside this container.

        The daemon reports the exec's host PID. Only PIDs currently in the
        container's own original cgroup are signalled. A process that detached
        from the command (a daemon it started) is not its descendant and keeps
        running, exactly as after a command that returned normally.
        """
        if type(pid) is not int or pid <= 0:
            return
        members = self._container_pids()
        parents = {}
        for member in members:
            try:
                fields = (PROC_ROOT / str(member) / "stat").read_text().rsplit(")", 1)[1].split()
                parents[member] = int(fields[1])
            except (OSError, IndexError, ValueError):
                continue
        doomed, frontier = set(), [pid]
        while frontier:
            parent = frontier.pop()
            for child, owner in parents.items():
                if owner == parent and child not in doomed:
                    doomed.add(child)
                    frontier.append(child)
        if pid in members:
            doomed.add(pid)
        for member in sorted(doomed, reverse=True):
            with suppress(ProcessLookupError, PermissionError):
                os.kill(member, signal.SIGKILL)

    def stop_and_confirm(self):
        with self._lock:
            self._closing = True
            if self._active is not None:
                self._active.cancel()
        # A retry can establish proof after a lost stop reply; it cannot execute.
        with self._stop_lock:
            deadline = time.monotonic() + CONTROL_SECONDS
            wire = _Wire(self.binding.socket_path)
            try:
                info = self._inspect(wire, deadline)
                if info["State"]["Running"]:
                    wire.control("POST", f"/containers/{self.environment_id}/stop?t=1", deadline,
                                 statuses=(204, 304))
                if self._inspect(wire, deadline)["State"]["Running"]:
                    raise TerminalFenced("Docker still reports the container running")
            except _ContainerMissing:
                # Native Harbor may remove the container before END even with
                # delete=False. An exact daemon 404 alone is not drain proof:
                # the original, durably bound cgroup must also be empty. No
                # readmission, replacement container or lost-daemon shortcut.
                if self.binding.compose_project is None:
                    raise
            self._confirm_cgroup_empty(deadline)
            return self.environment_id

    def _confirm_cgroup_empty(self, deadline):
        if not (CGROUP_ROOT / "cgroup.controllers").is_file():
            raise TerminalFenced("local cgroup v2 proof is unavailable")
        events = CGROUP_ROOT / self.binding.cgroup_path.lstrip("/") / "cgroup.events"
        while True:
            _remaining(deadline)
            try:
                state = dict(line.split() for line in events.read_text().splitlines())
            except FileNotFoundError:
                return
            if state.get("populated") == "0":
                return
            if state.get("populated") != "1":
                raise TerminalFenced("unrecognized container cgroup state")
            time.sleep(min(0.02, _remaining(deadline)))
