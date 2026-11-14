"""An unprivileged Apptainer transport for a task's container: a port for hosts with no Docker and no root.

Off by default: nothing in a study imports this module unless a run asks for it (``scripts/apptainer_replay.py``).
It stands where ``docker_terminal`` stands for the task's commands, and says where it cannot do the same:

- The image is the task's Docker image converted to a SIF, pulled by its manifest digest; the image's config digest
  is the Docker image ID a recorded trial names, so the two can be compared before anything runs.
- The container is one Apptainer instance in a root-mapped user namespace (``--fakeroot --ignore-fakeroot-command``):
  commands run as uid 0, as in the task's Docker container, with no LD_PRELOAD wrapper. Only uid 0 is mapped, so a
  chown to another user fails (EINVAL) where Docker's root succeeds, and files the image gives another owner appear
  as root's.
- The files are the image under a writable overlay directory on node-local disk. Its ``upper`` directory is what the
  task changed, which is what Docker's own overlay2 driver reads for its change list (``changes``).
- Each command is ``apptainer exec instance://<name>`` running ``/bin/sh -c <command>`` in the requested directory,
  with exactly the environment Docker gives an exec (the image's ``Env``, then ``HOSTNAME`` and ``HOME``) and empty
  standard input. Each stream keeps its first ``output_limit`` bytes and counts the rest, as the Docker backend does.
- A command past its time is ended by killing its process tree (the exec's host processes, all this job's own),
  and reports exit code 137 as Docker does. A process a command left running in the background lives on in the
  instance, as in Docker.
- The network is ``none`` (loopback only: Harbor's agent-phase rule for these tasks) or ``host`` (the verifier
  phase's public network). An instance's network is fixed when it starts, so a phase change restarts the instance
  on the same overlay (``restart``): the files carry over; processes the agent left running do not.
"""

from __future__ import annotations

import contextlib
import os
import posixpath
import signal
import stat
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from taste.brains.docker_checkpoint import NOT_THE_TASKS, _inside, _under_added
from taste.brains.terminal_broker import (
    MAX_TERMINAL_OUTPUT_BYTES,
    TerminalFenced,
    TerminalRequest,
    TerminalResult,
)

NETWORKS = ("none", "host")
KILLED = 128 + signal.SIGKILL  # Docker's exit code for an exec it ended with SIGKILL
READ_BYTES = 65536
CONTROL_SECONDS = 120
# After a command's own process ends, how long its streams may stay open (a background process holding them).
DRAIN_SECONDS = 5.0
# fuse-overlayfs and the kernel's overlayfs mark deletions and replaced directories in these ways.
WHITEOUT_PREFIX = ".wh."
OPAQUE_MARK = ".wh..wh..opq"
OPAQUE_XATTRS = ("user.fuseoverlayfs.opaque", "trusted.overlay.opaque", "user.overlay.opaque")
CHANGED, ADDED, DELETED = 0, 1, 2   # Docker's change kinds
LOWER_BATCH = 20000


@dataclass(frozen=True)
class ApptainerSpec:
    """One task container: the image, the overlay holding its changes, the instance and how each command runs."""

    image: str
    overlay: str
    name: str
    workdir: str = "/app"
    network: str = "none"
    hostname: str = "taskbox"
    env: tuple[tuple[str, str], ...] = ()
    binds: tuple[tuple[str, str], ...] = ()
    exec_user: str | None = None
    apptainer: str = "apptainer"

    def __post_init__(self):
        if self.network not in NETWORKS:
            raise ValueError("network must be none or host")
        if self.exec_user not in (None, "root", "0"):
            # One uid is mapped in an unprivileged root-mapped namespace: no other user can be entered.
            raise ValueError("only the task's root user can run commands here (one uid is mapped)")
        if not self.name or any(not (ch.isalnum() or ch in "-_.") for ch in self.name):
            raise ValueError("instance name: letters, digits, - _ and . only")
        for host, inside in self.binds:
            if not (posixpath.isabs(inside) and os.path.isabs(host)):
                raise ValueError("bind paths must be absolute")
        for key, _ in self.env:
            if not key or "=" in key or "\0" in key:
                raise ValueError("an environment name is malformed")


def namespace_args():
    """The user namespace every command runs in: root-mapped, without the host's fakeroot wrapper."""
    return ["--fakeroot", "--ignore-fakeroot-command"]


def start_argv(spec):
    """``apptainer instance start`` for the task's container: none of the host's home, temp or working folder
    mounted, a clean environment, its own IPC and host name, the overlay, the binds, and the network."""
    argv = [spec.apptainer, "instance", "start", *namespace_args(), "--no-mount", "home,tmp,cwd", "--no-home",
            "--cleanenv", "--ipc", "--uts", "--hostname", spec.hostname, "--overlay", spec.overlay]
    if spec.network == "none":
        argv += ["--net", "--network", "none"]
    for host, inside in spec.binds:
        argv += ["--bind", f"{host}:{inside}"]
    return [*argv, spec.image, spec.name]


SHELL = ("/bin/sh", "-c")  # the terminal's shell (docker_terminal); Harbor's own commands use ("bash", "-c")


def exec_argv(spec, cwd, command, shell=SHELL):
    """One command in the instance: ``/bin/sh -c`` in ``cwd`` with exactly the spec's environment."""
    environment = [f"{key}={value}" for key, value in spec.env]
    return [spec.apptainer, "exec", "--cleanenv", "--pwd", cwd, f"instance://{spec.name}",
            "/usr/bin/env", "-i", *environment, *shell, command]


def stop_argv(spec):
    return [spec.apptainer, "instance", "stop", spec.name]


def lower_argv(spec):
    """A shell in the image alone (no overlay, no binds), reading paths and printing each one's type."""
    script = ('while IFS= read -r p; do if [ -L "$p" ]; then echo l; elif [ -d "$p" ]; then echo d; '
              'elif [ -e "$p" ]; then echo f; else echo -; fi; done')
    return [spec.apptainer, "exec", *namespace_args(), "--no-mount", "home,tmp,cwd", "--no-home", "--cleanenv",
            "--pwd", "/", spec.image, "/bin/sh", "-c", script]


def docker_env(image_env, hostname, home="/root"):
    """The environment Docker gives an exec as root: the image's ``Env``, ``HOSTNAME``, and ``HOME`` if unset."""
    pairs, names = [], set()
    for item in image_env or ():
        key, _, value = str(item).partition("=")
        if key and key not in names:
            pairs.append((key, value))
            names.add(key)
    if "PATH" not in names:
        pairs.append(("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"))
    pairs.append(("HOSTNAME", hostname))
    if "HOME" not in names:
        pairs.append(("HOME", home))
    return tuple(pairs)


def _descendants(pid):
    """Every process descended from ``pid`` (read from /proc), deepest last."""
    found, queue = [], [pid]
    while queue:
        current = queue.pop(0)
        try:
            tasks = os.listdir(f"/proc/{current}/task")
        except OSError:
            continue
        for task in tasks:
            try:
                children = Path(f"/proc/{current}/task/{task}/children").read_text().split()
            except OSError:
                continue
            for child in children:
                if child.isdigit() and int(child) not in found:
                    found.append(int(child))
                    queue.append(int(child))
    return found


def _kill_tree(process):
    """SIGKILL a command's own process, everything still descended from it, and its own process group (the command
    starts in a session of its own, so the group holds only this command's processes)."""
    for pid in [process.pid, *_descendants(process.pid)]:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)


class _Reader(threading.Thread):
    """One stream's first ``limit`` bytes, and how many more were drained."""

    def __init__(self, stream, limit):
        super().__init__(daemon=True)
        self.stream, self.limit = stream, limit
        self.data, self.dropped = bytearray(), 0

    def run(self):
        try:
            while True:
                chunk = self.stream.read1(READ_BYTES) if hasattr(self.stream, "read1") else self.stream.read(READ_BYTES)
                if not chunk:
                    return
                keep = max(0, min(len(chunk), self.limit - len(self.data)))
                self.data.extend(chunk[:keep])
                self.dropped += len(chunk) - keep
        except (OSError, ValueError):
            return


@dataclass
class _Run:
    code: int
    stdout: bytes
    stderr: bytes
    dropped: tuple[int, int]
    terminated: str
    seconds: float = 0.0


@dataclass
class ApptainerTerminalBackend:
    """The task container's terminal: one instance, one command at a time."""

    spec: ApptainerSpec
    deadline_unix: float = float("inf")
    output_limit: int = MAX_TERMINAL_OUTPUT_BYTES
    started: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _active: subprocess.Popen | None = field(default=None, repr=False)

    @property
    def environment_id(self):
        return self.spec.name

    # -- lifecycle ------------------------------------------------------------

    def _control(self, argv, seconds=CONTROL_SECONDS):
        done = subprocess.run(argv, capture_output=True, timeout=seconds, stdin=subprocess.DEVNULL)
        if done.returncode != 0:
            message = done.stderr.decode("utf-8", "replace").strip()[-800:]
            raise RuntimeError(f"apptainer {argv[1]} failed ({done.returncode}): {message}")
        return done

    def start(self):
        """Start the instance; its overlay directory is made if it is not there."""
        if self.started:
            raise TerminalFenced("the instance is already started")
        Path(self.spec.overlay).mkdir(parents=True, exist_ok=True)
        self._control(start_argv(self.spec))
        self.started = True

    def stop(self):
        """Stop this trial's instance, by its exact name. Its overlay stays."""
        if not self.started:
            return
        with self._lock:
            self.started = False
        self._control(stop_argv(self.spec))

    def restart(self, network):
        """The same container files under another network (a phase change): stop, then start on the same overlay."""
        self.stop()
        self.spec = replace(self.spec, network=network)
        self.start()

    # -- commands -------------------------------------------------------------

    def run(self, command, cwd, timeout_seconds, *, stdin=None, shell=SHELL):
        """One command, its streams capped; ended at its time as Docker's terminal ends it."""
        if not self.started:
            raise TerminalFenced("the instance is not started")
        allowance = min(float(timeout_seconds), self.deadline_unix - time.time())
        if allowance <= 0:
            raise TimeoutError("the terminal's deadline has passed")
        with self._lock:
            if self._active is not None:
                raise TerminalFenced("the terminal is already executing")
            began = time.monotonic()
            process = subprocess.Popen(exec_argv(self.spec, cwd, command, shell),
                                       stdin=subprocess.DEVNULL if stdin is None else subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            self._active = process
        readers = [_Reader(process.stdout, self.output_limit), _Reader(process.stderr, self.output_limit)]
        try:
            for reader in readers:
                reader.start()
            if stdin is not None:
                feeder = threading.Thread(target=_feed, args=(process.stdin, stdin), daemon=True)
                feeder.start()
            terminated = ""
            try:
                process.wait(timeout=allowance)
            except subprocess.TimeoutExpired:
                terminated = "timeout"
                _kill_tree(process)
                process.wait(timeout=CONTROL_SECONDS)
            for reader in readers:
                reader.join(timeout=DRAIN_SECONDS)
            code = process.returncode
            if terminated or code < 0:
                code = KILLED if terminated else 128 - code
            return _Run(code, bytes(readers[0].data), bytes(readers[1].data),
                        (readers[0].dropped, readers[1].dropped), terminated, time.monotonic() - began)
        finally:
            # A stream a background process still holds open is left to its reader (closing it would wait for it).
            for stream, reader in zip((process.stdout, process.stderr), readers, strict=True):
                if not reader.is_alive():
                    with contextlib.suppress(OSError):
                        stream.close()
            with self._lock:
                self._active = None

    def execute(self, request: TerminalRequest) -> TerminalResult:
        if not isinstance(request, TerminalRequest):
            raise TypeError("a validated TerminalRequest is required")
        done = self.run(request.command, request.cwd, request.timeout_seconds)
        return TerminalResult(done.code, done.stdout, done.stderr, *done.dropped, done.terminated)

    def interrupt(self):
        with self._lock:
            process = self._active
        if process is not None:
            _kill_tree(process)

    # -- the task's changes -----------------------------------------------------

    def upper(self):
        return Path(self.spec.overlay) / "upper"

    def lower_types(self, paths):
        """{path: "f" | "d" | "l" | "-"}: each path's type in the image alone."""
        found = {}
        paths = [path for path in paths if "\n" not in path]
        for start in range(0, len(paths), LOWER_BATCH):
            batch = paths[start:start + LOWER_BATCH]
            done = subprocess.run(lower_argv(self.spec), input="".join(p + "\n" for p in batch).encode(),
                                  capture_output=True, timeout=CONTROL_SECONDS * 5)
            answers = done.stdout.decode().split()
            if done.returncode != 0 or len(answers) != len(batch):
                raise RuntimeError("the image's own files could not be listed")
            found.update(zip(batch, answers, strict=True))
        return found

    def changes(self):
        """What the task changed against its image, in Docker's kinds: [(path, kind)], read from the overlay."""
        entries = list(upper_entries(self.upper()))
        lower = self.lower_types([path for path, kind, _ in entries if kind != "whiteout"])
        return changes_from(entries, lower)

    def changed_paths(self, mounts=()):
        """The task's changed paths as a Docker checkpoint lists them (``taste.benchmarks.branch_trial``)."""
        entries = list(upper_entries(self.upper()))
        lower = self.lower_types([path for path, kind, _ in entries if kind != "whiteout"])
        types = {path: kind for path, kind, _ in entries}
        return select(changes_from(entries, lower), types, mounts)


def _feed(stream, data):
    try:
        stream.write(data)
    except (BrokenPipeError, OSError):
        pass
    finally:
        with contextlib.suppress(OSError):
            stream.close()


def _opaque(path, names):
    if OPAQUE_MARK in names:
        return True
    read = getattr(os, "getxattr", None)  # Linux only; elsewhere the marker file alone tells
    for key in OPAQUE_XATTRS if read else ():
        try:
            if read(path, key, follow_symlinks=False) == b"y":
                return True
        except OSError:
            continue
    return False


def upper_entries(upper):
    """Each path in an overlay's upper directory, as the container sees it: (path, kind, opaque), where kind is
    "file", "dir", "link", "other" or "whiteout" (a deleted path). The overlay's own markers are not entries."""
    upper = Path(upper)
    if not upper.is_dir():
        return
    stack = [(upper, "/")]
    while stack:
        directory, inside = stack.pop()
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            continue
        for name in names:
            if name == OPAQUE_MARK:
                continue
            path = directory / name
            if name.startswith(WHITEOUT_PREFIX):
                yield posixpath.join(inside, name[len(WHITEOUT_PREFIX):]), "whiteout", False
                continue
            info = os.lstat(path)
            target = posixpath.join(inside, name)
            if stat.S_ISCHR(info.st_mode) and info.st_rdev == 0:
                yield target, "whiteout", False
            elif stat.S_ISDIR(info.st_mode):
                yield target, "dir", _opaque(path, os.listdir(path))
                stack.append((path, target))
            elif stat.S_ISLNK(info.st_mode):
                yield target, "link", False
            elif stat.S_ISREG(info.st_mode):
                yield target, "file", False
            else:
                yield target, "other", False


def changes_from(entries, lower):
    """Docker's change list from an upper directory's entries and the image's types: a whiteout is a deletion; a
    path the image lacks is added; one it has is changed; a replaced (opaque) directory the image had is deleted
    and added again, as Docker's overlay2 driver lists it."""
    found = []
    for path, kind, opaque in entries:
        if kind == "whiteout":
            found.append((path, DELETED))
        elif lower.get(path, "-") == "-":
            found.append((path, ADDED))
        elif kind == "dir" and opaque:
            found += [(path, DELETED), (path, ADDED)]
        else:
            found.append((path, CHANGED))
    return found


def select(entries, types, mounts=()):
    """The paths a Docker checkpoint copies and deletes (``docker_checkpoint._take``): the top-most added paths and
    the changed files, then the deleted paths marked " (deleted)"; never /proc, /sys, /dev, /run or a mount."""
    roots_out = NOT_THE_TASKS + tuple(mounts)
    kept = sorted({(path, kind) for path, kind in entries if not _inside(path, roots_out)})
    added = {path for path, kind in kept if kind == ADDED}
    roots = {path for path in added if not _under_added(path, added)}
    parents = {posixpath.dirname(path) for path, _ in kept}
    changed = {path for path, kind in kept if kind == CHANGED and path not in parents
               and not _under_added(path, added) and types.get(path) != "dir"}
    deleted = sorted({path for path, kind in kept if kind == DELETED})
    return [*sorted(roots | changed), *(path + " (deleted)" for path in deleted)]
