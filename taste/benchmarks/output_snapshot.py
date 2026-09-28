"""Bounded, link-free Docker output snapshots for a private lifecycle owner.

Never mount the destination in the task. The controller must serialize copies
and keep destination ancestors outside worker write access. Outputs are fresh
snapshots, not overlays: an existing nonempty destination is refused. This also
prevents stale rewards from surviving a failed or repeated verifier download.
"""

from __future__ import annotations

import io
import math
import os
import stat
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from taste.brains.docker_terminal import DockerTerminalBackend, _remaining, _Wire


class OutputSnapshotError(RuntimeError):
    """Output could not be copied completely within its admitted limits."""


@dataclass(frozen=True)
class SnapshotLimits:
    archive_bytes: int = 16 * 1024 * 1024
    file_bytes: int = 4 * 1024 * 1024
    total_bytes: int = 8 * 1024 * 1024
    entries: int = 1024
    depth: int = 16

    def __post_init__(self):
        for name, maximum in (("archive_bytes", 64 * 1024 * 1024), ("file_bytes", 16 * 1024 * 1024),
                              ("total_bytes", 32 * 1024 * 1024), ("entries", 4096), ("depth", 32)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"invalid snapshot {name} limit")


DEFAULT_LIMITS = SnapshotLimits()


def _parts(name, limits):
    if (not isinstance(name, str) or not name or len(name.encode("utf-8")) > 4096
            or "\\" in name or any(ord(c) < 32 or ord(c) == 127 for c in name)):
        raise OutputSnapshotError("invalid output pathname")
    parts = name.rstrip("/").split("/")
    if len(parts) > limits.depth or any(part in ("", ".", "..") for part in parts):
        raise OutputSnapshotError("invalid output pathname")
    return tuple(parts)


def _decode(payload, root_name, limits, *, single_file=False):
    if type(payload) is not bytes or len(payload) > limits.archive_bytes:
        raise OutputSnapshotError("output archive exceeded its byte limit")
    root = _parts(root_name, limits)
    if len(root) != 1:
        raise OutputSnapshotError("output archive root must be one name")
    members, directories, files, total = set(), set(), {}, 0
    try:
        # No compression: neither expansion bombs nor task-supplied executables
        # participate in the Docker archive download or its host extraction.
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            for item in archive:
                parts = _parts(item.name, limits)
                if parts[0] != root_name or parts in members or len(members) >= limits.entries:
                    raise OutputSnapshotError("output archive has duplicate, excess or foreign entries")
                members.add(parts)
                if item.sparse is not None or not (item.isdir() or item.isreg()):
                    raise OutputSnapshotError("output links, sparse files and special files are refused")
                relative = parts[1:]
                if single_file and (relative or not item.isreg()):
                    raise OutputSnapshotError("output archive must contain exactly its regular file root")
                if item.isdir():
                    if item.size != 0:
                        raise OutputSnapshotError("output directory carries unexpected data")
                    directories.add(relative)
                    continue
                if (not relative and not single_file) or item.name.endswith("/") or not 0 <= item.size <= limits.file_bytes:
                    raise OutputSnapshotError("output file exceeded its byte limit")
                total += item.size
                if total > limits.total_bytes:
                    raise OutputSnapshotError("output snapshot exceeded its total byte limit")
                stream = archive.extractfile(item)
                if stream is None:
                    raise OutputSnapshotError("output file has no archive data")
                data = stream.read(limits.file_bytes + 1)
                if len(data) != item.size:
                    raise OutputSnapshotError("output file was truncated")
                files[relative] = data
        if single_file and set(files) != {()}:
            raise OutputSnapshotError("output archive omitted its regular file root")
        if not single_file and () not in directories:
            raise OutputSnapshotError("output archive omitted its directory root")
        for path in directories | files.keys():
            for size in range(1, len(path)):
                if path[:size] in files:
                    raise OutputSnapshotError("output file shadows a directory")
                directories.add(path[:size])
    except (tarfile.TarError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
        raise OutputSnapshotError("invalid output archive") from exc
    return directories, files


def _private_directory(path):
    """Open every ancestor without following links, before touching any output."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise OutputSnapshotError("snapshot destination must be an absolute private path")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            info = os.fstat(descriptor)
            sticky_root = info.st_uid == 0 and info.st_mode & stat.S_ISVTX
            if info.st_uid not in (0, os.geteuid()) or (info.st_mode & 0o022 and not sticky_root):
                raise OutputSnapshotError("snapshot destination has a writable or foreign ancestor")
        info = os.fstat(descriptor)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise OutputSnapshotError("snapshot destination must be private to its owner")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def extract_snapshot(payload: bytes, destination: Path, root_name: str, *, limits=DEFAULT_LIMITS):
    """Validate the whole archive before writing regular files into an empty 0700 directory.

    A failed filesystem write leaves an incomplete snapshot and raises; callers
    must never grade it or retry into the same directory. No metadata from the
    archive (ownership, mode, xattrs, timestamps) is applied to host files.
    """
    if not isinstance(limits, SnapshotLimits):
        raise TypeError("validated snapshot limits are required")
    directories, files = _decode(payload, root_name, limits)
    descriptor = _private_directory(destination)
    try:
        if os.listdir(descriptor):
            raise OutputSnapshotError("snapshot destination must be empty; refusing stale output")
        for path in sorted(directories - {()}, key=lambda p: (len(p), p)):
            os.mkdir("/".join(path), mode=0o700, dir_fd=descriptor)
        for path, data in files.items():
            fd = os.open("/".join(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=descriptor)
            with os.fdopen(fd, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"files": len(files), "bytes": sum(map(len, files.values())), "archive_bytes": len(payload)}


def extract_file_snapshot(payload: bytes, destination: Path, root_name: str, *, limits=DEFAULT_LIMITS):
    """Validate a one-file archive before exclusively creating its private host file.

    Its existing parent must be private. An existing destination (including a
    link) is never replaced. Failed writes remain incomplete and cannot be used
    for grading or retried into the same destination.
    """
    if not isinstance(limits, SnapshotLimits):
        raise TypeError("validated snapshot limits are required")
    _, files = _decode(payload, root_name, limits, single_file=True)
    destination = Path(destination)
    if not destination.is_absolute() or destination.name in ("", ".", ".."):
        raise OutputSnapshotError("snapshot destination must name an absolute file")
    descriptor = _private_directory(destination.parent)
    try:
        fd = os.open(destination.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=descriptor)
        with os.fdopen(fd, "wb") as output:
            output.write(files[()])
            output.flush()
            os.fsync(output.fileno())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"files": 1, "bytes": len(files[()]), "archive_bytes": len(payload)}


def _download_archive(backend, source, deadline_unix, limits):
    """Read bytes and confirm the original identity before host publication.

    Both file and directory snapshots use this same bounded transport. A
    stopped container can be copied; this never starts or executes it.
    """
    if not isinstance(backend, DockerTerminalBackend) or not isinstance(limits, SnapshotLimits):
        raise TypeError("a bound Docker backend and validated snapshot limits are required")
    if type(deadline_unix) not in (int, float) or not math.isfinite(deadline_unix):
        raise ValueError("a finite absolute snapshot deadline is required")
    if (not isinstance(source, str) or not source.startswith("/") or source.endswith("/")
            or not source[1:] or PurePosixPath(source).as_posix() != source):
        raise OutputSnapshotError("a canonical absolute container path is required")
    _parts(source[1:], limits)
    deadline = time.monotonic() + min(10, deadline_unix - time.time(),
                                     backend.binding.deadline_unix - time.time())
    _remaining(deadline)
    wire = _Wire(backend.binding.socket_path)
    try:
        backend._inspect(wire, deadline)
        with wire.request("GET", f"/containers/{backend.environment_id}/archive?path={quote(source, safe='')}",
                          deadline) as response:
            if response.status != 200 or response.getheader("Content-Type", "").split(";", 1)[0] != "application/x-tar":
                raise OutputSnapshotError("Docker did not provide an output archive")
            data = bytearray()
            while True:
                _remaining(deadline)
                chunk = response.read1(min(65536, limits.archive_bytes + 1 - len(data)))
                if not chunk:
                    if response.length not in (0, None):
                        raise OutputSnapshotError("Docker output archive was truncated")
                    break
                data.extend(chunk)
                if len(data) > limits.archive_bytes:
                    raise OutputSnapshotError("output archive exceeded its byte limit")
        backend._inspect(wire, deadline)
        return bytes(data)
    finally:
        wire.cancel()


def download_snapshot(backend: DockerTerminalBackend, source: str, destination: Path, *,
                      deadline_unix: float, limits=DEFAULT_LIMITS):
    """Read-only directory copy, bound to the original container and deadline.

    The caller owns this synchronous operation until it returns, including on
    async cancellation. It must not delete the container before it completes.
    """
    payload = _download_archive(backend, source, deadline_unix, limits)
    return extract_snapshot(payload, destination, PurePosixPath(source).name, limits=limits)


def download_file_snapshot(backend: DockerTerminalBackend, source: str, destination: Path, *,
                           deadline_unix: float, limits=DEFAULT_LIMITS):
    """Read-only file copy with the same ownership and deadline contract as directories."""
    payload = _download_archive(backend, source, deadline_unix, limits)
    return extract_file_snapshot(payload, destination, PurePosixPath(source).name, limits=limits)
