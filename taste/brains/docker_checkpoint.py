"""A checkpoint of what a task changed in its container: one tar, its deletions, a manifest.

Docker lists every path a container changed against its image: added, changed
or deleted. A checkpoint copies out, through the daemon's archive endpoint,
the added paths (a top-most added directory whole) and the changed files into
one tar whose member names are the paths without their leading slash, and
lists the deleted paths. A changed directory is not copied: its changed
children are listed and copied themselves. Paths under /proc, /sys, /dev and
/run, and under the container's mounts, are not the task's files; they are
left out and listed. The tar is stored under its own SHA-256, so the same
files are stored once.

Nothing runs inside the container. The caller holds the terminal between
commands: a checkpoint of files a command is still writing would be torn.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import json
import os
import posixpath
import tarfile
import tempfile
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA = "taste.checkpoint/1"
CHECKPOINT_BYTES = 512 * 1024 * 1024
CHECKPOINT_SECONDS = 600
CHANGES_BYTES = 64 * 1024 * 1024
KINDS = {0: "changed", 1: "added", 2: "deleted"}
NOT_THE_TASKS = ("/proc", "/sys", "/dev", "/run")
_GO_DIR = 1 << 31


@dataclass(frozen=True)
class CheckpointManifest:
    """What one checkpoint holds and what it does not."""

    environment_id: str
    image: str
    tar: str
    tar_sha256: str
    tar_bytes: int
    changes: tuple[tuple[str, str], ...]
    copied: tuple[str, ...]
    deleted: tuple[str, ...]
    left_out: tuple[str, ...]
    over_cap: tuple[str, ...]
    taken_at: float

    @property
    def partial(self) -> bool:
        return bool(self.over_cap)

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, **asdict(self), "partial": self.partial}


def _inside(path, roots):
    return any(path == root or path.startswith(root.rstrip("/") + "/") for root in roots)


def _under_added(path, added):
    parent = posixpath.dirname(path)
    while parent not in ("", "/"):
        if parent in added:
            return True
        parent = posixpath.dirname(parent)
    return False


def _changes(wire, container, deadline):
    from taste.brains.docker_terminal import READ_BYTES, DockerTransportError, _remaining

    with wire.request("GET", f"/containers/{container}/changes", deadline) as response:
        if response.status != 200:
            raise DockerTransportError(f"Docker changes request returned HTTP {response.status}")
        data = bytearray()
        while True:
            _remaining(deadline)
            chunk = response.read1(READ_BYTES)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > CHANGES_BYTES:
                raise DockerTransportError("Docker changes list exceeded its byte limit")
    try:
        entries = json.loads(data) if data else []
    except ValueError as exc:
        raise DockerTransportError("invalid Docker changes JSON") from exc
    if not isinstance(entries, list) or not all(
            isinstance(item, dict) and isinstance(item.get("Path"), str) and item["Path"].startswith("/")
            and item.get("Kind") in KINDS for item in entries):
        raise DockerTransportError("Docker changes list is malformed")
    return [(item["Path"], item["Kind"]) for item in entries]


@contextlib.contextmanager
def _archive(wire, container, path, deadline):
    """The daemon's tar of one path, and the path's stat, as one streamed response."""
    from taste.brains.docker_terminal import DockerTransportError

    query = urllib.parse.quote(path, safe="/")
    with wire.request("GET", f"/containers/{container}/archive?path={query}", deadline) as response:
        if response.status != 200:
            raise DockerTransportError(f"Docker archive request returned HTTP {response.status}")
        try:
            stat = json.loads(base64.b64decode(response.getheader("X-Docker-Container-Path-Stat") or ""))
        except (ValueError, binascii.Error) as exc:
            raise DockerTransportError("Docker archive reply has no readable path stat") from exc
        if not isinstance(stat, dict) or type(stat.get("mode")) is not int:
            raise DockerTransportError("Docker archive reply has no readable path stat")
        yield stat, response


def take(backend, wire, directory, *, cap_bytes=CHECKPOINT_BYTES) -> CheckpointManifest:
    """Copy out what the container changed against its image; the caller holds the terminal."""
    container = backend.environment_id
    deadline = time.monotonic() + min(CHECKPOINT_SECONDS, backend.binding.deadline_unix - time.time())
    info = backend._inspect(wire, deadline)
    mounts = tuple(item["Destination"] for item in info.get("Mounts") or ()
                   if isinstance(item, dict) and isinstance(item.get("Destination"), str))
    entries = _changes(wire, container, deadline)
    left_out = sorted(path for path, _ in entries if _inside(path, NOT_THE_TASKS + mounts))
    kept = sorted((path, kind) for path, kind in entries if not _inside(path, NOT_THE_TASKS + mounts))
    added = {path for path, kind in kept if kind == 1}
    roots = {path for path in added if not _under_added(path, added)}
    # Docker lists a directory as changed when anything under it changed: a
    # changed path with changes beneath it is a directory, not to be copied.
    parents = {posixpath.dirname(path) for path, _ in kept}
    changed = {path for path, kind in kept
               if kind == 0 and path not in parents and not _under_added(path, added)}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    copied, over_cap, total = [], [], 0
    handle, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tar", dir=directory)
    try:
        with os.fdopen(handle, "wb") as raw, tarfile.open(fileobj=raw, mode="w|", format=tarfile.PAX_FORMAT) as out:
            for path in sorted(roots | changed):
                if total >= cap_bytes:
                    over_cap.append(path)
                    continue
                with _archive(wire, container, path, deadline) as (stat, response):
                    if path in changed and stat["mode"] & _GO_DIR:
                        continue  # a changed directory: its changed children are listed themselves
                    parent = posixpath.dirname(path).lstrip("/")
                    with tarfile.open(fileobj=response, mode="r|") as source:
                        for member in source:
                            body = source.extractfile(member) if member.isfile() else None
                            member.name = posixpath.join(parent, member.name) if parent else member.name
                            if member.islnk():
                                member.linkname = posixpath.join(parent, member.linkname) if parent else member.linkname
                            out.addfile(member, body)
                            total += member.size
                copied.append(path)
        digest = hashlib.sha256()
        with open(temporary, "rb") as written:
            for block in iter(lambda: written.read(1 << 20), b""):
                digest.update(block)
        name = digest.hexdigest() + ".tar"
        size = os.path.getsize(temporary)
        if (directory / name).exists():
            os.unlink(temporary)
        else:
            os.replace(temporary, directory / name)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
    return CheckpointManifest(
        environment_id=container, image=str(info.get("Image") or ""), tar=name, tar_sha256=name[:-4],
        tar_bytes=size, changes=tuple((path, KINDS[kind]) for path, kind in kept), copied=tuple(copied),
        deleted=tuple(path for path, kind in kept if kind == 2), left_out=tuple(left_out),
        over_cap=tuple(over_cap), taken_at=time.time())
