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
# A summary names at most this many paths of each list, each cut to PATH_CHARS.
SUMMARY_PATHS = 40
PATH_CHARS = 300
_GO_DIR = 1 << 31


def _first(name, items, count):
    """The first ``count`` of ``items`` under ``name``, and how many more there are."""
    return {name: [item[:PATH_CHARS] for item in items[:count]], f"more_{name}": max(0, len(items) - count)}


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

    @classmethod
    def from_dict(cls, value) -> CheckpointManifest:
        """A manifest as ``to_dict`` wrote it, as when a later trial restores a saved checkpoint."""
        names = {item for item in cls.__dataclass_fields__}
        if not isinstance(value, dict) or value.get("schema") != SCHEMA or set(value) != {*names, "schema", "partial"}:
            raise ValueError("not a checkpoint manifest of schema " + SCHEMA)
        try:
            manifest = cls(**{name: value[name] for name in names if name not in (
                "changes", "copied", "deleted", "left_out", "over_cap")},
                changes=tuple((str(path), str(kind)) for path, kind in value["changes"]),
                **{name: tuple(str(path) for path in value[name])
                   for name in ("copied", "deleted", "left_out", "over_cap")})
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint manifest is malformed") from exc
        # The tar is named by its own digest, beside the manifest: never a path elsewhere.
        if (not isinstance(manifest.tar_sha256, str) or len(manifest.tar_sha256) != 64
                or manifest.tar != manifest.tar_sha256 + ".tar" or posixpath.basename(manifest.tar) != manifest.tar
                or manifest.partial != value["partial"]):
            raise ValueError("checkpoint manifest is malformed")
        return manifest

    def summary(self, *, paths=SUMMARY_PATHS) -> dict:
        """A bounded account for a planner or a reply: counts, size and the first paths.

        ``added`` counts top-most added paths (a new directory once, whatever it
        holds), ``changed`` the changed files, ``deleted`` the deleted paths.
        """
        kinds = dict(self.changes)
        return {"taken_at": round(self.taken_at, 3), "tar_sha256": self.tar_sha256, "bytes": self.tar_bytes,
                "added": sum(1 for path in self.copied if kinds.get(path) == "added"),
                "changed": sum(1 for path in self.copied if kinds.get(path) == "changed"),
                "deleted": len(self.deleted), "partial": self.partial, "left_out": len(self.left_out),
                **_first("paths", self.copied, paths), **_first("deleted_paths", self.deleted, paths),
                **_first("over_cap", self.over_cap, paths)}


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
    if entries is None:
        entries = []  # Docker answers null for a container that changed nothing
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


# -- restore ----------------------------------------------------------------------------

ROLE_LABEL = "taste.terminal.role"
RM_CHUNK = 256


@dataclass(frozen=True)
class RestoreReceipt:
    """What a restore did, and whether the container now matches its checkpoint."""

    checkpoint: str
    removed: tuple[str, ...]
    from_image: tuple[str, ...]
    deleted: tuple[str, ...]
    mismatches: tuple[str, ...]
    seconds: float

    @property
    def exact(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict:
        return {**asdict(self), "exact": self.exact}

    def summary(self, *, paths=SUMMARY_PATHS) -> dict:
        """A bounded account: whether the files now match, what was done, and how long it took."""
        return {"tar_sha256": self.checkpoint, "exact": self.exact, "seconds": self.seconds,
                "removed": len(self.removed), "from_image": len(self.from_image), "deleted": len(self.deleted),
                **_first("mismatches", self.mismatches, paths)}


def _remove(wire, container, paths, deadline):
    """``rm -rf`` inside the container, as root: the controller undoes what any user made."""
    from taste.brains.docker_terminal import DockerTransportError, _full_id

    for start in range(0, len(paths), RM_CHUNK):
        created = wire.control("POST", f"/containers/{container}/exec", deadline,
                               {"Cmd": ["rm", "-rf", "--", *paths[start:start + RM_CHUNK]], "User": "0",
                                "AttachStdout": False, "AttachStderr": False, "Tty": False}, statuses=(201,))
        exec_id = created.get("Id") if isinstance(created, dict) else None
        if not _full_id(exec_id):
            raise DockerTransportError("Docker did not identify the created exec")
        wire.control("POST", f"/exec/{exec_id}/start", deadline, {"Detach": True, "Tty": False})
        while True:
            state = wire.control("GET", f"/exec/{exec_id}/json", deadline)
            if not isinstance(state, dict) or type(state.get("Running")) is not bool:
                raise DockerTransportError("Docker exec has no readable state")
            if not state["Running"]:
                if state.get("ExitCode") != 0:
                    raise DockerTransportError("removing paths inside the container failed")
                break
            time.sleep(0.05)


def _put(wire, container, directory, body, deadline):
    from taste.brains.docker_terminal import DockerTransportError

    query = urllib.parse.quote(directory, safe="/")
    with wire.request("PUT", f"/containers/{container}/archive?path={query}", deadline, raw=body) as response:
        if response.status != 200:
            raise DockerTransportError(f"Docker archive upload returned HTTP {response.status}")


def _originals(wire, backend, image, paths, deadline):
    """Copy the image's own versions of ``paths`` into the task container.

    They come from a helper container made from the same image, never started,
    labelled with the terminal's owner token, and removed afterwards.
    """
    from taste.brains.docker_terminal import OWNER_LABEL, DockerTransportError, _full_id

    if not paths:
        return
    created = wire.control("POST", "/containers/create", deadline,
                           {"Image": image, "Cmd": ["true"], "NetworkDisabled": True,
                            "Labels": {OWNER_LABEL: backend.binding.owner_token,
                                       ROLE_LABEL: "checkpoint-originals"},
                            "HostConfig": {"NetworkMode": "none", "AutoRemove": False}}, statuses=(201,))
    helper = created.get("Id") if isinstance(created, dict) else None
    if not _full_id(helper):
        raise DockerTransportError("Docker did not identify the helper container")
    try:
        for path in paths:
            with _archive(wire, helper, path, deadline) as (_stat, response):
                body = response.read()
            _put(wire, backend.environment_id, posixpath.dirname(path) or "/", body, deadline)
    finally:
        wire.control("DELETE", f"/containers/{helper}?force=1", deadline, statuses=(204, 404))


def _kinds_of(wire, container, mounts, deadline):
    return {path: KINDS[kind] for path, kind in _changes(wire, container, deadline)
            if not _inside(path, NOT_THE_TASKS + mounts)}


def restore(backend, wire, manifest, directory, *, any_container=False) -> RestoreReceipt:
    """Return the task's files to ``manifest``; the caller holds the terminal.

    ``any_container``: the checkpoint may have been taken in another container
    of the same image, as when a branch trial restores the files of step k
    that an earlier trial rebuilt. The image must still be the same.

    Everything added since is removed (what the checkpoint holds comes back
    from its tar); paths the image holds that were changed or deleted since,
    and that the checkpoint does not hold, get the image's version back; the
    checkpoint's tar is put back and its deletions made again. Then the
    container's changes are compared with the checkpoint's; a directory Docker
    lists as changed only because something under it was touched is not a
    difference. Running processes are not restored.

    Refused before anything changes for a partial checkpoint (it lacks paths
    it would delete) and when too little time remains for one of this size:
    removals come before the copy back, and a restore cut short between them
    would leave the files mixed.
    """
    from taste.brains.docker_terminal import DockerTransportError
    from taste.brains.environment_records import restore_seconds
    from taste.brains.terminal_broker import TerminalConflict

    if manifest.over_cap:
        raise TerminalConflict("a partial checkpoint is not restored: it lacks paths it would delete")
    started = time.monotonic()
    container = backend.environment_id
    deadline = time.monotonic() + min(CHECKPOINT_SECONDS, backend.binding.deadline_unix - time.time())
    tar = Path(directory) / manifest.tar
    digest = hashlib.sha256()
    with open(tar, "rb") as stored:
        for block in iter(lambda: stored.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != manifest.tar_sha256:
        raise TerminalConflict("the checkpoint's tar does not match its checksum")
    info = backend._inspect(wire, deadline)
    if str(info.get("Image") or "") != manifest.image or (container != manifest.environment_id
                                                          and not any_container):
        raise TerminalConflict("the checkpoint is of another image or container")
    mounts = tuple(item["Destination"] for item in info.get("Mounts") or ()
                   if isinstance(item, dict) and isinstance(item.get("Destination"), str))
    current = _kinds_of(wire, container, mounts, deadline)
    target = dict(manifest.changes)
    parents = {posixpath.dirname(path) for path in current} | {posixpath.dirname(path) for path in target}
    added_now = {path for path, kind in current.items() if kind == "added"}
    removed = sorted(path for path in added_now if not _under_added(path, added_now))
    from_image, deleted_roots = [], set()
    for path in sorted(current):
        kind = current[path]
        if path in target or kind == "added" or (kind == "changed" and path in parents):
            continue
        if any(path.startswith(root + "/") for root in deleted_roots):
            continue  # inside a deleted tree that comes back whole
        from_image.append(path)
        if kind == "deleted":
            deleted_roots.add(path)
    deleted = sorted(path for path, kind in target.items() if kind == "deleted")
    if deadline - time.monotonic() < restore_seconds(manifest.tar_bytes):
        raise TerminalConflict("too little time remains to restore this checkpoint without leaving it half done")
    _remove(wire, container, removed, deadline)
    _originals(wire, backend, manifest.image, from_image, deadline)
    with open(tar, "rb") as stored:
        _put(wire, container, "/", stored, deadline)
    _remove(wire, container, deleted, deadline)
    after = _kinds_of(wire, container, mounts, deadline)
    mismatches = []
    for path in sorted(set(after) | set(target)):
        if after.get(path) == target.get(path):
            continue
        if "changed" in (after.get(path), target.get(path)) and None in (after.get(path), target.get(path)):
            # Listed as changed on one side only: a directory touched along the way is no difference.
            try:
                with _archive(wire, container, path, deadline) as (stat, _response):
                    if stat["mode"] & _GO_DIR:
                        continue
            except DockerTransportError:
                pass  # not there to look at: a difference
        mismatches.append(path)
    return RestoreReceipt(checkpoint=manifest.tar_sha256, removed=tuple(removed), from_image=tuple(from_image),
                          deleted=tuple(deleted), mismatches=tuple(mismatches),
                          seconds=round(time.monotonic() - started, 3))
