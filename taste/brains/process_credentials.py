"""Private systemd launch material, separate from durable scope metadata.

Only names and content digests enter a ScopeSpec. The outside controller keeps
the bytes in its private scope directory; systemd copies them into a read-only
credential mount for the service. No credential value enters argv or the unit
environment. These are trusted runtime credentials, never task-provided files.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

MAX_CREDENTIAL_BYTES = 64 * 1024
MAX_CREDENTIALS = 8


@dataclass(frozen=True)
class ScopeCredential:
    name: str
    sha256: str

    def __post_init__(self):
        if (not isinstance(self.name, str)
                or re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", self.name) is None):
            raise ValueError("scope credential name must be a simple identifier")
        if not isinstance(self.sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("scope credential requires its exact SHA-256 digest")

    @classmethod
    def from_bytes(cls, name: str, value: bytes):
        _bounded(value)
        return cls(name, hashlib.sha256(value).hexdigest())


def _bounded(value):
    if not isinstance(value, bytes) or not 0 < len(value) <= MAX_CREDENTIAL_BYTES:
        raise ValueError("scope credentials must contain 1..65536 bytes")


def credential_values(descriptors, values):
    """Copy and check the complete mapping before creating any durable state."""
    if values is None:
        values = {}
    if not isinstance(values, Mapping) or set(values) != {item.name for item in descriptors}:
        raise ValueError("scope credential values must match the admitted names exactly")
    result = dict(values)
    for item in descriptors:
        value = result[item.name]
        _bounded(value)
        if hashlib.sha256(value).hexdigest() != item.sha256:
            raise ValueError("scope credential content differs from the admitted digest")
    return result


def write_credentials(parent_fd, values):
    if not values:
        return
    os.mkdir("credentials", mode=0o700, dir_fd=parent_fd)
    directory = os.open("credentials", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    try:
        for name, value in values.items():
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=directory)
            with os.fdopen(fd, "wb") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
        os.fsync(directory)
        os.fsync(parent_fd)
    finally:
        os.close(directory)


def load_credential_properties(directory: Path, descriptors):
    """Validate the copied bytes and every pathname ancestor before launch.

    A worker-writable ancestor could replace an otherwise root-owned file
    between validation and systemd's read. Open each component without following
    symlinks and admit only controller/root-owned directories. Root-owned sticky
    temporary directories are safe because the next component is also owned by
    the controller. Mutations by the trusted root controller remain out of scope.
    """
    path = Path(directory)
    raw = str(directory)
    if (not path.is_absolute() or str(path) != raw or ".." in path.parts
            or any(c.isspace() or c in "%:\\\x00" for c in raw)):
        raise ValueError("scope credential directory must be an absolute, unexpanded path")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
            info = os.fstat(fd)
            sticky_root = info.st_uid == 0 and bool(info.st_mode & stat.S_ISVTX)
            if (info.st_uid not in {0, os.geteuid()}
                    or (info.st_mode & 0o022 and not sticky_root)):
                raise ValueError("scope credential path is replaceable by another user")
        info = os.fstat(fd)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("scope credential directory must be private and controller-owned")
        for item in descriptors:
            source = os.open(item.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
            with os.fdopen(source, "rb") as handle:
                info = os.fstat(handle.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                        or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1
                        or not 0 < info.st_size <= MAX_CREDENTIAL_BYTES):
                    raise ValueError("scope credential must be a private, owned, bounded regular file")
                value = handle.read(MAX_CREDENTIAL_BYTES + 1)
                _bounded(value)
                if hashlib.sha256(value).hexdigest() != item.sha256:
                    raise ValueError("scope credential content differs from the admitted digest")
        return [f"--property=LoadCredential={item.name}:{path / item.name}" for item in descriptors]
    finally:
        os.close(fd)
