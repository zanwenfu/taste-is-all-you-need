"""Unambiguous addresses for work in flight, with a conservative v1 upgrade.

The flat v1 address concatenated session, branch and suffix with dots. Those
names also permit dots, so the address could describe several owners. Never
guess during an upgrade: stop all old workers first, and resolve ambiguous
files explicitly. Mixed old/new writer versions are not supported.
"""

from __future__ import annotations

import fcntl
import os
import re
import tempfile
from contextlib import ExitStack
from pathlib import Path

from taste.memstore.backend import GitBackend
from taste.memstore.objects import BadName, BranchBusy

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_KIND = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_SUFFIX = re.compile(r"(?:\.[A-Za-z0-9._-]{1,127})?\Z")
TURN_SUFFIX = re.compile(r"\.(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_LEGACY_SUFFIXES = {
    **dict.fromkeys(
        ("lease", "intent", "acked", "tip", "rewinds", "session-mirror", "planner-transport-lock"),
        re.compile(r"\Z"),
    ),
    "turns": TURN_SUFFIX,
    "monitor": re.compile(r"(?:\.contract-[0-9a-f]{64})?\Z"),
    "runtime-session": re.compile(r"\.[0-9a-f]{24}\Z"),
    "worker-budget": re.compile(r"\.[0-9a-f]{64}\.jsonl\Z"),
}


class SidecarMigrationError(RuntimeError):
    """Recovery files cannot be assigned safely; original bytes are retained."""


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir_durable(path: Path) -> None:
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    for directory in reversed(missing):
        _fsync_directory(directory.parent)


class Sidecars:
    def __init__(self, backend: GitBackend) -> None:
        self.backend = backend
        self.root = backend.common_dir / "memstore-sidecars"
        self.layout = self.root / "layout"
        self._initialize()

    def path(self, session: str, branch: str, kind: str, suffix: str = "") -> Path:
        for value in (session, branch):
            if not _NAME.fullmatch(value):
                raise BadName(f"invalid sidecar identity: {value!r}")
        if not _KIND.fullmatch(kind) or not _SUFFIX.fullmatch(suffix):
            raise BadName(f"invalid sidecar kind or suffix: {kind!r}, {suffix!r}")
        parent = self.root / "v2" / session / branch
        _mkdir_durable(parent)
        return parent / f"{kind}{suffix}"

    def _ready(self) -> bool:
        try:
            version = self.layout.read_bytes()
        except FileNotFoundError:
            return False
        if version != b"2\n":
            raise SidecarMigrationError(f"unsupported sidecar layout in {self.layout}")
        return True

    def _migration_plan(self) -> list[tuple[Path, str, str, str, str]]:
        prefix = "refs/heads/mem/"
        owners = []
        for ref, _ in self.backend.for_each_ref(prefix):
            parts = ref.removeprefix(prefix).split("/")
            if len(parts) == 2 and all(_NAME.fullmatch(part) for part in parts):
                owners.append(tuple(parts))
        plan = []
        for source in sorted(self.backend.common_dir.glob("memstore.*")):
            kind = source.name.split(".", 2)[1]
            suffix_pattern = _LEGACY_SUFFIXES.get(kind)
            if suffix_pattern is None:
                continue  # The repo lock and temporary Git indexes are not sidecars.
            candidates = []
            for session, branch in owners:
                stem = f"memstore.{kind}.{session}.{branch}"
                if source.name.startswith(stem):
                    suffix = source.name[len(stem):]
                    if suffix_pattern.fullmatch(suffix):
                        candidates.append((source, session, branch, kind, suffix))
            if len(candidates) != 1:
                reason = "ambiguous ownership" if candidates else "no identifiable owner"
                raise SidecarMigrationError(f"sidecar migration: {reason} for {source}")
            _, session, branch, _, suffix = candidates[0]
            destination = self.root / "v2" / session / branch / f"{kind}{suffix}"
            if source.is_symlink() or not source.is_file() or destination.exists() or destination.is_symlink():
                raise SidecarMigrationError(f"sidecar migration: conflicting or unsafe file {source}")
            plan.append(candidates[0])
        return plan

    def _initialize(self) -> None:
        if self._ready():
            return
        with self.backend.lock(), ExitStack() as locks:
            if self._ready():
                return
            plan = self._migration_plan()  # Validate every owner before moving any file.
            lock_paths = {source for source, _, _, kind, _ in plan
                          if kind in {"lease", "planner-transport-lock", "worker-budget"}}
            # A previous upgrade may have stopped after moving some leases.
            # Lock both layouts before resuming it.
            for kind in ("lease", "planner-transport-lock", "worker-budget.*.jsonl"):
                lock_paths.update((self.root / "v2").glob(f"*/*/{kind}"))
            for path in sorted(lock_paths):
                handle = locks.enter_context(path.open("r+b"))
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise BranchBusy(f"sidecar migration requires all writers stopped: {path}") from exc
            for source, session, branch, kind, suffix in plan:
                destination = self.path(session, branch, kind, suffix)
                os.replace(source, destination)
                _fsync_directory(destination.parent)
                _fsync_directory(source.parent)
            _mkdir_durable(self.root)
            # A missing marker means an interrupted migration is retried. It
            # is published only after all recovery files have durable names.
            with tempfile.NamedTemporaryFile(dir=self.root, prefix="layout-", delete=False) as marker:
                try:
                    marker.write(b"2\n")
                    marker.flush()
                    os.fsync(marker.fileno())
                    os.replace(marker.name, self.layout)
                    _fsync_directory(self.root)
                finally:
                    Path(marker.name).unlink(missing_ok=True)
