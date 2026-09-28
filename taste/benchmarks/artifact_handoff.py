"""Complete, bounded artifact collection before a separate verifier is admitted.

The outside trial owner supplies the exact source/destination contract. Native
Harbor collection is best effort; this owner retains failures independently and
checks the captured bytes again before upload. The object is one-use and cannot
resume a dead trial. All destinations remain outside task write access.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from taste.benchmarks.output_snapshot import (
    DEFAULT_LIMITS,
    OutputSnapshotError,
    SnapshotLimits,
    _parts,
    _private_directory,
    download_file_snapshot,
    download_snapshot,
)
from taste.brains.docker_terminal import DockerTerminalBackend
from taste.brains.owned_thread import start_owned_thread
from taste.brains.terminal_broker import _settle


@dataclass(frozen=True)
class ArtifactTarget:
    source: str
    destination: Path
    kind: str
    service: str = "main"

    def __post_init__(self):
        if (not isinstance(self.source, str) or not self.source.startswith("/")
                or PurePosixPath(self.source).as_posix() != self.source or self.source.endswith("/")):
            raise ValueError("artifact source must be a canonical absolute container path")
        _parts(self.source[1:], DEFAULT_LIMITS)
        destination = Path(self.destination)
        if not destination.is_absolute() or ".." in destination.parts or not destination.name:
            raise ValueError("artifact destination must be an absolute host path")
        if self.kind not in {"file", "directory"}:
            raise ValueError("artifact kind must be file or directory")
        if not isinstance(self.service, str) or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,62}", self.service) is None:
            raise ValueError("artifact service must be an admitted compose service name")
        object.__setattr__(self, "destination", destination)


def _fingerprint(target, limits):
    """Read only ordinary, private, bounded host files without following links."""
    rows, size, count = [], 0, 0

    def visit(parent, name, relative):
        nonlocal size, count
        count += 1
        if count > limits.entries or len(relative) > limits.depth:
            raise OutputSnapshotError("artifact fingerprint exceeded its entry/depth limit")
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            info = os.fstat(fd)
            if info.st_uid != os.geteuid():
                raise OutputSnapshotError("artifact ownership changed")
            kind = "directory" if stat.S_ISDIR(info.st_mode) else "file"
            if not relative and kind != target.kind:
                raise OutputSnapshotError("artifact root type changed")
            if kind == "directory":
                if stat.S_IMODE(info.st_mode) != 0o700:
                    raise OutputSnapshotError("artifact directory is not private")
                rows.append({"path": "/".join(relative), "kind": kind})
                for child in sorted(os.listdir(fd)):
                    _parts(child, limits)
                    visit(fd, child, (*relative, child))
            else:
                if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                        or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > limits.file_bytes):
                    raise OutputSnapshotError("artifact is not an ordinary private bounded file")
                digest, read = hashlib.sha256(), 0
                while chunk := os.read(fd, min(65536, limits.file_bytes + 1 - read)):
                    read += len(chunk)
                    size += len(chunk)
                    if read > limits.file_bytes or size > limits.total_bytes:
                        raise OutputSnapshotError("artifact fingerprint exceeded its byte limit")
                    digest.update(chunk)
                after = os.fstat(fd)
                if read != info.st_size or any(getattr(after, key) != getattr(info, key) for key in
                        ("st_dev", "st_ino", "st_mode", "st_uid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")):
                    raise OutputSnapshotError("artifact changed during fingerprinting")
                rows.append({"path": "/".join(relative), "kind": kind, "bytes": read, "sha256": digest.hexdigest()})
        finally:
            os.close(fd)

    parent = _private_directory(target.destination.parent)
    try:
        visit(parent, target.destination.name, ())
    finally:
        os.close(parent)
    return {"entries": rows, "bytes": size}


class ArtifactHandoff:
    """One collection, permanently failed after any incomplete or cancelled copy.

    Limits apply both per archive and to aggregate captured file bytes/entries.
    Empty directories and empty files are legitimate outputs. A missing output
    is not silently treated as empty. Source modes are intentionally not applied
    to the host; executables/links require a separately admitted transfer policy.
    """

    def __init__(self, backend, targets, *, service_backends=None, limits=DEFAULT_LIMITS):
        targets = tuple(targets)
        if (not isinstance(backend, DockerTerminalBackend)
                or not isinstance(limits, SnapshotLimits) or not 1 <= len(targets) <= 64
                or any(not isinstance(target, ArtifactTarget) for target in targets)):
            raise ValueError("a bounded nonempty artifact contract is required")
        self.targets = MappingProxyType({target.source: target for target in targets})
        if len(self.targets) != len(targets):
            raise ValueError("artifact sources must be unique")
        for i, target in enumerate(targets):
            for other in targets[i + 1:]:
                a, b = target.destination, other.destination
                if a == b or a in b.parents or b in a.parents:
                    raise ValueError("artifact host destinations overlap")
        needed = {target.service for target in targets} - {"main"}
        services = {} if service_backends is None else service_backends
        if (not isinstance(services, Mapping) or set(services) != needed or len(services) > 15
                or any(not isinstance(value, DockerTerminalBackend) for value in services.values())):
            raise ValueError("each artifact service needs exactly one original Docker binding")
        bound = {"main": DockerTerminalBackend(backend.binding),
                 **{name: DockerTerminalBackend(value.binding) for name, value in services.items()}}
        identities = {(value.binding.socket_path, value.environment_id) for value in bound.values()}
        if len(identities) != len(bound):
            raise ValueError("different artifact services cannot alias the same container")
        self._backends = MappingProxyType(bound)
        self._backend = bound["main"]
        self._main_stopped, self._main_draining = False, False
        self.limits, self.phase = limits, "collecting"
        self._pending, self._receipts = set(), {}

    async def drain_main(self):
        """Stop the original main container before reading helper-service state.

        The trial owner must first settle agent processes and seal terminal
        grants. Harbor can swallow a stop_service error; retain it here so no
        later helper capture or seal can turn that failure into a valid result.
        Independent lifecycle cleanup remains mandatory after controller death.
        """
        operation = None
        try:
            required = {source for source, target in self.targets.items() if target.service == "main"}
            if (self.phase != "collecting" or self._main_draining or self._pending
                    or not required.issubset(self._receipts)):
                raise OutputSnapshotError("main artifact collection must finish before main drainage")
            self._main_draining = True
            operation = start_owned_thread(self._backend.stop_and_confirm)
            await asyncio.wait((operation,))
            if operation.result() != self._backend.environment_id or self.phase != "collecting":
                raise OutputSnapshotError("main container drainage did not confirm its original identity")
            self._main_stopped = True
            return asdict(self._backend.binding)
        except BaseException:
            self.phase = "failed"
            if operation is not None:
                await _settle(operation)
            raise
        finally:
            self._main_draining = False

    async def collect(self, source, destination, kind, *, deadline_unix, service=None):
        target = self.targets.get(source)
        operation, claimed = None, False
        try:
            service = "main" if service is None else service
            if (self.phase != "collecting" or self._main_draining or target is None or target.service != service
                    or target.destination != Path(destination) or target.kind != kind
                    or source in self._pending or source in self._receipts
                    or (service != "main" and not self._main_stopped)):
                raise OutputSnapshotError("artifact copy differs from its unused admitted target")
            self._pending.add(source)
            claimed = True

            def capture():
                # A service name is not authority to rediscover a replacement
                # container. Copies use the original binding, including start.
                backend = self._backends[target.service]
                if service != "main":
                    self._backend.stop_and_confirm()
                function = download_snapshot if kind == "directory" else download_file_snapshot
                summary = function(backend, source, target.destination, deadline_unix=deadline_unix, limits=self.limits)
                fingerprint = _fingerprint(target, self.limits)
                if fingerprint["bytes"] != summary["bytes"]:
                    raise OutputSnapshotError("artifact bytes changed after capture")
                if service != "main":
                    self._backend.stop_and_confirm()
                return {"source": source, "destination": str(target.destination), "kind": kind, "service": service,
                        "binding": asdict(backend.binding), "snapshot": summary, **fingerprint}

            operation = start_owned_thread(capture)
            await asyncio.wait((operation,))
            receipt = operation.result()
            if (sum(row["bytes"] for row in self._receipts.values()) + receipt["bytes"] > self.limits.total_bytes
                    or sum(len(row["entries"]) for row in self._receipts.values()) + len(receipt["entries"]) > self.limits.entries):
                raise OutputSnapshotError("artifact collection exceeded its aggregate limit")
            self._receipts[source] = receipt
            return copy.deepcopy(receipt)
        except BaseException:
            self.phase = "failed"
            if operation is not None:
                await _settle(operation)
            raise
        finally:
            if claimed:
                self._pending.discard(source)

    def seal(self):
        """Refuse incomplete collection even if upstream swallowed its errors."""
        if (self.phase != "collecting" or self._pending or self._main_draining
                or (len(self._backends) > 1 and not self._main_stopped)
                or set(self._receipts) != set(self.targets)):
            self.phase = "failed"
            raise OutputSnapshotError("artifact collection did not complete")
        self.phase = "sealed"
        return self.verify()

    def seal_harbor_manifest(self, payload: bytes, artifacts_root: Path):
        """Also require Harbor's complete successful collection report.

        A wrapper can fail after our bytes were captured (for example while
        recording a receipt). Native Harbor records that failure and continues;
        it must not be masked by a successful low-level snapshot.
        """
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise OutputSnapshotError("duplicate artifact manifest field")
                result[key] = value
            return result

        try:
            if type(payload) is not bytes or len(payload) > 65536:
                raise OutputSnapshotError("artifact manifest exceeded its limit")
            rows = json.loads(payload, object_pairs_hook=unique)
            if not isinstance(rows, list) or len(rows) != len(self.targets):
                raise OutputSnapshotError("artifact manifest is incomplete")
            seen = set()
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("source"), str):
                    raise OutputSnapshotError("invalid artifact manifest entry")
                target = self.targets.get(row["source"])
                if target is None or row["source"] in seen:
                    raise OutputSnapshotError("artifact manifest has an unadmitted or duplicate source")
                expected = {"source": target.source, "destination": "artifacts/" +
                    target.destination.relative_to(artifacts_root).as_posix(), "type": target.kind,
                    "status": "ok", "service": row.get("service"), "exclude": []}
                service = "main" if row.get("service") is None else row["service"]
                if row != expected or service != target.service:
                    raise OutputSnapshotError("Harbor did not report the exact successful artifact collection")
                seen.add(row["source"])
            return self.seal()
        except BaseException:
            self.phase = "failed"
            raise

    def verify(self):
        """Check captured bytes before upload; return receipts for durable reporting."""
        try:
            if self.phase != "sealed" or self._pending:
                raise OutputSnapshotError("artifact collection is not sealed")
            for source, target in self.targets.items():
                observed = _fingerprint(target, self.limits)
                receipt = self._receipts[source]
                if any(observed[key] != receipt[key] for key in ("entries", "bytes")):
                    raise OutputSnapshotError("captured artifact changed before verification")
            # A caller cannot mutate our admitted evidence through the result.
            return copy.deepcopy(list(self._receipts.values()))
        except BaseException:
            self.phase = "failed"
            raise
