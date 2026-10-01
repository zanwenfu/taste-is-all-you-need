"""Bounded file tools for exactly the artifacts named in a worker assignment.

Every lookup starts at an owned worktree directory descriptor. Parent links
and special files are rejected; tools never evaluate a shell or task code.
These effects belong to the memory worktree and are restored by its rollback.
External terminal effects require the separate persistent terminal broker.

The arguments are named ``artifact`` and ``body``, not ``path`` and
``content``. A benchmark that reads a run's tool calls classifies a call by
its argument names, and one carrying a path and content is read as an edit to
the task's repository. These files are in this system's memory; the record
must not say otherwise.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import stat
import uuid

from taste.brains.delivery import validate_artifact_path
from taste.brains.records import Assignment
from taste.brains.responses_conversation import ResponsesTool, ToolOutcome
from taste.memstore import Branch

ARTIFACT_TOOLS_VERSION = "taste.brains/ArtifactTools/2"
_MAX_WRITE = 65_536
_MAX_READ = 8192


class ArtifactTools:
    def __init__(self, branch: Branch, assignment: Assignment):
        if branch.name != assignment.worker:
            raise ValueError("artifact tools require their assigned worker branch")
        self.branch = branch
        self._pid = os.getpid()
        self._lease = branch._lease
        self._owned()
        info = branch.worktree.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("artifact worktree must be a directory")
        self._root_identity = (info.st_dev, info.st_ino)
        self.readable = frozenset(item.path for item in (*assignment.inputs, *assignment.outputs))
        self.writable = frozenset(item.path for item in assignment.outputs)
        for path in self.readable:
            validate_artifact_path(path)

    def _owned(self):
        if (os.getpid() != self._pid or self._lease is None or self._lease.closed
                or self.branch._lease is not self._lease):
            raise ValueError("artifact tools require their original active branch lease")

    def _admit(self):
        # Check before acquiring a possibly inherited lock after fork. These
        # handlers have no await points, so cancellation cannot split an effect.
        self._owned()
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise asyncio.CancelledError

    def _arguments(self, arguments, *, write=False, fields=()):
        if not isinstance(arguments, dict) or set(arguments) != {"artifact", *fields}:
            raise ValueError("tool arguments must exactly match the declared schema")
        path = arguments["artifact"]
        if not isinstance(path, str) or path not in (self.writable if write else self.readable):
            raise ValueError("path is not an artifact admitted for this operation")

    @contextlib.contextmanager
    def _parent(self, path, *, create=False):
        self._owned()
        parts = validate_artifact_path(path).split("/")
        descriptor = os.open(self.branch.worktree, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(descriptor)
            if (info.st_dev, info.st_ino) != self._root_identity:
                raise ValueError("artifact worktree directory changed identity")
            for part in parts[:-1]:
                if create:
                    try:
                        os.mkdir(part, mode=0o755, dir_fd=descriptor)
                        os.fsync(descriptor)
                    except FileExistsError:
                        pass
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            yield descriptor, parts[-1]
        finally:
            os.close(descriptor)

    @staticmethod
    def _regular(info):
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("artifact must be a regular file with one link")

    def _validate_read(self, arguments):
        self._arguments(arguments, fields=("offset", "limit"))
        if (type(arguments["offset"]) is not int or not 0 <= arguments["offset"] <= 1_073_741_824
                or type(arguments["limit"]) is not int or not 1 <= arguments["limit"] <= _MAX_READ):
            raise ValueError("read requires a nonnegative byte offset and a limit of 1-8192 bytes")

    def _validate_write(self, arguments):
        self._arguments(arguments, write=True, fields=("body", "executable"))
        if (not isinstance(arguments["body"], str) or len(arguments["body"].encode()) > _MAX_WRITE
                or type(arguments["executable"]) is not bool):
            raise ValueError("write requires UTF-8 text up to 64 KiB and a boolean executable flag")

    async def read(self, _effect_id, call):
        self._admit()
        self._validate_read(call.arguments)
        try:
            with self.branch._mutation_lock, self._parent(call.arguments["artifact"]) as (parent, leaf):
                # O_NONBLOCK prevents a substituted FIFO from stalling before
                # fstat can reject it. No-follow protects the final component.
                descriptor = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
                try:
                    info = os.fstat(descriptor)
                    self._regular(info)
                    os.lseek(descriptor, call.arguments["offset"], os.SEEK_SET)
                    data = os.read(descriptor, call.arguments["limit"])
                finally:
                    os.close(descriptor)
            try:
                content, encoding = data.decode("utf-8"), "utf-8"
            except UnicodeDecodeError:
                content, encoding = base64.b64encode(data).decode("ascii"), "base64"
            offset = call.arguments["offset"] + len(data)
            return ToolOutcome(json.dumps({"content": content, "encoding": encoding, "size": info.st_size,
                                           "next_offset": offset, "eof": offset >= info.st_size,
                                           "executable": bool(info.st_mode & 0o111)}, ensure_ascii=False))
        except (OSError, ValueError) as exc:
            return ToolOutcome("Artifact read refused: " + type(exc).__name__, True)

    async def write(self, _effect_id, call):
        self._admit()
        self._validate_write(call.arguments)
        try:
            with self.branch._mutation_lock, self._parent(call.arguments["artifact"], create=True) as (parent, leaf):
                with contextlib.suppress(FileNotFoundError):
                    self._regular(os.stat(leaf, dir_fd=parent, follow_symlinks=False))
                temporary = ".taste-artifact-" + uuid.uuid4().hex
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                     0o600, dir_fd=parent)
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(call.arguments["body"].encode())
                        stream.flush()
                        os.fchmod(stream.fileno(), 0o755 if call.arguments["executable"] else 0o644)
                        os.fsync(stream.fileno())
                    os.replace(temporary, leaf, src_dir_fd=parent, dst_dir_fd=parent)
                    os.fsync(parent)
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(temporary, dir_fd=parent)
            return ToolOutcome("Artifact saved.")
        except (OSError, ValueError) as exc:
            # An error after replace may mean the write took effect. The
            # returned error does not assert absence; the worker can read it.
            return ToolOutcome("Artifact write was not confirmed: " + type(exc).__name__, True)

    async def remove(self, _effect_id, call):
        self._admit()
        self._arguments(call.arguments, write=True)
        try:
            with self.branch._mutation_lock, self._parent(call.arguments["artifact"]) as (parent, leaf):
                info = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
                if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
                    raise ValueError("cannot remove a special file or directory")
                os.unlink(leaf, dir_fd=parent)
                os.fsync(parent)
            return ToolOutcome("Artifact removed.")
        except FileNotFoundError:
            return ToolOutcome("Artifact is absent.")
        except (OSError, ValueError) as exc:
            return ToolOutcome("Artifact removal was not confirmed: " + type(exc).__name__, True)

    def tools(self):
        def schema(properties):
            return {"type": "object", "properties": {"artifact": {"type": "string"}, **properties},
                    "required": ["artifact", *properties], "additionalProperties": False}

        return {
            "read_artifact": ResponsesTool(
                "Read up to 8192 bytes of an assigned input or output artifact, named by its "
                "assignment path. Artifacts are files in this system's memory workspace, not in "
                "any task environment. Binary or partial UTF-8 data is base64.",
                schema({"offset": {"type": "integer", "minimum": 0},
                        "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_READ}}),
                self._validate_read, self.read),
            "write_artifact": ResponsesTool(
                "Atomically replace an assigned output artifact with UTF-8 text, up to 64 KiB. "
                "This writes to this system's memory workspace, never to a task environment.",
                schema({"body": {"type": "string"}, "executable": {"type": "boolean"}}),
                self._validate_write, self.write),
            "remove_artifact": ResponsesTool(
                "Remove one assigned output artifact from this system's memory workspace. "
                "Directories cannot be removed.",
                schema({}), lambda arguments: self._arguments(arguments, write=True), self.remove),
        }
