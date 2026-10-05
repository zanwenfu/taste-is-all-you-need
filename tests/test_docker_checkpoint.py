"""A checkpoint of what a task changed in its container, taken against a fake daemon."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import posixpath
import tarfile
import time
import urllib.parse
from pathlib import Path

import pytest

from taste.brains.docker_terminal import DockerTerminalBackend
from taste.brains.terminal_broker import TerminalConflict, TerminalFenced
from tests.test_docker_terminal import CONTAINER, TOKEN, backend
from tests.test_docker_terminal import daemon as _daemon

daemon = _daemon

GO_DIR = 1 << 31
CHANGED, ADDED, DELETED = 0, 1, 2

# The container's files after some work: None is a directory.
FILES = {
    "/app": None, "/app/new.py": b"print('new')\n", "/app/pkg": None, "/app/pkg/a.py": b"A = 1\n",
    "/etc": None, "/etc/app.conf": b"x = 2\n", "/tmp": None, "/logs": None, "/logs/agent.txt": b"log\n",
}
CHANGES = [{"Path": "/app", "Kind": CHANGED}, {"Path": "/app/new.py", "Kind": ADDED},
           {"Path": "/app/pkg", "Kind": ADDED}, {"Path": "/app/pkg/a.py", "Kind": ADDED},
           {"Path": "/etc", "Kind": CHANGED}, {"Path": "/etc/app.conf", "Kind": CHANGED},
           {"Path": "/tmp", "Kind": CHANGED}, {"Path": "/tmp/old", "Kind": DELETED},
           {"Path": "/logs/agent.txt", "Kind": ADDED}, {"Path": "/run/app.pid", "Kind": ADDED}]


def _tar_of(path):
    """What the daemon's archive endpoint returns for one path: names relative to its parent."""
    buffer = io.BytesIO()
    parent = path.rsplit("/", 1)[0]
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(item for item in FILES if item == path or item.startswith(path + "/")):
            info = tarfile.TarInfo(name[len(parent) + 1:])
            content = FILES[name]
            if content is None:
                info.type, info.mode = tarfile.DIRTYPE, 0o755
                archive.addfile(info)
            else:
                info.size, info.mode = len(content), 0o644
                archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def serve(daemon, *, mounts=("/logs",)):
    daemon.info["Mounts"] = [{"Destination": item} for item in mounts]
    prefix = f"/containers/{CONTAINER}/archive?path="

    def override(handler, path):
        if path == f"/containers/{CONTAINER}/changes":
            daemon.reply(handler, CHANGES)
            return True
        if path.startswith(prefix):
            wanted = urllib.parse.unquote(path[len(prefix):])
            data = _tar_of(wanted)
            is_dir = FILES[wanted] is None
            stat = {"name": wanted.rsplit("/", 1)[1], "size": 0 if is_dir else len(FILES[wanted]),
                    "mode": (GO_DIR | 0o755) if is_dir else 0o644, "mtime": "2026-10-05T00:00:00Z",
                    "linkTarget": ""}
            handler.send_response(200)
            handler.send_header("X-Docker-Container-Path-Stat", base64.b64encode(json.dumps(stat).encode()).decode())
            handler.send_header("Content-Type", "application/x-tar")
            handler.send_header("Content-Length", str(len(data)))
            handler.send_header("Connection", "close")
            handler.end_headers()
            handler.wfile.write(data)
            return True
        return False

    daemon.override = override


def test_a_checkpoint_holds_what_the_task_added_and_changed(daemon, tmp_path):
    serve(daemon)
    manifest = backend(daemon).checkpoint(tmp_path / "checkpoints")
    stored = tmp_path / "checkpoints" / manifest.tar
    assert stored.name == manifest.tar_sha256 + ".tar"
    assert hashlib.sha256(stored.read_bytes()).hexdigest() == manifest.tar_sha256
    with tarfile.open(stored) as archive:
        members = {member.name: member for member in archive.getmembers()}
        assert sorted(members) == ["app/new.py", "app/pkg", "app/pkg/a.py", "etc/app.conf"]
        assert archive.extractfile("etc/app.conf").read() == b"x = 2\n"
        assert members["app/pkg"].isdir()
    # The top-most added directory is copied whole; a changed directory is not copied at all.
    assert manifest.copied == ("/app/new.py", "/app/pkg", "/etc/app.conf")
    assert manifest.deleted == ("/tmp/old",)
    # Mounts and the kernel's own trees are not the task's files.
    assert manifest.left_out == ("/logs/agent.txt", "/run/app.pid")
    assert manifest.over_cap == () and not manifest.partial
    assert ("/app/pkg/a.py", "added") in manifest.changes and ("/app", "changed") in manifest.changes


def test_the_same_files_are_stored_once(daemon, tmp_path):
    serve(daemon)
    first = backend(daemon).checkpoint(tmp_path / "checkpoints")
    second = backend(daemon).checkpoint(tmp_path / "checkpoints")
    assert first.tar_sha256 == second.tar_sha256
    assert sorted(path.name for path in (tmp_path / "checkpoints").iterdir()) == [first.tar]


def test_a_checkpoint_over_its_cap_is_partial_and_says_what_it_left(daemon, tmp_path):
    serve(daemon)
    manifest = backend(daemon).checkpoint(tmp_path / "checkpoints", cap_bytes=1)
    assert manifest.partial and manifest.copied == ("/app/new.py",)
    assert manifest.over_cap == ("/app/pkg", "/etc/app.conf")


def test_no_checkpoint_is_taken_while_a_command_runs(daemon, tmp_path):
    serve(daemon)
    executor = backend(daemon)
    executor._active = object()
    with pytest.raises(TerminalFenced, match="between commands"):
        executor.checkpoint(tmp_path / "checkpoints")


# -- restore: a fake daemon over a small filesystem ------------------------------------------

HELPER = "e" * 64


class Filesystem:
    """An image's files and a container's, and Docker's list of what differs (None: a directory)."""

    def __init__(self, image):
        self.image, self.files = dict(image), dict(image)

    def changes(self):
        entries = {}
        for path in sorted(set(self.image) | set(self.files)):
            if path in self.files and path not in self.image:
                entries[path] = ADDED
            elif path in self.image and path not in self.files:
                if posixpath.dirname(path) in self.files or posixpath.dirname(path) == "/":
                    entries[path] = DELETED  # Docker lists a deleted tree by its top
            elif self.files[path] != self.image[path]:
                entries[path] = CHANGED
        for path in list(entries):
            parent = posixpath.dirname(path)
            while parent != "/":
                if parent in self.image and parent in self.files and parent not in entries:
                    entries[parent] = CHANGED
                parent = posixpath.dirname(parent)
        return [{"Path": path, "Kind": kind} for path, kind in sorted(entries.items())]

    def tar(self, tree, path):
        buffer = io.BytesIO()
        parent = posixpath.dirname(path)
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name in sorted(item for item in tree if item == path or item.startswith(path + "/")):
                info = tarfile.TarInfo(name[len(parent):].lstrip("/"))
                if tree[name] is None:
                    info.type, info.mode = tarfile.DIRTYPE, 0o755
                    archive.addfile(info)
                else:
                    info.size, info.mode = len(tree[name]), 0o644
                    archive.addfile(info, io.BytesIO(tree[name]))
        return buffer.getvalue()

    def put(self, directory, data):
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive.getmembers():
                path = posixpath.normpath(posixpath.join(directory, member.name))
                self.files[path] = None if member.isdir() else archive.extractfile(member).read()

    def remove(self, paths):
        for path in paths:
            for name in [item for item in self.files if item == path or item.startswith(path + "/")]:
                del self.files[name]


def restoring(daemon, filesystem, mounts=()):
    """Serve changes, archives, puts, the helper container and rm -rf execs from ``filesystem``."""
    daemon.info["Mounts"] = [{"Destination": item} for item in mounts]
    daemon.info["Image"] = "sha256:" + "f" * 64
    pending = {}

    def stat(tree, path):
        is_dir = tree[path] is None
        value = {"name": posixpath.basename(path), "size": 0 if is_dir else len(tree[path]),
                 "mode": (GO_DIR | 0o755) if is_dir else 0o644, "mtime": "", "linkTarget": ""}
        return base64.b64encode(json.dumps(value).encode()).decode()

    def send(handler, status, data=b"", headers=()):
        handler.send_response(status)
        for key, value in headers:
            handler.send_header(key, value)
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(data)

    def override(handler, path):
        method = handler.command
        for owner, tree in ((CONTAINER, filesystem.files), (HELPER, filesystem.image)):
            prefix = f"/containers/{owner}/archive?path="
            if path.startswith(prefix):
                wanted = urllib.parse.unquote(path[len(prefix):])
                if method == "PUT":
                    filesystem.put(wanted, handler.body)
                    send(handler, 200)
                elif wanted not in tree:
                    send(handler, 404)
                else:
                    send(handler, 200, filesystem.tar(tree, wanted),
                         (("X-Docker-Container-Path-Stat", stat(tree, wanted)), ("Content-Type", "application/x-tar")))
                return True
        if path == f"/containers/{CONTAINER}/changes":
            # As the real daemon: null, not [], when nothing changed.
            daemon.reply(handler, filesystem.changes() or None)
        elif path.startswith("/containers/create"):
            body = json.loads(handler.body)
            assert body["Image"] == daemon.info["Image"] and body["Labels"]
            daemon.helpers.append("created")
            daemon.reply(handler, {"Id": HELPER}, status=201)
        elif path.startswith(f"/containers/{HELPER}") and method == "DELETE":
            daemon.helpers.append("removed")
            send(handler, 204)
        elif path == f"/containers/{CONTAINER}/exec":
            body = json.loads(handler.body)
            assert body["Cmd"][:3] == ["rm", "-rf", "--"] and body["User"] == "0"
            pending["cmd"] = body["Cmd"]
            daemon.reply(handler, {"Id": EXEC_RM}, status=201)
        elif path == f"/exec/{EXEC_RM}/start":
            assert json.loads(handler.body) == {"Detach": True, "Tty": False}
            filesystem.remove(pending["cmd"][3:])
            send(handler, 200)
        elif path == f"/exec/{EXEC_RM}/json":
            daemon.reply(handler, {"ID": EXEC_RM, "ContainerID": CONTAINER, "Running": False, "ExitCode": 0})
        else:
            return False
        return True

    daemon.helpers = []
    daemon.override = override


EXEC_RM = "9" * 64


@pytest.fixture
def raw_daemon(tmp_path, monkeypatch):
    """The terminal tests' fake daemon, answering PUT and DELETE too, with raw request bodies."""
    import socketserver
    import tempfile
    import threading
    from http.server import BaseHTTPRequestHandler

    from taste.brains import docker_terminal as transport
    from tests.test_docker_terminal import Daemon

    fake = Daemon(tmp_path)
    monkeypatch.setattr(transport, "CGROUP_ROOT", fake.cgroups)
    monkeypatch.setattr(transport, "PROC_ROOT", fake.proc)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _any(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                self.body = self.rfile.read(length) if length else b""
                path = self.path.removeprefix("/v1.51")
                fake.calls.append((self.command, path, None))
                if not (fake.override and fake.override(self, path)):
                    if self.command == "GET" and path == f"/containers/{CONTAINER}/json":
                        fake.reply(self, fake.info)
                    else:
                        raise AssertionError(f"unexpected daemon operation: {self.command} {path}")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except BaseException as exc:
                fake.errors.append(exc)
            finally:
                self.close_connection = True

        do_GET = do_POST = do_PUT = do_DELETE = _any

        def log_message(self, *_args):
            pass

    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        pass

    with tempfile.TemporaryDirectory(prefix="taste-wire-", dir="/tmp") as socket_dir:
        fake.socket = str(Path(socket_dir) / "docker.sock")
        with Server(fake.socket, Handler) as server:
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
            thread.start()
            try:
                yield fake
            finally:
                server.shutdown()
                thread.join(3)
    assert not fake.errors, fake.errors


IMAGE = {"/app": None, "/app/main.py": b"print('v1')\n", "/etc": None, "/etc/app.conf": b"x = 1\n",
         "/etc/hosts.allow": b"ALL\n", "/usr": None, "/usr/lib": None, "/usr/lib/lib.so": b"\x7fELF"}


def test_a_restore_returns_the_container_to_its_checkpoint(raw_daemon, tmp_path):
    filesystem = Filesystem(IMAGE)
    restoring(raw_daemon, filesystem)
    # Work up to the checkpoint: a new module, a changed config, a deleted file.
    filesystem.files.update({"/app/util.py": b"U = 1\n", "/etc/app.conf": b"x = 2\n"})
    del filesystem.files["/etc/hosts.allow"]
    at_checkpoint = dict(filesystem.files)
    executor = backend(raw_daemon)
    manifest = executor.checkpoint(tmp_path / "checkpoints")
    # Later work that breaks things: a new tree, the module rewritten, the config and a
    # library the image holds changed, the deleted file back, and the main file deleted.
    filesystem.files.update({"/app/build": None, "/app/build/out.o": b"obj", "/app/util.py": b"broken",
                             "/etc/app.conf": b"x = 3\n", "/usr/lib/lib.so": b"patched",
                             "/etc/hosts.allow": b"ALL\n"})
    del filesystem.files["/app/main.py"]
    receipt = executor.restore(manifest, tmp_path / "checkpoints")
    assert filesystem.files == at_checkpoint
    assert receipt.exact and receipt.mismatches == ()
    assert receipt.removed == ("/app/build", "/app/util.py")
    assert receipt.from_image == ("/app/main.py", "/usr/lib/lib.so")
    assert receipt.deleted == ("/etc/hosts.allow",)
    # The image's originals came from a helper container, created and then removed.
    assert raw_daemon.helpers == ["created", "removed"]


def test_a_container_that_changed_nothing_is_checkpointed_and_restored_to(raw_daemon, tmp_path):
    """Found against a real daemon: it lists no changes as null."""
    filesystem = Filesystem(IMAGE)
    restoring(raw_daemon, filesystem)
    executor = backend(raw_daemon)
    manifest = executor.checkpoint(tmp_path / "checkpoints")
    assert manifest.changes == () and manifest.copied == () and manifest.deleted == ()
    filesystem.files.update({"/app/new.py": b"N = 1\n", "/etc/app.conf": b"x = 9\n"})
    del filesystem.files["/usr/lib/lib.so"]
    receipt = executor.restore(manifest, tmp_path / "checkpoints")
    assert filesystem.files == IMAGE and receipt.exact
    assert receipt.removed == ("/app/new.py",) and receipt.from_image == ("/etc/app.conf", "/usr/lib/lib.so")


def test_a_partial_checkpoint_is_refused_before_anything_changes(raw_daemon, tmp_path):
    """A partial checkpoint lacks paths it would delete: found by review, seen in the pilot."""
    filesystem = Filesystem(IMAGE)
    restoring(raw_daemon, filesystem)
    filesystem.files.update({"/app/util.py": b"U = 1\n", "/app/vendor": None, "/app/vendor/big.bin": b"x" * 64})
    executor = backend(raw_daemon)
    manifest = executor.checkpoint(tmp_path / "checkpoints", cap_bytes=1)
    assert manifest.partial and manifest.over_cap
    filesystem.files["/app/util.py"] = b"broken"
    before = dict(filesystem.files)
    with pytest.raises(TerminalConflict, match="partial"):
        executor.restore(manifest, tmp_path / "checkpoints")
    assert filesystem.files == before and raw_daemon.helpers == []


def test_a_restore_without_the_time_its_size_needs_changes_nothing(raw_daemon, tmp_path):
    filesystem = Filesystem(IMAGE)
    restoring(raw_daemon, filesystem)
    filesystem.files["/app/util.py"] = b"U = 1\n"
    executor = DockerTerminalBackend.admit(raw_daemon.socket, CONTAINER, TOKEN, time.time() + 8)
    manifest = executor.checkpoint(tmp_path / "checkpoints")
    filesystem.files.update({"/app/build": None, "/app/build/out.o": b"obj"})
    before = dict(filesystem.files)
    with pytest.raises(TerminalConflict, match="too little time"):
        executor.restore(manifest, tmp_path / "checkpoints")
    assert filesystem.files == before and raw_daemon.helpers == []


def test_a_restore_refuses_a_checkpoint_of_another_image(raw_daemon, tmp_path):
    filesystem = Filesystem(IMAGE)
    restoring(raw_daemon, filesystem)
    executor = backend(raw_daemon)
    manifest = executor.checkpoint(tmp_path / "checkpoints")
    raw_daemon.info["Image"] = "sha256:" + "1" * 64
    with pytest.raises(TerminalConflict, match="another image"):
        executor.restore(manifest, tmp_path / "checkpoints")


def test_a_checkpoint_whose_tar_changed_is_not_restored(raw_daemon, tmp_path):
    filesystem = Filesystem(IMAGE)
    restoring(raw_daemon, filesystem)
    filesystem.files["/app/util.py"] = b"U = 1\n"
    executor = backend(raw_daemon)
    manifest = executor.checkpoint(tmp_path / "checkpoints")
    (tmp_path / "checkpoints" / manifest.tar).write_bytes(b"tampered")
    with pytest.raises(TerminalConflict, match="checksum"):
        executor.restore(manifest, tmp_path / "checkpoints")
