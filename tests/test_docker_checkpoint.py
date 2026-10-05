"""A checkpoint of what a task changed in its container, taken against a fake daemon."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
import urllib.parse

import pytest

from taste.brains.terminal_broker import TerminalFenced
from tests.test_docker_terminal import CONTAINER, backend
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
