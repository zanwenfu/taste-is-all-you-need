"""Host output admission, including a real Unix HTTP boundary with a fake daemon."""

import gzip
import io
import os
import tarfile
import time
from dataclasses import replace
from urllib.parse import quote

import pytest

from taste.benchmarks.output_snapshot import (
    OutputSnapshotError,
    SnapshotLimits,
    download_file_snapshot,
    download_snapshot,
    extract_file_snapshot,
    extract_snapshot,
)
from taste.brains.terminal_broker import TerminalConflict
from tests.test_docker_terminal import CONTAINER, backend
from tests.test_docker_terminal import daemon as docker_daemon

daemon = docker_daemon


def archive(*entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as output:
        root = tarfile.TarInfo("verifier")
        root.type = tarfile.DIRTYPE
        output.addfile(root)
        for name, content, kind in entries:
            item = tarfile.TarInfo(name)
            item.type, item.mode, item.uid, item.gid = kind, 0o4777, 1234, 5678
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                item.linkname = content.decode()
            else:
                item.size = len(content)
            output.addfile(item, io.BytesIO(content) if item.isreg() else None)
    return stream.getvalue()


def reward(data=b"1\n"):
    return ("verifier/reward.txt", data, tarfile.REGTYPE)


@pytest.fixture
def destination(tmp_path):
    path = tmp_path / "output"
    path.mkdir(mode=0o700)
    return path


def test_snapshot_preserves_bytes_only_and_creates_confined_directories(destination):
    data = bytes(range(256))
    payload = archive(reward(), ("verifier/nested/raw.bin", data, tarfile.REGTYPE))
    result = extract_snapshot(payload, destination, "verifier")
    assert result == {"files": 2, "bytes": 258, "archive_bytes": len(payload)}
    assert (destination / "reward.txt").read_bytes() == b"1\n"
    assert (destination / "nested/raw.bin").read_bytes() == data
    assert (destination / "nested").stat().st_mode & 0o7777 == 0o700
    info = (destination / "nested/raw.bin").stat()
    assert info.st_mode & 0o7777 == 0o600 and info.st_uid == os.geteuid() and info.st_nlink == 1


@pytest.mark.parametrize("name", ["/verifier/escape", "verifier/../escape", "verifier//escape",
    "verifier/./escape", "verifier/a\\b", "foreign/escape", "verifier/evil\npath",
    "verifier/" + "x/" * 16 + "deep"])
def test_path_escape_or_ambiguity_rejects_entire_snapshot(destination, name):
    with pytest.raises(OutputSnapshotError):
        extract_snapshot(archive(reward(), (name, b"bad", tarfile.REGTYPE)), destination, "verifier")
    assert not list(destination.iterdir())


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE,
                                 tarfile.CHRTYPE, tarfile.BLKTYPE, b"X"])
def test_links_and_special_files_never_reach_host(destination, kind):
    canary = destination.parent / "private"
    canary.write_bytes(b"controller-private")
    with pytest.raises(OutputSnapshotError):
        extract_snapshot(archive(reward(), ("verifier/escape", str(canary).encode(), kind)),
                         destination, "verifier")
    assert canary.read_bytes() == b"controller-private"
    assert not list(destination.iterdir())


@pytest.mark.parametrize("entries", [
    [reward(), reward(b"0")],
    [("verifier/a", b"file", tarfile.REGTYPE), ("verifier/a/child", b"child", tarfile.REGTYPE)],
    [("verifier/a/child", b"child", tarfile.REGTYPE), ("verifier/a", b"file", tarfile.REGTYPE)],
])
def test_duplicate_and_file_directory_collisions_are_rejected_before_writes(destination, entries):
    with pytest.raises(OutputSnapshotError):
        extract_snapshot(archive(*entries), destination, "verifier")
    assert not list(destination.iterdir())


@pytest.mark.parametrize("limits", [SnapshotLimits(archive_bytes=1024), SnapshotLimits(file_bytes=1),
    SnapshotLimits(total_bytes=1), SnapshotLimits(entries=1)])
def test_independent_byte_and_entry_limits(destination, limits):
    with pytest.raises(OutputSnapshotError):
        extract_snapshot(archive(reward()), destination, "verifier", limits=limits)
    assert not list(destination.iterdir())


@pytest.mark.parametrize("payload", [b"", b"garbage", archive(reward(b"x" * 1024))[:1600]])
def test_malformed_or_truncated_archive_has_no_partial_reward(destination, payload):
    with pytest.raises(OutputSnapshotError):
        extract_snapshot(payload, destination, "verifier")
    assert not list(destination.iterdir())


def test_compressed_and_sparse_archives_are_refused(destination):
    sparse = archive(("verifier/sparse", b"", tarfile.GNUTYPE_SPARSE))
    for payload in (gzip.compress(archive(reward())), sparse):
        with pytest.raises(OutputSnapshotError):
            extract_snapshot(payload, destination, "verifier")
        assert not list(destination.iterdir())


def test_empty_snapshot_and_stale_output_refusal(destination):
    extract_snapshot(archive(), destination, "verifier")
    extract_snapshot(archive(reward()), destination, "verifier")
    with pytest.raises(OutputSnapshotError, match="stale"):
        extract_snapshot(archive(), destination, "verifier")
    assert (destination / "reward.txt").read_bytes() == b"1\n"


@pytest.mark.parametrize("place", ["leaf", "ancestor"])
def test_host_symlink_destination_cannot_escape(destination, place):
    link = destination.parent / "link"
    link.symlink_to(destination if place == "leaf" else destination.parent, target_is_directory=True)
    target = link if place == "leaf" else link / destination.name
    with pytest.raises((OSError, OutputSnapshotError)):
        extract_snapshot(archive(reward()), target, "verifier")
    assert not list(destination.iterdir())


def test_host_writable_ancestor_and_nonprivate_destination_refused(destination):
    destination.chmod(0o755)
    with pytest.raises(OutputSnapshotError, match="private"):
        extract_snapshot(archive(reward()), destination, "verifier")
    destination.chmod(0o700)
    destination.parent.chmod(0o777)
    with pytest.raises(OutputSnapshotError, match="ancestor"):
        extract_snapshot(archive(reward()), destination, "verifier")


@pytest.mark.parametrize("field,value", [("file_bytes", True), ("archive_bytes", 65 * 1024 * 1024),
    ("total_bytes", 0), ("entries", 4097), ("depth", -1)])
def test_limits_cannot_be_disabled(field, value):
    with pytest.raises(ValueError):
        SnapshotLimits(**{field: value})


def serve_archive(daemon, payload, *, extra_length=0, chunked=False, after=None, source="/logs/verifier", container_id=CONTAINER):
    def handle(handler, path):
        if "/archive?" not in path:
            return False
        assert path == f"/containers/{container_id}/archive?path={quote(source, safe='')}"
        handler.send_response(200)
        handler.send_header("Content-Type", "application/x-tar")
        handler.send_header("Connection", "close")
        handler.send_header("Transfer-Encoding" if chunked else "Content-Length",
                            "chunked" if chunked else str(len(payload) + extra_length))
        handler.end_headers()
        if after:
            after()
        if chunked:
            for start in range(0, len(payload), 137):
                part = payload[start:start + 137]
                handler.wfile.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
            handler.wfile.write(b"0\r\n\r\n")
        else:
            handler.wfile.write(payload)
        return True
    daemon.override = handle


@pytest.mark.parametrize("chunked,stopped", [(False, False), (True, False), (False, True)])
def test_real_http_snapshot_uses_bound_identity_without_terminal_exec(daemon, destination, chunked, stopped):
    owner = backend(daemon)
    if stopped:
        daemon.info["State"]["Running"] = False
    serve_archive(daemon, archive(reward()), chunked=chunked)
    download_snapshot(owner, "/logs/verifier", destination, deadline_unix=time.time() + 5)
    assert (destination / "reward.txt").read_bytes() == b"1\n"
    assert all(method == "GET" for method, _, _ in daemon.calls)


@pytest.mark.parametrize("failure", ["truncated", "excess", "restarted"])
def test_http_failure_never_publishes_a_reward(daemon, destination, failure):
    owner = backend(daemon)
    limits = SnapshotLimits(archive_bytes=1024) if failure == "excess" else SnapshotLimits()
    def restart():
        daemon.info["State"]["StartedAt"] = "different"
    serve_archive(daemon, archive(reward()), extra_length=10 if failure == "truncated" else 0,
                  after=restart if failure == "restarted" else None)
    with pytest.raises((OutputSnapshotError, TerminalConflict)):
        download_snapshot(owner, "/logs/verifier", destination, deadline_unix=time.time() + 5, limits=limits)
    assert not list(destination.iterdir())


def test_original_deadline_is_not_extended_for_download(daemon, destination):
    owner = backend(daemon)
    owner.binding = replace(owner.binding, deadline_unix=time.time() - 1)
    calls = len(daemon.calls)
    with pytest.raises(TimeoutError):
        download_snapshot(owner, "/logs/verifier", destination, deadline_unix=time.time() + 5)
    assert len(daemon.calls) == calls and not list(destination.iterdir())


def file_archive(*entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as output:
        for name, data, kind in entries:
            item = tarfile.TarInfo(name)
            item.type, item.size, item.mode = kind, len(data), 0o4777
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                item.linkname, item.size = data.decode(), 0
            output.addfile(item, io.BytesIO(data) if item.isreg() else None)
    return stream.getvalue()


@pytest.mark.parametrize("data", [b"", bytes(range(256))])
def test_individual_file_snapshot_preserves_binary_bytes_without_metadata(destination, data):
    payload = file_archive(("result.bin", data, tarfile.REGTYPE))
    target = destination / "renamed.bin"
    summary = extract_file_snapshot(payload, target, "result.bin")
    assert summary == {"files": 1, "bytes": len(data), "archive_bytes": len(payload)}
    assert target.read_bytes() == data and target.stat().st_mode & 0o7777 == 0o600
    assert target.stat().st_uid == os.geteuid() and target.stat().st_nlink == 1


@pytest.mark.parametrize("entries", [[], [("result", b"", tarfile.DIRTYPE)],
    [("result", b"/private", tarfile.SYMTYPE)], [("result", b"/private", tarfile.LNKTYPE)],
    [("result", b"", tarfile.FIFOTYPE)], [("result/", b"1", tarfile.REGTYPE)],
    [("result/child", b"1", tarfile.REGTYPE)], [("foreign", b"1", tarfile.REGTYPE)],
    [("result", b"1", tarfile.REGTYPE), ("result", b"2", tarfile.REGTYPE)],
    [("result", b"1", tarfile.REGTYPE), ("result/child", b"2", tarfile.REGTYPE)],
])
def test_individual_file_rejects_wrong_shape_before_creating_any_output(destination, entries):
    with pytest.raises(OutputSnapshotError):
        extract_file_snapshot(file_archive(*entries), destination / "target", "result")
    assert not list(destination.iterdir())


@pytest.mark.parametrize("existing", ["file", "symlink", "directory"])
def test_individual_file_never_replaces_an_existing_host_target(destination, existing):
    target = destination / "target"
    canary = destination / "canary"
    canary.write_bytes(b"old")
    if existing == "file":
        target.write_bytes(b"old")
    elif existing == "symlink":
        target.symlink_to(canary)
    else:
        target.mkdir()
    with pytest.raises(FileExistsError):
        extract_file_snapshot(file_archive(("result", b"new", tarfile.REGTYPE)), target, "result")
    assert canary.read_bytes() == b"old"
    assert target.is_dir() if existing == "directory" else target.read_bytes() == b"old"


@pytest.mark.parametrize("damage", ["parent_link", "parent_public", "truncated", "oversize"])
def test_individual_file_failed_admission_has_no_host_output(destination, damage):
    payload = file_archive(("result", b"data", tarfile.REGTYPE))
    target, limits = destination / "target", SnapshotLimits()
    if damage == "parent_link":
        link = destination.parent / "link"
        link.symlink_to(destination, target_is_directory=True)
        target = link / "target"
    elif damage == "parent_public":
        destination.chmod(0o755)
    elif damage == "truncated":
        payload = payload[:514]
    else:
        limits = SnapshotLimits(file_bytes=2)
    with pytest.raises((OutputSnapshotError, OSError)):
        extract_file_snapshot(payload, target, "result", limits=limits)
    assert not list(destination.iterdir())


@pytest.mark.parametrize("failure", [None, "restarted", "truncated"])
def test_individual_file_crosses_real_docker_http_boundary(daemon, destination, failure):
    owner = backend(daemon)
    source = "/app/file with space.bin"
    payload = file_archive(("file with space.bin", bytes(range(256)), tarfile.REGTYPE))

    def restart():
        daemon.info["State"]["StartedAt"] = "replacement"
    serve_archive(daemon, payload, source=source, after=restart if failure == "restarted" else None,
                  extra_length=10 if failure == "truncated" else 0)
    target = destination / "target"
    if failure:
        with pytest.raises((OutputSnapshotError, TerminalConflict)):
            download_file_snapshot(owner, source, target, deadline_unix=time.time() + 5)
        assert not target.exists()
    else:
        download_file_snapshot(owner, source, target, deadline_unix=time.time() + 5)
        assert target.read_bytes() == bytes(range(256))
    assert all(method == "GET" for method, _, _ in daemon.calls)
