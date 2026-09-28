"""A best-effort collector cannot turn incomplete artifact copies into grading."""
import asyncio
import json
import os
import tarfile
import threading
import time

import pytest

from taste.benchmarks.artifact_handoff import ArtifactHandoff, ArtifactTarget
from taste.benchmarks.output_snapshot import OutputSnapshotError, SnapshotLimits
from tests.test_docker_terminal import backend
from tests.test_docker_terminal import daemon as docker_daemon
from tests.test_output_snapshot import archive, file_archive, reward, serve_archive
from tests.test_thread_shutdown_ownership import cancel_loop_tasks

daemon = docker_daemon


@pytest.fixture
def collection(daemon, tmp_path):
    target = tmp_path / "file"
    targets = [ArtifactTarget("/tmp/result", target, "file")]
    item = ArtifactHandoff(backend(daemon), targets)
    serve_archive(daemon, file_archive(("result", b"correct", tarfile.REGTYPE)), source="/tmp/result")
    return item, target


async def collect(item, target):
    return await item.collect("/tmp/result", target, "file", deadline_unix=time.time() + 5)


def test_http_capture_seals_original_bytes_and_returned_receipts_cannot_change_authority(collection):
    item, target = collection
    collected = asyncio.run(collect(item, target))
    collected["entries"].clear()
    collected["binding"].clear()
    receipts = item.seal()
    assert receipts[0]["source"] == "/tmp/result" and receipts[0]["bytes"] == 7
    assert receipts[0]["binding"]["container_id"]
    assert target.read_bytes() == b"correct"
    receipts[0]["entries"].clear()
    assert len(item.verify()[0]["entries"]) == 1
    assert item.phase == "sealed"


def test_directory_snapshot_detects_new_files_before_upload(collection, daemon):
    _, target = collection
    target.mkdir(mode=0o700)
    item = ArtifactHandoff(backend(daemon), [ArtifactTarget("/logs/verifier", target, "directory")])
    serve_archive(daemon, archive(reward(), ("verifier/empty", b"", tarfile.DIRTYPE)))
    asyncio.run(item.collect("/logs/verifier", target, "directory", deadline_unix=time.time() + 5))
    assert len(item.seal()[0]["entries"]) == 3
    (target / "injected").write_bytes(b"after collection")
    (target / "injected").chmod(0o600)
    with pytest.raises(OutputSnapshotError, match="changed"):
        item.verify()


@pytest.mark.parametrize("damage", [None, "failed", "skipped", "destination", "source", "duplicate", "missing", "filter"])
def test_harbor_report_cannot_mask_a_failure_after_successful_capture(collection, damage):
    item, target = collection
    asyncio.run(collect(item, target))
    row = {"source": "/tmp/result", "destination": "artifacts/file", "type": "file",
           "status": "ok", "service": None, "exclude": []}
    rows = [row]
    if damage in {"failed", "skipped"}:
        row["status"] = damage
    elif damage in {"destination", "source"}:
        row[damage] = "/other"
    elif damage == "duplicate":
        rows.append(row)
    elif damage == "missing":
        rows = []
    elif damage == "filter":
        row["exclude"] = ["*"]
    if damage is None:
        item.seal_harbor_manifest(json.dumps(rows).encode(), target.parent)
        assert item.phase == "sealed"
    else:
        with pytest.raises(OutputSnapshotError):
            item.seal_harbor_manifest(json.dumps(rows).encode(), target.parent)
        assert item.phase == "failed"


@pytest.mark.parametrize("damage", ["content", "delete", "link", "mode", "hardlink"])
def test_changed_artifact_never_reaches_verifier(collection, damage):
    item, target = collection
    asyncio.run(collect(item, target))
    item.seal()
    if damage == "content":
        target.write_bytes(b"changed")
    elif damage == "delete":
        target.unlink()
    elif damage == "link":
        target.unlink()
        target.symlink_to("/etc/passwd")
    elif damage == "mode":
        target.chmod(0o644)
    else:
        os.link(target, target.parent / "alias")
    with pytest.raises((OSError, OutputSnapshotError)):
        item.verify()
    assert item.phase == "failed"


def test_upstream_swallowing_a_copy_failure_cannot_seal_or_retry(collection, daemon):
    item, target = collection
    serve_archive(daemon, file_archive(("result", b"outside", tarfile.SYMTYPE)), source="/tmp/result")
    with pytest.raises(OutputSnapshotError):
        asyncio.run(collect(item, target))
    # Harbor may catch the error and continue. Our independent gate cannot.
    with pytest.raises(OutputSnapshotError, match="did not complete"):
        item.seal()
    calls = len(daemon.calls)
    with pytest.raises(OutputSnapshotError):
        asyncio.run(collect(item, target))
    assert len(daemon.calls) == calls and not target.exists()


def test_omitted_copy_or_wrong_destination_is_not_an_empty_artifact(collection, daemon):
    item, target = collection
    calls = len(daemon.calls)
    with pytest.raises(OutputSnapshotError):
        asyncio.run(item.collect("/tmp/result", target.parent / "other", "file", deadline_unix=time.time() + 5))
    with pytest.raises(OutputSnapshotError):
        item.seal()
    assert len(daemon.calls) == calls


def test_missing_collection_entry_blocks_grading_even_if_other_entry_succeeded(collection, daemon):
    _, target = collection
    item = ArtifactHandoff(backend(daemon), [ArtifactTarget("/tmp/result", target, "file"),
        ArtifactTarget("/tmp/missing", target.parent / "missing", "file")])
    asyncio.run(collect(item, target))
    with pytest.raises(OutputSnapshotError, match="did not complete"):
        item.seal()


def test_aggregate_budget_applies_across_individually_bounded_files(collection, daemon):
    _, target = collection
    second = target.parent / "second"
    item = ArtifactHandoff(backend(daemon), [ArtifactTarget("/tmp/result", target, "file"),
        ArtifactTarget("/tmp/second", second, "file")], limits=SnapshotLimits(total_bytes=8))
    asyncio.run(collect(item, target))
    serve_archive(daemon, file_archive(("second", b"xx", tarfile.REGTYPE)), source="/tmp/second")
    with pytest.raises(OutputSnapshotError, match="aggregate"):
        asyncio.run(item.collect("/tmp/second", second, "file", deadline_unix=time.time() + 5))
    assert item.phase == "failed"
    with pytest.raises(OutputSnapshotError):
        item.seal()


@pytest.mark.parametrize("whole_loop", [False, True])
def test_cancelled_collection_waits_for_host_writer_and_remains_failed(collection, monkeypatch, whole_loop):
    item, target = collection
    entered, release = threading.Event(), threading.Event()

    def blocked(*_args, **_kwargs):
        entered.set()
        assert release.wait(5)
        target.write_bytes(b"late")
        target.chmod(0o600)
        return {"files": 1, "bytes": 4, "archive_bytes": 10240}
    monkeypatch.setattr("taste.benchmarks.artifact_handoff.download_file_snapshot", blocked)

    async def scenario():
        task = asyncio.create_task(collect(item, target))
        while not entered.is_set():
            await asyncio.sleep(0.001)
        if whole_loop:
            cancel_loop_tasks()
        else:
            task.cancel()
        try:
            await asyncio.sleep(0.02)
            assert not task.done() and not target.exists()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert target.read_bytes() == b"late" and item.phase == "failed"
        with pytest.raises(OutputSnapshotError):
            item.seal()
    asyncio.run(scenario())


@pytest.mark.parametrize("overlap", ["same", "child", "source"])
def test_overlapping_contract_is_rejected_before_collection(collection, daemon, overlap):
    _, target = collection
    second = target if overlap == "same" else target / "child" if overlap == "child" else target.parent / "other"
    with pytest.raises(ValueError, match=r"overlap|unique"):
        ArtifactHandoff(backend(daemon), [ArtifactTarget("/tmp/result", target, "file"),
            ArtifactTarget("/tmp/result" if overlap == "source" else "/tmp/other", second, "file")])
