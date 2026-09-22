"""Interrupted and overlapping Git tree edits must not share index ownership."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest
from git import Git
from git.exc import GitCommandError

from taste.memstore import ObjectType, Store
from taste.memstore.backend import EMPTY_TREE, GitBackend


@pytest.fixture
def backend(tmp_path: Path):
    item = GitBackend.init(tmp_path / "repo")
    try:
        yield item
    finally:
        item.close()


def _index_env(git, kwargs):
    return {**os.environ, **git.environment(), **kwargs.get("env", {})}


def _kill_first_writer(monkeypatch, target):
    """Kill a real Git process after it acquires this operation's index lock."""
    execute = Git.execute
    killed = []

    def intercept(git, command, **kwargs):
        env = _index_env(git, kwargs)
        if git is target and "update-index" in command and not killed:
            index = Path(env["GIT_INDEX_FILE"])
            lock = Path(f"{index}.lock")
            # update-index holds its lock while waiting for --index-info stdin.
            # Keeping the pipe open gives us a deterministic kill point.
            proc = subprocess.Popen(
                ["git", "update-index", "--index-info"], cwd=git.working_dir,
                env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            try:
                deadline = time.monotonic() + 5
                while not lock.exists() and proc.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.005)
                assert lock.exists() and proc.poll() is None, "Git writer never acquired its lock"
                proc.kill()
                proc.communicate(timeout=5)
                assert proc.returncode == -9 and lock.exists(), "SIGKILL did not leave a Git lock"
                killed.append(index)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.communicate(timeout=5)
            raise GitCommandError(command, -9, stderr="controlled SIGKILL after index lock acquisition")
        return execute(git, command, **kwargs)

    monkeypatch.setattr(Git, "execute", intercept)
    return killed


def test_retry_after_real_git_writer_kill_preserves_the_default_index(backend, monkeypatch):
    (backend.path / "staged.txt").write_text("original stage\n")
    backend.stage_all()
    index = Path(backend.repo.index.path)
    before = index.read_bytes()
    blob = backend.hash_blob("replacement\n")
    killed = _kill_first_writer(monkeypatch, backend.repo.git)

    with pytest.raises(GitCommandError, match="controlled SIGKILL"):
        backend.tree_with_blob(EMPTY_TREE, "changed.txt", blob)
    assert len(killed) == 1
    result = backend.tree_with_blob(EMPTY_TREE, "changed.txt", blob)

    assert backend.blob_at(result, "changed.txt") == blob
    assert backend.blob_at(result, "staged.txt") is None
    assert index.read_bytes() == before
    assert not killed[0].exists() and not Path(f"{killed[0]}.lock").exists()


def test_typed_record_merge_can_retry_after_its_git_writer_is_killed(tmp_path, monkeypatch):
    with closing(Store.open(tmp_path / "repo", "scratch-retry")) as store:
        ours = store.branch("ours")
        ours.write("shared.json", "{}\n")
        ours.publish("shared", "shared.json", type=ObjectType.RECORD)
        base = ours.checkpoint("base")
        theirs = store.branch("theirs", from_state=base)
        ours.checkpoint("ours edits", records={"shared.json": {"ours": 1}})
        theirs.checkpoint("theirs edits", records={"shared.json": {"theirs": 2}})
        killed = _kill_first_writer(monkeypatch, ours.backend.repo.git)

        failed = ours.merge(theirs, reason="interrupted writer")
        assert not failed.ok and len(killed) == 1
        assert "controlled SIGKILL" in failed.conflicts[0].detail
        assert ours.head.record("shared.json") == {"ours": 1}
        assert theirs.head.record("shared.json") == {"theirs": 2}
        retried = ours.merge(theirs, reason="retry after writer exit")
        assert retried.ok and retried.state.record("shared.json") == {"ours": 1, "theirs": 2}


def test_existing_ambiguous_index_and_lock_are_neither_reused_nor_deleted(backend):
    previous = backend.common_dir / f"memstore.index.{os.getpid()}.{threading.get_ident()}"
    previous.write_bytes(b"unreconciled index evidence")
    lock = Path(f"{previous}.lock")
    lock.write_bytes(b"another writer may still own this")
    default_lock = Path(f"{backend.repo.index.path}.lock")
    default_lock.write_bytes(b"default index belongs to another operation")
    blob = backend.hash_blob("new operation")

    result = backend.tree_with_blob(EMPTY_TREE, "new.txt", blob)

    assert backend.blob_at(result, "new.txt") == blob
    assert previous.read_bytes() == b"unreconciled index evidence"
    assert lock.read_bytes() == b"another writer may still own this"
    assert default_lock.read_bytes() == b"default index belongs to another operation"


@pytest.mark.parametrize("failure", [OSError("write interrupted"), KeyboardInterrupt("stop")])
def test_failed_operation_cleans_its_private_index_and_lock(backend, monkeypatch, failure):
    execute = Git.execute
    touched = []

    def intercept(git, command, **kwargs):
        if git is backend.repo.git and "update-index" in command:
            index = Path(_index_env(git, kwargs)["GIT_INDEX_FILE"])
            Path(f"{index}.lock").write_bytes(b"interrupted operation")
            touched.append(index)
            raise failure
        return execute(git, command, **kwargs)

    monkeypatch.setattr(Git, "execute", intercept)
    with pytest.raises(type(failure)) as caught:
        backend.tree_with_blob(EMPTY_TREE, "f", backend.hash_blob("value"))
    assert caught.value is failure and len(touched) == 1
    assert not touched[0].exists() and not Path(f"{touched[0]}.lock").exists()


def test_overlapping_tree_edits_on_one_backend_keep_their_own_content(backend, monkeypatch):
    (backend.path / "staged.txt").write_text("default index\n")
    backend.stage_all()
    before = Path(backend.repo.index.path).read_bytes()
    original_env = dict(backend.repo.git.environment())
    blobs = [backend.hash_blob(f"content {i}") for i in range(2)]
    both_read = threading.Barrier(2)
    execute = Git.execute

    def intercept(git, command, **kwargs):
        result = execute(git, command, **kwargs)
        if git is backend.repo.git and "read-tree" in command:
            both_read.wait(timeout=5)
        return result

    monkeypatch.setattr(Git, "execute", intercept)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(backend.tree_with_blob, EMPTY_TREE, f"{i}.txt", blob)
                   for i, blob in enumerate(blobs)]
        trees = [future.result(timeout=10) for future in futures]

    for i, tree in enumerate(trees):
        assert backend.blob_at(tree, f"{i}.txt") == blobs[i]
        assert backend.blob_at(tree, f"{1-i}.txt") is None
        assert backend.blob_at(tree, "staged.txt") is None
    assert Path(backend.repo.index.path).read_bytes() == before
    assert backend.repo.git.environment() == original_env


def test_paused_tree_edit_does_not_redirect_another_default_index_read(backend, monkeypatch):
    (backend.path / "staged.txt").write_text("default index\n")
    backend.stage_all()
    expected = backend.write_tree()
    blob = backend.hash_blob("separate scratch tree")
    ready, release = threading.Event(), threading.Event()
    execute = Git.execute

    def intercept(git, command, **kwargs):
        result = execute(git, command, **kwargs)
        if git is backend.repo.git and "read-tree" in command:
            ready.set()
            assert release.wait(timeout=5), "reader did not release scratch writer"
        return result

    monkeypatch.setattr(Git, "execute", intercept)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(backend.tree_with_blob, EMPTY_TREE, "scratch.txt", blob)
        try:
            assert ready.wait(timeout=5), "scratch writer did not start"
            actual = backend.write_tree()
        finally:
            release.set()
        scratch = future.result(timeout=10)
    assert actual == expected
    assert backend.blob_at(scratch, "scratch.txt") == blob
