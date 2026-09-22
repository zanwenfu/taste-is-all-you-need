"""Cleanup must preserve files even when Git would ignore or omit them."""

from __future__ import annotations

import os
from contextlib import closing
from pathlib import Path

import pytest
from git import Repo

from taste.memstore import Store


@pytest.mark.parametrize("ignore_source", ["gitignore", "info-exclude"])
def test_ignored_output_is_visible_checkpointed_and_recoverable_after_removal(
    tmp_path: Path, ignore_source: str,
) -> None:
    root = tmp_path / "repo"
    with closing(Store.open(root, "s")) as store:
        branch = store.branch("worker")
        if ignore_source == "gitignore":
            branch.write(".gitignore", "build/\n")
        else:
            store.backend.write_excludes(["build/"])
        branch.checkpoint("before output")
        branch.write("build/result.bin", b"\0precious\xff")
        assert "build/result.bin" in branch.dirty_paths()
        assert "build/result.bin" in branch.backend.untracked_paths()
        # A fresh supervisor process must see and capture the ignored output.
    with closing(Store.open(root, "s")) as supervisor:
        supervisor.remove_branch("worker")
        state = supervisor.view("worker").head
        assert state.read_bytes("build/result.bin") == b"\0precious\xff"
        assert not supervisor.worktree_path_for("worker").exists()
        recreated = supervisor.branch("worker")
        assert recreated.path("build/result.bin").read_bytes() == b"\0precious\xff"
        assert not recreated.is_dirty()


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("operation", ["checkpoint", "remove", "rollback"])
def test_nested_repository_is_rejected_without_discarding_either_copy_of_work(
    tmp_path: Path, committed: bool, operation: str,
) -> None:
    with closing(Store.open(tmp_path / "repo", "s")) as store:
        branch = store.branch("worker")
        original = branch.head
        branch.write("solution.txt", "ordinary work must survive")
        nested = branch.path("project")
        nested.mkdir()
        with Repo.init(nested) as inner:
            with inner.config_writer() as config:
                config.set_value("user", "name", "capture-test")
                config.set_value("user", "email", "capture-test@example.invalid")
            (nested / "result.txt").write_text("nested work must survive")
            if committed:
                inner.git.add("--all")
                inner.git.commit("-m", "nested history must survive")
        with pytest.raises(RuntimeError, match="nested repository"):
            if operation == "checkpoint":
                branch.checkpoint("capture everything")
            elif operation == "remove":
                store.remove_branch("worker")
            else:
                branch.rollback(original, "return to start")
        assert branch.head == original
        assert branch.path("solution.txt").read_text() == "ordinary work must survive"
        assert (nested / "result.txt").read_text() == "nested work must survive"
        with Repo(nested) as preserved:
            assert preserved.head.is_valid() == committed


def test_ignored_nested_repository_cannot_be_deleted_as_a_clean_worktree(tmp_path: Path) -> None:
    with closing(Store.open(tmp_path / "repo", "s")) as store:
        branch = store.branch("worker")
        branch.write(".gitignore", "project/\n")
        branch.checkpoint("ignore project")
        nested = branch.path("project")
        with Repo.init(nested):
            (nested / "result.txt").write_text("only copy")
        with pytest.raises(RuntimeError, match="nested repository"):
            store.remove_branch("worker")
        assert (nested / "result.txt").read_text() == "only copy"


def test_special_file_blocks_capture_without_reading_or_removing_it(tmp_path: Path) -> None:
    with closing(Store.open(tmp_path / "repo", "s")) as store:
        branch = store.branch("worker")
        branch.write("valuable.txt", "keep")
        fifo = branch.path("pipe")
        os.mkfifo(fifo)
        with pytest.raises(RuntimeError, match="unsupported filesystem entry"):
            store.remove_branch("worker")
        assert fifo.exists()
        assert branch.path("valuable.txt").read_text() == "keep"


def test_capture_refuses_an_unreadable_directory_without_silently_skipping_it(tmp_path: Path) -> None:
    with closing(Store.open(tmp_path / "repo", "s")) as store:
        branch = store.branch("worker")
        branch.write("private/result.txt", "keep")
        private = branch.path("private")
        private.chmod(0o000)
        try:
            with pytest.raises(OSError):
                store.remove_branch("worker")
            assert branch.worktree.exists()
        finally:
            private.chmod(0o700)
        assert branch.path("private/result.txt").read_text() == "keep"


def test_index_gitlink_without_git_marker_cannot_omit_nested_work(tmp_path: Path) -> None:
    with closing(Store.open(tmp_path / "repo", "s")) as store:
        branch = store.branch("worker")
        original = branch.head
        branch.write("project/only.txt", "keep")
        branch.backend.repo.git.update_index(
            "--add", "--cacheinfo", f"160000,{original.id},project"
        )
        assert not branch.path("project/.git").exists()
        with pytest.raises(RuntimeError, match="nested repository"):
            store.remove_branch("worker")
        assert branch.head == original
        assert branch.path("project/only.txt").read_text() == "keep"


def test_nested_repository_created_during_staging_cannot_publish_gitlink(
    tmp_path: Path, monkeypatch,
) -> None:
    with closing(Store.open(tmp_path / "repo", "s")) as store:
        branch = store.branch("worker")
        original = branch.head
        branch.write("project/only.txt", "keep")
        real_check = branch.backend.check_capture_safe

        def create_after_scan():
            real_check()
            with Repo.init(branch.path("project")) as inner:
                inner.git.add("--all")
                inner.git.execute([
                    "git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "created after scan",
                ])

        monkeypatch.setattr(branch.backend, "check_capture_safe", create_after_scan)
        with pytest.raises(RuntimeError, match="nested repository"):
            branch.checkpoint("race with nested repository creation")
        assert branch.head == original
        assert branch.path("project/only.txt").read_text() == "keep"
