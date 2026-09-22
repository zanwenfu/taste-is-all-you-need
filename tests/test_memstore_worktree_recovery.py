"""Recovery from real stale/partial Git worktrees preserves the original files."""

from __future__ import annotations

import json
import shutil
from contextlib import closing
from pathlib import Path

import pytest
from git import Repo
from git.exc import GitCommandError

from taste.memstore import BadName, BranchBusy, Store


def _materialize(root: Path, instruction: str) -> None:
    root.mkdir(parents=True)
    with Repo.init(root) as repo:
        (root / "INSTRUCTION.md").write_text(instruction)
        repo.git.add("--all")
        repo.git.execute([
            "git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", "task environment",
        ])


def _preserved(store: Store, worker: str = "worker") -> Path:
    copies = store.worktree_recoveries(worker)
    assert len(copies) == 1
    metadata = json.loads((copies[0].parent / "recovery.json").read_text())
    assert metadata["source"] == str(store.worktree_path_for(worker))
    assert metadata["ref"] == store.ref_for(worker)
    return copies[0]


def test_rematerialized_repository_reopens_with_new_task_and_preserves_old_work(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _materialize(root, "first task")
    with closing(Store.open(root, "s")) as first:
        branch = first.branch("worker")
        branch.write("solution.txt", "old work, only copy")
        old_worktree = branch.worktree
    shutil.rmtree(root)
    _materialize(root, "second task")
    with closing(Store.open(root, "s")) as second:
        branch = second.branch("worker")
        assert branch.worktree == old_worktree
        assert branch.read("INSTRUCTION.md") == "second task"
        assert branch.path("INSTRUCTION.md").read_text() == "second task"
        assert not branch.path("solution.txt").exists()
        saved = _preserved(second)
        assert (saved / "INSTRUCTION.md").read_text() == "first task"
        assert (saved / "solution.txt").read_text() == "old work, only copy"


def test_partial_removal_restores_exact_head_and_keeps_uncommitted_remainder(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    with closing(Store.open(root, "s")) as store:
        branch = store.branch("worker")
        branch.write("result.txt", "committed")
        state = branch.checkpoint("durable result")
        branch.write("result.txt", "uncommitted, only copy")
        worktree = branch.worktree
    (worktree / ".git").unlink()
    with closing(Store.open(root, "s")) as reopened:
        branch = reopened.branch("worker")
        assert branch.head.id == state.id
        assert branch.path("result.txt").read_text() == "committed"
        saved = _preserved(reopened)
        assert (saved / "result.txt").read_text() == "uncommitted, only copy"
        branch.close()
        reopened.branch("worker").checkpoint("can continue")
        assert reopened.worktree_recoveries("worker") == (saved,)


def test_live_lease_prevents_repair_even_if_git_administration_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    with closing(Store.open(root, "s")) as owner, closing(Store.open(root, "s")) as other:
        branch = owner.branch("worker")
        branch.write("only.txt", "keep")
        (branch.worktree / ".git").unlink()
        with pytest.raises(BranchBusy):
            other.branch("worker")
        assert branch.path("only.txt").read_text() == "keep"
        branch.close()
        other.branch("worker")
        assert (_preserved(other) / "only.txt").read_text() == "keep"


def test_removal_holds_the_lease_through_git_cleanup(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    with closing(Store.open(root, "s")) as owner, closing(Store.open(root, "s")) as other:
        branch = owner.branch("worker")
        branch.write("only.txt", "keep")
        real_remove = owner.backend.worktree_remove

        def observe_lease(path: Path) -> None:
            with pytest.raises(BranchBusy):
                other.branch("worker")
            real_remove(path)

        monkeypatch.setattr(owner.backend, "worktree_remove", observe_lease)
        owner.remove_branch("worker")
        assert other.branch("worker").path("only.txt").read_text() == "keep"


def test_read_only_remainder_raises_on_removal_then_reopens_safely(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    try:
        with closing(Store.open(root, "s")) as store:
            branch = store.branch("worker")
            branch.write("gomodcache/pkg/dep.go", "package dep\n")
            branch.path("gomodcache/pkg").chmod(0o555)
            branch.path("gomodcache").chmod(0o555)
            worktree = branch.worktree
            with pytest.raises(GitCommandError):
                store.remove_branch("worker")
            assert worktree.exists()
            assert store.view("worker").head.read("gomodcache/pkg/dep.go") == "package dep\n"
        with closing(Store.open(root, "s")) as reopened:
            branch = reopened.branch("worker")
            assert branch.path("gomodcache/pkg/dep.go").read_text() == "package dep\n"
            assert (_preserved(reopened) / "gomodcache/pkg/dep.go").read_text() == "package dep\n"
    finally:
        for directory in tmp_path.rglob("*"):
            if directory.is_dir() and not directory.is_symlink():
                directory.chmod(0o700)


def test_failure_after_preserving_worktree_can_be_retried_without_losing_copy(
    tmp_path: Path, monkeypatch,
) -> None:
    with closing(Store.open(tmp_path / "repo", "s")) as store:
        branch = store.branch("worker")
        branch.write("only.txt", "keep")
        branch.close()
        (branch.worktree / ".git").unlink()
        with monkeypatch.context() as fault:
            def fail_add(*_args):
                raise OSError("simulated checkout failure")

            fault.setattr(store.backend, "worktree_add", fail_add)
            with pytest.raises(OSError, match="checkout failure"):
                store.branch("worker")
        saved = _preserved(store)
        assert (saved / "only.txt").read_text() == "keep"
        store.branch("worker")
        assert store.worktree_recoveries("worker") == (saved,)


def test_worktree_path_symlink_is_refused_without_touching_its_target(tmp_path: Path) -> None:
    with closing(Store.open(tmp_path / "repo", "s")) as store:
        branch = store.branch("worker")
        worktree = branch.worktree
        branch.close()
        preserved = tmp_path / "foreign"
        worktree.rename(preserved)
        worktree.symlink_to(preserved, target_is_directory=True)
        with pytest.raises(BadName, match="symlink"):
            store.branch("worker")
        assert worktree.is_symlink()
        assert (preserved / ".git").is_file()


@pytest.mark.parametrize("after_rename", [False, True])
def test_interruption_around_preservation_rename_retains_files_and_releases_lease(
    tmp_path: Path, monkeypatch, after_rename: bool,
) -> None:
    root = tmp_path / "repo"
    with closing(Store.open(root, "s")) as store:
        branch = store.branch("worker")
        branch.write("only.txt", "keep")
        worktree = branch.worktree
        branch.close()
        (worktree / ".git").unlink()
        real_rename = Path.rename

        def interrupt(source, destination):
            if source == worktree:
                if after_rename:
                    real_rename(source, destination)
                raise SystemExit("interrupted recovery")
            return real_rename(source, destination)

        with monkeypatch.context() as fault:
            fault.setattr(Path, "rename", interrupt)
            with pytest.raises(SystemExit, match="interrupted recovery"):
                store.branch("worker")
    with closing(Store.open(root, "s")) as reopened:
        reopened.branch("worker")
        assert (_preserved(reopened) / "only.txt").read_text() == "keep"


def test_strict_recovery_does_not_create_a_replacement_branch(tmp_path: Path) -> None:
    with closing(Store.open(tmp_path / "repo", "s")) as store:
        before = store.backend.for_each_ref("refs/")
        with pytest.raises(RuntimeError, match="Missing worker branch ref"):
            store.branch("worker", repair=False)
        assert store.backend.for_each_ref("refs/") == before
        assert not store.worktree_path_for("worker").exists()
