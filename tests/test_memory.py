"""Memory primitives.

These are the load-bearing operations the whole harness stands on: open a
session branch, checkpoint, rollback, demand-page a file from any ref, read
the log. Every other module assumes these work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taste.memory import Memory, _parse_commit_message


def _write(repo: Path, rel: str, content: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# ------------------------------------------------------------------ session


def test_open_session_creates_branch(tiny_repo: Path) -> None:
    memory = Memory.open_session(tiny_repo, "abc123")
    assert memory.branch == "taste/session-abc123"
    assert memory.repo.active_branch.name == "taste/session-abc123"


def test_open_session_is_idempotent(tiny_repo: Path) -> None:
    Memory.open_session(tiny_repo, "s1")
    # Re-opening the same session should resume on the same branch, not error.
    memory = Memory.open_session(tiny_repo, "s1")
    assert memory.branch == "taste/session-s1"


# ------------------------------------------------------------------ checkpoint


def test_checkpoint_commits_changes_and_tags_step_id(tiny_repo: Path) -> None:
    memory = Memory.open_session(tiny_repo, "cp")
    _write(tiny_repo, "hello.txt", "hello world\n")

    cp = memory.checkpoint("step-01", "add world")
    assert cp.step_id == "step-01"
    assert cp.message == "add world"
    assert cp.parent_sha is not None

    # The step_id survives through git log.
    log = memory.log(limit=1)
    assert log[0].step_id == "step-01"


def test_checkpoint_noop_when_no_changes(tiny_repo: Path) -> None:
    memory = Memory.open_session(tiny_repo, "noop")
    head_before = memory.head()
    cp = memory.checkpoint("step-x", "empty step")
    assert cp.sha == head_before.sha  # nothing committed
    assert cp.message.startswith("(no-op)")


# ------------------------------------------------------------------ rollback


def test_rollback_to_undoes_commits(tiny_repo: Path) -> None:
    memory = Memory.open_session(tiny_repo, "rb")
    before = memory.head()

    _write(tiny_repo, "a.txt", "a")
    memory.checkpoint("step-01", "add a")
    _write(tiny_repo, "b.txt", "b")
    memory.checkpoint("step-02", "add b")

    assert (tiny_repo / "a.txt").exists()
    assert (tiny_repo / "b.txt").exists()

    memory.rollback_to(before)

    assert memory.head().sha == before.sha
    assert not (tiny_repo / "a.txt").exists()
    assert not (tiny_repo / "b.txt").exists()


def test_rollback_last_moves_one_commit_back(tiny_repo: Path) -> None:
    memory = Memory.open_session(tiny_repo, "rbl")
    _write(tiny_repo, "a.txt", "a")
    cp1 = memory.checkpoint("step-01", "add a")
    _write(tiny_repo, "b.txt", "b")
    memory.checkpoint("step-02", "add b")

    new_head = memory.rollback_last()
    assert new_head.sha == cp1.sha
    assert (tiny_repo / "a.txt").exists()
    assert not (tiny_repo / "b.txt").exists()


def test_rollback_last_raises_without_history(tiny_repo: Path) -> None:
    memory = Memory.open_session(tiny_repo, "empty")
    with pytest.raises(RuntimeError):
        memory.rollback_last()


# ------------------------------------------------------------------ demand paging


def test_show_reads_file_at_ref(tiny_repo: Path) -> None:
    memory = Memory.open_session(tiny_repo, "paging")
    _write(tiny_repo, "note.txt", "alpha")
    cp1 = memory.checkpoint("step-01", "write alpha")
    _write(tiny_repo, "note.txt", "beta")
    memory.checkpoint("step-02", "overwrite with beta")

    # Demand-page the old version without touching the working tree.
    assert memory.show(cp1.sha, "note.txt").rstrip() == "alpha"
    assert (tiny_repo / "note.txt").read_text() == "beta"


def test_show_raises_for_missing_file(tiny_repo: Path) -> None:
    memory = Memory.open_session(tiny_repo, "missing")
    with pytest.raises(FileNotFoundError):
        memory.show("HEAD", "does/not/exist.txt")


# ------------------------------------------------------------------ trailer parsing


def test_parse_commit_message_extracts_step_id() -> None:
    raw = "add world\n\nTaste-Checkpoint: step-07"
    subject, step_id = _parse_commit_message(raw)
    assert subject == "add world"
    assert step_id == "step-07"


def test_parse_commit_message_without_trailer() -> None:
    subject, step_id = _parse_commit_message("initial")
    assert subject == "initial"
    assert step_id is None
