"""Imported Git commits are ancestry, but are not agent execution states."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest

from taste.memstore import NoSuchState, Store, Verdict
from taste.memstore.backend import GitBackend
from taste.memstore.objects import Manifest, Meta, now_iso
from taste.memstore.store import NOTES


def _ordinary_repository(root: Path) -> list[str]:
    commits = []
    with closing(GitBackend.init(root)) as backend:
        for index in range(3):
            (root / "unchanged.txt").write_text("imported data")
            (root / "changing.txt").write_text(str(index))
            backend.repo.git.add("--all")
            backend.repo.git.commit("-m", f"ordinary commit {index}")
            commits.append(backend.head_commit())
    return commits


def test_resume_verdicts_and_origins_stop_at_the_import_boundary(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    imported = _ordinary_repository(root)
    with closing(Store.open(root, "s")) as store:
        branch = store.branch("worker")
        assert not branch.resume().crashed
        assert branch.read("changing.txt") == "2"
        state_root = store.provenance(branch.head)[-1]
        assert state_root.meta.kind == "root"
        assert state_root.parents == []
        assert store.backend.is_ancestor(imported[0], branch.head.id)
        assert store.backend.head_commit() == imported[-1]
        assert all(store.backend.note_get(NOTES["meta"], sha) is None for sha in imported)
        assert store.origin(branch.head, "unchanged.txt") == state_root

        branch.turn(content="reasoned from imported source")
        good = branch.checkpoint("first agent state")
        store.judge(good, Verdict("fail", by="monitor", detail="needs a fix"))
        branch.write("changing.txt", "mistake")
        bad = branch.checkpoint("second agent state")
        branch.rollback(good, "restore first state")
        assert branch.resume().unacked[0].detail == "needs a fix"
        branch.acknowledge()
        assert not branch.resume().unacked
        assert branch.read("changing.txt") == "2"
        assert store.state(bad.id).read("changing.txt") == "mistake"
        assert branch.head.transcript.turns[0]["content"] == "reasoned from imported source"
    with closing(Store.open(root, "s")) as reopened:
        assert not reopened.branch("worker").resume().unacked
        assert reopened.origin(reopened.view("worker").head, "unchanged.txt").id == state_root.id


def test_sessions_import_the_same_git_tree_without_sharing_root_metadata(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _ordinary_repository(root)
    roots = []
    for session in ("first", "second"):
        with closing(Store.open(root, session)) as store:
            branch = store.branch("worker")
            state_root = store.provenance(branch.head)[-1]
            assert state_root.meta.session == session
            assert state_root.meta.branch == ""
            roots.append(state_root.id)
            assert not branch.resume().crashed
    assert roots[0] != roots[1]


def test_legacy_root_on_an_ordinary_commit_remains_resumable(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    imported = _ordinary_repository(root)
    # Exact previous layout: annotate the existing HEAD and leave its ordinary
    # ancestors unannotated. Upgrading must not rewrite existing states/refs.
    with closing(GitBackend(root)) as backend:
        meta = Meta(branch="", kind="root", reason="legacy root", producer="",
                    parents=(), created_at=now_iso(), session="legacy")
        backend.note_set(NOTES["meta"], imported[-1], meta.to_json())
        backend.note_set(NOTES["manifest"], imported[-1], Manifest().to_json())
        backend.note_set(NOTES["transcript"], imported[-1], "")
        assert backend.cas_update_ref("refs/taste/memstore/legacy/root", imported[-1], None)
    with closing(Store.open(root, "legacy")) as store:
        branch = store.branch("worker")
        assert not branch.resume().crashed
        assert store.provenance(branch.head)[-1].id == imported[-1]
        assert store.origin(branch.head, "unchanged.txt").id == imported[-1]


def test_missing_state_metadata_is_not_mistaken_for_imported_history(tmp_path: Path) -> None:
    with closing(Store.open(tmp_path / "repo", "s")) as store:
        branch = store.branch("worker")
        damaged = branch.checkpoint("metadata will be lost")
        branch.checkpoint("later state")
        store.backend.repo.git.notes("--ref", NOTES["meta"], "remove", damaged.id)
        with pytest.raises(NoSuchState):
            branch.resume()


def test_same_named_branch_in_another_session_is_a_history_boundary(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    with closing(Store.open(root, "source")) as source:
        worker = source.branch("worker")
        worker.turn(content="source reasoning")
        state = worker.checkpoint("source work")
        with closing(Store.open(root, "destination")) as destination:
            adopted = destination.branch("worker", from_state=state)
            assert len(adopted.history()) == 1
            assert adopted.history()[0].meta.session == "destination"
            assert state.id in [item.id for item in destination.provenance(adopted.head)]
            assert adopted.head.transcript.turns[0]["content"] == "source reasoning"
