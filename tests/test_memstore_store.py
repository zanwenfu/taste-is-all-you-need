"""The memory layer's invariants, each one a test.

Nothing is lost. A checkpoint is atomic. The store is the truth. If any of
these stop holding, everything built above the layer is standing on sand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taste.memstore import (
    BranchBusy,
    ForeignHead,
    NotAnAncestor,
    ObjectType,
    PublishError,
    StaleBranch,
    Store,
    Transcript,
    Verdict,
)
from taste.memstore.store import NOTES


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "repo", "s1")
    yield s
    s.close()


# ------------------------------------------------------------------ opening


def test_open_creates_repo_and_seeds_branch(store: Store) -> None:
    a = store.branch("a", producer="brain-a")
    assert (a.worktree / ".git").exists()
    head = a.head
    assert head.meta.kind == "branch"
    assert head.meta.producer == "brain-a"
    assert head.manifest.entries == {}
    assert len(head.transcript) == 0
    assert store.branches() == ["a"]


def test_reopen_is_idempotent(tmp_path: Path) -> None:
    s1 = Store.open(tmp_path / "repo", "s1")
    a = s1.branch("a")
    a.write("f.txt", "one\n")
    first = a.checkpoint("first")
    s1.close()
    s2 = Store.open(tmp_path / "repo", "s1")
    assert s2.branch("a").head == first
    s2.close()


# ------------------------------------------------------------------ checkpoint


def test_checkpoint_holds_artifacts_transcript_and_reason(store: Store) -> None:
    a = store.branch("a")
    a.write("src/x.py", "x = 1\n")
    t = Transcript().append(role="user", content="write x").append(role="assistant", content="done")
    st = a.checkpoint("wrote x", transcript=t, verdict=Verdict("pass", by="monitor"))
    assert st.read("src/x.py") == "x = 1\n"
    assert st.meta.reason == "wrote x"
    assert st.meta.kind == "checkpoint"
    assert st.meta.verdict is not None and st.meta.verdict.status == "pass"
    assert st.transcript.turns[1]["content"] == "done"
    assert st.parents[0].meta.kind == "branch"
    assert a.head == st


def test_transcript_carries_forward_when_not_given(store: Store) -> None:
    a = store.branch("a")
    a.write("f", "1")
    s1 = a.checkpoint("one", transcript=Transcript().append(role="user", content="q"))
    a.write("f", "2")
    s2 = a.checkpoint("two")
    assert s2.transcript == s1.transcript
    assert s2.read("f") == "2"


def test_checkpoint_with_no_changes_is_still_a_state(store: Store) -> None:
    a = store.branch("a")
    a.write("f", "1")
    s1 = a.checkpoint("one")
    s2 = a.checkpoint("thought about it", transcript=Transcript().append(role="assistant", content="hm"))
    assert s2 != s1
    assert s2.parents == [s1]
    assert len(s2.transcript) == 1


def test_records_are_typed_files(store: Store) -> None:
    a = store.branch("a")
    a.publish("revenue", "out/revenue.json", type=ObjectType.RECORD, description="Q3 revenue summary")
    st = a.checkpoint("computed revenue", records={"out/revenue.json": {"q3": 100, "unit": "usd"}})
    assert st.record("out/revenue.json") == {"q3": 100, "unit": "usd"}
    entry = st.manifest.entries["revenue"]
    assert entry.type is ObjectType.RECORD
    assert entry.blob == st.blob("out/revenue.json")
    assert entry.published_at is not None


def test_publish_requires_the_path_to_exist(store: Store) -> None:
    a = store.branch("a")
    a.publish("ghost", "nope.txt")
    with pytest.raises(PublishError):
        a.checkpoint("oops")
    # Nothing was published; the branch did not move.
    assert a.head.meta.kind == "branch"


# ------------------------------------------------------------------ nothing is lost


def test_rollback_is_an_append_and_the_failed_state_survives(store: Store) -> None:
    a = store.branch("a")
    a.write("f", "good")
    good = a.checkpoint("good", transcript=Transcript().append(role="assistant", content="good idea"))
    a.write("f", "bad")
    bad = a.checkpoint("bad", transcript=Transcript().append(role="assistant", content="bad idea"))

    back = a.rollback(good, "monitor failed the bad state")

    assert back.meta.kind == "rollback"
    assert back.meta.restores == good.id
    assert back.meta.rolled_back_from == bad.id
    assert back.parents == [bad]  # the failed state is the parent: reachable forever
    assert back.read("f") == "good"
    assert (a.worktree / "f").read_text() == "good"
    assert back.transcript == good.transcript  # the brain resumes with the context it had
    # and the failed context is one parent away, intact
    assert bad.transcript.turns[0]["content"] == "bad idea"
    assert [s.meta.reason for s in a.history()] == [
        "monitor failed the bad state",
        "bad",
        "good",
        "branch a from session root",
    ]


def test_rollback_captures_uncommitted_work_first(store: Store) -> None:
    a = store.branch("a")
    a.write("f", "good")
    good = a.checkpoint("good")
    a.write("f", "half-done edit never checkpointed")
    back = a.rollback(good, "abandon")
    capture = back.parents[0]
    assert capture.meta.reason.startswith("capture before rollback")
    assert capture.read("f") == "half-done edit never checkpointed"
    assert back.read("f") == "good"


def test_rollback_target_must_be_own_history(store: Store) -> None:
    a = store.branch("a")
    b = store.branch("b")
    b.write("g", "1")
    theirs = b.checkpoint("b's state")
    with pytest.raises(NotAnAncestor):
        a.rollback(theirs, "wrong branch")


def test_rollback_restores_manifest_of_target(store: Store) -> None:
    a = store.branch("a")
    a.write("out/a.json", "{}")
    a.publish("a", "out/a.json", type=ObjectType.RECORD)
    s1 = a.checkpoint("publish a")
    a.write("out/b.json", "{}")
    a.publish("b", "out/b.json", type=ObjectType.RECORD)
    a.checkpoint("publish b")
    back = a.rollback(s1, "drop b")
    assert set(back.manifest.entries) == {"a"}


# ------------------------------------------------------------------ atomicity


def test_a_branch_is_one_address_space(tmp_path: Path) -> None:
    s1 = Store.open(tmp_path / "repo", "s1")
    a1 = s1.branch("a")
    s2 = Store.open(tmp_path / "repo", "s1")
    with pytest.raises(BranchBusy):
        s2.branch("a")
    # A different branch of the same session is fine: concurrency is across branches.
    b2 = s2.branch("b")
    b2.write("g", "1")
    b2.checkpoint("b works while a is held")
    # Releasing the lease lets the next holder in.
    a1.close()
    a2 = s2.branch("a")
    assert a2.head.meta.kind == "branch"
    s1.close()
    s2.close()


def test_lost_compare_and_swap_publishes_nothing(store: Store) -> None:
    a = store.branch("a")
    a.write("f", "0")
    base = a.checkpoint("base")
    a.write("f", "1")
    winner = a.checkpoint("first writer")
    # A writer that still believes ``base`` is the head must lose, and lose cleanly:
    # this is the window a crashed lease-holder leaves behind.
    a.write("f", "2")
    built = a.build("stale writer")
    object.__setattr__(built, "expected_head", base.id)  # pretend it was built earlier
    with pytest.raises(StaleBranch):
        a.publish_state(built)
    assert a.head == winner
    assert a.head.read("f") == "1"


@pytest.mark.parametrize("fail_at", ["commit_tree", "note_meta", "note_manifest", "note_transcript", "cas"])
def test_crash_anywhere_in_the_sequence_leaves_the_store_consistent(
    store: Store, monkeypatch: pytest.MonkeyPatch, fail_at: str
) -> None:
    a = store.branch("a")
    a.write("f", "1")
    before = a.checkpoint("before")
    a.write("f", "2")

    backend = a.backend
    calls = {"note": 0}

    class Boom(RuntimeError):
        pass

    if fail_at == "commit_tree":
        def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise Boom

        monkeypatch.setattr(backend, "commit_tree", boom)
    elif fail_at.startswith("note_"):
        real_note = backend.note_set
        which = {"note_meta": 0, "note_manifest": 1, "note_transcript": 2}[fail_at]

        def boom_note(ns: str, commit: str, text: str, **kw: object) -> None:
            if calls["note"] == which:
                raise Boom
            calls["note"] += 1
            real_note(ns, commit, text, **kw)

        monkeypatch.setattr(backend, "note_set", boom_note)
    else:

        def boom_cas(ref: str, new: str, old: str | None) -> bool:
            raise Boom

        monkeypatch.setattr(backend, "cas_update_ref", boom_cas)

    with pytest.raises(Boom):
        a.checkpoint("after")

    # The branch did not move and its head is a complete state.
    monkeypatch.undo()
    assert a.head == before
    assert a.head.meta.reason == "before"
    assert a.head.manifest is not None and a.head.transcript is not None
    # And the layer is still fully usable afterwards.
    after = a.checkpoint("after, retried")
    assert after.read("f") == "2"
    assert after.parents == [before]


# ------------------------------------------------------------------ the store is the truth


def test_everything_is_reconstructible_from_refs_and_notes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    s = Store.open(root, "s1")
    a = s.branch("a", producer="A")
    a.write("f", "1")
    a.publish("f", "f", description="the f file")
    s1 = a.checkpoint("one", transcript=Transcript().append(role="user", content="hello"))
    a.write("f", "2")
    s2 = a.checkpoint("two")
    back = a.rollback(s1, "back")
    snapshot = {
        "head": back.id,
        "history": [x.id for x in a.history()],
        "manifest": back.manifest.to_json(),
        "transcript": back.transcript.to_jsonl(),
        "meta": back.meta.to_json(),
        "s2_read": s2.read("f"),
    }
    s.close()

    # Remove every non-git sidecar the wider harness might have left, and the lock.
    taste_dir = root / ".git" / "taste"
    if taste_dir.exists():
        for p in taste_dir.rglob("*"):
            if p.is_file():
                p.unlink()
    for lock in (root / ".git").glob("memstore.*"):
        lock.unlink()

    s = Store.open(root, "s1")
    a = s.branch("a")
    assert a.head.id == snapshot["head"]
    assert [x.id for x in a.history()] == snapshot["history"]
    assert a.head.manifest.to_json() == snapshot["manifest"]
    assert a.head.transcript.to_jsonl() == snapshot["transcript"]
    assert a.head.meta.to_json() == snapshot["meta"]
    assert s.state(s2.id).read("f") == snapshot["s2_read"]
    s.close()


def test_notes_live_under_their_namespaces(store: Store) -> None:
    a = store.branch("a")
    a.write("f", "1")
    st = a.checkpoint("one")
    for ns in ("meta", "manifest", "transcript"):
        assert store.backend.note_get(NOTES[ns], st.id) is not None


# ------------------------------------------------------------------ provenance and diff


def test_provenance_and_origin(store: Store) -> None:
    a = store.branch("a", producer="A")
    a.write("f", "v1")
    s1 = a.checkpoint("v1")
    a.write("g", "other")
    s2 = a.checkpoint("touch g")
    a.write("f", "v2")
    s3 = a.checkpoint("v2")
    chain = store.provenance(s3)
    assert [c.id for c in chain[:3]] == [s3.id, s2.id, s1.id]
    assert all(c.meta.producer == "A" for c in chain[:3])
    # f took its current form at s3; at s2 it was still the s1 version.
    assert store.origin(s3, "f") == s3
    assert store.origin(s2, "f") == s1
    assert store.origin(s3, "missing") is None


def test_typed_diff_reports_record_keys(store: Store) -> None:
    a = store.branch("a")
    a.publish("r", "r.json", type=ObjectType.RECORD)
    s1 = a.checkpoint("r1", records={"r.json": {"a": 1, "b": 2}})
    a.write("plain.txt", "x")
    s2 = a.checkpoint("r2", records={"r.json": {"a": 1, "b": 3, "c": 4}})
    d = s2.diff(s1)
    by_path = {e.path: e for e in d.entries}
    assert by_path["r.json"].type is ObjectType.RECORD
    assert json.loads(by_path["r.json"].detail) == {"added": ["c"], "removed": [], "changed": ["b"]}
    assert by_path["plain.txt"].type is ObjectType.FILE
    assert by_path["plain.txt"].status == "A"


# ------------------------------------------------------------------ branches


def test_branch_from_state_inherits_artifacts_manifest_and_transcript(store: Store) -> None:
    a = store.branch("a")
    a.write("shared.txt", "from a")
    a.publish("shared", "shared.txt")
    s = a.checkpoint("a's work", transcript=Transcript().append(role="assistant", content="ctx"))
    b = store.branch("b", from_state=s, producer="B")
    assert b.head.meta.kind == "branch"
    assert b.head.read("shared.txt") == "from a"
    assert "shared" in b.head.manifest
    assert b.head.transcript == s.transcript
    assert (b.worktree / "shared.txt").read_text() == "from a"


def test_branches_are_isolated_working_trees(store: Store) -> None:
    a = store.branch("a")
    b = store.branch("b")
    a.write("only-a.txt", "a")
    a.checkpoint("a writes")
    assert not (b.worktree / "only-a.txt").exists()
    assert b.head.read("only-a.txt") is None


# ------------------------------------------------- a head this layer did not make


def _commit_inside_the_worktree(worktree, message: str) -> str:
    """What a coding brain does without being told not to: commit its work."""
    import subprocess

    subprocess.run(["git", "add", "-A"], cwd=worktree, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=w", "-c", "user.email=w@w.local", "commit", "-qm", message],
        cwd=worktree,
        check=True,
        capture_output=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=worktree, capture_output=True, text=True
    ).stdout.strip()


def test_a_head_memstore_did_not_create_is_named_not_guessed_at(store: Store) -> None:
    """A worker's own ``git commit`` moves the branch out from under the layer.

    ``Bash`` is in the worker tool set and the worktree *is* a branch head, so
    this is an ordinary thing for a brain to do and fatal here. Measured live:
    5 of 7 runs corrupted their branch this way -- exactly the runs where the
    model got far enough to have something to commit.

    Before this, the next checkpoint died as ``NoSuchState`` from six frames
    deep in the transcript walk, naming a commit id and nothing else. An
    orchestrator cannot course-correct a condition that has no name.
    """
    a = store.branch("a")
    a.write("f", "1")
    last_good = a.checkpoint("the last state this layer built")

    a.write("add.py", "def add(x, y): return x + y\n")
    foreign = _commit_inside_the_worktree(a.worktree, "Add add.py with add(x, y)")
    assert foreign != last_good.id

    for attempt in (
        lambda: a.checkpoint("terminal work"),
        lambda: a.build("terminal work"),
        lambda: a.rollback(last_good, "undo"),
    ):
        with pytest.raises(ForeignHead, match=foreign[:10]):
            attempt()


def test_a_branch_with_a_foreign_head_can_still_be_read(store: Store) -> None:
    """Detection sits on the write paths for a reason.

    Whoever repairs this -- a monitor, a dashboard, the central brain deciding
    whether to re-plan -- has to be able to look at the branch first. A read
    that raised would turn one worker's corrupted branch into a crash for
    every observer of it.
    """
    a = store.branch("a")
    a.write("out/result.json", "{}")
    a.publish("result", "out/result.json", type=ObjectType.RECORD)
    last_good = a.checkpoint("the last state this layer built")

    a.write("add.py", "def add(x, y): return x + y\n")
    foreign = _commit_inside_the_worktree(a.worktree, "Add add.py with add(x, y)")

    view = store.view("a")
    assert view.exists()
    assert view.head.id == foreign
    assert store.heads()["a"].id == foreign
    # The catalog answers from the head's manifest note, which a foreign head
    # has none of: the branch's published work becomes invisible rather than
    # raising. That silence is why detection has to live on the write paths.
    assert store.catalog() == []
    # And what this layer did build is still exactly where it was.
    assert last_good.read("out/result.json") == "{}"
    assert set(last_good.manifest.entries) == {"result"}
