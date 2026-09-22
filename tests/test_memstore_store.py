"""The memory layer's invariants, each one a test.

Nothing is lost. A checkpoint is atomic. The store is the truth. If any of
these stop holding, everything built above the layer is standing on sand.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from taste.memstore import (
    Branch,
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
    shutil.rmtree(root / ".git" / "memstore-sidecars")

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


def _git(worktree, *args: str) -> None:
    import subprocess

    subprocess.run(
        ["git", "-c", "user.name=w", "-c", "user.email=w@w.local", *args],
        cwd=worktree,
        check=True,
        capture_output=True,
    )


def test_every_way_a_worker_can_take_its_branch_is_refused(store: Store) -> None:
    """Four shapes, and the first version of this guard caught one of them.

    ``ForeignHead`` originally asked one question -- does the head carry state
    metadata -- which names ``git commit``, ``--amend`` and ``revert``, because
    all three land on a commit this layer never made.

    ``checkout --detach`` never moves the ref at all: the tree leaves the
    branch, and the next checkpoint stages that tree and moves the ref onto it,
    publishing another commit's content as this branch's work. That one is
    caught by the worktree-attachment check.

    **``reset --hard`` is deliberately absent and still undetected.** It lands
    on a commit this layer did create, so the metadata question answers
    "healthy" while the branch has silently forked backwards. A forward-only
    check on the ref catches it -- and also refuses the coordinator's own
    recovery from a rewound control branch, which
    ``test_raw_control_ref_rewind_cannot_erase_cost_or_repeat_provider_call``
    and its siblings exist to prove works. Detection has to be something a
    repair path can opt out of, and that is a contract decision this test will
    be extended to cover once it is made.
    """
    for shape, mutate in (
        ("commit", lambda b, first: (b.write("extra.py", "x\n"),
                                     _git(b.worktree, "add", "-A"),
                                     _git(b.worktree, "commit", "-qm", "mine"))),
        ("amend", lambda b, first: _git(b.worktree, "commit", "--amend", "-qm", "amended")),
        ("revert", lambda b, first: _git(b.worktree, "revert", "--no-edit", "HEAD")),
        ("detach", lambda b, first: _git(b.worktree, "checkout", "--detach", first.id)),
    ):
        branch = store.branch(f"w-{shape}")
        branch.write("f", "1")
        first = branch.checkpoint("first")
        branch.write("f", "2")
        branch.checkpoint("second")

        mutate(branch, first)

        with pytest.raises(ForeignHead):
            branch.checkpoint(f"after {shape}")
        branch.close()


def test_the_guard_leaves_every_legitimate_move_alone(store: Store) -> None:
    """A false refusal here wedges a branch, so forward motion must be free.

    ``rollback`` is the one that looks like a rewind and is not: it *appends*,
    keeping the superseded head as its parent, so the previous value stays an
    ancestor of the new one. ``merge`` likewise moves forward.
    """
    a = store.branch("a")
    a.write("f", "1")
    first = a.checkpoint("one")
    a.write("f", "2")
    a.checkpoint("two")

    back = a.rollback(first, "undo two")
    assert back.meta.kind == "rollback"
    a.write("f", "3")
    assert a.checkpoint("after rollback").read("f") == "3"

    b = store.branch("b")
    b.write("g", "B")
    b.checkpoint("b work")
    merged = a.merge(b, reason="take b")
    assert not merged.conflicts
    a.write("f", "4")
    assert a.checkpoint("after merge").read("f") == "4"


# ------------------------------------------------------------ silent rewind
#
# `git reset --hard` lands on a commit memstore *did* create, carrying every
# note, so the metadata guard sees a healthy branch while two states have left
# it. Measured on a scratch repo: three checkpoints, a reset to the first, and
# the next checkpoint was accepted with states 2 and 3 simply gone. Delivery
# then projected an orphan into the shared integration branch, so integration
# advertised bytes the worker's own history no longer contained.
#
# "Nothing is lost" is this layer's first rule, so the gap matters more than
# its rarity. The layer could not see it because it kept no record of how far
# a branch had ever got: history() walks back from the current head, so after
# a rewind the short history *is* the whole history.


def test_a_silent_rewind_is_refused(store: Store) -> None:
    branch = store.branch("worker", producer="worker")
    ids = []
    for index in (1, 2, 3):
        branch.write("f.txt", f"{index}\n")
        ids.append(branch.checkpoint(f"state {index}").id)

    branch.backend.repo.git.reset("--hard", ids[0])

    branch.write("f.txt", "4\n")
    with pytest.raises(ForeignHead) as caught:
        branch.checkpoint("after the reset")
    message = str(caught.value)
    assert ids[2][:10] in message, "the refusal must name what went missing"
    assert "reset" in message.lower()
    branch.close()


def test_an_ordinary_advance_is_untouched(store: Store) -> None:
    branch = store.branch("worker", producer="worker")
    for index in range(4):
        branch.write("f.txt", f"{index}\n")
        branch.checkpoint(f"state {index}")
    assert len(branch.history()) == 5
    branch.close()


def test_a_rollback_is_not_a_rewind(store: Store) -> None:
    """Rollback appends: the superseded state stays reachable, so the tip
    never moves backwards and the guard must not fire."""
    branch = store.branch("worker", producer="worker")
    branch.write("f.txt", "one\n")
    first = branch.checkpoint("one")
    branch.write("f.txt", "two\n")
    branch.checkpoint("two")

    branch.rollback(first, "back to one")
    branch.write("f.txt", "three\n")
    branch.checkpoint("three")
    assert branch.head.read("f.txt") == "three\n"
    branch.close()


def test_a_branch_with_no_recorded_tip_still_works(tmp_path: Path) -> None:
    """A branch this layer did not record must keep working.

    Failing closed on a missing tip would make memstore unusable against any
    repository it did not create from scratch, which is a worse failure than
    the one being prevented.
    """
    store = Store.open(tmp_path / "repo", "no-tip")
    branch = store.branch("worker", producer="worker")
    branch.write("f.txt", "one\n")
    branch.checkpoint("one")
    store.sidecar("tip", "worker").unlink()

    branch.write("f.txt", "two\n")
    assert branch.checkpoint("two")
    branch.close()
    store.close()


def _rewound(store: Store, name: str = "control") -> tuple[Branch, list[str]]:
    """A branch whose ref was moved back behind what it published."""
    branch = store.branch(name, producer="coordinator")
    ids = []
    for index in (1, 2, 3):
        branch.write("f.txt", f"{index}\n")
        ids.append(branch.checkpoint(f"state {index}").id)
    store.backend.cas_update_ref(branch.ref, ids[0], ids[2])
    branch.backend.reset_hard_to_head()
    return branch, ids


def test_a_repair_may_accept_the_rewind_with_evidence(store: Store) -> None:
    """The coordinator repairs a branch it owns, so refusal alone is not
    enough -- there has to be a way through that records itself."""
    branch, ids = _rewound(store)
    with pytest.raises(ForeignHead):
        branch.checkpoint("blocked until accepted")

    accepted = branch.accept_rewind(
        evidence="planner-receipts:request-1",
        reason="control ref was rewound; replaying from the receipt journal",
    )
    assert accepted == ids[0]
    # And the branch works normally afterwards.
    branch.write("f.txt", "four\n")
    assert branch.checkpoint("four")
    branch.close()


def test_accepting_a_rewind_must_present_evidence(store: Store) -> None:
    branch, _ = _rewound(store)
    with pytest.raises(ValueError, match="evidence"):
        branch.accept_rewind(evidence="   ", reason="no evidence")
    with pytest.raises(ValueError, match="reason"):
        branch.accept_rewind(evidence="planner-receipts:r1", reason="  ")
    branch.close()


def test_an_accepted_rewind_records_what_stopped_being_claimed(store: Store) -> None:
    branch, ids = _rewound(store)
    branch.accept_rewind(
        evidence="planner-receipts:request-1",
        reason="replaying from the receipt journal",
    )
    entries = [
        json.loads(line)
        for line in store.sidecar("rewinds", "control").read_text().splitlines()
        if line.strip()
    ]
    assert len(entries) == 1
    recorded = entries[0]
    assert recorded["evidence"] == "planner-receipts:request-1"
    assert recorded["accepted_head"] == ids[0]
    assert set(recorded["orphaned"]) == {ids[1], ids[2]}
    # Nothing was deleted: the orphans are still in the object store.
    assert store.state(ids[2]).read("f.txt") == "3\n"
    branch.close()
