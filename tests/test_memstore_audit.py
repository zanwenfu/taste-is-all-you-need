"""One test per defect an independent audit found in the first version.

Every test here failed before its fix. They are kept together because they
are the record of what the layer got wrong, and because a regression in any
of them is a silent data-loss bug rather than a visible error.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest

from taste.memstore import (
    BadName,
    BranchBusy,
    ObjectType,
    SchemaMismatch,
    Store,
    Transcript,
    Verdict,
)
from taste.memstore.backend import MODE_EXEC, MODE_SYMLINK, GitBackend, NoteConflict
from taste.memstore.merge import merge_records
from taste.memstore.store import NOTES, TRANSCRIPT_SNAPSHOT_EVERY


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "repo", "s1")
    yield s
    s.close()


# ------------------------------------------------------------------ CRITICAL: wrong repository


def test_sibling_repositories_do_not_share_a_worktree(tmp_path: Path) -> None:
    """Two repos side by side once resolved to the same working tree, so one
    session committed into the other's object store."""
    a_root, b_root = tmp_path / "repoA", tmp_path / "repoB"
    sa, sb = Store.open(a_root, "s"), Store.open(b_root, "s")
    ba, bb = sa.branch("w"), sb.branch("w")

    assert ba.worktree != bb.worktree
    assert ba.backend.common_dir == sa.backend.common_dir
    assert bb.backend.common_dir == sb.backend.common_dir

    ba.write("mine.txt", "A")
    ba.checkpoint("A work")
    bb.write("mine.txt", "B")
    st = bb.checkpoint("B work")

    # Each store answers about itself, and B's commit is in B's object store.
    assert sb.state(st.id).read("mine.txt") == "B"
    assert sa.branch("w").head.read("mine.txt") == "A"
    assert sa.backend.ref_sha(sa.ref_for("w")) != sb.backend.ref_sha(sb.ref_for("w"))
    sa.close()
    sb.close()


def test_a_foreign_worktree_is_refused_not_adopted(tmp_path: Path) -> None:
    """If some other repo's checkout is sitting at our path, say so."""
    root, other = tmp_path / "repo", tmp_path / "other"
    s = Store.open(root, "s")
    GitBackend.init(other)
    (other / "f").write_text("x")
    other_repo = GitBackend(other)
    other_repo.stage_all()
    other_repo.repo.git.commit("-m", "seed")

    path = s.worktree_path_for("w")
    path.parent.mkdir(parents=True, exist_ok=True)
    other_repo.repo.git.worktree("add", str(path), "-b", "squatter")
    with pytest.raises(BadName, match="belongs to"):
        s.branch("w")
    s.close()
    other_repo.close()


# ------------------------------------------------------------------ CRITICAL: destroyed transcript


def test_two_sessions_in_one_repo_never_collide(tmp_path: Path) -> None:
    """Identical trees, parents and messages once minted identical commit ids,
    and `git notes add -f` then destroyed the first session's transcript."""
    root = tmp_path / "repo"
    s1, s2 = Store.open(root, "alpha"), Store.open(root, "beta")
    a, b = s1.branch("worker"), s2.branch("worker")
    a.write("out.txt", "same")
    b.write("out.txt", "same")
    x = a.checkpoint("do the thing", transcript=Transcript().append(who="alpha"))
    y = b.checkpoint("do the thing", transcript=Transcript().append(who="beta"))

    assert x.id != y.id
    assert s1.state(x.id).transcript.turns == ({"who": "alpha"},)
    assert s2.state(y.id).transcript.turns == ({"who": "beta"},)
    assert x.meta.session == "alpha" and y.meta.session == "beta"
    s1.close()
    s2.close()


def test_a_note_is_never_silently_overwritten(store: Store) -> None:
    """Defence in depth behind unique ids: a differing note raises."""
    a = store.branch("a")
    a.write("f", "1")
    st = a.checkpoint("one")
    store.backend.note_set(NOTES["meta"], st.id, st.meta.to_json(), overwrite=False)  # identical is fine
    with pytest.raises(NoteConflict):
        store.backend.note_set(NOTES["meta"], st.id, '{"schema": "taste.memstore/1"}', overwrite=False)


# ------------------------------------------------------------------ nothing is lost, everywhere


def test_merge_captures_uncommitted_work(store: Store) -> None:
    """Only rollback used to capture; merge reset --hard over the brain's work."""
    a, b = store.branch("a"), store.branch("b")
    b.write("theirs.txt", "b")
    b.checkpoint("b works")
    a.write("precious.txt", "never checkpointed")
    res = a.merge(b, reason="pull b")
    assert res.ok
    assert res.state.read("precious.txt") == "never checkpointed"
    assert any(st.read("precious.txt") == "never checkpointed" for st in a.history())


def test_removing_a_worktree_captures_uncommitted_work(store: Store) -> None:
    a = store.branch("a")
    a.write("precious.txt", "never checkpointed")
    store.remove_branch("a")
    assert store.view("a").head.read("precious.txt") == "never checkpointed"


def test_nothing_is_silently_excluded(store: Store) -> None:
    """The first version inherited the legacy harness's exclude list, so
    caches and bytecode were never checkpointed by a nothing-is-lost layer."""
    a = store.branch("a")
    a.write("__pycache__/x.cpython-312.pyc", "bytecode")
    a.write(".pytest_cache/v/results", "cache")
    st = a.checkpoint("everything")
    assert st.read("__pycache__/x.cpython-312.pyc") == "bytecode"
    assert st.read(".pytest_cache/v/results") == "cache"


def test_names_are_rejected_not_mangled(store: Store) -> None:
    """`worker/1` and `worker-1` once slugged to one address space."""
    ok = store.branch("worker-1")
    for bad in ("worker/1", "worker 1", "..", ".hidden", "a" * 200, "x.lock", ""):
        with pytest.raises(BadName):
            store.branch(bad)
    assert store.branches() == ["worker-1"]
    assert ok.name == "worker-1"


# ------------------------------------------------------------------ data fidelity


def test_binary_and_non_utf8_content_round_trips(store: Store) -> None:
    a = store.branch("a")
    raw = bytes(range(256))
    a.write("blob.bin", raw)
    st = a.checkpoint("binary")
    assert st.read_bytes("blob.bin") == raw


def test_non_utf8_merges_are_values_not_exceptions(store: Store) -> None:
    """A binary clash once raised UnicodeEncodeError out of the merge, which
    broke the promise that a conflict is a value the caller can act on."""
    a = store.branch("a")
    a.write("d.bin", bytes([0xFF, 0xFE]))
    a.write("f.bin", bytes([0xFF, 0xFE]))
    a.publish("d", "d.bin", type=ObjectType.NOTE)
    base = a.checkpoint("base")
    b = store.branch("b", from_state=base)
    a.write("d.bin", bytes([0xFF, 0x01]))
    a.write("f.bin", bytes([0xFF, 0x0A]))
    a.checkpoint("a edits")
    b.write("d.bin", bytes([0xFF, 0x03]))
    b.write("f.bin", bytes([0xFF, 0x0B]))
    b.checkpoint("b edits")

    res = a.merge(b, reason="binary")  # must not raise

    # The untyped binary file is a conflict the caller is handed, not a crash.
    assert res.state is None
    paths = {c.path for c in res.conflicts}
    assert "f.bin" in paths
    assert a.head.meta.kind == "conflict"


def test_symlinks_and_exec_bits_are_not_flattened(store: Store) -> None:
    """tree_with_blob hardcoded 100644, turning symlinks into files and
    dropping the executable bit."""
    a = store.branch("a")
    a.write("script.sh", "#!/bin/sh\necho one\n")
    os.chmod(a.path("script.sh"), 0o755)
    a.publish("s", "script.sh", type=ObjectType.NOTE)
    base = a.checkpoint("base")
    assert a.backend.mode_at(base.id, "script.sh") == MODE_EXEC

    b = store.branch("b", from_state=base)
    a.write("script.sh", "#!/bin/sh\necho a\n")
    os.chmod(a.path("script.sh"), 0o755)
    a.checkpoint("a edits")
    b.write("script.sh", "#!/bin/sh\necho b\n")
    os.chmod(b.path("script.sh"), 0o755)
    b.checkpoint("b edits")
    res = a.merge(b, reason="scripts")
    assert res.ok
    assert a.backend.mode_at(res.state.id, "script.sh") == MODE_EXEC

    # A symlink keeps its mode rather than being flattened into a file.
    (a.path("link")).symlink_to("one")
    a.publish("l", "link", type=ObjectType.NOTE)
    link_base = a.checkpoint("a links")
    assert a.backend.mode_at(link_base.id, "link") == MODE_SYMLINK


def test_record_merge_is_type_aware(store: Store) -> None:
    """`1 == True == 1.0` in Python once dropped a real change with no clash."""
    merged, clashes = merge_records({"k": 1}, {"k": True}, {"k": 2})
    assert clashes == ["k"]
    merged, clashes = merge_records({"k": 1}, {"k": 1.0}, {"k": 9})
    assert clashes == ["k"]
    # And nested objects are merged rather than compared whole.
    merged, clashes = merge_records(
        {"cfg": {"a": 1, "b": 2}}, {"cfg": {"a": 9, "b": 2}}, {"cfg": {"a": 1, "b": 8}}
    )
    assert clashes == [] and merged == {"cfg": {"a": 9, "b": 8}}


# ------------------------------------------------------------------ reading without the lease


def _holder(root: str, ready, release) -> None:  # type: ignore[no-untyped-def]
    s = Store.open(Path(root), "s1")
    b = s.branch("held")
    b.write("live.txt", "written while held")
    b.checkpoint("holder works")
    ready.set()
    release.wait(timeout=60)
    s.close()


def test_a_monitor_can_read_a_branch_that_a_brain_is_writing(tmp_path: Path) -> None:
    """Every named read path once took the WRITE lease, so a second brain
    could not read a working branch at all."""
    root = tmp_path / "repo"
    Store.open(root, "s1").close()
    ctx = mp.get_context("fork")
    ready, release = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_holder, args=(str(root), ready, release))
    holder.start()
    assert ready.wait(timeout=60)

    s = Store.open(root, "s1")
    view = s.view("held")
    assert view.head.read("live.txt") == "written while held"
    assert next(st.meta.reason for st in view.history()) == "holder works"
    assert "held" in s.heads()                      # heads() no longer opens branches
    assert s.catalog() is not None
    with pytest.raises(BranchBusy):
        s.branch("held")                            # writing is still exclusive

    # A monitor records its judgment without the lease.
    s.judge(view.head, Verdict("fail", by="monitor", detail="regression", failure_class="TESTS_FAIL"))
    assert [v.status for v in s.view("held").head.verdicts] == ["fail"]
    assert s.view("held").head.verdicts[0].at is not None

    release.set()
    holder.join(timeout=60)
    s.close()


def test_is_dirty_does_not_stage(store: Store) -> None:
    """The read-shaped predicate used to mutate the index."""
    a = store.branch("a")
    a.write("f", "1")
    a.checkpoint("one")
    a.write("untracked.txt", "x")
    before = a.backend.repo.git.ls_files("--stage")
    assert a.is_dirty() and a.dirty_paths() == ["untracked.txt"]
    assert a.backend.repo.git.ls_files("--stage") == before


# ------------------------------------------------------------------ transcripts at scale


def test_transcript_is_stored_as_a_delta(store: Store) -> None:
    """The whole context was re-serialized at every checkpoint: 600 turns
    cost 38 MB of loose objects."""
    a = store.branch("a")
    t = Transcript()
    for i in range(10):
        t = t.append(role="assistant", content=f"turn {i}")
        a.write("f", str(i))
        st = a.checkpoint(f"turn {i}", transcript=t)
    # Each note holds one turn, not the history.
    raw = store.backend.note_get(NOTES["transcript"], st.id)
    assert len(Transcript.from_jsonl(raw)) == 1
    assert st.meta.transcript_from is not None
    # And the full context still reads back exactly.
    assert st.transcript.turns == t.turns
    assert len(st.turns(-3)) == 3


def test_transcript_reconstructs_across_a_snapshot_boundary(store: Store) -> None:
    a = store.branch("a")
    n = TRANSCRIPT_SNAPSHOT_EVERY + 5
    t = Transcript()
    for i in range(n):
        t = t.append(i=i)
        a.write("f", str(i))
        st = a.checkpoint(f"t{i}", transcript=t)
    assert len(st.transcript) == n
    assert st.transcript.turns == t.turns
    depths = [s.meta.transcript_from for s in a.history()]
    assert any(d is None for d in depths)  # a snapshot was taken


def test_turn_appends_without_rewriting(store: Store) -> None:
    a = store.branch("a")
    a.turn(role="user", content="do it")
    a.turn(role="assistant", content="done")
    st = a.checkpoint("worked")
    assert len(st.transcript) == 2
    a.turn(role="user", content="again")
    st2 = a.checkpoint("worked again")
    assert len(st2.transcript) == 3
    assert len(Transcript.from_jsonl(store.backend.note_get(NOTES["transcript"], st2.id))) == 1


# ------------------------------------------------------------------ the communicator


def test_adopt_records_cross_branch_provenance(store: Store) -> None:
    """A took B's bytes and the resulting state recorded nothing about it."""
    b = store.branch("b", producer="brain-b")
    b.publish("revenue", "out/rev.json", type=ObjectType.RECORD, description="Q3 revenue")
    b_state = b.checkpoint("computed", records={"out/rev.json": {"emea": 1}})

    a = store.branch("a", producer="brain-a")
    (hit,) = store.search("revenue")
    a.adopt(hit, as_="incoming/rev.json")
    st = a.checkpoint("used b's numbers")

    assert st.record("incoming/rev.json") == {"emea": 1}
    (src,) = st.meta.sources
    assert src.branch == "b" and src.state == b_state.id and src.as_path == "incoming/rev.json"
    # provenance follows the data across the branch boundary
    assert store.origin(st, "incoming/rev.json") == b_state


def test_inbox_carries_a_request_between_brains(store: Store) -> None:
    """A cannot write in B's address space, so the ask must be shared state."""
    a, b = store.branch("a", producer="brain-a"), store.branch("b", producer="brain-b")
    mid = store.send("b", {"need": "revenue_summary"}, sender="brain-a")
    msgs = b.resume().inbox
    assert [m["body"]["need"] for m in msgs] == ["revenue_summary"]
    assert msgs[0]["sender"] == "brain-a" and msgs[0]["id"] == mid
    store.mark_inbox_seen("b", mid)
    assert b.resume().inbox == ()
    store.send("b", {"need": "something else"})
    assert len(b.resume().inbox) == 1
    assert a.resume().inbox == ()


def test_catalog_lets_a_brain_look_around_before_it_can_ask(store: Store) -> None:
    """Search only answers questions whose vocabulary the brain already has."""
    b = store.branch("b")
    b.publish("revenue", "r.json", type=ObjectType.RECORD, description="Q3 revenue")
    b.checkpoint("published", records={"r.json": {}})
    assert store.search("money") == []          # the wrong word finds nothing
    entries = [(h.branch, h.entry.name) for h in store.catalog()]
    assert entries == [("b", "revenue")]        # orientation needs no word at all


# ------------------------------------------------------------------ resume


def test_resume_tells_a_brain_what_it_was_doing(store: Store) -> None:
    """The one thing a failed turn lost was what it was trying to do."""
    a = store.branch("a")
    a.write("f", "1")
    a.checkpoint("first")
    a.intend("rewrite the parser to handle nested quotes")
    a.write("parser.py", "half a rewrite")

    r = a.resume()
    assert r.crashed
    assert r.intent == "rewrite the parser to handle nested quotes"
    assert r.dirty_paths == ("parser.py",)
    assert r.last_reason == "first"

    a.checkpoint("finished the rewrite")
    assert a.resume().intent is None
    assert not a.resume().crashed


def test_resume_reports_open_conflicts(store: Store) -> None:
    a, b = store.branch("a"), store.branch("b")
    a.write("f.txt", "A")
    base = a.checkpoint("base")
    c = store.branch("c", from_state=base)
    a.write("f.txt", "AA")
    a.checkpoint("a")
    c.write("f.txt", "CC")
    c.checkpoint("c")
    a.merge(c, reason="try")
    assert [x.path for x in a.resume().open_conflicts] == ["f.txt"]
    assert b.resume().open_conflicts == ()


# ------------------------------------------------------------------ repair and schema


def test_a_deleted_worktree_is_repaired_not_fatal(tmp_path: Path) -> None:
    """Deleting the sidecar directory once bricked every branch permanently."""
    import shutil

    root = tmp_path / "repo"
    s = Store.open(root, "s1")
    a = s.branch("a")
    a.write("f", "1")
    a.checkpoint("one")
    wt = a.worktree
    s.close()
    shutil.rmtree(wt)

    s = Store.open(root, "s1")
    a = s.branch("a")                      # repairs itself
    assert a.head.read("f") == "1"
    a.write("f", "2")
    assert a.checkpoint("two").read("f") == "2"
    s.close()


def test_a_future_schema_is_refused_not_misread(store: Store) -> None:
    a = store.branch("a")
    a.write("f", "1")
    st = a.checkpoint("one")
    raw = json.loads(store.backend.note_get(NOTES["meta"], st.id))
    raw["schema"] = "taste.memstore/99"
    store.backend.note_set(NOTES["meta"], st.id, json.dumps(raw))
    with pytest.raises(SchemaMismatch):
        _ = store.state(st.id).meta


# ------------------------------------------------------------------ handoff and housekeeping


def test_a_brain_can_hand_its_branch_to_another(store: Store) -> None:
    """The lease had no exit but process death, so one brain could never pass
    a branch to another without a restart."""
    a = store.branch("w", producer="brain-a")
    a.write("f", "1")
    a.checkpoint("a's work")
    assert a.holder is not None and a.holder["producer"] == "brain-a"

    a.release()
    b = store.branch("w", producer="brain-b")
    assert b.holder["producer"] == "brain-b"
    b.write("f", "2")
    st = b.checkpoint("b continues where a left off")
    assert st.read("f") == "2"
    assert st.meta.producer == "brain-b"
    assert [s.meta.producer for s in b.history()][:2] == ["brain-b", "brain-a"]


def test_gc_collects_what_a_lost_race_left_behind(store: Store) -> None:
    """A build that never publishes leaves an unreferenced commit and notes."""
    a = store.branch("a")
    a.write("f", "1")
    head = a.checkpoint("one")
    a.write("f", "2")
    orphan = a.build("never published")
    assert store.backend.note_get(NOTES["meta"], orphan.sha) is not None

    store.gc()
    assert store.backend.note_get(NOTES["meta"], orphan.sha) is None
    # and the real state is untouched
    assert a.head == head and head.manifest is not None and head.transcript is not None
