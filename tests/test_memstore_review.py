"""One test per defect the second independent review found.

The first audit went after the layer as it was. This round went after the
fixes, and after the question the fixes existed to answer: can a brain
actually be built on this? Several of these are gaps rather than bugs -- the
monitor could not see work in flight, and nothing gave it patch text -- but a
gap that forces the layer above to reach around the API is a defect in the
API, so they are pinned the same way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taste.memstore import (
    BranchBusy,
    NotAnAncestor,
    Store,
    Verdict,
)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "repo", "s1")
    yield s
    s.close()


# ------------------------------------------------ rollback stays in its branch


def test_rollback_will_not_reinstate_another_branchs_state(store: Store) -> None:
    """After a merge, the other branch's states are git ancestors of ours.

    Ancestry was the only guard, so ``rollback`` would happily adopt one --
    taking that branch's tree, manifest and transcript, and dropping our own
    files, with no error and nothing in the return value to say so.
    """
    a = store.branch("a")
    a.write("a.txt", "mine\n")
    a.checkpoint("a's own work")

    b = store.branch("b", from_state=a.head)
    b.write("b.txt", "theirs\n")
    theirs = b.checkpoint("b's own work")

    a.merge(b, reason="take b's work")
    assert store.backend.is_ancestor(theirs.id, a.head.id), "the setup must make it an ancestor"

    with pytest.raises(NotAnAncestor):
        a.rollback(theirs, "roll back to a state that was never mine")

    assert a.head.read("a.txt") == "mine\n"


def test_rollback_still_works_within_the_branch(store: Store) -> None:
    """The guard must not break the operation it guards."""
    b = store.branch("worker")
    b.write("f.txt", "one\n")
    good = b.checkpoint("a good state")
    b.write("f.txt", "two\n")
    b.checkpoint("a bad state")

    b.rollback(good, "back to the good one")
    assert b.head.read("f.txt") == "one\n"
    assert b.head.meta.restores == good.id


# ------------------------------------------------ branch creation is one write


def test_a_new_branch_never_exists_pointing_at_another_branchs_state(
    store: Store,
) -> None:
    """Creation used to claim the ref and seed it second, so a crash between
    the two left the branch permanently showing another branch's state with an
    empty history -- and reopening skipped the seed, so it never repaired."""
    a = store.branch("a")
    a.write("a.txt", "mine\n")
    base = a.checkpoint("a's work")

    b = store.branch("b", from_state=base)
    head = b.head
    assert head.meta.branch == "b", f"branch b opened onto {head.meta.branch}'s state"
    assert b.history(), "a freshly created branch must have its own first state"


# ------------------------------------------- a checkpoint survives another gc


def test_a_checkpoint_is_not_broken_by_a_gc_in_another_process(
    tmp_path: Path,
) -> None:
    """``gc`` prunes unreachable objects, and a state being built is briefly
    unreachable. It must not be able to land inside a checkpoint."""
    root = tmp_path / "repo"
    writer = Store.open(root, "s1")
    b = writer.branch("worker")
    b.write("f.txt", "work\n")

    tidier = Store.open(root, "s1")
    tidier.gc()

    st = b.checkpoint("the work that must not be lost")
    assert st.read("f.txt") == "work\n"
    assert b.head.id == st.id
    tidier.close()
    writer.close()


# ------------------------------------------- a monitor sees work in flight


def test_a_monitor_sees_intent_before_the_brain_checkpoints(store: Store) -> None:
    """The monitor's whole job is catching a wrong turn before a minute of
    tool time is spent on it. It cannot wait for a checkpoint."""
    b = store.branch("worker")
    b.checkpoint("start")
    b.intend("try approach X with the frobnicator")

    view = store.view("worker")
    assert view.intent == "try approach X with the frobnicator"


def test_a_monitor_sees_uncommitted_files_and_their_patch(store: Store) -> None:
    b = store.branch("worker")
    b.write("kept.txt", "one\n")
    b.checkpoint("a committed state")
    b.write("kept.txt", "two\n")
    b.write("brand-new.txt", "hello\n")

    view = store.view("worker")
    assert set(view.dirty_paths()) == {"kept.txt", "brand-new.txt"}

    patch = view.pending_diff()
    assert "kept.txt" in patch and "brand-new.txt" in patch, "untracked work is still work"
    assert "+two" in patch, "a monitor grades the code, not the filenames"
    assert "+hello" in patch


def test_asking_what_changed_does_not_change_what_is_captured(store: Store) -> None:
    """The patch must come from a read, not from staging the tree."""
    b = store.branch("worker")
    b.checkpoint("base")
    b.write("new.txt", "content\n")

    view = store.view("worker")
    view.pending_diff()
    view.dirty_paths()

    assert b.dirty_paths() == ["new.txt"]
    st = b.checkpoint("after being watched")
    assert st.read("new.txt") == "content\n"


def test_a_monitor_sees_turns_the_brain_has_not_checkpointed(store: Store) -> None:
    b = store.branch("worker")
    b.checkpoint("start")
    b.turn(role="assistant", content="I am going to try X")

    view = store.view("worker")
    assert [t["content"] for t in view.pending_turns()] == ["I am going to try X"]


def test_a_monitor_can_see_who_holds_a_branch(store: Store) -> None:
    b = store.branch("worker")
    b.checkpoint("start")

    view = store.view("worker")
    holder = view.holder
    assert holder is not None and "pid" in holder

    b.release()
    assert store.view("worker").holder is None


def test_none_of_the_monitors_reads_take_the_lease(tmp_path: Path) -> None:
    """If watching required the lease, the brain would have to stop to be
    watched -- which is the opposite of what a monitor is for."""
    root = tmp_path / "repo"
    writer = Store.open(root, "s1")
    b = writer.branch("worker")
    b.checkpoint("start")
    b.intend("doing something long")
    b.turn(role="assistant", content="thinking")
    b.write("wip.txt", "half done\n")

    # A second opener of the same branch is refused the lease...
    second = Store.open(root, "s1")
    with pytest.raises(BranchBusy):
        second.branch("worker")

    # ...but every read a monitor needs answers anyway, from that same store.
    view = second.view("worker")
    assert view.intent == "doing something long"
    assert view.dirty_paths() == ["wip.txt"]
    assert [t["content"] for t in view.pending_turns()] == ["thinking"]
    assert view.head.id == b.head.id
    assert view.pending_diff()

    # and the brain never had to stop
    assert b.head.id == view.head.id
    b.checkpoint("still going")

    second.close()
    writer.close()


# ------------------------------------------------------------ verdict routing


def test_a_monitor_judges_work_it_watched_without_the_lease(tmp_path: Path) -> None:
    """The whole loop, end to end: watch in flight, judge, and be heard."""
    root = tmp_path / "repo"
    writer = Store.open(root, "s1")
    b = writer.branch("worker")
    b.write("bad.py", "def f(): return None\n")
    st = b.checkpoint("an approach that will not work")

    monitor = Store.open(root, "s1")
    view = monitor.view("worker")
    assert "bad.py" in view.head.files()
    monitor.judge(view.head, Verdict(status="fail", by="monitor", detail="returns None"))
    monitor.close()

    r = b.resume()
    assert r.failed and [v.detail for v in r.unacked] == ["returns None"]
    assert st.id == r.head_id
    writer.close()


def test_a_dead_holders_label_does_not_make_a_free_branch_look_busy(
    tmp_path: Path,
) -> None:
    """A crashed brain frees the lease -- the lock is in the kernel -- but its
    label stays on the file. Reporting that as the holder would tell a
    monitor a branch is occupied when anyone may take it."""
    import json
    import os
    import socket

    root = tmp_path / "repo"
    s = Store.open(root, "s1")
    b = s.branch("worker")
    b.checkpoint("start")
    b.release()

    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child exits immediately
        os._exit(0)
    os.waitpid(pid, 0)  # now certainly not running

    s.sidecar("lease", "worker").write_text(
        json.dumps({"pid": pid, "host": socket.gethostname(), "producer": "ghost"})
    )

    assert s.view("worker").holder is None
    s.branch("worker")  # and it can actually be taken
    s.close()


def test_an_alive_process_label_without_the_kernel_lease_is_not_a_holder(
    tmp_path: Path,
) -> None:
    """PID liveness cannot substitute for the branch's actual flock."""
    import json
    import os
    import socket

    s = Store.open(tmp_path / "repo", "s1")
    branch = s.branch("worker")
    branch.checkpoint("start")
    branch.release()
    s.sidecar("lease", "worker").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "producer": "stale-but-alive",
            }
        )
    )

    assert s.view("worker").holder is None
    replacement = s.branch("worker")
    assert replacement.holder is not None
    s.close()


def test_a_rollback_gives_back_clean_context_without_discarding_the_failure(
    store: Store,
) -> None:
    """Rollback restores the context the brain had at the good state -- that is
    the point of it -- so the failed attempt's reasoning is deliberately not in
    the new transcript.

    That is only acceptable because it is still *reachable*: the failed state
    is one parent away and the new state names it. If either were untrue,
    rollback would be discarding the very material the vision says a failure
    exists to leave behind.
    """
    b = store.branch("worker")
    b.turn(role="assistant", content="a sound plan")
    good = b.checkpoint("a good state")

    b.turn(role="assistant", content="the mistake, and why I made it")
    b.write("broken.py", "syntax error\n")
    bad = b.checkpoint("the attempt that fails")

    rolled = b.rollback(good, "that did not work")

    # clean context, as documented
    assert [t["content"] for t in rolled.transcript.turns] == ["a sound plan"]
    assert rolled.read("broken.py") is None

    # but nothing was discarded
    assert rolled.meta.rolled_back_from == bad.id
    assert rolled.meta.restores == good.id
    recovered = store.state(bad.id)
    assert recovered.read("broken.py") == "syntax error\n"
    assert "the mistake, and why I made it" in [
        t["content"] for t in recovered.transcript.turns
    ]
    assert bad.id in [p.id for p in rolled.parents]
