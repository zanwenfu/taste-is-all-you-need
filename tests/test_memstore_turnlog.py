"""A brain's turns must survive the death of the brain.

``intend`` is written to disk before the work, so a killed brain wakes knowing
what it was attempting. Its *turns* were not: they accumulated in a list in
memory and reached git only at the next checkpoint. A brain killed between the
model call and the checkpoint therefore lost the model's reasoning and the
tool results -- the expensive part, and precisely the material it would need to
improve from the failure.

That is a nothing-is-lost violation at the one layer whose whole purpose is
that nothing is lost, so it is fixed here rather than worked around above.
"""

from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest

from taste.memstore import StaleBranch, Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "repo", "s1")
    yield s
    s.close()


def _in_a_doomed_child(work) -> None:
    """Run ``work`` in a forked child that is SIGKILLed before it can return.

    SIGKILL is the honest test: no atexit, no finally, no flush on the way
    out. Whatever survives survived because it was already on disk.
    """
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never returns to pytest
        try:
            work()
        finally:
            os.kill(os.getpid(), signal.SIGKILL)
    _, status = os.waitpid(pid, 0)
    assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL


# ------------------------------------------------------------------ the defect


def test_turns_survive_a_kill_between_turn_and_checkpoint(tmp_path: Path) -> None:
    """The model's output is paid for the moment it arrives, so it is durable
    the moment it arrives -- not at the next checkpoint."""
    root = tmp_path / "repo"
    Store.open(root, "s1").close()

    def brain() -> None:
        s = Store.open(root, "s1")
        b = s.branch("worker")
        b.intend("implement the parser")
        b.turn(role="assistant", content="EXPENSIVE REASONING")
        b.turn(role="tool", content="pytest: 3 failed")

    _in_a_doomed_child(brain)

    s = Store.open(root, "s1")
    b = s.branch("worker")
    r = b.resume()
    assert r.intent == "implement the parser"
    contents = [t["content"] for t in r.recovered_turns]
    assert contents == ["EXPENSIVE REASONING", "pytest: 3 failed"]
    assert r.crashed
    s.close()


def test_recovered_turns_land_in_the_next_state_exactly_once(tmp_path: Path) -> None:
    """The recovered work is real work: it belongs in the next checkpoint, and
    it belongs there once."""
    root = tmp_path / "repo"
    Store.open(root, "s1").close()

    def brain() -> None:
        s = Store.open(root, "s1")
        b = s.branch("worker")
        b.turn(role="assistant", content="first")
        b.turn(role="tool", content="second")

    _in_a_doomed_child(brain)

    s = Store.open(root, "s1")
    b = s.branch("worker")
    b.turn(role="assistant", content="third")
    st = b.checkpoint("carrying on from where I died")
    assert [t["content"] for t in st.transcript.turns] == ["first", "second", "third"]
    s.close()


def test_a_consumed_journal_never_replays(tmp_path: Path) -> None:
    """A crash *after* the state is published must not re-apply the same turns.

    The journal is keyed on the state the turns extend, so once the branch has
    moved on the old journal can no longer be mistaken for pending work.
    """
    root = tmp_path / "repo"
    s = Store.open(root, "s1")
    b = s.branch("worker")
    b.turn(role="assistant", content="only once")
    st = b.checkpoint("published")
    assert [t["content"] for t in st.transcript.turns] == ["only once"]
    s.close()

    # Simulate the crash window: the ref moved, but cleanup never ran.
    common = Store.open(root, "s1").backend.common_dir
    leftovers = list(common.glob("memstore.turns.*"))
    for stale in leftovers:
        stale.write_text('{"role": "assistant", "content": "only once"}\n')

    s = Store.open(root, "s1")
    b = s.branch("worker")
    assert b.resume().recovered_turns == ()
    st2 = b.checkpoint("the next state")
    assert [t["content"] for t in st2.transcript.turns] == ["only once"]
    s.close()


def test_a_turn_journal_is_private_to_its_branch_and_session(tmp_path: Path) -> None:
    """One brain's uncommitted reasoning must never surface in another's."""
    root = tmp_path / "repo"
    s1 = Store.open(root, "s1")
    a, b = s1.branch("alpha"), s1.branch("beta")
    a.turn(role="assistant", content="alpha thinking")

    assert [t["content"] for t in a.resume().recovered_turns] == ["alpha thinking"]
    assert b.resume().recovered_turns == ()

    s2 = Store.open(root, "s2")
    other = s2.branch("alpha")
    assert other.resume().recovered_turns == ()
    s2.close()
    s1.close()


def test_publishing_clears_the_journal(tmp_path: Path) -> None:
    """The journal is a staging area, not a second copy of history."""
    root = tmp_path / "repo"
    s = Store.open(root, "s1")
    b = s.branch("worker")
    b.turn(role="assistant", content="pending")
    common = s.backend.common_dir
    assert list(common.glob("memstore.turns.s1.worker.*"))

    b.checkpoint("done")
    assert not list(common.glob("memstore.turns.s1.worker.*"))
    s.close()


def test_turns_are_durable_without_a_checkpoint_or_a_clean_exit(tmp_path: Path) -> None:
    """No flush, no close, no context manager -- the write itself is the commit
    to disk, because a killed process runs none of those."""
    root = tmp_path / "repo"
    Store.open(root, "s1").close()

    def brain() -> None:
        s = Store.open(root, "s1")
        b = s.branch("worker")
        for i in range(5):
            b.turn(role="assistant", content=f"turn {i}")

    _in_a_doomed_child(brain)

    s = Store.open(root, "s1")
    recovered = s.branch("worker").resume().recovered_turns
    assert [t["content"] for t in recovered] == [f"turn {i}" for i in range(5)]
    s.close()


def test_gc_sweeps_journals_that_no_branch_can_still_be_holding(tmp_path: Path) -> None:
    """Stale journals are debris, not memory: collectable, and only once the
    branch has moved past the state they extend."""
    root = tmp_path / "repo"
    s = Store.open(root, "s1")
    b = s.branch("worker")
    b.turn(role="assistant", content="live")
    common = s.backend.common_dir
    (common / "memstore.turns.s1.worker.deadbeefdeadbeefdeadbeefdeadbeefdeadbeef").write_text(
        '{"role": "assistant", "content": "stale"}\n'
    )

    s.gc()

    assert [t["content"] for t in b.resume().recovered_turns] == ["live"]
    assert not list(common.glob("*deadbeef*"))
    s.close()


def test_turns_survive_the_removal_of_the_working_tree(tmp_path: Path) -> None:
    """Dropping a worktree drops files, never reasoning.

    ``_capture`` saves uncommitted *files* before a destructive operation, but
    it only fires on a dirty tree. A brain that had thought without yet
    touching a file would fall through it, so the journal has to survive on
    its own -- which it does, because removal keeps the branch and its head,
    and the journal is named for that head.
    """
    root = tmp_path / "repo"
    s = Store.open(root, "s1")
    b = s.branch("worker")
    b.turn(role="assistant", content="thought about it, wrote nothing")
    assert not b.is_dirty()

    s.remove_branch("worker")

    again = s.branch("worker")
    recovered = again.resume().recovered_turns
    assert [t["content"] for t in recovered] == ["thought about it, wrote nothing"]
    st = again.checkpoint("picking it back up")
    assert [t["content"] for t in st.transcript.turns] == ["thought about it, wrote nothing"]
    s.close()


def test_a_turn_recorded_while_a_state_is_being_built_is_not_destroyed(tmp_path: Path) -> None:
    """Building a state takes a git commit and four notes writes -- a fifth of
    a second in which the brain is still thinking.

    ``build`` reads the journal and ``publish_state`` consumed it by name, so
    everything recorded in between was deleted unread. The fold has to be by
    position, not by filename, and whatever arrived after it has to carry over
    to the journal of the state that was just published.
    """
    root = tmp_path / "repo"
    s = Store.open(root, "s1")
    b = s.branch("worker")

    b.turn(role="assistant", content="before the build")
    built = b.build("a state that takes a while to publish")
    b.turn(role="assistant", content="thought during the build")
    published = b.publish_state(built)

    assert [t["content"] for t in published.transcript.turns] == ["before the build"]

    # The turn that arrived mid-build is still pending, not destroyed.
    assert [t["content"] for t in b.resume().recovered_turns] == ["thought during the build"]
    nxt = b.checkpoint("the next state")
    assert [t["content"] for t in nxt.transcript.turns] == [
        "before the build",
        "thought during the build",
    ]
    s.close()


def test_a_lost_race_loses_the_state_but_never_the_turns(tmp_path: Path) -> None:
    """Losing the compare-and-swap discards a state, not a brain's reasoning.

    Every turn must end up in exactly one of two places -- committed to the
    branch, or still pending -- and never in neither. It may legitimately be
    in the *winning* state rather than pending, which is why the assertion is
    on the union rather than on the journal alone.
    """
    root = tmp_path / "repo"
    s = Store.open(root, "s1")
    b = s.branch("worker")

    b.turn(role="assistant", content="folded into the doomed build")
    built = b.build("about to lose the race")
    b.turn(role="assistant", content="arrived during the doomed build")

    # Someone else moves the branch first.
    other = b.build("the winner")
    b.publish_state(other)

    with pytest.raises(StaleBranch):
        b.publish_state(built)

    committed = [t["content"] for t in b.head.transcript.turns]
    pending = [t["content"] for t in b.resume().recovered_turns]
    for turn in ("folded into the doomed build", "arrived during the doomed build"):
        assert turn in committed or turn in pending, f"{turn!r} is in neither"
    # and never in both, which would duplicate it at the next checkpoint
    assert not (set(committed) & set(pending))
    s.close()


def test_no_turn_is_lost_when_a_brain_keeps_thinking_through_a_checkpoint(
    tmp_path: Path,
) -> None:
    """The realistic shape of the bug: a brain streaming turns while a
    checkpoint runs.

    A checkpoint is a git commit plus four notes writes -- a fifth of a second
    during which a brain with a streaming callback, or a monitor thread, is
    still recording. Every one of those turns has to end up committed or
    pending, and each exactly once.
    """
    import threading

    root = tmp_path / "repo"
    s = Store.open(root, "s1")
    b = s.branch("worker")

    stop = threading.Event()
    produced: list[str] = []

    def keep_thinking() -> None:
        i = 0
        while not stop.is_set():
            content = f"t{i}"
            b.turn(role="assistant", content=content)
            produced.append(content)
            i += 1
            stop.wait(0.005)

    thinker = threading.Thread(target=keep_thinking)
    thinker.start()
    try:
        st = b.checkpoint("checkpoint while the brain is still talking")
    finally:
        stop.set()
        thinker.join()

    committed = [t["content"] for t in st.transcript.turns]
    pending = [t["content"] for t in b.resume().recovered_turns]

    assert len(produced) > 1, "the thread never got to record anything"
    missing = [t for t in produced if t not in committed and t not in pending]
    assert not missing, f"lost turns: {missing}"
    assert not (set(committed) & set(pending)), "a turn was both committed and pending"

    final = b.checkpoint("and now everything settles")
    assert [t["content"] for t in final.transcript.turns] == produced
    s.close()
