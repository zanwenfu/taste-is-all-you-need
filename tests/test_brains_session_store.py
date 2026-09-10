"""The memstore-backed SessionStore, against the SDK's own conformance suite.

The 14 behavioural contracts are Anthropic's, not ours, which is the point:
the adapter is checked against the specification its consumer actually holds
rather than against my reading of it. Our own tests then cover what the
conformance suite has no opinion about -- that a transcript is real history in
the branch, that it survives a kill, and that it moves with a rollback.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from taste.memstore import Store

pytest.importorskip("claude_agent_sdk", reason="the brain layer needs claude-agent-sdk")

from claude_agent_sdk.testing import run_session_store_conformance

from taste.brains.session_store import MAIN_SUBPATH, MemstoreSessionStore


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "repo", "s1")
    yield s
    s.close()


def _run(coro):
    """This repo has no async pytest plugin; four coroutines do not justify one."""
    return asyncio.run(coro)


def _entry(**kw):
    return {"type": "x", **kw}


# ------------------------------------------------------------------ the contract


def test_it_satisfies_the_sdks_own_conformance_suite(tmp_path: Path) -> None:
    """All 14 contracts, including every optional method we implement."""
    made: list[Store] = []

    def make_store() -> MemstoreSessionStore:
        # A fresh repository per contract: the suite calls this once per
        # contract precisely so state cannot leak between them.
        root = tmp_path / f"repo{len(made)}"
        s = Store.open(root, "s1")
        made.append(s)
        return MemstoreSessionStore(s, "brain")

    try:
        asyncio.run(run_session_store_conformance(make_store))
    finally:
        for s in made:
            s.close()


# ------------------------------------------------------------------ our own claims


def test_a_transcript_becomes_history_at_the_brains_own_checkpoint(store: Store) -> None:
    """A brain's conversation and its artifacts are one history.

    The adapter itself never commits: ``checkpoint`` stages the whole
    worktree, so a mirror batch landing mid-edit would commit half-written
    code under a reason claiming to be a transcript write. It writes durably
    and lets the sub-brain's own checkpoint -- where the tree is coherent and
    the reason is true -- turn that into history.
    """
    sess = MemstoreSessionStore(store, "brain")
    key = {"project_key": "proj", "session_id": "s-1"}
    _run(sess.append(key, [_entry(uuid="a", text="thinking")]))

    branch = store.branch("brain")
    assert branch.head.read(sess._path(key)) is None, "the adapter must not commit"
    assert _run(sess.load(key)) == [_entry(uuid="a", text="thinking")]

    st = branch.checkpoint("the brain's own work")
    assert any(f.endswith(f"{MAIN_SUBPATH}.jsonl") for f in st.files()), st.files()


def test_entries_are_stored_verbatim(store: Store) -> None:
    """The transcript format is an internal union; a store that interprets it
    breaks on the next CLI release."""
    sess = MemstoreSessionStore(store, "brain")
    key = {"project_key": "proj", "session_id": "s-1"}
    odd = _entry(
        uuid="a",
        nested={"deep": [1, 2, {"x": None}]},
        unicode="café 日本語 🙂",
        empty_list=[],
        false=False,
        zero=0,
    )
    _run(sess.append(key, [odd]))
    assert _run(sess.load(key)) == [odd]


def test_a_transcript_survives_the_death_of_the_process(tmp_path: Path) -> None:
    """append() returns only once the entries are committed, so a store that
    is reopened from scratch has them."""
    root = tmp_path / "repo"
    s1 = Store.open(root, "s1")
    sess = MemstoreSessionStore(s1, "brain")
    key = {"project_key": "proj", "session_id": "s-1"}
    _run(sess.append(key, [_entry(uuid="a", text="expensive reasoning")]))
    s1.close()  # nothing else runs; no flush, no drain

    s2 = Store.open(root, "s1")
    again = MemstoreSessionStore(s2, "brain")
    assert _run(again.load(key)) == [_entry(uuid="a", text="expensive reasoning")]
    s2.close()


def test_a_rollback_takes_the_conversation_with_it(store: Store) -> None:
    """A brain rolled back to an earlier state must not remember what it said
    afterwards -- and the abandoned conversation must still be reachable."""
    sess = MemstoreSessionStore(store, "brain")
    key = {"project_key": "proj", "session_id": "s-1"}

    _run(sess.append(key, [_entry(uuid="a", text="sound plan")]))
    good = store.branch("brain").checkpoint("a sound plan")

    _run(sess.append(key, [_entry(uuid="b", text="the mistake")]))
    bad = store.branch("brain").checkpoint("the mistake")
    assert len(_run(sess.load(key))) == 2

    store.branch("brain").rollback(good, "that approach was wrong")

    # the brain no longer has the mistake in its context...
    assert [e["text"] for e in _run(sess.load(key))] == ["sound plan"]
    # ...but nothing was destroyed: it is one state away, still readable.
    restored = store.state(bad.id)
    raw = restored.read(sess._path(key))
    assert raw is not None and "the mistake" in raw


def test_subagent_transcripts_are_discoverable(store: Store) -> None:
    """Without list_subkeys the SDK materializes only the main transcript on
    resume, and every subagent's reasoning is silently dropped."""
    sess = MemstoreSessionStore(store, "brain")
    key = {"project_key": "proj", "session_id": "s-1"}
    _run(sess.append(key, [_entry(uuid="m")]))
    _run(sess.append({**key, "subpath": "subagents/agent-1"}, [_entry(uuid="s1")]))
    _run(sess.append({**key, "subpath": "subagents/agent-2"}, [_entry(uuid="s2")]))

    assert _run(sess.list_subkeys(key)) == ["subagents/agent-1", "subagents/agent-2"]
    assert _run(sess.load({**key, "subpath": "subagents/agent-1"})) == [_entry(uuid="s1")]


def test_a_hostile_project_key_cannot_escape_the_transcript_directory(
    store: Store,
) -> None:
    """project_key is caller-defined and defaults to a sanitized cwd; it must
    not be able to write outside the transcript tree."""
    sess = MemstoreSessionStore(store, "brain")
    key = {"project_key": "../../etc", "session_id": "../../passwd"}
    _run(sess.append(key, [_entry(uuid="a")]))

    for path in store.branch("brain").head.files():
        assert path.startswith("sdk-sessions/")
        # escaped, not merely absent: no segment may be a traversal
        assert ".." not in path.split("/"), path
    assert _run(sess.load(key)) == [_entry(uuid="a")]


def test_two_brains_keep_separate_transcripts(store: Store) -> None:
    """One store per branch: a brain's memory is its own."""
    a = MemstoreSessionStore(store, "alpha")
    b = MemstoreSessionStore(store, "beta")
    key = {"project_key": "proj", "session_id": "same-id"}

    _run(a.append(key, [_entry(uuid="a", who="alpha")]))
    _run(b.append(key, [_entry(uuid="b", who="beta")]))

    assert _run(a.load(key)) == [_entry(uuid="a", who="alpha")]
    assert _run(b.load(key)) == [_entry(uuid="b", who="beta")]


def test_the_brain_layer_states_its_dependency_at_import(tmp_path: Path) -> None:
    """A missing SDK must fail at import with a clear message, not mid-append.

    ``fold_session_summary`` is on the hot path -- every main-transcript write
    maintains the summary sidecar -- so a lazy import would surface as a
    traceback from inside a batch the caller already believed was durable.
    """
    import subprocess
    import sys

    # find_spec, not the removed find_module: Python 3.12 ignores the latter,
    # so a blocker written that way silently blocks nothing.
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.startswith('claude_agent_sdk'):\n"
        "            raise ImportError('absent')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "try:\n"
        "    import taste.brains.session_store\n"
        "except ImportError as e:\n"
        "    print('OK' if 'claude-agent-sdk' in str(e) else f'unclear: {e}')\n"
        "else:\n"
        "    print('imported without the SDK')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "OK", f"{out.stdout}{out.stderr}"


def test_a_long_key_resolves_to_the_same_place_in_every_process() -> None:
    """A brain must find its own transcript after a restart.

    ``_safe`` shortens an over-long key with a digest. Python randomizes
    string hashing per process, so ``hash()`` would put the same session in a
    different directory on every run and resume would silently find nothing --
    and a realistic worktree realpath is already ~127 characters, so this is
    the ordinary path rather than an edge case.
    """
    import subprocess
    import sys

    long_key = (
        "/private/var/folders/w5/_y5qsxd54h7g1bgp2wv659h0000gn/T/brain-spike-abc123"
        "/.taste-worktrees/9f2a1b3c4d5e/session-name/worker-01"
    ).replace("/", "-")
    assert len(long_key) > 120, "the fixture must actually exercise the shortening path"

    code = (
        "from taste.brains.session_store import MemstoreSessionStore as M\n"
        f"print(M._safe({long_key!r}))\n"
    )
    seen = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120).stdout.strip()
        for _ in range(3)
    }
    assert len(seen) == 1, f"the same key resolved to {len(seen)} different paths: {seen}"


def test_no_entry_is_lost_across_many_appends(tmp_path: Path) -> None:
    """Batching commits must never cost entries.

    ``branch.read`` returns the last *committed* state while ``branch.write``
    writes the working tree, so a read-modify-write rebased every batch onto a
    base that only advanced at a checkpoint. At checkpoint_every=5, eight of
    ten entries were destroyed before any commit could preserve them, and
    nothing raised -- the obvious cure for the adapter's latency was the thing
    that deleted the brain's reasoning.
    """
    s = Store.open(tmp_path / "repo", "s1")
    try:
        sess = MemstoreSessionStore(s, "brain")
        key = {"project_key": "p", "session_id": "sess"}
        for i in range(10):
            _run(sess.append(key, [_entry(uuid=f"u{i}")]))
        got = _run(sess.load(key))
        assert got is not None, "every append returned successfully; load found nothing"
        assert [e["uuid"] for e in got] == [f"u{i}" for i in range(10)]
    finally:
        s.close()


def test_a_resume_sees_entries_that_are_not_committed_yet(tmp_path: Path) -> None:
    """The tail of a transcript is the part a resuming brain needs most.

    Entries written since the last checkpoint are real -- they are fsynced --
    so a read that consulted only committed state would hand the brain a
    transcript missing its most recent turns.
    """
    s = Store.open(tmp_path / "repo", "s1")
    try:
        sess = MemstoreSessionStore(s, "brain")
        key = {"project_key": "p", "session_id": "sess"}
        _run(sess.append(key, [_entry(uuid="a", text="the newest thinking")]))

        assert s.branch("brain").head.read(sess._path(key)) is None, (
            "the fixture must leave the entry uncommitted"
        )
        got = _run(sess.load(key))
        assert got == [_entry(uuid="a", text="the newest thinking")]
    finally:
        s.close()


def test_a_mirror_batch_never_commits_the_brains_half_written_work(store: Store) -> None:
    """``Branch.checkpoint`` stages the whole worktree.

    An adapter that committed per mirror batch photographed whatever the brain
    happened to be mid-keystroke on, and labelled it a transcript write -- a
    state no monitor could interpret and a rollback would faithfully restore.
    """
    sess = MemstoreSessionStore(store, "brain")
    branch = store.branch("brain")
    branch.write("auth.py", "def login():\n    # half-written\n    return (\n")

    _run(sess.append({"project_key": "p", "session_id": "s"}, [_entry(uuid="u0")]))

    assert not branch.history()[:-1] or "auth.py" not in branch.head.files(), (
        "a transcript mirror committed the brain's broken mid-edit file"
    )
    assert branch.is_dirty(), "the brain's work should still be its own to commit"


def test_a_redelivered_batch_is_not_stored_twice(store: Store) -> None:
    """The SDK retries a failed batch three times.

    The protocol says most entries carry a stable uuid to be treated as an
    idempotency key; without that, a retry after a partial success doubles the
    brain's own turns and it resumes reading itself twice.
    """
    sess = MemstoreSessionStore(store, "brain")
    key = {"project_key": "p", "session_id": "s"}
    batch = [_entry(uuid="u1"), _entry(uuid="u2")]

    _run(sess.append(key, batch))
    _run(sess.append(key, batch))

    assert [e["uuid"] for e in _run(sess.load(key))] == ["u1", "u2"]


def test_entries_without_a_uuid_are_never_deduplicated(store: Store) -> None:
    """Titles, tags and mode markers carry no identity; two of them are two
    entries, not a duplicate."""
    sess = MemstoreSessionStore(store, "brain")
    key = {"project_key": "p", "session_id": "s"}
    _run(sess.append(key, [{"type": "title"}, {"type": "title"}]))
    assert len(_run(sess.load(key))) == 2


def test_a_brain_finds_its_memory_after_its_worktree_moves(tmp_path: Path) -> None:
    """A brain's memory follows its identity, not its filesystem location.

    The SDK derives ``project_key`` from the realpath of cwd and offers no way
    to set it, so a worktree recreated at a different path addressed a
    different transcript and the brain woke amnesiac -- with no error. Pinning
    the scope to the branch makes worktrees disposable, which is what the
    architecture assumes: the branch is the address space, the worktree is
    scratch.
    """
    s = Store.open(tmp_path / "repo", "s1")
    try:
        sess = MemstoreSessionStore(s, "brain", project_key="mem/s1/brain")
        before = {"project_key": "-Users-someone-wt-abc123", "session_id": "s-1"}
        _run(sess.append(before, [_entry(uuid="a", text="what I learned")]))

        # the worktree is recreated somewhere else; the SDK derives a new key
        after = {"project_key": "-private-var-folders-T-wt-zzz999", "session_id": "s-1"}
        assert _run(sess.load(after)) == [_entry(uuid="a", text="what I learned")]
    finally:
        s.close()


def test_two_projects_stay_isolated_when_the_scope_is_not_pinned(tmp_path: Path) -> None:
    """Pinning is opt-in; without it the SDK's own scoping must still hold,
    because the protocol requires distinct projects to be isolated."""
    s = Store.open(tmp_path / "repo", "s1")
    try:
        sess = MemstoreSessionStore(s, "brain")
        a = {"project_key": "A", "session_id": "s1"}
        b = {"project_key": "B", "session_id": "s1"}
        _run(sess.append(a, [_entry(uuid="a", who="A")]))
        _run(sess.append(b, [_entry(uuid="b", who="B")]))
        assert _run(sess.load(a)) == [_entry(uuid="a", who="A")]
        assert _run(sess.load(b)) == [_entry(uuid="b", who="B")]
    finally:
        s.close()


def test_a_rollback_lets_a_re_mirrored_entry_back_in(tmp_path: Path) -> None:
    """The dedup cache is keyed on the head for a reason.

    A rollback removes turns from the transcript, and a cache that outlived it
    swallowed the SDK's re-mirror of exactly those entries -- the brain lost
    them for good, silently, having been told they were already stored.
    """
    s = Store.open(tmp_path / "repo", "s1")
    try:
        sess = MemstoreSessionStore(s, "brain")
        key = {"project_key": "p", "session_id": "sess"}
        branch = s.branch("brain")

        _run(sess.append(key, [_entry(uuid="u1")]))
        good = branch.checkpoint("a good state")
        _run(sess.append(key, [_entry(uuid="u2")]))
        branch.checkpoint("a state we will abandon")

        branch.rollback(good, "that was wrong")
        _run(sess.append(key, [_entry(uuid="u2")]))

        assert [e["uuid"] for e in _run(sess.load(key))] == ["u1", "u2"]
    finally:
        s.close()


def test_dedup_still_holds_within_one_state(tmp_path: Path) -> None:
    """Keying the cache on the head must not disable the dedup it exists for."""
    s = Store.open(tmp_path / "repo", "s1")
    try:
        sess = MemstoreSessionStore(s, "brain")
        key = {"project_key": "p", "session_id": "sess"}
        batch = [_entry(uuid="u1")]
        _run(sess.append(key, batch))
        _run(sess.append(key, batch))
        assert [e["uuid"] for e in _run(sess.load(key))] == ["u1"]
    finally:
        s.close()
