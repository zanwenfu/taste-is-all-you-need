"""The memstore-backed SessionStore, against the SDK's own conformance suite.

The 14 behavioural contracts are Anthropic's, not ours, which is the point:
the adapter is checked against the specification its consumer actually holds
rather than against my reading of it. Our own tests then cover what the
conformance suite has no opinion about -- that a transcript is real history in
the branch, that it survives a kill, and that it moves with a rollback.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import ExitStack, closing
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


def test_colliding_legacy_names_keep_separate_sdk_recovery_ledgers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    key = {"project_key": "proj", "session_id": "same-sdk-session"}
    identities = (("tb", "task1.worker"), ("tb.task1", "worker"))

    def interrupted(record):
        raise OSError("interrupted after transcript persistence")

    with ExitStack() as stack:
        for index, (session, branch) in enumerate(identities):
            store = stack.enter_context(closing(Store.open(root, session)))
            adapter = MemstoreSessionStore(store, branch)
            with monkeypatch.context() as patch:
                patch.setattr(adapter, "_apply_append_sidecars", interrupted)
                with pytest.raises(OSError, match="interrupted"):
                    _run(adapter.append(key, [_entry(uuid="same-uuid", text=f"reasoning {index}")]))
            store.branch(branch).turn(content=f"pending {index}")
            store.gc()
    for index, (session, branch) in enumerate(identities):
        with closing(Store.open(root, session)) as store:
            adapter = MemstoreSessionStore(store, branch)
            assert _run(adapter.load(key)) == [_entry(uuid="same-uuid", text=f"reasoning {index}")]
            state = store.branch(branch).checkpoint("persist SDK recovery")
            assert [turn["content"] for turn in state.transcript.turns] == [f"pending {index}"]
            ledger = adapter._read_mirror_ledger()
            assert len(ledger["history"]) == 1
            assert not ledger["unresolved"]


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


def test_sdk_mirror_and_worker_resume_on_a_repository_with_git_history(store: Store) -> None:
    for index in range(3):
        (store.root / "source.py").write_text(f"VERSION = {index}\n")
        store.backend.repo.git.add("--all")
        store.backend.repo.git.commit("-m", f"ordinary source history {index}")
    key = {"project_key": "proj", "session_id": "s-1"}
    adapter = MemstoreSessionStore(store, "brain")
    _run(adapter.append(key, [_entry(uuid="a", text="understood existing source")]))
    worker = store.branch("brain")
    assert worker.resume().head_id == worker.head.id
    worker.checkpoint("persist conversation with source")
    worker.close()
    recovered = MemstoreSessionStore(store, "brain")
    assert _run(recovered.load(key)) == [_entry(uuid="a", text="understood existing source")]
    assert store.branch("brain").resume().recovered_turns == ()
    assert store.view("brain").read("source.py") == "VERSION = 2\n"


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
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        ).stdout.strip()
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


def _fsync_bytes(path: Path, payload: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _leave_fsynced_transcript_before_sidecars(
    sess: MemstoreSessionStore,
    key: dict[str, str],
    batch: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    """Inject the crash boundary immediately after transcript fsync."""

    def crash_before_sidecars(record):
        raise OSError("killed after transcript fsync")

    monkeypatch.setattr(sess, "_apply_append_sidecars", crash_before_sidecars)
    with pytest.raises(OSError, match="after transcript fsync"):
        _run(sess.append(key, batch))
    raw = sess._read_mirror_ledger()
    assert len(raw["unresolved"]) == 1
    return next(iter(raw["unresolved"].values()))


def test_fsynced_uuidless_batch_recovers_after_process_reopen_without_redelivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kill after fsync must not require the SDK to resend opaque markers."""
    root = tmp_path / "repo"
    key = {"project_key": "p", "session_id": "sess"}
    batch = [{"type": "title", "title": "the same marker may occur twice"}]

    first = Store.open(root, "s1")
    sess = MemstoreSessionStore(first, "brain")
    record = _leave_fsynced_transcript_before_sidecars(sess, key, batch, monkeypatch)
    assert record["status"] == "failed"
    assert record["transcript"]["payload"] == sess._entry_payload(batch)
    assert record["evidence"][-1]["classification"] == "complete"
    assert record["failures"][-1]["message"] == "killed after transcript fsync"
    first.close()

    # A new process has no in-memory UUID cache and the SDK never redelivers.
    second = Store.open(root, "s1")
    try:
        again = MemstoreSessionStore(second, "brain")
        assert again.unresolved_mirror_batches() == ()
        assert _run(again.load(key)) == batch
        assert len(_run(again.list_session_summaries("p"))) == 1

        history = again._read_mirror_ledger()["history"]
        assert history[-1]["status"] == "recovered"
        assert history[-1]["failures"][-1]["message"] == "killed after transcript fsync"
        assert history[-1]["evidence"][-1]["classification"] == "complete"
    finally:
        second.close()


@pytest.mark.parametrize("classification", ["absent", "partial", "mismatch"])
def test_recovery_distinguishes_noncomplete_transcript_writes_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    classification: str,
) -> None:
    root = tmp_path / classification
    key = {"project_key": "p", "session_id": "sess"}
    # No UUID: retry safety must come from the byte-range journal, not dedup.
    batch = [{"type": "mode", "mode": "careful"}]

    first = Store.open(root, "s1")
    sess = MemstoreSessionStore(first, "brain")
    record = _leave_fsynced_transcript_before_sidecars(sess, key, batch, monkeypatch)
    target = first.branch("brain").path(sess._path(key))
    before_size = record["transcript"]["before"]["size"]
    original = target.read_bytes()
    prefix = original[:before_size]
    payload = record["transcript"]["payload"].encode()
    if classification == "absent":
        changed = prefix
    elif classification == "partial":
        changed = prefix + payload[: max(1, len(payload) // 2)]
    else:
        changed = prefix + b'{"type": "different"}\n'
    _fsync_bytes(target, changed)
    first.close()

    second = Store.open(root, "s1")
    try:
        again = MemstoreSessionStore(second, "brain")
        unresolved = again.unresolved_mirror_batches()
        assert len(unresolved) == 1
        assert unresolved[0]["status"] == classification
        assert unresolved[0]["evidence"][-1]["classification"] == classification

        if classification == "mismatch":
            with pytest.raises(RuntimeError, match="divergent"):
                _run(again.append(key, batch))
            assert again.unresolved_mirror_batches()
        else:
            # Exact absent/partial prefixes can be completed only when the SDK
            # supplies the same fingerprint again.
            _run(again.append(key, batch))
            assert again.unresolved_mirror_batches() == ()
            assert _run(again.load(key)) == batch
    finally:
        second.close()


def test_redelivery_after_summary_failure_reconstructs_summary_and_meta(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dedup of transcript UUIDs must not skip the failed sidecar transition."""
    sess = MemstoreSessionStore(store, "brain")
    key = {"project_key": "p", "session_id": "sess"}
    batch = [
        _entry(
            uuid="u1",
            timestamp="2024-01-01T00:00:00.000Z",
            customTitle="durable title",
        )
    ]

    def fail_summary(*args, **kwargs):
        raise OSError("summary disk failed")

    monkeypatch.setattr(sess, "_fold_summary", fail_summary)
    with pytest.raises(OSError, match="summary disk failed"):
        _run(sess.append(key, batch))

    raw = sess._read_mirror_ledger()
    pending = next(iter(raw["unresolved"].values()))
    assert pending["failures"][-1]["message"] == "summary disk failed"
    assert not store.branch("brain").path(sess._summary_path("p", "sess")).exists()

    # Recovery runs before fresh-entry dedup and uses the journalled fold, so
    # the retry neither duplicates the transcript nor clears a false alarm.
    _run(sess.append(key, batch))
    assert sess.unresolved_mirror_batches() == ()
    assert _run(sess.load(key)) == batch
    [summary] = _run(sess.list_session_summaries("p"))
    assert summary["data"]["custom_title"] == "durable title"
    assert _run(sess.list_sessions("p"))[0]["mtime"] == summary["mtime"]
    assert sess._read_mirror_ledger()["history"][-1]["failures"][-1]["message"] == (
        "summary disk failed"
    )


def test_recovery_never_overwrites_a_divergent_summary_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    key = {"project_key": "p", "session_id": "sess"}
    batch = [_entry(uuid="u1", customTitle="expected")]

    first = Store.open(root, "s1")
    sess = MemstoreSessionStore(first, "brain")
    _leave_fsynced_transcript_before_sidecars(sess, key, batch, monkeypatch)
    summary_path = first.branch("brain").path(sess._summary_path("p", "sess"))
    divergent = b'{"session_id":"somebody-else","mtime":1,"data":{}}\n'
    _fsync_bytes(summary_path, divergent)
    first.close()

    second = Store.open(root, "s1")
    try:
        again = MemstoreSessionStore(second, "brain")
        unresolved = again.unresolved_mirror_batches()
        assert len(unresolved) == 1
        assert unresolved[0]["status"] == "failed"
        assert unresolved[0]["failures"][-1]["stage"] == "reconcile_sidecars"
        assert "divergent sidecar" in unresolved[0]["error"]
        assert summary_path.read_bytes() == divergent
    finally:
        second.close()
