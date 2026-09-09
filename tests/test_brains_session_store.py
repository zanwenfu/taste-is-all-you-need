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

from claude_agent_sdk.testing import run_session_store_conformance  # noqa: E402

from taste.brains.session_store import MAIN_SUBPATH, MemstoreSessionStore  # noqa: E402


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


def test_a_transcript_is_real_history_not_a_side_file(store: Store) -> None:
    """The whole reason for this adapter: a brain's conversation and its
    artifacts are one history, so they move together."""
    sess = MemstoreSessionStore(store, "brain")
    key = {"project_key": "proj", "session_id": "s-1"}
    _run(sess.append(key, [_entry(uuid="a", text="thinking")]))

    branch = store.branch("brain")
    files = branch.head.files()
    assert any(f.endswith(f"{MAIN_SUBPATH}.jsonl") for f in files), files
    assert branch.head.meta.reason.startswith("transcript:")


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
    good = store.branch("brain").head

    _run(sess.append(key, [_entry(uuid="b", text="the mistake")]))
    bad = store.branch("brain").head
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
