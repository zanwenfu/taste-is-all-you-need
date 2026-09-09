"""Filenames are content too.

The audit fix made file *contents* byte-faithful, but paths crossed the git
boundary as text that git had already quoted: ``core.quotePath`` defaults on,
so ``ls-tree`` and ``status --porcelain`` reported ``"caf\\303\\251.txt"`` for
``café.txt``. Every comparison against a real path then failed, and a memory
layer holding arbitrary agent-authored files sees non-ASCII names as ordinary
input.

The parsers now read NUL-delimited output, which git never quotes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taste.memstore import ObjectType, PublishError, Store

AWKWARD = [
    pytest.param("café.txt", id="non-ascii"),
    pytest.param("日本語.txt", id="cjk"),
    pytest.param("a b.txt", id="space"),
    pytest.param("emoji-🙂.txt", id="emoji"),
    pytest.param("back\\slash.txt", id="backslash"),
]


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "repo", "s1")
    yield s
    s.close()


@pytest.mark.parametrize("name", AWKWARD)
def test_an_awkward_path_can_be_published_and_found(store: Store, name: str) -> None:
    b = store.branch("worker")
    b.write(name, "one\n")
    b.publish("artifact", name, type=ObjectType.FILE, description="awkward")
    st = b.checkpoint("publish an awkward name")

    assert st.read(name) == "one\n"
    assert st.blob(name) is not None, "the blob is invisible, so provenance is too"
    assert name in st.files()
    assert store.origin(st, name) is not None
    assert st.manifest.entries["artifact"].path == name


@pytest.mark.parametrize("name", AWKWARD)
def test_an_awkward_path_is_reported_as_dirty_under_its_real_name(
    store: Store, name: str
) -> None:
    b = store.branch("worker")
    b.checkpoint("clean")
    b.write(name, "changed\n")

    dirty = b.dirty_paths()
    assert name in dirty, f"got {dirty!r}"
    assert (b.worktree / name).exists(), "the reported path must exist on disk"


def test_an_awkward_path_shows_up_in_a_diff(store: Store) -> None:
    b = store.branch("worker")
    b.write("café.txt", "one\n")
    first = b.checkpoint("one")
    b.write("café.txt", "two\n")
    second = b.checkpoint("two")

    assert "café.txt" in second.diff(first).paths()


def test_a_record_at_an_awkward_path_still_merges_as_a_record(store: Store) -> None:
    """A quoted name never matched its manifest entry, so records silently
    degraded to opaque files and stopped merging by key."""
    b = store.branch("worker")
    b.checkpoint("base", records={"café.json": {"a": 1}})
    b.publish("notes", "café.json", type=ObjectType.RECORD)
    base = b.checkpoint("publish the record")

    other = store.branch("other", from_state=base)
    other.checkpoint("theirs", records={"café.json": {"a": 1, "b": 2}})
    b.checkpoint("ours", records={"café.json": {"a": 1, "c": 3}})

    result = b.merge(other, reason="merge the record")
    assert result.ok, f"conflicts: {result.conflicts}"
    assert b.head.record("café.json") == {"a": 1, "b": 2, "c": 3}


def test_a_rejected_publish_does_not_wedge_the_branch(store: Store) -> None:
    """A publish naming a path that is not there is a rejected request, not a
    permanent inability to commit.

    The entry used to stay pending, so every later checkpoint re-raised -- and
    since ``_capture`` checkpoints before rollback and merge, one bad publish
    left the branch unable to save anything at all.
    """
    b = store.branch("worker")
    b.write("real.txt", "here\n")
    b.publish("ghost", "not-there.txt")

    with pytest.raises(PublishError):
        b.checkpoint("publishing something absent")

    st = b.checkpoint("the branch still works")
    assert st.read("real.txt") == "here\n"
    assert not b.is_dirty()
