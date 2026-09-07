"""Typed merge: the type decides. Conflicts are values, not exceptions."""

from __future__ import annotations

from pathlib import Path

import pytest

from taste.memstore import ObjectType, Store, Transcript
from taste.memstore.merge import merge_records
from taste.memstore.store import NOTES


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "repo", "s1")
    yield s
    s.close()


# ------------------------------------------------------------------ merge_records


def test_records_disjoint_keys_combine() -> None:
    merged, clashes = merge_records({}, {"a": 1}, {"b": 2})
    assert merged == {"a": 1, "b": 2} and clashes == []


def test_records_one_side_changed_is_taken() -> None:
    merged, clashes = merge_records({"a": 1, "b": 2}, {"a": 9, "b": 2}, {"a": 1, "b": 2})
    assert merged == {"a": 9, "b": 2} and clashes == []
    merged, clashes = merge_records({"a": 1, "b": 2}, {"a": 1, "b": 2}, {"a": 1, "b": 7})
    assert merged == {"a": 1, "b": 7} and clashes == []


def test_records_deletion_on_one_side_is_taken() -> None:
    merged, clashes = merge_records({"a": 1, "b": 2}, {"b": 2}, {"a": 1, "b": 2})
    assert merged == {"b": 2} and clashes == []


def test_records_same_change_both_sides_agrees() -> None:
    merged, clashes = merge_records({"a": 1}, {"a": 5}, {"a": 5})
    assert merged == {"a": 5} and clashes == []
    merged, clashes = merge_records({"a": 1}, {}, {})
    assert merged == {} and clashes == []


def test_records_different_changes_clash_per_key() -> None:
    merged, clashes = merge_records({"a": 1, "b": 1}, {"a": 2, "b": 3}, {"a": 4, "b": 3})
    assert clashes == ["a"]
    assert merged["b"] == 3  # the agreed key still merges
    merged, clashes = merge_records({"a": 1}, {"a": 2}, {})  # modify vs delete
    assert clashes == ["a"]


def test_records_non_object_values() -> None:
    assert merge_records([1], [1, 2], [1]) == ([1, 2], [])
    assert merge_records(1, 2, 3) == (2, ["<value>"])


# ------------------------------------------------------------------ merge_branches


def _two_branches(store: Store):  # type: ignore[no-untyped-def]
    a = store.branch("a", producer="A")
    a.write("shared.json", "{}\n")
    a.publish("shared", "shared.json", type=ObjectType.RECORD, description="shared record")
    base = a.checkpoint("base", transcript=Transcript().append(role="assistant", content="A's ctx"))
    b = store.branch("b", from_state=base, producer="B")
    return a, b, base


def test_disjoint_files_merge_clean(store: Store) -> None:
    a, b, _ = _two_branches(store)
    a.write("a.txt", "from a\n")
    a.publish("a-out", "a.txt", description="a's output")
    a_head = a.checkpoint("a writes")
    b.write("b.txt", "from b\n")
    b.publish("b-out", "b.txt", description="b's output")
    b_head = b.checkpoint("b writes")

    res = a.merge(b, reason="bring in b")
    assert res.ok and res.state is not None
    m = res.state
    assert m.meta.kind == "merge"
    assert m.meta.parents == (a_head.id, b_head.id)
    assert m.read("a.txt") == "from a\n" and m.read("b.txt") == "from b\n"
    assert set(m.manifest.entries) == {"shared", "a-out", "b-out"}  # union
    assert m.transcript == a_head.transcript  # transcripts never merge
    assert (a.worktree / "b.txt").read_text() == "from b\n"  # working tree updated
    assert a.head == m


def test_records_merge_by_key(store: Store) -> None:
    a, b, _ = _two_branches(store)
    a.checkpoint("a adds k1", records={"shared.json": {"k1": 1}})
    b.checkpoint("b adds k2", records={"shared.json": {"k2": 2}})
    res = a.merge(b, reason="combine")
    assert res.ok
    assert res.state.record("shared.json") == {"k1": 1, "k2": 2}


def test_record_key_clash_is_a_conflict_value(store: Store) -> None:
    a, b, _ = _two_branches(store)
    a_head = a.checkpoint("a sets k", records={"shared.json": {"k": "a"}})
    b.checkpoint("b sets k", records={"shared.json": {"k": "b"}})
    res = a.merge(b, reason="combine")
    assert not res.ok and res.state is None
    (c,) = res.conflicts
    assert c.path == "shared.json" and c.type is ObjectType.RECORD and "k" in c.detail
    # nothing published; the conflict is attached to our head for a brain to find
    assert a.head == a_head
    assert a.head.conflicts[0].path == "shared.json"
    assert store.backend.note_get(NOTES["conflicts"], a_head.id) is not None


def test_file_content_conflict(store: Store) -> None:
    a, b, _ = _two_branches(store)
    a.write("f.txt", "line A\n")
    a.checkpoint("a")
    b.write("f.txt", "line B\n")
    b.checkpoint("b")
    res = a.merge(b, reason="combine")
    assert not res.ok
    (c,) = res.conflicts
    assert c.path == "f.txt" and c.type is ObjectType.FILE


def test_resolved_merge_records_both_parents(store: Store) -> None:
    a, b, _ = _two_branches(store)
    a.write("f.txt", "line A\n")
    a_head = a.checkpoint("a")
    b.write("f.txt", "line B\n")
    b_head = b.checkpoint("b")
    assert not a.merge(b, reason="try").ok
    # A brain resolves: writes what it wants, then records the merge.
    a.write("f.txt", "line A\nline B\n")
    res = a.merge(b, reason="resolved by hand", resolved=True)
    assert res.ok
    assert res.state.meta.parents == (a_head.id, b_head.id)
    assert res.state.read("f.txt") == "line A\nline B\n"
    # and now b is in a's history, so merging again is a no-op
    assert a.merge(b, reason="again").state == res.state


def test_merge_of_already_merged_branch_is_noop(store: Store) -> None:
    a, b, _ = _two_branches(store)
    b.write("x", "1")
    b.checkpoint("b")
    first = a.merge(b, reason="once").state
    assert a.merge(b, reason="twice").state == first
