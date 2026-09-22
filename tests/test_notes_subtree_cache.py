"""Reuse immutable note subtrees while validating every newly selected tree."""

from collections import Counter
from tempfile import TemporaryFile

import pytest

from taste.memstore import Store
from taste.memstore import backend as module

NAMESPACE = "refs/notes/subtree-test"
SUFFIX = "1" * 38


@pytest.fixture
def backend(tmp_path):
    store = Store.open(tmp_path / "repo", "note-subtrees")
    try:
        yield store.backend
    finally:
        store.close()


def tree(backend, entries):
    with TemporaryFile() as handle:
        for name, kind, sha in sorted(entries):
            mode = "040000" if kind == "tree" else "100644"
            handle.write(f"{mode} {kind} {sha}\t{name}\n".encode())
        handle.seek(0)
        return backend.repo.git.mktree(istream=handle)


def publish(backend, entries):
    before = backend.ref_sha(NAMESPACE)
    sha = backend.commit_tree(tree(backend, entries), [before] if before else [], "note update")
    assert backend.cas_update_ref(NAMESPACE, sha, before)
    return sha


def test_unchanged_subtree_is_not_reloaded_when_another_note_changes(backend, monkeypatch):
    stable = tree(backend, [(SUFFIX, "blob", backend.hash_blob("stable"))])
    stream = backend.repo.odb.stream
    loads = Counter()

    def counted(sha):
        loads[sha.hex()] += 1
        return stream(sha)

    monkeypatch.setattr(backend.repo.odb, "stream", counted)
    for version in range(6):
        changed = tree(backend, [(SUFFIX, "blob", backend.hash_blob(str(version)))])
        publish(backend, [("ab", "tree", stable), ("cd", "tree", changed)])
        assert backend.note_get(NAMESPACE, "ab" + SUFFIX) == "stable"
        assert backend.note_get(NAMESPACE, "cd" + SUFFIX) == str(version)
    assert loads[stable] == 1, loads[stable]


def test_duplicate_flat_and_nested_target_is_rejected_even_for_identical_bytes(backend):
    blob = backend.hash_blob("same bytes")
    subtree = tree(backend, [(SUFFIX, "blob", blob)])
    publish(backend, [("ab", "tree", subtree), ("ab" + SUFFIX, "blob", blob)])
    with pytest.raises(ValueError, match="duplicate target"):
        backend.note_get(NAMESPACE, "ab" + SUFFIX)


def test_subtree_relocation_uses_new_prefix_and_validates_remaining_hash_width(backend):
    subtree = tree(backend, [(SUFFIX, "blob", backend.hash_blob("note"))])
    publish(backend, [("ab", "tree", subtree)])
    assert backend.note_get(NAMESPACE, "ab" + SUFFIX) == "note"
    publish(backend, [("cd", "tree", subtree)])
    assert backend.note_get(NAMESPACE, "ab" + SUFFIX) is None
    assert backend.note_get(NAMESPACE, "cd" + SUFFIX) == "note"
    publish(backend, [("c", "tree", subtree)])
    with pytest.raises(ValueError, match="invalid target path"):
        backend.note_get(NAMESPACE, "cd" + SUFFIX)


def test_empty_nested_tree_is_rejected_after_a_warm_read(backend):
    valid = tree(backend, [(SUFFIX, "blob", backend.hash_blob("note"))])
    publish(backend, [("ab", "tree", valid)])
    assert backend.note_get(NAMESPACE, "ab" + SUFFIX) == "note"
    publish(backend, [("ab", "tree", valid), ("cd", "tree", module.EMPTY_TREE)])
    with pytest.raises(ValueError, match="invalid tree prefix"):
        backend.note_get(NAMESPACE, "ab" + SUFFIX)


def test_subtree_cache_respects_entry_and_target_bounds_and_can_be_cleared(backend, monkeypatch):
    monkeypatch.setattr(module, "_NOTE_SUBTREE_CACHE_MAX_ENTRIES", 2)
    monkeypatch.setattr(module, "_NOTE_SUBTREE_CACHE_MAX_TARGETS", 2)
    for version in range(5):
        subtree = tree(backend, [(SUFFIX, "blob", backend.hash_blob(str(version)))])
        publish(backend, [("ab", "tree", subtree)])
        assert backend.note_get(NAMESPACE, "ab" + SUFFIX) == str(version)
    assert len(backend._note_subtree_cache) <= 2
    assert backend._note_subtree_cache_size <= 2
    backend._clear_object_caches()
    assert not backend._note_subtree_cache and backend._note_subtree_cache_size == 0
    assert backend.note_get(NAMESPACE, "ab" + SUFFIX) == "4"
