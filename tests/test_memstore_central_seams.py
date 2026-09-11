"""Regressions at the memstore seams used by the central brain.

These are deliberately process-shaped rather than API-unit-shaped: the central
brain cleans up a worker through a freshly opened Store, holds catalog hits
while producers keep moving, and acknowledges inbox messages after restarts.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from taste.memstore import (
    BranchBusy,
    Hit,
    ObjectType,
    PublishError,
    Source,
    StaleBranch,
    Store,
)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    opened = Store.open(tmp_path / "repo", "s1")
    yield opened
    opened.close()


def test_fresh_store_captures_dirty_work_before_removing_a_worktree(
    tmp_path: Path,
) -> None:
    """A supervisor does not own the Branch object its dead child used."""
    root = tmp_path / "repo"
    worker = Store.open(root, "s1")
    branch = worker.branch("worker")
    branch.write("precious.txt", "the child died before checkpointing\n")
    branch.release()  # the process is gone; its worktree remains dirty
    worker.close()

    supervisor = Store.open(root, "s1")
    assert supervisor.view("worker").dirty_paths() == ["precious.txt"]

    supervisor.remove_branch("worker")

    assert supervisor.view("worker").head.read("precious.txt") == (
        "the child died before checkpointing\n"
    )
    assert not supervisor.worktree_path_for("worker").exists()
    supervisor.close()


def test_fresh_store_refuses_to_remove_a_live_workers_worktree(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    worker = Store.open(root, "s1")
    branch = worker.branch("worker")
    branch.write("in-flight.txt", "still working\n")
    supervisor = Store.open(root, "s1")

    with pytest.raises(BranchBusy):
        supervisor.remove_branch("worker")

    assert supervisor.worktree_path_for("worker").exists()
    assert supervisor.view("worker").dirty_paths() == ["in-flight.txt"]
    supervisor.close()
    worker.close()


def test_a_published_path_tracks_the_bytes_at_each_checkpoint(store: Store) -> None:
    producer = store.branch("producer")
    producer.write("out/data.txt", "v1\n")
    producer.publish("data", "out/data.txt", description="the live result")
    first = producer.checkpoint("publish v1")

    producer.write("out/data.txt", "v2\n")
    second = producer.checkpoint("revise the published result")

    old = first.manifest.entries["data"]
    current = second.manifest.entries["data"]
    assert current.blob == second.blob("out/data.txt")
    assert current.blob != old.blob
    assert current.published_at is not None
    assert old.published_at is not None
    assert current.published_at > old.published_at


def test_deleting_a_live_published_path_removes_it_from_the_catalog(store: Store) -> None:
    producer = store.branch("producer")
    producer.write("out/data.txt", "v1\n")
    producer.publish("data", "out/data.txt")
    producer.checkpoint("publish")

    producer.path("out/data.txt").unlink()
    state = producer.checkpoint("the live result no longer exists")

    assert "data" not in state.manifest
    assert store.catalog() == []


def test_a_catalog_hit_is_a_pinned_snapshot_when_the_producer_moves(
    store: Store,
) -> None:
    producer = store.branch("producer")
    producer.write("out/data.txt", "v1\n")
    producer.publish("data", "out/data.txt")
    producer.checkpoint("publish v1")
    (hit,) = store.catalog()

    producer.write("out/data.txt", "v2\n")
    producer.checkpoint("producer moves on")

    consumer = store.branch("consumer")
    consumer.adopt(hit, as_="input/data.txt")
    adopted = consumer.checkpoint("adopt the advertised snapshot")

    assert adopted.read("input/data.txt") == "v1\n"
    assert hit.entry.blob == hit.state.blob(hit.entry.path)
    assert adopted.meta.sources[0].blob == hit.entry.blob
    assert adopted.meta.sources[0].session == "s1"


def test_adopt_rejects_a_hit_whose_advertised_blob_does_not_match(
    store: Store,
) -> None:
    producer = store.branch("producer")
    producer.write("out/data.txt", "v1\n")
    producer.publish("data", "out/data.txt")
    producer.checkpoint("publish")
    (hit,) = store.catalog()
    corrupt = Hit(
        branch=hit.branch,
        state=hit.state,
        entry=replace(hit.entry, blob="0" * 40),
        score=hit.score,
    )

    with pytest.raises(PublishError, match="advertised blob"):
        store.branch("consumer").adopt(corrupt)


def test_modifying_an_adopted_file_makes_the_new_state_its_origin(
    store: Store,
) -> None:
    producer = store.branch("producer")
    producer.write("out/data.txt", "upstream\n")
    upstream = producer.checkpoint("produce")

    consumer = store.branch("consumer")
    consumer.adopt(upstream, "out/data.txt", as_="input/data.txt")
    consumer.write("input/data.txt", "derived locally\n")
    derived = consumer.checkpoint("adopt, then derive")

    source = derived.meta.sources[0]
    assert source.state == upstream.id
    assert source.blob == upstream.blob("out/data.txt")
    assert store.origin(derived, "input/data.txt") == derived


def test_two_adopts_to_one_path_follow_the_source_whose_blob_won(
    store: Store,
) -> None:
    first = store.branch("first")
    first.write("value.txt", "first\n")
    first_state = first.checkpoint("first value")
    second = store.branch("second")
    second.write("value.txt", "second\n")
    second_state = second.checkpoint("second value")

    consumer = store.branch("consumer")
    consumer.adopt(first_state, "value.txt", as_="input.txt")
    consumer.adopt(second_state, "value.txt", as_="input.txt")
    state = consumer.checkpoint("the second adoption wins")

    assert len(state.meta.sources) == 2
    assert store.origin(state, "input.txt") == second_state


def test_source_deserializes_states_written_before_pinned_identity_fields() -> None:
    old = Source.from_dict(
        {"branch": "producer", "state": "abc", "path": "x", "as_path": "y"}
    )

    assert old.blob is None
    assert old.session == ""


def test_inbox_acknowledgement_cannot_rewind_or_leave_its_chain(
    store: Store,
) -> None:
    first = store.send("worker", {"n": 1})
    second = store.send("worker", {"n": 2})

    store.mark_inbox_seen("worker", second)
    store.mark_inbox_seen("worker", second)  # idempotent
    assert store.inbox("worker") == []

    with pytest.raises(ValueError, match="rewind"):
        store.mark_inbox_seen("worker", first)
    unrelated = store.branch("unrelated").head.id
    with pytest.raises(ValueError, match="not in the inbox"):
        store.mark_inbox_seen("worker", unrelated)
    assert store.inbox("worker") == []


def test_inbox_acknowledgement_reports_a_lost_compare_and_swap(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    message = store.send("worker", {"n": 1})
    monkeypatch.setattr(store.backend, "cas_update_ref", lambda *args: False)

    with pytest.raises(StaleBranch, match="seen marker"):
        store.mark_inbox_seen("worker", message)


def test_inbox_rejects_non_json_numbers_before_publishing(store: Store) -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        store.send("worker", {"cost": float("nan")})
    assert store.inbox("worker") == []


def test_inbox_cursor_without_its_message_log_fails_closed(store: Store) -> None:
    seen_ref = f"{store.SEEN_REF}/{store.session}/worker"
    marker = store.branch("unrelated").head.id
    assert store.backend.cas_update_ref(seen_ref, marker, None)

    with pytest.raises(ValueError, match="cursor but no message log"):
        store.inbox("worker")


def test_unpublish_is_not_discarded_by_a_merge(store: Store) -> None:
    ours = store.branch("ours")
    ours.write("published.txt", "old\n")
    ours.publish("published", "published.txt", type=ObjectType.FILE)
    base = ours.checkpoint("publish")

    theirs = store.branch("theirs", from_state=base)
    theirs.write("theirs.txt", "new\n")
    theirs.checkpoint("their work")

    ours.unpublish("published")
    merged = ours.merge(theirs, reason="merge while unpublishing")

    assert merged.ok
    assert "published" not in ours.head.manifest
