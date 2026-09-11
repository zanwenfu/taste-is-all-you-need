"""Product delivery is a byte-exact projection, not a raw worker merge."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from taste.brains.delivery import (
    DELIVERY_NOTES,
    DeliveryIdentityConflict,
    DeliveryRecoveryRequired,
    InvalidArtifactPath,
    ReservedArtifactPath,
    deliver_product,
)
from taste.memstore import ObjectType, Store, Transcript


@pytest.fixture
def store(tmp_path: Path) -> Store:
    opened = Store.open(tmp_path / "repo", "delivery-test")
    yield opened
    opened.close()


def _fork(store: Store):  # type: ignore[no-untyped-def]
    integration = store.branch("integration", producer="central")
    integration.write("app.py", "VALUE = 'base'\n")
    integration.write("kept.txt", "base file\n")
    integration.publish("app", "app.py", type=ObjectType.FILE)
    base = integration.checkpoint("integration base")
    worker = store.branch("worker", from_state=base, producer="worker")
    return integration, worker, base


def test_delivery_uses_the_pinned_state_and_only_selected_artifacts(store: Store) -> None:
    integration, worker, base = _fork(store)
    worker.write("app.py", "VALUE = 'pinned'\n")
    worker.write("scratch.txt", "exploration, not an output\n")
    worker.write("contract.json", '{"private":"worker"}\n')
    worker.write("assignment.json", '{"private":"assignment"}\n')
    worker.write("worker-report.json", '{"private":"report"}\n')
    worker.write("sdk-sessions/project/session/transcript.jsonl", "private reasoning\n")
    worker.write(".taste/runtime/events.jsonl", "private events\n")
    worker.publish("worker-app", "app.py", description="selected output")
    pinned = worker.checkpoint(
        "worker finished",
        transcript=Transcript().append(role="assistant", content="private thought"),
    )

    # The branch moves after the report's final_state_id was captured.  A
    # delivery implementation that peeks at worker.head will ship this value.
    worker.write("app.py", "VALUE = 'later and wrong for this delivery'\n")
    worker.write("late.txt", "also later\n")
    worker.checkpoint("worker kept moving")

    result = deliver_product(
        store,
        integration,
        delivery_id="assignment-7/attempt-1",
        final_state_id=pinned.id,
        base_state_id=base.id,
        artifact_paths=("app.py",),
    )

    assert result.ok
    assert result.integration_state.read("app.py") == "VALUE = 'pinned'\n"
    assert result.integration_state.read("kept.txt") == "base file\n"
    assert result.integration_state.read("scratch.txt") is None
    assert result.integration_state.read("late.txt") is None
    assert result.record.source_state_id == pinned.id
    assert result.record.entries[0].blob == pinned.blob("app.py")

    projected_files = set(result.projection.files())
    assert projected_files == {"app.py", "kept.txt"}
    assert result.projection.transcript.turns == ()
    assert set(result.projection.manifest.entries) == {"app", "worker-app"}
    assert pinned.id in {state.id for state in store.provenance(result.projection)}


def test_delivery_branch_movement_cannot_change_the_state_being_merged(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration, worker, base = _fork(store)
    worker.write("selected.py", "SELECTED = True\n")
    final = worker.checkpoint("selected output")
    real_merge = integration.merge
    raced_projection: list[str] = []

    def move_delivery_then_merge(other: Any, *, reason: str, resolved: bool = False):  # type: ignore[no-untyped-def]
        raced_projection.append(other.head.id)
        mutable_delivery = store.branch(other.name)
        mutable_delivery.write("evil.py", "EVIL = 'unselected later state'\n")
        mutable_delivery.checkpoint("delivery branch moved after validation")
        mutable_delivery.close()
        return real_merge(other, reason=reason, resolved=resolved)

    monkeypatch.setattr(integration, "merge", move_delivery_then_merge)
    result = deliver_product(
        store,
        integration,
        delivery_id="pinned-merge-view",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("selected.py",),
    )

    assert result.ok
    assert raced_projection == [result.projection.id]
    assert result.integration_state.meta.parents[1] == result.projection.id
    assert result.integration_state.read("selected.py") == "SELECTED = True\n"
    assert result.integration_state.read("evil.py") is None

    retry = deliver_product(
        store,
        integration,
        delivery_id="pinned-merge-view",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("selected.py",),
    )
    assert retry.ok and retry.reused_projection and retry.reused_outcome
    assert retry.projection.id == result.projection.id


def test_private_changes_on_both_sides_neither_leak_nor_conflict(store: Store) -> None:
    integration, worker, base = _fork(store)
    worker.write("new.py", "WORKER = True\n")
    worker.write("contract.json", "worker contract\n")
    worker.write("sdk-sessions/p/s/log.jsonl", "worker transcript\n")
    final = worker.checkpoint("worker output")

    # Even a pre-existing private/control conflict is outside the product
    # merge.  The target is sanitized in its own durable checkpoint first.
    integration.write("contract.json", "different central control bytes\n")
    integration.write("sdk-sessions/central/s/log.jsonl", "different transcript\n")
    integration.checkpoint("accidental private files on integration")

    result = deliver_product(
        store,
        integration,
        delivery_id="private-conflict",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("new.py",),
    )

    assert result.ok
    assert result.integration_state.read("new.py") == "WORKER = True\n"
    assert not any(
        path == "contract.json" or path.startswith("sdk-sessions/")
        for path in result.integration_state.files()
    )
    assert not any(
        path == "contract.json" or path.startswith("sdk-sessions/")
        for path in result.projection.files()
    )


def test_control_file_in_the_recorded_base_is_stripped_on_both_sides(store: Store) -> None:
    integration = store.branch("integration", producer="central")
    integration.write("app.py", "BASE = True\n")
    integration.write("contract.json", "obsolete shared scaffolding\n")
    base = integration.checkpoint("legacy base containing control state")
    worker = store.branch("worker", from_state=base)
    worker.write("app.py", "WORKER = True\n")
    worker.write("contract.json", "worker-private revision\n")
    final = worker.checkpoint("worker output")
    integration.write("contract.json", "central-private revision\n")
    integration.checkpoint("central control revision")

    result = deliver_product(
        store,
        integration,
        delivery_id="legacy-control-base",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("app.py",),
    )

    assert result.ok
    assert result.integration_state.read("app.py") == "WORKER = True\n"
    assert result.integration_state.read("contract.json") is None
    assert result.projection.read("contract.json") is None


def test_worker_that_absorbed_a_newer_target_is_not_projected_from_a_stale_base(
    store: Store,
) -> None:
    integration, worker, base = _fork(store)
    integration.write("central.py", "CENTRAL = 2\n")
    integration.checkpoint("target advanced")
    assert worker.merge(integration.view, reason="worker absorbed target").ok
    worker.write("app.py", "WORKER = 3\n")
    final = worker.checkpoint("worker output after rebase-like merge")

    with pytest.raises(ValueError, match="exact worker/target merge base"):
        deliver_product(
            store,
            integration,
            delivery_id="stale-recorded-base",
            final_state_id=final.id,
            base_state_id=base.id,
            artifact_paths=("app.py",),
        )


def test_criss_cross_history_with_two_best_merge_bases_is_refused(store: Store) -> None:
    rootline = store.branch("rootline")
    rootline.write("app.py", "VALUE = 'root'\n")
    root = rootline.checkpoint("shared root")
    side_a = store.branch("side-a", from_state=root)
    side_a.write("a.txt", "from A\n")
    a1 = side_a.checkpoint("A1")
    side_b = store.branch("side-b", from_state=root)
    side_b.write("b.txt", "from B\n")
    b1 = side_b.checkpoint("B1")

    worker = store.branch("worker", from_state=a1)
    assert worker.merge(side_b.view, reason="A2 absorbs B1").ok
    integration = store.branch("integration", from_state=b1)
    assert integration.merge(side_a.view, reason="B2 absorbs A1").ok
    worker.write("app.py", "VALUE = 'worker'\n")
    final = worker.checkpoint("worker changes selected app only")

    with pytest.raises(ValueError, match="unique exact worker/target merge base"):
        deliver_product(
            store,
            integration,
            delivery_id="criss-cross",
            final_state_id=final.id,
            base_state_id=b1.id,
            artifact_paths=("app.py",),
        )


def test_genuine_product_conflict_remains_a_memstore_conflict(store: Store) -> None:
    integration, worker, base = _fork(store)
    worker.write("app.py", "VALUE = 'worker'\n")
    final = worker.checkpoint("worker changes product")
    integration.write("app.py", "VALUE = 'central'\n")
    integration.checkpoint("central changes product")

    result = deliver_product(
        store,
        integration,
        delivery_id="product-conflict",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("app.py",),
    )

    assert not result.ok and result.integration_state is None
    assert [conflict.path for conflict in result.conflicts] == ["app.py"]
    assert integration.head.meta.kind == "conflict"
    assert integration.head.meta.merged == result.projection.id


def test_selected_record_keeps_its_type_and_uses_typed_merge(store: Store) -> None:
    integration = store.branch("integration")
    integration.write("config.json", '{}\n')
    integration.publish("config", "config.json", type=ObjectType.RECORD)
    base = integration.checkpoint("record base")
    worker = store.branch("worker", from_state=base)
    worker.checkpoint("worker record key", records={"config.json": {"worker": 1}})
    final = worker.head
    integration.checkpoint("central record key", records={"config.json": {"central": 2}})

    result = deliver_product(
        store,
        integration,
        delivery_id="typed-record",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("config.json",),
    )

    assert result.ok
    assert result.integration_state.record("config.json") == {"central": 2, "worker": 1}
    assert result.projection.manifest.type_of("config.json") is ObjectType.RECORD


def test_successful_retry_is_a_noop(store: Store) -> None:
    integration, worker, base = _fork(store)
    worker.write("new.py", "NEW = 1\n")
    final = worker.checkpoint("worker output")
    first = deliver_product(
        store,
        integration,
        delivery_id="retry-success",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("new.py",),
    )
    first_head = integration.head.id
    first_projection = first.projection.id

    second = deliver_product(
        store,
        integration,
        delivery_id="retry-success",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("new.py",),
    )

    assert second.ok and second.reused_projection and second.reused_outcome
    assert integration.head.id == first_head
    assert second.projection.id == first_projection


def test_retry_identity_survives_process_store_reopen(tmp_path: Path) -> None:
    root = tmp_path / "durable-repo"
    first_store = Store.open(root, "durable-delivery")
    integration, worker, base = _fork(first_store)
    worker.write("durable.py", "DURABLE = True\n")
    final = worker.checkpoint("durable output")
    first = deliver_product(
        first_store,
        integration,
        delivery_id="survives-reopen",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("durable.py",),
    )
    integration_id = integration.head.id
    projection_id = first.projection.id
    final_id, base_id = final.id, base.id
    first_store.close()

    second_store = Store.open(root, "durable-delivery")
    try:
        reopened_target = second_store.branch("integration")
        second = deliver_product(
            second_store,
            reopened_target,
            delivery_id="survives-reopen",
            final_state_id=final_id,
            base_state_id=base_id,
            artifact_paths=("durable.py",),
        )
        assert second.ok and second.reused_projection and second.reused_outcome
        assert reopened_target.head.id == integration_id
        assert second.projection.id == projection_id
    finally:
        second_store.close()


def test_crash_after_source_only_seed_cannot_rebind_identity(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration, worker, base = _fork(store)
    worker.write("one.py", "ONE = 1\n")
    worker.write("two.py", "TWO = 2\n")
    final = worker.checkpoint("two possible outputs")
    real_note_set = store.backend.note_set

    def crash_before_projection_note(
        namespace: str,
        commit: str,
        text: str,
        *,
        overwrite: bool = True,
    ) -> None:
        if namespace == DELIVERY_NOTES:
            raise RuntimeError("injected death after delivery seed")
        real_note_set(namespace, commit, text, overwrite=overwrite)

    monkeypatch.setattr(store.backend, "note_set", crash_before_projection_note)
    with pytest.raises(RuntimeError, match="injected death"):
        deliver_product(
            store,
            integration,
            delivery_id="seed-crash",
            final_state_id=final.id,
            base_state_id=base.id,
            artifact_paths=("one.py",),
        )
    monkeypatch.setattr(store.backend, "note_set", real_note_set)

    # The public seed knows only the source.  The earlier atomic claim is what
    # prevents rebinding that seed to another path after restart.
    with pytest.raises(DeliveryIdentityConflict, match="claimed for different work"):
        deliver_product(
            store,
            integration,
            delivery_id="seed-crash",
            final_state_id=final.id,
            base_state_id=base.id,
            artifact_paths=("two.py",),
        )

    resumed = deliver_product(
        store,
        integration,
        delivery_id="seed-crash",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("one.py",),
    )
    assert resumed.ok and resumed.integration_state.read("one.py") == "ONE = 1\n"
    assert resumed.integration_state.read("two.py") is None


def test_retry_preserves_ambiguous_publish_before_reset_tree_and_fails_closed(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration, worker, base = _fork(store)
    worker.write("new.py", "NEW = True\n")
    final = worker.checkpoint("worker output")
    real_reset = integration.backend.reset_hard_to_head

    def die_before_reset(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("injected death before target reset")

    monkeypatch.setattr(integration.backend, "reset_hard_to_head", die_before_reset)
    with pytest.raises(RuntimeError, match="before target reset"):
        deliver_product(
            store,
            integration,
            delivery_id="target-reset-crash",
            final_state_id=final.id,
            base_state_id=base.id,
            artifact_paths=("new.py",),
        )
    assert integration.head.read("new.py") == "NEW = True\n"
    assert "new.py" in integration.dirty_paths()
    merge_state = integration.head
    monkeypatch.setattr(integration.backend, "reset_hard_to_head", real_reset)

    with pytest.raises(DeliveryRecoveryRequired, match="preserved dirty paths"):
        deliver_product(
            store,
            integration,
            delivery_id="target-reset-crash",
            final_state_id=final.id,
            base_state_id=base.id,
            artifact_paths=("new.py",),
        )
    captured = integration.head
    assert captured.parents[0] == merge_state
    assert captured.read("new.py") is None
    assert merge_state.read("new.py") == "NEW = True\n"
    assert integration.dirty_paths() == []


def test_merge_recovery_never_resets_additional_dirty_work(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    integration, worker, base = _fork(store)
    worker.write("new.py", "NEW = True\n")
    final = worker.checkpoint("worker output")
    real_reset = integration.backend.reset_hard_to_head
    monkeypatch.setattr(
        integration.backend,
        "reset_hard_to_head",
        lambda: (_ for _ in ()).throw(RuntimeError("injected reset death")),
    )
    with pytest.raises(RuntimeError, match="reset death"):
        deliver_product(
            store,
            integration,
            delivery_id="dirty-recovery",
            final_state_id=final.id,
            base_state_id=base.id,
            artifact_paths=("new.py",),
        )
    merge_head = integration.head.id
    monkeypatch.setattr(integration.backend, "reset_hard_to_head", real_reset)
    integration.write("precious.txt", "unique work after the crash\n")

    with pytest.raises(DeliveryRecoveryRequired, match="preserved dirty paths"):
        deliver_product(
            store,
            integration,
            delivery_id="dirty-recovery",
            final_state_id=final.id,
            base_state_id=base.id,
            artifact_paths=("new.py",),
        )
    assert integration.head.id != merge_head
    assert integration.head.read("precious.txt") == "unique work after the crash\n"
    assert integration.head.read("new.py") is None
    assert integration.dirty_paths() == []


def test_conflicted_retry_does_not_publish_duplicate_conflict_states(store: Store) -> None:
    integration, worker, base = _fork(store)
    worker.write("app.py", "VALUE = 'worker'\n")
    final = worker.checkpoint("worker output")
    integration.write("app.py", "VALUE = 'central'\n")
    integration.checkpoint("central output")
    first = deliver_product(
        store,
        integration,
        delivery_id="retry-conflict",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("app.py",),
    )
    conflict_head = integration.head.id

    second = deliver_product(
        store,
        integration,
        delivery_id="retry-conflict",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("app.py",),
    )

    assert not first.ok and not second.ok
    assert second.reused_projection and second.reused_outcome
    assert integration.head.id == conflict_head
    assert second.conflicts == first.conflicts


def test_retry_fails_closed_if_delivered_product_was_rolled_back(store: Store) -> None:
    integration, worker, base = _fork(store)
    worker.write("new.py", "NEW = True\n")
    final = worker.checkpoint("worker output")
    first = deliver_product(
        store,
        integration,
        delivery_id="rolled-back-delivery",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("new.py",),
    )
    assert first.ok

    integration.rollback(base, "operator rolls back delivered product")
    assert integration.head.read("new.py") is None

    with pytest.raises(DeliveryRecoveryRequired, match="no longer matches"):
        deliver_product(
            store,
            integration,
            delivery_id="rolled-back-delivery",
            final_state_id=final.id,
            base_state_id=base.id,
            artifact_paths=("new.py",),
        )


def test_retry_fails_closed_if_conflict_note_was_deleted(store: Store) -> None:
    integration, worker, base = _fork(store)
    worker.write("app.py", "VALUE = 'worker'\n")
    final = worker.checkpoint("worker output")
    integration.write("app.py", "VALUE = 'central'\n")
    integration.checkpoint("central output")
    first = deliver_product(
        store,
        integration,
        delivery_id="deleted-conflict-note",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("app.py",),
    )
    assert first.conflicts
    store.backend.repo.git.notes(
        "--ref=refs/notes/taste/conflicts",
        "remove",
        integration.head.id,
    )

    with pytest.raises(DeliveryRecoveryRequired, match="lost its durable conflict evidence"):
        deliver_product(
            store,
            integration,
            delivery_id="deleted-conflict-note",
            final_state_id=final.id,
            base_state_id=base.id,
            artifact_paths=("app.py",),
        )


def test_retry_fails_closed_if_conflict_note_was_rewritten(store: Store) -> None:
    integration, worker, base = _fork(store)
    worker.write("app.py", "VALUE = 'worker'\n")
    final = worker.checkpoint("worker output")
    integration.write("app.py", "VALUE = 'central'\n")
    integration.checkpoint("central output")
    first = deliver_product(
        store,
        integration,
        delivery_id="rewritten-conflict-note",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("app.py",),
    )
    assert first.conflicts
    namespace = "refs/notes/taste/conflicts"
    raw = store.backend.note_get(namespace, integration.head.id)
    assert raw is not None
    forged = json.loads(raw)
    forged[0]["detail"] = "forged conflict evidence"
    store.backend.note_set(
        namespace,
        integration.head.id,
        json.dumps(forged),
        overwrite=True,
    )

    with pytest.raises(DeliveryRecoveryRequired, match="lost its durable conflict evidence"):
        deliver_product(
            store,
            integration,
            delivery_id="rewritten-conflict-note",
            final_state_id=final.id,
            base_state_id=base.id,
            artifact_paths=("app.py",),
        )


def test_delivery_merge_and_target_checkpoint_share_one_worktree_lock(
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    integration, worker, base = _fork(store)
    worker.write("new.py", "NEW = True\n")
    final = worker.checkpoint("worker output")
    reset_entered = threading.Event()
    permit_reset = threading.Event()
    checkpoint_done = threading.Event()
    real_reset = integration.backend.reset_hard_to_head
    errors: list[BaseException] = []

    def blocked_reset() -> None:
        reset_entered.set()
        if not permit_reset.wait(5):
            raise RuntimeError("test did not release delivery reset")
        real_reset()

    monkeypatch.setattr(integration.backend, "reset_hard_to_head", blocked_reset)

    def deliver() -> None:
        try:
            deliver_product(
                store,
                integration,
                delivery_id="serialized-target-worktree",
                final_state_id=final.id,
                base_state_id=base.id,
                artifact_paths=("new.py",),
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def checkpoint() -> None:
        try:
            integration.checkpoint("concurrent central checkpoint")
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            checkpoint_done.set()

    delivery_thread = threading.Thread(target=deliver)
    checkpoint_thread = threading.Thread(target=checkpoint)
    delivery_thread.start()
    assert reset_entered.wait(5)
    checkpoint_thread.start()
    assert not checkpoint_done.wait(0.1)
    permit_reset.set()
    delivery_thread.join(5)
    checkpoint_thread.join(5)

    assert not delivery_thread.is_alive()
    assert not checkpoint_thread.is_alive()
    assert errors == []
    assert integration.head.read("new.py") == "NEW = True\n"


def test_absence_check_and_merge_share_one_target_lock(
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    integration, worker, base = _fork(store)
    final = worker.checkpoint("gone.txt is intentionally absent")
    merge_entered = threading.Event()
    permit_merge = threading.Event()
    addition_done = threading.Event()
    real_merge = integration.merge
    results: list[Any] = []
    errors: list[BaseException] = []

    def blocked_merge(other: Any, *, reason: str, resolved: bool = False):  # type: ignore[no-untyped-def]
        merge_entered.set()
        if not permit_merge.wait(5):
            raise RuntimeError("test did not release delivery merge")
        return real_merge(other, reason=reason, resolved=resolved)

    monkeypatch.setattr(integration, "merge", blocked_merge)

    def deliver() -> None:
        try:
            results.append(
                deliver_product(
                    store,
                    integration,
                    delivery_id="serialized-absence-check",
                    final_state_id=final.id,
                    base_state_id=base.id,
                    artifact_paths=("gone.txt",),
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def add_target_path() -> None:
        try:
            integration.write("gone.txt", "added after delivery\n")
            integration.checkpoint("later target addition")
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            addition_done.set()

    delivery_thread = threading.Thread(target=deliver)
    addition_thread = threading.Thread(target=add_target_path)
    delivery_thread.start()
    assert merge_entered.wait(5)
    addition_thread.start()
    assert not addition_done.wait(0.1)
    permit_merge.set()
    delivery_thread.join(5)
    addition_thread.join(5)

    assert not delivery_thread.is_alive() and not addition_thread.is_alive()
    assert errors == []
    assert len(results) == 1 and results[0].ok
    assert results[0].integration_state.read("gone.txt") is None
    assert integration.head.read("gone.txt") == "added after delivery\n"


@pytest.mark.parametrize(
    "change",
    ["source", "target", "projection", "base"],
)
def test_delivery_identity_cannot_be_reused_for_different_work(
    store: Store, change: str
) -> None:
    integration, worker, base = _fork(store)
    worker.write("one.py", "ONE = 1\n")
    worker.write("two.py", "TWO = 2\n")
    final = worker.checkpoint("worker output")
    deliver_product(
        store,
        integration,
        delivery_id="immutable-id",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("one.py",),
    )

    kwargs = {
        "final_state_id": final.id,
        "base_state_id": base.id,
        "artifact_paths": ("one.py",),
    }
    target = integration
    if change == "source":
        worker.write("one.py", "ONE = 9\n")
        kwargs["final_state_id"] = worker.checkpoint("different final state").id
    elif change == "target":
        target = store.branch("other-target", from_state=base)
    elif change == "projection":
        kwargs["artifact_paths"] = ("two.py",)
    else:
        # A later state is an ancestor of the source only when source is also
        # moved, so use the session root shared by base and source.
        kwargs["base_state_id"] = base.parents[0].id

    with pytest.raises(DeliveryIdentityConflict):
        deliver_product(
            store,
            target,
            delivery_id="immutable-id",
            **kwargs,
        )


def test_selected_deletion_and_file_mode_are_exact(store: Store) -> None:
    integration, worker, base = _fork(store)
    integration.write("delete-me.txt", "old\n")
    base = integration.checkpoint("base with deletion target")
    worker.close()
    worker = store.branch("deleting-worker", from_state=base)
    worker.path("delete-me.txt").unlink()
    worker.write("run.sh", "#!/bin/sh\nexit 0\n")
    worker.path("run.sh").chmod(0o755)
    final = worker.checkpoint("delete and executable")

    result = deliver_product(
        store,
        integration,
        delivery_id="delete-and-mode",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("delete-me.txt", "run.sh"),
    )

    assert result.ok
    assert result.integration_state.read("delete-me.txt") is None
    assert result.integration_state.read("run.sh") == "#!/bin/sh\nexit 0\n"
    assert store.backend.mode_at(result.integration_state.id, "run.sh") == "100755"


def test_explicit_absence_conflicts_with_a_concurrent_target_addition(store: Store) -> None:
    integration, worker, base = _fork(store)
    # Selecting a path missing from the pinned final state explicitly asks for
    # absence.  It was also absent at base, while target added it concurrently.
    final = worker.checkpoint("gone.txt is intentionally absent")
    integration.write("gone.txt", "concurrent target addition\n")
    integration.checkpoint("target adds the path")

    result = deliver_product(
        store,
        integration,
        delivery_id="absence-vs-addition",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("gone.txt",),
    )

    assert not result.ok and result.integration_state is None
    assert [conflict.path for conflict in result.conflicts] == ["gone.txt"]
    assert "requires absence" in result.conflicts[0].detail
    assert integration.head.read("gone.txt") == "concurrent target addition\n"
    conflict_head = integration.head.id

    retry = deliver_product(
        store,
        integration,
        delivery_id="absence-vs-addition",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("gone.txt",),
    )
    assert not retry.ok and retry.reused_outcome
    assert integration.head.id == conflict_head


def test_absence_conflict_does_not_hide_an_ordinary_product_conflict(store: Store) -> None:
    integration, worker, base = _fork(store)
    worker.write("app.py", "VALUE = 'worker'\n")
    final = worker.checkpoint("worker changes app and requires gone absent")
    integration.write("app.py", "VALUE = 'central'\n")
    integration.write("gone.txt", "concurrent target addition\n")
    target_before = integration.checkpoint("target conflicts in two ways")

    result = deliver_product(
        store,
        integration,
        delivery_id="complete-conflict-set",
        final_state_id=final.id,
        base_state_id=base.id,
        artifact_paths=("app.py", "gone.txt"),
    )

    assert not result.ok
    assert [conflict.path for conflict in result.conflicts] == ["app.py", "gone.txt"]
    assert integration.head.read("app.py") == target_before.read("app.py")
    assert integration.head.read("gone.txt") == target_before.read("gone.txt")
    assert integration.head.conflicts == list(result.conflicts)


@pytest.mark.parametrize(
    "path, error",
    [
        ("contract.json", ReservedArtifactPath),
        ("sdk-sessions/p/s/log.jsonl", ReservedArtifactPath),
        (".taste/runtime.json", ReservedArtifactPath),
        ("../escape.py", InvalidArtifactPath),
        ("/absolute.py", InvalidArtifactPath),
        ("a/../b.py", InvalidArtifactPath),
    ],
)
def test_invalid_or_private_artifact_paths_are_refused(
    store: Store, path: str, error: type[Exception]
) -> None:
    integration, worker, base = _fork(store)
    final = worker.checkpoint("worker done")
    with pytest.raises(error):
        deliver_product(
            store,
            integration,
            delivery_id=f"bad-{path}",
            final_state_id=final.id,
            base_state_id=base.id,
            artifact_paths=(path,),
        )


def test_abbreviated_state_ids_are_refused_even_when_git_could_resolve_them(store: Store) -> None:
    integration, worker, base = _fork(store)
    final = worker.checkpoint("worker done")
    with pytest.raises(KeyError, match="full, lowercase state id"):
        deliver_product(
            store,
            integration,
            delivery_id="abbreviation",
            final_state_id=final.id[:10],
            base_state_id=base.id,
            artifact_paths=("app.py",),
        )
