"""Cycle history is audited incrementally without trusting rewritten ancestry."""

import pytest

from taste.brains.central_runtime import (
    CYCLE_SCHEMA,
    CoordinatorCorruption,
    _cycle_root,
    _index_path,
)
from taste.memstore import Store
from tests.test_brains_central_runtime import FakeLauncher, ScriptedTransport, simple_goal, stack


@pytest.fixture
def runtime(tmp_path):
    store = Store.open(tmp_path / "repo", "cycle-audit")
    runtime, _ = stack(store, simple_goal(), ScriptedTransport(lambda *args: ()), FakeLauncher())
    try:
        yield runtime
    finally:
        store.close()


def append_cycle(runtime, records):
    sequence = len(records) + 1
    identity = f"cycle-{sequence}"
    root = _cycle_root(runtime.goal.goal_id, sequence, identity)
    runtime._append_cycle_intent(records, root, {
        "schema": CYCLE_SCHEMA, "sequence": sequence, "cycle_id": identity,
    })
    return runtime._cycle_records()


def test_incremental_cycles_do_not_reread_all_historical_indexes(runtime, monkeypatch):
    index_path = _index_path(runtime.goal.goal_id)
    show = runtime.store.backend.show
    reads = []

    def counted(commit, path):
        if path == index_path:
            reads.append(commit)
        return show(commit, path)

    monkeypatch.setattr(runtime.store.backend, "show", counted)
    records = runtime._cycle_records()
    for _ in range(20):
        records = append_cycle(runtime, records)
    assert len(records) == 20
    # Includes current-index reads and append checks, not just history reads.
    # Re-reading every growing historical prefix needs >200 reads here.
    assert len(reads) <= 85, len(reads)


@pytest.mark.parametrize("cold", [False, True])
def test_temporary_index_rollback_is_detected_even_after_restoration(runtime, cold):
    records = append_cycle(runtime, runtime._cycle_records())
    records = append_cycle(runtime, records)
    path = _index_path(runtime.goal.goal_id)
    original = runtime.control.head.record(path)
    runtime.control.checkpoint("inject temporary rollback", records={
        path: {**original, "cycles": original["cycles"][:1]},
    })
    runtime.control.checkpoint("restore index", records={path: original})
    if cold:
        runtime, _ = stack(runtime.store, runtime.goal, ScriptedTransport(lambda *args: ()),
                           FakeLauncher(), shared=(runtime.control, runtime.integration, runtime.control_lock))
    with pytest.raises(CoordinatorCorruption, match=r"cycle index.*(rolled back|rewritten)"):
        runtime._cycle_records()


def test_cached_index_does_not_hide_deleted_intents(runtime):
    records = append_cycle(runtime, runtime._cycle_records())
    runtime.control.path(f"{records[0][3]}/intent.json").unlink()
    runtime.control.checkpoint("delete indexed intent")
    with pytest.raises(CoordinatorCorruption, match="indexed cycle intent"):
        runtime._cycle_records()


@pytest.mark.parametrize("move", ["rollback", "second_parent"])
def test_cached_authority_requires_first_parent_ancestry(runtime, move):
    base = runtime.control.head.id
    append_cycle(runtime, runtime._cycle_records())
    audited = runtime.control.head.id
    target = base
    if move == "second_parent":
        target = runtime.store.backend.commit_tree(
            runtime.store.backend.repo.commit(audited).tree.hexsha,
            [base, audited], "reparent with audited head only as second parent",
        )
    assert runtime.store.backend.cas_update_ref(runtime.control.ref, target, audited)
    with pytest.raises(CoordinatorCorruption, match="first-parent"):
        runtime._cycle_records()


def test_unrelated_control_records_and_fresh_runtime_keep_complete_history(runtime):
    records = runtime._cycle_records()
    for _ in range(4):
        records = append_cycle(runtime, records)
        runtime.control.checkpoint("unrelated control record", records={"other.json": {"n": len(records)}})
    assert runtime._cycle_records() == records
    fresh, _ = stack(runtime.store, runtime.goal, ScriptedTransport(lambda *args: ()), FakeLauncher(),
                      shared=(runtime.control, runtime.integration, runtime.control_lock))
    assert fresh._cycle_records() == records
