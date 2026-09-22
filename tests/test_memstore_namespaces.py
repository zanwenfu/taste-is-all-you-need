"""Session isolation includes pending recovery data, locks and cleanup."""

from __future__ import annotations

import fcntl
import os
from contextlib import ExitStack, closing
from pathlib import Path

import pytest

from taste.memstore import BadName, BranchBusy, Store


def _legacy_layout(store: Store) -> tuple[Path, dict[str, bytes]]:
    """Create a pre-v2 on-disk fixture, including uncheckpointed reasoning."""
    branch = store.branch("worker")
    branch.write("product.txt", "checkpointed product")
    branch.checkpoint("base")
    branch.intend("finish the product")
    branch.turn(role="assistant", content="pending reasoning")
    common = store.backend.common_dir
    sources = [
        (store.sidecar(kind, "worker", suffix), f"memstore.{kind}.old.worker{suffix}")
        for kind, suffix in (
            ("lease", ""), ("tip", ""), ("intent", ""), ("turns", f".{branch.head.id}"),
        )
    ]
    branch.close()
    for source, old_name in sources:
        target = common / old_name
        if source != target:
            source.replace(target)
    (common / "memstore-sidecars" / "layout").unlink(missing_ok=True)
    return common, {name: (common / name).read_bytes() for _, name in sources}


def test_dotted_names_do_not_share_leases_or_pending_reasoning(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    with ExitStack() as stack:
        first = stack.enter_context(closing(Store.open(root, "tb")))
        second = stack.enter_context(closing(Store.open(root, "tb.task1")))
        a = first.branch("task1.worker")
        b = second.branch("worker")
        a.intend("first intent")
        b.intend("second intent")
        a.turn(content="first pending")
        b.turn(content="second pending")
        # Cleanup in either session must preserve both live writers.
        first.gc()
        second.gc()
        assert a.resume().intent == "first intent"
        assert b.resume().intent == "second intent"
        a.close()
        b.close()
        first.gc()
        second.gc()
    with ExitStack() as stack:
        first = stack.enter_context(closing(Store.open(root, "tb")))
        second = stack.enter_context(closing(Store.open(root, "tb.task1")))
        assert [t["content"] for t in first.branch("task1.worker").resume().recovered_turns] == ["first pending"]
        assert [t["content"] for t in second.branch("worker").resume().recovered_turns] == ["second pending"]


def test_namespace_component_boundaries_cover_kind_and_suffix(tmp_path: Path) -> None:
    with closing(Store.open(tmp_path / "repo", "s")) as store:
        assert store.sidecar("monitor", "worker", ".contract-abc") != store.sidecar(
            "monitor", "worker.contract-abc"
        )
        # Both names are legal Git identities; their combined length must not
        # overflow one filesystem filename in recovery plumbing.
    with closing(Store.open(tmp_path / "repo", "s" * 120)) as store:
        worker = store.branch("w" * 120)
        worker.turn(content="long identity")
        assert worker.resume().recovered_turns[0]["content"] == "long identity"


def test_unambiguous_legacy_recovery_data_is_migrated_and_reopens(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    with closing(Store.open(root, "old")) as old:
        common, originals = _legacy_layout(old)
    for _ in range(2):
        with closing(Store.open(root, "old")) as upgraded:
            resumed = upgraded.branch("worker").resume()
            assert resumed.intent == "finish the product"
            assert [t["content"] for t in resumed.recovered_turns] == ["pending reasoning"]
            assert upgraded.view("worker").read("product.txt") == "checkpointed product"
    assert not any((common / name).exists() for name in originals)


def test_migration_refuses_a_live_legacy_writer_before_moving_data(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    with closing(Store.open(root, "old")) as old:
        common, originals = _legacy_layout(old)
    with (common / "memstore.lease.old.worker").open("a+") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BranchBusy, match="migration"):
            Store.open(root, "old")
        assert {name: (common / name).read_bytes() for name in originals} == originals
    with closing(Store.open(root, "old")) as upgraded:
        assert upgraded.branch("worker").resume().recovered_turns


def test_ambiguous_legacy_ownership_is_preserved_for_recovery(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    with closing(Store.open(root, "a")) as first:
        first.branch("b.worker").close()
        common = first.backend.common_dir
    with closing(Store.open(root, "a.b")) as second:
        second.branch("worker").close()
    # One historical filename can belong to either durable branch. No choice
    # of owner is justified, even if one branch happens to be opened first.
    legacy = common / "memstore.intent.a.b.worker"
    legacy.write_text("whose pending work?")
    (common / "memstore-sidecars" / "layout").unlink(missing_ok=True)
    with pytest.raises(RuntimeError, match="ambiguous"):
        Store.open(root, "a")
    assert legacy.read_text() == "whose pending work?"


def test_interrupted_migration_resumes_without_losing_pending_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    with closing(Store.open(root, "old")) as old:
        common, _ = _legacy_layout(old)
    replace = os.replace
    moved = 0

    def interrupted(source, destination):
        nonlocal moved
        if Path(source).parent == common and Path(source).name.startswith("memstore."):
            moved += 1
            if moved == 2:
                raise OSError("simulated migration interruption")
        return replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupted)
        with pytest.raises(OSError, match="simulated migration"):
            Store.open(root, "old")
    assert moved == 2
    with closing(Store.open(root, "old")) as upgraded:
        resumed = upgraded.branch("worker").resume()
        assert resumed.intent == "finish the product"
        assert resumed.recovered_turns[0]["content"] == "pending reasoning"


def test_gc_does_not_sweep_a_journal_held_by_another_writer(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    with closing(Store.open(root, "s")) as writer, closing(Store.open(root, "s")) as reader:
        branch = writer.branch("worker")
        stale = writer.sidecar("turns", "worker", "." + "d" * 40)
        stale.write_text('{"content":"publication still in progress"}\n')
        reader.gc()
        assert stale.exists(), "GC must not race a writer's journal handoff"
        branch.close()
        reader.gc()
        assert not stale.exists()


def test_migration_preserves_all_runtime_sidecar_families(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    with closing(Store.open(root, "old")) as old:
        common, _ = _legacy_layout(old)
    families = [
        ("acked", ""), ("rewinds", ""), ("session-mirror", ""),
        ("planner-transport-lock", ""), ("monitor", ""),
        ("monitor", ".contract-" + "a" * 64),
        ("runtime-session", "." + "b" * 24),
        ("worker-budget", "." + "c" * 64 + ".jsonl"),
    ]
    for kind, suffix in families:
        (common / f"memstore.{kind}.old.worker{suffix}").write_bytes(kind.encode() + b"\0\xff")
    with closing(Store.open(root, "old")) as upgraded:
        for kind, suffix in families:
            assert upgraded.sidecar(kind, "worker", suffix).read_bytes() == kind.encode() + b"\0\xff"
            assert not (common / f"memstore.{kind}.old.worker{suffix}").exists()


def test_conflicting_upgrade_destination_never_overwrites_either_copy(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    with closing(Store.open(root, "old")) as old:
        common, originals = _legacy_layout(old)
        destination = common / "memstore-sidecars" / "v2" / "old" / "worker" / "intent"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("another copy")
    with pytest.raises(RuntimeError, match="conflicting"):
        Store.open(root, "old")
    assert destination.read_text() == "another copy"
    assert {name: (common / name).read_bytes() for name in originals} == originals


def test_unknown_owner_or_layout_never_silently_discards_recovery_data(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    with closing(Store.open(root, "old")) as old:
        common, originals = _legacy_layout(old)
    orphan = common / "memstore.intent.missing.worker"
    orphan.write_text("pending work with no branch ref")
    with pytest.raises(RuntimeError, match="no identifiable owner"):
        Store.open(root, "old")
    assert orphan.read_text() == "pending work with no branch ref"
    assert {name: (common / name).read_bytes() for name in originals} == originals

    marker = common / "memstore-sidecars" / "layout"
    marker.parent.mkdir(exist_ok=True)
    marker.write_text("99\n")
    with pytest.raises(RuntimeError, match="unsupported sidecar layout"):
        Store.open(root, "old")
    assert marker.read_text() == "99\n"


@pytest.mark.parametrize(("kind", "suffix"), [
    ("../outside", ""), ("monitor.contract-x", ""), ("monitor", "/../../outside"),
])
def test_sidecar_components_cannot_escape_the_namespace(
    tmp_path: Path, kind: str, suffix: str,
) -> None:
    with closing(Store.open(tmp_path / "repo", "s")) as store, pytest.raises(BadName):
        store.sidecar(kind, "worker", suffix)
