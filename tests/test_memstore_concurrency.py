"""Concurrency is across branches, never within one.

Two processes each own a branch and checkpoint as fast as they can while a
third keeps reading. Every state on every branch is complete; nothing is
lost. A second process on an already-held branch is refused at open.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest

from taste.memstore import BranchBusy, Store


def _writer(root: str, who: str, rounds: int, q: mp.Queue) -> None:  # type: ignore[type-arg]
    store = Store.open(Path(root), "race")
    b = store.branch(who, producer=who)
    for i in range(rounds):
        b.write("counter.txt", f"{who}-{i}\n")
        b.publish("counter", "counter.txt", description=f"{who}'s counter")
        b.checkpoint(f"{who} round {i}")
    store.close()
    q.put((who, rounds))


def _holder(root: str, ready: mp.Event, release: mp.Event) -> None:  # type: ignore[type-arg]
    store = Store.open(Path(root), "race")
    store.branch("held")
    ready.set()
    release.wait(timeout=60)
    store.close()


def test_parallel_branches_lose_nothing(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    Store.open(root, "race").close()
    ctx = mp.get_context("fork")
    q: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    rounds = 12
    ps = [ctx.Process(target=_writer, args=(str(root), who, rounds, q)) for who in ("p1", "p2", "p3")]
    for p in ps:
        p.start()
    for p in ps:
        p.join(timeout=180)
    done = {who: n for who, n in (q.get(timeout=5) for _ in ps)}
    assert done == {"p1": rounds, "p2": rounds, "p3": rounds}

    s = Store.open(root, "race")
    for who in ("p1", "p2", "p3"):
        b = s.branch(who)
        history = b.history()
        assert len(history) == rounds + 1  # rounds checkpoints + the branch record
        assert history[0].read("counter.txt") == f"{who}-{rounds - 1}\n"
        for st in history:
            assert st.meta is not None and st.manifest is not None and st.transcript is not None
    # and the communicator's view across all of them is consistent
    hits = s.search("counter")
    assert sorted(h.branch for h in hits) == ["p1", "p2", "p3"]
    s.close()


def test_a_held_branch_refuses_a_second_process(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    Store.open(root, "race").close()
    ctx = mp.get_context("fork")
    ready, release = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_holder, args=(str(root), ready, release))
    holder.start()
    assert ready.wait(timeout=60)
    s = Store.open(root, "race")
    with pytest.raises(BranchBusy):
        s.branch("held")
    release.set()
    holder.join(timeout=60)
    # Once the holder is gone, the lease is free: no cleanup needed.
    assert s.branch("held").head.meta.kind == "branch"
    s.close()
