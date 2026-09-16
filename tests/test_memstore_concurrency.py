"""Concurrency is across branches, never within one.

Two processes each own a branch and checkpoint as fast as they can while a
third keeps reading. Every state on every branch is complete; nothing is
lost. A second process on an already-held branch is refused at open.

The repo lock has to hold that line at both levels. ``flock`` excludes other
processes, which is the architecture: one OS process per worker, each with its
own worktree. Within a process it excludes nothing on its own -- the
description is already open, so a second thread skips the ``flock`` and only
increments a depth counter. These tests pin both halves.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from taste.memstore import BranchBusy, Store
from taste.memstore.backend import _HELD


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


def test_the_repo_lock_excludes_another_thread(tmp_path: Path) -> None:
    """``flock`` is per open file description, so re-entrancy must not leak.

    One description is held for the outermost holder and nested holders count
    up, which is right for a re-entrant call on one thread and wrong for a
    second thread: it finds the description already open, skips the ``flock``
    entirely, and walks into the critical section beside the first.

    Asserted on the mechanism rather than on lost notes. Two threads writing
    notes concurrently corrupt each other only when the interleaving lands
    exactly wrong, so a test written that way passes on most runs and proves
    nothing on any of them.
    """
    store = Store.open(tmp_path / "repo", "threads")
    backend = store.backend
    inside = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with backend.lock():
            inside.set()
            release.wait(timeout=10)

    def contend() -> None:
        assert inside.wait(timeout=10), "the first thread never entered"
        with backend.lock():
            second_entered.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            holder = pool.submit(hold)
            contender = pool.submit(contend)
            assert inside.wait(timeout=10)
            # The contender must still be waiting: the lock is held.
            assert not second_entered.wait(timeout=1.0), (
                "a second thread entered the repo lock while another held it"
            )
            release.set()
            holder.result(timeout=10)
            contender.result(timeout=10)
            assert second_entered.is_set(), "the lock never handed over"
    finally:
        release.set()
        store.close()


def test_the_repo_lock_is_still_re_entrant_on_one_thread(tmp_path: Path) -> None:
    """Nesting on one thread must not deadlock against itself.

    Written nested on purpose. ``flock`` is per open file description, so the
    inner acquisition must find the outer one and count rather than open a
    second description and block on it forever. Combining the two contexts on
    one line is the same thing at runtime and reads as two independent
    acquisitions, which is not what is under test.
    """
    store = Store.open(tmp_path / "repo", "reentrant")
    backend = store.backend
    key = (backend.common_dir / "memstore.lock", threading.get_ident())
    try:
        with backend.lock():
            assert _HELD[key][1] == 1
            with backend.lock():
                # Counted, not re-locked: one description for both holders.
                assert _HELD[key][1] == 2
                store.branch("a").close()
            assert _HELD[key][1] == 1
        assert key not in _HELD, "the outermost holder must release the description"
    finally:
        store.close()


def test_two_threads_checkpointing_one_store_lose_no_notes(tmp_path: Path) -> None:
    """The consequence the lock exists to prevent, end to end.

    Production runs one worker per process, but a worker's monitor judges on
    an ``asyncio.to_thread`` worker against the same ``Store``. Nothing is
    lost is the layer's first rule, so it has to hold when two threads
    checkpoint different branches at once.
    """
    store = Store.open(tmp_path / "repo", "threads")
    rounds = 8
    barrier = threading.Barrier(2)

    def write(name: str) -> None:
        branch = store.branch(name, producer=name)
        barrier.wait(timeout=10)
        for index in range(rounds):
            branch.write("counter.txt", f"{name}-{index}\n")
            branch.checkpoint(f"{name} round {index}")
        branch.close()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(write, "t1")
            second = pool.submit(write, "t2")
            first.result(timeout=120)
            second.result(timeout=120)

        for name in ("t1", "t2"):
            history = store.view(name).history()
            assert len(history) == rounds + 1, f"{name} lost a state"
            for state in history:
                assert state.meta is not None, f"{name} lost a meta note"
                assert state.manifest is not None, f"{name} lost a manifest note"
                assert state.transcript is not None, f"{name} lost a transcript note"
    finally:
        store.close()
