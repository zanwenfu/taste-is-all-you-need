"""What a killed process leaves behind must not stop the next one, or be mistaken for work.

A SIGKILLed git leaves its lock files, and a killed writer leaves a half-finished
record in the working tree. The tests that matter most here are the negative
ones: a lock that a live process may hold is never removed.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from taste.brains.central_host import open_coordinator_branch
from taste.memstore import BranchBusy, Store

REPOSITORY = Path(__file__).resolve().parents[1]


def _wait_for(check, seconds=20.0):
    limit = time.monotonic() + seconds
    while time.monotonic() < limit:
        if check():
            return True
        time.sleep(0.02)
    return False


def _index_lock(store: Store, name: str) -> Path:
    branch = store.branch(name, producer="test")
    path = Path(branch.backend.repo.git_dir) / "index.lock"
    branch.close()
    return path


def test_git_commands_take_no_optional_locks(tmp_path):
    """A reader's ``git status`` must not be able to hold the lock a restart clears."""
    store = Store.open(tmp_path / "repo", "s")
    branch = store.branch("worker", producer="test")
    assert branch.backend.repo.git.environment()["GIT_OPTIONAL_LOCKS"] == "0"
    store.close()


def test_a_lock_a_living_git_still_holds_survives_the_death_of_its_parent(tmp_path):
    """The parent dies, its ``git add`` lives on: the branch stays busy and the lock stays."""
    store = Store.open(tmp_path / "repo", "s")
    lock = _index_lock(store, "worker")
    release = tmp_path / "release"
    clean = tmp_path / "clean"
    clean.write_text(f'#!/bin/sh\nwhile [ ! -e "{release}" ]; do sleep 0.02; done\ncat\n')
    clean.chmod(0o755)
    common = store.backend.common_dir
    (common / "info").mkdir(exist_ok=True)
    (common / "info" / "attributes").write_text("*.slow filter=slow\n")
    store.backend.repo.git.config("filter.slow.clean", str(clean))
    worker = textwrap.dedent(f"""
        from taste.memstore import Store
        store = Store.open({str(tmp_path / 'repo')!r}, "s")
        branch = store.branch("worker", producer="test")
        branch.write("work.slow", "the work")
        branch.checkpoint("the work")
    """)
    parent = subprocess.Popen([sys.executable, "-c", worker], cwd=REPOSITORY,
                              env={**os.environ, "PYTHONPATH": str(REPOSITORY)})
    try:
        assert _wait_for(lock.exists), "git add never took its lock"
        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=10)
        # The orphan keeps the locks it inherited, so opening waits for it, or
        # reports the branch busy. Either way it must not touch the lock file.
        opening = threading.Thread(target=_opens, args=(store,), daemon=True)
        opening.start()
        opening.join(1.0)
        assert lock.exists(), "a lock a living git holds must never be removed"
    finally:
        release.write_text("")
        if parent.poll() is None:
            parent.kill()
    opening.join(20.0)
    assert not opening.is_alive(), "opening must finish once the orphan has"
    assert _wait_for(lambda: not lock.exists()), "the orphan git never finished"
    assert _wait_for(lambda: _opens(store))
    store.close()


def _opens(store: Store) -> bool:
    try:
        store.branch("worker", producer="test").close()
    except BranchBusy:
        return False
    return True


def test_a_lock_held_open_by_a_process_outside_the_store_is_left_alone(tmp_path):
    store = Store.open(tmp_path / "repo", "s")
    lock = _index_lock(store, "worker")
    with lock.open("w"):
        store.branch("worker", producer="test").close()
        assert lock.exists(), "something has this lock open, so it is not ours to remove"
    assert lock.exists()
    store.branch("worker", producer="test").close()
    assert not lock.exists(), "nothing holds it now and the lease is ours"
    store.close()


def test_a_stale_lock_is_cleared_only_with_the_lease_in_hand(tmp_path):
    store = Store.open(tmp_path / "repo", "s")
    lock = _index_lock(store, "worker")
    holder = store.branch("worker", producer="test")
    lock.write_text("")
    with pytest.raises(BranchBusy):
        Store.open(tmp_path / "repo", "s").branch("worker", producer="other")
    assert lock.exists()
    lock.unlink()
    holder.close()
    store.close()


def test_a_released_branch_can_still_be_read(tmp_path):
    store = Store.open(tmp_path / "repo", "s")
    branch = store.branch("worker", producer="test")
    branch.release()
    assert branch.backend.dirty_paths() == []
    store.close()


def test_reads_survive_the_death_of_the_object_helper_and_closing_does_too(tmp_path):
    """Stopping a service signals every process in it, the helper first among equals."""
    store = Store.open(tmp_path / "repo", "s")
    branch = store.branch("worker", producer="test")
    head = branch.backend.ref_sha(branch.ref)
    git = branch.backend.repo.git

    def kill_the_helper():
        reader = git.cat_file_header
        assert reader is not None, "the object helper must be running for this to test anything"
        reader.proc.kill()
        reader.proc.wait()

    kill_the_helper()
    assert branch.backend.ref_sha(branch.ref) == head
    kill_the_helper()
    branch.close()
    store.close()


def test_a_coordinator_branch_drops_what_a_dead_writer_never_committed(tmp_path):
    store = Store.open(tmp_path / "repo", "s")
    control = store.branch("central-control", producer="test")
    control.checkpoint("a committed record", records={"kept.json": {"kept": True}})
    control.write("half-written.json", "{}\n")
    control.backend.stage_all()
    control.write("never-staged.json", "{}\n")
    control.close()

    opened = open_coordinator_branch(store, "central-control", producer="test")
    assert opened.dirty_paths() == []
    state = opened.checkpoint("the next record", records={"next.json": {"next": True}})
    assert state.read("half-written.json") is None and state.read("never-staged.json") is None
    assert state.read("kept.json") is not None and state.read("next.json") is not None
    store.close()


def test_a_worker_branch_keeps_its_uncommitted_work(tmp_path):
    """A worker's unfinished files are its work: only a ledger drops what it did not commit."""
    store = Store.open(tmp_path / "repo", "s")
    worker = store.branch("worker", producer="test")
    worker.write("unfinished.py", "work\n")
    worker.close()
    assert store.branch("worker", producer="test").dirty_paths() == ["unfinished.py"]
    store.close()
