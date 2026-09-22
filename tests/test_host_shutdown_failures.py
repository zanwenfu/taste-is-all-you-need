"""Partial shutdown preserves ownership, attempts all cleanup and remains retryable."""

import pytest

from taste.brains import central_host
from taste.brains.central_host import compose_central_runtime
from taste.brains.planner_transport import PLANNER_RECEIPT_BRANCH
from taste.brains.supervisor import CentralSupervisor
from taste.memstore import Store
from tests.test_brains_central_host import NoCallPlannerLLM, NoLaunchLauncher, goal, repository


@pytest.mark.parametrize("resource", ["planner_journal", "integration", "control"])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_host_attempts_all_owned_closes_and_can_retry_incomplete_cleanup(tmp_path, monkeypatch, resource, failure):
    root = repository(tmp_path)
    store = Store.open(root, "host-close-failure")
    host = compose_central_runtime(root, "host-close-failure", goal(), store=store,
                                   planner_llm=NoCallPlannerLLM())
    target = getattr(host, resource)
    original = target.close

    def fail_close():
        raise failure("synthetic close failure")

    monkeypatch.setattr(target, "close", fail_close)
    try:
        with pytest.raises(failure, match="synthetic close failure"):
            host.close()
        assert not host.closed, "closed must mean owned resources have actually been released"
        for name in ("planner_journal", "integration", "control"):
            if name != resource:
                assert getattr(host, name)._lease is None, name
        with pytest.raises(RuntimeError, match="clos"):
            host.cycle()
        with pytest.raises(RuntimeError, match="clos"):
            host.__enter__()
        monkeypatch.setattr(target, "close", original)
        host.close()
        assert host.closed and target._lease is None
    finally:
        monkeypatch.setattr(target, "close", original)
        store.close()


def test_store_close_does_not_abandon_later_branch_leases(tmp_path, monkeypatch):
    store = Store.open(tmp_path / "repo", "store-close-failure")
    first = store.branch("first")
    second = store.branch("second")
    original = first.release

    def fail_close():
        raise OSError("first branch close failed")

    monkeypatch.setattr(first, "release", fail_close)
    try:
        with pytest.raises(OSError, match="first branch close failed"):
            store.close()
        assert second._lease is None
        assert first._lease is not None
        monkeypatch.setattr(first, "release", original)
        store.close()
        assert first._lease is None
    finally:
        monkeypatch.setattr(first, "release", original)
        store.close()


def test_host_preserves_simultaneous_errors_and_retries_only_pending_owners(tmp_path, monkeypatch):
    root = repository(tmp_path)
    store = Store.open(root, "host-multiple-close")
    host = compose_central_runtime(root, "host-multiple-close", goal(), store=store,
                                   planner_llm=NoCallPlannerLLM())
    originals = {name: getattr(host, name).close for name in ("planner_journal", "integration", "control")}
    control_closes = []

    def fail_journal():
        raise RuntimeError("journal close failed")

    def interrupt_integration():
        raise KeyboardInterrupt("integration close interrupted")

    def close_control():
        control_closes.append(True)
        originals["control"]()

    monkeypatch.setattr(host.planner_journal, "close", fail_journal)
    monkeypatch.setattr(host.integration, "close", interrupt_integration)
    monkeypatch.setattr(host.control, "close", close_control)
    try:
        with pytest.raises(BaseExceptionGroup) as errors:
            host.close()
        assert {type(item) for item in errors.value.exceptions} == {RuntimeError, KeyboardInterrupt}
        assert control_closes == [True] and host.control._lease is None
        assert not host.closed
        for name in ("planner_journal", "integration"):
            monkeypatch.setattr(getattr(host, name), "close", originals[name])
        host.close()
        assert host.closed and control_closes == [True]
    finally:
        for name, original in originals.items():
            monkeypatch.setattr(getattr(host, name), "close", original)
        store.close()


def test_failed_host_cleanup_never_closes_borrowed_branches(tmp_path, monkeypatch):
    root = repository(tmp_path)
    store = Store.open(root, "borrowed-close")
    control = store.branch("central-control")
    journal = store.branch(PLANNER_RECEIPT_BRANCH)
    host = compose_central_runtime(root, "borrowed-close", goal(), store=store, control=control,
                                   planner_journal=journal, planner_llm=NoCallPlannerLLM())
    original = host.integration.close

    def fail_close():
        raise OSError("integration close failed")

    monkeypatch.setattr(host.integration, "close", fail_close)
    try:
        with pytest.raises(OSError, match="integration close failed"):
            host.close()
        assert control._lease is not None and journal._lease is not None
        monkeypatch.setattr(host.integration, "close", original)
        host.close()
        assert host.closed and host.integration._lease is None
        assert control._lease is not None and journal._lease is not None
    finally:
        monkeypatch.setattr(host.integration, "close", original)
        store.close()


def test_store_releases_other_leases_and_root_backend_despite_multiple_failures(tmp_path, monkeypatch):
    store = Store.open(tmp_path / "repo", "store-multiple-close")
    first, second = store.branch("first"), store.branch("second")
    release, close_backend = first.release, first.backend.close
    close_root = store.backend.close
    root_closes = []

    def fail_release():
        raise RuntimeError("first release failed")

    def fail_backend():
        raise OSError("first backend failed")

    def record_root_close():
        root_closes.append(True)
        close_root()

    monkeypatch.setattr(first, "release", fail_release)
    monkeypatch.setattr(first.backend, "close", fail_backend)
    monkeypatch.setattr(store.backend, "close", record_root_close)
    try:
        with pytest.raises(ExceptionGroup) as errors:
            store.close()
        assert {str(item) for item in errors.value.exceptions} == {"first release failed", "first backend failed"}
        assert second._lease is None and root_closes == [True]
    finally:
        monkeypatch.setattr(first, "release", release)
        monkeypatch.setattr(first.backend, "close", close_backend)
        store.close()


def test_supervisor_attempts_both_owned_branches_when_one_close_fails(tmp_path, monkeypatch):
    store = Store.open(tmp_path / "repo", "supervisor-close")
    supervisor = CentralSupervisor(store, launcher=NoLaunchLauncher())
    original = supervisor.integration.close

    def fail_close():
        raise RuntimeError("integration close failed")

    monkeypatch.setattr(supervisor.integration, "close", fail_close)
    try:
        with pytest.raises(RuntimeError, match="integration close failed"):
            supervisor.close()
        assert supervisor.control._lease is None
    finally:
        monkeypatch.setattr(supervisor.integration, "close", original)
        store.close()


def test_branch_backend_is_closed_even_when_releasing_its_lease_fails(tmp_path, monkeypatch):
    store = Store.open(tmp_path / "repo", "branch-close-failure")
    branch = store.branch("first")
    release = branch.release
    close_backend = branch.backend.close
    closed = []

    def fail_release():
        raise OSError("lease release failed")

    def record_close():
        closed.append(True)
        close_backend()

    monkeypatch.setattr(branch, "release", fail_release)
    monkeypatch.setattr(branch.backend, "close", record_close)
    try:
        with pytest.raises(OSError, match="lease release failed"):
            branch.close()
        assert closed == [True]
    finally:
        monkeypatch.setattr(branch, "release", release)
        store.close()


def test_partial_composition_cleanup_failure_does_not_leak_other_owned_branches(tmp_path, monkeypatch):
    root = repository(tmp_path)
    store = Store.open(root, "compose-close-failure")
    originals = {}

    def fail_runtime(*args, **kwargs):
        # Construction has opened all three owned branches by this point.
        for name, branch in store._branches.items():
            originals[name] = branch.close
        journal = store._branches[PLANNER_RECEIPT_BRANCH]

        def fail_close():
            raise OSError("journal cleanup failed")

        monkeypatch.setattr(journal, "close", fail_close)
        raise RuntimeError("runtime construction failed")

    monkeypatch.setattr(central_host, "CentralRuntime", fail_runtime)
    try:
        with pytest.raises((OSError, ExceptionGroup)):
            compose_central_runtime(root, "compose-close-failure", goal(), store=store,
                                    planner_llm=NoCallPlannerLLM())
        remaining = set(store._branches)
        assert remaining == {PLANNER_RECEIPT_BRANCH}, remaining
    finally:
        for name, branch in list(store._branches.items()):
            if name in originals:
                monkeypatch.setattr(branch, "close", originals[name])
        store.close()
