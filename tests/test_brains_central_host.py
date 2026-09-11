"""The central production host shares one exact Store, Branch pair, and RLock."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

pytest.importorskip("claude_agent_sdk", reason="the central host imports the worker process")

from taste.brains.central_communication import CentralCommunication
from taste.brains.central_host import CentralRuntimeHost, compose_central_runtime
from taste.brains.central_planner import CentralPlanner, Goal
from taste.brains.central_runtime import CentralRuntime
from taste.brains.communication import Communicator
from taste.brains.planner_transport import PLANNER_RECEIPT_BRANCH, LLMPlannerTransport
from taste.brains.supervisor import CentralSupervisor, SubprocessLauncher
from taste.llm import LLM
from taste.memstore import Store


class NoCallPlannerLLM:
    def __init__(self) -> None:
        self.ready_calls: list[tuple[str, ...]] = []

    def ensure_ready(self, *models: str) -> None:
        self.ready_calls.append(models)

    def call(self, **_kwargs):
        raise AssertionError("composition must not call the planner model")


class NoCallTransport:
    def complete(self, *, request_id: str, system: str, prompt: str) -> str:
        raise AssertionError((request_id, system, prompt))


class NoLaunchLauncher:
    def launch(self, spec):
        raise AssertionError(spec)

    def recover(self, spec):
        return None


def goal() -> Goal:
    return Goal(
        goal_id="composition-goal",
        task="produce one exact result",
        success_criteria=("the result is integrated",),
        budget_usd=5.0,
    )


def repository(tmp_path: Path, name: str = "repo") -> Path:
    root = tmp_path / name
    root.mkdir()
    return root


def test_default_composition_wires_every_production_component_without_calling_model(
    tmp_path: Path,
) -> None:
    root = repository(tmp_path)
    store = Store.open(root, "host-test")
    llm = NoCallPlannerLLM()

    host = compose_central_runtime(
        root,
        "host-test",
        goal(),
        store=store,
        planner_llm=llm,
        worker_environment={"TASTE_COMPOSITION_TEST": "1"},
    )
    try:
        assert isinstance(host, CentralRuntimeHost)
        assert isinstance(host.transport, LLMPlannerTransport)
        assert isinstance(host.planner, CentralPlanner)
        assert isinstance(host.launcher, SubprocessLauncher)
        assert isinstance(host.supervisor, CentralSupervisor)
        assert isinstance(host.communication, CentralCommunication)
        assert isinstance(host.runtime, CentralRuntime)

        assert host.planner_llm is llm
        assert host.transport.llm is llm
        assert host.transport.store is store
        assert host.transport.control is host.control
        assert host.transport.journal is host.planner_journal
        assert host.planner_journal is not host.control
        assert host.transport.mutation_lock is host.control_lock
        assert host.launcher.env == {"TASTE_COMPOSITION_TEST": "1"}
        assert callable(host.launcher.command)
        assert llm.ready_calls == []

        assert host.planner.store is store
        assert host.planner.control is host.control
        assert host.planner.mutation_lock is host.control_lock
        assert host.supervisor.store is store
        assert host.supervisor.control is host.control
        assert host.supervisor.integration is host.integration
        assert host.supervisor.control_lock is host.control_lock
        assert host.communication.store is store
        assert host.communication.control is host.control
        assert host.communication.control_lock is host.control_lock
        assert host.runtime.store is store
        assert host.runtime.control is host.control
        assert host.runtime.integration is host.integration
        assert host.runtime.control_lock is host.control_lock
        assert host.runtime.communication is host.communication
    finally:
        host.close()
        store.close()

    assert host.closed
    with pytest.raises(RuntimeError, match="closed"):
        host.cycle()


def test_default_planner_llm_caps_the_same_billed_currency_as_global_ledger(
    tmp_path: Path,
) -> None:
    root = repository(tmp_path)
    store = Store.open(root, "default-llm-test")

    host = compose_central_runtime(
        root,
        "default-llm-test",
        goal(),
        store=store,
        launcher=NoLaunchLauncher(),
    )
    try:
        assert isinstance(host.planner_llm, LLM)
        assert host.planner_llm.cap_on == "billed"
        assert host.planner_llm.budget_usd == goal().budget_usd
        assert host.planner_llm.max_attempts == 1
        assert host.planner_llm.stats.total_cost_usd == 0.0
    finally:
        host.close()
        store.close()


@pytest.mark.parametrize(
    ("planner_budget", "cap_on", "message"),
    [
        (5.0, "work", "cap_on='billed'"),
        (None, "billed", "budget_usd"),
        (6.0, "billed", "budget_usd"),
    ],
)
def test_budgeted_goal_rejects_mismatched_injected_real_planner_llm(
    tmp_path: Path,
    planner_budget: float | None,
    cap_on: str,
    message: str,
) -> None:
    root = repository(tmp_path)
    store = Store.open(root, "mismatched-llm-test")
    llm = LLM(env_dir=root, budget_usd=planner_budget, cap_on=cap_on)

    with pytest.raises(ValueError, match=message):
        compose_central_runtime(
            root,
            "mismatched-llm-test",
            goal(),
            store=store,
            planner_llm=llm,
            launcher=NoLaunchLauncher(),
        )

    assert store.branches() == []
    store.close()


def test_budgeted_goal_accepts_aligned_injected_real_planner_llm(tmp_path: Path) -> None:
    root = repository(tmp_path)
    store = Store.open(root, "aligned-llm-test")
    llm = LLM(
        env_dir=root,
        budget_usd=goal().budget_usd,
        cap_on="billed",
        max_attempts=1,
    )

    host = compose_central_runtime(
        root,
        "aligned-llm-test",
        goal(),
        store=store,
        planner_llm=llm,
        launcher=NoLaunchLauncher(),
    )
    try:
        assert host.planner_llm is llm
        assert host.transport.llm is llm
    finally:
        host.close()
        store.close()


def test_budgeted_goal_rejects_multi_attempt_planner_llm(tmp_path: Path) -> None:
    root = repository(tmp_path)
    store = Store.open(root, "retrying-llm-test")
    llm = LLM(
        env_dir=root,
        budget_usd=goal().budget_usd,
        cap_on="billed",
        max_attempts=2,
    )

    with pytest.raises(ValueError, match="max_attempts=1"):
        compose_central_runtime(
            root,
            "retrying-llm-test",
            goal(),
            store=store,
            planner_llm=llm,
            launcher=NoLaunchLauncher(),
        )

    assert store.branches() == []
    store.close()


def test_injected_branches_lock_and_boundaries_remain_caller_owned(tmp_path: Path) -> None:
    root = repository(tmp_path)
    store = Store.open(root, "injected-test")
    control = store.branch("central-control", producer="test-central")
    integration = store.branch("integration", producer="test-integration")
    lock = threading.RLock()
    transport = NoCallTransport()
    launcher = NoLaunchLauncher()
    communicator = Communicator(store)

    host = compose_central_runtime(
        root,
        "injected-test",
        goal(),
        store=store,
        control=control,
        integration=integration,
        control_lock=lock,
        transport=transport,
        launcher=launcher,
        communicator=communicator,
    )
    assert host.transport is transport
    assert host.launcher is launcher
    assert host.communication.communicator is communicator
    assert host.control is control
    assert host.integration is integration
    assert host.control_lock is lock

    host.close()
    host.close()
    assert control.holder is not None
    assert integration.holder is not None
    store.close()


def test_injected_llm_transport_journal_is_exact_and_remains_caller_owned(
    tmp_path: Path,
) -> None:
    root = repository(tmp_path)
    store = Store.open(root, "injected-llm-transport-test")
    control = store.branch("central-control", producer="test-central")
    integration = store.branch("integration", producer="test-integration")
    journal = store.branch(PLANNER_RECEIPT_BRANCH, producer="test-planner-receipts")
    lock = threading.RLock()
    transport = LLMPlannerTransport(
        NoCallPlannerLLM(),
        store=store,
        control=control,
        journal=journal,
        mutation_lock=lock,
    )

    host = compose_central_runtime(
        root,
        "injected-llm-transport-test",
        goal(),
        store=store,
        control=control,
        integration=integration,
        control_lock=lock,
        transport=transport,
        launcher=NoLaunchLauncher(),
    )
    assert host.planner_journal is journal
    host.close()

    assert control.holder is not None
    assert integration.holder is not None
    assert journal.holder is not None
    store.close()


@pytest.mark.parametrize("foreign_binding", ["store", "control", "journal", "lock"])
def test_injected_planner_transport_must_share_receipt_identity(
    tmp_path: Path,
    foreign_binding: str,
) -> None:
    root = repository(tmp_path)
    other_root = repository(tmp_path, "other-transport-repo")
    store = Store.open(root, "transport-binding-test")
    other_store = Store.open(other_root, "other-transport-test")
    control = store.branch("central-control", producer="test-central")
    shared_lock = threading.RLock()
    transport_store = other_store if foreign_binding == "store" else store
    if foreign_binding == "store":
        transport_control = other_store.branch("central-control", producer="foreign-transport")
    elif foreign_binding == "control":
        transport_control = store.branch("wrong-control", producer="foreign-transport")
    else:
        transport_control = control
    if foreign_binding == "store":
        transport_journal = other_store.branch(
            PLANNER_RECEIPT_BRANCH,
            producer="foreign-transport",
        )
    elif foreign_binding == "journal":
        transport_journal = store.branch("wrong-journal", producer="foreign-transport")
    else:
        transport_journal = store.branch(
            PLANNER_RECEIPT_BRANCH,
            producer="planner-receipt-test",
        )
    transport_lock = threading.RLock() if foreign_binding == "lock" else shared_lock
    transport = LLMPlannerTransport(
        NoCallPlannerLLM(),
        store=transport_store,
        control=transport_control,
        journal=transport_journal,
        mutation_lock=transport_lock,
    )

    with pytest.raises(ValueError, match=r"exact shared Store|planner journal"):
        compose_central_runtime(
            root,
            "transport-binding-test",
            goal(),
            store=store,
            control=control,
            control_lock=shared_lock,
            transport=transport,
            launcher=NoLaunchLauncher(),
        )

    assert store.view("central-control").holder is not None
    assert store.view("integration").holder is None
    if transport_control is not control:
        transport_control.close()
    control.close()
    other_store.close()
    store.close()


def test_host_releases_branches_it_opens_on_an_injected_store(tmp_path: Path) -> None:
    root = repository(tmp_path)
    store = Store.open(root, "owned-branch-test")
    host = compose_central_runtime(
        root,
        "owned-branch-test",
        goal(),
        store=store,
        transport=NoCallTransport(),
        launcher=NoLaunchLauncher(),
    )
    assert store.view("central-control").holder is not None
    assert store.view("integration").holder is not None

    host.close()

    assert store.view("central-control").holder is None
    assert store.view("integration").holder is None
    probe = store.branch("probe", producer="still-open")
    assert probe.holder is not None
    store.close()


def test_partial_composition_failure_releases_new_branches(tmp_path: Path) -> None:
    root = repository(tmp_path)
    other_root = repository(tmp_path, "other")
    store = Store.open(root, "failure-test")
    other_store = Store.open(other_root, "other-test")

    with pytest.raises(Exception, match="communicator must use the exact shared Store"):
        compose_central_runtime(
            root,
            "failure-test",
            goal(),
            store=store,
            transport=NoCallTransport(),
            launcher=NoLaunchLauncher(),
            communicator=Communicator(other_store),
        )

    assert store.view("central-control").holder is None
    assert store.view("integration").holder is None
    other_store.close()
    store.close()
