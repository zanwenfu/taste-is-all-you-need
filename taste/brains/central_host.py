"""Production composition root for the central Agent OS process.

The durable coordinator components deliberately accept their Store, Branch,
and lock dependencies as objects.  That makes their invariants testable, but
it also means a production caller must take care not to open two writable
``central-control`` branches or give planner and supervisor look-alike locks.
This module is the one place which assembles that object graph.

``compose_central_runtime`` performs no model call and starts no worker.  The
returned :class:`CentralRuntimeHost` advances only when :meth:`cycle` is
called, so a service, CLI, or test retains control of polling and shutdown
policy.  Authentication remains in the inherited environment; no secret is
accepted by this API or copied into worker argv.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from taste.brains.central_communication import CentralCommunication
from taste.brains.central_planner import CentralPlanner, Goal, PlannerTransport
from taste.brains.central_runtime import CentralRuntime, CycleOutcome
from taste.brains.communication import Communicator
from taste.brains.planner_transport import PLANNER_RECEIPT_BRANCH, LLMPlannerTransport
from taste.brains.supervisor import (
    CentralSupervisor,
    ProcessLauncher,
    SubprocessLauncher,
)
from taste.brains.worker_entrypoint import worker_command_factory
from taste.llm import LLM, MODEL_MONITOR, MODEL_PLANNER
from taste.memstore import Branch, Store

__all__ = ["CentralRuntimeHost", "compose_central_runtime"]

_RLOCK_TYPE = type(threading.RLock())


def _validate_budgeted_planner_llm(llm: Any, goal: Goal) -> None:
    """Keep a real injected LLM's private cap aligned with the global ledger."""
    if not isinstance(llm, LLM) or goal.budget_usd is None:
        return
    if llm.cap_on != "billed":
        raise ValueError("a budgeted Goal requires planner LLM cap_on='billed'")
    budget = llm.budget_usd
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(float(budget))
        or float(budget) != goal.budget_usd
    ):
        raise ValueError("a budgeted Goal requires planner LLM budget_usd to equal Goal.budget_usd")
    if llm.max_attempts != 1:
        raise ValueError(
            "a budgeted Goal requires planner LLM max_attempts=1 so every "
            "possibly billed request has its own durable receipt"
        )


class CentralRuntimeHost:
    """The fully wired central process and its explicit resource boundary.

    Callers may inspect the public component attributes for observability or
    testing.  Components injected into :func:`compose_central_runtime` remain
    caller-owned.  A Store or Branch opened by the factory is closed by this
    host, including on partial construction failure.
    """

    def __init__(
        self,
        *,
        store: Store,
        goal: Goal,
        control: Branch,
        integration: Branch,
        planner_journal: Branch | None,
        control_lock: threading.RLock,
        planner_llm: Any | None,
        transport: PlannerTransport,
        planner: CentralPlanner,
        launcher: ProcessLauncher,
        supervisor: CentralSupervisor,
        communication: CentralCommunication,
        runtime: CentralRuntime,
        owns_store: bool,
        owns_control: bool,
        owns_integration: bool,
        owns_planner_journal: bool,
    ) -> None:
        self.store = store
        self.goal = goal
        self.control = control
        self.integration = integration
        self.planner_journal = planner_journal
        self.control_lock = control_lock
        self.planner_llm = planner_llm
        self.transport = transport
        self.planner = planner
        self.launcher = launcher
        self.supervisor = supervisor
        self.communication = communication
        self.runtime = runtime
        self._owns_store = owns_store
        self._owns_control = owns_control
        self._owns_integration = owns_integration
        self._owns_planner_journal = owns_planner_journal
        self._closed = False
        self._lifecycle_lock = threading.RLock()

    @property
    def closed(self) -> bool:
        """Whether this host has released its owned central resources."""
        with self._lifecycle_lock:
            return self._closed

    def cycle(self) -> CycleOutcome:
        """Run one crash-replayable reconciliation cycle."""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("central runtime host is closed")
            return self.runtime.cycle()

    def close(self) -> None:
        """Release exactly the resources opened by the composition factory."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            try:
                # The supervisor received injected shared branches, so this is
                # normally a no-op.  Keep it in the lifecycle sequence in case
                # it later acquires another explicitly owned resource.
                self.supervisor.close()
            finally:
                if self._owns_store:
                    # Store.close owns every branch it opened and the backend.
                    self.store.close()
                else:
                    if self._owns_planner_journal and self.planner_journal is not None:
                        self.planner_journal.close()
                    if self._owns_integration:
                        self.integration.close()
                    if self._owns_control:
                        self.control.close()

    def __enter__(self) -> CentralRuntimeHost:
        if self.closed:
            raise RuntimeError("central runtime host is closed")
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def compose_central_runtime(
    repo_root: Path | str,
    session: str,
    goal: Goal,
    *,
    store: Store | None = None,
    control: Branch | None = None,
    integration: Branch | None = None,
    planner_journal: Branch | None = None,
    control_lock: threading.RLock | None = None,
    planner_llm: Any | None = None,
    transport: PlannerTransport | None = None,
    launcher: ProcessLauncher | None = None,
    communicator: Communicator | None = None,
    control_branch: str = "central-control",
    integration_branch: str = "integration",
    planner_journal_branch: str = PLANNER_RECEIPT_BRANCH,
    planner_model: str = MODEL_PLANNER,
    planner_max_tokens: int = 8192,
    monitor_model: str = MODEL_MONITOR,
    python_executable: str | None = None,
    worker_environment: Mapping[str, str] | None = None,
    launcher_handshake_timeout: float = 5.0,
    default_wall_timeout_seconds: float = 900.0,
    supervisor_poll_interval: float = 0.05,
    supervisor_termination_grace: float = 2.0,
) -> CentralRuntimeHost:
    """Build the concrete central coordinator around one exact shared state.

    ``store``, ``control`` and ``integration`` are optional injection seams.
    When omitted, the factory opens and owns them.  When supplied, they must
    have the exact repository, session, and branch identities named here and
    remain caller-owned.  Supplying ``transport`` or ``launcher`` replaces
    only the external model/process boundary; all central components remain
    the production implementations.
    """
    if not isinstance(goal, Goal):
        raise TypeError("goal must be a Goal")
    if len({control_branch, integration_branch, planner_journal_branch}) != 3:
        raise ValueError("control, integration, and planner journal branch names must be distinct")
    root = Path(repo_root).expanduser().resolve(strict=True)
    if store is not None:
        if not isinstance(store, Store):
            raise TypeError("store must be a Store")
        if store.root != root or store.session != session:
            raise ValueError("injected store does not match repo_root and session")
    if control is not None and (
        not isinstance(control, Branch)
        or store is None
        or control.store is not store
        or control.name != control_branch
    ):
        raise ValueError("injected control branch has the wrong store or identity")
    if integration is not None and (
        not isinstance(integration, Branch)
        or store is None
        or integration.store is not store
        or integration.name != integration_branch
    ):
        raise ValueError("injected integration branch has the wrong store or identity")
    if control is not None and integration is not None and control is integration:
        raise ValueError("control and integration branches must be distinct")
    if planner_journal is not None and (
        not isinstance(planner_journal, Branch)
        or store is None
        or planner_journal.store is not store
        or planner_journal.name != planner_journal_branch
        or planner_journal is control
        or planner_journal is integration
    ):
        raise ValueError("injected planner journal branch has the wrong store or identity")
    if control_lock is not None and not isinstance(control_lock, _RLOCK_TYPE):
        raise TypeError("control_lock must be a threading.RLock")
    if transport is not None and planner_llm is not None:
        raise ValueError("planner_llm is unused when a planner transport is injected")
    if planner_llm is not None:
        _validate_budgeted_planner_llm(planner_llm, goal)
    if transport is not None and not isinstance(transport, PlannerTransport):
        raise TypeError("transport must implement PlannerTransport")
    if launcher is not None and not isinstance(launcher, ProcessLauncher):
        raise TypeError("launcher must implement ProcessLauncher")

    opened_store = store
    owns_store = opened_store is None
    opened_control = control
    owns_control = opened_control is None
    opened_integration = integration
    owns_integration = opened_integration is None
    opened_planner_journal = planner_journal
    owns_planner_journal = False
    try:
        if opened_store is None:
            opened_store = Store.open(root, session)
        if opened_control is None:
            opened_control = opened_store.branch(control_branch, producer="central-runtime")
        if opened_integration is None:
            opened_integration = opened_store.branch(
                integration_branch, producer="central-integration"
            )
        lock = control_lock or threading.RLock()

        concrete_llm = planner_llm
        concrete_transport = transport
        if concrete_transport is None:
            if opened_planner_journal is None:
                opened_planner_journal = opened_store.branch(
                    planner_journal_branch,
                    producer="central-planner-receipts",
                )
                owns_planner_journal = True
            if concrete_llm is None:
                concrete_llm = LLM(
                    env_dir=root,
                    budget_usd=goal.budget_usd,
                    run_id=f"central-planner.{goal.goal_id}",
                    max_attempts=1,
                    # CentralRuntime's durable global ledger is billed USD;
                    # its process-local guard must enforce the same currency.
                    cap_on="billed",
                )
            _validate_budgeted_planner_llm(concrete_llm, goal)
            concrete_transport = LLMPlannerTransport(
                concrete_llm,
                store=opened_store,
                control=opened_control,
                journal=opened_planner_journal,
                mutation_lock=lock,
                model=planner_model,
                max_tokens=planner_max_tokens,
            )
        elif isinstance(concrete_transport, LLMPlannerTransport):
            if opened_planner_journal is None:
                opened_planner_journal = concrete_transport.journal
            elif concrete_transport.journal is not opened_planner_journal:
                raise ValueError(
                    "injected LLMPlannerTransport must use the exact planner journal Branch"
                )
            if (
                concrete_transport.store is not opened_store
                or concrete_transport.control is not opened_control
                or concrete_transport.journal is not opened_planner_journal
                or concrete_transport.journal.name != planner_journal_branch
                or concrete_transport.mutation_lock is not lock
            ):
                raise ValueError(
                    "injected LLMPlannerTransport must use the exact shared Store, "
                    "control Branch, planner journal Branch, and mutation RLock"
                )
            _validate_budgeted_planner_llm(concrete_transport.llm, goal)

        concrete_planner = CentralPlanner(
            opened_store,
            transport=concrete_transport,
            control=opened_control,
            control_branch=opened_control.name,
            integration_branch=opened_integration.name,
            mutation_lock=lock,
        )

        concrete_launcher = launcher
        if concrete_launcher is None:
            command = worker_command_factory(
                root,
                session,
                python_executable=python_executable,
                monitor_model=monitor_model,
            )
            concrete_launcher = SubprocessLauncher(
                command,
                env=worker_environment,
                handshake_timeout=launcher_handshake_timeout,
            )

        concrete_supervisor = CentralSupervisor(
            opened_store,
            launcher=concrete_launcher,
            control_branch=opened_control,
            integration_branch=opened_integration,
            control_lock=lock,
            poll_interval=supervisor_poll_interval,
            termination_grace=supervisor_termination_grace,
        )
        concrete_communication = CentralCommunication(
            opened_store,
            goal_id=goal.goal_id,
            control=opened_control,
            control_lock=lock,
            communicator=communicator,
        )
        runtime = CentralRuntime(
            opened_store,
            goal,
            planner=concrete_planner,
            supervisor=concrete_supervisor,
            control=opened_control,
            integration=opened_integration,
            control_lock=lock,
            default_wall_timeout_seconds=default_wall_timeout_seconds,
            communication=concrete_communication,
        )
        return CentralRuntimeHost(
            store=opened_store,
            goal=goal,
            control=opened_control,
            integration=opened_integration,
            planner_journal=opened_planner_journal,
            control_lock=lock,
            planner_llm=concrete_llm,
            transport=concrete_transport,
            planner=concrete_planner,
            launcher=concrete_launcher,
            supervisor=concrete_supervisor,
            communication=concrete_communication,
            runtime=runtime,
            owns_store=owns_store,
            owns_control=owns_control,
            owns_integration=owns_integration,
            owns_planner_journal=owns_planner_journal,
        )
    except BaseException:
        if opened_store is not None:
            if owns_store:
                opened_store.close()
            else:
                if owns_integration and opened_integration is not None:
                    opened_integration.close()
                if owns_planner_journal and opened_planner_journal is not None:
                    opened_planner_journal.close()
                if owns_control and opened_control is not None:
                    opened_control.close()
        raise
