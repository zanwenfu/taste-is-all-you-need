"""Production process boundary for one supervised worker.

The central supervisor launches this module through :class:`SubprocessLauncher`::

    launcher = SubprocessLauncher(
        worker_command_factory(repo_root, session)
    )

The resulting argv is equivalent to::

    python -m taste.brains.worker_entrypoint \
        --repo-root /absolute/repo --session SESSION \
        --worker WORKER --model MODEL --prepared-state STATE \
        [--monitor-budget-usd ASSIGNMENT_BUDGET]

Authentication stays in the inherited environment.  No token or API key is
accepted on the command line, printed, or copied into a diagnostic.
``worker_command`` derives the optional monitor budget from the exact typed
Assignment; callers cannot supply a separate launcher-side override.

The first half of this module is deliberately lease-free.  It reads the exact
checkpointed Assignment and Contract through a BranchView, validates their
canonical bytes and launch bindings, and only then constructs SubBrain (which
takes the branch write lease).  WorkerRuntime repeats the durable validation
after taking the lease, closing the read/acquire race without ever taking two
leases for one worker.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import re
import signal
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

from taste.brains.communication import Communicator
from taste.brains.contract import CONTRACT_PATH, Contract
from taste.brains.monitor import BATCH_SIZE, MonitorBrain
from taste.brains.monitor_judge import LLMMonitorJudge
from taste.brains.records import Assignment
from taste.brains.subbrain import SubBrain, SubBrainResult
from taste.brains.supervisor import LaunchSpec, mark_worker_ready
from taste.brains.worker_runtime import (
    ASSIGNMENT_PATH,
    ContractMismatch,
    ShutdownUnconfirmed,
    WorkerRuntime,
)
from taste.llm import LLM, MODEL_MONITOR
from taste.memstore import Store
from taste.memstore.store import _check_name

__all__ = [
    "DurableWorkerInput",
    "EntrypointConfig",
    "EntrypointInputError",
    "WorkerExitCode",
    "assignment_run_id",
    "execute_worker",
    "main",
    "worker_command",
    "worker_command_factory",
]

_EXACT_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_RUN_ID = re.compile(r"worker-run\.[0-9a-f]{64}\Z")
_LAUNCH_TOKEN = re.compile(r"[0-9a-f]{32}\Z")
_REGULAR_GIT_MODES = frozenset({"100644"})

# A disconnect which cannot prove child reaping is a process quarantine, not
# an ordinary exception.  Keeping strong references prevents CPython's cyclic
# GC from closing the flock between ``execute_worker`` returning and the
# supervisor observing process exit.  The OS releases everything at process
# death; tests may explicitly close their injected Store afterwards.
_QUARANTINED_RESOURCES: list[tuple[Store, SubBrain | None]] = []


class WorkerExitCode(IntEnum):
    """Stable process outcomes consumed alongside the durable WorkerReport.

    ``INCOMPLETE`` means a validated runtime reached its reporting boundary
    but did not certify completion.  The remaining non-zero codes mean no
    validated terminal report may be inferred from process exit alone.
    """

    COMPLETED = 0
    INCOMPLETE = 10
    INPUT_REJECTED = 65
    RUNTIME_FAILURE = 70
    INFRA_FAILURE = 71
    SHUTDOWN_UNCONFIRMED = 72
    INTERRUPTED = 130


class EntrypointInputError(RuntimeError):
    """The process launch is not bound to one exact durable assignment."""


@dataclass(frozen=True, slots=True)
class EntrypointConfig:
    """Non-secret, immutable worker process configuration."""

    repo_root: Path
    session: str
    worker: str
    expected_model: str
    prepared_state_id: str
    monitor_model: str = MODEL_MONITOR
    monitor_max_tokens: int = 1024
    monitor_batch_size: int = BATCH_SIZE
    monitor_budget_usd: float | None = None
    poll_interval: float = 0.05
    terminal_quiet_period: float = 0.25
    shutdown_timeout: float = 10.0

    def __post_init__(self) -> None:
        root = Path(self.repo_root).expanduser()
        try:
            root = root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("repo_root must name an existing directory") from exc
        if not root.is_dir() or not (root / ".git").exists():
            raise ValueError("repo_root must name an existing Git worktree")
        object.__setattr__(self, "repo_root", root)
        object.__setattr__(self, "session", _check_name(self.session, "session"))
        object.__setattr__(self, "worker", _check_name(self.worker, "worker"))
        for name in ("expected_model", "monitor_model"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
                or len(value) > 256
                or any(ord(character) < 32 for character in value)
            ):
                raise ValueError(f"{name} must be a stable non-empty model id")
        if _EXACT_OBJECT_ID.fullmatch(self.prepared_state_id) is None:
            raise ValueError("prepared_state_id must be a full lowercase object id")
        for name in ("monitor_max_tokens", "monitor_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.monitor_budget_usd is not None and (
            isinstance(self.monitor_budget_usd, bool)
            or not isinstance(self.monitor_budget_usd, (int, float))
            or not 0 < float(self.monitor_budget_usd) < float("inf")
        ):
            raise ValueError("monitor_budget_usd must be finite and positive")
        if self.monitor_budget_usd is not None:
            object.__setattr__(self, "monitor_budget_usd", float(self.monitor_budget_usd))
        for name in ("poll_interval", "terminal_quiet_period", "shutdown_timeout"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 < float(value) < float("inf")
            ):
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class DurableWorkerInput:
    """Lease-free, exact launch input accepted from the worker branch."""

    assignment: Assignment
    contract: Contract
    head_state_id: str
    run_id: str


def assignment_run_id(assignment: Assignment) -> str:
    """The same content-bound run identity minted by CentralSupervisor."""
    token = hashlib.sha256(assignment.to_json().encode("utf-8")).hexdigest()
    return f"worker-run.{token}"


def _assignment_monitor_budget_usd(assignment: Assignment) -> float | None:
    if "monitor_budget_usd" not in assignment.resources:
        return None
    value = assignment.resources["monitor_budget_usd"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < float(value) < float("inf")
    ):
        raise ValueError("Assignment.resources.monitor_budget_usd must be finite and positive")
    return float(value)


def _expected_readiness_path(store: Store, run_id: str) -> Path:
    key = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    return store.backend.common_dir / f"taste-supervisor.{store.session}.{key}.ready.json"


def _validated_environment(
    store: Store,
    assignment: Assignment,
    environ: Mapping[str, str],
) -> str:
    run_id = environ.get("TASTE_WORKER_RUN_ID", "")
    if _RUN_ID.fullmatch(run_id) is None or run_id != assignment_run_id(assignment):
        raise EntrypointInputError("supervisor run id does not match the exact assignment")
    token = environ.get("TASTE_WORKER_LAUNCH_TOKEN", "")
    if _LAUNCH_TOKEN.fullmatch(token) is None:
        raise EntrypointInputError("supervisor launch token is missing or malformed")
    raw_ready_path = environ.get("TASTE_WORKER_READY_PATH", "")
    ready_path = Path(raw_ready_path)
    if not ready_path.is_absolute() or ready_path != _expected_readiness_path(store, run_id):
        raise EntrypointInputError("supervisor readiness path is not bound to this run")
    return run_id


def _require_regular_record(store: Store, state_id: str, path: str) -> str:
    entry = store.backend.entry_at(state_id, path)
    if entry is None or entry.mode not in _REGULAR_GIT_MODES:
        raise EntrypointInputError(f"durable {path} is not a regular control record")
    try:
        raw = store.state(state_id).read(path)
    except (OSError, UnicodeError) as exc:
        raise EntrypointInputError(f"durable {path} is not readable UTF-8") from exc
    if raw is None:
        raise EntrypointInputError(f"durable {path} is missing")
    return raw


def _load_durable_input(
    store: Store,
    config: EntrypointConfig,
    environ: Mapping[str, str],
) -> DurableWorkerInput:
    """Read and validate launch truth without opening a writable Branch."""
    view = store.view(config.worker)
    if not view.exists():
        raise EntrypointInputError("worker branch does not exist")
    try:
        head = view.head
    except Exception as exc:
        raise EntrypointInputError("worker branch head is unreadable") from exc
    if head.meta.session != config.session or head.meta.branch != config.worker:
        raise EntrypointInputError("worker branch identity does not match this process")
    if head.id != config.prepared_state_id:
        raise EntrypointInputError("worker head changed after exact assignment preparation")

    contract_text = _require_regular_record(store, head.id, CONTRACT_PATH)
    assignment_text = _require_regular_record(store, head.id, ASSIGNMENT_PATH)
    try:
        contract = Contract.from_json(contract_text)
        assignment = Assignment.from_json(assignment_text)
    except Exception as exc:
        raise EntrypointInputError("durable worker control records are malformed") from exc

    # Canonical bytes reject duplicate/unknown fields even though the legacy
    # Contract decoder itself remains permissive for old callers.
    if contract.to_json() != contract_text:
        raise EntrypointInputError("durable contract is not canonical")
    if assignment.to_json() != assignment_text:
        raise EntrypointInputError("durable assignment is not canonical")
    if assignment.contract != contract or assignment.contract.to_json() != contract_text:
        raise EntrypointInputError("durable assignment and contract disagree")
    if assignment.worker != config.worker:
        raise EntrypointInputError("assignment worker identity does not match this process")
    if assignment.model != config.expected_model:
        raise EntrypointInputError("assignment model does not match this process")
    if contract.inputs != tuple(item.path for item in assignment.inputs):
        raise EntrypointInputError("contract inputs do not match the exact structured input paths")
    if contract.outputs != tuple(item.path for item in assignment.outputs):
        raise EntrypointInputError(
            "contract outputs do not match the exact structured output paths"
        )
    try:
        assignment_monitor_budget = _assignment_monitor_budget_usd(assignment)
    except ValueError as exc:
        raise EntrypointInputError("assignment monitor budget is invalid") from exc
    if config.monitor_budget_usd != assignment_monitor_budget:
        raise EntrypointInputError("monitor budget does not match the exact durable assignment")

    try:
        prepared = store.state(config.prepared_state_id)
    except Exception as exc:
        raise EntrypointInputError("prepared state does not exist") from exc
    if prepared.meta.session != config.session or prepared.meta.branch != config.worker:
        raise EntrypointInputError("prepared state belongs to another worker or session")
    if (
        _require_regular_record(store, prepared.id, CONTRACT_PATH) != contract_text
        or _require_regular_record(store, prepared.id, ASSIGNMENT_PATH) != assignment_text
    ):
        raise EntrypointInputError("prepared control records differ from the live assignment")

    run_id = _validated_environment(store, assignment, environ)
    return DurableWorkerInput(
        assignment=assignment,
        contract=contract,
        head_state_id=head.id,
        run_id=run_id,
    )


def _validate_acquired_brain(
    brain: SubBrain,
    store: Store,
    config: EntrypointConfig,
    durable: DurableWorkerInput,
) -> None:
    """Close the lease-free read/acquire race before constructing an SDK client."""
    if brain.store is not store or brain.contract != durable.contract:
        raise EntrypointInputError("worker factory changed the exact store or contract")
    if brain.branch.name != config.worker or brain.branch.head.id != config.prepared_state_id:
        raise EntrypointInputError("worker head changed while acquiring its lease")
    dirty = brain.branch.dirty_paths()
    if dirty:
        raise EntrypointInputError("worker worktree was dirty when its lease was acquired")


def worker_command(
    spec: LaunchSpec,
    *,
    repo_root: Path,
    session: str,
    python_executable: str | None = None,
    monitor_model: str = MODEL_MONITOR,
) -> tuple[str, ...]:
    """Build the non-secret argv for one exact LaunchSpec."""
    executable = python_executable or sys.executable
    if not executable or "\x00" in executable:
        raise ValueError("python_executable is invalid")
    root = Path(repo_root).expanduser().resolve(strict=True)
    _check_name(session, "session")
    if spec.run_id != assignment_run_id(spec.assignment):
        raise ValueError("LaunchSpec run_id does not match its exact assignment")
    if _EXACT_OBJECT_ID.fullmatch(spec.prepared_state_id) is None:
        raise ValueError("LaunchSpec prepared_state_id is not exact")
    if (
        not isinstance(monitor_model, str)
        or not monitor_model.strip()
        or monitor_model != monitor_model.strip()
        or len(monitor_model) > 256
        or any(ord(character) < 32 for character in monitor_model)
    ):
        raise ValueError("monitor_model is not a stable model id")
    command = [
        executable,
        "-m",
        "taste.brains.worker_entrypoint",
        "--repo-root",
        str(root),
        "--session",
        session,
        "--worker",
        spec.assignment.worker,
        "--model",
        spec.assignment.model,
        "--prepared-state",
        spec.prepared_state_id,
        "--monitor-model",
        monitor_model,
    ]
    monitor_budget = _assignment_monitor_budget_usd(spec.assignment)
    if monitor_budget is not None:
        command.extend(("--monitor-budget-usd", str(monitor_budget)))
    return tuple(command)


def worker_command_factory(
    repo_root: Path,
    session: str,
    *,
    python_executable: str | None = None,
    monitor_model: str = MODEL_MONITOR,
) -> Callable[[LaunchSpec], tuple[str, ...]]:
    """Return the command callable accepted by SubprocessLauncher."""

    def command(spec: LaunchSpec) -> tuple[str, ...]:
        return worker_command(
            spec,
            repo_root=repo_root,
            session=session,
            python_executable=python_executable,
            monitor_model=monitor_model,
        )

    return command


async def execute_worker(
    config: EntrypointConfig,
    *,
    environ: Mapping[str, str] | None = None,
    store: Store | None = None,
    store_factory: Callable[[Path, str], Store] = Store.open,
    llm_factory: Callable[..., Any] = LLM,
    client_factory: Callable[[Any], Any] | None = None,
    brain_factory: Callable[..., SubBrain] = SubBrain,
    judge_factory: Callable[..., Any] = LLMMonitorJudge,
    monitor_factory: Callable[..., Any] = MonitorBrain,
    runtime_factory: Callable[..., WorkerRuntime] = WorkerRuntime,
    ready_callback: Callable[[], Any] = mark_worker_ready,
) -> WorkerExitCode:
    """Run one worker and return a classified process exit.

    An injected ``store`` remains caller-owned.  A production store opened by
    this function is closed on every confirmed-shutdown path.  On
    ``SHUTDOWN_UNCONFIRMED`` it deliberately remains open until process exit,
    retaining the branch lease while an SDK child may still exist.
    """
    environment = os.environ if environ is None else environ
    owned_store = store is None
    shutdown_unconfirmed = False
    opened = store
    brain: SubBrain | None = None
    if opened is None:
        try:
            opened = store_factory(config.repo_root, config.session)
        except Exception:
            return WorkerExitCode.INFRA_FAILURE
    elif opened.root != config.repo_root or opened.session != config.session:
        return WorkerExitCode.INPUT_REJECTED

    outcome = WorkerExitCode.INFRA_FAILURE
    try:
        try:
            durable = _load_durable_input(opened, config, environment)
            if durable.contract.budget_usd is not None and client_factory is not None:
                raise EntrypointInputError(
                    "budgeted workers cannot use an injected SDK client factory"
                )
        except EntrypointInputError:
            outcome = WorkerExitCode.INPUT_REJECTED
        else:
            # Credential/pricing validation precedes the worker lease. A
            # broken monitor provider must not strand a runnable branch.
            try:
                monitor_llm = llm_factory(
                    env_dir=config.repo_root,
                    budget_usd=config.monitor_budget_usd,
                    run_id=f"{durable.run_id}.monitor",
                    # One monitor judgement is one independently durable unit.
                    # Retrying inside the facade would reserve (and could bill)
                    # several complete calls before the monitor can persist an
                    # outcome; let the worker/runtime classify one failed call
                    # instead.
                    max_attempts=1,
                    # WorkerReport and the central goal ledger both account
                    # invoice (billed) dollars, so the process-local monitor
                    # guard must enforce that same currency.
                    cap_on="billed",
                )
                ensure_ready = getattr(monitor_llm, "ensure_ready", None)
                if not callable(ensure_ready):
                    raise TypeError("LLM factory returned no ensure_ready()")
                ensure_ready(config.monitor_model)
                judge = judge_factory(
                    monitor_llm,
                    model=config.monitor_model,
                    max_tokens=config.monitor_max_tokens,
                )
            except Exception:
                outcome = WorkerExitCode.INFRA_FAILURE
            else:
                try:
                    brain = brain_factory(
                        opened,
                        durable.contract,
                        model=durable.assignment.model,
                    )
                except Exception:
                    outcome = WorkerExitCode.INFRA_FAILURE
                else:
                    try:
                        _validate_acquired_brain(brain, opened, config, durable)
                    except EntrypointInputError:
                        outcome = WorkerExitCode.INPUT_REJECTED
                    else:
                        try:
                            monitor = monitor_factory(
                                opened,
                                durable.contract,
                                judge,
                                batch_size=config.monitor_batch_size,
                            )

                            def ready(identity: str) -> Any:
                                if identity != durable.assignment.worker:
                                    raise ContractMismatch("runtime readiness identity changed")
                                return ready_callback()

                            runtime_kwargs: dict[str, Any] = {
                                "assignment": durable.assignment,
                                "run_id": durable.run_id,
                                "communicator": Communicator(opened),
                                "ready": ready,
                                "poll_interval": config.poll_interval,
                                "terminal_quiet_period": config.terminal_quiet_period,
                                "shutdown_timeout": config.shutdown_timeout,
                            }
                            if client_factory is not None:
                                runtime_kwargs["client_factory"] = client_factory
                            runtime = runtime_factory(brain, monitor, **runtime_kwargs)
                        except Exception:
                            outcome = WorkerExitCode.INFRA_FAILURE
                        else:
                            try:
                                result: SubBrainResult = await runtime.run()
                            except ShutdownUnconfirmed:
                                shutdown_unconfirmed = True
                                outcome = WorkerExitCode.SHUTDOWN_UNCONFIRMED
                            except asyncio.CancelledError:
                                outcome = WorkerExitCode.INTERRUPTED
                            except ContractMismatch:
                                outcome = WorkerExitCode.INPUT_REJECTED
                            except Exception:
                                outcome = WorkerExitCode.RUNTIME_FAILURE
                            else:
                                outcome = (
                                    WorkerExitCode.COMPLETED
                                    if result.completed
                                    else WorkerExitCode.INCOMPLETE
                                )
    finally:
        cleanup_failed = False
        if shutdown_unconfirmed:
            _QUARANTINED_RESOURCES.append((opened, brain))
        else:
            if brain is not None:
                try:
                    brain.close()
                except Exception:
                    cleanup_failed = True
            if owned_store:
                try:
                    opened.close()
                except Exception:
                    cleanup_failed = True
            if cleanup_failed:
                # Keep possibly-live lease objects reachable and never let a
                # cleanup failure preserve a reassuring completion exit.
                _QUARANTINED_RESOURCES.append((opened, brain))
                outcome = WorkerExitCode.RUNTIME_FAILURE
    return outcome


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one exact supervised Taste worker")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--session", required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--model", required=True, dest="expected_model")
    parser.add_argument("--prepared-state", required=True, dest="prepared_state_id")
    parser.add_argument("--monitor-model", default=MODEL_MONITOR)
    parser.add_argument("--monitor-max-tokens", default=1024, type=_positive_int)
    parser.add_argument("--monitor-batch-size", default=BATCH_SIZE, type=_positive_int)
    parser.add_argument("--monitor-budget-usd", type=_positive_float)
    parser.add_argument("--poll-interval", default=0.05, type=_positive_float)
    parser.add_argument("--terminal-quiet-period", default=0.25, type=_positive_float)
    parser.add_argument("--shutdown-timeout", default=10.0, type=_positive_float)
    return parser


def _run_with_signals(config: EntrypointConfig) -> WorkerExitCode:
    async def supervise_signals() -> WorkerExitCode:
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(execute_worker(config), name="taste-worker-entrypoint")
        installed: list[signal.Signals] = []
        for caught in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(caught, task.cancel)
            except (NotImplementedError, RuntimeError):
                continue
            installed.append(caught)
        try:
            return await task
        finally:
            for caught in installed:
                loop.remove_signal_handler(caught)

    try:
        return asyncio.run(supervise_signals())
    except KeyboardInterrupt:
        return WorkerExitCode.INTERRUPTED


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint. Diagnostics are categorical and never include secrets."""
    namespace = _parser().parse_args(argv)
    try:
        config = EntrypointConfig(**vars(namespace))
    except ValueError:
        return int(WorkerExitCode.INPUT_REJECTED)
    return int(_run_with_signals(config))


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``
    raise SystemExit(main())
