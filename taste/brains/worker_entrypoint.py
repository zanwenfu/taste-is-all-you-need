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
import os
import signal
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from taste.brains.communication import Communicator
from taste.brains.monitor import BATCH_SIZE, MonitorBrain
from taste.brains.monitor_judge import LLMMonitorJudge
from taste.brains.subbrain import SubBrain, SubBrainResult
from taste.brains.supervisor import mark_worker_ready
from taste.brains.worker_admission import (
    DurableWorkerInput,
    EntrypointConfig,
    EntrypointInputError,
    WorkerExitCode,
    _load_durable_input,
    validate_acquired_branch,
)
from taste.brains.worker_launch import worker_command, worker_command_factory
from taste.brains.worker_protocol import (
    ContractMismatch,
    ShutdownUnconfirmed,
    assignment_run_id,
)
from taste.brains.worker_runtime import WorkerRuntime
from taste.llm import LLM, MODEL_MONITOR
from taste.memstore import Store

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

# A disconnect which cannot prove child reaping is a process quarantine, not
# an ordinary exception.  Keeping strong references prevents CPython's cyclic
# GC from closing the flock between ``execute_worker`` returning and the
# supervisor observing process exit.  The OS releases everything at process
# death; tests may explicitly close their injected Store afterwards.
_QUARANTINED_RESOURCES: list[tuple[Store, SubBrain | None]] = []


def _validate_acquired_brain(
    brain: SubBrain,
    store: Store,
    config: EntrypointConfig,
    durable: DurableWorkerInput,
) -> None:
    """Close the lease-free read/acquire race before constructing an SDK client."""
    if brain.store is not store or brain.contract != durable.contract:
        raise EntrypointInputError("worker factory changed the exact store or contract")
    validate_acquired_branch(brain.branch, store, config, durable)


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
            traceback.print_exc()
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
        except EntrypointInputError as exc:
            # The reason is the whole value of this check. Measured: a live
            # worker exited 65 and 70 with zero bytes on any stream and no
            # durable record, so diagnosing it meant calling this loader by
            # hand from a saved repository. An exit code is not evidence.
            print(f"worker input rejected: {exc}", file=sys.stderr, flush=True)
            outcome = WorkerExitCode.INPUT_REJECTED
        else:
            # Credential/pricing validation precedes the worker lease. A
            # broken monitor provider must not strand a runnable branch.
            try:
                monitor_llm = llm_factory(
                    env_dir=config.repo_root,
                    # Provider configuration comes from the launch owner.
                    # A task-controlled .env must never modify this process.
                    load_env_file=False,
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
                                # Same reason: this bare catch turned every
                                # worker fault into an opaque 70 with nothing
                                # written anywhere. The supervisor inherits
                                # stderr precisely so this is visible.
                                traceback.print_exc()
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
