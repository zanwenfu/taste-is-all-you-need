"""Explicit Azure worker process entrypoint; never imports the Claude harness.

This admits a fresh exact assignment once. Recovery is deliberately separate:
an existing or partially initialized run directory is an error, not permission
to create a new spending allowance. The central supervisor must reap a previous
process before attempting recovery or preparing another assignment.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import signal
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from taste.brains.azure_worker_policy import AzureWorkerPolicy
from taste.brains.azure_worker_runtime import AzureWorkerRuntime
from taste.brains.monitor import MonitorBrain
from taste.brains.responses_monitor import ResponsesMonitorJudge
from taste.brains.responses_session import ResponsesSession
from taste.brains.supervisor import mark_worker_ready
from taste.brains.worker_admission import (
    EntrypointConfig,
    EntrypointInputError,
    WorkerExitCode,
    _load_durable_input,
    validate_acquired_branch,
)
from taste.memstore import Store


def run_directory(store: Store, run_id: str) -> Path:
    return store.backend.common_dir / (
        f"taste.azure.{store.session}." + hashlib.sha256(run_id.encode()).hexdigest())


async def execute_worker(config: EntrypointConfig, *, environ: Mapping[str, str] | None = None,
                         store: Store | None = None,
                         ready_callback: Callable[[], None] | None = None) -> WorkerExitCode:
    environment = os.environ if environ is None else environ
    owned = store is None
    branch = session = None
    outcome = WorkerExitCode.INFRA_FAILURE
    try:
        if store is None:
            store = Store.open(config.repo_root, config.session)
        durable = _load_durable_input(store, config, environment)
        policy = AzureWorkerPolicy.from_assignment(durable.assignment)
        policy.validate_launch(config)
        azure = policy.azure_config(environment)
        directory = run_directory(store, durable.run_id)
        if os.path.lexists(directory):
            raise EntrypointInputError("existing Azure run requires explicit recovery")
        branch = store.branch(config.worker)
        validate_acquired_branch(branch, store, config, durable)
        if branch.view.pending_turns():
            raise EntrypointInputError("fresh Azure launch cannot inherit pending worker turns")
        monitor = MonitorBrain(store, durable.contract, None, batch_size=policy.monitor_batch_size,
                               run_id=durable.run_id)
        if monitor._state_path().exists():
            raise EntrypointInputError("existing Azure monitor requires explicit recovery")
        # Exclusive parent creation is the irreversible admission marker. Any
        # partial initialization remains fenced, including a missing child DB.
        directory.mkdir(mode=0o700)
        fd = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        session = ResponsesSession.create(directory / "worker", policy.worker, azure)
        judge = ResponsesMonitorJudge.create(directory / "monitor", policy.monitor, azure)
        monitor = MonitorBrain(store, durable.contract, judge, batch_size=policy.monitor_batch_size,
                               run_id=durable.run_id)
        runtime = AzureWorkerRuntime(branch, durable.assignment, session, monitor)
        if ready_callback is None:
            mark_worker_ready(environ=environment)
        else:
            ready_callback()
        result = await runtime.run()
        outcome = (WorkerExitCode.INTERRUPTED if result.interrupted else
                   WorkerExitCode.COMPLETED if result.report.completed else WorkerExitCode.INCOMPLETE)
    except EntrypointInputError as exc:
        print(f"Azure worker input rejected: {exc}", file=sys.stderr, flush=True)
        outcome = WorkerExitCode.INPUT_REJECTED
    except asyncio.CancelledError:
        outcome = WorkerExitCode.INTERRUPTED
    except Exception as exc:
        print(f"Azure worker failed: {type(exc).__name__}", file=sys.stderr, flush=True)
        outcome = WorkerExitCode.RUNTIME_FAILURE if session is not None else WorkerExitCode.INFRA_FAILURE
    finally:
        # Model and monitor operations retain ownership through cancellation;
        # runtime return/exception cannot leave a live provider thread behind.
        for resource in (session, branch, store if owned else None):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    print(f"Azure worker cleanup failed: {type(exc).__name__}", file=sys.stderr, flush=True)
                    outcome = WorkerExitCode.RUNTIME_FAILURE
    return outcome


def _parser():
    parser = argparse.ArgumentParser(description="Run one exact Azure Responses worker")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--session", required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--model", required=True, dest="expected_model")
    parser.add_argument("--prepared-state", required=True, dest="prepared_state_id")
    parser.add_argument("--monitor-model", required=True)
    parser.add_argument("--monitor-max-tokens", required=True, type=int)
    parser.add_argument("--monitor-batch-size", required=True, type=int)
    parser.add_argument("--monitor-budget-usd", required=True, type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = _parser().parse_args(argv)
    try:
        config = EntrypointConfig(**vars(namespace))
    except ValueError:
        return int(WorkerExitCode.INPUT_REJECTED)

    async def run():
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(execute_worker(config))
        installed = []
        for caught in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(caught, task.cancel)
                installed.append(caught)
            except (NotImplementedError, RuntimeError):
                continue
        try:
            return await task
        finally:
            for caught in installed:
                loop.remove_signal_handler(caught)
    try:
        return int(asyncio.run(run()))
    except KeyboardInterrupt:
        return int(WorkerExitCode.INTERRUPTED)


if __name__ == "__main__":
    raise SystemExit(main())
