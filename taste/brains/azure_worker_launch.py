"""Opt-in Azure process commands; the historical worker launcher is unchanged."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from taste.brains.azure_worker_policy import AzureWorkerPolicy
from taste.brains.python_process import isolated_python_argv
from taste.brains.supervisor import LaunchSpec
from taste.brains.terminal_worker_policy import TerminalWorkerPolicy, install_terminal_credential
from taste.brains.worker_admission import EntrypointConfig
from taste.brains.worker_protocol import assignment_run_id
from taste.memstore import Store


def worker_command(spec: LaunchSpec, *, repo_root: Path, session: str,
                   python_executable: str | None = None, terminal_credential_provider=None) -> tuple[str, ...]:
    policy = AzureWorkerPolicy.from_assignment(spec.assignment)
    if spec.run_id != assignment_run_id(spec.assignment):
        raise ValueError("LaunchSpec run_id differs from its assignment")
    config = EntrypointConfig(
        repo_root, session, spec.assignment.worker, policy.worker.model, spec.prepared_state_id,
        monitor_model=policy.monitor.model, monitor_max_tokens=policy.monitor.max_output_tokens,
        monitor_batch_size=policy.monitor_batch_size, monitor_budget_usd=policy.monitor.budget_usd)
    executable = python_executable or sys.executable
    if not isinstance(executable, str) or not executable or "\x00" in executable:
        raise ValueError("invalid Python executable")
    terminal = TerminalWorkerPolicy.from_assignment(spec.assignment)
    if terminal is not None:
        if terminal_credential_provider is None:
            raise ValueError("terminal assignment requires a trusted credential provider")
        credential = terminal_credential_provider(spec)
        store = Store.open(repo_root, session)
        try:
            install_terminal_credential(store, spec.assignment, spec.prepared_state_id, credential)
        finally:
            store.close()
    return tuple(isolated_python_argv(executable,
        "import runpy; runpy.run_module('taste.brains.azure_worker_entrypoint', "
        "run_name='__main__', alter_sys=True)", [
            "--repo-root", str(config.repo_root), "--session", config.session,
            "--worker", config.worker, "--model", config.expected_model,
            "--prepared-state", config.prepared_state_id, "--monitor-model", config.monitor_model,
            "--monitor-max-tokens", str(config.monitor_max_tokens),
            "--monitor-batch-size", str(config.monitor_batch_size),
            "--monitor-budget-usd", str(config.monitor_budget_usd),
        ]))


def worker_command_factory(repo_root: Path, session: str, *, python_executable: str | None = None,
                           terminal_credential_provider=None,
                           ) -> Callable[[LaunchSpec], tuple[str, ...]]:
    def command(spec):
        return worker_command(spec, repo_root=repo_root, session=session, python_executable=python_executable,
                              terminal_credential_provider=terminal_credential_provider)
    return command
