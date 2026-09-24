"""Nonsecret worker launch commands, independent of model SDK imports."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from pathlib import Path

from taste.brains.supervisor import LaunchSpec
from taste.brains.worker_protocol import _assignment_monitor_budget_usd, assignment_run_id
from taste.llm import MODEL_MONITOR
from taste.memstore.store import _check_name

_EXACT_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")

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
    from taste.brains.python_process import isolated_python_argv

    command = isolated_python_argv(executable,
        "import runpy; runpy.run_module('taste.brains.worker_entrypoint', "
        "run_name='__main__', alter_sys=True)", [
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
    ])
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

