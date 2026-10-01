"""Run or settle one already admitted goal in an owned outer process scope.

The launch owner supplies a hash-pinned, non-secret input file and retains it
outside task write access. Preparation makes no model call and launches no
worker. Execution never creates missing admission limits or refreshes their
deadline. A successful process exit is not a benchmark reward; the durable
GoalOutcome and confirmed outer-scope drainage must be inspected separately.

This is a trusted runtime entrypoint, not a sandbox or a Harbor adapter. The
owner must enforce hard termination (including descendants) outside this
process. Authentication comes from the trusted service environment.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import signal
import stat
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from taste.brains.central_host import (
    _await_owned_task,
    _owned_call,
    compose_central_runtime,
)
from taste.brains.central_planner import Goal, _goal_path
from taste.brains.central_runtime import GoalOutcome, _digest, _goal_root
from taste.brains.input_limits import MAX_GOAL_INPUT_BYTES
from taste.brains.owned_thread import start_owned_thread
from taste.brains.python_process import isolated_python_argv
from taste.brains.worker_protocol import GOAL_TASK_PATH
from taste.llm import MODEL_MONITOR, MODEL_PLANNER
from taste.memstore import Store
from taste.memstore.store import _check_name

_SCHEMA = "taste.brains/GoalProcessInput/1"
_MAX_INPUT_BYTES = MAX_GOAL_INPUT_BYTES


class GoalInputError(ValueError):
    """The launch input does not match the prepared durable goal."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise GoalInputError("duplicate input key")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise GoalInputError("non-finite input number")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)


def python_source_digest() -> str:
    """Pin this package's Python bytes; dependency/image pins remain external."""
    package = Path(__file__).resolve().parents[1]
    manifest = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(package.rglob("*.py"))}
    return hashlib.sha256(_canonical(manifest).encode()).hexdigest()


@dataclass(frozen=True)
class GoalProcessInput:
    repo_root: str
    session: str
    goal: Goal
    prepared_state_id: str
    run_limits_json: str
    python_source_sha256: str
    planner_model: str = MODEL_PLANNER
    monitor_model: str = MODEL_MONITOR
    planner_max_tokens: int = 8192

    def __post_init__(self):
        root = Path(self.repo_root)
        if (not isinstance(self.repo_root, str) or not root.is_absolute()
                or str(root.resolve(strict=True)) != self.repo_root
                or not root.is_dir() or not (root / ".git").exists()):
            raise GoalInputError("repo_root must be an existing canonical Git worktree")
        _check_name(self.session, "session")
        if not isinstance(self.goal, Goal) or self.goal.budget_usd is None or self.goal.budget_usd <= 0:
            raise GoalInputError("goal processes require a positive finite budget")
        for value, pattern in ((self.prepared_state_id, r"[0-9a-f]{40}|[0-9a-f]{64}"),
                               (self.python_source_sha256, r"[0-9a-f]{64}")):
            if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
                raise GoalInputError("invalid prepared state or Python source digest")
        for value in (self.planner_model, self.monitor_model):
            if (not isinstance(value, str) or not value or value != value.strip()
                    or len(value) > 256 or any(ord(char) < 32 for char in value)):
                raise GoalInputError("invalid model identifier")
        if type(self.planner_max_tokens) is not int or self.planner_max_tokens <= 0:
            raise GoalInputError("planner_max_tokens must be a positive integer")
        limits = _decode(self.run_limits_json)
        fields = {"schema", "goal_digest", "max_generations", "wall_clock_seconds",
                  "max_planner_failures", "deadline_at"}
        if (not isinstance(limits, dict) or set(limits) != fields
                or limits["schema"] != "taste.brains/GoalRunLimits/1"
                or limits["goal_digest"] != _digest(self.goal.to_json())
                or any(type(limits[key]) is not int or limits[key] <= 0
                       for key in ("max_generations", "max_planner_failures"))
                or type(limits["wall_clock_seconds"]) not in (int, float)
                or not math.isfinite(limits["wall_clock_seconds"])
                or limits["wall_clock_seconds"] <= 0):
            raise GoalInputError("invalid prepared run limits")
        deadline = datetime.fromisoformat(limits["deadline_at"].replace("Z", "+00:00"))
        if deadline.tzinfo is None or deadline.utcoffset() is None:
            raise GoalInputError("prepared deadline must include its timezone")
        object.__setattr__(self, "run_limits_json", _canonical(limits))
        if len(self.to_bytes()) > _MAX_INPUT_BYTES:
            raise GoalInputError("goal input exceeds 512 KiB")

    @property
    def limits(self):
        return _decode(self.run_limits_json)

    def to_bytes(self) -> bytes:
        return (_canonical({"schema": _SCHEMA, "repo_root": self.repo_root,
                            "session": self.session, "goal": self.goal.to_dict(),
                            "prepared_state_id": self.prepared_state_id,
                            "run_limits": self.limits, "python_source_sha256": self.python_source_sha256,
                            "planner_model": self.planner_model, "monitor_model": self.monitor_model,
                            "planner_max_tokens": self.planner_max_tokens}) + "\n").encode()

    @classmethod
    def from_bytes(cls, raw: bytes):
        if len(raw) > _MAX_INPUT_BYTES:
            raise GoalInputError("goal input exceeds 512 KiB")
        value = _decode(raw)
        fields = {"schema", "repo_root", "session", "goal", "prepared_state_id", "run_limits",
                  "python_source_sha256", "planner_model", "monitor_model", "planner_max_tokens"}
        if not isinstance(value, dict) or set(value) != fields or value.pop("schema") != _SCHEMA:
            raise GoalInputError("invalid goal process input fields or schema")
        value["goal"] = Goal.from_dict(value["goal"])
        value["run_limits_json"] = _canonical(value.pop("run_limits"))
        return cls(**value)


@contextmanager
def _closing_host(host):
    try:
        yield host
    except BaseException as original:
        try:
            host.close()
        except BaseException as cleanup:
            raise BaseExceptionGroup("goal operation and resource close failed", [original, cleanup]) from None
        raise
    else:
        host.close()


def prepare_goal_process(
    repo_root, session, goal, *, max_generations, wall_clock_seconds, deadline_at,
    max_planner_failures=3, planner_model=MODEL_PLANNER, monitor_model=MODEL_MONITOR,
    planner_max_tokens=8192, host_factory=compose_central_runtime, share_task=False,
) -> GoalProcessInput:
    """Prepare once before launch; the owner persists the returned exact bytes.

    The Python-only factory seam replaces external boundaries in tests. Input
    files cannot name an import, factory, executable, or environment override.
    With ``share_task`` the goal's task is committed to the integration branch
    before any plan, so each worker branch carries the original text.
    """
    if not isinstance(goal, Goal) or goal.budget_usd is None or goal.budget_usd <= 0:
        raise GoalInputError("goal processes require a positive finite budget")
    with _closing_host(host_factory(
        repo_root, session, goal, planner_model=planner_model, monitor_model=monitor_model,
        planner_max_tokens=planner_max_tokens, python_executable=sys.executable,
    )) as host:
        if share_task:
            shared = host.integration.head.read(GOAL_TASK_PATH)
            if shared is None:
                host.integration.write(GOAL_TASK_PATH, goal.task)
                host.integration.checkpoint("the goal's original task, for every worker")
            elif shared != goal.task:
                raise GoalInputError("integration already carries another goal's task")
        limits = host.prepare_run(max_generations=max_generations, wall_clock_seconds=wall_clock_seconds,
                                  deadline_at=deadline_at, max_planner_failures=max_planner_failures)
        return GoalProcessInput(
            str(host.store.root), session, goal, host.control.head.id, _canonical(limits),
            python_source_digest(), planner_model, monitor_model, planner_max_tokens,
        )


def _validate_admission(store, config):
    prepared = store.state(config.prepared_state_id)
    current = store.view("central-control").head
    if (prepared.meta.session != config.session or prepared.branch != "central-control"
            or not store.backend.is_ancestor(prepared.id, current.id)):
        raise GoalInputError("prepared admission is outside current control history")
    path = f"{_goal_root(config.goal.goal_id)}/run-limits.json"
    for state in (prepared, current):
        if (_canonical(state.record(path)) != config.run_limits_json
                or _canonical(state.record(_goal_path(config.goal.goal_id)))
                != _canonical(config.goal.to_dict())):
            raise GoalInputError("prepared goal or deadline differs from durable state")


async def execute_goal(config: GoalProcessInput, *, mode="run", host_factory=compose_central_runtime,
                       on_settled=None) -> GoalOutcome:
    """Hold ownership through repeated cancellation, drainage and resource close."""
    if mode not in {"run", "settle"}:
        raise GoalInputError("mode must be run or settle")
    if on_settled is not None and mode != "settle":
        raise GoalInputError("settled evidence can only be exported during settlement")
    if config.python_source_sha256 != python_source_digest():
        raise GoalInputError("Python source differs from prepared launch input")
    # Check before writable composition so recovery cannot silently create a
    # missing control branch. Repeat under the host's write lease as well.
    with _closing_host(Store.open(Path(config.repo_root), config.session)) as store:
        _validate_admission(store, config)
        with _closing_host(host_factory(
            config.repo_root, config.session, config.goal, store=store,
            planner_model=config.planner_model, monitor_model=config.monitor_model,
            planner_max_tokens=config.planner_max_tokens, python_executable=sys.executable,
        )) as host:
            _validate_admission(store, config)
            if mode == "run":
                limits = config.limits
                return await host.run_async(**{key: limits[key] for key in (
                    "max_generations", "wall_clock_seconds", "max_planner_failures",
                )})
            # Settlement must never enter run(), plan, launch, or refresh a bound.
            task = start_owned_thread(
                _owned_call, lambda: host.stop_and_drain("outer process scope terminated"),
            )
            try:
                await asyncio.wait((task,))
                outcome = task.result()
            except BaseException as original:
                try:
                    await _await_owned_task(task)
                except BaseException as cleanup:
                    if cleanup is not original:
                        raise BaseExceptionGroup("goal settlement failed", [original, cleanup]) from None
                raise
            if on_settled is not None:
                on_settled(host, outcome)
            return outcome


def load_goal_input(path: Path, expected_sha256: str) -> GoalProcessInput:
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise GoalInputError("invalid input digest")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise GoalInputError("goal input must be a regular file")
        raw = stream.read(_MAX_INPUT_BYTES + 1)
    if len(raw) > _MAX_INPUT_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise GoalInputError("goal input size or digest differs")
    return GoalProcessInput.from_bytes(raw)


def goal_command(input_path: Path, expected_sha256: str, *, mode="run", python_executable=None) -> list[str]:
    """Bind trusted source imports, excluding task cwd, PYTHONPATH and user site."""
    if not input_path.is_absolute() or mode not in {"run", "settle"}:
        raise GoalInputError("input path must be absolute and mode must be run or settle")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise GoalInputError("invalid input digest")
    return isolated_python_argv(
        python_executable or sys.executable,
        "from taste.brains.goal_entrypoint import main; raise SystemExit(main())",
        ["--input", str(input_path), "--sha256", expected_sha256, "--mode", mode],
    )


async def _run_with_signals(config, mode):
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(execute_goal(config, mode=mode))
    installed = []
    try:
        for caught in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(caught, task.cancel)
            installed.append(caught)
        return await task
    finally:
        for caught in installed:
            loop.remove_signal_handler(caught)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or settle one prepared Taste goal")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--mode", choices=("run", "settle"), default="run")
    args = parser.parse_args(argv)
    try:
        config = load_goal_input(args.input, args.sha256)
    except (ValueError, TypeError, KeyError, AttributeError, OSError):
        print("goal input rejected", file=sys.stderr)
        return 65
    try:
        outcome = asyncio.run(_run_with_signals(config, args.mode))
    except (asyncio.CancelledError, KeyboardInterrupt):
        return 130
    except BaseException:
        # Provider exceptions may contain keys or task data; the journal is
        # the recovery source. Do not echo arbitrary exception text to logs.
        print("goal execution or cleanup failed; inspect durable state", file=sys.stderr)
        return 70
    return 0 if outcome.complete else 10


if __name__ == "__main__":
    raise SystemExit(main())
