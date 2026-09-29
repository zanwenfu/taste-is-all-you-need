"""Durable, bounded exchange between an outside owner and Azure goal services.

Preparation never calls a model. Run uses systemd credentials exclusively;
settlement uses none. Each operation claims a fresh private result directory
before doing work. A missing result is incomplete execution, never success or
permission to replay. The outside owner must drain the whole service before
reading a result, and before launching a separate settlement operation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
import os
import re
import stat
import sys
import time
import uuid
from pathlib import Path

from taste.brains.azure_execution_policy import AzureExecutionPolicy
from taste.brains.azure_goal_entrypoint import _policy, _run, prepare_azure_goal_process
from taste.brains.central_planner import Goal
from taste.brains.central_runtime import GoalOutcome
from taste.brains.goal_entrypoint import (
    GoalInputError,
    GoalProcessInput,
    _canonical,
    _decode,
    load_goal_input,
    python_source_digest,
)
from taste.brains.input_limits import MAX_GOAL_INPUT_BYTES
from taste.brains.python_process import isolated_python_argv
from taste.memstore.store import _check_name

MAX_RESULT_BYTES = 1024 * 1024
_PREPARATION_SCHEMA = "taste.brains/AzureGoalPreparation/1"
_RESULT_SCHEMA = "taste.brains/AzureGoalHandoff/1"


def preparation_bytes(repo_root, session, goal, policy, *, max_generations, wall_clock_seconds,
                      max_planner_failures=3):
    """Create non-secret, hash-pinned input for an unprivileged preparation service."""
    value = {"schema": _PREPARATION_SCHEMA, "repo_root": str(repo_root), "session": session,
             "goal": goal.to_dict(), "policy": policy.to_dict(), "max_generations": max_generations,
             "wall_clock_seconds": wall_clock_seconds, "max_planner_failures": max_planner_failures,
             "python_source_sha256": python_source_digest()}
    _preparation(value)
    raw = (_canonical(value) + "\n").encode()
    if len(raw) > MAX_GOAL_INPUT_BYTES:
        raise GoalInputError("Azure preparation input exceeds 512 KiB")
    return raw


def _preparation(value):
    if (not isinstance(value, dict) or set(value) != {"schema", "repo_root", "session", "goal", "policy",
            "max_generations", "wall_clock_seconds", "max_planner_failures", "python_source_sha256"}
            or value["schema"] != _PREPARATION_SCHEMA
            or value["python_source_sha256"] != python_source_digest()):
        raise GoalInputError("Azure preparation differs from its admitted source or schema")
    for field in ("max_generations", "max_planner_failures"):
        if type(value[field]) is not int or not 1 <= value[field] <= 10000:
            raise GoalInputError("invalid Azure preparation count limit")
    wall = value["wall_clock_seconds"]
    if type(wall) not in (int, float) or not math.isfinite(wall) or not 0 < wall <= 604800:
        raise GoalInputError("invalid Azure preparation wall limit")
    if not isinstance(value["repo_root"], str) or not isinstance(value["session"], str):
        raise GoalInputError("invalid Azure preparation workspace")
    _check_name(value["session"], "session")
    root = Path(value["repo_root"])
    if not root.is_absolute() or str(root.resolve(strict=True)) != str(root) or not root.is_dir():
        raise GoalInputError("Azure preparation requires an existing canonical workspace")
    # Memory branches live in sibling .taste-worktrees directories. A fresh
    # trial needs a writable enclosing state directory, separate from the
    # outside owner's protected input, credentials and result exchange paths.
    if not os.access(root, os.W_OK | os.X_OK) or not os.access(root.parent, os.W_OK | os.X_OK):
        raise GoalInputError("Azure preparation requires a writable enclosing workspace directory")
    goal, policy = Goal.from_dict(value["goal"]), AzureExecutionPolicy.from_dict(value["policy"])
    if goal.budget_usd is None or goal.budget_usd <= 0:
        raise GoalInputError("Azure preparation requires a positive goal budget")
    if not 0 < policy.deadline_unix - time.time() <= wall:
        raise GoalInputError("Azure preparation deadline is expired or exceeds its wall limit")
    return root, goal, policy


def _read(fd, name, maximum, *, uid=None, mode=None):
    descriptor = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=fd)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not 0 < info.st_size <= maximum
                or (uid is not None and info.st_uid != uid)
                or (mode is not None and stat.S_IMODE(info.st_mode) != mode)):
            raise GoalInputError("handoff input must be a bounded owned regular file")
        data = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
        if (len(data) != info.st_size or len(data) > maximum
                or (info.st_mtime_ns, info.st_ctime_ns) != (after.st_mtime_ns, after.st_ctime_ns)):
            raise GoalInputError("handoff file changed during observation")
        return data


def _write(fd, name, data):
    if len(data) > MAX_RESULT_BYTES:
        raise GoalInputError("Azure goal result exceeds 1 MiB")
    temporary = ".handoff-" + uuid.uuid4().hex
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Exclusive publication; an existing result must never be overwritten.
        os.link(temporary, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
    finally:
        os.unlink(temporary, dir_fd=fd)
        os.fsync(fd)


def _open_result_directory(path, uid):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    info = os.fstat(descriptor)
    if info.st_uid != uid or stat.S_IMODE(info.st_mode) != 0o700:
        os.close(descriptor)
        raise GoalInputError("handoff directory must be private to its service UID")
    return descriptor


def read_handoff(directory, input_sha256, operation, *, service_uid):
    """Outside owner only, AFTER service drainage; absence is an incomplete operation.

    Directory ancestors must remain controlled by the owner. No model worker
    may write this exchange directory. Return canonical raw result data and its
    digest so the owner can bind its own durable lifecycle record to the copy.
    """
    _arguments(input_sha256, operation)
    fd = _open_result_directory(directory, service_uid)
    try:
        intent = _decode(_read(fd, "intent.json", 1024, uid=service_uid, mode=0o600))
        raw = _read(fd, "result.json", MAX_RESULT_BYTES, uid=service_uid, mode=0o600)
    finally:
        os.close(fd)
    expected = {"schema": _RESULT_SCHEMA, "input_sha256": input_sha256, "operation": operation}
    value = _decode(raw)
    if (intent != expected or not isinstance(value, dict) or set(value) != {*expected, "result"}
            or any(value[key] != item for key, item in expected.items())):
        raise GoalInputError("handoff result differs from its admitted operation")
    return value["result"], hashlib.sha256(raw).hexdigest()


def settled_outcome(value, config: GoalProcessInput):
    """Validate reported goal identity/accounting before deciding whether to grade.

    Unknown spend is a valid interrupted outcome. It remains visibly unknown;
    callers must use grading_ready(), not just outcome.complete.
    """
    try:
        outcome = GoalOutcome.from_dict(value)
        if _canonical(outcome.to_dict()) != _canonical(value) or outcome.goal_id != config.goal.goal_id:
            raise ValueError
        if type(outcome.complete) is not bool or not isinstance(outcome.stop_reason, str) or not outcome.stop_reason:
            raise ValueError
        if outcome.complete != (outcome.stop_reason == "complete"):
            raise ValueError
        for count in (outcome.cycles, outcome.generations):
            if type(count) is not int or count < 0:
                raise ValueError
        if any(not isinstance(item, str) for item in (outcome.detail, outcome.completion_reason)):
            raise ValueError
        budget = outcome.budget
        if type(budget.limit_usd) not in (int, float) or budget.limit_usd != config.goal.budget_usd:
            raise ValueError
        for number in (budget.known_spent_usd, budget.reserved_usd, budget.planner_spent_usd, budget.worker_spent_usd):
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise ValueError
        if not math.isclose(budget.known_spent_usd, budget.planner_spent_usd + budget.worker_spent_usd,
                            rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError
        for identities in (budget.unknown_run_ids, budget.unbounded_live_run_ids,
                           budget.unknown_planner_attempt_ids, outcome.delivered_assignment_ids):
            if any(not isinstance(item, str) or not item for item in identities) or len(set(identities)) != len(identities):
                raise ValueError
        return outcome
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
        raise GoalInputError("goal handoff has invalid identity or accounting") from None


def grading_ready(outcome):
    """Model completion is separate from benchmark reward; this is only admission."""
    return (outcome.budget.enforceable and outcome.budget.reserved_usd == 0
            and outcome.budget.known_spent_usd <= outcome.budget.limit_usd
            and outcome.stop_reason in {"complete", "budget_blocked", "generation_bound", "wall_clock"})


def _arguments(digest, operation):
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise GoalInputError("invalid handoff input digest")
    if operation not in {"prepare", "run", "settle"}:
        raise GoalInputError("invalid handoff operation")


def handoff_command(input_path, input_sha256, output_dir, *, operation, python_executable=None):
    _arguments(input_sha256, operation)
    if any(not Path(path).is_absolute() for path in (input_path, output_dir)):
        raise GoalInputError("handoff paths must be absolute")
    return isolated_python_argv(python_executable or sys.executable,
        "from taste.brains.azure_goal_handoff import main; raise SystemExit(main())",
        ["--input", str(input_path), "--sha256", input_sha256, "--output", str(output_dir), "--operation", operation])


def perform(input_path, digest, output_dir, *, operation):
    _arguments(digest, operation)
    if operation == "prepare":
        raw = _read(None, input_path, MAX_GOAL_INPUT_BYTES)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise GoalInputError("preparation input digest differs")
        value = _decode(raw)
        root, goal, policy = _preparation(value)
        # A restarted preparer must not import already-mutated state as a new
        # initial workspace or refresh the prepared goal's deadline.
        if any(root.iterdir()):
            raise GoalInputError("trial preparation requires a fresh empty workspace")
    else:
        config = load_goal_input(input_path, digest)
        _policy(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(mode=0o700)  # Existing/incomplete exchanges are never reused.
    fd = _open_result_directory(output_dir, os.geteuid())
    try:
        binding = {"schema": _RESULT_SCHEMA, "input_sha256": digest, "operation": operation}
        _write(fd, "intent.json", (_canonical(binding) + "\n").encode())
        if operation == "prepare":
            config = prepare_azure_goal_process(root, value["session"], goal, policy=policy, environment={},
                max_generations=value["max_generations"], wall_clock_seconds=value["wall_clock_seconds"],
                max_planner_failures=value["max_planner_failures"])
            result = _decode(config.to_bytes())
        else:
            # The production entrypoint owns signals, host closure and worker
            # drainage. A cancelled call publishes no successful result here.
            outcome = asyncio.run(_run(config, operation, systemd_credentials=True))
            result = settled_outcome(outcome.to_dict(), config).to_dict()
        _write(fd, "result.json", (_canonical({**binding, "result": result}) + "\n").encode())
        return result
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operation", choices=("prepare", "run", "settle"), required=True)
    args = parser.parse_args(argv)
    try:
        perform(args.input, args.sha256, args.output, operation=args.operation)
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 130
    except BaseException:
        print("Azure goal handoff incomplete; retain state for recovery", file=sys.stderr)
        return 70
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
