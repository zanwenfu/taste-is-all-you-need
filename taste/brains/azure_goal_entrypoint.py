"""Run or settle a prepared Azure goal in an externally owned process scope.

The existing GoalProcessInput wire format pins the non-secret Azure policy in
Goal.metadata. Credentials remain in the host environment. An outer owner must
still bound and drain this entire process scope, including worker descendants.
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from taste.brains.azure_central_host import compose_azure_central_runtime
from taste.brains.azure_execution_policy import POLICY_KEY, AzureExecutionPolicy
from taste.brains.goal_entrypoint import (
    GoalInputError,
    execute_goal,
    goal_command,
    load_goal_input,
    prepare_goal_process,
)
from taste.brains.python_process import isolated_python_argv
from taste.providers.azure_openai import AZURE_MONITOR_MODEL, AZURE_PLANNER_MODEL
from taste.providers.base import ProtocolFailure


def _host_factory(*args, policy, environment, planner_model, monitor_model, planner_max_tokens,
                  settlement_only=False, **kwargs):
    if (planner_model != AZURE_PLANNER_MODEL or monitor_model != AZURE_MONITOR_MODEL
            or planner_max_tokens != policy.planner_max_output_tokens):
        raise GoalInputError("Azure goal model options differ from the admitted policy")
    return compose_azure_central_runtime(*args, policy=policy, environment=environment,
                                         settlement_only=settlement_only, **kwargs)


def prepare_azure_goal_process(
    repo_root, session, goal, *, policy: AzureExecutionPolicy,
    max_generations, wall_clock_seconds, max_planner_failures=3, environment=None,
):
    """Admit exact Azure policy and its original deadline without model calls."""
    return prepare_goal_process(
        repo_root, session, policy.bind_goal(goal), max_generations=max_generations,
        wall_clock_seconds=wall_clock_seconds, max_planner_failures=max_planner_failures,
        deadline_at=datetime.fromtimestamp(policy.deadline_unix, UTC),
        planner_model=AZURE_PLANNER_MODEL, monitor_model=AZURE_MONITOR_MODEL,
        planner_max_tokens=policy.planner_max_output_tokens,
        host_factory=partial(_host_factory, policy=policy, environment=environment),
    )


def _policy(config):
    try:
        policy = AzureExecutionPolicy.from_dict(config.goal.metadata.get(POLICY_KEY))
        if (config.planner_model != AZURE_PLANNER_MODEL or config.monitor_model != AZURE_MONITOR_MODEL
                or config.planner_max_tokens != policy.planner_max_output_tokens
                or datetime.fromisoformat(config.limits["deadline_at"].replace("Z", "+00:00")).timestamp()
                != datetime.fromtimestamp(policy.deadline_unix, UTC).timestamp()):
            raise ValueError("model or deadline differs")
    except (ValueError, TypeError, OverflowError, ProtocolFailure) as exc:
        raise GoalInputError("goal input does not bind one exact Azure execution policy") from exc
    return policy


async def execute_azure_goal(config, *, mode="run", environment=None):
    policy = _policy(config)
    return await execute_goal(config, mode=mode,
                              host_factory=partial(_host_factory, policy=policy, environment=environment,
                                                   settlement_only=mode == "settle"))


def azure_goal_command(input_path, expected_sha256, *, mode="run", python_executable=None):
    # Share the input/path/mode checks with the historical entrypoint.
    goal_command(input_path, expected_sha256, mode=mode, python_executable=python_executable)
    return isolated_python_argv(
        python_executable or sys.executable,
        "from taste.brains.azure_goal_entrypoint import main; raise SystemExit(main())",
        ["--input", str(input_path), "--sha256", expected_sha256, "--mode", mode],
    )


async def _run(config, mode):
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(execute_azure_goal(config, mode=mode))
    installed = []
    try:
        for caught in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(caught, task.cancel)
            installed.append(caught)
        return await task
    finally:
        for caught in installed:
            loop.remove_signal_handler(caught)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run or settle one prepared Azure Taste goal")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--mode", choices=("run", "settle"), default="run")
    args = parser.parse_args(argv)
    try:
        config = load_goal_input(args.input, args.sha256)
        _policy(config)
    except (ValueError, TypeError, KeyError, AttributeError, OSError):
        print("Azure goal input rejected", file=sys.stderr)
        return 65
    try:
        outcome = asyncio.run(_run(config, args.mode))
    except (asyncio.CancelledError, KeyboardInterrupt):
        return 130
    except BaseException:
        print("Azure goal execution or cleanup failed; inspect durable state", file=sys.stderr)
        return 70
    return 0 if outcome.complete else 10


if __name__ == "__main__":
    raise SystemExit(main())
