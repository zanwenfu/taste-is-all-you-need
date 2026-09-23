"""Owned goal failures must not escape through asyncio's default log handler."""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta

import pytest

from taste.brains.central_host import compose_central_runtime
from taste.brains.goal_entrypoint import execute_goal, prepare_goal_process
from tests.test_brains_central_runtime import FakeLauncher, ScriptedTransport, simple_goal
from tests.test_goal_cancellation import _leaves, wait_event


@pytest.mark.parametrize("boundary", ["driver", "cleanup", "settlement"])
def test_owned_failure_reaches_caller_without_an_unsolicited_loop_diagnostic(tmp_path, boundary):
    entered, release = threading.Event(), threading.Event()
    marker = f"test-only-private-{boundary}-diagnostic"
    hosts = []

    def no_provider(*_args):
        pytest.fail("this failure test must not call a model")

    def factory(*args, **kwargs):
        host = compose_central_runtime(*args, **kwargs, transport=ScriptedTransport(no_provider),
                                       launcher=FakeLauncher())
        hosts.append(host)
        return host

    def delayed_failure(*_args, **_kwargs):
        entered.set()
        assert release.wait(10)
        raise RuntimeError(marker)

    repo = tmp_path / "repo"
    repo.mkdir()
    goal = simple_goal(budget_usd=5)
    if boundary == "settlement":
        config = prepare_goal_process(
            repo, "diagnostics", goal, max_generations=1, wall_clock_seconds=30,
            deadline_at=datetime.now(UTC) + timedelta(seconds=30), host_factory=factory,
        )

        def failing_factory(*args, **kwargs):
            host = factory(*args, **kwargs)
            host.stop_and_drain = delayed_failure
            return host

        def run():
            return execute_goal(config, mode="settle", host_factory=failing_factory)
    else:
        host = factory(repo, "diagnostics", goal)
        if boundary == "driver":
            host.run = delayed_failure
        else:
            def immediate_failure(**_kwargs):
                raise RuntimeError("test-only initial driver failure")

            host.run = immediate_failure
            host.stop_and_drain = delayed_failure

        def run():
            return host.run_async(max_generations=1, wall_clock_seconds=30)

    async def scenario():
        loop_errors = []
        asyncio.get_running_loop().set_exception_handler(lambda _loop, context: loop_errors.append(context))
        task = asyncio.create_task(run())
        try:
            await wait_event(entered)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0.02)
                assert not task.done(), "caller returned before the owned operation settled"
            release.set()
            with pytest.raises(BaseExceptionGroup) as caught:
                await task
            assert marker in [str(error) for error in _leaves(caught.value)]
            await asyncio.sleep(0)
            assert not loop_errors, "private failure escaped to an unsolicited event-loop diagnostic"
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    try:
        asyncio.run(scenario())
    finally:
        for host in hosts:
            host.close()
