"""Stop requests cross async callers, the central driver, and worker cleanup."""

from __future__ import annotations

import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from taste.brains.central_host import compose_central_runtime
from taste.brains.central_planner import PlannerTransportError
from taste.brains.planner_transport import PlannerTelemetry, PlannerUsage
from taste.brains.supervisor import SubprocessLauncher
from tests.test_brains_central_runtime import (
    FakeLauncher,
    ScriptedTransport,
    assignment_for,
    simple_goal,
)


def assignments(request, _prompt):
    return (assignment_for(request, "build", "worker-build", "product.txt", budget_usd=1),)


@pytest.fixture
def make_host(tmp_path):
    hosts = []

    def make(*, responder=assignments, launcher=None, telemetry=None, budget=None):
        root = tmp_path / f"repo-{len(hosts)}"
        root.mkdir()
        host = compose_central_runtime(
            root, "cancellation-test", simple_goal(budget_usd=budget),
            transport=ScriptedTransport(responder, telemetry=telemetry),
            launcher=launcher or FakeLauncher(), supervisor_termination_grace=0.05,
        )
        hosts.append(host)
        return host

    try:
        yield make
    finally:
        for host in hosts:
            # Test-owned fake launchers need explicit cleanup on failed baseline
            # assertions too; the production host's old close does not stop them.
            for run in host.supervisor.runs():
                host.supervisor.stop(run.run_id, "test_cleanup")
            host.close()


def bounds():
    return {"max_generations": 5, "wall_clock_seconds": 3.0}


async def run_async(host, **kwargs):
    # Exercise the currently-used integration on the unpatched baseline.
    # The new host API must replace this unowned to_thread cancellation path.
    if hasattr(host, "run_async"):
        return await host.run_async(**kwargs)
    return await asyncio.to_thread(host.run, **kwargs)


async def wait_event(event):
    assert await asyncio.to_thread(event.wait, 10), "test boundary was not reached"


def test_stop_before_run_makes_no_provider_call_or_worker_launch(make_host):
    host = make_host()
    host.request_stop("caller cancelled before admission")
    outcome = host.run(**bounds())
    assert outcome.stop_reason == "cancelled" and not outcome.complete
    assert outcome.detail == "caller cancelled before admission"
    assert host.transport.calls == [] and host.launcher.launch_calls == []
    assert host.outcome() == outcome
    host.request_stop("a later request must not overwrite the first")
    assert host.run(**bounds()) == outcome


def test_stop_between_prepare_and_launch_never_spawns_the_prepared_run(make_host):
    host = make_host()

    def at_boundary(stage, _record):
        if stage == "decision_effect:prepare":
            host.request_stop("cancelled after preparation")

    host.runtime.fault_injector = at_boundary
    outcome = host.run(**bounds())
    assert outcome.stop_reason == "cancelled"
    assert host.launcher.launch_calls == []
    runs = host.supervisor.runs()
    assert len(runs) == 1 and runs[0].terminal and runs[0].recovery_status == "complete"
    assert not host.store.worktree_path_for(runs[0].assignment.worker).exists()


def test_stop_request_does_not_wait_for_the_planner_or_host_locks(make_host):
    entered, release = threading.Event(), threading.Event()

    def respond(request, prompt):
        entered.set()
        assert release.wait(10)
        return assignments(request, prompt)

    host = make_host(responder=respond)
    with ThreadPoolExecutor(max_workers=2) as pool:
        driver = pool.submit(host.run, **bounds())
        try:
            assert entered.wait(10)
            stop = pool.submit(host.request_stop, "external stop while planner is active")
            stop.result(timeout=1)
        finally:
            release.set()
        outcome = driver.result(timeout=10)
    assert outcome.stop_reason == "cancelled"
    assert len(host.transport.calls) == 1 and host.launcher.launch_calls == []


@pytest.mark.parametrize("cancel_count", [1, 3])
@pytest.mark.parametrize("provider_failure", [False, True])
def test_async_cancel_waits_for_planner_receipt_and_driver_exit(
    make_host, monkeypatch, cancel_count, provider_failure,
):
    entered, release, done = threading.Event(), threading.Event(), threading.Event()

    def respond(request, prompt):
        entered.set()
        assert release.wait(10)
        if provider_failure:
            raise PlannerTransportError("provider reply lost after dispatch")
        return assignments(request, prompt)

    telemetry = PlannerTelemetry(
        source="provider_completion", cost_known=True,
        requested_model="claude-sonnet-5", model="claude-sonnet-5", provider="anthropic",
        usage=PlannerUsage(100, 20, 0, 0, 0), billed_usd=0.25, work_usd=0.5,
        pricing_table_sha="test-pricing-table", pricing_as_of="2026-09-10",
    )
    host = make_host(responder=respond, telemetry=telemetry, budget=5)
    original = host.run

    def tracked(**kwargs):
        try:
            return original(**kwargs)
        finally:
            done.set()

    monkeypatch.setattr(host, "run", tracked)

    async def scenario():
        task = asyncio.create_task(run_async(host, **bounds()))
        try:
            await wait_event(entered)
            for _ in range(cancel_count):
                task.cancel()
                await asyncio.sleep(0.02)
                assert not task.done(), "cancellation returned while the goal driver was still active"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert done.is_set()
            outcome = host.outcome()
            assert outcome.stop_reason == "cancelled" and not outcome.complete
            if provider_failure:
                assert not outcome.budget.enforceable
                assert outcome.budget.unknown_planner_attempt_ids
            else:
                assert outcome.budget.planner_spent_usd == pytest.approx(0.25)
                assert outcome.budget.enforceable
            assert host.launcher.launch_calls == []
        finally:
            release.set()
            await wait_event(done)
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_async_cancel_stops_real_worker_writes_and_preserves_dirty_work(make_host):
    def command(spec):
        return [sys.executable, "-c", (
            "import time\nfrom pathlib import Path\n"
            f"path=Path({str(spec.worktree / 'product.txt')!r})\n"
            "while True:\n"
            "    with path.open('a') as out: out.write('worker effect\\n')\n"
            "    time.sleep(0.01)\n"
        )]

    host = make_host(launcher=SubprocessLauncher(command))
    worker_path = host.store.worktree_path_for("worker-build") / "product.txt"

    async def scenario():
        task = asyncio.create_task(run_async(host, **bounds()))
        try:
            for _ in range(1000):
                if worker_path.exists() and worker_path.stat().st_size:
                    break
                assert not task.done(), "driver ended before the real worker wrote"
                await asyncio.sleep(0.01)
            assert worker_path.exists() and worker_path.stat().st_size
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            outcome = host.outcome()
            assert outcome is not None and outcome.stop_reason == "cancelled"
            run, = host.supervisor.runs()
            assert run.terminal and run.recovery_status == "complete"
            assert host.store.state(run.recovery_state_id).read("product.txt")
            assert not worker_path.parent.exists()
            await asyncio.sleep(0.05)
            assert not worker_path.exists(), "worker effects continued after cancellation returned"
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            # Old to_thread cancellation loses the future; wait on the host's
            # lifecycle lock before allowing fixture cleanup to touch its Store.
            await asyncio.to_thread(host.outcome)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_stop_attempts_every_worker_and_retries_cleanup_without_new_work(make_host, monkeypatch, failure):
    def respond(request, _prompt):
        return tuple(assignment_for(request, name, f"worker-{name}", f"{name}.txt")
                     for name in ("first", "second"))

    host = make_host(responder=respond)
    host.cycle()
    host.cycle()
    original = host.supervisor.stop
    attempts = []
    fail = True

    def stop(run_id, reason="cancelled"):
        attempts.append(run_id)
        if fail and host.supervisor.get(run_id).assignment.assignment_id == "first":
            raise failure("first worker cleanup failed")
        return original(run_id, reason)

    monkeypatch.setattr(host.supervisor, "stop", stop)
    try:
        with pytest.raises(failure, match="first worker cleanup failed"):
            host.stop_and_drain("caller cancelled the goal")
        assert set(attempts) == {run.run_id for run in host.supervisor.runs()}
        assert host.outcome() is None
        calls = list(host.launcher.launch_calls)
        fail = False
        outcome = host.stop_and_drain("retry must retain the original stop reason")
        assert outcome.stop_reason == "cancelled" and outcome.detail == "caller cancelled the goal"
        assert all(run.terminal and run.recovery_status == "complete" for run in host.supervisor.runs())
        assert host.launcher.launch_calls == calls and len(host.transport.calls) == 1
    finally:
        monkeypatch.setattr(host.supervisor, "stop", original)


def _leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in _leaves(child)]
    return [error]


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_async_cancel_preserves_cleanup_failure_and_keeps_the_event_loop_alive(make_host, monkeypatch, failure):
    host = make_host()
    host.cycle()
    host.cycle()
    entered, release = threading.Event(), threading.Event()
    original = host.launcher.cancel

    def between_cycles():
        entered.set()
        assert release.wait(10)

    def fail_cancel(*args, **kwargs):
        raise failure("worker cleanup did not settle")

    monkeypatch.setattr(host.launcher, "cancel", fail_cancel)

    async def scenario():
        task = asyncio.create_task(run_async(host, **bounds(), between_cycles=between_cycles))
        try:
            await wait_event(entered)
            task.cancel()
            release.set()
            with pytest.raises(BaseExceptionGroup) as caught:
                await task
            errors = _leaves(caught.value)
            assert any(isinstance(error, asyncio.CancelledError) for error in errors)
            assert any(isinstance(error, failure) for error in errors)
            assert host.outcome() is None
            monkeypatch.setattr(host.launcher, "cancel", original)
            settled = await asyncio.to_thread(host.stop_and_drain)
            assert settled.stop_reason == "cancelled"
            assert all(run.terminal for run in host.supervisor.runs())
        finally:
            release.set()
            monkeypatch.setattr(host.launcher, "cancel", original)
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.to_thread(host.outcome)

    asyncio.run(scenario())


def test_async_driver_validation_failure_still_drains_existing_workers(make_host):
    host = make_host()
    host.cycle()
    host.cycle()

    async def scenario():
        with pytest.raises(ValueError, match="max_generations"):
            await run_async(host, max_generations=0, wall_clock_seconds=3)
        assert all(run.terminal and run.recovery_status == "complete" for run in host.supervisor.runs())
        assert host.outcome() is not None

    asyncio.run(scenario())
