"""Durable goal bounds and deadlines crossing planner/provider boundaries."""
import threading
import time
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from taste.brains.central_host import compose_central_runtime
from taste.brains.central_runtime import CoordinatorError, _goal_root
from taste.brains.supervisor import ProcessExit
from taste.llm import LLM, MODEL_PLANNER, InfraFailure
from taste.memstore import Store
from taste.providers._anthropic import AnthropicProvider
from tests.test_brains_central_runtime import FakeLauncher, ScriptedTransport, simple_goal, stack
from tests.test_brains_runtime_boundaries import one
from tests.test_brains_supervisor import FakeClock


def test_real_anthropic_sdk_request_respects_short_llm_timeout():
    import anthropic

    requests = []

    class SlowHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(self.path)
            time.sleep(0.4)
            self.send_response(500)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = anthropic.Anthropic(api_key="test-only", base_url=f"http://127.0.0.1:{server.server_port}",
                                 max_retries=0, timeout=10)
    provider = AnthropicProvider(api_key="test-only")
    provider._client = client
    llm = LLM(max_attempts=1)
    # Keep the real facade, provider adapter and installed HTTP client; only
    # the remote service is a local server which deliberately stalls.
    llm.provider_for = lambda model: provider
    try:
        started = time.monotonic()
        with pytest.raises(InfraFailure):
            llm.call(model=MODEL_PLANNER, system="test", messages=[{"role": "user", "content": "test"}],
                     timeout_seconds=0.05)
        assert time.monotonic() - started < 2
        assert len(requests) == 1
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_expired_durable_deadline_survives_runtime_reconstruction(tmp_path):
    store = Store.open(tmp_path / "repo", "deadline")
    try:
        transport, launcher = ScriptedTransport(one), FakeLauncher()
        runtime, shared = stack(store, simple_goal(), transport, launcher)
        runtime.cycle()
        runtime.cycle()
        # Simulate the state left by a crashed host after publishing its trial
        # bounds. Restart must use that absolute deadline, not a new allowance.
        from taste.brains.central_runtime import _digest
        limits = {"schema": "taste.brains/GoalRunLimits/1", "goal_digest": _digest(runtime.goal.to_json()),
                  "max_generations": 2, "wall_clock_seconds": 30.0, "max_planner_failures": 3,
                  "deadline_at": (datetime.now(UTC) - timedelta(seconds=1)).isoformat()}
        runtime.control.checkpoint("prior host run bounds", records={
            f"{_goal_root(runtime.goal.goal_id)}/run-limits.json": limits,
        })
        restored, _ = stack(store, simple_goal(), transport, launcher, shared=shared)
        started = time.monotonic()
        outcome = restored.run(max_generations=2, wall_clock_seconds=30)
        assert time.monotonic() - started < 3
        assert outcome.stop_reason == "wall_clock"
        assert len(transport.calls) == len(launcher.launch_calls) == 1
        assert all(run.terminal for run in restored.supervisor.runs())
    finally:
        store.close()


def test_worker_deadline_is_capped_by_remaining_goal_time(tmp_path):
    store = Store.open(tmp_path / "repo", "deadline")
    try:
        runtime, _ = stack(store, simple_goal(), ScriptedTransport(one), FakeLauncher())
        observed = []

        def finish():
            for run in runtime.supervisor.runs():
                observed.append(run.wall_timeout_seconds)
                runtime.supervisor.launcher.handles[run.run_id].exit = ProcessExit(exit_code=1, reaped=True)

        runtime.run(max_generations=1, wall_clock_seconds=10, between_cycles=finish)
        assert observed and 0 < observed[0] <= 10
    finally:
        store.close()


def test_composed_planner_passes_remaining_deadline_to_llm(tmp_path):
    class DeadlineLLM:
        max_attempts = 1

        def ensure_ready(self, *models):
            pass

        def call(self, **kwargs):
            assert 0 < kwargs["timeout_seconds"] <= 5
            self.timeout = kwargs["timeout_seconds"]
            raise TimeoutError("deliberate test timeout")

    llm = DeadlineLLM()
    root = tmp_path / "repo"
    root.mkdir()
    host = compose_central_runtime(root, "deadline", simple_goal(), planner_llm=llm,
                                   launcher=FakeLauncher())
    try:
        outcome = host.run(max_generations=1, wall_clock_seconds=5, max_planner_failures=1)
        assert llm.timeout <= 5
        assert outcome.stop_reason == "planner_failed"
        assert len(outcome.budget.unknown_planner_attempt_ids) == 1
    finally:
        host.close()


def test_reentrant_run_cannot_overwrite_the_active_deadline(tmp_path):
    store = Store.open(tmp_path / "repo", "deadline")
    try:
        runtime, _ = stack(store, simple_goal(), ScriptedTransport(one), FakeLauncher())

        def nested_driver():
            runtime.run(max_generations=9, wall_clock_seconds=99)

        with pytest.raises(CoordinatorError, match="active run"):
            runtime.run(max_generations=1, wall_clock_seconds=10, between_cycles=nested_driver)
        assert runtime.outcome().stop_reason == "runtime_error"
        assert all(run.terminal for run in runtime.supervisor.runs())
    finally:
        store.close()


def test_expensive_preparation_cannot_launch_after_goal_deadline(tmp_path):
    store = Store.open(tmp_path / "repo", "deadline")
    try:
        launcher, clock = FakeLauncher(), FakeClock()
        runtime, _ = stack(store, simple_goal(), ScriptedTransport(one), launcher)
        runtime.clock = runtime.supervisor.clock = clock
        prepare = runtime.supervisor.prepare

        def slow_prepare(*args, **kwargs):
            result = prepare(*args, **kwargs)
            clock.advance(11)
            return result

        runtime.supervisor.prepare = slow_prepare
        outcome = runtime.run(max_generations=1, wall_clock_seconds=10)
        assert outcome.stop_reason == "wall_clock"
        assert launcher.launch_calls == []
        assert all(run.terminal for run in runtime.supervisor.runs())
    finally:
        store.close()


def test_recovered_prepared_run_uses_containing_goal_deadline(tmp_path):
    store = Store.open(tmp_path / "repo", "deadline")
    try:
        launcher, clock = FakeLauncher(), FakeClock()
        transport = ScriptedTransport(one)
        runtime, shared = stack(store, simple_goal(), transport, launcher)
        runtime.clock = runtime.supervisor.clock = clock
        plan = runtime.cycle().plan
        prepared = runtime.supervisor.prepare(plan.assignments[0], wall_timeout_seconds=60)
        restored, _ = stack(store, simple_goal(), transport, launcher, shared=shared)
        restored.clock = restored.supervisor.clock = clock
        expected_deadline = clock() + timedelta(seconds=10)

        def inspect_and_expire():
            run = restored.supervisor.get(prepared.run_id)
            assert datetime.fromisoformat(run.deadline_at) == expected_deadline
            clock.advance(11)

        outcome = restored.run(max_generations=1, wall_clock_seconds=10, between_cycles=inspect_and_expire)
        assert outcome.stop_reason == "wall_clock"
        assert len(launcher.launch_calls) == 1
        assert all(run.terminal for run in restored.supervisor.runs())
    finally:
        store.close()


@pytest.mark.parametrize("timeout", [True, None, "10", 0, -1, float("nan"), float("inf")])
def test_invalid_timeout_is_rejected_before_worker_preparation(tmp_path, timeout):
    store = Store.open(tmp_path / "repo", "deadline")
    try:
        runtime, _ = stack(store, simple_goal(), ScriptedTransport(one), FakeLauncher())
        assignment = runtime.cycle().plan.assignments[0]
        before = runtime.control.head.id
        with pytest.raises(ValueError, match="finite and positive"):
            runtime.supervisor.prepare(assignment, wall_timeout_seconds=timeout)
        assert runtime.control.head.id == before
        assert not runtime.supervisor.runs()
        assert not store.worktree_path_for(assignment.worker).exists()
    finally:
        store.close()
