"""Bounded planner recovery and the actual SDK inheritance boundary."""
import json
import os
import sys

import pytest

from taste.brains.central_planner import PlannerTransportError
from taste.brains.contract import Contract
from taste.brains.python_process import isolated_python_argv
from taste.brains.subbrain import SubBrain
from taste.brains.supervisor import CentralSupervisor, SubprocessLauncher
from taste.brains.worker_runtime import ContractMismatch, WorkerRuntime
from taste.memstore import Store
from tests.test_brains_central_runtime import (
    FakeLauncher,
    ScriptedTransport,
    complete_response,
    settle,
    simple_goal,
    stack,
)
from tests.test_brains_runtime_boundaries import one
from tests.test_brains_supervisor import assignment_for


@pytest.fixture
def store(tmp_path):
    opened = Store.open(tmp_path / "repo", "retry")
    yield opened
    opened.close()


def test_transient_failure_next_cycle_uses_fresh_world_and_preserves_old_attempt(store):
    failed = False

    def respond(request, prompt):
        nonlocal failed
        if request.generation == 1:
            return one(request, prompt)
        if not failed:
            failed = True
            raise RuntimeError("HTTP 529")
        return complete_response(request, prompt)

    launcher, transport = FakeLauncher(), ScriptedTransport(respond)
    runtime, _ = stack(store, simple_goal(), transport, launcher)
    runtime.cycle()
    runtime.cycle()
    settle(runtime, launcher, "build")
    with pytest.raises(PlannerTransportError, match="529"):
        runtime.cycle()
    assert runtime.cycle().complete
    assert len(transport.calls) == 3
    attempts = runtime.planner.planner_attempts(runtime.goal.goal_id)
    assert len(attempts) == 3
    assert sum(not a.telemetry.cost_known for a in attempts) == 1


@pytest.mark.parametrize("mode", ["transient", "invalid", "permanent"])
def test_run_recovers_planner_errors_with_finite_attempts(store, mode):
    calls = 0

    def respond(request, prompt):
        nonlocal calls
        calls += 1
        if calls == 1 or mode == "permanent":
            if mode == "invalid":
                return "not a planner JSON object"
            raise RuntimeError("HTTP 529")
        return one(request, prompt)

    launcher, transport = FakeLauncher(), ScriptedTransport(respond)
    runtime, _ = stack(store, simple_goal(), transport, launcher)
    outcome = runtime.run(max_generations=1, wall_clock_seconds=30,
                          between_cycles=lambda: settle(runtime, launcher, "build"))
    if mode == "permanent":
        assert outcome.stop_reason == "planner_failed"
        assert calls == 3 and not launcher.launch_calls
    else:
        assert outcome.stop_reason == "generation_bound"
        assert calls == 2 and len(launcher.launch_calls) == 1
        assert outcome.delivered_assignment_ids == ("build",)


def test_ambiguous_paid_error_stops_without_retry(store):
    def respond(request, prompt):
        raise RuntimeError("connection lost after dispatch")

    transport = ScriptedTransport(respond, call_ceiling_usd=0.5)
    runtime, _ = stack(store, simple_goal(budget_usd=5), transport, FakeLauncher())
    outcome = runtime.run(max_generations=3, wall_clock_seconds=30)
    assert outcome.stop_reason == "budget_blocked"
    assert len(transport.calls) == 1
    assert len(outcome.budget.unknown_planner_attempt_ids) == 1


def test_direct_sdk_client_refuses_inherited_overrides_without_mutating_parent(store, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://test.invalid")
    brain = SubBrain(store, Contract(identity="worker", task="test", success_criteria=("done",)))
    try:
        options = brain.options()
        assert "ANTHROPIC_BASE_URL" not in options.env
        with pytest.raises(ContractMismatch, match="inherited"):
            WorkerRuntime._make_client(options)
        assert os.environ["ANTHROPIC_BASE_URL"] == "https://test.invalid"
        assert options.max_turns is None
    finally:
        brain.close()


def test_real_worker_to_installed_sdk_environment_is_filtered(store, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://test.invalid")
    output = tmp_path / "sdk-environment.json"
    code = f"""
import asyncio, json
from pathlib import Path
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk._internal.transport import subprocess_cli as cli
async def capture(*args, **kwargs):
    env = kwargs['env']
    Path({str(output)!r}).write_text(json.dumps({{'override_present': 'ANTHROPIC_BASE_URL' in env,
                                               'ordinary': env.get('TASTE_TEST_ORDINARY')}}))
    raise RuntimeError('test captured process boundary; no CLI started')
cli.anyio.open_process = capture
transport = cli.SubprocessCLITransport(prompt='test', options=ClaudeAgentOptions(cli_path='/bin/false'))
try:
    asyncio.run(transport.connect())
except Exception:
    pass
"""
    launcher = SubprocessLauncher(isolated_python_argv(sys.executable, code, []), env={
        "ANTHROPIC_BASE_URL": "https://also.invalid",
        "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1", "TASTE_TEST_ORDINARY": "preserved",
    })
    supervisor = CentralSupervisor(store, launcher=launcher)
    run = supervisor.prepare(assignment_for(supervisor), wall_timeout_seconds=20)
    try:
        supervisor.start(run.run_id, active_generation=1)
        supervisor.wait(run.run_id, active_generation=1)
        assert json.loads(output.read_text()) == {"override_present": False, "ordinary": "preserved"}
        assert os.environ["ANTHROPIC_BASE_URL"] == "https://test.invalid"
    finally:
        supervisor.stop(run.run_id)
        supervisor.close()
