"""An agent run alone: the same goal machinery with the planner, monitors and certifier off."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sqlite3
import tempfile
from dataclasses import replace
from pathlib import Path

from taste.brains import azure_worker_launch, benchmark_reply
from taste.brains.azure_central_host import compose_azure_central_runtime
from taste.brains.azure_worker_entrypoint import run_directory
from taste.brains.central_planner import Goal
from taste.brains.records import WorkerReport
from taste.brains.single_run import FIXED_PLAN_MODEL, single_run_proposal
from taste.brains.terminal_broker import TerminalBroker, TerminalResult
from taste.brains.terminal_service import TerminalCredential, TerminalGrant, TerminalService
from taste.brains.worker_protocol import WORKER_REPORT_PATH
from tests.test_azure_central_host import environment, install_planner
from tests.test_azure_central_host import policy as _policy
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_azure_terminal_policy import bound
from tests.test_azure_worker_process import BOOTSTRAP
from tests.test_terminal_broker import Environment

policy, sdk_transport = _policy, _sdk_transport
SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

ALONE = BOOTSTRAP.replace("install(network)", """
import json
from tests.test_openai_responses import function_call, message
def agent(number, payload):
    if number == 1:
        return [message("I ran the tests and they pass."),
                function_call(json.dumps({"command": "make test"}), name="bash", call_id="c1", id="fc_c1")]
    return [function_call(json.dumps({"command": "echo SENTINEL"}), name="bash", call_id="c2", id="fc_c2")]
install(network, worker_reply=agent)
""".replace("SENTINEL", SENTINEL))


def alone(policy):
    return replace(bound(policy), worker_agent="mini-swe-agent", services="none",
                   planner_model=FIXED_PLAN_MODEL, planner_deployment="fixed-plan")


class Scripted(Environment):
    def execute(self, request):
        self.calls.append(request)
        if SENTINEL in request.command:
            return TerminalResult(0, f"{SENTINEL}\n".encode(), b"")
        return TerminalResult(0, b"3 passed\n", b"")


def test_the_agent_alone_runs_once_unjudged_and_its_own_words_are_the_reply(tmp_path, policy, sdk_transport, monkeypatch):
    sent, _ = install_planner(sdk_transport)
    admitted = alone(policy)
    goal = Goal(goal_id="azure-goal", task="Make the tests pass.", success_criteria=("make test passes",),
                budget_usd=100, metadata={benchmark_reply.KEY: benchmark_reply.SCHEMA,
                                          benchmark_reply.RESERVE_KEY: 10})
    original = azure_worker_launch.worker_command

    def command(*args, **kwargs):
        argv = list(original(*args, **kwargs))
        argv[3] = argv[3].replace("import runpy;", ALONE + "\nsys.modules.pop('taste.brains.azure_worker_entrypoint', None)\nimport runpy;", 1)
        return tuple(argv)

    monkeypatch.setattr(azure_worker_launch, "worker_command", command)

    async def scenario():
        env = Scripted()
        owner = TerminalBroker.create(tmp_path / "terminal-ledger", admitted.terminal.binding, env)
        with tempfile.TemporaryDirectory(prefix="taste-alone-rpc-", dir="/tmp") as directory:
            socket_path = str(Path(directory) / "service" / "terminal.sock")
            seed = TerminalCredential(socket_path, os.geteuid(), os.geteuid(),
                                      TerminalGrant(owner.binding, "controller_bootstrap", 5), "a" * 64)
            service = TerminalService(owner, [seed])
            await service.start()
            loop = asyncio.get_running_loop()

            def issue(spec):
                credential = TerminalCredential(socket_path, os.geteuid(), os.geteuid(),
                    admitted.terminal.grant(spec.assignment), secrets.token_hex(32))

                async def register():
                    service.authorize(credential)

                asyncio.run_coroutine_threadsafe(register(), loop).result(timeout=5)
                return credential

            root = tmp_path / "repo"
            root.mkdir()
            try:
                with compose_azure_central_runtime(root, "alone", goal, policy=admitted,
                        environment=environment(), terminal_credential_provider=issue) as runtime:
                    result = await runtime.run_async(max_generations=1, wall_clock_seconds=60)
                    assert result.stop_reason == "generation_bound" and not result.complete, json.dumps(
                        {k: v for k, v in result.to_dict().items() if k in ("detail", "budget", "runs")})
                    plan = runtime.planner.current_plan(goal.goal_id)
                    assert plan.metadata["proposal"]["final_reply"] == "I ran the tests and they pass."
                    (run,) = runtime.supervisor.runs()
                    # The task as given, verbatim, was the agent's whole assignment.
                    assert run.assignment.contract.task == "Make the tests pass."
                    report = WorkerReport.from_json(
                        runtime.store.view(run.assignment.worker).head.read(WORKER_REPORT_PATH))
                    # The agent submitted; with nothing to certify it, its claim stands as made.
                    assert report.completed and report.metadata["supervised"] is False
                    with sqlite3.connect(run_directory(runtime.store, run.run_id) / "monitor" / "calls.sqlite3") as journal:
                        assert journal.execute("select count(*) from calls").fetchone() == (0,)
                    assert [call.command.split("{\n", 1)[1] for call in env.calls] == [
                        "make test\n} 2>&1", f"echo {SENTINEL}\n}} 2>&1"]
                assert sent == []  # nothing reached the Azure planner: the plan is fixed
            finally:
                await service.close()
                owner.close()
    asyncio.run(scenario())


def test_the_agent_alone_continues_in_the_same_environment_until_the_generations_run_out(
        tmp_path, policy, sdk_transport, monkeypatch):
    sent, _ = install_planner(sdk_transport)
    admitted = alone(policy)
    goal = Goal(goal_id="azure-goal", task="Make the tests pass.", success_criteria=("make test passes",),
                budget_usd=100, metadata={benchmark_reply.KEY: benchmark_reply.SCHEMA,
                                          benchmark_reply.RESERVE_KEY: 10})
    original = azure_worker_launch.worker_command

    def command(*args, **kwargs):
        argv = list(original(*args, **kwargs))
        argv[3] = argv[3].replace("import runpy;", ALONE + "\nsys.modules.pop('taste.brains.azure_worker_entrypoint', None)\nimport runpy;", 1)
        return tuple(argv)

    monkeypatch.setattr(azure_worker_launch, "worker_command", command)

    async def scenario():
        env = Scripted()
        owner = TerminalBroker.create(tmp_path / "terminal-ledger", admitted.terminal.binding, env)
        with tempfile.TemporaryDirectory(prefix="taste-alone-rpc-", dir="/tmp") as directory:
            socket_path = str(Path(directory) / "service" / "terminal.sock")
            seed = TerminalCredential(socket_path, os.geteuid(), os.geteuid(),
                                      TerminalGrant(owner.binding, "controller_bootstrap", 5), "a" * 64)
            service = TerminalService(owner, [seed])
            await service.start()
            loop = asyncio.get_running_loop()

            def issue(spec):
                credential = TerminalCredential(socket_path, os.geteuid(), os.geteuid(),
                    admitted.terminal.grant(spec.assignment), secrets.token_hex(32))

                async def register():
                    service.authorize(credential)

                asyncio.run_coroutine_threadsafe(register(), loop).result(timeout=5)
                return credential

            root = tmp_path / "repo"
            root.mkdir()
            try:
                with compose_azure_central_runtime(root, "alone", goal, policy=admitted,
                        environment=environment(), terminal_credential_provider=issue) as runtime:
                    result = await runtime.run_async(max_generations=3, wall_clock_seconds=120)
                    assert result.stop_reason == "generation_bound", result.detail
                    runs = sorted(runtime.supervisor.runs(), key=lambda run: run.assignment.generation)
                    # Three runs, each given the task as given, in the one environment.
                    assert [run.assignment.assignment_id for run in runs] == ["agent-run", "agent-run-2",
                                                                              "agent-run-3"]
                    assert {run.assignment.contract.task for run in runs} == {"Make the tests pass."}
                    assert len(env.calls) == 6
                    plan = runtime.planner.current_plan(goal.goal_id)
                    assert plan.metadata["proposal"]["final_reply"] == "I ran the tests and they pass."
                assert sent == []
            finally:
                await service.close()
                owner.close()
    asyncio.run(scenario())


def test_a_later_generation_runs_the_agent_again_on_the_task_as_given():
    """The continue control (#34): the agent alone, run again in the same environment each
    generation the runtime allows, given the task verbatim; nothing plans, judges or certifies."""
    exemplar = {"model": "m", "resources": {}, "contract": {"budget_usd": 2.0, "max_turns": 1000}}
    payload = {"request": {"operation_id": "op", "generation": 2, "parent_plan": {"plan_id": "p"},
                           "goal": {"task": "Make the tests pass.", "success_criteria": ["make test passes"]},
                           "world": {"integration_state_id": "a" * 40}},
               "required_output_shape": {"assignments": [exemplar]}}
    proposal = json.loads(single_run_proposal(json.dumps(payload)))
    [run] = proposal["assignments"]
    assert run["assignment_id"] == "agent-run-2" and run["generation"] == 2
    assert run["contract"]["identity"] == "agent-2" and run["contract"]["task"] == "Make the tests pass."
    assert proposal["complete"] is False


def test_the_closing_reply_without_a_report_says_so():
    payload = {"request": {"operation_id": benchmark_reply.CLOSING_OPERATION_PREFIX + "x", "generation": 2,
                           "goal": {"task": "t"}, "world": {"integration_state_id": "a" * 40, "outcomes": []}},
               "required_output_shape": {"assignments": [{}]},
               "standing_criteria": [{"criterion_id": "c1", "text": "done"}]}
    proposal = json.loads(single_run_proposal(json.dumps(payload)))
    assert proposal["assignments"] == [] and proposal["complete"] is False
    assert proposal["metadata"]["final_reply"] == "The agent ran once and ended without a report."
    assert proposal["assessment"] == [{"criterion_id": "c1", "verdict": "not_met",
                                       "evidence": "No one judged the agent's work in this run."}]
