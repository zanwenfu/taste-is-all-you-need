"""A hosted agent as a worker: the real entrypoint, Azure wire and terminal RPC, mini-swe-agent unchanged."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from taste.benchmarks.goal_trajectory import _hosted_calls
from taste.benchmarks.worker_trajectory import hosted_trajectory
from taste.brains.azure_worker_entrypoint import execute_worker, run_directory
from taste.brains.azure_worker_policy import AzureWorkerPolicy
from taste.brains.hosted_worker import hosted_task
from taste.brains.records import ArtifactSpec, contract_digest
from taste.brains.responses_session import ResponsesSession
from taste.brains.terminal_broker import TerminalResult
from taste.brains.worker_admission import WorkerExitCode
from taste.brains.worker_commands import recorded_commands
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_azure_terminal_worker import terminal
from tests.test_azure_worker_runtime import install, report
from tests.test_azure_worker_runtime import worker as _worker
from tests.test_openai_responses import function_call, message
from tests.test_responses_monitor import verdict
from tests.test_worker_admission import install_assignment
from tests.test_worker_admission import launch as _launch

sdk_transport = _sdk_transport
launch = _launch
worker = _worker
SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def hosted(worker, **policy):
    source = worker.assignment
    contract = replace(source.contract, task="Make the parser tests pass.", inputs=(), outputs=("report.md",),
                       success_criteria=("make test exits 0",))
    raw = {**source.resources["azure_openai"], "worker_agent": "mini-swe-agent", **policy}
    assignment = replace(source, contract=contract, contract_digest=contract_digest(contract), inputs=(),
                         outputs=(ArtifactSpec("report", "report.md", kind="report"),),
                         resources={**source.resources, "azure_openai": raw})
    worker.assignment = assignment
    install_assignment(worker, assignment)
    return assignment


def bash(command, call_id):
    return function_call(json.dumps({"command": command}), name="bash", call_id=call_id, id="fc_" + call_id)


def scripted(env, outputs):
    calls = env.calls

    def execute(request):
        calls.append(request)
        for key, result in outputs.items():
            if key in request.command:
                return result
        return TerminalResult(0, b"", b"")

    env.execute = execute


def judged(severity):
    def reply(role, payload):
        if role == "terminal":
            return verdict(terminal=True)
        value = json.loads(verdict())
        value.update(severity=severity, reason=f"judged {severity}")
        return json.dumps(value)
    return reply


def test_an_unchanged_agent_fixes_the_task_and_its_report_is_certified(worker, sdk_transport, tmp_path):
    replies = {1: [message("Let me run the tests."), bash("make test", "c1")],
               2: [bash("sed -i 's/x/y/' parser.py && make test", "c2")],
               3: [bash(f"echo {SENTINEL}", "c3")]}
    sent, calls, _ = install(sdk_transport, worker_reply=lambda number, _: replies[number])

    async def scenario():
        hosted(worker)
        async with terminal(worker, tmp_path) as t:
            scripted(t.env, {"sed -i": TerminalResult(0, b"4 passed\n", b""),
                             "make test": TerminalResult(1, b"FAILED test_parse\n", b""),
                             SENTINEL: TerminalResult(0, f"{SENTINEL}\n".encode(), b"")})
            assert await execute_worker(worker.config, store=worker.store, environ=worker.environ) == WorkerExitCode.COMPLETED
            result = report(worker)
            assert result.completed and not result.uncertain, result.uncertainty_reasons
            assert result.metadata["harness"] == "hosted/1"
            assert result.metadata["agent"]["name"] == "mini-swe-agent"
            # The agent's own commands ran in the task container, in its own shell form.
            commands = [call.command for call in t.env.calls]
            assert [command.split("{\n", 1)[1] for command in commands] == [
                "make test\n} 2>&1", "sed -i 's/x/y/' parser.py && make test\n} 2>&1", f"echo {SENTINEL}\n}} 2>&1"]
            assert all(call.timeout_seconds == 5 for call in t.env.calls)  # the grant's limit under upstream's 30
            # Its requests are its own: its system message, its bash tool, no instructions of ours.
            first = json.loads(sent[0].content)
            assert "instructions" not in first and first["tools"][0]["name"] == "bash"
            assert first["input"][0] == {"role": "system",
                                         "content": "You are a helpful assistant that can interact with a computer."}
            assert first["input"][1]["content"].startswith("Please solve this issue: Make the parser tests pass.")
            # The report the coordinator gets says how it ended and what ran.
            body = worker.store.state(result.final_state_id).read("report.md")
            assert "Exit: Submitted" in body and "Let me run the tests." in body
            assert "ran: make test -> exit 1" in body
            turns = worker.store.state(result.final_state_id).transcript.turns
            assert len(recorded_commands(turns, run_id=result.run_id)) == 3
            trace = json.loads((run_directory(worker.store, result.run_id) / "worker/trajectory.worker.json").read_text())
            assert trace["agent"]["name"] == "mini-swe-agent"
            assert [step["source"] for step in trace["steps"]] == ["user", "agent", "agent", "agent"]
            assert trace["extra"]["exit"] == {"exit_status": "Submitted", "stopped_by": ""}
            assert calls["worker"] == 3
            # Settlement reads the same record back against the paid receipts.
            policy = AzureWorkerPolicy.from_assignment(worker.assignment)
            session = ResponsesSession.open(run_directory(worker.store, result.run_id) / "worker", policy.worker,
                                            policy.azure_config(worker.environ))
            try:
                rows, evidence = session.conversation_audit(), session.call_evidence()
            finally:
                session.close()
            projected, gaps = hosted_trajectory(rows, run_id=result.run_id), []
            _hosted_calls(projected, rows, evidence, gaps, result.run_id)
            assert gaps == [] and not projected["extra"]["incomplete_worker_trace"]
            agent_steps = [step for step in projected["steps"] if step["source"] == "agent"]
            assert len(agent_steps) == 3 and all("metrics" in step for step in agent_steps)
    asyncio.run(scenario())


def test_an_agent_its_monitor_judges_wrong_is_stopped_at_its_next_call(worker, sdk_transport, tmp_path):
    _, calls, _ = install(sdk_transport, worker_reply=lambda number, _: [bash("rm -rf src", f"c{number}")],
                             monitor_reply=judged("wrong"))

    async def scenario():
        hosted(worker, monitor_batch_size=1)
        worker.config = replace(worker.config, monitor_batch_size=1)
        async with terminal(worker, tmp_path) as t:
            assert await execute_worker(worker.config, store=worker.store, environ=worker.environ) == WorkerExitCode.INCOMPLETE
            result = report(worker)
            assert not result.completed
            assert len(t.env.calls) == 1 and calls["worker"] == 1
            body = worker.store.state(result.final_state_id).read("report.md")
            assert "(stopped: monitor_wrong)" in body
    asyncio.run(scenario())


def test_a_hosted_agent_needs_the_task_terminal_and_one_report(worker, sdk_transport):
    install(sdk_transport)
    hosted(worker)
    # No terminal admitted: refused before any paid call.
    assert asyncio.run(execute_worker(worker.config, store=worker.store, environ=worker.environ)) in {
        WorkerExitCode.INPUT_REJECTED, WorkerExitCode.RUNTIME_FAILURE, WorkerExitCode.INFRA_FAILURE}


def test_the_agent_is_given_the_original_task_alone_when_that_is_its_assignment(worker):
    assignment = hosted(worker)
    contract = replace(assignment.contract, task="Fix the build.\n")
    verbatim = replace(assignment, contract=contract, contract_digest=contract_digest(contract))
    assert hosted_task(verbatim, "Fix the build.") == "Fix the build."
    planned = hosted_task(assignment, "The original words of the task.")
    assert planned.startswith("Make the parser tests pass.\n\nDone when:\n- make test exits 0")
    assert planned.endswith("The original words of the task.")
    assert hosted_task(assignment, None) == "Make the parser tests pass.\n\nDone when:\n- make test exits 0"


@pytest.mark.parametrize("damage", [{"worker_agent": "claude-code"}, {"worker_agent": ""}])
def test_an_unknown_agent_is_refused_at_admission(worker, damage):
    from taste.brains.azure_worker_policy import AzureWorkerPolicy
    from taste.brains.worker_admission import EntrypointInputError

    assignment = hosted(worker, **damage)
    with pytest.raises(EntrypointInputError):
        AzureWorkerPolicy.from_assignment(assignment)
