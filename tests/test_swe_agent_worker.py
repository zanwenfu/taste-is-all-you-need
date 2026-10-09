"""SWE-agent through the real worker: entrypoint, Azure wire, terminal RPC, its record exported and branched.

As tests/test_branch_worker.py does for mini-swe-agent: a base run recorded as a
trial records it, its replay script exported from the worker's journal and the
terminal ledger, then branched by a fresh worker. SWE-agent runs commands of its
own before its first model call (its shell session, variables, tools): the
script's setup, rebuilt and answered from the record like the steps.
"""

from __future__ import annotations

import hashlib

import pytest

from taste.agents.swe_agent import session_command
from taste.benchmarks.replay_export import encode_script, export_script
from taste.brains.branch_replay import REJECT_EXIT, BranchPolicy
from taste.brains.terminal_broker import TerminalResult
from taste.brains.worker_admission import WorkerExitCode
from tests.test_branch_worker import launch, open_worker, run_agent, sdk_transport, worker
from tests.test_hosted_worker import bash
from tests.test_openai_responses import function_call, message

# The fixtures of the worker these tests run, as tests/test_branch_worker.py has them.
FIXTURES = (launch, sdk_transport, worker)

PATCH = "diff --git a/parser.py b/parser.py\n-x\n+y\n"
MARKER = "<<SWE_AGENT_SUBMISSION>>"
REJECTION = "Submission rejected by a reviewer: tabs are still dropped (tests/test_tabs.py fails)."


def submit(call_id):
    return function_call("{}", name="submit", call_id=call_id, id="fc_" + call_id)


REPLIES = {1: [message("Run the tests."), bash("make test", "c1")], 2: [submit("c2")]}
OUTPUTS = {session_command("make test"): TerminalResult(1, b"FAILED test_parse\n", b""),
           session_command("submit"): TerminalResult(0, f"{MARKER}\n{PATCH}{MARKER}\n".encode(), b""),
           "cat -- /root/model.patch": TerminalResult(0, PATCH.encode(), b"")}


@pytest.fixture(autouse=True)
def roomy_terminal(monkeypatch):
    """The harness's terminal allows 10 commands, as many as mini-swe-agent's runs need; SWE-agent's
    setup alone runs about two dozen."""
    import tests.test_azure_terminal_worker as harness

    binding = harness.TerminalBinding
    monkeypatch.setattr(harness, "TerminalBinding",
                        lambda trial, environment, deadline, _commands: binding(trial, environment, deadline, 500))


@pytest.fixture
def other(tmp_path):
    value = open_worker(tmp_path / "branch")
    yield value
    value.store.close()


@pytest.fixture
def base(worker, sdk_transport, tmp_path):
    run = run_agent(worker, sdk_transport, tmp_path, REPLIES, OUTPUTS, worker_agent="swe-agent")
    assert run.code == WorkerExitCode.COMPLETED and run.calls == 2, run.trace["extra"].get("exit")
    run.script = export_script(run.journal, ledger=run.ledger, trial="base")
    return run


def branch(other, sdk_transport, tmp_path, script, step, *, replies, **options):
    path = tmp_path / f"script-{step}.json"
    raw = encode_script(script)
    path.write_bytes(raw)
    fork = BranchPolicy(str(path), hashlib.sha256(raw).hexdigest(), step, **options)
    directory = tmp_path / f"branch-{step}"
    directory.mkdir()
    run = run_agent(other, sdk_transport, directory, replies, OUTPUTS, worker_agent="swe-agent",
                    branch=fork.to_dict())
    run.path = path
    return run


def test_the_export_of_a_swe_agent_run_holds_its_setup_and_its_steps(base):
    script = base.script
    assert script["source"]["agent"]["name"] == "swe-agent" and script["exit"]["exit_status"] == "Submitted"
    setup = [run["command"] for run in script["setup"]]
    assert setup[0].startswith("# swe-rex: start the shell session") and len(setup) >= 10
    assert len(script["steps"][0]["runs"]) == 3
    assert script["steps"][1]["runs"][0]["command"] == "submit" and script["submission_step"] == 2
    # In the trial's record the setup is the system's, before any reply: no step of the agent.
    before = [step for step in base.trace["steps"] if step["message"] == "command run before any model reply"]
    assert len(before) == len(setup) and all(step["source"] == "system" for step in before)


def test_a_branch_rebuilds_and_replays_the_setup_then_goes_live(base, other, sdk_transport, tmp_path):
    run = branch(other, sdk_transport, tmp_path, base.script, 1, replies={1: [submit("c3")]})
    assert run.code == WorkerExitCode.COMPLETED and run.calls == 1
    executed = [item["executed"] for item in base.script["setup"]] + [
        item["executed"] for item in base.script["steps"][0]["runs"]]
    assert run.commands[:len(executed)] == executed
    account = run.trace["extra"]["branch"]
    assert account["faithful"] and account["went_live"] and account["context"] == "matched"
    assert account["rebuild"]["commands"] == len(executed) and account["rebuild"]["divergent"] == 0
    # Its own record, exported, has the same setup, the replayed step and its live one.
    again = export_script(run.journal, ledger=run.ledger, parent=run.path, trial="branch")
    assert again["setup"] == base.script["setup"]
    assert [step["replayed"] for step in again["steps"]] == [True, False] and again["submission_step"] == 2


def test_a_rejected_submission_keeps_swe_agent_working(base, other, sdk_transport, tmp_path):
    run = branch(other, sdk_transport, tmp_path, base.script, 2, override="reject", note=REJECTION,
                 replies={1: [submit("c4")]})
    assert run.code == WorkerExitCode.COMPLETED and run.calls == 1
    last = run.requests[0]["input"][-1]
    assert last["type"] == "function_call_output" and REJECTION in last["output"]
    again = export_script(run.journal, ledger=run.ledger, parent=run.path, trial="branch")
    rejected = again["steps"][1]["runs"]
    assert rejected[0]["output"] == REJECTION and rejected[0]["returncode"] == REJECT_EXIT
    # The patch read the first submission did is left out: the rejection kept the agent from it.
    assert [item["command"] for item in rejected] == [
        item["command"] for item in base.script["steps"][1]["runs"]][:len(rejected)]
    assert len(rejected) < len(base.script["steps"][1]["runs"])
    assert [step["replayed"] for step in again["steps"]] == [True, True, False] and again["submission_step"] == 3
    assert again["source"]["parent"]["override"] == "reject"
    replayed_setup = [step for step in run.trace["steps"] if step["message"] == "command run before any model reply"]
    assert len(replayed_setup) == len(base.script["setup"])
