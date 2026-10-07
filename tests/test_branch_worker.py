"""Branches of a recorded run, through the real worker: entrypoint, Azure wire, terminal RPC, mini-swe-agent unchanged.

A base run is recorded as a trial records it; its replay script is exported
from the worker's journal and the terminal ledger; then a fresh worker, in a
store of its own, branches it at step k. The model is the SDK wire's fake.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from taste.agents.trajectory_reader import steps_from_trajectory
from taste.benchmarks.goal_trajectory import _hosted_calls
from taste.benchmarks.replay_export import encode_script, export_script
from taste.benchmarks.worker_trajectory import hosted_trajectory
from taste.brains.azure_worker_entrypoint import execute_worker, run_directory
from taste.brains.azure_worker_policy import AzureWorkerPolicy
from taste.brains.branch_replay import REJECT_EXIT, BranchPolicy
from taste.brains.contract import CONTRACT_PATH, Contract
from taste.brains.records import ArtifactRef, ArtifactSpec, Assignment, contract_digest
from taste.brains.responses_session import ResponsesSession
from taste.brains.terminal_broker import TerminalResult
from taste.brains.worker_admission import EntrypointConfig, WorkerExitCode
from taste.brains.worker_protocol import ASSIGNMENT_PATH, assignment_run_id
from taste.memstore import Store
from taste.providers.azure_openai import AZURE_MONITOR_MODEL, AZURE_WORKER_MODEL
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_azure_terminal_worker import terminal
from tests.test_azure_worker_policy import assignment as policy_assignment
from tests.test_azure_worker_policy import environment
from tests.test_azure_worker_runtime import install, report
from tests.test_azure_worker_runtime import worker as _worker
from tests.test_hosted_worker import SENTINEL, bash, hosted, scripted
from tests.test_openai_responses import message
from tests.test_worker_admission import install_assignment
from tests.test_worker_admission import launch as _launch

sdk_transport = _sdk_transport
launch = _launch
worker = _worker

REPLIES = {1: [message("Let me run the tests."), bash("make test", "c1")],
           2: [bash("sed -i 's/x/y/' parser.py && make test", "c2")],
           3: [bash(f"echo {SENTINEL}", "c3")]}
OUTPUTS = {"sed -i": TerminalResult(0, b"4 passed\n", b""),
           "make test": TerminalResult(1, b"FAILED test_parse\n", b""),
           SENTINEL: TerminalResult(0, f"{SENTINEL}\n".encode(), b"")}
REJECTION = "Submission rejected by a reviewer: tabs are still dropped (tests/test_tabs.py fails)."


def open_worker(directory):
    """Another worker in a store of its own, as the launch and worker fixtures make one."""
    store = Store.open(directory / "repo", "admission")
    producer = store.branch("producer")
    producer.write("input.txt", "exact input")
    source = producer.checkpoint("input source")
    producer.close()
    branch = store.branch("worker")
    brief = Contract("worker", "produce output", inputs=("input.txt",), outputs=("output.txt",),
                     success_criteria=("the output is correct",), budget_usd=30)
    assignment = Assignment(
        "task", 1, 0, brief, contract_digest(brief), branch.head.id,
        inputs=(ArtifactRef("input", "producer", source.id, "input.txt", source.blob("input.txt")),),
        outputs=(ArtifactSpec("output", "output.txt"),), model=AZURE_WORKER_MODEL,
        resources={"monitor_budget_usd": 30})
    branch.write("input.txt", source.read_bytes("input.txt"))
    branch.write(CONTRACT_PATH, brief.to_json())
    branch.write(ASSIGNMENT_PATH, assignment.to_json())
    prepared = branch.checkpoint("prepared assignment")
    branch.close()
    config = EntrypointConfig(store.root, store.session, "worker", AZURE_WORKER_MODEL, prepared.id,
                              monitor_model=AZURE_MONITOR_MODEL, monitor_budget_usd=30)
    run_id = assignment_run_id(assignment)
    key = hashlib.sha256(run_id.encode()).hexdigest()
    value = SimpleNamespace(store=store, assignment=assignment, config=config, environ={
        "TASTE_WORKER_RUN_ID": run_id, "TASTE_WORKER_LAUNCH_TOKEN": "a" * 32,
        "TASTE_WORKER_READY_PATH": str(store.backend.common_dir / f"taste-supervisor.admission.{key}.ready.json")})
    value.assignment = replace(value.assignment, resources=policy_assignment(monitor_max_calls=20).resources)
    install_assignment(value, value.assignment)
    value.config = replace(value.config, monitor_max_tokens=256, monitor_budget_usd=20)
    value.environ.update(environment())
    return value


@pytest.fixture
def other(tmp_path):
    value = open_worker(tmp_path / "branch")
    yield value
    value.store.close()


def run_agent(worker, sdk_transport, directory, replies, outputs, task=None, **policy):
    sent, calls, _ = install(sdk_transport, worker_reply=lambda number, _: replies[number])
    # One test's wire keeps every request sent in it: this run's come after the ones before.
    start = len(sent)
    seen = {}

    async def scenario():
        assignment = hosted(worker, services="none", **policy)
        if task is not None:
            contract = replace(assignment.contract, task=task)
            worker.assignment = replace(assignment, contract=contract, contract_digest=contract_digest(contract))
            install_assignment(worker, worker.assignment)
        async with terminal(worker, directory) as t:
            scripted(t.env, outputs)
            seen["code"] = await execute_worker(worker.config, store=worker.store, environ=worker.environ)
            seen["commands"] = [call.command for call in t.env.calls]
    asyncio.run(scenario())
    result = report(worker)
    run = run_directory(worker.store, result.run_id) / "worker"
    return SimpleNamespace(code=seen["code"], commands=seen["commands"], calls=calls["worker"], result=result,
                           requests=[json.loads(wire.content) for wire in sent[start:]], journal=run / "calls.sqlite3",
                           ledger=directory / "terminal-ledger" / "terminal.sqlite3",
                           trace=json.loads((run / "trajectory.worker.json").read_text()))


@pytest.fixture
def base(worker, sdk_transport, tmp_path):
    """The run every branch here continues: a failing test, a fix, a submission."""
    run = run_agent(worker, sdk_transport, tmp_path, REPLIES, OUTPUTS)
    assert run.code == WorkerExitCode.COMPLETED and run.calls == 3
    run.script = export_script(run.journal, ledger=run.ledger, trial="base")
    return run


def branch(other, sdk_transport, tmp_path, script, step, *, replies=None, outputs=OUTPUTS, where="branch",
           task=None, **options):
    path = tmp_path / f"script-{where}-{step}.json"
    raw = encode_script(script)
    path.write_bytes(raw)
    fork = BranchPolicy(str(path), hashlib.sha256(raw).hexdigest(), step, **options)
    run = run_agent(other, sdk_transport, tmp_path / where, replies or {}, outputs, task=task,
                    branch=fork.to_dict())
    run.path = path
    return run


def test_the_export_holds_each_step_as_the_agent_had_it(base):
    script = base.script
    assert script["task"].startswith("Make the parser tests pass.") and script["submission_step"] == 3
    assert script["exit"]["exit_status"] == "Submitted" and script["source"]["dropped_steps"] == 0
    first = script["steps"][0]
    assert first["text"] == ["Let me run the tests."] and first["reply"]["cost_usd"] > 0
    assert [item["type"] for item in first["reply"]["output"]] == ["message", "function_call"]
    [run] = first["runs"]
    assert run["command"] == "make test" and run["executed"].endswith("{\nmake test\n} 2>&1")
    assert (run["output"], run["returncode"], run["output_exact"], run["timed_out"]) == (
        "FAILED test_parse\n", 1, True, False)
    # Each request is the conversation before it plus what came since.
    assert [step["messages"]["base"] for step in script["steps"]] == [0, 2, 5]


def test_the_command_line_finds_a_trials_records_from_harbors_trial(base, tmp_path, capsys):
    """A trial's directory as its owner leaves it, reached from Harbor's trial directory."""
    spec = importlib.util.spec_from_file_location(
        "replay_script", Path(__file__).resolve().parents[1] / "scripts" / "replay_script.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    trial = tmp_path / "trials" / "f00d"
    worker = trial / "agent-state/workspace/.git" / ("taste.azure.terminal-trial." + "a" * 64) / "worker"
    worker.mkdir(parents=True)
    shutil.copy(base.journal, worker / "calls.sqlite3")
    (trial / "controller/terminal").mkdir(parents=True)
    shutil.copy(base.ledger, trial / "controller/terminal/terminal.sqlite3")
    harbor = tmp_path / "jobs/job/task__1"
    harbor.mkdir(parents=True)
    (harbor / "result.json").write_text(json.dumps({"agent_result": {"metadata": {"taste": {"trial": "f00d"}}}}))
    out = tmp_path / "script.json"
    assert cli.main([str(harbor), "-o", str(out), "--trials", str(tmp_path / "trials")]) == 0
    written = json.loads(out.read_text())
    assert written["source"]["trial"] == "f00d"
    assert {key: value for key, value in written.items() if key != "source"} == {
        key: value for key, value in base.script.items() if key != "source"}
    summary = json.loads(capsys.readouterr().out)
    assert summary["steps"] == 3 and summary["rebuildable"] and summary["outputs_exact"] == 3
    with pytest.raises(ValueError, match="worker runs"):
        cli.main([str(trial), "-o", str(out), "--run", "another-run"])


def test_a_branch_rebuilds_k_steps_replays_them_and_its_first_live_request_is_the_recorded_one(
        base, other, sdk_transport, tmp_path):
    run = branch(other, sdk_transport, tmp_path, base.script, 2, replies={1: REPLIES[3]})
    assert run.code == WorkerExitCode.COMPLETED and run.calls == 1
    # The recorded commands 1..2, exactly as the container ran them, then the live one.
    executed = [item["runs"][0]["executed"] for item in base.script["steps"]]
    assert run.commands == executed
    # The agent's context at step 3 is the recorded run's, to the byte.
    assert run.requests[0]["input"] == base.requests[2]["input"]
    account = run.trace["extra"]["branch"]
    assert account["faithful"] and account["went_live"] and account["context"] == "matched"
    assert account["replayed_steps"] == 2 and account["rebuild"]["commands"] == 2
    assert account["rebuild"]["divergent"] == 0
    replayed = [step for step in run.trace["steps"] if step["extra"].get("replayed")]
    assert len(replayed) == 2 and all("metrics" not in step and step["llm_call_count"] == 0 for step in replayed)
    assert replayed[0]["observation"]["results"][0]["content"] == "FAILED test_parse\n"
    assert run.trace["extra"]["exit"] == {"exit_status": "Submitted", "stopped_by": ""}
    # Replayed steps were not bought: the run's accounting has the one live call.
    assert run.result.metadata["worker_accounting"]["completed_calls"] == 1
    assert run.result.metadata["branch"]["replayed_steps"] == 2


def test_a_settled_branch_record_matches_its_receipts(base, other, sdk_transport, tmp_path):
    run = branch(other, sdk_transport, tmp_path, base.script, 2, replies={1: REPLIES[3]})
    policy = AzureWorkerPolicy.from_assignment(other.assignment)
    session = ResponsesSession.open(run.journal.parent, policy.worker, policy.azure_config(other.environ))
    try:
        rows, evidence = session.conversation_audit(), session.call_evidence()
    finally:
        session.close()
    projected, gaps = hosted_trajectory(rows, run_id=run.result.run_id), []
    _hosted_calls(projected, rows, evidence, gaps, run.result.run_id)
    assert gaps == [] and not projected["extra"]["incomplete_worker_trace"]


def test_reject_at_the_submission_keeps_the_agent_working_in_place(base, other, sdk_transport, tmp_path):
    run = branch(other, sdk_transport, tmp_path, base.script, 3, override="reject", note=REJECTION,
                 replies={1: [bash("make test", "c4")], 2: [bash(f"echo {SENTINEL}", "c5")]})
    assert run.code == WorkerExitCode.COMPLETED and run.calls == 2
    # Its first live request ends with the rejection, as the submission's result, exit 1.
    last = run.requests[0]["input"][-1]
    assert last["type"] == "function_call_output" and REJECTION in last["output"]
    assert f'"returncode": {REJECT_EXIT}' in last["output"]
    # The three recorded commands rebuilt, then the two live ones.
    assert len(run.commands) == 5 and run.trace["extra"]["branch"]["override"] == "reject"
    # Its own record, exported, is the run the agent had: the rejection at step 3, then two live steps.
    again = export_script(run.journal, ledger=run.ledger, parent=run.path, trial="branch")
    assert len(again["steps"]) == 5 and again["submission_step"] == 5
    assert [step["replayed"] for step in again["steps"]] == [True, True, True, False, False]
    assert again["steps"][2]["runs"][0]["output"] == REJECTION
    assert again["steps"][2]["runs"][0]["returncode"] == REJECT_EXIT
    assert again["source"]["parent"]["step"] == 3 and again["source"]["parent"]["override"] == "reject"


def test_a_second_round_branches_the_first_rounds_record(base, other, sdk_transport, tmp_path):
    """Continue with feedback, twice: round 2 rebuilds round 1's run, rejected submission included."""
    first = branch(other, sdk_transport, tmp_path, base.script, 3, override="reject", note=REJECTION,
                   replies={1: [bash("make test", "c4")], 2: [bash(f"echo {SENTINEL}", "c5")]})
    record = export_script(first.journal, ledger=first.ledger, parent=first.path, trial="round-1")
    third = open_worker(tmp_path / "round2")
    try:
        again = "Submission rejected by a reviewer: tests/test_tabs.py still fails."
        second = branch(third, sdk_transport, tmp_path, record, 5, override="reject", note=again, where="round2",
                        replies={1: [bash(f"echo {SENTINEL}", "c6")]})
    finally:
        third.store.close()
    assert second.code == WorkerExitCode.COMPLETED and second.calls == 1
    account = second.trace["extra"]["branch"]
    # The first submission, rejected in round 1, printed the sentinel when it ran again: no divergence.
    assert account["faithful"] and account["rebuild"]["commands"] == 5 and account["rebuild"]["divergent"] == 0
    assert record["steps"][2]["runs"][0]["printed"]["returncode"] == 0
    context = json.dumps(second.requests[0]["input"])
    assert "tabs are still dropped" in context and "test_tabs.py still fails" in context


def test_steps_are_the_trajectory_readers_steps(base, other, sdk_transport, tmp_path):
    """Step k is the same step to the map, the reader and a branch: the readers' steps are the script's."""
    run = branch(other, sdk_transport, tmp_path, base.script, 3, override="reject", note=REJECTION,
                 replies={1: [bash("make test", "c4")], 2: [bash(f"echo {SENTINEL}", "c5")]})
    again = export_script(run.journal, ledger=run.ledger, parent=run.path, trial="branch")
    for trace, script in ((base.trace, base.script), (run.trace, again)):
        read = steps_from_trajectory(trace)
        assert [step.number for step in read] == [step["step"] for step in script["steps"]]
        assert [step.command for step in read] == ["\n".join(item["command"] for item in step["runs"])
                                                   for step in script["steps"]]
        assert [step.returncode for step in read] == [step["runs"][-1]["returncode"] for step in script["steps"]]


def test_with_the_replay_off_another_agent_starts_fresh_on_the_last_steps_files(
        base, other, sdk_transport, tmp_path):
    """A checker's trial: the run's final files rebuilt, then its own agent with its own task."""
    task = "Review whether the parser tests pass now. Do not change any file."
    run = branch(other, sdk_transport, tmp_path, base.script, 3, replay=False, task=task,
                 replies={1: [bash("make test", "c1")], 2: [bash(f"echo {SENTINEL}", "c2")]})
    assert run.code == WorkerExitCode.COMPLETED and run.calls == 2
    # The three recorded commands rebuilt the files; nothing of the run's context was replayed.
    executed = [item["runs"][0]["executed"] for item in base.script["steps"]]
    assert run.commands[:3] == executed and len(run.commands) == 5
    first = run.requests[0]["input"]
    assert len(first) == 2 and first[1]["content"].startswith("Please solve this issue: " + task)
    account = run.trace["extra"]["branch"]
    assert account["replay"] is False and account["replayed_steps"] == 0 and account["faithful"]
    assert account["rebuild"]["commands"] == 3 and account["context"] == "unchecked"


def test_append_shows_the_note_after_step_ks_output(base, other, sdk_transport, tmp_path):
    note = "Note: an attempt that continued from here was rejected by a reviewer: tabs are dropped."
    run = branch(other, sdk_transport, tmp_path, base.script, 1, override="append", note=note,
                 replies={1: REPLIES[2], 2: REPLIES[3]})
    assert run.code == WorkerExitCode.COMPLETED and run.calls == 2
    last = run.requests[0]["input"][-1]
    assert "FAILED test_parse" in last["output"] and "tabs are dropped" in last["output"]
    assert run.trace["extra"]["branch"]["context"] == "unchecked"


def test_live_off_grades_step_k_with_no_model_call(base, other, sdk_transport, tmp_path):
    run = branch(other, sdk_transport, tmp_path, base.script, 2, live=False)
    assert run.calls == 0 and run.code == WorkerExitCode.INCOMPLETE
    assert len(run.commands) == 2
    assert run.trace["extra"]["exit"]["stopped_by"] == "branch_live_off"
    account = run.trace["extra"]["branch"]
    assert account["faithful"] and not account["went_live"] and account["prefix_complete"]


def test_a_restored_branch_runs_nothing_before_going_live(base, other, sdk_transport, tmp_path):
    run = branch(other, sdk_transport, tmp_path, base.script, 2, mode="restore", replies={1: REPLIES[3]})
    assert run.code == WorkerExitCode.COMPLETED and run.calls == 1
    assert run.commands == [base.script["steps"][2]["runs"][0]["executed"]]
    assert run.trace["extra"]["branch"]["rebuild"] is None and run.trace["extra"]["branch"]["faithful"]


def test_a_rebuild_that_diverges_is_unfaithful_and_never_goes_live(base, other, sdk_transport, tmp_path):
    changed = {**OUTPUTS, "make test": TerminalResult(0, b"collected 0 items\n", b"")}
    run = branch(other, sdk_transport, tmp_path, base.script, 2, outputs=changed, replies={1: REPLIES[3]})
    assert run.calls == 0 and run.code == WorkerExitCode.INCOMPLETE
    # It stopped rebuilding at the first divergent command.
    assert len(run.commands) == 1
    account = run.trace["extra"]["branch"]
    assert not account["faithful"] and account["unfaithful"]["reason"] == "rebuild"
    [divergence] = account["rebuild"]["divergences"]
    assert (divergence["returncode"], divergence["recorded_returncode"]) == (0, 1)
    assert run.trace["extra"]["exit"]["stopped_by"] == "branch_unfaithful"


def test_another_task_is_unfaithful_before_anything_runs(base, other, sdk_transport, tmp_path):
    script = {**base.script, "task": "Make the lexer tests pass."}
    run = branch(other, sdk_transport, tmp_path, script, 2, replies={1: REPLIES[3]})
    assert run.calls == 0 and run.commands == []
    account = run.trace["extra"]["branch"]
    assert not account["faithful"] and account["unfaithful"]["reason"] == "task"
    assert account["task_matches"] is False


def test_a_script_that_is_not_the_admitted_one_is_refused(base, other, sdk_transport, tmp_path):
    path = tmp_path / "script.json"
    path.write_bytes(encode_script(base.script))
    fork = BranchPolicy(str(path), "0" * 64, 2)
    install(sdk_transport, worker_reply=lambda number, _: REPLIES[number])

    async def scenario():
        hosted(other, services="none", branch=fork.to_dict())
        async with terminal(other, tmp_path / "branch") as t:
            code = await execute_worker(other.config, store=other.store, environ=other.environ)
            return code, len(t.env.calls)
    code, commands = asyncio.run(scenario())
    assert code != WorkerExitCode.COMPLETED and commands == 0
