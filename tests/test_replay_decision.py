"""Replaying a recorded decision rebuilds exactly what was decided on, from a copy of the memory."""

import asyncio
import hashlib
import importlib.util
import threading
from pathlib import Path

import pytest

from taste.brains.central_planner import CentralPlanner, Goal, InvalidPlannerOutput
from taste.brains.monitor import Judgement, MonitorBrain, Severity
from taste.brains.planner_transport import PLANNER_RECEIPT_BRANCH
from taste.brains.subbrain import SubBrain
from taste.memstore import Store
from tests.test_brains_monitor_judge import scaffold
from tests.test_brains_planner_transport import FakeTurn, ReadyFakeLLM, make_transport

_SPEC = importlib.util.spec_from_file_location(
    "replay_decision", Path(__file__).resolve().parents[1] / "scripts" / "replay_decision.py")
replay = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(replay)


def test_a_coordinator_prompt_is_rebuilt_byte_for_byte_from_a_copy(tmp_path):
    store = Store.open(tmp_path / "repo", "replay")
    store.branch("integration", producer="replay-test").close()
    control = store.branch("central-control", producer="replay-test")
    lock = threading.RLock()
    transport = make_transport(ReadyFakeLLM([FakeTurn(text="not a plan", input_tokens=10, output_tokens=5)]),
                               store, control=control, mutation_lock=lock,
                               journal=store.branch(PLANNER_RECEIPT_BRANCH, producer="replay-test"),
                               max_tokens=2048)
    central = CentralPlanner(store, transport=transport, control=control, mutation_lock=lock)
    goal = Goal(goal_id="replay-goal", task="produce report.md", success_criteria=("report.md exists",))
    with pytest.raises(InvalidPlannerOutput):
        central.plan(goal)
    store.close()
    before = sorted(path.name for path in (tmp_path / "repo").iterdir())

    copy = replay.copy_workspace(tmp_path / "repo", into=tmp_path / "replay")
    opened = Store.open(copy, "replay")
    try:
        request, prompt, recorded, policy = replay.coordinator_prompt(opened)
    finally:
        opened.close()
    assert request.generation == 1 and policy is None
    # The prompt the run sent, rebuilt: its digest is the one recorded before sending.
    assert recorded == [hashlib.sha256(prompt.encode()).hexdigest()]
    # The original was only read.
    assert sorted(path.name for path in (tmp_path / "repo").iterdir()) == before


def test_a_monitor_stop_is_rebuilt_from_the_runs_monitor_state(tmp_path):
    store = Store.open(tmp_path / "repo", "replay")
    brain, _ = scaffold(store)
    brain.wal.intent("Bash", "c1", {"command": "make test"})
    brain.wal.result("Bash", "c1", ok=False, summary="1 failed")
    verdicts = iter([Judgement(Severity.FINE, "nothing contradicts the contract"),
                     Judgement(Severity.WRONG, "off the contract")])
    monitor = MonitorBrain(store, brain.contract, lambda *_: next(verdicts), batch_size=1)
    monitor.tick()
    monitor.tick()
    identity = brain.contract.identity
    store.close()

    copy = replay.copy_workspace(tmp_path / "repo", into=tmp_path / "replay")
    opened = Store.open(copy, "replay")
    try:
        name, _assignment, contract, batch, view, action = replay.monitor_judgement(opened, copy, "replay", identity)
    finally:
        opened.close()
    assert name == identity and contract == brain.contract
    assert action["judgement"]["severity"] == "wrong"
    # The batch that stopped the run, the event before it, and a worker still running.
    assert batch == [view.observed_events[1]] and list(view.earlier_events) == [view.observed_events[0]]
    assert view.worker_running is True and view.head.id == action["observed_head"]


def test_a_certification_is_rebuilt_from_the_runs_monitor_state(tmp_path):
    from tests.test_brains_monitor import FakeClient, ScriptedTerminalJudge, a_contract

    store = Store.open(tmp_path / "repo", "replay")
    brain = SubBrain(store, a_contract())
    brain.install_contract()
    brain.checkpoint("durable contract")
    judge = ScriptedTerminalJudge(Severity.WRONG)
    monitor = MonitorBrain(store, brain.contract, judge, batch_size=1)
    brain.wal.intent("Bash", "bad-start", {"command": "pytest"})
    asyncio.run(monitor.respond(monitor.tick(), FakeClient()))
    brain.branch.write("parser.py", "def parse(text):\n    return text\n")
    work_state = brain.checkpoint("corrected terminal work")
    asyncio.run(monitor.certify_terminal(work_state, context={"tests": {"passed": True}}))
    [(state, context, findings)] = judge.terminal_calls
    identity = brain.contract.identity
    store.close()

    copy = replay.copy_workspace(tmp_path / "repo", into=tmp_path / "replay")
    opened = Store.open(copy, "replay")
    try:
        name, _assignment, contract, rebuilt, rebuilt_context, rebuilt_findings, assessment = (
            replay.certifier_judgement(opened, copy, identity))
    finally:
        opened.close()
    # Exactly what the certifier was given, rebuilt from the copy.
    assert name == identity and contract == brain.contract
    assert rebuilt.id == state.id == work_state.id
    assert rebuilt_context == context and rebuilt_findings == findings and len(findings) == 1
    assert assessment.acceptable
