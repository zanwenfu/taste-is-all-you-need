"""The production monitor judge is strict, pinned and fully auditable."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from taste.brains.contract import CONTRACT_PATH, Contract
from taste.brains.monitor import MonitorBrain, Severity
from taste.brains.monitor_judge import (
    ASSIGNMENT_PATH,
    JUDGEMENT_SCHEMA,
    TERMINAL_JUDGEMENT_SCHEMA,
    LLMMonitorJudge,
    MonitorObservationError,
    MonitorResponseError,
    build_monitor_observation,
    build_terminal_observation,
    parse_monitor_response,
    parse_terminal_response,
)
from taste.brains.records import ArtifactSpec, Assignment, contract_digest
from taste.brains.subbrain import SubBrain
from taste.llm import MODEL_MONITOR
from taste.memstore import Store

from .fakes import FakeLLM, FakeTurn


def contract(**changes: Any) -> Contract:
    values = {
        "identity": "worker-1",
        "task": "build the parser",
        "outputs": ("parser.py",),
        "success_criteria": ("the parser tests pass", "no TODO remains"),
    }
    return Contract(**{**values, **changes})


def assignment(brief: Contract, *, base_state_id: str) -> Assignment:
    return Assignment(
        assignment_id="build-parser",
        generation=3,
        attempt=1,
        contract=brief,
        contract_digest=contract_digest(brief),
        base_state_id=base_state_id,
        outputs=(
            ArtifactSpec(
                artifact_id="parser",
                path="parser.py",
                description="tested parser implementation",
            ),
        ),
        model="claude-sonnet-4-6",
    )


def response(**changes: Any) -> str:
    payload = {
        "schema": JUDGEMENT_SCHEMA,
        "severity": "drifting",
        "reason": "The parser was edited before its failing test was isolated.",
        "evidence": ["The pytest tool result reports one failure."],
        "suggestion": "Inspect that failure before making another edit.",
    }
    payload.update(changes)
    return json.dumps(payload, sort_keys=True)


def terminal_response(*, finding_ids: tuple[str, ...] = (), **changes: Any) -> str:
    payload = {
        "schema": TERMINAL_JUDGEMENT_SCHEMA,
        "severity": "fine",
        "reason": "The exact work state satisfies the assignment.",
        "evidence": ["The terminal context records a passing test run."],
        "suggestion": "",
        "resolved_finding_ids": list(finding_ids),
        "unresolved_finding_ids": [],
    }
    payload.update(changes)
    return json.dumps(payload, sort_keys=True)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    opened = Store.open(tmp_path / "repo", "session-1")
    yield opened
    opened.close()


def scaffold(store: Store, *, typed: bool = True) -> tuple[SubBrain, Assignment | None]:
    brief = contract()
    brain = SubBrain(store, brief)
    brain.install_contract()
    item = assignment(brief, base_state_id=brain.branch.head.id) if typed else None
    if item is not None:
        brain.branch.write(ASSIGNMENT_PATH, item.to_json())
    brain.checkpoint("central accepted the durable worker brief")
    return brain, item


def test_llm_judge_receives_exact_pinned_contract_assignment_and_events(store: Store) -> None:
    brain, item = scaffold(store)
    assert item is not None
    brain.wal.intent("Bash", "test-1", {"command": "pytest tests/test_parser.py"})
    brain.wal.result("Bash", "test-1", ok=False, summary="1 failed")
    fake = FakeLLM([FakeTurn(text=response())], model=MODEL_MONITOR)
    judge = LLMMonitorJudge(fake)
    monitor = MonitorBrain(store, brain.contract, judge, batch_size=2)

    judgement = monitor.tick()

    assert judgement is not None
    assert judgement.severity is Severity.DRIFTING
    assert judgement.model == MODEL_MONITOR
    assert judgement.raw_response == response()
    assert judgement.cost_usd is not None and judgement.cost_usd > 0
    assert fake.call_count == 1
    call = fake.calls[0]
    assert call["model"] == MODEL_MONITOR
    assert call["role"] == "monitor"
    assert call["temperature"] == 0.0
    assert call["tools"] is None
    assert JUDGEMENT_SCHEMA in call["system"]
    prompt = call["messages"][0]["content"]
    assert brain.contract.to_json() in prompt
    assert item.to_json() in prompt
    assert "tested parser implementation" in prompt
    assert '"tool_use_id": "test-1"' in prompt
    assert f"observed_head: {monitor.pending_actions[0].observed_head}" in prompt
    brain.close()


def test_monitor_rejects_provider_model_substitution_before_pricing() -> None:
    backing = FakeLLM([FakeTurn(text=response())], model=MODEL_MONITOR)

    class SubstitutingLLM:
        def call(self, **kwargs: Any):
            return replace(backing.call(**kwargs), model="claude-opus-5")

    judge = LLMMonitorJudge(SubstitutingLLM(), model=MODEL_MONITOR)

    with pytest.raises(MonitorResponseError, match=r"returned model.*expected"):
        judge._completion(system="monitor system", prompt="exact observation")


def test_contract_only_worker_is_explicitly_presented_without_an_assignment(store: Store) -> None:
    brain, _ = scaffold(store, typed=False)
    observation = build_monitor_observation(
        brain.contract,
        [{"kind": "assistant", "text": "still investigating"}],
        store.view(brain.contract.identity),
    )

    prompt = observation.prompt()
    assert brain.contract.to_json() in prompt
    assert "<exact-assignment-json>" not in prompt
    assert '"assignment_id": null' in prompt
    brain.close()


def test_judge_reads_state_fields_from_the_pinned_head_while_worker_moves(store: Store) -> None:
    brain, _ = scaffold(store)
    brain.wal.intent("Read", "read-1", {"path": "parser.py"})
    observed_head = brain.branch.head.id

    class MovingLLM(FakeLLM):
        def call(self, **kwargs: Any):
            brain.branch.write("created-after-observation.txt", "later")
            brain.checkpoint("worker moved while the monitor model was thinking")
            return super().call(**kwargs)

    fake = MovingLLM([FakeTurn(text=response(severity="fine"))], model=MODEL_MONITOR)
    monitor = MonitorBrain(
        store,
        brain.contract,
        LLMMonitorJudge(fake),
        batch_size=1,
    )

    assert monitor.tick() is not None
    prompt = fake.calls[0]["messages"][0]["content"]
    assert f"observed_head: {observed_head}" in prompt
    assert "created-after-observation.txt" not in prompt
    assert brain.branch.head.id != observed_head
    assert monitor.pending_actions[0].observed_head == observed_head
    brain.close()


def test_invalid_or_split_durable_brief_is_rejected_before_calling_model(store: Store) -> None:
    brief = contract()
    brain = SubBrain(store, brief)
    brain.install_contract()
    malformed = json.loads(assignment(brief, base_state_id=brain.branch.head.id).to_json())
    malformed["unexpected"] = True
    brain.branch.write(ASSIGNMENT_PATH, json.dumps(malformed))
    brain.checkpoint("malformed assignment")
    fake = FakeLLM([FakeTurn(text=response())], model=MODEL_MONITOR)

    with pytest.raises(MonitorObservationError, match="Assignment"):
        LLMMonitorJudge(fake)(brief, [{"kind": "event"}], store.view(brief.identity))
    assert fake.call_count == 0

    revised = contract(task="silently revised task")
    brain.branch.write(CONTRACT_PATH, revised.to_json())
    brain.checkpoint("split contract")
    with pytest.raises(MonitorObservationError, match="differs"):
        LLMMonitorJudge(fake)(brief, [{"kind": "event"}], store.view(brief.identity))
    assert fake.call_count == 0
    brain.close()


def test_assignment_must_be_bound_to_the_exact_monitor_contract(store: Store) -> None:
    brief = contract()
    brain = SubBrain(store, brief)
    brain.install_contract()
    different = contract(task="build a lexer instead")
    brain.branch.write(
        ASSIGNMENT_PATH,
        assignment(different, base_state_id=brain.branch.head.id).to_json(),
    )
    brain.checkpoint("assignment and contract disagree")
    fake = FakeLLM([FakeTurn(text=response())], model=MODEL_MONITOR)

    with pytest.raises(MonitorObservationError, match="Assignment and Contract disagree"):
        LLMMonitorJudge(fake)(brief, [{"kind": "event"}], store.view(brief.identity))
    assert fake.call_count == 0
    brain.close()


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("```json\n{}\n```", "not one valid JSON object"),
        (
            '{"schema":"taste.brains/MonitorJudgement/1","severity":"fine",'
            '"severity":"lost","reason":"x","evidence":[],"suggestion":""}',
            "duplicate key",
        ),
        (response(extra="not allowed"), "unknown fields"),
        (response(reason=""), "non-empty string"),
        (response(evidence=[""]), "non-empty strings"),
        (response(severity="uncertain"), "invalid severity"),
    ],
)
def test_response_parser_rejects_malformed_or_ambiguous_output(raw: str, match: str) -> None:
    with pytest.raises(MonitorResponseError, match=match):
        parse_monitor_response(raw)


@pytest.mark.parametrize(
    "turn",
    [
        FakeTurn(text=response(), stop_reason="max_tokens"),
        FakeTurn(text=response(), tool_calls=[("invented_tool", {})]),
    ],
    ids=["truncated", "tool-call"],
)
def test_incomplete_or_non_json_mode_completion_is_never_a_verdict(
    store: Store,
    turn: FakeTurn,
) -> None:
    brain, _ = scaffold(store)
    fake = FakeLLM([turn], model=MODEL_MONITOR)
    judge = LLMMonitorJudge(fake)

    with pytest.raises(MonitorResponseError):
        judge(brain.contract, [{"kind": "event"}], store.view(brain.contract.identity))
    brain.close()


def test_model_audit_metadata_survives_monitor_restart(store: Store) -> None:
    brain, _ = scaffold(store)
    brain.wal.intent("Bash", "test-1", {"command": "pytest"})
    monitor = MonitorBrain(
        store,
        brain.contract,
        LLMMonitorJudge(
            FakeLLM([FakeTurn(text=response(severity="fine"))], model=MODEL_MONITOR)
        ),
        batch_size=1,
    )
    original = monitor.tick()
    assert original is not None

    restarted = MonitorBrain(
        store,
        brain.contract,
        lambda *_args: pytest.fail("persisted work must not be judged twice"),
        batch_size=1,
    )
    restored = restarted.state.judgements[0]
    assert restored.model == MODEL_MONITOR
    assert restored.raw_response == original.raw_response
    assert restored.cost_usd == original.cost_usd
    assert restarted.pending_actions[0].judgement.raw_response == original.raw_response
    brain.close()


def test_terminal_judge_receives_the_exact_state_context_outputs_and_findings(
    store: Store,
) -> None:
    brain, item = scaffold(store)
    assert item is not None
    brain.branch.write("parser.py", "def parse(text):\n    return text.strip()\n")
    brain.wal.intent("Bash", "terminal-test", {"command": "pytest"})
    brain.wal.result("Bash", "terminal-test", ok=True, summary="12 passed")
    work_state = brain.checkpoint("immutable terminal work")
    findings = [
        {
            "id": "finding-1",
            "severity": "wrong",
            "reason": "the earlier parser did not strip input",
            "evidence": ["a test failed"],
            "suggestion": "strip input",
        }
    ]
    fake = FakeLLM(
        [FakeTurn(text=terminal_response(finding_ids=("finding-1",)))],
        model=MODEL_MONITOR,
    )

    decision = LLMMonitorJudge(fake).judge_terminal(
        brain.contract,
        work_state,
        {"terminal": {"tests": {"passed": 12}}},
        findings,
    )

    assert decision.judgement.severity is Severity.FINE
    assert decision.resolved_finding_ids == ("finding-1",)
    assert decision.judgement.model == MODEL_MONITOR
    prompt = fake.calls[0]["messages"][0]["content"]
    assert f"work_state: {work_state.id}" in prompt
    assert brain.contract.to_json() in prompt
    assert item.to_json() in prompt
    assert "def parse(text)" in prompt
    assert '"tool_use_id": "terminal-test"' in prompt
    assert '"passed": 12' in prompt
    assert '"id": "finding-1"' in prompt
    assert TERMINAL_JUDGEMENT_SCHEMA in fake.calls[0]["system"]
    brain.close()


def test_terminal_observation_stays_on_the_supplied_state_if_branch_moves(
    store: Store,
) -> None:
    brain, _ = scaffold(store)
    brain.branch.write("parser.py", "old exact contents\n")
    work_state = brain.checkpoint("state to certify")

    class MovingLLM(FakeLLM):
        def call(self, **kwargs: Any):
            brain.branch.write("parser.py", "newer unassessed contents\n")
            brain.checkpoint("branch moved after certification began")
            return super().call(**kwargs)

    fake = MovingLLM([FakeTurn(text=terminal_response())], model=MODEL_MONITOR)
    LLMMonitorJudge(fake).judge_terminal(brain.contract, work_state, {}, [])

    prompt = fake.calls[0]["messages"][0]["content"]
    assert f"work_state: {work_state.id}" in prompt
    assert "old exact contents" in prompt
    assert "newer unassessed contents" not in prompt
    assert brain.branch.head.id != work_state.id
    brain.close()


@pytest.mark.parametrize(
    ("raw", "finding_ids", "match"),
    [
        (terminal_response(), ("f1",), "partition exactly"),
        (
            terminal_response(finding_ids=("f1",), resolved_finding_ids=["f1", "extra"]),
            ("f1",),
            "partition exactly",
        ),
        (
            terminal_response(
                finding_ids=("f1",),
                unresolved_finding_ids=["f1"],
            ),
            ("f1",),
            "overlap",
        ),
        (
            terminal_response(
                finding_ids=("f1",),
                severity="fine",
                resolved_finding_ids=[],
                unresolved_finding_ids=["f1"],
            ),
            ("f1",),
            "fine terminal decision",
        ),
        (
            terminal_response(finding_ids=("f1",), resolved_finding_ids=["f1", "f1"]),
            ("f1",),
            "duplicate",
        ),
    ],
)
def test_terminal_parser_requires_an_exact_nonambiguous_finding_partition(
    raw: str,
    finding_ids: tuple[str, ...],
    match: str,
) -> None:
    with pytest.raises(MonitorResponseError, match=match):
        parse_terminal_response(raw, finding_ids=finding_ids)


def test_terminal_observation_rejects_a_split_durable_assignment(store: Store) -> None:
    brain, _ = scaffold(store)
    work_state = brain.checkpoint("valid state")
    different = contract(task="build a lexer instead")
    brain.branch.write(
        ASSIGNMENT_PATH,
        assignment(different, base_state_id=work_state.id).to_json(),
    )
    split_state = brain.checkpoint("split terminal control records")

    with pytest.raises(MonitorObservationError, match="Assignment and Contract disagree"):
        build_terminal_observation(brain.contract, split_state, {}, [])
    brain.close()
