"""Large pinned evidence must not expand into an unbounded model request."""
import json

import pytest

from taste.brains.monitor_judge import (
    MAX_MONITOR_PROMPT_BYTES,
    LLMMonitorJudge,
    MonitorObservationError,
    build_monitor_observation,
    build_terminal_observation,
)
from taste.memstore import State, Store
from tests.test_brains_monitor_judge import scaffold


@pytest.fixture
def store(tmp_path):
    opened = Store.open(tmp_path / "repo", "monitor-bounds")
    yield opened
    opened.close()


def test_large_binary_is_identified_without_reading_or_base64_expansion(store, monkeypatch):
    brain, _ = scaffold(store)
    payload = bytes(range(256)) * 8192
    brain.branch.write("parser.py", payload)
    state = brain.checkpoint("large generated artifact")
    original = State.read_bytes

    def forbid_large_read(self, path):
        if path == "parser.py":
            pytest.fail("large artifact was read into the prompt builder")
        return original(self, path)

    monkeypatch.setattr(State, "read_bytes", forbid_large_read)
    try:
        observation = build_terminal_observation(brain.contract, state, {}, [])
        artifacts = json.loads(observation.payload)["state"]["output_artifacts"]
        assert artifacts[0]["blob_id"] == state.blob("parser.py")
        assert artifacts[0]["content"]["byte_size"] == len(payload)
        assert artifacts[0]["content"]["omitted"] is True
        assert len(observation.prompt().encode()) < MAX_MONITOR_PROMPT_BYTES
    finally:
        brain.close()


def test_large_context_and_event_batch_are_explicitly_partial(store):
    brain, _ = scaffold(store)
    try:
        state = brain.checkpoint("large context")
        context = {"tool_output": "x" * 2_000_000}
        terminal = build_terminal_observation(brain.contract, state, context, [])
        evidence = json.loads(terminal.payload)["terminal_context"]
        assert evidence["truncated"] and evidence["serialized_bytes"] > 2_000_000
        assert len(evidence["sha256"]) == 64
        assert len(terminal.prompt().encode()) < MAX_MONITOR_PROMPT_BYTES
        ordinary = build_monitor_observation(brain.contract, [context], brain.branch.view)
        assert json.loads(ordinary.payload)["events"]["truncated"]
        assert len(ordinary.prompt().encode()) < MAX_MONITOR_PROMPT_BYTES
    finally:
        brain.close()


def test_too_many_exact_finding_ids_fail_before_a_model_call(store):
    class NeverCall:
        def call(self, **kwargs):
            pytest.fail("oversized exact finding set reached the provider")

    brain, _ = scaffold(store)
    try:
        state = brain.checkpoint("exact finding identity set")
        findings = [{"id": f"finding-{i}-" + "a" * 64} for i in range(4000)]
        with pytest.raises(MonitorObservationError, match="input budget"):
            LLMMonitorJudge(NeverCall()).judge_terminal(brain.contract, state, {}, findings)
    finally:
        brain.close()
