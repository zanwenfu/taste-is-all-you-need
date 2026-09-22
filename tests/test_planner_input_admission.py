"""Planner failures before provider dispatch retain known-zero accounting."""

from types import SimpleNamespace

import pytest

from taste.brains import planner_transport as module
from taste.brains.central_planner import CentralPlanner, Goal, PlannerTransportError
from taste.brains.planner_transport import PlannerCompletionError, load_planner_transport_evidence
from taste.memstore import Store
from tests.fakes import FakeTurn
from tests.test_brains_planner_transport import ReadyFakeLLM, make_transport


@pytest.fixture
def store(tmp_path):
    opened = Store.open(tmp_path / "repo", "planner-admission")
    try:
        yield opened
    finally:
        opened.close()


def test_default_input_bound_prevents_an_oversized_provider_call(store):
    llm = ReadyFakeLLM([FakeTurn(text='{"ok":true}')])
    transport = make_transport(llm, store)
    with pytest.raises(PlannerCompletionError, match=r"input.*bytes") as error:
        transport.complete(request_id="too-large", system="system", prompt="x" * 1_048_576)
    assert error.value.telemetry.cost_known and error.value.telemetry.billed_usd == 0
    assert llm.call_count == 0 and llm.ready_models == []
    assert load_planner_transport_evidence(transport.journal.head, "too-large") is None


@pytest.mark.parametrize("overflow", [False, True])
def test_input_bound_counts_combined_utf8_bytes_and_accepts_exact_limit(store, overflow):
    llm = ReadyFakeLLM([FakeTurn(text='{"ok":true}')])
    transport = make_transport(llm, store, max_prompt_bytes=8)
    prompt = "界界" + ("x" if overflow else "")
    if overflow:
        with pytest.raises(PlannerCompletionError):
            transport.complete(request_id="utf8", system="é", prompt=prompt)
        assert llm.call_count == 0
    else:
        transport.complete(request_id="utf8", system="é", prompt=prompt)
        assert llm.call_count == 1


@pytest.mark.parametrize("bound", [0, -1, True, 1.5])
def test_input_bound_requires_a_positive_integer(store, bound):
    with pytest.raises(ValueError, match="max_prompt_bytes"):
        make_transport(ReadyFakeLLM(), store, max_prompt_bytes=bound)


@pytest.mark.parametrize("failure", ["readiness", "oversized_world"])
def test_preflight_failure_keeps_central_planner_spending_known_zero(store, failure):
    class NotReady(ReadyFakeLLM):
        def ensure_ready(self, *models):
            raise RuntimeError("credential preflight failed")

    llm = NotReady() if failure == "readiness" else ReadyFakeLLM([FakeTurn(text='{"ok":true}')])
    transport = make_transport(llm, store)
    integration = store.branch("integration", producer="test")
    if failure == "oversized_world":
        integration.write("large.txt", "x" * 1_048_576)
    central = CentralPlanner(store, transport=transport, control=transport.control,
                             mutation_lock=transport.mutation_lock)
    goal = Goal(goal_id="goal", task="synthetic work", success_criteria=("done",))
    with pytest.raises(PlannerTransportError, match=r"credential preflight failed|input"):
        central.plan(goal)
    assert central.planner_cost(goal.goal_id, currency="billed") == 0.0
    assert central.planner_cost(goal.goal_id, currency="work") == 0.0
    assert llm.call_count == 0


def test_deadline_expiring_during_admission_leaves_replayable_zero_receipt(store, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    llm = ReadyFakeLLM([FakeTurn(text='{"ok":true}')])
    transport = make_transport(llm, store)
    transport.bind_deadline(remaining_seconds=1.0)
    commit = transport._commit_authoritative_pair

    def expire(path, *args, **kwargs):
        result = commit(path, *args, **kwargs)
        if path.endswith("/intent.json"):
            now[0] = 102.0
        return result

    monkeypatch.setattr(transport, "_commit_authoritative_pair", expire)
    with pytest.raises(PlannerCompletionError, match="deadline") as error:
        transport.complete(request_id="expired", system="system", prompt="prompt")
    assert error.value.telemetry.cost_known and error.value.telemetry.billed_usd == 0
    assert llm.call_count == 0
    receipt = load_planner_transport_evidence(transport.journal.head, "expired")
    assert receipt.telemetry == error.value.telemetry
    with pytest.raises(PlannerCompletionError, match="deadline") as replay:
        transport.complete(request_id="expired", system="system", prompt="prompt")
    assert replay.value.telemetry == error.value.telemetry and llm.call_count == 0


@pytest.mark.parametrize("field,value", [("billed_usd", 1.0), ("requested_model", None), ("provider", "invented")])
def test_not_dispatched_receipts_cannot_claim_paid_work_or_a_provider(field, value):
    raw = module.PlannerTelemetry.not_dispatched("requested-model").to_dict()
    raw[field] = value
    with pytest.raises(ValueError):
        module.PlannerTelemetry.from_dict(raw)
