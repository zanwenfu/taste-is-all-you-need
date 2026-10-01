"""Azure planner routes survive paid receipts, process reopen and failures."""
from __future__ import annotations

import json
from contextlib import closing
from dataclasses import replace
from types import MappingProxyType

import pytest

from taste.brains.planner_transport import PlannerCompletionError, PlannerReceiptError
from taste.llm import LLM
from taste.memstore import Store
from taste.providers.azure_openai import AZURE_PLANNER_MODEL, AzureDeployment
from tests.test_azure_openai import config, httpx, success
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_brains_planner_transport import evidence, make_transport
from tests.test_openai_responses import response

sdk_transport = _sdk_transport

def planner(store, azure=None, **kwargs):
    llm = LLM(azure_openai=azure or config(), load_env_file=False, max_attempts=1)
    return make_transport(llm, store, model=AZURE_PLANNER_MODEL, max_tokens=128, **kwargs)


def call(transport, request_id="azure-attempt"):
    return transport.complete(request_id=request_id, system="planner system", prompt="world")


@pytest.mark.parametrize("change", ["endpoint", "deployment", "direct_provider"])
def test_reopened_planner_cannot_replay_paid_receipt_under_another_route(tmp_path, sdk_transport, change):
    sent, _ = sdk_transport(success)
    root = tmp_path / "repo"
    with closing(Store.open(root, "azure-planner")) as store:
        first = planner(store)
        original = call(first)
        original_receipt = evidence(first, "azure-attempt")
        assert original.telemetry.cost_known and len(sent) == 1
        control_id, journal_id = first.control.head.id, first.journal.head.id
    with closing(Store.open(root, "azure-planner")) as store:
        if change == "endpoint":
            altered = planner(store, config(base_url="https://other.openai.azure.com/openai/v1/"))
        elif change == "deployment":
            altered = planner(store, config(deployments=(AzureDeployment(AZURE_PLANNER_MODEL, "other"),)))
        else:
            altered = make_transport(LLM(load_env_file=False, max_attempts=1), store,
                                     model=AZURE_PLANNER_MODEL, max_tokens=128)
        with pytest.raises(PlannerReceiptError, match="call config"):
            call(altered)
        assert len(sent) == 1
        assert altered.control.head.id == control_id
        assert altered.journal.head.id == journal_id
        assert evidence(altered, "azure-attempt") == original_receipt


def test_a_chosen_reasoning_effort_is_sent_and_bound_into_the_receipt(tmp_path, sdk_transport):
    sent, _ = sdk_transport(success)
    root = tmp_path / "repo"
    with closing(Store.open(root, "azure-planner")) as store:
        # Unchosen, the request names no effort: the provider's default applies.
        call(planner(store), "default-effort")
        assert "reasoning" not in json.loads(sent[0].content)
        reasoning = planner(store, effort="medium")
        call(reasoning, "chosen-effort")
        assert json.loads(sent[1].content)["reasoning"] == {"effort": "medium"}
    with closing(Store.open(root, "azure-planner")) as store:
        # A receipt made at one effort is not replayed as if made at another.
        with pytest.raises(PlannerReceiptError, match="call config"):
            call(planner(store, effort="high"), "chosen-effort")
        assert call(planner(store, effort="medium"), "chosen-effort").telemetry.cost_known
        assert len(sent) == 2


def test_rotated_azure_key_replays_old_receipt_and_only_new_request_uses_new_key(tmp_path, sdk_transport):
    sent, _ = sdk_transport(success)
    root = tmp_path / "repo"
    with closing(Store.open(root, "azure-planner")) as store:
        original = call(planner(store))
    with closing(Store.open(root, "azure-planner")) as store:
        restarted = planner(store, config(api_key="rotated-test-key"))
        assert call(restarted) == original
        assert restarted.llm.stats.totals.calls == 0
        assert len(sent) == 1
        assert call(restarted, "new-attempt").telemetry.cost_known
        assert [wire.headers["authorization"] for wire in sent] == [
            "Bearer azure-test-only", "Bearer rotated-test-key"]
        assert all(json.loads(wire.content)["model"] == "gpt-6-astra" for wire in sent)
        assert all(str(wire.url) == config().base_url + "responses" for wire in sent)
        for branch in (restarted.control, restarted.journal):
            for path in branch.head.files():
                raw = branch.head.read(path)
                if raw is not None:
                    assert "azure-test-only" not in raw and "rotated-test-key" not in raw


@pytest.mark.parametrize("field", ["route", "endpoint", "deployment", "deployment_type", "served_model"])
def test_mismatched_completion_route_is_unknown_and_replay_never_redispatches(
        tmp_path, sdk_transport, monkeypatch, field):
    sent, _ = sdk_transport(success)
    with closing(Store.open(tmp_path / "repo", "azure-planner")) as store:
        transport = planner(store)
        real_call = transport.llm.call

        def mismatched(**kwargs):
            completion = real_call(**kwargs)
            return replace(completion, provenance={**completion.provenance, field: "changed"})

        monkeypatch.setattr(transport.llm, "call", mismatched)
        with pytest.raises(PlannerCompletionError) as raised:
            call(transport)
        assert raised.value.telemetry.source == "azure_route_mismatch"
        assert not raised.value.telemetry.cost_known
        assert raised.value.telemetry.billed_usd is None
        assert transport.llm.stats.total_cost_usd > 0  # a paid HTTP reply really occurred
        replay = planner(store, control=transport.control, journal=transport.journal,
                         mutation_lock=transport.mutation_lock)
        with pytest.raises(PlannerCompletionError) as replayed:
            call(replay)
        assert replayed.value.telemetry == raised.value.telemetry
        assert len(sent) == 1


def test_immutable_route_mapping_is_accepted_and_dynamic_region_does_not_rebind(tmp_path, sdk_transport, monkeypatch):
    sent, _ = sdk_transport(success)
    with closing(Store.open(tmp_path / "repo", "azure-planner")) as store:
        transport = planner(store)
        real_call = transport.llm.call

        def immutable(**kwargs):
            completion = real_call(**kwargs)
            return replace(completion, provenance=MappingProxyType({
                **completion.provenance, "region": "westus", "model_session": "new-service-session"}))

        monkeypatch.setattr(transport.llm, "call", immutable)
        assert call(transport).telemetry.cost_known
        assert len(sent) == 1


@pytest.mark.parametrize("failure", ["lost_reply", "wrong_snapshot", "truncated"])
def test_actual_sdk_failures_and_paid_invalid_output_remain_receipted_across_reopen(
        tmp_path, sdk_transport, failure):
    def handler(wire):
        if failure == "lost_reply":
            raise httpx.ReadError("response lost", request=wire)
        if failure == "wrong_snapshot":
            return httpx.Response(200, json=response(model="gpt-6-astra"),
                                  headers={"x-ms-served-model": "unadmitted-snapshot"})
        return httpx.Response(200, json=response(model=AZURE_PLANNER_MODEL, status="incomplete"))

    sent, _ = sdk_transport(handler)
    root = tmp_path / "repo"
    with closing(Store.open(root, "azure-planner")) as store:
        transport = planner(store)
        with pytest.raises(PlannerCompletionError) as raised:
            call(transport)
        telemetry = raised.value.telemetry
        assert telemetry.cost_known == (failure == "truncated")
        receipt = evidence(transport, "azure-attempt")
    with closing(Store.open(root, "azure-planner")) as store:
        reopened = planner(store)
        with pytest.raises(PlannerCompletionError) as raised:
            call(reopened)
        assert raised.value.telemetry == telemetry
        assert evidence(reopened, "azure-attempt") == receipt
        assert len(sent) == 1
