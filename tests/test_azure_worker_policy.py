"""Assignment, process launch, Azure route and durable spending agree exactly."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from taste.brains.azure_worker_policy import AZURE_WORKER_POLICY_SCHEMA, AzureWorkerPolicy
from taste.brains.contract import Contract
from taste.brains.records import Assignment, contract_digest
from taste.brains.responses_session import ResponsesConflict, ResponsesFenced, ResponsesSession
from taste.brains.worker_admission import EntrypointConfig, EntrypointInputError
from taste.memstore import Store
from taste.pricing import table_sha
from taste.providers.azure_openai import AZURE_MONITOR_MODEL, AZURE_WORKER_MODEL
from tests.test_azure_openai import config, success
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_responses_session import binding, call

sdk_transport = _sdk_transport


def assignment(**changes):
    brief = Contract("worker", "produce the artifact", success_criteria=("output is correct",),
                     budget_usd=30, max_turns=4)
    raw = {
        "schema": AZURE_WORKER_POLICY_SCHEMA, "endpoint": config().base_url,
        "worker_deployment": "gpt-6-sol", "monitor_deployment": "gpt-6-sol",
        "deadline_unix": binding().deadline_unix, "worker_max_calls": 4, "worker_max_output_tokens": 128,
        "monitor_max_calls": 3, "monitor_max_output_tokens": 256,
        "max_request_bytes": 196_608, "monitor_batch_size": 10, "pricing_sha": table_sha(),
        **changes,
    }
    return Assignment("task", 1, 0, brief, contract_digest(brief), "a" * 40,
                      model=AZURE_WORKER_MODEL, resources={"azure_openai": raw, "monitor_budget_usd": 20})


def environment(**changes):
    return {"AZURE_OPENAI_BASE_URL": config().base_url, "AZURE_OPENAI_API_KEY": "azure-secret",
            "OPENAI_API_KEY": "personal-key-must-not-be-used", "ANTHROPIC_API_KEY": "never-claude",
            **changes}


def test_assignment_roundtrip_preserves_exact_worker_and_monitor_bindings():
    original = assignment()
    policy = AzureWorkerPolicy.from_assignment(original)
    reopened = AzureWorkerPolicy.from_assignment(Assignment.from_json(original.to_json()))
    assert policy == reopened
    assert policy.worker.budget_usd == 30 and policy.monitor.budget_usd == 20
    assert policy.worker.deadline_unix == policy.monitor.deadline_unix
    assert policy.monitor.run_id == policy.worker.run_id + ".monitor"
    assert policy.monitor.role == "monitor" and policy.worker.role == "worker"
    azure = policy.azure_config(environment())
    assert azure.api_key == "azure-secret" and "azure-secret" not in repr(azure)
    assert "secret" not in original.to_json()


@pytest.mark.parametrize("changes", [
    {"worker_max_calls": True}, {"worker_max_calls": 5}, {"worker_max_calls": 0},
    {"monitor_max_calls": 10_001}, {"worker_max_output_tokens": 128_001},
    {"monitor_max_output_tokens": False}, {"max_request_bytes": 1_048_577},
    {"monitor_batch_size": True}, {"monitor_batch_size": 1001}, {"deadline_unix": 0},
    {"deadline_unix": True}, {"deadline_unix": "tomorrow"}, {"monitor_deployment": "different"},
    {"worker_deployment": "gpt-6-sol/invalid"}, {"endpoint": "https://api.openai.com/v1/"},
    {"endpoint": "https://resource.openai.azure.com/openai/v1/?api-key=secret"},
    {"endpoint": config().base_url.rstrip("/")}, {"pricing_sha": "wrong"},
    {"schema": "future"}, {"api_key": "must-not-be-in-assignment"},
    {"request_seconds": 0}, {"request_seconds": True}, {"request_seconds": "45"}, {"request_seconds": 3601},
])
def test_invalid_or_unbound_policy_is_rejected_before_dispatch(changes):
    with pytest.raises(EntrypointInputError):
        AzureWorkerPolicy.from_assignment(assignment(**changes))


@pytest.mark.parametrize("budget", [None, 0])
def test_worker_requires_an_explicit_positive_budget(budget):
    source = assignment()
    contract = replace(source.contract, budget_usd=budget)
    source = replace(source, contract=contract, contract_digest=contract_digest(contract))
    with pytest.raises(EntrypointInputError):
        AzureWorkerPolicy.from_assignment(source)


def test_monitor_requires_an_explicit_positive_budget_and_both_routes_are_azure():
    source = assignment()
    with pytest.raises(EntrypointInputError):
        AzureWorkerPolicy.from_assignment(replace(source, resources={"azure_openai": source.resources["azure_openai"]}))
    with pytest.raises(EntrypointInputError):
        AzureWorkerPolicy.from_assignment(replace(source, model="claude-sonnet-5"))
    with pytest.raises(EntrypointInputError):
        AzureWorkerPolicy.from_assignment(replace(source, resources={}))


@pytest.mark.parametrize("changes", [
    {"AZURE_OPENAI_API_KEY": ""}, {"AZURE_OPENAI_API_KEY": " whitespace "},
    {"AZURE_OPENAI_BASE_URL": ""},
    {"AZURE_OPENAI_BASE_URL": "https://other.openai.azure.com/openai/v1/"},
    {"AZURE_OPENAI_BASE_URL": "https://api.openai.com/v1/"},
])
def test_environment_cannot_change_route_or_fall_back_to_personal_credentials(changes):
    policy = AzureWorkerPolicy.from_assignment(assignment())
    with pytest.raises(EntrypointInputError):
        policy.azure_config(environment(**changes))


def test_process_options_must_match_the_immutable_policy(tmp_path):
    policy = AzureWorkerPolicy.from_assignment(assignment())
    store = Store.open(tmp_path / "repo", "policy")
    try:
        launch = EntrypointConfig(store.root, store.session, "worker", AZURE_WORKER_MODEL, "a" * 40,
                                  monitor_model=AZURE_MONITOR_MODEL, monitor_budget_usd=20,
                                  monitor_max_tokens=256, monitor_batch_size=10)
        policy.validate_launch(launch)
        for changes in ({"monitor_model": "claude-sonnet-5"}, {"expected_model": "gpt-6-sol"},
                        {"monitor_budget_usd": 21}, {"monitor_max_tokens": 257}, {"monitor_batch_size": 11}):
            with pytest.raises(EntrypointInputError):
                policy.validate_launch(replace(launch, **changes))
    finally:
        store.close()


def test_policy_reopen_keeps_spending_and_changed_assignment_cannot_reuse_journal(tmp_path, sdk_transport):
    sent, _ = sdk_transport(success)
    source = assignment()
    policy = AzureWorkerPolicy.from_assignment(source)
    directory = tmp_path / "responses"
    session = ResponsesSession.create(directory, policy.worker, policy.azure_config(environment()))
    try:
        first = asyncio.run(call(session))
    finally:
        session.close()
    reopened_policy = AzureWorkerPolicy.from_assignment(Assignment.from_json(source.to_json()))
    session = ResponsesSession.open(directory, reopened_policy.worker, reopened_policy.azure_config(environment()))
    try:
        assert asyncio.run(call(session)) == first
        assert session.known_cost_usd == pytest.approx(0.000366)
        assert len(sent) == 1
    finally:
        session.close()
    changed = AzureWorkerPolicy.from_assignment(replace(source, attempt=1))
    with pytest.raises(ResponsesConflict, match="binding changed"):
        ResponsesSession.open(directory, changed.worker, changed.azure_config(environment()))
    assert len(sent) == 1


def test_expired_assignment_deadline_does_not_restart_when_reconstructed(tmp_path, sdk_transport):
    sent, _ = sdk_transport(success)
    source = assignment(deadline_unix=1)
    policy = AzureWorkerPolicy.from_assignment(Assignment.from_json(source.to_json()))
    session = ResponsesSession.create(tmp_path / "responses", policy.worker, policy.azure_config(environment()))
    try:
        with pytest.raises(ResponsesFenced, match="deadline"):
            asyncio.run(call(session))
        assert not sent
        assert session.known_cost_usd == 0
    finally:
        session.close()
