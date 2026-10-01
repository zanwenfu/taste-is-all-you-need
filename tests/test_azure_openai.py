"""Azure routing, real SDK authentication, provenance and uncertain spending."""

from __future__ import annotations

import json

import pytest

from taste import providers
from taste.llm import LLM, InfraFailure
from taste.pricing import max_call_cost_usd, rates_for
from taste.providers.azure_openai import (
    AZURE_MONITOR_MODEL,
    AZURE_PLANNER_MODEL,
    AZURE_WORKER_MODEL,
    AzureDeployment,
    AzureOpenAIConfig,
)
from taste.providers.base import ProtocolFailure
from tests.test_openai_responses import response

httpx = pytest.importorskip("httpx")
pytest.importorskip("openai")

ENDPOINT = "https://test-resource.openai.azure.com/openai/v1"
BINDINGS = (
    AzureDeployment(AZURE_PLANNER_MODEL, "gpt-6-astra"),
    AzureDeployment(AZURE_WORKER_MODEL, "gpt-6-sol"),
)


def config(**overrides):
    return AzureOpenAIConfig(**{
        "api_key": "azure-test-only", "base_url": ENDPOINT,
        "deployments": BINDINGS, **overrides,
    })


def invoke(llm, model=AZURE_WORKER_MODEL):
    return llm.call(model=model, system="Be precise.", messages=[], max_tokens=128,
                    effort="low", timeout_seconds=2)


@pytest.fixture
def sdk_transport(monkeypatch):
    original = httpx.Client
    clients, sent, options = [], [], []

    def install(handler):
        def handle(wire):
            sent.append(wire)
            return handler(wire)

        class Client(original):
            def __init__(self, **kwargs):
                options.append(kwargs.copy())
                super().__init__(**kwargs, transport=httpx.MockTransport(handle))
                clients.append(self)

        monkeypatch.setattr(httpx, "Client", Client)
        return sent, options

    yield install
    for client in clients:
        client.close()


def success(wire):
    deployment = json.loads(wire.content)["model"]
    model = next(b.model for b in BINDINGS if b.deployment == deployment)
    return httpx.Response(200, json=response(model=deployment), headers={
        "x-ms-served-model": model, "azureml-model-session": "session_test",
        "x-ms-region": "eastus",
    })


def test_azure_uses_only_explicit_credentials_endpoint_and_deployment(monkeypatch, tmp_path, sdk_transport):
    for key, value in {
        "OPENAI_API_KEY": "personal-do-not-use", "ANTHROPIC_API_KEY": "claude-do-not-use",
        "OPENAI_BASE_URL": "https://wrong.invalid/v1", "OPENAI_ORG_ID": "personal-org",
        "OPENAI_PROJECT_ID": "personal-project", "HTTPS_PROXY": "http://wrong.invalid:9000",
    }.items():
        monkeypatch.setenv(key, value)
    (tmp_path / ".env").write_text("AZURE_OPENAI_BASE_URL=https://wrong.invalid/v1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(providers._CACHE, "openai", object())
    sent, options = sdk_transport(success)
    llm = LLM(azure_openai=config())  # even the default load_env_file=True must not discover task files
    result = invoke(llm)
    assert len(sent) == 1
    wire = sent[0]
    assert str(wire.url) == ENDPOINT + "/responses"
    assert wire.headers["authorization"] == "Bearer azure-test-only"
    assert wire.headers.get("openai-organization", "") == ""
    assert wire.headers.get("openai-project", "") == ""
    assert json.loads(wire.content)["model"] == "gpt-6-sol"
    assert options == [{"trust_env": False, "follow_redirects": False}]
    assert result.model == AZURE_WORKER_MODEL
    assert result.provenance == {
        "route": "azure_openai", "endpoint": ENDPOINT + "/", "deployment": "gpt-6-sol",
        "deployment_type": "GlobalStandard", "served_model": AZURE_WORKER_MODEL,
        "model_session": "session_test", "region": "eastus",
    }
    assert llm.stats.per_model[AZURE_WORKER_MODEL].calls == 1


def test_direct_key_never_substitutes_for_missing_azure_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "personal-do-not-use")
    with pytest.raises(ProtocolFailure, match="AZURE_OPENAI_API_KEY"):
        AzureOpenAIConfig.from_environment({
            "OPENAI_API_KEY": "personal-do-not-use", "AZURE_OPENAI_BASE_URL": ENDPOINT,
        }, deployments=BINDINGS)


def test_sdk_environment_headers_cannot_override_azure_authorization(monkeypatch, sdk_transport):
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer personal-do-not-use")
    sent, options = sdk_transport(success)
    with pytest.raises(ProtocolFailure, match="OPENAI_CUSTOM_HEADERS"):
        LLM(azure_openai=config()).ensure_ready(AZURE_WORKER_MODEL)
    assert sent == options == []


@pytest.mark.parametrize("endpoint", [
    "https://api.openai.com/v1", "http://test.openai.azure.com/openai/v1",
    "https://test.openai.azure.com.evil.invalid/openai/v1",
    "https://user:secret@test.openai.azure.com/openai/v1",
    "https://test.openai.azure.com:9443/openai/v1",
    ENDPOINT + "?api-key=secret", ENDPOINT + "#fragment", ENDPOINT + "/responses",
    "https://test.openai.azure.com/openai/%76%31", "\n" + ENDPOINT,
])
def test_non_azure_or_ambiguous_endpoints_fail_before_client_creation(endpoint):
    with pytest.raises(ProtocolFailure):
        config(base_url=endpoint)


@pytest.mark.parametrize("model", ["claude-sonnet-4-6", "gpt-5.6-terra", "gpt-6-sol-2026-09-22"])
def test_undeclared_models_cannot_fall_back_to_another_provider(model, sdk_transport):
    sent, _ = sdk_transport(success)
    llm = LLM(azure_openai=config(deployments=(BINDINGS[0],)))
    with pytest.raises(ProtocolFailure, match="not admitted"):
        invoke(llm, model)
    assert sent == []


@pytest.mark.parametrize("overrides", [
    {"api_key": "personal"}, {"api_keys": {"anthropic": "personal"}},
    {"max_attempts": 2}, {"max_attempts": True},
])
def test_mixed_credentials_and_paid_retry_loops_are_rejected(overrides):
    with pytest.raises(ValueError):
        LLM(azure_openai=config(), **overrides)


def test_duplicate_or_unverified_deployment_bindings_are_rejected():
    with pytest.raises(ProtocolFailure, match="unique"):
        config(deployments=(BINDINGS[0], BINDINGS[0]))
    with pytest.raises(ProtocolFailure, match="GlobalStandard"):
        AzureDeployment(AZURE_WORKER_MODEL, "gpt-6-sol", "DataZoneStandard")
    with pytest.raises(ProtocolFailure, match="pricing"):
        AzureDeployment("claude-opus-5", "claude-opus-5")
    assert "azure-test-only" not in repr(config())


@pytest.mark.parametrize("failure", ["lost_reply", "500", "bad_usage", "wrong_snapshot", "missing_snapshot", "redirect"])
def test_failed_dispatch_retains_budget_and_fences_all_further_calls(sdk_transport, failure):
    def handler(wire):
        if failure == "lost_reply":
            raise httpx.ReadError("reply lost", request=wire)
        if failure == "500":
            # A server error may follow a reply that was produced and charged.
            return httpx.Response(500, json={"error": {"message": "server error"}})
        if failure == "redirect":
            return httpx.Response(307, headers={"location": "https://api.openai.com/v1/responses"})
        if failure == "bad_usage":
            return httpx.Response(200, json=response(model=AZURE_WORKER_MODEL, usage={}))
        if failure == "wrong_snapshot":
            return httpx.Response(200, json=response(model="gpt-6-sol"), headers={"x-ms-served-model": "gpt-6-sol-2099-01-01"})
        return httpx.Response(200, json=response(model="gpt-6-sol"))

    sent, _ = sdk_transport(handler)
    cap = max_call_cost_usd(AZURE_WORKER_MODEL, max_output_tokens=128)
    llm = LLM(azure_openai=config(), budget_usd=cap, cap_on="billed")
    with pytest.raises((ProtocolFailure, InfraFailure)):
        invoke(llm)
    assert len(sent) == 1
    assert llm._reserved_usd == cap
    assert llm.stats.totals.calls == 0
    with pytest.raises(ProtocolFailure, match="unsettled"):
        invoke(llm)
    assert len(sent) == 1


def refused(seconds="0"):
    return httpx.Response(429, json={"error": {"code": "rate_limit_exceeded", "message": "rate limited"}},
                          headers={"retry-after": seconds})


def test_a_rate_limit_refusal_is_waited_out_and_the_same_request_sent_again(sdk_transport, monkeypatch):
    # The service declined to run the request: nothing was produced, nothing
    # billed. Treated as a lost paid reply it ended a worker's whole run.
    waits = []
    monkeypatch.setattr("taste.providers._openai.time.sleep", waits.append)
    refusals = iter([refused("7"), refused("not-a-number")])

    def handler(wire):
        return next(refusals, None) or success(wire)

    sent, _ = sdk_transport(handler)
    cap = max_call_cost_usd(AZURE_WORKER_MODEL, max_output_tokens=128)
    llm = LLM(azure_openai=config(), budget_usd=10 * cap, cap_on="billed")
    completion = llm.call(model=AZURE_WORKER_MODEL, system="Be precise.", messages=[], max_tokens=128,
                          effort="low", timeout_seconds=600)
    assert completion.provenance["served_model"] == AZURE_WORKER_MODEL
    # What the service asked for, then a bounded default when it did not say.
    assert waits == [7.0, 4.0] and len(sent) == 3
    assert len({wire.content for wire in sent}) == 1, "the request sent again is the same request"
    # One call was made and paid for; its reservation was released on settlement.
    assert llm.stats.totals.calls == 1 and llm._reserved_usd == 0
    assert invoke(llm).provenance["served_model"] == AZURE_WORKER_MODEL


def test_a_refusal_that_outlasts_the_bounded_waits_fails_as_before(sdk_transport, monkeypatch):
    monkeypatch.setattr("taste.providers._openai.time.sleep", lambda _seconds: None)
    sent, _ = sdk_transport(lambda _wire: refused("1"))
    cap = max_call_cost_usd(AZURE_WORKER_MODEL, max_output_tokens=128)
    llm = LLM(azure_openai=config(), budget_usd=cap, cap_on="billed")
    with pytest.raises(InfraFailure):
        llm.call(model=AZURE_WORKER_MODEL, system="Be precise.", messages=[], max_tokens=128,
                 effort="low", timeout_seconds=600)
    assert len(sent) == 6 and llm.stats.totals.calls == 0
    with pytest.raises(ProtocolFailure, match="unsettled"):
        invoke(llm)


def test_a_refusal_is_not_waited_out_past_the_calls_own_deadline(sdk_transport, monkeypatch):
    waits = []
    monkeypatch.setattr("taste.providers._openai.time.sleep", waits.append)
    sent, _ = sdk_transport(lambda _wire: refused("30"))
    llm = LLM(azure_openai=config())
    with pytest.raises(InfraFailure):
        invoke(llm)  # two seconds allowed; the service asks for thirty
    assert len(sent) == 1 and waits == []


def test_separate_llms_do_not_reuse_other_azure_credentials(sdk_transport):
    sent, _ = sdk_transport(success)
    first, second = LLM(azure_openai=config()), LLM(azure_openai=config(api_key="second-azure"))
    invoke(first)
    invoke(second)
    assert [w.headers["authorization"] for w in sent] == ["Bearer azure-test-only", "Bearer second-azure"]


def test_body_snapshot_can_identify_the_model_without_a_header(sdk_transport):
    sdk_transport(lambda wire: httpx.Response(200, json=response(model=AZURE_WORKER_MODEL)))
    assert invoke(LLM(azure_openai=config())).model == AZURE_WORKER_MODEL


def test_verified_role_prices_and_whole_request_long_context_boundary():
    assert AZURE_MONITOR_MODEL == AZURE_WORKER_MODEL
    for model, short_input, short_output in [(AZURE_PLANNER_MODEL, 10, 50), (AZURE_WORKER_MODEL, 2, 10)]:
        short = rates_for(model, 272_000)
        long = rates_for(model, 272_001)
        assert (short.input, short.output) == (short_input, short_output)
        assert (long.input, long.output) == (short_input * 2, short_output * 1.5)
        assert short.cache_write == short.input * 1.25
