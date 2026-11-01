"""The Anthropic route: Claude through the first-party Messages API, with the Bedrock route's rules.

Every call goes to a mock transport: no request leaves this machine.
"""

from __future__ import annotations

import json
import time

import pytest

from taste.benchmarks.harbor_settings import TrialSettings, model_route
from taste.brains.hosted_worker import _reply
from taste.llm import LLM, InfraFailure
from taste.pricing import max_call_cost_usd
from taste.providers.azure_openai import (
    ANTHROPIC_MODELS,
    AZURE_LUNA6_MODEL,
    AzureDeployment,
    AzureOpenAIConfig,
    route_of,
)
from taste.providers.base import ProtocolFailure

httpx = pytest.importorskip("httpx")

ENDPOINT = "https://api.anthropic.com"
KEY = "anthropic-test-only"
SONNET = "claude-sonnet-5-5"
BINDINGS = (AzureDeployment(SONNET, "claude-sonnet-5-5"),)
BASH = {"type": "function", "name": "bash", "description": "Run a command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}, "strict": False}


def route(**overrides):
    return AzureOpenAIConfig(**{"base_url": ENDPOINT, "api_key": KEY, "deployments": BINDINGS, **overrides})


def message(*blocks, stop="end_turn", usage=None, model=SONNET):
    """A Messages reply as the Anthropic API writes one."""
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": model,
            "content": list(blocks) or [{"type": "text", "text": "Done."}], "stop_reason": stop,
            "stop_sequence": None,
            "usage": usage or {"input_tokens": 12, "output_tokens": 5, "cache_read_input_tokens": 1000,
                               "cache_creation_input_tokens": 200}}


def ok(*blocks, **options):
    return lambda wire: httpx.Response(200, json=message(*blocks, **options), headers={"request-id": "req_test"})


@pytest.fixture
def wire(monkeypatch):
    """Install a handler behind every httpx client; returns what was sent."""
    original = httpx.Client
    clients, sent = [], []

    def install(handler):
        def handle(request):
            sent.append(request)
            return handler(request)

        class Client(original):
            def __init__(self, **kwargs):
                super().__init__(**kwargs, transport=httpx.MockTransport(handle))
                clients.append(self)

        monkeypatch.setattr(httpx, "Client", Client)
        return sent

    yield install
    for client in clients:
        client.close()


def body(request):
    return json.loads(request.content)


def ask(llm, model=SONNET, **overrides):
    return llm.call(**{"model": model, "system": "", "messages": [{"role": "user", "content": "hi"}],
                       "tools": [BASH], "max_tokens": 256, "effort": "low", "temperature": None,
                       "timeout_seconds": 600, **overrides})


def native(item):
    return {"role": "user", "content": [{"type": "_openai_item", "item": item}]}


# ---------------------------------------------------------------- the route


def test_the_anthropic_route_admits_sonnet_only_under_its_api_model_id():
    config = route()
    assert config.base_url == ENDPOINT + "/" and config.route == "anthropic" == route_of(ENDPOINT)
    assert KEY not in repr(config)
    assert ANTHROPIC_MODELS == {SONNET: "claude-sonnet-5-5"}
    with pytest.raises(ProtocolFailure, match="API model id"):
        route(deployments=(AzureDeployment(SONNET, "global.anthropic.claude-sonnet-5-5"),))
    # Luna is not Anthropic's, and Haiku's long-prompt price tier is not in the table: neither is admitted.
    for model, deployment in ((AZURE_LUNA6_MODEL, "gpt-6-luna"), ("claude-haiku-5-5", "claude-haiku-5-5")):
        with pytest.raises(ProtocolFailure, match="API model id"):
            route(deployments=(AzureDeployment(model, deployment),))


@pytest.mark.parametrize("endpoint", [
    "http://api.anthropic.com", ENDPOINT + ":443", ENDPOINT + "/v1", ENDPOINT + "/?x=1",
    "https://user:pw@api.anthropic.com", "https://api.anthropic.com.evil.invalid", "https://anthropic.com",
    "\n" + ENDPOINT,
])
def test_other_or_ambiguous_anthropic_endpoints_are_refused(endpoint):
    with pytest.raises(ProtocolFailure):
        route(base_url=endpoint)


# ---------------------------------------------------------------- the request


def test_sonnet_is_asked_through_messages_with_its_tools_effort_adaptive_thinking_and_two_cache_points(wire):
    sent = wire(ok())
    completion = ask(LLM(azure_openai=route()), system="A long, stable system prompt.",
                     messages=[{"role": "user", "content": "first"}, {"role": "assistant", "content": "ok"},
                               {"role": "user", "content": "second"}], temperature=1.0)
    request = sent[0]
    assert str(request.url) == ENDPOINT + "/v1/messages"
    assert request.headers["x-api-key"] == KEY and request.headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in request.headers
    sent_body = body(request)
    assert sent_body["model"] == "claude-sonnet-5-5" and sent_body["max_tokens"] == 256
    assert sent_body["system"] == [{"type": "text", "text": "A long, stable system prompt.",
                                    "cache_control": {"type": "ephemeral"}}]
    assert sent_body["messages"][-1] == {"role": "user", "content": [
        {"type": "text", "text": "second", "cache_control": {"type": "ephemeral"}}]}
    assert request.content.decode().count("cache_control") == 2
    assert sent_body["tools"] == [{"name": "bash", "description": "Run a command.",
                                   "input_schema": BASH["parameters"]}]
    assert sent_body["thinking"] == {"type": "adaptive"} and sent_body["output_config"] == {"effort": "low"}
    assert "temperature" not in request.content.decode()
    assert completion.effective_sampling == {"temperature": None, "effort": "low", "thinking": "adaptive",
                                             "thinking_carried": False, "dropped": ["temperature"]}
    assert completion.provider == "anthropic_api"
    assert completion.provenance["route"] == "anthropic" and completion.provenance["request_id"] == "req_test"
    assert completion.provenance["served_model"] == SONNET


def test_a_hosted_agents_items_become_alternating_turns_with_tool_results_first(wire):
    sent = wire(ok())
    messages = [{"role": "user", "content": "go"},
                native({"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "Listing.", "annotations": []}]}),
                native({"type": "function_call", "call_id": "toolu_1", "name": "bash",
                        "arguments": '{"command": "ls"}', "status": "completed"}),
                native({"type": "function_call_output", "call_id": "toolu_1", "output": "src\n"}),
                {"role": "user", "content": "and now?"},
                native({"type": "reasoning", "id": "rs_1", "summary": []})]
    completion = ask(LLM(azure_openai=route()), messages=messages)
    turns = body(sent[0])["messages"]
    assert [turn["role"] for turn in turns] == ["user", "assistant", "user"]
    assert turns[1]["content"] == [{"type": "text", "text": "Listing."},
                                   {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "ls"}}]
    assert turns[2]["content"][0] == {"type": "tool_result", "tool_use_id": "toolu_1", "content": "src\n"}
    assert turns[2]["content"][1]["text"] == "and now?"
    assert completion.effective_sampling["reasoning_items_not_sent"] == 1


def test_a_command_that_printed_nothing_is_a_tool_result_with_no_content(wire):
    sent = wire(ok())
    ask(LLM(azure_openai=route()), messages=[
        {"role": "user", "content": "go"},
        native({"type": "function_call", "call_id": "t", "name": "bash", "arguments": "{}", "status": "completed"}),
        native({"type": "function_call_output", "call_id": "t", "output": ""})])
    assert body(sent[0])["messages"][-1]["content"][0] == {"type": "tool_result", "tool_use_id": "t",
                                                          "cache_control": {"type": "ephemeral"}}


# ---------------------------------------------------------------- the reply


def test_a_reply_comes_back_as_responses_items_without_its_thinking(wire):
    sent = wire(ok({"type": "thinking", "thinking": "Let me look.", "signature": "c2ln"},
                   {"type": "redacted_thinking", "data": "b3BhcXVl"},
                   {"type": "text", "text": "I will list the files."},
                   {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "ls"}},
                   stop="tool_use"))
    llm = LLM(azure_openai=route())
    completion = ask(llm)
    assert "Let me look." not in json.dumps(completion.transcript_blocks) and "c2ln" not in json.dumps(completion.transcript_blocks)
    [call] = completion.tool_calls
    assert (call.id, call.name, call.arguments) == ("toolu_1", "bash", {"command": "ls"})
    reply = _reply(completion, 0.001)
    assert [dict(item) for item in reply.output] == [
        {"type": "message", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "I will list the files.", "annotations": []}]},
        {"type": "function_call", "call_id": "toolu_1", "name": "bash", "arguments": '{"command": "ls"}',
         "status": "completed"}]
    again = [{"role": "user", "content": "go"}, *(native(dict(item)) for item in reply.output),
             native({"type": "function_call_output", "call_id": "toolu_1", "output": "src"})]
    ask(llm, messages=again)
    assert body(sent[1])["messages"][1] == {"role": "assistant", "content": [
        {"type": "text", "text": "I will list the files."},
        {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "ls"}}]}


@pytest.mark.parametrize("stop", ["max_tokens", "model_context_window_exceeded", "refusal", "pause_turn"])
def test_a_reply_cut_short_hands_over_no_tool_call(wire, stop):
    wire(ok({"type": "text", "text": "Halfway"},
            {"type": "tool_use", "id": "t", "name": "bash", "input": {"command": "rm"}}, stop=stop))
    completion = ask(LLM(azure_openai=route()))
    assert completion.tool_calls == () and completion.text_blocks == ("Halfway",)
    assert _reply(completion, 0.0).output == ()


@pytest.mark.parametrize("stop, canonical", [
    ("end_turn", "end_turn"), ("stop_sequence", "end_turn"), ("max_tokens", "max_tokens"),
    ("refusal", "refusal"), ("model_context_window_exceeded", "context_window"), ("pause_turn", "other"),
    ("something_new", "other")])
def test_each_stop_is_named_so_it_can_be_counted(wire, stop, canonical):
    wire(ok({"type": "text", "text": "No."}, stop=stop))
    assert ask(LLM(azure_openai=route())).stop_reason == canonical


@pytest.mark.parametrize("usage", [
    {"input_tokens": 1, "output_tokens": 1, "cache_creation_input_tokens": 0},
    {"input_tokens": 1, "output_tokens": -1, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    {"input_tokens": "1", "output_tokens": 1, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": None, "cache_creation_input_tokens": 0},
])
def test_usage_must_report_every_bucket(wire, usage):
    sent = wire(ok(usage=usage))
    cap = max_call_cost_usd(SONNET, max_output_tokens=256)
    llm = LLM(azure_openai=route(), budget_usd=cap, cap_on="billed")
    with pytest.raises(ProtocolFailure, match="usage"):
        ask(llm)
    with pytest.raises(ProtocolFailure, match="unsettled"):
        ask(llm)
    assert len(sent) == 1


def test_a_reply_from_another_model_is_refused(wire):
    wire(ok(model="claude-opus-5-5"))
    with pytest.raises(ProtocolFailure, match="served"):
        ask(LLM(azure_openai=route()))


def test_cache_reads_and_writes_are_priced_apart(wire):
    wire(ok(usage={"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 10_000,
                   "cache_creation_input_tokens": 2_000}))
    completion = ask(LLM(azure_openai=route()))
    usage = completion.usage
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens, usage.cache_write_tokens) == (
        100, 50, 10_000, 2_000)


# ---------------------------------------------------------------- waits and failures


def refused(status):
    kind = {429: "rate_limit_error", 529: "overloaded_error"}[status]
    return lambda _request: httpx.Response(status, json={"type": "error", "error": {"type": kind, "message": "wait"}},
                                           headers={"retry-after": "3"})


@pytest.mark.parametrize("status", [429, 529])
def test_a_declined_request_is_waited_out_and_its_waits_stay_on_the_record(wire, monkeypatch, status):
    waits = []
    monkeypatch.setattr("taste.providers._anthropic_route.time.sleep", waits.append)
    refusals = iter([True, True])
    sent = wire(lambda request: refused(status)(request) if next(refusals, False) else ok()(request))
    llm = LLM(azure_openai=route())
    completion = ask(llm)
    assert waits == [3.0, 3.0] and len(sent) == 3
    assert len({request.content for request in sent}) == 1
    assert completion.provenance["throttled_waits"] == "2"
    moments = [float(value) for value in completion.provenance["throttled_at_unix"].split(",")]
    assert len(moments) == 2 and all(abs(moment - time.time()) < 60 for moment in moments)


def test_a_request_that_never_left_this_machine_is_sent_again(wire, monkeypatch):
    waits = []
    monkeypatch.setattr("taste.providers._anthropic_route.time.sleep", waits.append)
    failures = iter([httpx.ConnectError("no connection")])

    def handler(request):
        failure = next(failures, None)
        if failure is not None:
            raise failure
        return ok()(request)

    sent = wire(handler)
    llm = LLM(azure_openai=route())
    assert ask(llm).provenance["throttled_waits"] == "0"
    assert len(sent) == 2 and llm.provider_for(SONNET).unconnected == 1


@pytest.mark.parametrize("failure,transient", [
    ("read_timeout", True), ("dropped", True), ("500", True), ("400", False), ("401", False), ("307", False)])
def test_a_request_that_may_have_arrived_is_never_sent_again(wire, monkeypatch, failure, transient):
    monkeypatch.setattr("taste.providers._anthropic_route.time.sleep", lambda _seconds: None)

    def handler(request):
        if failure == "read_timeout":
            raise httpx.ReadTimeout("no answer", request=request)
        if failure == "dropped":
            raise httpx.RemoteProtocolError("server disconnected", request=request)
        return httpx.Response(int(failure), json={"type": "error", "error": {"type": "x", "message": "refused"}},
                              headers={"location": "https://elsewhere.invalid/"})

    sent = wire(handler)
    llm = LLM(azure_openai=route())
    with pytest.raises(InfraFailure) as raised:
        ask(llm)
    assert len(sent) == 1 and raised.value.transient is transient
    assert KEY not in str(raised.value) and KEY not in repr(raised.value.last_error)


# ---------------------------------------------------------------- the host picks a route per model


def test_claude_takes_the_anthropic_route_when_the_host_names_it_and_luna_keeps_bedrock():
    bedrock = {"TASTE_BEDROCK_URL": "https://bedrock-runtime.us-east-1.amazonaws.com",
               "AWS_BEARER_TOKEN_BEDROCK": "b"}
    anthropic = {"TASTE_ANTHROPIC_URL": ENDPOINT, "ANTHROPIC_API_KEY": KEY}
    both = {**bedrock, **anthropic}
    assert model_route(both, model=SONNET) == (ENDPOINT + "/", KEY)
    assert model_route(both, model=AZURE_LUNA6_MODEL) == (bedrock["TASTE_BEDROCK_URL"] + "/", "b")
    assert model_route(both) == (bedrock["TASTE_BEDROCK_URL"] + "/", "b")
    # Without TASTE_ANTHROPIC_URL a key alone changes nothing: Claude stays on Bedrock.
    assert model_route({**bedrock, "ANTHROPIC_API_KEY": KEY}, model=SONNET) == (bedrock["TASTE_BEDROCK_URL"] + "/", "b")
    with pytest.raises(RuntimeError) as raised:
        model_route({**bedrock, "TASTE_ANTHROPIC_URL": ENDPOINT}, model=SONNET)
    assert KEY not in str(raised.value)


def test_an_agent_run_alone_on_the_anthropic_route_uses_its_api_model_id():
    settings = TrialSettings.from_options({"model": SONNET, "agent": "mini-swe-agent", "services": "none",
                                           "worker_effort": "low"})
    policy = settings.policy(ENDPOINT + "/", time.time() + 1500, owner_token="a" * 32,
                             container_id="c" * 64, workdir="/app")
    assert (policy.planner_deployment, policy.worker_deployment) == ("fixed-plan", "claude-sonnet-5-5")
    config = policy.azure_config({"AZURE_OPENAI_BASE_URL": policy.endpoint, "AZURE_OPENAI_API_KEY": KEY})
    assert config.route == "anthropic"


# ---------------------------------------------------------------- an organization key names its workspace

WORKSPACE = "wrkspc_013testWorkspace0001"


def test_an_organization_key_names_its_workspace_in_the_endpoint_and_each_request_carries_it(wire):
    config = route(base_url=f"{ENDPOINT}/workspaces/{WORKSPACE}")
    assert config.route == "anthropic" and config.anthropic_workspace == WORKSPACE
    assert config.base_url == f"{ENDPOINT}/workspaces/{WORKSPACE}/"
    assert route().anthropic_workspace is None
    sent = wire(ok())
    completion = ask(LLM(azure_openai=config))
    assert str(sent[0].url) == ENDPOINT + "/v1/messages"
    assert sent[0].headers["anthropic-workspace-id"] == WORKSPACE
    assert completion.provenance["endpoint"] == config.base_url
    # Without a workspace in the endpoint, no workspace header is sent.
    sent.clear()
    ask(LLM(azure_openai=route()))
    assert "anthropic-workspace-id" not in sent[0].headers


@pytest.mark.parametrize("path", ["/workspaces/", "/workspaces/default", "/workspaces/wrkspc_bad-id",
                                  f"/workspaces/{WORKSPACE}/v1", f"/workspace/{WORKSPACE}"])
def test_only_a_well_formed_workspace_path_is_admitted(path):
    with pytest.raises(ProtocolFailure):
        route(base_url=ENDPOINT + path)


def test_the_host_names_the_workspace_in_taste_anthropic_url():
    environment = {"TASTE_ANTHROPIC_URL": f"{ENDPOINT}/workspaces/{WORKSPACE}", "ANTHROPIC_API_KEY": KEY}
    assert model_route(environment, model=SONNET) == (f"{ENDPOINT}/workspaces/{WORKSPACE}/", KEY)
