"""The Bedrock route: Converse on the wire, Responses items on the harness side, and the Azure route's rules.

Every call goes to a mock transport: no request leaves this machine.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from taste import providers
from taste.agents import HostedStop, ModelReply, ShellResult
from taste.benchmarks.harbor_settings import TrialSettings, model_route
from taste.brains.branch_replay import ReplayHost, request_sha
from taste.brains.hosted_worker import _reply
from taste.brains.responses_session import ResponsesBinding, ResponsesSession
from taste.llm import LLM, InfraFailure
from taste.pricing import PRICES, call_cost, max_call_cost_usd
from taste.providers.azure_openai import (
    AZURE_LUNA6_MODEL,
    AZURE_WORKER_MODEL,
    BEDROCK_PROFILES,
    AzureDeployment,
    AzureOpenAIConfig,
)
from taste.providers.base import ProtocolFailure
from tests.test_branch_replay import OUTPUTS, SENTINEL, Agent, Live, Recorder

httpx = pytest.importorskip("httpx")

ENDPOINT = "https://bedrock-runtime.us-east-1.amazonaws.com"
KEY = "bedrock-test-only"
SONNET = "claude-sonnet-5-5"
BINDINGS = (AzureDeployment(AZURE_LUNA6_MODEL, "global.openai.gpt-6-luna"),
            AzureDeployment(SONNET, "global.anthropic.claude-sonnet-5-5"))
BASH = {"type": "function", "name": "bash", "description": "Run a command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}, "strict": False}


def route(**overrides):
    return AzureOpenAIConfig(**{"base_url": ENDPOINT, "api_key": KEY, "deployments": BINDINGS, **overrides})


def converse(*blocks, stop="end_turn", usage=None):
    """A Converse reply as Bedrock writes one."""
    return {"output": {"message": {"role": "assistant", "content": list(blocks) or [{"text": "Done."}]}},
            "stopReason": stop, "metrics": {"latencyMs": 412},
            "usage": usage or {"inputTokens": 12, "outputTokens": 5, "cacheReadInputTokens": 1000,
                               "cacheWriteInputTokens": 200, "totalTokens": 1217}}


def ok(*blocks, **options):
    return lambda wire: httpx.Response(200, json=converse(*blocks, **options),
                                       headers={"x-amzn-requestid": "req-test"})


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


def ask(llm, model=AZURE_LUNA6_MODEL, **overrides):
    return llm.call(**{"model": model, "system": "", "messages": [{"role": "user", "content": "hi"}],
                       "tools": [BASH], "max_tokens": 256, "effort": "medium", "temperature": None,
                       "timeout_seconds": 600, **overrides})


def native(item):
    return {"role": "user", "content": [{"type": "_openai_item", "item": item}]}


# ---------------------------------------------------------------- the route


def test_a_bedrock_route_admits_each_model_only_through_its_global_profile():
    config = route()
    assert config.base_url == ENDPOINT + "/" and config.route == "bedrock"
    assert KEY not in repr(config)
    # A geographic profile is billed 10% more than the table says; another
    # model's profile would price one model as another.
    for deployment in ("us.openai.gpt-6-luna", "global.openai.gpt-6-sol", "gpt-6-luna"):
        with pytest.raises(ProtocolFailure, match="global profile"):
            route(deployments=(AzureDeployment(AZURE_LUNA6_MODEL, deployment),))
    # Azure serves only its own models.
    with pytest.raises(ProtocolFailure, match="Azure deployment"):
        AzureOpenAIConfig("https://test-resource.openai.azure.com/openai/v1", KEY,
                          (AzureDeployment(SONNET, "claude-sonnet-5-5"),))
    assert BEDROCK_PROFILES[AZURE_WORKER_MODEL] == "global.openai.gpt-6-sol"


@pytest.mark.parametrize("endpoint", [
    "http://bedrock-runtime.us-east-1.amazonaws.com", ENDPOINT + ":443", ENDPOINT + "/model",
    ENDPOINT + "/?x=1", "https://user:pw@bedrock-runtime.us-east-1.amazonaws.com",
    "https://bedrock-runtime.us-east-1.amazonaws.com.evil.invalid", "https://bedrock-mantle.us-east-1.api.aws",
    "https://bedrock.us-east-1.amazonaws.com", "\n" + ENDPOINT,
])
def test_other_or_ambiguous_endpoints_are_refused(endpoint):
    with pytest.raises(ProtocolFailure):
        route(base_url=endpoint)


def test_a_model_the_route_does_not_admit_is_never_sent(wire):
    sent = wire(ok())
    with pytest.raises(ProtocolFailure, match="not admitted"):
        ask(LLM(azure_openai=route(deployments=BINDINGS[1:])))
    assert sent == []
    # Without the route, the Claude models are not reachable at all.
    with pytest.raises(ProtocolFailure, match="unknown provider"):
        providers.for_model(SONNET)


# ---------------------------------------------------------------- the request


def test_luna_is_asked_through_converse_with_its_tools_and_reasoning_effort(wire):
    sent = wire(ok({"text": "Hello."}))
    llm = LLM(azure_openai=route())
    completion = ask(llm, temperature=1.0, tools=[BASH, {"name": "read", "input_schema": {"type": "object"}}])
    [request] = sent
    assert request.url == ENDPOINT + "/model/global.openai.gpt-6-luna/converse"
    assert request.headers["authorization"] == "Bearer " + KEY
    sent_body = body(request)
    assert sent_body["inferenceConfig"] == {"maxTokens": 256}
    assert sent_body["additionalModelRequestFields"] == {"reasoning": {"effort": "medium"}}
    assert sent_body["toolConfig"]["tools"] == [
        {"toolSpec": {"name": "bash", "description": "Run a command.", "inputSchema": {"json": BASH["parameters"]}}},
        {"toolSpec": {"name": "read", "inputSchema": {"json": {"type": "object"}}}}]
    assert "temperature" not in request.content.decode() and "system" not in sent_body
    assert completion.provider == "bedrock" and completion.model == AZURE_LUNA6_MODEL
    assert completion.effective_sampling == {"temperature": None, "effort": "medium", "dropped": ["temperature"]}
    assert completion.provenance == {
        "route": "bedrock", "endpoint": ENDPOINT + "/", "deployment": "global.openai.gpt-6-luna",
        "deployment_type": "GlobalStandard", "served_model": AZURE_LUNA6_MODEL, "request_id": "req-test",
        "throttled_waits": "0"}
    assert llm.azure_route_for(AZURE_LUNA6_MODEL) == {key: completion.provenance[key] for key in (
        "route", "endpoint", "deployment", "deployment_type", "served_model")}
    # Converse's buckets are disjoint already, and priced as the table says.
    usage = completion.usage
    assert (usage.input_tokens, usage.cache_read_tokens, usage.cache_write_tokens, usage.output_tokens) == (
        12, 1000, 200, 5)
    billed, _ = call_cost(AZURE_LUNA6_MODEL, input_tokens=12, output_tokens=5, cache_read_tokens=1000,
                          cache_write_tokens=200)
    assert llm.stats.total_cost_usd == pytest.approx(billed)


def test_a_hosted_agents_responses_items_become_alternating_converse_turns(wire):
    # What mini-swe-agent sends after two steps on the Azure route: its
    # system and task messages, then each reply's items (reasoning encrypted
    # for Azure, a message, a call) and each call's output.
    sent = wire(ok())
    messages = [
        {"role": "system", "content": "You are a careful engineer."},
        {"role": "user", "content": "Please solve this issue."},
        native({"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAA-azure-only", "summary": []}),
        native({"type": "message", "role": "assistant", "id": "msg_1",
                "content": [{"type": "output_text", "text": "Looking first.", "annotations": []}]}),
        native({"type": "function_call", "call_id": "call_1", "name": "bash", "arguments": '{"command": "ls"}'}),
        native({"type": "function_call_output", "call_id": "call_1", "output": "<returncode>0</returncode>\nsrc"}),
        native({"type": "reasoning", "id": "rs_2", "encrypted_content": "gAAAA-azure-only", "summary": []}),
        native({"type": "function_call", "call_id": "call_2", "name": "bash", "arguments": '{"command": "cat a"}'}),
        {"role": "user", "content": "Note: keep going."},
        native({"type": "function_call_output", "call_id": "call_2", "output": "<returncode>1</returncode>\n"}),
    ]
    completion = ask(LLM(azure_openai=route()), messages=messages)
    sent_body = body(sent[0])
    assert sent_body["system"] == [{"text": "You are a careful engineer."}]
    assert sent_body["messages"] == [
        {"role": "user", "content": [{"text": "Please solve this issue."}]},
        {"role": "assistant", "content": [
            {"text": "Looking first."},
            {"toolUse": {"toolUseId": "call_1", "name": "bash", "input": {"command": "ls"}}}]},
        {"role": "user", "content": [
            {"toolResult": {"toolUseId": "call_1", "content": [{"text": "<returncode>0</returncode>\nsrc"}]}}]},
        {"role": "assistant", "content": [
            {"toolUse": {"toolUseId": "call_2", "name": "bash", "input": {"command": "cat a"}}}]},
        # The tool's result first, as Claude requires; then the text after it.
        {"role": "user", "content": [
            {"toolResult": {"toolUseId": "call_2", "content": [{"text": "<returncode>1</returncode>\n"}]}},
            {"text": "Note: keep going."}]},
    ]
    assert "gAAAA" not in sent[0].content.decode()
    assert completion.effective_sampling["reasoning_items_not_sent"] == 2


def test_a_command_that_printed_nothing_is_sent_as_empty_text_not_as_no_content(wire):
    # GPT-6 Luna refuses a tool result with no content; a command such as
    # `cd` or `touch` prints nothing, and its step must still go through.
    sent = wire(ok())
    ask(LLM(azure_openai=route()), messages=[
        {"role": "user", "content": "Please solve this issue."},
        native({"type": "function_call", "call_id": "call_1", "name": "bash", "arguments": '{"command": "touch a"}'}),
        native({"type": "function_call_output", "call_id": "call_1", "output": ""}),
    ])
    assert body(sent[0])["messages"][-1]["content"] == [{"toolResult": {"toolUseId": "call_1", "content": [{"text": ""}]}}]


@pytest.mark.parametrize("messages,why", [
    ([{"role": "user", "content": "hi"}, {"role": "system", "content": "late"}], "no system message after"),
    ([native({"type": "function_call", "call_id": "c", "name": "bash", "arguments": "{"})], "unparseable"),
    ([native({"type": "web_search_call", "id": "w"})], "no Converse form"),
    ([{"role": "assistant", "content": "I start."}], "starts with the user"),
])
def test_a_request_converse_cannot_carry_is_refused_before_it_is_sent(wire, messages, why):
    sent = wire(ok())
    with pytest.raises(ProtocolFailure, match=why):
        ask(LLM(azure_openai=route()), messages=messages)
    assert sent == []


def test_claude_reads_the_cache_its_last_step_wrote_and_is_sent_its_effort_with_adaptive_thinking(wire):
    """Unpinned, Sonnet 5.5 thinks adaptively at its default (high) effort while the record would say
    none: the effort asked for is sent, with adaptive thinking, and recorded. Its thinking blocks are
    never sent back (as on Luna's Bedrock route, where reasoning is not carried between turns)."""
    sent = wire(ok())
    completion = ask(LLM(azure_openai=route()), model=SONNET, system="A long, stable system prompt.",
                     messages=[{"role": "user", "content": "first"}, {"role": "assistant", "content": "ok"},
                               {"role": "user", "content": "second"}], temperature=1.0)
    sent_body = body(sent[0])
    assert sent[0].url == ENDPOINT + "/model/global.anthropic.claude-sonnet-5-5/converse"
    assert sent_body["system"] == [{"text": "A long, stable system prompt."}, {"cachePoint": {"type": "default"}}]
    assert sent_body["messages"][-1] == {"role": "user", "content": [
        {"text": "second"}, {"cachePoint": {"type": "default"}}]}
    assert sent[0].content.decode().count("cachePoint") == 2
    assert sent_body["additionalModelRequestFields"] == {"thinking": {"type": "adaptive"},
                                                          "output_config": {"effort": "medium"}}
    assert "temperature" not in sent[0].content.decode()
    assert completion.effective_sampling == {"temperature": None, "effort": "medium", "thinking": "adaptive",
                                             "thinking_carried": False, "dropped": ["temperature"]}


def test_claudes_thinking_is_left_out_of_the_reply_it_sends_back(wire):
    sent = wire(ok({"reasoningContent": {"reasoningText": {"text": "Let me look.", "signature": "c2ln"}}},
                   {"text": "I will list the files."},
                   {"toolUse": {"toolUseId": "t1", "name": "bash", "input": {"command": "ls"}}},
                   stop="tool_use"))
    completion = ask(LLM(azure_openai=route()), model=SONNET)
    assert "Let me look." not in json.dumps(completion.transcript_blocks) and "c2ln" not in json.dumps(completion.transcript_blocks)
    assert [call.name for call in completion.tool_calls] == ["bash"]


# ---------------------------------------------------------------- the reply


def test_a_reply_comes_back_as_the_responses_items_a_hosted_agent_reads(wire):
    sent = wire(ok({"text": "I will list the files."},
                   {"toolUse": {"toolUseId": "tooluse_1", "name": "bash", "input": {"command": "ls"}}},
                   stop="tool_use"))
    llm = LLM(azure_openai=route())
    completion = ask(llm)
    [call] = completion.tool_calls
    assert (call.id, call.name, call.arguments) == ("tooluse_1", "bash", {"command": "ls"})
    assert completion.stop_reason == "tool_use"
    reply = _reply(completion, 0.001)
    assert reply.status == "completed" and reply.incomplete_reason is None
    assert [dict(item) for item in reply.output] == [
        {"type": "message", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "I will list the files.", "annotations": []}]},
        {"type": "function_call", "call_id": "tooluse_1", "name": "bash", "arguments": '{"command": "ls"}',
         "status": "completed"}]
    # Sent back as the agent sends them, they are the same turn again.
    again = [{"role": "user", "content": "go"}, *(native(dict(item)) for item in reply.output),
             native({"type": "function_call_output", "call_id": "tooluse_1", "output": "src"})]
    ask(llm, messages=again)
    assert body(sent[1])["messages"][1] == {"role": "assistant", "content": [
        {"text": "I will list the files."},
        {"toolUse": {"toolUseId": "tooluse_1", "name": "bash", "input": {"command": "ls"}}}]}


@pytest.mark.parametrize("stop", ["max_tokens", "content_filtered", "model_context_window_exceeded"])
def test_a_reply_cut_short_hands_over_no_tool_call(wire, stop):
    wire(ok({"text": "Halfway"}, {"toolUse": {"toolUseId": "t", "name": "bash", "input": {"command": "rm"}}},
            stop=stop))
    completion = ask(LLM(azure_openai=route()))
    assert completion.tool_calls == () and completion.text_blocks == ("Halfway",)
    assert completion.transcript_blocks == ({"type": "text", "text": "Halfway"},)
    assert _reply(completion, 0.0).output == ()


@pytest.mark.parametrize("usage", [
    {"inputTokens": 1, "outputTokens": 1, "cacheWriteInputTokens": 0},
    {"inputTokens": 1, "outputTokens": -1, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0},
    {"inputTokens": "1", "outputTokens": 1, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0},
    {"inputTokens": 1, "outputTokens": 1, "cacheReadInputTokens": 5, "cacheWriteInputTokens": 0, "totalTokens": 2},
])
def test_usage_must_report_every_bucket_and_add_up(wire, usage):
    sent = wire(ok(usage=usage))
    cap = max_call_cost_usd(AZURE_LUNA6_MODEL, max_output_tokens=256)
    llm = LLM(azure_openai=route(), budget_usd=cap, cap_on="billed")
    with pytest.raises(ProtocolFailure, match="usage"):
        ask(llm)
    # A reply the ledger cannot price leaves the call unsettled, as on Azure.
    with pytest.raises(ProtocolFailure, match="unsettled"):
        ask(llm)
    assert len(sent) == 1


# ---------------------------------------------------------------- waits and failures


def throttled(_request):
    return httpx.Response(429, json={"message": "Too many requests, please wait before trying again."},
                          headers={"x-amzn-errortype": "ThrottlingException:http://internal.amazon.com/coral/"})


def test_a_throttled_request_is_waited_out_and_its_waits_stay_on_the_record(wire, monkeypatch):
    waits = []
    monkeypatch.setattr("taste.providers._bedrock.time.sleep", waits.append)
    refusals = iter([True, True])
    sent = wire(lambda request: throttled(request) if next(refusals, False) else ok()(request))
    llm = LLM(azure_openai=route())
    completion = ask(llm)
    assert waits == [2.0, 4.0] and len(sent) == 3
    assert len({request.content for request in sent}) == 1, "the request sent again is the same request"
    assert completion.provenance["throttled_waits"] == "2"
    moments = [float(value) for value in completion.provenance["throttled_at_unix"].split(",")]
    assert len(moments) == 2 and all(abs(moment - time.time()) < 60 for moment in moments)
    assert llm.provider_for(AZURE_LUNA6_MODEL).throttled == 2


def test_throttling_that_outlasts_the_waits_fails_and_fences_the_facade(wire, monkeypatch):
    monkeypatch.setattr("taste.providers._bedrock.time.sleep", lambda _seconds: None)
    sent = wire(throttled)
    llm = LLM(azure_openai=route())
    with pytest.raises(InfraFailure) as raised:
        ask(llm)
    assert len(sent) == 6 and raised.value.transient
    with pytest.raises(ProtocolFailure, match="unsettled"):
        ask(llm)


def test_a_request_that_never_left_this_machine_is_sent_again(wire, monkeypatch):
    waits = []
    monkeypatch.setattr("taste.providers._bedrock.time.sleep", waits.append)
    failures = iter([httpx.ConnectError("no connection")])

    def handler(request):
        failure = next(failures, None)
        if failure is not None:
            raise failure
        return ok()(request)

    sent = wire(handler)
    llm = LLM(azure_openai=route())
    assert ask(llm).provenance["throttled_waits"] == "0"
    assert len(sent) == 2 and waits == [2.0]
    assert llm.provider_for(AZURE_LUNA6_MODEL).unconnected == 1


@pytest.mark.parametrize("failure,transient", [
    ("read_timeout", True), ("dropped", True), ("500", True), ("503", True),
    ("400", False), ("403", False), ("307", False),
])
def test_a_request_that_may_have_arrived_is_never_sent_again(wire, monkeypatch, failure, transient):
    monkeypatch.setattr("taste.providers._bedrock.time.sleep", lambda _seconds: None)

    def handler(request):
        if failure == "read_timeout":
            raise httpx.ReadTimeout("no answer", request=request)
        if failure == "dropped":
            raise httpx.RemoteProtocolError("server disconnected", request=request)
        return httpx.Response(int(failure), json={"message": "refused"},
                              headers={"x-amzn-errortype": "ValidationException",
                                       "location": "https://elsewhere.invalid/"})

    sent = wire(handler)
    llm = LLM(azure_openai=route())
    with pytest.raises(InfraFailure) as raised:
        ask(llm)
    assert len(sent) == 1 and raised.value.transient is transient
    assert KEY not in str(raised.value) and KEY not in repr(raised.value.last_error)


# ---------------------------------------------------------------- the journal


def test_a_bedrock_journal_keeps_its_receipts_across_a_restart(tmp_path, wire):
    sent = wire(ok({"text": "Hello."}))
    limits = ResponsesBinding(run_id="worker-run.bedrock", model=AZURE_LUNA6_MODEL, endpoint=route().base_url,
                              deployment="global.openai.gpt-6-luna", budget_usd=5, max_calls=4,
                              max_output_tokens=256, deadline_unix=time.time() + 60)

    async def call(session):
        return await session.complete("turn-1", system="", messages=[{"role": "user", "content": "hi"}],
                                      tools=[BASH], effort="medium")

    first = ResponsesSession.create(tmp_path / "session", limits, route())
    result = asyncio.run(call(first))
    first.close()
    resumed = ResponsesSession.open(tmp_path / "session", limits, route())
    try:
        assert asyncio.run(call(resumed)) == result and len(sent) == 1
        assert result.provider == "bedrock" and not resumed.unsettled
        assert resumed.known_cost_usd == pytest.approx(call_cost(
            AZURE_LUNA6_MODEL, input_tokens=12, output_tokens=5, cache_read_tokens=1000,
            cache_write_tokens=200)[0])
    finally:
        resumed.close()


# ---------------------------------------------------------------- a branch of an Azure run


def azure_reply(command, index):
    """A recorded Azure reply: an encrypted reasoning item, a message and a call."""
    return ModelReply(output=(
        {"type": "reasoning", "id": f"rs_{index}", "encrypted_content": "gAAAA-azure-only", "summary": []},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Next step."}]},
        {"type": "function_call", "call_id": f"call_{index}", "name": "bash",
         "arguments": json.dumps({"command": command})}),
        status="completed", incomplete_reason=None, usage={"input_tokens": 100, "output_tokens": 20},
        model=AZURE_LUNA6_MODEL, cost_usd=0.25)


class BedrockLive:
    """The live side of a branch on the Bedrock route, as a hosted worker asks: the agent's items as native blocks."""

    def __init__(self, llm):
        self.llm, self.asked, self.ran = llm, [], []

    def ask(self, *, messages, tools, effort):
        self.asked.append(request_sha(messages, tools, effort))
        canonical = [message if set(message) == {"role", "content"} and isinstance(message["content"], str)
                     else native(dict(message)) for message in messages]
        completion = self.llm.call(model=AZURE_LUNA6_MODEL, system="", messages=canonical, tools=tools,
                                   max_tokens=256, effort=effort, temperature=None, timeout_seconds=600)
        return _reply(completion, 0.0)

    def run(self, command, *, cwd, timeout_seconds, shown=None):
        self.ran.append(shown)
        output, code = OUTPUTS[shown]
        return ShellResult(output, code)


def test_a_branch_of_an_azure_run_replays_its_record_and_goes_live_on_bedrock(wire):
    # The digest a branch checks is the agent's own request, before any
    # translation: the record of an Azure run matches whichever route the
    # branch then goes live on, and a request that differs still stops it.
    replies = [azure_reply("make test", 0), azure_reply("sed -i 's/x/y/' parser.py && make test", 1),
               azure_reply(f"echo {SENTINEL}", 2)]
    recorder = Recorder(Live(replies, OUTPUTS))
    assert Agent().run(recorder)[0] == "Submitted"
    script = recorder.script()
    sent = wire(ok({"toolUse": {"toolUseId": "tooluse_9", "name": "bash", "input": {"command": f"echo {SENTINEL}"}}},
                   stop="tool_use"))
    live, events = BedrockLive(LLM(azure_openai=route())), []
    host = ReplayHost(live, script, 2, record=lambda kind, payload: events.append((kind, payload)))
    status, _ = Agent().run(host)
    assert status == "Submitted" and host.faithful and host.replayed == 2
    assert live.asked == [script.steps[2].request_sha]
    assert events[-1][1] == {"live": True, "context": "matched"}
    [request] = sent
    turns = body(request)["messages"]
    assert [turn["role"] for turn in turns] == ["user", "assistant", "user", "assistant", "user"]
    assert [block["toolUse"]["input"]["command"] for turn in turns for block in turn["content"]
            if "toolUse" in block] == ["make test", "sed -i 's/x/y/' parser.py && make test"]
    assert "gAAAA" not in request.content.decode() and live.ran == [f"echo {SENTINEL}"]
    # The fidelity check is not loosened: a different task is still a different request.
    other = BedrockLive(LLM(azure_openai=route()))
    stopped = ReplayHost(other, script, 2, record=lambda kind, payload: None)
    with pytest.raises(HostedStop):
        Agent(task="Another task.").run(stopped)
    assert other.asked == [] and len(sent) == 1


# ---------------------------------------------------------------- trial settings and prices


def test_a_host_names_exactly_one_model_route():
    azure = {"AZURE_OPENAI_BASE_URL": "https://r.openai.azure.com/openai/v1", "AZURE_OPENAI_API_KEY": "a"}
    bedrock = {"TASTE_BEDROCK_URL": ENDPOINT, "AWS_BEARER_TOKEN_BEDROCK": KEY}
    assert model_route(azure) == ("https://r.openai.azure.com/openai/v1/", "a")
    assert model_route(bedrock) == (ENDPOINT + "/", KEY)
    for environment in ({**azure, **bedrock}, {}, {"TASTE_BEDROCK_URL": ENDPOINT}):
        with pytest.raises(RuntimeError) as raised:
            model_route(environment)
        assert KEY not in str(raised.value)


def test_an_agent_run_alone_on_bedrock_uses_its_models_global_profile():
    for model, profile in (("gpt-6-luna", "global.openai.gpt-6-luna"),
                           (SONNET, "global.anthropic.claude-sonnet-5-5")):
        settings = TrialSettings.from_options({"model": model, "agent": "mini-swe-agent", "services": "none",
                                               "worker_effort": "medium"})
        policy = settings.policy(ENDPOINT + "/", time.time() + 1500, owner_token="a" * 32,
                                 container_id="c" * 64, workdir="/app")
        assert (policy.planner_deployment, policy.worker_deployment) == ("fixed-plan", profile)
        assert policy.endpoint == ENDPOINT + "/"
        config = policy.azure_config({"AZURE_OPENAI_BASE_URL": policy.endpoint, "AZURE_OPENAI_API_KEY": KEY})
        assert config.route == "bedrock"


def test_bedrock_prices_are_its_global_list_prices():
    for model, rates in (("claude-sonnet-5-5", (2.00, 10.00, 0.10, 2.50)),
                         ("claude-opus-5-5", (4.00, 20.00, 0.20, 5.00)),
                         ("claude-haiku-5-5", (0.10, 0.50, 0.01, 0.125))):
        price = PRICES[model]
        assert price.provider == "bedrock" and price.context_window == 200_000 and len(price.tiers) == 1
        r = price.tiers[0][1]
        assert (r.input, r.output, r.cache_read, r.cache_write) == rates
        assert r.cache_write == pytest.approx(r.input * 1.25), model
    # Luna and Sol cost on Bedrock's global profiles what they cost on Azure.
    for model, short in ((AZURE_LUNA6_MODEL, (0.10, 0.50, 0.01, 0.125)), (AZURE_WORKER_MODEL, (2.0, 10.0, 0.2, 2.5))):
        r = PRICES[model].tiers[0][1]
        assert (r.input, r.output, r.cache_read, r.cache_write) == short
