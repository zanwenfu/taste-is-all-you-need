"""Responses wire contracts through the real SDK, without paid requests.

The transport supplies HTTP payloads, not fake SDK response classes. These
tests cover the adapter -> SDK -> transcript -> next request and cost seams.
"""

from __future__ import annotations

import copy
import json

import pytest

from taste.llm import LLM
from taste.pricing import call_cost, max_call_cost_usd
from taste.providers._openai import OpenAIProvider
from taste.providers.base import CompletionRequest, ProtocolFailure, SamplingConfig

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")

MODEL = "gpt-5.6-terra"
TOOL = {
    "name": "read_artifact",
    "description": "Read a named artifact.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}},
        "required": ["path"],
    },
}


def response(*, output=None, status="completed", usage=None, **extra):
    return {
        "id": "resp_test", "object": "response", "created_at": 1,
        "model": MODEL, "status": status,
        "output": output if output is not None else [message("done")],
        "usage": usage if usage is not None else {
            "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
            "input_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 40},
            "output_tokens_details": {"reasoning_tokens": 10},
        },
        **extra,
    }


def message(text, *, phase="final_answer"):
    return {
        "type": "message", "id": "msg_test", "role": "assistant",
        "status": "completed", "phase": phase,
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def function_call(arguments='{"path":"result.txt"}', **extra):
    return {
        "type": "function_call", "id": "fc_test", "call_id": "call_test",
        "name": "read_artifact", "arguments": arguments, "status": "completed", **extra,
    }


def request(**overrides):
    return CompletionRequest(**{
        "model": MODEL, "system": [{"type": "text", "text": "Be precise."}],
        "messages": [{"role": "user", "content": "Read the result."}],
        "tools": [TOOL], "max_tokens": 128,
        "sampling": SamplingConfig(effort="low"), "role": "worker", **overrides,
    })


@pytest.fixture
def transport():
    clients = []

    def build(*payloads):
        pending = iter(payloads)
        sent = []

        def handle(wire):
            sent.append(json.loads(wire.content))
            return httpx.Response(200, json=next(pending))

        provider = OpenAIProvider(api_key="test-only-key")
        provider._client = openai.OpenAI(
            api_key="test-only-key", base_url="https://api.test.invalid/v1",
            max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        )
        clients.append(provider._client)
        return provider, sent

    yield build
    for client in clients:
        client.close()


def test_output_limit_matches_the_facade_budget_reservation(transport):
    provider, sent = transport(response())
    cap = max_call_cost_usd(MODEL, max_output_tokens=128, max_attempts=1)
    llm = LLM(load_env_file=False, budget_usd=cap, cap_on="billed", max_attempts=1)
    llm._providers["openai"] = provider
    llm.call(model=MODEL, system="Short answer.", messages=[], max_tokens=128)
    assert sent[0]["max_output_tokens"] == 128
    assert llm.stats.totals.calls == 1


def test_stateless_reasoning_and_tool_results_survive_json_round_trip(transport):
    native = [
        {"type": "reasoning", "id": "rs_test", "summary": [], "encrypted_content": "opaque"},
        message("I will read it.", phase="commentary"), function_call(),
    ]
    provider, sent = transport(response(output=native), response())
    first = provider.complete(request())
    restored = json.loads(json.dumps(first.transcript_blocks))
    followup = request(messages=[
        *request().messages, {"role": "assistant", "content": restored},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_test", "content": "42"},
        ]},
    ])
    provider.complete(followup)
    assert sent[0]["store"] is False
    assert "reasoning.encrypted_content" in sent[0]["include"]
    assert sent[1]["input"] == [
        *request().messages, *native,
        {"type": "function_call_output", "call_id": "call_test", "output": "42"},
    ]
    assert first.summary_text == "I will read it."
    assert len(first.tool_calls) == 1


def test_existing_paired_text_transcripts_do_not_duplicate_assistant_messages(transport):
    provider, sent = transport(response())
    provider.complete(request(messages=[{"role": "assistant", "content": [
        {"type": "text", "text": "previous"},
        {"type": "_openai_item", "item": message("previous")},
        {"type": "text", "text": "new annotation"},
    ]}]))
    assert sent[0]["input"] == [message("previous"), {"role": "assistant", "content": "new annotation"}]


def test_optional_tool_fields_remain_optional(transport):
    provider, sent = transport(response())
    provider.complete(request())
    assert sent[0]["tools"][0]["strict"] is False
    assert sent[0]["tools"][0]["parameters"] == TOOL["input_schema"]


def test_cache_writes_are_disjoint_and_billed_at_the_write_rate(transport):
    provider, _ = transport(response())
    llm = LLM(load_env_file=False, max_attempts=1)
    llm._providers["openai"] = provider
    completion = llm.call(model=MODEL, system="s", messages=[])
    usage = completion.usage
    assert (usage.input_tokens, usage.cache_read_tokens, usage.cache_write_tokens) == (30, 30, 40)
    assert usage.prompt_total == 100
    billed, work = call_cost(MODEL, input_tokens=30, output_tokens=20,
                             cache_read_tokens=30, cache_write_tokens=40)
    assert llm.stats.total_cost_usd == pytest.approx(billed)
    assert llm.stats.total_work_usd == pytest.approx(work)


@pytest.mark.parametrize(("bucket", "name", "value"), [
    (None, "input_tokens", -1), (None, "output_tokens", -1),
    (None, "input_tokens", True), (None, "input_tokens", 1.5),
    (None, "output_tokens", "20"), (None, "input_tokens", None),
    ("input_tokens_details", "cached_tokens", -1),
    ("input_tokens_details", "cached_tokens", 101),
    ("input_tokens_details", "cache_write_tokens", 80),
    ("input_tokens_details", "cache_write_tokens", True),
    ("output_tokens_details", "reasoning_tokens", 21),
    ("output_tokens_details", "reasoning_tokens", -1),
])
def test_invalid_usage_never_becomes_a_successful_accounted_call(transport, bucket, name, value):
    payload = response()
    target = payload["usage"] if bucket is None else payload["usage"][bucket]
    target[name] = value
    provider, _ = transport(payload)
    with pytest.raises(ProtocolFailure):
        provider.complete(request())


@pytest.mark.parametrize("missing", ["input_tokens", "output_tokens", "input_tokens_details", "cache_write_tokens"])
def test_missing_billable_usage_is_not_assumed_free(transport, missing):
    payload = response()
    target = payload["usage"]["input_tokens_details"] if missing == "cache_write_tokens" else payload["usage"]
    del target[missing]
    provider, _ = transport(payload)
    with pytest.raises(ProtocolFailure):
        provider.complete(request())


@pytest.mark.parametrize("arguments", ["[]", "null", "7", '"text"', '{"offset":NaN}', '{"offset":1e999}', ""])
def test_tool_arguments_must_be_a_finite_json_object(transport, arguments):
    provider, _ = transport(response(output=[function_call(arguments)]))
    with pytest.raises(ProtocolFailure):
        provider.complete(request())


@pytest.mark.parametrize("calls", [
    [function_call(call_id="")], [function_call(name="")],
    [function_call(), function_call(id="fc_second")],
])
def test_tool_call_identity_is_complete_and_unambiguous(transport, calls):
    provider, _ = transport(response(output=calls))
    with pytest.raises(ProtocolFailure):
        provider.complete(request())


@pytest.mark.parametrize("reason", ["max_output_tokens", "content_filter"])
def test_incomplete_response_never_exposes_executable_tool_calls(transport, reason):
    provider, _ = transport(response(status="incomplete", output=[function_call()],
                                    incomplete_details={"reason": reason}))
    result = provider.complete(request())
    assert result.stop_reason == ("max_tokens" if reason == "max_output_tokens" else "content_filter")
    assert result.tool_calls == ()


def test_refusal_is_not_reported_as_success(transport):
    item = copy.deepcopy(message("unused"))
    item["content"] = [{"type": "refusal", "refusal": "Cannot do this."}]
    provider, _ = transport(response(output=[item]))
    assert provider.complete(request()).stop_reason == "refusal"


def test_zero_usage_is_valid_when_explicit(transport):
    payload = response()
    payload["usage"] = {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    provider, _ = transport(payload)
    result = provider.complete(request())
    assert result.usage.prompt_total == result.usage.output_tokens == 0


@pytest.mark.parametrize("status", ["failed", "cancelled", "queued", "in_progress"])
def test_unsuccessful_response_never_exposes_executable_tool_calls(transport, status):
    provider, _ = transport(response(status=status, output=[function_call()]))
    result = provider.complete(request())
    assert result.stop_reason == "other"
    assert result.tool_calls == ()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_output_limit_is_rejected_before_any_http(transport, limit):
    provider, sent = transport()
    with pytest.raises(ProtocolFailure):
        provider.complete(request(max_tokens=limit))
    assert sent == []
