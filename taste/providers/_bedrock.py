"""Amazon Bedrock adapter, on the Converse API.

**Why Converse.** On Bedrock the GPT-6 models refuse the OpenAI-shaped
Responses API (``bedrock-mantle``: "does not support the '/v1/responses'
API"), and its Chat Completions route refuses function tools with a reasoning
effort, as OpenAI's own does. Converse takes both: tools, and the effort as
``additionalModelRequestFields={"reasoning": {"effort": ...}}``. Claude models
take the same request. Found by calling the real service on 2026-10-16.

**The harness side stays Responses-shaped.** Hosted agents send and read
Responses items (``_openai_item`` blocks; see ``taste.agents``), so this
adapter writes each reply as those items too: its text as one assistant
``message`` item, each tool call as a ``function_call`` item beside its
canonical ``tool_use`` block. Sent back, they become Converse turns again. An
agent's request digest is computed on its own messages, before any
translation, so a branch replays the same requests whichever route serves it.

**No reasoning comes back.** Converse returns no reasoning content for the
GPT-6 models, so none is replayed on the next turn: each turn reasons afresh.
Reasoning items a request carries (an Azure run's, encrypted for Azure) cannot
be sent and are left out; ``effective_sampling`` counts them. This is a
difference from the Azure route, which replays them, and is disclosed with
results.

**Caching.** GPT-6 models cache prompt prefixes by themselves. Claude caches
only up to a ``cachePoint``: one goes after the system prompt (it covers the
tools too) and one at the end of the conversation, so each step reads the
prefix the step before it wrote. Converse's usage buckets are already disjoint:
``inputTokens`` excludes the cached reads and writes, and ``totalTokens`` is
checked against their sum.

**Waits.** As on the Azure route, a request is sent again only when that is
provably free: a 429 (the service declined to process it) or a connection
that was never made. Every other failure may hide a billed reply and is left
to the caller's settlement. Each wait is recorded on the reply's provenance
(``throttled_waits`` and when), so the throttled minutes stay on the record.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any
from urllib.parse import quote

from taste.providers._openai import _NATIVE, _resend_wait, _sending_limits
from taste.providers.azure_openai import AzureOpenAIConfig
from taste.providers.base import (
    Completion,
    CompletionRequest,
    ProtocolFailure,
    ToolCall,
    Usage,
    UsageSchemaError,
)

_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
# How many times one call is sent again after a sending that provably cost
# nothing, and the longest single wait before the next (the Azure route's).
_UNBILLED_RESENDS = 5

_STOP_REASONS = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "end_turn",
    "guardrail_intervened": "content_filter",
    "content_filtered": "content_filter",
    # Not in Converse's StopReason list (botocore 2023-09-30), but Claude's own.
    "refusal": "refusal",
    "model_context_window_exceeded": "context_window",
    "malformed_model_output": "other",
    "malformed_tool_use": "other",
}
# Claude allows four cache points in one request; this adapter writes two.
_CACHE_POINT = {"cachePoint": {"type": "default"}}


class BedrockError(RuntimeError):
    """Bedrock answered with an error status: its type and message, never the request."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.status_code = response.status_code
        self.error_type = (response.headers.get("x-amzn-errortype") or "").split(":", 1)[0]
        try:
            message = str(response.json().get("message", ""))
        except Exception:
            message = ""
        super().__init__(f"Bedrock {self.status_code} {self.error_type}: {message[:500]}")


class BedrockProvider:
    name = "bedrock"

    def __init__(self, *, route: AzureOpenAIConfig) -> None:
        if not isinstance(route, AzureOpenAIConfig) or route.route != "bedrock":
            raise ProtocolFailure("the Bedrock adapter serves only an explicit Bedrock route")
        self._route = route
        self._client: Any = None
        # Rate-limit refusals this provider waited out, and requests it sent
        # again because no connection had been made.
        self.throttled = 0
        self.unconnected = 0

    def ensure_ready(self) -> None:
        self._ensure_client()

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ProtocolFailure("the `httpx` package is not installed") from exc
            # No proxies or redirects from the environment: the request goes
            # to the admitted endpoint or nowhere.
            self._client = httpx.Client(trust_env=False, follow_redirects=False, timeout=60.0)
        return self._client

    # ------------------------------------------------------------ calls

    def complete(self, request: CompletionRequest) -> Completion:
        if type(request.max_tokens) is not int or request.max_tokens < 1:
            raise ProtocolFailure("max_tokens must be a positive integer")
        if request.sampling.json_output:
            raise ProtocolFailure("the Bedrock route has no JSON reply mode")
        binding = self._route.deployment_for(request.model)
        claude = binding.deployment.split(".")[1] == "anthropic"
        system, messages, dropped_items = _to_converse(request.system, request.messages)
        if claude:
            if system:
                system.append(dict(_CACHE_POINT))
            if messages:
                messages[-1]["content"].append(dict(_CACHE_POINT))
        body: dict[str, Any] = {"messages": messages, "inferenceConfig": {"maxTokens": request.max_tokens}}
        if system:
            body["system"] = system
        if request.tools:
            body["toolConfig"] = {"tools": [_to_tool(tool) for tool in request.tools]}
        # temperature is never sent: the GPT-6 models refuse it beside a
        # reasoning effort, Claude Opus refuses it outright, and the hosted
        # agents' sessions send none. A drop is recorded, never hidden.
        dropped = ["temperature"] if request.sampling.temperature is not None else []
        if request.sampling.effort:
            if claude:
                # Unpinned, Claude 5.x thinks adaptively at its default (high) effort; the effort asked
                # for is sent with adaptive thinking. Its thinking blocks are never sent back (the reply
                # drops them), as reasoning is not carried between turns on Luna's Bedrock route either.
                body["additionalModelRequestFields"] = {"thinking": {"type": "adaptive"},
                                                        "output_config": {"effort": request.sampling.effort}}
            else:
                body["additionalModelRequestFields"] = {"reasoning": {"effort": request.sampling.effort}}

        payload, headers, waits = self._send(binding.deployment, body, request.timeout_seconds,
                                             request.request_seconds)
        completion = self._to_completion(payload, request, dropped, dropped_items,
                                         claude_thinking=claude and bool(request.sampling.effort))
        provenance = {
            "route": "bedrock", "endpoint": self._route.base_url,
            "deployment": binding.deployment, "deployment_type": binding.deployment_type,
            # Bedrock echoes no model; the profile admitted for it pins it.
            "served_model": binding.model,
            "request_id": headers.get("x-amzn-requestid", ""),
            "throttled_waits": str(len(waits)),
        }
        if waits:
            provenance["throttled_at_unix"] = ",".join(f"{moment:.3f}" for moment in waits)
        return replace(completion, provenance=provenance)

    def _send(self, profile: str, body: dict[str, Any], timeout_seconds: float | None,
              request_seconds: float | None) -> tuple[dict[str, Any], Any, list[float]]:
        """Send one request; send it again only when the last sending provably cost nothing.

        ``timeout_seconds`` is the call's deadline, waits included;
        ``request_seconds`` bounds each sending inside it. Returns the reply,
        its headers and when each refusal that was waited out came.
        """
        import httpx

        client = self._ensure_client()
        url = f"{self._route.base_url}model/{quote(profile, safe='')}/converse"
        headers = {"authorization": f"Bearer {self._route.api_key}", "content-type": "application/json",
                   "accept": "application/json"}
        content = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        waits: list[float] = []
        for resend in range(_UNBILLED_RESENDS + 1):
            seconds = request_seconds
            if deadline is not None:
                left = max(0.001, deadline - time.monotonic())
                seconds = left if seconds is None else min(left, seconds)
            timeout = _sending_limits(seconds) if seconds is not None else httpx.Timeout(60.0, connect=30.0)
            try:
                response = client.post(url, content=content, headers=headers, timeout=timeout)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as failure:
                # No connection, no request: nothing reached the service.
                unbilled, counter = failure, "unconnected"
            else:
                if response.status_code == 429:
                    unbilled, counter = BedrockError(response), "throttled"
                elif response.status_code != 200:
                    # A redirect, a refusal or a server fault: never followed
                    # or sent again here.
                    raise BedrockError(response)
                else:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise ProtocolFailure("Converse reply is not JSON") from exc
                    if not isinstance(payload, dict):
                        raise ProtocolFailure("Converse reply must be a JSON object")
                    return payload, response.headers, waits
            pause = _resend_wait(unbilled, resend)
            if resend == _UNBILLED_RESENDS or (
                    deadline is not None and time.monotonic() + pause >= deadline):
                raise unbilled
            setattr(self, counter, getattr(self, counter) + 1)
            if counter == "throttled":
                waits.append(time.time())
            time.sleep(pause)
        raise AssertionError("unreachable")  # pragma: no cover

    def is_retryable(self, exc: Exception) -> bool:
        # As on the Azure route: a lost reply may already have been billed,
        # so whether to ask again is the caller's settled decision.
        return False

    def is_transient(self, exc: Exception) -> bool:
        """Whether the same request, asked again, could be answered (the Azure route's rule)."""
        try:
            import httpx
        except ImportError:  # pragma: no cover
            return False
        if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
            return True
        status = getattr(exc, "status_code", None)
        return isinstance(status, int) and status in _RETRYABLE_STATUS

    # ------------------------------------------------------------ response

    def _to_completion(self, payload: dict[str, Any], request: CompletionRequest, dropped: list[str],
                       dropped_items: int, *, claude_thinking: bool = False) -> Completion:
        usage = _to_usage(payload.get("usage"))
        message = (payload.get("output") or {}).get("message") or {}
        content = message.get("content")
        if message.get("role") != "assistant" or not isinstance(content, list):
            raise ProtocolFailure("Converse reply has no assistant message")
        stop = _STOP_REASONS.get(payload.get("stopReason"), "other")
        complete = stop in ("end_turn", "tool_use")
        texts: list[str] = []
        calls: list[ToolCall] = []
        transcript: list[dict[str, Any]] = []
        pending: list[str] = []

        def flush():
            if pending:
                transcript.append({"type": _NATIVE, "item": {
                    "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []} for text in pending]}})
                pending.clear()

        for block in content:
            if not isinstance(block, dict):
                raise ProtocolFailure("Converse content block must be an object")
            if "text" in block:
                if not isinstance(block["text"], str):
                    raise ProtocolFailure("Converse text must be a string")
                texts.append(block["text"])
                if block["text"]:
                    pending.append(block["text"])
            elif "toolUse" in block:
                # Even a well-formed call in a reply cut short is not an
                # authorized action; it is never handed to a tool dispatcher.
                if not complete:
                    continue
                use = block["toolUse"]
                call_id, name, arguments = use.get("toolUseId"), use.get("name"), use.get("input")
                if not isinstance(call_id, str) or not call_id or any(c.id == call_id for c in calls):
                    raise ProtocolFailure("tool call id is missing or duplicated")
                if not isinstance(name, str) or not name:
                    raise ProtocolFailure("tool name is missing")
                if not isinstance(arguments, dict):
                    raise ProtocolFailure("tool arguments must be a JSON object")
                raw_arguments = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
                call = ToolCall(id=call_id, name=name, arguments=arguments, raw_arguments=raw_arguments)
                calls.append(call)
                flush()
                transcript.append({"type": _NATIVE, "item": {
                    "type": "function_call", "call_id": call_id, "name": name,
                    "arguments": raw_arguments, "status": "completed"}})
                transcript.append({"type": "tool_use", "id": call_id, "name": name, "input": arguments})
            # Anything else (Claude's thinking blocks, which are never sent
            # back, and any unknown block) is not replayed and is left out.
        flush()
        if not complete:
            # As on the Responses route: a reply cut short is kept as the text
            # it reached, not as items to send back.
            transcript = [{"type": "text", "text": text} for text in texts if text]
        elif calls:
            stop = "tool_use"

        sampling: dict[str, Any] = {"temperature": None, "effort": request.sampling.effort}
        if "effort" in dropped:
            sampling["effort"] = None
        if claude_thinking:
            sampling.update(thinking="adaptive", thinking_carried=False)
        if dropped:
            sampling["dropped"] = dropped
        if dropped_items:
            sampling["reasoning_items_not_sent"] = dropped_items
        return Completion(
            text_blocks=tuple(texts), tool_calls=tuple(calls), stop_reason=stop,
            model=request.model, provider=self.name, usage=usage,
            transcript_blocks=tuple(transcript), effective_sampling=sampling, raw=payload,
        )


def _to_usage(usage: Any) -> Usage:
    """Converse usage -> disjoint buckets. Every counter must be there; absent is not zero."""
    if not isinstance(usage, dict):
        raise UsageSchemaError("Converse reply has no usage")
    counts = {}
    for name in ("inputTokens", "outputTokens", "cacheReadInputTokens", "cacheWriteInputTokens"):
        value = usage.get(name)
        if type(value) is not int or value < 0:
            raise UsageSchemaError(f"Converse usage {name} must be a non-negative integer")
        counts[name] = value
    total = usage.get("totalTokens")
    if total is not None and total != sum(counts.values()):
        # The buckets are disjoint only if they add up to the total; if they
        # do not, a cached token would be priced twice or not at all.
        raise UsageSchemaError("Converse usage buckets do not add up to its total")
    return Usage(
        input_tokens=counts["inputTokens"], output_tokens=counts["outputTokens"],
        cache_read_tokens=counts["cacheReadInputTokens"], cache_write_tokens=counts["cacheWriteInputTokens"],
        raw=dict(counts),
    )


# ---------------------------------------------------------------- request


def _to_converse(system_blocks: list[dict[str, Any]], messages: list[dict[str, Any]]):
    """Canonical system and messages -> Converse ``system`` and alternating ``messages``.

    Messages may hold canonical blocks (text, tool_use, tool_result) or native
    Responses items (a hosted agent's). A system or developer message before
    the conversation starts joins the system prompt; Converse has no place for
    one later. Consecutive turns of one role are merged, and in a user turn the
    tool results come first, as Claude requires. Returns the two lists and how
    many reasoning items were left out.
    """
    system = [{"text": block["text"]} for block in system_blocks if block.get("text")]
    turns: list[tuple[str, dict[str, Any]]] = []
    dropped = 0

    def say(role, block):
        if role in ("system", "developer"):
            if turns:
                raise ProtocolFailure("Converse has no system message after the conversation starts")
            if block.get("text"):
                system.append(block)
            return
        if role not in ("user", "assistant"):
            raise ProtocolFailure(f"no Converse turn for role {role!r}")
        if "text" in block and not block["text"]:
            return  # Converse refuses empty text, and it says nothing
        turns.append((role, block))

    for message in messages:
        role, content = message.get("role"), message.get("content")
        if isinstance(content, str):
            say(role, {"text": content})
            continue
        native_calls = {block["item"].get("call_id") for block in content or []
                        if block.get("type") == _NATIVE and block["item"].get("type") == "function_call"}
        for block in content or []:
            kind = block.get("type")
            if kind == _NATIVE:
                dropped += _native(block["item"], say)
            elif kind == "text":
                say(role, {"text": block["text"]})
            elif kind == "tool_use":
                # A tool_use travelling beside its native twin is that same call.
                if block["id"] not in native_calls:
                    say("assistant", {"toolUse": {"toolUseId": block["id"], "name": block["name"],
                                                  "input": block.get("input") or {}}})
            elif kind == "tool_result":
                say("user", _tool_result(block["tool_use_id"], block.get("content", "")))
            else:
                raise ProtocolFailure(f"no Converse form for a {kind!r} block")

    merged: list[dict[str, Any]] = []
    for role, block in turns:
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"].append(block)
        else:
            merged.append({"role": role, "content": [block]})
    for turn in merged:
        if turn["role"] == "user":
            turn["content"].sort(key=lambda block: "toolResult" not in block)
    if merged and merged[0]["role"] != "user":
        raise ProtocolFailure("a Converse conversation starts with the user")
    return system, merged, dropped


def _native(item: dict[str, Any], say) -> int:
    """One Responses item as Converse blocks; 1 if it is reasoning, which is left out."""
    kind = item.get("type")
    if kind == "reasoning":
        return 1
    if kind == "function_call":
        try:
            arguments = json.loads(item.get("arguments") or "{}")
        except ValueError as exc:
            raise ProtocolFailure("a recorded tool call has unparseable arguments") from exc
        if not isinstance(arguments, dict):
            raise ProtocolFailure("a recorded tool call's arguments are not an object")
        say("assistant", {"toolUse": {"toolUseId": item["call_id"], "name": item["name"], "input": arguments}})
    elif kind == "function_call_output":
        say("user", _tool_result(item["call_id"], item.get("output", "")))
    elif kind in (None, "message"):
        content = item.get("content")
        if isinstance(content, str):
            say(item.get("role"), {"text": content})
        else:
            for part in content or []:
                if part.get("type") in ("output_text", "input_text", "text"):
                    say(item.get("role"), {"text": part.get("text", "")})
                elif part.get("type") != "refusal":
                    raise ProtocolFailure(f"no Converse form for a {part.get('type')!r} part")
    else:
        raise ProtocolFailure(f"no Converse form for a {kind!r} item")
    return 0


def _tool_result(call_id: str, output: Any) -> dict[str, Any]:
    # An empty output stays one empty text block: GPT-6 Luna refuses a result
    # with no content ("Invalid 'input'"), and both it and Claude take empty
    # text (live calls, 2026-10-16).
    text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
    return {"toolResult": {"toolUseId": call_id, "content": [{"text": text}]}}


def _to_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """A canonical or Responses function tool -> a Converse tool specification."""
    if tool.get("type") == "function" and "parameters" in tool:
        name, description, schema = tool["name"], tool.get("description", ""), tool["parameters"]
    else:
        name, description = tool["name"], tool.get("description", "")
        schema = tool.get("input_schema", {"type": "object"})
    spec: dict[str, Any] = {"name": name, "inputSchema": {"json": schema}}
    if description:
        spec["description"] = description
    return {"toolSpec": spec}
