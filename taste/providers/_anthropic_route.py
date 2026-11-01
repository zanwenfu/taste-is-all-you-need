"""Anthropic route adapter: Claude through Anthropic's own Messages API, with the Bedrock route's rules.

**Same shape as the Bedrock route.** Hosted agents send and read Responses
items; this adapter turns them into Messages turns and writes each reply back as
those items (its text as one assistant ``message`` item, each tool call as a
``function_call`` item beside its canonical ``tool_use`` block), exactly as the
Converse adapter does, so an agent's request digest and its branches do not
depend on which route served it.

**Effort and thinking.** The effort asked for is sent as
``output_config.effort`` with adaptive thinking, the fields the Bedrock route
passes through to the same models. Thinking blocks are never sent back (the
reply drops them), as on the Bedrock route: each turn thinks afresh.

**Caching.** Two cache breakpoints, as the Bedrock route writes them: after the
system prompt (which covers the tools) and on the last block of the
conversation, so each step reads the prefix the step before it wrote. The
Messages API's usage buckets are disjoint: ``input_tokens`` excludes the cached
reads (``cache_read_input_tokens``) and writes (``cache_creation_input_tokens``).
Every bucket must be reported; absent is never taken as zero.

**Waits.** A request is sent again only when that is provably free: a 429 (rate
limited) or a 529 (overloaded), both declined before processing, or a
connection that was never made. Every other failure may hide a billed reply and
is left to the caller's settlement. Each wait is recorded on the reply's
provenance (``throttled_waits`` and when).
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any

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

API_VERSION = "2023-06-01"
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})
_DECLINED_STATUS = frozenset({429, 529})
_UNBILLED_RESENDS = 5
_STOP_REASONS = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "end_turn",
    "refusal": "refusal",
    "model_context_window_exceeded": "context_window",
    # A server-side tool paused the turn; this route sends no server tools.
    "pause_turn": "other",
}
_CACHE = {"type": "ephemeral"}


class AnthropicRouteError(RuntimeError):
    """Anthropic answered with an error status: its type and message, never the request."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.status_code = response.status_code
        try:
            error = response.json().get("error") or {}
            self.error_type, message = str(error.get("type", "")), str(error.get("message", ""))
        except Exception:
            self.error_type, message = "", ""
        super().__init__(f"Anthropic {self.status_code} {self.error_type}: {message[:500]}")


class AnthropicRouteProvider:
    name = "anthropic_api"

    def __init__(self, *, route: AzureOpenAIConfig) -> None:
        if not isinstance(route, AzureOpenAIConfig) or route.route != "anthropic":
            raise ProtocolFailure("the Anthropic adapter serves only an explicit Anthropic route")
        self._route = route
        self._client: Any = None
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
            raise ProtocolFailure("the Anthropic route has no JSON reply mode")
        binding = self._route.deployment_for(request.model)
        system, messages, dropped_items = _to_messages(request.system, request.messages)
        if system:
            system[-1]["cache_control"] = dict(_CACHE)
        if messages:
            messages[-1]["content"][-1]["cache_control"] = dict(_CACHE)
        body: dict[str, Any] = {"model": binding.deployment, "max_tokens": request.max_tokens, "messages": messages}
        if system:
            body["system"] = system
        if request.tools:
            body["tools"] = [_to_tool(tool) for tool in request.tools]
        # temperature is never sent, as on the Bedrock route; a drop is recorded, never hidden.
        dropped = ["temperature"] if request.sampling.temperature is not None else []
        if request.sampling.effort:
            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": request.sampling.effort}

        payload, headers, waits = self._send(body, request.timeout_seconds, request.request_seconds)
        served = payload.get("model")
        if served != binding.deployment:
            raise ProtocolFailure("the reply was served by another model than the one admitted")
        completion = self._to_completion(payload, request, dropped, dropped_items,
                                         thinking=bool(request.sampling.effort))
        provenance = {
            "route": "anthropic", "endpoint": self._route.base_url,
            "deployment": binding.deployment, "deployment_type": binding.deployment_type,
            "served_model": binding.model,
            "request_id": headers.get("request-id", ""),
            "throttled_waits": str(len(waits)),
        }
        if waits:
            provenance["throttled_at_unix"] = ",".join(f"{moment:.3f}" for moment in waits)
        return replace(completion, provenance=provenance)

    def _send(self, body: dict[str, Any], timeout_seconds: float | None,
              request_seconds: float | None) -> tuple[dict[str, Any], Any, list[float]]:
        """Send one request; send it again only when the last sending provably cost nothing."""
        import httpx

        client = self._ensure_client()
        url = f"{self._route.base_url}v1/messages"
        headers = {"x-api-key": self._route.api_key, "anthropic-version": API_VERSION,
                   "content-type": "application/json", "accept": "application/json"}
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
                unbilled, counter = failure, "unconnected"
            else:
                if response.status_code in _DECLINED_STATUS:
                    unbilled, counter = AnthropicRouteError(response), "throttled"
                elif response.status_code != 200:
                    raise AnthropicRouteError(response)
                else:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise ProtocolFailure("Messages reply is not JSON") from exc
                    if not isinstance(payload, dict):
                        raise ProtocolFailure("Messages reply must be a JSON object")
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
        # A lost reply may already have been billed: whether to ask again is the caller's settled decision.
        return False

    def is_transient(self, exc: Exception) -> bool:
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
                       dropped_items: int, *, thinking: bool) -> Completion:
        usage = _to_usage(payload.get("usage"))
        content = payload.get("content")
        if payload.get("role") != "assistant" or not isinstance(content, list):
            raise ProtocolFailure("Messages reply has no assistant content")
        stop = _STOP_REASONS.get(payload.get("stop_reason"), "other")
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
                raise ProtocolFailure("Messages content block must be an object")
            kind = block.get("type")
            if kind == "text":
                if not isinstance(block.get("text"), str):
                    raise ProtocolFailure("Messages text must be a string")
                texts.append(block["text"])
                if block["text"]:
                    pending.append(block["text"])
            elif kind == "tool_use":
                # Even a well-formed call in a reply cut short is not an authorized action.
                if not complete:
                    continue
                call_id, name, arguments = block.get("id"), block.get("name"), block.get("input")
                if not isinstance(call_id, str) or not call_id or any(c.id == call_id for c in calls):
                    raise ProtocolFailure("tool call id is missing or duplicated")
                if not isinstance(name, str) or not name:
                    raise ProtocolFailure("tool name is missing")
                if not isinstance(arguments, dict):
                    raise ProtocolFailure("tool arguments must be a JSON object")
                raw_arguments = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
                calls.append(ToolCall(id=call_id, name=name, arguments=arguments, raw_arguments=raw_arguments))
                flush()
                transcript.append({"type": _NATIVE, "item": {
                    "type": "function_call", "call_id": call_id, "name": name,
                    "arguments": raw_arguments, "status": "completed"}})
                transcript.append({"type": "tool_use", "id": call_id, "name": name, "input": arguments})
            # Thinking, redacted thinking and any unknown block are not replayed and are left out.
        flush()
        if not complete:
            transcript = [{"type": "text", "text": text} for text in texts if text]
        elif calls:
            stop = "tool_use"

        sampling: dict[str, Any] = {"temperature": None, "effort": request.sampling.effort}
        if thinking:
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
    """Messages usage -> disjoint buckets. Every counter must be there; absent is not zero."""
    if not isinstance(usage, dict):
        raise UsageSchemaError("Messages reply has no usage")
    counts = {}
    for name in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        value = usage.get(name)
        if type(value) is not int or value < 0:
            raise UsageSchemaError(f"Messages usage {name} must be a non-negative integer")
        counts[name] = value
    return Usage(
        input_tokens=counts["input_tokens"], output_tokens=counts["output_tokens"],
        cache_read_tokens=counts["cache_read_input_tokens"], cache_write_tokens=counts["cache_creation_input_tokens"],
        raw=dict(counts),
    )


# ---------------------------------------------------------------- request


def _to_messages(system_blocks: list[dict[str, Any]], messages: list[dict[str, Any]]):
    """Canonical system and messages -> Messages ``system`` blocks and alternating ``messages``.

    As the Converse adapter: a system or developer message before the
    conversation joins the system prompt; consecutive turns of one role are
    merged; in a user turn the tool results come first; reasoning items are left
    out and counted.
    """
    system = [{"type": "text", "text": block["text"]} for block in system_blocks if block.get("text")]
    turns: list[tuple[str, dict[str, Any]]] = []
    dropped = 0

    def say(role, block):
        if role in ("system", "developer"):
            if turns:
                raise ProtocolFailure("Messages has no system message after the conversation starts")
            if block.get("text"):
                system.append({"type": "text", "text": block["text"]})
            return
        if role not in ("user", "assistant"):
            raise ProtocolFailure(f"no Messages turn for role {role!r}")
        if block.get("type") == "text" and not block["text"]:
            return  # the Messages API refuses empty text, and it says nothing
        turns.append((role, block))

    for message in messages:
        role, content = message.get("role"), message.get("content")
        if isinstance(content, str):
            say(role, {"type": "text", "text": content})
            continue
        native_calls = {block["item"].get("call_id") for block in content or []
                        if block.get("type") == _NATIVE and block["item"].get("type") == "function_call"}
        for block in content or []:
            kind = block.get("type")
            if kind == _NATIVE:
                dropped += _native(block["item"], say)
            elif kind == "text":
                say(role, {"type": "text", "text": block["text"]})
            elif kind == "tool_use":
                if block["id"] not in native_calls:
                    say("assistant", {"type": "tool_use", "id": block["id"], "name": block["name"],
                                      "input": block.get("input") or {}})
            elif kind == "tool_result":
                say("user", _tool_result(block["tool_use_id"], block.get("content", "")))
            else:
                raise ProtocolFailure(f"no Messages form for a {kind!r} block")

    merged: list[dict[str, Any]] = []
    for role, block in turns:
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"].append(block)
        else:
            merged.append({"role": role, "content": [block]})
    for turn in merged:
        if turn["role"] == "user":
            turn["content"].sort(key=lambda block: block.get("type") != "tool_result")
    if merged and merged[0]["role"] != "user":
        raise ProtocolFailure("a Messages conversation starts with the user")
    return system, merged, dropped


def _native(item: dict[str, Any], say) -> int:
    """One Responses item as Messages blocks; 1 if it is reasoning, which is left out."""
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
        say("assistant", {"type": "tool_use", "id": item["call_id"], "name": item["name"], "input": arguments})
    elif kind == "function_call_output":
        say("user", _tool_result(item["call_id"], item.get("output", "")))
    elif kind in (None, "message"):
        content = item.get("content")
        if isinstance(content, str):
            say(item.get("role"), {"type": "text", "text": content})
        else:
            for part in content or []:
                if part.get("type") in ("output_text", "input_text", "text"):
                    say(item.get("role"), {"type": "text", "text": part.get("text", "")})
                elif part.get("type") != "refusal":
                    raise ProtocolFailure(f"no Messages form for a {part.get('type')!r} part")
    else:
        raise ProtocolFailure(f"no Messages form for a {kind!r} item")
    return 0


def _tool_result(call_id: str, output: Any) -> dict[str, Any]:
    # An empty output is a result with no content: the Messages API refuses empty text blocks.
    text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
    result: dict[str, Any] = {"type": "tool_result", "tool_use_id": call_id}
    if text:
        result["content"] = text
    return result


def _to_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """A canonical or Responses function tool -> a Messages tool."""
    if tool.get("type") == "function" and "parameters" in tool:
        name, description, schema = tool["name"], tool.get("description", ""), tool["parameters"]
    else:
        name, description = tool["name"], tool.get("description", "")
        schema = tool.get("input_schema", {"type": "object"})
    spec: dict[str, Any] = {"name": name, "input_schema": schema}
    if description:
        spec["description"] = description
    return spec
