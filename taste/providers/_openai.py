"""OpenAI adapter, on the Responses API.

**Why Responses and not Chat Completions.** The reasoning models reject the
combination this harness needs — function tools together with any reasoning
effort — on ``/v1/chat/completions``, with an error naming ``/v1/responses``
as the place to do it. Since tool use *is* the harness's core operation and
reasoning is the reason to pick these models, Responses is the only surface
that supports the actual workload. Discovered by calling the real API; no
unit test against our own fakes could have found it.

**Reasoning items must be replayed.** With ``store=False`` the provider keeps
no server-side state, so a reasoning item produced before a tool call has to
be sent back on the next turn or the model loses its own train of thought.
They are carried through the canonical transcript as opaque blocks that only
this adapter interprets — other providers skip them.

**The usage subtraction.** OpenAI's ``input_tokens`` is the *total* prompt,
cached portion included; Anthropic's excludes it. Normalizing to disjoint
buckets here is what makes a dollar figure comparable across families;
getting it wrong double-counts every cached token on one side of the study.

**Reasoning tokens are inside ``output_tokens``** and are reported separately
for analysis, never added. They also consume ``max_output_tokens``. A small
ceiling may produce no visible output; the adapter must honor it anyway,
because the facade reserved a budget for precisely that ceiling.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from taste.providers.azure_openai import AzureOpenAIConfig
from taste.providers.base import (
    Completion,
    CompletionRequest,
    ProtocolFailure,
    ToolCall,
    Usage,
    UsageSchemaError,
)

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

# Blocks carrying provider-native items through the canonical transcript.
_NATIVE = "_openai_item"

_STOP_REASONS = {
    "completed": "end_turn",
    "incomplete": "max_tokens",
    "failed": "other",
    "cancelled": "other",
}


class OpenAIProvider:
    name = "openai"

    def __init__(self, *, api_key: str | None = None, azure: AzureOpenAIConfig | None = None) -> None:
        if azure is not None and api_key is not None:
            raise ProtocolFailure("Azure credentials must come from the explicit Azure configuration")
        self._azure = azure
        self._api_key = azure.api_key if azure is not None else api_key or os.environ.get("OPENAI_API_KEY")
        self._client: Any = None

    def ensure_ready(self) -> None:
        if not self._api_key:
            raise ProtocolFailure("OPENAI_API_KEY is not set. Put it in .env or export it.")
        self._ensure_client()

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ProtocolFailure(
                    "the `openai` package is not installed; `pip install openai`"
                ) from exc
            # max_retries=0: the facade owns retry policy so it cannot differ
            # between providers or compound with the SDK's own.
            if self._azure is None:
                self._client = openai.OpenAI(api_key=self._api_key, max_retries=0)
            else:
                import httpx

                if os.environ.get("OPENAI_CUSTOM_HEADERS"):
                    # The SDK merges this variable into explicit headers,
                    # including Authorization. Refuse before creating a client
                    # rather than mutate process environment across threads.
                    raise ProtocolFailure("Azure-only calls forbid OPENAI_CUSTOM_HEADERS")
                self._client = openai.OpenAI(
                    api_key=self._api_key, base_url=self._azure.base_url,
                    organization="", project="", admin_api_key="", webhook_secret="",
                    max_retries=0, timeout=60.0,
                    http_client=httpx.Client(trust_env=False, follow_redirects=False),
                )
        return self._client

    # ------------------------------------------------------------ calls

    def complete(self, request: CompletionRequest) -> Completion:
        if type(request.max_tokens) is not int or request.max_tokens < 1:
            raise ProtocolFailure("max_tokens must be a positive integer")
        binding = self._azure.deployment_for(request.model) if self._azure is not None else None
        client = self._ensure_client()

        kwargs: dict[str, Any] = {
            "model": binding.deployment if binding is not None else request.model,
            "instructions": _instructions(request.system),
            "input": self._to_input(request.messages),
            "max_output_tokens": request.max_tokens,
            # No server-side state: a run must be reproducible from its own
            # transcript, not from something the provider remembers.
            "store": False,
            # Required by Azure's stateless Responses contract; also accepted
            # by OpenAI versions that now include encrypted content by default.
            "include": ["reasoning.encrypted_content"],
        }
        if request.tools:
            kwargs["tools"] = [_to_tool(t) for t in request.tools]
        if request.timeout_seconds is not None:
            kwargs["timeout"] = request.timeout_seconds
        if request.sampling.effort:
            kwargs["reasoning"] = {"effort": request.sampling.effort}

        # temperature is not accepted alongside reasoning on these models.
        # Dropping it silently would leave the manifest claiming a setting
        # that never applied, so the drop is recorded instead.
        dropped = ["temperature"] if request.sampling.temperature is not None else []

        raw = client.responses.with_raw_response.create(**kwargs)
        # Validate counters before the SDK's permissive model construction:
        # it can coerce strings/bools into integer counters, concealing an
        # incompatible wire schema from the accounting boundary.
        payload = raw.http_response.json()
        if not isinstance(payload, dict):
            raise ProtocolFailure("Responses payload must be a JSON object")
        usage = self._to_usage(payload.get("usage"))
        completion = self._to_completion(raw.parse(), request, dropped, usage=usage)
        if binding is not None:
            served = raw.headers.get("x-ms-served-model") or payload.get("model")
            if served != binding.model or payload.get("model") not in {binding.model, binding.deployment}:
                raise ProtocolFailure("Azure served-model identity differs from the pinned deployment")
            completion = replace(completion, model=served, provenance={
                "route": "azure_openai", "endpoint": self._azure.base_url,
                "deployment": binding.deployment, "deployment_type": binding.deployment_type,
                "served_model": served,
                "model_session": raw.headers.get("azureml-model-session", ""),
                "region": raw.headers.get("x-ms-region", ""),
            })
        return completion

    def is_retryable(self, exc: Exception) -> bool:
        if self._azure is not None:
            # A lost reply may already have been billed. The new Azure path
            # requires explicit settlement, never automatic paid replays.
            return False
        try:
            import openai
        except ImportError:  # pragma: no cover
            return False
        if isinstance(exc, openai.APIConnectionError):
            return True
        status = getattr(exc, "status_code", None)
        return isinstance(status, int) and status in _RETRYABLE_STATUS

    # ------------------------------------------------------------ request

    def _to_input(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Canonical messages -> Responses ``input`` items."""
        items: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if isinstance(content, str):
                items.append({"role": role, "content": content})
                continue

            # Older saved turns contain both canonical text and a native
            # message with that text. Replay the native message exactly once,
            # preserving its phase, annotations and provider-specific fields.
            # Count matching shadows so unrelated added text is retained.
            shadows: Counter[str] = Counter()
            for block in content or []:
                native = block.get("item", {}) if block.get("type") == _NATIVE else {}
                if isinstance(native, dict) and native.get("type") == "message":
                    shadows.update(
                        p["text"] for p in native.get("content", [])
                        if p.get("type") == "output_text"
                    )
            for block in content or []:
                kind = block.get("type")
                if kind == _NATIVE:
                    # Reasoning and function_call items, replayed verbatim.
                    items.append(block["item"])
                elif kind == "text":
                    if shadows[block["text"]]:
                        shadows[block["text"]] -= 1
                    else:
                        items.append({"role": role, "content": block["text"]})
                elif kind == "tool_result":
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": block["tool_use_id"],
                            "output": str(block.get("content", "")),
                        }
                    )
                # A canonical tool_use block always travels beside its native
                # twin, so it needs no separate translation.
        return items

    # ------------------------------------------------------------ response

    def _to_completion(
        self, response: Any, request: CompletionRequest, dropped: list[str],
        *, usage: Usage | None = None,
    ) -> Completion:
        texts: list[str] = []
        calls: list[ToolCall] = []
        transcript: list[dict[str, Any]] = []
        status = getattr(response, "status", "") or ""
        stop = _STOP_REASONS.get(status, "other")
        if status == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
            if reason == "content_filter":
                stop = "content_filter"
        refused = False
        call_ids: set[str] = set()

        for item in getattr(response, "output", []) or []:
            kind = getattr(item, "type", None)
            native = _as_dict(item)

            if kind == "message":
                for part in getattr(item, "content", []) or []:
                    if getattr(part, "type", None) == "output_text":
                        texts.append(part.text)
                    elif getattr(part, "type", None) == "refusal":
                        refused = True
                # The message item itself is replayed so the model sees its
                # own prior turn in the same shape it produced it.
                transcript.append({"type": _NATIVE, "item": native})

            elif kind == "function_call":
                # Even a parseable call in a truncated/failed response is not
                # an authorized action. Never expose it to a tool dispatcher.
                if status != "completed":
                    continue
                raw_arguments = getattr(item, "arguments", "")
                try:
                    arguments = json.loads(raw_arguments, parse_constant=_reject_constant)
                    # JSON numeric overflow (1e999) becomes inf without going
                    # through parse_constant. Reject it recursively as well.
                    json.dumps(arguments, allow_nan=False)
                except (TypeError, ValueError) as exc:
                    raise ProtocolFailure(
                        f"tool call {getattr(item, 'name', '?')!r} had unparseable "
                        "JSON arguments"
                    ) from exc
                if not isinstance(arguments, dict):
                    raise ProtocolFailure("tool arguments must be a JSON object")
                call_id = getattr(item, "call_id", None)
                name = getattr(item, "name", None)
                if not isinstance(call_id, str) or not call_id or call_id in call_ids:
                    raise ProtocolFailure("tool call_id is missing or duplicated")
                if not isinstance(name, str) or not name:
                    raise ProtocolFailure("tool name is missing")
                call_ids.add(call_id)
                call = ToolCall(
                    id=call_id,
                    name=name,
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                )
                calls.append(call)
                transcript.append({"type": _NATIVE, "item": native})
                transcript.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )

            elif kind == "reasoning":
                # Opaque, and required: dropping it with store=False loses the
                # model's own context across a tool call.
                transcript.append({"type": _NATIVE, "item": native})

        if refused:
            stop = "refusal"
            calls.clear()
        elif calls and stop == "end_turn":
            stop = "tool_use"

        sampling: dict[str, Any] = {"temperature": None, "effort": request.sampling.effort}
        if dropped:
            sampling["dropped"] = dropped

        return Completion(
            text_blocks=tuple(texts),
            tool_calls=tuple(calls),
            stop_reason=stop,
            model=getattr(response, "model", "") or request.model,
            provider=self.name,
            usage=usage if usage is not None else self._to_usage(response.usage),
            transcript_blocks=tuple(transcript),
            effective_sampling=sampling,
            raw=response,
        )

    def _to_usage(self, usage: Any) -> Usage:
        total_prompt = _counter(usage, "input_tokens")
        output = _counter(usage, "output_tokens")
        details = _field(usage, "input_tokens_details")
        cached = _counter(details, "cached_tokens")
        written = _counter(details, "cache_write_tokens")
        reasoning = _counter(_field(usage, "output_tokens_details"), "reasoning_tokens")
        if cached + written > total_prompt or reasoning > output:
            raise UsageSchemaError("Responses usage buckets exceed their reported total")

        # GPT-5.6+ bills writes separately, at 1.25x ordinary input, rather
        # than adding a surcharge to tokens already counted as uncached.
        return Usage(
            input_tokens=total_prompt - cached - written,
            output_tokens=output,
            cache_read_tokens=cached,
            cache_write_tokens=written,
            reasoning_tokens=reasoning,
            raw={"input_tokens": total_prompt, "cached_tokens": cached,
                 "cache_write_tokens": written},
        )


def _instructions(system: list[dict[str, Any]]) -> str:
    """System blocks collapse to one instructions string.

    ``cache_control`` markers are dropped rather than translated: caching is
    automatic on this API, so carrying them would imply a control it does not
    offer.
    """
    return "\n\n".join(b.get("text", "") for b in system if b.get("text")).strip()


def _to_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Canonical tool schema -> Responses function tool (flat, not nested).

    Explicit non-strict mode preserves the existing schemas' optional fields.
    Responses may otherwise normalize an omitted ``strict`` into strict mode.
    Tool handlers still validate arguments before any effect.
    """
    return {
        "type": "function",
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool.get("input_schema", {"type": "object"}),
        "strict": False,
    }


def _as_dict(item: Any) -> Any:
    if hasattr(item, "model_dump"):
        try:
            return item.model_dump(exclude_none=True)
        except Exception:
            pass
    return item


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _counter(value: Any, name: str) -> int:
    count = _field(value, name)
    if type(count) is not int or count < 0:
        raise UsageSchemaError(f"Responses usage {name} must be a non-negative integer")
    return count


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")
