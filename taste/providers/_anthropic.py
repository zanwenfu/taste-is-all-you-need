"""Anthropic adapter.

The reference implementation, and the cheapest one: the canonical IR *is* the
Anthropic wire shape, so translation here is close to the identity function.
That is the point of choosing it as the IR — the adapter that has to be
exactly right is the one that barely does anything.
"""

from __future__ import annotations

import os
from typing import Any

from taste.providers.base import (
    Completion,
    CompletionRequest,
    ProtocolFailure,
    ToolCall,
    Usage,
    require,
)

# Statuses worth another attempt: rate limits, transient server faults, overload.
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

_STOP_REASONS = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "end_turn",
    "refusal": "refusal",
}


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client: Any = None

    def ensure_ready(self) -> None:
        if not self._api_key:
            raise ProtocolFailure(
                "ANTHROPIC_API_KEY is not set. Put it in .env or export it."
            )
        self._ensure_client()

    def _ensure_client(self) -> Any:
        if self._client is None:
            import anthropic

            # max_retries=0: the facade owns retry policy, so it cannot
            # differ between providers or double up with the SDK's own.
            self._client = anthropic.Anthropic(api_key=self._api_key, max_retries=0)
        return self._client

    # ------------------------------------------------------------ calls

    def complete(self, request: CompletionRequest) -> Completion:
        client = self._ensure_client()
        kwargs: dict[str, Any] = {
            "model": request.model,
            "system": request.system,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
        }
        if request.tools:
            kwargs["tools"] = request.tools
        if request.sampling.temperature is not None:
            kwargs["temperature"] = request.sampling.temperature

        message = client.messages.create(**kwargs)
        return self._to_completion(message, request)

    def is_retryable(self, exc: Exception) -> bool:
        import anthropic

        if isinstance(exc, anthropic.APIConnectionError):  # includes timeouts
            return True
        if isinstance(exc, anthropic.APIStatusError):
            return exc.status_code in _RETRYABLE_STATUS
        return False

    # ------------------------------------------------------------ translation

    def _to_completion(self, message: Any, request: CompletionRequest) -> Completion:
        texts: list[str] = []
        calls: list[ToolCall] = []
        transcript: list[dict[str, Any]] = []

        for block in message.content:
            kind = getattr(block, "type", None)
            if kind == "text":
                texts.append(block.text)
                transcript.append({"type": "text", "text": block.text})
            elif kind == "tool_use":
                arguments = dict(block.input or {})
                calls.append(ToolCall(id=block.id, name=block.name, arguments=arguments))
                transcript.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": arguments,
                    }
                )
            else:
                # Thinking and any future block type pass through untouched,
                # so a transcript stays replayable even for blocks we do not
                # interpret.
                transcript.append(
                    block.model_dump() if hasattr(block, "model_dump") else {"type": kind}
                )

        return Completion(
            text_blocks=tuple(texts),
            tool_calls=tuple(calls),
            stop_reason=_STOP_REASONS.get(message.stop_reason or "", "other"),
            model=getattr(message, "model", "") or request.model,
            provider=self.name,
            usage=self._to_usage(message.usage),
            transcript_blocks=tuple(transcript),
            effective_sampling={"temperature": request.sampling.temperature},
            raw=message,
        )

    def _to_usage(self, usage: Any) -> Usage:
        # input_tokens already excludes the cached portion here, which is why
        # this adapter needs no subtraction — unlike OpenAI's.
        write = require(usage, "cache_creation_input_tokens")
        long_ttl = self._one_hour_writes(usage)
        if long_ttl:
            raise ProtocolFailure(
                f"{long_ttl} tokens were written to the 1-hour cache, billed at 2x "
                "rather than the 1.25x this ledger assumes. We never request 1h "
                "caching, so this is a bug rather than a price to absorb."
            )
        return Usage(
            input_tokens=require(usage, "input_tokens"),
            output_tokens=require(usage, "output_tokens"),
            cache_read_tokens=require(usage, "cache_read_input_tokens"),
            cache_write_tokens=write,
            reasoning_tokens=0,
            raw=_usage_dict(usage),
        )

    @staticmethod
    def _one_hour_writes(usage: Any) -> int:
        detail = getattr(usage, "cache_creation", None)
        if detail is None:
            return 0
        return int(getattr(detail, "ephemeral_1h_input_tokens", 0) or 0)


def _usage_dict(usage: Any) -> dict[str, Any]:
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:
            pass
    return {
        k: v for k, v in vars(usage).items() if isinstance(v, int | float | str | bool | type(None))
    }
