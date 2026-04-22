"""Anthropic client wrapper with prompt-caching helpers.

Kept intentionally thin: one ``call`` method that returns the raw message,
plus small helpers for turning strings into cacheable system blocks and
counting cache hit/miss tokens for telemetry.

The harness components (planner, worker, monitor) drive their own tool-use
loops — this module never owns the loop. That separation is what lets the
kernel insert a checkpoint commit between any two model turns.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message

# Default model assignments. Planner gets the strongest reasoning (long-horizon
# decomposition is the task that benefits most from it), Worker gets Sonnet
# (implementation sweet-spot), Monitor gets Haiku (cheap, high-frequency eval).
MODEL_PLANNER = "claude-opus-4-7"
MODEL_WORKER = "claude-sonnet-4-6"
MODEL_MONITOR = "claude-haiku-4-5-20251001"


@dataclass
class CacheStats:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_read_tokens + self.cache_creation_tokens + self.input_tokens
        return self.cache_read_tokens / total if total else 0.0

    def add(self, message: Message) -> None:
        usage = message.usage
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0


class LLM:
    """A thin Anthropic wrapper focused on caching + observability."""

    def __init__(self, *, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it or pass api_key=... ."
            )
        self.client = Anthropic(api_key=key)
        self.stats = CacheStats()

    def call(
        self,
        *,
        model: str,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> Message:
        """Single API call. Returns the full Message; caller owns the loop."""
        kwargs: dict[str, Any] = {
            "model": model,
            "system": _as_system_blocks(system),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
        message = self.client.messages.create(**kwargs)
        self.stats.add(message)
        return message


def cached(text: str) -> dict[str, Any]:
    """Wrap a block of text as a cacheable system block."""
    return {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}


def _as_system_blocks(system: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize system into block form so cache_control is always respected."""
    if isinstance(system, str):
        return [cached(system)]
    return system
