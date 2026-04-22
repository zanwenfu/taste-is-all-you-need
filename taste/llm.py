"""Anthropic client wrapper with prompt-caching helpers + run telemetry.

Kept intentionally thin: one ``call`` method that returns the raw message,
plus helpers for turning strings into cacheable system blocks and tracking
token usage / cache hit / estimated cost per model.

The harness components (planner, worker, monitor) drive their own tool-use
loops — this module never owns the loop. That separation is what lets the
kernel insert a checkpoint commit between any two model turns.
"""

from __future__ import annotations

import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message
from dotenv import load_dotenv

# Default model assignments. Planner gets the strongest reasoning (long-horizon
# decomposition is the task that benefits most from it), Worker gets Sonnet
# (implementation sweet-spot), Monitor gets Haiku (cheap, high-frequency eval).
MODEL_PLANNER = "claude-opus-4-7"
MODEL_WORKER = "claude-sonnet-4-6"
MODEL_MONITOR = "claude-haiku-4-5-20251001"


# USD per 1M tokens. Approximate public list prices; the harness reports
# "estimated cost" and defers to Anthropic's invoice for ground truth. Edit
# here when pricing changes; no other module depends on these numbers.
_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-7":            {"input": 15.00, "output": 75.00, "cache_write": 18.75, "cache_read": 1.50},
    "claude-sonnet-4-6":          {"input":  3.00, "output": 15.00, "cache_write":  3.75, "cache_read": 0.30},
    "claude-haiku-4-5-20251001":  {"input":  0.80, "output":  4.00, "cache_write":  1.00, "cache_read": 0.08},
}
_FALLBACK_PRICING = _PRICING["claude-sonnet-4-6"]


@dataclass
class ModelUsage:
    """Per-model token totals."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def add(self, message: Message) -> None:
        u = message.usage
        self.calls += 1
        self.input_tokens += u.input_tokens
        self.output_tokens += u.output_tokens
        self.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
        self.cache_creation_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0

    def cost_usd(self, model: str) -> float:
        rates = _PRICING.get(model, _FALLBACK_PRICING)
        return (
            self.input_tokens * rates["input"]
            + self.output_tokens * rates["output"]
            + self.cache_creation_tokens * rates["cache_write"]
            + self.cache_read_tokens * rates["cache_read"]
        ) / 1_000_000


@dataclass
class RunStats:
    """Aggregate + per-model usage for one kernel run. Thread-safe: parallel
    workers on worktrees share one RunStats via the shared ``LLM`` client.
    """

    per_model: dict[str, ModelUsage] = field(default_factory=lambda: defaultdict(ModelUsage))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, model: str, message: Message) -> None:
        with self._lock:
            self.per_model[model].add(message)

    @property
    def totals(self) -> ModelUsage:
        total = ModelUsage()
        for u in self.per_model.values():
            total.calls += u.calls
            total.input_tokens += u.input_tokens
            total.output_tokens += u.output_tokens
            total.cache_read_tokens += u.cache_read_tokens
            total.cache_creation_tokens += u.cache_creation_tokens
        return total

    @property
    def total_cost_usd(self) -> float:
        return sum(u.cost_usd(m) for m, u in self.per_model.items())

    @property
    def cache_hit_rate(self) -> float:
        t = self.totals
        denom = t.cache_read_tokens + t.cache_creation_tokens + t.input_tokens
        return t.cache_read_tokens / denom if denom else 0.0


class LLM:
    """A thin Anthropic wrapper focused on caching + observability.

    Auto-loads `.env` from the caller's working directory (walking up) before
    reading ``ANTHROPIC_API_KEY``. The key is never logged, echoed, or
    exposed through the public API.
    """

    def __init__(self, *, api_key: str | None = None, env_dir: Path | None = None) -> None:
        load_dotenv(_find_env(env_dir), override=False)
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Put it in .env or export it in your shell."
            )
        self.client = Anthropic(api_key=key)
        self.stats = RunStats()

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
        self.stats.record(model, message)
        return message


def cached(text: str) -> dict[str, Any]:
    """Wrap a block of text as a cacheable system block."""
    return {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}


def _as_system_blocks(system: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize system into block form so cache_control is always respected."""
    if isinstance(system, str):
        return [cached(system)]
    return system


def _find_env(start: Path | None) -> Path | None:
    """Walk up from ``start`` (default: cwd) to find a .env file."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        env = candidate / ".env"
        if env.is_file():
            return env
    return None
