"""Anthropic client wrapper: prompt caching, retries, budget caps, run telemetry.

Kept intentionally thin: one ``call`` method that returns the raw message,
plus helpers for cacheable system blocks and per-role token/cost accounting.

Experiment-integrity guarantees this layer owns (research_plan.md Wave 0):

* **Typed failure classes.** Transient API errors are retried with exponential
  backoff; if they persist, ``InfraFailure`` is raised so infra outages are
  never recorded as task failures. ``BudgetExceeded`` enforces a hard per-run
  dollar cap — the cost-matching instrument *is* the budget guard.
* **No silent mispricing.** An unknown model ID raises ``PricingError`` before
  any API call instead of silently billing at a fallback rate.
* **Per-role attribution.** Usage is recorded per (role, model) so planner /
  worker / monitor spend can be separated in cost-matched comparisons.

The harness components (planner, worker, monitor) drive their own tool-use
loops — this module never owns the loop. That separation is what lets the
kernel insert a checkpoint commit between any two model turns.
"""

from __future__ import annotations

import hashlib
import os
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
from anthropic import Anthropic
from anthropic.types import Message
from dotenv import load_dotenv

# Default model assignments. Planner gets the strongest reasoning (long-horizon
# decomposition is the task that benefits most from it), Worker gets Sonnet
# (implementation sweet-spot), Monitor gets Haiku (cheap, high-frequency eval).
MODEL_PLANNER = "claude-opus-4-7"
MODEL_WORKER = "claude-sonnet-4-6"
MODEL_MONITOR = "claude-haiku-4-5-20251001"

# Single sampling default, surfaced here so the kernel can log it in the run
# manifest. Experiments override per call; nothing hardcodes temperature inline.
DEFAULT_TEMPERATURE = 1.0


# USD per 1M tokens, verified against https://platform.claude.com/docs/en/about-claude/pricing
# Cache columns are the 5-minute-TTL write (1.25x base input) and cache hit
# (0.1x base input). We never request 1-hour caching (2x write).
#
# Cost is a PRIMARY dependent variable in the experiments, so these numbers are
# load-bearing: a wrong rate silently corrupts every cost-matched comparison.
# Re-verify against the live page before each sweep and bump PRICING_AS_OF.
#
# WARNING when adding models: Claude 4.7+ uses a newer tokenizer that emits
# ~30% more tokens for the same text than 4.6-and-earlier. Token counts are
# therefore NOT comparable across that boundary — always match on dollars.
PRICING_AS_OF = "2026-08-13"

_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-5":              {"input":  5.00, "output": 25.00, "cache_write":  6.25, "cache_read": 0.50},
    "claude-opus-4-7":            {"input":  5.00, "output": 25.00, "cache_write":  6.25, "cache_read": 0.50},
    "claude-sonnet-5":            {"input":  2.00, "output": 10.00, "cache_write":  2.50, "cache_read": 0.20},
    "claude-sonnet-4-6":          {"input":  3.00, "output": 15.00, "cache_write":  3.75, "cache_read": 0.30},
    "claude-haiku-4-5-20251001":  {"input":  1.00, "output":  5.00, "cache_write":  1.25, "cache_read": 0.10},
}

# HTTP statuses worth retrying: rate limits, transient server errors, overload.
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}

# Shared across every LLM instance in the process so parallel kernels can't
# stampede the API. Override the size with TASTE_API_CONCURRENCY.
_API_SEMAPHORE = threading.Semaphore(int(os.environ.get("TASTE_API_CONCURRENCY", "8")))


class PricingError(RuntimeError):
    """Raised before any API call when a model has no pricing entry.

    A silent fallback rate would corrupt every cost-matched comparison the
    experiments run, so unknown models fail fast instead. Classified as an
    "infra" failure (configuration problem, not an agent task failure).
    """

    failure_kind = "infra"


class BudgetExceeded(RuntimeError):
    """The per-run dollar cap was hit. The run must halt, typed as 'budget'."""

    failure_kind = "budget"

    def __init__(self, spent_usd: float, budget_usd: float) -> None:
        super().__init__(
            f"budget exceeded: ${spent_usd:.4f} spent >= ${budget_usd:.4f} cap"
        )
        self.spent_usd = spent_usd
        self.budget_usd = budget_usd


class InfraFailure(RuntimeError):
    """The API kept failing after retries. Must never count as a task failure."""

    failure_kind = "infra"

    def __init__(self, message: str, *, attempts: int, last_error: Exception | None) -> None:
        super().__init__(f"{message} (attempts={attempts}, last={last_error!r})")
        self.attempts = attempts
        self.last_error = last_error


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, anthropic.APIConnectionError):  # includes APITimeoutError
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS
    return False


@dataclass
class ModelUsage:
    """Token totals for one (role, model) cell."""

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

    def merge(self, other: ModelUsage) -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_creation_tokens += other.cache_creation_tokens

    def cost_usd(self, model: str) -> float:
        rates = _require_pricing(model)
        return (
            self.input_tokens * rates["input"]
            + self.output_tokens * rates["output"]
            + self.cache_creation_tokens * rates["cache_write"]
            + self.cache_read_tokens * rates["cache_read"]
        ) / 1_000_000


@dataclass
class RunStats:
    """Aggregate + per-(role, model) usage for one kernel run. Thread-safe:
    parallel workers on worktrees share one RunStats via the shared ``LLM``
    client. One RunStats belongs to exactly one run — construct a fresh LLM
    per run so cost attribution never bleeds across trials.

    The running dollar total is maintained incrementally under the lock so the
    per-call budget check is an O(1) read that can never race a concurrent
    ``record()`` (iterating the cells unlocked could observe a mid-insert dict).
    """

    per_role_model: dict[tuple[str, str], ModelUsage] = field(
        default_factory=lambda: defaultdict(ModelUsage)
    )
    _total_cost_usd: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, model: str, message: Message, *, role: str = "unspecified") -> None:
        with self._lock:
            self.per_role_model[(role, model)].add(message)
            self._total_cost_usd += _message_cost_usd(model, message)

    def _snapshot(self) -> list[tuple[tuple[str, str], ModelUsage]]:
        with self._lock:
            return list(self.per_role_model.items())

    @property
    def per_model(self) -> dict[str, ModelUsage]:
        """Role-agnostic aggregation (the CLI usage table reads this).

        Derived snapshot — mutating the returned objects records nothing.
        """
        agg: dict[str, ModelUsage] = defaultdict(ModelUsage)
        for (_role, model), usage in self._snapshot():
            agg[model].merge(usage)
        return dict(agg)

    @property
    def totals(self) -> ModelUsage:
        total = ModelUsage()
        for _key, usage in self._snapshot():
            total.merge(usage)
        return total

    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return self._total_cost_usd

    @property
    def cache_hit_rate(self) -> float:
        t = self.totals
        denom = t.cache_read_tokens + t.cache_creation_tokens + t.input_tokens
        return t.cache_read_tokens / denom if denom else 0.0


class LLM:
    """A thin Anthropic wrapper focused on caching, retries, and observability.

    Auto-loads `.env` from the caller's working directory (walking up) before
    reading ``ANTHROPIC_API_KEY``. The key is never logged, echoed, or
    exposed through the public API.

    Parameters
    ----------
    budget_usd:
        Hard per-run dollar cap. Checked before every call; once estimated
        spend reaches the cap, :class:`BudgetExceeded` is raised. The check is
        check-then-act, so calls already in flight complete: overshoot is
        bounded by the number of concurrently in-flight calls (≤ the kernel's
        max_parallel) times one call's cost. Cost-matched analysis must use
        realized spend from RunStats, never assume spend == cap.
    max_attempts / backoff_base:
        Retry policy for transient API errors (429/5xx/529/connection). Delays
        are ``backoff_base * 2^attempt`` capped at 30s with jitter. SDK-level
        retries are disabled so this layer owns the classification.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        env_dir: Path | None = None,
        budget_usd: float | None = None,
        max_attempts: int = 5,
        backoff_base: float = 1.0,
        semaphore: threading.Semaphore | None = None,
    ) -> None:
        load_dotenv(_find_env(env_dir), override=False)
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Put it in .env or export it in your shell."
            )
        self.client = Anthropic(api_key=key, max_retries=0)
        self.stats = RunStats()
        self.budget_usd = budget_usd
        self.max_attempts = max(1, max_attempts)
        self.backoff_base = backoff_base
        self._semaphore = semaphore or _API_SEMAPHORE

    def call(
        self,
        *,
        model: str,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = DEFAULT_TEMPERATURE,
        role: str = "unspecified",
    ) -> Message:
        """Single API call with retry/backoff. Returns the full Message.

        Raises :class:`PricingError` for unknown models, :class:`BudgetExceeded`
        when the per-run cap is reached, and :class:`InfraFailure` when
        transient errors persist past ``max_attempts``. Non-retryable API
        errors (4xx other than 408/409/429) propagate unchanged.
        """
        _require_pricing(model)
        if self.budget_usd is not None and self.stats.total_cost_usd >= self.budget_usd:
            raise BudgetExceeded(self.stats.total_cost_usd, self.budget_usd)

        kwargs: dict[str, Any] = {
            "model": model,
            "system": _as_system_blocks(system),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools

        last_exc: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                with self._semaphore:
                    message = self.client.messages.create(**kwargs)
            except anthropic.APIError as exc:
                if not _is_retryable(exc):
                    # Non-retryable API errors (credit exhausted, auth revoked,
                    # malformed request) are environment problems, not agent
                    # task failures — surface them typed so the kernel
                    # classifies the run as infra instead of crashing raw.
                    raise InfraFailure(
                        "non-retryable API error",
                        attempts=attempt + 1,
                        last_error=exc,
                    ) from exc
                last_exc = exc
                if attempt < self.max_attempts - 1:
                    delay = min(self.backoff_base * (2**attempt), 30.0)
                    time.sleep(delay * (0.5 + random.random()))
                continue
            self.stats.record(model, message, role=role)
            return message

        raise InfraFailure(
            "API call failed on transient errors",
            attempts=self.max_attempts,
            last_error=last_exc,
        ) from last_exc


def cached(text: str) -> dict[str, Any]:
    """Wrap a block of text as a cacheable system block (one cache breakpoint)."""
    return {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}


def static_system(*parts: str) -> dict[str, Any]:
    """Join static prompt parts into ONE cacheable block.

    Prompt caching has a minimum cacheable-prefix length (1024 tokens on
    Sonnet-class models); several small blocks each below the minimum cache
    nothing. Consolidating the static prefix into a single block (tools +
    system share the prefix) is the Wave-0 cache fix — verify with
    ``scripts/cache_smoke.py``.
    """
    return cached("\n\n".join(p.strip() for p in parts if p and p.strip()))


def prompt_sha(text: str) -> str:
    """Short stable hash of a prompt, for run manifests."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def ensure_priced(model: str) -> None:
    """Raise :class:`PricingError` if ``model`` has no pricing entry.

    The kernel calls this for every role's model at run start so unpriced
    configurations fail fast, before any money is spent.
    """
    _require_pricing(model)


def _message_cost_usd(model: str, message: Message) -> float:
    rates = _require_pricing(model)
    u = message.usage
    return (
        u.input_tokens * rates["input"]
        + u.output_tokens * rates["output"]
        + (getattr(u, "cache_creation_input_tokens", 0) or 0) * rates["cache_write"]
        + (getattr(u, "cache_read_input_tokens", 0) or 0) * rates["cache_read"]
    ) / 1_000_000


def _require_pricing(model: str) -> dict[str, float]:
    if model not in _PRICING:
        known = ", ".join(sorted(_PRICING))
        raise PricingError(
            f"no pricing entry for model {model!r}; add it to taste.llm._PRICING "
            f"(known: {known}). Refusing to run with silently mispriced cost telemetry."
        )
    return _PRICING[model]


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
