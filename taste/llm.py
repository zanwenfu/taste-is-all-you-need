"""The model facade: one call surface, whichever provider answers.

Providers translate wire formats and nothing else (see ``taste/providers``).
Everything a run's integrity depends on lives here, once, so it cannot drift
between families:

* **Typed failures.** Transient errors retry with backoff; if they persist,
  :class:`InfraFailure`. Non-retryable API errors — a revoked key, an
  exhausted balance — surface typed as well, so an environment problem is
  never recorded as an agent task failure.
* **A hard dollar cap** per run, checked before every call
  (:class:`BudgetExceeded`). The cost-matching instrument *is* the budget guard.
* **Per-(role, model) accounting**, in both currencies. ``billed`` is the
  invoice; ``work`` prices every prompt token as if no cache existed. See
  :mod:`taste.pricing` for why an experiment caps on one and reports both.
* **Unpriced models refuse to run**, rather than being billed at a guess.

The cores drive their own tool-use loops; this module never owns one. That
separation is what lets the kernel insert a checkpoint between any two turns.
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

from dotenv import load_dotenv

from taste import providers
from taste.pricing import PricingError, call_cost, ensure_priced
from taste.providers.base import (
    Completion,
    CompletionRequest,
    ProtocolFailure,
    SamplingConfig,
)

# Default model per role. The Planner gets the strongest reasoning
# (long-horizon decomposition benefits most), the Worker a mid tier, the
# Monitor a cheap one it calls often.
MODEL_PLANNER = "claude-opus-4-7"
MODEL_WORKER = "claude-sonnet-4-6"
MODEL_MONITOR = "claude-haiku-4-5-20251001"

DEFAULT_TEMPERATURE = 1.0

# Measured minimum cacheable prefix, by model family. A block shorter than
# this caches nothing at all — silently, with no error and no warning.
#
# Verified empirically 2026-08-15 (two identical calls, checking whether the
# second reads from cache): Sonnet 4.6 caches at ~1.4k tokens; Haiku 4.5 does
# NOT cache at 3.5k but does at 7k. The consequence is worth stating plainly:
# the Monitor runs on Haiku with a short system prompt and a diff, so in
# practice **the Monitor never caches**. Its cost is what it appears to be,
# and a run's cache-hit rate is effectively the Worker's alone.
CACHE_MINIMUM_TOKENS = {"opus": 1024, "sonnet": 1024, "haiku": 4096}


def cache_minimum_for(model: str) -> int:
    """Tokens a system prefix must exceed before caching engages at all."""
    for family, minimum in CACHE_MINIMUM_TOKENS.items():
        if family in model:
            return minimum
    return 1024

# Shared across every LLM in the process so parallel kernels cannot stampede
# a provider. Size with TASTE_API_CONCURRENCY.
_API_SEMAPHORE = threading.Semaphore(int(os.environ.get("TASTE_API_CONCURRENCY", "8")))


class BudgetExceeded(RuntimeError):
    """The per-run dollar cap was reached. The run halts, typed 'budget'."""

    failure_kind = "budget"

    def __init__(self, spent_usd: float, budget_usd: float) -> None:
        super().__init__(f"budget exceeded: ${spent_usd:.4f} spent >= ${budget_usd:.4f} cap")
        self.spent_usd = spent_usd
        self.budget_usd = budget_usd


class InfraFailure(RuntimeError):
    """The provider could not be used. Never counts as a task failure."""

    failure_kind = "infra"

    def __init__(self, message: str, *, attempts: int, last_error: Exception | None) -> None:
        super().__init__(f"{message} (attempts={attempts}, last={last_error!r})")
        self.attempts = attempts
        self.last_error = last_error


@dataclass
class ModelUsage:
    """Token totals and both costs for one (role, model) cell."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0
    billed_usd: float = 0.0
    work_usd: float = 0.0

    def add(self, completion: Completion, model: str) -> None:
        u = completion.usage
        self.calls += 1
        self.input_tokens += u.input_tokens
        self.output_tokens += u.output_tokens
        self.cache_read_tokens += u.cache_read_tokens
        self.cache_creation_tokens += u.cache_write_tokens
        self.reasoning_tokens += u.reasoning_tokens
        # Priced per call: a provider whose long-context tier reprices the
        # whole request gives a different answer if tokens are summed first.
        billed, work = call_cost(
            model,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_tokens=u.cache_read_tokens,
            cache_write_tokens=u.cache_write_tokens,
            reasoning_tokens=u.reasoning_tokens,
        )
        self.billed_usd += billed
        self.work_usd += work

    def merge(self, other: ModelUsage) -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.billed_usd += other.billed_usd
        self.work_usd += other.work_usd

    def cost_usd(self, model: str = "") -> float:
        """Billed cost. ``model`` is accepted for backward compatibility."""
        return self.billed_usd


@dataclass
class RunStats:
    """Per-(role, model) usage for one run. Thread-safe: parallel workers
    share one instance through the shared client.

    One RunStats belongs to exactly one run — construct a fresh LLM per run,
    or cost attribution bleeds across trials. Running totals are maintained
    incrementally under the lock, so the pre-call budget check is an O(1)
    read that cannot race a concurrent record.
    """

    per_role_model: dict[tuple[str, str], ModelUsage] = field(
        default_factory=lambda: defaultdict(ModelUsage)
    )
    _billed_usd: float = field(default=0.0, repr=False)
    _work_usd: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, model: str, completion: Completion, *, role: str = "unspecified") -> None:
        billed, work = call_cost(
            model,
            input_tokens=completion.usage.input_tokens,
            output_tokens=completion.usage.output_tokens,
            cache_read_tokens=completion.usage.cache_read_tokens,
            cache_write_tokens=completion.usage.cache_write_tokens,
        )
        with self._lock:
            self.per_role_model[(role, model)].add(completion, model)
            self._billed_usd += billed
            self._work_usd += work

    def _snapshot(self) -> list[tuple[tuple[str, str], ModelUsage]]:
        with self._lock:
            return list(self.per_role_model.items())

    @property
    def per_model(self) -> dict[str, ModelUsage]:
        """Role-agnostic aggregation. A derived snapshot — mutating the
        returned objects records nothing."""
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
        """Billed cost — what the invoice will say."""
        with self._lock:
            return self._billed_usd

    @property
    def total_work_usd(self) -> float:
        """Cost with every prompt token at the uncached rate.

        Order-invariant, so it is the unit an experiment caps on: billed cost
        depends on what ran before this run and would make each trial's
        budget a function of its neighbours.
        """
        with self._lock:
            return self._work_usd

    @property
    def cache_delta_usd(self) -> float:
        """work - billed. SIGNED: negative when cache writes outweigh reads.

        Reported rather than presented as a saving, because a policy that
        discards its cache prefix pays a real tax that must stay visible.
        """
        with self._lock:
            return self._work_usd - self._billed_usd

    @property
    def cache_hit_rate(self) -> float:
        t = self.totals
        denom = t.cache_read_tokens + t.cache_creation_tokens + t.input_tokens
        return t.cache_read_tokens / denom if denom else 0.0


class LLM:
    """The single call surface. Owns retries, budget, and telemetry.

    Auto-loads `.env` from the working directory upwards. Keys are never
    logged, echoed, or exposed through the public API.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_keys: dict[str, str] | None = None,
        env_dir: Path | None = None,
        budget_usd: float | None = None,
        max_attempts: int = 5,
        backoff_base: float = 1.0,
        semaphore: threading.Semaphore | None = None,
        run_id: str = "",
    ) -> None:
        load_dotenv(_find_env(env_dir), override=False)
        self._api_keys = dict(api_keys or {})
        if api_key:  # legacy single-key form: the Anthropic slot
            self._api_keys.setdefault("anthropic", api_key)
        self.stats = RunStats()
        self.budget_usd = budget_usd
        self.max_attempts = max(1, max_attempts)
        self.backoff_base = backoff_base
        self.run_id = run_id
        self._semaphore = semaphore or _API_SEMAPHORE
        self._providers: dict[str, Any] = {}

    # ------------------------------------------------------------ providers

    def provider_for(self, model: str) -> Any:
        """The adapter serving ``model``, cached per instance."""
        from taste.pricing import provider_for as _provider_for

        name = _provider_for(model)
        if name not in self._providers:
            self._providers[name] = providers.get(name, api_key=self._api_keys.get(name))
        return self._providers[name]

    def ensure_ready(self, *models: str) -> None:
        """Validate pricing and credentials for the models a run will use.

        Called once at run start, for exactly the providers that run declares
        — so an OpenAI-only run does not require an Anthropic key to exist.
        """
        for model in {m for m in models if m}:
            ensure_priced(model)
            self.provider_for(model).ensure_ready()

    # ------------------------------------------------------------ the call

    def call(
        self,
        *,
        model: str,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = DEFAULT_TEMPERATURE,
        effort: str | None = None,
        role: str = "unspecified",
    ) -> Completion:
        """One model turn, retried on transient failure.

        Raises :class:`PricingError` for unpriced models, :class:`BudgetExceeded`
        at the cap, and :class:`InfraFailure` when the provider cannot serve
        the request. All three are typed so the kernel classifies the run
        rather than crashing.
        """
        ensure_priced(model)
        if self.budget_usd is not None and self.stats.total_cost_usd >= self.budget_usd:
            raise BudgetExceeded(self.stats.total_cost_usd, self.budget_usd)

        provider = self.provider_for(model)
        request = CompletionRequest(
            model=model,
            system=_as_system_blocks(system),
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            sampling=SamplingConfig(temperature=temperature, effort=effort),
            role=role,
            run_id=self.run_id,
        )

        last_exc: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                with self._semaphore:
                    completion = provider.complete(request)
            except ProtocolFailure:
                raise  # already typed and already infra; retrying will not help
            except Exception as exc:
                if not provider.is_retryable(exc):
                    # Credit exhausted, key revoked, malformed request: an
                    # environment problem, surfaced typed so the kernel never
                    # records it as the agent failing its task.
                    raise InfraFailure(
                        "non-retryable provider error", attempts=attempt + 1, last_error=exc
                    ) from exc
                last_exc = exc
                if attempt < self.max_attempts - 1:
                    delay = min(self.backoff_base * (2**attempt), 30.0)
                    time.sleep(delay * (0.5 + random.random()))
                continue
            self.stats.record(model, completion, role=role)
            return completion

        raise InfraFailure(
            "provider failed on transient errors",
            attempts=self.max_attempts,
            last_error=last_exc,
        ) from last_exc


# ---------------------------------------------------------------- prompts


def cached(text: str) -> dict[str, Any]:
    """A cacheable system block (one cache breakpoint)."""
    return {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}


def static_system(*parts: str) -> dict[str, Any]:
    """Join static prompt parts into ONE cacheable block.

    Caching has a minimum cacheable-prefix length; several small blocks each
    below it cache nothing. Consolidating the static prefix is what makes the
    cache actually engage — verify with ``scripts/cache_smoke.py``.
    """
    return cached("\n\n".join(p.strip() for p in parts if p and p.strip()))


def prompt_sha(text: str) -> str:
    """Short stable hash of a prompt, for run manifests."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _as_system_blocks(system: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(system, str):
        return [cached(system)]
    return system


def _find_env(start: Path | None) -> Path | None:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        env = candidate / ".env"
        if env.is_file():
            return env
    return None


__all__ = [
    "DEFAULT_TEMPERATURE",
    "LLM",
    "MODEL_MONITOR",
    "MODEL_PLANNER",
    "MODEL_WORKER",
    "BudgetExceeded",
    "Completion",
    "InfraFailure",
    "ModelUsage",
    "PricingError",
    "RunStats",
    "cached",
    "ensure_priced",
    "prompt_sha",
    "static_system",
]
