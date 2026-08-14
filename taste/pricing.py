"""Token prices, and the two ways to add them up.

Cost is a primary dependent variable in this project's experiments, which
makes this table load-bearing: a wrong rate does not crash anything, it
quietly corrupts every comparison built on top of it. So unknown models raise
rather than falling back to a guess, every entry is stamped with the date it
was verified, and the cache multipliers are checked against the published
structure in the tests.

**Two costs, and the difference matters.**

``billed_cost`` is what the invoice says: cached reads are cheap, cache writes
carry a surcharge. ``work_cost`` prices every prompt token at the model's
uncached input rate, as if no cache existed.

The experiments cap on ``work_cost`` and report both. The reason is subtle
and decides whether a result means anything: prefix caches are scoped to an
organization, not to a run, so what a run is billed depends on what ran
*before* it. A cap on billed cost would make each run's budget a function of
its neighbours, which breaks the independence a paired comparison needs.
``work_cost`` is order-invariant and cache-mechanism-agnostic, so it is the
honest unit for a *budget*.

But reporting only ``work_cost`` would flatter the intervention. An arm that
grows one long transcript caches well; an arm that resets destroys its cache
prefix by construction and pays more real money for identical work. Declaring
"rollback wins at matched work" while it spent more dollars is exactly what a
cost-matched study exists to prevent — so both are reported, and their signed
difference is the cache tax the design imposes.

**Tiers.** Some providers surcharge long prompts, and the surcharge applies to
the *whole request*, not the overflow. The tier is therefore selected by that
request's own prompt size, and cost must be accumulated per call — summing
tokens first and multiplying once gives a different, wrong answer as soon as
any call crosses the threshold.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


class PricingError(RuntimeError):
    """No verified price for this model.

    Raised before any request is sent. A silent fallback rate would corrupt
    every cost-matched comparison downstream, so an unpriced model is a
    configuration error, not something to guess at.
    """

    failure_kind = "infra"


@dataclass(frozen=True)
class Rates:
    """USD per 1M tokens."""

    input: float
    output: float
    cache_read: float
    cache_write: float


@dataclass(frozen=True)
class ModelPrice:
    provider: str
    tiers: tuple[tuple[int | None, Rates], ...]
    """(max prompt tokens for this tier, rates). ``None`` means unbounded and
    must be last. Ordered ascending."""
    context_window: int
    as_of: str

    def rates_for(self, prompt_tokens: int) -> Rates:
        for limit, rates in self.tiers:
            if limit is None or prompt_tokens <= limit:
                return rates
        return self.tiers[-1][1]


def _flat(provider: str, i: float, o: float, cr: float, cw: float, *, window: int, as_of: str) -> ModelPrice:
    return ModelPrice(provider, ((None, Rates(i, o, cr, cw)),), window, as_of)


# Anthropic: verified 2026-08-13 against platform.claude.com/docs/en/about-claude/pricing
# Cache write is 1.25x base input (5-minute TTL); a cache read is 0.10x. We
# never request the 1-hour TTL, which is billed at 2x.
_ANTHROPIC = {
    "claude-opus-5":             _flat("anthropic", 5.00, 25.00, 0.50, 6.25, window=200_000, as_of="2026-08-13"),
    "claude-opus-4-7":           _flat("anthropic", 5.00, 25.00, 0.50, 6.25, window=200_000, as_of="2026-08-13"),
    "claude-sonnet-5":           _flat("anthropic", 2.00, 10.00, 0.20, 2.50, window=200_000, as_of="2026-08-13"),
    "claude-sonnet-4-6":         _flat("anthropic", 3.00, 15.00, 0.30, 3.75, window=200_000, as_of="2026-08-13"),
    "claude-haiku-4-5-20251001": _flat("anthropic", 1.00,  5.00, 0.10, 1.25, window=200_000, as_of="2026-08-13"),
}

# OpenAI: verified 2026-08-15 against developers.openai.com/api/docs/pricing
# Long context begins above 272K prompt tokens and repricies the WHOLE
# request: input and cached-input double, output is 1.5x.
_LONG_CONTEXT_THRESHOLD = 272_000


def _tiered(short: Rates, long: Rates, *, window: int, as_of: str) -> ModelPrice:
    return ModelPrice(
        "openai", ((_LONG_CONTEXT_THRESHOLD, short), (None, long)), window, as_of
    )


_OPENAI = {
    "gpt-5.6-sol": _tiered(
        Rates(5.00, 30.00, 0.50, 6.25), Rates(10.00, 45.00, 1.00, 12.50),
        window=1_050_000, as_of="2026-08-15",
    ),
    "gpt-5.6-terra": _tiered(
        Rates(2.00, 12.00, 0.20, 2.50), Rates(4.00, 18.00, 0.40, 5.00),
        window=1_050_000, as_of="2026-08-15",
    ),
    "gpt-5.6-luna": _tiered(
        Rates(0.20, 1.20, 0.02, 0.25), Rates(0.40, 1.80, 0.04, 0.50),
        window=1_050_000, as_of="2026-08-15",
    ),
}

PRICES: dict[str, ModelPrice] = {**_ANTHROPIC, **_OPENAI}

# Names that resolve to a different snapshot over time. A run manifest
# recording an alias cannot say which model actually answered, so an alias is
# refused outright rather than silently pinned.
ALIAS_DENYLIST = frozenset(
    {
        "gpt-5.6",
        "daybreak-blue-latest",
        "gpt-5.6-latest",
        "claude-opus-latest",
        "claude-sonnet-latest",
        "claude-haiku-latest",
        "gemini-pro-latest",
        "gemini-flash-latest",
    }
)


def ensure_priced(model: str) -> ModelPrice:
    """The price for ``model``, or raise. Call before spending anything."""
    if model in ALIAS_DENYLIST:
        raise PricingError(
            f"{model!r} is an alias that repoints over time. Pin an exact "
            "snapshot — a manifest recording an alias cannot say which model ran."
        )
    price = PRICES.get(model)
    if price is None:
        raise PricingError(
            f"no verified price for {model!r}; add it to taste.pricing.PRICES "
            f"(known: {', '.join(sorted(PRICES))}). Refusing to run with "
            "cost telemetry that would be silently wrong."
        )
    return price


def provider_for(model: str) -> str:
    return ensure_priced(model).provider


def rates_for(model: str, prompt_tokens: int) -> Rates:
    """Rates for a request of this prompt size — the tier is per request."""
    return ensure_priced(model).rates_for(prompt_tokens)


def call_cost(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> tuple[float, float]:
    """``(billed_cost, work_cost)`` in USD for one call.

    ``input_tokens`` must be the *uncached* billable prompt tokens; the cached
    buckets are disjoint from it. Adapters are responsible for that
    normalization — providers disagree about whether their reported input
    total includes the cached portion, and getting it wrong double-counts.
    """
    prompt_total = input_tokens + cache_read_tokens + cache_write_tokens
    r = rates_for(model, prompt_total)

    billed = (
        input_tokens * r.input
        + cache_read_tokens * r.cache_read
        + cache_write_tokens * r.cache_write
        + output_tokens * r.output
    ) / 1_000_000
    work = (prompt_total * r.input + output_tokens * r.output) / 1_000_000
    return billed, work


def table_sha() -> str:
    """Digest of the price table, for the run manifest.

    Two runs priced by different tables are not cost-comparable; recording
    this makes that detectable instead of invisible.
    """
    payload = {
        model: {
            "provider": p.provider,
            "as_of": p.as_of,
            "tiers": [
                [limit, [r.input, r.output, r.cache_read, r.cache_write]]
                for limit, r in p.tiers
            ],
        }
        for model, p in sorted(PRICES.items())
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
