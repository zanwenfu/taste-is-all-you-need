"""Prices, tiers, and the two ways of adding them up.

Every assertion here protects a number that appears in the paper. A wrong
rate does not crash anything — it quietly corrupts the comparison built on
top of it — so these tests exist to make the failure loud.
"""

from __future__ import annotations

import pytest

from taste.pricing import (
    _LONG_CONTEXT_THRESHOLD,
    PRICES,
    PricingError,
    call_cost,
    ensure_priced,
    max_call_cost_usd,
    provider_for,
    rates_for,
    request_token_bound,
    table_sha,
)

# ------------------------------------------------------------------ table


def test_every_priced_model_declares_a_verification_date() -> None:
    for model, price in PRICES.items():
        assert price.as_of, f"{model} has no as_of date"
        assert price.provider in {"anthropic", "openai", "fixed"}
        # The only fixed price is the fixed plan, which sends nothing anywhere.
        assert price.provider != "fixed" or model == "taste-fixed-plan"


def test_anthropic_cache_multipliers_match_the_published_structure() -> None:
    """1.25x for a 5-minute cache write, 0.10x for a read. A rate violating
    this is a transcription error, not a price change."""
    for model, price in PRICES.items():
        if price.provider != "anthropic":
            continue
        r = price.tiers[0][1]
        assert r.cache_write == pytest.approx(r.input * 1.25), model
        assert r.cache_read == pytest.approx(r.input * 0.10), model


def test_openai_long_context_doubles_input_and_halves_again_on_output() -> None:
    """Input and cached-input double; output is 1.5x. Not the same factor."""
    for model, price in PRICES.items():
        if price.provider != "openai":
            continue
        short, long = price.tiers[0][1], price.tiers[1][1]
        assert long.input == pytest.approx(short.input * 2), model
        assert long.cache_read == pytest.approx(short.cache_read * 2), model
        assert long.output == pytest.approx(short.output * 1.5), model


def test_gpt_5_6_luna_on_azure_carries_its_dated_global_standard_price() -> None:
    # Azure retail prices for the deployment's version 2026-07-09, Global
    # Standard, checked 2026-10-04: per million tokens, input, output, cached
    # input and cache write, short context and then long.
    price = PRICES["gpt-5.6-luna-2026-07-09"]
    short, long = price.tiers[0][1], price.tiers[1][1]
    assert (short.input, short.output, short.cache_read, short.cache_write) == (0.20, 1.20, 0.02, 0.25)
    assert (long.input, long.output, long.cache_read, long.cache_write) == (0.40, 1.80, 0.04, 0.50)
    assert price.tiers[0][0] == _LONG_CONTEXT_THRESHOLD and price.as_of == "2026-10-04"


def test_gpt_6_luna_on_azure_carries_its_dated_global_standard_price() -> None:
    # Azure retail prices for the deployment's version 2026-09-22, Global
    # Standard, checked 2026-10-07: per million tokens, input, output, cached
    # input and cache write, short context and then long.
    price = PRICES["gpt-6-luna-2026-09-22"]
    short, long = price.tiers[0][1], price.tiers[1][1]
    assert (short.input, short.output, short.cache_read, short.cache_write) == (0.10, 0.50, 0.01, 0.125)
    assert (long.input, long.output, long.cache_read, long.cache_write) == (0.20, 0.75, 0.02, 0.25)
    assert price.tiers[0][0] == _LONG_CONTEXT_THRESHOLD and price.as_of == "2026-10-07"


def test_only_the_fixed_plan_may_cost_nothing() -> None:
    from taste.pricing import ModelPrice, Rates

    assert max_call_cost_usd("taste-fixed-plan", max_output_tokens=16384, cap_on="billed") == 0.0
    zero = ModelPrice("openai", ((None, Rates(0.0, 0.0, 0.0, 0.0)),), 1000, "2026-10-04")
    PRICES["a-model-priced-at-zero"] = zero
    try:
        with pytest.raises(PricingError, match="invalid"):
            max_call_cost_usd("a-model-priced-at-zero", max_output_tokens=10, cap_on="billed")
    finally:
        del PRICES["a-model-priced-at-zero"]


def test_default_role_models_are_priced() -> None:
    from taste.llm import MODEL_MONITOR, MODEL_PLANNER, MODEL_WORKER

    for model in (MODEL_PLANNER, MODEL_WORKER, MODEL_MONITOR):
        ensure_priced(model)


# ------------------------------------------------------------------ refusals


def test_unknown_model_raises_before_anything_is_spent() -> None:
    with pytest.raises(PricingError, match="no verified price"):
        ensure_priced("gpt-9-imaginary")


def test_aliases_are_refused() -> None:
    """A manifest recording an alias cannot say which model actually ran."""
    with pytest.raises(PricingError, match="repoints"):
        ensure_priced("gpt-5.6")
    with pytest.raises(PricingError, match="repoints"):
        ensure_priced("claude-sonnet-latest")


def test_pricing_error_is_classified_as_infra() -> None:
    """A configuration fault must never be recorded as an agent task failure."""
    assert PricingError.failure_kind == "infra"


# ------------------------------------------------------------------ tiers


def test_tier_boundary_is_inclusive_and_reprices_the_whole_request() -> None:
    at = rates_for("gpt-5.6-terra", _LONG_CONTEXT_THRESHOLD)
    over = rates_for("gpt-5.6-terra", _LONG_CONTEXT_THRESHOLD + 1)

    assert at.input == 2.00
    assert over.input == 4.00, "one token over the threshold reprices everything"


def test_flat_priced_models_ignore_prompt_size() -> None:
    small = rates_for("claude-sonnet-4-6", 10)
    huge = rates_for("claude-sonnet-4-6", 500_000)
    assert small == huge


# ------------------------------------------------------------------ the two costs


def test_work_cost_ignores_the_cache_and_billed_cost_does_not() -> None:
    billed, work = call_cost(
        "claude-sonnet-4-6",
        input_tokens=1_000,
        cache_read_tokens=99_000,
        output_tokens=0,
    )
    # work prices all 100k prompt tokens at the full input rate.
    assert work == pytest.approx(100_000 * 3.00 / 1e6)
    # billed pays the cheap rate for the 99k that were cached.
    assert billed == pytest.approx((1_000 * 3.00 + 99_000 * 0.30) / 1e6)
    assert billed < work


def test_a_cold_call_can_cost_more_than_its_work(  ) -> None:
    """The invariant is NOT work >= billed: a cache write is a surcharge.

    This is why the cache delta is reported signed rather than as a saving.
    """
    billed, work = call_cost(
        "claude-sonnet-4-6", input_tokens=0, cache_write_tokens=100_000, output_tokens=0
    )
    assert billed > work


def test_cache_delta_identity_holds_exactly() -> None:
    """work - billed must equal the per-bucket rate differences."""
    model, inp, cr, cw, out = "gpt-5.6-terra", 500, 4_000, 2_000, 300
    billed, work = call_cost(
        model,
        input_tokens=inp,
        cache_read_tokens=cr,
        cache_write_tokens=cw,
        output_tokens=out,
    )
    r = rates_for(model, inp + cr + cw)
    expected = (cr * (r.input - r.cache_read) + cw * (r.input - r.cache_write)) / 1e6
    assert (work - billed) == pytest.approx(expected)


def test_per_call_accumulation_differs_from_summing_tokens_first() -> None:
    """Why cost must accumulate per call once a tier can be crossed.

    Two small calls priced individually are cheaper than one call carrying
    their combined prompt across the threshold.
    """
    half = _LONG_CONTEXT_THRESHOLD // 2 + 1
    per_call = sum(
        call_cost("gpt-5.6-terra", input_tokens=half, output_tokens=100)[0] for _ in range(2)
    )
    combined = call_cost("gpt-5.6-terra", input_tokens=half * 2, output_tokens=200)[0]
    assert combined > per_call, "the surcharge applies to the whole request"


def test_reasoning_tokens_are_not_added_to_output() -> None:
    """Reasoning tokens are a reported SUBSET of output, never an addition."""
    a, _ = call_cost("gpt-5.6-terra", input_tokens=10, output_tokens=100)
    b, _ = call_cost(
        "gpt-5.6-terra", input_tokens=10, output_tokens=100, reasoning_tokens=80
    )
    assert a == b


def test_zero_usage_costs_nothing() -> None:
    assert call_cost("claude-sonnet-4-6", input_tokens=0, output_tokens=0) == (0.0, 0.0)


def test_max_call_cost_prices_full_context_output_and_retries() -> None:
    price = PRICES["claude-haiku-4-5-20251001"]
    expected_once = (
        price.context_window * 1.25 + 1024 * 5.00
    ) / 1_000_000
    assert max_call_cost_usd(
        "claude-haiku-4-5-20251001",
        max_output_tokens=1024,
    ) == pytest.approx(expected_once)
    assert max_call_cost_usd(
        "claude-haiku-4-5-20251001",
        max_output_tokens=1024,
        max_attempts=3,
    ) == pytest.approx(expected_once * 3)


def test_max_call_cost_uses_the_selected_cap_currency() -> None:
    billed = max_call_cost_usd(
        "claude-sonnet-4-6",
        max_output_tokens=1,
        cap_on="billed",
    )
    work = max_call_cost_usd(
        "claude-sonnet-4-6",
        max_output_tokens=1,
        cap_on="work",
    )
    assert billed > work, "cache-write rates are the worst billed prompt bucket"


def test_a_call_is_bounded_by_the_request_it_sends_when_that_is_known() -> None:
    # gpt-6-astra: a prompt token costs at most 12.50 a million (a cache write)
    # in a request of up to 272,000, and 25.00 in a longer one, where output
    # also rises from 50 to 75. Bounded by the whole 1,050,000-token window a
    # call's worst case was $27.48, whatever it sent.
    model = "gpt-6-astra-2026-09-03"
    assert max_call_cost_usd(model, max_output_tokens=16_384) == pytest.approx(27.4788)
    assert max_call_cost_usd(
        model, max_output_tokens=16_384, max_prompt_tokens=200_000
    ) == pytest.approx(3.3192)
    # The boundary is inclusive, and one token more reprices the whole request.
    assert max_call_cost_usd(
        model, max_output_tokens=16_384, max_prompt_tokens=272_000
    ) == pytest.approx(4.2192)
    assert max_call_cost_usd(
        model, max_output_tokens=16_384, max_prompt_tokens=272_001
    ) == pytest.approx(8.028825)
    # A bound beyond the model's window is the window.
    assert max_call_cost_usd(
        model, max_output_tokens=16_384, max_prompt_tokens=9_000_000
    ) == pytest.approx(27.4788)
    # The work currency prices the same bound at the uncached input rate.
    assert max_call_cost_usd(
        model, max_output_tokens=16_384, max_prompt_tokens=200_000, cap_on="work"
    ) == pytest.approx(2.8192)


def test_a_request_holds_no_more_tokens_than_its_bytes_and_an_allowance() -> None:
    assert request_token_bound(0) == 4_096
    assert request_token_bound(100_000) == 104_096
    for invalid in (-1, True, 1.5, None):
        with pytest.raises(ValueError, match="request_bytes"):
            request_token_bound(invalid)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_output_tokens": 0}, "max_output_tokens"),
        ({"max_output_tokens": 1, "max_attempts": False}, "max_attempts"),
        ({"max_output_tokens": 1, "cap_on": "tokens"}, "cap_on"),
        ({"max_output_tokens": 1, "max_prompt_tokens": 0}, "max_prompt_tokens"),
        ({"max_output_tokens": 1, "max_prompt_tokens": True}, "max_prompt_tokens"),
    ],
)
def test_max_call_cost_rejects_an_unbounded_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        max_call_cost_usd("claude-haiku-4-5-20251001", **kwargs)


# ------------------------------------------------------------------ provenance


def test_provider_resolution() -> None:
    assert provider_for("claude-sonnet-4-6") == "anthropic"
    assert provider_for("gpt-5.6-terra") == "openai"


def test_table_sha_is_stable_and_sensitive() -> None:
    """Two runs priced by different tables are not cost-comparable."""
    first = table_sha()
    assert first == table_sha()

    original = PRICES["gpt-5.6-terra"]
    try:
        PRICES["gpt-5.6-terra"] = original.__class__(
            provider=original.provider,
            tiers=((None, rates_for("gpt-5.6-luna", 10)),),
            context_window=original.context_window,
            as_of=original.as_of,
        )
        assert table_sha() != first
    finally:
        PRICES["gpt-5.6-terra"] = original
    assert table_sha() == first
