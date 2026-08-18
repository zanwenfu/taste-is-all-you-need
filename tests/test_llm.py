"""The model facade: retries, typed failures, budget caps, telemetry.

These behaviors live above the provider boundary precisely so they cannot
drift between families — so they are tested there too, against a scripted
provider rather than a scripted wire format.
"""

from __future__ import annotations

import pytest

from taste.llm import (
    LLM,
    MODEL_WORKER,
    BudgetExceeded,
    InfraFailure,
    PricingError,
    RunStats,
    cached,
    static_system,
)
from taste.providers.base import Completion, Usage
from tests.fakes import FakeProvider, FakeTurn


def _llm(provider: FakeProvider, **kwargs) -> LLM:
    """A facade wired to a scripted provider. backoff_base=0 keeps it instant."""
    llm = LLM(api_key="unused-in-tests", backoff_base=0.0, **kwargs)
    llm._providers["anthropic"] = provider
    return llm


def _call(llm: LLM, **overrides):
    payload = {
        "model": MODEL_WORKER,
        "system": "sys",
        "messages": [{"role": "user", "content": "hi"}],
    }
    payload.update(overrides)
    return llm.call(**payload)


def _usage_completion(**usage) -> Completion:
    return Completion(
        text_blocks=("ok",),
        tool_calls=(),
        stop_reason="end_turn",
        model=MODEL_WORKER,
        provider="fake",
        usage=Usage(**usage),
        transcript_blocks=(),
    )


# ------------------------------------------------------------------ retries


def test_retries_transient_errors_then_succeeds() -> None:
    provider = FakeProvider(
        [FakeTurn(text="done")],
        errors=[RuntimeError("overloaded"), RuntimeError("overloaded"), None],
    )
    completion = _call(_llm(provider))

    assert completion.stop_reason == "end_turn"
    assert provider.call_count == 3


def test_infra_failure_after_exhausted_retries() -> None:
    provider = FakeProvider(errors=[RuntimeError("boom")] * 5)
    llm = _llm(provider, max_attempts=3)

    with pytest.raises(InfraFailure) as excinfo:
        _call(llm)
    assert provider.call_count == 3
    assert excinfo.value.attempts == 3
    assert llm.stats.totals.calls == 0, "a failed call records no usage"


def test_non_retryable_error_is_typed_and_immediate() -> None:
    """Credit exhaustion or a revoked key is an environment fault, and must
    never be recorded as the agent failing its task."""
    provider = FakeProvider(errors=[RuntimeError("insufficient balance")], retryable=False)
    llm = _llm(provider)

    with pytest.raises(InfraFailure) as excinfo:
        _call(llm)
    assert provider.call_count == 1, "non-retryable means no retries"
    assert excinfo.value.failure_kind == "infra"


# ------------------------------------------------------------------ budget


def test_budget_cap_blocks_the_next_call() -> None:
    # 1M input tokens on Sonnet 4.6 is $3 — over a $1 cap.
    provider = FakeProvider([FakeTurn(text="x", input_tokens=1_000_000)])
    llm = _llm(provider, budget_usd=1.0)

    _call(llm)
    with pytest.raises(BudgetExceeded) as excinfo:
        _call(llm)
    assert excinfo.value.spent_usd >= excinfo.value.budget_usd
    assert provider.call_count == 1, "the capped call must not reach the provider"


def test_no_budget_means_no_cap() -> None:
    provider = FakeProvider([FakeTurn(text="x", input_tokens=1_000_000)])
    llm = _llm(provider)
    for _ in range(3):
        _call(llm)
    assert llm.stats.totals.calls == 3


# ------------------------------------------------------------------ pricing gate


def test_unknown_model_raises_before_reaching_the_provider() -> None:
    provider = FakeProvider([FakeTurn(text="x")])
    llm = _llm(provider)

    with pytest.raises(PricingError):
        _call(llm, model="mystery-model-9000")
    assert provider.call_count == 0


def test_default_role_models_are_priced() -> None:
    from taste.llm import MODEL_MONITOR, MODEL_PLANNER, ensure_priced

    for model in (MODEL_PLANNER, MODEL_WORKER, MODEL_MONITOR):
        ensure_priced(model)


# ------------------------------------------------------------------ telemetry


def test_per_role_attribution_and_aggregation() -> None:
    llm = _llm(FakeProvider([FakeTurn(text="x")]))
    _call(llm, role="planner")
    _call(llm, role="worker")
    _call(llm, role="worker")

    assert llm.stats.per_role_model[("planner", MODEL_WORKER)].calls == 1
    assert llm.stats.per_role_model[("worker", MODEL_WORKER)].calls == 2
    assert llm.stats.per_model[MODEL_WORKER].calls == 3
    assert llm.stats.totals.calls == 3


def test_stats_are_fresh_per_instance() -> None:
    """One RunStats per run, or cost attribution bleeds across trials."""
    a = _llm(FakeProvider([FakeTurn(text="x")]))
    b = _llm(FakeProvider([FakeTurn(text="x")]))
    _call(a)
    assert a.stats.totals.calls == 1
    assert b.stats.totals.calls == 0


def test_cache_hit_rate_uses_cache_read_tokens() -> None:
    stats = RunStats()
    stats.record(MODEL_WORKER, _usage_completion(input_tokens=100, cache_read_tokens=900))
    assert stats.cache_hit_rate == pytest.approx(0.9)


def test_billed_and_work_costs_are_both_tracked() -> None:
    """Both currencies, because an arm that resets destroys its cache prefix
    and pays more real money for identical work."""
    stats = RunStats()
    stats.record(MODEL_WORKER, _usage_completion(input_tokens=1_000, cache_read_tokens=99_000))

    assert stats.total_work_usd > stats.total_cost_usd
    assert stats.cache_delta_usd == pytest.approx(
        stats.total_work_usd - stats.total_cost_usd
    )


def test_cache_delta_is_signed_not_a_saving() -> None:
    """A cold call with cache writes costs MORE than its work."""
    stats = RunStats()
    stats.record(MODEL_WORKER, _usage_completion(cache_write_tokens=100_000))
    assert stats.cache_delta_usd < 0


def test_reasoning_tokens_are_recorded_but_not_double_counted() -> None:
    stats = RunStats()
    stats.record(MODEL_WORKER, _usage_completion(output_tokens=100, reasoning_tokens=80))
    assert stats.totals.reasoning_tokens == 80
    assert stats.totals.output_tokens == 100


# ------------------------------------------------------------------ prompts


def test_static_system_joins_into_a_single_cached_block() -> None:
    block = static_system("part one", "  ", "part two")
    assert block["type"] == "text"
    assert block["cache_control"] == {"type": "ephemeral"}
    assert block["text"] == "part one\n\npart two"


def test_cached_block_shape() -> None:
    assert cached("hello") == {
        "type": "text",
        "text": "hello",
        "cache_control": {"type": "ephemeral"},
    }


def test_string_system_is_normalized_to_a_block() -> None:
    provider = FakeProvider([FakeTurn(text="x")])
    _call(_llm(provider), system="plain string")
    assert provider.calls[0].system == [cached("plain string")]
