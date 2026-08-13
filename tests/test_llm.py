"""Wave-0 LLM layer: retries, typed failures, budget caps, pricing, telemetry.

These guarantees are experimental-integrity load-bearing: infra errors must
never pollute task-failure rates, unknown models must never be silently
mispriced, and per-role attribution must survive concurrent recording.
"""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import httpx
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

# ------------------------------------------------------------------ fixtures


def _message(input_tokens: int = 10, output_tokens: int = 5, cache_read: int = 0):
    return SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=0,
        ),
        content=[],
        stop_reason="end_turn",
    )


def _status_error(code: int) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(code, request=request)
    return anthropic.APIStatusError(f"http {code}", response=response, body=None)


def _llm(create_fn, **kwargs) -> LLM:
    """LLM with a fake client; backoff_base=0 keeps retries instant."""
    llm = LLM(api_key="test-key-not-used", backoff_base=0.0, **kwargs)
    llm.client = SimpleNamespace(messages=SimpleNamespace(create=create_fn))
    return llm


def _call(llm: LLM, **overrides):
    payload = {
        "model": MODEL_WORKER,
        "system": "sys",
        "messages": [{"role": "user", "content": "hi"}],
    }
    payload.update(overrides)
    return llm.call(**payload)


# ------------------------------------------------------------------ retries


def test_retries_transient_errors_then_succeeds() -> None:
    calls = {"n": 0}

    def create(**kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _status_error(529)
        return _message()

    llm = _llm(create)
    message = _call(llm)
    assert message.stop_reason == "end_turn"
    assert calls["n"] == 3
    assert llm.stats.totals.calls == 1  # only the success is recorded


def test_infra_failure_after_exhausted_retries() -> None:
    calls = {"n": 0}

    def create(**kwargs):
        calls["n"] += 1
        raise _status_error(429)

    llm = _llm(create, max_attempts=3)
    with pytest.raises(InfraFailure) as excinfo:
        _call(llm)
    assert calls["n"] == 3
    assert excinfo.value.attempts == 3
    assert llm.stats.totals.calls == 0  # nothing recorded for failed calls


def test_non_retryable_error_becomes_typed_infra_failure_immediately() -> None:
    """Credit exhaustion / auth errors (4xx) must not retry AND must surface
    typed, so the kernel classifies the run as infra instead of crashing raw."""
    calls = {"n": 0}

    def create(**kwargs):
        calls["n"] += 1
        raise _status_error(400)

    llm = _llm(create)
    with pytest.raises(InfraFailure) as excinfo:
        _call(llm)
    assert calls["n"] == 1  # no retries for non-retryable errors
    assert isinstance(excinfo.value.last_error, anthropic.APIStatusError)
    assert excinfo.value.__cause__ is excinfo.value.last_error


# ------------------------------------------------------------------ budget


def test_budget_cap_blocks_next_call() -> None:
    # One call at 1M input tokens on Sonnet ≈ $3 — over a $1 cap.
    llm = _llm(lambda **kwargs: _message(input_tokens=1_000_000), budget_usd=1.0)
    _call(llm)  # first call goes through and records the spend
    with pytest.raises(BudgetExceeded) as excinfo:
        _call(llm)
    assert excinfo.value.spent_usd >= excinfo.value.budget_usd
    assert llm.stats.totals.calls == 1


def test_no_budget_means_no_cap() -> None:
    llm = _llm(lambda **kwargs: _message(input_tokens=1_000_000))
    for _ in range(3):
        _call(llm)
    assert llm.stats.totals.calls == 3


# ------------------------------------------------------------------ pricing


def test_every_default_role_model_is_priced() -> None:
    """A default model with no pricing entry would halt every real run at
    _validate_models — and an unnoticed wrong rate corrupts the cost DV."""
    from taste.llm import MODEL_MONITOR, MODEL_PLANNER, ensure_priced

    for model in (MODEL_PLANNER, MODEL_WORKER, MODEL_MONITOR):
        ensure_priced(model)  # raises PricingError if missing


def test_cache_rate_multipliers_match_published_structure() -> None:
    """Anthropic prices 5m cache writes at 1.25x input and hits at 0.1x.
    A rate that violates this is a transcription error, not a price change."""
    from taste.llm import _PRICING

    for model, r in _PRICING.items():
        assert r["cache_write"] == pytest.approx(r["input"] * 1.25), f"{model} cache_write"
        assert r["cache_read"] == pytest.approx(r["input"] * 0.10), f"{model} cache_read"


def test_unknown_model_raises_before_api_call() -> None:
    calls = {"n": 0}

    def create(**kwargs):
        calls["n"] += 1
        return _message()

    llm = _llm(create)
    with pytest.raises(PricingError, match="no pricing entry"):
        _call(llm, model="mystery-model-9000")
    assert calls["n"] == 0


# ------------------------------------------------------------------ telemetry


def test_per_role_attribution_and_aggregation() -> None:
    llm = _llm(lambda **kwargs: _message())
    _call(llm, role="planner")
    _call(llm, role="worker")
    _call(llm, role="worker")

    assert llm.stats.per_role_model[("planner", MODEL_WORKER)].calls == 1
    assert llm.stats.per_role_model[("worker", MODEL_WORKER)].calls == 2
    # Role-agnostic view (CLI table) still aggregates per model.
    assert llm.stats.per_model[MODEL_WORKER].calls == 3
    assert llm.stats.totals.calls == 3


def test_stats_are_fresh_per_instance() -> None:
    a = _llm(lambda **kwargs: _message())
    b = _llm(lambda **kwargs: _message())
    _call(a)
    assert a.stats.totals.calls == 1
    assert b.stats.totals.calls == 0


def test_cache_hit_rate_uses_cache_read_tokens() -> None:
    stats = RunStats()
    stats.record(MODEL_WORKER, _message(input_tokens=100, cache_read=900))
    assert stats.cache_hit_rate == pytest.approx(0.9)


# ------------------------------------------------------------------ system blocks


def test_static_system_joins_into_single_cached_block() -> None:
    block = static_system("part one", "  ", "part two")
    assert block["type"] == "text"
    assert block["cache_control"] == {"type": "ephemeral"}
    # One consolidated block (single cache breakpoint), blank parts dropped,
    # parts joined in order — pins the consolidation contract exactly.
    assert block["text"] == "part one\n\npart two"


def test_cached_block_shape() -> None:
    block = cached("hello")
    assert block == {
        "type": "text",
        "text": "hello",
        "cache_control": {"type": "ephemeral"},
    }
