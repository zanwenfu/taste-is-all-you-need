"""Provider registry.

Adapters are constructed lazily and cached per process, so importing this
package costs nothing and an OpenAI-only run never needs an Anthropic key to
exist. Which provider serves a model is decided by the price table, because
a model the ledger cannot price must not be reachable at all.
"""

from __future__ import annotations

from taste.pricing import provider_for
from taste.providers.base import (
    Completion,
    CompletionRequest,
    ProtocolFailure,
    Provider,
    SamplingConfig,
    ToolCall,
    Usage,
    UsageSchemaError,
)

_BUILDERS = {}
_CACHE: dict[str, Provider] = {}


def _register() -> None:
    from taste.providers._anthropic import AnthropicProvider
    from taste.providers._openai import OpenAIProvider

    _BUILDERS["anthropic"] = AnthropicProvider
    _BUILDERS["openai"] = OpenAIProvider


def get(name: str, *, api_key: str | None = None) -> Provider:
    if not _BUILDERS:
        _register()
    if name not in _BUILDERS:
        raise ProtocolFailure(f"unknown provider {name!r}; known: {sorted(_BUILDERS)}")
    if api_key is not None:
        return _BUILDERS[name](api_key=api_key)
    if name not in _CACHE:
        _CACHE[name] = _BUILDERS[name]()
    return _CACHE[name]


def for_model(model: str, *, api_key: str | None = None) -> Provider:
    """The adapter that serves ``model``, per the price table."""
    return get(provider_for(model), api_key=api_key)


def reset_cache() -> None:
    """Drop cached adapters — for tests that swap keys."""
    _CACHE.clear()


__all__ = [
    "Completion",
    "CompletionRequest",
    "ProtocolFailure",
    "Provider",
    "SamplingConfig",
    "ToolCall",
    "Usage",
    "UsageSchemaError",
    "for_model",
    "get",
    "reset_cache",
]
