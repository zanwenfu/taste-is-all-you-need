"""The provider boundary: one protocol, and the types that cross it.

**The Anthropic block shape is the internal IR, not the provider.** System
blocks, transcript messages and tool schemas stay in ``{"type": "text", ...}``
/ ``{"name", "description", "input_schema"}`` form everywhere in the kernel
and cores; non-Anthropic adapters translate to their native shape on every
call and back again on the way out. This is a deliberate choice rather than a
leaked abstraction: it means adding a provider touches one file instead of
every call site, and the 200-odd existing tests keep their meaning.

**Disjoint prompt buckets are the whole ballgame for cost.** Providers
disagree about whether the "input tokens" they report include the cached
portion. Anthropic's excludes it; OpenAI's includes it. Normalizing to
disjoint buckets — uncached, cache-read, cache-write — is what makes a dollar
figure comparable across families, and getting it wrong double-counts every
cached token in the study.

Adapters own translation and nothing else. Retries, budget caps, pricing and
telemetry live in the facade above them, so those behaviors cannot drift
between providers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

StopReason = Literal[
    "end_turn", "tool_use", "max_tokens", "refusal", "content_filter", "interrupted", "other"
]


class ProtocolFailure(RuntimeError):
    """The provider replied, but not in a shape the harness can use.

    Distinct from an API error: the request succeeded. Classified as infra
    because it is our integration at fault, never the agent's work.
    """

    failure_kind = "infra"


class UsageSchemaError(ProtocolFailure):
    """A usage field the cost ledger depends on was absent.

    Absent is not zero. Silently treating a missing counter as zero would
    understate cost invisibly, so it is an error.
    """


@dataclass(frozen=True)
class Usage:
    """Token counts with **disjoint** prompt buckets.

    ``input_tokens`` is defined as *uncached billable prompt tokens*.
    ``output_tokens`` is the billable superset and already includes
    ``reasoning_tokens``, which is reported for analysis and must never be
    added to anything.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def prompt_total(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


@dataclass(frozen=True)
class ToolCall:
    id: str
    """The provider's own call id, echoed back verbatim in the result. A
    mismatch is an API error, not a nicety."""
    name: str
    arguments: dict[str, Any]
    """Always decoded. OpenAI hands back a JSON string; the adapter parses it
    so the cores never have to know which provider they are talking to."""
    raw_arguments: str | None = None


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float | None = None
    effort: str | None = None
    verbosity: str | None = None


@dataclass(frozen=True)
class CompletionRequest:
    model: str
    system: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    max_tokens: int
    sampling: SamplingConfig
    role: str
    run_id: str = ""


@dataclass(frozen=True)
class Completion:
    """One model turn, in the harness's own vocabulary."""

    text_blocks: tuple[str, ...]
    tool_calls: tuple[ToolCall, ...]
    stop_reason: StopReason
    model: str
    """The id the provider echoed back — not the one requested. They differ
    when an alias resolves, which is worth catching."""
    provider: str
    usage: Usage
    transcript_blocks: tuple[dict[str, Any], ...]
    """The assistant turn to append, in canonical form."""
    effective_sampling: Mapping[str, Any] = field(default_factory=dict)
    """What was actually sent, for the manifest — a provider that rejects or
    ignores a parameter must not leave the record claiming otherwise."""
    raw: Any = field(default=None, compare=False, repr=False)

    @property
    def summary_text(self) -> str:
        return "\n".join(self.text_blocks).strip()


class Provider(Protocol):
    """Translate a request, translate the response. Nothing else."""

    name: str

    def ensure_ready(self) -> None:
        """Raise if this provider cannot be used — missing key, missing SDK.

        Called once per run for the providers a run actually declares, so an
        OpenAI-only run does not require an Anthropic key to exist.
        """
        ...

    def complete(self, request: CompletionRequest) -> Completion: ...

    def is_retryable(self, exc: Exception) -> bool:
        """Whether this exception is worth another attempt.

        Provider-specific because the exception hierarchies differ, but the
        retry loop itself lives in the facade so the policy cannot diverge.
        """
        ...


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def tool_use_block(call: ToolCall) -> dict[str, Any]:
    return {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}


def require(usage: Any, *names: str) -> int:
    """Read the first present usage counter, or raise.

    ``None`` coerces to zero — providers legitimately send null for a bucket
    that saw no traffic. A wholly absent attribute is a schema change, and
    guessing at it would misprice silently.
    """
    for name in names:
        if hasattr(usage, name):
            return int(getattr(usage, name) or 0)
        if isinstance(usage, dict) and name in usage:
            return int(usage[name] or 0)
    raise UsageSchemaError(
        f"usage reported none of {names!r}; the cost ledger cannot be trusted "
        "until the adapter is updated"
    )
