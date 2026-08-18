"""OpenAI adapter, on the Responses API.

**Why Responses and not Chat Completions.** The reasoning models reject the
combination this harness needs — function tools together with any reasoning
effort — on ``/v1/chat/completions``, with an error naming ``/v1/responses``
as the place to do it. Since tool use *is* the harness's core operation and
reasoning is the reason to pick these models, Responses is the only surface
that supports the actual workload. Discovered by calling the real API; no
unit test against our own fakes could have found it.

**Reasoning items must be replayed.** With ``store=False`` the provider keeps
no server-side state, so a reasoning item produced before a tool call has to
be sent back on the next turn or the model loses its own train of thought.
They are carried through the canonical transcript as opaque blocks that only
this adapter interprets — other providers skip them.

**The usage subtraction.** OpenAI's ``input_tokens`` is the *total* prompt,
cached portion included; Anthropic's excludes it. Normalizing to disjoint
buckets here is what makes a dollar figure comparable across families;
getting it wrong double-counts every cached token on one side of the study.

**Reasoning tokens are inside ``output_tokens``** and are reported separately
for analysis, never added. They also consume ``max_output_tokens``, which is
why a ceiling sized for a one-word verdict can be spent entirely on thinking
and return nothing — see ``_REASONING_HEADROOM``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from taste.providers.base import (
    Completion,
    CompletionRequest,
    ProtocolFailure,
    ToolCall,
    Usage,
    require,
)

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

# Blocks carrying provider-native items through the canonical transcript.
_NATIVE = "_openai_item"

# Reasoning tokens are billed as output AND drawn from max_output_tokens, so a
# ceiling sized for the visible answer can be consumed entirely by thinking,
# returning an empty message the Monitor would score as a failed step. This
# floor keeps a small verdict call from failing for a reason unrelated to the
# task. It raises the ceiling, not the spend: unused tokens cost nothing.
_REASONING_HEADROOM = 2048

_STOP_REASONS = {
    "completed": "end_turn",
    "incomplete": "max_tokens",
    "failed": "other",
    "cancelled": "other",
}


class OpenAIProvider:
    name = "openai"

    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client: Any = None

    def ensure_ready(self) -> None:
        if not self._api_key:
            raise ProtocolFailure("OPENAI_API_KEY is not set. Put it in .env or export it.")
        self._ensure_client()

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ProtocolFailure(
                    "the `openai` package is not installed; `pip install openai`"
                ) from exc
            # max_retries=0: the facade owns retry policy so it cannot differ
            # between providers or compound with the SDK's own.
            self._client = openai.OpenAI(api_key=self._api_key, max_retries=0)
        return self._client

    # ------------------------------------------------------------ calls

    def complete(self, request: CompletionRequest) -> Completion:
        client = self._ensure_client()

        kwargs: dict[str, Any] = {
            "model": request.model,
            "instructions": _instructions(request.system),
            "input": self._to_input(request.messages),
            "max_output_tokens": max(request.max_tokens, _REASONING_HEADROOM),
            # No server-side state: a run must be reproducible from its own
            # transcript, not from something the provider remembers.
            "store": False,
        }
        if request.tools:
            kwargs["tools"] = [_to_tool(t) for t in request.tools]
        if request.sampling.effort:
            kwargs["reasoning"] = {"effort": request.sampling.effort}

        # temperature is not accepted alongside reasoning on these models.
        # Dropping it silently would leave the manifest claiming a setting
        # that never applied, so the drop is recorded instead.
        dropped = ["temperature"] if request.sampling.temperature is not None else []

        response = client.responses.create(**kwargs)
        return self._to_completion(response, request, dropped)

    def is_retryable(self, exc: Exception) -> bool:
        try:
            import openai
        except ImportError:  # pragma: no cover
            return False
        if isinstance(exc, openai.APIConnectionError):
            return True
        status = getattr(exc, "status_code", None)
        return isinstance(status, int) and status in _RETRYABLE_STATUS

    # ------------------------------------------------------------ request

    def _to_input(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Canonical messages -> Responses ``input`` items."""
        items: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if isinstance(content, str):
                items.append({"role": role, "content": content})
                continue

            for block in content or []:
                kind = block.get("type")
                if kind == _NATIVE:
                    # Reasoning and function_call items, replayed verbatim.
                    items.append(block["item"])
                elif kind == "text":
                    items.append({"role": role, "content": block["text"]})
                elif kind == "tool_result":
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": block["tool_use_id"],
                            "output": str(block.get("content", "")),
                        }
                    )
                # A canonical tool_use block always travels beside its native
                # twin, so it needs no separate translation.
        return items

    # ------------------------------------------------------------ response

    def _to_completion(
        self, response: Any, request: CompletionRequest, dropped: list[str]
    ) -> Completion:
        texts: list[str] = []
        calls: list[ToolCall] = []
        transcript: list[dict[str, Any]] = []

        for item in getattr(response, "output", []) or []:
            kind = getattr(item, "type", None)
            native = _as_dict(item)

            if kind == "message":
                for part in getattr(item, "content", []) or []:
                    if getattr(part, "type", None) == "output_text":
                        texts.append(part.text)
                        transcript.append({"type": "text", "text": part.text})
                # The message item itself is replayed so the model sees its
                # own prior turn in the same shape it produced it.
                transcript.append({"type": _NATIVE, "item": native})

            elif kind == "function_call":
                raw_arguments = getattr(item, "arguments", "") or "{}"
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    # The cores index into these; a string would fail far from
                    # here with a confusing error.
                    raise ProtocolFailure(
                        f"tool call {getattr(item, 'name', '?')!r} had unparseable "
                        f"arguments: {raw_arguments[:200]!r}"
                    ) from exc
                call = ToolCall(
                    id=getattr(item, "call_id", "") or getattr(item, "id", ""),
                    name=getattr(item, "name", ""),
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                )
                calls.append(call)
                transcript.append({"type": _NATIVE, "item": native})
                transcript.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )

            elif kind == "reasoning":
                # Opaque, and required: dropping it with store=False loses the
                # model's own context across a tool call.
                transcript.append({"type": _NATIVE, "item": native})

        stop = _STOP_REASONS.get(getattr(response, "status", "") or "", "other")
        if stop == "max_tokens" and calls:
            # Truncated after a usable tool call: the call is still actionable.
            stop = "tool_use"
        elif calls and stop == "end_turn":
            stop = "tool_use"

        sampling: dict[str, Any] = {"temperature": None, "effort": request.sampling.effort}
        if dropped:
            sampling["dropped"] = dropped

        return Completion(
            text_blocks=tuple(texts),
            tool_calls=tuple(calls),
            stop_reason=stop,
            model=getattr(response, "model", "") or request.model,
            provider=self.name,
            usage=self._to_usage(response.usage),
            transcript_blocks=tuple(transcript),
            effective_sampling=sampling,
            raw=response,
        )

    def _to_usage(self, usage: Any) -> Usage:
        total_prompt = require(usage, "input_tokens", "prompt_tokens")
        output = require(usage, "output_tokens", "completion_tokens")

        cached = 0
        details = getattr(usage, "input_tokens_details", None) or getattr(
            usage, "prompt_tokens_details", None
        )
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)

        reasoning = 0
        out_details = getattr(usage, "output_tokens_details", None) or getattr(
            usage, "completion_tokens_details", None
        )
        if out_details is not None:
            reasoning = int(getattr(out_details, "reasoning_tokens", 0) or 0)

        # THE subtraction: the ledger's input bucket is the uncached remainder.
        return Usage(
            input_tokens=max(total_prompt - cached, 0),
            output_tokens=output,
            cache_read_tokens=cached,
            # Caching is automatic here: prefixes are cached without a write
            # charge, so there is no bucket to bill.
            cache_write_tokens=0,
            reasoning_tokens=reasoning,
            raw={"input_tokens": total_prompt, "cached_tokens": cached},
        )


def _instructions(system: list[dict[str, Any]]) -> str:
    """System blocks collapse to one instructions string.

    ``cache_control`` markers are dropped rather than translated: caching is
    automatic on this API, so carrying them would imply a control it does not
    offer.
    """
    return "\n\n".join(b.get("text", "") for b in system if b.get("text")).strip()


def _to_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Canonical tool schema -> Responses function tool (flat, not nested).

    ``strict`` is left unset deliberately: it constrains the schema dialect,
    and a tool that silently stops being callable is worse than one that
    occasionally needs a retry.
    """
    return {
        "type": "function",
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool.get("input_schema", {"type": "object"}),
    }


def _as_dict(item: Any) -> Any:
    if hasattr(item, "model_dump"):
        try:
            return item.model_dump(exclude_none=True)
        except Exception:
            pass
    return item
