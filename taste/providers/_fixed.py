"""A provider that is a fixed rule, not a model: the planner of an agent run alone.

It answers only the planner's requests, from the prompt alone, with nothing
sent anywhere and nothing charged (see ``taste.brains.single_run``).
"""

from __future__ import annotations

from taste.providers.base import Completion, CompletionRequest, ProtocolFailure, Usage


class FixedPlanProvider:
    name = "fixed"
    # No SDK client to close: hosts that own their planner's clients find none.
    _client = None

    def __init__(self, *, api_key=None):
        del api_key  # nothing to authenticate

    def ensure_ready(self) -> None:
        return None

    def complete(self, request: CompletionRequest) -> Completion:
        from taste.brains.single_run import FIXED_PLAN_MODEL, single_run_proposal

        if request.model != FIXED_PLAN_MODEL or request.role != "planner" or len(request.messages) != 1:
            raise ProtocolFailure("the fixed plan answers only the planner's own requests")
        try:
            text = single_run_proposal(request.messages[0]["content"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolFailure(f"the fixed plan cannot answer this request: {exc}") from exc
        return Completion(
            text_blocks=(text,), tool_calls=(), stop_reason="end_turn", model=FIXED_PLAN_MODEL,
            provider=self.name, usage=Usage(0, 0, 0, 0), transcript_blocks=({"type": "text", "text": text},),
            effective_sampling={}, provenance={"route": "fixed"},
        )

    def is_retryable(self, exc: Exception) -> bool:
        return False

    def is_transient(self, exc: Exception) -> bool:
        return False
