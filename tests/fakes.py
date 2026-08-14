"""Scripted LLM stand-in, so the real cores can be tested with zero API calls.

The kernel's existing hermetic tests bypass the model entirely via
``plan_override`` / ``worker_override``. That leaves the code that actually
talks to the model — :func:`taste.cores.execute`'s tool-use loop — untested.
:class:`FakeLLM` closes that gap: it presents the same surface the real
:class:`taste.llm.LLM` does (``.call`` returning a Message-shaped object,
plus a real ``.stats``), so the cores run unmodified.

Responses are scripted in order. Each one records real token usage into a
real :class:`~taste.llm.RunStats`, so cost and cache telemetry are exercised
too — not just control flow.

    llm = FakeLLM([
        FakeTurn(tool_calls=[("write_file", {"path": "a.py", "content": "x"})]),
        FakeTurn(text="done"),
    ])
    result = cores.execute(llm, spec=spec, step=step, plan_context="", tools=tools)
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from taste.llm import MODEL_WORKER, RunStats


@dataclass
class FakeTurn:
    """One scripted model turn.

    ``stop_reason`` is inferred when omitted: ``tool_use`` if the turn calls
    tools, ``end_turn`` otherwise — matching what the API actually returns.
    """

    text: str | None = None
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    stop_reason: str | None = None
    input_tokens: int = 100
    output_tokens: int = 50
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def resolved_stop_reason(self) -> str:
        if self.stop_reason is not None:
            return self.stop_reason
        return "tool_use" if self.tool_calls else "end_turn"


class ScriptExhausted(RuntimeError):
    """The code under test made more calls than the script provides.

    Loud on purpose: silently repeating the last turn would let an infinite
    loop in the core under test masquerade as a passing test.
    """


class FakeLLM:
    """A scripted stand-in for :class:`taste.llm.LLM`.

    Parameters
    ----------
    turns:
        Scripted responses, consumed in order.
    model:
        Recorded in stats. Must be a priced model id, since ``RunStats``
        computes cost eagerly and rejects unknown models.
    """

    def __init__(
        self,
        turns: list[FakeTurn] | None = None,
        *,
        model: str = MODEL_WORKER,
    ) -> None:
        self._turns = list(turns or [])
        self._next = itertools.count()
        self.model = model
        self.stats = RunStats()
        # A snapshot of every call's kwargs — what was actually sent at that
        # moment, not what the transcript later became.
        self.calls: list[dict[str, Any]] = []

    # ------------------------------------------------------------ surface

    def call(self, **kwargs: Any) -> SimpleNamespace:
        # Deep copy, because cores.execute mutates ONE messages list in place
        # across turns. Storing it by reference would make every recorded call
        # show the final transcript, silently voiding per-call assertions.
        self.calls.append(copy.deepcopy(kwargs))
        index = next(self._next)
        if index >= len(self._turns):
            raise ScriptExhausted(
                f"FakeLLM ran out of scripted turns: call #{index + 1} was made "
                f"but only {len(self._turns)} were scripted"
            )
        turn = self._turns[index]
        message = _message_for(turn, call_index=index)
        self.stats.record(
            kwargs.get("model", self.model),
            message,
            role=kwargs.get("role", "unspecified"),
        )
        return message

    # ------------------------------------------------------------ helpers

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def roles_called(self) -> list[str]:
        return [c.get("role", "unspecified") for c in self.calls]

    def last_user_content(self) -> Any:
        """Content of the final user message sent — the feedback channel."""
        messages = self.calls[-1]["messages"]
        for msg in reversed(messages):
            if msg["role"] == "user":
                return msg["content"]
        raise AssertionError("no user message in the last call")


def _message_for(turn: FakeTurn, *, call_index: int) -> SimpleNamespace:
    """Build an object shaped like ``anthropic.types.Message``.

    Only the attributes the cores actually read are provided: block ``.type``
    / ``.text`` / ``.id`` / ``.name`` / ``.input``, plus ``.stop_reason`` and
    a ``.usage`` carrying the same field names the SDK uses.
    """
    blocks: list[SimpleNamespace] = []
    if turn.text is not None:
        blocks.append(SimpleNamespace(type="text", text=turn.text))
    for tool_index, (name, payload) in enumerate(turn.tool_calls):
        blocks.append(
            SimpleNamespace(
                type="tool_use",
                id=f"toolu_{call_index:02d}_{tool_index:02d}",
                name=name,
                input=payload,
            )
        )
    return SimpleNamespace(
        content=blocks,
        stop_reason=turn.resolved_stop_reason(),
        usage=SimpleNamespace(
            input_tokens=turn.input_tokens,
            output_tokens=turn.output_tokens,
            cache_read_input_tokens=turn.cache_read_tokens,
            cache_creation_input_tokens=turn.cache_creation_tokens,
        ),
    )


def plan_turn(steps: list[dict[str, Any]], **usage: int) -> FakeTurn:
    """A scripted Planner response calling ``submit_plan``."""
    return FakeTurn(tool_calls=[("submit_plan", {"steps": steps})], **usage)


def verdict_turn(*, passed: bool, reason: str = "scripted", **usage: int) -> FakeTurn:
    """A scripted Monitor response calling ``report_verdict``."""
    return FakeTurn(
        tool_calls=[("report_verdict", {"passed": passed, "reason": reason})], **usage
    )
