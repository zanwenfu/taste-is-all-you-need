"""One probe with memory tools: the model may search and fetch before it answers, within stated limits.

Every call and every answer is kept whole, so "looks" (did it search), "finds" (did what came back hold the rule)
and "uses" (does the answer obey it) are read from the record and never inferred. A model that keeps calling tools
past the round limit is asked once more with no tools offered. Tool output past the memory budget is refused
rather than cut, so the model knows it got nothing and the budget is the same for every arm.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from taste.memstore.tools import FETCH, SEARCH


@dataclass
class ToolRun:
    final: str = ""
    finish_reason: str = ""
    calls: list[dict] = field(default_factory=list)
    searches: int = 0
    fetches: int = 0
    rounds: int = 0
    tool_errors: int = 0
    unparsed_arguments: int = 0
    tool_tokens: int = 0
    budget_refusals: int = 0
    round_limit_hit: bool = False
    action: dict | None = None
    usage: list[tuple[int, int]] = field(default_factory=list)


def _error(text: str) -> str:
    return json.dumps({"error": text})


def ask_with_tools(chat: Callable[[list[dict], list | None], Any], messages: list[dict], tools: list[dict],
                   answer: Callable[[str, dict], tuple[str, bool]], *, max_rounds: int, budget: int,
                   count: Callable[[str], int], actions: frozenset[str] = frozenset()) -> ToolRun:
    """Until the model answers. A call to one of ``actions`` is what the agent chose to do, and ends the run there."""
    sent = [dict(message) for message in messages]
    run = ToolRun()
    for round_number in range(max_rounds + 1):
        offered = tools if round_number < max_rounds else None
        reply = chat(sent, offered)
        choice = reply.choices[0]
        run.usage.append((reply.usage.prompt_tokens, reply.usage.completion_tokens))
        calls = choice.message.tool_calls or []
        if not calls or offered is None:
            run.final, run.finish_reason = choice.message.content or "", choice.finish_reason
            run.round_limit_hit = offered is None
            return run
        run.rounds += 1
        sent.append({"role": "assistant", "content": choice.message.content,
                     "tool_calls": [{"id": c.id, "type": "function",
                                     "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                    for c in calls]})
        for call in calls:
            name = call.function.name
            if name in actions:
                run.action = {"name": name, "arguments": call.function.arguments}
                run.final, run.finish_reason = choice.message.content or "", choice.finish_reason
                return run
            run.searches += name == SEARCH
            run.fetches += name == FETCH
            try:
                arguments = json.loads(call.function.arguments or "{}")
                parsed = isinstance(arguments, dict)
            except json.JSONDecodeError:
                parsed = False
            if not parsed:
                output, failed = _error("arguments must be one JSON object"), True
                run.unparsed_arguments += 1
            else:
                output, failed = answer(name, arguments)
                if not failed and run.tool_tokens + count(output) > budget:
                    output, failed = _error("the memory budget for this question is used up"), True
                    run.budget_refusals += 1
                elif not failed:
                    run.tool_tokens += count(output)
            run.tool_errors += failed
            run.calls.append({"round": round_number, "name": name, "arguments": call.function.arguments,
                              "output": output, "failed": failed})
            sent.append({"role": "tool", "tool_call_id": call.id, "content": output})
    raise AssertionError("unreachable: the last round offers no tools")


def effort_arg(value: str) -> str | None:
    """A command line's reasoning effort: "none" sends none, for a model that takes no such setting."""
    return None if value == "none" else value


def served_chat(client: Any, model: str, *, max_tokens: int, effort: str | None = "low") -> Callable[[list[dict], list | None], Any]:
    """The served model through its OpenAI-compatible endpoint, as the rule-loss package asks it."""

    def chat(messages: list[dict], tools: list | None) -> Any:
        extra = {"tools": tools, "tool_choice": "auto"} if tools else {}
        if effort is not None:
            extra["reasoning_effort"] = effort
        return client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, **extra)

    return chat


def _reply(raw: dict) -> SimpleNamespace:
    choice = raw["choices"][0]
    message = choice.get("message") or {}
    calls = [SimpleNamespace(id=call.get("id", ""), type="function",
                             function=SimpleNamespace(name=call["function"]["name"],
                                                      arguments=call["function"].get("arguments") or ""))
             for call in message.get("tool_calls") or []]
    usage = raw.get("usage") or {}
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=message.get("content"), tool_calls=calls or None),
                                 finish_reason=choice.get("finish_reason") or "")],
        usage=SimpleNamespace(prompt_tokens=usage.get("prompt_tokens", 0),
                              completion_tokens=usage.get("completion_tokens", 0)))


def http_chat(url: str, model: str, *, max_tokens: int, effort: str | None = "low", timeout: float = 1800,
              tries: int = 3) -> Callable[[list[dict], list | None], Any]:
    """The same call as ``served_chat`` over plain HTTP, for an environment with no OpenAI package. A refusal
    by the server (4xx) is raised at once; a dropped connection or a server error is tried again. An effort of
    None sends none."""
    endpoint = url.rstrip("/") + "/chat/completions"

    def chat(messages: list[dict], tools: list | None) -> Any:
        body = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if effort is not None:
            body["reasoning_effort"] = effort
        if tools:
            body |= {"tools": tools, "tool_choice": "auto"}
        request = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        for attempt in range(tries):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return _reply(json.load(response))
            except urllib.error.HTTPError as error:
                if error.code < 500 or attempt == tries - 1:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == tries - 1:
                    raise
            time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    return chat


def grade(text: str, letter: str) -> bool | None:
    """The package's rule: the last non-empty line is the answer; anything but A or B is unreadable."""
    lines = [line for line in (text or "").strip().splitlines() if line.strip()]
    answer = lines[-1].strip().upper() if lines else ""
    return (answer == letter) if answer in ("A", "B") else None


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def holds(value: str, text: str) -> bool:
    """Whether ``value`` holds all of ``text``, whatever the case: the benchmark's rule text starts with a capital and
    its histories hold it lower-cased, mid-sentence."""
    return text.casefold() in value.casefold()


def found_in(run: ToolRun, text: str) -> bool:
    """Whether the whole of ``text`` reached the model in any one piece of tool output."""
    for call in run.calls:
        try:
            values = list(_strings(json.loads(call["output"])))
        except json.JSONDecodeError:
            values = [call["output"]]
        if any(holds(value, text) for value in values):
            return True
    return False
