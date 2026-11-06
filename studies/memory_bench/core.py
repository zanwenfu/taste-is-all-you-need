"""What every system in the comparison shares: ranked items, one token budget, one reader prompt."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

# LongMemEval's own reader prompt with its reading method on, byte for byte
# (src/generation/run_generation.py at commit 9e0b455, lines 55 and 252; "json" history format).
_READER = (
    "I will give you several history chats between you and a user. Please answer the question based on the "
    "relevant chat history. Answer the question step by step: first extract all the relevant information, and "
    "then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\n"
    "Question: {}\nAnswer (step by step):"
)
_CHUNK = "\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n"


@dataclass(frozen=True)
class Item:
    """One thing a system offers the reader, in its rank order."""

    session: str  # neutral id of the session this came from, as far as the system can say
    date: str
    turns: tuple[dict, ...]  # as the reader's prompt shows them
    rounds: tuple[int, ...] = ()  # which rounds of that session; empty when the system cannot say
    source: str = ""  # the system's own handle on it


@dataclass(frozen=True)
class Delivered:
    items: tuple[Item, ...]
    tokens: int
    offered: int
    cut: int  # items delivered in part
    dropped: int  # items offered and not delivered at all


def chunk_text(item: Item, number: int = 1) -> str:
    return _CHUNK.format(number, item.date, "\n" + json.dumps(list(item.turns)))


def item_text(item: Item) -> str:
    """What a budget charges: the memory's own words, with no heading or role, so a system that returns many small pieces
    pays the same per word as one that returns a few large ones."""
    return "\n".join(str(turn["content"]) for turn in item.turns)


def deliver(items: Iterable[Item], budget: int, count: Callable[[str], int],
            render: Callable[[Item], str] = item_text) -> Delivered:
    """Fill the budget in the system's own rank order, in tokens, and stop at the first item that does
    not fit whole. That item is cut between turns, never inside one, and the cut part cannot claim its
    rounds: what arrived is no longer known round by round. A row that must fit the reader's window charges what
    the prompt will hold instead, through ``render``."""
    offered = list(items)
    taken: list[Item] = []
    used = cut = 0
    for position, item in enumerate(offered):
        cost = count(render(item))
        if used + cost <= budget:
            taken.append(item)
            used += cost
            continue
        for keep in range(len(item.turns) - 1, 0, -1):
            part = Item(item.session, item.date, item.turns[:keep], (), item.source)
            cost = count(render(part))
            if used + cost <= budget:
                taken.append(part)
                used += cost
                cut = 1
                break
        return Delivered(tuple(taken), used, len(offered), cut, len(offered) - position - cut)
    return Delivered(tuple(taken), used, len(offered), 0, 0)


def reader_prompt(items: Iterable[Item], question_date: str, question: str) -> str:
    """The benchmark's prompt. Like its own script, it shows what was retrieved in date order. What came from one session
    sits under one heading in that session's own order, as LoCoMo's prompt does, so that pieces of a session are not shown
    as sessions of their own."""
    groups: dict[str, list[Item]] = {}
    for item in items:
        groups.setdefault(item.session, []).append(item)
    history = []
    for number, parts in enumerate(sorted(groups.values(), key=lambda parts: parts[0].date), 1):
        parts = sorted(parts, key=lambda part: part.rounds[0] if part.rounds else 1 << 30)
        turns = [turn for part in parts for turn in part.turns]
        history.append(_CHUNK.format(number, parts[0].date, "\n" + json.dumps(turns)))
    return _READER.format("".join(history), question_date, question)


def rankings_line(qid: str, system: str, items: Sequence[Item], *, ingest_seconds: float, retrieve_seconds: float,
                  ledger: dict[str, Any], **extra: Any) -> str:
    """One question's ranked list, as an adapter in another environment writes it."""
    return json.dumps({"qid": qid, "system": system, "items": [asdict(item) for item in items],
                       "ingest_seconds": ingest_seconds, "retrieve_seconds": retrieve_seconds, "ledger": ledger, **extra})


def no_memory_prompt(question: str) -> str:
    """The benchmark's own form for a reader given no history (run_generation.py lines 48-50)."""
    return question + "Answer step by step."


def fill(texts: list[str], cap: int, count: Callable[[str], int]) -> str:
    """In the system's own order until the cap. The one that does not fit is cut between words, and filling stops."""
    kept, used = [], 0
    for text in texts:
        size = count(text)
        if used + size <= cap:
            kept.append(text)
            used += size
            continue
        words = text.split(" ")
        low, high = 0, len(words)
        while low < high:
            middle = (low + high + 1) // 2
            if count(" ".join(words[:middle])) <= cap - used:
                low = middle
            else:
                high = middle - 1
        if low:
            kept.append(" ".join(words[:low]))
        break
    return "\n\n".join(kept)
