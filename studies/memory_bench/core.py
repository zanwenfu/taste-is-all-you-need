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


def deliver(items: Iterable[Item], budget: int, count: Callable[[str], int],
            render: Callable[[Item], str] = chunk_text) -> Delivered:
    """Fill the budget in the system's own rank order, in tokens, and stop at the first item that does
    not fit whole. That item is cut between turns, never inside one, and the cut part cannot claim its
    rounds: what arrived is no longer known round by round."""
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
    """The benchmark's prompt. Like its own script, it shows what was retrieved in date order."""
    ordered = sorted(items, key=lambda item: item.date)
    history = "".join(chunk_text(item, number) for number, item in enumerate(ordered, 1))
    return _READER.format(history, question_date, question)


def rankings_line(qid: str, system: str, items: Sequence[Item], *, ingest_seconds: float, retrieve_seconds: float,
                  ledger: dict[str, Any], **extra: Any) -> str:
    """One question's ranked list, as an adapter in another environment writes it."""
    return json.dumps({"qid": qid, "system": system, "items": [asdict(item) for item in items],
                       "ingest_seconds": ingest_seconds, "retrieve_seconds": retrieve_seconds, "ledger": ledger, **extra})


def no_memory_prompt(question: str) -> str:
    """The benchmark's own form for a reader given no history (run_generation.py lines 48-50)."""
    return question + "Answer step by step."
