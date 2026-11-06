"""Retrieval scores: sessions at a rank and at the budget, rounds at the budget only.

LongMemEval's own Recall@k is ``recall_all``, and MemPalace's 96.6 is ``recall_any``, so both are always
printed. Abstention questions have no right place to look, and have no retrieval score. Five whole sessions hold far more
rounds than five rounds do, so a round score at a rank would compare unlike things; at the budget every system has paid the
same. A system that cannot say which round it drew on gets no round score, not a zero; one that can, of which nothing whole arrived,
reads zero. For a memory that names its round, a round score means something drawn from that round arrived, not that the fact
survived: reading the answers tells that.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from studies.memory_bench.core import Item
from studies.memory_bench.lme import Gold

CUTS = (1, 3, 5, 10, 30, 50)


def _dedupe(values: Iterable) -> list:
    return list(dict.fromkeys(values))


def _recalls(prefix: str, found: Iterable, gold: frozenset, tag: str) -> dict[str, float]:
    held = set(found) & gold
    return {
        f"{prefix}_recall_any@{tag}": float(bool(held)),
        f"{prefix}_recall_all@{tag}": float(held == gold),
    }


def _ndcg(ranked: Sequence, gold: frozenset, cut: int) -> float:
    gains = sum(1 / math.log2(place + 2) for place, value in enumerate(ranked[:cut]) if value in gold)
    ideal = sum(1 / math.log2(place + 2) for place in range(min(len(gold), cut)))
    return gains / ideal if ideal else 0.0


def _round_keys(items: Iterable[Item]) -> list[tuple[str, int]]:
    return [(item.session, number) for item in items for number in item.rounds]


def retrieval_scores(ranked: Sequence[Item], delivered: Sequence[Item], gold: Gold) -> dict[str, float]:
    if gold.abstention:
        return {}
    sessions = _dedupe(item.session for item in ranked)
    scores: dict[str, float] = {}
    for cut in CUTS:
        scores.update(_recalls("session", sessions[:cut], gold.sessions, str(cut)))
    scores["session_ndcg@10"] = _ndcg(sessions, gold.sessions, 10)
    scores.update(_recalls("session", (item.session for item in delivered), gold.sessions, "budget"))
    if any(item.rounds for item in ranked):
        scores.update(_recalls("round", _round_keys(delivered), gold.rounds, "budget"))
    if gold.evidence and not any(turn.get("role") == "memory" for item in delivered for turn in item.turns):
        # Judge-free, for systems that hand over the words as they were said. A memory in other words can never hold a whole
        # turn, so asking it would read zero by construction.
        text = "\n".join(str(turn.get("content", "")) for item in delivered for turn in item.turns)
        scores["evidence_survived"] = sum(words in text for words in gold.evidence) / len(gold.evidence)
    return scores
