"""Retrieval scores: session and round level, any and all, at a rank and at the budget.

LongMemEval's own Recall@k is ``recall_all``, and MemPalace's 96.6 is ``recall_any``, so both are always
printed. Abstention questions have no right place to look, and have no retrieval score.
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
        scores.update(_recalls("round", _round_keys(ranked[:cut]), gold.rounds, str(cut)))
    scores["session_ndcg@10"] = _ndcg(sessions, gold.sessions, 10)
    scores.update(_recalls("session", (item.session for item in delivered), gold.sessions, "budget"))
    scores.update(_recalls("round", _round_keys(delivered), gold.rounds, "budget"))
    if gold.evidence:
        # Judge-free: a memory can name the right session and still have lost the fact.
        text = "\n".join(str(turn.get("content", "")) for item in delivered for turn in item.turns)
        scores["evidence_survived"] = sum(words in text for words in gold.evidence) / len(gold.evidence)
    return scores
