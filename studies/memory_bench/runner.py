"""One question, from a fresh store to a complete record, and the checks before any aggregate.

No model is called here. The prompt goes to the reader and the answer to the judges in later stages, so
what each system offered can be read and counted before a dollar or a GPU-hour is spent on it.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from typing import Any

from studies.memory_bench import locomo
from studies.memory_bench.core import deliver, no_memory_prompt, reader_prompt
from studies.memory_bench.lme import Gold
from studies.memory_bench.metrics import retrieval_scores
from studies.memory_bench.systems import System
from studies.memory_ledger import check

MAX_RANKED_KEPT = 200


def retrieve_one(view: dict, gold: Gold, system: System, *, budget: int, count: Callable[[str], int]) -> dict[str, Any]:
    try:
        started = time.perf_counter()
        system.ingest(view["sessions"], unit=view["qid"])
        ingested = time.perf_counter()
        ranked = system.retrieve(view["question"], qid=view["qid"])
        retrieved = time.perf_counter()
        measured = getattr(system, "measured", None)
        ingest_seconds, retrieve_seconds = (
            measured(view["qid"]) if measured else (ingested - started, retrieved - ingested)
        )
        delivered = deliver(ranked, budget, count)
        if system.no_retrieval:
            prompt = no_memory_prompt(view["question"])
        else:
            prompt = reader_prompt(delivered.items, view["question_date"], view["question"])
        ledger = system.ledger()
    finally:
        system.close()
    return {
        "qid": view["qid"],
        "question_id": gold.question_id,
        "question_type": gold.question_type,
        "system": getattr(system, "label", None) or type(system).__name__,
        "budget": budget,
        "ranked": [{"session": i.session, "rounds": list(i.rounds)} for i in ranked[:MAX_RANKED_KEPT]],
        "delivered": [{"session": i.session, "rounds": list(i.rounds)} for i in delivered.items],
        "offered": delivered.offered,
        "cut": delivered.cut,
        "dropped": delivered.dropped,
        "tokens_delivered": delivered.tokens,
        "scores": retrieval_scores(ranked, delivered.items, gold),
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8", "surrogateescape")).hexdigest(),
        "ingest_seconds": ingest_seconds,
        "retrieve_seconds": retrieve_seconds,
        "ledger": ledger,
    }


def retrieve_conversation(view: dict, golds: dict, system: System, *, budget: int,
                          count: Callable[[str], int]) -> list[dict[str, Any]]:
    """One LoCoMo conversation: one store, one ingest, every question of it in turn."""
    records = []
    try:
        started = time.perf_counter()
        system.ingest(view["sessions"], unit=view["conversation"])
        ingested = time.perf_counter() - started
        for question in view["questions"]:
            gold = golds[question["qid"]]
            asked = time.perf_counter()
            ranked = system.retrieve(question["query"], qid=question["qid"])
            retrieved = time.perf_counter()
            measured = getattr(system, "measured", None)
            ingest_seconds, retrieve_seconds = (
                measured(question["qid"]) if measured else (ingested, retrieved - asked)
            )
            delivered = deliver(ranked, budget, count, locomo.dialog_text)
            prompt = locomo.reader_prompt(delivered.items, view, question)
            records.append({
                "qid": question["qid"],
                "question_id": gold.question_id,
                "question_type": gold.question_type,
                "category": gold.category,
                "conversation": view["conversation"],
                "system": getattr(system, "label", None) or type(system).__name__,
                "budget": budget,
                "ranked": [{"session": i.session, "rounds": list(i.rounds)} for i in ranked[:MAX_RANKED_KEPT]],
                "delivered": [{"session": i.session, "rounds": list(i.rounds)} for i in delivered.items],
                "offered": delivered.offered,
                "cut": delivered.cut,
                "dropped": delivered.dropped,
                "tokens_delivered": delivered.tokens,
                "scores": locomo.locomo_scores(ranked, delivered.items, gold),
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8", "surrogateescape")).hexdigest(),
                "ingest_seconds": ingest_seconds,
                "retrieve_seconds": retrieve_seconds,
                "ledger": system.ledger(),
            })
    finally:
        system.close()
    return records


def validate(records: list[dict[str, Any]], *, expected: set[str]) -> dict[str, Any]:
    """Every question exactly once, no empty prompt, every record's ledger clean. Before any aggregate."""
    seen: set[str] = set()
    for record in records:
        if record["qid"] in seen:
            raise ValueError(f"{record['qid']} appears twice")
        seen.add(record["qid"])
    if expected - seen:
        raise ValueError(f"missing {len(expected - seen)} questions")
    if seen - expected:
        raise ValueError(f"{len(seen - expected)} questions that were not asked for")
    for record in records:
        if not record["prompt"].strip():
            raise ValueError(f"{record['qid']} has an empty prompt")
    return check([record["ledger"] for record in records])
