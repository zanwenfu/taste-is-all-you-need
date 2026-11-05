"""One rule-loss case with memory tools, from a fresh store to a complete record.

The probe sees what the package's probe sees (system, developer, the summary that replaced the history, the
question) and, in addition, the two memory tools. Condition (ii) adds one neutral sentence saying a searchable
record exists; it never mentions rules or instructions. The record keeps every call whole and the three steps
apart: looked (searched at all), found (the rule's whole text reached the model), compliant (the answer obeyed).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from studies.memory_probe.stores import GitTurns
from studies.memory_probe.tool_loop import ask_with_tools, found_in, grade
from taste.memstore import BadPointer, CommitHeader
from taste.memstore.tools import FETCH, SEARCH, definitions

SENTENCE = "A searchable record of the earlier conversation is available through the memory_search and memory_fetch tools."
ARMS = {
    "a5": "Taste git memory, keyword search and fetch, headers off (ranks as a flat file by construction)",
    "a4": "Taste git memory, keyword search and fetch, model-written headers",
}


def case_messages(case: dict, summary: str, *, condition: str, upper: bool) -> list[dict]:
    if condition not in ("i", "ii"):
        raise ValueError(f"unknown condition {condition!r}")
    developer = case["developer_prompt"] + (f"\n\n{SENTENCE}" if condition == "ii" else "")
    return [{"role": "system", "content": case["system_prompt"]},
            {"role": "developer", "content": developer},
            {"role": "assistant", "content": summary},
            {"role": "user", "content": case["post_sc_probe_prompt"] if upper else case["probe_prompt"]}]


def _turns_reached(store: GitTurns, run: Any) -> list[int]:
    """The turns whose pointers reached the model, in search results or as fetched."""
    import json

    pointers = []
    for call in run.calls:
        if call["failed"]:
            continue
        if call["name"] == SEARCH:
            pointers += [result["pointer"] for result in json.loads(call["output"]).get("results", [])]
        elif call["name"] == FETCH:
            pointers.append(json.loads(call["arguments"]).get("pointer", ""))
    turns = set()
    for pointer in pointers:
        try:
            turn = store.turn_of(pointer)
        except BadPointer:
            continue
        if turn is not None:
            turns.add(turn)
    return sorted(turns)


def run_case(case: dict, summary: str, *, chat: Callable, arm: str, condition: str, workdir: Path, max_rounds: int,
             budget: int, count: Callable[[str], int], prompt: str, k: int,
             header: Callable[[int, dict], CommitHeader] | None = None, upper: bool = False) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if arm == "a4" and header is None:
        raise ValueError("arm a4 needs its headers")
    started = time.perf_counter()
    store = GitTurns(Path(workdir), header=header if arm == "a4" else None)
    try:
        store.ingest(case["messages"])
        ingested = time.perf_counter()
        run = ask_with_tools(chat, case_messages(case, summary, condition=condition, upper=upper),
                             definitions("chat"), store.answer, max_rounds=max_rounds, budget=budget, count=count)
        asked = time.perf_counter()
        reached = _turns_reached(store, run)
        ledger, commits = store.ledger(), store.commits
    finally:
        store.close()
    rule_turns = [n for n, turn in enumerate(case["messages"]) if case["sc_text"] in str(turn.get("content", ""))]
    return {
        **{key: case[key] for key in ("id", "haystack_id", "sc_id", "sc_type", "compliant_letter")},
        "arm": arm, "condition": condition, "prompt": prompt, "k": k, "upper": upper,
        "final": run.final, "finish_reason": run.finish_reason, "compliant": grade(run.final, case["compliant_letter"]),
        "looked": run.searches > 0, "found": found_in(run, case["sc_text"]),
        "found_turns": reached, "rule_turns": rule_turns, "rule_turn_reached": bool(set(reached) & set(rule_turns)),
        "searches": run.searches, "fetches": run.fetches, "rounds": run.rounds, "tool_errors": run.tool_errors,
        "unparsed_arguments": run.unparsed_arguments, "budget_refusals": run.budget_refusals,
        "round_limit_hit": run.round_limit_hit, "tool_tokens": run.tool_tokens, "usage": run.usage,
        "calls": run.calls, "commits": commits, "ingest_seconds": ingested - started, "probe_seconds": asked - ingested,
        "ledger": ledger,
    }
