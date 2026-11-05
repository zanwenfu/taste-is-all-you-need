"""One rule-loss case with memory tools, from a fresh store to a complete record.

The probe sees what the package's probe sees (system, developer, the summary that replaced the history, the
question) and, in addition, the two memory tools. The record keeps every call whole and the three steps apart: looked
(searched at all), found (the rule's whole text reached the model), compliant (the answer obeyed).

The package's developer prompt says "Do not execute a tool for the current request", and a model told so does not
search, so five conditions separate that from what an agent does when it may look:

    i    the benchmark's words                              iii  that clause narrowed to the tools that carry out a request
    ii   i and one sentence saying a searchable record      iv   iii and the sentence
         exists; it never mentions rules or instructions    v    iv and one sentence telling it to check that record

Arm a5q leaves nothing to the agent: the harness searches once with the request's own words and delivers what it
finds, as memory products usually do. That measures finding alone.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from studies.memory_bench.core import fill
from studies.memory_ledger import BASELINE, stand_in_record
from studies.memory_probe.stores import GitTurns, NotesStore
from studies.memory_probe.tool_loop import ToolRun, ask_with_tools, found_in, grade, holds
from taste.memstore import BadPointer, CommitHeader
from taste.memstore.tools import FETCH, MAX_RESULTS, SEARCH, definitions

SENTENCE = "A searchable record of the earlier conversation is available through the memory_search and memory_fetch tools."
LOOK = "Before you answer a request, check that record for anything that bears on it."
_FORBIDS = "Do not execute a tool for the current request."
_NARROWED = "Do not execute a tool to carry out the current request."
# condition: (narrow the clause, say a record exists, tell it to check)
CONDITIONS = {"i": (False, False, False), "ii": (False, True, False), "iii": (True, False, False),
              "iv": (True, True, False), "v": (True, True, True)}
_REQUEST = re.compile(r'Request from the user:\n"(.*?)"\n\nTwo candidate', re.DOTALL)
ARMS = {
    "a5": "Taste git memory, keyword search and fetch, headers off (ranks as a flat file by construction)",
    "a5q": "a5's store, searched once by the harness with the request's own words, what it finds delivered; no tools",
    "a4": "Taste git memory, keyword search and fetch, model-written headers",
    "a6": "rule records written by the study's reader, delivered with the summary at every call; no tools",
    "a7": "a4 and a6 together",
    "a2a": "the whole conversation as a flat file, cut to the memory budget from a stated end, delivered; no tools",
    "a2c": "a notes file one model kept as the conversation went, its last version only, searched with the tools",
    "a2d": "the same notes file with every version kept in git, searched with the tools",
}
TOOL_ARMS = ("a5", "a4", "a7", "a2c", "a2d")
# The same words the earlier record arms used, so a6 is the same delivery measured in this harness.
DELIVERED_HEADER = "Standing instructions the user gave in this session, verbatim, from the record:"
FOUND_HEADER = "From the record of this session, what may bear on this request:"
FLAT_HEADER = "The conversation so far, from a notes file:"


def flat_file(messages: list[dict], *, budget: int, keep: str, count: Callable[[str], int]) -> str:
    """Every turn as "[role]: content", one per line, cut to ``budget`` tokens from the stated end. The turn at the
    cut is shortened from its far side, a word at a time, as the earlier record arms' recent text was."""
    if keep not in ("newest", "oldest"):
        raise ValueError(f"keep must be newest or oldest, not {keep!r}")
    lines = [f"[{turn['role']}]: {turn['content']}" for turn in messages]
    ordered = list(reversed(lines)) if keep == "newest" else lines
    kept, used = [], 0
    for line in ordered:
        size = count(line)
        if used + size <= budget:
            kept.append(line)
            used += size
            continue
        words = line.split()
        while words and count(" ".join(words)) > budget - used:
            words = words[1:] if keep == "newest" else words[:-1]
        if words:
            kept.append(" ".join(words))
        break
    return "\n".join(reversed(kept) if keep == "newest" else kept)


def case_messages(case: dict, summary: str, *, condition: str, upper: bool, block: str | None = None,
                  block_header: str = "") -> list[dict]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    narrow, say, tell = CONDITIONS[condition]
    developer = case["developer_prompt"]
    if narrow:
        if _FORBIDS not in developer:
            raise ValueError("this case's developer prompt does not hold the clause to narrow")
        developer = developer.replace(_FORBIDS, _NARROWED)
    developer += f"\n\n{SENTENCE}" if say else ""
    developer += f" {LOOK}" if tell else ""
    context = summary if block is None else f"{summary}\n\n{block_header}\n{block}"
    return [{"role": "system", "content": case["system_prompt"]},
            {"role": "developer", "content": developer},
            {"role": "assistant", "content": context},
            {"role": "user", "content": case["post_sc_probe_prompt"] if upper else case["probe_prompt"]}]


def _search_for_it(store: GitTurns, case: dict, budget: int, count: Callable[[str], int]) -> tuple[str, list[dict]]:
    """One search with the request's own words, each hit fetched in rank order, all of it cut to the budget."""
    request = _REQUEST.search(case["probe_prompt"]).group(1)
    arguments = {"description": request, "limit": MAX_RESULTS}
    output, failed = store.answer(SEARCH, arguments)
    calls = [{"name": SEARCH, "arguments": json.dumps(arguments), "output": output, "failed": failed}]
    pieces = []
    for hit in json.loads(output).get("results", []):
        arguments = {"pointer": hit["pointer"]}
        output, failed = store.answer(FETCH, arguments)
        calls.append({"name": FETCH, "arguments": json.dumps(arguments), "output": output, "failed": failed})
        if not failed:
            pieces.append(json.loads(output)["text"])
    return fill(pieces, budget, count), calls


def _turns_reached(store: GitTurns, run: Any) -> list[int]:
    """The turns whose pointers reached the model, in search results or as fetched."""
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


def _without_tools(chat: Callable, messages: list[dict]) -> ToolRun:
    reply = chat(messages, None)
    choice = reply.choices[0]
    return ToolRun(final=choice.message.content or "", finish_reason=choice.finish_reason,
                   usage=[(reply.usage.prompt_tokens, reply.usage.completion_tokens)])


def _rule_turns(case: dict) -> list[int]:
    return [n for n, turn in enumerate(case["messages"]) if holds(str(turn.get("content", "")), case["sc_text"])]


def _delivered_search(case: dict, summary: str, *, chat: Callable, condition: str, workdir: Path, budget: int,
                      count: Callable[[str], int], prompt: str, k: int, upper: bool) -> dict[str, Any]:
    """Arm a5q: the store is searched by the harness, so "looked" is true by construction and "found" is all it says."""
    started = time.perf_counter()
    store = GitTurns(Path(workdir), header=None)
    try:
        store.ingest(case["messages"])
        ingested = time.perf_counter()
        block, calls = _search_for_it(store, case, budget, count)
        messages = case_messages(case, summary, condition=condition, upper=upper, block=block, block_header=FOUND_HEADER)
        run = _without_tools(chat, messages)
        asked = time.perf_counter()
        run.calls, run.searches, run.fetches = calls, 1, len(calls) - 1
        reached = _turns_reached(store, run)
        ledger, commits = store.ledger("single_query"), store.commits
    finally:
        store.close()
    rule_turns = _rule_turns(case)
    return {
        **{key: case[key] for key in ("id", "haystack_id", "sc_id", "sc_type", "compliant_letter")},
        "arm": "a5q", "condition": condition, "prompt": prompt, "k": k, "upper": upper, "keep": None,
        "delivered_tokens": count(block), "final": run.final, "finish_reason": run.finish_reason,
        "compliant": grade(run.final, case["compliant_letter"]), "looked": True, "found": holds(block, case["sc_text"]),
        "found_turns": reached, "rule_turns": rule_turns, "rule_turn_reached": bool(set(reached) & set(rule_turns)),
        "searches": 1, "fetches": len(calls) - 1, "rounds": 0, "tool_errors": sum(call["failed"] for call in calls),
        "unparsed_arguments": 0, "budget_refusals": 0, "round_limit_hit": False, "tool_tokens": 0, "usage": run.usage,
        "calls": calls, "commits": commits, "ingest_seconds": ingested - started, "probe_seconds": asked - ingested,
        "ledger": ledger,
    }


def run_case(case: dict, summary: str, *, chat: Callable, arm: str, condition: str, workdir: Path, max_rounds: int,
             budget: int, count: Callable[[str], int], prompt: str, k: int,
             header: Callable[[int, dict], CommitHeader] | None = None, upper: bool = False,
             delivered: str | None = None, keep: str = "newest", notes: list[str] | None = None) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if arm in ("a4", "a7") and header is None:
        raise ValueError(f"arm {arm} needs its headers")
    if arm in ("a6", "a7") and delivered is None:
        raise ValueError(f"arm {arm} needs its delivered records")
    if arm in ("a2c", "a2d") and not notes:
        raise ValueError(f"arm {arm} needs the notes versions")
    block, block_header = None, ""
    if arm == "a5q":
        return _delivered_search(case, summary, chat=chat, condition=condition, workdir=workdir, budget=budget,
                                 count=count, prompt=prompt, k=k, upper=upper)
    if arm in ("a6", "a7"):
        block, block_header = delivered, DELIVERED_HEADER
    elif arm == "a2a":
        block, block_header = flat_file(case["messages"], budget=budget, keep=keep, count=count), FLAT_HEADER
    messages = case_messages(case, summary, condition=condition, upper=upper, block=block, block_header=block_header)
    started = time.perf_counter()
    reached: list[int] = []
    commits = 0
    if arm in TOOL_ARMS:
        if arm in ("a2c", "a2d"):
            store = NotesStore(Path(workdir), notes or [], history=arm == "a2d")
        else:
            store = GitTurns(Path(workdir), header=header if arm in ("a4", "a7") else None)
        try:
            if isinstance(store, GitTurns):
                store.ingest(case["messages"])
            ingested = time.perf_counter()
            run = ask_with_tools(chat, messages, definitions("chat"), store.answer, max_rounds=max_rounds,
                                 budget=budget, count=count)
            asked = time.perf_counter()
            reached = _turns_reached(store, run)
            ledger, commits = store.ledger(), store.commits
        finally:
            store.close()
    else:
        ingested = time.perf_counter()
        run = _without_tools(chat, messages)
        asked = time.perf_counter()
        source = "studies/context_rot/criteria.py records" if arm == "a6" else f"flat file, keep {keep}"
        ledger = stand_in_record(BASELINE, code=f"studies/memory_probe/cases.py:{arm} ({source})")
    rule_turns = _rule_turns(case)
    return {
        **{key: case[key] for key in ("id", "haystack_id", "sc_id", "sc_type", "compliant_letter")},
        "arm": arm, "condition": condition, "prompt": prompt, "k": k, "upper": upper,
        "keep": keep if arm == "a2a" else None, "delivered_tokens": count(block) if block is not None else 0,
        "final": run.final, "finish_reason": run.finish_reason, "compliant": grade(run.final, case["compliant_letter"]),
        "looked": run.searches > 0, "found": found_in(run, case["sc_text"]),
        "found_turns": reached, "rule_turns": rule_turns, "rule_turn_reached": bool(set(reached) & set(rule_turns)),
        "searches": run.searches, "fetches": run.fetches, "rounds": run.rounds, "tool_errors": run.tool_errors,
        "unparsed_arguments": run.unparsed_arguments, "budget_refusals": run.budget_refusals,
        "round_limit_hit": run.round_limit_hit, "tool_tokens": run.tool_tokens, "usage": run.usage,
        "calls": run.calls, "commits": commits, "ingest_seconds": ingested - started, "probe_seconds": asked - ingested,
        "ledger": ledger,
    }
