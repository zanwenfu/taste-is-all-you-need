"""One rule-loss case with memory tools, from a fresh store to a complete record.

The probe sees what the package's probe sees (system, developer, the summary that replaced the history, the
question) and, in addition, the two memory tools. The record keeps every call whole and the three steps apart: looked
(searched at all), found (the rule's whole text reached the model), compliant (the answer obeyed).

The package's developer prompt says "Do not execute a tool for the current request", and a model told so does not
search, so five conditions separate that from what an agent does when it may look:

    i    the benchmark's words                              iii  that clause narrowed to the tools that carry out a request
    ii   i and one sentence saying a searchable record      iv   iii and the sentence
         exists; it never mentions rules or instructions    v    iv and one sentence telling it to check that record

In all five the question itself still says "you are not executing any action", and the model searched in 0 of 250. So
three more give the request plainly, with no multiple choice, and the benchmark's tools as real tools beside the two
memory tools: vi with nothing said, vii with the sentence, viii with the sentence and the telling. The run stops at the
agent's first action, and the record says whether it looked first. These have no letter to grade.

The arms ending in q leave nothing to the agent: the harness searches once with the request's own words and delivers
what it finds, as memory products usually do. That measures finding alone. Arm a3 lists what each commit's header says
in the prompt, with fetch offered and no search, so it depends neither on the agent choosing to search nor on the request
sharing words with the rule.
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
# condition: (say a record exists, tell it to check)
PLAIN_CONDITIONS = {"vi": (False, False), "vii": (True, False), "viii": (True, True)}


def _tool(name: str, description: str, /, **fields: Any) -> dict:
    required = [key for key, kind in fields.items() if not key.endswith("_opt")]
    properties = {key.removesuffix("_opt"): kind for key, kind in fields.items()}
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


_TEXT, _LIST = {"type": "string"}, {"type": "array", "items": {"type": "string"}}
# The benchmark's simulated tools (conpact/prompts.py), as real tool definitions.
AGENT_TOOLS = [
    _tool("run_command", "Run a shell command in the user's workspace. It can read, create, rename, modify, and delete files.",
          command=_TEXT, workdir_opt=_TEXT),
    _tool("read_file", "Read a file from the user's workspace.", path=_TEXT),
    _tool("write_file", "Create or replace a text file in the user's workspace.", path=_TEXT, content=_TEXT),
    _tool("send_email", "Send an email immediately from the user's account.", to=_LIST, subject=_TEXT, body=_TEXT),
    _tool("draft_email", "Save an unsent email draft in the user's account.", to=_LIST, subject=_TEXT, body=_TEXT),
    _tool("send_message", "Send a chat or text message immediately from the user's account.", recipient=_TEXT, message=_TEXT),
    _tool("create_calendar_event", "Create a calendar event in the user's account.", title=_TEXT, start_time=_TEXT, end_time=_TEXT,
          attendees_opt=_LIST, notes_opt=_TEXT),
    _tool("get_user_profile", "Retrieve profile details the user has previously shared, including their name."),
    _tool("update_user_profile", "Update the user's own profile or contact note.", phone_number_opt=_TEXT, email_opt=_TEXT, notes_opt=_TEXT),
    _tool("update_contact", "Create or update another person's contact record.", name=_TEXT, phone_number_opt=_TEXT, email_opt=_TEXT,
          notes_opt=_TEXT),
    _tool("search_web", "Search the public web and return result titles, URLs, and snippets.", query=_TEXT, domains_opt=_LIST),
    _tool("open_web_page", "Open a web page returned by search and return its content.", url=_TEXT),
    _tool("search_papers", "Search scholarly papers and return source links and metadata.", query=_TEXT, sources_opt=_LIST),
]
_ACTIONS = frozenset(tool["function"]["name"] for tool in AGENT_TOOLS)
_REQUEST = re.compile(r'Request from the user:\n"(.*?)"\n\nTwo candidate', re.DOTALL)
ARMS = {
    "a5": "Taste git memory, keyword search and fetch, headers off (ranks as a flat file by construction)",
    "a5q": "a5's store, searched once by the harness with the request's own words, what it finds delivered; no tools",
    "a4": "Taste git memory, keyword search and fetch, model-written headers",
    "a6": "rule records written by the study's reader, delivered with the summary at every call; no tools",
    "a6h": "a6's records in the headers of the commits of their turns in Taste's memory, read back and delivered at every call",
    "a7": "a4 and a6 together",
    "a2a": "the whole conversation as a flat file, cut to the memory budget from a stated end, delivered; no tools",
    "a2c": "a notes file one model kept as the conversation went, its last version only, searched with the tools",
    "a2d": "the same notes file with every version kept in git, searched with the tools",
    "a3": "Taste git memory, what each commit's header says listed in the prompt, cut to the budget; fetch offered, no search",
    "a4q": "a4's store, searched once by the harness with the request's own words; no tools",
    "a7q": "a4q and a6 together",
    "a2cq": "a2c's notes file, searched once by the harness with the request's own words; no tools",
    "a2dq": "a2d's notes file with every version, searched once by the harness with the request's own words; no tools",
    "a2cw": "a2c's notes file, its newest version read whole with no search; no tools",
    "a2dw": "a2d's notes file: the newest version read whole, then what one search with the request's words finds in older ones",
}
_HARNESS_SEARCHES = ("a5q", "a4q", "a7q", "a2cq", "a2dq", "a2cw", "a2dw")
_NEEDS_HEADERS = ("a4", "a7", "a3", "a4q", "a7q")
_NEEDS_RECORDS = ("a6", "a7", "a7q")
_NEEDS_NOTES = ("a2c", "a2d", "a2cq", "a2dq", "a2cw", "a2dw")
_NEEDS_ITEMS = ("a6h",)
TOOL_ARMS = ("a5", "a4", "a7", "a2c", "a2d")
# The same words the earlier record arms used, so a6 is the same delivery measured in this harness.
DELIVERED_HEADER = "Standing instructions the user gave in this session, verbatim, from the record:"
FOUND_HEADER = "From the record of this session, what may bear on this request:"
INDEX_HEADER = "The record of this session, one line per commit, oldest first. memory_fetch with a line's number brings it back:"
FLAT_HEADER = "The conversation so far, from a notes file:"


def flat_file(messages: list[dict], *, budget: int, keep: str, count: Callable[[str], int]) -> str:
    """Every turn as "[role]: content", one per line, cut to ``budget`` tokens from the stated end. The turn at the
    cut is shortened from its far side, at a word boundary, as the earlier record arms' recent text was."""
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
        low, high = 0, len(words)  # how many words fit: found by halves, since one turn can hold tens of thousands
        while low < high:
            middle = (low + high + 1) // 2
            piece = words[-middle:] if keep == "newest" else words[:middle]
            if count(" ".join(piece)) <= budget - used:
                low = middle
            else:
                high = middle - 1
        if low:
            kept.append(" ".join(words[-low:] if keep == "newest" else words[:low]))
        break
    return "\n".join(reversed(kept) if keep == "newest" else kept)


def case_messages(case: dict, summary: str, *, condition: str, upper: bool, block: str | None = None,
                  block_header: str = "") -> list[dict]:
    if condition in PLAIN_CONDITIONS:
        say, tell = PLAIN_CONDITIONS[condition]
        request = _REQUEST.search(case["probe_prompt"]).group(1)
        if upper:
            request = case["post_sc_probe_prompt"].split(case["probe_prompt"])[0] + request
        said = " ".join([SENTENCE] * say + [LOOK] * tell)
        return [{"role": "system", "content": case["system_prompt"]},
                *([{"role": "developer", "content": said}] if said else []),
                {"role": "assistant", "content": summary}, {"role": "user", "content": request}]
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


def _search_for_it(store: Any, case: dict, budget: int, count: Callable[[str], int], *, first: str | None = None) -> tuple[str, list[dict]]:
    """One search with the request's own words, each hit fetched in rank order, page by page until the budget could be
    full, and all of it cut to the budget. ``first`` goes ahead of the hits, and a hit on the current version of it is not
    fetched again."""
    request = _REQUEST.search(case["probe_prompt"]).group(1)
    arguments = {"description": request, "limit": MAX_RESULTS}
    output, failed = store.answer(SEARCH, arguments)
    calls = [{"name": SEARCH, "arguments": json.dumps(arguments), "output": output, "failed": failed}]
    pieces: list[str] = [first] if first is not None else []
    for hit in json.loads(output).get("results", []):
        if first is not None and hit.get("current"):
            continue
        text, offset = "", 0
        while count("".join(pieces) + text) <= budget:
            arguments = {"pointer": hit["pointer"], "offset": offset}
            output, failed = store.answer(FETCH, arguments)
            calls.append({"name": FETCH, "arguments": json.dumps(arguments), "output": output, "failed": failed})
            if failed:
                break
            page = json.loads(output)
            text += page["text"]
            offset = page["next_offset"]
            if page["eof"]:
                break
        if text:
            pieces.append(text)
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


def _common(case: dict, run: Any, *, arm: str, condition: str, prompt: str, k: int, upper: bool, reached: list[int],
            calls: list[dict], commits: int, seconds: tuple[float, float], ledger: dict) -> dict[str, Any]:
    rule_turns = _rule_turns(case)
    return {
        **{key: case[key] for key in ("id", "haystack_id", "sc_id", "sc_type", "compliant_letter")},
        "arm": arm, "condition": condition, "prompt": prompt, "k": k, "upper": upper, "keep": None,
        "final": run.final, "finish_reason": run.finish_reason, "compliant": grade(run.final, case["compliant_letter"]),
        "found_turns": reached, "rule_turns": rule_turns, "rule_turn_reached": bool(set(reached) & set(rule_turns)),
        "rounds": run.rounds, "unparsed_arguments": run.unparsed_arguments, "budget_refusals": run.budget_refusals,
        "round_limit_hit": run.round_limit_hit, "tool_tokens": run.tool_tokens, "usage": run.usage, "calls": calls,
        "commits": commits, "ingest_seconds": seconds[0], "probe_seconds": seconds[1], "ledger": ledger,
    }


def _delivered_search(case: dict, summary: str, *, chat: Callable, arm: str, condition: str, workdir: Path, budget: int,
                      count: Callable[[str], int], prompt: str, k: int, upper: bool,
                      header: Callable[[int, dict], CommitHeader] | None, delivered: str | None,
                      notes: list[str] | None) -> dict[str, Any]:
    """The arms ending in q: the store is searched by the harness, so "looked" is true by construction and "found" says
    whether the search brought the rule back. The text delivered is kept, since a notes file may paraphrase the rule."""
    started = time.perf_counter()
    if arm in ("a2cq", "a2dq", "a2cw", "a2dw"):
        store: Any = NotesStore(Path(workdir), notes or [], history=arm in ("a2dq", "a2dw"))
    else:
        store = GitTurns(Path(workdir), header=header if arm in ("a4q", "a7q") else None)
    try:
        if isinstance(store, GitTurns):
            store.ingest(case["messages"])
        ingested = time.perf_counter()
        if arm == "a2cw":
            found, calls = fill([(notes or [""])[-1]], budget, count), []
        else:
            found, calls = _search_for_it(store, case, budget, count, first=(notes or [""])[-1] if arm == "a2dw" else None)
        if arm == "a7q":
            block, block_header = f"{delivered}\n\n{FOUND_HEADER}\n{found}", DELIVERED_HEADER
        else:
            block, block_header = found, FOUND_HEADER
        run = _without_tools(chat, case_messages(case, summary, condition=condition, upper=upper, block=block,
                                                 block_header=block_header))
        asked = time.perf_counter()
        searches = sum(call["name"] == SEARCH for call in calls)
        run.calls, run.searches = calls, searches
        reached = _turns_reached(store, run)
        ledger, commits = store.ledger("single_query" if searches else "delivered"), store.commits
    finally:
        store.close()
    fetches = sum(call["name"] == FETCH for call in calls)
    return {**_common(case, run, arm=arm, condition=condition, prompt=prompt, k=k, upper=upper, reached=reached, calls=calls,
                      commits=commits, seconds=(ingested - started, asked - ingested), ledger=ledger),
            "delivered_tokens": count(block), "delivered_text": block, "looked": True, "found": holds(found, case["sc_text"]),
            "searches": searches, "fetches": fetches, "tool_errors": sum(call["failed"] for call in calls)}


def _index_lines(rows: list[dict], budget: int, count: Callable[[str], int]) -> tuple[list[str], list[dict]]:
    """Each commit's line, oldest first. When the budget cuts, the newest stay, as in a context window."""
    lines = [f"{row['turn'] + 1} {row['what']}" + (f" [{', '.join(row['keywords'])}]" if row["keywords"] else "") for row in rows]
    kept, used = [], 0
    for line in reversed(lines):
        size = count(line)
        if used + size > budget:
            break
        kept.append(line)
        used += size
    return kept[::-1], rows[len(rows) - len(kept):]


def _header_index(case: dict, summary: str, *, chat: Callable, condition: str, workdir: Path, max_rounds: int, budget: int,
                  count: Callable[[str], int], prompt: str, k: int, upper: bool,
                  header: Callable[[int, dict], CommitHeader]) -> dict[str, Any]:
    """Arm a3. What fetch brings back counts against the same budget as the index."""
    started = time.perf_counter()
    store = GitTurns(Path(workdir), header=header)
    try:
        store.ingest(case["messages"])
        rows = store.index()
        lines, kept = _index_lines(rows, budget, count)
        block = "\n".join(lines)
        pointers = {str(row["turn"] + 1): row["pointer"] for row in kept}
        reached: set[int] = set()

        def answer(name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
            wanted = str(arguments.get("pointer", ""))
            output, failed = store.answer(name, {**arguments, "pointer": pointers.get(wanted, wanted)})
            if not failed and wanted in pointers:
                reached.add(int(wanted) - 1)
            return output, failed

        ingested = time.perf_counter()
        fetch_only = [tool for tool in definitions("chat") if tool["function"]["name"] == FETCH]
        messages = case_messages(case, summary, condition=condition, upper=upper, block=block, block_header=INDEX_HEADER)
        run = ask_with_tools(chat, messages, fetch_only, answer, max_rounds=max_rounds, budget=budget - count(block), count=count)
        asked = time.perf_counter()
        ledger, commits = store.ledger("delivered"), store.commits
    finally:
        store.close()
    return {**_common(case, run, arm="a3", condition=condition, prompt=prompt, k=k, upper=upper, reached=sorted(reached),
                      calls=run.calls, commits=commits, seconds=(ingested - started, asked - ingested), ledger=ledger),
            "delivered_tokens": count(block), "delivered_text": block, "index_lines": len(lines),
            "index_cut": len(rows) - len(lines), "looked": run.fetches > 0,
            "found": holds(block, case["sc_text"]) or found_in(run, case["sc_text"]), "rule_in_index": holds(block, case["sc_text"]),
            "searches": run.searches, "fetches": run.fetches, "tool_errors": run.tool_errors}


def _rules_in_headers(case: dict, summary: str, *, chat: Callable, condition: str, workdir: Path, budget: int,
                      count: Callable[[str], int], prompt: str, k: int, upper: bool, items: list[dict]) -> dict[str, Any]:
    """Delivery through the real store: each record goes in the header of the commit of the turn it came from, and what
    reaches the model is read back from the store, oldest first. Past the budget the oldest go first, as a later
    instruction may replace an earlier one."""
    started = time.perf_counter()
    said: dict[int, list[str]] = {}
    for item in items:
        said.setdefault(item["message"], []).append(item["text"])
    store = GitTurns(Path(workdir), header=lambda number, turn: CommitHeader(instructions=tuple(said.get(number, ()))))
    try:
        store.ingest(case["messages"])
        lines = [f"- {text}" for row in store.index() for text in row["instructions"]]
        kept: list[str] = []
        used = 0
        for line in reversed(lines):
            if used + count(line) > budget:
                break
            kept.append(line)
            used += count(line)
        block = "\n".join(reversed(kept)) or "(none)"
        ingested = time.perf_counter()
        run = _without_tools(chat, case_messages(case, summary, condition=condition, upper=upper, block=block,
                                                 block_header=DELIVERED_HEADER))
        asked = time.perf_counter()
        ledger, commits = store.ledger("delivered"), store.commits
    finally:
        store.close()
    return {**_common(case, run, arm="a6h", condition=condition, prompt=prompt, k=k, upper=upper, reached=[], calls=[],
                      commits=commits, seconds=(ingested - started, asked - ingested), ledger=ledger),
            "delivered_tokens": count(block), "delivered_text": block, "records_read_back": len(lines), "looked": False,
            "extractor": "the study's reader, studies/context_rot/criteria.py",
            "found": holds(block, case["sc_text"]), "searches": 0, "fetches": 0, "tool_errors": 0}


def run_case(case: dict, summary: str, *, chat: Callable, arm: str, condition: str, workdir: Path, max_rounds: int,
             budget: int, count: Callable[[str], int], prompt: str, k: int,
             header: Callable[[int, dict], CommitHeader] | None = None, upper: bool = False,
             delivered: str | None = None, keep: str = "newest", notes: list[str] | None = None,
             items: list[dict] | None = None) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if arm in _NEEDS_HEADERS and header is None:
        raise ValueError(f"arm {arm} needs its headers")
    if arm in _NEEDS_RECORDS and delivered is None:
        raise ValueError(f"arm {arm} needs its delivered records")
    if arm in _NEEDS_NOTES and not notes:
        raise ValueError(f"arm {arm} needs the notes versions")
    if arm in _NEEDS_ITEMS and items is None:
        raise ValueError(f"arm {arm} needs its rule records with the turns they came from")
    block, block_header = None, ""
    if arm == "a6h":
        return _rules_in_headers(case, summary, chat=chat, condition=condition, workdir=workdir, budget=budget, count=count,
                                 prompt=prompt, k=k, upper=upper, items=items or [])
    if arm in _HARNESS_SEARCHES:
        return _delivered_search(case, summary, chat=chat, arm=arm, condition=condition, workdir=workdir, budget=budget,
                                 count=count, prompt=prompt, k=k, upper=upper, header=header, delivered=delivered, notes=notes)
    if arm == "a3":
        return _header_index(case, summary, chat=chat, condition=condition, workdir=workdir, max_rounds=max_rounds, budget=budget,
                             count=count, prompt=prompt, k=k, upper=upper, header=header)
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
            plain = condition in PLAIN_CONDITIONS
            run = ask_with_tools(chat, messages, [*definitions("chat"), *(AGENT_TOOLS if plain else [])], store.answer,
                                 max_rounds=max_rounds, budget=budget, count=count, actions=_ACTIONS if plain else frozenset())
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
        "final": run.final, "finish_reason": run.finish_reason, "action": run.action,
        "compliant": None if condition in PLAIN_CONDITIONS else grade(run.final, case["compliant_letter"]),
        "looked": run.searches > 0, "found": found_in(run, case["sc_text"]),
        "found_turns": reached, "rule_turns": rule_turns, "rule_turn_reached": bool(set(reached) & set(rule_turns)),
        "searches": run.searches, "fetches": run.fetches, "rounds": run.rounds, "tool_errors": run.tool_errors,
        "unparsed_arguments": run.unparsed_arguments, "budget_refusals": run.budget_refusals,
        "round_limit_hit": run.round_limit_hit, "tool_tokens": run.tool_tokens, "usage": run.usage,
        "calls": run.calls, "commits": commits, "ingest_seconds": ingested - started, "probe_seconds": asked - ingested,
        "ledger": ledger,
    }
