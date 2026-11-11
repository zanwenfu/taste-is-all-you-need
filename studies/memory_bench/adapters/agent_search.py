"""A coding agent's way with memory, as Claude Code does it: the history stays as plain files, one per chat session, and the model looks
through them itself with a search and a read, deciding what to read. Everything it reads is charged against the same token budget every
other system fills, counted the same way (the memory's own words), and what it read becomes its ranked list, in the order it read it. The
answer is then written from that list by the benchmark's own reader prompt, the same as for every other system, so this arm differs from
the others only in who chooses what reaches the reader: the model, with its own searches, instead of a ranking.

The model sees file names, dates and round counts for free, as the reader's prompt shows session dates for free. A search shows a short
excerpt of each matching round, also free and never passed on to the answer, as a search engine's snippets are; only rounds it reads are
kept. A round is one user message and the replies after it, the unit the keyword and git systems rank, so its rounds score retrieval the same
way.

    python -m studies.memory_bench.adapters.agent_search --views views.jsonl --url URL --model M --budget 2000 \\
        --tokenizer TOKENIZER --out rankings.jsonl --trace trace.jsonl
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from studies.memory_bench.core import Item, item_text
from studies.memory_bench.lme import rounds

SYSTEM = (
    "You answer questions about a user's past chats with an assistant. The chats are stored as plain text files, one file per chat "
    "session, oldest first. You cannot see their text yet. Find what you need with these commands, one per reply, written alone on the "
    "last line of your reply:\n"
    "SEARCH: <words or a regular expression>\n"
    "  lists the rounds that match, oldest first, case-insensitive, each with a short excerpt (at most {hits}). A round is one user "
    "message and the assistant's replies to it.\n"
    "READ: <file> <round>   or   READ: <file> <first round>-<last round>\n"
    "  reads those rounds in full and keeps them for answering.\n"
    "DONE\n"
    "  when you have read what you need.\n"
    "Only the rounds you READ are kept for the answer; excerpts from SEARCH are not. You may read at most {budget} tokens in all. "
    "You have at most {steps} replies. Do not answer the question yourself: it is answered afterwards from what you read."
)

FIRST = "Question date: {date}\nQuestion: {question}\n\nFiles (name, session date, rounds):\n{files}"

_COMMAND = re.compile(r"^\s*(SEARCH|READ|DONE)\b:?\s*(.*)$", re.IGNORECASE)
_RANGE = re.compile(r"^(\S+)\s+(\d+)(?:\s*-\s*(\d+))?\s*$")


@dataclass(frozen=True)
class Round:
    file: str
    sid: str
    date: str
    number: int
    turns: tuple[dict, ...]

    def text(self) -> str:
        return " ".join(f"{turn['role']}: {turn['content']}" for turn in self.turns)

    def item(self) -> Item:
        return Item(self.sid, self.date, self.turns, (self.number,), f"{self.file}/r{self.number}")


def files_of(sessions: list[dict]) -> dict[str, list[Round]]:
    """One file per session, oldest first, with a neutral name."""
    out: dict[str, list[Round]] = {}
    for index, session in enumerate(sessions, 1):
        name = f"session-{index:03d}"
        out[name] = [Round(name, session["sid"], session["date"], number, tuple(part))
                     for number, part in enumerate(rounds(session))]
    return out


def excerpt(text: str, start: int, end: int, width: int = 200) -> str:
    """About ``width`` characters around a match, on one line."""
    left = max(0, start - width // 2)
    right = min(len(text), max(end, left + width))
    piece = text[left:right].replace("\n", " ")
    return ("..." if left else "") + piece + ("..." if right < len(text) else "")


def search(files: dict[str, list[Round]], pattern: str, *, hits: int = 30) -> str:
    try:
        found = re.compile(pattern, re.IGNORECASE)
    except re.error:
        found = re.compile(re.escape(pattern), re.IGNORECASE)
    lines, total = [], 0
    for name, parts in files.items():
        for part in parts:
            text = part.text()
            match = found.search(text)
            if not match:
                continue
            total += 1
            if len(lines) < hits:
                lines.append(f"{name} ({part.date}) round {part.number}: {excerpt(text, match.start(), match.end())}")
    if not total:
        return f"No round matches {pattern!r}."
    more = f"\n... and {total - len(lines)} more; search more narrowly to see them." if total > len(lines) else ""
    return f"{total} rounds match {pattern!r}:\n" + "\n".join(lines) + more


class Reading:
    """What the model has read so far, charged as every system is charged: ``count(item_text(item))`` per round, never twice."""

    def __init__(self, budget: int, count: Callable[[str], int]) -> None:
        self.budget, self.count = budget, count
        self.used = 0
        self.items: list[Item] = []
        self.seen: set[tuple[str, int]] = set()

    @property
    def left(self) -> int:
        return self.budget - self.used

    def read(self, files: dict[str, list[Round]], name: str, first: int, last: int) -> str:
        if name not in files:
            return f"There is no file {name!r}."
        parts = files[name]
        if not 0 <= first <= last < len(parts):
            return f"{name} has rounds 0 to {len(parts) - 1}."
        shown = []
        for part in parts[first:last + 1]:
            key = (name, part.number)
            if key in self.seen:
                shown.append(f"[{name} round {part.number}: already read]")
                continue
            item = part.item()
            cost = self.count(item_text(item))
            if cost > self.left:
                shown.append(f"[{name} round {part.number}: {cost} tokens, does not fit in the {self.left} left; not read]")
                continue
            self.seen.add(key)
            self.items.append(item)
            self.used += cost
            body = "\n".join(f"{turn['role']}: {turn['content']}" for turn in part.turns)
            shown.append(f"[{name} round {part.number}, {part.date}]\n{body}")
        return "\n\n".join(shown) + f"\n\n({self.used} of {self.budget} tokens read; {self.left} left)"


def command(reply: str) -> tuple[str, str] | None:
    """The last line of the reply that is a command."""
    for line in reversed([line for line in reply.splitlines() if line.strip()]):
        found = _COMMAND.match(line.strip().strip("`"))
        if found:
            return found.group(1).upper(), found.group(2).strip()
    return None


class AgentSearch:
    """``ask(messages) -> dict`` with at least ``text``; a fresh reading per question."""

    def __init__(self, ask: Callable[[list[dict]], dict], count: Callable[[str], int], budget: int, *, steps: int = 12,
                 hits: int = 30, trace: Path | None = None) -> None:
        self.ask, self.count, self.budget, self.steps, self.hits, self.trace = ask, count, budget, steps, hits, trace
        self.lock = threading.Lock()

    def ingest(self, sessions: list[dict]) -> None:
        self.files = files_of(sessions)

    def retrieve(self, query: str, *, date: str | None = None, qid: str | None = None) -> list[Item]:
        reading = Reading(self.budget, self.count)
        listing = "\n".join(f"{name}  {parts[0].date if parts else ''}  {len(parts)}" for name, parts in self.files.items())
        messages = [{"role": "system", "content": SYSTEM.format(hits=self.hits, budget=self.budget, steps=self.steps)},
                    {"role": "user", "content": FIRST.format(date=date or "", question=query, files=listing)}]
        log, stop = [], "steps"
        for _ in range(self.steps):
            reply = self.ask(messages)
            text = reply.get("text") or ""
            messages.append({"role": "assistant", "content": text})
            parsed = command(text)
            entry = {"command": parsed, "finish_reason": reply.get("finish_reason"),
                     "prompt_tokens": reply.get("prompt_tokens"), "completion_tokens": reply.get("completion_tokens")}
            if parsed is None:
                result = "No command found. End your reply with one line: SEARCH: ..., READ: ... or DONE."
            elif parsed[0] == "DONE":
                log.append(entry)
                stop = "done"
                break
            elif parsed[0] == "SEARCH":
                result = search(self.files, parsed[1], hits=self.hits) if parsed[1] else "SEARCH needs words or a pattern."
            else:
                found = _RANGE.match(parsed[1])
                if not found:
                    result = "Write READ: <file> <round> or READ: <file> <first>-<last>."
                else:
                    first = int(found.group(2))
                    last = int(found.group(3)) if found.group(3) else first
                    result = reading.read(self.files, found.group(1), first, last)
            entry["result_chars"] = len(result)
            log.append(entry)
            if reading.left <= 0:
                stop = "budget"
                break
            messages.append({"role": "user", "content": result})
        if self.trace:
            with self.lock, Path(self.trace).open("a") as written:
                written.write(json.dumps({"qid": qid, "query": query, "budget": self.budget, "used": reading.used,
                                          "stop": stop, "steps": log, "read": [item.source for item in reading.items]}) + "\n")
        return reading.items

    def close(self) -> None:
        pass


def chat_ask(client: Any, model: str, *, max_tokens: int, effort: str | None) -> Callable[[list[dict]], dict]:
    extra = {"reasoning_effort": effort} if effort else {}

    def ask(messages: list[dict]) -> dict:
        reply = client.chat.completions.create(model=model, messages=messages, n=1, temperature=0, max_tokens=max_tokens, **extra)
        choice = reply.choices[0]
        return {"text": choice.message.content or "", "finish_reason": choice.finish_reason,
                "prompt_tokens": reply.usage.prompt_tokens, "completion_tokens": reply.usage.completion_tokens}

    return ask


class _WithQid:
    """``run_adapter`` asks ``retrieve(query, date=...)``; the trace wants the question's id too, which the views file holds."""

    def __init__(self, inner: AgentSearch, qids: dict[str, str]) -> None:
        self.inner, self.qids = inner, qids

    def ingest(self, sessions: list[dict]) -> None:
        self.inner.ingest(sessions)

    def retrieve(self, query: str, *, date: str | None = None) -> list[Item]:
        return self.inner.retrieve(query, date=date, qid=self.qids.get(query))

    def close(self) -> None:
        self.inner.close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    from studies.memory_bench.adapters.common import run_adapter
    from studies.memory_bench.run import _counter

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("views", "url", "model", "tokenizer", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--trace", help="every step of every question, apart from the rankings")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--hits", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard", default="0/1")
    arguments = parser.parse_args(argv)
    from openai import OpenAI

    client = OpenAI(base_url=arguments.url, api_key="unused", timeout=600, max_retries=3)
    ask = chat_ask(client, arguments.model, max_tokens=arguments.max_tokens, effort=arguments.effort)
    count = _counter(arguments.tokenizer)
    units = [json.loads(line) for line in Path(arguments.views).read_text().splitlines()]
    qids = {query["query"]: query["qid"] for unit in units for query in unit["queries"]}
    shard = tuple(int(part) for part in arguments.shard.split("/"))

    def make() -> Any:
        return _WithQid(AgentSearch(ask, count, arguments.budget, steps=arguments.steps, hits=arguments.hits,
                                    trace=Path(arguments.trace) if arguments.trace else None), qids)

    settings = {"budget": arguments.budget, "steps": arguments.steps, "hits": arguments.hits, "max_tokens": arguments.max_tokens,
                "effort": arguments.effort, "model": arguments.model}
    counts = run_adapter(Path(arguments.views), Path(arguments.out), make, system=f"agent_search_{arguments.budget}",
                         code="studies/memory_bench/adapters/agent_search.py", versions={"agent_search": settings},
                         limit=arguments.limit, shard=(shard[0], shard[1]))
    print(json.dumps({**counts, **settings}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
