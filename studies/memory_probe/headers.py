"""Headers for the header arm: one model-written "what" line and keywords per distinct turn, written once.

The 15 cases of a history share every turn but the one holding the rule, so a turn's header is keyed by its words
and asked for once. A turn longer than ``MAX_TURN_CHARS`` is cut before the header model sees it (full-text search
over the turn itself is the backstop). The parse is the comparison's own; every call is counted for the ledger.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from studies.memory_bench.headers import _parsed
from taste.memstore import CommitHeader

MAX_TURN_CHARS = 24_000
PROMPT = (
    "Below is one turn of a past conversation between a user and an assistant. Say what it contains, so that "
    "someone looking for something later can tell whether to open it. Reply with one JSON object and nothing "
    'else, in this form: {"what": "one sentence saying what this turn contains", "keywords": ["up to 12 words '
    'or short names someone might search for"]}\n\nTurn:\n'
)


def turn_key(turn: dict) -> str:
    text = json.dumps({"role": turn["role"], "content": turn["content"]}, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()


def write_turn_headers(cases: list[dict], ask: Callable[[str], dict], path: Path, *, workers: int) -> dict[str, int]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = {row["key"] for row in map(json.loads, path.read_text().splitlines()) if row["ok"]} if path.exists() else set()
    turns = [turn for case in cases for turn in case["messages"]]
    todo = {turn_key(turn): turn for turn in turns if turn_key(turn) not in stored}
    lock, failed = threading.Lock(), [0]

    def one(item: tuple[str, dict]) -> None:
        key, turn = item
        try:
            reply = ask(PROMPT + f"{turn['role']}: {str(turn['content'])[:MAX_TURN_CHARS]}")
        except Exception:
            with lock:
                failed[0] += 1
            return
        parsed = None if reply["finish_reason"] == "length" else _parsed(reply["text"])
        row = {"key": key, "what": parsed[0] if parsed else "", "keywords": parsed[1] if parsed else [],
               "ok": parsed is not None, "finish_reason": reply["finish_reason"], "prompt_tokens": reply["prompt_tokens"],
               "completion_tokens": reply["completion_tokens"], "raw": reply["text"]}
        with lock, path.open("a") as written:
            written.write(json.dumps(row) + "\n")
            failed[0] += parsed is None

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, todo.items()))
    return {"turns": len(turns), "calls": len(todo), "failed": failed[0]}


def served_ask(url: str, model: str, *, max_tokens: int = 1024) -> Callable[[str], dict]:
    """One user message to the served auxiliary model, over plain HTTP, no tools."""
    from studies.memory_probe.tool_loop import http_chat

    chat = http_chat(url, model, max_tokens=max_tokens)

    def ask(prompt: str) -> dict:
        reply = chat([{"role": "user", "content": prompt}], None)
        choice = reply.choices[0]
        return {"text": choice.message.content or "", "finish_reason": choice.finish_reason,
                "prompt_tokens": reply.usage.prompt_tokens, "completion_tokens": reply.usage.completion_tokens}

    return ask


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Write the header arm's headers, once per distinct turn.")
    for name in ("inputs", "url", "model", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args(argv)
    cases = [json.loads(line) for line in (Path(arguments.inputs) / "cases.jsonl").read_text().splitlines()]
    cases = cases[: arguments.limit] if arguments.limit else cases
    counts = write_turn_headers(cases, served_ask(arguments.url, arguments.model), Path(arguments.out),
                                workers=arguments.workers)
    print(json.dumps(counts))
    return 0 if counts["failed"] == 0 else 1


class StoredTurnHeaders:
    """Gives each commit the header stored for its turn's words; a turn with none is counted."""

    def __init__(self, path: Path) -> None:
        self.headers: dict[str, Any] = {row["key"]: row for row in map(json.loads, Path(path).read_text().splitlines())}
        self.missing = 0

    def __call__(self, number: int, turn: dict) -> CommitHeader:
        row = self.headers.get(turn_key(turn))
        if row is None or not row["ok"]:
            self.missing += 1
            return CommitHeader()
        return CommitHeader(what=row["what"], keywords=tuple(row["keywords"]))


if __name__ == "__main__":
    raise SystemExit(main())
