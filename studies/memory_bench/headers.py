"""The header's model-written part, written once before any retrieval run, with its calls on the ledger.

Mechanical fields add almost no new words, so a header with only those would test little. The user's idea is
telling an agent what the content is, and that needs a model: the one auxiliary model every system that needs
one shares. It sees only answer-stripped text, and never a session id or a date. The same words get the same
header, so the same session in two questions costs one call. A reply that is not the right JSON is counted and
leaves a header that says nothing; full-text search over the contents is the backstop.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from studies.memory_bench.lme import turns_text
from studies.memory_bench.reader import served_ask  # noqa: F401 - the same call as the reader's
from taste.memstore import CommitHeader

PROMPT = (
    "Below is one past conversation. Say what it contains, so that someone "
    "looking for something later can tell whether to open it. Reply with one JSON object and nothing else, in "
    'this form: {"what": "one sentence saying what this conversation contains", "keywords": ["up to 12 words '
    'or short names someone might search for"]}\n\nConversation:\n'
)


def _key(session: dict) -> str:
    return hashlib.sha256(json.dumps(session["turns"], sort_keys=True).encode("utf-8", "surrogateescape")).hexdigest()


def _parsed(text: str) -> tuple[str, list[str]] | None:
    start, end = text.find("{"), text.rfind("}")
    try:
        value = json.loads(text[start : end + 1]) if start >= 0 else None
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("what"), str):
        return None
    words = value.get("keywords")
    return value["what"], [word for word in words if isinstance(word, str)] if isinstance(words, list) else []


def write_all(views: list[dict], ask: Callable[[str], dict], path: Path, *, workers: int, settings: dict[str, Any] | None = None) -> dict[str, int]:
    """Ask for a header for every distinct session not yet stored, and count the calls. ``settings`` (model, cap, the prompt's hash) goes on every row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = {json.loads(line)["key"] for line in path.read_text().splitlines()} if path.exists() else set()
    sessions = [session for view in views for session in view["sessions"]]
    todo = {_key(session): session for session in sessions if _key(session) not in stored}
    lock = threading.Lock()
    failed = [0]
    errors = [0]
    tokens = [0, 0]

    def one(item: tuple[str, dict]) -> None:
        key, session = item
        try:
            reply = ask(PROMPT + turns_text(session["turns"]))
        except Exception:
            with lock:
                failed[0] += 1
                errors[0] += 1
            return
        parsed = None if reply.get("finish_reason") == "length" else _parsed(reply["text"])
        row = {"key": key, "what": parsed[0] if parsed else "", "keywords": parsed[1] if parsed else [],
               "ok": parsed is not None, "finish_reason": reply.get("finish_reason"), "prompt_tokens": reply["prompt_tokens"],
               "completion_tokens": reply["completion_tokens"], "raw": reply["text"], **(settings or {})}
        with lock, path.open("a") as written:
            written.write(json.dumps(row) + "\n")
            failed[0] += parsed is None
            tokens[0] += reply["prompt_tokens"]
            tokens[1] += reply["completion_tokens"]

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, todo.items()))
    return {"sessions": len(sessions), "calls": len(todo), "failed": failed[0], "errors": errors[0],
            "prompt_tokens": tokens[0], "completion_tokens": tokens[1]}


class StoredHeaders:
    """Gives a commit the header stored for its session's words; a session with none is counted."""

    def __init__(self, path: Path) -> None:
        rows = map(json.loads, Path(path).read_text().splitlines())
        self.headers: dict[str, Any] = {row["key"]: row for row in rows}
        self.missing = 0

    def __call__(self, session: dict) -> CommitHeader:
        row = self.headers.get(_key(session))
        if row is None:
            self.missing += 1
            return CommitHeader()
        return CommitHeader(what=row["what"], keywords=tuple(row["keywords"]))


def stored_writer(path: Path) -> StoredHeaders:
    return StoredHeaders(path)
