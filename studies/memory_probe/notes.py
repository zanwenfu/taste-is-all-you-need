"""A notes file one model keeps as a conversation goes, for the direct test of version control against flat markdown.

The conversation is read in rounds (a user turn and the replies after it). After each round the model rewrites the
whole notes file, keeping what it judges an assistant may need later, as agents that keep their own notes do. The
same versions then serve two arms: the notes file as it stands at the end (a2c), and the same file with every
earlier version kept in git and searchable (a2d). Same writes, with and without history.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAX_TURN_CHARS = 24_000
_PROMPT = (
    "You keep the notes file of an assistant that is helping a user. Its conversation history will later be "
    "summarized and most of it removed, and the notes file is kept. Rewrite the whole notes file so that it keeps "
    "anything from the conversation the assistant may need later{told}. Reply with the new notes file only.\n\n"
    "The notes file now:\n{notes}\n\nThe newest part of the conversation:\n{part}"
)
# The one clause under test: whether the note-taker is told that what the user asked for matters.
STYLES = {"told": ", including anything the user asked it to keep doing", "neutral": ""}


def chunks(messages: list[dict]) -> list[list[dict]]:
    """Rounds: a user turn and the replies after it; anything before the first user turn is its own round."""
    out: list[list[dict]] = []
    for turn in messages:
        if turn["role"] == "user" or not out:
            out.append([])
        out[-1].append(turn)
    return out


def take_notes(messages: list[dict], ask: Callable[[str], dict], *, style: str = "told") -> list[str]:
    """Every version, in order. A reply cut off by the output cap stops the case: a notes file cut at the cap would lose
    its end silently, and the notes arms would look worse for no real reason."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}")
    notes, versions = "", []
    for part in chunks(messages):
        text = "\n".join(f"[{turn['role']}]: {str(turn['content'])[:MAX_TURN_CHARS]}" for turn in part)
        reply = ask(_PROMPT.format(told=STYLES[style], notes=notes or "(no notes yet)", part=text))
        if reply["finish_reason"] == "length":
            raise ValueError("a notes version was cut off by the output cap")
        notes = reply["text"]
        versions.append(notes)
    return versions


def write_notes(cases: list[dict], ask: Callable[[str], dict], path: Path, *, workers: int, style: str = "told") -> dict[str, int]:
    """Every case's versions, written as each case finishes; a case that failed is asked again next time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in path.read_text().splitlines()} if path.exists() else set()
    todo = [case for case in cases if case["id"] not in done]
    lock, failed = threading.Lock(), [0]

    def one(case: dict) -> None:
        try:
            versions = take_notes(case["messages"], ask, style=style)
        except Exception:
            with lock:
                failed[0] += 1
            return
        with lock, path.open("a") as written:
            written.write(json.dumps({"id": case["id"], "versions": versions}) + "\n")

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, todo))
    return {"cases": len(cases), "written": len(todo) - failed[0], "failed": failed[0]}


def main(argv=None) -> int:
    import argparse

    from studies.memory_probe.headers import served_ask

    parser = argparse.ArgumentParser(description="Write each case's notes versions with the served model.")
    for name in ("inputs", "url", "model", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--style", default="told", choices=tuple(STYLES))
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args(argv)
    cases = [json.loads(line) for line in (Path(arguments.inputs) / "cases.jsonl").read_text().splitlines()]
    cases = cases[: arguments.limit] if arguments.limit else cases
    ask = served_ask(arguments.url, arguments.model, max_tokens=arguments.max_tokens)
    counts = write_notes(cases, ask, Path(arguments.out), workers=arguments.workers, style=arguments.style)
    print(json.dumps(counts))
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
