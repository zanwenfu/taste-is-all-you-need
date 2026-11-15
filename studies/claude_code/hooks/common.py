"""What the study's hooks share: Claude Code's hook input, a function named "module:function", and the cap on context a hook adds."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

# Claude Code saves hook context over 10,000 characters to a file and shows the agent only a 2,000-character preview, so a longer
# block would reach the agent cut. Every block the hooks return stays under CONTEXT_LIMIT, well inside the cap.
CONTEXT_CAP = 10_000
CONTEXT_LIMIT = 9_000
_NOTE_ROOM = 80  # the line saying how many entries were left out


def now() -> str:
    """The time in UTC, to the millisecond, as the driver's turn lines carry it, so the scorer can place a hook's line in a turn."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def read_event(stream: TextIO) -> dict[str, Any]:
    """The hook's input. Empty or broken input reads as an empty event, so a hook never fails the user's prompt on it."""
    text = stream.read()
    try:
        event = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        return {}
    return event if isinstance(event, dict) else {}


def load(target: str) -> Callable[..., Any]:
    """The function named "package.module:function"."""
    module, _, name = target.partition(":")
    if not module or not name:
        raise ValueError(f'expected "module:function", got {target!r}')
    return getattr(importlib.import_module(module), name)


def fit(header: str, ranked: list[tuple[Any, str]], limit: int = CONTEXT_LIMIT) -> str:
    """The header and the entries that fit in ``limit`` characters, taken in ``ranked`` order (most wanted first) and shown sorted by
    their key. Each entry is one line; the most wanted one is cut rather than left out when it cannot fit whole."""
    kept: list[tuple[Any, str]] = []
    used = len(header) + _NOTE_ROOM
    for key, text in ranked:
        line = "\n- " + " ".join(str(text).split())
        if used + len(line) > limit:
            if not kept:
                kept.append((key, line[: max(limit - used - 3, 0)] + "..."))
            break
        kept.append((key, line))
        used += len(line)
    left = len(ranked) - len(kept)
    note = f"\n({left} more left out to stay under the size limit.)" if left else ""
    return header + note + "".join(line for _, line in sorted(kept, key=lambda item: item[0]))


def context_output(event_name: str, text: str) -> str:
    """The JSON a hook prints to add context; Claude Code ignores additionalContext at the top level, so it goes in hookSpecificOutput."""
    return json.dumps({"hookSpecificOutput": {"hookEventName": event_name, "additionalContext": text}})


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(record) + "\n")
