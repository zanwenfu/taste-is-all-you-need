"""Claude Code hooks for the git memory (agent-os session, Nov 16 2026). Standard library only: they run inside the task container,
whose Python may be 3.8, and reach the memory service on the machine through the socket in the mounted memory folder.

    prompt   UserPromptSubmit: the project's house rules, with every user message
    post     PostToolUse: when Claude first reads, edits, searches or runs a command on a file or folder since the last
             compaction, the instructions for its folder and the notes on that file
    compact  SessionStart, matcher "compact": the house rules again, and what was delivered is forgotten so it comes again when needed

Every delivery is logged (mode, items, characters) to deliveries.jsonl beside the socket. The hook never blocks Claude: if the
service does not answer it says so in the log and gives nothing.

    python3 -B git_memory_hook.py prompt --memory /cc/memory < hook-input.json
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import time
from pathlib import Path

RULES_HEADER = ("Project house rules, from the project memory. Follow every one exactly, in every reply and every tool call; they "
                "override your default habits.")
TOOLS_LINE = ("The project memory also holds instructions for each folder and notes on the code. A folder's instructions and a "
              "file's notes arrive when you first open or change it; for anything else use the memory_search and memory_fetch tools.")
TOUCH_HEADER = "From the project memory, for what you just touched:"
_PATHISH = re.compile(r"(?:/app/)?((?:openlibrary|scripts|tests|conf|static|vendor|infogami)(?:/[\w.\-]+)+)")


def ask(memory: Path, request: dict, timeout: float = 30) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout)
        conn.connect(str(memory / "memory.sock"))
        conn.sendall((json.dumps(request) + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
    return json.loads(data or b"{}")


def touched_paths(event: dict) -> list:
    """Repository paths a tool call names: its file or path argument, and paths written in a shell command or a search pattern."""
    given = event.get("tool_input") or {}
    texts = [str(given.get(key, "")) for key in ("file_path", "path", "notebook_path", "command", "pattern", "glob")]
    found = []
    for text in texts:
        for match in _PATHISH.finditer(text):
            path = match.group(1).rstrip("/.,;:)'\"")
            if path not in found:
                found.append(path)
    return found


def _state(memory: Path) -> Path:
    return memory / "delivered.json"


def _delivered(memory: Path, session: str) -> set:
    try:
        return set(json.loads(_state(memory).read_text()).get(session, []))
    except (OSError, ValueError):
        return set()


def _remember(memory: Path, session: str, ids: set) -> None:
    try:
        data = json.loads(_state(memory).read_text())
    except (OSError, ValueError):
        data = {}
    data[session] = sorted(ids)
    _state(memory).write_text(json.dumps(data))


def _log(memory: Path, record: dict) -> None:
    try:
        with (memory / "deliveries.jsonl").open("a") as log:
            log.write(json.dumps({"time": time.time(), **record}) + "\n")
    except OSError:
        pass


def _output(event_name: str, text: str) -> str:
    return json.dumps({"hookSpecificOutput": {"hookEventName": event_name, "additionalContext": text}})


def rules_text(memory: Path) -> str:
    rules = ask(memory, {"op": "rules"}).get("rules") or []
    return RULES_HEADER + "\n" + "".join(f"- {rule}\n" for rule in rules) + "\n" + TOOLS_LINE


def run(mode: str, event: dict, memory: Path) -> str:
    session = str(event.get("session_id") or "")
    try:
        if mode == "prompt":
            text = rules_text(memory)
            _log(memory, {"mode": mode, "chars": len(text)})
            return _output("UserPromptSubmit", text)
        if mode == "compact":
            if event.get("source", "compact") != "compact":
                return ""
            _remember(memory, session, set())
            text = rules_text(memory)
            _log(memory, {"mode": mode, "chars": len(text)})
            return _output("SessionStart", text)
        paths = touched_paths(event)
        if not paths:
            return ""
        delivered = _delivered(memory, session)
        items = [i for i in ask(memory, {"op": "items_for", "paths": paths}).get("items", []) if i["id"] not in delivered]
        if not items:
            return ""
        _remember(memory, session, delivered | {i["id"] for i in items})
        text = TOUCH_HEADER + "\n" + "".join(f"- {i['text']}\n" for i in items)
        _log(memory, {"mode": mode, "tool": event.get("tool_name"), "paths": paths[:5], "items": len(items),
                      "folder_items": sum(i["kind"] == "folder" for i in items), "chars": len(text)})
        return _output("PostToolUse", text)
    except Exception as error:  # the memory being down must never stop Claude: log it and give nothing
        _log(memory, {"mode": mode, "error": f"{type(error).__name__}: {error}"[:300]})
        return ""


def main(argv=None, stdin=None, stdout=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("prompt", "post", "compact"))
    parser.add_argument("--memory", type=Path, default=Path("/cc/memory"))
    args = parser.parse_args(argv)
    raw = (stdin or sys.stdin).read()
    try:
        event = json.loads(raw) if raw.strip() else {}
    except ValueError:
        event = {}
    output = run(args.mode, event, args.memory)
    if output:
        (stdout or sys.stdout).write(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
