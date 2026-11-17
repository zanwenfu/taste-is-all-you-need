"""Claude Code hooks for the git memory (agent-os session, Nov 16 2026). Standard library only: they run inside the task container,
whose Python may be 3.8, and reach the memory service on the machine through the socket in the mounted memory folder.

    prompt   UserPromptSubmit: the project's house rules, with every user message
    post     PostToolUse: when Claude first reads, edits, searches or runs a command on a file or folder since the last
             compaction, the instructions for its folder and the notes on that file
    compact  SessionStart, matcher "compact": the house rules again, and what was delivered is forgotten so it comes again when needed
    stop     Stop (the fair test, Nov 17): the turn just finished is recorded as one commit (the task, Claude's reply, the files changed,
             the tests run); with --reader, the small reader reads the user's message and saves any standing rule Claude did not

With --refresh-folders (the design revised after the first run, Nov 16 ~8:40 pm ET: a folder's instruction delivered once was
skipped when later tasks came back to that folder), each new user message forgets which folder instructions were delivered, so a
folder's instruction comes again with every task that touches it; a file's notes still come once until a compaction.

Every delivery is logged (mode, items, characters) to deliveries.jsonl beside the socket. The hook never blocks Claude: if the
service does not answer it says so in the log and gives nothing.

    python3 -B git_memory_hook.py prompt --memory /cc/memory < hook-input.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

RULES_HEADER = ("Project house rules, from the project memory. Follow every one exactly, in every reply and every tool call; they "
                "override your default habits.")
TOOLS_LINE = ("The project memory also holds instructions for each folder and notes on the code. A folder's instructions and a "
              "file's notes arrive when you first open or change it; for anything else use the memory_search and memory_fetch tools.")
TOUCH_HEADER = "From the project memory, for what you just touched:"
LIVE_HEADER = ("Standing rules and decisions from the project memory, oldest first; where two disagree, the later one stands. Follow "
               "every one exactly, in every reply and every tool call; they override your default habits.")
LIVE_TOOLS = ("The project memory keeps what matters across tasks and sessions. Save rules, decisions and facts with memory_save; a "
              "file's notes arrive when you first open it; for anything else, including what was done in earlier tasks, use "
              "memory_search and memory_fetch.")
MEMORY_FILES = ("CLAUDE.md", "CLAUDE.local.md", ".claude")
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


def _load(memory: Path) -> dict:
    try:
        return json.loads(_state(memory).read_text())
    except (OSError, ValueError):
        return {}


def _delivered(memory: Path, session: str) -> set:
    entry = _load(memory).get(session, {})
    return set(entry.get("ids", [])) if isinstance(entry, dict) else set(entry)


def _remember(memory: Path, session: str, ids: set, folders: set = frozenset()) -> None:
    """What has been delivered in this context: every item's id, and which of them are folder instructions."""
    data = _load(memory)
    old = data.get(session, {})
    kept = set(old.get("folders", [])) if isinstance(old, dict) else set()
    data[session] = {"ids": sorted(ids), "folders": sorted((kept | set(folders)) & set(ids))}
    _state(memory).write_text(json.dumps(data))


def _forget_folders(memory: Path, session: str) -> None:
    data = _load(memory)
    entry = data.get(session, {})
    if isinstance(entry, dict) and entry.get("folders"):
        folders = set(entry["folders"])
        data[session] = {"ids": sorted(set(entry.get("ids", [])) - folders), "folders": []}
        _state(memory).write_text(json.dumps(data))


def _log(memory: Path, record: dict) -> None:
    try:
        with (memory / "deliveries.jsonl").open("a") as log:
            log.write(json.dumps({"time": time.time(), **record}) + "\n")
    except OSError:
        pass


def _output(event_name: str, text: str) -> str:
    return json.dumps({"hookSpecificOutput": {"hookEventName": event_name, "additionalContext": text}})


def rules_text(memory: Path, live: bool = False) -> str:
    rules = ask(memory, {"op": "rules"}).get("rules") or []
    if live:  # the fair test: what Claude and the reader saved, and how to save and look up the rest
        return (LIVE_HEADER + "\n" + "".join(f"- {rule}\n" for rule in rules) + "\n" if rules else "") + LIVE_TOOLS
    return RULES_HEADER + "\n" + "".join(f"- {rule}\n" for rule in rules) + "\n" + TOOLS_LINE


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in content or [] if isinstance(c, dict) and c.get("type") == "text")


def last_turn(transcript: Path) -> dict:
    """From Claude Code's own session transcript: the user's last message, Claude's last words after it, and its shell commands."""
    rows = []
    for line in transcript.read_text(errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    start = None
    for index, row in enumerate(rows):
        message = row.get("message") or {}
        content = message.get("content")
        # a compaction summary is stored as a user message; it is not the user's (the fair test: the reader took 13 paraphrased
        # "rules" from summaries, Nov 17)
        if row.get("isCompactSummary") or row.get("isVisibleInTranscriptOnly"):
            continue
        if row.get("type") == "user" and not row.get("isMeta") and _text_of(content).strip() and not (
                isinstance(content, list) and any(isinstance(c, dict) and c.get("type") == "tool_result" for c in content)):
            start = index
    if start is None:
        return {"user": "", "reply": "", "commands": []}
    reply, commands = "", []
    for row in rows[start + 1:]:
        if row.get("type") != "assistant":
            continue
        for c in (row.get("message") or {}).get("content") or []:
            if isinstance(c, dict) and c.get("type") == "text" and c.get("text", "").strip():
                reply = c["text"]
            if isinstance(c, dict) and c.get("type") == "tool_use" and c.get("name") == "Bash":
                commands.append(str((c.get("input") or {}).get("command", "")))
    return {"user": _text_of((rows[start].get("message") or {}).get("content")), "reply": reply, "commands": commands}


def changed_files(project: Path) -> list:
    done = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(project), "status", "--porcelain", "--untracked-files=all"],
                          capture_output=True, text=True, timeout=60)
    paths = [line[3:].split(" -> ")[-1].strip('"') for line in done.stdout.splitlines() if len(line) > 3]
    return [p for p in paths if not any(p == m or p.startswith(m + "/") for m in MEMORY_FILES)]


def task_title(user: str) -> str:
    title = re.search(r"(?im)^\W*title:?\s*(.+)$", user)
    if title:
        return title.group(1).strip()[:160]
    body = user.split("<pr_description>", 1)[-1]
    return next((line.strip(" #*") for line in body.splitlines() if len(line.strip(" #*")) > 15 and not line.strip().startswith("<")),
                user.strip()[:160])[:160]


def record_turn(event: dict, memory: Path, project: Path, reader: bool) -> dict:
    turn = last_turn(Path(str(event.get("transcript_path") or "")))
    # Claude Code hands the Stop hook the final reply itself; the transcript may not hold it yet when the hook runs (the fair trial:
    # every turn record had an empty reply, Nov 17)
    turn["reply"] = str(event.get("last_assistant_message") or "") or turn["reply"]
    files = changed_files(project)
    tests = [c.replace("\n", " ")[:200] for c in turn["commands"] if "pytest" in c][:8]
    text = (f"Turn record. Task: {task_title(turn['user'])}\nClaude's reply: {turn['reply'][:1500]}\n"
            f"Files changed so far in this task: {', '.join(files) or 'none'}\nTests run: {'; '.join(tests) or 'none'}")
    saved = ask(memory, {"op": "save", "kind": "turn", "source": "hook", "text": text})
    if "error" in saved:  # a failed save is logged as an error, never passed over (the fair trial, Nov 17)
        raise RuntimeError(f"turn not saved: {saved['error']}")
    found, kept = [], 0
    if reader and turn["user"].strip():
        sys.path.insert(0, os.environ.get("CC_STUDY_CODE", "/cc/code"))
        from studies.claude_code.readers import claude_reader  # noqa: E402  (the small model, on the subscription)
        found = claude_reader(turn["user"])
        for rule in found:
            reply = ask(memory, {"op": "save", "kind": "rule", "source": "reader", "text": rule})
            if "error" in reply:
                raise RuntimeError(f"reader's rule not saved: {reply['error']}")
            kept += "saved" in reply
    return {"turn": saved.get("saved"), "reader_found": len(found), "reader_saved": kept, "files": len(files)}


def run(mode: str, event: dict, memory: Path, refresh_folders: bool = False, live: bool = False, reader: bool = False,
        project: Path = Path("/app")) -> str:
    session = str(event.get("session_id") or "")
    try:
        if mode == "stop":  # never blocks Claude: it prints nothing
            if not event.get("stop_hook_active"):
                _log(memory, {"mode": mode, **record_turn(event, memory, project, reader)})
            return ""
        if mode == "prompt":
            if refresh_folders:
                _forget_folders(memory, session)
            text = rules_text(memory, live)
            _log(memory, {"mode": mode, "chars": len(text)})
            return _output("UserPromptSubmit", text)
        if mode == "compact":
            if event.get("source", "compact") != "compact":
                return ""
            _remember(memory, session, set())
            text = rules_text(memory, live)
            _log(memory, {"mode": mode, "chars": len(text)})
            return _output("SessionStart", text)
        paths = touched_paths(event)
        if not paths:
            return ""
        delivered = _delivered(memory, session)
        items = [i for i in ask(memory, {"op": "items_for", "paths": paths}).get("items", []) if i["id"] not in delivered]
        if not items:
            return ""
        _remember(memory, session, delivered | {i["id"] for i in items}, {i["id"] for i in items if i["kind"] == "folder"})
        text = TOUCH_HEADER + "\n" + "".join(f"- {i['text']}\n" for i in items)
        _log(memory, {"mode": mode, "tool": event.get("tool_name"), "paths": paths[:5], "items": len(items),
                      "folder_items": sum(i["kind"] == "folder" for i in items), "chars": len(text)})
        return _output("PostToolUse", text)
    except Exception as error:  # the memory being down must never stop Claude: log it and give nothing
        _log(memory, {"mode": mode, "error": f"{type(error).__name__}: {error}"[:300]})
        return ""


def main(argv=None, stdin=None, stdout=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("prompt", "post", "compact", "stop"))
    parser.add_argument("--memory", type=Path, default=Path("/cc/memory"))
    parser.add_argument("--refresh-folders", action="store_true", help="a folder's instruction again with every task that touches it")
    parser.add_argument("--live", action="store_true", help="the fair test: deliver what was saved, and say how to save")
    parser.add_argument("--reader", action="store_true", help="stop mode: the small reader saves standing rules Claude did not")
    args = parser.parse_args(argv)
    raw = (stdin or sys.stdin).read()
    try:
        event = json.loads(raw) if raw.strip() else {}
    except ValueError:
        event = {}
    output = run(args.mode, event, args.memory, args.refresh_folders, args.live, args.reader)
    if output:
        (stdout or sys.stdout).write(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
