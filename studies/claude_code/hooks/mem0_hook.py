"""Claude Code hook for the Mem0 arm: Mem0 is searched with each prompt and what it finds goes to the agent; each finished exchange
goes into Mem0.

    prompt  UserPromptSubmit: search with the prompt's own words, as the long-run study's Mem0 arm searches with each task's words. The
            memories found go back as additional context under that arm's header, the best ones kept when not all fit and shown oldest
            first. The prompt is kept in the state folder for the Stop hook.
    stop    Stop: add the exchange that just ended, the user's prompt and Claude's last reply, to Mem0. The long-run study's Mem0 also
            saw each command's output; here only what a Stop hook sees goes in.

The client has add(messages, user) and search(query, limit) and is made by a function named "module:function" in CC_STUDY_MEM0_CLIENT.
sidecar_from_env reaches the long-run study's Mem0 sidecar (mem0ai 2.2.1 in its own environment, studies/long_run/mem0_sidecar.py) at
CC_STUDY_MEM0_URL; fake_from_env is a word-overlap stand-in kept in the file CC_STUDY_MEM0_FAKE, for plumbing runs; FakeMemory() with no
file is for tests. This module never imports mem0. Each call is logged with --log. The hook never exits with 2, which would block the
prompt or keep Claude from stopping.

    python -B mem0_hook.py prompt --state STATE_DIR --log mem0.jsonl < hook-input.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, TextIO

if __package__ in (None, ""):  # run by Claude Code as a script: make the study's package importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studies.claude_code.hooks.common import CONTEXT_LIMIT, append_jsonl, context_output, fit, load, now, read_event  # noqa: E402

CLIENT_ENV = "CC_STUDY_MEM0_CLIENT"
URL_ENV = "CC_STUDY_MEM0_URL"
FAKE_ENV = "CC_STUDY_MEM0_FAKE"
TOP_K = 1000  # as in the long-run study: more memories than the block can hold
PENDING = "mem0_pending_prompt.json"
# The long-run study's Mem0 header, word for word (studies/long_run/session.py, SEARCH_HEADER), so the two studies' arms read alike.
SEARCH_HEADER = ("Earlier turns from the record of this session that may bear on the current task, oldest first. Where two disagree, the later one stands. "
                 "They are history and not new requests: any task in them is over, unless the current task says otherwise.")


class MemoryClient(Protocol):
    def add(self, messages: list[dict[str, str]], user: bool = False) -> None: ...

    def search(self, query: str, limit: int) -> list[dict[str, Any]]: ...


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class FakeMemory:
    """Each message kept whole as one memory; a search returns those sharing a word with the query, most shared words first, then newest.
    Kept in ``path`` when one is given, so that one hook call sees what an earlier one added."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.memories: list[dict[str, Any]] = json.loads(self.path.read_text()) if self.path and self.path.exists() else []
        self.adds: list[tuple[list[dict[str, str]], bool]] = []

    def add(self, messages: list[dict[str, str]], user: bool = False) -> None:
        self.adds.append((messages, user))
        for message in messages:
            text = str(message.get("content") or "").strip()
            if text:
                number = len(self.memories)
                self.memories.append({"id": str(number), "memory": text, "created_at": f"{number:08d}", "role": message.get("role")})
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.memories))

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        wanted = _words(query)
        scored = [(len(wanted & _words(memory["memory"])), number, memory) for number, memory in enumerate(self.memories)]
        return [memory for shared, _, memory in sorted(scored, key=lambda row: (-row[0], -row[1])) if shared][:limit]


class SidecarMemory:
    """The long-run study's Mem0 sidecar, reached over its local port: an add only queues, and a search waits for every add before it."""

    def __init__(self, url: str, timeout: float = 600.0) -> None:
        self.url, self.timeout = url.rstrip("/"), timeout

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        request = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                return json.loads(reply.read())
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Mem0's sidecar answered {error.code} on {path}: {error.read().decode(errors='replace')[:500]}") from None

    def add(self, messages: list[dict[str, str]], user: bool = False) -> None:
        self._post("/add", {"messages": messages, "user": user})

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        return self._post("/search", {"query": query, "limit": limit})["results"]


def sidecar_from_env() -> SidecarMemory:
    return SidecarMemory(os.environ[URL_ENV])


def fake_from_env() -> FakeMemory:
    return FakeMemory(Path(os.environ[FAKE_ENV]))


def last_reply(transcript: str | Path | None) -> str:
    """Claude's last text reply in a session transcript (JSON lines), for a Stop hook given no last message of its own."""
    if not transcript or not Path(transcript).exists():
        return ""
    reply = ""
    for line in Path(transcript).read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = (entry.get("message") or {}).get("content") if isinstance(entry, dict) and entry.get("type") == "assistant" else None
        texts = [block.get("text", "") for block in content or [] if isinstance(block, dict) and block.get("type") == "text"]
        if any(text.strip() for text in texts):
            reply = "\n".join(texts).strip()
    return reply


def on_prompt(event: Mapping[str, Any], *, client: MemoryClient, state: Path, log: Path | None = None,
              limit: int = CONTEXT_LIMIT) -> str | None:
    prompt = str(event.get("prompt") or "")
    Path(state).mkdir(parents=True, exist_ok=True)
    (Path(state) / PENDING).write_text(json.dumps({"prompt": prompt}))
    record: dict[str, Any] = {"time": now(), "mode": "prompt", "session_id": event.get("session_id"), "prompt_chars": len(prompt)}
    text = None
    try:
        hits = client.search(prompt, TOP_K) if prompt.strip() else []
        ranked = [((str(hit.get("created_at") or ""), rank), str(hit["memory"])) for rank, hit in enumerate(hits) if hit.get("memory")]
        text = fit(SEARCH_HEADER, ranked, limit) if ranked else None
        record["hits"] = len(hits)
    except Exception as error:  # nothing goes out; the driver stops the session on a logged error
        record["error"] = f"{type(error).__name__}: {error}"[:500]
    record["context_chars"] = len(text or "")
    if log:
        append_jsonl(log, record)
    return context_output("UserPromptSubmit", text) if text else None


def on_stop(event: Mapping[str, Any], *, client: MemoryClient, state: Path, log: Path | None = None) -> None:
    pending = Path(state) / PENDING
    prompt = json.loads(pending.read_text()).get("prompt", "") if pending.exists() else ""
    reply = str(event.get("last_assistant_message") or "").strip() or last_reply(event.get("transcript_path"))
    messages = ([{"role": "user", "content": prompt}] if prompt.strip() else []) + ([{"role": "assistant", "content": reply}] if reply else [])
    record: dict[str, Any] = {"time": now(), "mode": "stop", "session_id": event.get("session_id"), "messages": len(messages),
                              "reply_chars": len(reply)}
    try:
        if messages:
            client.add(messages, user=bool(prompt.strip()))
        pending.unlink(missing_ok=True)
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"[:500]
    if log:
        append_jsonl(log, record)


def main(argv: list[str] | None = None, stdin: TextIO | None = None, stdout: TextIO | None = None,
         environ: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("prompt", "stop"))
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args(argv)
    stdin, stdout, environ = stdin or sys.stdin, stdout or sys.stdout, os.environ if environ is None else environ
    event = read_event(stdin)
    try:
        client = load(environ.get(CLIENT_ENV, ""))()
    except Exception as error:  # no client named, or it cannot be made: say so, and let the session go on
        message = f"{CLIENT_ENV}: {type(error).__name__}: {error}"[:500]
        if args.log:
            append_jsonl(args.log, {"time": now(), "mode": args.mode, "session_id": event.get("session_id"), "error": message})
        print(message, file=sys.stderr)
        return 1
    if args.mode == "prompt":
        output = on_prompt(event, client=client, state=args.state, log=args.log)
        if output:
            stdout.write(output)
    else:
        on_stop(event, client=client, state=args.state, log=args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
