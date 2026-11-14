"""Mem0 as a long session's memory, in the place of Taste's store in the search arm.

Mem0 2.2.1 lives in its own environment (the comparison's venv-mem0), so it runs in a sidecar process
(``mem0_sidecar``) that this side reaches over a local port. What it is given is what Taste's search arm stores:
every message the agent records, in order. Mem0 takes a user side and an assistant side, so each add is one
exchange, the turns since the agent last spoke and the agent's reply, and a command's output goes in as what the
user side saw. At each new task the harness searches with the task's own words, as it does for Taste, and the
memories that come back go in front of the agent under the same header and the same token budget.

The session's own model writes Mem0's memories, with the session's own reasoning effort, as it writes the agent's own
notes in the notes arms. Mem0 makes its adds in order while the session goes on, and a search waits for every add
before it (see ``mem0_sidecar``). No taste/ code is in this path, so its ledger entry is a baseline.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from studies.memory_bench.core import fill
from studies.memory_ledger import BASELINE, stand_in_record

TOP_K = 1000  # as in the comparison: more memories than any budget here can hold
CODE = "studies/long_run/mem0_store.py with studies/long_run/mem0_sidecar.py (mem0ai 2.2.1)"


class Mem0Turns:
    """One session's Mem0. ``client`` has add(messages), search(query, limit), stats() and close()."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.pending: list[dict] = []
        self.user_spoke = False  # the waiting exchange holds a message from the user (not the harness's, not a command's output)
        self.commits = 0  # adds made, so that a search before anything is stored can be skipped, as for Taste
        self.searches = 0
        self.search_seconds = 0.0  # what the searches waited, for Mem0's adds before them and for the search itself

    def add(self, turn: dict) -> None:
        role = turn["role"]
        if role == "assistant":
            self.pending.append({"role": "assistant", "content": turn["content"]})
            self.flush()
            return
        users_words = role == "user" and not turn.get("from_harness")
        if users_words and self.pending:  # what came before the user's message goes in on its own, so Mem0's model, which thinks on the
            self.flush()                  # user's words, reads them with the agent's reply alone, as Taste's rule reader reads them alone
        content = turn["content"] if role == "user" else f"Command output:\n{turn['content']}"
        self.pending.append({"role": "user", "content": content})
        self.user_spoke = self.user_spoke or users_words

    def flush(self) -> None:
        """Add what is waiting as one exchange: before a search, so that it sees everything recorded, and at the end."""
        if self.pending:
            self.client.add(self.pending, user=self.user_spoke)
            self.pending, self.user_spoke = [], False
            self.commits += 1

    def search_with(self, request: str, budget: int, count: Callable[[str], int]) -> tuple[str, list[dict], list[dict], int]:
        """One search with these words. The best memories are kept in Mem0's order until the budget is full, the last cut
        at a word, and shown oldest first, as Taste's search shows what it found. Returns (the block, the call, the
        memories kept, their tokens)."""
        self.searches += 1
        started = time.monotonic()
        hits = self.client.search(request, limit=TOP_K)
        self.search_seconds += time.monotonic() - started
        call = {"name": "mem0.search", "arguments": json.dumps({"query": request, "limit": TOP_K}), "output": json.dumps(hits), "failed": False}
        kept, used = [], 0
        for rank, hit in enumerate(hits):
            if used >= budget:
                break
            text = fill([hit["memory"]], budget - used, count)
            if not text:
                break
            kept.append({"rank": rank, "created_at": hit.get("created_at") or "", "id": hit.get("id"), "memory": text})
            used += count(text)
        shown = sorted(kept, key=lambda memory: (memory["created_at"], memory["rank"]))
        return "\n".join(f"- {memory['memory']}" for memory in shown), [call], kept, used

    def drain(self) -> None:
        """Wait until Mem0 has made every add sent to it."""
        self.client.drain()

    def stats(self) -> dict[str, Any]:
        return {"adds": self.commits, "searches": self.searches, "search_seconds": round(self.search_seconds, 1), **self.client.stats()}

    def ledger(self, mode: str = "single_query") -> dict[str, Any]:
        """A baseline: no taste/ code in the path. The harness searched it once at each new task."""
        return stand_in_record(BASELINE, code=f"{CODE}, searched by the harness at each new task")

    def close(self) -> None:
        try:
            self.flush()
        finally:  # the sidecar does not outlive the session, even when the last add fails (seed 135's did, Nov 14)
            self.client.close()


class SidecarClient:
    """The session's side of the sidecar: JSON over a local port, standard library only. An add only queues its exchange,
    so it answers at once. A search and a drain wait for every add before them, with no time limit of their own: each
    model call has its own time-out and tries, and a sidecar that stops closes the connection."""

    def __init__(self, url: str, *, process: subprocess.Popen | None = None, timeout: float = 120.0) -> None:
        self.url, self.process, self.timeout = url.rstrip("/"), process, timeout

    def _post(self, path: str, body: dict, timeout: float | None) -> Any:
        request = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as reply:
                return json.loads(reply.read())
        except urllib.error.HTTPError as error:  # the session's record says why, not only that the sidecar answered 500
            raise RuntimeError(f"Mem0's sidecar answered {error.code} on {path}: {error.read().decode(errors='replace')[:1000]}") from None

    def add(self, messages: list[dict], user: bool = False) -> None:
        self._post("/add", {"messages": messages, "user": user}, self.timeout)

    def search(self, query: str, limit: int) -> list[dict]:
        return self._post("/search", {"query": query, "limit": limit}, None)["results"]

    def drain(self) -> None:
        self._post("/drain", {}, None)

    def stats(self) -> dict[str, Any]:
        with urllib.request.urlopen(self.url + "/stats", timeout=60) as reply:
            return json.loads(reply.read())

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()


def start_sidecar(command: list[str], *, url: str, model: str, max_tokens: int, effort: str | None, thinking: str, thinking_max_tokens: int,
                  embedder: str, dims: int, log: Path, workdir: Path, wait: float = 900.0) -> SidecarClient:
    """Start Mem0 in its own environment (``command`` runs that environment's Python) and wait for its ready line. Its
    model calls carry ``effort`` as the session's own calls do; ``thinking`` is on, off or user (see ``mem0_sidecar``); each reply goes to log.replies.jsonl and each call's
    seconds, finish reason and tokens to log.calls.jsonl."""
    root = Path(__file__).resolve().parents[2]
    log.parent.mkdir(parents=True, exist_ok=True)
    argv = [*command, "-m", "studies.long_run.mem0_sidecar", "--url", url, "--model", model, "--max-tokens", str(max_tokens),
            "--embedder", embedder, "--dims", str(dims), "--workdir", str(workdir), "--replies", str(log.with_suffix(".replies.jsonl")),
            "--calls", str(log.with_suffix(".calls.jsonl")), "--failures", str(log.with_suffix(".failed.jsonl")), "--thinking", thinking,
            "--thinking-max-tokens", str(thinking_max_tokens), *(["--effort", effort] if effort else [])]
    with log.open("a") as errors:
        process = subprocess.Popen(argv, cwd=root, stdout=subprocess.PIPE, stderr=errors, text=True)
    started = time.monotonic()
    assert process.stdout is not None
    for line in process.stdout:
        if line.startswith("{") and '"ready"' in line:
            port = json.loads(line)["ready"]
            return SidecarClient(f"http://127.0.0.1:{port}", process=process)
        if time.monotonic() - started > wait:
            break
    process.kill()
    raise RuntimeError(f"Mem0's sidecar did not start; see {log}")
