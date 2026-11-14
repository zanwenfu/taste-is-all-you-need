"""Mem0 for one long session, in Mem0's own environment, reached over a local port (see ``mem0_store``).

    MEM0_TELEMETRY=False <venv-mem0 python> -m studies.long_run.mem0_sidecar --url URL --model NAME --max-tokens N \\
        --effort low --thinking off --json-mode off --embedder sentence-transformers/all-MiniLM-L6-v2 --dims 384 --workdir DIR --replies replies.jsonl

It builds Mem0 exactly as the comparison does (``mem0_arm.mem0_config``: the served model through Mem0's vLLM
provider, the local embedder, an in-memory vector store), and its model calls carry what every other call in the
session carries: the reasoning effort, a 30-minute time-out and three tries (``tool_loop.http_chat``). With --thinking
off they also ask the served model not to think first (``enable_thinking`` false in its chat template), as Mem0 expects a
model that answers at once: with thinking on, Laguna spent the whole output cap thinking about a command's output on
most exchanges that held one, and Mem0 got an empty reply. With --json-mode off they go without the response_format Mem0
asks for (vLLM's guided JSON): with thinking off and JSON mode, Laguna answered an empty list even to the session's first
message, which holds two standing instructions, and without JSON mode it wrote all of them down; Mem0 reads a JSON
object from a plain reply as well, and a reply it cannot read is counted. It counts every
model call, its tokens and finish reason, and every reply that does not parse, prints one line {"ready": PORT}, and
answers POST /add, POST /search, POST /drain and GET /stats until it is stopped. Mem0 is imported only in ``main``, so
the server itself can be tested without it.

Adds are made by one worker thread, one at a time in the order they came, so the session goes on while Mem0 writes. A
search first waits until every add before it is made, so Mem0 holds at each search exactly what it would hold had the
session waited for each add; only the wall time changes.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

USER = "session"


class Adds:
    """Mem0's adds, made in order by one worker thread. The first add that fails is kept, no later add is made, and every
    later request answers with that failure, so the session ends with it as it would have at the add itself."""

    def __init__(self, memory: Any) -> None:
        self.memory, self.waiting = memory, queue.Queue()
        self.made, self.seconds, self.longest, self.error = 0, 0.0, 0.0, None
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self) -> None:
        while True:
            messages = self.waiting.get()
            try:
                if self.error is None:
                    started = time.perf_counter()
                    self.memory.add(messages, user_id=USER, infer=True)
                    took = time.perf_counter() - started
                    self.made, self.seconds, self.longest = self.made + 1, self.seconds + took, max(self.longest, took)
            except Exception as error:
                self.error = f"{type(error).__name__}: {error}"[:1000]
            finally:
                self.waiting.task_done()

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"a Mem0 add failed: {self.error}")

    def put(self, messages: list[dict]) -> int:
        self.check()
        self.waiting.put(messages)
        return self.waiting.unfinished_tasks

    def drain(self) -> None:
        self.waiting.join()
        self.check()

    def stats(self) -> dict[str, Any]:
        return {"adds_made": self.made, "adds_waiting": self.waiting.unfinished_tasks, "add_seconds": round(self.seconds, 1),
                "add_seconds_longest": round(self.longest, 1), "add_error": self.error}


class Calls:
    """Every model call Mem0 makes, as the server answered it: seconds, finish reason and tokens, one line each in ``log``,
    and their totals. Wraps ``completions.create`` of Mem0's OpenAI client in place. A reply cut off by the output cap
    reaches Mem0 as it is, as it would in Mem0's own use, and is counted. Without ``json_mode``, Mem0's response_format is
    left off each call."""

    def __init__(self, completions: Any, log: Path | None = None, json_mode: bool = True) -> None:
        self.create, self.log, self.json_mode = completions.create, log, json_mode
        self.calls, self.prompt_tokens, self.completion_tokens, self.cut_off = 0, 0, 0, 0
        completions.create = self

    def __call__(self, **params: Any) -> Any:
        if not self.json_mode:
            params.pop("response_format", None)
        started = time.perf_counter()
        reply = self.create(**params)
        finish = reply.choices[0].finish_reason if reply.choices else None
        usage = getattr(reply, "usage", None)
        prompt, completion = (getattr(usage, "prompt_tokens", 0) or 0), (getattr(usage, "completion_tokens", 0) or 0)
        self.calls, self.prompt_tokens, self.completion_tokens = self.calls + 1, self.prompt_tokens + prompt, self.completion_tokens + completion
        self.cut_off += finish == "length"
        if self.log is not None:
            with self.log.open("a") as written:
                written.write(json.dumps({"seconds": round(time.perf_counter() - started, 1), "finish": finish, "prompt_tokens": prompt,
                                          "completion_tokens": completion}) + "\n")
        return reply

    def totals(self) -> dict[str, int]:
        return {"model_calls": self.calls, "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "cut_off": self.cut_off}


def serve(memory: Any, *, stats: Callable[[], dict], port: int = 0) -> ThreadingHTTPServer:
    """A server on 127.0.0.1 for one Mem0 ``memory``; the caller runs serve_forever. Adds go to one worker, in order."""
    adds = Adds(memory)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: Any, status: int = 200) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                if self.path == "/add":
                    self._send({"waiting": adds.put(body["messages"])})
                elif self.path == "/drain":
                    adds.drain()
                    self._send({"waiting": 0})
                elif self.path == "/search":
                    adds.drain()
                    found = memory.search(body["query"], filters={"user_id": USER}, top_k=int(body["limit"]), threshold=0.0)
                    self._send({"results": [{key: hit.get(key) for key in ("memory", "created_at", "id", "score")} for hit in found["results"]]})
                else:
                    self._send({"error": "no such path"}, 404)
            except Exception as error:  # the session's side sees the failure and ends the arm with it
                self._send({"error": f"{type(error).__name__}: {error}"[:1000]}, 500)

        def do_GET(self) -> None:  # noqa: N802
            self._send({**stats(), **adds.stats()}) if self.path == "/stats" else self._send({"error": "no such path"}, 404)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("url", "model", "embedder", "workdir", "replies"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--dims", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--effort", help="the reasoning effort on every model call, as the session's own calls send it")
    parser.add_argument("--thinking", choices=("on", "off"), default="on", help="off asks the model to answer without thinking first")
    parser.add_argument("--json-mode", choices=("on", "off"), default="on", help="off sends Mem0's calls without its response_format")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds for one model call, as the session's own calls")
    parser.add_argument("--tries", type=int, default=3, help="tries of a model call that fails, as the session's own calls")
    parser.add_argument("--calls", help="a file for one line per model call: seconds, finish reason and tokens")
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args(argv)

    if os.environ.get("MEM0_TELEMETRY") != "False":
        raise SystemExit("set MEM0_TELEMETRY=False before starting: Mem0 reads it once, at import")
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        os.environ.pop(key, None)
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"  # any accidental OpenAI client fails here, locally
    os.environ.setdefault("MEM0_DIR", str(Path(arguments.workdir) / "mem0"))

    from mem0 import Memory

    from studies.memory_bench.adapters.mem0_arm import Counted, check_search_parts, mem0_config

    memory = Memory.from_config(mem0_config(arguments.url, arguments.model, arguments.embedder, arguments.dims, arguments.device,
                                            max_tokens=arguments.max_tokens))
    if type(memory.llm).__name__ != "VllmLLM" or type(memory.embedding_model).__name__ != "HuggingFaceEmbedding":
        raise SystemExit("Mem0 is not using the served model and the local embedder")
    check_search_parts(memory)
    memory.llm.client = memory.llm.client.with_options(timeout=arguments.timeout, max_retries=arguments.tries - 1)
    calls = Calls(memory.llm.client.chat.completions, log=Path(arguments.calls) if arguments.calls else None, json_mode=arguments.json_mode == "on")
    extra: dict[str, Any] = {"reasoning_effort": arguments.effort} if arguments.effort else {}
    if arguments.thinking == "off":
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    counted = Counted(memory.llm, log=Path(arguments.replies), extra=extra)
    memory.llm = counted
    server = serve(memory, stats=lambda: {"llm_calls": counted.calls, "llm_unparsed": counted.unparsed, **calls.totals()})
    print(json.dumps({"ready": server.server_address[1]}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
