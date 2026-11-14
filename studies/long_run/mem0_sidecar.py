"""Mem0 for one long session, in Mem0's own environment, reached over a local port (see ``mem0_store``).

    MEM0_TELEMETRY=False <venv-mem0 python> -m studies.long_run.mem0_sidecar --url URL --model NAME --max-tokens 4096 \\
        --thinking user --thinking-max-tokens 16384 --effort low --embedder sentence-transformers/all-MiniLM-L6-v2 --dims 384 \\
        --workdir DIR --replies replies.jsonl --calls calls.jsonl

It builds Mem0 exactly as the comparison does (``mem0_arm.mem0_config``: the served model through Mem0's vLLM
provider, the local embedder, an in-memory vector store), and Mem0 makes its own requests, JSON mode included. Its model
calls carry what every other call in the session carries: the reasoning effort, a 30-minute time-out and three tries
(``tool_loop.http_chat``). It counts every model call, its tokens and finish reason, and every reply that does not parse,
prints one line {"ready": PORT}, and answers POST /add, POST /search, POST /drain and GET /stats until it is stopped.
Mem0 is imported only in ``main``, so the server itself can be tested without it.

Thinking (--thinking user). An add whose exchange holds a message from the user is written with the model thinking, as
Taste's rule reader reads each user message, under --thinking-max-tokens; a reply cut off there is asked for once more
without thinking (see ``Calls``). An add of command output only is written without thinking, at the chat comparison's
cap. On a three-task session before the full run, thinking replies that ended used 299 to 6,214 tokens, and the one that
did not end ran to the 32,768 cap on all three tries, twelve minutes each. Measured on Laguna
before the full run: with thinking on, a command's output led it to think until the cap on most such exchanges (8,192
tokens in 173 s and 177 s, an empty reply), so thinking everywhere would cost hours per session and lose those exchanges
anyway; with thinking off, JSON mode gave an empty list even for the session's first message, which holds two standing
instructions, and without JSON mode it now and then answered as the coding agent instead of writing memories. With thinking
on and JSON mode, the first message's three instructions were written down. --thinking on or off applies to every add.

Adds are made by one worker thread, one at a time in the order they came, so the session goes on while Mem0 writes. A
search first waits until every add before it is made, so Mem0 holds at each search exactly what it would hold had the
session waited for each add; only the wall time changes.

An add that fails keeps nothing of its exchange and the session goes on, as Mem0 itself does with a reply it cannot parse:
Mem0 2.2.1 reads each item of the model's memory list with .get, and on Laguna one list held a memory and then two strings
(seed 135, Nov 14), so that add raised before anything was stored, and the session, which then ended at the first failed add,
lost three hours. Each failure is counted and written to --failures with its exchange. Three adds failing in a row end the
session as before: that is a broken Mem0 or server, not one bad reply.
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
FAILS_IN_A_ROW = 3


class Adds:
    """Mem0's adds, made in order by one worker thread. ``before`` is told, before each add, whether its exchange holds a
    message from the user. An add that fails keeps nothing of its exchange; it is counted, written to ``failures`` with the
    exchange, and the adds go on. Once FAILS_IN_A_ROW fail in a row, no later add is made and every later request answers
    with that failure, so the session ends with it."""

    def __init__(self, memory: Any, before: Callable[[bool], None] = lambda user: None, failures: Path | None = None) -> None:
        self.memory, self.before, self.failures, self.waiting = memory, before, failures, queue.Queue()
        self.made, self.failed, self.in_a_row, self.seconds, self.longest, self.error = 0, 0, 0, 0.0, 0.0, None
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self) -> None:
        while True:
            messages, user = self.waiting.get()
            try:
                if self.error is None:
                    started = time.perf_counter()
                    self.before(user)
                    try:
                        self.memory.add(messages, user_id=USER, infer=True)
                    except Exception as error:
                        self._failed(messages, f"{type(error).__name__}: {error}"[:1000])
                    else:
                        took = time.perf_counter() - started
                        self.made, self.seconds, self.longest, self.in_a_row = self.made + 1, self.seconds + took, max(self.longest, took), 0
            except Exception as error:
                self.error = f"{type(error).__name__}: {error}"[:1000]
            finally:
                self.waiting.task_done()

    def _failed(self, messages: list[dict], error: str) -> None:
        self.failed, self.in_a_row = self.failed + 1, self.in_a_row + 1
        if self.failures is not None:
            with self.failures.open("a") as written:
                written.write(json.dumps({"add": self.made + self.failed, "error": error, "messages": messages}) + "\n")
        if self.in_a_row >= FAILS_IN_A_ROW:
            self.error = f"{FAILS_IN_A_ROW} Mem0 adds in a row failed, the last: {error}"

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError(self.error)

    def put(self, messages: list[dict], user: bool = False) -> int:
        self.check()
        self.waiting.put((messages, user))
        return self.waiting.unfinished_tasks

    def drain(self) -> None:
        self.waiting.join()
        self.check()

    def stats(self) -> dict[str, Any]:
        return {"adds_made": self.made, "adds_failed": self.failed, "adds_waiting": self.waiting.unfinished_tasks, "add_seconds": round(self.seconds, 1),
                "add_seconds_longest": round(self.longest, 1), "add_error": self.error}


class Calls:
    """Every model call Mem0 makes, as the server answered it, wrapping ``completions.create`` of Mem0's OpenAI client in
    place. ``thinking`` (set before each add) chooses the call's form: with it, the model thinks under ``thinking_cap``, and
    a reply cut off by that cap is asked for once more without thinking, under ``plain_cap``; without it, the model answers
    at once under ``plain_cap``. Asking a cut-off thinking reply again the same way does not help: on Laguna before the full
    run, every one of six such tries was cut off again, at about twelve minutes each, and Mem0 then kept nothing of the
    exchange. Each try is one line in ``log`` (seconds, thinking, finish reason, tokens); the totals count every try, the
    plain tries made after a cut-off, and a reply that reached Mem0 still cut off."""

    def __init__(self, completions: Any, *, plain_cap: int, thinking_cap: int, log: Path | None = None) -> None:
        self.create, self.log, self.plain_cap, self.thinking_cap = completions.create, log, plain_cap, thinking_cap
        self.thinking = False
        self.calls = self.thinking_calls = self.prompt_tokens = self.completion_tokens = self.cut_off = self.fell_back = self.left_cut_off = 0
        completions.create = self

    def _one(self, params: dict, thinking: bool, attempt: int) -> tuple[Any, str | None]:
        extra = dict(params.get("extra_body") or {})
        extra["chat_template_kwargs"] = {**extra.get("chat_template_kwargs", {}), "enable_thinking": thinking}
        started = time.perf_counter()
        reply = self.create(**{**params, "extra_body": extra, "max_tokens": self.thinking_cap if thinking else self.plain_cap})
        finish = reply.choices[0].finish_reason if reply.choices else None
        usage = getattr(reply, "usage", None)
        prompt, completion = (getattr(usage, "prompt_tokens", 0) or 0), (getattr(usage, "completion_tokens", 0) or 0)
        self.calls, self.thinking_calls = self.calls + 1, self.thinking_calls + thinking
        self.prompt_tokens, self.completion_tokens = self.prompt_tokens + prompt, self.completion_tokens + completion
        self.cut_off += finish == "length"
        if self.log is not None:
            with self.log.open("a") as written:
                written.write(json.dumps({"seconds": round(time.perf_counter() - started, 1), "thinking": thinking, "try": attempt,
                                          "finish": finish, "prompt_tokens": prompt, "completion_tokens": completion}) + "\n")
        return reply, finish

    def __call__(self, **params: Any) -> Any:
        thinking = self.thinking
        reply, finish = self._one(params, thinking, 1)
        if thinking and finish == "length":
            self.fell_back += 1
            reply, finish = self._one(params, False, 2)
        self.left_cut_off += finish == "length"
        return reply

    def totals(self) -> dict[str, int]:
        return {"model_calls": self.calls, "thinking_calls": self.thinking_calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "cut_off": self.cut_off, "fell_back": self.fell_back, "left_cut_off": self.left_cut_off}


def serve(memory: Any, *, stats: Callable[[], dict], port: int = 0, before: Callable[[bool], None] = lambda user: None,
          failures: Path | None = None) -> ThreadingHTTPServer:
    """A server on 127.0.0.1 for one Mem0 ``memory``; the caller runs serve_forever. Adds go to one worker, in order."""
    adds = Adds(memory, before, failures)

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
                    self._send({"waiting": adds.put(body["messages"], bool(body.get("user")))})
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
    parser.add_argument("--max-tokens", type=int, default=2000, help="the output cap of a call without thinking")
    parser.add_argument("--thinking", choices=("on", "off", "user"), default="on",
                        help="user: think only on an add whose exchange holds a message from the user")
    parser.add_argument("--thinking-max-tokens", type=int, default=16384, help="the output cap of a call with thinking")
    parser.add_argument("--effort", help="the reasoning effort on every model call, as the session's own calls send it")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds for one model call, as the session's own calls")
    parser.add_argument("--tries", type=int, default=3, help="tries of a model call that fails to answer")
    parser.add_argument("--calls", help="a file for one line per model call: seconds, thinking, finish reason and tokens")
    parser.add_argument("--failures", help="a file for one line per add that failed: its number, the error and its exchange")
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
    calls = Calls(memory.llm.client.chat.completions, plain_cap=arguments.max_tokens, thinking_cap=arguments.thinking_max_tokens,
                  log=Path(arguments.calls) if arguments.calls else None)
    counted = Counted(memory.llm, log=Path(arguments.replies), extra={"reasoning_effort": arguments.effort} if arguments.effort else None)
    memory.llm = counted

    def before(user: bool) -> None:
        calls.thinking = arguments.thinking == "on" or (arguments.thinking == "user" and user)

    server = serve(memory, stats=lambda: {"llm_calls": counted.calls, "llm_unparsed": counted.unparsed, **calls.totals()}, before=before,
                   failures=Path(arguments.failures) if arguments.failures else None)
    print(json.dumps({"ready": server.server_address[1]}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
