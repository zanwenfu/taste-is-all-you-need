"""Mem0 for one long session, in Mem0's own environment, reached over a local port (see ``mem0_store``).

    MEM0_TELEMETRY=False <venv-mem0 python> -m studies.long_run.mem0_sidecar --url URL --model NAME --max-tokens N \\
        --embedder sentence-transformers/all-MiniLM-L6-v2 --dims 384 --workdir DIR --replies replies.jsonl

It builds Mem0 exactly as the comparison does (``mem0_arm.mem0_config``: the served model through Mem0's vLLM
provider, the local embedder, an in-memory vector store), counts every model call and every reply that does not
parse, prints one line {"ready": PORT}, and answers POST /add, POST /search and GET /stats until it is stopped.
Mem0 is imported only in ``main``, so the server itself can be tested without it.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

USER = "session"


def serve(memory: Any, *, stats: Callable[[], dict], port: int = 0) -> ThreadingHTTPServer:
    """A server on 127.0.0.1 for one Mem0 ``memory``; the caller runs serve_forever. One request at a time, as one session adds."""

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
                    result = memory.add(body["messages"], user_id=USER, infer=True)
                    self._send({"results": (result or {}).get("results", [])})
                elif self.path == "/search":
                    found = memory.search(body["query"], filters={"user_id": USER}, top_k=int(body["limit"]), threshold=0.0)
                    self._send({"results": [{key: hit.get(key) for key in ("memory", "created_at", "id", "score")} for hit in found["results"]]})
                else:
                    self._send({"error": "no such path"}, 404)
            except Exception as error:  # the session's side sees the failure and ends the arm with it
                self._send({"error": f"{type(error).__name__}: {error}"[:1000]}, 500)

        def do_GET(self) -> None:  # noqa: N802
            self._send(stats()) if self.path == "/stats" else self._send({"error": "no such path"}, 404)

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
    counted = Counted(memory.llm, log=Path(arguments.replies))
    memory.llm = counted
    server = serve(memory, stats=lambda: {"llm_calls": counted.calls, "llm_unparsed": counted.unparsed})
    print(json.dumps({"ready": server.server_address[1]}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
