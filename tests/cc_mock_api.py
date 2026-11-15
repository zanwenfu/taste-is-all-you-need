"""A local stand-in for the Messages API, so that the real ``claude`` binary can be tested with no model and no network.

Claude Code is pointed at it with ``ANTHROPIC_BASE_URL=http://127.0.0.1:<port>`` and a dummy key that is not a key.
Each request of the agent loop (one that offers the Bash tool) gets the next reply of a script, as a stream of
server-sent events or as one JSON body, whichever the request asks for; any other request (a title, a probe) gets a
short text. Every request body is logged, one JSON line each, so a test can read the system prompt, the tools and the
tool results Claude Code sent.

    python3 -I tests/cc_mock_api.py --script turns.json --log requests.jsonl --port-file port.txt

The script is a JSON list of replies; each reply is a list of content blocks: {"type": "text", "text": ...} or
{"type": "tool_use", "name": ..., "input": {...}}. Past its end, every reply is a short text.
"""

from __future__ import annotations

import argparse
import itertools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_IDS = itertools.count(1)


def _usage():
    return {"input_tokens": 10, "output_tokens": 5, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


def _blocks(reply):
    out = []
    for block in reply:
        if block["type"] == "tool_use":
            out.append({"type": "tool_use", "id": f"toolu_mock{next(_IDS):04d}", "name": block["name"],
                        "input": block["input"]})
        else:
            out.append({"type": "text", "text": block["text"]})
    return out


def _message(model, blocks):
    stop = "tool_use" if any(block["type"] == "tool_use" for block in blocks) else "end_turn"
    return {"id": f"msg_mock{next(_IDS):04d}", "type": "message", "role": "assistant", "model": model,
            "content": blocks, "stop_reason": stop, "stop_sequence": None, "usage": _usage()}


def _events(message):
    """The message as the streaming API sends it."""
    head = {**message, "content": [], "stop_reason": None}
    yield "message_start", {"type": "message_start", "message": head}
    for index, block in enumerate(message["content"]):
        if block["type"] == "tool_use":
            yield "content_block_start", {"type": "content_block_start", "index": index,
                                          "content_block": {**block, "input": {}}}
            yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                          "delta": {"type": "input_json_delta",
                                                    "partial_json": json.dumps(block["input"])}}
        else:
            yield "content_block_start", {"type": "content_block_start", "index": index,
                                          "content_block": {"type": "text", "text": ""}}
            yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                          "delta": {"type": "text_delta", "text": block["text"]}}
        yield "content_block_stop", {"type": "content_block_stop", "index": index}
    yield "message_delta", {"type": "message_delta", "delta": {"stop_reason": message["stop_reason"],
                                                               "stop_sequence": None},
                            "usage": {"output_tokens": 5}}
    yield "message_stop", {"type": "message_stop"}


class Script:
    def __init__(self, replies, log):
        self.replies, self.log, self.lock, self.turn = list(replies), Path(log), threading.Lock(), 0

    def next(self, body):
        with self.lock:
            with self.log.open("a") as handle:
                handle.write(json.dumps(body) + "\n")
            tools = [tool.get("name") for tool in body.get("tools") or ()]
            if "Bash" not in tools:
                return [{"type": "text", "text": "ok"}]
            reply = self.replies[self.turn] if self.turn < len(self.replies) else [{"type": "text", "text": "done"}]
            self.turn += 1
            return reply


def handler(script):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            size = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(size) or b"{}")
            if not self.path.split("?")[0].endswith("/v1/messages"):
                with script.lock, script.log.open("a") as handle:
                    handle.write(json.dumps({"_path": self.path}) + "\n")
                self.send_response(404)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"type":"error","error":{"type":"not_found_error","message":"mock"}}')
                return
            message = _message(body.get("model", "mock"), _blocks(script.next(body)))
            if body.get("stream"):
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for name, data in _events(message):
                    self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
                    self.wfile.flush()
            else:
                raw = json.dumps(message).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
    return Handler


def serve(script_path, log, port_file):
    script = Script(json.loads(Path(script_path).read_text()), log)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(script))
    Path(port_file).write_text(str(server.server_address[1]))
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--port-file", required=True)
    args = parser.parse_args()
    serve(args.script, args.log, args.port_file)
