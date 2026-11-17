"""The git memory's search and fetch as an MCP tool server for Claude Code (agent-os session, Nov 16 2026). Standard library only:
it runs inside the task container (Python 3.8 or later) and passes each call to the memory service through the socket in the
mounted memory folder. Newline-delimited JSON-RPC 2.0 on stdin and stdout, as MCP's stdio transport.

    python3 -B memory_mcp.py --memory /cc/memory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from git_memory_hook import ask  # noqa: E402

PROTOCOL = "2025-06-18"


def respond(request: dict, memory: Path, saving: bool = False) -> dict | None:
    method, rid = request.get("method"), request.get("id")
    if rid is None:  # a notification (initialized, cancelled): no reply
        return None
    try:
        if method == "initialize":
            wanted = (request.get("params") or {}).get("protocolVersion") or PROTOCOL
            result = {"protocolVersion": wanted, "capabilities": {"tools": {}}, "serverInfo": {"name": "memory", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": ask(memory, {"op": "definitions", "saving": saving}).get("tools", [])}
        elif method == "tools/call":
            params = request.get("params") or {}
            reply = ask(memory, {"op": "tool", "name": params.get("name"), "arguments": params.get("arguments") or {}}, timeout=60)
            if "error" in reply:
                result = {"content": [{"type": "text", "text": reply["error"]}], "isError": True}
            else:
                result = {"content": [{"type": "text", "text": reply.get("text", "")}], "isError": bool(reply.get("failed"))}
        elif method == "ping":
            result = {}
        else:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"no method {method}"}}
    except Exception as error:  # the service being down is a tool error the agent can read, never a crash
        result = {"content": [{"type": "text", "text": f"memory unavailable: {type(error).__name__}"}], "isError": True}
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def main(argv=None, stdin=None, stdout=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", type=Path, default=Path("/cc/memory"))
    parser.add_argument("--saving", action="store_true", help="also offer memory_save (the fair test)")
    args = parser.parse_args(argv)
    out = stdout or sys.stdout
    for line in stdin or sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except ValueError:
            continue
        reply = respond(request, args.memory, args.saving)
        if reply is not None:
            out.write(json.dumps(reply) + "\n")
            out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
