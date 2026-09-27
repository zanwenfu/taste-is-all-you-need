"""Model-visible terminal commands and bounded pages of durable output.

The injected client holds credentials; tool descriptions, arguments and
results contain only public scope and receipts. Commands affect the shared
task container. Memory rollback does not undo those external effects.
"""

from __future__ import annotations

import base64
import json

from taste.brains.responses_conversation import ResponsesTool, ToolOutcome
from taste.brains.terminal_broker import (
    TerminalConflict,
    TerminalFenced,
    TerminalRequest,
    _identifier,
)
from taste.brains.terminal_service import TerminalClient, TerminalUnavailable

PAGE_BYTES = 4096


class TerminalTools:
    def __init__(self, client: TerminalClient):
        if not isinstance(client, TerminalClient):
            raise TypeError("terminal tools require an authenticated TerminalClient")
        self.client = client

    def instructions(self):
        return (
            "The terminal tools execute in one shared task container, using a fresh /bin/sh -c "
            "for each command and the explicit cwd you supply. Terminal actions are serial. "
            "Files, installed packages and services persist across successful commands and memory "
            "rollback; shell variables and cwd do not carry over automatically. These tools do "
            "not execute on the controller host or in the artifact memory workspace. "
            "Timeout or cancellation of an active command ends the task environment. "
            "Outputs are bounded binary-preserving prefixes. Read retained output pages with "
            "read_terminal_output; dropped bytes cannot be recovered from the receipt. "
            "Do not claim that truncated output is complete. Public terminal scope: "
            + json.dumps(self.client.credential.grant.to_dict(), sort_keys=True)
        )

    def _validate_execute(self, arguments):
        if not isinstance(arguments, dict) or set(arguments) != {"command", "cwd", "timeout_seconds"}:
            raise ValueError("terminal execution requires command, cwd and timeout_seconds")
        request = TerminalRequest("validation", self.client.credential.grant.actor_id, **arguments)
        if request.timeout_seconds > self.client.credential.grant.max_timeout_seconds:
            raise ValueError("terminal command timeout exceeds the admitted limit")

    @staticmethod
    def _validate_read(arguments):
        if not isinstance(arguments, dict) or set(arguments) != {"request_id", "stream", "offset", "limit"}:
            raise ValueError("terminal output requires request_id, stream, offset and limit")
        _identifier(arguments["request_id"])
        if (arguments["stream"] not in ("stdout", "stderr")
                or type(arguments["offset"]) is not int or not 0 <= arguments["offset"] <= 1048576
                or type(arguments["limit"]) is not int or not 1 <= arguments["limit"] <= PAGE_BYTES):
            raise ValueError("terminal pages require a valid stream, byte offset and limit of 1-4096")

    @staticmethod
    def _page(data, dropped, offset=0, limit=PAGE_BYTES):
        page = data[offset:offset + limit]
        try:
            content, encoding = page.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            content, encoding = base64.b64encode(page).decode("ascii"), "base64"
        return {"content": content, "encoding": encoding, "offset": offset,
                "next_offset": min(offset + len(page), len(data)), "captured_eof": offset + len(page) >= len(data),
                "captured_bytes": len(data), "dropped_bytes": dropped}

    async def execute(self, effect_id, call):
        self._validate_execute(call.arguments)
        request = TerminalRequest(effect_id, self.client.credential.grant.actor_id, **call.arguments)
        # Unconfirmed transport outcomes propagate. ResponsesConversation then
        # retains its pending tool intent and forbids automatic effect replay.
        result = await self.client.execute(request)
        return ToolOutcome(json.dumps({"request_id": effect_id, "return_code": result.return_code,
            "stdout": self._page(result.stdout, result.stdout_dropped_bytes),
            "stderr": self._page(result.stderr, result.stderr_dropped_bytes)}, ensure_ascii=False),
            is_error=result.return_code != 0)

    async def read(self, _effect_id, call):
        self._validate_read(call.arguments)
        try:
            result = await self.client.lookup(call.arguments["request_id"])
        except (TerminalConflict, TerminalFenced, TerminalUnavailable):
            return ToolOutcome("No confirmed terminal output is available for this actor and request.", True)
        if result is None:
            return ToolOutcome("No terminal receipt exists for this actor and request.", True)
        stream = call.arguments["stream"]
        return ToolOutcome(json.dumps({"request_id": call.arguments["request_id"], "stream": stream,
            "return_code": result.return_code,
            **self._page(getattr(result, stream), getattr(result, stream + "_dropped_bytes"),
                         call.arguments["offset"], call.arguments["limit"])}, ensure_ascii=False))

    def tools(self):
        def schema(properties):
            return {"type": "object", "properties": properties, "required": list(properties),
                    "additionalProperties": False}

        return {
            "terminal_exec": ResponsesTool(
                "Execute one bounded command in the shared task container. Returns a durable receipt and output prefixes; external effects persist across memory rollback.",
                schema({"command": {"type": "string"}, "cwd": {"type": "string"},
                        "timeout_seconds": {"type": "number", "exclusiveMinimum": 0,
                                            "maximum": self.client.credential.grant.max_timeout_seconds}}),
                self._validate_execute, self.execute),
            "read_terminal_output": ResponsesTool(
                "Read up to 4096 retained bytes from this actor's completed terminal receipt without executing another command. Dropped bytes are not recoverable from the receipt.",
                schema({"request_id": {"type": "string"}, "stream": {"type": "string", "enum": ["stdout", "stderr"]},
                        "offset": {"type": "integer", "minimum": 0, "maximum": 1048576},
                        "limit": {"type": "integer", "minimum": 1, "maximum": PAGE_BYTES}}),
                self._validate_read, self.read),
        }
