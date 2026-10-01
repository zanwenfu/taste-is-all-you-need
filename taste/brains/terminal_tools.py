"""Model-visible terminal commands and pages of their retained output.

The injected client holds credentials; tool descriptions, arguments and
results contain only public scope and receipts. Commands affect the shared
task container. Memory rollback does not undo those external effects.

A result is plain text, as a shell would show it: the exit status, then what
the command printed. Long output is shown as its beginning and its end, with
the number of characters left out stated where they were, and the rest stays
readable by page. Nothing the worker was not shown is ever described as shown.
"""

from __future__ import annotations

import json
import posixpath

from taste.brains.responses_conversation import ResponsesTool, ToolOutcome
from taste.brains.terminal_broker import (
    TerminalConflict,
    TerminalFenced,
    TerminalRequest,
    _identifier,
)
from taste.brains.terminal_service import TerminalClient, TerminalUnavailable

PAGE_BYTES = 16384
DEFAULT_TIMEOUT_SECONDS = 120
# Characters of one command's output shown inline, across both streams.
SHOWN_CHARS = 14000
_STDERR_SHARE = 5000


def _text(data):
    return data.decode("utf-8", errors="replace")


def _shown(text, limit, stream, request_id):
    """The beginning and end of long output, saying how much is between them."""
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    return (text[:head]
            + f"\n... [cut: {len(text) - limit:,} more characters of {stream}; "
              f"read_terminal_output with request_id \"{request_id}\" pages through them]\n"
            + text[-tail:])


def render(request_id, timeout_seconds, result):
    """A command's receipt as the text its worker is given, and a record keeps."""
    code = result.return_code
    if result.terminated == "timeout":
        head = (f"timed out after {timeout_seconds:g}s: the command and its child "
                f"processes were killed (exit {code}). Output before that:")
    elif result.terminated == "cancelled":
        head = f"cancelled: the command and its child processes were killed (exit {code})."
    else:
        head = f"exit {code}"
    out, err = _text(result.stdout), _text(result.stderr)
    # Whichever stream is short is shown whole; the other takes the rest.
    if len(out) <= SHOWN_CHARS - _STDERR_SHARE:
        out_limit, err_limit = len(out), SHOWN_CHARS - len(out)
    elif len(err) <= _STDERR_SHARE:
        out_limit, err_limit = SHOWN_CHARS - len(err), len(err)
    else:
        out_limit, err_limit = SHOWN_CHARS - _STDERR_SHARE, _STDERR_SHARE
    parts = [head]
    if out:
        parts.append(_shown(out, out_limit, "stdout", request_id))
    if err:
        parts.append("[stderr]")
        parts.append(_shown(err, err_limit, "stderr", request_id))
    for stream, dropped in (("stdout", result.stdout_dropped_bytes), ("stderr", result.stderr_dropped_bytes)):
        if dropped:
            parts.append(f"[cut: {dropped:,} more bytes of {stream} were beyond what is kept "
                         "and cannot be read]")
    return "\n".join(part.rstrip("\n") for part in parts) + "\n"


class TerminalTools:
    def __init__(self, client: TerminalClient, *, workdir: str | None = None):
        if not isinstance(client, TerminalClient):
            raise TypeError("terminal tools require an authenticated TerminalClient")
        if workdir is not None and (not isinstance(workdir, str) or not posixpath.isabs(workdir)
                                    or "\x00" in workdir):
            raise ValueError("the terminal working directory must be an absolute path")
        self.client = client
        self.workdir = workdir or "/"

    @property
    def _limit(self):
        return self.client.credential.grant.max_timeout_seconds

    def instructions(self):
        limit = self._limit
        return (
            "Terminal tools run in the task's own container, which all workers share.\n"
            f"- terminal_exec runs one command with /bin/sh -c. cwd defaults to {self.workdir}; "
            "a relative cwd is resolved against it. Every call is a fresh shell: `cd` and "
            "shell variables do not carry over. Files, installed packages and running "
            "services do persist, across commands and across memory rollback.\n"
            "- Commands run one at a time across all workers.\n"
            f"- A command still running after timeout_seconds (default "
            f"{min(DEFAULT_TIMEOUT_SECONDS, limit):g}, at most {limit:g}) is killed together "
            "with its child processes, and you receive what it had printed. Its effects may "
            "be partial. Give builds and test suites enough time.\n"
            "- To leave a process running, detach it and redirect its output, for example "
            "`nohup server > /tmp/server.log 2>&1 &`. A background process that keeps the "
            "command's output open makes the call wait until the timeout.\n"
            "- Long output is shown as its beginning and end around a `[cut: N more "
            "characters ...]` marker. read_terminal_output pages through what was kept. "
            "Never describe output you were not shown as if you had seen it.\n"
            "- These tools do not run on the controller host and cannot see the memory "
            "workspace; the artifact tools cannot see the container.\n"
            # Measured on real trials: a build run in place left a binary, and
            # another 4,553 generated files, in a developer's repository, and
            # neither worker's claim said so.
            "- The container is someone's working environment. Change only what your "
            "assignment needs. Keep build output and scratch files out of the task's own "
            "directories when you can (/tmp is yours), and remove what you no longer need.\n"
            "- Before your final claim, look at what your work left behind there (in a git "
            "repository, `git status --short`). List in your evidence every file it created, "
            "changed or deleted that is still there, including files a build or a test "
            "generated. Leave none out for seeming unimportant.\n"
            "Public terminal scope: "
            + json.dumps(self.client.credential.grant.to_dict(), sort_keys=True)
        )

    def _request(self, effect_id, arguments):
        if (not isinstance(arguments, dict) or "command" not in arguments
                or set(arguments) - {"command", "cwd", "timeout_seconds"}):
            raise ValueError("terminal execution takes command, and optionally cwd and timeout_seconds")
        cwd, timeout = arguments.get("cwd"), arguments.get("timeout_seconds")
        if cwd is None:
            cwd = self.workdir
        elif not isinstance(cwd, str) or "\x00" in cwd or not cwd:
            raise ValueError("cwd must be a path in the task container")
        else:
            cwd = posixpath.normpath(posixpath.join(self.workdir, cwd))
        if timeout is None:
            timeout = min(DEFAULT_TIMEOUT_SECONDS, self._limit)
        request = TerminalRequest(effect_id, self.client.credential.grant.actor_id,
                                  arguments["command"], cwd, timeout)
        if request.timeout_seconds > self._limit:
            raise ValueError(f"terminal command timeout exceeds the admitted {self._limit:g} seconds")
        return request

    def _validate_execute(self, arguments):
        self._request("validation", arguments)

    @staticmethod
    def _validate_read(arguments):
        if (not isinstance(arguments, dict) or not {"request_id", "stream"} <= set(arguments)
                or set(arguments) - {"request_id", "stream", "offset", "limit"}):
            raise ValueError("terminal output takes request_id and stream, and optionally offset and limit")
        _identifier(arguments["request_id"])
        offset, limit = arguments.get("offset", 0), arguments.get("limit", PAGE_BYTES)
        if (arguments["stream"] not in ("stdout", "stderr")
                or type(offset) is not int or not 0 <= offset <= 1048576
                or type(limit) is not int or not 1 <= limit <= PAGE_BYTES):
            raise ValueError(f"terminal pages require a stream, a byte offset and a limit of 1-{PAGE_BYTES}")

    def _render(self, request, result):
        return render(request.request_id, request.timeout_seconds, result)

    async def execute(self, effect_id, call):
        request = self._request(effect_id, call.arguments)
        # Unconfirmed transport outcomes propagate. ResponsesConversation then
        # retains its pending tool intent and forbids automatic effect replay.
        result = await self.client.execute(request)
        return ToolOutcome(self._render(request, result),
                           is_error=result.return_code != 0 or bool(result.terminated))

    async def read(self, _effect_id, call):
        self._validate_read(call.arguments)
        identifier, stream = call.arguments["request_id"], call.arguments["stream"]
        try:
            result = await self.client.lookup(identifier)
        except (TerminalConflict, TerminalFenced, TerminalUnavailable):
            return ToolOutcome("No confirmed terminal output is available for this actor and request.", True)
        if result is None:
            return ToolOutcome("No terminal receipt exists for this actor and request.", True)
        data, dropped = getattr(result, stream), getattr(result, stream + "_dropped_bytes")
        offset = call.arguments.get("offset", 0)
        page = data[offset:offset + call.arguments.get("limit", PAGE_BYTES)]
        end = offset + len(page)
        head = f"{stream} bytes {offset:,}-{end:,} of {len(data):,} kept for request {identifier}"
        if end < len(data):
            head += f"; continue from offset {end}"
        elif dropped:
            head += f"; [cut: {dropped:,} more bytes were beyond what is kept and cannot be read]"
        else:
            head += "; end of output"
        return ToolOutcome(head + "\n" + _text(page))

    def tools(self):
        limit = self._limit
        return {
            "terminal_exec": ResponsesTool(
                "Run one command in the shared task container with /bin/sh -c and return its exit "
                "status and output. A command that outlives its timeout is killed. Effects on the "
                "container persist.",
                {"type": "object", "additionalProperties": False, "required": ["command"],
                 "properties": {
                     "command": {"type": "string"},
                     "cwd": {"type": "string",
                             "description": f"Working directory; defaults to {self.workdir}."},
                     "timeout_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": limit,
                                         "description": f"Defaults to {min(DEFAULT_TIMEOUT_SECONDS, limit):g}."}}},
                self._validate_execute, self.execute),
            "read_terminal_output": ResponsesTool(
                f"Read up to {PAGE_BYTES} bytes kept from one of this worker's finished commands, "
                "without running anything. Use it for the part of a long output that was cut.",
                {"type": "object", "additionalProperties": False, "required": ["request_id", "stream"],
                 "properties": {
                     "request_id": {"type": "string"},
                     "stream": {"type": "string", "enum": ["stdout", "stderr"]},
                     "offset": {"type": "integer", "minimum": 0, "maximum": 1048576},
                     "limit": {"type": "integer", "minimum": 1, "maximum": PAGE_BYTES}}},
                self._validate_read, self.read),
        }
