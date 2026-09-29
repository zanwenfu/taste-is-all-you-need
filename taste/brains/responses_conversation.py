"""A Responses worker's model/tool turns, projected into brain memory.

The private ResponsesSession owns spending; the branch owns active context.
Record a request intent before calling the model, and its full native reply
before executing tools. A reply lost only at the memory boundary is recovered
from the session receipt. A tool intent without a result is never retried.

This is the turn driver, not worker admission or terminal certification. Tool
handlers are trusted controller code: they validate arguments, enforce their
own filesystem/container boundary, and retain ownership through cancellation.
No task code, shell, or provider SDK runs in the memory worktree implicitly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from taste.brains.responses_session import (
    ResponsesConflict,
    ResponsesFenced,
    ResponsesSession,
    _completion_payload,
)
from taste.memstore import Branch
from taste.providers.base import Completion, ToolCall

_PREFIX = "responses_"
_SCHEMA = "taste.brains/ResponsesConversation/1"


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _copy(value):
    return json.loads(_json(value))


def _sha(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class ToolOutcome:
    content: str
    is_error: bool = False

    def __post_init__(self):
        if not isinstance(self.content, str) or len(self.content.encode()) > 65_536:
            raise ValueError("tool result must be text of at most 64 KiB")
        if type(self.is_error) is not bool:
            raise ValueError("tool result is_error must be boolean")


@dataclass(frozen=True)
class ResponsesTool:
    """A schema plus argument admission and an owned, bounded implementation.

    validate raises ValueError to deny a call without executing it. execute
    receives a stable controller effect ID as well as the provider's call ID.
    External effects need their own non-rollback ledger (e.g. TerminalBroker).
    """

    description: str
    input_schema: dict[str, Any]
    validate: Callable[[dict[str, Any]], None]
    execute: Callable[[str, ToolCall], Awaitable[ToolOutcome]]


@dataclass
class _Context:
    messages: list[dict[str, Any]]
    inputs: dict[str, str]
    request_id: str | None = None
    completion: Completion | None = None
    next_tool: int = 0
    tool_intent: str | None = None
    input_since_response: bool = False
    submitted_inputs: dict[str, str] = field(default_factory=dict)

    @property
    def pending(self):
        return self.request_id is not None and (
            self.completion is None or self.next_tool < len(self.completion.tool_calls)
        )


@dataclass(frozen=True)
class CompletedResponsesTurn:
    """A paid reply and the exact controller inputs submitted before it."""

    request_id: str
    completion: Completion
    submitted_inputs: Mapping[str, str]


class ResponsesConversation:
    """Advance one model reply and its serial tool calls per step.

    Concurrent step/observe calls are refused. This object borrows the branch
    and session leases; the caller closes them only after step has settled.
    Reopen with the same inputs after a clean checkpoint or process restart.
    """

    def __init__(self, branch: Branch, session: ResponsesSession, *, system: str,
                 tools: Mapping[str, ResponsesTool], effort: str = "low"):
        if not isinstance(system, str) or not system.strip():
            raise ValueError("a nonempty worker system prompt is required")
        if not isinstance(effort, str) or not effort:
            raise ValueError("worker reasoning effort is required")
        self.branch, self.session = branch, session
        if (not session.directory.resolve().is_relative_to(branch.store.backend.common_dir.resolve())
                or session.directory.resolve().is_relative_to(branch.worktree.resolve())):
            raise ResponsesConflict("Responses ledger must be in controller storage outside the worker worktree")
        self._tools = dict(tools)
        if len(self._tools) > 32:
            raise ValueError("at most 32 worker tools may be declared")
        for name, tool in self._tools.items():
            if not isinstance(name, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name) is None:
                raise ValueError("invalid worker tool name")
            if not isinstance(tool, ResponsesTool) or not callable(tool.validate) or not callable(tool.execute):
                raise ValueError("worker tools require validation and execution handlers")
        self._request_config = _copy({
            "system": system, "effort": effort,
            "tools": [{"name": name, "description": tool.description,
                       "input_schema": tool.input_schema} for name, tool in self._tools.items()],
        })
        self._binding = {
            "schema": _SCHEMA, "run_id": session.binding.run_id,
            "session": branch.store.session, "branch": branch.name,
            "model": session.binding.model, "endpoint": session.binding.endpoint,
            "deployment": session.binding.deployment, "request_config": self._request_config,
        }
        # Several controller objects can borrow Store's cached Branch. Keep a
        # single synchronous admission flag on that shared branch, not per view.
        if not hasattr(branch, "_responses_active"):
            branch._responses_active = False
        with branch._mutation_lock:
            if branch._responses_active:
                raise ResponsesConflict("a Responses turn already owns this branch")
            events = self._events()
            if not events:
                self._append("binding", binding=self._binding)
            self.session.reconcile_conversation_audit(self._events())
            self._context()

    def _events(self):
        _ = self.session.known_cost_usd  # check ownership before memory access
        # A checkpoint folds the pending journal into a new head. Read both
        # halves under the writer's reentrant lock, so a controller thread
        # cannot splice the old committed prefix with the new empty journal.
        with self.branch._mutation_lock:
            return [event for event in (*self.branch.head.transcript.turns, *self.branch.view.pending_turns())
                    if str(event.get("kind", "")).startswith(_PREFIX)]

    def _append(self, kind, **payload):
        event = {"kind": _PREFIX + kind, **_copy(payload)}
        identifier = self.session.record_conversation_event(self._events(), event)
        self.branch.turn(**event)
        self.session.publish_conversation_event(identifier)

    def _request(self, messages):
        return {**_copy(self._request_config), "messages": _copy(messages)}

    def _effect_id(self, request_id, call):
        return "effect_" + _sha([self.session.binding.run_id, request_id, call.id])

    def _context(self, *, stop_at_completion: str | None = None) -> _Context:
        events = self._events()
        if not events or events[0] != {"kind": "responses_binding", "binding": self._binding}:
            raise ResponsesConflict("worker context differs from the admitted Responses binding")
        context = _Context([], {})
        request_ids, call_ids = set(), set()
        for event in events[1:]:
            kind = event["kind"]
            if kind == "responses_input":
                identifier, content = event["id"], event["content"]
                if context.pending or identifier in context.inputs or not isinstance(content, str):
                    raise ResponsesConflict("worker input is duplicated or crossed a pending turn")
                context.inputs[identifier] = content
                context.messages.append({"role": "user", "content": content})
                context.input_since_response = True
            elif kind == "responses_request":
                identifier = event["id"]
                if context.pending or identifier in request_ids or event["request_sha"] != _sha(self._request(context.messages)):
                    raise ResponsesConflict("Responses intent does not match its exact context")
                request_ids.add(identifier)
                context.request_id, context.completion = identifier, None
                context.submitted_inputs = dict(context.inputs)
                context.next_tool, context.tool_intent = 0, None
                context.input_since_response = False
            elif kind == "responses_completion":
                if context.request_id != event["id"] or context.completion is not None:
                    raise ResponsesConflict("Responses reply has no unique pending intent")
                reply = self.session.lookup(event["id"])
                if reply is None or _completion_payload(reply) != event["completion"]:
                    raise ResponsesConflict("worker reply differs from the durable provider receipt")
                if len(reply.tool_calls) > 32 or any(call.id in call_ids for call in reply.tool_calls):
                    raise ResponsesConflict("worker reply has excessive or reused tool call IDs")
                call_ids.update(call.id for call in reply.tool_calls)
                context.completion = reply
                context.messages.append({"role": "assistant", "content": list(reply.transcript_blocks)})
                if stop_at_completion == context.request_id:
                    return context
            elif kind in {"responses_tool_intent", "responses_tool_result"}:
                reply = context.completion
                if reply is None or context.next_tool >= len(reply.tool_calls):
                    raise ResponsesConflict("worker tool event has no pending provider call")
                call = reply.tool_calls[context.next_tool]
                effect_id = self._effect_id(context.request_id, call)
                if event["effect_id"] != effect_id or event["call"] != _copy({"id": call.id, "name": call.name, "arguments": call.arguments}):
                    raise ResponsesConflict("worker tool event changed its admitted effect")
                if kind == "responses_tool_intent":
                    if context.tool_intent is not None:
                        raise ResponsesConflict("worker tool intent was duplicated")
                    context.tool_intent = effect_id
                else:
                    if context.tool_intent != effect_id:
                        raise ResponsesConflict("worker tool result has no durable intent")
                    result = ToolOutcome(**event["result"])
                    context.messages.append({"role": "user", "content": [{
                        "type": "tool_result", "tool_use_id": call.id,
                        "content": result.content, "is_error": result.is_error,
                    }]})
                    context.next_tool += 1
                    context.tool_intent = None
            else:
                raise ResponsesConflict("unknown Responses memory event")
        return context

    @property
    def messages(self):
        with self.branch._mutation_lock:
            return _copy(self._context().messages)

    def completed_turn(self, request_id: str | None = None) -> CompletedResponsesTurn:
        """Read a receipt for feedback acceptance, including after a restart.

        Inputs queued after this request are excluded. A historical receipt is
        reconstructed from its verified memory prefix and private provider
        receipt; callers cannot invent a submission by writing an ack marker.
        """
        with self.branch._mutation_lock:
            context = self._context(stop_at_completion=request_id)
            if (context.completion is None or context.request_id is None
                    or (request_id is not None and context.request_id != request_id)):
                raise ResponsesConflict("no completed Responses turn matches this request")
            return CompletedResponsesTurn(
                context.request_id, context.completion,
                MappingProxyType(dict(context.submitted_inputs)),
            )

    def observe(self, identifier: str, content: str):
        """Durably supply a contract or feedback; this alone accepts no inbox message."""
        if not isinstance(identifier, str) or not identifier or len(identifier) > 256:
            raise ValueError("worker input requires a stable identifier")
        if not isinstance(content, str) or len(content.encode()) > 65_536:
            raise ValueError("worker input must be text of at most 64 KiB")
        with self.branch._mutation_lock:
            if self.branch._responses_active:
                raise ResponsesConflict("cannot insert feedback during an active Responses turn")
            context = self._context()
            if identifier in context.inputs:
                if context.inputs[identifier] != content:
                    raise ResponsesConflict("worker input ID was reused with different content")
                return
            if context.pending:
                raise ResponsesFenced("settle the pending Responses turn before adding feedback")
            self._append("input", id=identifier, content=content)

    def _check_effect_admission(self):
        if self.session.fenced or self.session.unsettled or time.time() >= self.session.binding.deadline_unix:
            raise ResponsesFenced("worker effects are fenced or their deadline elapsed")
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError()

    async def step(self) -> Completion:
        with self.branch._mutation_lock:
            if self.branch._responses_active:
                raise ResponsesConflict("a Responses turn already owns this branch")
            self.branch._responses_active = True
        try:
            context = self._context()
            if context.tool_intent is not None:
                raise ResponsesFenced("a worker tool has an unknown outcome; automatic replay is forbidden")
            if (not context.pending and context.completion is not None
                    and context.completion.stop_reason != "tool_use" and not context.input_since_response):
                return context.completion
            if not context.pending:
                self._check_effect_admission()
                request_id = "response." + uuid.uuid4().hex
                self._append("request", id=request_id, request_sha=_sha(self._request(context.messages)))
                context = self._context()
            request_id = context.request_id
            if context.completion is None:
                reply = await self.session.complete(request_id, **self._request(context.messages))
                current = self._context()
                if current.request_id != request_id or current.completion is not None:
                    raise ResponsesConflict("worker memory changed while its model call was active")
                self._append("completion", id=request_id, completion=_completion_payload(reply))
                context = self._context()
            reply = context.completion
            while context.next_tool < len(reply.tool_calls):
                self._check_effect_admission()
                call = reply.tool_calls[context.next_tool]
                effect_id = self._effect_id(request_id, call)
                payload = {"effect_id": effect_id, "call": {"id": call.id, "name": call.name, "arguments": call.arguments}}
                tool = self._tools.get(call.name)
                denial = None
                if tool is None:
                    denial = "Tool is not declared in this worker's contract."
                else:
                    try:
                        tool.validate(_copy(call.arguments))
                    except ValueError as exc:
                        denial = str(exc)[:2000] or "Tool arguments were rejected."
                self._append("tool_intent", **payload)
                result = (ToolOutcome(denial, True) if denial is not None
                          else await tool.execute(effect_id, ToolCall(call.id, call.name, _copy(call.arguments), call.raw_arguments)))
                if not isinstance(result, ToolOutcome):
                    raise ResponsesConflict("worker tool returned no validated result")
                current = self._context()
                if current.request_id != request_id or current.tool_intent != effect_id:
                    raise ResponsesConflict("worker memory changed during its tool effect")
                self._append("tool_result", **payload,
                             result={"content": result.content, "is_error": result.is_error})
                context = self._context()
            return reply
        finally:
            self.branch._responses_active = False
