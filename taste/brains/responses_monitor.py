"""Azure monitor judgements backed by a private model-call journal.

MonitorBrain invokes its synchronous judge in an owned thread. Each operation
opens the Responses journal and runs its event loop wholly in that thread;
SQLite connections and asyncio leases never cross threads or event loops.
The journal persists beyond the operation, the process and memory rollback.

Create this journal once at controller admission, outside all task-writable
paths. Reopen it explicitly on recovery. Missing/corrupt journals are errors,
never a reason to silently start a fresh spending allowance.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from taste.brains.monitor_judge import (
    _SYSTEM_PROMPT,
    MAX_MONITOR_PROMPT_BYTES,
    MAX_TRANSCRIPT_VIEW_BYTES,
    TRANSCRIPT_VIEW_BYTES,
    LLMMonitorJudge,
    PinnedMonitorObservation,
)
from taste.brains.responses_session import ResponsesBinding, ResponsesConflict, ResponsesSession
from taste.brains.worker_protocol import ModelCallAccounting
from taste.providers.azure_openai import AzureOpenAIConfig


class _MonitorCalls:
    def __init__(self, directory: Path, binding: ResponsesBinding, azure: AzureOpenAIConfig):
        self.directory = Path(directory).absolute()
        self.binding, self.azure = binding, azure

    def call(self, *, model, system, messages, tools, max_tokens, temperature, role):
        if (model != self.binding.model or max_tokens != self.binding.max_output_tokens
                or role != "monitor" or tools is not None or temperature != 0.0
                or len(messages) != 1 or messages[0].get("role") != "user"):
            raise ResponsesConflict("monitor request differs from its admitted configuration")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise ResponsesConflict("use MonitorBrain's async API or an owned judge thread")

        request = {"system": system, "messages": messages, "tools": None, "effort": "low"}
        canonical = json.dumps(request, sort_keys=True, ensure_ascii=True,
                               allow_nan=False, separators=(",", ":"))
        request_id = "monitor." + hashlib.sha256(canonical.encode()).hexdigest()
        session = ResponsesSession.open(self.directory, self.binding, self.azure)
        try:
            # One exact pinned observation/prompt has one paid reply. A crash
            # before the monitor sidecar commits reuses that reply; a pending
            # provider outcome refuses dispatch. No SDK retry can spend twice.
            return asyncio.run(session.complete(request_id, **request))
        finally:
            session.close()

    def call_accounting(self) -> ModelCallAccounting:
        session = ResponsesSession.open(self.directory, self.binding, self.azure)
        try:
            return session.call_accounting()
        finally:
            session.close()


class ResponsesMonitorJudge(LLMMonitorJudge):
    """Strict incremental and terminal judges using Azure Responses only.

    The outer trial owner must bound the whole monitor thread/process. A
    request deadline is supplied to the provider by ResponsesSession, and a
    cancelled MonitorBrain operation still owns its thread through settlement.
    """

    @classmethod
    def create(cls, directory: Path, binding: ResponsesBinding, azure: AzureOpenAIConfig):
        return cls(directory, binding, azure, fresh=True)

    @classmethod
    def open(cls, directory: Path, binding: ResponsesBinding, azure: AzureOpenAIConfig):
        return cls(directory, binding, azure, fresh=False)

    def __init__(self, directory, binding, azure, *, fresh):
        if binding.role != "monitor":
            raise ResponsesConflict("monitor journal requires an explicit monitor role")
        operation = ResponsesSession.create if fresh else ResponsesSession.open
        session = operation(directory, binding, azure)
        session.close()
        super().__init__(
            _MonitorCalls(directory, binding, azure), model=binding.model,
            max_tokens=binding.max_output_tokens, json_prefill=False,
        )
        # A certifier held to a 64 KiB view of a long run saw every result
        # shortened and refused correct work three times over for "an evidence
        # gap" (measured: a four-minute fix became four workers and $18). The
        # view grows with what this journal admits; rendering and the request's
        # own JSON escaping can each enlarge it, so it takes under half.
        room = binding.max_request_bytes - MAX_MONITOR_PROMPT_BYTES
        self.transcript_view_bytes = max(TRANSCRIPT_VIEW_BYTES, min(MAX_TRANSCRIPT_VIEW_BYTES, room // 2))
        self.max_prompt_bytes = max(MAX_MONITOR_PROMPT_BYTES, binding.max_request_bytes * 3 // 4)

    def call_accounting(self) -> ModelCallAccounting:
        return self.llm.call_accounting()

    def ensure_ready(self) -> None:
        calls = self.llm
        session = ResponsesSession.open(calls.directory, calls.binding, calls.azure)
        try:
            session.ensure_ready()
        finally:
            session.close()

    @staticmethod
    def _observation_id(observation):
        evidence = json.loads(observation.payload)
        # Elapsed time belongs to the original observation. Advancing a wall
        # clock after a crash must not make unchanged evidence a new paid call.
        evidence.pop("elapsed")
        normalized = replace(observation, payload=json.dumps(evidence, sort_keys=True, allow_nan=False))
        return "monitor-context." + hashlib.sha256(
            (_SYSTEM_PROMPT + normalized.prompt()).encode()
        ).hexdigest()

    def _observation(self, contract, batch, view):
        observation = super()._observation(contract, batch, view)
        identity = self._observation_id(observation)
        calls = self.llm
        session = ResponsesSession.open(calls.directory, calls.binding, calls.azure)
        try:
            pinned = PinnedMonitorObservation(**session.pin_context(identity, asdict(observation)))
            if self._observation_id(pinned) != identity:
                raise ResponsesConflict("pinned monitor observation differs from its exact evidence")
            return pinned
        finally:
            session.close()
