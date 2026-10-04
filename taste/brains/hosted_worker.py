"""A worker whose agent someone else wrote, run unchanged.

The agent's own loop runs in a thread of the worker process. Its two calls
out, a model request and a shell command, come back to this runtime on the
event loop. There they go through the worker's journaled model session and its
terminal client, exactly as Taste's own worker's calls do, and are recorded in
the worker's memory branch, where its monitor and the coordinator read them.
The agent is told nothing about Taste.

An agent that cannot take feedback is supervised by stopping it. Its monitor
judges what it does as for any worker. A judgement that it is doing the wrong
thing ("wrong") or that more of the same will not help ("lost") ends its run
at its next call, and the coordinator plans again with that on record. Its
exit, final message, submission and a mechanical record of its commands become
the assignment's one output, its report, which is certified like any other.

A hosted agent works only in the task's container. It cannot read or write
memory artifacts, take inbox messages or acknowledge verdicts, so none of
these can hold up its report.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import threading
import time
import uuid
from dataclasses import asdict

from taste.agents import HostedStop, ModelReply, ShellResult
from taste.brains.azure_worker_policy import AzureWorkerPolicy
from taste.brains.azure_worker_runtime import LOST_REPLIES, AzureWorkerRuntime
from taste.brains.records import Assignment
from taste.brains.responses_feedback import WorkerClaim
from taste.brains.responses_session import ResponsesFenced, ResponsesSession
from taste.brains.terminal_broker import TerminalRequest
from taste.brains.terminal_worker_policy import TerminalWorkerPolicy
from taste.brains.worker_commands import bounded, ran, unfinished
from taste.brains.worker_protocol import GOAL_TASK_PATH, ContractMismatch
from taste.llm import BudgetExceeded, InfraFailure
from taste.providers._openai import _NATIVE

PREFIX = "hosted_"
_SCHEMA = "taste.brains/HostedAgent/1"
# A judgement that the agent must be stopped, or that more will not help.
STOP_SEVERITIES = frozenset({"wrong", "lost"})
# What one command's output keeps in memory; the agent itself saw all of it.
OUTPUT_CHARS = 16_000
COMMAND_CHARS = 65_536
# The report is a worker output, which is bounded like any artifact write.
REPORT_CHARS = 60_000
SUMMARY_CHARS = 4_000


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _sha(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _excerpt(text, limit):
    if len(text) <= limit:
        return text
    head = limit * 2 // 5
    return text[:head] + f"\n... [cut: {len(text) - limit:,} more characters]\n" + text[-(limit - head):]


def _text(data):
    return data.decode("utf-8", errors="replace")


def hosted_task(assignment: Assignment, original: str | None) -> str:
    """What the agent is asked to do, as plain text for its own task template.

    An assignment that is the original task verbatim is given as that text
    alone, as the agent would be given it without Taste.
    """
    contract = assignment.contract
    if original is not None and contract.task.strip() == original.strip():
        return original
    parts = [contract.task.strip()]
    if contract.success_criteria:
        parts.append("Done when:\n" + "\n".join("- " + item for item in contract.success_criteria))
    if contract.notes.strip():
        parts.append(contract.notes.strip())
    if original:
        parts.append("For reference, the original task as its author wrote it. Your work is "
                     "what is asked above.\n\n" + original)
    return "\n\n".join(parts)


def _reply(completion, cost):
    """A completion as the provider gave it: its output items, in order."""
    items = tuple(block["item"] for block in completion.transcript_blocks if block.get("type") == _NATIVE)
    status, reason = {"max_tokens": ("incomplete", "max_output_tokens"),
                      "content_filter": ("incomplete", "content_filter")}.get(
                          completion.stop_reason, ("completed", None))
    return ModelReply(output=items, status=status, incomplete_reason=reason,
                      usage=asdict(completion.usage), model=completion.model, cost_usd=cost)


class _Host:
    """The agent thread's way out; every call runs on the worker's event loop."""

    def __init__(self, runtime, loop):
        self.runtime, self.loop = runtime, loop
        self._lock = threading.Lock()
        self._stopped = None
        self._active = None

    @property
    def stopped(self):
        return self._stopped

    def stop(self, reason, *, now=True):
        """End the agent's run: at once, or (``now=False``) at its next call."""
        with self._lock:
            if self._stopped is None:
                self._stopped = reason
            active = self._active
        if now and active is not None:
            active.cancel()

    def _call(self, factory):
        with self._lock:
            if self._stopped is not None:
                raise HostedStop(self._stopped)
            future = asyncio.run_coroutine_threadsafe(factory(), self.loop)
            self._active = future
        try:
            return future.result()
        except concurrent.futures.CancelledError:
            raise HostedStop(self._stopped or "cancelled") from None
        finally:
            with self._lock:
                self._active = None

    def ask(self, *, messages, tools, effort):
        return self._call(lambda: self.runtime._ask(messages, tools, effort))

    def run(self, command, *, cwd, timeout_seconds, shown=None):
        return self._call(lambda: self.runtime._run(command, cwd, timeout_seconds, shown))


class HostedWorkerRuntime(AzureWorkerRuntime):
    HARNESS = "hosted/1"
    ACKNOWLEDGES_FEEDBACK = False

    def __init__(self, branch, assignment: Assignment, session: ResponsesSession, monitor, *,
                 terminal_client, agent, model_name: str):
        # Not AzureWorkerRuntime.__init__: there is no Responses conversation
        # and no feedback channel to build, only the same admission checks.
        self.branch, self.assignment = branch, assignment
        self.session, self.monitor = session, monitor
        self.prepared = branch.head
        if (monitor.store is not branch.store or monitor.contract != assignment.contract
                or monitor.run_id != session.binding.run_id):
            raise ContractMismatch("hosted monitor differs from the admitted worker run")
        terminal_policy = TerminalWorkerPolicy.from_assignment(assignment)
        if terminal_policy is None:
            raise ContractMismatch("a hosted agent works in the task's terminal, which this assignment lacks")
        if terminal_client is None or terminal_client.credential.grant != terminal_policy.grant(assignment):
            raise ContractMismatch("hosted terminal client differs from the admitted assignment")
        present = [spec for spec in assignment.outputs if spec.disposition == "present"]
        if len(present) != 1 or len(assignment.outputs) != 1:
            raise ContractMismatch("a hosted agent's assignment names exactly one output, its report")
        self.report_spec = present[0]
        self.terminal_client, self.workdir = terminal_client, terminal_policy.workdir or "/"
        self.agent, self.model_name = agent, model_name
        # An agent run alone is neither judged nor certified: it is the baseline.
        self.CERTIFIES = AzureWorkerPolicy.from_assignment(assignment).supervised
        self.host = None
        self._commands = 0
        self.task = hosted_task(assignment, self.prepared.read(GOAL_TASK_PATH))
        if not self._events():
            self._append("binding", binding={
                "schema": _SCHEMA, "run_id": session.binding.run_id, "session": branch.store.session,
                "branch": branch.name, "agent": agent.identity(), "model": session.binding.model,
                "endpoint": session.binding.endpoint, "deployment": session.binding.deployment,
                "workdir": self.workdir})
            self._append("task", content=self.task)
        self.session.reconcile_conversation_audit(self._events())

    # -- memory record ----------------------------------------------------

    def _events(self):
        with self.branch._mutation_lock:
            return [event for event in (*self.branch.head.transcript.turns, *self.branch.view.pending_turns())
                    if str(event.get("kind", "")).startswith(PREFIX)]

    def _append(self, kind, **payload):
        event = {"kind": PREFIX + kind, **json.loads(_json(payload))}
        identifier = self.session.record_conversation_event(self._events(), event)
        self.branch.turn(**event)
        self.session.publish_conversation_event(identifier)

    # -- the agent's two calls, on the event loop ---------------------------

    def _admitted(self):
        if time.time() >= self.session.binding.deadline_unix:
            raise HostedStop("deadline")

    async def _ask(self, messages, tools, effort):
        lost = 0
        while True:
            self._admitted()
            request_id = "hosted." + uuid.uuid4().hex
            self._append("request", id=request_id,
                         request_sha=_sha({"messages": messages, "tools": tools, "effort": effort}))
            try:
                completion = await self.session.complete(request_id, system="", messages=messages,
                                                         tools=tools, effort=effort)
            except InfraFailure as failure:
                # Asked again as a new call, as Taste's own worker does; the
                # journal charges the lost one its worst case.
                lost += 1
                if not failure.transient or lost > LOST_REPLIES:
                    raise HostedStop("model_unavailable") from None
                self.session.forfeit(request_id)
                self._append("lost", id=request_id)
                continue
            except BudgetExceeded:
                raise HostedStop("budget") from None
            except ResponsesFenced:
                raise HostedStop("model_session_closed") from None
            except ValueError:
                raise HostedStop("request_too_large") from None
            cost = self.session._receipt_cost(completion)
            self._append("completion", id=request_id, text=list(completion.text_blocks),
                         calls=[{"id": call.id, "name": call.name, "arguments": call.arguments}
                                for call in completion.tool_calls],
                         stop_reason=completion.stop_reason, model=completion.model, cost_usd=cost)
            return _reply(completion, cost)

    async def _run(self, command, cwd, timeout_seconds, shown=None):
        self._admitted()
        grant = self.terminal_client.credential.grant
        timeout = min(float(timeout_seconds), float(grant.max_timeout_seconds))
        self._commands += 1
        effect_id = "effect_" + _sha([self.session.binding.run_id, "command", self._commands])
        # The record shows the command as the agent wrote it; the ledger keeps
        # the exact text the container ran.
        self._append("command", effect_id=effect_id, command=_excerpt(shown or command, COMMAND_CHARS),
                     cwd=cwd, timeout_seconds=timeout)
        try:
            result = await self.terminal_client.execute(
                TerminalRequest(effect_id, grant.actor_id, command, cwd, timeout))
        except (asyncio.CancelledError, HostedStop):
            raise
        except Exception:
            raise HostedStop("terminal_unavailable") from None
        output = _text(result.stdout) + _text(result.stderr)
        self._append("output", effect_id=effect_id, returncode=result.return_code,
                     terminated=result.terminated or "", output=_excerpt(output, OUTPUT_CHARS),
                     output_chars=len(output),
                     dropped_bytes=result.stdout_dropped_bytes + result.stderr_dropped_bytes)
        judged = await self._monitored(lambda: self.monitor.drain(None)) if self.CERTIFIES else ()
        stops = [judgement.severity.value for judgement, _ in judged or ()
                 if judgement.severity.value in STOP_SEVERITIES]
        if stops and self.host is not None:
            # The agent still sees this command's result; its next call ends it.
            self.host.stop("monitor_" + stops[0], now=False)
        return ShellResult(output, result.return_code, timed_out=result.terminated == "timeout")

    # -- the run --------------------------------------------------------------

    def _pending_inbox(self):
        try:
            return [item.inbox_id for item in self._communicator().pending(self.assignment.worker)]
        except Exception:
            return []

    def _communicator(self):
        from taste.brains.communication import Communicator

        return Communicator(self.branch.store)

    def _command_lines(self):
        started, lines = {}, []
        for event in self._events():
            if event["kind"] == "hosted_command":
                started[event["effect_id"]] = event["command"]
            elif event["kind"] == "hosted_output" and event["effect_id"] in started:
                lines.append(ran(started.pop(event["effect_id"]),
                                 f"exit {event['returncode']}\n{event['output']}"))
        lines.extend(unfinished(command) for command in started.values())
        return lines

    def _activity(self, *, last=40, limit=12_000):
        lines = bounded(self._command_lines(), last=last, limit=limit)
        return ["Mechanical record of this agent's commands, oldest first:", *lines] if lines else []

    @staticmethod
    def _final_text(messages):
        for message in reversed(list(messages or ())):
            if message.get("object") == "response":
                texts = [part.get("text", "") for item in message.get("output", ())
                         if item.get("type") == "message" for part in item.get("content", ())
                         if part.get("type") == "output_text"]
                if any(texts):
                    return "\n".join(text for text in texts if text)
        return ""

    def _report_text(self, exit_status, stopped_by, final_text, submission):
        identity = self.agent.identity()
        lines = [f"# Report of {identity['name']} {identity['version']}", "",
                 f"Exit: {exit_status}" + (f" (stopped: {stopped_by})" if stopped_by else ""), "",
                 "## Its final message", "", final_text or "(none)", "",
                 "## Its submission", "", submission or "(empty)", "",
                 "## Commands it ran (mechanical record, oldest first)", ""]
        lines.extend(bounded(self._command_lines(), last=60, limit=REPORT_CHARS // 2) or ["(none)"])
        return _excerpt("\n".join(lines) + "\n", REPORT_CHARS)

    async def _drive(self) -> WorkerClaim:
        loop = asyncio.get_running_loop()
        self.host = _Host(self, loop)
        done = loop.create_future()

        def settle(outcome):
            if not done.done():
                done.set_result(outcome)

        def target():
            try:
                outcome = (self.agent.run(self.task, self.host, cwd=self.workdir, model_name=self.model_name), None)
            except BaseException as error:  # the agent's own loop ended by an exception
                outcome = (None, error)
            loop.call_soon_threadsafe(settle, outcome)

        thread = threading.Thread(target=target, name="taste-hosted-agent", daemon=True)
        thread.start()
        cancelled = None
        while not done.done():
            try:
                await asyncio.wait((done,))
            except asyncio.CancelledError as exc:
                # Stop at the agent's next call (its in-flight one is cancelled
                # and settles first), then report it as interrupted.
                cancelled = exc
                self.host.stop("interrupted")
        result, error = done.result()
        if cancelled is not None:
            raise cancelled
        if error is not None and not isinstance(error, HostedStop):
            raise error
        stopped_by = (self.host.stopped or str(error) or "stopped") if isinstance(error, HostedStop) else None
        exit_status = result.exit_status if result is not None else "Stopped"
        submission = result.submission if result is not None else ""
        final_text = self._final_text(result.messages if result is not None else ())
        self._append("exit", exit_status=exit_status, stopped_by=stopped_by or "",
                     submission=_excerpt(submission, SUMMARY_CHARS))
        path = self.branch.path(self.report_spec.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._report_text(exit_status, stopped_by, final_text, submission), encoding="utf-8")
        status = "completed" if exit_status == "Submitted" and not stopped_by else "blocked"
        summary = _excerpt(final_text or submission or f"{exit_status}"
                           + (f" (stopped: {stopped_by})" if stopped_by else ""), SUMMARY_CHARS)
        return WorkerClaim(status, summary, tuple(self._activity()), (), {})

    def _extra_metadata(self):
        return {"agent": self.agent.identity(), "supervised": self.CERTIFIES}
