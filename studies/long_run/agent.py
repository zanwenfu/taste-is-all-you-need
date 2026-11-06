"""One coding agent's long run with compaction on, a message planted mid-run, and Taste's memory in the memory arms.

The agent is mini-swe-agent 2.4.6's ``DefaultAgent`` with three hooks, all in ``query`` and ``add_messages``:

* **Planted message.** At a chosen model call, one user message is added before the call, once. The harness never
  sends it again; after a compaction it survives only if the summary kept it or the agent wrote it down.
* **Compaction.** When the conversation passes a stated number of tokens, everything but the system prompt and
  the newest turns that fit a stated budget is replaced by one summary. A kept part never starts with a tool
  result whose call was summarized. A later compaction gets the previous summary apart from the turns to fold in.
  Each compaction is logged: sizes, the summary, and whether each watched text is in the summary or the kept turns.
* **Memory.** In the memory arms every message is committed to Taste's memory as it is added, so compaction never
  touches it, and the model is offered search and fetch beside bash, answered on the host; nothing runs in the
  task's container for them. The memory's own traffic is left out of it: a store that holds its searches and their results finds
  them again, and by the third search they fill most of what it returns.

The agent is no longer "run unchanged": the paper says so. Which name applies (brief, section 2): the compaction-only
arm is a baseline; an arm whose memory is ``taste.memstore`` is "Taste memory, study-driven".
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any

from jinja2 import StrictUndefined, Template
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import FormatError, InterruptAgentFlow
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
)
from pydantic import Field

from taste.memstore.tools import FETCH, SEARCH, definitions

SUMMARY_LEAD = "The conversation so far was compacted. Summary of the earlier part:\n\n"
OUTPUT_KEPT = 200_000  # characters of a command's output kept in the action log, and so scored; a memory call's output is kept whole
_SENT_KEYS = ("role", "content", "tool_calls", "tool_call_id")


def render(message: dict) -> str:
    """A message as text: what it says, and any tool calls it made."""
    text = str(message.get("content") or "")
    for call in message.get("tool_calls") or []:
        text += f"\n[{call['function']['name']}] {call['function']['arguments']}"
    return text.strip()


class ToolModel:
    """mini-swe-agent's model interface over a chat function: bash, and in the memory arms the two memory tools."""

    def __init__(self, chat: Callable[[list[dict], list | None], Any], *, memory_tools: bool,
                 observation_template: str = "{{output.output}}", format_error_template: str = "{{error}}") -> None:
        self.chat, self.memory_tools = chat, memory_tools
        self.last_text = self.last_finish = self.last_error = ""
        self.observation_template, self.format_error_template = observation_template, format_error_template
        self.tools = [BASH_TOOL, *(definitions("chat") if memory_tools else [])]

    def _refuse(self, error: str, *, has_tool_calls: bool = True) -> FormatError:
        """As mini-swe-agent does: its published template tells the model how to call a tool and how to submit."""
        self.last_error = error
        content = Template(self.format_error_template, undefined=StrictUndefined).render(
            error=error, actions=[], has_tool_calls=has_tool_calls, finish_reason=self.last_finish)
        return FormatError({"role": "user", "content": content, "extra": {"interrupt_type": "FormatError"}})

    def _actions(self, calls: list) -> list[dict]:
        if not calls:
            raise self._refuse("No tool calls found in the response. Every response MUST include at least one tool call.",
                               has_tool_calls=False)
        allowed = {"bash", SEARCH, FETCH} if self.memory_tools else {"bash"}
        actions = []
        for call in calls:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as error:
                raise self._refuse(f"Error parsing tool call arguments: {error}.") from None
            if name not in allowed:
                raise self._refuse(f"Unknown tool '{name}'.")
            if not isinstance(arguments, dict):
                raise self._refuse("Tool call arguments must be one JSON object.")
            if name == "bash":
                if "command" not in arguments:
                    raise self._refuse("Missing 'command' argument in bash tool call.")
                actions.append({"command": arguments["command"], "tool_call_id": call.id})
            else:
                actions.append({"memory_tool": name, "arguments": arguments, "tool_call_id": call.id,
                                "command": f"[{name}] {json.dumps(arguments, ensure_ascii=False)}"})
        return actions

    def query(self, messages: list[dict], **kwargs: Any) -> dict:
        sent = [{key: message[key] for key in _SENT_KEYS if key in message} for message in messages]
        reply = self.chat(sent, self.tools)
        message = reply.choices[0].message
        self.last_text, self.last_finish = message.content or "", reply.choices[0].finish_reason or ""
        calls = message.tool_calls or []
        actions = self._actions(calls)
        return {"role": "assistant", "content": message.content or "",
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls],
                "extra": {"actions": actions, "cost": 0.0, "timestamp": time.time(),
                          "usage": [reply.usage.prompt_tokens, reply.usage.completion_tokens],
                          "finish_reason": reply.choices[0].finish_reason}}

    def format_message(self, **kwargs: Any) -> dict:
        return dict(**kwargs)

    def format_observation_messages(self, message: dict, outputs: list[dict], template_vars: dict | None = None) -> list[dict]:
        return format_toolcall_observation_messages(actions=message["extra"]["actions"], outputs=outputs,
                                                    observation_template=self.observation_template,
                                                    template_vars=template_vars)

    def get_template_vars(self, **kwargs: Any) -> dict:
        return {}

    def serialize(self) -> dict:
        return {"info": {"config": {"model": {"memory_tools": self.memory_tools,
                                              "tools": [t["function"]["name"] for t in self.tools]}}}}


class MemoryEnvironment:
    """The task's environment, with the memory tools answered on the host instead of in the container."""

    def __init__(self, inner: Any, answer: Callable[[str, dict], tuple[str, bool]] | None) -> None:
        self.inner, self.answer = inner, answer

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        if "memory_tool" in action:
            if self.answer is None:
                return {"output": "No memory in this run.", "returncode": 1, "exception_info": ""}
            text, failed = self.answer(action["memory_tool"], action["arguments"])
            return {"output": text, "returncode": int(failed), "exception_info": ""}
        return self.inner.execute(action, cwd, timeout=timeout)

    def get_template_vars(self, **kwargs: Any) -> dict:
        return self.inner.get_template_vars(**kwargs)

    def serialize(self) -> dict:
        return self.inner.serialize()


class LongRunConfig(AgentConfig):
    compact_at_tokens: int = 0  # 0: never compact
    keep_recent_tokens: int = 0
    planted: dict[int, str] = Field(default_factory=dict)  # model call -> one user message added just before it
    watch: dict[str, str] = Field(default_factory=dict)  # texts whose survival each compaction reports
    mentions: dict[str, str] = Field(default_factory=dict)  # patterns for the same, since a summary puts a rule in its own words
    task_calls: int = 0  # 0: a task may take any number of model calls; past it the next task follows
    stop_at_tokens: int = 0  # 0: never; the conversation ends here rather than overflow the model's window


class LongRunAgent(DefaultAgent):
    def __init__(self, model: Any, env: Any, *, count: Callable[[str], int],
                 summarize: Callable[[list[dict], str | None], str] | None = None,
                 record: Callable[[dict], None] | None = None,
                 on_action: Callable[[dict], None] | None = None,
                 standing: Callable[[], str] | None = None,
                 on_compact: Callable[[list[dict]], None] | None = None,
                 on_prompt: Callable[[dict], None] | None = None, **kwargs: Any) -> None:
        super().__init__(model, env, config_class=LongRunConfig, **kwargs)
        self.count, self.summarize, self.record, self.on_action = count, summarize, record, on_action
        self.standing, self.on_compact, self.on_prompt = standing, on_compact, on_prompt
        self.resent: dict[str, int] = {}
        self._told: set[int] = set()
        self._system_base: str | None = None
        self.task_name = ""
        self.events: list[dict] = []
        self.recorded = 0
        self.previous_summary: str | None = None
        self._summary_message: dict | None = None
        self._planted_sent: set[int] = set()
        self._memory_calls: set[str] = set()
        self.outcomes: list[dict] = []

    def add_messages(self, *messages: dict) -> list[dict]:
        added = super().add_messages(*messages)
        if self.record is not None:
            for message in messages:
                # The system prompt is in front of the model at every call, so a store has no need of it.
                if message.get("role") not in ("exit", "system"):
                    said = self._talk(message)
                    if said:
                        self.record({"role": message["role"], "content": said,
                                     "from_harness": "interrupt_type" in message.get("extra", {})})
                        self.recorded += 1
        return added

    def _talk(self, message: dict) -> str:
        """What a message adds to the memory: all of it but the memory's own calls and their results."""
        own = {action["tool_call_id"] for action in message.get("extra", {}).get("actions", []) if "memory_tool" in action}
        self._memory_calls |= own
        if message.get("role") == "tool" and message.get("tool_call_id") in self._memory_calls:
            return ""
        return render({**message, "tool_calls": [call for call in message.get("tool_calls") or [] if call["id"] not in own]})

    def tokens(self, *, delivered: bool = True) -> int:
        """What the conversation holds. ``delivered=False`` leaves out a memory's block in the system prompt, so that every arm
        compacts at the same point of the same talk: a block is put back whole at every call, and folding turns cannot shrink it."""
        total = sum(self.count(render(message)) for message in self.messages)
        if not delivered and self._system_base is not None:
            total += self.count(self._system_base) - self.count(render(self.messages[0]))
        return total

    def _check_resent(self) -> None:
        """What the protocol promises: past the messages that first tell a rule, the harness never sends it. A model's own summary and
        command output are not the harness's. Counted per model call, read from what is about to be sent."""
        for message in self.messages[1:]:
            if message.get("role") != "user" or message is self._summary_message or id(message) in self._told:
                continue
            text = str(message.get("content") or "")
            for name, watched in self.config.watch.items():
                if watched in text:
                    self.resent[name] = self.resent.get(name, 0) + 1

    def query(self) -> dict:
        call = self.n_calls + 1
        if call in self.config.planted and call not in self._planted_sent:
            self._planted_sent.add(call)
            self.add_messages({"role": "user", "content": self.config.planted[call]})
            self._told.add(id(self.messages[-1]))
        folded = len(self.events)
        if self.summarize is not None and 0 < self.config.compact_at_tokens < self.tokens(delivered=False):
            self._compact(call)
        if self.standing is not None:
            if self._system_base is None:
                self._system_base = self.messages[0]["content"]
            block = self.standing()
            self.messages[0] = {**self.messages[0], "content": f"{self._system_base}\n\n{block}" if block else self._system_base}
        self._check_resent()
        if self.on_prompt is not None and len(self.events) > folded:
            self.on_prompt({"call": call, "messages": [{"role": m["role"], "content": render(m)} for m in self.messages]})
        return super().query()

    def _compact(self, call: int) -> None:
        system, body = self.messages[0], self.messages[1:]
        keep_from, used = len(body), 0
        for index in range(len(body) - 1, -1, -1):
            size = self.count(render(body[index]))
            if used + size > self.config.keep_recent_tokens:
                break
            used, keep_from = used + size, index
        while keep_from < len(body) and body[keep_from].get("role") == "tool":
            keep_from += 1
        older = [message for message in body[:keep_from] if message is not self._summary_message]
        recent = body[keep_from:]
        if not older:
            return
        before = self.tokens()
        plain = [{"role": m["role"] if m["role"] in ("system", "user", "assistant") else "user", "content": render(m)}
                 for m in older]
        summary = self.summarize(plain, self.previous_summary)
        if self.on_compact is not None:
            self.on_compact(plain)
        self.previous_summary = summary
        self._summary_message = {"role": "user", "content": SUMMARY_LEAD + summary}
        self.messages[:] = [system, self._summary_message, *recent]
        self.events.append({
            "call": call, "tokens_before": before, "tokens_after": self.tokens(), "summarized": len(older),
            "kept": len(recent), "summary": summary, "delivered": self.standing() if self.standing else None,
            "watched_in_summary": {name: text in summary for name, text in self.config.watch.items()},
            "watched_in_kept_turns": {name: any(text in render(m) for m in recent)
                                      for name, text in self.config.watch.items()},
            "mentioned_in_summary": {name: bool(re.search(pattern, summary)) for name, pattern in self.config.mentions.items()},
            "mentioned_in_kept_turns": {name: any(re.search(pattern, render(m)) for m in recent)
                                        for name, pattern in self.config.mentions.items()},
        })

    def execute_actions(self, message: dict) -> list[dict]:
        """As the default agent does, with each action logged: compaction drops old messages, the log keeps them all."""
        outputs = []
        for action in message.get("extra", {}).get("actions", []):
            kind = action.get("memory_tool", "bash")
            row = {"kind": kind, "call": self.n_calls, "task": self.task_name, "compactions": len(self.events),
                   "command": action["command"], "said": str(message.get("content") or "")[:OUTPUT_KEPT]}
            try:
                output = self.env.execute(action)
            except Exception as error:
                if self.on_action:
                    extra = (getattr(error, "messages", None) or [{}])[0].get("extra", {})
                    self.on_action({**row, "submission": extra["submission"]} if "submission" in extra else row)
                raise
            if self.on_action:
                self.on_action({**row, "returncode": output.get("returncode"), "output": str(output.get("output", ""))
                                if kind != "bash" else str(output.get("output", ""))[:OUTPUT_KEPT]})
            outputs.append(output)
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def _until_exit(self) -> dict:
        """The default agent's loop, after its first two messages."""
        started = self.n_calls
        self.n_consecutive_format_errors = 0
        while True:
            if 0 < self.config.stop_at_tokens < self.tokens():
                return {"exit_status": "WindowFull", "submission": ""}
            if 0 < self.config.task_calls <= self.n_calls - started:
                return {"exit_status": "TaskCallLimit", "submission": ""}
            try:
                self.step()
                self.n_consecutive_format_errors = 0
            except FormatError as error:
                if self.on_action:
                    self.on_action({"kind": "no_command", "call": self.n_calls, "task": self.task_name, "compactions": len(self.events),
                                    "command": "", "output": getattr(self.model, "last_text", "")[:OUTPUT_KEPT],
                                    "finish_reason": getattr(self.model, "last_finish", ""),
                                    "error": getattr(self.model, "last_error", "")})
                self.cost += error.messages[0].get("extra", {}).get("cost", 0.0)
                self.n_consecutive_format_errors += 1
                limit = self.config.max_consecutive_format_errors
                self.add_messages(*error.messages)
                if 0 < limit <= self.n_consecutive_format_errors:
                    self.add_messages({"role": "exit", "content": "RepeatedFormatError",
                                       "extra": {"exit_status": "RepeatedFormatError", "submission": ""}})
            except InterruptAgentFlow as error:
                self.add_messages(*error.messages)
            except Exception as error:
                self.handle_uncaught_exception(error)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                return self.messages[-1].get("extra", {})

    def run_tasks(self, tasks: list[dict], make_env: Callable[[dict], Any],
                  worded: Callable[[dict, int], str] | None = None) -> list[dict]:
        """Several tasks in one conversation, each in its own container. A submitted task's exit message is never sent
        to the model: the next task follows as a user message, and only the container changes. A task past its call limit, or
        one where the model stops issuing commands, is left the same way. Anything else ends the session."""
        self.outcomes = []
        opened = False
        machines = 0  # the tasks that got a machine so far: the first of them is where a session's rules are told
        try:
            for task in tasks:
                self.task_name = task["name"]
                self._close_machine()
                try:
                    self.env.inner = make_env(task)
                except Exception as error:  # one machine that will not build leaves its task, not the session
                    self.outcomes.append({"task": task["name"], "exit_status": "MachineFailed", "reason": f"{type(error).__name__}: {error}"[:500],
                                          "calls": self.n_calls, "compactions": len(self.events)})
                    continue
                text = worded(task, machines) if worded else task["instruction"]
                machines += 1
                if not opened:
                    opened = True
                    self.extra_template_vars |= {"task": text}
                    self.messages = []
                    self.add_messages(
                        self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
                        self.model.format_message(role="user", content=self._render_template(self.config.instance_template)))
                    self._told.add(id(self.messages[1]))
                else:
                    if self.messages[-1].get("role") == "exit":
                        self.messages.pop()
                    self.add_messages({"role": "user", "content": text})
                ended = self._until_exit()
                self.outcomes.append({"task": task["name"], "exit_status": ended.get("exit_status"), "calls": self.n_calls,
                                      "compactions": len(self.events)})
                if ended.get("exit_status") not in ("Submitted", "TaskCallLimit", "RepeatedFormatError"):
                    break
        finally:
            self._close_machine()
        return self.outcomes

    def _close_machine(self) -> None:
        """A finished task's machine holds memory on a shared node, so it goes before the next one starts."""
        machine, self.env.inner = self.env.inner, None
        getattr(machine, "cleanup", lambda: None)()

    def serialize(self, *extra_dicts: dict) -> dict:
        return super().serialize({"info": {"compactions": self.events, "recorded_to_memory": self.recorded}},
                                 *extra_dicts)
