"""mini-swe-agent (Lieret and Jimenez; MIT licence), run unchanged on Taste.

What stays mini-swe-agent's: its agent loop (``DefaultAgent``), its ``mini``
configuration (the prompts, templates and limits Harbor runs it with on
Terminal-Bench), its bash tool, how it parses a reply into commands and
formats their outputs, and when it stops.

What Taste supplies, as two classes behind mini-swe-agent's own interfaces:

- ``TasteResponsesModel`` in place of its LiteLLM Responses model class. It
  builds the same request (the same input items, tool definition and reasoning
  effort) and sends it through the host; the rest of ``query`` follows
  upstream's code path. Taste's transport has its own resend rules, so
  upstream's retry loop is not used, and LiteLLM-only options (its timeout,
  parameter dropping, extra headers) have no meaning here.
- ``TasteEnvironment`` in place of its local environment, which under Harbor
  runs inside the task container. Commands go to the container through the
  host, with standard error folded into the output, upstream's 30-second
  default limit, its timeout message and its submission protocol.

Differences, disclosed with results: Taste caps each reply's output tokens
(a budget needs a ceiling per call); the environment's template values leave
out the host process's environment variables (they can hold credentials, and
the ``mini`` prompts use none of them); and output beyond the terminal
broker's retention limit is cut where upstream would keep it.

mini-swe-agent is installed without its dependencies (LiteLLM would replace
the ``openai`` package this project needs); what these modules import is in
the ``agents`` extra.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

import minisweagent
import yaml
from jinja2 import StrictUndefined, Template
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir
from minisweagent.exceptions import FormatError, Submitted
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_toolcall_response import (
    BASH_TOOL_RESPONSE_API,
    finish_reason_from_responses_api,
    format_toolcall_observation_messages,
    parse_toolcall_actions_response,
)
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content
from minisweagent.utils.serialize import recursive_merge
from pydantic import BaseModel

from taste.agents import AgentExit, ModelReply, agentrewind, uname_override
from taste.providers._openai import _NATIVE

NAME = "mini-swe-agent"
# LiteLLM transport options in upstream configurations; nothing to send here.
_TRANSPORT_ONLY = frozenset({"drop_params", "timeout", "extra_headers"})
# Upstream's defaults (LitellmModelConfig), for configurations that set none.
_OBSERVATION_TEMPLATE = (
    "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
    "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
)


def canonical_messages(items):
    """Responses input items, as Taste's provider replays them byte for byte.

    A plain role message keeps its form; every other item (reasoning, a
    function call or its output, a structured message) travels as a native
    item, which the provider sends back exactly as it is.
    """
    messages = []
    for item in items:
        if set(item) == {"role", "content"} and isinstance(item["content"], str):
            messages.append({"role": item["role"], "content": item["content"]})
        else:
            messages.append({"role": "user", "content": [{"type": _NATIVE, "item": copy.deepcopy(dict(item))}]})
    return messages


class _Response:
    """A reply in the shape mini-swe-agent's Responses model class reads."""

    def __init__(self, reply: ModelReply):
        self.output = [copy.deepcopy(dict(item)) for item in reply.output]
        self.status = reply.status
        self.incomplete_details = {"reason": reply.incomplete_reason} if reply.incomplete_reason else None
        self.usage = dict(reply.usage)
        self.model = reply.model
        self.cost = reply.cost_usd

    def model_dump(self, mode=None):
        return copy.deepcopy({"object": "response", "model": self.model, "status": self.status,
                              "output": self.output, "usage": self.usage,
                              "incomplete_details": self.incomplete_details})


class TasteResponsesModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict[str, Any] = {}
    format_error_template: str = "{{ error }}"
    observation_template: str = _OBSERVATION_TEMPLATE
    multimodal_regex: str = ""


class TasteResponsesModel:
    """mini-swe-agent's Responses model class, with the host as its transport."""

    def __init__(self, *, host, effort, config_class=TasteResponsesModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        options = dict(self.config.model_kwargs)
        reasoning = options.pop("reasoning", None)
        for name in options:
            if name not in _TRANSPORT_ONLY:
                raise ValueError(f"model option {name!r} has no meaning on Taste's transport")
        if reasoning is not None and reasoning != {"effort": effort}:
            raise ValueError("the reasoning effort is the trial's, set once")
        self.host, self.effort = host, effort

    def _prepare_messages_for_api(self, messages):
        """Upstream's: flatten response objects into their output items for stateless calls."""
        result = []
        for msg in messages:
            if msg.get("object") == "response":
                for item in msg.get("output", []):
                    result.append({k: v for k, v in item.items() if k != "extra"})
            else:
                result.append({k: v for k, v in msg.items() if k != "extra"})
        return result

    def _query(self, messages, **kwargs):
        if kwargs:
            raise ValueError("per-call model options are not supported")
        return _Response(self.host.ask(messages=canonical_messages(messages),
                                       tools=[copy.deepcopy(BASH_TOOL_RESPONSE_API)], effort=self.effort))

    def query(self, messages, **kwargs):
        response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        cost_output = {"cost": response.cost}
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        try:
            actions = self._parse_actions(response)
        except FormatError as error:
            error.messages[0]["extra"].update(cost_output)
            error.messages[0]["extra"]["response"] = response.model_dump(mode="json")
            raise
        message = response.model_dump()
        message["extra"] = {"actions": actions, **cost_output, "timestamp": time.time()}
        return message

    def _parse_actions(self, response):
        return parse_toolcall_actions_response(
            getattr(response, "output", []),
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": finish_reason_from_responses_api(response)},
        )

    def format_message(self, **kwargs):
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(self, message, outputs, template_vars=None):
        return format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []), outputs=outputs,
            observation_template=self.config.observation_template, template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs):
        return self.config.model_dump()

    def serialize(self):
        return {"info": {"config": {"model": self.config.model_dump(mode="json"),
                                    "model_type": f"{type(self).__module__}.{type(self).__name__}"}}}


class RewindResponsesModel(TasteResponsesModel):
    """The rewind arm's model class (``taste.agents.agentrewind``): AgentRewind's two tools beside the bash tool
    and its guidance after the system prompt on every call, and from call ``at_call`` on its notebook after
    the messages that call restores. What the agent keeps is its own: the additions are what is sent."""

    def __init__(self, *, notes=(), at_call=None, **kwargs):
        super().__init__(**kwargs)
        self.notes, self.at_call = [dict(note) for note in notes], at_call
        self._calls, self._note_at = 0, None

    def _query(self, messages, **kwargs):
        if kwargs:
            raise ValueError("per-call model options are not supported")
        sent = canonical_messages(messages)
        if not sent or sent[0].get("role") != "system" or not isinstance(sent[0].get("content"), str):
            raise ValueError("the rewind arm's guidance follows the system prompt, and this request has none")
        sent[0] = {**sent[0], "content": agentrewind.with_guidance(sent[0]["content"])}
        self._calls += 1
        if self.at_call is not None and self._calls >= self.at_call:
            # Where the restored messages end, the first time; there it stays, as the rewound context grows.
            self._note_at = len(sent) if self._note_at is None else self._note_at
            sent.insert(self._note_at, agentrewind.notebook_message(self.notes))
        tools = [copy.deepcopy(BASH_TOOL_RESPONSE_API), *agentrewind.tools()]
        return _Response(self.host.ask(messages=sent, tools=tools, effort=self.effort))

    def _parse_actions(self, response):
        """Upstream's parsing for its bash calls; a call of the two tools becomes a command the host answers."""
        calls = [item for item in getattr(response, "output", []) or [] if item.get("type") == "function_call"]
        if not any(item.get("name") in agentrewind.TOOL_NAMES for item in calls):
            return super()._parse_actions(response)
        actions = []
        for item in calls:
            if item.get("name") not in agentrewind.TOOL_NAMES:
                actions += parse_toolcall_actions_response(
                    [item], format_error_template=self.config.format_error_template,
                    template_kwargs={"finish_reason": finish_reason_from_responses_api(response)})
                continue
            try:
                arguments = json.loads(item.get("arguments", "{}"))
            except ValueError as error:
                arguments = f"Error parsing tool call arguments: {error}."
            if not isinstance(arguments, dict):
                text = arguments if isinstance(arguments, str) else "Tool call arguments must be a JSON object."
                raise FormatError({"type": "message", "role": "user", "extra": {"interrupt_type": "FormatError"},
                                   "content": [{"type": "input_text", "text": Template(
                                       self.config.format_error_template, undefined=StrictUndefined).render(
                                       error=text, actions=[], has_tool_calls=True,
                                       finish_reason=finish_reason_from_responses_api(response))}]})
            actions.append({"command": agentrewind.tool_command(item["name"], arguments),
                            "tool_call_id": item.get("call_id") or item.get("id"), "rewind_tool": item["name"]})
        return actions


class TasteEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 30


class TasteEnvironment:
    """mini-swe-agent's local environment, with the task's container as its machine."""

    def __init__(self, *, host, config_class=TasteEnvironmentConfig, uname=None, **kwargs):
        self.config = config_class(**kwargs)
        if not self.config.cwd.startswith("/"):
            raise ValueError("a hosted agent works in the task's directory, an absolute path")
        self.host = host
        # The machine the agent is told it runs on, field by field over the host's (agent_uname).
        self.uname = dict(uname or {})

    def _shell(self, command):
        exports = " ".join(f"{name}={shlex.quote(value)}" for name, value in self.config.env.items())
        # One stream in arrival order, as upstream's stderr=STDOUT gives; a
        # group, so that a heredoc ending the command still closes.
        return (f"export {exports}; " if exports else "") + "{\n" + command + "\n} 2>&1"

    def execute(self, action, cwd="", *, timeout=None):
        command = action.get("command", "")
        limit = timeout or self.config.timeout
        if action.get("rewind_tool"):
            # A call of the rewind arm's tools: the host answers it; nothing runs in the container.
            result = self.host.run(command, cwd=cwd or self.config.cwd, timeout_seconds=limit, shown=command)
            return {"output": result.output, "returncode": result.returncode, "exception_info": ""}
        result = self.host.run(self._shell(command), cwd=cwd or self.config.cwd, timeout_seconds=limit,
                               shown=command)
        if result.timed_out:
            error = subprocess.TimeoutExpired(command, limit)
            output = {"output": result.output, "returncode": -1,
                      "exception_info": f"An error occurred while executing the command: {error}",
                      "extra": {"exception_type": type(error).__name__, "exception": str(error)}}
        else:
            output = {"output": result.output, "returncode": result.returncode, "exception_info": ""}
        self._check_finished(output)
        return output

    def _check_finished(self, output):
        """Upstream's submission protocol: the sentinel first, and a clean exit."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted({"role": "exit", "content": submission,
                             "extra": {"exit_status": "Submitted", "submission": submission}})

    def get_template_vars(self, **kwargs):
        return recursive_merge(self.config.model_dump(), {**platform.uname()._asdict(), **self.uname}, kwargs)

    def serialize(self):
        return {"info": {"config": {"environment": self.config.model_dump(mode="json"),
                                    "environment_type": f"{type(self).__module__}.{type(self).__name__}"}}}


class MiniSweAgent:
    """mini-swe-agent with one of its own configurations, hosted on Taste."""

    name = NAME

    def __init__(self, *, effort, config="mini", uname="", rewind=None):
        path = Path(builtin_config_dir) / f"{config}.yaml"
        self._raw = path.read_bytes()
        loaded = yaml.safe_load(self._raw)
        self._configs = {part: dict(loaded.get(part) or {}) for part in ("agent", "model", "environment")}
        self.config_name, self.effort = config, effort
        # agent_uname: not part of the identity, which a branch compares with its record's; it is disclosed
        # with the trial's settings instead.
        self.uname = uname_override(uname)
        # The rewind arm: {"notes", "at_call"}; disclosed with the branch it runs in, not in the identity.
        if rewind is not None and (not isinstance(rewind, dict) or set(rewind) != {"notes", "at_call"}):
            raise ValueError("the rewind arm's agent is given its notes and the call they go in at")
        self.rewind = rewind

    def configs(self):
        return copy.deepcopy(self._configs)

    def identity(self):
        return {"name": NAME, "version": minisweagent.__version__, "config": self.config_name,
                "config_sha256": hashlib.sha256(self._raw).hexdigest()}

    def _agent(self, host, *, cwd, model_name):
        configs = self.configs()
        model = (TasteResponsesModel if self.rewind is None else RewindResponsesModel)(
            host=host, effort=self.effort, model_name=model_name, **configs["model"],
            **({} if self.rewind is None else self.rewind))
        return DefaultAgent(model, TasteEnvironment(host=host, cwd=cwd, uname=self.uname, **configs["environment"]),
                            **configs["agent"])

    def run(self, task, host, *, cwd, model_name):
        agent = self._agent(host, cwd=cwd, model_name=model_name)
        result = agent.run(task)
        return AgentExit(result.get("exit_status", ""), result.get("submission", ""),
                         tuple(copy.deepcopy(agent.messages)), agent.n_calls)
