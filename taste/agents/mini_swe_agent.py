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

from taste.agents import AgentExit, ModelReply
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


class TasteEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 30


class TasteEnvironment:
    """mini-swe-agent's local environment, with the task's container as its machine."""

    def __init__(self, *, host, config_class=TasteEnvironmentConfig, **kwargs):
        self.config = config_class(**kwargs)
        if not self.config.cwd.startswith("/"):
            raise ValueError("a hosted agent works in the task's directory, an absolute path")
        self.host = host

    def _shell(self, command):
        exports = " ".join(f"{name}={shlex.quote(value)}" for name, value in self.config.env.items())
        # One stream in arrival order, as upstream's stderr=STDOUT gives; a
        # group, so that a heredoc ending the command still closes.
        return (f"export {exports}; " if exports else "") + "{\n" + command + "\n} 2>&1"

    def execute(self, action, cwd="", *, timeout=None):
        command = action.get("command", "")
        limit = timeout or self.config.timeout
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
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), kwargs)

    def serialize(self):
        return {"info": {"config": {"environment": self.config.model_dump(mode="json"),
                                    "environment_type": f"{type(self).__module__}.{type(self).__name__}"}}}


class MiniSweAgent:
    """mini-swe-agent with one of its own configurations, hosted on Taste."""

    name = NAME

    def __init__(self, *, effort, config="mini"):
        path = Path(builtin_config_dir) / f"{config}.yaml"
        self._raw = path.read_bytes()
        loaded = yaml.safe_load(self._raw)
        self._configs = {part: dict(loaded.get(part) or {}) for part in ("agent", "model", "environment")}
        self.config_name, self.effort = config, effort

    def configs(self):
        return copy.deepcopy(self._configs)

    def identity(self):
        return {"name": NAME, "version": minisweagent.__version__, "config": self.config_name,
                "config_sha256": hashlib.sha256(self._raw).hexdigest()}

    def run(self, task, host, *, cwd, model_name):
        configs = self.configs()
        agent = DefaultAgent(
            TasteResponsesModel(host=host, effort=self.effort, model_name=model_name, **configs["model"]),
            TasteEnvironment(host=host, cwd=cwd, **configs["environment"]),
            **configs["agent"],
        )
        result = agent.run(task)
        return AgentExit(result.get("exit_status", ""), result.get("submission", ""),
                         tuple(copy.deepcopy(agent.messages)), agent.n_calls)
