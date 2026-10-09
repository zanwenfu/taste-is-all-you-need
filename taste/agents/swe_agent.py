"""SWE-agent (Yang, Jimenez, Wettig, Lieret, Yao, Narasimhan and Press; MIT licence), run unchanged on Taste.

The configuration is the one SWE-Bench Pro's public leaderboard is run with:
Scale's ``config/tool_use.yaml`` (scaleapi/SWE-agent, branch
scale-customizations, commit 1ac1581, SWE-agent 1.1.0) with its tool bundles
``registry``, ``edit_anthropic`` and ``submit``. The files under
``swe_agent_files`` are that commit's, byte for byte, except
``tools/registry/lib``, which that commit dropped although its editor imports
it; it is its parent's (8ea180b), as SWE-agent ships it.

What stays SWE-agent's: its agent loop (``DefaultAgent``: setup, steps,
requeries after format, blocklist and bash syntax errors, autosubmission after
an error), its environment class (``SWEEnv``: the shell session, the
repository, the tool bundles' installation and state commands), its tools and
their function-calling parser, its templates, and when it stops.

What Taste supplies, as three classes behind SWE-agent's and SWE-ReX's own
interfaces:

- ``TasteModel`` in place of its LiteLLM model class. Upstream's own code
  turns the history into chat messages and counts the cost against its
  limits; the messages are sent through the host as Responses items (a tool
  call and its result as function call items), and the reply comes back as
  LiteLLM returns one: its text and its function calls. Taste's transport has
  its own resend rules, so upstream's retry loop is not used.
- ``TasteRuntime`` in place of SWE-ReX's runtime, which runs a server inside
  the task container holding one interactive bash session. Here every
  operation is one command in the task container through the host. The
  session is carried from one command to the next in files under
  ``/root/.taste-swe-agent``: each command starts, as SWE-ReX's session does,
  from an empty environment, gets back the variables, functions, aliases and
  directory the last one left, is checked with ``bash -n`` first (SWE-ReX's
  syntax check), runs, and leaves its own. Files are read, written and
  uploaded with commands too, shown in the record as ``# swe-rex: ...``.
- ``TasteDeployment``: that runtime, already running (Harbor started the
  container).

Differences, disclosed with results: shell options, background jobs, history
expansion and the interactive shell's job control do not carry over or apply
(each command runs as a script in a fresh shell, its session state put back);
a command's time limit is at most what the trial's terminal allows (the
configuration asks 450 s); the task container has no network, so the editor bundle's install of
tree-sitter fails as SWE-agent tolerates and its outline of long Python files
falls back to the file's start; the repository is the task container's own and
is not reset to a base commit (Harbor starts it there); reasoning items are not
kept in the history (LiteLLM's chat interface keeps none); function tools are
sent non-strict, as a chat completion sends them; output beyond the terminal
broker's retention limit is cut.

SWE-agent is installed without its dependencies (LiteLLM would replace the
``openai`` package this project needs); what its modules import is in the
``swe-agent`` extra.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import os
import posixpath
import re
import shlex
import tempfile
from pathlib import Path
from typing import Any

# SWE-agent finds its configuration and tools beside its package; here they
# are Taste's copies (FILES). Its logging stays quiet, and LiteLLM (imported,
# never called) reads its bundled price table instead of fetching one.
os.environ.setdefault("SWE_AGENT_CONFIG_DIR", str(Path(__file__).resolve().with_name("swe_agent_files") / "config"))
os.environ.setdefault("SWE_AGENT_TOOLS_DIR", str(Path(__file__).resolve().with_name("swe_agent_files") / "tools"))
os.environ.setdefault("SWE_AGENT_TRAJECTORY_DIR", tempfile.gettempdir())
os.environ.setdefault("SWE_AGENT_LOG_STREAM_LEVEL", "ERROR")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import sweagent
import yaml
from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig
from sweagent.agent.models import AbstractModel, GenericAPIModelConfig, InstanceStats, LiteLLMModel
from sweagent.agent.problem_statement import TextProblemStatement
from sweagent.environment.repo import PreExistingRepoConfig
from sweagent.environment.swe_env import SWEEnv
from sweagent.tools.tools import ToolHandler
from sweagent.utils.log import get_logger
from swerex.deployment.abstract import AbstractDeployment
from swerex.exceptions import (
    BashIncorrectSyntaxError,
    CommandTimeoutError,
    NonZeroExitCodeError,
    SwerexException,
)
from swerex.runtime.abstract import (
    AbstractRuntime,
    BashInterruptAction,
    BashObservation,
    CloseBashSessionResponse,
    CloseResponse,
    CommandResponse,
    CreateBashSessionResponse,
    IsAliveResponse,
    ReadFileResponse,
    UploadResponse,
    WriteFileResponse,
)

from taste.agents import AgentExit, HostedStop
from taste.agents.protocols import SWE_AGENT_RUNTIME
from taste.providers._openai import _NATIVE

FILES = Path(__file__).resolve().with_name("swe_agent_files")
NAME = "swe-agent"
CONFIG = "tool_use"
COMMIT = "1ac158175178c9c40ed903fb59cc7aa3c87c75ae"
# SWE-ReX's prompt string, which its session exports.
PS1 = "SHELLPS1PREFIX"
SESSION_DIR = "/root/.taste-swe-agent"
# Printed, with bash -n's exit code, when a command fails SWE-ReX's syntax check.
SYNTAX_ERROR = "TASTE_SWE_AGENT_SYNTAX_ERROR"
SYNTAX_EXIT = 97
MISSING_EXIT = 66
# What a call with no time limit is given: the terminal caps it at its own maximum.
NO_LIMIT_SECONDS = 3600.0
# Starting the session sources the image's startup file; SWE-ReX gives that 10 s inside the
# container, and a command here also pays for starting a process in it.
STARTUP_FLOOR_SECONDS = 30.0
FILE_SECONDS = 60.0
# Variables a shell sets for itself (the host's name among them: another container's is not this
# one's), and the wrapper's own: not part of the session.
_NOT_SESSION = ("BASH|BASH_*|BASHOPTS|BASHPID|COLUMNS|COMP_*|DIRSTACK|EPOCHREALTIME|EPOCHSECONDS|EUID|"
                "FUNCNAME|GROUPS|HISTCMD|HISTFILE|HISTFILESIZE|HISTSIZE|HOSTNAME|HOSTTYPE|LINENO|LINES|"
                "MACHTYPE|OPTERR|OPTIND|OSTYPE|PIPESTATUS|PPID|PWD|RANDOM|SECONDS|SHELLOPTS|SHLVL|SRANDOM|UID|"
                "_|__taste_*")
# The session's state after a command: its variables, functions, aliases and directory. The
# directory (@D@) is written out, so that the agent's command sees no variable of the wrapper's.
_SAVE = r"""__taste_status=$?
{ for __taste_name in $(compgen -v); do case "$__taste_name" in @NOT_SESSION@) ;; *) declare -p "$__taste_name" ;; esac; done; } > @D@/vars.new 2>/dev/null && mv -f @D@/vars.new @D@/vars
declare -f > @D@/functions 2>/dev/null
alias -p > @D@/aliases 2>/dev/null
pwd > @D@/cwd 2>/dev/null
exit "$__taste_status"
""".replace("@NOT_SESSION@", _NOT_SESSION)
_START = r"""exec 2>&1
mkdir -p @D@ && cd /
@STARTUP@
""" + _SAVE
_COMMAND = r"""exec 2>&1
{ . @D@/vars; . @D@/functions; . @D@/aliases; } >/dev/null 2>&1
shopt -s expand_aliases
cd -- "$(cat @D@/cwd 2>/dev/null || echo /)" >/dev/null 2>&1
@WRITE@
if ! /bin/bash -n < @D@/command > @D@/syntax 2>&1; then printf '%s\n' @SYNTAX_ERROR@; cat @D@/syntax; exit @SYNTAX_EXIT@; fi
. @D@/command
""".replace("@SYNTAX_ERROR@", SYNTAX_ERROR).replace("@SYNTAX_EXIT@", str(SYNTAX_EXIT)) + _SAVE
_ANSI = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
# Taste's terminal takes a command of at most 64 KiB. A longer one (an editor call that creates a
# large file) is written to the session's directory first, in parts of PART_CHARS, then run from there.
COMMAND_BYTES = 60_000
PART_CHARS = 48_000


def _encoded(text: str) -> str:
    return base64.b64encode(text.encode("utf-8", errors="surrogateescape")).decode()


def _session(write: str, directory: str) -> str:
    script = _COMMAND.replace("@WRITE@", write).replace("@D@", shlex.quote(directory))
    return "exec env -i bash -c " + shlex.quote(script)


def session_command(command: str, *, directory: str = SESSION_DIR) -> str:
    """What the task container runs for one command of the shell session."""
    return _session(f"printf '%s' '{_encoded(command)}' | base64 -d > @D@/command", directory)


def command_parts(command: str, *, directory: str = SESSION_DIR) -> list[str]:
    """For a command too long to send whole: the commands that write it to the session's directory."""
    encoded, target = _encoded(command), shlex.quote(directory) + "/command.b64"
    parts = [encoded[start:start + PART_CHARS] for start in range(0, len(encoded), PART_CHARS)]
    return [f"mkdir -p {shlex.quote(directory)} && printf '%s' '{part}' {'>' if index == 0 else '>>'} {target}"
            for index, part in enumerate(parts)]


def stored_session_command(*, directory: str = SESSION_DIR) -> str:
    """What the task container runs for a command ``command_parts`` wrote."""
    return _session("base64 -d < @D@/command.b64 > @D@/command", directory)


def start_command(startup: list[str], *, directory: str = SESSION_DIR) -> str:
    """What the task container runs to start the session: SWE-ReX's environment and startup commands."""
    script = _START.replace("@D@", shlex.quote(directory)).replace("@STARTUP@", " ; ".join(startup))
    return f"exec env -i PS1={PS1} PS2= PS0= HISTFILE=/dev/null bash -i -c " + shlex.quote(script)


def strip_control_chars(text: str) -> str:
    """SWE-ReX's ``_strip_control_chars``."""
    return _ANSI.sub("", text)


class _Guarded:
    """The host, remembering the first time Taste ended the run.

    SWE-agent catches every exception in its loop and tries to submit what is
    there; a stop is one. The runtime then reports itself dead, so nothing more
    runs, and the run's end raises the stop again.
    """

    def __init__(self, host):
        self.host, self.stop = host, None

    def ask(self, **request):
        try:
            return self.host.ask(**request)
        except HostedStop as stop:
            self.stop = self.stop or stop
            raise

    def run(self, command, **request):
        try:
            return self.host.run(command, **request)
        except HostedStop as stop:
            self.stop = self.stop or stop
            raise


# -- the runtime: SWE-ReX's, with the task's container behind the host -------------------------


class TasteRuntime(AbstractRuntime):
    """SWE-ReX's runtime interface, each operation one command in the task's container."""

    def __init__(self, host, *, files=FILES, session_dir=SESSION_DIR):
        self.host = host if isinstance(host, _Guarded) else _Guarded(host)
        self.files, self.session_dir = Path(files), session_dir
        self.logger = get_logger("swea-taste-runtime", emoji="🧭")

    def _run(self, command, *, timeout, shown):
        return self.host.run(command, cwd="/", timeout_seconds=float(timeout), shown=shown)

    @property
    def stopped(self):
        return self.host.stop

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        if self.stopped is not None:
            return IsAliveResponse(is_alive=False, message=f"Taste ended the run: {self.stopped}")
        return IsAliveResponse(is_alive=True)

    async def create_session(self, request) -> CreateBashSessionResponse:
        if request.session != "default":
            raise SwerexException("one shell session is kept per task container")
        startup = [f"source {path}" for path in request.startup_source]
        startup += [f"export PS1='{PS1}'", "export PS2=''", "export PS0=''"]
        result = self._run(start_command(startup, directory=self.session_dir),
                           timeout=max(request.startup_timeout, STARTUP_FLOOR_SECONDS),
                           shown=SWE_AGENT_RUNTIME + "start the shell session: " + " ; ".join(startup))
        if result.timed_out:
            raise CommandTimeoutError("timeout while starting the shell session")
        return CreateBashSessionResponse(output=strip_control_chars(result.output))

    async def run_in_session(self, action) -> BashObservation:
        if isinstance(action, BashInterruptAction):
            # The terminal ended the command when its time ran out; there is nothing left to interrupt.
            return BashObservation(output="", exit_code=0, expect_string=PS1)
        if action.session != "default":
            raise SwerexException("one shell session is kept per task container")
        if action.is_interactive_command or action.is_interactive_quit:
            raise SwerexException("interactive commands are not supported on Taste's runtime")
        limit = action.timeout if action.timeout is not None else NO_LIMIT_SECONDS
        text = session_command(action.command, directory=self.session_dir)
        if len(text.encode()) > COMMAND_BYTES:
            parts = command_parts(action.command, directory=self.session_dir)
            for index, part in enumerate(parts, 1):
                written = self._run(part, timeout=FILE_SECONDS,
                                    shown=SWE_AGENT_RUNTIME + f"write the command, part {index} of {len(parts)}")
                if written.returncode != 0 or written.timed_out:
                    raise SwerexException(f"could not write the command: {written.output[:500]}")
            text = stored_session_command(directory=self.session_dir)
        result = self._run(text, timeout=limit, shown=action.command)
        if result.timed_out:
            raise CommandTimeoutError(f"timeout after {action.timeout} seconds while running command "
                                      f"{action.command!r}")
        if result.returncode == SYNTAX_EXIT and result.output.startswith(SYNTAX_ERROR + "\n"):
            stderr = result.output[len(SYNTAX_ERROR) + 1:]
            raise BashIncorrectSyntaxError(
                f"Error (exit code 2) while checking bash command \n{action.command!r}\n"
                f"---- Stderr ----\n{stderr}\n---- Stdout ----\n",
                extra_info={"bash_stdout": "", "bash_stderr": stderr})
        output = strip_control_chars(result.output).strip()
        if action.check == "ignore":
            return BashObservation(output=output, exit_code=None, expect_string=PS1)
        if action.check == "raise" and result.returncode != 0:
            message = (f"Command {action.command!r} failed with exit code {result.returncode}. "
                       f"Here is the output:\n{output!r}")
            raise NonZeroExitCodeError(f"{action.error_msg}: {message}" if action.error_msg else message)
        return BashObservation(output=output, exit_code=result.returncode, expect_string=PS1)

    async def close_session(self, request) -> CloseBashSessionResponse:
        return CloseBashSessionResponse()

    async def execute(self, command) -> CommandResponse:
        text = command.command if isinstance(command.command, str) else shlex.join(command.command)
        program = f"/bin/sh -c {shlex.quote(text)}" if command.shell or isinstance(command.command, str) else text
        if command.env is not None:
            program = "env -i " + " ".join(f"{name}={shlex.quote(str(value))}"
                                           for name, value in command.env.items()) + " " + program
        if command.cwd:
            program = f"cd {shlex.quote(command.cwd)} && {program}"
        limit = command.timeout if command.timeout is not None else NO_LIMIT_SECONDS
        result = self._run(program + " 2>&1", timeout=limit, shown=SWE_AGENT_RUNTIME + "execute " + text)
        if result.timed_out:
            raise CommandTimeoutError(f"Timeout ({command.timeout}s) exceeded while running command")
        response = CommandResponse(stdout=result.output, stderr="", exit_code=result.returncode)
        if command.check and result.returncode != 0:
            message = (f"Command {command.command!r} failed with exit code {result.returncode}. "
                       f"Stdout:\n{response.stdout!r}\nStderr:\n''")
            raise NonZeroExitCodeError(f"{command.error_msg}: {message}" if command.error_msg else message)
        return response

    async def read_file(self, request) -> ReadFileResponse:
        path = shlex.quote(request.path)
        result = self._run(f"if [ -e {path} ]; then cat -- {path}; else exit {MISSING_EXIT}; fi",
                           timeout=FILE_SECONDS, shown=SWE_AGENT_RUNTIME + "read_file " + request.path)
        if result.returncode == MISSING_EXIT:
            raise FileNotFoundError(request.path)
        if result.returncode != 0 or result.timed_out:
            raise SwerexException(f"could not read {request.path}: {result.output[:500]}")
        return ReadFileResponse(content=result.output)

    async def write_file(self, request) -> WriteFileResponse:
        path = shlex.quote(request.path)
        parent = shlex.quote(posixpath.dirname(request.path) or "/")
        result = self._run(f"mkdir -p {parent} && printf '%s' '{_encoded(request.content)}' | base64 -d > {path}",
                           timeout=FILE_SECONDS, shown=SWE_AGENT_RUNTIME + "write_file " + request.path)
        if result.returncode != 0 or result.timed_out:
            raise SwerexException(f"could not write {request.path}: {result.output[:500]}")
        return WriteFileResponse()

    async def upload(self, request) -> UploadResponse:
        """Every file under the source, written below the target; the same command wherever Taste runs."""
        source = Path(request.source_path)
        files = sorted(path for path in source.rglob("*") if path.is_file()) if source.is_dir() else [source]
        named = [(path.relative_to(source).as_posix() if source.is_dir() else source.name, path) for path in files]
        parents = sorted({posixpath.dirname(name) for name, _ in named} - {""})
        target = request.target_path if source.is_dir() else posixpath.dirname(request.target_path) or "/"
        if not source.is_dir():
            named = [(posixpath.basename(request.target_path), files[0])]
        parts = [f"mkdir -p {shlex.quote(target)}", f"cd {shlex.quote(target)}"]
        parts += [f"mkdir -p {shlex.quote(parent)}" for parent in parents]
        parts += [f"printf '%s' '{base64.b64encode(path.read_bytes()).decode()}' | base64 -d > {shlex.quote(name)}"
                  for name, path in named]
        try:
            shown = source.relative_to(self.files).as_posix()
        except ValueError:
            shown = source.name
        result = self._run(" && ".join(parts), timeout=FILE_SECONDS,
                           shown=SWE_AGENT_RUNTIME + f"upload {shown} -> {request.target_path}")
        if result.returncode != 0 or result.timed_out:
            raise SwerexException(f"could not upload to {request.target_path}: {result.output[:500]}")
        return UploadResponse()

    async def close(self) -> CloseResponse:
        return CloseResponse()


class TasteDeployment(AbstractDeployment):
    """SWE-ReX's deployment: the task's container, which Harbor started and will stop."""

    def __init__(self, runtime: TasteRuntime):
        self._runtime = runtime
        self.logger = runtime.logger

    def add_hook(self, hook):
        pass

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        return await self._runtime.is_alive(timeout=timeout)

    async def start(self, *args, **kwargs):
        pass

    async def stop(self, *args, **kwargs):
        pass

    @property
    def runtime(self) -> TasteRuntime:
        return self._runtime


# -- the model: upstream's messages and accounting, with the host as its transport ---------------


def canonical_messages(items):
    """Responses input items, as Taste's provider replays them byte for byte (as for mini-swe-agent)."""
    messages = []
    for item in items:
        if set(item) == {"role", "content"} and isinstance(item["content"], str):
            messages.append({"role": item["role"], "content": item["content"]})
        else:
            messages.append({"role": "user", "content": [{"type": _NATIVE, "item": copy.deepcopy(dict(item))}]})
    return messages


def request_items(messages):
    """Chat messages, as upstream sends them to LiteLLM, as Responses input items."""
    items = []
    for message in messages:
        if not isinstance(message["content"], str):
            raise ValueError("only text messages are sent: the configuration shows the model no images")
        if message["role"] == "tool":
            items.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                          "output": message["content"]})
            continue
        items.append({"role": message["role"], "content": message["content"]})
        for call in message.get("tool_calls") or ():
            items.append({"type": "function_call", "call_id": call["id"], "name": call["function"]["name"],
                          "arguments": call["function"]["arguments"]})
    return items


def responses_tool(tool):
    """A chat function tool as a Responses function tool, non-strict as a chat completion's is."""
    function = tool["function"]
    return {"type": "function", "name": function["name"], "description": function.get("description", ""),
            "parameters": copy.deepcopy(function["parameters"]), "strict": False}


class TasteModel(AbstractModel):
    """SWE-agent's model interface, with the host as its transport and upstream's own accounting."""

    def __init__(self, *, host, effort, config: GenericAPIModelConfig, tools):
        self.config = config.model_copy(deep=True)
        self.stats = InstanceStats()
        self.tools = tools
        self.logger = get_logger("swea-lm", emoji="🤖")
        self.host, self.effort = host, effort
        self._tools = [responses_tool(tool) for tool in tools.tools] if tools.use_function_calling else []

    @property
    def instance_cost_limit(self) -> float:
        return self.config.per_instance_cost_limit

    def query(self, history, n: int | None = 1, temperature: float | None = None) -> dict:
        if n not in (None, 1) or temperature is not None:
            raise ValueError("one sample at the trial's settings")
        messages = LiteLLMModel._history_to_messages(self, history)
        reply = self.host.ask(messages=canonical_messages(request_items(messages)), tools=copy.deepcopy(self._tools),
                              effort=self.effort)
        texts = [part.get("text", "") for item in reply.output if item.get("type") == "message"
                 for part in item.get("content") or () if part.get("type") == "output_text"]
        output: dict[str, Any] = {"message": "\n".join(text for text in texts if text)}
        if self.tools.use_function_calling:
            output["tool_calls"] = [{"function": {"arguments": item["arguments"], "name": item["name"]},
                                     "id": item["call_id"], "type": "function"}
                                    for item in reply.output if item.get("type") == "function_call"]
        usage = reply.usage
        sent = sum(int(usage.get(key) or 0) for key in ("input_tokens", "cache_read_tokens", "cache_write_tokens"))
        LiteLLMModel._update_stats(self, input_tokens=sent, output_tokens=int(usage.get("output_tokens") or 0),
                                   cost=reply.cost_usd)
        return output


# -- the agent ----------------------------------------------------------------------------------


def _files_sha256(root=FILES):
    digest = hashlib.sha256()
    for path in sorted(item for item in Path(root).rglob("*") if item.is_file() and "__pycache__" not in item.parts):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


class SweAgent:
    """SWE-agent with Scale's SWE-Bench Pro configuration, hosted on Taste."""

    name = NAME

    def __init__(self, *, effort, config=CONFIG):
        self._raw = (FILES / "config" / f"{config}.yaml").read_bytes()
        self._loaded = yaml.safe_load(self._raw)
        self.config_name, self.effort = config, effort
        self._sha256 = _files_sha256()

    def identity(self):
        return {"name": NAME, "version": sweagent.__version__, "config": self.config_name,
                "config_sha256": self._sha256, "commit": COMMIT}

    def agent_config(self, model_name) -> DefaultAgentConfig:
        """The configuration as SWE-agent reads it, its bundles found in Taste's copies."""
        agent = copy.deepcopy(self._loaded["agent"])
        for bundle in agent["tools"]["bundles"]:
            bundle["path"] = str(FILES / bundle["path"])
        return DefaultAgentConfig(**agent, model={"name": model_name})

    def run(self, task, host, *, cwd, model_name):
        config = self.agent_config(model_name)
        guarded = _Guarded(host)
        agent = DefaultAgent(templates=config.templates, tools=ToolHandler(config.tools),
                             history_processors=config.history_processors,
                             model=TasteModel(host=guarded, effort=self.effort, config=config.model, tools=config.tools),
                             max_requeries=config.max_requeries, action_sampler_config=config.action_sampler)
        name = cwd.strip("/")
        repo = PreExistingRepoConfig(repo_name=name, reset=False) if name else None
        env = SWEEnv(deployment=TasteDeployment(TasteRuntime(guarded)), repo=repo, post_startup_commands=[])
        env.start()
        with tempfile.TemporaryDirectory(prefix="taste-swe-agent-") as output:
            result = agent.run(env=env, problem_statement=TextProblemStatement(text=task), output_dir=Path(output))
        if guarded.stop is not None:
            raise guarded.stop
        exit_status = str(result.info.get("exit_status") or "")
        return AgentExit("Submitted" if exit_status == "submitted" else exit_status,
                         result.info.get("submission") or "", tuple(copy.deepcopy(agent.history)),
                         agent.model.stats.api_calls)
