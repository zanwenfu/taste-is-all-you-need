"""A second reviewer family: the study's checker run as Claude Code in script mode (``claude -p``).

The recovery study's reviewer (``taste.agents.checker``) is one model, GPT-6 Luna, which also judges its own agent's
work. This runs the same review with another family, Claude (Haiku 5.5 by default), through the real ``claude`` CLI
billed to the user's Claude subscription, so that its verdicts can be read beside Luna's by the same analyses.

What is the same as the checker:

- its system prompt and task text, word for word, except that the verdict is given by Claude Code's
  ``StructuredOutput`` tool (whose schema is the checker's ``submit_verdict`` parameters) in place of
  ``submit_verdict``;
- its one tool, a shell in the task container: every Bash command is rewritten by a hook (``cc_hook``) into
  ``docker exec -w <cwd> <container> /bin/sh -c ...`` running the checker's own wrapper (``_shell``), with the
  checker's per-command time limit and the checker's observation (``exit code: N``, then the output cut to its
  beginning and end at the same length); commands that name hidden tests, graders or reference solutions are refused
  with the checker's message, and nothing runs;
- its limits: 30 model calls (``--max-turns``), the last one announced before it is made; 900 seconds for the whole
  review; $2 of (notional) model spending; a verdict that fails the checker's validation is sent back with the
  checker's correction;
- its record: the submission (schema ``taste.agents/CheckerVerdict/1``), with the verdict, the evidence, how the
  review ended, its steps, commands, refusals, cost and tokens, checked by ``parse_submission``.

What confines it: only the Bash and StructuredOutput tools exist (``--tools Bash``); the hook rewrites every command
into the container or denies it, and denies every other tool; the permission rules allow Bash solely as
``docker exec -w <cwd> <container> /bin/sh -c ...``, so a command for any other container or any host command left
unrewritten is refused; settings, CLAUDE.md files, memory, MCP servers and claude.ai connectors are not loaded; the
child process gets a fresh home and configuration directory and an environment built from an allowlist, so no API
key, cloud credential or provider route reaches it; credentials are scrubbed from the hook's and the commands'
environments.

The subscription's token is read from ``CLAUDE_CODE_OAUTH_TOKEN`` (the environment, or one line of an env file), and
passed to ``claude`` only through its environment: never on a command line, never written or logged. ``--bare`` is
never used (it ignores the token). A usage-limit, rate-limit or login failure reported by Claude Code raises
``StopJob``: the caller ends the whole job and reports; nothing here retries it.
"""

from __future__ import annotations

import contextlib
import copy
import glob
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from taste.agents.checker import (
    PROMPTS,
    SCHEMA,
    SUBMIT_TOOL,
    VERSION,
    CheckerSettings,
    InvalidVerdict,
    _shell,
    parse_submission,
    system_prompt,
    validate_verdict,
)

FAMILY = "claude"
VIA = "claude-code-print"
DEFAULT_MODEL = "claude-haiku-5-5"
TOKEN_NAME = "CLAUDE_CODE_OAUTH_TOKEN"
REPO = Path(__file__).resolve().parents[2]
HOOK = Path(__file__).resolve().with_name("cc_hook.py")

# The checker names its verdict tool; Claude Code's is StructuredOutput. These are the only words that change.
VERDICT_TOOL = "StructuredOutput"
_SYSTEM_CHANGE = ("When you have decided, call submit_verdict once with:",
                  f"When you have decided, call {VERDICT_TOOL} once with:")
_TASK_CHANGE = ("Gather your evidence in the container, then call submit_verdict.",
                f"Gather your evidence in the container, then call {VERDICT_TOOL}.")
# The checker's LAST_STEP, "not run: no steps are left", and its correction of a verdict it does not accept.
LAST_STEP = (f"This is your last reply: no more commands will run. Call {VERDICT_TOOL} now, with your verdict from "
             "the evidence you have.")
NO_STEPS = f"not run: no steps are left; call {VERDICT_TOOL}"
CORRECTION = "Your verdict was not accepted: {error}. Call " + VERDICT_TOOL + " with a corrected verdict."
BACKGROUND = "not run: commands run in the foreground; run it again without run_in_background"
EMPTY = "error: bash needs a command, as text"
# The verdict's schema: the checker's submit_verdict parameters, unchanged.
VERDICT_SCHEMA = copy.deepcopy(SUBMIT_TOOL["parameters"])

# What the child process may inherit from this one: nothing that routes, bills or authenticates a model.
ENV_ALLOW = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "USER", "LOGNAME", "SHELL", "TERM",
             "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CERT_PATH", "DOCKER_TLS_VERIFY", "DOCKER_API_VERSION",
             "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY",
             "https_proxy", "http_proxy", "no_proxy")

# What Claude Code says when the subscription cannot serve: matched only on a failed run, never on a verdict.
STOPS = (
    ("usage_limit", re.compile(r"you've hit your[^\n]*limit|usage limit|session limit|weekly limit|"
                               r"limit (?:reached|will reset)|resets? (?:at|in) |out of (?:extra )?usage", re.I)),
    ("spend_limit", re.compile(r"spend limit|credit balance is too low|exceeds your group's budget|"
                               r"usage credits required", re.I)),
    ("rate_limit", re.compile(r"\b429\b|rate[ _-]?limit|temporarily limiting requests|request rejected|"
                              r"too many requests", re.I)),
    ("auth", re.compile(r"oauth|login expired|invalid auth|\b401\b|\b403\b|failed to authenticate|"
                        r"not logged in|does not have access to claude|claude login not accepted|invalid api key|"
                        r"please run /login", re.I)),
)
# How a Claude Code run that ended without a verdict ended, in the checker's words.
SUBTYPE_ENDING = {"error_max_turns": "steps", "error_max_budget_usd": "usd",
                  "error_max_structured_output_retries": "format_errors"}


class StopJob(RuntimeError):
    """The subscription cannot serve (a usage, spend or rate limit, or the login): end the whole job, do not retry."""

    def __init__(self, kind, message):
        super().__init__(f"{kind}: {message}")
        self.kind, self.message = kind, message


class ReviewFailed(RuntimeError):
    """A review that ended in an error of Claude Code or the container, not at a limit: no verdict, no stop."""


@dataclass(frozen=True)
class ReviewSettings:
    """The reviewer's settings: the checker's limits, and how Claude Code is run."""

    model: str = DEFAULT_MODEL
    effort: str = "medium"
    checker_version: str = VERSION
    max_steps: int = 30                 # model calls, the last one included (--max-turns)
    max_seconds: float = 900.0          # the whole review; the last reply is announced when the next would not fit
    kill_grace_seconds: float = 120.0   # past max_seconds, the run is stopped
    command_seconds: float = 120.0      # one command, in the container
    output_chars: int = 10_000          # of one command's output, shown cut to its beginning and end
    max_usd: float = 2.0                # Claude Code's own estimate (not billed on a subscription)
    output_tokens: int = 16_384         # per reply, the study's cap
    max_corrections: int = 3            # verdicts sent back for correction, within the steps and the time
    cli_retries: int = 1                # Claude Code's own retries of a failed request (CLAUDE_CODE_MAX_RETRIES)
    claude: str = "claude"
    docker: str = "docker"
    python: str = sys.executable

    def __post_init__(self):
        if self.checker_version not in PROMPTS:
            raise ValueError(f"no checker version {self.checker_version!r}")
        if self.effort not in ("low", "medium", "high"):
            raise ValueError("effort must be low, medium or high")
        for name in ("max_steps", "output_chars", "output_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 2:
                raise ValueError(f"{name} must be an integer of at least 2")
        for name in ("max_corrections", "cli_retries"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a whole number")
        for name in ("max_seconds", "kill_grace_seconds", "command_seconds", "max_usd"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

    @property
    def tool_timeout_ms(self):
        """Claude Code's own limit on a command: past the container's, so that the checker's observation arrives."""
        return int((self.command_seconds + 30) * 1000)


# -- what the reviewer is told ------------------------------------------------------------------------------------


def _replace_once(text, old, new, what):
    if text.count(old) != 1:
        raise RuntimeError(f"the checker's {what} no longer holds {old!r} once")
    return text.replace(old, new)


def reviewer_system_prompt(cwd, settings: ReviewSettings):
    """The checker's system prompt for ``cwd``, its limits, with the verdict tool's name changed."""
    checker = CheckerSettings(max_steps=settings.max_steps, command_seconds=settings.command_seconds,
                              output_chars=settings.output_chars, max_seconds=settings.max_seconds,
                              max_usd=settings.max_usd)
    return _replace_once(system_prompt(cwd, checker, settings.checker_version), *_SYSTEM_CHANGE, "system prompt")


def reviewer_task(task_text):
    """The checker's task text (``checker_task``, or ``requirement_task``) with the verdict tool's name changed."""
    if not isinstance(task_text, str) or not task_text.strip():
        raise ValueError("the reviewer needs the checker's task text")
    return _replace_once(task_text, *_TASK_CHANGE, "task text")


def prompts_sha256(settings: ReviewSettings):
    """Everything that shapes what this reviewer is told, beyond the checker's own prompts."""
    return hashlib.sha256(json.dumps(
        [PROMPTS[settings.checker_version][1], _SYSTEM_CHANGE, _TASK_CHANGE, LAST_STEP, NO_STEPS, CORRECTION,
         BACKGROUND, EMPTY, VERDICT_SCHEMA], sort_keys=True).encode()).hexdigest()


# -- the container command ----------------------------------------------------------------------------------------


_THOUSANDS = '__k() { __n=$1; __o=""; while [ ${#__n} -gt 3 ]; do __r=${__n%???}; __o=",${__n#"$__r"}$__o"; ' \
             '__n=$__r; done; printf "%s%s" "$__n" "$__o"; }'


def container_script(command, settings: ReviewSettings, *, timeout=True):
    """The POSIX shell script that runs one command as the checker does, and prints the checker's observation.

    The command runs in the checker's wrapper (``_shell``: pagers off, standard error folded in) under the
    per-command limit (``timeout``, when the container has it); its output is kept in a temporary file and shown
    as ``_observation`` shows it: ``exit code: N`` (or the time-limit line), ``output:``, then the output, cut to its
    first two fifths and last three fifths past ``output_chars`` with the count of characters cut (bytes here), or
    ``(none)``. The script itself always exits 0, so that Claude Code shows the observation whole.
    """
    limit = settings.output_chars
    head = limit * 2 // 5
    seconds = f"{settings.command_seconds:g}"
    inner = shlex.quote(_shell(command))
    run = f"timeout -k 5 {seconds} /bin/sh -c {inner}" if timeout else f"/bin/sh -c {inner}"
    return "\n".join((
        _THOUSANDS,
        "__t=$(mktemp 2>/dev/null) || __t=''",
        'if [ -z "$__t" ]; then',
        f"  {run}",
        '  echo "(exit code: $?)"; exit 0',
        "fi",
        "__s=$(date +%s)",
        f'{run} > "$__t" 2>&1',
        "__c=$?",
        "__e=$(( $(date +%s) - __s ))",
        f'if [ "$__c" -ge 124 ] && [ "$__e" -ge {int(settings.command_seconds)} ]; then',
        f'  echo "exit code: none (ended after {seconds} seconds; its output so far is below)"',
        "else",
        '  echo "exit code: $__c"',
        "fi",
        'echo "output:"',
        '__n=$(wc -c < "$__t" | tr -d " ")',
        'if [ "$__n" -eq 0 ]; then',
        '  echo "(none)"',
        f'elif [ "$__n" -le {limit} ]; then',
        '  cat "$__t"',
        "else",
        f'  head -c {head} "$__t"',
        f'  printf "\\n[... %s characters cut ...]\\n" "$(__k $((__n - {limit})))"',
        f'  tail -c {limit - head} "$__t"',
        "fi",
        'rm -f "$__t"',
        "exit 0",
    ))


def exec_prefix(container, cwd, user=None, docker="docker"):
    """The fixed start of every command the reviewer may run: ``docker exec -w <cwd> [-u <user>] <container>``."""
    parts = [docker, "exec", "-w", cwd] + (["-u", user] if user else []) + [container, "/bin/sh", "-c"]
    return " ".join(shlex.quote(part) for part in parts)


def allow_rule(container, cwd, user=None, docker="docker"):
    """The one Bash permission rule: the exec prefix, then anything (the hook writes the rest)."""
    return f"Bash({exec_prefix(container, cwd, user, docker)} *)"


def container_command(command, container, cwd, settings: ReviewSettings, *, user=None, timeout=True):
    return exec_prefix(container, cwd, user, settings.docker) + " " + shlex.quote(
        container_script(command, settings, timeout=timeout))


_RUN = re.compile(r'\n__s=\$\(date \+%s\)\n(?:timeout -k 5 \S+ )?/bin/sh -c (.*) > "\$__t" 2>&1\n__c=\$\?\n', re.S)


def command_from_script(script):
    """The reviewer's own command inside a ``container_script`` (for test stand-ins of the container), or None."""
    found = _RUN.search(script or "")
    if found is None:
        return None
    try:
        (inner,) = shlex.split(found.group(1))
    except ValueError:
        return None
    head, tail = _shell("\x00").split("\x00")
    if not (inner.startswith(head) and inner.endswith(tail)):
        return None
    return inner[len(head):len(inner) - len(tail)]


# -- the token and the child's environment ------------------------------------------------------------------------


def read_token(env_file=None, environ=None):
    """The subscription's token: from ``environ`` (this process's environment), else the one line of ``env_file``
    that sets ``CLAUDE_CODE_OAUTH_TOKEN``. No other line is kept; the value is never shown."""
    environ = os.environ if environ is None else environ
    value = environ.get(TOKEN_NAME, "")
    if not value and env_file:
        with open(env_file, encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("export "):
                    stripped = stripped[len("export "):].lstrip()
                if stripped.startswith(TOKEN_NAME + "="):
                    value = stripped.split("=", 1)[1].strip().strip("'\"")
                    break
    if not value or any(char.isspace() for char in value):
        raise ValueError(f"no {TOKEN_NAME} found")
    return value


def child_env(token, *, home, config, settings: ReviewSettings, environ=None):
    """The ``claude`` process's whole environment: an allowlist of this one's, the token, the reviewer's switches."""
    environ = os.environ if environ is None else environ
    env = {name: environ[name] for name in ENV_ALLOW if name in environ}
    env.update({
        TOKEN_NAME: token, "HOME": str(home), "CLAUDE_CONFIG_DIR": str(config), "TMPDIR": str(home),
        # Nothing from the host's own configuration reaches the reviewer's context.
        "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        "ENABLE_CLAUDEAI_MCP_SERVERS": "false", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_TELEMETRY": "1", "DISABLE_AUTOUPDATER": "1",
        # Credentials are stripped from the hook's and the commands' environments.
        "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB": "1",
        # A command runs in the foreground, under the checker's limit (the container's fires first).
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        "BASH_DEFAULT_TIMEOUT_MS": str(settings.tool_timeout_ms), "BASH_MAX_TIMEOUT_MS": str(settings.tool_timeout_ms),
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(settings.output_tokens),
        "CLAUDE_CODE_MAX_RETRIES": str(settings.cli_retries),
    })
    return env


def redact(text, token):
    """``text`` with the token, and anything shaped like an Anthropic key or token, removed."""
    text = str(text)
    if token:
        text = text.replace(token, "[redacted]")
    return re.sub(r"sk-ant-[A-Za-z0-9_-]+", "[redacted]", text)


# -- one review ---------------------------------------------------------------------------------------------------


def claude_settings(container, cwd, user, hook_config, settings: ReviewSettings):
    """The settings file: the one allowed Bash form, the hooks, and nothing of the host's configuration."""
    hook = f"{shlex.quote(settings.python)} -I {shlex.quote(str(HOOK))} {shlex.quote(str(hook_config))}"
    return {
        "permissions": {"allow": [allow_rule(container, cwd, user, settings.docker)],
                        "deny": ["Read", "Edit", "Write", "NotebookEdit", "WebFetch", "WebSearch", "Agent", "Task",
                                 "Glob", "Grep", "mcp__*"]},
        # Bypass mode is on (the user's yes, Sun Nov 15 2026, about 5:20 pm ET): Claude Code refused complex scripts
        # it could not parse (heredocs, rm -rf in /tmp, inline Python) even with the hook's approval, and nobody can
        # approve in -p. The hook stays the boundary: every Bash command is rewritten into the review's own container,
        # every other tool is denied, and --tools Bash leaves no other tool to call.
        # Before a tool runs: a hook that fails for any reason blocks the call (exit 2). After a batch: it counts
        # the step and announces the last; failing there only loses the count.
        # A hook that times out does not block its call, so the limit is far past the hook's few milliseconds.
        # Bypass mode still asks before a 'sh -c' script with rm that Claude Code cannot parse, and a PreToolUse
        # "allow" does not answer it; under --permission-prompts none only a PermissionRequest hook can. The hook
        # approves only a command it rewrote itself; with no answer (or a failure) the call is denied.
        "hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": hook + " || exit 2",
                                                              "timeout": 300}]}],
                  "PermissionRequest": [{"matcher": "*", "hooks": [{"type": "command", "command": hook + " || exit 2",
                                                                     "timeout": 300}]}],
                  "PostToolBatch": [{"hooks": [{"type": "command", "command": hook, "timeout": 300}]}]},
        "disableAllHooks": False, "attribution": {"commit": "", "pr": ""}, "includeGitInstructions": False,
        "disableClaudeAiConnectors": True, "enableAllProjectMcpServers": False, "autoMemoryEnabled": False,
    }


def command_line(settings: ReviewSettings, paths, *, container, cwd, user, session, resume=False, turns=None):
    """The ``claude`` command line, with no secret in it. The prompt goes on standard input."""
    args = [settings.claude, "-p", "--output-format", "json", "--model", settings.model, "--effort", settings.effort,
            "--settings", str(paths["settings"]), "--setting-sources", "", "--strict-mcp-config",
            "--tools", "Bash", "--allowedTools", allow_rule(container, cwd, user, settings.docker),
            "--disallowedTools", "mcp__*", "--permission-prompts", "none", "--permission-mode", "bypassPermissions",
            "--max-turns", str(turns if turns is not None else settings.max_steps),
            "--max-budget-usd", f"{settings.max_usd:g}",
            "--json-schema", json.dumps(VERDICT_SCHEMA, separators=(",", ":"))]
    if resume:
        args += ["--resume", session]
    else:
        args += ["--system-prompt-file", str(paths["system"]), "--session-id", session]
    return args


def _stop_kind(text):
    for kind, pattern in STOPS:
        if pattern.search(text or ""):
            return kind
    return None


def _result_json(stdout):
    """The result object Claude Code printed: the whole output, or its last line that is one JSON object."""
    stdout = (stdout or "").strip()
    for candidate in [stdout, *reversed(stdout.splitlines())]:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("type") == "result":
            return value
    return None


def transcript_counts(config, session, *, tools=False):
    """From the session's transcript: the model calls (assistant messages, by ID) and their tokens; with ``tools``,
    also the tool calls by name (each counted once)."""
    calls, uses = {}, {}
    for path in glob.glob(str(Path(config) / "projects" / "*" / f"{session}.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                message = entry.get("message") if isinstance(entry, dict) else None
                if entry.get("type") != "assistant" or not isinstance(message, dict):
                    continue
                if message.get("model") == "<synthetic>" or not message.get("id"):
                    continue
                calls[message["id"]] = message.get("usage") or {}
                for block in message.get("content") or ():
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        uses[block.get("id")] = block.get("name")

    def total(name):
        return sum(int(usage.get(name) or 0) for usage in calls.values())
    split = {name: total(name) for name in ("input_tokens", "cache_creation_input_tokens",
                                            "cache_read_input_tokens", "output_tokens")}
    if tools:
        named = {}
        for name in uses.values():
            named[name] = named.get(name, 0) + 1
        return len(calls), split, named
    return len(calls), split


# The first line the container wrapper prints for every command it ran (``container_script``): ``exit code: N`` (or
# ``exit code: none ...`` at the time limit), or ``(exit code: N)`` when it could not make its temporary file. A result
# without it did not come from the wrapper: Docker's own error (no daemon, no such container, a lost connection).
WRAPPER_LINE = re.compile(r"^\(?exit code: ", re.M)


def _result_text(content):
    """A tool result's text, as the transcript holds it: a string, or a list of text blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_result_text(block.get("text", "")) if isinstance(block, dict) else str(block) for block in content)
    return ""


def tool_results(config, session):
    """From the session's transcript: each tool call's result as the model got it, by tool-use ID."""
    found = {}
    for path in glob.glob(str(Path(config) / "projects" / "*" / f"{session}.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                message = entry.get("message") if isinstance(entry, dict) else None
                if entry.get("type") != "user" or not isinstance(message, dict) or not isinstance(message.get("content"), list):
                    continue
                for block in message["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        found[block.get("tool_use_id")] = _result_text(block.get("content"))
    return found


def transport_failures(config, session, rewritten):
    """(lost, unanswered): the commands sent into the container whose result the model got lacks the wrapper's exit
    line, and those with no result in the transcript at all."""
    results = tool_results(config, session)
    return ([call for call in rewritten if call in results and not WRAPPER_LINE.search(results[call])],
            [call for call in rewritten if call not in results])


@dataclass
class Review:
    """One review's outcome: the checker-format submission, or why there is none."""

    reply: dict | None
    claude_code: dict = field(default_factory=dict)
    hook: dict = field(default_factory=dict)
    error: str | None = None


class Runner:
    """Runs the reviewer on one container at a time; holds the token in memory only."""

    def __init__(self, token, settings: ReviewSettings | None = None, *, environ=None, clock=time.monotonic):
        if not token:
            raise ValueError("the reviewer needs the subscription's token")
        self._token = token
        self.settings = settings or ReviewSettings()
        self.environ = os.environ if environ is None else environ
        self.clock = clock

    def identity(self, version=None):
        """The reviewer's label for every record: its family, model, route, prompts and limits."""
        return {"family": FAMILY, "model": self.settings.model, "effort": self.settings.effort, "via": VIA,
                # cost_usd in each reply is Claude Code's own estimate at API prices; the subscription bills none.
                "cost_is_notional": True,
                "claude_code_version": version, "checker_version": self.settings.checker_version,
                "checker_prompts_sha256": PROMPTS[self.settings.checker_version][1],
                "prompts_sha256": prompts_sha256(self.settings),
                "settings": {key: value for key, value in asdict(self.settings).items()
                             if key not in ("claude", "docker", "python")}}

    def version(self):
        try:
            done = subprocess.run([self.settings.claude, "--version"], capture_output=True, text=True, timeout=60,
                                  env={"PATH": self.environ.get("PATH", os.defpath)})
            return done.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    def preflight(self, container, cwd, user=None):
        """The container is running and has the tools the command wrapper uses; True if it has ``timeout``."""
        probe = "cd " + shlex.quote(cwd) + " || exit 3; for t in mktemp wc head tail cat date rm tr; do " \
                "command -v $t >/dev/null || echo missing:$t; done; " \
                "if command -v timeout >/dev/null; then echo has:timeout; fi; exit 0"
        args = [self.settings.docker, "exec", "-w", cwd] + (["-u", user] if user else []) + \
               [container, "/bin/sh", "-c", probe]
        try:
            done = subprocess.run(args, capture_output=True, text=True, timeout=60,
                                  env={name: self.environ[name] for name in ENV_ALLOW if name in self.environ}
                                  | {"HOME": self.environ.get("HOME", "/")})
        except (OSError, subprocess.SubprocessError) as error:
            raise ReviewFailed(f"the container could not be reached: {error}") from None
        if done.returncode != 0 or "missing:" in done.stdout:
            raise ReviewFailed("container preflight failed: " + redact((done.stdout + done.stderr)[-500:], self._token))
        return "has:timeout" in done.stdout

    def started_at(self, container):
        """When the container last started (``docker inspect``); ``ReviewFailed`` if it cannot be read."""
        args = [self.settings.docker, "inspect", "-f", "{{.State.StartedAt}}", container]
        try:
            done = subprocess.run(args, capture_output=True, text=True, timeout=60,
                                  env={name: self.environ[name] for name in ENV_ALLOW if name in self.environ}
                                  | {"HOME": self.environ.get("HOME", "/")})
        except (OSError, subprocess.SubprocessError) as error:
            raise ReviewFailed(f"the container could not be inspected: {error}") from None
        if done.returncode != 0:
            raise ReviewFailed("the container could not be inspected: " + redact(done.stderr[-300:], self._token))
        return done.stdout.strip()

    def review(self, container, cwd, task_text, work_dir, *, user=None, preflight=True):
        """Review the work in ``container`` at ``cwd``; ``task_text`` is the checker's task text as the checker gets
        it. Raises ``StopJob`` when the subscription cannot serve, ``ReviewFailed`` on other errors."""
        settings = self.settings
        if not isinstance(cwd, str) or not cwd.startswith("/"):
            raise ValueError("the reviewer works in the task's directory, an absolute path")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(container)):
            raise ValueError("a container is named by its ID or name")
        prompt = reviewer_task(task_text)
        timeout = self.preflight(container, cwd, user) if preflight else True
        # When the container started: a restart during the review (the machine powered off, Docker restarted) means
        # the commands may have run on another state than the one under review.
        began_at = self.started_at(container) if preflight else None
        work = Path(work_dir)
        if work.exists():
            shutil.rmtree(work)
        paths = {name: work / name for name in ("home", "config", "cwd")}
        for path in paths.values():
            path.mkdir(parents=True)
        paths.update(system=work / "system.txt", settings=work / "settings.json", hook=work / "hook.json",
                     state=work / "hook-state.json")
        paths["system"].write_text(reviewer_system_prompt(cwd, settings))
        started = time.time()
        hook_config = {
            "container": container, "cwd": cwd, "user": user, "docker": settings.docker, "repo": str(REPO),
            "state": str(paths["state"]), "max_steps": settings.max_steps, "max_seconds": settings.max_seconds,
            "started": started, "tool_timeout_ms": settings.tool_timeout_ms, "timeout": timeout,
            "settings": {key: getattr(settings, key) for key in ("command_seconds", "output_chars")},
            "last_step": LAST_STEP, "no_steps": NO_STEPS, "background": BACKGROUND, "empty": EMPTY,
            "verdict_tool": VERDICT_TOOL}
        paths["hook"].write_text(json.dumps(hook_config, indent=1))
        paths["state"].write_text(json.dumps({"commands": 0, "refused": 0, "denied": 0, "other_tools": 0,
                                              "batches": 0, "batch_times": [], "last_step": None, "events": []}))
        paths["settings"].write_text(json.dumps(claude_settings(container, cwd, user, paths["hook"], settings),
                                                indent=1))
        env = child_env(self._token, home=paths["home"], config=paths["config"], settings=settings,
                        environ=self.environ)
        session, runs = str(uuid.uuid4()), []
        try:
            ended, result, verdict, corrections = self._loop(paths, env, session, prompt, container, cwd, user, runs)
        except ReviewFailed:
            self._tripwire(paths, session)
            raise
        state = self._tripwire(paths, session)
        # Fail closed: every command the hook sent into the container must be known by its tool-use ID, and each
        # one's result must be the wrapper's. Without the IDs nothing could be checked, so the review is not kept.
        rewritten, commands = state.get("rewritten") or [], int(state["commands"])
        if commands and (len(rewritten) != commands or not all(rewritten)):
            raise ReviewFailed(f"transport_unverified: {commands} commands went into the container, the hook knows "
                               f"{sum(1 for call in rewritten if call)} of them by ID; the review is not kept")
        lost, unanswered = transport_failures(paths["config"], session, rewritten)
        if lost:
            raise ReviewFailed(f"transport_failed: {len(lost)} of {len(rewritten)} commands did not reach the "
                               "container (no wrapper exit line in their result); the review is not kept")
        if unanswered and ended == "verdict":
            raise ReviewFailed(f"transport_unverified: {len(unanswered)} of {len(rewritten)} commands have no result "
                               "in the transcript of a review that gave a verdict; the review is not kept")
        if began_at is not None and self.started_at(container) != began_at:
            raise ReviewFailed("container_restarted: the container started again during the review; it is not kept")
        if ended != "verdict" and state.get("last_step") and ended in ("steps", "format_errors"):
            ended = state["last_step"]
        calls, split = transcript_counts(paths["config"], session)
        result = result or {}
        submission = {"schema": SCHEMA, **(verdict or {"verdict": None, "confidence": None}), "ended": ended,
                      "steps": calls, "commands": int(state["commands"]), "refused": int(state["refused"]),
                      "cost_usd": round(float(result.get("total_cost_usd") or 0.0), 6),
                      "tokens": {"input": split["input_tokens"] + split["cache_creation_input_tokens"]
                                 + split["cache_read_input_tokens"], "output": split["output_tokens"]}}
        reply = parse_submission(submission)
        claude_code = {
            "session_id": session, "runs": runs, "corrections": corrections, "subtype": result.get("subtype"),
            "is_error": result.get("is_error"), "num_turns": result.get("num_turns"), "model_calls": calls,
            "duration_ms": result.get("duration_ms"), "duration_api_ms": result.get("duration_api_ms"),
            "total_cost_usd": result.get("total_cost_usd"), "tokens": split,
            "model_usage": result.get("modelUsage"), "permission_denials": len(result.get("permission_denials") or ()),
            "container_timeout": timeout, "started_unix": round(started, 3), "seconds": round(time.time() - started, 3)}
        hook = {key: state.get(key) for key in ("commands", "refused", "denied", "other_tools", "batches", "last_step",
                                                "token_in_env")}
        hook["events"] = state.get("events", [])[-60:]
        return Review(reply=reply, claude_code=json.loads(redact(json.dumps(claude_code), self._token)),
                      hook=json.loads(redact(json.dumps(hook), self._token)))

    def _tripwire(self, paths, session):
        """The hook's state, once every tool call in the transcript is known to have passed through the hook; else
        ``StopJob``: the confinement did not hold, and nothing more may run."""
        state = json.loads(paths["state"].read_text())
        _, _, named = transcript_counts(paths["config"], session, tools=True)
        seen = int(state["commands"]) + int(state["refused"]) + int(state["denied"])
        others = sum(count for name, count in named.items() if name not in ("Bash", VERDICT_TOOL))
        if named.get("Bash", 0) > seen or others > int(state.get("other_tools", 0)):
            raise StopJob("hook_bypassed", f"{named.get('Bash', 0)} Bash calls, {seen} seen by the hook; "
                                           f"{others} other tool calls, {state.get('other_tools', 0)} seen")
        return state

    def _loop(self, paths, env, session, prompt, container, cwd, user, runs):
        """The ``claude`` runs of one review: the first, then one per verdict the checker would not accept, sent back
        with its correction. Returns how the review ended, the last result object, the verdict, the corrections."""
        settings = self.settings
        began = self.clock()
        result, ended, corrections, verdict, problem = None, None, 0, None, None
        while True:
            used, _ = transcript_counts(paths["config"], session)
            turns = settings.max_steps - used
            left = settings.max_seconds + settings.kill_grace_seconds - (self.clock() - began)
            if turns < 1 or left <= 0:
                ended = "steps" if turns < 1 else "seconds"
                break
            args = command_line(settings, paths, container=container, cwd=cwd, user=user, session=session,
                                resume=bool(runs), turns=turns)
            done = self._run(args, prompt if not runs else CORRECTION.format(error=problem), env, paths["cwd"], left)
            runs.append({key: done[key] for key in ("exit_code", "timed_out", "seconds")})
            result = done["result"] or result
            if done["timed_out"]:
                ended = "seconds"
                break
            failed = done["result"] is None or done["result"].get("is_error")
            if failed and (done["result"] or {}).get("subtype") not in SUBTYPE_ENDING:
                text = redact(json.dumps(done["result"]) if done["result"] else done["stdout"][-2000:], self._token)
                text += "\n" + redact(done["stderr"][-2000:], self._token)
                kind = _stop_kind(text)
                status = (done["result"] or {}).get("api_error_status")
                if kind is None and status in (401, 403, 429):
                    kind = "rate_limit" if status == 429 else "auth"
                if kind is not None:
                    raise StopJob(kind, text.strip()[-1500:])
                raise ReviewFailed(f"claude ended with exit code {done['exit_code']}: {text.strip()[-1500:]}")
            subtype = done["result"].get("subtype")
            if subtype in SUBTYPE_ENDING:
                ended = SUBTYPE_ENDING[subtype]
                break
            found = done["result"].get("structured_output")
            if not isinstance(found, dict):
                ended = "format_errors"
                break
            try:
                verdict = validate_verdict(found)
                ended = "verdict"
                break
            except InvalidVerdict as error:
                problem = str(error)
                corrections += 1
                if corrections > settings.max_corrections:
                    ended = "format_errors"
                    break
        return ended, result, verdict, corrections

    def _run(self, args, prompt, env, cwd, seconds):
        """One ``claude`` process; stopped (SIGINT, then SIGTERM, then SIGKILL to its group) past ``seconds``."""
        began = self.clock()
        process = subprocess.Popen(args, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True)
        timed_out = False
        try:
            stdout, stderr = process.communicate(prompt, timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out, stdout, stderr = True, "", ""
            for sig, wait in ((signal.SIGINT, 15), (signal.SIGTERM, 10), (signal.SIGKILL, 10)):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, sig)
                try:
                    stdout, stderr = process.communicate(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        return {"exit_code": process.returncode, "timed_out": timed_out, "stdout": stdout or "",
                "stderr": stderr or "", "result": _result_json(stdout), "seconds": round(self.clock() - began, 3)}
