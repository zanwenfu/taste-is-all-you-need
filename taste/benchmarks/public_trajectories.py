"""Published mini-swe-agent runs, read back as the exact commands they executed.

SWE-bench's bash-only leaderboard publishes one trajectory file per run of
mini-swe-agent (Lieret and Jimenez, 2025). A replay re-executes a run's
commands in the benchmark's image, so the one thing that must be exactly right
is the command list: every command the scaffold executed, in order, and
nothing it did not. A reply the scaffold rejected (a *format error*) ran
nothing; a command that timed out ran and was killed; the command whose output
began with the submission sentinel ended the run, and nothing after it ran.

The scaffold wrote three shapes over its life. Each is read with the
scaffold's own rule for that shape, never a looser guess:

- **Text actions** -- ``trajectory_format`` ``"mini-swe-agent-1"`` (v1.x,
  August to December 2025) or undeclared (micro- and mini-swe-agent before
  1 August 2025). Messages are ``{role, content}``. The command is the single
  regex match in the assistant's text (stripped); the next user message is
  its result: an observation (``<returncode>`` / ``<output>``, or
  ``<output_head>`` / ``<output_tail>`` when the output reached 10,000
  characters), a timeout notice (no return code), or -- as the last message
  of a submitted run -- the submission itself. Zero or several matches is a
  format error: the next message says so and nothing ran. The regex changed
  three times and older files do not record theirs, so it is *proved* from
  the record: every candidate the scaffold used in that era is tried, and a
  reading counts only if it explains every turn (one match exactly where the
  scaffold executed, otherwise none or several). Two readings that disagree
  on a command make the run unreadable.
- **Tool calls** -- ``"mini-swe-agent-1.1"`` (v2.x, 2026). The assistant
  message carries the executed list in ``extra.actions`` (checked against its
  ``tool_calls``), possibly several per turn; one ``tool`` message per call
  follows, paired by ``tool_call_id``, with the full output in
  ``extra.raw_output`` and the exit status in ``extra.returncode``. A format
  error leaves no assistant message, only a user message with
  ``extra.interrupt_type == "FormatError"``. A run ends with a ``role: exit``
  message; a submitting turn has no results (the sentinel interrupts it).
- **The Responses API** -- also ``"mini-swe-agent-1.1"``, for models served
  that way. The turn is a response object (``object: "response"``) whose
  ``function_call`` items are the actions; results are
  ``function_call_output`` items paired by ``call_id``.

Anything else raises ValueError: a SWE-agent ``.traj`` (the leaderboard's
"mini-v0.0.0" entries are SWE-agent 1.0.1 runs, with a persistent shell and a
``submit`` tool), an unknown format, a record two readings explain
differently, a command that may have run but has no recorded result. A run
dropped loudly costs one data point; a run replayed with the wrong commands
costs the study its meaning.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

#: The sentinel of today's protocol, and the two it replaced: ``MICRO_`` until
#: the rename to mini-swe-agent (23 July 2025), ``MINI_`` until v2 dropped it.
CURRENT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
OLD_SENTINELS = ("MINI_SWE_AGENT_FINAL_OUTPUT", "MICRO_SWE_AGENT_FINAL_OUTPUT")
SENTINELS = (CURRENT_SENTINEL, *OLD_SENTINELS)

TEXT_FORMAT = "mini-swe-agent-1"
TOOL_FORMAT = "mini-swe-agent-1.1"

# The text-action regexes, oldest first. Files record theirs only from v1.16
# (``config.agent.action_regex``); before that the record has to prove which.
#: 17-22 July 2025: any fence, and a bare or trailing fence counts as an action.
REGEX_LOOSE = r"```(?:bash)?\n?(.*?)(?=\n```|```)"
#: 22 July - 16 September 2025: only ```bash blocks.
REGEX_BASH = r"```bash\n(.*?)\n```"
#: From 16 September 2025 (whitespace after the language tag); v1.16's default.
REGEX_BASH_WS = r"```bash\s*\n(.*?)\n```"

# Every format-error template the scaffold shipped starts with this.
_FORMAT_ERROR_PREFIX = "Please always provide EXACTLY ONE action"

_OBSERVATION = re.compile(r"\A<returncode>(?P<rc>-?\d+)</returncode>\n<output>\n(?P<output>.*)</output>\Z", re.DOTALL)
_OBSERVATION_TRUNCATED = re.compile(
    r"\A<returncode>(?P<rc>-?\d+)</returncode>\n<warning>\n.*?\n</warning>"
    r"<output_head>\n(?P<head>.*?)\n</output_head>\n<elided_chars>\n\d+ characters elided\n</elided_chars>\n"
    r"<output_tail>\n(?P<tail>.*)\n</output_tail>\Z",
    re.DOTALL,
)
_PLEASE = r"Please try another command and make sure to avoid those requiring interactive input\."
# The timeout notice, newest first. Dec 2025 (#639) truncates the partial
# output; 22 July (#236) names the command; 18 July (#211) printed the whole
# action dict instead; before that the notice carried nothing.
_TIMEOUTS = (
    re.compile(
        r"\AThe last command <command>(?P<command>.*?)</command> timed out and has been killed\.\n"
        r"The output of the command was:\n<output>\n(?P<output>.*)\n</output>\n" + _PLEASE + r"\Z", re.DOTALL),
    re.compile(
        r"\AThe last command <command>(?P<command>.*?)</command> timed out and has been killed\.\n"
        r"The output of the command was:\n<warning>Output was too long and has been truncated\.</warning>\n"
        r"<output_head>\n(?P<head>.*?)\n</output_head>\n<elided_chars>\d+ characters elided</elided_chars>\n"
        r"<output_tail>\n(?P<tail>.*)\n</output_tail>\n" + _PLEASE + r"\Z", re.DOTALL),
    re.compile(
        r"\AThe last command <command>(?P<command>.*?)</command> timed out and has been killed\.\n"
        r"The output of the command was:\n <output>\n(?P<output>.*)\n</output>\n" + _PLEASE + r"\Z", re.DOTALL),
    re.compile(
        r"\AThe last command (?!<command>).*? timed out and has been killed\.\n"
        r"The output of the command was:\n(?P<output>.*)\n\n" + _PLEASE + r"\Z", re.DOTALL),
    re.compile(r"\AThe command timed out\. Please change your command and make sure it doesn't require input\.\Z"),
)


@dataclass(frozen=True)
class RecordedCommand:
    """One command the scaffold executed, and what the trajectory recorded for it.

    ``output`` is the command's output, unwrapped from the observation
    template. When ``output_truncated`` is set the model was shown only the
    first and last 5,000 characters: text-action files kept only those
    (``output`` is head + tail), tool-call files kept everything (``output``
    is ``raw_output``, complete), so ``out[:5000] == output[:5000] and
    out[-5000:] == output[-5000:]`` is the comparison that holds for both.
    For the submitting command ``output`` is the submission the scaffold
    extracted -- its output minus the sentinel line.

    ``returncode`` is None when the record holds none: a timeout, an
    execution error, the submitting command of a text-action run (v1 accepted
    the sentinel whatever the exit status), a command that ran unrecorded in
    the same turn as the submission. A tool-call submission records 0: v2
    accepts the sentinel only on a zero exit.
    """

    index: int  # 0-based execution order
    command: str  # exactly as executed by the scaffold
    returncode: int | None
    output: str
    output_truncated: bool  # the scaffold elided part of the output it showed
    timed_out: bool


@dataclass(frozen=True)
class PublicTrajectory:
    """One published run: its commands in execution order and the settings they ran under.

    The settings come from ``info.config`` when the file recorded it (from
    v1.9); otherwise ``timeout``/``step_limit``/``cost_limit`` are None and
    ``cwd``/``env``/``environment_class``/``image``/``interpreter`` are empty
    -- not recorded, not invented. (Those runs used the stock SWE-bench
    config of their day: docker, ``/testbed``, 60 s, the PAGER/LESS/TQDM
    variables, ``bash -lc``; a replay should fall back to the same.)
    """

    instance_id: str
    mini_version: str
    trajectory_format: str  # "" when the file declared none (before 1 August 2025)
    exit_status: str
    submission: str
    commands: tuple[RecordedCommand, ...]
    submit_index: int | None  # the command whose output carried the sentinel; None if never submitted
    sentinel: str | None  # the sentinel word that run used
    unexecuted_turns: int  # model turns that executed nothing (format errors)
    timeout: int | None  # environment timeout from the config
    env: dict[str, str]
    cwd: str
    environment_class: str
    step_limit: int | None
    cost_limit: float | None
    api_calls: int | None
    cost_usd: float | None
    image: str = ""
    #: The argv the command was appended to: v1's DockerEnvironment hard-codes
    #: ``bash -lc`` (a login shell, so the image's conda env is active); v2
    #: records it, and its SWE-bench config ran ``bash -c`` -- without
    #: ``BASH_ENV`` until v2.4, so outside the testbed env.
    interpreter: tuple[str, ...] = field(default=())


def parse_trajectory(raw: dict, *, mini_version: str = "") -> PublicTrajectory:
    """Read one published ``.traj.json``. ``mini_version`` (the leaderboard's
    label) is used only when the file records no version of its own."""
    if not isinstance(raw, dict):
        raise ValueError("a trajectory is a JSON object")
    if "trajectory" in raw and "messages" not in raw:
        raise ValueError(
            "SWE-agent trajectory (the leaderboard's mini-v0.0.0 entries ran SWE-agent 1.x): a persistent shell "
            "and a `submit` tool, not mini-swe-agent's one-subshell-per-command protocol"
        )
    info, messages = raw.get("info"), raw.get("messages")
    if not isinstance(info, dict) or not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("not a mini-swe-agent trajectory: needs 'info' and at least the system and task messages")
    if not all(isinstance(m, dict) for m in messages):
        raise ValueError("every message must be a JSON object")
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise ValueError("the first two messages must be the system prompt and the task")
    declared = raw.get("trajectory_format")
    try:
        if declared in (None, TEXT_FORMAT):
            commands, submit_index, unexecuted = _read_text_actions(info, messages)
        elif declared == TOOL_FORMAT:
            commands, submit_index, unexecuted = _read_tool_actions(info, messages)
        else:
            raise ValueError(f"unknown trajectory_format {declared!r}")
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f"malformed {declared or 'undeclared'} trajectory: {type(exc).__name__}: {exc}") from exc

    sentinel = None
    if submit_index is not None:
        sentinel = CURRENT_SENTINEL if declared == TOOL_FORMAT else _first_sentinel(commands[submit_index].command)
    config = _dict(info.get("config"))
    environment, agent = _dict(config.get("environment")), _dict(config.get("agent"))
    environment_class = str(config.get("environment_type") or environment.get("environment_class") or "")
    if "interpreter" in environment:
        interpreter = tuple(str(part) for part in environment["interpreter"])
    elif declared != TOOL_FORMAT and environment_class.endswith(".DockerEnvironment"):
        interpreter = ("bash", "-lc")
    else:
        interpreter = ()
    stats = _dict(info.get("model_stats"))
    return PublicTrajectory(
        instance_id=str(raw.get("instance_id") or ""),
        mini_version=str(info.get("mini_version") or mini_version or ""),
        trajectory_format=str(declared or ""),
        exit_status=str(info.get("exit_status") or ""),
        submission=str(info.get("submission") or ""),
        commands=tuple(commands),
        submit_index=submit_index,
        sentinel=sentinel,
        unexecuted_turns=unexecuted,
        timeout=_number(environment.get("timeout"), int),
        env={str(k): str(v) for k, v in _dict(environment.get("env")).items()},
        cwd=str(environment.get("cwd") or ""),
        environment_class=environment_class,
        step_limit=_number(agent.get("step_limit"), int),
        cost_limit=_number(agent.get("cost_limit"), float),
        api_calls=_number(stats.get("api_calls"), int),
        cost_usd=_number(stats.get("instance_cost"), float),
        image=str(environment.get("image") or ""),
        interpreter=interpreter,
    )


def replay_commands(traj: PublicTrajectory) -> list[str]:
    """The commands to re-execute, in order, with the old sentinel rewritten to
    COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT so a replay environment that knows only
    the current protocol ends where the original did. Nothing after submit_index."""
    commands = [c.command for c in traj.commands]
    if traj.submit_index is None:
        return commands
    if traj.sentinel is None:
        raise ValueError(
            "the submitting command names no known sentinel, so it cannot be rewritten for the current protocol"
        )
    commands = commands[: traj.submit_index + 1]
    if traj.sentinel != CURRENT_SENTINEL:
        # Only the submitting command: an earlier command that merely mentions the
        # old word (a grep, a failed attempt) must not start ending the run.
        commands[-1] = commands[-1].replace(traj.sentinel, CURRENT_SENTINEL)
    return commands


def early_sentinel_indices(traj: PublicTrajectory) -> tuple[int, ...]:
    """Commands before the submission that name a sentinel.

    Each one did not end the original run (it failed, timed out, or printed
    the word somewhere other than its first line), but it could end a replay
    whose environment behaves differently -- a submit that timed out at 60 s
    may finish in time. A replay that stops at one of these did not
    reproduce the run.
    """
    end = len(traj.commands) if traj.submit_index is None else traj.submit_index
    return tuple(c.index for c in traj.commands[:end] if any(s in c.command for s in SENTINELS))


# -- text actions ---------------------------------------------------------------------------------


def _read_text_actions(info: dict, messages: list[dict]) -> tuple[list[RecordedCommand], int | None, int]:
    exit_status = str(info.get("exit_status") or "")
    submitted = exit_status == "Submitted"
    agent = _dict(_dict(info.get("config")).get("agent"))
    format_prefix = _literal_prefix(agent.get("format_error_template")) or _FORMAT_ERROR_PREFIX

    # The loop is strict: every reply is followed by exactly one user message
    # (its result, its format error, or the submission). The only other user
    # message is LimitsExceeded's empty one, raised before a model call, so it
    # follows a result rather than a reply.
    turns: list[tuple[int, str, str]] = []
    i = 2
    while i < len(messages):
        role = messages[i].get("role")
        if role == "assistant":
            if i + 1 == len(messages):
                raise ValueError(
                    f"message {i}: the run ends on a reply with no recorded result; whether its command ran is unknown"
                )
            if messages[i + 1].get("role") != "user":
                raise ValueError(f"message {i + 1}: a reply must be followed by a user message")
            turns.append((i, _text(messages[i].get("content")), _text(messages[i + 1].get("content"))))
            i += 2
        elif role == "user" and i == len(messages) - 1 and not submitted:
            i += 1
        else:
            raise ValueError(f"message {i}: unexpected {role!r} message in a text-action trajectory")

    # What each result says happened, independent of any regex.
    results: list[tuple[str, Any]] = []
    for n, (i, _reply, result) in enumerate(turns):
        if submitted and n == len(turns) - 1:
            if result != str(info.get("submission") or ""):
                raise ValueError(f"message {i + 1}: the final message is not the recorded submission")
            results.append(("submit", result))
        elif (observed := _text_observation(result)) is not None:
            results.append(("observation", observed))
        elif (timed_out := _text_timeout(result)) is not None:
            results.append(("timeout", timed_out))
        elif result.startswith((format_prefix, _FORMAT_ERROR_PREFIX)):
            results.append(("format_error", None))
        elif result.startswith(("The last command", "The command timed out", "<returncode>")):
            raise ValueError(f"message {i + 1}: a timeout or observation in a template this reader does not know")
        else:
            raise ValueError(f"message {i + 1}: unrecognised result {result[:80]!r}")

    readings: dict[str, list[str | None]] = {}
    first_misfit: dict[str, int] = {}
    for regex in _text_regexes(info, messages):
        reading: list[str | None] = []
        for (i, reply, _result), (kind, payload) in zip(turns, results, strict=True):
            matches = [m.strip() for m in re.findall(regex, reply, re.DOTALL)]
            ran = len(matches) == 1
            fits = ran == (kind != "format_error")
            if fits and kind == "timeout" and payload[0] is not None:
                fits = payload[0] == matches[0]  # the notice names the command it killed
            if not fits:
                first_misfit[regex] = i
                break
            reading.append(matches[0] if ran else None)
        else:
            readings[regex] = reading
    if not readings:
        detail = ", ".join(f"{rx!r} fails at message {i}" for rx, i in first_misfit.items())
        raise ValueError(f"no action regex the scaffold used explains this record ({detail})")
    distinct = {tuple(r) for r in readings.values()}
    if len(distinct) > 1:
        raise ValueError(f"action regexes {sorted(readings)} all explain this record but extract different commands")
    reading = next(iter(readings.values()))

    commands: list[RecordedCommand] = []
    submit_index = None
    for command, (kind, payload) in zip(reading, results, strict=True):
        if command is None:
            continue
        n = len(commands)
        if kind == "observation":
            returncode, output, truncated = payload
            commands.append(RecordedCommand(n, command, returncode, output, truncated, False))
        elif kind == "timeout":
            _named, output, truncated = payload
            commands.append(RecordedCommand(n, command, None, output, truncated, True))
        else:
            commands.append(RecordedCommand(n, command, None, payload, False, False))
            submit_index = n
    return commands, submit_index, sum(1 for c in reading if c is None)


def _text_regexes(info: dict, messages: list[dict]) -> tuple[str, ...]:
    configured = _dict(_dict(info.get("config")).get("agent")).get("action_regex")
    if configured:
        return (str(configured),)
    if info.get("mini_version"):  # recorded from 6 August 2025: after the loose regex, around the \s* change
        return (REGEX_BASH, REGEX_BASH_WS)
    # Undeclared: July 2025. The MINI_ sentinel dates a run after the rename,
    # by when only ```bash blocks counted; MICRO_ allows either.
    prompt = _text(messages[0].get("content")) + _text(messages[1].get("content"))
    if "MINI_SWE_AGENT_FINAL_OUTPUT" in prompt:
        return (REGEX_BASH,)
    return (REGEX_LOOSE, REGEX_BASH)


def _text_observation(text: str) -> tuple[int, str, bool] | None:
    if m := _OBSERVATION.match(text):
        return int(m["rc"]), m["output"], False
    if m := _OBSERVATION_TRUNCATED.match(text):
        return int(m["rc"]), m["head"] + m["tail"], True
    return None


def _text_timeout(text: str) -> tuple[str | None, str, bool] | None:
    """(the command the notice names, if any; the partial output; truncated)."""
    for pattern in _TIMEOUTS:
        if m := pattern.match(text):
            groups = m.groupdict()
            if groups.get("head") is not None:
                return groups.get("command"), groups["head"] + groups["tail"], True
            return groups.get("command"), groups.get("output") or "", False
    return None


# -- tool calls -----------------------------------------------------------------------------------


def _read_tool_actions(info: dict, messages: list[dict]) -> tuple[list[RecordedCommand], int | None, int]:
    exit_status = str(info.get("exit_status") or "")
    commands: list[RecordedCommand] = []
    submit_index, unexecuted = None, 0
    i = 2
    while i < len(messages):
        message = messages[i]
        extra = _dict(message.get("extra"))
        if message.get("role") == "assistant" or message.get("object") == "response":
            actions = _actions(message, i, info)
            results = []
            j = i + 1
            while j < len(messages) and len(results) < len(actions) and _is_result(messages[j]):
                results.append(messages[j])
                j += 1
            if results:
                if len(results) != len(actions):
                    raise ValueError(f"message {i}: {len(actions)} actions but {len(results)} results")
                for (command, call_id), result in zip(actions, results, strict=True):
                    result_id = result.get("tool_call_id") or result.get("call_id")
                    if result_id != call_id:
                        raise ValueError(f"message {i}: result tool_call_id {result_id!r} does not pair with {call_id!r}")
                    recorded = _tool_result(result)
                    if recorded is not None:  # None: the scaffold's own "action was not executed" padding
                        commands.append(RecordedCommand(len(commands), command, *recorded))
                i = j
                continue
            # A turn without results: the sentinel interrupted it, or the run crashed in it.
            end = messages[j] if j < len(messages) else {}
            ended = _dict(end.get("extra")).get("exit_status")
            if end.get("role") != "exit" or j != len(messages) - 1 or ended != "Submitted" or exit_status != "Submitted":
                raise ValueError(
                    f"message {i}: the run ended ({ended or exit_status or 'no exit message'}) right after this turn, "
                    "which has no recorded result; whether its commands ran is unknown"
                )
            k = _submitting_action(actions, i)
            # The environment runs a turn's calls in order until one submits:
            # the ones before it ran (unrecorded), the ones after never did.
            for command, _call_id in actions[:k]:
                commands.append(RecordedCommand(len(commands), command, None, "", False, False))
            commands.append(RecordedCommand(len(commands), actions[k][0], 0, _text(end.get("content")), False, False))
            submit_index = len(commands) - 1
            i = j
        elif extra.get("interrupt_type") == "FormatError":
            unexecuted += 1
            i += 1
        elif message.get("role") == "exit":
            if i != len(messages) - 1:
                raise ValueError(f"message {i}: the exit message is not the last")
            if exit_status == "Submitted" and _text(message.get("content")) != str(info.get("submission") or ""):
                raise ValueError(f"message {i}: the exit message is not the recorded submission")
            i += 1
        elif _is_result(message):
            raise ValueError(f"message {i}: a result with no turn before it")
        else:
            raise ValueError(f"message {i}: unexpected message ({message.get('role') or message.get('type')!r})")
    if exit_status == "Submitted" and submit_index is None:
        raise ValueError("the run is recorded as submitted but no turn ended in the submission")
    return commands, submit_index, unexecuted


def _actions(message: dict, i: int, info: dict) -> list[tuple[str, str | None]]:
    """The executed list (``extra.actions``), checked against what the model emitted."""
    recorded = _dict(message.get("extra")).get("actions")
    if not isinstance(recorded, list) or not recorded:
        raise ValueError(f"message {i}: a model turn without recorded actions")
    actions = []
    for action in recorded:
        if not isinstance(action, dict) or not isinstance(action.get("command"), str):
            raise ValueError(f"message {i}: an action without a command")
        actions.append((action["command"], action.get("tool_call_id")))
    if message.get("object") == "response":
        emitted = [(_command(item["arguments"]), item.get("call_id") or item.get("id"))
                   for item in message.get("output") or [] if item.get("type") == "function_call"]
    elif message.get("tool_calls"):
        emitted = [(_command(call["function"]["arguments"]), call.get("id")) for call in message["tool_calls"]]
    else:
        # The text-based model class in the v2 layout: actions are the regex matches of the reply.
        regex = _dict(_dict(info.get("config")).get("model")).get("action_regex")
        if not regex or any(call_id is not None for _command_text, call_id in actions):
            raise ValueError(f"message {i}: recorded actions but neither tool calls nor an action regex")
        emitted = [(m.strip(), None) for m in re.findall(str(regex), _text(message.get("content")), re.DOTALL)]
    if emitted != actions:
        raise ValueError(f"message {i}: the recorded actions are not the tool calls the model made")
    return actions


def _command(arguments: Any) -> str:
    args = json.loads(arguments) if isinstance(arguments, str) else arguments
    if not isinstance(args, dict) or not isinstance(args.get("command"), str):
        raise ValueError("a bash tool call without a command")
    return args["command"]


def _is_result(message: dict) -> bool:
    if message.get("role") == "tool" or message.get("type") == "function_call_output":
        return True
    return message.get("role") == "user" and "raw_output" in _dict(message.get("extra"))


def _tool_result(message: dict) -> tuple[int | None, str, bool, bool] | None:
    extra = _dict(message.get("extra"))
    exception = str(extra.get("exception_info") or "")
    if exception == "action was not executed":
        return None
    if "raw_output" not in extra:
        raise ValueError("a tool result without raw_output")
    shown = _text(message.get("output") if message.get("type") == "function_call_output" else message.get("content"))
    timed_out = extra.get("exception_type") in ("TimeoutExpired", "TimeoutError") or bool(
        re.search(r"timed out after [\d.]+ seconds", exception)
    )
    # On an execution error the scaffold writes -1 as a placeholder, not an exit status.
    returncode = None if exception else _number(extra.get("returncode"), int)
    if not exception and returncode is None:
        raise ValueError("a tool result without a return code")
    # The template's elided branch is the only one that ends in </output_tail>
    # (an output that merely contains that text still ends in </output>).
    truncated = shown.endswith("</output_tail>")
    return returncode, str(extra.get("raw_output") or ""), truncated, timed_out


def _submitting_action(actions: list[tuple[str, str | None]], i: int) -> int:
    if len(actions) == 1:
        return 0
    named = [k for k, (command, _call_id) in enumerate(actions) if CURRENT_SENTINEL in command]
    if len(named) != 1:
        raise ValueError(f"message {i}: cannot tell which of {len(actions)} calls in the submitting turn submitted")
    return named[0]


# -- shared helpers -------------------------------------------------------------------------------


def _text(content: Any) -> str:
    """Message content as one string. Cache-control rewrites (Anthropic, v1)
    and Responses-API inputs store it as a list of text blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") not in ("text", "input_text", "output_text"):
                raise ValueError(f"a content block this reader does not know: {str(block)[:80]!r}")
            parts.append(str(block.get("text") or ""))
        return "".join(parts)
    raise ValueError(f"message content of type {type(content).__name__}")


def _literal_prefix(template: Any) -> str:
    """The fixed text a Jinja template renders before its first tag."""
    if not isinstance(template, str):
        return ""
    return re.split(r"\{[{%#]", template, maxsplit=1)[0]


def _first_sentinel(command: str) -> str | None:
    found = sorted((command.find(word), word) for word in SENTINELS if word in command)
    return found[0][1] if found else None


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _number(value: Any, kind: type) -> Any:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if kind is int:
        return int(number) if number == int(number) else None
    return number
