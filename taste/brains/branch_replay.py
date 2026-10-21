"""A hosted agent's recorded run, branched at step k: the prefix from the record, then live.

A replay script (``scripts/replay_script.py`` writes one from a finished
trial) holds a run's steps in order. Step i is the agent's i-th model call and
the commands it ran before its next one: the request's digest, the reply as
the provider gave it, and for each command the text the agent wrote, the exact
text the task container ran, its directory, time limit, output, exit code and
whether it timed out. Steps are numbered as the study numbers them everywhere,
and as ``taste.agents.trajectory_reader.steps_from_trajectory`` reads them from
a trial's record: one per model reply the agent received, in order, replayed
ones included; a reply the run's end cut off and a call given up as lost are
not steps. "Step k" is the same step to the map, the reader and a branch.

A branch from step k brings back the state at step k, then lets the agent go on:

- The task's files. ``rebuild`` runs the recorded commands of steps 1..k again,
  in order, in a fresh task container, through the worker's terminal like the
  agent's own commands, and compares each with the record (the rule below).
  ``restore`` puts back a checkpoint that a rebuild took (the trial owner does
  that before the agent starts) and runs nothing.
- The agent's context. ``ReplayHost`` answers the agent's first k model calls
  with the recorded replies and its commands with the recorded outputs, asking
  no model and running nothing. Each request must equal the recorded one, and
  each command the recorded one; otherwise the branch is unfaithful and the
  agent is stopped there.
- Then, live: the agent's next model call is a fresh sample. With live off it
  is stopped instead, so that the task's verifier grades the files of step k.

With the replay off, only the files of step k are brought back, and the
trial's own agent starts fresh with its own task: a checker given a run's
final files, without the coding agent's context.

Commands an agent runs before its first model call (SWE-agent starts its
shell session, sets variables and installs its tools) are the script's
setup: a rebuild runs them first, and a branch answers them from the record
before step 1, as it answers the steps.

A change at step k: ``append`` shows the agent step k's output followed by a
note; ``reject``, at the submission step, shows it a rejection text with exit
code 1 instead, so that the agent's submission protocol does not end its run
and it works on in place. Which of step k's commands is changed is the
agent's (``taste.agents.protocols``): mini-swe-agent's one command, and
SWE-agent's action, not its state reads after it. A rejection keeps the
agent from what its submission would have run next (SWE-agent reads its
patch); those recorded commands are left out when it asks its model again,
and the record says how many. A branch's own record exports as the run the agent
had, so it can be branched again (another round); a changed step keeps what
the container printed beside what the agent was shown, and a rebuild checks
the former.

Rebuild fidelity rule. A rebuilt command diverges from the record when its
exit code differs, when one timed out and the other did not, or when the
similarity of their outputs is below OUTPUT_FLOOR. A command the record shows
cut off by its time limit is not compared: where a limit cuts a command
depends on the host's speed, so neither its exit code nor its output says
anything about the files (its similarity is still recorded), and the commands
after it, compared as usual, show whether the files differ. Nor is which
lines a cut-off directory walk printed: a command that only reads and cuts a
walk's output to its first or last lines (``grep -r``, ``rg``, ``find`` or
``fd``, through filters that keep the order, into ``head`` or ``tail``;
``host_ordered``) prints the lines the host lists first, and hosts, even two
containers of one host, list a directory in different orders. Such a command
that would diverge only by its output is not divergent when its exit code,
its time-out and its number of normalized lines match the record's; the row
says ``host_ordered``, and the commands after it, compared as usual, show
whether the files differ. A command the
record shows finishing is given REBUILD_TIME_FACTOR times its limit to finish
again (within the terminal's maximum); one cut off keeps its own limit, so
that its effects are cut off as they were. Similarity is the Dice
coefficient of the two outputs' line multisets, 2|A&B| / (|A| + |B|), after
each line is normalized (runs of seven or more hexadecimal digits become '#',
runs of digits become '0', whitespace is collapsed, empty lines are dropped);
two empty outputs are identical. Timings, sizes, process IDs and hashes thus
do not count, a changed line among many counts a little, and a different
one-line answer counts fully. A rebuild with more divergent commands than its
tolerance (zero unless set) is unfaithful: it stops there, and the agent
replays its context but does not go live.

Replayed replies cost nothing: the agent is told each one cost $0, so its own
cost limit counts only live calls, and Taste's model journal never sees them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from posixpath import isabs

from taste.agents import HostedStop, ModelReply, ShellResult, agentrewind
from taste.agents.protocols import MINI_SENTINEL, protocol
from taste.pricing import request_token_bound
from taste.providers._openai import _NATIVE

SCRIPT_SCHEMA = "taste.branch/ReplayScript/1"
MODES = ("rebuild", "restore")
# rewind: step k's call is made again, live, with the rewind arm's notebook after the messages it restores.
OVERRIDES = ("append", "reject", "rewind")
# mini-swe-agent's submission protocol (TasteEnvironment._check_finished): the
# output's first line is this, and the command exited 0.
SENTINEL = MINI_SENTINEL
REJECT_EXIT = 1
OUTPUT_FLOOR = 0.5
# How much longer than its recorded limit a rebuilt command that the record
# shows finishing may take: a host slower than the recording's must not cut it off.
REBUILD_TIME_FACTOR = 2.0
# A note travels in the trial's policy, beside the task, inside the goal's input limit.
MAX_NOTE_CHARS = 16_384
MAX_SCRIPT_BYTES = 256 * 1024 * 1024
# What one replayed or rebuilt output keeps in memory; the ledger and the
# script keep all of it.
REBUILD_OUTPUT_CHARS = 2_000
_SHA = re.compile(r"[0-9a-f]{64}")
_HEX = re.compile(r"\b(?:0x)?[0-9a-fA-F]{7,}\b")
_DIGITS = re.compile(r"\d+")


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def request_sha(messages, tools, effort):
    """The digest a hosted worker records for one model request (its ``request_sha``)."""
    return digest({"messages": messages, "tools": tools, "effort": effort})


def message_shas(messages):
    return [digest(message) for message in messages]


def replay_sha(messages, tools, effort):
    """The digest a replay compares a request by: without what the rewind arm adds to every call
    (``taste.agents.agentrewind.strip``), the same as ``request_sha`` for any other request."""
    messages, tools = agentrewind.strip(messages, tools)
    return request_sha(messages, tools, effort)


def replay_message_shas(messages):
    return message_shas(agentrewind.strip(messages, None)[0])


def tool_run(run):
    """Whether a recorded command is a call of the rewind arm's tools, which changes no file."""
    return agentrewind.parse_tool_command(run.command) is not None


def measured(messages, tools, effort, prompt_tokens):
    """A request as its answer measured it, for bounding the next: its tools and effort and its messages as a
    replay compares them (without the rewind arm's additions), and its tokens (with whatever it carried)."""
    messages, tools = agentrewind.strip(messages, tools)
    return (digest(tools), effort), tuple(message_shas(messages)), prompt_tokens


def measured_at(script, step, model):
    """Step ``step``'s request as the recorded run measured it; None if it was not measured on ``model``."""
    if not step or script.source.get("model") != model or script.source.get("tools_sha256") is None:
        return None
    usage = script.steps[step - 1].reply.usage
    counts = [usage.get(name, 0) for name in ("input_tokens", "cache_read_tokens", "cache_write_tokens")]
    if "input_tokens" not in usage or any(type(count) is not int or count < 0 for count in counts):
        return None
    return (script.source["tools_sha256"], script.source.get("effort")), tuple(script.message_shas(step)), sum(counts)


def prompt_bound(messages, tools, effort, earlier=None):
    """The most prompt tokens a request can hold.

    A request is measured only once it is answered. An earlier one that was
    (``measured``), with the same tools and effort, whose messages begin this
    one's, bounds what they share by its measure; each message added since
    is bounded by its bytes, as ``request_token_bound`` bounds a request.
    Both are compared without what the rewind arm adds to every call, and
    what it adds to this one is counted by its bytes too. Without such a
    measure, the request's bytes bound it whole.
    """
    whole = request_token_bound(len(_json({"messages": messages, "tools": tools, "effort": effort}).encode()))
    if earlier is None:
        return whole
    frame, shas, tokens = earlier
    plain, kept = agentrewind.strip(messages, tools)
    if frame != (digest(kept), effort) or len(plain) < len(shas) or tuple(message_shas(plain[:len(shas)])) != shas:
        return whole
    additions = len(_json(messages).encode()) - len(_json(plain).encode())
    if tools is not None:
        additions += len(_json(tools).encode()) - len(_json(kept).encode())
    return min(whole, tokens + request_token_bound(len(_json(plain[len(shas):]).encode()) + additions))


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def submitted(output, returncode):
    """Whether a command's result ends mini-swe-agent's run as its submission."""
    lines = str(output).lstrip().splitlines()
    return bool(lines) and lines[0].strip() == SENTINEL and returncode == 0


def with_note(output, note):
    """A command's output with a note after it, on a line of its own."""
    return output + ("" if not output or output.endswith("\n") else "\n") + note


def normalized_lines(text):
    lines = []
    for line in str(text).splitlines():
        line = " ".join(_DIGITS.sub("0", _HEX.sub("#", line)).split())
        if line:
            lines.append(line)
    return lines


def similarity(recorded, observed):
    """The Dice coefficient of the two outputs' normalized line multisets; 1.0 for two empty ones."""
    first, second = Counter(normalized_lines(recorded)), Counter(normalized_lines(observed))
    total = sum(first.values()) + sum(second.values())
    return 1.0 if total == 0 else 2 * sum((first & second).values()) / total


def cut_off(run):
    """Whether the record shows the command cut off by its time limit (as the container printed it)."""
    return bool((run.printed or {}).get("timed_out", run.timed_out))


def rebuild_timeout(run, limit):
    """The time a rebuilt command is given: its own limit if the record shows it cut off, else
    REBUILD_TIME_FACTOR times that; never more than ``limit``, the terminal's maximum."""
    seconds = run.timeout_seconds if cut_off(run) else REBUILD_TIME_FACTOR * run.timeout_seconds
    return min(float(seconds), float(limit))


# Reading a command as the shell would (``host_ordered``). Anything it cannot read for certain makes it
# not host-ordered, and the command is compared as every other one is.
_PUNCTUATION = ";&|<>()\n"
_OPERATOR = re.compile(r"&&|\|\||\|&|;;|&>>|&>|>>|>&|>\||<<<|<<-|<<|<&|[;&|<>()\n]")
_SEPARATORS = frozenset({";", "&&", "||", "\n", "(", ")"})
_PIPES = frozenset({"|", "|&"})
_TO_NULL = frozenset({">", ">>", ">|", "&>", "&>>"})
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
# Programs that read files or their input and write only to their output, with the options ruled out below.
_READERS = frozenset({"cd", "pwd", "ls", "cat", "head", "tail", "wc", "cut", "tr", "nl", "sort", "uniq", "grep",
                      "egrep", "fgrep", "rg", "find", "fd", "fdfind", "sed", "git", "echo", "printf", "true"})
_GIT_READS = frozenset({"grep", "log", "show", "diff", "ls-files", "rev-parse", "blame"})
_FIND_WRITES = frozenset({"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"})
_SED_OPTIONS = frozenset({"-n", "--quiet", "--silent", "-E", "-r", "--regexp-extended"})
_SED_PRINT = re.compile(r"\s*(?:\d+|\$)?\s*(?:,\s*(?:\d+|\$)\s*)?p\s*")
# grep's short options that take an argument, and the long ones that switch recursion on.
_GREP_ARGUMENT = frozenset("ABCDdefm")
_GREP_LONG_ARGUMENT = frozenset({"--regexp", "--file", "--max-count", "--after-context", "--before-context",
                                 "--context", "--devices", "--binary-files", "--label", "--include", "--exclude",
                                 "--exclude-dir", "--exclude-from", "--group-separator"})
_GREP_RECURSIVE = frozenset({"--recursive", "--dereference-recursive", "--directories=recurse"})
# Stages that keep the order of the lines they pass on, between a walk and the head or tail that cuts it.
_ORDER_KEEPING = frozenset({"grep", "egrep", "fgrep", "rg", "cut", "tr", "sed", "cat", "nl", "head", "tail"})
# mini-swe-agent's command as its environment runs it (``TasteEnvironment._shell``): variables, then the
# command in a group with standard error folded in.
_MINI_WRAPPER = re.compile(r"(?:export(?: [A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_./:@%+,=-]*)+; )?"
                           r"\{\n(?P<command>.*)\n\} 2>&1", re.DOTALL)


def _tokens(command):
    """The shell's words and operators, quotes removed; a newline separates commands, and ``#`` is a word."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=_PUNCTUATION)
    lexer.whitespace, lexer.whitespace_split, lexer.commenters = " \t\r", True, ""
    tokens = []
    for token in lexer:
        if token and all(char in _PUNCTUATION for char in token):
            operators = _OPERATOR.findall(token)
            if "".join(operators) != token:
                raise ValueError(f"unread operator {token!r}")
            tokens.extend(operators)
        else:
            tokens.append(token)
    return tokens


def _operator(token):
    return bool(token) and all(char in _PUNCTUATION for char in token)


def _pipelines(tokens):
    """The pipelines, each a list of simple commands' words; None for anything else (redirection into a file,
    a heredoc, a command in the background)."""
    pipelines, pipeline, words, index = [], [], [], 0
    while index < len(tokens):
        token = tokens[index]
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        if token in _SEPARATORS or token in _PIPES:
            if words:
                pipeline.append(words)
                words = []
            elif token in _PIPES:
                return None
            if token in _SEPARATORS and pipeline:
                pipelines.append(pipeline)
                pipeline = []
        elif token in _TO_NULL:
            if following != "/dev/null":
                return None
            index += 1
        elif token in (">&", "<&"):
            if following not in ("1", "2", "-"):
                return None
            index += 1
        elif token == "<":
            if following is None or _operator(following):
                return None
            index += 1
        elif _operator(token):
            return None
        else:
            words.append(token)
        index += 1
    if words:
        pipeline.append(words)
    if pipeline:
        pipelines.append(pipeline)
    return pipelines


def _program(words):
    """A simple command's program and arguments, past its variable assignments."""
    index = 0
    while index < len(words) and _ASSIGNMENT.match(words[index]):
        index += 1
    return (words[index], words[index + 1:]) if index < len(words) else ("", [])


def _sed_reads(arguments):
    """sed that only prints line ranges: ``sed -n '1,40p' file``."""
    scripts, operands, given = [], [], iter(arguments)
    for argument in given:
        if argument == "-e":
            scripts.append(next(given, ""))
        elif argument in _SED_OPTIONS:
            continue
        elif argument.startswith("-"):
            return False
        else:
            operands.append(argument)
    if not scripts and operands:
        scripts.append(operands.pop(0))
    return bool(scripts) and all(_SED_PRINT.fullmatch(script) for script in scripts)


def _reads_only(words):
    """Whether a simple command only reads: a program of ``_READERS`` with none of its writing options."""
    program, arguments = _program(words)
    if not program:
        return True
    if program not in _READERS:
        return False
    if program == "sed":
        return _sed_reads(arguments)
    if program == "find":
        return not _FIND_WRITES.intersection(arguments)
    if program in ("fd", "fdfind"):
        return not any(argument in ("-x", "-X") or argument.startswith("--exec") for argument in arguments)
    if program == "rg":
        return not any(argument.startswith("--pre") for argument in arguments)
    if program == "sort":
        return not any(argument.startswith("--output") or (argument.startswith("-") and not argument.startswith("--")
                                                            and "o" in argument[1:]) for argument in arguments)
    if program == "uniq":
        # A second operand is the file uniq writes.
        return len([argument for argument in arguments if not argument.startswith("-")]) <= 1
    if program == "git":
        return bool(arguments) and arguments[0] in _GIT_READS and not any(
            argument.startswith("--output") for argument in arguments)
    return True


def _recursive_grep(arguments):
    given = iter(arguments)
    for argument in given:
        if argument == "--":
            return False
        if argument in _GREP_RECURSIVE:
            return True
        if argument in ("-d", "--directories"):
            if next(given, None) == "recurse":
                return True
        elif argument in _GREP_LONG_ARGUMENT:
            next(given, None)  # its argument is the next word
        elif argument.startswith("-") and not argument.startswith("--") and len(argument) > 1:
            for position, option in enumerate(argument[1:], 1):
                if option in "rR":
                    return True
                if option in _GREP_ARGUMENT:
                    if position == len(argument) - 1:
                        next(given, None)  # its argument is the next word
                    break
    return False


def _walks(words):
    """Whether a simple command prints files or matches in the order the host lists directories."""
    program, arguments = _program(words)
    if program in ("find", "fd", "fdfind"):
        return True
    if program == "rg":
        return not any(argument.startswith("--sort") for argument in arguments)
    return program in ("grep", "egrep", "fgrep") and _recursive_grep(arguments)


def _cuts_a_walk(pipeline):
    walking = False
    for words in pipeline:
        program = _program(words)[0]
        if walking and program in ("head", "tail"):
            return True
        if _walks(words):
            walking = True
        elif program not in _ORDER_KEEPING:
            walking = False
    return False


def host_ordered(command):
    """Whether a command only reads, and cuts a directory walk's output to its first or last lines.

    A walk is a recursive grep, rg (unsorted), find or fd; a cut is ``head`` or ``tail`` after it in one pipeline,
    with only order-keeping filters between (``_ORDER_KEEPING``). Which lines such a command prints is the order the
    host lists directories in. Every part of the command must read only (``_READERS``, with no output into a file
    but /dev/null); anything this cannot read for certain (substitutions, heredocs, quotes left open) is not
    host-ordered.
    """
    if not isinstance(command, str) or any(mark in command for mark in ("`", "$(", "<(", ">(")):
        return False
    try:
        pipelines = _pipelines(_tokens(command))
    except ValueError:
        return False
    if not pipelines or not all(_reads_only(words) for pipeline in pipelines for words in pipeline):
        return False
    return any(_cuts_a_walk(pipeline) for pipeline in pipelines)


def _host_ordered_run(run):
    """``host_ordered`` for a recorded command, which ran as mini-swe-agent's environment wraps it."""
    found = _MINI_WRAPPER.fullmatch(run.executed or "")
    return found is not None and found["command"] == run.command and host_ordered(run.command)


def compare(run, output, returncode, timed_out):
    """One rebuilt command against what the container printed when it was recorded."""
    recorded = run.printed or {"output": run.output, "returncode": run.returncode, "timed_out": run.timed_out}
    score = round(similarity(recorded["output"], output), 4)
    if recorded["timed_out"]:
        # Cut off when recorded: not compared (see the fidelity rule).
        divergent = False
    else:
        divergent = returncode != recorded["returncode"] or bool(timed_out) or score < OUTPUT_FLOOR
    row = {"returncode": returncode, "recorded_returncode": recorded["returncode"],
           "timed_out": bool(timed_out), "recorded_timed_out": recorded["timed_out"],
           "similarity": score, "divergent": divergent}
    if (divergent and returncode == recorded["returncode"] and not timed_out
            and len(normalized_lines(output)) == len(normalized_lines(recorded["output"]))
            and _host_ordered_run(run)):
        # A cut-off walk printed the lines this host lists first (see the fidelity rule).
        row.update(divergent=False, host_ordered=True)
    return row


def _problem(message):
    return ValueError("replay script: " + message)


def _text(value, name):
    if not isinstance(value, str):
        raise _problem(f"{name} must be text")
    return value


def _sha(value, name):
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise _problem(f"{name} must be a SHA-256 digest")
    return value


@dataclass(frozen=True)
class RecordedRun:
    """One command an agent ran, as it wrote it and as the task container ran it.

    ``output``, ``returncode`` and ``timed_out`` are what the agent was shown.
    ``printed`` is what the container printed, when a branch showed the agent
    something else (an appended note, a rejection); a rebuild is checked
    against that.
    """

    command: str
    executed: str | None
    cwd: str
    timeout_seconds: float
    output: str
    returncode: int
    timed_out: bool
    output_exact: bool
    printed: Mapping | None = None

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping):
            raise _problem("a run must be an object")
        executed = value.get("executed")
        timeout = value.get("timeout_seconds")
        if (executed is not None and not isinstance(executed, str)) or type(value.get("returncode")) is not int:
            raise _problem("a run's executed text or exit code is malformed")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise _problem("a run's time limit must be positive")
        if type(value.get("timed_out")) is not bool or type(value.get("output_exact")) is not bool:
            raise _problem("a run's time-out and exactness must be true or false")
        cwd = _text(value.get("cwd"), "a run's directory")
        if not isabs(cwd):
            raise _problem("a run's directory must be absolute")
        printed = value.get("printed")
        if printed is not None and (
                not isinstance(printed, Mapping) or set(printed) != {"output", "returncode", "timed_out"}
                or not isinstance(printed["output"], str) or type(printed["returncode"]) is not int
                or type(printed["timed_out"]) is not bool):
            raise _problem("what a run printed must have its output, exit code and time-out")
        return cls(_text(value.get("command"), "a run's command"), executed, cwd, float(timeout),
                   _text(value.get("output"), "a run's output"), value["returncode"], value["timed_out"],
                   value["output_exact"], None if printed is None else dict(printed))

    def to_dict(self):
        value = {"command": self.command, "executed": self.executed, "cwd": self.cwd,
                 "timeout_seconds": self.timeout_seconds, "output": self.output, "returncode": self.returncode,
                 "timed_out": self.timed_out, "output_exact": self.output_exact}
        if self.printed is not None:
            value["printed"] = dict(self.printed)
        return value


def reply_from_dict(value):
    if not isinstance(value, Mapping) or not isinstance(value.get("output"), list):
        raise _problem("a reply must have its output items")
    if not all(isinstance(item, Mapping) for item in value["output"]):
        raise _problem("a reply's output items must be objects")
    cost = value.get("cost_usd")
    if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
        raise _problem("a reply's cost must be a nonnegative number")
    reason = value.get("incomplete_reason")
    if reason is not None and not isinstance(reason, str):
        raise _problem("a reply's incomplete reason must be text or null")
    if not isinstance(value.get("usage"), Mapping):
        raise _problem("a reply's usage must be an object")
    return ModelReply(output=tuple(value["output"]), status=_text(value.get("status"), "a reply's status"),
                      incomplete_reason=reason, usage=dict(value["usage"]),
                      model=_text(value.get("model"), "a reply's model"), cost_usd=float(cost))


def reply_from_completion(payload, cost):
    """A journaled completion as the hosted worker gave it to its agent (``hosted_worker._reply``)."""
    items = tuple(block["item"] for block in payload["transcript_blocks"] if block.get("type") == _NATIVE)
    status, reason = {"max_tokens": ("incomplete", "max_output_tokens"),
                      "content_filter": ("incomplete", "content_filter")}.get(
                          payload["stop_reason"], ("completed", None))
    return ModelReply(output=items, status=status, incomplete_reason=reason, usage=dict(payload["usage"]),
                      model=payload["model"], cost_usd=cost)


def reply_to_dict(reply):
    return {"output": [dict(item) for item in reply.output], "status": reply.status,
            "incomplete_reason": reply.incomplete_reason, "usage": dict(reply.usage), "model": reply.model,
            "cost_usd": reply.cost_usd}


@dataclass(frozen=True)
class RecordedStep:
    """One model call and the commands run after it, before the next call."""

    number: int
    request_sha: str
    base: int
    added: tuple[str, ...]
    reply: ModelReply
    text: tuple[str, ...]
    calls: tuple[Mapping, ...]
    runs: tuple[RecordedRun, ...]
    replayed: bool = False

    @classmethod
    def from_dict(cls, value, number):
        if not isinstance(value, Mapping) or value.get("step") != number:
            raise _problem(f"step {number} is missing or out of order")
        messages = value.get("messages")
        if (not isinstance(messages, Mapping) or type(messages.get("base")) is not int
                or messages["base"] < 0 or not isinstance(messages.get("added"), list)):
            raise _problem(f"step {number}'s request messages are malformed")
        text, calls, runs = value.get("text"), value.get("calls"), value.get("runs")
        if (not isinstance(text, list) or not all(isinstance(item, str) for item in text)
                or not isinstance(calls, list) or not all(isinstance(item, Mapping) for item in calls)
                or not isinstance(runs, list) or type(value.get("replayed", False)) is not bool):
            raise _problem(f"step {number}'s text, calls or runs are malformed")
        return cls(number, _sha(value.get("request_sha"), "a request digest"), messages["base"],
                   tuple(_sha(item, "a message digest") for item in messages["added"]),
                   reply_from_dict(value.get("reply")), tuple(text), tuple(dict(item) for item in calls),
                   tuple(RecordedRun.from_dict(item) for item in runs), value.get("replayed", False))

    def to_dict(self):
        return {"step": self.number, "request_sha": self.request_sha,
                "messages": {"base": self.base, "added": list(self.added)},
                "reply": reply_to_dict(self.reply), "text": list(self.text),
                "calls": [dict(item) for item in self.calls], "runs": [run.to_dict() for run in self.runs],
                "replayed": self.replayed}



@dataclass(frozen=True)
class ReplayScript:
    """A finished run's steps, in order, and what the agent was given."""

    task: str
    steps: tuple[RecordedStep, ...]
    source: Mapping
    exit: Mapping | None
    # The commands the agent ran before its first model call.
    setup: tuple[RecordedRun, ...] = ()

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or value.get("schema") != SCRIPT_SCHEMA:
            raise _problem("not a replay script of schema " + SCRIPT_SCHEMA)
        steps, setup = value.get("steps"), value.get("setup", [])
        if not isinstance(steps, list) or not isinstance(value.get("source"), Mapping):
            raise _problem("it needs its steps and its source")
        if not isinstance(setup, list):
            raise _problem("its setup must be a list of runs")
        ended = value.get("exit")
        if ended is not None and not isinstance(ended, Mapping):
            raise _problem("its exit must be an object or null")
        return cls(_text(value.get("task"), "the task"),
                   tuple(RecordedStep.from_dict(item, number) for number, item in enumerate(steps, 1)),
                   dict(value["source"]), None if ended is None else dict(ended),
                   tuple(RecordedRun.from_dict(item) for item in setup))

    @classmethod
    def from_bytes(cls, raw):
        if len(raw) > MAX_SCRIPT_BYTES:
            raise _problem("it is larger than its limit")
        try:
            value = json.loads(raw)
        except ValueError as exc:
            raise _problem("it is not JSON") from exc
        return cls.from_dict(value)

    def to_dict(self):
        submission = next((step.number for step in self.steps if self.submission(step.number)), None)
        value = {"schema": SCRIPT_SCHEMA, "source": dict(self.source), "task": self.task,
                 "steps": [step.to_dict() for step in self.steps], "submission_step": submission,
                 "exit": None if self.exit is None else dict(self.exit)}
        if self.setup:
            value["setup"] = [run.to_dict() for run in self.setup]
        return value

    @property
    def protocol(self):
        """The recorded agent's submission protocol (``taste.agents.protocols``)."""
        return protocol((self.source.get("agent") or {}).get("name"))

    def submission(self, number):
        """Whether step ``number`` ended the run as its submission."""
        return self.protocol.submission_index(self.steps[number - 1].runs) is not None

    def override_index(self, number, override):
        """Which of step ``number``'s commands an override changes; None if it can change none."""
        runs = self.steps[number - 1].runs
        return self.protocol.submission_index(runs) if override == "reject" else self.protocol.note_index(runs)

    def message_shas(self, number):
        """The digests of the messages of step ``number``'s request, as recorded."""
        shas = []
        for step in self.steps[:number]:
            shas = shas[:step.base] + list(step.added)
        return shas

    def diagnose(self, number, messages, tools, effort):
        """Where a request differs from step ``number``'s: the first message that differs, and more."""
        recorded, observed = self.message_shas(number), replay_message_shas(messages)
        first = next((index for index, (left, right) in enumerate(zip(recorded, observed, strict=False))
                      if left != right), None)
        if first is None and len(recorded) != len(observed):
            first = min(len(recorded), len(observed))
        detail = {"messages": len(observed), "recorded_messages": len(recorded), "first_differing_message": first}
        if self.source.get("effort") is not None and effort != self.source["effort"]:
            detail["effort"] = {"recorded": self.source["effort"], "now": effort}
        if (self.source.get("tools_sha256") is not None
                and digest(agentrewind.strip((), tools)[1]) != self.source["tools_sha256"]):
            detail["tools_differ"] = True
        compared = agentrewind.strip(messages, None)[0]
        if first is not None and first < len(compared) and isinstance(compared[first], Mapping):
            detail["role"] = compared[first].get("role")
        return detail


def check_branch(script, step, *, mode, override="", note="", replay=True):
    """Refuse a branch this script cannot serve, saying why; nothing otherwise."""
    if type(step) is not int or not 0 <= step <= len(script.steps):
        raise ValueError(f"branch step must be between 0 and {len(script.steps)}, the steps this record holds")
    if mode not in MODES:
        raise ValueError("branch mode must be rebuild or restore")
    commands = [*(script.setup if step else ()), *(run for item in script.steps[:step] for run in item.runs)]
    if mode == "rebuild" and any(run.executed is None and not tool_run(run) for run in commands):
        raise ValueError("this script lacks the exact commands the container ran (the trial's terminal "
                         "ledger), which a rebuild runs again")
    if override and not replay:
        raise ValueError("an override changes what the replayed agent sees: it needs the replay")
    if override:
        if override not in OVERRIDES:
            raise ValueError("branch override must be append, reject or rewind")
        if not note:
            raise ValueError("a branch override needs its note")
        if override == "rewind":
            return  # it changes the next call, not a command's output: any step can be rewound to
        if step == 0 or script.protocol.note_index(script.steps[step - 1].runs) is None:
            raise ValueError("an override changes the output of step k's command, and this step ran none")
        if override == "reject" and not script.submission(step):
            raise ValueError("reject replaces a submission, and step k is not one")
        if override == "append" and script.submission(step):
            raise ValueError("step k is the submission: a note after it would not be read; use reject")


@dataclass(frozen=True)
class BranchPolicy:
    """What a worker needs to branch its hosted agent's run: the script and step k, how, and what changes."""

    script: str
    script_sha256: str
    step: int
    mode: str = "rebuild"
    live: bool = True
    override: str = ""
    note: str = ""
    tolerance: int = 0
    # False: only the files of step k are brought back; the trial's own agent
    # (a checker given a run's final files, say) starts fresh with its own task.
    replay: bool = True
    # The rewind arm (taste.agents.agentrewind): its tools on every call, the notes its earlier commits carried
    # back ({"memory_summary", "target_record_uid", "reason"}), and what an attempt must have left to commit.
    rewind: Mapping | None = None

    def __post_init__(self):
        if not isinstance(self.script, str) or not isabs(self.script) or "\x00" in self.script:
            raise ValueError("the branch script must be an absolute path")
        if not isinstance(self.script_sha256, str) or _SHA.fullmatch(self.script_sha256) is None:
            raise ValueError("the branch script's digest must be a SHA-256")
        if type(self.step) is not int or not 0 <= self.step <= 100_000:
            raise ValueError("branch step must be a nonnegative integer")
        if self.mode not in MODES:
            raise ValueError("branch mode must be rebuild or restore")
        if type(self.live) is not bool:
            raise ValueError("branch live must be true or false")
        if self.override not in ("", *OVERRIDES):
            raise ValueError("branch override must be append, reject or rewind")
        if not isinstance(self.note, str) or len(self.note) > MAX_NOTE_CHARS:
            raise ValueError(f"a branch note is text of at most {MAX_NOTE_CHARS} characters")
        if bool(self.override) != bool(self.note):
            raise ValueError("a branch override needs a note, and a note an override")
        if self.override and not self.live:
            raise ValueError("an override changes what the agent sees next: it needs branch_live=on")
        if type(self.tolerance) is not int or self.tolerance < 0:
            raise ValueError("branch tolerance must be a nonnegative integer")
        if type(self.replay) is not bool:
            raise ValueError("branch replay must be true or false")
        if self.override and not self.replay:
            raise ValueError("an override changes what the replayed agent sees: it needs branch_replay=on")
        if self.rewind is not None:
            _check_rewind(self.rewind)
        if self.override == "rewind" and self.rewind is None:
            raise ValueError("a rewind needs the rewind arm's notes")
        if self.override == "rewind" and self.note != agentrewind.notebook(self.rewind["notes"]):
            raise ValueError("a rewind's note is the notebook of its notes")

    def to_dict(self):
        value = {"script": self.script, "script_sha256": self.script_sha256, "step": self.step,
                 "mode": self.mode, "live": self.live, "tolerance": self.tolerance, "replay": self.replay}
        if self.override:
            value.update(override=self.override, note=self.note)
        if self.rewind is not None:
            value["rewind"] = {**self.rewind, "notes": [dict(note) for note in self.rewind["notes"]]}
        return value

    def agent_rewind(self):
        """What the rewind arm's agent is given (``MiniSweAgent(rewind=...)``): the notes, and the call their notebook
        goes in at: a rewind's step k + 1, or for a later round of a rewound run the call it first went in at
        (``at_call``), where it stays; None outside the arm."""
        if self.rewind is None:
            return None
        return {"notes": [dict(note) for note in self.rewind["notes"]],
                "at_call": self.step + 1 if self.override == "rewind" else self.rewind.get("at_call")}

    @classmethod
    def from_dict(cls, value):
        required = {"script", "script_sha256", "step", "mode", "live", "tolerance", "replay"}
        if not isinstance(value, Mapping) or not required <= set(value) <= {*required, "override", "note", "rewind"}:
            raise ValueError("invalid branch policy fields")
        if "override" in value and not value.get("override"):
            raise ValueError("invalid branch policy fields")
        return cls(**dict(value))


def _check_rewind(rewind):
    required, optional = {"notes", "restart_usd"}, {"restart_seconds", "spend_cap_usd", "at_call"}
    if not isinstance(rewind, Mapping) or not required <= set(rewind) <= required | optional:
        raise ValueError("the rewind arm's settings are its notes and restart_usd, and optionally restart_seconds, "
                         "spend_cap_usd and at_call")
    notes = rewind["notes"]
    # A list, or a tuple as a worker's frozen assignment holds it.
    if not isinstance(notes, (list, tuple)) or not all(
            isinstance(note, Mapping) and set(note) <= {"memory_summary", "target_record_uid", "reason"}
            and isinstance(note.get("memory_summary"), str) and note["memory_summary"].strip()
            and isinstance(note.get("target_record_uid", ""), str)
            and (note.get("reason") is None or isinstance(note["reason"], str)) for note in notes):
        raise ValueError("each rewind note is a non-empty memory_summary, its target and its reason")
    for name in ("restart_usd", "restart_seconds", "spend_cap_usd"):
        value = rewind.get(name, 0.0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"the rewind arm's {name} is a nonnegative number")
    at_call = rewind.get("at_call")
    if at_call is not None and (type(at_call) is not int or at_call < 1):
        raise ValueError("the rewind arm's at_call is the call its notebook first went in at, 1 or more")


async def rebuild(script, step, execute, *, tolerance=0, on_row=None):
    """Run the recorded commands of steps 1..``step`` again, in order, the setup first; return the account.

    ``execute(number, run)`` runs one recorded command and returns its
    ``(output, returncode, timed_out)``; the setup's commands are step 0's.
    Each result is compared with the record (``compare``); the rebuild stops
    once more commands diverged than ``tolerance`` allows. ``on_row`` sees
    each comparison as it is made.
    """
    rows, divergent = [], 0
    commands = [(0, index, run) for index, run in enumerate(script.setup if step else ())]
    commands += [(item.number, index, run) for item in script.steps[:step] for index, run in enumerate(item.runs)
                 if not tool_run(run)]
    for number, index, run in commands:
        output, returncode, timed_out = await execute(number, run)
        row = {"step": number, "run": index, **compare(run, output, returncode, timed_out)}
        rows.append(row)
        if on_row is not None:
            on_row(row, output)
        divergent += row["divergent"]
        if divergent > tolerance:
            return {"commands": len(rows), "divergent": divergent, "faithful": False, "rows": rows}
    return {"commands": len(rows), "divergent": divergent, "faithful": True, "rows": rows}


class ReplayHost:
    """An agent's host that answers its first k steps from a replay script, then hands it to a live host.

    ``record(kind, payload)`` puts each replayed step on the run's record; it
    may raise ``HostedStop`` to end the run. ``faithful=False`` says the files
    were not brought back: the context is still replayed, but the agent is
    stopped at its first call after it. ``context=False`` leaves the first live
    request unchecked against the record's next one, as for another agent
    given only the files of step k.
    """

    def __init__(self, live, script, step, *, record, live_after=True, override="", note="",
                 faithful=True, context=True):
        self.live, self.script, self.step = live, script, step
        self.record = record
        self.live_after, self.override, self.note = live_after, override, note
        self.faithful, self.context = faithful, context
        self.replayed = 0
        # The recorded commands still to come before the agent's next call: the setup's, then each step's.
        self._runs = list(script.setup) if step else []
        self._position = 0  # of the next one, in its step
        self._target = None  # which of step k's commands the override changes
        self._rejected = False
        self._done = step == 0
        self._handed_over = False

    @property
    def prefix_complete(self):
        """Whether all k recorded steps have been served."""
        return self._done

    @property
    def stopped(self):
        return self.live.stopped

    def stop(self, reason, *, now=True):
        self.live.stop(reason, now=now)

    def _diverged(self, number, what, detail):
        self.faithful = False
        self.record("unfaithful", {"step": number, "reason": what, "detail": detail})
        raise HostedStop("branch_unfaithful")

    def _prefix_done(self, dropped=0):
        self._done = True
        self.record("prefix", {"steps": self.step, **({"dropped": dropped} if dropped else {})})

    def _hand_over(self, sha=None):
        """The first call after the prefix: check it, then let the agent go on live or stop it."""
        if self._handed_over:
            return
        if not self.faithful:
            raise HostedStop("branch_unfaithful")
        context = "unchecked"
        if (sha is not None and self.context and self.override in ("", "rewind")
                and self.step < len(self.script.steps)):
            # No output of step k was changed: the next request must be the recorded one (a rewind's
            # notebook is not compared).
            expected = self.script.steps[self.step].request_sha
            context = "matched" if sha == expected else "differs"
            if context == "differs":
                self._diverged(self.step + 1, "context", {"note": "the first request after the prefix differs "
                                                          "from the recorded one"})
        self.record("live", {"live": self.live_after, "context": context})
        if not self.live_after:
            raise HostedStop("branch_live_off")
        self._handed_over = True

    def ask(self, *, messages, tools, effort):
        sha = replay_sha(messages, tools, effort)
        if not self._done and self._runs and self._rejected:
            # The rejection kept the agent from what its submission would have run next.
            dropped, self._runs = len(self._runs), []
            self._prefix_done(dropped)
        if self._done:
            self._hand_over(sha)
            return self.live.ask(messages=messages, tools=tools, effort=effort)
        number = self.replayed + 1
        if self._runs:
            self._diverged(self.replayed, "ask", {"note": "the agent asked its model before running every "
                                                  "recorded command of its step", "commands_left": len(self._runs)})
        step = self.script.steps[self.replayed]
        if sha != step.request_sha:
            self._diverged(number, "request", self.script.diagnose(number, messages, tools, effort))
        self.replayed = number
        self._runs, self._position = list(step.runs), 0
        if number == self.step and self.override in ("append", "reject"):
            self._target = self.script.override_index(number, self.override)
        self.record("replay", {"step": number, "request_sha": sha, "text": list(step.text),
                               "calls": [dict(call) for call in step.calls], "model": step.reply.model,
                               "recorded_cost_usd": step.reply.cost_usd})
        if not self._runs and number == self.step:
            self._prefix_done()
        # Paid for by the recorded run, not by this one.
        return replace(step.reply, cost_usd=0.0)

    def run(self, command, *, cwd, timeout_seconds, shown=None):
        if self._done:
            self._hand_over()
            return self.live.run(command, cwd=cwd, timeout_seconds=timeout_seconds, shown=shown)
        if not self._runs:
            self._diverged(self.replayed, "command", {"note": "the agent ran a command the record does not have"})
        expected, index = self._runs.pop(0), self._position
        self._position += 1
        same = (command == expected.executed if expected.executed is not None
                else (shown if shown is not None else command) == expected.command)
        if not same or cwd != expected.cwd:
            self._diverged(self.replayed, "command", {
                "command_matches": same, "cwd_matches": cwd == expected.cwd})
        output, returncode, timed_out = expected.output, expected.returncode, expected.timed_out
        at_step = self.step and self.replayed == self.step
        override = self.override if at_step and index == self._target else ""
        if override == "append":
            output = with_note(output, self.note)
        elif override == "reject":
            output, returncode, timed_out = self.note, REJECT_EXIT, False
            self._rejected = True
        # A call of the rewind arm's tools is kept whole: a later timeline reads it back (taste.agents.agentrewind).
        self.record("replay_output", {"step": self.replayed,
                                      "command": expected.command if tool_run(expected) else _cut(expected.command),
                                      "cwd": expected.cwd, "returncode": returncode, "timed_out": timed_out,
                                      "override": override, "output": _cut(output, 16_000),
                                      "output_chars": len(output), "output_sha256": text_sha(output)})
        if at_step and not self._runs:
            self._prefix_done()
        return ShellResult(output, returncode, timed_out=timed_out)


def _cut(text, limit=65_536):
    if len(text) <= limit:
        return text
    head = limit * 2 // 5
    return text[:head] + f"\n... [cut: {len(text) - limit:,} more characters]\n" + text[-(limit - head):]
