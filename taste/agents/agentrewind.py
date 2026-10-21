"""AgentRewind (Zhuang et al., arXiv 2608.14380) as an arm of the recovery study: what its agent is given, and
how its two tools are answered.

What stays AgentRewind's, unchanged (``taste.agents.agentrewind_upstream``, its released code at commit
6661046): the two tools' names, descriptions and schemas; the guidance for using them, as its default
policies give it (the agent names the record; its messages and its files roll back; memory accumulates);
the checks a commit's memory must pass and their texts (a feasibility verdict; a note 75% like an earlier
one); the notebook's header, cuts (the last 20 notes, 900 characters each, 8,000 in all) and rendering; and
each candidate's rendering, with its 400-character previews and the cap of 80 candidates.

What Taste supplies, disclosed with results, because the released code leaves it to a benchmark's adapter,
which is not public:

- the tools in the Responses API's flat form, beside mini-swe-agent's bash tool, on every call of every round;
- the guidance after the agent's system prompt, two newlines apart, on every call;
- the notebook as a user turn after the restored messages (Bedrock's Converse takes no system message after
  the conversation starts), built from the notes in full and rendered as upstream renders it, and kept at
  that place on every later call of the rewound run, later rounds included (upstream's public runner applies
  its override to the rewound call; its private adapter is presumed to keep it, as the context it made does);
- record ids ``llm-<step>``, the study's own step numbers in the current timeline; after a rewind the
  re-made call takes the number of the call it replaces; there are no tool record ids, so upstream's mapping
  of a tool id to its owning call has nothing to map;
- the candidate list as JSON; ``created_at`` null (replayed steps keep no clock time); a candidate's tool
  record as the command's exit code and output as recorded, with no file effects (Taste does not snapshot
  files per command);
- each tool's answer as the tool's output: an error with exit code 1, the agent then going on as upstream's
  tools raise to it. A target that is not one of the timeline's calls is such an error, which the agent can
  correct; upstream's commit lets it through and its runner then fails the rewind. An accepted commit ends
  the run, and the recovery driver branches from the chosen step (``taste.brains.branch_replay``, override
  ``rewind``);
- a commit refused when the attempt's budget left cannot start another round (the study's episode rule), in
  dollars against the trial's own cap and in seconds against its deadline;
- the context-window guard (``taste.brains.branch_replay.prompt_bound``) counts the arm's additions to a
  request by their bytes on top of the last measure, so for Claude (200K) this arm's runs end somewhat
  earlier than the others' near the window.

The recovery driver's ``agentrewind_norestore`` twin (``taste.recovery_study.recovery_driver``) is a rewind,
the same way, to where the agent began to rewind (the step after the last one that ran a container command),
so its files are its final files and the rollback guidance is true of it. Its note has the same memory and
reason; only its target tag differs, naming that call. The memory still speaks of the call the agent chose.

Files: a rewind's come back by running the run's recorded commands again in a fresh container and checking
them against the record (``taste.brains.branch_replay.rebuild``), not from a snapshot as upstream's git
workspace does; the twin's are its trial's saved final files.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from taste.agents.agentrewind_upstream import (
    BACKTRACK_TOOL_DEFINITIONS,
    BACKTRACK_TOOL_INSTRUCTIONS,
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_MEMORY_PREFIX,
    MEMORY_DUPLICATE_REJECTION,
    MEMORY_VERDICT_REJECTION,
    candidate_from_segment,
    find_duplicate_memory_overlap,
    find_feasibility_verdict,
    render_backtrack_memory_notebook,
)

CANDIDATES, COMMIT = "backtrack_candidates", "backtrack_commit"
TOOL_NAMES = frozenset({CANDIDATES, COMMIT})
# The default policies' guidance: upstream's backtrack_tool_instructions() with nothing set returns this text.
GUIDANCE = BACKTRACK_TOOL_INSTRUCTIONS
GUIDANCE_SEPARATOR = "\n\n"
# A tool call as the agent's command: no bash command mini-swe-agent's environment writes starts with a NUL.
TOOL_PREFIX = "\x00taste-agentrewind\x00"
_RECORD = re.compile(r"llm-([1-9][0-9]*)")
REFUSED_FOR_BUDGET = ("backtrack_commit refused: the budget left in this attempt cannot start another round "
                      "from an earlier point. Continue from the current state.")


def tools():
    """The two tools, as the Responses API takes a function tool."""
    return [{"type": "function", **copy.deepcopy(definition)} for definition in BACKTRACK_TOOL_DEFINITIONS]


def with_guidance(system):
    return system + GUIDANCE_SEPARATOR + GUIDANCE


def notebook(notes):
    """The memory notebook of these notes (``{"memory_summary", "target_record_uid", "reason"}``), as upstream
    renders it after a rewind under its default policies."""
    return render_backtrack_memory_notebook([dict(note) for note in notes])


def notebook_message(notes):
    return {"role": "user", "content": notebook(notes)}


def is_notebook(message):
    return (isinstance(message, Mapping) and message.get("role") == "user" and isinstance(message.get("content"), str)
            and message["content"].strip().startswith(DEFAULT_MEMORY_PREFIX.strip()))


def strip(messages, tools):
    """A request without what this arm adds to every call (its tools, the guidance after the system prompt, a
    notebook), so that a run recorded without them, or with another notebook, replays as the same requests."""
    suffix = GUIDANCE_SEPARATOR + GUIDANCE
    kept = []
    for message in messages:
        if is_notebook(message):
            continue
        if (isinstance(message, Mapping) and message.get("role") == "system" and isinstance(message.get("content"), str)
                and message["content"].endswith(suffix)):
            message = {**message, "content": message["content"][:-len(suffix)]}
        kept.append(message)
    return kept, (None if tools is None else [tool for tool in tools if tool.get("name") not in TOOL_NAMES])


def tool_command(name, arguments):
    return TOOL_PREFIX + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False, sort_keys=True)


def parse_tool_command(command):
    """(name, arguments) of a tool call written as a command; None for any other command."""
    if not isinstance(command, str) or not command.startswith(TOOL_PREFIX):
        return None
    value = json.loads(command[len(TOOL_PREFIX):])
    return value["name"], value["arguments"]


@dataclass(frozen=True)
class Step:
    """One model call of the current timeline: its text, its tool calls, and what each returned to the agent."""

    number: int
    text: tuple[str, ...] = ()
    calls: tuple[Mapping, ...] = ()  # {"name", "arguments"}, in order
    results: tuple[Mapping, ...] = ()  # {"name", "arguments", "output", "error"}, in order


def record_uid(number):
    return f"llm-{number}"


def _result(command, returncode, output, timed_out):
    try:
        tool = parse_tool_command(command)
    except ValueError:  # a tool call recorded cut: still a call of the tools, its arguments unreadable
        tool = ("unreadable tool call", {"command": command})
    name, arguments = tool if tool is not None else ("bash", {"command": command})
    return {"name": name, "arguments": arguments, "output": {"returncode": returncode, "output": output},
            "error": "timeout" if timed_out else None}


def timeline(events, prefix="hosted_"):
    """The current timeline's model calls, from a hosted run's record (``taste.brains.hosted_worker``): one per
    reply the agent received, the replayed ones first, numbered as the study numbers steps; each with the
    commands that followed it and what they returned. Setup commands and replies the run's end cut off are not
    steps."""
    steps, commands = [], {}
    for event in events:
        kind = str(event.get("kind", ""))
        kind = kind[len(prefix):] if kind.startswith(prefix) else ""
        if kind == "replay" or (kind == "completion" and not event.get("cut_off")):
            calls = tuple({"name": call["name"], "arguments": call["arguments"]} for call in event.get("calls") or ())
            steps.append((tuple(event.get("text") or ()), calls, []))
        elif kind == "replay_output" and event.get("step") and steps:
            steps[-1][2].append(_result(event["command"], event["returncode"], event["output"], event["timed_out"]))
        elif kind == "command" and steps:
            commands[event["effect_id"]] = event["command"]
        elif kind == "output" and event.get("effect_id") in commands:
            steps[-1][2].append(_result(commands.pop(event["effect_id"]), event["returncode"], event["output"],
                                        event.get("terminated") == "timeout"))
    return [Step(number, text, calls, tuple(results)) for number, (text, calls, results) in enumerate(steps, 1)]


def candidates(steps: Iterable[Step], limit=DEFAULT_CANDIDATE_LIMIT):
    """The last ``limit`` model calls of the timeline as rewind targets, each rendered by upstream's
    ``candidate_from_segment``."""
    found = []
    for step in steps:
        record = {"record_uid": record_uid(step.number), "kind": "llm", "metadata": {"created_at": None},
                  "output": {"content": "\n".join(step.text),
                             "tool_calls": [{"name": call["name"], "arguments": json.dumps(call["arguments"])}
                                            for call in step.calls]}}
        segment = [{"kind": "tool", "input": {"tool_name": result["name"], "arguments": result["arguments"]},
                    "output": {"value": result["output"]}, "error": result.get("error"), "effects": None}
                   for result in step.results]
        found.append(candidate_from_segment(record, segment))
    return found[-limit:] if limit else found


def _tool_calls(step):
    """The step's commands, each as (name, arguments) if it called one of the two tools, else None; [] for none.
    A tool call's encoding holds no newline (JSON escapes them), so a step's commands split on newlines."""
    lines = [line for line in str(step.command).split("\n") if line]
    return [parse_tool_command(line) if line.startswith(TOOL_PREFIX) else None for line in lines]


def _ran_a_command(step):
    return any(call is None for call in _tool_calls(step))


def ended_at_commit(steps):
    """For a run that ended at an accepted commit, from its steps (``taste.agents.trajectory_reader.Step``):
    ``(Commit, where the agent began to rewind, whether the commit's own step ran a container command)``; None
    if the last tool call is not a commit.

    Where it began to rewind is the step after the last one that ran a container command: from there on it
    only called the two tools or ran nothing, so the files there are the run's final files. It is the call a
    commit that restores nothing goes back to (the rewind arm's twin); when the commit's own step ran a
    command too, it is that step, whose command going back undoes.
    """
    steps = list(steps)
    calls = [call for call in (_tool_calls(steps[-1]) if steps else ()) if call is not None]
    if not calls or calls[-1][0] != COMMIT:
        return None
    arguments = calls[-1][1] if isinstance(calls[-1][1], Mapping) else {}
    match = _RECORD.fullmatch(str(arguments.get("record_uid", "")).strip())
    summary = arguments.get("memory_summary")
    if match is None or not isinstance(summary, str) or not summary.strip():
        return None
    ran = _ran_a_command(steps[-1])
    first = len(steps)
    while not ran and first > 1 and not _ran_a_command(steps[first - 2]):
        first -= 1
    reason = arguments.get("reason")
    return Commit(int(match.group(1)), summary, reason if isinstance(reason, str) and reason else None), first, ran


@dataclass(frozen=True)
class Commit:
    """An accepted commit: the step to re-decide, and the note it carries back."""

    step: int
    memory_summary: str
    reason: str | None = None

    def note(self):
        return {"memory_summary": self.memory_summary, "target_record_uid": record_uid(self.step),
                "reason": self.reason}


def answer(name, arguments, steps, notes=(), *, can_restart=True):
    """(output, exit code, an accepted ``Commit`` or None) for one call of the two tools.

    A commit's checks run in upstream's order: a record id and a memory are required, a feasibility verdict is
    refused, the target must be one of the timeline's model calls (where upstream's commit lets an unknown one
    through and its runner fails the rewind), and a note too like an earlier one is refused; then the study's
    own rule, that the attempt can still start another round.
    """
    arguments = arguments if isinstance(arguments, Mapping) else {}
    steps = list(steps)
    if name == CANDIDATES:
        reason = arguments.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return "backtrack_candidates requires a non-empty reason.", 1, None
        return json.dumps(candidates(steps), ensure_ascii=False, sort_keys=True), 0, None
    if name != COMMIT:
        return f"Unknown tool '{name}'.", 1, None
    uid, summary, reason = arguments.get("record_uid"), arguments.get("memory_summary"), arguments.get("reason")
    if not isinstance(uid, str) or not uid.strip():
        return "backtrack_commit requires a non-empty record_uid.", 1, None
    if not isinstance(summary, str) or not summary.strip():
        return "backtrack_commit requires a non-empty memory_summary.", 1, None
    phrase = find_feasibility_verdict(summary)
    if phrase is not None:
        return MEMORY_VERDICT_REJECTION.format(phrase=phrase), 1, None
    match = _RECORD.fullmatch(uid.strip())
    if match is None or int(match.group(1)) not in {step.number for step in steps}:
        return f"backtrack target not found: {uid}", 1, None
    overlap = find_duplicate_memory_overlap(summary, [dict(note) for note in notes])
    if overlap is not None:
        return MEMORY_DUPLICATE_REJECTION.format(overlap=overlap), 1, None
    if not can_restart:
        return REFUSED_FOR_BUDGET, 1, None
    commit = Commit(int(match.group(1)), summary, reason if isinstance(reason, str) and reason else None)
    return f"backtrack_commit accepted: the run rewinds to {record_uid(commit.step)}.", 0, commit
