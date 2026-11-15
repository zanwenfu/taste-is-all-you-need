"""A Claude Code session's score, read from its saved stream alone: for each standing rule, which of Claude Code's own Bash commands it
applied to and which obeyed it, turn by turn; and when the session's ls rule lost to the project file's.

The stream is the driver's stream.jsonl: Claude Code's stream-json lines, with the driver's own "study_turn" line before each call,
naming the turn and the rules its message told. Each Bash tool call (an assistant content block of type tool_use, name Bash, with
input.command) is split into its simple commands (split_commands), and each simple command is read for the program it runs:

    python_B     every run of Python (python, python3 or python3.x, by any path) from the turn the rule was told in; obeyed when -B is
                 among the interpreter's own options, before the script, -c or -m. "python -c" counts, as in the long-run study, and is
                 also counted apart, so an analysis can leave it out. Setting PYTHONDONTWRITEBYTECODE instead is not obeying, and tools
                 that run Python under another name (pytest, uv run) are not counted.
    ls_one       every ls from the turn the rule was told in; obeyed when 1 is among its short options, as in the long-run study's
                 pattern (so ls -1a obeys too). The other forms counted are ls -la (l and a, the project file's form) and other.
    session_tag  every task after the one that told it; obeyed when, within the task, an echo or printf ran in a Bash call whose
                 command or output holds the tag. A task with no such call, or no Bash call at all, counts as not obeyed.

As in studies.long_run.score, a rule applies to each command of its kind from the turn it was told in, ``by_task`` holds the counts
turn by turn (arms can end early, and only turns both arms reached compare), and ``by_compactions`` by how many compactions came first.
Compactions are counted from the stream's compact_boundary system lines. The PreCompact and PostCompact hooks' log is read beside them
and placed in turns by time, as a check that both agree, and in case the stream has no such lines.

    python -m studies.claude_code.score SESSION_FOLDER    # reads session.json, stream.jsonl and hooks/compaction.jsonl; writes summary.json
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import shlex
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

RULES = ("python_B", "ls_one", "session_tag")  # the names the driver's turn lines use, as in the long-run study
LS_ONE, LS_LA, LS_OTHER = "ls -1", "ls -la", "other"
_PYTHON = re.compile(r"^python(?:\d+(?:\.\d+)*)?$")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_KEYWORDS = {"do", "then", "else", "elif", "if", "while", "until", "!", "{", "}", "time", "exec", "command", "builtin", "nohup"}
_HEREDOC = re.compile(r"<<(-?)[ \t]*\\?(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
_ECHOES = ("echo", "printf")


# Commands

def _separator(command: str, i: int) -> str:
    """The command separator starting at ``i``, outside quotes, or ''. Redirections such as 2>&1, &> and >| are not separators."""
    pair, char = command[i:i + 2], command[i]
    before = command[i - 1] if i else ""
    if pair in ("&&", "||", ";;", "|&", "$("):
        return pair
    if char in ";()`\n":
        return char
    if char == "|":
        return "" if before == ">" else "|"
    if char == "&":
        return "" if before in ("<", ">") or pair == "&>" else "&"
    return ""


def _after_heredocs(command: str, start: int, documents: list[tuple[str, bool]]) -> int:
    """Where the command line goes on after the bodies of the here-documents opened on the line that just ended."""
    i = start
    for word, strip_tabs in documents:
        while i < len(command):
            end = command.find("\n", i)
            line = command[i:] if end < 0 else command[i:end]
            i = len(command) if end < 0 else end + 1
            if (line.lstrip("\t") if strip_tabs else line).rstrip() == word:
                break
    return i


def split_commands(command: str) -> list[str]:
    """The simple commands of a shell command line, in order: split at ; && || | & and newlines, at subshell parentheses, backquotes
    and $( ), outside quotes. Comments and here-document bodies are dropped, since the shell does not run them. Substitutions inside
    double quotes stay in their command: a reading of what runs, not a shell."""
    segments: list[str] = []
    current: list[str] = []
    documents: list[tuple[str, bool]] = []
    quote, word_start, i = "", True, 0

    def cut() -> None:
        text = "".join(current).strip()
        if text:
            segments.append(text)
        current.clear()

    while i < len(command):
        char = command[i]
        if quote:
            current.append(char)
            if quote == '"' and char == "\\" and i + 1 < len(command):
                current.append(command[i + 1])
                i += 1
            elif char == quote:
                quote = ""
            i += 1
            continue
        if char == "\\" and i + 1 < len(command):
            current.append(" " if command[i + 1] == "\n" else command[i:i + 2])  # a line continuation is a space
            word_start = command[i + 1] == "\n"
            i += 2
            continue
        if char in "'\"":
            quote, word_start = char, False
            current.append(char)
            i += 1
            continue
        if char == "#" and word_start:
            while i < len(command) and command[i] != "\n":
                i += 1
            continue
        heredoc = _HEREDOC.match(command, i) if command.startswith("<<", i) and not command.startswith("<<<", i) else None
        if heredoc:
            documents.append((heredoc.group(3), heredoc.group(1) == "-"))
            current.append(heredoc.group(0))
            i, word_start = heredoc.end(), False
            continue
        separator = _separator(command, i)
        if separator:
            cut()
            i += len(separator)
            if separator == "\n":
                i = _after_heredocs(command, i, documents)
                documents = []
            word_start = True
            continue
        current.append(char)
        word_start = char in " \t"
        i += 1
    cut()
    return segments


def words(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:  # an unclosed quote: the plain words still name the program
        return segment.split()


def program(argv: list[str]) -> tuple[str, list[str]]:
    """The program a simple command runs, by its base name, and its arguments: past variable assignments, shell keywords, and the
    wrappers sudo, env, nice and timeout."""
    rest = list(argv)
    while rest:
        head = rest[0]
        if _ASSIGNMENT.match(head) or head in _KEYWORDS:
            rest.pop(0)
        elif head in ("sudo", "env", "nice"):
            rest.pop(0)
            if head == "nice" and rest[:1] == ["-n"]:
                rest = rest[2:]
            while rest and rest[0].startswith("-"):
                rest.pop(0)
        elif head == "timeout":
            rest.pop(0)
            while rest and rest[0].startswith("-"):
                if rest.pop(0) in ("-s", "-k", "--signal", "--kill-after") and rest:
                    rest.pop(0)
            rest = rest[1:]  # the duration
        else:
            break
    return (rest[0].rsplit("/", 1)[-1], rest[1:]) if rest else ("", [])


def python_options(args: list[str]) -> tuple[bool, bool]:
    """Whether -B is among the interpreter's own options, and whether it runs a -c string. Options end at the script, -, -c or -m."""
    has_b, i = False, 0
    while i < len(args):
        arg = args[i]
        if arg in ("-", "--") or not arg.startswith("-"):
            break
        if arg.startswith("--"):
            i += 2 if arg == "--check-hash-based-pycs" else 1
            continue
        letters = arg[1:]
        for position, letter in enumerate(letters):
            if letter == "B":
                has_b = True
            elif letter in "cm":
                return has_b, letter == "c"
            elif letter in "WX":  # these take an argument: the rest of the word, or the next word
                i += position == len(letters) - 1
                break
        i += 1
    return has_b, False


def ls_form(args: list[str]) -> str:
    letters: set[str] = set()
    for arg in args:
        if arg == "--":
            break
        if arg == "--all":
            letters.add("a")
        elif arg.startswith("-") and not arg.startswith("--"):
            letters.update(arg[1:])
    if "1" in letters:
        return LS_ONE
    return LS_LA if {"l", "a"} <= letters else LS_OTHER


# The stream

def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(block.get("text") or "") for block in content if isinstance(block, dict) and block.get("type") == "text")
    return ""


def _time(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _new_turn(marker: dict[str, Any], compactions: int) -> dict[str, Any]:
    return {"id": str(marker.get("turn")), "kind": marker.get("kind"), "tells": list(marker.get("tells") or []), "time": marker.get("time"),
            "bash": [], "texts": [], "result": None, "compactions": 0, "compactions_at_start": compactions, "compactions_at_end": compactions}


def read_stream(lines: Iterable[str | dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The session's turns in order, each with its Bash calls (command, output, compactions before it, context tokens at it) and its
    result line; and totals over the whole stream. A tool call seen twice (same id) counts once."""
    turns: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    tools: Counter[str] = Counter()
    compactions = broken = bash_calls = subagent_calls = 0
    context: int | None = None
    turn: dict[str, Any] | None = None
    for raw in lines:
        if isinstance(raw, str):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                broken += 1
                continue
        else:
            event = raw
        if not isinstance(event, dict):
            broken += 1
            continue
        kind = event.get("type")
        if kind == "study_turn" or turn is None:
            turn = _new_turn(event if kind == "study_turn" else {"turn": "unmarked", "kind": "unmarked"}, compactions)
            turns.append(turn)
            if kind == "study_turn":
                continue
        if kind == "system" and event.get("subtype") == "compact_boundary":
            compactions += 1
            turn["compactions"] += 1
        elif kind == "assistant":
            message = event.get("message") or {}
            usage = message.get("usage") or {}
            if usage:
                context = sum(int(usage.get(key) or 0) for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    turn["texts"].append(str(block.get("text") or ""))
                elif block.get("type") == "tool_use":
                    key = str(block.get("id") or f"no-id-{len(seen)}")
                    if key in seen:
                        continue
                    seen.add(key)
                    tools[str(block.get("name"))] += 1
                    if block.get("name") == "Bash":
                        subagent = event.get("parent_tool_use_id") is not None
                        call = {"id": key, "command": str((block.get("input") or {}).get("command") or ""), "output": "",
                                "compactions": compactions, "context_tokens": context, "subagent": subagent}
                        turn["bash"].append(call)
                        by_id[key] = call
                        bash_calls += 1
                        subagent_calls += subagent
        elif kind == "user":
            content = (event.get("message") or {}).get("content")
            for block in content if isinstance(content, list) else []:
                if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id") in by_id:
                    by_id[block["tool_use_id"]]["output"] = _text(block.get("content"))
        elif kind == "result":
            turn["result"] = {"subtype": event.get("subtype"), "is_error": event.get("is_error"), "cost_usd": event.get("total_cost_usd"),
                              "num_turns": event.get("num_turns"), "session_id": event.get("session_id"),
                              "permission_denials": len(event.get("permission_denials") or []), "usage": event.get("usage") or {}}
        turn["compactions_at_end"] = compactions
    return turns, {"compactions": compactions, "tools": dict(tools), "bash_calls": bash_calls, "subagent_bash_calls": subagent_calls,
                   "broken_lines": broken}


def log_by_turn(turns: list[dict[str, Any]], entries: Iterable[dict[str, Any]], event: str = "PreCompact") -> list[int]:
    """How many compaction-log lines of ``event`` fall in each turn, from its start line's time to the next one's."""
    starts = [_time(turn["time"]) for turn in turns]
    counts = [0] * len(turns)
    if not turns or any(start is None for start in starts):
        return counts
    for entry in entries:
        when = _time(entry.get("time")) if entry.get("event") == event else None
        index = bisect.bisect_right(starts, when) - 1 if when else -1
        if index >= 0:
            counts[index] += 1
    return counts


# Scores

def _mark(score: dict[str, Any], task: str, ok: bool, compactions: int) -> None:
    score["applies"] += 1
    score["obeyed"] += ok
    cell = score["by_task"].setdefault(task, {"obeyed": 0, "applies": 0})
    cell["obeyed"] += ok
    cell["applies"] += 1
    bucket = score["by_compactions"].setdefault(str(compactions), [0, 0])
    bucket[0] += ok
    bucket[1] += 1


def override(turns: list[dict[str, Any]], told: int | None, log_counts: list[int]) -> dict[str, Any]:
    """When the session's ls rule first lost: the first ls -la (the project file's form) after the turn that told it, and the first ls
    of any other form than -1; each with the turns, Bash calls, ls calls and compactions since the rule was told."""
    if told is None:
        return {"told": False}
    start = turns[told]
    points: dict[str, dict[str, Any] | None] = {"first_ls_la": None, "first_not_ls_1": None}
    bash_calls = ls_calls = 0
    for index in range(told, len(turns)):
        for call in turns[index]["bash"]:
            for segment in split_commands(call["command"]):
                name, args = program(words(segment))
                if name != "ls":
                    continue
                form = ls_form(args)
                point = {"turn": turns[index]["id"], "turns_after": index - told, "bash_calls_before": bash_calls,
                         "ls_calls_before": ls_calls, "compactions_since": call["compactions"] - start["compactions_at_start"],
                         "log_compactions_by_turn_end": sum(log_counts[told:index + 1]), "context_tokens": call["context_tokens"],
                         "command": segment}
                if form != LS_ONE and points["first_not_ls_1"] is None:
                    points["first_not_ls_1"] = point
                if form == LS_LA and points["first_ls_la"] is None:
                    points["first_ls_la"] = point
                ls_calls += 1
            bash_calls += 1
    return {"told": True, "told_at": start["id"], **points, "ls_calls_after": ls_calls, "bash_calls_after": bash_calls,
            "compactions_after": turns[-1]["compactions_at_end"] - start["compactions_at_start"]}


def score_session(lines: Iterable[str | dict[str, Any]], tag: str, compaction_log: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
    turns, totals = read_stream(lines)
    entries = list(compaction_log)
    log_counts = log_by_turn(turns, entries)
    told: dict[str, int] = {}
    for index, turn in enumerate(turns):
        for rule in turn["tells"]:
            told.setdefault(rule, index)
    scores = {rule: {"told_at": turns[told[rule]]["id"] if rule in told else None, "applies": 0, "obeyed": 0, "by_task": {},
                     "by_compactions": {}} for rule in RULES}
    rows: dict[str, dict[str, Any]] = {}
    usage: Counter[str] = Counter()
    for index, turn in enumerate(turns):
        row = {"kind": turn["kind"], "bash_calls": len(turn["bash"]), "python_with_B": 0, "python_without_B": 0, "python_c": 0,
               "ls": {LS_ONE: 0, LS_LA: 0, LS_OTHER: 0}, "tag_echoed": False, "compactions": turn["compactions"],
               "compactions_in_log": log_counts[index]}
        for call in turn["bash"]:
            for segment in split_commands(call["command"]):
                name, args = program(words(segment))
                if _PYTHON.match(name):
                    has_b, runs_c = python_options(args)
                    row["python_with_B" if has_b else "python_without_B"] += 1
                    row["python_c"] += runs_c
                    if index >= told.get("python_B", len(turns)):
                        _mark(scores["python_B"], turn["id"], has_b, call["compactions"])
                elif name == "ls":
                    form = ls_form(args)
                    row["ls"][form] += 1
                    if index >= told.get("ls_one", len(turns)):
                        _mark(scores["ls_one"], turn["id"], form == LS_ONE, call["compactions"])
                elif name in _ECHOES and tag and (tag in segment or tag in call["output"]):
                    row["tag_echoed"] = True
        # The tag is due "before you finish": a task that never finished (no result line, as when a session is stopped mid-task)
        # is not scored for it.
        if turn["kind"] == "task" and index > told.get("session_tag", len(turns)) and turn["result"] is not None:
            _mark(scores["session_tag"], turn["id"], row["tag_echoed"], turn["compactions_at_end"])
        row["tag_mentioned"] = bool(tag) and any(tag in text for text in [*turn["texts"], *(call["command"] for call in turn["bash"])])
        result = turn["result"] or {}
        row |= {"result": result.get("subtype"), "is_error": result.get("is_error"), "cost_usd": result.get("cost_usd"),
                "num_turns": result.get("num_turns"), "permission_denials": result.get("permission_denials")}
        usage.update({key: int(value) for key, value in (result.get("usage") or {}).items() if isinstance(value, int)})
        rows[turn["id"]] = row
    read, fresh = usage["cache_read_input_tokens"], usage["input_tokens"] + usage["cache_creation_input_tokens"]
    return {"valid": bool(turns) and all(turn["result"] for turn in turns), "turns": len(turns),
            "turns_with_result": sum(turn["result"] is not None for turn in turns), "scores": scores,
            "override": override(turns, told.get("ls_one"), log_counts), "tasks": rows,
            "compactions": {"stream": totals["compactions"], "log_pre": sum(entry.get("event") == "PreCompact" for entry in entries),
                            "log_post": sum(entry.get("event") == "PostCompact" for entry in entries)},
            # total_cost_usd is the resumed session's running total, so the session's cost is the largest one reported, not their sum
            "cost_usd": round(max((row["cost_usd"] or 0 for row in rows.values()), default=0.0), 6), "usage": dict(usage),
            "cache_read_share": read / (read + fresh) if read + fresh else None, **{key: totals[key] for key in
                                                                                   ("tools", "bash_calls", "subagent_bash_calls", "broken_lines")}}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize(folder: Path) -> dict[str, Any]:
    """Score one session folder as the driver leaves it, and write its summary.json beside it."""
    folder = Path(folder)
    record = json.loads((folder / "session.json").read_text())
    summary = score_session((folder / "stream.jsonl").read_text().splitlines(), record["tag"], _jsonl(folder / "hooks" / "compaction.jsonl"))
    summary |= {"arm": record.get("arm"), "seed": record.get("seed"), "ended_by": record.get("ended_by")}
    (folder / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", type=Path)
    for folder in parser.parse_args(argv).folders:
        summary = summarize(folder)
        rules = ", ".join(f"{rule} {s['obeyed']}/{s['applies']}" for rule, s in summary["scores"].items())
        first = (summary["override"].get("first_ls_la") or {}).get("turn", "none")
        print(f"{folder}: {summary['arm']} | {rules} | compactions {summary['compactions']['stream']} in the stream, "
              f"{summary['compactions']['log_pre']} logged | first ls -la after the override: {first} | ${summary['cost_usd']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
