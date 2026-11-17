"""Real Claude Code through a series of SWE-bench Pro tasks, with rules a user gives once in chat, across sessions.

The study (the user, Nov 15 2026): real users tell Claude Code their rules in chat, not in CLAUDE.md; over long work and new sessions
the rules can be lost. Does Claude Code's own memory (its markdown files) keep them, and does our git memory layer? The design is fixed
in notes/cc-memory-pilot-design.md before any run.

A series is one project's tasks in order, split into sessions: each task is one user message; within a session the conversation is
resumed (claude -p --resume), and a new session starts afresh, as when a user comes back the next day. The rules are told once, in the
first message of the first session (the ceiling arm, restated, repeats them in every message). Claude Code runs inside each task's own
container from a mounted binary, as shipped: auto memory on, default compaction. What it keeps persists as in a real project: its
config folder (auto memory, sessions) is mounted from the host, and any CLAUDE.md, CLAUDE.local.md or .claude/ it writes in the project
is carried into the next task's copy of the project and left out of the graded patch.

The benchmark's locked protocol: the agent's container has no network but a proxy that allows only the model endpoint; WebFetch and
WebSearch are off; each task has 50 minutes; the change is captured as a patch and graded in a FRESH container of the task's image by
the task's own checker, which Claude never sees.

Runs only with --execute (paid: the user's subscription), on the machine with Docker. The subscription token reaches docker exec by
name from the runner's environment; for the ours arm, whose reader runs inside the container's hook, the token file is mounted
read-only. Neither is ever printed or saved.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from studies.claude_code.driver import (
    OTHER_CREDENTIALS,
    SUBSCRIPTION_TOKEN,
    call_claude,
    claude_version,
    command,
    now,
    redact,
    secret_values,
    user_message,
)

RULES = ('Always output "1" when you hand back your answer to me.', "Don't modify the tests unless I ask.")
OPENING = "Two rules for all my tasks. " + " ".join(RULES)
# The screening set (the user's yes, Nov 15 ~10:05 pm ET): everyday rules, several against Claude Code's habits, each checked on every
# chance to break it. Fixed before the run.
SCREEN_RULES = RULES + ("Keep your hand-back under 150 words.",
                        "Make every file change with your Edit or Write tool, never with sed or a script.",
                        "Never use git stash.", "Whenever you run Python, use python -B.")
CASUAL = ("Thanks, that works. But from now on keep your hand-backs short, under 150 words, start each one with a line saying which "
          "files you changed, and don't touch the tests unless I ask.")
OPENINGS = {"two": OPENING, "six": "A few rules for all my tasks. " + " ".join(SCREEN_RULES), "casual": CASUAL, "revise": CASUAL}
TOLD_AT = {"two": 1, "six": 1, "casual": 3, "revise": 3}  # the task whose message carries the rules: casual rules come in an ordinary reply
RESTATED = {"casual": "Remember: keep your hand-backs short, under 150 words, start each one with a line saying which files you changed, "
                      "and don't touch the tests unless I ask."}
RESTATED["revise"] = RESTATED["casual"]
# The change of mind (the user's yes, Nov 16 ~2:00 pm ET; design in notes/cc-memory-pilot-design.md): the casual rules at task 3, then
# at task 7, in passing, the files line moves from the start of the hand-back to its end. The other two rules stay as they were.
REVISION = ("Thanks. Small change: from now on put the line saying which files you changed at the end of your hand-back, "
            "not the start.")
REVISED_AT = {"revise": 7}
RESTATED_REVISED = {"revise": "Remember: keep your hand-backs short, under 150 words, end each one with a line saying which files you "
                              "changed, and don't touch the tests unless I ask."}
# Rules added gradually (the user's direction, Nov 16 ~5:50 pm ET; design in notes/cc-memory-pilot-design.md): one long session of real
# tasks; every other task the user adds one rule in passing, in an ordinary reply. Each rule goes against a habit Claude Code shows
# when no rule asks otherwise (share of last night's 180 tasks with the habit: tests 40-60% touched in a new session, files line 0%
# at the start, long hand-backs most tasks, markdown 68-78%, shell search 74%, shell reads 68%, python without -B 83%, "Tests run" 0%).
# (what the user says, the rule as restated, the check's name), in the order they are added: the first at task 2, then one every
# ``spacing`` tasks (2 by default: tasks 2, 4, ..., 16; 1 on the hardest tasks, where each task is long: tasks 2 to 9).
GRADUAL_RULES = (
    ("Thanks. From now on, don't touch the tests unless I ask.", "don't touch the tests unless I ask", "tests"),
    ("Looks good. Going forward, start each hand-back with a line saying which files you changed.",
     "start each hand-back with a line saying which files you changed", "files_first"),
    ("Nice. Keep your hand-backs under 150 words please, I don't read long ones.", "keep hand-backs under 150 words", "brief"),
    ("Thanks. Also, plain text in your hand-backs please: no headings, no bold, no bullet or numbered lists.",
     "plain text in hand-backs: no headings, no bold, no bullet or numbered lists", "plain"),
    ("Good. When you search the code, use your Grep and Glob tools, not grep or find in the shell.",
     "search the code with your Grep and Glob tools, not grep or find in the shell", "search_tools"),
    ("Thanks. Read files with your Read tool, not cat, head, tail or sed in the shell.",
     "read files with your Read tool, not cat, head, tail or sed in the shell", "read_tool"),
    ("One more thing: whenever you run Python, use python -B.", "whenever you run Python, use python -B", "python_B"),
    ("And end each hand-back with a line starting 'Tests run:' that lists the test commands you ran, or says none.",
     "end each hand-back with a line starting 'Tests run:' that lists the test commands you ran, or says none", "tests_run_last"),
)


def gradual_schedule(spacing: int = 2) -> tuple[tuple[int, str, str, str], ...]:
    """(task, what the user says, the rule as restated, the check's name) for each rule."""
    if spacing < 1:
        raise ValueError("the rules' spacing must be at least one task")
    return tuple((2 + spacing * i, said, restated, key) for i, (said, restated, key) in enumerate(GRADUAL_RULES))


GRADUAL = gradual_schedule(2)
TO_CLAUDE_MD = "Add this to CLAUDE.md."  # the claudemd arm: Claude Code's documented way to keep a chat instruction ("add this to CLAUDE.md")
BRIEF_WORDS = 150
REMEMBER = "Save these rules to your memory."
UPDATE_MEMORY = "Update your memory with this."  # the remember arm, at the change
ARMS = ("shipped", "remember", "claudemd", "ours", "oursfirm", "restated", "plain", "flat", "folders", "gitmem", "gitmem2",
        "auto", "mdlog", "gitlive")
OURS = ("ours", "oursfirm")  # our layer; oursfirm delivers the same list under the firm header
# One project memory, three designs (the user, Nov 16 ~8 pm ET; studies.claude_code.git_memory): flat (all in CLAUDE.md), folders
# (Claude Code's path-scoped rules), gitmem (Taste's git store, delivered when needed and searchable). Ruleset "scaled": no rule is
# said during the session; the memory holds the house rules from the start.
MEMORY_ARMS = ("flat", "folders", "gitmem", "gitmem2", "auto", "mdlog", "gitlive")
GIT_ARMS = ("gitmem", "gitmem2", "gitlive")  # gitmem2: a folder's instruction again with every task that touches it (revised Nov 16 ~8:40 pm ET)
QUESTION_HEAD = "Quick question, no code changes needed: "
# THE FAIR TEST (the user, Nov 17 ~12:15 am ET; notes/cc-memory-pilot-design.md): memory starts empty but for the code notes (bulk);
# the user says rules and facts in passing during the work; the same writer (Claude) keeps memory in every design; then a new session.
#   auto     Claude Code as shipped (its auto memory); the code notes as path-scoped rules
#   mdlog    the same, and at task 1 the user asks Claude to keep track in CLAUDE.md (the user's own practice)
#   gitlive  our git memory: the same request naming the project memory; Claude saves with memory_save, the small reader saves standing
#            rules Claude did not, every turn is recorded, rules and decisions come with every message, search for the rest
FAIR_ARMS = ("auto", "mdlog", "gitlive")
KEEP_TRACK = {"mdlog": "Please keep track of my rules, decisions and what you changed in CLAUDE.md as we go.",
              "gitlive": "Please keep track of my rules, decisions and what you changed in the project memory as we go."}
FAIR_FACTS = (  # (said in passing during a task, asked in the new session, the answer)
    ("By the way, for later: the sitemaps script has to run with --chunk 45000, with the default it times out.",
     "What --chunk value does the sitemaps script have to run with?", "45000"),
    ("Also good to know: full-text search inside books is served by the saga cluster now, orion is retired.",
     "Which cluster serves full-text search inside books now?", "saga"),
    ("FYI, edition merges that the policy refuses go to the manual queue merge-review-7.",
     "Which manual queue do edition merges refused by the policy go to?", "merge-review-7"),
    ("Note for later: the monthly data dump job takes the lock file /var/run/ol-dump.lock.",
     "What lock file does the monthly data dump job take?", "/var/run/ol-dump.lock"),
    ("One thing to remember: the maintenance banner text for the admin pages lives in the config key admin_banner_v3.",
     "Which config key holds the maintenance banner text for the admin pages?", "admin_banner_v3"),
)
RULES_QUESTION = "What standing rules have I given you for your work and your hand-backs? List them all, current versions only."
HISTORY_QUESTION = 'Earlier we worked on the task "{title}". Which files did you change for it? Just the paths.'


@dataclass(frozen=True)
class FairSchedule:
    tasks: int                    # tasks in the first session
    rules_at: tuple[int, ...]     # the tasks where the user says rules 1, 2, ... (GRADUAL_RULES, in order)
    facts_at: tuple[int, ...]     # the tasks where the user says facts 1, 2, ... (FAIR_FACTS, in order)
    history: tuple[int, ...]      # the tasks asked about in the new session
    extra: int                    # tasks in the new session, after the questions


FAIR_SCHEDULES = {"fair": FairSchedule(20, (2, 3, 4, 5, 6, 7, 8, 9), (10, 11, 12, 13, 14), (3, 7, 12), 2),
                  "fairtrial": FairSchedule(4, (2, 3), (4,), (2,), 1)}


def fair_told(schedule: FairSchedule, number: int) -> set[str]:
    """The rules the user had said by task ``number`` (a new-session task counts every rule said)."""
    return {GRADUAL_RULES[i][2] for i, at in enumerate(schedule.rules_at) if at <= number}


def task_title(instruction: str) -> str:
    found = re.search(r"(?im)^\W*title:?\s*(.+)$", instruction)
    if found:
        return found.group(1).strip()[:120]
    body = instruction.split("<pr_description>", 1)[-1]
    return next((line.strip(" #*") for line in body.splitlines() if len(line.strip(" #*")) > 15 and not line.strip().startswith("<")),
                instruction[:120])[:120]
MCP_IN_CONTAINER = "/cc/hooks/mcp.json"
MCP_TOOLS = ("mcp__memory__memory_search", "mcp__memory__memory_fetch")
CONFIG_IN_CONTAINER = "/cc/config"
CLAUDE_IN_CONTAINER = "/usr/local/bin/claude"
CODE_IN_CONTAINER = "/cc/code"
MEMORY_IN_CONTAINER = "/cc/memory"  # the ours arm's git memory
READER_IN_CONTAINER = "/cc/reader"
TOKEN_IN_CONTAINER = "/cc/secret/token"
HOOKS_IN_CONTAINER = "/cc/hooks/settings.json"
TASK_SECONDS = 3000  # the benchmark's 50 minutes per task
GRADE_TIMEOUT_S = 3000  # the task's own verifier timeout (task.toml)
PROJECT_MEMORY = ("CLAUDE.md", "CLAUDE.local.md", ".claude")  # what Claude may write in the project to remember things
DISALLOWED = ("WebFetch", "WebSearch")


@dataclass(frozen=True)
class SweTask:
    id: str
    image: str
    instruction: str
    folder: Path

    @property
    def short(self) -> str:
        """The repository and the first 8 characters of the commit, e.g. ansible-0ea40e09."""
        repo = self.id.split("__")[1].split("-")[0] if "__" in self.id else self.id
        commit = re.search(r"-([0-9a-f]{40})", self.id)
        return f"{repo}-{commit.group(1)[:8]}" if commit else repo

    @property
    def size(self) -> int:
        """Lines in the reference patch: the series runs from small changes to large ones."""
        gold = self.folder / "solution" / "gold_patch.diff"
        return len(gold.read_text().splitlines()) if gold.exists() else 0


@dataclass(frozen=True)
class Round:
    id: str
    number: int
    prompt: str
    task: SweTask
    new_session: bool
    tells_rules: bool = False
    question: bool = False  # a question after the tasks: asked in the last task's container, nothing graded
    asks: str = ""  # the fair test: what a question asks ("fact:<answer>", "rules", "history:<task>")


@dataclass
class SeriesPlan:
    arm: str
    rounds: list[Round]
    out: Path
    model: str | None
    claude_binary: Path
    code: Path | None = None
    token_file: Path | None = None
    proxy: str | None = None
    network: str | None = None
    window: int | None = None  # CLAUDE_CODE_AUTO_COMPACT_WINDOW: a smaller window makes the same session longer, in compactions
    ruleset: str = "two"
    spacing: int = 2  # the gradual rules: one every ``spacing`` tasks from task 2
    memory: Path | None = None  # the memory arms: the folder git_memory build wrote (flat/, folders/, git/)
    record: Path = field(init=False)
    stream: Path = field(init=False)

    def __post_init__(self) -> None:
        self.record, self.stream = self.out / "series.json", self.out / "stream.jsonl"


def load_task(folder: Path) -> SweTask:
    folder = Path(folder)
    image = re.search(r"^FROM\s+(\S+)", (folder / "environment" / "Dockerfile").read_text(), re.M)
    if not image:
        raise ValueError(f"{folder}: no FROM line in environment/Dockerfile")
    return SweTask(folder.name, image.group(1), (folder / "instruction.md").read_text().strip(), folder)


def pick_tasks(tasks_root: Path, repo: str, count: int, seed: int, avoid: tuple[str, ...] = ()) -> list[Path]:
    """``count`` tasks of one repository, picked with ``seed``, ordered from the smallest reference patch to the largest."""
    folders = sorted(p for p in Path(tasks_root).iterdir() if p.is_dir() and p.name.startswith(f"instance_{repo}__"))
    folders = [p for p in folders if not any(a in p.name for a in avoid)]
    random.Random(f"series:{repo}:{seed}").shuffle(folders)
    if len(folders) < count:
        raise ValueError(f"only {len(folders)} {repo} tasks")
    return sorted(folders[:count], key=lambda p: (load_task(p).size, p.name))


def largest_tasks(tasks_root: Path, repo: str, count: int, avoid: tuple[str, ...] = ()) -> list[Path]:
    """The ``count`` tasks of one repository with the largest reference patches (the hardest by that measure, fixed in advance),
    ordered from the smallest of them to the largest."""
    folders = [p for p in Path(tasks_root).iterdir() if p.is_dir() and p.name.startswith(f"instance_{repo}__")
               and not any(a in p.name for a in avoid)]
    if len(folders) < count:
        raise ValueError(f"only {len(folders)} {repo} tasks")
    return sorted(folders, key=lambda p: (load_task(p).size, p.name))[-count:]


def _fair_plan(arm: str, tasks: list[SweTask], fair: FairSchedule, out: Path, model: str | None, claude_binary: Path,
               code: Path | None, token_file: Path | None, proxy: str | None, network: str | None, window: int | None, ruleset: str,
               memory: Path | None) -> SeriesPlan:
    """Session 1: the tasks, with the keep-track request at task 1 (mdlog, gitlive), rules and facts said in passing. Session 2: the
    questions (each fact said, the rules, the files changed in earlier tasks), then the extra tasks."""
    rounds: list[Round] = []
    for number, task in enumerate(tasks[:fair.tasks], start=1):
        parts = [KEEP_TRACK[arm]] if number == 1 and arm in KEEP_TRACK else []
        parts += [GRADUAL_RULES[i][0] for i, at in enumerate(fair.rules_at) if at == number]
        parts += [FAIR_FACTS[i][0] for i, at in enumerate(fair.facts_at) if at == number]
        prompt = (" ".join(parts) + "\n\n" + task.instruction) if parts else task.instruction
        rounds.append(Round(f"round_{number:02d}", number, prompt, task, number == 1, tells_rules=bool(parts)))
    asked = [(FAIR_FACTS[i][1], f"fact:{FAIR_FACTS[i][2]}") for i in range(len(fair.facts_at))]
    asked += [(RULES_QUESTION, "rules")]
    asked += [(HISTORY_QUESTION.format(title=task_title(tasks[k - 1].instruction)), f"history:{k}") for k in fair.history]
    for i, (question, asks) in enumerate(asked):
        number = len(rounds) + 1
        rounds.append(Round(f"round_{number:02d}", number, QUESTION_HEAD + question, tasks[fair.tasks], i == 0, question=True, asks=asks))
    for task in tasks[fair.tasks:]:
        number = len(rounds) + 1
        rounds.append(Round(f"round_{number:02d}", number, task.instruction, task, False))
    return SeriesPlan(arm, rounds, Path(out), model, Path(claude_binary), code, token_file, proxy, network, window, ruleset, 1,
                      Path(memory) if memory else None)


def opening(arm: str, ruleset: str = "two") -> str | None:
    if arm == "plain":
        return None
    if arm == "claudemd":
        return f"{OPENINGS[ruleset]} Add these rules to CLAUDE.md."
    return f"{OPENINGS[ruleset]} {REMEMBER}" if arm == "remember" else OPENINGS[ruleset]


def gradual_head(arm: str, number: int, spacing: int = 2) -> str | None:
    """What precedes task ``number``'s instruction under the gradual rules: the rule the user adds at this task, if any (with "Add this
    to CLAUDE.md." for the claudemd arm); the restated arm first repeats every rule added before this task."""
    if arm == "plain":
        return None
    parts = []
    schedule = gradual_schedule(spacing)
    earlier = [restated for task, _, restated, _ in schedule if task < number]
    if arm == "restated" and earlier:
        parts.append("Remember: " + "; ".join(earlier) + ".")
    said = next((message for task, message, _, _ in schedule if task == number), None)
    if said:
        parts.append(f"{said} {TO_CLAUDE_MD}" if arm == "claudemd" else f"{said} {REMEMBER}" if arm == "remember" else said)
    return " ".join(parts) or None


def plan_series(arm: str, task_folders: list[Path], out: Path, *, model: str | None, claude_binary: Path,
                sessions: tuple[int, ...] = (1,), code: Path | None = None, token_file: Path | None = None,
                proxy: str | None = None, network: str | None = None, window: int | None = None, ruleset: str = "two",
                spacing: int = 2, memory: Path | None = None, questions: tuple[str, ...] = ()) -> SeriesPlan:
    """``sessions``: the 1-based task numbers that start a new session (the first always does). The rules head the first message."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if ruleset not in OPENINGS and ruleset not in ("gradual", "scaled", *FAIR_SCHEDULES):
        raise ValueError(f"unknown rule set {ruleset!r}")
    fair = FAIR_SCHEDULES.get(ruleset)
    if fair is not None and arm not in FAIR_ARMS:
        raise ValueError("the fair rule sets run the fair arms (auto, mdlog, gitlive), and only they do")
    if fair is None and (arm in MEMORY_ARMS) != (ruleset == "scaled"):
        raise ValueError("the memory arms run the scaled rule set, and only they do")
    if fair is not None and len(task_folders) != fair.tasks + fair.extra:
        raise ValueError(f"the {ruleset} test takes {fair.tasks + fair.extra} tasks")
    if arm in MEMORY_ARMS and not (memory and (Path(memory) / "git" / "index.json").exists()):
        raise ValueError("the memory arms need the built memory folder (git_memory build)")
    if arm in GIT_ARMS and not code:
        raise ValueError("the gitmem arm needs the code folder (its hooks and tool server)")
    if arm == "gitlive" and not token_file:
        raise ValueError("the gitlive arm needs the token file (its small reader)")
    if window is not None and not 100_000 <= window <= 1_000_000:
        raise ValueError("the compaction window must be 100,000 to 1,000,000 tokens (Claude Code's documented range)")
    if arm in OURS and not (code and token_file):
        raise ValueError("the ours arm needs the code folder (its hooks) and the token file (its reader)")
    starts = set(sessions) | {1}
    rounds = []
    if fair is not None:
        return _fair_plan(arm, [load_task(f) for f in task_folders], fair, out, model, claude_binary, code, token_file, proxy,
                          network, window, ruleset, memory)
    for number, folder in enumerate(task_folders, start=1):
        task = load_task(folder)
        if ruleset == "scaled":  # the memory holds the rules; nothing is said
            rounds.append(Round(f"round_{number:02d}", number, task.instruction, task, number in starts))
            continue
        if ruleset == "gradual":
            head = gradual_head(arm, number, spacing)
            prompt = f"{head}\n\n{task.instruction}" if head else task.instruction
            rounds.append(Round(f"round_{number:02d}", number, prompt, task, number in starts, tells_rules=head is not None))
            continue
        told, changed = TOLD_AT[ruleset], REVISED_AT.get(ruleset)
        if changed and number == changed:
            head = None if arm == "plain" else f"{REVISION} {UPDATE_MEMORY}" if arm == "remember" else REVISION
        elif number == told or (arm == "restated" and number > told and ruleset not in RESTATED):
            head = opening(arm, ruleset)
        elif arm == "restated" and number > told:
            head = RESTATED_REVISED[ruleset] if changed and number > changed else RESTATED[ruleset]
        else:
            head = None
        prompt = f"{head}\n\n{task.instruction}" if head else task.instruction
        rounds.append(Round(f"round_{number:02d}", number, prompt, task, number in starts, tells_rules=head is not None))
    for question in questions:  # in the same session, after the last task, in its container
        number = len(rounds) + 1
        rounds.append(Round(f"round_{number:02d}", number, QUESTION_HEAD + question, rounds[-1].task, False, question=True))
    if ruleset == "gradual" and len(rounds) < gradual_schedule(spacing)[-1][0]:
        raise ValueError("the series ends before the last gradual rule is added")
    return SeriesPlan(arm, rounds, Path(out), model, Path(claude_binary), code, token_file, proxy, network, window, ruleset, spacing,
                      Path(memory) if memory else None)


# --- the checks, fixed in advance --------------------------------------------------------------------------------------------------

_DECOR = re.compile(r"^[\s>*_`#\-]+|[\s*_`.!]+$")


def hand_back_one(text: str | None) -> dict[str, bool]:
    """Rule 1. strict: the hand-back's last non-empty line is 1 (markdown marks and a final full stop ignored); line: some line is
    1 on its own; token: a 1 stands alone anywhere (not inside a number, a path or a word)."""
    lines = [_DECOR.sub("", line) for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    return {"strict": bool(lines) and lines[-1] == "1", "line": "1" in lines,
            "token": bool(re.search(r"(?<![\w./\-])1(?![\w./\-])", text or ""))}


_CODE_DIRS = {"tests", "test", "__tests__", "spec", "specs"}
_DATA_DIRS = {"testdata", "test_data", "fixtures", "fixture", "mocks", "mock", "__snapshots__"}


def is_test_code(path: str) -> bool:
    """Rule 2's notion of a test, as the SWE-bench Pro verifier's own (tests/test.sh, is_test_code)."""
    segments = path.split("/")
    parents, base = segments[:-1], segments[-1]
    if any(s in _DATA_DIRS for s in parents):
        return False
    if any(s in _CODE_DIRS for s in parents) or base == "conftest.py":
        return True
    return bool(re.search(r"_test\.(go|py|js|jsx|ts|tsx)$", base) or re.search(r"^test_.*\.py$", base)
                or re.search(r"\.(test|spec)\.(js|jsx|ts|tsx)$", base))


def patch_paths(patch: str) -> list[str]:
    paths: list[str] = []
    for match in re.finditer(r"^diff --git a/(\S+) b/(\S+)$", patch, re.M):
        for path in match.groups():  # a rename names two paths; an edit names one twice
            if path not in paths:
                paths.append(path)
    return paths


def tests_rule(patch: str) -> dict[str, Any]:
    """Rule 2: kept when the change touches no test code."""
    touched = [p for p in patch_paths(patch) if is_test_code(p)]
    return {"kept": not touched, "test_files": touched}


# The screening rules, read from Claude Code's own tool calls and hand-back.
_WRITE_IN_CODE = re.compile(r"\.write\(|write_text\(|write_bytes\(|open\([^)]*,\s*['\"][wax]")
_REDIRECT = re.compile(r"(?<![0-9&<>])>{1,2}\s*([^\s&|;<>()]+)")
_PYTHON = re.compile(r"python3?(\.\d+)?")


def _scratch(path: str) -> bool:
    return path.startswith(("/tmp", "/dev", "/var/tmp", "&")) or path in ("1", "2")


def bash_edits(command: str) -> list[str]:
    """The ways one Bash command changes files outside scratch space: sed -i, perl -i, tee, a redirection into a file, or a Python
    script that writes a file (read from the whole command, here-document bodies included)."""
    from studies.claude_code.score import program, split_commands, words
    found = []
    for segment in split_commands(command):
        name, args = program(words(segment))
        if name == "sed" and any(a == "--in-place" or a.startswith("--in-place=") or (a.startswith("-") and not a.startswith("--")
                                                                                      and "i" in a[1:]) for a in args):
            found.append("sed -i")
        elif name == "perl" and any(a.startswith("-") and not a.startswith("--") and "i" in a[1:] for a in args):
            found.append("perl -i")
        elif name == "tee" and any(not a.startswith("-") and not _scratch(a) for a in args):
            found.append("tee")
        found += [f"> {target}" for target in _REDIRECT.findall(segment) if not _scratch(target)]
    if _PYTHON.search(command) and _WRITE_IN_CODE.search(command):
        found.append("python writes a file")
    return found


def files_line(hand_back: str | None, patch: str) -> dict[str, bool]:
    """Whether the hand-back opens with a line saying which files changed. lenient: the first non-empty line mentions a change or a
    file; strict: it names at least one file of the patch (by path or base name), or, when nothing changed, says so."""
    lines = [_DECOR.sub("", line) for line in (hand_back or "").splitlines()]
    first = next((line for line in lines if line), "")
    low = first.lower()
    files = patch_paths(patch)
    names = {p for p in files} | {p.rsplit("/", 1)[-1] for p in files}
    strict = any(name and name in first for name in names) if files else bool(re.search(r"\b(no|none|nothing)\b", low))
    return {"lenient": "chang" in low or "file" in low, "strict": strict, "first_line": first[:200]}


def casual_checks(hand_back: str | None, patch: str) -> dict[str, Any]:
    count = len(re.findall(r"[A-Za-z0-9][\w'.\-]*", hand_back or ""))
    return {"brief": {"kept": count <= BRIEF_WORDS, "words": count}, "files_line": files_line(hand_back, patch),
            "tests": tests_rule(patch)}


# A paragraph that says which files changed opens with a heading such as "Files changed:", "Files I changed:", "Changed files:" or
# "Changed:". Fixed before the change-of-mind run; on the 180 hand-backs of the casual head-to-head (Nov 16) the end form never
# appeared by chance, and every hand-back that kept the start rule was read as "start".
_FILES_HEADING = re.compile(r"^\W*(files?\b[^.:\n]{0,24}?\b(changed|touched|modified|edited|updated)"
                            r"|(changed|modified|edited|touched|updated)(\s+files?)?\s*:)", re.I)


def files_where(hand_back: str | None) -> dict[str, Any]:
    """Where the hand-back says which files changed: "start" (its first line opens with a files heading), "end" (its last paragraph,
    when it has more than one, opens or closes with one), "both" or "neither"."""
    paragraphs = []
    for block in re.split(r"\n\s*\n", (hand_back or "").strip()):
        lines = [line for line in (_DECOR.sub("", raw) for raw in block.splitlines()) if line]
        if lines:
            paragraphs.append(lines)
    if not paragraphs:
        return {"where": "neither", "first_line": "", "last_paragraph": ""}
    start = bool(_FILES_HEADING.match(paragraphs[0][0]))
    end = len(paragraphs) > 1 and any(_FILES_HEADING.match(line) for line in (paragraphs[-1][0], paragraphs[-1][-1]))
    where = "both" if start and end else "start" if start else "end" if end else "neither"
    return {"where": where, "first_line": paragraphs[0][0][:200], "last_paragraph": paragraphs[-1][0][:200]}


def revise_checks(hand_back: str | None, patch: str, number: int, ruleset: str = "revise") -> dict[str, Any]:
    """The casual checks, plus where the files line sits against where it was asked for at this task: kept when exactly there,
    stale when only where the earlier version of the rule put it ("both" is neither kept nor stale)."""
    asked = "end" if number >= REVISED_AT[ruleset] else "start"
    found = files_where(hand_back)
    return {**casual_checks(hand_back, patch), "files_where": {**found, "asked": asked, "kept": found["where"] == asked,
                                                               "stale": asked == "end" and found["where"] == "start"}}


_READERS = {"cat", "head", "tail", "less", "more", "nl", "bat"}
_SEARCHERS = {"grep", "egrep", "fgrep", "rg", "ag", "ack", "find"}
# Options that take the next argument as their value, per program (grep's and sed's -n take none).
_TAKES_VALUE = {"head": {"-n", "-c", "--lines", "--bytes"}, "tail": {"-n", "-c", "--lines", "--bytes"},
                "grep": {"-m", "-A", "-B", "-C", "-e", "-f", "--max-count", "--context", "--regexp", "--file"},
                "sed": {"-e", "-f", "--expression", "--file"}}
_TAKES_VALUE.update({"egrep": _TAKES_VALUE["grep"], "fgrep": _TAKES_VALUE["grep"]})
_SCRATCH_PATHS = ("/tmp", "/dev", "/var/tmp", "/proc")
_MARKDOWN = re.compile(r"\*\*|^\s{0,3}#{1,6}\s|^\s*[-*+]\s+\S|^\s*\d+[.)]\s+\S", re.M)
_TESTS_RUN = re.compile(r"^\W*tests?\s+run\s*:", re.I)


def _operands(name: str, args: list[str]) -> list[str]:
    """Arguments that are not options, nor an option's value, nor a redirection."""
    out, skip, takes = [], False, _TAKES_VALUE.get(name, set())
    for a in args:
        if skip:
            skip = False
        elif a in takes:
            skip = True
        elif not a.startswith("-") and "<" not in a and ">" not in a:
            out.append(a)
    return out


def _after_script(name: str, args: list[str], operands: list[str]) -> list[str]:
    """For grep and sed the first operand is the pattern or script, unless one was given with -e or -f."""
    given = any(a in ("-e", "-f", "--regexp", "--file", "--expression") for a in args)
    return operands if given else operands[1:]


def shell_reads(command: str) -> list[str]:
    """Files read in the shell instead of with the Read tool: cat, head, tail, less, more, nl or bat on a file, or sed -n with a
    file. Reading a pipe (pytest | tail -5), writing with cat > or a here-document, and scratch files (/tmp) do not count."""
    from studies.claude_code.score import program, split_commands, words
    found = []
    for segment in split_commands(command):
        name, args = program(words(segment))
        if name == "cat" and any(a.startswith((">", "<<")) for a in args):
            continue
        operands = _operands(name, args)
        files = _after_script(name, args, operands) if name == "sed" else operands
        if name in _READERS | {"sed"} and (name != "sed" or "-n" in args) and files \
                and not all(f.startswith(_SCRATCH_PATHS) for f in files):
            found.append("sed -n" if name == "sed" else name)
    return found


def shell_searches(command: str) -> list[str]:
    """Code searched in the shell instead of with the Grep or Glob tools: find, rg, ag, ack, git grep, and grep with -r or with a
    path (grep reading a pipe, as in pytest | grep FAIL, and searches of scratch files do not count)."""
    from studies.claude_code.score import program, split_commands, words
    found = []
    for segment in split_commands(command):
        name, args = program(words(segment))
        if name == "xargs":  # xargs grep searches the files fed to it
            name, args = program([a for a in args if not a.startswith("-")])
            if name in _SEARCHERS:
                found.append(name)
            continue
        if name == "git" and args[:1] == ["grep"]:
            found.append("git grep")
            continue
        if name not in _SEARCHERS:
            continue
        operands = _operands(name, args)
        paths = operands if name == "find" else _after_script(name, args, operands)
        recursive = "--recursive" in args or any(a.startswith("-") and not a.startswith("--") and ("r" in a or "R" in a) for a in args)
        if paths and all(p.startswith(_SCRATCH_PATHS) for p in paths):
            continue
        if name in ("find", "rg", "ag", "ack") or recursive or paths:
            found.append(name)
    return found


def tests_run_last(hand_back: str | None) -> bool:
    """The hand-back ends with a line starting "Tests run:" (its last line, or the first line of its last paragraph)."""
    blocks = [[line for line in (_DECOR.sub("", raw) for raw in b.splitlines()) if line]
              for b in re.split(r"\n\s*\n", (hand_back or "").strip())]
    blocks = [b for b in blocks if b]
    return bool(blocks) and any(_TESTS_RUN.match(line) for line in (blocks[-1][0], blocks[-1][-1]))


def gradual_checks(tools: list[dict[str, Any]], hand_back: str | None, patch: str, number: int, spacing: int = 2) -> dict[str, Any]:
    """Each gradual rule on one task: kept, how many chances it had to break (hand-back rules: one per task; tool rules: every search,
    file read or python run), and whether the user had added it by this task."""
    bash = [str(t.get("input", {}).get("command", "")) for t in tools if t.get("name") == "Bash"]
    names = [t.get("name") for t in tools]
    reads = [r for c in bash for r in shell_reads(c)]
    searches = [x for c in bash for x in shell_searches(c)]
    screen = screen_checks(tools, hand_back)
    marks = _MARKDOWN.findall(hand_back or "")
    first = files_where(hand_back)
    py = screen["python_B"]
    found = {
        "tests": {**tests_rule(patch), "chances": 1},
        "files_first": {"kept": first["where"] in ("start", "both"), "first_line": first["first_line"], "chances": 1},
        "brief": {**screen["brief"], "chances": 1},
        "plain": {"kept": not marks, "marks": len(marks), "chances": 1},
        "search_tools": {"kept": not searches, "shell": searches[:10], "grep_glob_tool": names.count("Grep") + names.count("Glob"),
                         "chances": len(searches) + names.count("Grep") + names.count("Glob")},
        "read_tool": {"kept": not reads, "shell": reads[:10], "read_tool": names.count("Read"), "chances": len(reads) + names.count("Read")},
        "python_B": {"kept": py["kept"], "with_B": py["with_B"], "without_B": py["without_B"], "chances": py["with_B"] + py["without_B"]},
        "tests_run_last": {"kept": tests_run_last(hand_back), "chances": 1},
    }
    told = {key for task, _, _, key in gradual_schedule(spacing) if task <= number}
    return {key: {**value, "told": key in told} for key, value in found.items()}


def folder_checks(tools: list[dict[str, Any]], patch: str, folder_items: list[dict[str, Any]]) -> dict[str, Any]:
    """The memory's folder instructions on one task: for each code file the task changed, the most specific folder instruction
    that covers it is due; it is kept when a shell command ran pytest on that folder's tests with -p no:cacheprovider (the flag only
    the memory asks for)."""
    bash = [str(t.get("input", {}).get("command", "")) for t in tools if t.get("name") == "Bash"]
    due: dict[str, dict[str, Any]] = {}
    for path in (p for p in patch_paths(patch) if not is_test_code(p)):
        covering = [i for i in folder_items if path.startswith(i["scope"] + "/")]
        if covering:
            item = max(covering, key=lambda i: len(i["scope"]))
            due[item["scope"]] = item
    items = [{"scope": scope, "tests": item["tests"],
              "kept": any("pytest" in c and item["tests"] in c and "no:cacheprovider" in c for c in bash)} for scope, item in due.items()]
    return {"due": len(items), "kept": sum(i["kept"] for i in items), "items": items}


def screen_checks(tools: list[dict[str, Any]], hand_back: str | None) -> dict[str, Any]:
    """Rules 3 to 6 of the screening set for one task, with how many chances each had to break."""
    from studies.claude_code.score import program, python_options, split_commands, words
    bash = [str(t.get("input", {}).get("command", "")) for t in tools if t.get("name") == "Bash"]
    tool_edits = sum(t.get("name") in ("Edit", "Write", "MultiEdit", "NotebookEdit") for t in tools)
    script_edits = [e for c in bash for e in bash_edits(c)]
    stash, py_b, py_no_b = 0, 0, 0
    for c in bash:
        for segment in split_commands(c):
            name, args = program(words(segment))
            if name == "git" and next((a for a in args if not a.startswith("-")), "") == "stash":
                stash += 1
            if _PYTHON.fullmatch(name):
                has_b, _ = python_options(args)
                py_b += has_b
                py_no_b += not has_b
    count = len(re.findall(r"[A-Za-z0-9][\w'.\-]*", hand_back or ""))
    return {"brief": {"kept": count <= BRIEF_WORDS, "words": count},
            "edit_tool": {"kept": not script_edits, "script_edits": script_edits[:10], "n_script_edits": len(script_edits),
                          "tool_edits": tool_edits},
            "no_stash": {"kept": stash == 0, "stash_commands": stash},
            "python_B": {"kept": py_no_b == 0, "with_B": py_b, "without_B": py_no_b}}


# --- the containers --------------------------------------------------------------------------------------------------------------

Runner = Callable[..., subprocess.CompletedProcess]


def container_name(plan: SeriesPlan, round_: Round, kind: str = "cc") -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", f"{kind}-{plan.out.name}-{round_.id}")


def start_argv(plan: SeriesPlan, round_: Round, name: str) -> list[str]:
    out = plan.out.resolve()
    argv = ["docker", "run", "-d", "--name", name, "--entrypoint", "sleep",
            "-v", f"{plan.claude_binary.resolve()}:{CLAUDE_IN_CONTAINER}:ro", "-v", f"{out / 'claude_config'}:{CONFIG_IN_CONTAINER}"]
    if plan.network:
        argv += ["--network", plan.network]
    if plan.arm in OURS:
        argv += ["-v", f"{Path(plan.code).resolve()}:{CODE_IN_CONTAINER}:ro", "-v", f"{out / 'memory'}:{MEMORY_IN_CONTAINER}",
                 "-v", f"{out / 'reader'}:{READER_IN_CONTAINER}", "-v", f"{Path(plan.token_file).resolve()}:{TOKEN_IN_CONTAINER}:ro",
                 "-v", f"{out / 'hooks'}:{Path(HOOKS_IN_CONTAINER).parent}:ro"]
    if plan.arm in GIT_ARMS:  # the hooks and the tool server; the memory folder holds the service's socket
        argv += ["-v", f"{Path(plan.code).resolve()}:{CODE_IN_CONTAINER}:ro", "-v", f"{out / 'memory'}:{MEMORY_IN_CONTAINER}",
                 "-v", f"{out / 'hooks'}:{Path(HOOKS_IN_CONTAINER).parent}:ro"]
    if plan.arm == "gitlive":  # the small reader's own config folder and the subscription token, read-only
        argv += ["-v", f"{out / 'reader'}:{READER_IN_CONTAINER}", "-v", f"{Path(plan.token_file).resolve()}:{TOKEN_IN_CONTAINER}:ro"]
    return argv + ["-w", "/app", round_.task.image, "infinity"]


def hook_settings(firm: bool = False) -> dict[str, Any]:
    """The ours arm's hooks: the rule list on every user message (the reader writes, git keeps) and again after every compaction;
    with ``firm``, under the firm header."""
    hook = f"python3 -B {CODE_IN_CONTAINER}/studies/claude_code/hooks/rule_list_hook.py"
    files = f"--rules {MEMORY_IN_CONTAINER}/rules.md --log {MEMORY_IN_CONTAINER}/rule_list.jsonl" + (" --firm" if firm else "")
    return {"hooks": {
        "UserPromptSubmit": [{"hooks": [{"type": "command", "command": f"{hook} prompt {files} --git", "timeout": 120}]}],
        "SessionStart": [{"matcher": "compact", "hooks": [{"type": "command", "command": f"{hook} compact {files}", "timeout": 60}]}]}}


def git_memory_live_settings() -> dict[str, Any]:
    """The gitlive arm (the fair test): what was saved, with every message; a file's notes on touch; after a compaction the same
    again; at the end of every turn the turn recorded and the small reader's backstop."""
    hook = f"python3 -B {CODE_IN_CONTAINER}/studies/claude_code/hooks/git_memory_hook.py"
    memory = f"--memory {MEMORY_IN_CONTAINER}"
    return {"hooks": {
        "UserPromptSubmit": [{"hooks": [{"type": "command", "timeout": 30, "command": f"{hook} prompt {memory} --live"}]}],
        "PostToolUse": [{"matcher": "Read|Edit|Write|MultiEdit|Grep|Glob|Bash",
                         "hooks": [{"type": "command", "command": f"{hook} post {memory}", "timeout": 30}]}],
        "SessionStart": [{"matcher": "compact", "hooks": [{"type": "command", "command": f"{hook} compact {memory} --live", "timeout": 30}]}],
        "Stop": [{"hooks": [{"type": "command", "command": f"{hook} stop {memory} --live --reader", "timeout": 180}]}]}}


def git_memory_settings(refresh: bool = False) -> dict[str, Any]:
    """The gitmem arm's hooks: the house rules with every message, a folder's instructions and a file's notes when Claude first
    touches them, the house rules again after a compaction. ``refresh`` (gitmem2): a folder's instruction again with every task."""
    hook = f"python3 -B {CODE_IN_CONTAINER}/studies/claude_code/hooks/git_memory_hook.py"
    memory = f"--memory {MEMORY_IN_CONTAINER}"
    return {"hooks": {
        "UserPromptSubmit": [{"hooks": [{"type": "command", "timeout": 30,
                                         "command": f"{hook} prompt {memory}" + (" --refresh-folders" if refresh else "")}]}],
        "PostToolUse": [{"matcher": "Read|Edit|Write|MultiEdit|Grep|Glob|Bash",
                         "hooks": [{"type": "command", "command": f"{hook} post {memory}", "timeout": 30}]}],
        "SessionStart": [{"matcher": "compact", "hooks": [{"type": "command", "command": f"{hook} compact {memory}", "timeout": 30}]}]}}


def mcp_config(saving: bool = False) -> dict[str, Any]:
    """memory_search and memory_fetch (and, with ``saving``, memory_save) as an MCP server over the same store."""
    return {"mcpServers": {"memory": {"command": "python3", "args": [
        "-B", f"{CODE_IN_CONTAINER}/studies/claude_code/hooks/memory_mcp.py", "--memory", MEMORY_IN_CONTAINER]
        + (["--saving"] if saving else [])}}}


def exec_env(plan: SeriesPlan) -> list[str]:
    """docker exec -e flags: the token by NAME only (docker takes its value from the runner's environment), the rest by value."""
    env = {"CLAUDE_CONFIG_DIR": CONFIG_IN_CONTAINER, "DISABLE_AUTOUPDATER": "1", "HOME": "/root",
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
           "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "0"}  # auto memory ON in every arm, as the design says, whatever the default is
    if plan.arm == "gitlive":  # our layer replaces Claude Code's built-in memory: with both on, Claude saved to its auto memory
        env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"  # (which it also calls "memory") and never to ours (the fair trial, Nov 17)
    if plan.window:
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(plan.window)
    if plan.proxy:
        env |= {"HTTPS_PROXY": plan.proxy, "HTTP_PROXY": plan.proxy, "NO_PROXY": "localhost,127.0.0.1"}
    if plan.arm in OURS:
        env |= {"PYTHONPATH": CODE_IN_CONTAINER, "CC_STUDY_RULE_READER": "studies.claude_code.readers:claude_reader",
                "CC_STUDY_READER_HOME": READER_IN_CONTAINER, "CC_STUDY_TOKEN_FILE": TOKEN_IN_CONTAINER}
    if plan.arm == "gitlive":
        env |= {"PYTHONPATH": CODE_IN_CONTAINER, "CC_STUDY_CODE": CODE_IN_CONTAINER, "CC_STUDY_READER_HOME": READER_IN_CONTAINER,
                "CC_STUDY_TOKEN_FILE": TOKEN_IN_CONTAINER}
    flags = ["-e", SUBSCRIPTION_TOKEN]
    for key, value in env.items():
        flags += ["-e", f"{key}={value}"]
    return flags


def claude_argv(plan: SeriesPlan, name: str, *, budget_usd: float, resume: str | None) -> list[str]:
    inner = command(budget_usd=budget_usd, resume=resume, model=plan.model, claude=CLAUDE_IN_CONTAINER)
    inner += ["--disallowedTools", *DISALLOWED]
    if plan.arm in OURS:
        inner += ["--settings", HOOKS_IN_CONTAINER]
    if plan.arm in GIT_ARMS:
        at = inner.index("--permission-mode")
        inner = inner[:at] + list(MCP_TOOLS) + (["mcp__memory__memory_save"] if plan.arm == "gitlive" else []) + inner[at:]
        inner += ["--settings", HOOKS_IN_CONTAINER, "--mcp-config", MCP_IN_CONTAINER, "--strict-mcp-config"]
    return ["docker", "exec", "-i", "-w", "/app", *exec_env(plan), name, *inner]


_EXCLUDE = " ".join(f"':(exclude){p}'" for p in PROJECT_MEMORY)
PATCH_SCRIPT = (f"cd /app && cp .git/index /tmp/cc-index && GIT_INDEX_FILE=/tmp/cc-index git add -A -- . {_EXCLUDE} >/dev/null 2>&1 && "
                "GIT_INDEX_FILE=/tmp/cc-index git diff --cached --binary HEAD; rm -f /tmp/cc-index")
MEMORY_OUT = "cd /app && ls -d " + " ".join(PROJECT_MEMORY) + " 2>/dev/null | tar cf - -T - 2>/dev/null"


def patch_argv(name: str) -> list[str]:
    """The task's change against its base commit, new files included, project memory files left out, the repository's index untouched."""
    return ["docker", "exec", name, "bash", "-c", PATCH_SCRIPT]


def run(argv: list[str], *, runner: Runner = subprocess.run, timeout: float = 600, **options: Any) -> subprocess.CompletedProcess:
    return runner(argv, capture_output=True, text=True, timeout=timeout, check=False, **options)


def carry_out(name: str, carry: Path, *, runner: Runner = subprocess.run) -> list[str]:
    """Save what Claude wrote in the project to remember things, for the next task's copy of the project; returns the names."""
    done = runner(["docker", "exec", name, "bash", "-c", MEMORY_OUT], capture_output=True, timeout=120, check=False)
    data = done.stdout if isinstance(done.stdout, bytes) else (done.stdout or "").encode()
    carry.write_bytes(data)
    listed = runner(["tar", "tf", str(carry)], capture_output=True, text=True, timeout=60, check=False) if data else None
    return sorted({line.split("/")[0] for line in (listed.stdout.splitlines() if listed else []) if line})


def carry_in(name: str, carry: Path, *, runner: Runner = subprocess.run) -> None:
    if carry.exists() and carry.stat().st_size:
        with carry.open("rb") as data:
            runner(["docker", "exec", "-i", name, "tar", "xf", "-", "-C", "/app"], stdin=data, capture_output=True, timeout=120, check=False)


def grade_fresh(plan: SeriesPlan, round_: Round, patch_file: Path, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    """The locked protocol: the captured patch applied to a fresh container of the task's image, graded by the task's own checker."""
    name = container_name(plan, round_, "grade")
    run(["docker", "rm", "-f", name], runner=runner, timeout=120)
    started = run(["docker", "run", "-d", "--name", name, "--entrypoint", "sleep", "-w", "/app", round_.task.image, "infinity"],
                  runner=runner, timeout=1800)
    if started.returncode != 0:
        return {"reward": None, "error": f"grading container: {started.stderr[-300:]}"}
    try:
        applied = True
        if patch_file.stat().st_size:
            run(["docker", "cp", str(patch_file), f"{name}:/tmp/model.patch"], runner=runner, timeout=120)
            done = run(["docker", "exec", name, "bash", "-c", "cd /app && (git apply --verbose /tmp/model.patch || "
                        "patch --fuzz=5 -p1 -i /tmp/model.patch)"], runner=runner, timeout=300)
            applied = done.returncode == 0
        run(["docker", "cp", f"{round_.task.folder / 'tests'}/.", f"{name}:/tests"], runner=runner, timeout=120)
        t0 = time.monotonic()
        done = run(["docker", "exec", name, "bash", "-c", "mkdir -p /logs/verifier && bash /tests/test.sh > /logs/verifier/test-sh.log 2>&1; "
                                                        "cat /logs/verifier/reward.txt"], runner=runner, timeout=GRADE_TIMEOUT_S + 120)
        last = (done.stdout or "").strip().splitlines()[-1:] or [""]
        reward = int(last[0]) if last[0] in ("0", "1") else None
        return {"reward": reward, "applied": applied, "seconds": round(time.monotonic() - t0, 1),
                **({} if reward is not None else {"error": (done.stderr or "")[-300:]})}
    finally:
        run(["docker", "rm", "-f", name], runner=runner, timeout=300)


def memory_snapshot(plan: SeriesPlan, round_: Round, carried: list[str]) -> dict[str, Any]:
    """What each memory store holds after a task: Claude Code's auto memory files, project files it wrote, our rules file."""
    config = plan.out / "claude_config"
    auto = sorted(str(p.relative_to(config)) for p in config.glob("projects/*/memory/**/*") if p.is_file()) if config.exists() else []
    texts = {name: (config / name).read_text(errors="replace") for name in auto}
    rules = plan.out / "memory" / "rules.md"
    if rules.exists():
        texts["rules.md"] = rules.read_text(errors="replace")
    keep = plan.out / "memory-after" / round_.id
    keep.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (keep / name.replace("/", "__")).write_text(text)
    joined = "\n".join(texts.values())
    return {"auto_memory_files": auto, "project_memory": carried, "rules_file": rules.exists(),
            "holds_rule_1": bool(re.search(r'output\W+1\W|"1"', joined)), "holds_rule_2": bool(re.search(r"modify the tests|tests unless", joined, re.I))}


# --- the series ------------------------------------------------------------------------------------------------------------------

def seed_project_memory(folder: Path, carry: Path) -> list[str]:
    """The flat and folders arms: the memory's files (CLAUDE.md, .claude/rules/) as the project's carried files before task 1."""
    import tarfile
    names = sorted(str(p.relative_to(folder)) for p in Path(folder).rglob("*") if p.is_file())
    with tarfile.open(carry, "w") as tar:
        for name in names:
            tar.add(Path(folder) / name, arcname=name)
    return names


def start_memory_service(plan: SeriesPlan, *, wait_s: float = 180) -> subprocess.Popen:
    """The gitmem arm: Taste's store served on this machine through the socket in the mounted memory folder (the store needs Python
    3.11 and GitPython; CC_STUDY_PYLIB names the folder holding GitPython). No credential reaches it."""
    folder = plan.out / "memory"
    sock = folder / "memory.sock"
    env = {k: v for k, v in os.environ.items() if k not in (SUBSCRIPTION_TOKEN, *OTHER_CREDENTIALS)}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(Path(plan.code).resolve()), os.environ.get("CC_STUDY_PYLIB", "")) if p)
    log = (folder / "service.log").open("a")
    process = subprocess.Popen([sys.executable, "-B", "-m", "studies.claude_code.git_memory", "serve", "--memory", str(folder),
                                "--socket", str(sock)], env=env, stdout=log, stderr=log, cwd=str(Path(plan.code).resolve()))
    deadline = time.time() + wait_s
    while time.time() < deadline and process.poll() is None and not sock.exists():
        time.sleep(0.5)
    if process.poll() is not None or not sock.exists():
        process.kill()
        raise RuntimeError("the memory service did not start: " + (folder / "service.log").read_text()[-400:])
    return process


@contextlib.contextmanager
def memory_service(plan: SeriesPlan, service: Callable[[SeriesPlan], Any] | None):
    """The gitmem arm's service for the length of one batch; nothing for the other arms."""
    server = (service or start_memory_service)(plan) if plan.arm in GIT_ARMS else None
    try:
        yield server
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except Exception:
                server.kill()


def earlier_state(plan: SeriesPlan, first: int) -> tuple[dict[str, Any] | None, str | None, float]:
    """For a batch that starts at round ``first`` (1-based): the record so far, the session to resume and its running total. A batch
    must follow on exactly from the rounds already recorded."""
    if first == 1:
        if plan.record.exists():
            raise RuntimeError(f"{plan.record} exists: this series has begun; continue it from the round after its last one")
        return None, None, 0.0
    record = json.loads(plan.record.read_text())
    done = record["rounds"]
    if len(done) != first - 1 or record.get("ended_by"):
        raise RuntimeError(f"cannot start at round {first}: {len(done)} rounds recorded, ended_by {record.get('ended_by')}")
    totals = [r["reported_total_usd"] for r in done if r.get("reported_total_usd") is not None]
    return record, done[-1]["session_id"], (totals[-1] if totals else 0.0)


def run_series(plan: SeriesPlan, *, execute: bool = False, per_call_usd: float, session_usd: float, timeout_s: float = TASK_SECONDS,
               caller=None, runner: Runner = subprocess.run, base_env: dict[str, str] | None = None,
               say: Callable[[str], None] = print, first: int = 1, last: int | None = None, remove_images: bool = False,
               prefetch: Callable[[str], Any] | None = None, stop_on_break: bool = False,
               service: Callable[[SeriesPlan], Any] | None = None) -> dict[str, Any]:
    """Paid: only with execute=True. Runs rounds ``first`` to ``last`` (1-based, inclusive); saves the record after every round. Ends
    early when the budget is spent, when a call gives no result, when a container will not start, or (stop_on_break) at the first
    task that breaks a rule."""
    if not execute:
        raise RuntimeError("run_series starts paid Claude Code calls; call it with execute=True, after the user's OK")
    last = len(plan.rounds) if last is None else last
    if not 1 <= first <= last <= len(plan.rounds):
        raise ValueError(f"rounds {first} to {last} of {len(plan.rounds)}")
    env = dict(os.environ if base_env is None else base_env)
    for name in OTHER_CREDENTIALS:
        env.pop(name, None)
    if not env.get(SUBSCRIPTION_TOKEN):
        raise RuntimeError(f"no {SUBSCRIPTION_TOKEN}: the study runs on the user's Claude subscription")
    secrets = secret_values(env)
    caller = caller or call_claude
    for sub in ("claude_config", "patches", "memory", "reader", "hooks"):
        (plan.out / sub).mkdir(parents=True, exist_ok=True)
    if plan.arm in OURS:
        (plan.out / "hooks" / "settings.json").write_text(json.dumps(hook_settings(firm=plan.arm == "oursfirm"), indent=1) + "\n")
    carry = plan.out / "project-memory.tar"
    earlier, session_id, reported_total = earlier_state(plan, first)
    folder_items: list[dict[str, Any]] = []
    if plan.arm in MEMORY_ARMS:
        import shutil
        index = json.loads((plan.memory / "git" / "index.json").read_text())
        folder_items = [{"scope": r["scope"], "tests": r["tests"]} for r in index if r["kind"] == "folder"]
        if earlier is None and plan.arm in ("flat", "folders"):
            seed_project_memory(plan.memory / plan.arm, carry)
        if earlier is None and plan.arm in ("auto", "mdlog"):  # only the code notes, as path-scoped rules; no CLAUDE.md to start
            import tempfile
            seed = Path(tempfile.mkdtemp(prefix="seed-", dir=plan.out))
            shutil.copytree(plan.memory / "folders" / ".claude", seed / ".claude")
            seed_project_memory(seed, carry)
        if plan.arm in GIT_ARMS:
            if not (plan.out / "memory" / "index.json").exists():
                from studies.claude_code.git_memory import copy_memory
                copy_memory(plan.memory / "git", plan.out / "memory")
            settings = git_memory_live_settings() if plan.arm == "gitlive" else git_memory_settings(refresh=plan.arm == "gitmem2")
            (plan.out / "hooks" / "settings.json").write_text(json.dumps(settings, indent=1) + "\n")
            (plan.out / "hooks" / "mcp.json").write_text(json.dumps(mcp_config(saving=plan.arm == "gitlive"), indent=1) + "\n")
    record: dict[str, Any] = earlier or {
        "arm": plan.arm, "model": plan.model, "opening": opening(plan.arm), "claude_version": claude_version(str(plan.claude_binary)),
        "tasks": [r.task.id for r in plan.rounds], "sessions": [r.number for r in plan.rounds if r.new_session], "proxy": plan.proxy,
        "network": plan.network, "window": plan.window, "ruleset": plan.ruleset, "spacing": plan.spacing, "started": now(), "rounds": [], "spent_usd": 0.0, "ended_by": None, "batches": []}
    if record["tasks"] != [r.task.id for r in plan.rounds]:
        raise RuntimeError("the recorded series has other tasks than this plan")
    record.setdefault("batches", []).append({"first": first, "last": last, "started": now()})
    owner = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else None
    prefetch = prefetch or (lambda image: subprocess.Popen(["docker", "pull", "-q", image], stdout=subprocess.DEVNULL,
                                                           stderr=subprocess.DEVNULL))
    with plan.stream.open("a") as stream, memory_service(plan, service):
        for index in range(first - 1, last):
            round_ = plan.rounds[index]
            left = session_usd - record["spent_usd"]
            if left < 0.05:
                record["ended_by"] = "budget"
                break
            name = container_name(plan, round_)
            run(["docker", "rm", "-f", name], runner=runner, timeout=120)
            started = run(start_argv(plan, round_, name), runner=runner, timeout=1800)  # the first start pulls the image
            if started.returncode != 0:
                record["ended_by"] = f"container: {started.stderr[-300:]}"
                break
            carry_in(name, carry, runner=runner)
            resume = None if round_.new_session else session_id
            stream.write(json.dumps({"type": "study_turn", "turn": round_.id, "kind": "task", "task": round_.task.id, "index": index,
                                     "new_session": round_.new_session, "tells": ["rules"] if round_.tells_rules else [],
                                     "time": now(), "prompt": round_.prompt}) + "\n")
            stream.flush()
            seen: dict[str, Any] = {}

            def on_line(line: str) -> None:
                stream.write(redact(line.rstrip("\n"), secrets) + "\n")
                stream.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    return
                if isinstance(event, dict) and event.get("session_id"):
                    seen["session_id"] = event["session_id"]
                if isinstance(event, dict) and event.get("type") == "result":
                    seen["result"] = event
                if isinstance(event, dict) and event.get("type") == "system" and event.get("subtype") == "compact_boundary":
                    seen["compactions"] = seen.get("compactions", 0) + 1
                if isinstance(event, dict) and event.get("type") == "assistant":
                    seen.setdefault("tools", []).extend(b for b in (event.get("message") or {}).get("content", [])
                                                        if isinstance(b, dict) and b.get("type") == "tool_use")
                    usage = (event.get("message") or {}).get("usage") or {}
                    size = sum(usage.get(k, 0) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
                    seen["context"] = max(seen.get("context", 0), size)
                    seen.setdefault("contexts", []).append(size)

            if index + 1 < last:  # the next task's image downloads while this one runs
                prefetch(plan.rounds[index + 1].task.image)
            budget = min(per_call_usd, left)
            base = reported_total if resume else 0.0  # a new session's reported total starts again from zero
            argv = claude_argv(plan, name, budget_usd=base + budget, resume=resume)
            t0 = time.monotonic()
            code = caller(argv, env=env, cwd=plan.out, stdin_line=user_message(round_.prompt), timeout_s=timeout_s, on_line=on_line,
                          stderr_path=plan.out / "stderr.log")
            seconds = round(time.monotonic() - t0, 1)
            result = seen.get("result") or {}
            total = float(result.get("total_cost_usd") or 0.0)
            cost = total - base if total >= base else total
            if result:
                reported_total = total
            record["spent_usd"] = round(record["spent_usd"] + cost, 6)
            session_id = result.get("session_id") or seen.get("session_id") or (None if round_.new_session else session_id)
            patch = run(patch_argv(name), runner=runner, timeout=300).stdout or ""
            patch_file = plan.out / "patches" / f"{round_.id}.diff"
            patch_file.write_text(patch)
            carried = carry_out(name, carry, runner=runner)
            if owner:  # Claude Code runs as root in the container; its folders go back to the runner's user
                run(["docker", "exec", name, "chown", "-R", owner, CONFIG_IN_CONTAINER, *([MEMORY_IN_CONTAINER, READER_IN_CONTAINER]
                     if plan.arm in (*OURS, "gitlive") else [MEMORY_IN_CONTAINER] if plan.arm in GIT_ARMS else [])],
                    runner=runner, timeout=120)
            run(["docker", "rm", "-f", name], runner=runner, timeout=300)
            graded = ({"reward": None, "skipped": "question"} if round_.question else
                      grade_fresh(plan, round_, patch_file, runner=runner) if result else {"reward": None, "error": "no result"})
            if remove_images:
                run(["docker", "rmi", round_.task.image], runner=runner, timeout=300)
            one, tests = hand_back_one(result.get("result")), tests_rule(patch)
            memory = memory_snapshot(plan, round_, carried)
            row = {"round": round_.id, "task": round_.task.id, "new_session": round_.new_session, "exit": code, "seconds": seconds,
                   "cost_usd": round(cost, 6), "reported_total_usd": total if result else None, "is_error": result.get("is_error"),
                   "num_turns": result.get("num_turns"), "compactions": seen.get("compactions", 0),
                   "largest_context": seen.get("context", 0), "rule_1": one,
                   **({"screen": screen_checks(seen.get("tools", []), result.get("result"))} if plan.ruleset == "six" else {}),
                   **({"casual": casual_checks(result.get("result"), patch), "rules_told": round_.number >= TOLD_AT["casual"]}
                      if plan.ruleset == "casual" else {}),
                   **({"casual": revise_checks(result.get("result"), patch, round_.number, plan.ruleset),
                       "rules_told": round_.number >= TOLD_AT[plan.ruleset]} if plan.ruleset in REVISED_AT else {}),
                   **({"gradual": gradual_checks(seen.get("tools", []), result.get("result"), patch, round_.number, plan.spacing)}
                      if plan.ruleset == "gradual" else {}),
                   **({"gradual": gradual_checks(seen.get("tools", []), result.get("result"), patch, 10 ** 6, 1),
                       "folder_items": folder_checks(seen.get("tools", []), patch, folder_items)}
                      if plan.ruleset == "scaled" and not round_.question else {}),
                   **({"gradual": {k: {**v, "told": k in fair_told(FAIR_SCHEDULES[plan.ruleset], round_.number)}
                                   for k, v in gradual_checks(seen.get("tools", []), result.get("result"), patch, 10 ** 6, 1).items()}}
                      if plan.ruleset in FAIR_SCHEDULES and not round_.question else {}),
                   **({"asks": round_.asks} if round_.asks else {}),
                   **({"question": round_.prompt[len(QUESTION_HEAD):], "answer": result.get("result"),
                       "memory_tool_calls": sum(1 for t in seen.get("tools", []) if str(t.get("name", "")).startswith("mcp__memory__"))}
                      if round_.question else {}),
                   "context_calls": len(seen.get("contexts", [])),
                   "context_mean": round(sum(seen.get("contexts", [])) / len(seen["contexts"])) if seen.get("contexts") else 0,
                   "rule_2": tests, "patch_lines": patch.count("\n"),
                   **{f"grade_{k}": v for k, v in graded.items()}, "memory": memory, "session_id": session_id}
            record["rounds"].append(row)
            stream.write(json.dumps({"type": "study_call_end", "turn": round_.id, "exit": code, "time": now()}) + "\n")
            stream.flush()
            so_far = sum(r.get("compactions", 0) for r in record["rounds"])
            say(f"{plan.arm} {round_.id}{' (new session)' if round_.new_session else ''} {round_.task.short}: exit {code}, {seconds}s, "
                f"compactions {row['compactions']} (all {so_far}), context up to {row['largest_context']}, "
                f"${cost:.2f} (series ${record['spent_usd']:.2f}), rule 1 {one['strict']}, rule 2 {tests['kept']}, "
                f"reward {graded.get('reward')}, memory auto {len(memory['auto_memory_files'])} project {memory['project_memory']}"
                + ("" if "screen" not in row else " | brief {b[kept]} ({b[words]} words), edit tool {e[kept]} ({e[n_script_edits]} by script, "
                   "{e[tool_edits]} by tool), no stash {n[kept]}, python -B {p[kept]} ({p[with_B]} with, {p[without_B]} without)".format(
                       b=row["screen"]["brief"], e=row["screen"]["edit_tool"], n=row["screen"]["no_stash"], p=row["screen"]["python_B"]))
                + ("" if "casual" not in row else " | rules told {t} | brief {b[kept]} ({b[words]} words), files line {f[lenient]}/{f[strict]} "
                   "({f[first_line]!r:.70}), tests {x[kept]}".format(t=row["rules_told"], b=row["casual"]["brief"], f=row["casual"]["files_line"],
                                                                      x=row["casual"]["tests"]))
                + ("" if "files_where" not in row.get("casual", {}) else " | files at {w[where]} (asked {w[asked]}, kept {w[kept]}, "
                   "stale {w[stale]})".format(w=row["casual"]["files_where"]))
                + ("" if "gradual" not in row else " | rules kept {k}/{n} told{broken}".format(
                    k=sum(v["kept"] for v in row["gradual"].values() if v["told"] and v["chances"]),
                    n=sum(1 for v in row["gradual"].values() if v["told"] and v["chances"]),
                    broken="".join(f", broke {key}" for key, v in row["gradual"].items() if v["told"] and v["chances"] and not v["kept"])))
                + ("" if "folder_items" not in row else " | folder items kept {f[kept]}/{f[due]}, context mean {c}".format(
                    f=row["folder_items"], c=row["context_mean"]))
                + ("" if "answer" not in row else " | answer ({m} memory tool calls): {a!r:.150}".format(
                    m=row["memory_tool_calls"], a=(row["answer"] or "").replace("\n", " "))))
            broke = not one["strict"] or not tests["kept"]
            if not result:
                record["ended_by"] = "no_result"
            elif stop_on_break and broke:
                record["ended_by"] = f"rule_broken at {round_.id}"
            plan.record.write_text(json.dumps(record, indent=1) + "\n")  # after every round, so a crash loses at most one
            if record["ended_by"]:
                break
    stderr = plan.out / "stderr.log"
    if stderr.exists():
        stderr.write_text(redact(stderr.read_text(), secrets))
    record["batches"][-1]["finished"] = record["finished"] = now()
    plan.record.write_text(json.dumps(record, indent=1) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-root", type=Path, required=True, help="the v2/tasks folder of SWE-bench_Pro-os v2.0.0")
    parser.add_argument("--repo", default="ansible")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--avoid", default="40ade1f8", help="comma-separated id parts to leave out (40ade1f8: timing-sensitive test)")
    parser.add_argument("--sessions", default="1,6", help="task numbers that start a new session")
    parser.add_argument("--arm", choices=ARMS, default="shipped")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="claude-sonnet-5-5")
    parser.add_argument("--claude-binary", type=Path, required=True)
    parser.add_argument("--code", type=Path, help="this code's folder, mounted for the ours arm's hooks")
    parser.add_argument("--token-file", type=Path, help="the subscription token's file, mounted read-only for the ours arm's reader")
    parser.add_argument("--proxy", help="the model-only proxy, e.g. http://172.30.0.1:8888")
    parser.add_argument("--network", help="the docker network with no way out but the proxy")
    parser.add_argument("--window", type=int, help="CLAUDE_CODE_AUTO_COMPACT_WINDOW, 100000 to 1000000")
    parser.add_argument("--stop-on-break", action="store_true", help="end the series at the first task that breaks a rule")
    parser.add_argument("--memory", type=Path, help="the memory arms: the folder git_memory build wrote")
    parser.add_argument("--questions", type=Path, help="a JSON list of {question, answer}, asked after the last task")
    parser.add_argument("--ruleset", choices=(*OPENINGS, "gradual", "scaled", *FAIR_SCHEDULES), default="two",
                        help="two: the 1 and the tests; six: the screening set; casual: three rules said in passing at task 3; "
                             "revise: the casual rules, then at task 7 the files line moves to the end")
    parser.add_argument("--largest", action="store_true", help="the --count tasks with the largest reference fixes, not a random draw")
    parser.add_argument("--avoid-largest", type=int, default=0, help="leave the N largest tasks out of a random draw (the fair test's trial)")
    parser.add_argument("--rule-spacing", type=int, default=2, help="gradual rules: one every this many tasks from task 2")
    parser.add_argument("--per-call-usd", type=float, default=3.0)
    parser.add_argument("--session-usd", type=float, default=5.0)
    parser.add_argument("--rounds", default=None, help="FIRST:LAST (1-based, inclusive) for one batch")
    parser.add_argument("--remove-images", action="store_true", help="delete each task's image after it is graded (disk)")
    parser.add_argument("--execute", action="store_true", help="run Claude Code for real: paid, only after the user's OK")
    args = parser.parse_args(argv)
    avoid = tuple(a for a in args.avoid.split(",") if a)
    if args.avoid_largest:
        avoid += tuple(f.name for f in largest_tasks(args.tasks_root, args.repo, args.avoid_largest, avoid))
    fair = FAIR_SCHEDULES.get(args.ruleset)
    if fair is not None and args.largest:  # the first session on the hardest tasks, the new session on the next ones
        top = largest_tasks(args.tasks_root, args.repo, fair.tasks, avoid)
        folders = top + [f for f in largest_tasks(args.tasks_root, args.repo, fair.tasks + fair.extra, avoid) if f not in top]
    else:
        folders = (largest_tasks(args.tasks_root, args.repo, args.count, avoid) if args.largest
                   else pick_tasks(args.tasks_root, args.repo, args.count, args.seed, avoid))
    plan = plan_series(args.arm, folders, args.out, model=args.model, claude_binary=args.claude_binary,
                       sessions=tuple(int(x) for x in args.sessions.split(",") if x), code=args.code, token_file=args.token_file,
                       proxy=args.proxy, network=args.network, window=args.window, ruleset=args.ruleset, spacing=args.rule_spacing,
                       memory=args.memory,
                       questions=tuple(q["question"] for q in json.loads(args.questions.read_text())) if args.questions else ())
    if not args.execute:
        name = container_name(plan, plan.rounds[0])
        print(json.dumps({"arm": plan.arm, "tasks": [[r.number, r.task.short, r.task.size, r.new_session] for r in plan.rounds],
                          "first_prompt_head": plan.rounds[0].prompt[:240], "start": start_argv(plan, plan.rounds[0], name),
                          "claude": claude_argv(plan, name, budget_usd=args.per_call_usd, resume=None)}, indent=1))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    first, last = (int(x) for x in args.rounds.split(":")) if args.rounds else (1, len(plan.rounds))
    record = run_series(plan, execute=True, per_call_usd=args.per_call_usd, session_usd=args.session_usd, first=first, last=last,
                        remove_images=args.remove_images, stop_on_break=args.stop_on_break)
    print(json.dumps({key: record[key] for key in ("arm", "spent_usd", "ended_by")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
