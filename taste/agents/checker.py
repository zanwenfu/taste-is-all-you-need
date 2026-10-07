"""The checker: a reviewer that decides whether a coding agent's submission is done.

Part of the recovery study (which recovery brings a failed coding-agent run
back). When a coding agent submits, the checker is run on a fresh copy of the
agent's final files and decides, from evidence it gathers itself, whether the
work does what the task asks. Its ``not_done`` verdict triggers a recovery, and
its findings are the feedback the coding agent is shown.

It is a small tool-using agent, hosted like mini-swe-agent through Taste's
``Host``: ``ask`` for its model, ``run`` for a shell in the task container. Its
model is the trial's, fixed by the study; nothing here calls a model itself.

Given: one text (``checker_task``) holding the task exactly as the coding agent
got it, the agent's final message and its submission, and the instructions;
and the shell. It sees what changed (``git status`` and ``git diff`` in a
repository, otherwise the changed paths), runs the repository's existing tests
for the change, writes and runs small tests of its own from the task text, and
runs what the task names. The container is a copy, so its own test files reach
no one.

Never given, and refused: hidden benchmark tests, a grader and its results,
reference solutions. The prompt says so, and a command that names one of
``FORBIDDEN_PATHS`` (directly, through ``..``, by a glob or brace expansion
that could reach one, or relative to a directory that holds one) or a
benchmark's published sources (``FORBIDDEN_WORDS``) is refused before it
reaches the container. That catches the obvious ways. A search of the whole
file system (``find /``, ``grep -r ... /``) is not caught: the container the
checker runs in must not hold these files, and a fresh task container does not
(Harbor copies tests and solutions in only to grade).

It ends with one JSON object, its submission (``SCHEMA``): the verdict; for
``not_done`` the unmet requirement and the evidence (the check, its command, at
most 30 lines of output, expected and observed, and no solution code); its
confidence; how it ended; its steps, commands, refusals, cost and tokens. Its
limits are its settings (``CheckerSettings``): at its last affordable step it
is told to give its verdict, and a review that ends without one still submits
the object, with a null verdict and the limit that ended it. A trial's own
limits (``worker_max_calls``, ``spend_cap_usd``, its deadline) stop it
outright, with no submission, if they come first.

``feedback`` renders a ``not_done`` verdict for the coding agent: the evidence
(``"evidence"``), or the bare rejection (``"bare"``) used as an ablation.
"""

from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
import math
import posixpath
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from taste.agents import AgentExit
from taste.providers._openai import _NATIVE

NAME = "taste-checker"
VERSION = "1"
SCHEMA = "taste.agents/CheckerVerdict/1"
VERDICTS = ("done", "not_done")
# How a review ended: with a verdict, or without one at a limit.
ENDINGS = ("verdict", "steps", "usd", "seconds", "format_errors")
BARE_FEEDBACK = "A reviewer rejected the submission; keep working."

# Where benchmarks keep what a reviewer must never see. Harbor (Terminal-Bench
# and its SWE-bench adapters) copies a task's tests to /tests to grade it and
# writes the verifier's results (reward.txt, reward.json, ctrf.json,
# test-stdout.txt) to /logs/verifier; it copies the reference solution
# (solve.sh) to /solution for its oracle agent. The SWE-bench harness grades
# with /eval.sh. The others are the same things by other names.
FORBIDDEN_PATHS = ("/tests", "/solution", "/solutions", "/oracle", "/grader", "/graders", "/grading",
                   "/logs/verifier", "/eval.sh")
# Where benchmarks publish their tasks, tests and solutions; matched as whole
# words, so "testbench" is not "tbench".
FORBIDDEN_WORDS = ("terminal-bench", "terminal_bench", "terminalbench", "tbench", "t-bench", "swe-bench",
                   "swe_bench", "swebench", "laude-institute", "harbor-framework")

# What the model's verdict may hold. Evidence is for the coding agent to read,
# so it is short; and the whole submission stays within the hosted worker's
# 4,000-character summary (taste.brains.hosted_worker.SUMMARY_CHARS), which
# becomes a trial's final reply, so a driver can read it there whole.
EVIDENCE_LINES = 30
COMMAND_LINES = 40
FIELD_CHARS = {"unmet_requirement": 500, "check": 300, "command": 1_200, "output": 2_400,
               "expected": 400, "observed": 400}
VERDICT_CHARS = 3_500
SUBMISSION_CHARS = 3_900
EVIDENCE_FIELDS = ("check", "command", "output", "expected", "observed")


@dataclass(frozen=True)
class CheckerSettings:
    """The checker's own limits, fixed for a study and disclosed with its identity."""

    max_steps: int = 30             # model calls, the last one included
    max_usd: float = 2.0            # what its model calls may cost together
    max_seconds: float = 900.0      # the whole review, so that a trial's deadline does not cut it off
    command_seconds: float = 120.0  # one command's time limit
    output_chars: int = 10_000      # of one command's output, shown to it
    max_format_errors: int = 3      # replies in a row with neither a command nor a verdict

    def __post_init__(self):
        for name, least in (("max_steps", 1), ("output_chars", 200), ("max_format_errors", 1)):
            if type(getattr(self, name)) is not int or getattr(self, name) < least:
                raise ValueError(f"{name} must be an integer of at least {least}")
        for name in ("max_usd", "max_seconds", "command_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


SYSTEM_TEMPLATE = """\
You review a coding agent's work. The agent was given a task, worked on it in a container and \
submitted. You decide whether its work does what the task asks, from evidence you gather yourself.

You work in a copy of that container, holding the agent's final files; the working directory is \
{cwd}. The bash tool runs one command there in a fresh shell (cd and variables do not carry over), \
with standard error folded into the output, a limit of {seconds} seconds, and long output cut to \
its beginning and end. The copy is yours: you may write test files and run them. Nothing you change \
reaches the agent.

How to check:
1. See what changed: git status and git diff in a git repository; otherwise the changed paths if \
you are given them, the files the task names, or those the agent says it changed.
2. Run the repository's existing tests that cover the change.
3. Write small tests of your own for what the task asks, taking the expected results from the task \
text, never from the agent's code, and run them.
4. Run what the task names (a command, a script, a program, an output file) and compare what \
happens with what the task asks.

Rules:
- Never use, look for or read hidden benchmark tests, a grader or its results, or reference \
solutions: not in /tests, /solution, /oracle, /grader or /logs/verifier, not elsewhere in the file \
system, not on the internet. Commands that name such paths are refused, and nothing is run. Judge \
only from the task text, the agent's work and the checks you run.
- Judge the task as written: do not add requirements it does not state, and do not reject work \
for its style.
- A requirement is unmet only when a check you ran shows it. If your checks pass and nothing the \
task asks for is missing, the work is done.
- Do not fix the work. What you submit must not contain solution code or a patch: say what is \
wrong and show the evidence, not how to fix it.
- You have at most {steps} replies. A few decisive checks are better than many.

When you have decided, call submit_verdict once with:
- verdict: "done" if the work does what the task asks, "not_done" if it does not;
- for not_done, unmet_requirement: the requirement of the task that is not met, in one or two \
sentences; and evidence: check (what you checked, in one line), command (the command you ran), \
output (the part of its output that shows the problem, at most 30 lines), expected (what the task \
requires) and observed (what happened instead);
- confidence: a number from 0 to 1, how likely your verdict is right."""

TASK_TEMPLATE = """\
A coding agent was given the task below, worked on it in this container and submitted. Review its \
work: decide whether it does what the task asks.

<task>
{task}
</task>

The agent's final message and its submission are its own account of its work: check what they \
claim, do not take them as evidence.

<final_message>
{final_message}
</final_message>

<submission>
{submission}
</submission>
{changed}
Gather your evidence in the container, then call submit_verdict."""

CHANGED_TEMPLATE = """
The paths the agent's work changed, from a record of the container's files:
<changed_paths>
{paths}
</changed_paths>
"""

LAST_STEP = ("This is your last reply: no more commands will run. Call submit_verdict now, with "
             "your verdict from the evidence you have.")
NO_ACTION = ("Your reply ran no command and gave no verdict. Call bash to run a command, or "
             "submit_verdict to give your verdict.")
CUT_OFF = ("Your reply was cut off at its length limit before it called a tool. Reply more "
           "briefly, with one tool call.")
REFUSED = ("refused: the command names {name}, which a reviewer may not use (hidden tests, a "
           "grader or its results, reference solutions, a benchmark's published sources). Nothing "
           "was run.")

BASH_TOOL = {
    "type": "function", "name": "bash", "strict": False,
    "description": "Run one shell command in the container, in a fresh shell, and see its exit code and output.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["command"],
                   "properties": {"command": {"type": "string", "description": "The command to run."}}},
}
SUBMIT_TOOL = {
    "type": "function", "name": "submit_verdict", "strict": False,
    "description": "Give your verdict on the agent's work. Call it once, when you have decided; it ends your review.",
    "parameters": {
        "type": "object", "additionalProperties": False, "required": ["verdict", "confidence"],
        "properties": {
            "verdict": {"type": "string", "enum": list(VERDICTS)},
            "unmet_requirement": {"type": "string", "description": (
                "For not_done: the requirement of the task that is not met, in one or two sentences.")},
            "evidence": {
                "type": "object", "additionalProperties": False, "required": list(EVIDENCE_FIELDS),
                "description": "For not_done: the check that shows the requirement unmet. No solution code.",
                "properties": {
                    "check": {"type": "string", "description": "What you checked, in one line."},
                    "command": {"type": "string", "description": "The command you ran."},
                    "output": {"type": "string", "description": (
                        f"The part of its output that shows the problem, at most {EVIDENCE_LINES} lines.")},
                    "expected": {"type": "string", "description": "What the task requires."},
                    "observed": {"type": "string", "description": "What happened instead."},
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1,
                           "description": "How likely your verdict is right, from 0 to 1."},
        },
    },
}
TOOLS = (BASH_TOOL, SUBMIT_TOOL)
# Everything that shapes what the model is told, and what its commands may
# name, so that a change shows in a trial's record.
PROMPTS_SHA256 = hashlib.sha256(json.dumps(
    [SYSTEM_TEMPLATE, TASK_TEMPLATE, CHANGED_TEMPLATE, LAST_STEP, NO_ACTION, CUT_OFF, REFUSED, TOOLS,
     FORBIDDEN_PATHS, FORBIDDEN_WORDS], sort_keys=True).encode()).hexdigest()


def system_prompt(cwd: str, settings: CheckerSettings) -> str:
    return SYSTEM_TEMPLATE.format(cwd=cwd, seconds=f"{settings.command_seconds:g}", steps=settings.max_steps)


def checker_task(task: str, final_message: str, submission: str, *, changed_paths=None) -> str:
    """The checker's task text: the coding task as the agent got it, the agent's own account, the instructions.

    ``changed_paths``, when known (a checkpoint's manifest lists them), is given
    for work outside a git repository, where the checker cannot ask git.
    """
    if not isinstance(task, str) or not task.strip():
        raise ValueError("the checker needs the task the coding agent was given")
    changed = ""
    if changed_paths:
        changed = CHANGED_TEMPLATE.format(paths="\n".join(str(path) for path in changed_paths))
    return TASK_TEMPLATE.format(task=task, final_message=(final_message or "").strip() or "(none)",
                                submission=(submission or "").strip() or "(empty)", changed=changed)


# -- the verdict ----------------------------------------------------------------


class InvalidVerdict(ValueError):
    """A verdict that does not meet the schema; its message says what to correct."""


# Code fences and patch lines: what solution code in a finding would look like.
_CODE = re.compile(r"```|^(?:@@ |\+\+\+ |--- |diff --git )", re.M)
_PATCH = re.compile(r"^(?:@@ |\+\+\+ |--- |diff --git )", re.M)


def _number(value):
    """A finite int or float; never a bool."""
    return type(value) in (int, float) and math.isfinite(value)


def _text(value, name, errors, *, required=True):
    if not isinstance(value, str) or (required and not value.strip()):
        errors.append(f"{name} must be text" + (", not empty" if required else ""))
        return ""
    value = value.strip("\n") if name.endswith("output") else value.strip()
    if len(value) > FIELD_CHARS[name.rsplit(".", 1)[-1]]:
        errors.append(f"{name} is {len(value)} characters; at most {FIELD_CHARS[name.rsplit('.', 1)[-1]]}")
    return value


def validate_verdict(value) -> dict:
    """The model's verdict, checked and normalised; raises ``InvalidVerdict`` naming every problem.

    Unknown keys are dropped. A ``done`` verdict carries no evidence; one that
    names an unmet requirement contradicts itself and is refused.
    """
    if not isinstance(value, Mapping):
        raise InvalidVerdict("the verdict must be one JSON object")
    errors = []
    verdict = value.get("verdict")
    if verdict not in VERDICTS:
        errors.append('verdict must be "done" or "not_done"')
    confidence = value.get("confidence")
    if not _number(confidence) or not 0 <= confidence <= 1:
        errors.append("confidence must be a number from 0 to 1")
    result = {"verdict": verdict}
    if verdict == "done":
        if str(value.get("unmet_requirement") or "").strip():
            errors.append("a done verdict names no unmet requirement")
    elif verdict == "not_done":
        requirement = _text(value.get("unmet_requirement"), "unmet_requirement", errors)
        evidence = value.get("evidence")
        if not isinstance(evidence, Mapping):
            errors.append("evidence must be an object with " + ", ".join(EVIDENCE_FIELDS))
            evidence = {}
        shown = {name: _text(evidence.get(name), "evidence." + name, errors, required=name != "output")
                 for name in EVIDENCE_FIELDS}
        if shown["output"].count("\n") + 1 > EVIDENCE_LINES:
            errors.append(f"evidence.output must be at most {EVIDENCE_LINES} lines")
        if shown["command"].count("\n") + 1 > COMMAND_LINES:
            errors.append(f"evidence.command must be at most {COMMAND_LINES} lines")
        for name, text in (("unmet_requirement", requirement), *((f"evidence.{key}", shown[key])
                                                                  for key in ("check", "expected", "observed"))):
            if _CODE.search(text):
                errors.append(f"{name} must say what is wrong, without code blocks or a patch")
        if _PATCH.search(shown["command"]):
            errors.append("evidence.command must be the check you ran, not a patch")
        result.update(unmet_requirement=requirement, evidence=shown)
    if not errors:
        result["confidence"] = round(float(confidence), 3)
        size = len(json.dumps(result, ensure_ascii=False))
        if size > VERDICT_CHARS:
            errors.append(f"the verdict is {size} characters; at most {VERDICT_CHARS}: shorten the evidence")
    if errors:
        raise InvalidVerdict("; ".join(errors))
    return result


def json_object(text):
    """The one JSON object a reply's text holds: the whole text, a fenced block, or its outer braces."""
    text = (text or "").strip()
    candidates = [text]
    candidates.extend(match.group(1) for match in re.finditer(r"```(?:json)?\s*\n(.*?)```", text, re.S))
    if "{" in text and "}" in text:
        candidates.append(text[text.index("{"):text.rindex("}") + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_submission(text) -> dict:
    """The checker's submission, from its JSON text or a hosted report that holds it; validated."""
    if isinstance(text, Mapping):
        return _check_submission(text)
    if not isinstance(text, str):
        raise ValueError("a checker submission is JSON text")
    marker = "## Its submission\n\n"
    if marker in text:
        text = text.split(marker, 1)[1].split("\n\n## ", 1)[0]
    try:
        value = json.loads(text.strip())
    except ValueError:
        raise ValueError("a checker submission is one JSON object") from None
    return _check_submission(value)


def _check_submission(value):
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        raise ValueError("not a checker submission of schema " + SCHEMA)
    result = dict(value)
    if value.get("verdict") is None:
        if value.get("ended") == "verdict" or value.get("confidence") is not None:
            raise ValueError("a submission without a verdict ended at a limit and has no confidence")
    else:
        model = {key: value[key] for key in ("verdict", "unmet_requirement", "evidence", "confidence") if key in value}
        result.update(validate_verdict(model))
        if value.get("ended") != "verdict":
            raise ValueError("a submission with a verdict ended with it")
    if value.get("ended") not in ENDINGS:
        raise ValueError("a submission's ending must be one of " + ", ".join(ENDINGS))
    for name in ("steps", "commands", "refused"):
        if type(value.get(name)) is not int or value[name] < 0:
            raise ValueError(f"a submission's {name} must be a count")
    if not _number(value.get("cost_usd")) or value["cost_usd"] < 0:
        raise ValueError("a submission's cost must be a nonnegative number")
    return result


def feedback(submission, variant: str = "evidence") -> str:
    """What the coding agent is shown after a ``not_done`` verdict.

    ``submission`` is the checker's (its text, or parsed), or a verdict alone.
    ``"evidence"``: the unmet requirement and the reviewer's check, written to
    follow "rejected by a reviewer: ". ``"bare"``: the rejection alone, the
    ablation that tells the agent nothing about what is wrong.
    """
    if variant not in ("evidence", "bare"):
        raise ValueError("feedback variant must be evidence or bare")
    if isinstance(submission, Mapping) and "schema" not in submission:
        value = validate_verdict(submission)
    else:
        value = parse_submission(submission)
    if value["verdict"] != "not_done":
        raise ValueError("feedback is written only for a not_done verdict")
    if variant == "bare":
        return BARE_FEEDBACK
    evidence = value["evidence"]
    lines = [value["unmet_requirement"], "", "The reviewer's check: " + evidence["check"],
             "$ " + evidence["command"]]
    if evidence["output"]:
        lines.append(evidence["output"])
    lines.extend(["Expected: " + evidence["expected"], "Observed: " + evidence["observed"]])
    return "\n".join(lines)


# -- the tool layer --------------------------------------------------------------


_SOURCES = re.compile("(?<![a-z0-9])(?:" + "|".join(re.escape(word) for word in FORBIDDEN_WORDS) + ")(?![a-z0-9])",
                      re.I)
_BRACES = re.compile(r"\{([^{}]*,[^{}]*)\}")
_GLOB = re.compile(r"[*?\[]")


def _expand_braces(word, limit=64):
    """A word's brace expansions (``/{tests,app}`` -> ``/tests``, ``/app``), as far as ``limit``."""
    pending, done = [word], []
    while pending and len(done) + len(pending) < limit:
        current = pending.pop()
        match = _BRACES.search(current)
        if match is None:
            done.append(current)
            continue
        pending.extend(current[:match.start()] + option + current[match.end():]
                       for option in match.group(1).split(","))
    return done + pending


def _normal(path):
    return posixpath.normpath(re.sub("/+", "/", path))


def _paths(command, cwd):
    """Every path a command could name: its words, the relative ones from the directory it is in by then."""
    # Quotes and escapes only join what the shell joins: /te"st"s is /tests.
    text = re.sub(r"[\\'\"]", "", command)
    words = []
    for raw in re.split(r"[\s`;|&<>()]+", text):
        for word in _expand_braces(raw):
            words.extend(part for part in re.split(r"[=:,]", word) if part)
    base, previous = _normal(cwd), ""
    for word in words:
        if word.startswith(("-", "$", "~")):
            previous = word
            continue
        path = _normal(word if word.startswith("/") else posixpath.join(base, word))
        yield path
        if previous in ("cd", "pushd"):
            base = path
        previous = word


def _names(path, forbidden):
    """Whether ``path`` (a glob, perhaps) is ``forbidden`` or inside it, or could expand to it."""
    parts, wanted = path.strip("/").split("/"), forbidden.strip("/").split("/")
    if parts == [""] or len(parts) < len(wanted):
        return False
    return all(fnmatch.fnmatchcase(want, part) if _GLOB.search(part) else want == part
               for part, want in zip(parts[:len(wanted)], wanted, strict=True))


def forbidden_reference(command: str, cwd: str = "/") -> str | None:
    """The forbidden path or word a command names, or None; ``cwd`` is where it runs."""
    found = _SOURCES.search(re.sub(r"[\\'\"]", "", command))
    if found is not None:
        return found.group(0)
    for path in _paths(command, cwd):
        for forbidden in FORBIDDEN_PATHS:
            if _names(path, forbidden):
                return forbidden
    return None


_ENVIRONMENT = "export PAGER=cat MANPAGER=cat GIT_PAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1; "


def _shell(command):
    # One stream in arrival order; a group, so that a heredoc ending the command still closes.
    return _ENVIRONMENT + "{\n" + command + "\n} 2>&1"


def _excerpt(text, limit):
    if len(text) <= limit:
        return text
    head = limit * 2 // 5
    return text[:head] + f"\n[... {len(text) - limit:,} characters cut ...]\n" + text[-(limit - head):]


def _observation(result, settings):
    if result.timed_out:
        status = f"exit code: none (ended after {settings.command_seconds:g} seconds; its output so far is below)"
    else:
        status = f"exit code: {result.returncode}"
    return status + "\noutput:\n" + (_excerpt(result.output, settings.output_chars) or "(none)")


def _canonical(items):
    """Responses input items, as Taste's provider replays them byte for byte (as for mini-swe-agent)."""
    messages = []
    for item in items:
        if set(item) == {"role", "content"} and isinstance(item["content"], str):
            messages.append({"role": item["role"], "content": item["content"]})
        else:
            messages.append({"role": "user", "content": [{"type": _NATIVE, "item": copy.deepcopy(dict(item))}]})
    return messages


# -- the agent -----------------------------------------------------------------


class Checker:
    """The checker as a hosted agent: its prompts, its two tools, its own limits."""

    name = NAME

    def __init__(self, *, effort, settings: CheckerSettings | None = None, clock=time.monotonic):
        self.effort = effort
        self.settings = settings or CheckerSettings()
        self.clock = clock

    def identity(self):
        return {"name": NAME, "version": VERSION, "schema": SCHEMA, "prompts_sha256": PROMPTS_SHA256,
                "settings": asdict(self.settings)}

    def run(self, task, host, *, cwd, model_name):
        """Review the work in ``cwd``; the model is the host's, whatever ``model_name`` says."""
        return _Review(self, host, cwd).run(task)


class _Review:
    def __init__(self, checker, host, cwd):
        if not isinstance(cwd, str) or not cwd.startswith("/"):
            raise ValueError("the checker works in the task's directory, an absolute path")
        self.settings, self.effort, self.host, self.cwd = checker.settings, checker.effort, host, _normal(cwd)
        self.clock = checker.clock
        self.items = []
        self.steps = self.commands = self.refused = self.format_errors = 0
        self.spent = self.costliest = self.slowest = 0.0
        self.tokens = {"input": 0, "output": 0}

    def _limit(self, ahead):
        """The limit that leaves no room for ``ahead`` more steps, if one does.

        A step is expected to cost as much, and take as long, as the costliest
        and the slowest so far.
        """
        limits, elapsed = self.settings, self.clock() - self.started
        if self.steps + ahead > limits.max_steps:
            return "steps"
        if self.spent + ahead * self.costliest > limits.max_usd:
            return "usd"
        if elapsed + ahead * self.slowest > limits.max_seconds:
            return "seconds"
        return None

    def run(self, task):
        self.started = self.clock()
        self.items = [{"role": "system", "content": system_prompt(self.cwd, self.settings)},
                      {"role": "user", "content": task}]
        while True:
            ended = self._limit(1)
            if ended is not None:
                return self._end(None, ended)
            # The last step is the one no other could follow.
            last = self._limit(2)
            if last is not None:
                self.items.append({"role": "user", "content": LAST_STEP})
            began = self.clock()
            reply = self.host.ask(messages=_canonical(self.items), tools=copy.deepcopy(list(TOOLS)),
                                  effort=self.effort)
            self._count(reply)
            verdict = self._answer(reply, last is not None)
            self.slowest = max(self.slowest, self.clock() - began)
            if verdict is not None:
                return self._end(verdict, "verdict")
            if last is not None:
                return self._end(None, last)
            if self.format_errors >= self.settings.max_format_errors:
                return self._end(None, "format_errors")

    def _count(self, reply):
        cost = reply.cost_usd
        if not _number(cost) or cost < 0:
            raise ValueError("a reply's cost must be a finite, nonnegative number")
        self.steps += 1
        self.spent += cost
        self.costliest = max(self.costliest, cost)
        usage = reply.usage or {}

        def tokens(name):
            value = usage.get(name)
            return value if type(value) is int and value >= 0 else 0

        self.tokens["input"] += sum(tokens(name) for name in ("input_tokens", "cache_read_tokens", "cache_write_tokens"))
        self.tokens["output"] += tokens("output_tokens")

    def _answer(self, reply, last):
        output = [copy.deepcopy(dict(item)) for item in reply.output]
        self.items.extend(output)
        calls = [item for item in output if item.get("type") == "function_call"]
        if not calls:
            return self._text_verdict(output, reply.status)
        self.format_errors = 0
        verdict = None
        for call in calls:
            result = "not run: the verdict ended the review"
            if verdict is None:
                result, verdict = self._call(call, last)
            self.items.append({"type": "function_call_output", "call_id": call.get("call_id", ""), "output": result})
        return verdict

    def _call(self, call, last):
        name, arguments = call.get("name"), call.get("arguments")
        if not isinstance(arguments, dict):
            try:
                arguments = json.loads(arguments or "{}")
            except (TypeError, ValueError):
                arguments = None
        if not isinstance(arguments, dict):
            return "error: the arguments must be one JSON object", None
        if name == "submit_verdict":
            try:
                return "accepted", validate_verdict(arguments)
            except InvalidVerdict as error:
                return f"not accepted: {error}. Call submit_verdict again with a corrected verdict.", None
        if name != "bash":
            return f"error: there is no tool named {name!r}; the tools are bash and submit_verdict", None
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return "error: bash needs a command, as text", None
        if last:
            return "not run: no steps are left; call submit_verdict", None
        return self._bash(command), None

    def _bash(self, command):
        forbidden = forbidden_reference(command, self.cwd)
        if forbidden is not None:
            self.refused += 1
            return REFUSED.format(name=forbidden)
        self.commands += 1
        result = self.host.run(_shell(command), cwd=self.cwd, timeout_seconds=self.settings.command_seconds,
                               shown=command)
        return _observation(result, self.settings)

    def _text_verdict(self, output, status):
        """A reply with no tool call: a verdict written as JSON text, or a format error."""
        text = "\n".join(part.get("text", "") for item in output if item.get("type") == "message"
                         for part in item.get("content") or () if part.get("type") == "output_text")
        value = json_object(text)
        if value is not None and "verdict" in value:
            try:
                return validate_verdict(value)
            except InvalidVerdict as error:
                self.items.append({"role": "user", "content": (
                    f"Your verdict was not accepted: {error}. Call submit_verdict with a corrected verdict.")})
                return None
        self.format_errors += 1
        self.items.append({"role": "user", "content": CUT_OFF if status == "incomplete" else NO_ACTION})
        return None

    def _end(self, verdict, ended):
        submission = {"schema": SCHEMA, **(verdict or {"verdict": None, "confidence": None}), "ended": ended,
                      "steps": self.steps, "commands": self.commands, "refused": self.refused,
                      "cost_usd": round(self.spent, 6), "tokens": dict(self.tokens)}
        text = json.dumps(submission, ensure_ascii=False, allow_nan=False)
        if len(text) > SUBMISSION_CHARS:
            raise AssertionError("a validated verdict always fits the submission")
        status = ("Submitted" if verdict is not None else
                  "FormatError" if ended == "format_errors" else "LimitsExceeded")
        return AgentExit(status, text, tuple(copy.deepcopy(self.items)), self.steps)
