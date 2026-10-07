"""Where a failed run went wrong: the trajectory reader, and two rules that need no model.

In the recovery study a failed run can be rewound to a step and continued from
there. These choose where, from the run's record alone, as a deployed harness
could; each is scored against the point of no return measured by reruns.

- ``read_trajectory`` asks a model, through a callable the caller supplies, to
  name the first step where the agent went wrong, with a one-line reason and a
  confidence. Rewind to just before it: ``rewind_point(reading)``.
- ``last_passing_tests``: the last step after which the visible tests passed,
  read from what the agent's test commands printed.
- ``before_last_large_edit``: the step before the last large edit, read from
  the commands that wrote files.

(The third rule, the start, is rewind point 0 and needs no code.)

Steps are numbered as the study numbers them: step i is the agent's i-th model
call and the command it ran. A rewind point k is the state after step k; 0 is
the start. ``steps_from_trajectory`` reads the steps of a hosted agent's run
from its ATIF record (``taste.benchmarks.worker_trajectory.hosted_trajectory``).

The model is given, per step, the command, its exit code and the first and
last lines of its output; the agent's final message; and the checker's
findings (``taste.agents.checker``). Its reply is one JSON object
``{"step": int, "reason": str, "confidence": float}``, checked here; a reply
that fails the check is asked for again once, with what was wrong.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from taste.agents.checker import feedback, json_object, parse_submission

READER_SYSTEM = ("You read the record of a coding agent's run that failed, and find the first step "
                 "where it went wrong.")

READER_TEMPLATE = """\
A coding agent was given the task below and worked on it step by step. Its final work was found not \
to do what the task asks. Find the first step where it went wrong.

<task>
{task}
</task>

Its steps, in order. Each shows the command it ran, the command's exit code, and the first and last \
lines of what it printed.
<steps>
{steps}
</steps>

Its final message:
<final_message>
{final_message}
</final_message>

A reviewer checked its final work and found:
<findings>
{findings}
</findings>

Name the earliest step after which the run, continued as it went, did not recover: for example a \
misreading of the task, a wrong diagnosis, an edit that broke something, or a wrong approach that \
later steps built on. A mistake the agent noticed and undid later is not it. Reply with one JSON \
object and nothing else:
{{"step": <a step number from {first} to {last}>, "reason": "<one line: what went wrong at that \
step>", "confidence": <a number from 0 to 1, how likely this is the step>}}"""

RETRY_TEMPLATE = ("That reply was not accepted: {error}. Reply with one JSON object and nothing else: "
                  '{{"step": <a step number from {first} to {last}>, "reason": "<one line>", '
                  '"confidence": <a number from 0 to 1>}}')

REASON_CHARS = 300


@dataclass(frozen=True)
class Step:
    """One step of a coding agent's run: the command(s) its model call ran, and what they printed."""

    number: int
    command: str = ""
    output: str = ""
    returncode: int | None = None


class InvalidReading(ValueError):
    """The model's reply did not name a step as asked."""


def _steps(steps):
    steps = list(steps)
    if not steps:
        raise ValueError("a run has at least one step")
    previous = 0
    for step in steps:
        if not isinstance(step, Step) or type(step.number) is not int or step.number <= previous:
            raise ValueError("steps are Step records numbered from 1, in order")
        previous = step.number
    return steps


# -- the reader ------------------------------------------------------------------


def _lines(text, first, last, width):
    """A text's first and last lines, each cut to ``width``, with how many were left out between."""
    lines = [line if len(line) <= width else line[:width] + " [...]" for line in text.splitlines()]
    if len(lines) <= first + last:
        return lines
    return [*lines[:first], f"[... {len(lines) - first - last} lines not shown ...]",
            *(lines[-last:] if last else [])]


def render_steps(steps: Sequence[Step], *, command_lines=6, head_lines=4, tail_lines=6, width=200) -> str:
    """The step list as the reader is shown it: each command, its exit code, and its output's ends."""
    blocks = []
    for step in _steps(steps):
        if step.command.strip():
            command = _lines(step.command.strip(), command_lines, 0, width)
            block = [f"[step {step.number}] $ " + command[0], *command[1:]]
        else:
            block = [f"[step {step.number}] (no command)"]
        if step.returncode is not None:
            block.append(f"exit {step.returncode}")
        output = _lines(step.output.strip("\n"), head_lines, tail_lines, width)
        block.extend(output if output else ["(no output)"] if step.command.strip() else [])
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def _findings(findings):
    """The checker's findings as the reader is shown them: its evidence, as the coding agent sees it."""
    if isinstance(findings, str):
        try:
            findings = parse_submission(findings)
        except ValueError:
            return findings.strip() or "(none)"
    if not isinstance(findings, Mapping):
        raise ValueError("the checker's findings are its submission, its verdict or a text")
    if findings.get("verdict") == "not_done":
        return feedback(findings, "evidence")
    if findings.get("verdict") == "done":
        return "The reviewer judged the work done."
    return "The reviewer reached no verdict."


def reader_prompt(task: str, steps: Sequence[Step], final_message: str, findings) -> str:
    steps = _steps(steps)
    return READER_TEMPLATE.format(task=task, steps=render_steps(steps),
                                  final_message=(final_message or "").strip() or "(none)",
                                  findings=_findings(findings), first=steps[0].number, last=steps[-1].number)


def validate_reading(value, numbers) -> dict:
    """``{"step", "reason", "confidence"}`` from a reply's JSON object; raises ``InvalidReading``."""
    if not isinstance(value, Mapping):
        raise InvalidReading("the reply must be one JSON object")
    errors = []
    step = value.get("step")
    if type(step) is float and step.is_integer():
        step = int(step)
    if type(step) is not int or step not in numbers:
        errors.append(f"step must be one of the step numbers, {min(numbers)} to {max(numbers)}")
    reason = value.get("reason")
    reason = " ".join(reason.split()) if isinstance(reason, str) else ""
    if not reason:
        errors.append("reason must be one line of text")
    elif len(reason) > REASON_CHARS:
        errors.append(f"reason must be one line of at most {REASON_CHARS} characters")
    confidence = value.get("confidence")
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        errors.append("confidence must be a number from 0 to 1")
    if errors:
        raise InvalidReading("; ".join(errors))
    return {"step": step, "reason": reason, "confidence": round(float(confidence), 3)}


def read_trajectory(task: str, steps: Sequence[Step], final_message: str, findings, *,
                    ask: Callable[[list[dict[str, str]]], str], attempts: int = 2) -> dict:
    """The first step where the agent went wrong, as the model reads the record.

    ``ask`` takes the conversation (role and content messages) and returns the
    model's reply text; the caller chooses the model and counts its cost. A
    reply that is not a valid reading is answered with what was wrong, up to
    ``attempts`` replies in all; then ``InvalidReading`` is raised.
    """
    if type(attempts) is not int or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    steps = _steps(steps)
    numbers = {step.number for step in steps}
    messages = [{"role": "system", "content": READER_SYSTEM},
                {"role": "user", "content": reader_prompt(task, steps, final_message, findings)}]
    error = None
    for _ in range(attempts):
        reply = ask([dict(message) for message in messages])
        if not isinstance(reply, str):
            raise TypeError("ask must return the reply's text")
        try:
            return validate_reading(json_object(reply), numbers)
        except InvalidReading as failure:
            error = failure
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": RETRY_TEMPLATE.format(
                         error=error, first=steps[0].number, last=steps[-1].number)}]
    raise InvalidReading(f"no valid reading in {attempts} replies: {error}")


def rewind_point(reading: Mapping) -> int:
    """Where to rewind for a reading: just before the step it names."""
    return reading["step"] - 1


# -- the rules -------------------------------------------------------------------

# Test summaries, as runners print them. Each is (passed, failed): a pattern
# that counts passing tests in its first group, and one that shows failures.
_SUMMARIES = (
    # pytest: "== 5 passed, 1 skipped in 0.12s ==", or "5 passed in 0.12s" with -q.
    (re.compile(r"^[=\s]*(\d+) passed\b.*\bin [\d.]+s\b", re.M),
     re.compile(r"^[=\s]*(?:\d+ \w+, )*\d+ (?:failed|errors?)\b.*\bin [\d.]+s\b", re.M)),
    # unittest (and Django's runner): "Ran 5 tests in 0.01s" then "OK" or "FAILED (failures=1)".
    (re.compile(r"^Ran (\d+) tests? in [\d.]+s\s*\n+\s*OK\b", re.M), re.compile(r"^FAILED \(", re.M)),
    # go test: "ok  	example.com/pkg	0.01s"; "FAIL" or "--- FAIL".
    (re.compile(r"^ok\s+\S+\s+(?:\(cached\)|[\d.]+s)", re.M), re.compile(r"^(?:--- )?FAIL\b", re.M)),
    # cargo test: "test result: ok. 5 passed; 0 failed".
    (re.compile(r"^test result: ok\. (\d+) passed", re.M), re.compile(r"^test result: FAILED", re.M)),
    # jest / vitest: "Tests:       5 passed, 5 total".
    (re.compile(r"^\s*Tests:\s+(\d+) passed", re.M), re.compile(r"^\s*Tests:\s+.*\d+ failed", re.M)),
    # mocha: "5 passing"; "1 failing".
    (re.compile(r"^\s*(\d+) passing\b", re.M), re.compile(r"^\s*\d+ failing\b", re.M)),
    # rspec: "5 examples, 0 failures".
    (re.compile(r"^(\d+) examples?, 0 failures", re.M), re.compile(r"^\d+ examples?, [1-9]\d* failures?", re.M)),
    # phpunit: "OK (5 tests, 9 assertions)"; "FAILURES!".
    (re.compile(r"^OK \((\d+) tests?,", re.M), re.compile(r"^(?:FAILURES|ERRORS)!", re.M)),
    # Maven surefire: "Tests run: 5, Failures: 0, Errors: 0".
    (re.compile(r"Tests run: (\d+), Failures: 0, Errors: 0", re.M),
     re.compile(r"Tests run: \d+, Failures: (?:[1-9]\d*, Errors: \d+|\d+, Errors: [1-9])", re.M)),
)


def shows_passing_tests(step: Step) -> bool:
    """Whether a step's output shows visible tests passing: a runner's summary of at least one
    passing test, no runner's summary of a failure, and no failing exit code."""
    if step.returncode not in (None, 0):
        return False
    passed = False
    for passing, failing in _SUMMARIES:
        if failing.search(step.output):
            return False
        for match in passing.finditer(step.output):
            passed |= not match.groups() or int(match.group(1)) > 0
    return passed


def last_passing_tests(steps: Sequence[Step]) -> int | None:
    """The last step after which the visible tests passed (a rewind point), or None if none did."""
    found = [step.number for step in _steps(steps) if shows_passing_tests(step)]
    return found[-1] if found else None


_HEREDOC = re.compile(r"<<(-?)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
_PATCHERS = re.compile(r"\b(?:git\s+apply|patch|apply_patch|applypatch)\b")
_INTERPRETERS = re.compile(r"\b(?:python[\d.]*|perl|ruby|node|bash|sh)\b")
_WRITES = re.compile(r"open\([^)]*['\"][wa]b?\+?['\"]|write_text|write_bytes|\.write\(|json\.dump\(|"
                     r"sed\s+-i|>\s*[^\s&>]")
_INTO_FILE = re.compile(r"(?<![0-9&<>])>>?\s*(?!&)(['\"]?)(/[^\s'\"]*|[^\s'\"/&|;][^\s'\"&|;]*)\1|\btee\s+(?:-a\s+)?(\S+)")
_IN_PLACE = re.compile(r"\b(?:sed|perl)\s+(?:-\w+\s+)*-i\b|\bperl\s+-\w*i\w*\b")


def _into_file(line):
    """Whether a command line sends its output into a file other than a scratch one."""
    for match in _INTO_FILE.finditer(line):
        target = match.group(2) or match.group(3) or ""
        if target and not target.startswith(("/dev/", "/tmp/", "/proc/")):
            return True
    return False


def edit_lines(command: str) -> int:
    """About how many lines a command writes into files; 0 for one that edits nothing.

    A heredoc sent into a file (``cat <<EOF > f``, ``tee f <<EOF``) counts its
    lines; a patch applied (``git apply``, ``patch``) its added and removed
    lines; a heredoc script that writes files (``python - <<EOF`` with
    ``open(..., "w")``) its lines; an in-place ``sed`` or ``perl``, or an
    ``echo``/``printf`` into a file, one line for each. Writes into /tmp, /dev
    and /proc are scratch, not edits.
    """
    lines, total, index = command.split("\n"), 0, 0
    while index < len(lines):
        line = lines[index]
        markers = list(_HEREDOC.finditer(line))
        index += 1
        if not markers:
            if _IN_PLACE.search(line):
                total += max(1, len(re.findall(r"(?<![\w])s([/|#@,:]).*?\1.*?\1", line)))
            elif re.search(r"\b(?:echo|printf)\b", line) and _into_file(line):
                total += 1 + line.count("\\n")
            continue
        for marker in markers:
            body = []
            while index < len(lines):
                text = lines[index].lstrip("\t") if marker.group(1) else lines[index]
                index += 1
                if text == marker.group(3):
                    break
                body.append(lines[index - 1])
            if _PATCHERS.search(line):
                total += sum(1 for item in body if item[:1] in ("+", "-") and not item.startswith(("+++", "---")))
            elif _into_file(line) or (_INTERPRETERS.search(line) and any(_WRITES.search(item) for item in body)):
                total += len(body)
    return total


def before_last_large_edit(steps: Sequence[Step], *, min_lines: int = 20) -> int | None:
    """The step before the last edit of at least ``min_lines`` lines (a rewind point), or None."""
    found = [step.number for step in _steps(steps) if edit_lines(step.command) >= min_lines]
    return found[-1] - 1 if found else None


# -- steps from a record -----------------------------------------------------------


def steps_from_trajectory(trajectory: Mapping) -> list[Step]:
    """The steps of a hosted agent's run, from its ATIF record.

    One step per model reply the agent received, in order; a reply the run's
    end cut off (the agent never saw it) is not a step. A step's command is
    every command its reply ran, as the agent wrote them, and its output what
    they printed as recorded (long outputs are recorded cut in the middle).
    """
    steps = []
    for item in trajectory.get("steps") or ():
        extra = item.get("extra") or {}
        if item.get("source") != "agent" or extra.get("cut_off"):
            continue
        results = (item.get("observation") or {}).get("results") or []
        commands = [str((result.get("extra") or {}).get("command", "")) for result in results]
        if not results:
            commands = [_call_command(call) for call in item.get("tool_calls") or ()]
        codes = [(result.get("extra") or {}).get("returncode") for result in results]
        steps.append(Step(len(steps) + 1, "\n".join(command for command in commands if command),
                          "\n".join(str(result.get("content", "")) for result in results),
                          codes[-1] if codes and type(codes[-1]) is int else None))
    return steps


def _call_command(call):
    arguments = call.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return ""
    return str(arguments.get("command", "")) if isinstance(arguments, Mapping) else ""


def final_message(trajectory: Mapping) -> str:
    """The agent's last words in an ATIF record: the last reply it received that said anything."""
    for item in reversed(trajectory.get("steps") or ()):
        if (item.get("source") == "agent" and not (item.get("extra") or {}).get("cut_off")
                and str(item.get("message", "")).strip()):
            return str(item["message"])
    return ""
