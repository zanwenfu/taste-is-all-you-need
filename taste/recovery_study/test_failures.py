"""The hidden tests' own verdict on a graded trial: the required tests that did not pass, and what they printed.

The recovery study's "tests" message shows an agent what a perfect reviewer would: which of the task's hidden tests
fail on its final state, and an excerpt of their output. No harness could deploy it (it reads the hidden tests); it
is the ceiling the reviewer's feedback is measured against. Two graders are read from what they left in Harbor's
verifier folder, never run again:

- SWE-bench Pro's Harbor adapter writes ``output.json`` (``{"tests": [{"name", "status"}]}``, the tests it saw
  pass) and the run script's output;
- the SWE-bench harness (SWE-bench Multilingual) writes its eval log, the tests' output between
  ``>>>>> Start Test Output`` and ``>>>>> End Test Output``; the per-test results are read from it with swebench's
  own log parser for the repository, as the grader does, loaded without the rest of swebench (``swebench_parsers``).

A test is required when the task lists it to fail-to-pass or pass-to-pass (``tests/config.json``); it did not pass
unless the grader's results show it passed (swebench's rule: a required test that never ran failed).

Two richer forms of the message test whether its form, not its content, held it back (an agent told a hidden
test's name looks for it in its repository and finds nothing): the same message followed by the failing tests' own
code, from the task's test patch (``test_code``), and a wide excerpt with a larger cap (``failures(wide=True)``).
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import json
import re
import sys
import types
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

START, END = ">>>>> Start Test Output", ">>>>> End Test Output"
PASSING = ("PASSED", "XFAIL")
# Lines that carry a failure, for an excerpt when no failing test is named in the output.
FAILURE = re.compile(r"FAIL|Error|error:|panic|Traceback|assert|expected|Exception", re.IGNORECASE)
# Lines an excerpt leaves out: passing tests and log chatter.
NOISE = re.compile(r"--- PASS|\.\.\. ok\b|\bPASSED\b|^\s*PASS\s|^\s*[✓√]|\b(DEBUG|TRACE)\b|\tINFO\t")
# How graders join a test's parts ("file | describe | test", jest's "suite \u203a test", "module::test"): the
# last part is what a log usually prints.
PARTS = re.compile(" \\| | \u203a |::| > ")
# Go prints a failure before its test's name, jest and cargo after it.
CONTEXT_BEFORE, CONTEXT_AFTER, EXCERPT_LINES = 8, 8, 60
# The wide excerpt of the "full output" message: five times the context around each failing test, more lines.
WIDE_BEFORE, WIDE_AFTER, WIDE_LINES = 40, 40, 400
DEFAULT_CAP = 1200
# A diff section of a test patch: one file's changes.
SECTION = re.compile(r"^diff --git ", re.MULTILINE)
CODE_HEAD = "The failing tests' code (the project's test changes for this task, as a diff):\n"


@dataclass(frozen=True)
class Spec:
    """What swebench's log parsers read of a task instance."""

    repo: str
    instance_id: str
    version: str = ""


def _names(value):
    """A config's test list: a list, or its JSON or Python text."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = ast.literal_eval(value)
    return [str(name) for name in value]


def _config(task_dir):
    path = Path(task_dir) / "tests" / "config.json"
    return json.loads(path.read_text()) if path.exists() else None


def required(task_dir):
    """The task's required tests, fail-to-pass first; None when its config is not there."""
    config = _config(task_dir)
    if config is None:
        return None
    lower = "fail_to_pass" in config
    names = _names(config.get("fail_to_pass" if lower else "FAIL_TO_PASS"))
    names += _names(config.get("pass_to_pass" if lower else "PASS_TO_PASS"))
    return list(dict.fromkeys(names))


def failures(trial_dir, task_dir, *, parsers=None, wide=False):
    """``{"grader", "required", "failing", "excerpt"}`` for a graded trial; None when it has no grader output.

    ``failing`` is None when the results cannot be read (no parser for the repository). ``parsers`` maps a
    repository to its log parser (swebench's, by default). ``wide``: the excerpt keeps five times the context
    around each failing test (the "full output" message).
    """
    verifier = Path(trial_dir) / "verifier"
    names = required(task_dir)
    if (verifier / "output.json").exists():
        results = json.loads((verifier / "output.json").read_text()).get("tests") or []
        status = {str(item.get("name")): str(item.get("status")) for item in results}
        log = "\n".join(_read(verifier / name) for name in ("run-script-stdout.txt", "run-script-stderr.txt"))
        grader = "swebench-pro"
    elif (verifier / "test-stdout.txt").exists():
        whole = _read(verifier / "test-stdout.txt")
        log = _between(whole)
        config = _config(task_dir) or {}
        parsers = swebench_parsers() if parsers is None else parsers
        parse = (parsers or {}).get(config.get("repo"))
        status = None
        if parse is not None:
            spec = Spec(repo=config["repo"], instance_id=str(config.get("instance_id", "")),
                        version=str(config.get("version", "")))
            # As the grader: the marked span first, the whole log when the span yields no results.
            results = parse(log, spec) or parse(whole, spec)
            status = {str(name): getattr(value, "value", value) for name, value in results.items()}
        grader = "swebench-harness"
    else:
        return None
    # Without the task's list of required tests, which failed is unknown (not "none").
    failing = None if status is None or names is None else [name for name in names if status.get(name) not in PASSING]
    shape = (WIDE_BEFORE, WIDE_AFTER, WIDE_LINES) if wide else (CONTEXT_BEFORE, CONTEXT_AFTER, EXCERPT_LINES)
    return {"grader": grader, "required": None if names is None else len(names), "failing": failing,
            "excerpt": excerpt(log, failing or [], *shape)}


def leaf(name):
    """What a log or a test file usually shows of a test's name: its last part."""
    return PARTS.split(name)[-1].strip()


def excerpt(log, names, before=CONTEXT_BEFORE, after=CONTEXT_AFTER, most=EXCERPT_LINES):
    """The output around each failing test's last mention; else the lines that carry a failure."""
    lines = log.splitlines()
    keep = set()
    for name in names:
        hits = [index for index, line in enumerate(lines) if name in line]
        part = leaf(name)
        if not hits and len(part) >= 8:
            hits = [index for index, line in enumerate(lines) if part in line and not NOISE.search(line)]
        if hits:
            last = hits[-1]
            keep.update(range(max(0, last - before), min(len(lines), last + after + 1)))
    if not keep:
        keep = {index for index, line in enumerate(lines) if FAILURE.search(line)}
    chosen = [lines[index] for index in sorted(keep) if not NOISE.search(lines[index])]
    return "\n".join(chosen[-most:]).strip()


def test_patch(task_dir):
    """The task's hidden test changes (the patch its grader applies before running the tests); "" when not found."""
    config = _config(task_dir) or {}
    return str(config.get("test_patch") or "")


def test_code(task_dir, names, cap):
    """The parts of the task's test patch that hold the failing tests: its file sections that name a failing test
    (in full, or by the name's last part), those naming the most failing tests first, then in the patch's order,
    within ``cap`` characters. "" when there is no patch or no section names one: such a failing test is one the
    agent's repository already holds."""
    sections = [part for part in SECTION.split(test_patch(task_dir)) if part.strip()]
    marks = [[mark for mark in {name, leaf(name)} if len(mark) >= 4] for name in names or ()]
    scored = sorted((-sum(any(mark in section for mark in named) for named in marks), index)
                    for index, section in enumerate(sections))
    shown, text = [sections[index] for hits, index in scored if hits < 0], ""
    for section in shown:
        part = "diff --git " + section.rstrip("\n") + "\n"
        if len(text) + len(part) <= cap:
            text += part
            continue
        if cap - len(text) > 200:
            text += part[:cap - len(text) - 1] + "…"
        break
    return text.strip()


def message(found, cap=DEFAULT_CAP):
    """What the agent is told: the failing tests by name, then their output, within ``cap`` characters; "" when
    the grader left nothing to tell."""
    if not found:
        return ""
    failing = found.get("failing")
    if failing:
        head, shown = "", list(failing)
        while shown:
            rest = len(failing) - len(shown)
            head = "These required tests did not pass: " + ", ".join(shown) + (
                f" (and {rest} more)." if rest else ".")
            if len(head) <= cap // 2 or len(shown) == 1:
                break
            shown.pop()
    else:
        head = ""
    output = (found.get("excerpt") or "").strip()
    if not output:
        return head[:cap]
    text = (head + "\n\n" if head else "") + "Test output:\n"
    room = cap - len(text)
    if room <= 1:
        return head[:cap]
    # The excerpt opens on the first failing test's own output: keep its start.
    return text + (output if len(output) <= room else output[:room - 1] + "…")


def code_message(found, code, cap=DEFAULT_CAP):
    """The tests message (within ``cap``), then the failing tests' code when there is any (``test_code``)."""
    told = message(found, cap)
    return told + (("\n\n" if told else "") + CODE_HEAD + code if code else "")


def _read(path):
    return Path(path).read_text(errors="replace") if Path(path).exists() else ""


def _between(text):
    """The test output between the harness's markers, cut as its grader cuts it; the whole log without them."""
    if START in text and END in text:
        return text.split(START)[1].split(END)[0]
    return text


@functools.cache
def swebench_parsers():
    """swebench's log parsers by repository, loaded from its installed files without the rest of the package.

    swebench's package imports pull in dataset and container tooling the parsers never use; the parser files need
    only its test-status names, a type, and its escape-code stripper. None when swebench is not installed.
    """
    found = importlib.util.find_spec("swebench")
    if found is None or not found.submodule_search_locations:
        return None
    root = Path(next(iter(found.submodule_search_locations))) / "harness" / "log_parsers"
    if not root.is_dir():
        return None
    status = Enum("TestStatus", {name: name for name in ("FAILED", "PASSED", "SKIPPED", "ERROR", "XFAIL")})
    stubs = {
        "swebench": types.ModuleType("swebench"),
        "swebench.harness": types.ModuleType("swebench.harness"),
        "swebench.harness.constants": types.ModuleType("swebench.harness.constants"),
        "swebench.harness.utils": types.ModuleType("swebench.harness.utils"),
        "swebench.harness.test_spec": types.ModuleType("swebench.harness.test_spec"),
        "swebench.harness.test_spec.test_spec": types.ModuleType("swebench.harness.test_spec.test_spec"),
        "swebench.harness.log_parsers": types.ModuleType("swebench.harness.log_parsers"),
    }
    stubs["swebench.harness.constants"].TestStatus = status
    stubs["swebench.harness.utils"].ansi_escape = lambda text: re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)
    stubs["swebench.harness.test_spec.test_spec"].TestSpec = Spec
    saved = {name: sys.modules.get(name) for name in stubs}
    parsers = {}
    try:
        sys.modules.update(stubs)
        for path in sorted(root.glob("*.py")):
            if path.name == "__init__.py":
                continue
            name = f"swebench.harness.log_parsers.{path.stem}"
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            for key, value in vars(module).items():
                if key.startswith("MAP_REPO_TO_PARSER") and isinstance(value, dict):
                    parsers.update(value)
            saved.setdefault(name, None)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return parsers
