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
DEFAULT_CAP = 1200


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


def failures(trial_dir, task_dir, *, parsers=None):
    """``{"grader", "required", "failing", "excerpt"}`` for a graded trial; None when it has no grader output.

    ``failing`` is None when the results cannot be read (no parser for the repository). ``parsers`` maps a
    repository to its log parser (swebench's, by default).
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
    return {"grader": grader, "required": None if names is None else len(names), "failing": failing,
            "excerpt": excerpt(log, failing or [])}


def excerpt(log, names):
    """The output around each failing test's last mention; else the lines that carry a failure."""
    lines = log.splitlines()
    keep = set()
    for name in names:
        hits = [index for index, line in enumerate(lines) if name in line]
        leaf = PARTS.split(name)[-1].strip()
        if not hits and len(leaf) >= 8:
            hits = [index for index, line in enumerate(lines) if leaf in line and not NOISE.search(line)]
        if hits:
            last = hits[-1]
            keep.update(range(max(0, last - CONTEXT_BEFORE), min(len(lines), last + CONTEXT_AFTER + 1)))
    if not keep:
        keep = {index for index, line in enumerate(lines) if FAILURE.search(line)}
    chosen = [lines[index] for index in sorted(keep) if not NOISE.search(lines[index])]
    return "\n".join(chosen[-EXCERPT_LINES:]).strip()


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
