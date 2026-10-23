"""What the replayed public runs say, in the paper's quantities.

    python scripts/public_replay_report.py --root /data/pubreplay

Reads every finished cell under ``<root>/cells/<submission>/<task>/`` (written
by scripts/public_replay.py) and writes ``<root>/report/``: cells.csv,
intervals.csv, states.csv, models.csv, models_all_usable.csv, summary.json,
summary.md.

The definitions are the paper's. Each is one function below, so every number
can be traced to the line that computed it.

1. Timeline. The ``shadow.jsonl`` lines of the replay's session, in ``seq``
   order. ``run`` is the tree before command 1; ``tool`` with step id
   ``cmd-NNN`` is the tree after command NNN (1-based); ``final`` is the tree
   after the last replayed command -- the same tree as its predecessor, tested
   a second time, so it doubles as a flakiness check. A state's test result is
   the ``states.jsonl`` line with its sha.
2. Base set. The PASS_TO_PASS members passing at the ``run`` state (members
   minus failed, unreported and skipped). A cell whose run state did not run
   cleanly (not run, any infrastructure error, or timed out) is unusable.
3. Flaky filter. Every cell of a task starts from the same tree, so P for a
   task is the intersection of its usable cells' base sets: a member passing
   at some bases and not at others is excluded for the whole task (counted).
   The disagreement rate between the final duplicate and its predecessor
   (one tree, tested twice) is reported beside it.
4. State labels. For a later state: test breaks = P members that failed or
   were skipped (the official grader counts a skip as not passing). Load
   breaks = P members that could not run: the suite reported nothing at all
   ("no results for any graded test"), or every member of their group (a
   pytest file or class, a django TestCase class) went unreported. A P member
   missing from a group the runner otherwise reported is a parser miss: counted
   as unmeasured, never a break. Hang = ran and timed out. Unknown = never
   tested, not run (the reset or the patch failed), or any other infrastructure
   error. Broken = test or load breaks; a broken state's kind is load if it has
   load breaks, else test. Hangs and unknowns are never breaks.
5. Exposure. For command i in 1..N (N = commands replayed), state_after(i) is
   the last tool state with NNN <= i, else the run state, and state_before(i)
   = state_after(i-1). Exposure = share of commands after which the tree is
   broken: the chance a stop after a uniformly random command leaves a broken
   tree. Unmeasured states count as not broken, so exposure is a lower bound,
   and unmeasured_share says how far it could move. Exposure after the first
   edit counts only commands from the first observed tree change onward.
   Wasted share = share of commands that started on a broken tree.
6. Breaks (intervals). Consecutive broken states of one kind (the final
   duplicate is ignored; hang and unknown states neither open nor close one).
   Onset a = the command that produced the first; the break ends at the next
   clean state (ended = clean, repair b = its command), at the next state broken
   in the other kind (ended = changed, b = its command), or not at all (ended =
   None). Duration = b - a commands, or N - a + 1 when still open.
7. Log visibility, from the ORIGINAL recorded commands, as the model saw them
   (the first and last 5,000 characters of a long output). Window W = commands
   a..b (a..N when open). ran_tests = a test-runner invocation in W whose output
   shows tests ran. saw_failure = such a run that reported a failure (a missing
   runner or nothing collected is not one). named = some output in W that is not
   a pure listing (cat, grep, sed -n, git diff, ...) shows this break: a broken
   test on a result line (its name and a failure word on one line), or one of
   the break's own error lines (from the instrument's runs at its broken states,
   less those already at the base); a syntax error counts only with the file
   and line the instrument's run pointed at. Silent = not saw_failure. Unseen
   (the headline, conservative) = not (saw_failure or named).
8. Shipped. For runs that submitted: whether the official grade of the
   replayed submission fails a P member (the scaffold's later versions submit a
   file list the agent chose, not its tree); the end tree's label is kept as
   end_broken and compared with the grade. For runs that never submitted: stop
   broken = the end tree (the final duplicate's predecessor) was broken -- the
   stop at which a harness that submits on its way out would ship a break.
9. Fidelity tier. Both tiers need a healthy replay: every command it was meant
   to replay ran, no recorder error, no skipped sync path, no transport failure,
   no replay exception, and at least 90% of comparable exit statuses as
   recorded (a timeout on one side only counts against). Exact = the original
   and the replay submitted the same patch. Unsubmitted = neither submitted.
   Diverged otherwise. Main results use usable exact + unsubmitted cells; every
   model row is repeated over all usable cells as a sensitivity.
Progress (extra columns; no break metric uses them). The suite also runs the
FAIL_TO_PASS tests, but P is built from PASS_TO_PASS alone. Per measured
state, f2p_passing = FAIL_TO_PASS members not failed, unreported or skipped.
A state is solved when every FAIL_TO_PASS member passes and it is not broken;
first_solved_cmd = the first command after which the tree was solved, and
overshoot = solved at some point but not at the end.

10. Aggregates per model (submission), and overall with 95% intervals from a
   cluster bootstrap over tasks (2,000 resamples, seed 0). Across models,
   Spearman's rho between the leaderboard's own resolve rate (all of the
   submission's instances) and mean exposure, with a two-sided permutation
   p-value (10,000 permutations, seed 0), and the two rankings side by side.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste.replay import base_test_id

ARM_SUFFIX = "__MSWE__t1"
NO_RESULTS = "no results for any graded test"
MAIN_TIERS = ("exact", "unsubmitted")
RC_AGREEMENT_FLOOR = 0.9
BOOTSTRAP_RESAMPLES = 2000
PERMUTATIONS = 10000

_STEP = re.compile(r"^cmd-(\d+)$")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# ----------------------------------------------------------------- the timeline


@dataclass(frozen=True)
class State:
    """One observed tree: where it sits on the command grid, and its test result."""

    seq: int
    sha: str
    trigger: str
    command: int
    """0 for the run state, NNN for a tool state, N for the final duplicate."""
    record: dict | None


@dataclass(frozen=True)
class Label:
    status: str
    """``ok``, ``broken``, ``hang`` or ``unknown``."""
    test_breaks: frozenset = frozenset()
    load_breaks: frozenset = frozenset()
    unmeasured: frozenset = frozenset()
    """P members the runner did not report in a group it otherwise reported: a parser miss, not a break."""

    @property
    def kind(self) -> str | None:
        """A broken state's kind: ``load`` when a group of tests could not run at all, else ``test``."""
        if self.status != "broken":
            return None
        return "load" if self.load_breaks else "test"

    @property
    def broken(self) -> bool:
        return self.status == "broken"

    @property
    def measured(self) -> bool:
        return self.status in ("ok", "broken")

    def same_as(self, other: Label) -> bool:
        return (self.status, self.test_breaks, self.load_breaks) == (other.status, other.test_breaks, other.load_breaks)


UNKNOWN = Label("unknown")


@dataclass(frozen=True)
class Interval:
    """One break: consecutive broken states of one kind (a phase)."""

    onset: int
    repair: int | None
    """The command that ended it: the next clean state's, or the next state that broke differently."""
    duration: int
    tests: frozenset
    loads: frozenset
    ended: str | None = None
    """``clean`` (repaired), ``changed`` (the tree broke in another way), or None (still open at the end)."""
    states: tuple[int, ...] = ()
    """Timeline indices of the broken states in it: where its evidence is read."""

    @property
    def kind(self) -> str:
        return "load" if self.loads else "test"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_timeline(shadow_lines: Iterable[dict], states: Iterable[dict], session: str, commands: int) -> list[State]:
    """Definition 1. Raises ValueError on a timeline the definitions cannot read."""
    by_sha: dict[str, dict] = {}
    for record in states:
        by_sha.setdefault(record["sha"], record)
    timeline: list[State] = []
    for line in sorted((entry for entry in shadow_lines if entry.get("session") == session), key=lambda e: e["seq"]):
        trigger = line.get("trigger")
        if trigger == "run":
            command = 0
        elif trigger == "final":
            command = commands
        elif trigger == "tool":
            match = _STEP.match(str(line.get("step_id", "")))
            if not match:
                raise ValueError(f"a tool observation with step id {line.get('step_id')!r}")
            command = int(match.group(1))
        else:
            raise ValueError(f"an observation with trigger {trigger!r}")
        timeline.append(State(seq=int(line["seq"]), sha=line["sha"], trigger=trigger, command=command,
                              record=by_sha.get(line["sha"])))
    if not timeline or timeline[0].trigger != "run":
        raise ValueError("the timeline does not start at the run state")
    tools = [s.command for s in timeline if s.trigger == "tool"]
    if tools != sorted(tools) or len(set(tools)) != len(tools):
        raise ValueError("tool observations out of command order")
    if tools and tools[-1] > commands:
        raise ValueError(f"an observation after command {tools[-1]} of {commands}")
    return timeline


# ------------------------------------------------------------ base set, labels


def base_clean(record: dict | None) -> bool:
    """Definition 2: the run state ran, with no infrastructure error, in time."""
    return bool(record) and bool(record.get("ran")) and record.get("infra_error") is None and not record.get("timed_out")


def base_passing(record: dict, members: Sequence[str]) -> frozenset:
    not_passing = set(record.get("failed", [])) | set(record.get("unreported", [])) | set(record.get("skipped", []))
    return frozenset(m for m in members if m not in not_passing)


def task_base_set(base_sets: Sequence[frozenset]) -> tuple[frozenset, frozenset]:
    """Definition 3: (P for the task, the members excluded as flaky at the base)."""
    if not base_sets:
        return frozenset(), frozenset()
    union = frozenset().union(*base_sets)
    common = frozenset.intersection(*base_sets)
    return common, union - common


def container_of(test_id: str) -> str | None:
    """The group a runner loads together: a pytest file or class, a django TestCase class.

    ``path::Class::test_x[p]`` -> ``path::Class``; ``path::test_x`` -> ``path``;
    django's ``test_x (app.tests.Class)`` -> ``app.tests.Class``. None when the id
    names no group (sympy's bare function names).
    """
    text = test_id.strip()
    if "::" in text:
        return base_test_id(text).rsplit("::", 1)[0]
    if text.endswith(")") and " (" in text:
        name, inside = text.rsplit(" (", 1)
        inside = inside[:-1]
        # Python 3.11's unittest repeats the method: `test_x (app.tests.Cls.test_x)`.
        if inside.endswith("." + name.strip()):
            inside = inside[: -len(name.strip()) - 1]
        return inside
    return None


def label_state(record: dict | None, base: frozenset, suite: Sequence[str] = ()) -> Label:
    """Definition 4.

    A test break is a P member that failed or was skipped (the official grader
    counts a skip as not passing). A load break is a P member that could not run:
    the suite reported nothing at all, or every member of its group (``suite``:
    all the suite's members, to know a group whole) went unreported. A P member
    missing from a group the runner otherwise reported is a parser miss, counted
    as unmeasured and never as a break.
    """
    if record is None or not record.get("ran"):
        return UNKNOWN
    if record.get("timed_out"):
        return Label("hang")
    if record.get("infra_error") not in (None, NO_RESULTS):
        return UNKNOWN
    unreported = frozenset(record.get("unreported", []))
    tests = base & (frozenset(record.get("failed", [])) | frozenset(record.get("skipped", [])))
    if record.get("infra_error") == NO_RESULTS:
        loads, unmeasured = base & unreported, frozenset()
    else:
        groups: dict[str, list[str]] = {}
        for member in suite or base:
            group = container_of(member)
            if group is not None:
                groups.setdefault(group, []).append(member)
        dead = {group for group, members in groups.items() if all(m in unreported for m in members)}
        loads = frozenset(m for m in base & unreported if container_of(m) in dead)
        unmeasured = (base & unreported) - loads
    return Label("broken" if tests or loads else "ok", tests, loads, unmeasured)


def f2p_passing(record: dict | None, fail_to_pass: Sequence[str]) -> int | None:
    """FAIL_TO_PASS members passing at a state; None when the state was not measured."""
    if record is None or not record.get("ran") or record.get("timed_out"):
        return None
    if record.get("infra_error") not in (None, NO_RESULTS):
        return None
    not_passing = set(record.get("failed", [])) | set(record.get("unreported", [])) | set(record.get("skipped", []))
    return sum(1 for m in fail_to_pass if m not in not_passing)


def solved(label: Label, passing: int | None, fail_to_pass: Sequence[str]) -> bool | None:
    """Every FAIL_TO_PASS member passes and no P member is broken; None when unmeasured."""
    if not label.measured or passing is None:
        return None
    return passing == len(fail_to_pass) and not label.broken


# ---------------------------------------------------- exposure and intervals


def after_labels(timeline: Sequence[State], labels: Sequence[Label], commands: int) -> list[Label]:
    """``after[i]`` = the tree after command i, for i in 0..N; ``after[0]`` is the run state."""
    tools = [(state.command, label) for state, label in zip(timeline, labels, strict=True) if state.trigger == "tool"]
    after: list[Label] = []
    current, j = labels[0], 0
    for i in range(commands + 1):
        while j < len(tools) and tools[j][0] <= i:
            current = tools[j][1]
            j += 1
        after.append(current)
    return after


def exposure(after: Sequence[Label], commands: int, first_edit: int | None) -> dict[str, float | None]:
    """Definition 5."""
    if commands <= 0:
        return {"exposure": None, "exposure_after_first_edit": None, "wasted_share": None, "unmeasured_share": None}
    span = range(1, commands + 1)
    result: dict[str, float | None] = {
        "exposure": sum(after[i].broken for i in span) / commands,
        "exposure_after_first_edit": None,
        "wasted_share": sum(after[i - 1].broken for i in span) / commands,
        "unmeasured_share": sum(not after[i].measured for i in span) / commands,
    }
    if first_edit is not None and 1 <= first_edit <= commands:
        tail = range(first_edit, commands + 1)
        result["exposure_after_first_edit"] = sum(after[i].broken for i in tail) / len(tail)
    return result


def break_intervals(timeline: Sequence[State], labels: Sequence[Label], commands: int) -> list[Interval]:
    """Definition 6: phases. A phase is consecutive broken states of one kind; it ends at the
    next clean state (``clean``) or the next state that is broken in the other way (``changed``)."""
    intervals: list[Interval] = []
    onset: int | None = None
    kind: str | None = None
    tests: set = set()
    loads: set = set()
    members: list[int] = []

    def close(end: int | None, how: str | None) -> None:
        duration = (end - onset) if end is not None else commands - onset + 1
        intervals.append(Interval(onset, end, duration, frozenset(tests), frozenset(loads), how, tuple(members)))

    for index, (state, label) in enumerate(zip(timeline, labels, strict=True)):
        if state.trigger == "final" or not label.measured:
            continue
        if label.broken:
            if onset is not None and label.kind != kind:
                close(state.command, "changed")
                onset = None
            if onset is None:
                onset, kind, tests, loads, members = state.command, label.kind, set(), set(), []
            tests |= label.test_breaks
            loads |= label.load_breaks
            members.append(index)
        elif onset is not None:
            close(state.command, "clean")
            onset = None
    if onset is not None:
        close(None, None)
    return intervals


# ------------------------------------------------------------- log visibility


def test_function_name(test_id: str) -> str | None:
    """The function behind a test id, or None when the id names none.

    ``path::Class::test_x[p]`` -> ``test_x``; django's ``test_x (app.tests.Class)``
    -> ``test_x``; a docstring id (django prints a test's docstring in its place)
    names no function and is skipped rather than matched as prose.
    """
    base = base_test_id(test_id.strip())
    if " (" in base:
        base = base.split(" (", 1)[0]
    if "::" in base:
        base = base.rsplit("::", 1)[1]
    names = _IDENT.findall(base)
    name = names[-1] if names else ""
    return name if name.lower().startswith("test") and len(name) > 4 else None


# Command segments, for telling a test run or a listing from anything else.
_SEGMENT = re.compile(r"&&|\|\||;|\||\n")
_PREFIX = re.compile(r"^(?:cd\s+\S+|timeout\s+(?:-\S+\s+)*\d+[smh]?|env(?:\s+-\S+)*|[A-Z_][A-Z0-9_]*=\S*|sudo|time|nice)\s*")
_RUNNER = re.compile(
    # An optional path before the program (agents outside the project's env called it by its full path),
    # python flags with or without one argument (-W ignore, -X dev), then the runner.
    r"^(?:\S*/)?(?:pytest|py\.test|nosetests|tox|bin/test|runtests\.py|manage\.py\s+test"
    r"|python[\d.]*\s+(?:-[A-Za-z]\s+(?!-)\S+\s+|-\S+\s+)*(?:-m\s+(?:pytest|unittest|nose|django\s+test)"
    r"|(?:\S*/)?(?:runtests\.py|bin/test|manage\.py\s+test)))(?=\s|$)"
)
_LISTING_WORDS = {"cat", "head", "tail", "nl", "less", "more", "grep", "egrep", "fgrep", "rg", "ls", "find", "wc",
                  "diff", "echo", "printf", "pwd", "which", "true", ":"}
_LISTING_PAIRS = {("sed", "-n"), ("git", "diff"), ("git", "show"), ("git", "log"), ("git", "status"), ("git", "grep")}
# A runner that ran tests says how many, or names a collection.
_TESTS_RAN = re.compile(r"\bRan [1-9]\d* tests?\b|\b[1-9]\d* (?:passed|failed)\b|\bcollected [1-9]\d* items?\b")
_RUN_FAILED = re.compile(r"\bFAILED\b|^FAIL:|^ERROR:|\b\d+ failed\b|\b\d+ errors?\b|errors=\d+|failures=\d+", re.M)
_RESULT_FAILURE = re.compile(r"\b(?:FAIL|FAILED|ERROR|Error|failed|FAILURE)\b")


def _segments(command: str) -> list[str]:
    out = []
    for part in _SEGMENT.split(command or ""):
        part = part.strip().lstrip("(").strip()
        while True:
            stripped = _PREFIX.sub("", part, count=1).strip()
            if stripped == part:
                break
            part = stripped
        if part:
            out.append(part)
    return out


def is_test_command(command: str) -> bool:
    """A test runner invoked as a command (not a file named after one: `cat tests/runtests.py`)."""
    return any(_RUNNER.match(segment) for segment in _segments(command))


def is_listing(command: str) -> bool:
    """A command that only prints files or state: its output quotes source, not a run."""
    segments = _segments(command)
    if not segments:
        return False
    for segment in segments:
        words = segment.split()
        if words[0] in _LISTING_WORDS or tuple(words[:2]) in _LISTING_PAIRS:
            continue
        return False
    return True


def tests_ran(output: str) -> bool:
    return bool(_TESTS_RAN.search(output or ""))


def shows_failure(returncode: int | None, output: str) -> bool:
    """A test run that ran tests and reported a failure (a missing runner, or nothing collected, is not)."""
    if not tests_ran(output):
        return False
    return (returncode is not None and returncode != 0) or bool(_RUN_FAILED.search(output or ""))


_COLOUR = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def evidence_cut(record: dict | None) -> bool:
    """Whether a state's error lines may sit in the part of its output that was not kept.

    A state keeps its output's first 2,000 and last 6,000 characters; with no
    error line found in them and both at their limit, its own error may be in
    the middle, and a break built on it cannot be called unseen with certainty.
    """
    if not record or record.get("exceptions"):
        return False
    return len(record.get("output_head") or "") >= 2000 and len(record.get("output_tail") or "") >= 6000


def seen_output(recorded: Any) -> str:
    """What the model was shown, as text: a v2 record keeps the whole output, but the model saw
    its first and last 5,000 characters (a v1 record keeps only those already); colour codes
    (astropy and sphinx force coloured pytest output) are not text and are dropped."""
    output = recorded.output or ""
    if getattr(recorded, "output_truncated", False) and len(output) > 10000:
        output = output[:5000] + "\n" + output[-5000:]
    return _COLOUR.sub("", output)


def _norm(text: str) -> str:
    return " ".join(text.split())


_SYNTAX_TYPES = ("SyntaxError", "IndentationError", "TabError")
_LOCATION = re.compile(r'File "([^"]+)", line (\d+)')


def shows_evidence(output: str, names: Sequence[str], exceptions: Sequence[str], locations: Sequence[str]) -> bool:
    """Whether an output shows this break: a broken test reported failing, or the break's own error.

    A broken test counts on a result line (its name and a failure word on one
    line). An error counts by its line as the instrument's run printed it; a
    syntax error's message is generic, so it counts only with the file and line
    the instrument's run pointed at.
    """
    if not output:
        return False
    for line in output.splitlines():
        if _RESULT_FAILURE.search(line) and any(re.search(rf"\b{re.escape(name)}\b", line) for name in names):
            return True
    flat = _norm(output)
    for error in exceptions:
        kind = error.split(":", 1)[0].rsplit(".", 1)[-1]
        if kind in _SYNTAX_TYPES:
            if kind not in output:
                continue
            for location in locations:
                found = _LOCATION.search(location)
                if not found:
                    continue
                name, line_no = found.group(1).rsplit("/", 1)[-1], found.group(2)
                if re.search(rf'{re.escape(name)}", line {line_no}\b|\({re.escape(name)}, line {line_no}\)', output):
                    return True
        else:
            needle = _norm(error)[:120]
            if len(needle) >= 12 and needle in flat:
                return True
    return False


def visibility(interval: Interval, recorded: Sequence[Any], commands: int,
               exceptions: Sequence[str] = (), locations: Sequence[str] = ()) -> dict[str, Any]:
    """Definition 7, over the recorded commands (``recorded[k - 1]`` is command k), as the model saw them.

    ``exceptions`` and ``locations`` are the break's own error lines (from the
    instrument's runs at its broken states, less those already at the base).
    """
    last = interval.repair if interval.repair is not None else commands
    window = [recorded[k - 1] for k in range(interval.onset, last + 1) if 1 <= k <= len(recorded)]
    names = sorted({name for t in interval.tests | interval.loads if (name := test_function_name(t))})
    # A test run by its command, or by its output (a runner called from a one-liner or a script),
    # never a listing (an old log that mentions a test count).
    tested = [c for c in window if is_test_command(c.command) or (tests_ran(seen_output(c)) and not is_listing(c.command))]
    saw_failure = any(shows_failure(c.returncode, seen_output(c)) for c in tested)
    named = any(shows_evidence(seen_output(c), names, exceptions, locations) for c in window if not is_listing(c.command))
    return {
        "window": len(window),
        "ran_tests": any(tests_ran(seen_output(c)) for c in tested),
        "saw_failure": saw_failure,
        "named": named,
        "silent": not saw_failure,
        # The headline, and conservative: unseen only if no test run the agent made
        # reported any failure and no output showed this break's own evidence.
        "unseen": not (saw_failure or named),
    }


# ------------------------------------------------------- end state, fidelity


def final_verdict(timeline: Sequence[State], labels: Sequence[Label]) -> tuple[Label | None, str, bool | None]:
    """Definition 8: (the end tree's label, where it was read, duplicate agreement)."""
    finals = [i for i, state in enumerate(timeline) if state.trigger == "final"]
    if not finals:
        last = labels[-1]
        return (last if last.measured else None), "last", None
    i = finals[-1]
    final = labels[i]
    before = labels[i - 1] if i > 0 else None
    agree = before.same_as(final) if before is not None and before.measured and final.measured else None
    if before is not None and before.measured:
        return before, "predecessor", agree
    if final.measured:
        return final, "final", agree
    return None, "unmeasured", agree


def _verdict_index(timeline: Sequence[State], labels: Sequence[Label], source: str) -> int | None:
    """Where :func:`final_verdict` read the end tree, as an index into the timeline."""
    finals = [i for i, state in enumerate(timeline) if state.trigger == "final"]
    if source == "last":
        return len(timeline) - 1
    if source == "predecessor":
        return finals[-1] - 1
    if source == "final":
        return finals[-1]
    return None


def guard_counterfactual(timeline: Sequence[State], labels: Sequence[Label], solved_at: Sequence[bool | None],
                         end_index: int | None, commands: int | None = None) -> dict[str, Any]:
    """What returning to the last clean tree would have shipped in place of a broken end tree.

    The guard the labels make possible: at a stop, go back to the last measured
    state with no break (the run's base when there is none). Reported: whether
    that state holds any of the run's work, whether it already solved the task
    (every target test passing), and how many commands of work it discards (to
    the run's last command when ``commands`` is given, else to the end tree).
    All None when the end tree is not broken.
    """
    none = {"guard_keeps_work": None, "guard_solved": None, "guard_discards_cmds": None}
    if end_index is None or not labels[end_index].broken:
        return none
    for i in range(end_index - 1, -1, -1):
        if labels[i].status == "ok":
            return {
                "guard_keeps_work": timeline[i].trigger == "tool",
                "guard_solved": None if solved_at[i] is None else bool(solved_at[i]),
                "guard_discards_cmds": (commands if commands is not None else timeline[end_index].command)
                - timeline[i].command,
            }
    return none


def grade_agreement(verdict: Label | None, evidence: dict, base: frozenset) -> str | None:
    """Our end-tree breaks against the official grade's PASS_TO_PASS failures, both within P."""
    if verdict is None or evidence.get("resolved") is None or not evidence.get("grade"):
        return None
    official = frozenset(evidence.get("grade_failed") or []) & base
    ours = verdict.test_breaks | verdict.load_breaks
    if not official and not ours:
        return "both_clean"
    if official == ours:
        return "same_breaks"
    if not ours:
        return "official_only"
    if not official:
        return "ours_only"
    return "both_different"


def rc_agreement(fidelity: dict) -> float | None:
    compared = int(fidelity.get("rc_compared") or 0)
    return int(fidelity.get("rc_matched") or 0) / compared if compared else None


def fidelity_tier(fidelity: dict, manifest: dict) -> str:
    """Definition 9.

    Both tiers need a healthy replay: every command it was meant to replay ran
    (none stopped it early), no recorder error, no skipped sync path, no
    transport failure, no replay exception, and at least 90% of comparable exit
    statuses as recorded (a one-sided timeout counts against). Exact: the
    original submitted and the replay submitted the same patch. Unsubmitted:
    neither submitted.
    """
    healthy = (
        int(fidelity.get("shadow_errors") or manifest.get("shadow_errors") or 0) == 0
        and not (fidelity.get("sync_skipped_paths") or manifest.get("sync_skipped_paths"))
        and not (fidelity.get("replay_exception") or manifest.get("exception"))
        and int(manifest.get("transport_errors") or 0) == 0
        and manifest.get("commands_replayed") == manifest.get("commands_to_replay")
    )
    agreement = rc_agreement(fidelity)
    agrees = agreement is None or agreement >= RC_AGREEMENT_FLOOR
    if fidelity.get("recorded_submitted"):
        exact = healthy and agrees and fidelity.get("replay_submitted") and fidelity.get("same_submission") is True
        return "exact" if exact else "diverged"
    if healthy and not fidelity.get("replay_submitted") and agreement is not None and agrees:
        return "unsubmitted"
    return "diverged"


# ------------------------------------------------------------------ the cells


@dataclass
class Cell:
    submission: str
    task: str
    state: str
    fidelity: dict
    manifest: dict
    evidence: dict
    shadow: list[dict]
    states: list[dict]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


PLANTED = "planted"


def find_cells(root: Path, *, planted: bool = False) -> list[Cell]:
    """The study's cells; with ``planted``, only the known-answer runs (never data)."""
    cells = []
    for status_path in sorted((root / "cells").glob("*/*/status.json")):
        directory = status_path.parent
        task = directory.name
        if (directory.parent.name == PLANTED) != planted:
            continue
        cells.append(Cell(
            submission=directory.parent.name, task=task, state=str(_read_json(status_path).get("state", "")),
            fidelity=_read_json(directory / "fidelity.json"), manifest=_read_json(directory / "manifest.json"),
            evidence=_read_json(directory / "ledger" / "evidence" / f"{task}{ARM_SUFFIX}.json"),
            shadow=read_jsonl(directory / "shadow.jsonl"), states=read_jsonl(directory / "states.jsonl"),
        ))
    return cells


def load_members(root: Path) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Per task: (PASS_TO_PASS, FAIL_TO_PASS). P comes from the first alone."""
    from taste.benchmarks import swebench

    return {i.instance_id: (tuple(i.pass_to_pass), tuple(i.fail_to_pass))
            for i in swebench.load_dataset(root / "verified.jsonl")}


def leaderboard_rates(root: Path) -> dict[str, float]:
    """Each submission's resolve rate as the leaderboard reports it (its metadata's ``resolved``).

    Not recomputed from the per-instance details file: seven submissions publish
    none, and one publishes a file that marks all 500 instances unresolved while
    its metadata reports 69.6%. The details file is the fallback when no
    metadata rate exists.
    """
    rates: dict[str, float] = {}
    listing = root / "submissions.json"
    if listing.exists():
        raw = json.loads(listing.read_text())
        for row in raw.get("submissions", raw) if isinstance(raw, dict) else raw:
            if isinstance(row, dict) and row.get("resolved_pct") is not None:
                rates[str(row["name"])] = float(row["resolved_pct"]) / 100
    for path in sorted((root / "details").glob("*.json")):
        if path.stem in rates:
            continue
        rows = [row for row in json.loads(path.read_text()).values() if isinstance(row, dict) and "resolved" in row]
        if rows:
            rates[path.stem] = sum(bool(row["resolved"]) for row in rows) / len(rows)
    return rates


def load_recorded_commands(root: Path, submission: str, task: str, version: str = "") -> list[Any]:
    """The original run's commands, as the agent's log recorded them.

    The submitting command's recorded output is the submitted patch, which the
    model never saw (the sentinel ends the run), so it is blanked: it is not
    part of what the agent's log showed it.
    """
    import dataclasses

    from taste.benchmarks.public_trajectories import parse_trajectory

    raw = json.loads((root / "trajs" / submission / f"{task}.traj.json").read_text())
    trajectory = parse_trajectory(raw, mini_version=version)
    commands = list(trajectory.commands)
    if trajectory.submit_index is not None and trajectory.submit_index < len(commands):
        commands[trajectory.submit_index] = dataclasses.replace(commands[trajectory.submit_index], output="")
    return commands


def analyse(root: Path) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Every done cell under the definitions: (cell rows, interval rows, instrument checks)."""
    members = load_members(root)
    cells = find_cells(root)
    done = [c for c in cells if c.state == "done"]

    prepared: dict[tuple[str, str], tuple[list[State], bool, frozenset, str | None]] = {}
    for cell in done:
        problem = None
        try:
            timeline = load_timeline(cell.shadow, cell.states, str(cell.manifest.get("session", "")),
                                     int(cell.manifest.get("commands_replayed") or 0))
        except (ValueError, KeyError) as exc:
            timeline, problem = [], f"timeline: {exc}"
        run = timeline[0].record if timeline else None
        if problem is None and cell.task not in members:
            problem = "task not in the dataset"
        elif problem is None and not base_clean(run):
            problem = "base state not clean"
        usable = problem is None
        prepared[(cell.submission, cell.task)] = (
            timeline, usable, base_passing(run, members[cell.task][0]) if usable else frozenset(), problem)

    bases: dict[str, list[frozenset]] = {}
    for (_, task), (_, usable, base, _) in prepared.items():
        if usable:
            bases.setdefault(task, []).append(base)
    task_sets = {task: task_base_set(sets) for task, sets in bases.items()}
    checks: dict[str, Any] = {
        "flaky_excluded": {task: len(excluded) for task, (_, excluded) in task_sets.items()},
        "never_passing": {task: len(set(members[task][0]) - set().union(*bases[task])) for task in bases},
    }

    rows: list[dict] = []
    intervals: list[dict] = []
    state_rows: list[dict] = []
    for cell in done:
        timeline, usable, _, problem = prepared[(cell.submission, cell.task)]
        tier = fidelity_tier(cell.fidelity, cell.manifest)
        row: dict[str, Any] = {
            "submission": cell.submission, "task": cell.task, "repo": cell.task.split("__")[0],
            "tier": tier, "usable": usable, "problem": problem or "", "main": usable and tier in MAIN_TIERS,
            "recorded_submitted": bool(cell.fidelity.get("recorded_submitted")),
            "recorded_exit_status": str(cell.manifest.get("recorded_exit_status") or ""),
            "leaderboard_resolved": cell.fidelity.get("leaderboard_resolved"),
            "replay_resolved": cell.evidence.get("resolved", cell.fidelity.get("replay_resolved")),
            "rc_agreement": rc_agreement(cell.fidelity),
        }
        if not usable:
            rows.append(row)
            continue
        base, excluded = task_sets[cell.task]
        commands = int(cell.manifest.get("commands_replayed") or 0)
        suite = tuple(members[cell.task][0]) + tuple(members[cell.task][1])
        labels = [label_state(state.record, base, suite) for state in timeline]
        base_errors = set((timeline[0].record or {}).get("exceptions", []))
        fail_to_pass = members[cell.task][1]
        passing = [f2p_passing(state.record, fail_to_pass) for state in timeline]
        solved_at = [solved(label, count, fail_to_pass) for label, count in zip(labels, passing, strict=True)]
        for state, label, count in zip(timeline, labels, passing, strict=True):
            state_rows.append({
                "submission": cell.submission, "task": cell.task, "seq": state.seq, "trigger": state.trigger,
                "command": state.command, "status": label.status, "test_breaks": len(label.test_breaks),
                "load_breaks": len(label.load_breaks), "f2p_passing": count, "f2p_total": len(fail_to_pass),
            })
        after = after_labels(timeline, labels, commands)
        tool_commands = [state.command for state in timeline if state.trigger == "tool"]
        later = [label for state, label in zip(timeline, labels, strict=True) if state.trigger == "tool"]
        cell_intervals = break_intervals(timeline, labels, commands)
        verdict, source, agree = final_verdict(timeline, labels)
        first_solved = next((state.command for state, ok in zip(timeline, solved_at, strict=True)
                             if state.trigger == "tool" and ok), None)
        end_index = _verdict_index(timeline, labels, source)
        end_solved = solved_at[end_index] if end_index is not None else None
        recorded: list[Any] = []
        if cell_intervals:
            try:
                recorded = load_recorded_commands(root, cell.submission, cell.task,
                                                  str(cell.manifest.get("scaffold_version_recorded", "")))
            except Exception as exc:  # visibility unknown, and said so; never guessed
                row["problem"] = f"recorded commands: {type(exc).__name__}: {exc}"
        submitted = row["recorded_submitted"]
        row.update({
            "commands": commands, "observations": len(tool_commands), "base_size": len(base),
            "flaky_excluded": len(excluded),
            "states_broken": sum(label.broken for label in later),
            "states_hang": sum(label.status == "hang" for label in later),
            "states_unknown": sum(label.status == "unknown" for label in later),
            "states_with_unmeasured": sum(bool(label.unmeasured) for label in later),
            "any_break": any(label.broken for label in later),
            "test_break": any(label.test_breaks for label in later),
            "load_break": any(label.load_breaks for label in later),
            "intervals": len(cell_intervals),
            **exposure(after, commands, tool_commands[0] if tool_commands else None),
            "final_source": source, "final_agree": agree,
            # What the run shipped: the official grade of the submitted patch, restricted to P
            # (a later scaffold submits a file list the agent chose, not its tree).
            "shipped_any": (bool(base & set(cell.evidence.get("grade_failed") or []))
                            if submitted and cell.evidence.get("grade") else None),
            "end_broken": verdict.broken if submitted and verdict else None,
            "shipped_test": bool(verdict.test_breaks) if submitted and verdict else None,
            "shipped_load": bool(verdict.load_breaks) if submitted and verdict else None,
            "stop_broken": verdict.broken if not submitted and verdict else None,
            "grade_agreement": grade_agreement(verdict, cell.evidence, base) if submitted else None,
            "f2p_total": len(fail_to_pass),
            "f2p_passing_end": passing[end_index] if end_index is not None else None,
            "first_solved_cmd": first_solved,
            "solved_end": end_solved,
            "overshoot": (first_solved is not None and end_solved is False) if end_solved is not None else None,
            **guard_counterfactual(timeline, labels, solved_at, end_index, commands),
        })
        silent = 0
        for interval in cell_intervals:
            errors: list[str] = []
            places: list[str] = []
            for index in interval.states:
                record = timeline[index].record or {}
                errors += [e for e in record.get("exceptions", []) if e not in base_errors and e not in errors]
                places += [loc for loc in record.get("error_locations", []) if loc not in places]
            seen = visibility(interval, recorded, commands, errors, places) if recorded else {
                key: None for key in ("window", "ran_tests", "saw_failure", "named", "silent", "unseen")}
            seen["evidence_complete"] = not any(evidence_cut(timeline[index].record) for index in interval.states)
            silent += bool(seen["silent"])
            intervals.append({
                "submission": cell.submission, "task": cell.task, "tier": tier, "main": row["main"],
                "onset": interval.onset, "repair": interval.repair, "duration": interval.duration,
                "kind": interval.kind, "ended": interval.ended, "open_at_end": interval.repair is None,
                "evidence_lines": len(errors),
                "tests": ";".join(sorted(interval.tests)), "load_members": len(interval.loads), **seen,
            })
        row["silent_intervals"] = silent if recorded or not cell_intervals else None
        rows.append(row)

    usable_rows = [r for r in rows if r["usable"]]
    compared = [r for r in usable_rows if r.get("final_agree") is not None]
    checks["final_duplicate"] = {
        "compared": len(compared), "disagreed": sum(1 for r in compared if not r["final_agree"]),
        "rate": (sum(1 for r in compared if not r["final_agree"]) / len(compared)) if compared else None,
    }
    checks["grade_agreement"] = dict(Counter(r["grade_agreement"] for r in rows if r["main"] and r.get("grade_agreement")))
    checks["resolved_agreement_exact"] = dict(Counter(
        f"leaderboard={r['leaderboard_resolved']},replay={r['replay_resolved']}" for r in rows if r["tier"] == "exact"))
    checks["cells_by_state"] = dict(Counter(c.state for c in cells))
    # Per-command fidelity over every finished, usable run (main or not): what "exited as recorded" rests on.
    checks["exit_status_agreement"] = {
        "matched": sum(int(c.fidelity.get("rc_matched") or 0) for c in done if prepared[(c.submission, c.task)][1]),
        "compared": sum(int(c.fidelity.get("rc_compared") or 0) for c in done if prepared[(c.submission, c.task)][1]),
    }
    checks["state_rows"] = state_rows
    return rows, intervals, checks


# --------------------------------------------------------------- aggregation


def mean(values: Iterable[Any]) -> float | None:
    kept = [float(v) for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def median(values: Iterable[Any]) -> float | None:
    kept = [v for v in values if v is not None]
    return float(statistics.median(kept)) if kept else None


def summarise(cells: Sequence[dict], intervals: Sequence[dict]) -> dict[str, Any]:
    """The aggregate metrics over a set of analysed (usable) cells and their intervals."""
    submitted = [c for c in cells if c["recorded_submitted"]]
    unsubmitted = [c for c in cells if not c["recorded_submitted"]]
    return {
        "runs": len(cells),
        "resolve_rate_sample": mean(c["leaderboard_resolved"] for c in cells),
        "runs_with_break_any": mean(c["any_break"] for c in cells),
        "runs_with_break_test": mean(c["test_break"] for c in cells),
        "runs_with_break_load": mean(c["load_break"] for c in cells),
        "intervals": len(intervals),
        "intervals_test": sum(1 for i in intervals if i["kind"] == "test"),
        "intervals_load": sum(1 for i in intervals if i["kind"] == "load"),
        "intervals_per_run": mean(c["intervals"] for c in cells),
        "repaired_share": mean((i["ended"] == "clean") for i in intervals),
        "unseen_no_tests_share": mean((not i["ran_tests"]) for i in intervals if i["unseen"] is True),
        "unseen_repaired_share": mean((i["ended"] == "clean") for i in intervals if i["unseen"] is True),
        "unseen_test_share": mean(i["unseen"] for i in intervals if i["kind"] == "test"),
        "unseen_load_share": mean(i["unseen"] for i in intervals if i["kind"] == "load"),
        # Unseen and never repaired: the quiet damage a stop or a submission hands over.
        "unseen_open_breaks": sum(1 for i in intervals if i["unseen"] is True and i["ended"] is None),
        "unseen_open_test_breaks": sum(1 for i in intervals if i["unseen"] is True and i["ended"] is None
                                       and i["kind"] == "test"),
        "end_tree_broken_share": mean(c.get("end_broken") if c["recorded_submitted"] else c.get("stop_broken") for c in cells),
        "limit_stop_runs": sum(1 for c in unsubmitted if c.get("recorded_exit_status") == "LimitsExceeded"
                               and c["stop_broken"] is not None),
        "limit_stop_broken_share": mean(c["stop_broken"] for c in unsubmitted
                                        if c.get("recorded_exit_status") == "LimitsExceeded"),
        "median_interval_duration": median(i["duration"] for i in intervals),
        "silent_interval_share": mean(i["silent"] for i in intervals),
        "unseen_interval_share": mean(i["unseen"] for i in intervals),
        # Bounds for the error lines a state's stored output may have cut: unseen among breaks whose
        # evidence is complete, and a floor that counts every incomplete unseen break as seen.
        "unseen_complete_share": mean(i["unseen"] for i in intervals if i.get("evidence_complete")),
        "breaks_evidence_complete": sum(1 for i in intervals if i.get("evidence_complete")),
        "unseen_floor_share": mean((i["unseen"] is True and bool(i.get("evidence_complete"))) if i["unseen"] is not None
                                   else None for i in intervals),
        "exposure_mean": mean(c["exposure"] for c in cells),
        "exposure_after_first_edit_mean": mean(c["exposure_after_first_edit"] for c in cells),
        "wasted_share_mean": mean(c["wasted_share"] for c in cells),
        "unmeasured_share_mean": mean(c["unmeasured_share"] for c in cells),
        "shipped_runs": sum(1 for c in submitted if c["shipped_any"] is not None),
        "shipped_share": mean(c["shipped_any"] for c in submitted),
        "stop_runs": sum(1 for c in unsubmitted if c["stop_broken"] is not None),
        "stop_broken_share": mean(c["stop_broken"] for c in unsubmitted),
        "ever_solved_share": mean((c["first_solved_cmd"] is not None) for c in cells),
        "broken_end_runs": sum(1 for c in cells if c.get("guard_keeps_work") is not None),
        "guard_keeps_work_share": mean(c.get("guard_keeps_work") for c in cells),
        "guard_solved_share": mean(c.get("guard_solved") for c in cells),
        "overshoot_share": mean(c["overshoot"] for c in cells),
    }


CI_METRICS = ("runs_with_break_any", "runs_with_break_test", "runs_with_break_load", "silent_interval_share",
              "unseen_interval_share", "end_tree_broken_share", "limit_stop_broken_share",
              "exposure_mean", "exposure_after_first_edit_mean", "wasted_share_mean", "shipped_share",
              "stop_broken_share")


def percentile(ordered: Sequence[float], q: float) -> float | None:
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def cluster_bootstrap(cells: Sequence[dict], intervals: Sequence[dict], *, resamples: int = BOOTSTRAP_RESAMPLES,
                      seed: int = 0) -> dict[str, tuple[float | None, float | None]]:
    """95% percentile intervals for CI_METRICS, resampling whole tasks (runs of a task share its suite)."""
    tasks = sorted({c["task"] for c in cells})
    if not tasks:
        return {metric: (None, None) for metric in CI_METRICS}
    cells_by: dict[str, list[dict]] = {}
    intervals_by: dict[str, list[dict]] = {}
    for c in cells:
        cells_by.setdefault(c["task"], []).append(c)
    for i in intervals:
        intervals_by.setdefault(i["task"], []).append(i)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {metric: [] for metric in CI_METRICS}
    for _ in range(resamples):
        picked = [rng.choice(tasks) for _ in tasks]
        stats = summarise([c for t in picked for c in cells_by[t]], [i for t in picked for i in intervals_by.get(t, [])])
        for metric in CI_METRICS:
            if stats[metric] is not None:
                draws[metric].append(stats[metric])
    return {metric: (percentile(sorted(v), 0.025), percentile(sorted(v), 0.975)) for metric, v in draws.items()}


def ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks, ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda k: values[k])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / math.sqrt(sxx * syy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return pearson(ranks(xs), ranks(ys))


def permutation_p(xs: Sequence[float], ys: Sequence[float], *, permutations: int = PERMUTATIONS,
                  seed: int = 0) -> float | None:
    """Two-sided: the share of shuffles of ``ys`` whose |rho| reaches the observed one (+1 smoothing)."""
    observed = spearman(xs, ys)
    if observed is None:
        return None
    rng = random.Random(seed)
    rx, shuffled, hits = ranks(xs), list(ys), 0
    for _ in range(permutations):
        rng.shuffle(shuffled)
        rho = pearson(rx, ranks(shuffled))
        if rho is not None and abs(rho) >= abs(observed) - 1e-12:
            hits += 1
    return (hits + 1) / (permutations + 1)


def model_rows(rows: Sequence[dict], intervals: Sequence[dict], cells_by_state: dict[str, Counter],
               leaderboard: dict[str, float], *, main_only: bool) -> list[dict]:
    out = []
    for submission in sorted({r["submission"] for r in rows} | set(cells_by_state)):
        mine = [r for r in rows if r["submission"] == submission]
        chosen = [r for r in mine if r["main"]] if main_only else [r for r in mine if r["usable"]]
        keys = {(r["submission"], r["task"]) for r in chosen}
        tiers = Counter(r["tier"] for r in mine)
        states = cells_by_state.get(submission, Counter())
        out.append({
            "submission": submission, "cells": sum(states.values()), "not_done": sum(states.values()) - states.get("done", 0),
            "usable": sum(1 for r in mine if r["usable"]), "exact": tiers.get("exact", 0),
            "unsubmitted": tiers.get("unsubmitted", 0), "diverged": tiers.get("diverged", 0),
            "resolve_rate_leaderboard": leaderboard.get(submission),
            **summarise(chosen, [i for i in intervals if (i["submission"], i["task"]) in keys]),
        })
    return out


def cross_model(models: Sequence[dict]) -> dict[str, Any]:
    """Spearman between the leaderboard resolve rate and mean exposure, and both rankings."""
    kept = [m for m in models if m["resolve_rate_leaderboard"] is not None and m["exposure_mean"] is not None]
    xs = [m["resolve_rate_leaderboard"] for m in kept]
    ys = [m["exposure_mean"] for m in kept]
    by_resolve = ranks([-x for x in xs])
    by_exposure = ranks(ys)
    ranking = sorted(({"submission": m["submission"], "resolve_rate": x, "exposure": y,
                       "rank_resolve": rr, "rank_exposure": re_}
                      for m, x, y, rr, re_ in zip(kept, xs, ys, by_resolve, by_exposure, strict=True)),
                     key=lambda r: r["rank_resolve"])
    return {"models": len(kept), "spearman": spearman(xs, ys), "p_value": permutation_p(xs, ys), "ranking": ranking}


# ---------------------------------------------------------------------- output


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _ci(bounds: tuple[float | None, float | None]) -> str:
    low, high = bounds
    return "" if low is None else f" [{100 * low:.1f}, {100 * high:.1f}]"


def headline(summary: dict[str, Any]) -> str:
    main, ci = summary["overall_main"], summary["overall_main_ci"]
    cells = summary["cells"]
    corr = summary["cross_model"]
    rho = "n/a" if corr["spearman"] is None else f"{corr['spearman']:.2f} (p={corr['p_value']:.3f})"
    return "\n".join([
        f"cells: {cells['total']} (done {cells['done']}; usable {cells['usable']}; "
        f"exact {cells['tiers'].get('exact', 0)}, unsubmitted {cells['tiers'].get('unsubmitted', 0)}, "
        f"diverged {cells['tiers'].get('diverged', 0)}; main {cells['main']})",
        f"runs with a break: {_pct(main['runs_with_break_any'])}{_ci(ci['runs_with_break_any'])} "
        f"(test {_pct(main['runs_with_break_test'])}, load {_pct(main['runs_with_break_load'])})",
        f"break intervals: {main['intervals']}; unseen in the log: {_pct(main['unseen_interval_share'])}"
        f"{_ci(ci['unseen_interval_share'])}; no failing test run: {_pct(main['silent_interval_share'])}"
        f"{_ci(ci['silent_interval_share'])}; median length {main['median_interval_duration']} commands",
        f"exposure: {_pct(main['exposure_mean'])}{_ci(ci['exposure_mean'])}; after the first edit "
        f"{_pct(main['exposure_after_first_edit_mean'])}; wasted {_pct(main['wasted_share_mean'])}; "
        f"unmeasured {_pct(main['unmeasured_share_mean'])}",
        f"shipped (submitted runs): {_pct(main['shipped_share'])}{_ci(ci['shipped_share'])} of {main['shipped_runs']}; "
        f"stop broken (unsubmitted runs): {_pct(main['stop_broken_share'])}{_ci(ci['stop_broken_share'])} "
        f"of {main['stop_runs']}",
        f"broken end trees: {main['broken_end_runs']}; a last-clean-state guard keeps work in "
        f"{_pct(main['guard_keeps_work_share'])} and ships a solved tree in {_pct(main['guard_solved_share'])}",
        f"Spearman(leaderboard resolve rate, exposure) over {corr['models']} models: {rho}",
        f"final duplicate disagreement: {summary['checks']['final_duplicate']}",
    ])


def planted_checks(root: Path) -> dict[str, Any]:
    """The known-answer runs, labelled with the study's own definitions and compared with what was planted.

    ``root/planted/<task>.json`` holds ``expected.intervals`` as onset, repair and
    kind; a planted run is its own base (it has no sibling runs to intersect).
    """
    members = load_members(root)
    results: dict[str, Any] = {}
    for cell in find_cells(root, planted=True):
        spec_path = root / "planted" / f"{cell.task}.json"
        if cell.state != "done" or not spec_path.exists() or cell.task not in members:
            continue
        expected = [(i["onset_cmd"], i["repair_cmd"], i["kind"]) for i in json.loads(spec_path.read_text())["expected"]["intervals"]]
        commands = int(cell.manifest.get("commands_replayed") or 0)
        timeline = load_timeline(cell.shadow, cell.states, str(cell.manifest.get("session", "")), commands)
        p2p, f2p = members[cell.task]
        base = base_passing(timeline[0].record, p2p)
        labels = [label_state(state.record, base, tuple(p2p) + tuple(f2p)) for state in timeline]
        found = [(i.onset, i.repair, i.kind) for i in break_intervals(timeline, labels, commands)]
        results[cell.task] = {"expected": expected, "found": found, "pass": found == expected}
    return results


def run(root: Path, out: Path) -> dict[str, Any]:
    rows, intervals, checks = analyse(root)
    checks["planted"] = planted_checks(root)
    state_rows = checks.pop("state_rows")
    leaderboard = leaderboard_rates(root)
    cells_by_state: dict[str, Counter] = {}
    for cell in find_cells(root):
        cells_by_state.setdefault(cell.submission, Counter())[cell.state] += 1
    main = [r for r in rows if r["main"]]
    usable = [r for r in rows if r["usable"]]
    main_keys = {(r["submission"], r["task"]) for r in main}
    main_intervals = [i for i in intervals if (i["submission"], i["task"]) in main_keys]
    models = model_rows(rows, intervals, cells_by_state, leaderboard, main_only=True)
    models_all = model_rows(rows, intervals, cells_by_state, leaderboard, main_only=False)
    summary: dict[str, Any] = {
        "cells": {"total": sum(sum(c.values()) for c in cells_by_state.values()), "done": len(rows),
                  "usable": len(usable), "main": len(main), "tiers": dict(Counter(r["tier"] for r in rows)),
                  "by_state": checks["cells_by_state"]},
        "overall_main": summarise(main, main_intervals),
        "overall_main_ci": cluster_bootstrap(main, main_intervals),
        "overall_all_usable": summarise(usable, intervals),
        "cross_model": cross_model(models),
        "checks": {
            "flaky_excluded_total": sum(checks["flaky_excluded"].values()),
            "never_passing_total": sum(checks["never_passing"].values()),
            "tasks": len(checks["flaky_excluded"]),
            "final_duplicate": checks["final_duplicate"],
            "grade_agreement_main_submitted": checks["grade_agreement"],
            "resolved_agreement_exact": checks["resolved_agreement_exact"],
            "planted": checks["planted"],
            "exit_status_agreement": checks["exit_status_agreement"],
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "cells.csv", rows)
    write_csv(out / "intervals.csv", intervals)
    write_csv(out / "states.csv", state_rows)
    write_csv(out / "models.csv", models)
    write_csv(out / "models_all_usable.csv", models_all)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=list))
    text = headline(summary)
    lines = ["# Public-run replay: summary", "", "```", text, "```", "", "## Rankings (leaderboard resolve rate vs exposure)", "",
             "| submission | resolve rate | exposure | rank by resolve | rank by exposure (1 = safest) |", "|---|---|---|---|---|"]
    for r in summary["cross_model"]["ranking"]:
        lines.append(f"| {r['submission']} | {_pct(r['resolve_rate'])} | {_pct(r['exposure'])} | "
                     f"{r['rank_resolve']:g} | {r['rank_exposure']:g} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(text)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="/data/pubreplay")
    ap.add_argument("--out", default="", help="Default: <root>/report")
    args = ap.parse_args(argv)
    root = Path(args.root)
    run(root, Path(args.out) if args.out else root / "report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
