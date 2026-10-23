"""The replay study's analysis: each definition on its own, then a whole report over synthetic cells.

Hermetic and small: cells are written as scripts/public_replay.py writes them,
and the recorded commands (the original runs' logs) are stubbed at the one
seam that reads trajectories.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "public_replay_report", Path(__file__).resolve().parents[1] / "scripts" / "public_replay_report.py")
report = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = report  # its dataclasses resolve annotations through the module table
_SPEC.loader.exec_module(report)

T1 = "tests/test_mod.py::test_one"
T2 = "tests/test_mod.py::test_two[param]"
T3 = "tests/test_mod.py::test_three"
F1 = "tests/test_mod.py::test_fix"
NO_RESULTS = "no results for any graded test"


def rec(sha, *, ran=True, timed_out=False, infra=None, failed=(), unreported=(), skipped=()):
    """One states.jsonl line, shaped as public_replay.state_record writes it."""
    return {"sha": sha, "seconds": 1.0, "ran": ran, "exit_code": 1 if failed else 0,
            "timed_out": timed_out if ran else None, "infra_error": infra, "passed": 0,
            "failed": list(failed), "unreported": list(unreported), "skipped": list(skipped), "output_tail": ""}


def cmd(command, rc=0, output="", truncated=False):
    return SimpleNamespace(command=command, returncode=rc, output=output, timed_out=rc is None, output_truncated=truncated)


# ------------------------------------------------------------- the definitions


@pytest.mark.parametrize("test_id, name", [
    ("tests/test_mod.py::Cls::test_x[a-b]", "test_x"),
    ("test_y (app.tests.Cls)", "test_y"),
    ("test_y (app.tests.Cls.test_y)", "test_y"),
    ("test_Add", "test_Add"),
    ("mod.Class.test_w", "test_w"),
    ("A docstring that django printed in place of the name", None),
    ("test", None),
])
def test_a_broken_test_is_named_by_its_function(test_id, name) -> None:
    assert report.test_function_name(test_id) == name


@pytest.mark.parametrize("command, expected", [
    ("python -m pytest tests/test_mod.py -q", True),
    ("cd /testbed && ./tests/runtests.py queries --verbosity 2", True),
    ("cd /testbed && python tests/runtests.py aggregation", True),
    ("bin/test sympy/core/tests/test_basic.py", True),
    ("tox -e py39", True),
    ("python -m unittest discover", True),
    ("timeout 60 python -m pytest -x", True),
    ("cd /testbed && PYTHONPATH=. pytest -q tests", True),
    ("python -m django test --settings=x", True),
    ("python manage.py test app", True),
    ("cat ./tests/test_mod.py", False),
    ("grep -rn foo src/ && ls", False),
    ("sed -n '1,80p' src/_pytest/python.py", False),
    ("cat tox.ini", False),
    ("grep -n x tests/runtests.py", False),
    ("pip install pytest", False),
    ("ls bin/test", False),
    ("cat > test_x.py <<'EOF'\nimport unittest\nEOF", False),
    ("/opt/miniconda3/envs/testbed/bin/python -m pytest tests/test_mod.py", True),
    ("cd /testbed && /usr/bin/pytest -q", True),
    ("python -W ignore -m pytest tests", True),
    ("./runtests.py aggregation", True),
    ("python3.9 -X dev -m unittest tests.test_x", True),
    ("python manage.py testserver", False),
])
def test_test_runner_invocations(command, expected) -> None:
    assert report.is_test_command(command) is expected


def test_failure_signals() -> None:
    """A failure counts only from a run that ran tests: a missing runner or nothing collected is not one."""
    assert report.shows_failure(1, "Ran 3 tests in 0.01s\n\nFAILED (failures=1)") is True
    assert report.shows_failure(0, "FAILED tests/x.py::t - AssertionError\n1 failed, 2 passed in 0.1s") is True
    assert report.shows_failure(0, "5 passed in 0.10s") is False
    assert report.shows_failure(1, "") is False
    assert report.shows_failure(1, "/opt/miniconda3/bin/python: No module named pytest") is False
    assert report.shows_failure(5, "collected 0 items\n\nno tests ran in 0.01s") is False
    assert report.shows_failure(None, "Command timed out") is False
    assert report.shows_failure(1, "Traceback (most recent call last):\nNameError: x") is False


def test_state_labels_follow_definition_four() -> None:
    base = frozenset({T1, T2})
    broken = report.label_state(rec("s", failed=[T1, T3, F1]), base)
    assert broken.status == "broken" and broken.test_breaks == {T1} and not broken.load_breaks
    load = report.label_state(rec("s", infra=NO_RESULTS, unreported=[T1, T2, T3, F1]), base)
    assert load.status == "broken" and load.load_breaks == {T1, T2} and not load.test_breaks
    assert report.label_state(rec("s", timed_out=True, infra="timed out"), base).status == "hang"
    assert report.label_state(rec("s", ran=False, infra="patch failed: x"), base).status == "unknown"
    assert report.label_state(rec("s", infra="parse failed: boom", unreported=[T1, T2]), base).status == "unknown"
    assert report.label_state(None, base).status == "unknown"
    # a member outside P (never passing, or flaky at the base) never makes a break
    assert report.label_state(rec("s", failed=[T3, F1]), base).status == "ok"
    # a skipped P member is not passing, as the official grader counts it
    skipped = report.label_state(rec("s", skipped=[T2]), base)
    assert skipped.status == "broken" and skipped.test_breaks == {T2}


def test_a_test_missing_from_a_group_that_otherwise_ran_is_a_parser_miss_not_a_break() -> None:
    """Only a whole group that could not run is a load break (a module that will not import, a
    class whose set-up dies); one unreported test among reported ones is unmeasured."""
    suite = (T1, T2, T3, F1)
    base = frozenset({T1, T2})
    miss = report.label_state(rec("s", unreported=[T1]), base, suite)
    assert miss.status == "ok" and miss.unmeasured == {T1} and not miss.load_breaks
    whole = report.label_state(rec("s", unreported=[T1, T2, T3, F1]), base, suite)
    assert whole.status == "broken" and whole.load_breaks == {T1, T2} and whole.kind == "load"
    a1, a2, b1 = "test_a (app.tests.A)", "test_b (app.tests.A)", "test_c (app.tests.B)"
    django = report.label_state(rec("s", unreported=[a1, a2]), frozenset({a1, a2, b1}), (a1, a2, b1))
    assert django.load_breaks == {a1, a2} and django.status == "broken"
    assert report.container_of(a1) == "app.tests.A" and report.container_of(T2) == "tests/test_mod.py"
    assert report.container_of("tests/x.py::Cls::test_y[1]") == "tests/x.py::Cls"
    assert report.container_of("test_Add") is None
    # Python 3.11's unittest names a test with its method too: the group is still the class.
    assert report.container_of("test_x (app.tests.Cls.test_x)") == "app.tests.Cls"
    assert report.container_of("test_x (app.tests.Cls)") == "app.tests.Cls"


def test_the_base_set_is_previously_passing_tests_only_and_flaky_ones_leave_the_task() -> None:
    record = rec("s", failed=[F1], skipped=[T3])
    assert report.base_passing(record, [T1, T2, T3]) == {T1, T2}
    common, excluded = report.task_base_set([frozenset({T1, T2, T3}), frozenset({T1, T2})])
    assert common == {T1, T2} and excluded == {T3}
    assert report.task_base_set([]) == (frozenset(), frozenset())


def _states(*specs):
    """(trigger, command, label) triples -> (timeline, labels)."""
    timeline = [report.State(seq=i, sha=f"s{i}", trigger=t, command=c, record=None) for i, (t, c, _) in enumerate(specs)]
    return timeline, [label for _, _, label in specs]


OK = report.Label("ok")
BROKEN_T1 = report.Label("broken", frozenset({T1}))
BROKEN_LOAD = report.Label("broken", frozenset(), frozenset({T1, T2}))
HANG = report.Label("hang")


def test_exposure_and_wasted_share_follow_the_command_grid() -> None:
    timeline, labels = _states(("run", 0, OK), ("tool", 2, BROKEN_T1), ("tool", 4, OK), ("tool", 5, BROKEN_LOAD),
                               ("final", 6, BROKEN_LOAD))
    after = report.after_labels(timeline, labels, 6)
    assert [a.status for a in after] == ["ok", "ok", "broken", "broken", "ok", "broken", "broken"]
    measures = report.exposure(after, 6, first_edit=2)
    assert measures["exposure"] == pytest.approx(4 / 6)
    assert measures["exposure_after_first_edit"] == pytest.approx(4 / 5)
    assert measures["wasted_share"] == pytest.approx(3 / 6)
    assert measures["unmeasured_share"] == 0
    assert report.exposure(after, 0, None)["exposure"] is None


def test_intervals_close_only_on_a_clean_state_and_skip_unmeasured_ones() -> None:
    timeline, labels = _states(("run", 0, OK), ("tool", 2, BROKEN_T1), ("tool", 3, HANG), ("tool", 4, BROKEN_T1),
                               ("tool", 6, OK), ("tool", 7, BROKEN_LOAD), ("final", 9, OK))
    intervals = report.break_intervals(timeline, labels, 9)
    assert [(i.onset, i.repair, i.duration, i.kind) for i in intervals] == [(2, 6, 4, "test"), (7, None, 3, "load")]


def test_visibility_reads_the_original_log_for_this_breaks_own_evidence() -> None:
    """Seen means the log shows this break: a broken test reported failing, or the break's own error."""
    recorded = [
        cmd("ls"), cmd("sed -i 's/a/b/' pkg/mod.py"),
        cmd("python -m pytest tests/test_mod.py", 1, "FAILED tests/test_mod.py::test_one - AssertionError\n1 failed, 2 passed in 0.1s"),
        cmd("sed -i 's/b/a/' pkg/mod.py"), cmd("printf 'x = (' > pkg/mod.py"), cmd("git diff"),
    ]
    seen = report.visibility(report.Interval(2, 4, 2, frozenset({T1}), frozenset()), recorded, 6)
    assert seen == {"window": 3, "ran_tests": True, "saw_failure": True, "named": True, "silent": False, "unseen": False}
    quiet = report.visibility(report.Interval(5, None, 2, frozenset(), frozenset({T1})), recorded, 6)
    assert (quiet["ran_tests"], quiet["silent"], quiet["named"], quiet["unseen"]) == (False, True, False, True)

    name_error = "NameError: name 'having_group_by' is not defined"
    # The agent's reproduction script printed the break's own error: not a test run, but seen.
    repro = [cmd("sed -i '139d' compiler.py"), cmd("python repro.py", 1, f"Traceback (most recent call last):\n{name_error}")]
    shown = report.visibility(report.Interval(1, None, 2, frozenset(), frozenset({T1})), repro, 2, [name_error])
    assert (shown["silent"], shown["named"], shown["unseen"]) == (True, True, False)
    # Another error is not this break's.
    other = [cmd("python repro.py", 1, "NameError: name 'y' is not defined")]
    assert report.visibility(report.Interval(1, None, 1, frozenset(), frozenset({T1})), other, 1, [name_error])["unseen"] is True
    # A source listing that quotes an error type is not a run.
    listing = [cmd("grep -n 'except ImportError' pkg/mod.py", 0, "12: except ImportError: cannot import name 'Random'")]
    assert report.visibility(report.Interval(1, None, 1, frozenset(), frozenset({T1})), listing, 1,
                             ["ImportError: cannot import name 'Random'"])["unseen"] is True
    # A syntax error counts only at the file and line the instrument's run pointed at.
    where = ['File "/testbed/pylint/lint/pylinter.py", line 614']
    syntax = "SyntaxError: invalid syntax"
    at = [cmd("python -c 'import pylint.lint'", 1, f'  File "/testbed/pylint/lint/pylinter.py", line 614\n    if x(\n{syntax}')]
    elsewhere = [cmd("python -c 'import pylint.lint'", 1, f'  File "/testbed/pylint/lint/pylinter.py", line 700\n{syntax}')]
    pytest_form = [cmd("python -m pytest tests", 2, f"E   {syntax} (pylinter.py, line 614)\ncollected 0 items / 1 error")]
    interval = report.Interval(1, None, 1, frozenset(), frozenset({T1}))
    assert report.visibility(interval, at, 1, [syntax], where)["unseen"] is False
    assert report.visibility(interval, elsewhere, 1, [syntax], where)["unseen"] is True
    assert report.visibility(interval, pytest_form, 1, [syntax], where)["unseen"] is False
    # A runner that is not installed ran no test: no failure seen.
    missing = [cmd("python -m pytest tests", 1, "/usr/bin/python: No module named pytest")]
    assert report.visibility(interval, missing, 1)["saw_failure"] is False
    # Listing the test file names the test but reports nothing.
    names = [cmd("cat tests/test_mod.py", 0, "def test_one():\n    assert broken()  # Error")]
    assert report.visibility(report.Interval(1, None, 1, frozenset({T1}), frozenset()), names, 1)["named"] is False
    # A name must match as a word.
    near = [cmd("pytest -q", 1, "FAILED tests/test_mod.py::test_one_more\n1 failed in 0.1s")]
    assert report.visibility(report.Interval(1, None, 1, frozenset({T1}), frozenset()), near, 1)["named"] is False
    # A test run is known by its output too: django's runner called from a one-liner printed a failing run.
    inline = [cmd("python -c 'from django.test.runner import DiscoverRunner; DiscoverRunner().run_tests([\"t\"])'", 1,
                  "Ran 1 test in 0.01s\n\nFAILED (failures=1)")]
    assert report.visibility(interval, inline, 1)["saw_failure"] is True
    # but a listing of an old log that mentions a test count is not a run
    old_log = [cmd("cat test_output.txt", 0, "Ran 5 tests in 0.1s\n\nFAILED (failures=2)")]
    assert report.visibility(interval, old_log, 1)["saw_failure"] is False
    # Colour codes are not text: a coloured failing pytest run is a failing run, and its error line counts.
    coloured = [cmd("python -m pytest -q", 1, "\x1b[31mFAILED\x1b[0m tests/test_mod.py::\x1b[1mtest_one\x1b[0m - "
                    "AttributeError: 'Row' object has no attribute 'strip'\n\x1b[31m1 failed\x1b[0m, \x1b[32m2 passed\x1b[0m")]
    seen_coloured = report.visibility(report.Interval(1, None, 1, frozenset({T1}), frozenset()), coloured, 1,
                                      ["AttributeError: 'Row' object has no attribute 'strip'"])
    assert (seen_coloured["ran_tests"], seen_coloured["saw_failure"], seen_coloured["named"]) == (True, True, True)
    # The model saw the first and last 5,000 characters of a long output; evidence in the middle was never shown.
    middle = "x" * 6000 + "\n" + name_error + "\n" + "y" * 6000
    long = [cmd("python repro.py", 1, middle, truncated=True)]
    assert report.visibility(interval, long, 1, [name_error])["unseen"] is True
    assert report.visibility(interval, [cmd("python repro.py", 1, middle)], 1, [name_error])["unseen"] is False


def test_the_end_tree_is_read_from_the_duplicates_predecessor() -> None:
    timeline, labels = _states(("run", 0, OK), ("tool", 1, BROKEN_T1), ("final", 2, BROKEN_T1))
    verdict, source, agree = report.final_verdict(timeline, labels)
    assert verdict is BROKEN_T1 and source == "predecessor" and agree is True
    timeline, labels = _states(("run", 0, OK), ("tool", 1, HANG), ("final", 2, BROKEN_T1))
    verdict, source, agree = report.final_verdict(timeline, labels)
    assert verdict is BROKEN_T1 and source == "final" and agree is None


def test_fidelity_tiers() -> None:
    """A replay counts only if it ran every command it was meant to, healthily, and exited as recorded."""
    healthy = {"commands_replayed": 5, "commands_to_replay": 5, "transport_errors": 0}
    exact = {"recorded_submitted": True, "replay_submitted": True, "same_submission": True, "shadow_errors": 0,
             "sync_skipped_paths": [], "rc_compared": 10, "rc_matched": 10}
    assert report.fidelity_tier(exact, healthy) == "exact"
    assert report.fidelity_tier({**exact, "same_submission": False}, healthy) == "diverged"
    assert report.fidelity_tier({**exact, "replay_submitted": False}, healthy) == "diverged"
    assert report.fidelity_tier({**exact, "shadow_errors": 2}, healthy) == "diverged"
    assert report.fidelity_tier({**exact, "sync_skipped_paths": ["big.bin"]}, healthy) == "diverged"
    assert report.fidelity_tier({**exact, "rc_matched": 8}, healthy) == "diverged"
    assert report.fidelity_tier(exact, {**healthy, "exception": "RuntimeError: dead"}) == "diverged"
    assert report.fidelity_tier(exact, {**healthy, "commands_replayed": 3}) == "diverged"  # stopped early
    assert report.fidelity_tier(exact, {**healthy, "transport_errors": 1}) == "diverged"
    stopped = {"recorded_submitted": False, "replay_submitted": False, "rc_compared": 10, "rc_matched": 9}
    assert report.fidelity_tier(stopped, healthy) == "unsubmitted"
    assert report.fidelity_tier({**stopped, "rc_matched": 8}, healthy) == "diverged"
    assert report.fidelity_tier({**stopped, "rc_compared": 0, "rc_matched": 0}, healthy) == "diverged"
    assert report.fidelity_tier({**stopped, "replay_submitted": True}, healthy) == "diverged"  # ended at an early sentinel


def test_a_timeline_out_of_command_order_is_refused() -> None:
    shadow = [{"seq": 0, "sha": "a", "session": "s", "trigger": "run", "step_id": "run"},
              {"seq": 1, "sha": "b", "session": "s", "trigger": "tool", "step_id": "cmd-004"},
              {"seq": 2, "sha": "c", "session": "s", "trigger": "tool", "step_id": "cmd-002"}]
    with pytest.raises(ValueError):
        report.load_timeline(shadow, [], "s", 5)
    with pytest.raises(ValueError):
        report.load_timeline(shadow[:2], [], "s", 3)  # an observation after the last replayed command


def test_spearman_ranks_and_permutation() -> None:
    assert report.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert report.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert report.ranks([3, 1, 3, 2]) == [3.5, 1.0, 3.5, 2.0]
    assert report.spearman([1, 2], [1, 2]) is None
    assert report.spearman([1, 1, 1], [1, 2, 3]) is None
    p = report.permutation_p([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], permutations=2000)
    assert p == report.permutation_p([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], permutations=2000)  # seeded
    assert p < 0.01  # only the identity ordering (and its mirror) reaches |rho| = 1


def test_the_cluster_bootstrap_is_seeded_and_brackets_the_estimate() -> None:
    cells = []
    for t in range(6):
        for m in range(3):
            cells.append({"task": f"t{t}", "recorded_submitted": True, "leaderboard_resolved": True,
                          "any_break": (t + m) % 2 == 0, "test_break": False, "load_break": False, "intervals": 0,
                          "exposure": 0.1 * t, "exposure_after_first_edit": None, "wasted_share": 0.0,
                          "unmeasured_share": 0.0, "shipped_any": False, "stop_broken": None, "first_solved_cmd": None,
                          "overshoot": None})
    first = report.cluster_bootstrap(cells, [], resamples=300)
    assert first == report.cluster_bootstrap(cells, [], resamples=300)
    point = report.summarise(cells, [])
    low, high = first["exposure_mean"]
    assert low <= point["exposure_mean"] <= high
    assert first["silent_interval_share"] == (None, None)  # no intervals: no interval, rather than a fiction


# ------------------------------------------------------------- a whole report


def _instance(task, p2p, f2p):
    return {"instance_id": task, "repo": "pkg/pkg", "base_commit": "abc", "problem_statement": "fix",
            "test_patch": "", "version": "1.0", "FAIL_TO_PASS": json.dumps(f2p), "PASS_TO_PASS": json.dumps(p2p)}


def _cell(root, submission, task, *, states, commands, fidelity, evidence=None, session="s1", status="done"):
    """A cell directory as scripts/public_replay.py leaves it. ``states`` = (trigger, step_id, record or None)."""
    directory = root / "cells" / submission / task
    (directory / "ledger" / "evidence").mkdir(parents=True)
    shadow = [{"seq": i, "sha": f"{task}-{submission}-{i}", "session": session, "step_id": step, "attempt": 1,
               "trigger": trigger, "tool": "bash", "files": [], "cost_billed_usd": 0.0, "cost_work_usd": 0.0, "ts": 0.0}
              for i, (trigger, step, _) in enumerate(states)]
    shadow.append({**shadow[-1], "seq": 99, "session": "another-session", "trigger": "tool", "step_id": "cmd-999"})
    (directory / "shadow.jsonl").write_text("".join(json.dumps(line) + "\n" for line in shadow))
    lines = [{**record, "sha": f"{task}-{submission}-{i}"} for i, (_, _, record) in enumerate(states) if record]
    (directory / "states.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    full = {"recorded_submitted": False, "same_submission": None, "shadow_errors": 0, "replay_exception": None,
            "sync_skipped_paths": [], "rc_compared": commands, "rc_matched": commands, "leaderboard_resolved": False,
            "replay_resolved": None, "commands_replayed": commands, **fidelity}
    full.setdefault("replay_submitted", full["recorded_submitted"])
    (directory / "fidelity.json").write_text(json.dumps(full))
    (directory / "manifest.json").write_text(json.dumps({
        "session": session, "commands_replayed": commands, "commands_to_replay": commands, "transport_errors": 0,
        "shadow_errors": 0, "sync_skipped_paths": [], "exception": None, "scaffold_version_recorded": "2.0.0",
        "recorded_exit_status": "Submitted" if full["recorded_submitted"] else "LimitsExceeded"}))
    (directory / "ledger" / "evidence" / f"{task}__MSWE__t1.json").write_text(json.dumps(evidence or {"resolved": None}))
    (directory / "status.json").write_text(json.dumps({"state": status}))


@pytest.fixture
def study(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "pubreplay"
    root.mkdir()
    (root / "verified.jsonl").write_text("".join(json.dumps(i) + "\n" for i in (
        _instance("pkg__pkg-1", [T1, T2, T3], [F1]),
        _instance("pkg__pkg-2", ["U1", "U2"], ["G1"]),
        _instance("pkg__pkg-3", ["V1"], ["H1"]),
    )))
    (root / "details").mkdir()
    (root / "details" / "subA.json").write_text(json.dumps({
        "pkg__pkg-1": {"resolved": False}, "pkg__pkg-2": {"resolved": True}, "pkg__pkg-9": {"resolved": True}}))
    (root / "details" / "subB.json").write_text(json.dumps({"pkg__pkg-1": {"resolved": False}, "pkg__pkg-2": {"resolved": False}}))

    # A: submitted, replayed exactly. Breaks a test, repairs it (solving the task on the way), then
    # leaves the module unloadable and submits that.
    _cell(root, "subA", "pkg__pkg-1", commands=6, states=[
        ("run", "run", rec("x", failed=[F1])),
        ("tool", "cmd-002", rec("x", failed=[T1, F1])),
        ("tool", "cmd-004", rec("x")),
        ("tool", "cmd-005", rec("x", infra=NO_RESULTS, unreported=[T1, T2, T3, F1])),
        ("final", "final", rec("x", infra=NO_RESULTS, unreported=[T1, T2, T3, F1])),
    ], fidelity={"recorded_submitted": True, "same_submission": True, "leaderboard_resolved": False,
                 "replay_resolved": False},
       evidence={"resolved": False, "grade": {"pass_to_pass": "1/3"}, "grade_failed": [T1, T2, T3]})
    # B: never submitted (limits). T3 fails at its base -> flaky for the task; a hang, an unknown, then a break.
    _cell(root, "subB", "pkg__pkg-1", commands=3, states=[
        ("run", "run", rec("x", failed=[T3, F1])),
        ("tool", "cmd-001", rec("x", timed_out=True, infra="timed out")),
        ("tool", "cmd-002", rec("x", ran=False, infra="patch failed: x")),
        ("tool", "cmd-003", rec("x", failed=[T2, T3, F1])),
        ("final", "final", rec("x", failed=[T2, T3, F1])),
    ], fidelity={})
    # C: diverged (another patch); solved at its only edit; the duplicate disagrees with its predecessor.
    _cell(root, "subA", "pkg__pkg-2", commands=2, states=[
        ("run", "run", rec("x", failed=["G1"])),
        ("tool", "cmd-001", rec("x")),
        ("final", "final", rec("x", failed=["U1"])),
    ], fidelity={"recorded_submitted": True, "same_submission": False})
    # D: unusable, its base state never ran. E: crashed, never analysed.
    _cell(root, "subB", "pkg__pkg-2", commands=1, states=[
        ("run", "run", rec("x", ran=False, infra="patch failed: y")), ("final", "final", None),
    ], fidelity={"rc_compared": 0, "rc_matched": 0})
    _cell(root, "subB", "pkg__pkg-3", commands=1, states=[("run", "run", rec("x"))], fidelity={}, status="crashed")

    logs = {
        ("subA", "pkg__pkg-1"): [
            cmd("ls"), cmd("sed -i 's/a/b/' pkg/mod.py"),
            cmd("python -m pytest tests/test_mod.py", 1,
                "FAILED tests/test_mod.py::test_one - AssertionError\n1 failed, 2 passed in 0.1s"),
            cmd("sed -i 's/b/a/' pkg/mod.py"), cmd("printf 'x = (' > pkg/mod.py"),
            cmd("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff"),
        ],
        ("subB", "pkg__pkg-1"): [
            cmd("python setup.py build_ext -i", None, "timed out"), cmd("sed -i 's/a/b/' pkg/mod.py"),
            cmd("python -m pytest tests/test_mod.py -k two", 1, "FAILED tests/test_mod.py::test_two[param]\n1 failed in 0.1s"),
        ],
    }
    monkeypatch.setattr(report, "load_recorded_commands", lambda root, sub, task, version="": logs[(sub, task)])
    return root


def _csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_a_whole_report_over_synthetic_cells(study: Path) -> None:
    summary = report.run(study, study / "report")
    cells = {(r["submission"], r["task"]): r for r in _csv(study / "report" / "cells.csv")}
    assert len(cells) == 4  # the crashed cell is counted, never analysed

    a = cells[("subA", "pkg__pkg-1")]
    assert (a["tier"], a["usable"], a["main"]) == ("exact", "True", "True")
    assert a["base_size"] == "2" and a["flaky_excluded"] == "1"  # T3 is flaky at the base across A and B
    assert (a["any_break"], a["test_break"], a["load_break"], a["intervals"]) == ("True", "True", "True", "2")
    assert float(a["exposure"]) == pytest.approx(4 / 6) and float(a["wasted_share"]) == pytest.approx(0.5)
    assert float(a["exposure_after_first_edit"]) == pytest.approx(0.8)
    assert (a["shipped_any"], a["shipped_load"], a["shipped_test"]) == ("True", "True", "False")
    assert a["grade_agreement"] == "same_breaks" and a["final_agree"] == "True" and a["silent_intervals"] == "1"
    assert (a["first_solved_cmd"], a["solved_end"], a["overshoot"], a["f2p_passing_end"]) == ("4", "False", "True", "0")

    b = cells[("subB", "pkg__pkg-1")]
    assert (b["tier"], b["main"], b["recorded_submitted"]) == ("unsubmitted", "True", "False")
    assert (b["states_hang"], b["states_unknown"], b["states_broken"]) == ("1", "1", "1")
    assert float(b["exposure"]) == pytest.approx(1 / 3) and float(b["unmeasured_share"]) == pytest.approx(2 / 3)
    assert float(b["wasted_share"]) == 0 and b["stop_broken"] == "True" and b["shipped_any"] == ""
    assert b["first_solved_cmd"] == "" and b["overshoot"] == "False"

    c = cells[("subA", "pkg__pkg-2")]
    # no official grade on record for it, so nothing is said about what it shipped; its end tree was clean
    assert (c["tier"], c["main"], c["final_agree"], c["shipped_any"], c["end_broken"]) == ("diverged", "False", "False", "", "False")
    assert (c["first_solved_cmd"], c["solved_end"], c["overshoot"]) == ("1", "True", "False")

    d = cells[("subB", "pkg__pkg-2")]
    assert (d["usable"], d["problem"], d["tier"]) == ("False", "base state not clean", "diverged")

    intervals = _csv(study / "report" / "intervals.csv")
    shape = [(i["submission"], i["onset"], i["repair"], i["kind"], i["ended"], i["ran_tests"], i["saw_failure"],
              i["named"], i["silent"], i["unseen"]) for i in intervals]
    assert shape == [
        ("subA", "2", "4", "test", "clean", "True", "True", "True", "False", "False"),
        ("subA", "5", "", "load", "", "False", "False", "False", "True", "True"),
        ("subB", "3", "", "test", "", "True", "True", "True", "False", "False"),
    ]

    states = _csv(study / "report" / "states.csv")
    assert [(s["command"], s["status"], s["f2p_passing"]) for s in states if s["submission"] == "subA"
            and s["task"] == "pkg__pkg-1"] == [("0", "ok", "0"), ("2", "broken", "0"), ("4", "ok", "1"),
                                                ("5", "broken", "0"), ("6", "broken", "0")]

    models = {m["submission"]: m for m in _csv(study / "report" / "models.csv")}
    assert (models["subA"]["cells"], models["subA"]["exact"], models["subA"]["diverged"], models["subA"]["runs"]) == ("2", "1", "1", "1")
    assert (models["subB"]["cells"], models["subB"]["not_done"], models["subB"]["usable"], models["subB"]["runs"]) == ("3", "1", "1", "1")
    assert float(models["subA"]["resolve_rate_leaderboard"]) == pytest.approx(2 / 3)
    assert float(models["subA"]["silent_interval_share"]) == pytest.approx(0.5)
    assert float(models["subB"]["stop_broken_share"]) == pytest.approx(1.0)
    everything = {m["submission"]: m for m in _csv(study / "report" / "models_all_usable.csv")}
    assert everything["subA"]["runs"] == "2"  # the diverged cell joins the sensitivity rows

    main = summary["overall_main"]
    assert main["runs"] == 2 and main["intervals"] == 3
    assert (main["intervals_test"], main["intervals_load"]) == (2, 1)
    assert main["repaired_share"] == pytest.approx(1 / 3)  # A's test break ended clean; the other two stayed open
    assert main["unseen_no_tests_share"] == 1.0 and main["unseen_repaired_share"] == 0.0  # A's load break: no tests, open
    assert main["unseen_test_share"] == 0.0 and main["unseen_load_share"] == 1.0
    assert main["unseen_open_breaks"] == 1 and main["unseen_open_test_breaks"] == 0
    assert main["end_tree_broken_share"] == 1.0  # A shipped a broken end tree, B stopped on one
    assert main["limit_stop_runs"] == 1 and main["limit_stop_broken_share"] == 1.0
    assert main["silent_interval_share"] == pytest.approx(1 / 3)
    assert main["exposure_mean"] == pytest.approx((4 / 6 + 1 / 3) / 2)
    assert main["shipped_share"] == 1.0 and main["stop_broken_share"] == 1.0
    assert main["overshoot_share"] == pytest.approx(0.5) and main["ever_solved_share"] == pytest.approx(0.5)
    checks = summary["checks"]
    assert checks["flaky_excluded_total"] == 1 and checks["final_duplicate"] == {"compared": 3, "disagreed": 1, "rate": 1 / 3}
    assert checks["grade_agreement_main_submitted"] == {"same_breaks": 1}
    assert checks["resolved_agreement_exact"] == {"leaderboard=False,replay=False": 1}
    assert checks["exit_status_agreement"] == {"matched": 6 + 3 + 2, "compared": 6 + 3 + 2}  # usable runs A, B, C
    assert summary["cells"]["by_state"] == {"done": 4, "crashed": 1}
    assert summary["cross_model"]["spearman"] is None  # two models are not a correlation
    assert (study / "report" / "summary.md").read_text().startswith("# Public-run replay: summary")


def test_the_guard_returns_to_the_last_clean_tree_before_a_broken_end() -> None:
    """What a guard that returns to the last state with no break would have shipped instead."""
    timeline, labels = _states(("run", 0, OK), ("tool", 2, OK), ("tool", 5, BROKEN_T1), ("tool", 6, HANG),
                               ("tool", 8, BROKEN_LOAD), ("final", 9, BROKEN_LOAD))
    solved_at = [False, True, False, None, False, False]
    end = report._verdict_index(timeline, labels, report.final_verdict(timeline, labels)[1])
    guard = report.guard_counterfactual(timeline, labels, solved_at, end)
    assert guard == {"guard_keeps_work": True, "guard_solved": True, "guard_discards_cmds": 6}
    # No clean state after the base: the guard ships the base, with none of the run's work.
    timeline, labels = _states(("run", 0, OK), ("tool", 1, BROKEN_T1), ("final", 3, BROKEN_T1))
    guard = report.guard_counterfactual(timeline, labels, [False, False, False], 1)
    assert guard == {"guard_keeps_work": False, "guard_solved": False, "guard_discards_cmds": 1}
    # A clean end needs no guard.
    timeline, labels = _states(("run", 0, OK), ("tool", 1, OK), ("final", 2, OK))
    assert report.guard_counterfactual(timeline, labels, [False, True, True], 1)["guard_keeps_work"] is None


def test_a_break_that_changes_kind_is_two_phases_and_says_how_each_ended() -> None:
    """A tree that cannot load, then loads but fails a test: two breaks with their own evidence."""
    timeline, labels = _states(("run", 0, OK), ("tool", 2, BROKEN_LOAD), ("tool", 4, BROKEN_T1), ("tool", 6, OK),
                               ("tool", 8, BROKEN_T1), ("final", 9, BROKEN_T1))
    intervals = report.break_intervals(timeline, labels, 9)
    assert [(i.onset, i.repair, i.kind, i.ended, i.states) for i in intervals] == [
        (2, 4, "load", "changed", (1,)), (4, 6, "test", "clean", (2,)), (8, None, "test", None, (4,))]


def test_the_guard_counts_discarded_work_to_the_runs_last_command() -> None:
    timeline, labels = _states(("run", 0, OK), ("tool", 2, OK), ("tool", 5, BROKEN_T1), ("final", 9, BROKEN_T1))
    guard = report.guard_counterfactual(timeline, labels, [False, True, False, False], 2, commands=9)
    assert guard == {"guard_keeps_work": True, "guard_solved": True, "guard_discards_cmds": 7}


def test_planted_runs_are_checked_against_their_answers_and_never_counted(study: Path) -> None:
    """A known-answer run is calibration, not data: it must not join any model's numbers."""
    planted = study / "planted"
    planted.mkdir()
    (planted / "pkg__pkg-1.json").write_text(json.dumps({"expected": {"intervals": [
        {"onset_cmd": 1, "repair_cmd": 2, "kind": "load"}]}}))
    _cell(study, "planted", "pkg__pkg-1", commands=3, states=[
        ("run", "run", rec("x", failed=[F1])),
        ("tool", "cmd-001", rec("x", infra=NO_RESULTS, unreported=[T1, T2, T3, F1])),
        ("tool", "cmd-002", rec("x", failed=[F1])),
        ("final", "final", rec("x", failed=[F1])),
    ], fidelity={"recorded_submitted": True, "same_submission": True})
    summary = report.run(study, study / "report")
    assert summary["checks"]["planted"] == {"pkg__pkg-1": {"expected": [(1, 2, "load")], "found": [(1, 2, "load")],
                                                           "pass": True}}
    assert "planted" not in {m["submission"] for m in _csv(study / "report" / "models.csv")}
    assert summary["overall_main"]["runs"] == 2  # unchanged by the planted run


def test_the_submitting_commands_output_is_not_part_of_the_log(tmp_path: Path, monkeypatch) -> None:
    """Its recorded output is the submitted patch, which the model never saw: the sentinel ends the run."""
    from taste.benchmarks import public_trajectories

    recorded = (
        public_trajectories.RecordedCommand(index=0, command="sed -i 's/a/b/' x.py", returncode=0, output="",
                                            output_truncated=False, timed_out=False),
        public_trajectories.RecordedCommand(index=1, command="echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff",
                                            returncode=None, output="FAILED tests/test_mod.py::test_one",
                                            output_truncated=False, timed_out=False),
    )
    fake = SimpleNamespace(commands=recorded, submit_index=1)
    monkeypatch.setattr(public_trajectories, "parse_trajectory", lambda raw, mini_version="": fake)
    (tmp_path / "trajs" / "s").mkdir(parents=True)
    (tmp_path / "trajs" / "s" / "t.traj.json").write_text("{}")
    loaded = report.load_recorded_commands(tmp_path, "s", "t")
    assert [c.output for c in loaded] == ["", ""] and loaded[1].command.startswith("echo COMPLETE")


def test_a_coloured_state_cut_in_storage_may_hide_error_lines() -> None:
    # Evidence is read from the whole output when a state is tested; only coloured outputs, whose error
    # lines were missed before colour was stripped, had theirs recovered later from the kept head and tail.
    coloured = "\x1b[31m" + "h" * 2000
    assert report.evidence_cut({"output_head": coloured, "output_tail": "t" * 6000}) is True
    assert report.evidence_cut({"exceptions": ["NameError: x"], "output_head": coloured, "output_tail": "t" * 6000}) is True
    assert report.evidence_cut({"output_head": "h" * 2000, "output_tail": "t" * 6000}) is False
    assert report.evidence_cut({"output_head": "\x1b[31mshort", "output_tail": "short"}) is False
    assert report.evidence_cut(None) is False
    # Read again from the whole output: no longer in doubt.
    assert report.evidence_cut({"evidence_retested": True, "output_head": coloured, "output_tail": "t" * 6000}) is False
    assert report.evidence_cut({"evidence_retested": "verdicts differ", "output_head": coloured, "output_tail": "t" * 6000})


def test_unseen_bounds_turn_every_break_in_doubt_the_other_way() -> None:
    def brk(unseen, saw_failure, named, own, base):
        return {"unseen": unseen, "saw_failure": saw_failure, "named": named,
                "evidence_complete": own, "base_evidence_complete": base}

    breaks = [
        brk(True, False, False, True, True),  # unseen on complete evidence: unseen in both bounds
        brk(True, False, False, False, True),  # its own error lines may be cut: may be named, seen at the floor
        brk(False, False, True, True, False),  # named against a cut base: may be a base error, unseen at the ceiling
        brk(False, True, True, True, False),  # a failing test run: seen whatever the evidence
        brk(None, None, None, True, True),  # no recorded outputs: not judged
    ]
    assert report.unseen_bounds(breaks) == {"floor": 0.25, "ceiling": 0.75}


def test_break_flags_name_what_a_compile_check_catches_and_what_the_grade_ignores() -> None:
    test_break = report.Interval(onset=1, repair=3, duration=2, tests=frozenset({"t::a"}), loads=frozenset(),
                                 ended="clean", states=(1, 2))
    skipped = [{"failed": [], "skipped": ["t::a"]}, {"failed": [], "skipped": ["t::a"]}]
    assert report.break_flags(test_break, skipped, [], []) == {"syntax_error": False, "skip_only": True}
    failed_once = [{"failed": ["t::a"], "skipped": []}, {"failed": [], "skipped": ["t::a"]}]
    assert report.break_flags(test_break, failed_once, [], [])["skip_only"] is False
    load_break = report.Interval(onset=1, repair=None, duration=1, tests=frozenset(), loads=frozenset({"t::b"}),
                                 states=(1,))
    assert report.break_flags(load_break, [{}], ["IndentationError: unexpected indent"], []) == {
        "syntax_error": True, "skip_only": False}
    assert report.break_flags(load_break, [{}], [], ["pkg/x.py:3"])["syntax_error"] is True  # a syntax error's location
    assert report.break_flags(load_break, [{}], ["ImportError: cannot import name 'f'"], [])["syntax_error"] is False
