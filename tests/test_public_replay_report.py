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


def cmd(command, rc=0, output=""):
    return SimpleNamespace(command=command, returncode=rc, output=output, timed_out=rc is None)


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
    ("bin/test sympy/core/tests/test_basic.py", True),
    ("tox -e py39", True),
    ("python -m unittest discover", True),
    ("cat ./tests/test_mod.py", False),
    ("grep -rn foo src/ && ls", False),
])
def test_test_runner_invocations(command, expected) -> None:
    assert report.is_test_command(command) is expected


def test_failure_signals() -> None:
    assert report.shows_failure(1, "") is True
    assert report.shows_failure(0, "FAILED tests/x.py::t") is True
    assert report.shows_failure(0, "Traceback (most recent call last)") is True
    assert report.shows_failure(0, "5 passed in 0.10s") is False
    assert report.shows_failure(None, "Command timed out") is False  # a timeout counts only by its markers


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


def test_visibility_reads_the_original_log() -> None:
    recorded = [
        cmd("ls"), cmd("sed -i 's/a/b/' pkg/mod.py"),
        cmd("python -m pytest tests/test_mod.py", 1, "FAILED tests/test_mod.py::test_one - AssertionError"),
        cmd("sed -i 's/b/a/' pkg/mod.py"), cmd("printf 'x = (' > pkg/mod.py"), cmd("git diff"),
    ]
    seen = report.visibility(report.Interval(2, 4, 2, frozenset({T1}), frozenset()), recorded, 6)
    assert seen == {"window": 3, "ran_tests": True, "saw_failure": True, "named_test": True, "named_load": False,
                    "named": True, "silent": False, "unseen": False}
    quiet = report.visibility(report.Interval(5, None, 2, frozenset(), frozenset({T1})), recorded, 6)
    assert quiet["window"] == 2 and quiet["ran_tests"] is False and quiet["silent"] is True and quiet["named"] is False
    assert quiet["unseen"] is True
    # A reproduction script's traceback is not a test run, but it shows a load break: not unseen.
    repro = [cmd("printf 'x = (' > pkg/mod.py"), cmd("python repro.py", 1, "SyntaxError: invalid syntax")]
    shown = report.visibility(report.Interval(1, None, 2, frozenset(), frozenset({T1})), repro, 2)
    assert shown["silent"] is True and shown["named"] is True and shown["unseen"] is False
    # a name must match as a word: test_one is not named by test_one_more
    near = [cmd("pytest -q", 0, "test_one_more PASSED")]
    assert report.visibility(report.Interval(1, None, 1, frozenset({T1}), frozenset()), near, 1)["named_test"] is False


def test_the_end_tree_is_read_from_the_duplicates_predecessor() -> None:
    timeline, labels = _states(("run", 0, OK), ("tool", 1, BROKEN_T1), ("final", 2, BROKEN_T1))
    verdict, source, agree = report.final_verdict(timeline, labels)
    assert verdict is BROKEN_T1 and source == "predecessor" and agree is True
    timeline, labels = _states(("run", 0, OK), ("tool", 1, HANG), ("final", 2, BROKEN_T1))
    verdict, source, agree = report.final_verdict(timeline, labels)
    assert verdict is BROKEN_T1 and source == "final" and agree is None


def test_fidelity_tiers() -> None:
    exact = {"recorded_submitted": True, "same_submission": True, "shadow_errors": 0, "sync_skipped_paths": []}
    assert report.fidelity_tier(exact, {}) == "exact"
    assert report.fidelity_tier({**exact, "same_submission": False}, {}) == "diverged"
    assert report.fidelity_tier({**exact, "shadow_errors": 2}, {}) == "diverged"
    assert report.fidelity_tier({**exact, "sync_skipped_paths": ["big.bin"]}, {}) == "diverged"
    assert report.fidelity_tier(exact, {"exception": "RuntimeError: dead"}) == "diverged"
    stopped = {"recorded_submitted": False, "rc_compared": 10, "rc_matched": 9}
    assert report.fidelity_tier(stopped, {}) == "unsubmitted"
    assert report.fidelity_tier({**stopped, "rc_matched": 8}, {}) == "diverged"
    assert report.fidelity_tier({**stopped, "rc_compared": 0, "rc_matched": 0}, {}) == "diverged"


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
    (directory / "fidelity.json").write_text(json.dumps(full))
    (directory / "manifest.json").write_text(json.dumps({
        "session": session, "commands_replayed": commands, "shadow_errors": 0, "sync_skipped_paths": [],
        "exception": None, "scaffold_version_recorded": "2.0.0"}))
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
            cmd("python -m pytest tests/test_mod.py", 1, "FAILED tests/test_mod.py::test_one - AssertionError"),
            cmd("sed -i 's/b/a/' pkg/mod.py"), cmd("printf 'x = (' > pkg/mod.py"),
            cmd("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff"),
        ],
        ("subB", "pkg__pkg-1"): [
            cmd("python setup.py build_ext -i", None, "timed out"), cmd("sed -i 's/a/b/' pkg/mod.py"),
            cmd("python -m pytest tests/test_mod.py -k two", 1, "FAILED tests/test_mod.py::test_two[param]"),
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
    assert (c["tier"], c["main"], c["final_agree"], c["shipped_any"]) == ("diverged", "False", "False", "False")
    assert (c["first_solved_cmd"], c["solved_end"], c["overshoot"]) == ("1", "True", "False")

    d = cells[("subB", "pkg__pkg-2")]
    assert (d["usable"], d["problem"], d["tier"]) == ("False", "base state not clean", "diverged")

    intervals = _csv(study / "report" / "intervals.csv")
    shape = [(i["submission"], i["onset"], i["repair"], i["kind"], i["ran_tests"], i["saw_failure"], i["named"], i["silent"])
             for i in intervals]
    assert shape == [
        ("subA", "2", "4", "test", "True", "True", "True", "False"),
        ("subA", "5", "", "load", "False", "False", "False", "True"),
        ("subB", "3", "", "test", "True", "True", "True", "False"),
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
    assert main["silent_interval_share"] == pytest.approx(1 / 3)
    assert main["exposure_mean"] == pytest.approx((4 / 6 + 1 / 3) / 2)
    assert main["shipped_share"] == 1.0 and main["stop_broken_share"] == 1.0
    assert main["overshoot_share"] == pytest.approx(0.5) and main["ever_solved_share"] == pytest.approx(0.5)
    checks = summary["checks"]
    assert checks["flaky_excluded_total"] == 1 and checks["final_duplicate"] == {"compared": 3, "disagreed": 1, "rate": 1 / 3}
    assert checks["grade_agreement_main_submitted"] == {"same_breaks": 1}
    assert checks["resolved_agreement_exact"] == {"leaderboard=False,replay=False": 1}
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
