"""SWE-bench-Live adapter: their grading rulebook, mirrored and pinned.

Live's semantics differ from upstream SWE-bench in two ways that would each
silently reshape our numbers if mixed up: XFAIL counts as FAIL (upstream:
pass), and a graded test missing from the log blocks nothing (upstream:
counts as failed). These tests exist so neither rulebook can bleed into the
other without a red bar naming the difference.
"""

from __future__ import annotations

import json
from pathlib import Path

from taste.benchmarks.swebenchlive import (
    END_MARKER,
    START_MARKER,
    LiveInstance,
    build_live_probe_script,
    grade_live_in_sandbox,
    live_parity_check,
    load_live_dataset,
    parse_live_output,
    probe_files,
)
from taste.execution import ExecResult, ScriptedSandbox


def _instance(**overrides) -> LiveInstance:
    fields = dict(
        instance_id="acme__widget-7", repo="acme/widget", base_commit="c" * 40,
        problem_statement="fix it", test_patch="diff --git a/tests/test_w.py b/tests/test_w.py",
        fail_to_pass=("tests/test_w.py::test_new",),
        pass_to_pass=("tests/test_w.py::test_old", "tests/test_x.py::test_other"),
        test_cmds=("pytest -rA",), log_parser="pytest",
    )
    fields.update(overrides)
    return LiveInstance(**fields)


def _eval_log(*lines: str) -> str:
    return "\n".join([START_MARKER, *lines, END_MARKER])


# ------------------------------------------------------------------ loading


def test_the_loader_drops_the_gold_patch(tmp_path: Path) -> None:
    """Same containment as Verified: nothing downstream can leak to an agent
    a field the instance object never carried."""
    row = {
        "instance_id": "a__b-1", "repo": "a/b", "base_commit": "x",
        "problem_statement": "p", "test_patch": "t", "patch": "THE GOLD",
        "FAIL_TO_PASS": ["t::f"], "PASS_TO_PASS": ["t::p"],
        "test_cmds": ["pytest -rA"], "log_parser": "pytest",
    }
    path = tmp_path / "live.jsonl"
    path.write_text(json.dumps(row) + "\n")
    inst = load_live_dataset(path)[0]
    assert not hasattr(inst, "patch")
    assert "THE GOLD" not in repr(inst)


def test_published_image_uses_their_naming() -> None:
    inst = _instance(instance_id="aws-cloudformation__cfn-lint-3798")
    assert inst.published_image == (
        "starryzhang/sweb.eval.x86_64.aws-cloudformation_1776_cfn-lint-3798"
    )


# ------------------------------------------------------------------ probing


def test_probe_files_come_from_member_prefixes() -> None:
    assert probe_files(_instance()) == ("tests/test_w.py", "tests/test_x.py")


def test_probe_refuses_to_scope_when_any_member_is_not_a_path() -> None:
    """Half-scoped would silently shrink the oracle; all-or-nothing is loud."""
    inst = _instance(pass_to_pass=("tests/test_w.py::test_old", "not a path"))
    assert probe_files(inst) == ()
    assert "pytest -rA" in build_live_probe_script(inst)


def test_the_scoped_probe_restores_graded_files_around_the_run() -> None:
    script = build_live_probe_script(_instance())
    assert script.count("git checkout -q taste-baseline --") == 2
    assert "python -m pytest -rA tests/test_w.py tests/test_x.py" in script


def test_parse_reads_only_the_bracketed_slice() -> None:
    log = "PASSED tests/test_w.py::test_setup_noise\n" + _eval_log(
        "PASSED tests/test_w.py::test_old"
    )
    statuses = parse_live_output(_instance(), log)
    assert statuses == {"tests/test_w.py::test_old": "PASSED"}


# ------------------------------------------------------------------ grading


def _grade(log_lines, *, patch="diff --git a/w.py b/w.py", instance=None):
    sandbox = ScriptedSandbox().on(
        START_MARKER, ExecResult(0, _eval_log(*log_lines), "")
    )
    report = grade_live_in_sandbox(sandbox, instance or _instance(), patch)
    return report


def test_resolved_when_f2p_passes_and_nothing_fails() -> None:
    report = _grade([
        "PASSED tests/test_w.py::test_new",
        "PASSED tests/test_w.py::test_old",
        "PASSED tests/test_x.py::test_other",
    ])
    assert report is not None and report.resolved is True


def test_a_missing_p2p_test_does_not_block_resolution() -> None:
    """THEIR leniency, mirrored: upstream SWE-bench counts a missing graded
    test as failed; Live's rule only counts recorded failures. Flipping this
    would deflate our Live resolve rates against their own leaderboard."""
    report = _grade([
        "PASSED tests/test_w.py::test_new",
        "PASSED tests/test_w.py::test_old",
        # test_other never appears
    ])
    assert report is not None and report.resolved is True


def test_xfail_counts_as_fail_on_live() -> None:
    """Inverted from upstream, because their normaliser is 'pass' in status:
    'xfail' contains no 'pass'. Mirrored, not endorsed."""
    report = _grade([
        "PASSED tests/test_w.py::test_new",
        "XFAIL tests/test_w.py::test_old",
        "PASSED tests/test_x.py::test_other",
    ])
    assert report is not None and report.resolved is False


def test_an_unrunnable_grade_is_none_not_unresolved() -> None:
    sandbox = ScriptedSandbox().on(START_MARKER, ExecResult(127, "sh: boom", ""))
    assert grade_live_in_sandbox(sandbox, _instance(), "diff") is None


def test_a_test_patch_that_will_not_apply_is_infrastructure() -> None:
    sandbox = (
        ScriptedSandbox()
        .on("taste_test.diff", ExecResult(1, "", "corrupt"))
        .on(START_MARKER, ExecResult(0, _eval_log("PASSED t::x"), ""))
    )
    assert grade_live_in_sandbox(sandbox, _instance(), "diff") is None


def test_a_model_patch_that_will_not_apply_is_unresolved() -> None:
    sandbox = (
        ScriptedSandbox()
        .on("taste_pred.diff", ExecResult(1, "", "corrupt"))
        .on(START_MARKER, ExecResult(0, _eval_log("PASSED t::x"), ""))
    )
    report = grade_live_in_sandbox(sandbox, _instance(), "diff --git a/x b/x")
    assert report is not None and report.resolved is False


# ------------------------------------------------------------------ parity


def test_parity_passes_on_a_working_environment() -> None:
    sandbox = (
        ScriptedSandbox()
        .on("import sys", ExecResult(0, "3.8.20\n", ""))
        .on("--collect-only", ExecResult(0, "2 tests collected\n", ""))
    )
    assert live_parity_check(sandbox, _instance()) is None


def test_parity_refuses_when_collection_dies() -> None:
    """Collection imports the package under test; a bare uninstalled tree —
    bug 20's signature — dies exactly here, at $0."""
    sandbox = (
        ScriptedSandbox()
        .on("import sys", ExecResult(0, "3.8.20\n", ""))
        .on("--collect-only", ExecResult(2, "ImportError: no module named widget", ""))
    )
    reason = live_parity_check(sandbox, _instance())
    assert reason is not None and "collect" in reason
