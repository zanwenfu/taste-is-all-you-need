"""The SWE-bench adapter: an unmodified benchmark, observed more densely.

The properties under test are the ones the paper's defence rests on — that we
never alter the benchmark, that instances which cannot host the construct are
excluded before any arm runs, and that the Monitor's scope cannot overlap the
thing that scores it.

Hermetic: a synthetic dataset in the published schema, no Docker, no network.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from taste.benchmarks.swebench import (
    END_MARKER,
    START_MARKER,
    GradeReport,
    SWEInstance,
    build_eval_script,
    eligible,
    graded_test_files,
    load_dataset,
    monitor_scope,
    parse_eval_output,
    parse_report,
    pass_to_pass_suite,
    patch_for,
    stratified_sample,
    task_text,
)
from taste.execution import ExecResult

TEST_PATCH = """diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1,2 +1,3 @@
 def test_old(): pass
+def test_new(): pass
"""


def _row(
    instance_id: str,
    repo: str,
    *,
    p2p: list[str],
    f2p: list[str] | None = None,
    version: str = "4.1",
) -> dict:
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": "abc123",
        "problem_statement": "Something is broken.",
        "test_patch": TEST_PATCH,
        "version": version,
        "FAIL_TO_PASS": json.dumps(f2p or ["tests/test_core.py::test_new"]),
        "PASS_TO_PASS": json.dumps(p2p),
        "image": "swebench/x@sha256:deadbeef",
    }


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    rows = [
        _row("django__django-1", "django/django", p2p=["t::a", "t::b"]),
        _row("django__django-2", "django/django", p2p=["t::c"]),
        _row("django__django-3", "django/django", p2p=["t::d"]),
        _row("sympy__sympy-1", "sympy/sympy", p2p=["t::e"]),
        _row("astropy__astropy-1", "astropy/astropy", p2p=[]),  # no oracle
    ]
    path = tmp_path / "verified.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


# ------------------------------------------------------------------ loading


def test_loads_the_published_schema(dataset: Path) -> None:
    instances = load_dataset(dataset)
    assert len(instances) == 5
    first = instances[0]
    assert first.instance_id == "django__django-1"
    assert first.pass_to_pass == ("t::a", "t::b")
    assert first.repo_short == "django"


def test_test_lists_parse_whether_json_strings_or_lists(tmp_path: Path) -> None:
    row = _row("x__y-1", "x/y", p2p=["t::a"])
    row["PASS_TO_PASS"] = ["t::a", "t::b"]  # some snapshots ship real lists
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps(row))
    assert load_dataset(path)[0].pass_to_pass == ("t::a", "t::b")


def test_a_missing_field_fails_loudly_with_its_name(tmp_path: Path) -> None:
    """A schema change must not surface as a KeyError mid-sweep."""
    row = _row("x__y-1", "x/y", p2p=["t::a"])
    del row["PASS_TO_PASS"]
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps(row))

    with pytest.raises(ValueError, match="PASS_TO_PASS"):
        load_dataset(path)


def test_version_is_required_because_the_runner_depends_on_it(tmp_path: Path) -> None:
    """The test command is a function of (repo, version). A snapshot without
    it cannot be graded, and must fail at load rather than run the wrong
    command halfway through a sweep."""
    row = _row("x__y-1", "x/y", p2p=["t::a"])
    del row["version"]
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps(row))

    with pytest.raises(ValueError, match="version"):
        load_dataset(path)


def test_version_survives_loading(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps(_row("x__y-1", "django/django", p2p=["t::a"], version="3.2")))
    assert load_dataset(path)[0].version == "3.2"


# ------------------------------------------------------------------ exclusion


def test_instances_without_an_oracle_are_excluded_before_any_arm_runs(
    dataset: Path,
) -> None:
    """No PASS_TO_PASS means no previously-passing state, so the construct
    'something that used to work broke' is not expressible on that instance."""
    selection = eligible(load_dataset(dataset))

    assert [i.instance_id for i in selection.instances] == [
        "django__django-1", "django__django-2", "django__django-3", "sympy__sympy-1",
    ]
    assert len(selection.excluded) == 1
    assert "E1" in selection.excluded[0].reason


def test_an_empty_monitor_scope_is_excluded(dataset: Path) -> None:
    """With no test files outside the graded set, the harness's only
    verification signal would be the oracle itself."""
    instances = load_dataset(dataset)
    selection = eligible(instances, monitor_scopes={"django__django-2": 0})
    ids = [i.instance_id for i in selection.instances]

    assert "django__django-2" not in ids
    assert any("E3" in e.reason for e in selection.excluded)


def test_exclusions_are_recorded_with_reasons(dataset: Path) -> None:
    manifest = eligible(load_dataset(dataset)).manifest()
    assert manifest["n_excluded"] == 1
    assert manifest["excluded"][0]["instance_id"] == "astropy__astropy-1"
    assert "reason" in manifest["excluded"][0]


# ------------------------------------------------------------------ selection


def test_sampling_is_deterministic_for_a_seed(dataset: Path) -> None:
    frame = eligible(load_dataset(dataset))
    a = stratified_sample(frame, n=3, seed=11)
    b = stratified_sample(frame, n=3, seed=11)
    assert [i.instance_id for i in a.instances] == [i.instance_id for i in b.instances]


def test_sampling_keeps_the_dominant_repository_dominant(dataset: Path) -> None:
    """Proportional, not balanced: forcing balance produces a sample that is
    not the benchmark. The cost is that leave-one-repo-out is mandatory."""
    frame = eligible(load_dataset(dataset))
    sample = stratified_sample(frame, n=4, seed=3)
    assert sample.strata.get("django", 0) >= sample.strata.get("sympy", 0)


def test_selection_publishes_its_seed_and_strata(dataset: Path) -> None:
    manifest = stratified_sample(eligible(load_dataset(dataset)), n=3, seed=42).manifest()
    assert manifest["seed"] == 42
    assert manifest["strata"]
    assert len(manifest["selected"]) == manifest["n_selected"]


def test_a_development_holdout_is_never_sampled(dataset: Path) -> None:
    """The development slice is excluded from every reported number, forever."""
    frame = eligible(load_dataset(dataset))
    sample = stratified_sample(frame, n=4, seed=1, holdout={"django__django-1"})
    assert "django__django-1" not in [i.instance_id for i in sample.instances]


# ------------------------------------------------------------------ scoping


def test_graded_files_are_read_from_the_test_patch(dataset: Path) -> None:
    instance = load_dataset(dataset)[0]
    assert graded_test_files(instance) == {"tests/test_core.py"}


def test_monitor_scope_excludes_the_graded_files(tmp_path: Path) -> None:
    """Zero file-level overlap by construction: the Monitor cannot run the
    thing that scores it."""
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    for name in ("test_core.py", "test_other.py", "test_extra.py"):
        (repo / "tests" / name).write_text("def test_x(): pass\n")

    scope = monitor_scope(repo, graded_files={"tests/test_core.py"})

    assert "tests/test_core.py" not in scope
    assert set(scope) == {"tests/test_other.py", "tests/test_extra.py"}


def test_monitor_scope_can_be_empty_and_that_is_detectable(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_core.py").write_text("def test_x(): pass\n")
    assert monitor_scope(repo, graded_files={"tests/test_core.py"}) == []


# ------------------------------------------------------------------ probes


def test_the_probe_applies_the_gold_test_patch_first(dataset: Path) -> None:
    """The graded tests must be the benchmark's own, not any version the
    agent may have edited."""
    suite = pass_to_pass_suite(load_dataset(dataset)[0])
    assert "TASTE_TEST_PATCH" in suite.command
    assert "tests/test_core.py" in suite.command


def test_the_graded_tests_are_restored_before_the_patch_is_applied(
    dataset: Path,
) -> None:
    """The agent may have edited the test files. Applying the gold patch onto
    its edits conflicts, and the previous command's `|| true` swallowed that
    and then graded the agent's own tests -- in exactly the scenario the probe
    exists to detect."""
    script = build_eval_script(load_dataset(dataset)[0])
    checkout = script.index("git checkout")
    apply_at = script.index("git apply")
    assert checkout < apply_at, "tests must be restored before the patch"
    assert "|| true" not in script, "a failed apply must not be swallowed"


def test_the_environment_is_activated(dataset: Path) -> None:
    """The image puts conda base on PATH and activates testbed only from
    .bashrc, which a non-interactive shell never reads."""
    script = build_eval_script(load_dataset(dataset)[0])
    assert "conda activate testbed" in script


def test_output_is_bracketed_so_setup_noise_cannot_score(dataset: Path) -> None:
    script = build_eval_script(load_dataset(dataset)[0])
    assert script.index(START_MARKER) < script.index(END_MARKER)


def test_the_suite_grades_exactly_the_pass_to_pass_members(dataset: Path) -> None:
    suite = pass_to_pass_suite(load_dataset(dataset)[0])
    assert suite.members == ("t::a", "t::b")


def test_the_command_does_not_splat_test_ids_into_argv(dataset: Path) -> None:
    """One instance names 1,432 PASS_TO_PASS tests and another 2,476. As argv
    that approaches ARG_MAX, so the harness runs whole files and filters
    afterwards -- which is also what upstream does."""
    suite = pass_to_pass_suite(load_dataset(dataset)[0])
    for member in suite.members:
        assert member not in suite.command


def test_setup_output_outside_the_markers_is_never_parsed(dataset: Path) -> None:
    """A traceback line or a conda banner can match a runner's grammar."""
    instance = load_dataset(dataset)[0]
    noisy = (
        "PASSED tests/decoy.py::test_should_not_count\n"
        f"{START_MARKER}\n"
        "PASSED t::a\n"
        f"{END_MARKER}\n"
        "PASSED tests/after.py::test_also_not\n"
    )
    parsed = parse_eval_output(instance, noisy)
    assert "tests/decoy.py::test_should_not_count" not in parsed
    assert "tests/after.py::test_also_not" not in parsed


def test_a_script_that_died_before_the_markers_parses_to_nothing(
    dataset: Path,
) -> None:
    """Parsing the whole log would read setup output as results -- and a
    non-empty parse is what separates 'the suite ran' from 'a hole'."""
    instance = load_dataset(dataset)[0]
    assert parse_eval_output(instance, "conda: command not found") == {}


def test_probe_names_carry_the_instance(dataset: Path) -> None:
    assert pass_to_pass_suite(load_dataset(dataset)[0]).name == "p2p::django__django-1"


# ------------------------------------------------------------------ grading


def test_per_test_status_is_what_makes_a_timeline_possible(dataset: Path) -> None:
    """An aggregate would say only that something broke, not which thing or
    when — so the report must be parsed per test."""
    instance = load_dataset(dataset)[0]
    raw = {
        instance.instance_id: {
            "resolved": False,
            "tests_status": {
                "FAIL_TO_PASS": {"success": [], "failure": ["tests/test_core.py::test_new"]},
                "PASS_TO_PASS": {"success": ["t::a"], "failure": ["t::b"]},
            },
        }
    }
    report = parse_report(raw, instance)

    assert report.resolved is False
    assert report.pass_to_pass_passed == 1
    assert report.pass_to_pass_total == 2
    assert report.regressed_tests == ("t::b",)


def test_skipped_counts_as_regressed_and_xfail_does_not() -> None:
    """Verified against upstream grading.py, not remembered: ``test_passed``
    is PASSED-or-XFAIL, and ``test_failed`` explicitly lists SKIPPED. This
    test used to assert the opposite ("SKIPPED is maintained") on a claimed
    official semantics that was never checked — a patch that skips its way
    around the oracle would have graded as resolved."""
    report = GradeReport(
        instance_id="x", resolved=False,
        fail_to_pass_passed=1, fail_to_pass_total=1,
        pass_to_pass_passed=1, pass_to_pass_total=2,
        per_test={"t::a": "XFAIL", "t::b": "SKIPPED"},
    )
    assert report.regressed_tests == ("t::b",)


def test_a_fully_clean_report_shows_no_regression(dataset: Path) -> None:
    instance = load_dataset(dataset)[0]
    raw = {
        instance.instance_id: {
            "resolved": True,
            "tests_status": {
                "FAIL_TO_PASS": {"success": ["tests/test_core.py::test_new"], "failure": []},
                "PASS_TO_PASS": {"success": ["t::a", "t::b"], "failure": []},
            },
        }
    }
    assert parse_report(raw, instance).regressed_tests == ()


# ------------------------------------------------------------------ prediction


def test_the_prediction_excludes_test_files(tmp_path: Path) -> None:
    """The grader restores test files anyway; emitting them would make our
    prediction differ from the thing actually scored."""
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "src.py").write_text("x = 1\n")
    (repo / "tests" / "test_core.py").write_text("def test_a(): pass\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t.co", "commit", "-qm", "base"],
        cwd=repo, check=True,
    )
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()

    (repo / "src.py").write_text("x = 2\n")
    (repo / "tests" / "test_core.py").write_text("def test_a(): assert False\n")

    patch = patch_for(repo, base)
    assert "src.py" in patch
    assert "test_core.py" not in patch


# ------------------------------------------------------------------ protocol


def test_the_agent_is_never_told_the_test_names(dataset: Path) -> None:
    """Departing from the benchmark's own prompt protocol would forfeit
    comparability — and handing over the oracle would end the study."""
    instance = load_dataset(dataset)[0]
    text = task_text(instance)

    assert instance.problem_statement in text
    for node_id in instance.pass_to_pass + instance.fail_to_pass:
        assert node_id not in text
    assert "PASS_TO_PASS" not in text


def test_the_repo_is_the_clustering_unit(dataset: Path) -> None:
    instances = load_dataset(dataset)
    assert {i.repo_short for i in instances} == {"django", "sympy", "astropy"}


def test_the_markers_are_echoed_not_shell_no_ops(dataset: Path) -> None:
    """Found by running against a real SWE-bench image.

    `: 'MARKER'` is the shell null command: it accepts the argument and prints
    NOTHING. The marker never reached the log, the slice found no start, and
    parse_eval_output returned {} -- so every graded test was classified as an
    infrastructure hole on a run where pytest had reported all 13 of them
    perfectly. An entire sweep would have read as "no results anywhere".
    """
    script = build_eval_script(load_dataset(dataset)[0])
    assert f"echo '{START_MARKER}'" in script
    assert f"echo '{END_MARKER}'" in script
    assert f": '{START_MARKER}'" not in script, "the null command prints nothing"


def test_stderr_is_folded_into_stdout(dataset: Path) -> None:
    """django reports test results on STDERR while the markers go to STDOUT.

    Concatenated afterwards, the results land *after* the end marker and the
    slice misses them entirely -- so django, 46% of the frame, would report
    every test as a hole even with the markers fixed.
    """
    script = build_eval_script(load_dataset(dataset)[0])
    assert script.splitlines()[0] == "exec 2>&1", (
        "the redirect must come first, or anything failing before it is lost"
    )


def test_a_real_pytest_log_round_trips(tmp_path: Path) -> None:
    """The exact shape captured from a real image run.

    Uses a pytest repository deliberately: the fixture dataset is django,
    whose runner has a different grammar entirely, and feeding pytest output
    to the django parser is how this test was wrong the first time.
    """
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps(_row(
        "astropy__astropy-1", "astropy/astropy", version="5.1",
        p2p=["astropy/tests/test_x.py::test_a", "astropy/tests/test_x.py::test_b"],
    )))
    instance = load_dataset(path)[0]

    log = (
        "Applied patch cleanly.\n"
        f"{START_MARKER}\n"
        "PASSED astropy/tests/test_x.py::test_a\n"
        "FAILED astropy/tests/test_x.py::test_b\n"
        "===================== 1 failed, 1 passed in 0.42s =====================\n"
        f"{END_MARKER}\n"
    )
    parsed = parse_eval_output(instance, log)
    assert parsed.get("astropy/tests/test_x.py::test_a") == "PASSED"
    assert parsed.get("astropy/tests/test_x.py::test_b") == "FAILED"


def test_published_image_uses_upstreams_tag_separator() -> None:
    """One definition of the image tag, because a wrong one does not raise.

    A driver that derives the tag differently evaluates the instance in some
    other environment and reports the difference as a change in the agent's
    behaviour. ``__`` is not legal in a Docker tag; ``_1776_`` is upstream's
    substitution, not ours.
    """
    instance = SWEInstance(
        instance_id="psf__requests-5414", repo="psf/requests",
        base_commit="0" * 40, problem_statement="", test_patch="",
        version="2.0", fail_to_pass=(), pass_to_pass=(),
    )
    assert instance.published_image == (
        "swebench/sweb.eval.x86_64.psf_1776_requests-5414:latest"
    )
    assert "__" not in instance.published_image.split(":")[0].split(".")[-1]


# ------------------------------------------------------------------ grading


def _grade_instance() -> SWEInstance:
    return SWEInstance(
        instance_id="pytest-dev__pytest-1000", repo="pytest-dev/pytest",
        base_commit="0" * 40, problem_statement="", test_patch="diff", version="7.2",
        fail_to_pass=("testing/test_a.py::test_fixed",),
        pass_to_pass=("testing/test_a.py::test_old",),
    )


def _eval_log(*lines: str) -> str:
    return "\n".join([START_MARKER, *lines, END_MARKER])


def test_official_pass_semantics_xfail_passes_and_skipped_does_not() -> None:
    """The official grader's test_passed is PASSED-or-XFAIL. The previous
    implementation had both inversions: XFAIL counted as a regression and
    SKIPPED as a pass — the latter meaning a patch that skips its way around
    the oracle would grade as resolved."""
    report = GradeReport(
        instance_id="x", resolved=False,
        fail_to_pass_passed=0, fail_to_pass_total=0,
        pass_to_pass_passed=0, pass_to_pass_total=0,
        per_test={"a": "XFAIL", "b": "SKIPPED", "c": "PASSED", "d": "FAILED"},
    )
    assert report.regressed_tests == ("b", "d")


def test_grade_resolved_when_both_sets_pass(tmp_path: Path) -> None:
    from taste.benchmarks.swebench import grade_in_sandbox
    from taste.execution import ScriptedSandbox

    instance = _grade_instance()
    log = _eval_log(
        "PASSED testing/test_a.py::test_fixed",
        "PASSED testing/test_a.py::test_old",
    )
    sandbox = ScriptedSandbox().on("TASTE_START_TEST_OUTPUT", ExecResult(0, log, ""))
    report = grade_in_sandbox(sandbox, instance, "diff --git a/x b/x\n")
    assert report is not None and report.resolved is True
    assert report.fail_to_pass_passed == 1 and report.pass_to_pass_passed == 1


def test_grade_unresolved_when_a_graded_test_fails(tmp_path: Path) -> None:
    from taste.benchmarks.swebench import grade_in_sandbox
    from taste.execution import ScriptedSandbox

    instance = _grade_instance()
    log = _eval_log(
        "FAILED testing/test_a.py::test_fixed",
        "PASSED testing/test_a.py::test_old",
    )
    sandbox = ScriptedSandbox().on("TASTE_START_TEST_OUTPUT", ExecResult(1, log, ""))
    report = grade_in_sandbox(sandbox, instance, "diff --git a/x b/x\n")
    assert report is not None and report.resolved is False


def test_grade_with_no_markers_is_ungradable_not_unresolved(tmp_path: Path) -> None:
    """The script died before the markers: that is a missing measurement.
    Scoring it resolved=False would let infrastructure manufacture failure —
    the mirror image of the fabricated-regressions invariant."""
    from taste.benchmarks.swebench import grade_in_sandbox
    from taste.execution import ScriptedSandbox

    instance = _grade_instance()
    sandbox = ScriptedSandbox().on(
        "TASTE_START_TEST_OUTPUT", ExecResult(127, "conda: not found", "")
    )
    assert grade_in_sandbox(sandbox, instance, "diff --git a/x b/x\n") is None


def test_grade_scores_an_unappliable_patch_as_unresolved(tmp_path: Path) -> None:
    from taste.benchmarks.swebench import grade_in_sandbox
    from taste.execution import ExecResult as ER
    from taste.execution import ScriptedSandbox

    instance = _grade_instance()
    sandbox = ScriptedSandbox().on("git apply -v /tmp/taste_pred.diff", ER(1, "", "corrupt patch"))
    report = grade_in_sandbox(sandbox, instance, "diff --git a/x b/x\n")
    assert report is not None and report.resolved is False
    assert report.per_test == {}, "nothing ran; the verdict is about the patch"


def test_grade_of_the_empty_patch_runs_the_eval_and_fails_f2p(tmp_path: Path) -> None:
    """An agent that produced nothing still gets a real verdict: the base
    tree fails FAIL_TO_PASS by definition. None here would hide every
    do-nothing run from the resolve rate."""
    from taste.benchmarks.swebench import grade_in_sandbox
    from taste.execution import ScriptedSandbox

    instance = _grade_instance()
    log = _eval_log(
        "FAILED testing/test_a.py::test_fixed",
        "PASSED testing/test_a.py::test_old",
    )
    sandbox = ScriptedSandbox().on("TASTE_START_TEST_OUTPUT", ExecResult(1, log, ""))
    report = grade_in_sandbox(sandbox, instance, "")
    assert report is not None and report.resolved is False
    assert not any("taste_pred" in c for c in sandbox.commands), (
        "an empty patch must not be applied"
    )


def test_a_patch_that_kills_the_suite_is_a_failed_verdict_not_a_hole(tmp_path: Path) -> None:
    """Seven no-recovery cells vanished from the contrast's primary endpoint as
    'ungradable' because their patches made the suite uncollectable and no
    result appeared between the markers. The official grader scores that as
    every test failed. The baseline run disambiguates: results without the
    patch mean the environment is alive and the patch is what killed it."""
    from taste.benchmarks.swebench import grade_in_sandbox
    from taste.execution import ScriptedSandbox

    instance = _grade_instance()
    alive_baseline = _eval_log(
        "FAILED testing/test_a.py::test_fixed",
        "PASSED testing/test_a.py::test_old",
    )
    calls = {"n": 0}

    class _Box(ScriptedSandbox):
        def exec(self, command, *, timeout=600, env=None):
            if "TASTE_START_TEST_OUTPUT" in command:
                calls["n"] += 1
                # First eval (patched): the suite died before any result.
                # Second eval (baseline, after reset): alive.
                return ExecResult(2, "INTERNALERROR> SyntaxError in conftest" if calls["n"] == 1 else alive_baseline, "")
            return super().exec(command, timeout=timeout, env=env)

    report = grade_in_sandbox(_Box(), instance, "diff --git a/x b/x\n")
    assert report is not None, "a patch-killed suite must be a verdict, not a hole"
    assert report.resolved is False
    assert report.pass_to_pass_passed == 0 and report.fail_to_pass_passed == 0
    assert set(report.per_test.values()) == {"MISSING"}
    assert calls["n"] == 2, "the baseline disambiguation must actually run"


def test_a_dead_environment_is_still_a_hole(tmp_path: Path) -> None:
    """The mirror case: no results with OR without the patch is infrastructure."""
    from taste.benchmarks.swebench import grade_in_sandbox
    from taste.execution import ScriptedSandbox

    instance = _grade_instance()
    sandbox = ScriptedSandbox().on("TASTE_START_TEST_OUTPUT", ExecResult(127, "conda: not found", ""))
    assert grade_in_sandbox(sandbox, instance, "diff --git a/x b/x\n") is None


def test_the_prediction_excludes_the_harness_s_own_artifacts(tmp_path: Path) -> None:
    """Bug 32. .taste/plan.json and the monitor verdicts live in the tree and
    were leaking into the prediction diff — applied cleanly, graded
    harmlessly, which is exactly why nobody noticed. A prediction is the
    agent's source change and nothing else."""
    import subprocess as sp

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "lib.py").write_text("x = 1\n")
    sp.run(["git", "-C", str(ws), "init", "-q"], check=True)
    git = ["git", "-C", str(ws), "-c", "user.name=t", "-c", "user.email=t@l"]
    sp.run([*git, "add", "-A"], check=True)
    sp.run([*git, "commit", "-qm", "root"], check=True)
    root = sp.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (ws / "lib.py").write_text("x = 2\n")
    (ws / ".taste").mkdir()
    (ws / ".taste" / "plan.json").write_text("{}\n")
    (ws / ".taste" / "monitor").mkdir()
    (ws / ".taste" / "monitor" / "step-01.json").write_text("{}\n")
    sp.run([*git, "add", "-A"], check=True)

    patch = patch_for(ws, root)
    assert "lib.py" in patch
    assert ".taste" not in patch, "harness artifacts leaked into the prediction"


def test_rescore_wires_the_grader() -> None:
    """Bug 31. rescore.py predated make_grade and never passed it, so a
    'regrade' of seven dropped cells was a no-op that reported the same
    missing verdicts while the grader fix sat unused. A wiring guard: the
    re-scorer must construct its scorer with a grader."""
    src = (Path(__file__).resolve().parent.parent / "scripts" / "rescore.py").read_text()
    assert "grade=make_grade()" in src


def test_materialize_skips_untransportable_links_instead_of_aborting(tmp_path) -> None:
    """tox's image ships a fixture virtualenv with absolute symlinks; the host
    copy must skip and record them, never abort the cell (bug 29's family)."""
    import io
    import tarfile

    from taste.benchmarks.swebench import lenient_data_filter

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        data = b"print(1)\n"
        info = tarfile.TarInfo("pkg/a.py"); info.size = len(data); tar.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo("pkg/pip"); link.type = tarfile.SYMTYPE; link.linkname = "/usr/bin/pip"; tar.addfile(link)
    buf.seek(0)
    skipped: list[str] = []
    with tarfile.open(fileobj=buf) as tar:
        tar.extractall(tmp_path, filter=lenient_data_filter(skipped))
    assert (tmp_path / "pkg" / "a.py").read_text() == "print(1)\n"
    assert skipped == ["pkg/pip"]
