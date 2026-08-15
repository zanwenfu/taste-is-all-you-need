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
    GradeReport,
    eligible,
    graded_test_files,
    load_dataset,
    monitor_scope,
    parse_report,
    pass_to_pass_probe,
    patch_for,
    stratified_sample,
    task_text,
)

TEST_PATCH = """diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1,2 +1,3 @@
 def test_old(): pass
+def test_new(): pass
"""


def _row(instance_id: str, repo: str, *, p2p: list[str], f2p: list[str] | None = None) -> dict:
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": "abc123",
        "problem_statement": "Something is broken.",
        "test_patch": TEST_PATCH,
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
    probe = pass_to_pass_probe(load_dataset(dataset)[0])
    assert "TASTE_TEST_PATCH" in probe.command
    assert "tests/test_core.py" in probe.command


def test_the_probe_runs_exactly_the_pass_to_pass_set(dataset: Path) -> None:
    probe = pass_to_pass_probe(load_dataset(dataset)[0])
    assert "'t::a'" in probe.command and "'t::b'" in probe.command


def test_probe_names_carry_the_instance(dataset: Path) -> None:
    assert pass_to_pass_probe(load_dataset(dataset)[0]).name == "p2p::django__django-1"


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


def test_skipped_is_not_counted_as_regressed() -> None:
    """The official grader treats SKIPPED as maintained; we inherit that for
    the official verdict."""
    report = GradeReport(
        instance_id="x", resolved=True,
        fail_to_pass_passed=1, fail_to_pass_total=1,
        pass_to_pass_passed=1, pass_to_pass_total=2,
        per_test={"t::a": "PASSED", "t::b": "SKIPPED"},
    )
    assert report.regressed_tests == ()


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
