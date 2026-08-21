"""Linking a Monitor failure to the regression it was actually about.

The failure mode these tests exist to prevent is subtle and one-directional:
crediting an arm with *detecting* a regression because it happened to fail
nearby. That biases toward whichever arm fails most often, which is the
opposite of what the study claims to measure.

The second thing under test is the distinction between "measured, and nothing
links" and "we could not measure". Collapsing the second into the first is how
a pipeline manufactures silence.

Pure functions throughout — no Docker, no git, no API.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from taste.attribution import (
    CoverageMap,
    MonitorFailure,
    attribution_map,
    failed_at,
    harness_failures,
    read_coverage_sqlite,
    read_events,
    summarise_silence,
)
from taste.replay import RegressionEpisode
from taste.shadow import ShadowCommit


def _commit(seq: int, step_id: str, attempt: int, *, trigger: str = "worker") -> ShadowCommit:
    return ShadowCommit(
        seq=seq, sha=f"sha{seq:04d}", session="s", step_id=step_id,
        attempt=attempt, trigger=trigger,
    )


def _verdict(step_id: str, attempt: int, *, passed: bool, tests: tuple[str, ...] = ()) -> dict:
    return {
        "kind": "monitor.verdict",
        "ts": 1.0,
        "payload": {
            "id": step_id, "attempt": attempt, "passed": passed,
            "reason": "r", "failing_tests": list(tests),
        },
    }


# ------------------------------------------------------------------ the join


def test_a_failure_lands_on_the_tree_the_monitor_actually_graded() -> None:
    """The observation fires between the worker finishing and the Monitor
    evaluating, so (step_id, attempt) identifies the graded tree exactly."""
    timeline = [_commit(1, "step-01", 1), _commit(2, "step-01", 2), _commit(3, "step-02", 1)]
    events = [
        _verdict("step-01", 1, passed=False),
        _verdict("step-01", 2, passed=True),
        _verdict("step-02", 1, passed=False),
    ]

    failures = harness_failures(events, timeline)

    assert [(f.step_id, f.attempt, f.seq) for f in failures] == [
        ("step-01", 1, 1), ("step-02", 1, 3),
    ]


def test_passing_verdicts_are_not_failures() -> None:
    failures = harness_failures([_verdict("s", 1, passed=True)], [_commit(1, "s", 1)])
    assert failures == []


def test_a_failure_on_an_unchanged_tree_is_kept_not_dropped() -> None:
    """No shadow commit is written when the worker changed nothing, so there
    is no observation to index. Dropping the failure would silently convert a
    detected regression into a silent one."""
    failures = harness_failures([_verdict("step-09", 1, passed=False)], [_commit(1, "step-01", 1)])

    assert len(failures) == 1
    assert failures[0].seq is None
    assert failed_at(failures) == set()


def test_reverify_yields_two_verdicts_for_one_attempt() -> None:
    """REVERIFY re-runs the check without a new attempt, so the key repeats
    and both verdicts must survive."""
    events = [_verdict("step-01", 1, passed=False), _verdict("step-01", 1, passed=False)]
    assert len(harness_failures(events, [_commit(1, "step-01", 1)])) == 2


def test_unrelated_event_kinds_are_ignored() -> None:
    events = [{"kind": "worker.done", "payload": {"id": "x"}}, _verdict("s", 1, passed=False)]
    assert len(harness_failures(events, [_commit(1, "s", 1)])) == 1


def test_a_truncated_event_log_still_loads(tmp_path: Path) -> None:
    """A killed run leaves a partial final line; refusing the whole file
    would discard an otherwise complete cell."""
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"kind": "run.start", "payload": {}}) + "\n{\"kind\": \"mon")
    assert len(read_events(path)) == 1


# ------------------------------------------------------------------ coverage


def test_unknown_and_covers_nothing_are_different_answers() -> None:
    """The whole silent/unknown distinction rests on this."""
    cov = CoverageMap(
        instance_id="i", built_at_commit="c", method="declared",
        covers={"t_measured": frozenset()},
        uninstrumented=frozenset({"t_broken"}),
    )
    assert cov.files_for("t_measured") == frozenset()   # measured, covers nothing
    assert cov.files_for("t_broken") is None            # unmeasurable
    assert cov.files_for("t_absent") is None            # never seen


def test_coverage_is_read_from_a_real_coverage_database(tmp_path: Path) -> None:
    """Written by hand against coverage.py's schema so the reader is proven
    without installing the package."""
    db = tmp_path / ".coverage"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE file (id INTEGER PRIMARY KEY, path TEXT);
        CREATE TABLE context (id INTEGER PRIMARY KEY, context TEXT);
        CREATE TABLE line_bits (file_id INTEGER, context_id INTEGER, numbits BLOB);
        INSERT INTO file VALUES (1, '/testbed/pkg/core.py'), (2, '/testbed/pkg/util.py');
        INSERT INTO context VALUES (1, 'tests/test_a.py::test_one|call'), (2, '');
        INSERT INTO line_bits VALUES (1, 1, x'00'), (2, 1, x'00'), (1, 2, x'00');
        """
    )
    conn.commit()
    conn.close()

    cov = read_coverage_sqlite(db, instance_id="i", built_at_commit="c", root="/testbed")

    assert cov.files_for("tests/test_a.py::test_one") == frozenset({"pkg/core.py", "pkg/util.py"})
    assert cov.method == "pytest_cov_context"


# ------------------------------------------------------------------ the rule


MONITOR = CoverageMap(
    instance_id="i", built_at_commit="c", method="declared",
    covers={
        "mon_touching_core": frozenset({"pkg/core.py"}),
        "mon_elsewhere": frozenset({"pkg/other.py"}),
    },
)
PROBE = CoverageMap(
    instance_id="i", built_at_commit="c", method="declared",
    covers={
        "graded_core": frozenset({"pkg/core.py"}),
        "graded_other": frozenset({"pkg/far.py"}),
    },
)


def test_a_shared_changed_file_links_them() -> None:
    result = attribution_map(
        failures=[MonitorFailure("s", "st", 1, seq=5, failing_tests=("mon_touching_core",))],
        probe_tests=["graded_core", "graded_other"],
        monitor_coverage=MONITOR, probe_coverage=PROBE,
        modified_files_at={5: frozenset({"pkg/core.py"})},
    )
    assert result.by_seq == {5: {"graded_core"}}


def test_co_occurrence_alone_does_not_link() -> None:
    """The bias this whole module exists to remove."""
    result = attribution_map(
        failures=[MonitorFailure("s", "st", 1, seq=5, failing_tests=("mon_elsewhere",))],
        probe_tests=["graded_core"],
        monitor_coverage=MONITOR, probe_coverage=PROBE,
        modified_files_at={5: frozenset({"pkg/core.py", "pkg/other.py"})},
    )
    assert result.by_seq == {}
    assert result.decisions[0].verdict == "unattributed"


def test_a_shared_file_the_agent_did_not_touch_does_not_link() -> None:
    """Without this term, any two tests importing a shared utility are linked
    in every run forever — which would attribute everything and erase the
    phenomenon."""
    result = attribution_map(
        failures=[MonitorFailure("s", "st", 1, seq=5, failing_tests=("mon_touching_core",))],
        probe_tests=["graded_core"],
        monitor_coverage=MONITOR, probe_coverage=PROBE,
        modified_files_at={5: frozenset({"pkg/unrelated.py"})},
    )
    assert result.by_seq == {}


def test_missing_probe_coverage_is_unknown_never_unattributed() -> None:
    """Counting absence of data as evidence of silence would inflate the
    headline number in the direction of our own hypothesis."""
    result = attribution_map(
        failures=[MonitorFailure("s", "st", 1, seq=5, failing_tests=("mon_touching_core",))],
        probe_tests=["never_measured"],
        monitor_coverage=MONITOR, probe_coverage=PROBE,
        modified_files_at={5: frozenset({"pkg/core.py"})},
    )
    assert result.decisions[0].verdict == "unknown"
    assert result.unknown_rate == 1.0


def test_missing_monitor_coverage_is_unknown_too() -> None:
    result = attribution_map(
        failures=[MonitorFailure("s", "st", 1, seq=5, failing_tests=("mon_unmeasured",))],
        probe_tests=["graded_core"],
        monitor_coverage=MONITOR, probe_coverage=PROBE,
        modified_files_at={5: frozenset({"pkg/core.py"})},
    )
    assert result.decisions[0].verdict == "unknown"


def test_a_failure_naming_no_tests_is_unknown() -> None:
    """A Monitor that failed without identifying what failed tells us nothing
    about attribution — which is not the same as telling us it was unrelated."""
    result = attribution_map(
        failures=[MonitorFailure("s", "st", 1, seq=5, failing_tests=())],
        probe_tests=["graded_core"],
        monitor_coverage=MONITOR, probe_coverage=PROBE,
        modified_files_at={5: frozenset({"pkg/core.py"})},
    )
    assert result.decisions[0].verdict == "unknown"


def test_an_unindexed_failure_cannot_attribute_anything() -> None:
    result = attribution_map(
        failures=[MonitorFailure("s", "st", 1, seq=None, failing_tests=("mon_touching_core",))],
        probe_tests=["graded_core"],
        monitor_coverage=MONITOR, probe_coverage=PROBE,
        modified_files_at={},
    )
    assert result.by_seq == {}
    assert result.considered == 0


def test_the_map_plugs_straight_into_reconstruct() -> None:
    """Its shape is exactly what reconstruct already consumes, so the tested
    detection path is reused rather than duplicated."""
    result = attribution_map(
        failures=[MonitorFailure("s", "st", 1, seq=7, failing_tests=("mon_touching_core",))],
        probe_tests=["graded_core"],
        monitor_coverage=MONITOR, probe_coverage=PROBE,
        modified_files_at={7: frozenset({"pkg/core.py"})},
    )
    assert isinstance(result.by_seq, dict)
    assert all(isinstance(k, int) and isinstance(v, set) for k, v in result.by_seq.items())


# ------------------------------------------------------------------ reporting


def test_both_variants_are_reported_side_by_side() -> None:
    """The co-occurrence number is not deleted — a reader is entitled to see
    how much of the result rests on the coverage machinery."""
    episodes = [
        RegressionEpisode(probe="a", onset_seq=1, onset_sha="x",
                          detected_seq_attributed=None, detected_seq_any=3),
        RegressionEpisode(probe="b", onset_seq=2, onset_sha="y",
                          detected_seq_attributed=4, detected_seq_any=4),
        RegressionEpisode(probe="c", onset_seq=3, onset_sha="z"),
    ]
    report = summarise_silence(episodes, attribution_map(
        failures=[], probe_tests=[], monitor_coverage=MONITOR,
        probe_coverage=PROBE, modified_files_at={},
    ), method="pytest_cov_context")

    assert report.episodes == 3
    assert report.silent_attributed == 2      # a and c
    assert report.silent_unattributed == 1    # only c
    assert report.silence_rate_bound < report.silence_rate, (
        "the bound must under-count silence; if it ever exceeds the primary "
        "measure the two variants have been swapped"
    )


# ------------------------------------------------------------------ id shapes


def test_every_published_id_grammar_parses() -> None:
    """PASS_TO_PASS uses four grammars across the frame; a parser that handles
    only pytest node ids silently drops 38.5% of the graded tests."""
    from taste.attribution import parse_member_id

    cases = {
        "astropy/modeling/tests/test_separable.py::test_coord_matrix":
            ("test_coord_matrix", None, "test_separable"),
        "tests/_core/test_plot.py::TestInit::test_empty":
            ("test_empty", "TestInit", "test_plot"),
        "test_defaults (str.tests.SimpleTests)":
            ("test_defaults", "SimpleTests", "str.tests"),
        # django 4.2+ repeats the method at the end of the dotted path
        "test_x (user_commands.tests.UtilsTests.test_x)":
            ("test_x", "UtilsTests", "user_commands.tests"),
        "test_point3D": ("test_point3D", None, ""),
    }
    for raw, (func, cls, module) in cases.items():
        key = parse_member_id(raw)
        assert key is not None, raw
        assert (key.func, key.cls, key.module) == (func, cls, module), raw


def test_a_docstring_is_not_a_test_identifier() -> None:
    """django's runner prints a test's docstring instead of its name when it
    has one, so 6.1% of published ids are prose. Returning a bogus key would
    attribute coverage to a test that does not exist."""
    from taste.attribution import parse_member_id

    assert parse_member_id("Trailing zeros in the fractional part aren't truncated.") is None
    assert parse_member_id("[100%]") is None
    assert parse_member_id("") is None


def test_coverage_contexts_parse_from_any_runner() -> None:
    """coverage.py's dynamic_context names the context after the executing
    test function whoever invoked it -- which is why it works for django and
    sympy, and pytest-cov's --cov-context does not."""
    from taste.attribution import parse_context

    assert parse_context("pkg.mod.TestCls.test_thing|call").func == "test_thing"
    assert parse_context("pkg.mod.TestCls.test_thing").cls == "TestCls"
    assert parse_context("pkg.mod.test_thing").cls is None
    assert parse_context("pkg.mod.helper") is None, "not a test function"


def test_ids_and_contexts_reconcile_across_schemes() -> None:
    from taste.attribution import reconcile_contexts

    result = reconcile_contexts(
        ["str.tests.SimpleTests.test_defaults|call", "sympy.geometry.tests.test_point.test_point3D"],
        ["test_defaults (str.tests.SimpleTests)", "test_point3D"],
    )
    assert len(result.matched) == 2
    assert result.unresolved == ()


def test_a_class_disambiguates_a_shared_function_name() -> None:
    from taste.attribution import reconcile_contexts

    result = reconcile_contexts(
        ["m.AlphaTests.test_run", "m.BetaTests.test_run"],
        ["test_run (m.BetaTests)"],
    )
    assert result.matched["test_run (m.BetaTests)"] == "m.BetaTests.test_run"


def test_genuine_ambiguity_is_reported_never_guessed() -> None:
    """A wrong match attributes a Monitor failure to the wrong test, which is
    the exact error the coverage rule exists to prevent. sympy's bare names
    carry no module, so this case is real, not hypothetical."""
    from taste.attribution import reconcile_contexts

    result = reconcile_contexts(
        ["a.mod.Cls.test_x", "b.other.Other.test_x"], ["test_x"]
    )
    assert result.matched == {}
    assert result.ambiguous == ("test_x",)
    assert "test_x" in result.unresolved


def test_the_three_unresolved_kinds_stay_distinct() -> None:
    """Each says something different about why coverage is missing, and the
    protocol reports them separately."""
    from taste.attribution import reconcile_contexts

    result = reconcile_contexts(
        ["m.Cls.test_present", "a.test_dup", "b.test_dup"],
        ["test_present (m.Cls)", "[100%]", "test_never_ran", "test_dup"],
    )
    assert list(result.matched) == ["test_present (m.Cls)"]
    assert result.unmappable == ("[100%]",)
    assert result.unmatched == ("test_never_ran",)
    assert result.ambiguous == ("test_dup",)


def test_a_failure_lands_on_the_last_tree_of_the_attempt() -> None:
    """Exposed by the per-tool observation grid.

    The Monitor evaluates after the worker finishes, so the tree it graded is
    the LAST observation of that (step_id, attempt). The previous rule
    preferred a `worker`-triggered commit and otherwise kept the first — but
    under the fine grid the final tool observation already captures the tree
    the worker left, so the boundary commit dedupes away and no `worker`
    observation exists. The failure was then pinned to the earliest tree of
    the attempt, understating detection latency and blaming a tree that did
    not yet contain the break.
    """
    timeline = [
        _commit(1, "step-01", 1, trigger="tool"),
        _commit(2, "step-01", 1, trigger="tool"),
        _commit(3, "step-01", 1, trigger="tool"),
    ]
    failures = harness_failures([_verdict("step-01", 1, passed=False)], timeline)

    assert [f.seq for f in failures] == [3], "must be the last tree, not the first"


def test_the_coarse_grid_still_joins_the_same_way() -> None:
    """With one observation per attempt, last and worker are the same commit,
    so the fix cannot change any existing behaviour."""
    timeline = [_commit(1, "step-01", 1, trigger="worker"), _commit(2, "step-02", 1)]
    failures = harness_failures([_verdict("step-01", 1, passed=False)], timeline)
    assert [f.seq for f in failures] == [1]
