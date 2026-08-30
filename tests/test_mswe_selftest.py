"""The self-test classifier, against the cases a hand audit found it failing.

Every case here is a real command and a real output from an archived
mini-swe-agent trajectory, kept because the first version of the classifier
got it wrong and reported a number to match.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from mswe_selftest import (
    analyse_episode,
    covers,
    is_test_command,
    locatable,
    names_failure,
    parse_probe,
    ran_anything,
    targets_of,
)

DJANGO_PROBE = "test_BA_BCA__BAB_BAC_BCA (queries.tests.UnionTests)"
MPL_PROBE = "lib/mpl_toolkits/axes_grid1/tests/test_axes_grid1.py::test_anchored_locator_base_call"


def test_probe_ids_parse_into_name_and_one_specific_target() -> None:
    """One target, matched by label boundary in both directions: emitting the
    package as well is what let a sibling module count as coverage."""
    assert parse_probe(DJANGO_PROBE) == ("test_BA_BCA__BAB_BAC_BCA", ["queries.tests.UnionTests"])
    name, targets = parse_probe(MPL_PROBE)
    assert name == "test_anchored_locator_base_call"
    assert targets == ["lib/mpl_toolkits/axes_grid1/tests/test_axes_grid1.py"]
    # sympy emits bare function names: nothing can target them but the suite
    assert parse_probe("test_difference") == ("test_difference", [])
    assert not locatable("test_difference") and locatable(DJANGO_PROBE)


def test_a_runner_word_inside_a_path_is_not_a_test_command() -> None:
    assert not is_test_command("sed -i 's/x/y/' src/_pytest/python.py")
    assert not is_test_command("python -c \"import numpy.testing\"")
    assert is_test_command("cd /testbed && python -m pytest -x tests/")
    assert is_test_command("python tests/runtests.py delete --settings=tests.test_sqlite")
    assert is_test_command("python -c \"import sympy; sympy.test('sympy/sets/')\"")


def test_targets_are_the_runners_positional_arguments() -> None:
    assert targets_of("python tests/runtests.py delete --settings=tests.test_sqlite") == ["delete"]
    assert targets_of("cd /testbed && python tests/runtests.py --settings=tests.test_sqlite") == []
    assert targets_of("python -m pytest astropy/utils/tests/test_introspection.py -v") == ["astropy/utils/tests/test_introspection.py"]
    assert targets_of("python -m pytest -x -q") == []


def test_sibling_modules_do_not_cover_a_probe_in_another_module() -> None:
    """The audit's headline miscount: the string 'queries' is not coverage."""
    command = "python tests/runtests.py queries.test_query queries.test_qs_combinators --settings=tests.test_sqlite"
    output = "Ran 60 tests in 0.4s\n\nOK\n"
    name, targets = parse_probe(DJANGO_PROBE)
    assert covers(command, output, name, targets)[0] == "no"
    # the module that does hold it counts
    assert covers("python tests/runtests.py queries --settings=tests.test_sqlite", output, name, targets)[0] == "yes"


def test_a_k_filter_that_deselects_the_probe_is_not_coverage() -> None:
    command = "python -m pytest lib/mpl_toolkits/axes_grid1/tests/test_axes_grid1.py -k 'inset_axes_complete or anchored_locator_base_call'"
    deselected = "2 failed, 47 deselected in 3.1s\n"
    other = "lib/mpl_toolkits/axes_grid1/tests/test_axes_grid1.py::test_imagegrid_cbar_mode_edge"
    name, targets = parse_probe(other)
    assert covers(command, deselected, name, targets)[0] == "no"
    # the tests the filter did select are covered
    name2, targets2 = parse_probe(MPL_PROBE)
    assert covers(command, deselected, name2, targets2)[0] == "yes"


def test_a_runner_that_never_started_did_not_test() -> None:
    assert not ran_anything("\nbash: line 1: pytest: command not found\n")
    assert not ran_anything("/opt/conda/bin/python: No module named pytest\n")
    assert ran_anything("collected 74 items\n\n74 passed in 2.3s\n")
    assert ran_anything("Ran 52 tests in 1.2s\n\nOK\n")


def test_a_failure_counts_only_when_the_line_names_this_test() -> None:
    out = "FAILED testing/test_collection.py::Test_getinitialnodes::test_pkgfile - assert\n1 failed, 70 passed\n"
    assert names_failure(out, "test_pkgfile")
    assert not names_failure(out, "test_something_else")
    assert names_failure("ERROR: test_BA_BCA__BAB_BAC_BCA (queries.tests.UnionTests)\n", "test_BA_BCA__BAB_BAC_BCA")
    # a passing mention is not a failure
    assert not names_failure("test_pkgfile PASSED\n1 failed, 70 passed\n", "test_pkgfile")


def _episode(onset_seq: int = 2, recovered_seq: int | None = 3, probe: str = DJANGO_PROBE) -> dict:
    return {"probe": probe, "onset_seq": onset_seq, "recovered_seq": recovered_seq, "onset_sha": "abc"}


def test_the_window_ends_at_recovery() -> None:
    """A test run after the repair cannot have told the agent anything."""
    commands = [
        "sed -i 's/x/y/' django/db/models/sql/query.py",              # cmd-001 breaks
        "python tests/runtests.py queries --settings=tests.test_sqlite",  # cmd-002 (still broken)
        "git checkout -- django/db/models/sql/query.py",              # cmd-003 repairs
        "python tests/runtests.py queries --settings=tests.test_sqlite",  # cmd-004 after the fix
    ]
    outputs = ["", "ERROR: test_BA_BCA__BAB_BAC_BCA (queries.tests.UnionTests)\nRan 60 tests\nFAILED", "", "Ran 60 tests\n\nOK\n"]
    seqmap = {1: 1, 2: 2, 3: 3}  # observation seq -> command number
    row = analyse_episode(_episode(onset_seq=1, recovered_seq=3), commands, outputs, seqmap)
    assert row["tested"] and row["covered"] == "yes" and row["shown"] == "yes"
    assert row["first_showing_cmd"] == 2

    # the same run, but the only covering test happens *after* the repair
    outputs_late = ["", "", "", "Ran 60 tests\n\nOK\n"]
    row2 = analyse_episode(_episode(onset_seq=1, recovered_seq=3), commands, outputs_late, seqmap)
    assert not row2["tested"] and row2["covered"] == "no", "a post-repair run must not count"


def test_every_covering_run_is_inspected_not_only_the_first() -> None:
    """The first covering command is often a crashed runner."""
    commands = [
        "sed -i 's/a/b/' src/_pytest/python.py",
        "python -m pytest testing/test_collection.py",     # crashed: no module
        "python -m pytest testing/test_collection.py -v",  # the real run, shows the failure
        "sed -i 's/b/a/' src/_pytest/python.py",
    ]
    outputs = [
        "",
        "/opt/conda/bin/python: No module named pytest\n",
        "collected 71 items\nFAILED testing/test_collection.py::Test_getinitialnodes::test_pkgfile\n1 failed, 70 passed\n",
        "",
    ]
    probe = "testing/test_collection.py::Test_getinitialnodes::test_pkgfile"
    row = analyse_episode(_episode(onset_seq=1, recovered_seq=4, probe=probe), commands, outputs, {1: 1, 4: 4})
    assert row["tested"] and row["covered"] == "yes" and row["shown"] == "yes"
    assert row["first_covering_cmd"] == 3 and row["test_runs_in_window"] == 1


def test_an_unrecovered_episode_scans_to_the_end() -> None:
    commands = ["sed -i s/a/b/ mod.py", "python -m pytest"]
    outputs = ["", "collected 3 items\n3 passed\n"]
    row = analyse_episode(_episode(onset_seq=1, recovered_seq=None), commands, outputs, {1: 1})
    assert row["persisted_to_end"] and row["tested"] and row["covered"] == "yes" and row["shown"] == "no"


@pytest.mark.parametrize("probe,expected", [(DJANGO_PROBE, True), ("test_difference", False)])
def test_unlocatable_probe_ids_are_reported_not_guessed(probe: str, expected: bool) -> None:
    row = analyse_episode(_episode(probe=probe), ["x"], [""], {2: 1})
    assert row["locatable"] is expected


def test_an_early_stopping_or_truncated_run_is_unknown_not_covered() -> None:
    """The seaborn cell in the GPT sweep: `pytest -x ... | tail -40`.

    `-x` stops at the first failure, so tests ordered after it were never
    reached, and the pipe means the visible output is a window. Calling
    that "ran it and the agent still was not told" is a strong claim built
    on absent evidence; it accounted for 33 of 48 apparent coverages.
    """
    command = "cd /testbed && pytest -q tests/_core/test_plot.py -x --tb=short 2>&1 | tail -40"
    output = "F\n=== FAILURES ===\n_____ TestLayerAddition.test_without_data _____\n1 failed, 70 passed\n"
    probe = "tests/_core/test_plot.py::TestLabelVisibility::test_1d_column"
    name, targets = parse_probe(probe)
    verdict, why = covers(command, output, name, targets)
    assert verdict == "unknown" and "stopped at an earlier failure" in why

    # the same run without -x, but still piped: the window may hide the failure
    piped = "cd /testbed && pytest -q tests/_core/test_plot.py | tail -5"
    verdict2, why2 = covers(piped, "250 passed in 8.7s\n", name, targets)
    assert verdict2 == "unknown" and "truncated" in why2

    # full output, no early stop: the run really did execute it
    full = "cd /testbed && pytest -q tests/_core/test_plot.py"
    assert covers(full, "250 passed, 6 xfailed in 8.73s\n", name, targets)[0] == "yes"
