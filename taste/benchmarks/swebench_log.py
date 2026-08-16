"""Vendored SWE-bench log parsers: the layer that turns runner stdout into verdicts.

Every per-test status in our timeline is produced here. Nothing else in the
instrument decides whether a test passed; these functions do, by string
matching against a test runner's stdout. That places them directly under the
dependent variable.

**What breaks if this table is wrong.** A parser that fails to emit the key a
PASS_TO_PASS entry is written under does not report "unknown" — upstream's
grader (``get_eval_tests_report``) treats a missing key exactly as it treats a
failing test. A one-character drift in an id therefore manifests as *a test
that used to pass and now does not*: a fabricated regression, in the arm that
happened to be running, with no error, no warning, and no trace in the logs.
That is the exact event this paper measures. A parser bug does not add noise to
the measurement; it manufactures the signal. Hence: vendored verbatim from
upstream, one pinned commit, no improvements — including upstream's bugs, which
are documented below rather than fixed, because a number that disagrees with
the official harness is not comparable to the leaderboard and is worth nothing.

Upstream provenance
-------------------
Repository   https://github.com/SWE-bench/SWE-bench
Ref          ``main`` @ ``ca6e4e0d252f32f8762625b73575d5dee49d0a5a``
             (commit dated 2026-08-16T11:20:56Z, "Reject zero-count test
             summaries and keep task repo through re-grading"; fetched
             2026-08-17)
Files read   ``swebench/harness/log_parsers/python.py``     — the parsers
             ``swebench/harness/log_parsers/__init__.py``   — PARSER_REGISTRY keys
             ``swebench/harness/constants/__init__.py``     — TestStatus values
             ``swebench/harness/grading.py``                — how the maps are consumed

Why ``main`` and not the released ``v4.1.0``
--------------------------------------------
``pip install swebench`` currently yields v4.1.0 (@ ``726c5461e2ef52d8``), whose
``log_parsers/python.py`` differs from this ref. We pin ``main`` because
``docs/eval_protocol.md`` commits us to ``get_logs_eval``'s ``SUITE_RAN`` guard,
its test-exit-code cross-check and ``PARSER_REGISTRY[log_parser]`` — three
symbols that exist **only** on ``main``; v4.1.0 has no ``PARSER_REGISTRY`` at
all (it exposes ``MAP_REPO_TO_PARSER``) and no ``SUITE_RAN``. Mixing a v4.1.0
parser with a ``main`` grader would be an approximation of neither.

The four behavioural deltas v4.1.0 -> this ref, verified by diffing the two raw
files, all of them in code we vendor:

1. ``_is_skip_summary`` is new: ``SKIPPED [10] path:285: reason`` no longer
   registers a test named ``[10]``. Real line, from a captured matplotlib run:
   ``SKIPPED [10] lib/matplotlib/testing/compare.py:285: Don't know how to
   convert .svg files to png``. Deliberately *not* widened to ``PASSED [100%]``
   because pytest-dev__pytest-5262's PASS_TO_PASS literally contains the string
   ``[100%]`` — confirmed against ``data/verified.jsonl``.
2. ``parse_log_pytest_v2`` keys on ``" ".join(test_case[1:])`` instead of
   ``test_case[1]``, so ids containing spaces survive.
3. ``parse_log_pytest_v2`` drops a FAILED line's ``" - <message>"`` tail by
   splitting rather than by replacing ``" - "`` with a space.
4. ``parse_log_seaborn`` gained a ``len(parts) < 2`` guard; at v4.1.0 a bare
   ``FAILED`` line raises IndexError.

**Hazard carried by delta 2, stated because it is ours to carry.** pytest's
``-rA`` summary puts an xfail reason on the same line as the id:
``XFAIL xarray/tests/test_dataarray.py::TestReduce1D::test_idxmin[True-x5-0-1-None] - reason: dask operation 'argmin' breaks when dtype is datetime64 (M)``
(verbatim, captured run of pydata__xarray-4695). Under this ref
``parse_log_pytest_v2`` keys that as ``"<id> - reason: ..."``; under v4.1.0 it
keyed it as ``"<id>"``. ``grading.test_passed`` counts XFAIL as a pass, so on
astropy / scikit-learn / sphinx — the three repos on this parser — an xfailing
PASS_TO_PASS test resolves under v4.1.0 and reads as a regression under
``main``. No id in SWE-bench Verified contains a space (checked: 0 of 3735
astropy, 2093 scikit-learn, 1081 sphinx ids), so delta 2 buys us nothing on
this frame and costs us this. We still follow ``main``, because agreeing with
the grader we actually run beats being right on our own.

Deviation from upstream signatures (the only one)
-------------------------------------------------
Upstream's parsers take ``(log: str, test_spec: TestSpec)``. No Python parser
body at this ref references ``test_spec`` — checked by reading the whole file —
so it is dropped, keeping this module import-free and pure. Bodies are
otherwise transcribed line for line.

Coverage of our frame
---------------------
Upstream selects a parser by *repository only*; version never enters
``MAP_REPO_TO_PARSER_PY``. The 80 (repo, version) pairs of SWE-bench Verified
therefore reduce to 12 repositories, all 12 present below, and the 500
instances to 7 distinct parser bodies. Django alone is 231/500 — the
``parse_log_django`` transcription is load-bearing for 46% of the frame.

Real-log form catalogue for ``parse_log_django``
------------------------------------------------
Django's runner is stdlib ``unittest`` at ``--verbosity 2``, so its ids come in
two families and its status lines in five shapes. All lines below are verbatim
from captured SWE-bench ``test_output.txt`` files (sources in ``FIXTURES``):

* id family A, Django <= 4.2 — ``test_accent (dbshell.test_postgresql.PostgreSqlDbshellCommandTestCase) ... ok``
* id family B, Django 5.0 on Python 3.11 — unittest now repeats the method in
  the dotted path: ``test_aggregation_subquery_annotation_exists (aggregation.tests.AggregateTestCase.test_aggregation_subquery_annotation_exists) ... ok``
* **docstring displacement** — when a test has a docstring, ``getDescription``
  returns ``str(test) + "\\n" + docstring_first_line``, so the id lands on its
  own line and the status attaches to the *docstring*::

      test_sigint_handler (dbshell.test_postgresql.PostgreSqlDbshellCommandTestCase)
      SIGINT is ignored in Python and passed to psql to abort queries. ... skipped 'Requires a PostgreSQL connection'

  The parser therefore keys the docstring text, and so does the dataset:
  counting PASS_TO_PASS + FAIL_TO_PASS ids of this shape in
  ``data/verified.jsonl`` gives 0 for Django 1.11 and between 55 (3.1) and 972
  (4.0) for every later version — e.g. ``'Test change detection of new
  constraints.'``. This is not a bug to route around; it is the id.
* trailing blocks — ``test_null_display_for_field (admin_utils.tests.UtilsTests) ... FAIL``
  then later ``FAIL: test_null_display_for_field (admin_utils.tests.UtilsTests)``.
  Note what upstream does with the second form: ``line.split()[1]`` yields the
  *bare* method name, which matches no Verified id. The ``FAIL:``/``ERROR:``
  branches are near-inert on this frame; the ``... FAIL`` / ``... ERROR``
  suffix branches carry the verdict. Their one live use is subtests, where the
  suffix line never appears:
  ``ERROR: test_json_display_for_field (admin_utils.tests.UtilsTests) [<object object at 0x7ffab7f3aae0>] (value={('a', 'b'): 'c'})``.
* interleaving — Django's own output can land mid-line, which is what the
  ``prev_test`` machinery and the three trailing regexes exist for. Captured
  instance of the general failure, from django__django-12308::

      test_json_display_for_field (admin_utils.tests.UtilsTests) ... test_label_for_field (admin_utils.tests.UtilsTests) ... ok

  Upstream keys that whole string as one PASSED "test"; the real
  ``test_label_for_field`` id is never recorded. Reproduced faithfully.

``parse_log_sympy`` operates on a different runner entirely: sympy's ``bin/test
-C --verbose`` prints bare function names (``test_point3D``) with a trailing
``ok`` / ``F`` / ``E``, plus underscore banners naming failures as
``<path>.py:<test>``. Both key spaces are live in Verified: sympy ids are bare
function names (3993 of 3993 checked, none containing ``::`` or a space).
"""

from __future__ import annotations

import re
from typing import Callable, Literal, NamedTuple

TestStatus = Literal["PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL"]
"""The status vocabulary. Upstream is an ``enum.Enum`` in
``swebench/harness/constants/__init__.py``; a Literal keeps this module pure
while typing the same five strings."""

TEST_STATUS_VALUES: tuple[TestStatus, ...] = (
    "FAILED",
    "PASSED",
    "SKIPPED",
    "ERROR",
    "XFAIL",
)
"""Declaration order of upstream's ``TestStatus`` enum, preserved verbatim.
Several parsers iterate it inside ``any(line.startswith(...))``; order is not
semantically load-bearing there, but a reordering here would be an undetectable
divergence from the file we vendored, so it is fixed."""

Parser = Callable[[str], dict[str, str]]


# --------------------------------------------------------------- upstream code
# Everything from here to MAP_REPO_TO_PARSER_PY is a transcription of
# swebench/harness/log_parsers/python.py @ ca6e4e0d. Comments marked "upstream:"
# are upstream's own words, kept because they carry the reason a line exists.


_SKIP_SUMMARY_COUNT = re.compile(r"^\[\d+\]$")


def _is_skip_summary(status: str, name: str) -> bool:
    """True for pytest's ``SKIPPED [N] path:line: reason`` summary, where [N] is not a test.

    upstream: Deliberately scoped to SKIPPED. Wrapped progress lines produce a
    similar ``PASSED [100%]`` artifact, but PASS_TO_PASS for
    pytest-dev__pytest-5262 and -7521 literally expects ``[100%]``, so dropping
    it fails those instances.
    """
    return status == "SKIPPED" and bool(_SKIP_SUMMARY_COUNT.match(name))


def parse_log_pytest(log: str) -> dict[str, str]:
    """Parser for test logs generated with PyTest framework.

    Serves pytest-dev/pytest, pydata/xarray and pallets/flask on our frame: the
    repos whose ``-rA`` summary needs no post-processing of the id at all.
    """
    test_status_map: dict[str, str] = {}
    for line in log.split("\n"):
        if any([line.startswith(x) for x in TEST_STATUS_VALUES]):
            # upstream: Additional parsing for FAILED status
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            if _is_skip_summary(test_case[0], test_case[1]):
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


def parse_log_pytest_options(log: str) -> dict[str, str]:
    """Parser for test logs generated with PyTest framework with options.

    Exists because psf/requests and pylint-dev/pylint parametrize over
    filesystem paths, so the id printed at runtime embeds the container's
    tmpdir while the id recorded in the dataset embeds only the basename. The
    normalization below is what makes ``[/testbed/tests/test_utils.py]`` and
    ``[/test_utils.py]`` the same test.
    """
    option_pattern = re.compile(r"(.*?)\[(.*)\]")
    test_status_map: dict[str, str] = {}
    for line in log.split("\n"):
        if any([line.startswith(x) for x in TEST_STATUS_VALUES]):
            # upstream: Additional parsing for FAILED status
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            if _is_skip_summary(test_case[0], test_case[1]):
                continue
            has_option = option_pattern.search(test_case[1])
            if has_option:
                main, option = has_option.groups()
                if (
                    option.startswith("/")
                    and not option.startswith("//")
                    and "*" not in option
                ):
                    option = "/" + option.split("/")[-1]
                test_name = f"{main}[{option}]"
            else:
                test_name = test_case[1]
            test_status_map[test_name] = test_case[0]
    return test_status_map


def parse_log_django(log: str) -> dict[str, str]:
    """Parser for test logs generated with Django tester framework.

    46% of SWE-bench Verified routes through this function. Its shape is
    dictated by stdlib unittest putting a test's docstring on the line that
    carries the status, and by Django's own logging occasionally landing inside
    that line; see the form catalogue in the module docstring. Transcribed
    literally, brittle regexes and single-instance special cases included,
    because each one is load-bearing for some instance's ids.
    """
    test_status_map: dict[str, str] = {}
    lines = log.split("\n")

    prev_test = None
    for line in lines:
        line = line.strip()

        # upstream: This isn't ideal but the test output spans multiple lines
        if "--version is equivalent to version" in line:
            test_status_map["--version is equivalent to version"] = "PASSED"

        # upstream: Log it in case of error
        if " ... " in line:
            prev_test = line.split(" ... ")[0]

        pass_suffixes = (" ... ok", " ... OK", " ...  OK")
        for suffix in pass_suffixes:
            if line.endswith(suffix):
                # upstream TODO: Temporary, exclusive fix for django__django-7188
                # The proper fix should involve somehow getting the test results to
                # print on a separate line, rather than the same line
                if line.strip().startswith(
                    "Applying sites.0002_alter_domain_unique...test_no_migrations"
                ):
                    line = line.split("...", 1)[-1].strip()
                test = line.rsplit(suffix, 1)[0]
                test_status_map[test] = "PASSED"
                break
        if " ... skipped" in line:
            test = line.split(" ... skipped")[0]
            test_status_map[test] = "SKIPPED"
        if line.endswith(" ... FAIL"):
            test = line.split(" ... FAIL")[0]
            test_status_map[test] = "FAILED"
        if line.startswith("FAIL:"):
            test = line.split()[1].strip()
            test_status_map[test] = "FAILED"
        if line.endswith(" ... ERROR"):
            test = line.split(" ... ERROR")[0]
            test_status_map[test] = "ERROR"
        if line.startswith("ERROR:"):
            test = line.split()[1].strip()
            test_status_map[test] = "ERROR"

        if line.lstrip().startswith("ok") and prev_test is not None:
            # upstream: It means the test passed, but there's some additional output
            # (including new lines) between "..." and "ok" message
            test = prev_test
            test_status_map[test] = "PASSED"

    # upstream TODO: This is very brittle, we should do better
    # There's a bug in the django logger, such that sometimes a test output near the end
    # gets interrupted by a particular long multiline print statement.
    # We have observed this in one of 3 forms:
    # - "{test_name} ... Testing against Django installed in {*} silenced.\nok"
    # - "{test_name} ... Internal Server Error: \/(.*)\/\nok"
    # - "{test_name} ... System check identified no issues (0 silenced).\nok"
    patterns = [
        r"^(.*?)\s\.\.\.\sTesting\ against\ Django\ installed\ in\ ((?s:.*?))\ silenced\)\.\nok$",
        r"^(.*?)\s\.\.\.\sInternal\ Server\ Error:\ \/(.*)\/\nok$",
        r"^(.*?)\s\.\.\.\sSystem check identified no issues \(0 silenced\)\nok$",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, log, re.MULTILINE):
            test_name = match.group(1)
            test_status_map[test_name] = "PASSED"
    return test_status_map


def parse_log_pytest_v2(log: str) -> dict[str, str]:
    """Parser for test logs generated with PyTest framework (Later Version).

    Serves astropy, scikit-learn and sphinx. Two differences from
    ``parse_log_pytest`` earn it a separate body: these repos' images emit ANSI
    colour (``\\x1b[32mPASSED\\x1b[0m ...``), which is stripped here, and older
    pytest printed the status *after* the id, which the ``elif`` branch catches.
    See the module docstring for the xfail-id hazard this body carries.
    """
    test_status_map: dict[str, str] = {}
    escapes = "".join([chr(char) for char in range(1, 32)])
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line)
        translator = str.maketrans("", "", escapes)
        line = line.translate(translator)
        if any([line.startswith(x) for x in TEST_STATUS_VALUES]):
            if line.startswith("FAILED"):
                # upstream: drop the trailing " - <assertion message>" so it can't
                # enter the id
                line = line.split(" - ", 1)[0]
            test_case = line.split()
            if len(test_case) >= 2 and not _is_skip_summary(test_case[0], test_case[1]):
                test_status_map[" ".join(test_case[1:])] = test_case[0]
        # upstream: Support older pytest versions by checking if the line ends with
        # the test status
        elif any([line.endswith(x) for x in TEST_STATUS_VALUES]):
            test_case = line.split()
            if len(test_case) >= 2:
                test_status_map[" ".join(test_case[:-1])] = test_case[-1]
    return test_status_map


def parse_log_seaborn(log: str) -> dict[str, str]:
    """Parser for test logs generated with seaborn testing framework.

    Note what it does *not* do: an ``XFAIL`` line matches none of its three
    branches, so xfailing tests are simply absent from the map — where upstream
    grading counts absence as failure. Two Verified instances ride on this.
    """
    test_status_map: dict[str, str] = {}
    for line in log.split("\n"):
        parts = line.split()
        if len(parts) < 2:
            continue
        if line.startswith("FAILED"):
            test_case = parts[1]
            test_status_map[test_case] = "FAILED"
        elif " PASSED " in line:
            if parts[1] == "PASSED":
                test_case = parts[0]
                test_status_map[test_case] = "PASSED"
        elif line.startswith("PASSED"):
            test_case = parts[1]
            test_status_map[test_case] = "PASSED"
    return test_status_map


def parse_log_sympy(log: str) -> dict[str, str]:
    """Parser for test logs generated with Sympy framework.

    sympy does not use pytest node ids: its runner prints bare function names
    with a trailing ``ok`` / ``F`` / ``E``, and names failures a second time in
    an underscore banner as ``<path>.py:<test>``. The banner sweep runs first,
    over the *whole* log rather than line by line, so a failure keyed both ways
    ends up in the map twice under different keys — which is what the dataset's
    two id shapes require.
    """
    test_status_map: dict[str, str] = {}
    pattern = r"(_*) (.*)\.py:(.*) (_*)"
    matches = re.findall(pattern, log)
    for match in matches:
        test_case = f"{match[1]}.py:{match[2]}"
        test_status_map[test_case] = "FAILED"
    for line in log.split("\n"):
        line = line.strip()
        if line.startswith("test_"):
            if line.endswith(" E"):
                test = line.split()[0]
                test_status_map[test] = "ERROR"
            if line.endswith(" F"):
                test = line.split()[0]
                test_status_map[test] = "FAILED"
            if line.endswith(" ok"):
                test = line.split()[0]
                test_status_map[test] = "PASSED"
    return test_status_map


def parse_log_matplotlib(log: str) -> dict[str, str]:
    """Parser for test logs generated with PyTest framework.

    ``parse_log_pytest`` plus two literal substitutions: matplotlib
    parametrizes over ``MouseButton`` enum members, whose repr changed between
    the version that generated the dataset's ids and the version in the image.
    """
    test_status_map: dict[str, str] = {}
    for line in log.split("\n"):
        line = line.replace("MouseButton.LEFT", "1")
        line = line.replace("MouseButton.RIGHT", "3")
        if any([line.startswith(x) for x in TEST_STATUS_VALUES]):
            # upstream: Additional parsing for FAILED status
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            if _is_skip_summary(test_case[0], test_case[1]):
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


# Upstream aliases. They are not decoration: PARSER_REGISTRY is keyed by these
# names, so ``log_parser == "parse_log_xarray"`` must resolve, even though the
# body is parse_log_pytest's.
parse_log_astroid = parse_log_pytest
parse_log_flask = parse_log_pytest
parse_log_marshmallow = parse_log_pytest
parse_log_pvlib = parse_log_pytest
parse_log_pyvista = parse_log_pytest
parse_log_sqlfluff = parse_log_pytest
parse_log_xarray = parse_log_pytest

parse_log_pydicom = parse_log_pytest_options
parse_log_requests = parse_log_pytest_options
parse_log_pylint = parse_log_pytest_options

parse_log_astropy = parse_log_pytest_v2
parse_log_scikit = parse_log_pytest_v2
parse_log_sphinx = parse_log_pytest_v2


MAP_REPO_TO_PARSER_PY: dict[str, Parser] = {
    "astropy/astropy": parse_log_astropy,
    "django/django": parse_log_django,
    "marshmallow-code/marshmallow": parse_log_marshmallow,
    "matplotlib/matplotlib": parse_log_matplotlib,
    "mwaskom/seaborn": parse_log_seaborn,
    "pallets/flask": parse_log_flask,
    "psf/requests": parse_log_requests,
    "pvlib/pvlib-python": parse_log_pvlib,
    "pydata/xarray": parse_log_xarray,
    "pydicom/pydicom": parse_log_pydicom,
    "pylint-dev/astroid": parse_log_astroid,
    "pylint-dev/pylint": parse_log_pylint,
    "pytest-dev/pytest": parse_log_pytest,
    "pyvista/pyvista": parse_log_pyvista,
    "scikit-learn/scikit-learn": parse_log_scikit,
    "sqlfluff/sqlfluff": parse_log_sqlfluff,
    "sphinx-doc/sphinx": parse_log_sphinx,
    "sympy/sympy": parse_log_sympy,
}
"""Upstream's repo -> parser table, complete and verbatim (18 entries; six of
them are for repos absent from Verified and are kept so this table can be
diffed against upstream by eye)."""


PARSER_REGISTRY: dict[str, Parser] = {
    "parse_log_pytest": parse_log_pytest,
    "parse_log_pytest_options": parse_log_pytest_options,
    "parse_log_django": parse_log_django,
    "parse_log_pytest_v2": parse_log_pytest_v2,
    "parse_log_seaborn": parse_log_seaborn,
    "parse_log_sympy": parse_log_sympy,
    "parse_log_matplotlib": parse_log_matplotlib,
    "parse_log_astroid": parse_log_astroid,
    "parse_log_flask": parse_log_flask,
    "parse_log_marshmallow": parse_log_marshmallow,
    "parse_log_pvlib": parse_log_pvlib,
    "parse_log_pyvista": parse_log_pyvista,
    "parse_log_sqlfluff": parse_log_sqlfluff,
    "parse_log_xarray": parse_log_xarray,
    "parse_log_pydicom": parse_log_pydicom,
    "parse_log_requests": parse_log_requests,
    "parse_log_pylint": parse_log_pylint,
    "parse_log_astropy": parse_log_astropy,
    "parse_log_scikit": parse_log_scikit,
    "parse_log_sphinx": parse_log_sphinx,
}
"""The Python section of upstream's ``PARSER_REGISTRY``, keys verbatim, so a
``TestSpec.log_parser`` string lifted from the harness resolves here unchanged.
Upstream's registry also carries JavaScript / C / Go / Java / PHP / Ruby / Rust
entries; those are omitted because SWE-bench Verified is Python-only and
vendoring parsers we cannot exercise would be untested code pretending to be
provenance."""


# ---------------------------------------------------------------- our addition


class UnknownParserError(KeyError):
    """Raised when a log_parser name is not in PARSER_REGISTRY.

    A distinct type because the silent alternative is worse than a crash: a
    ``.get(name, parse_log_pytest)`` fallback would parse a Django log with a
    pytest parser and return ``{}``, which grading reads as "every test failed".
    """


def parse(parser: str, log: str) -> dict[str, str]:
    """Dispatch by upstream parser name, raising a clear error on an unknown key.

    Upstream reaches the parser by indexing ``PARSER_REGISTRY`` at the call
    site; we route through one function so the failure mode is a named
    exception listing what *is* available, rather than a bare KeyError from
    inside the grading path.
    """
    try:
        fn = PARSER_REGISTRY[parser]
    except KeyError:
        known = ", ".join(sorted(PARSER_REGISTRY))
        raise UnknownParserError(
            f"unknown log parser {parser!r}. This module vendors only the "
            f"Python parsers, which is all SWE-bench Verified needs; a "
            f"non-Python key means the caller is on a different benchmark. "
            f"Known keys: {known}"
        ) from None
    return fn(log)


VERIFIED_REPO_PAIRS: dict[str, int] = {
    "astropy/astropy": 6,
    "django/django": 9,
    "matplotlib/matplotlib": 6,
    "mwaskom/seaborn": 1,
    "pallets/flask": 1,
    "psf/requests": 7,
    "pydata/xarray": 4,
    "pylint-dev/pylint": 5,
    "pytest-dev/pytest": 10,
    "scikit-learn/scikit-learn": 4,
    "sphinx-doc/sphinx": 15,
    "sympy/sympy": 12,
}
"""Repository -> number of distinct ``version`` values present in SWE-bench
Verified (sums to the 80 (repo, version) pairs). Recorded so a test can assert
that ``MAP_REPO_TO_PARSER_PY`` covers the whole frame and notice immediately if
a future dataset revision introduces a repo we have no parser for."""

VERIFIED_INSTANCES: dict[str, int] = {
    "astropy/astropy": 22,
    "django/django": 231,
    "matplotlib/matplotlib": 34,
    "mwaskom/seaborn": 2,
    "pallets/flask": 1,
    "psf/requests": 8,
    "pydata/xarray": 22,
    "pylint-dev/pylint": 10,
    "pytest-dev/pytest": 19,
    "scikit-learn/scikit-learn": 32,
    "sphinx-doc/sphinx": 44,
    "sympy/sympy": 75,
}
"""Instance counts behind those pairs (sums to 500). The weights that decide
which transcription errors would be catastrophic and which merely bad."""


# -------------------------------------------------------------------- fixtures


class Fixture(NamedTuple):
    """One golden excerpt: real log text in, asserted status map out."""

    parser: str
    source: str
    """Where the text came from. ``captured:`` means it was lifted verbatim
    from a real SWE-bench ``test_output.txt``; ``hand-built:`` means it was
    written to exercise a branch no captured log we could obtain reaches."""
    log: str
    expected: dict[str, str]
    note: str = ""


FIXTURES: dict[str, Fixture] = {
    "pytest": Fixture(
        parser="parse_log_pytest",
        source=(
            "captured: pytest-dev__pytest-5262 (Verified, version 4.5), "
            "test_output.txt from a public `swebench` run_evaluation "
            "(github.com/Vexp-ai/vexp-swe-bench, run vexp-swebench-1774184993333); "
            "lines 1265, 1428, 1518, 1627 and 1631, each verbatim"
        ),
        log=(
            "PASSED\n"
            "PASSED                                                                   [100%]\n"
            "SKIPPED [1] testing/test_capture.py:894: python2 has no buffer\n"
            "PASSED testing/test_capture.py::test_dontreadfrominput_has_encoding\n"
            "PASSED testing/test_capture.py::test_capture_with_live_logging[capsys]\n"
        ),
        expected={
            "[100%]": "PASSED",
            "testing/test_capture.py::test_dontreadfrominput_has_encoding": "PASSED",
            "testing/test_capture.py::test_capture_with_live_logging[capsys]": "PASSED",
        },
        note=(
            "Three upstream behaviours at once: a bare status line is dropped "
            "(len <= 1); the wrapped progress artifact '[100%]' is kept, and "
            "must be, because this instance's PASS_TO_PASS contains the literal "
            "string '[100%]'; the 'SKIPPED [1] ...' summary is dropped by "
            "_is_skip_summary."
        ),
    ),
    "pytest_options": Fixture(
        parser="parse_log_pytest_options",
        source=(
            "captured: lines 3768-3771 of psf__requests-6070's test_output.txt "
            "(github.com/pouyafath/BenchmarkLLMAgent, run "
            "secondpaper_custom_gold_probe) plus line 448 of "
            "pylint-dev__pylint-8898 (Verified, version 3.0) and lines 1229/1252 "
            "of psf__requests-1142 (Verified, version 1.1), all verbatim"
        ),
        log=(
            "PASSED tests/test_utils.py::TestExtractZippedPaths::test_unzipped_paths_unchanged[/]\n"
            "PASSED tests/test_utils.py::TestExtractZippedPaths::test_unzipped_paths_unchanged[/testbed/tests/test_utils.py]\n"
            "PASSED tests/test_utils.py::TestExtractZippedPaths::test_unzipped_paths_unchanged[/opt/miniconda3/envs/testbed/lib/python3.10/site-packages/pytest/__init__.py]\n"
            "PASSED tests/test_utils.py::TestExtractZippedPaths::test_unzipped_paths_unchanged[/etc/invalid/location]\n"
            "PASSED tests/config/test_config.py::test_csv_regex_comma_in_quantifier[foo, bar-expected2]\n"
            "PASSED test_requests.py::RequestsTestCase::test_basic_building\n"
            "FAILED test_requests.py::RequestsTestCase::test_status_raising - TypeError: _...\n"
        ),
        expected={
            "tests/test_utils.py::TestExtractZippedPaths::test_unzipped_paths_unchanged[/]": "PASSED",
            "tests/test_utils.py::TestExtractZippedPaths::test_unzipped_paths_unchanged[/test_utils.py]": "PASSED",
            "tests/test_utils.py::TestExtractZippedPaths::test_unzipped_paths_unchanged[/__init__.py]": "PASSED",
            "tests/test_utils.py::TestExtractZippedPaths::test_unzipped_paths_unchanged[/location]": "PASSED",
            "tests/config/test_config.py::test_csv_regex_comma_in_quantifier[foo,": "PASSED",
            "test_requests.py::RequestsTestCase::test_basic_building": "PASSED",
            "test_requests.py::RequestsTestCase::test_status_raising": "FAILED",
        },
        note=(
            "The first four keys are the whole reason this parser exists: the "
            "container's absolute paths collapse to basenames, and the four "
            "results match psf__requests-6028's PASS_TO_PASS entries "
            "character for character (checked against data/verified.jsonl). "
            "psf__requests-6070 is a SWE-bench (full) instance, not a Verified "
            "one — it was used because no Verified requests log exercising this "
            "branch could be obtained; the runner, repo and id shape are "
            "identical. The pylint line shows upstream truncating an id at its "
            "first space, harmless here only because no Verified id contains "
            "one."
        ),
    ),
    "pytest_v2": Fixture(
        parser="parse_log_pytest_v2",
        source=(
            "captured: lines 1325-1328 of astropy__astropy-14365 (Verified, "
            "version 5.1) test_output.txt, verbatim including ANSI SGR bytes"
        ),
        log=(
            "\x1b[32mPASSED\x1b[0m astropy/io/ascii/tests/test_qdp.py::\x1b[1mtest_roundtrip_example_comma\x1b[0m\n"
            "\x1b[32mPASSED\x1b[0m astropy/io/ascii/tests/test_qdp.py::\x1b[1mtest_read_write_simple\x1b[0m\n"
        ),
        expected={
            "astropy/io/ascii/tests/test_qdp.py::test_roundtrip_example_comma": "PASSED",
            "astropy/io/ascii/tests/test_qdp.py::test_read_write_simple": "PASSED",
        },
        note=(
            "Colour is why this parser is not parse_log_pytest: the substitution "
            "removes '[32m'/'[1m'/'[0m' and the translate() drops the ESC bytes "
            "themselves. astropy's images emit colour; scikit-learn's and "
            "sphinx's do not, and take the same path harmlessly."
        ),
    ),
    "django_ok_and_docstring": Fixture(
        parser="parse_log_django",
        source=(
            "captured: lines 50, 57-60 of django__django-15851 (Verified, "
            "version 4.2), eval log from github.com/NL2Code/CodeR "
            "(logs/django__django-15851.20240604_CodeR.eval.log), verbatim"
        ),
        log=(
            "test_accent (dbshell.test_postgresql.PostgreSqlDbshellCommandTestCase) ... ok\n"
            "test_service (dbshell.test_postgresql.PostgreSqlDbshellCommandTestCase) ... ok\n"
            "test_sigint_handler (dbshell.test_postgresql.PostgreSqlDbshellCommandTestCase)\n"
            "SIGINT is ignored in Python and passed to psql to abort queries. ... skipped 'Requires a PostgreSQL connection'\n"
            "test_ssl_certificate (dbshell.test_postgresql.PostgreSqlDbshellCommandTestCase) ... ok\n"
        ),
        expected={
            "test_accent (dbshell.test_postgresql.PostgreSqlDbshellCommandTestCase)": "PASSED",
            "test_service (dbshell.test_postgresql.PostgreSqlDbshellCommandTestCase)": "PASSED",
            "SIGINT is ignored in Python and passed to psql to abort queries.": "SKIPPED",
            "test_ssl_certificate (dbshell.test_postgresql.PostgreSqlDbshellCommandTestCase)": "PASSED",
        },
        note=(
            "The docstring case is not an edge case: 'test_sigint_handler (...)' "
            "never appears as a key, and the dataset does not expect it to — the "
            "docstring line is the id. This instance's PASS_TO_PASS carries the "
            "'test_accent (...)' family verbatim."
        ),
    ),
    "django_fail_error_and_interleave": Fixture(
        parser="parse_log_django",
        source=(
            "captured: lines 73, 78, 87 and 109 of django__django-12308 "
            "(Verified, version 3.1) and lines 161/197 of django__django-13551 "
            "(Verified, version 3.2), both eval logs from github.com/NL2Code/CodeR, "
            "each line verbatim"
        ),
        log=(
            "test_json_display_for_field (admin_utils.tests.UtilsTests) ... test_label_for_field (admin_utils.tests.UtilsTests) ... ok\n"
            "test_null_display_for_field (admin_utils.tests.UtilsTests) ... FAIL\n"
            "test_10265 (auth_tests.test_tokens.TokenGeneratorTest) ... ERROR\n"
            "ERROR: test_json_display_for_field (admin_utils.tests.UtilsTests) [<object object at 0x7ffab7f3aae0>] (value={('a', 'b'): 'c'})\n"
            "FAIL: test_null_display_for_field (admin_utils.tests.UtilsTests)\n"
            "ERROR: test_10265 (auth_tests.test_tokens.TokenGeneratorTest)\n"
        ),
        expected={
            "test_json_display_for_field (admin_utils.tests.UtilsTests) ... test_label_for_field (admin_utils.tests.UtilsTests)": "PASSED",
            "test_null_display_for_field (admin_utils.tests.UtilsTests)": "FAILED",
            "test_10265 (auth_tests.test_tokens.TokenGeneratorTest)": "ERROR",
            "test_json_display_for_field": "ERROR",
            "test_null_display_for_field": "FAILED",
            "test_10265": "ERROR",
        },
        note=(
            "Three upstream facts a test should pin. (1) Interleaved output "
            "makes one composite key and silently loses test_label_for_field. "
            "(2) The 'FAIL:'/'ERROR:' branches key the bare method name, which "
            "matches no Verified django id — the '... FAIL' / '... ERROR' "
            "suffix lines carry the verdict; the bare keys are harmless "
            "residue except for subtests, where only the block form appears. "
            "(3) A subtest's parameter tail ('[<object ...>] (value=...)') is "
            "discarded by split()[1]."
        ),
    ),
    "django_python311_ids": Fixture(
        parser="parse_log_django",
        source=(
            "captured: lines 396-401 of django__django-17084 (Verified, version "
            "5.0) test_output.txt from github.com/Vexp-ai/vexp-swe-bench, verbatim"
        ),
        log=(
            "test_aggregation_order_by_not_selected_annotation_values (aggregation.tests.AggregateTestCase.test_aggregation_order_by_not_selected_annotation_values) ... ok\n"
            "test_aggregation_random_ordering (aggregation.tests.AggregateTestCase.test_aggregation_random_ordering)\n"
            "Random() is not included in the GROUP BY when used for ordering. ... ok\n"
            "test_aggregation_subquery_annotation (aggregation.tests.AggregateTestCase.test_aggregation_subquery_annotation)\n"
            "Subquery annotations are excluded from the GROUP BY if they are ... ok\n"
            "test_aggregation_subquery_annotation_exists (aggregation.tests.AggregateTestCase.test_aggregation_subquery_annotation_exists) ... ok\n"
        ),
        expected={
            "test_aggregation_order_by_not_selected_annotation_values (aggregation.tests.AggregateTestCase.test_aggregation_order_by_not_selected_annotation_values)": "PASSED",
            "Random() is not included in the GROUP BY when used for ordering.": "PASSED",
            "Subquery annotations are excluded from the GROUP BY if they are": "PASSED",
            "test_aggregation_subquery_annotation_exists (aggregation.tests.AggregateTestCase.test_aggregation_subquery_annotation_exists)": "PASSED",
        },
        note=(
            "Django 5.0 runs on Python 3.11, whose unittest repeats the method "
            "name inside the parentheses. All 827 non-docstring django 5.0 ids "
            "in Verified are this shape and none of the other eight versions' "
            "are; a parser that normalised ids would break exactly here. Note "
            "also the truncated docstring key — unittest prints only the "
            "docstring's first line, and the dataset stores that truncation."
        ),
    ),
    "sympy": Fixture(
        parser="parse_log_sympy",
        source=(
            "captured: lines 336-343 and 346-347 of sympy__sympy-13974 "
            "(Verified, version 1.1) test_output.txt from "
            "github.com/Vexp-ai/vexp-swe-bench, verbatim including trailing "
            "alignment spaces"
        ),
        log=(
            "sympy/physics/quantum/tests/test_tensorproduct.py[7] \n"
            "test_tensor_product_dagger E\n"
            "test_tensor_product_abstract ok\n"
            "test_tensor_product_expand ok\n"
            "test_tensor_product_commutator ok\n"
            "test_tensor_product_simp F\n"
            "test_issue_5923 ok\n"
            "test_eval_trace ok                                                        [FAIL]\n"
            "________________________________________________________________________________\n"
            "_ sympy/physics/quantum/tests/test_tensorproduct.py:test_tensor_product_dagger _\n"
        ),
        expected={
            "sympy/physics/quantum/tests/test_tensorproduct.py:test_tensor_product_dagger": "FAILED",
            "test_tensor_product_dagger": "ERROR",
            "test_tensor_product_abstract": "PASSED",
            "test_tensor_product_expand": "PASSED",
            "test_tensor_product_commutator": "PASSED",
            "test_tensor_product_simp": "FAILED",
            "test_issue_5923": "PASSED",
        },
        note=(
            "Bare function names, exactly as sympy's PASS_TO_PASS records them "
            "(this instance expects 'test_tensor_product_abstract' and "
            "'test_tensor_product_expand'). Two upstream properties to pin: the "
            "same failure is keyed twice, once by banner path and once by name; "
            "and 'test_eval_trace' is MISSING from the map, because the suite's "
            "'[FAIL]' verdict is appended to that last line so it no longer "
            "ends in ' ok'. Upstream loses a passing test there. Reproduced, "
            "not fixed."
        ),
    ),
    "seaborn": Fixture(
        parser="parse_log_seaborn",
        source=(
            "captured: lines 439-440 and 692 of mwaskom__seaborn-3187 "
            "(Verified, version 0.12) test_output.txt from "
            "github.com/Vexp-ai/vexp-swe-bench, verbatim"
        ),
        log=(
            "PASSED tests/_core/test_plot.py::TestInit::test_empty\n"
            "PASSED tests/_core/test_plot.py::TestInit::test_data_only\n"
            "XFAIL tests/_core/test_plot.py::TestScaling::test_log_scale_name - Custom log scale needs log name for consistency\n"
        ),
        expected={
            "tests/_core/test_plot.py::TestInit::test_empty": "PASSED",
            "tests/_core/test_plot.py::TestInit::test_data_only": "PASSED",
        },
        note=(
            "The XFAIL line produces no key at all. Upstream grading counts "
            "XFAIL as a pass when a parser reports it; this parser cannot, so "
            "an xfailing PASS_TO_PASS test on seaborn reads as failed. Two "
            "Verified instances are exposed to this."
        ),
    ),
    "matplotlib": Fixture(
        parser="parse_log_matplotlib",
        source=(
            "captured: lines 505, 1281 and 1283 of matplotlib__matplotlib-24627 "
            "(Verified, version 3.6) test_output.txt from "
            "github.com/Vexp-ai/vexp-swe-bench, verbatim"
        ),
        log=(
            "PASSED lib/matplotlib/tests/test_axes.py::test_invisible_axes[png]\n"
            "SKIPPED [10] lib/matplotlib/testing/compare.py:285: Don't know how to convert .svg files to png\n"
            "FAILED lib/matplotlib/tests/test_axes.py::test_cla_clears_children_axes_and_fig\n"
        ),
        expected={
            "lib/matplotlib/tests/test_axes.py::test_invisible_axes[png]": "PASSED",
            "lib/matplotlib/tests/test_axes.py::test_cla_clears_children_axes_and_fig": "FAILED",
        },
        note=(
            "UNVERIFIED-UNTIL-REAL-LOG for the MouseButton branch only: no "
            "captured matplotlib log we could obtain contains a "
            "'MouseButton.LEFT'/'MouseButton.RIGHT' id, so the two substitutions "
            "that distinguish this parser from parse_log_pytest are exercised by "
            "the hand-built fixture 'matplotlib_mousebutton' below, not by this "
            "one. The SKIPPED summary line is real and is dropped, as upstream "
            "intends."
        ),
    ),
    "matplotlib_mousebutton": Fixture(
        parser="parse_log_matplotlib",
        source=(
            "hand-built: UNVERIFIED-UNTIL-REAL-LOG. Written from upstream's "
            "substitution and from the id shape in matplotlib's PASS_TO_PASS; "
            "NOT captured from any run. Replace with a real captured line the "
            "first time a sweep produces one."
        ),
        log="PASSED lib/matplotlib/tests/test_backend_bases.py::test_toolbar_zoompan[MouseButton.LEFT]\n",
        expected={
            "lib/matplotlib/tests/test_backend_bases.py::test_toolbar_zoompan[1]": "PASSED"
        },
        note=(
            "Shows the intent of the substitution: the runtime repr collapses "
            "to the integer the dataset id was recorded with."
        ),
    ),
    "django_interrupted_ok": Fixture(
        parser="parse_log_django",
        source=(
            "hand-built: UNVERIFIED-UNTIL-REAL-LOG. Assembled from the three "
            "log shapes upstream names in its own comment above the trailing "
            "regexes, plus the prev_test path; NOT captured from any run. No "
            "obtainable log contained an interrupted 'ok'."
        ),
        log=(
            "test_check (check_framework.tests.SystemCheckFrameworkTests) ... System check identified no issues (0 silenced)\n"
            "ok\n"
        ),
        expected={
            "test_check (check_framework.tests.SystemCheckFrameworkTests)": "PASSED"
        },
        note=(
            "Two independent upstream mechanisms both fire on this input and "
            "agree: the prev_test carry-over (line starts with 'ok') and the "
            "third trailing regex. Kept because it is the only documentation "
            "of what those branches are for; demote to captured status as soon "
            "as a real instance produces one."
        ),
    ),
    "pytest_v2_trailing_status": Fixture(
        parser="parse_log_pytest_v2",
        source=(
            "hand-built: UNVERIFIED-UNTIL-REAL-LOG. Written to exercise the "
            "'older pytest prints the status last' elif branch; NOT captured. "
            "Every astropy / scikit-learn / sphinx log obtained used '-rA', "
            "whose summary puts the status first."
        ),
        log="sklearn/linear_model/tests/test_logistic.py::test_predict_2_classes PASSED\n",
        expected={
            "sklearn/linear_model/tests/test_logistic.py::test_predict_2_classes": "PASSED"
        },
        note="",
    ),
}
"""Golden fixtures, one per parser reachable from our 80 pairs plus three that
cover branches no obtainable log reaches.

Read the ``source`` field before trusting an entry. ``captured:`` entries are
verbatim lines from real SWE-bench ``test_output.txt`` files produced by the
official harness in the official images; the ``expected`` maps are this
module's actual output on them, and several encode upstream *losing* a test
(sympy's ``test_eval_trace``, django's ``test_label_for_field``, seaborn's
xfails) — that is upstream's behaviour, and a test that "fixes" those
expectations has broken agreement with the grader. ``hand-built:`` entries
carry the literal marker UNVERIFIED-UNTIL-REAL-LOG and must not be cited as
evidence about a real runner.
"""
