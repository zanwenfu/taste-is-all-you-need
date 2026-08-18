"""Frozen mirror of the official SWE-bench harness's per-(repo, version) run specs.

Every SWE-bench instance is graded by running a *repo-specific* test command over the
files touched by the gold ``test_patch``, then parsing that runner's stdout with a
*repo-specific* log parser. Neither is uniform: django shells out to
``./tests/runtests.py`` and demands dotted module paths, sympy shells out to
``bin/test``, sphinx goes through ``tox``, and the pytest-family repos disagree about
which of three parsers reads their output. Upstream keeps this in
``MAP_REPO_VERSION_TO_SPECS`` and ``MAP_REPO_TO_PARSER``; this module is a verbatim copy
of the slice of those tables that SWE-bench Verified actually exercises -- all 80
(repo, version) pairs, nothing else.

Why a copy rather than an import: the upstream package drags in docker, modal and the
datasets stack at import time, and -- see Provenance -- it has since deleted these tables
outright. Pinning them here is what keeps a measurement taken today comparable to one
taken a year from now, which is the entire point of an instrument.

What breaks if a row here is wrong: nothing raises. The wrong command runs, the parser
recognises none of the expected test names, every FAIL_TO_PASS test is recorded as "not
passing", and the instance is scored unresolved. A bad row does not surface as an error;
it surfaces as a *lower number*, applied to whichever arm happened to draw those
instances. That is precisely the shape of the experimental effect this instrument exists
to measure, and it is why `spec_for` raises instead of falling back to a plausible-looking
pytest default: a wrong answer here is far more expensive than a missing one.

Provenance
----------
Every value below was read from, and is traceable to:

    repo: https://github.com/SWE-bench/SWE-bench
    ref:  726c5461e2ef52d83cf1ea2107870a8bb3328d57   (tag v4.1.0)

    swebench/harness/constants/python.py    -- MAP_REPO_VERSION_TO_SPECS_PY
    swebench/harness/constants/__init__.py  -- NON_TEST_EXTS
    swebench/harness/log_parsers/python.py  -- MAP_REPO_TO_PARSER_PY
    swebench/harness/test_spec/python.py    -- get_test_directives (L230-261),
                                               make_eval_script_list_py (L405-425)

v4.1.0 is the newest upstream tag that still contains these tables. On ``main`` (checked
at 128cbd1a5759694874e6bd56624cb2fd6fb079e2, 2026-08-15) ``constants/python.py`` is gone
entirely and the environment specs have moved out of the package. Re-pin deliberately;
never bump the ref and assume the shape held.

The spec values were extracted by *executing* the upstream module and reading the
assembled dict, not by reading it top-down. That distinction is load-bearing:
``constants/python.py`` binds ``TEST_PYTEST`` twice -- line 2 as
``"pytest --no-header -rA --tb=no -p no:cacheprovider"``, then line 9 as ``"pytest -rA"``
-- and only the second binding is live by the time any ``SPECS_*`` dict is built. Any
transcription that trusts the first definition gets the test command wrong for eight of
the twelve repos here (astropy, matplotlib, flask, requests, xarray, pylint, pytest,
scikit-learn -- every row whose ``test_cmd`` is ``"pytest -rA"``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "LOG_PARSERS",
    "NON_TEST_EXTS",
    "SPECS",
    "RepoSpec",
    "UnknownRepoVersionError",
    "spec_for",
    "test_command",
    "test_directives",
]


class UnknownRepoVersionError(Exception):
    """Raised by `spec_for` for a (repo, version) pair this table does not cover.

    Deliberately *not* a KeyError subclass. Callers routinely wrap dict access in
    ``except KeyError`` and substitute a default; for this table that recovery path is
    the bug -- it is how an unknown pair quietly acquires a pytest command it does not
    understand. An unrelated exception type forces the miss to reach a human.
    """


# Verbatim from upstream constants/__init__.py. The missing leading dot on "csv" is an
# upstream quirk, not a typo on our side: the filter is `d.endswith(ext)`, so this entry
# matches any path ending in the bare letters "csv" rather than only the ".csv"
# extension. Reproduced exactly -- this filter decides which touched files become test
# directives, so a "corrected" copy would hand the runner a different set of files than
# the official harness does.
NON_TEST_EXTS: tuple[str, ...] = (
    ".json",
    ".png",
    "csv",
    ".txt",
    ".md",
    ".jpg",
    ".jpeg",
    ".pkl",
    ".yml",
    ".yaml",
    ".toml",
)

# Upstream keys parsers by repo, never by version, and reaches them through a layer of
# module-level aliases (`parse_log_requests = parse_log_pytest_options`). Those aliases
# are resolved here so the value names the parser that actually runs. Keeping the alias
# would hide the three real distinctions: requests/pylint need the *options* parser
# (which normalises `test[/some/path]` parametrisation ids), astropy/scikit-learn/sphinx
# need the *v2* parser (which strips ANSI escapes and accepts trailing-status lines),
# and only pytest-dev/pytest itself uses the plain one.
LOG_PARSERS: dict[str, str] = {
    "astropy/astropy": "pytest_v2",
    "django/django": "django",
    "matplotlib/matplotlib": "matplotlib",
    "mwaskom/seaborn": "seaborn",
    "pallets/flask": "pytest",
    "psf/requests": "pytest_options",
    "pydata/xarray": "pytest",
    "pylint-dev/pylint": "pytest_options",
    "pytest-dev/pytest": "pytest",
    "scikit-learn/scikit-learn": "pytest_v2",
    "sphinx-doc/sphinx": "pytest_v2",
    "sympy/sympy": "sympy",
}


@dataclass(frozen=True)
class RepoSpec:
    """One row of the harness's (repo, version) -> how-to-run-the-tests table.

    `install` and `python` are carried even though this process never builds an image:
    they narrow which upstream environment a result was produced under, so a run can be
    partially audited against the official Docker tags after the fact.

    They do not fully identify it. Upstream rows also carry `pre_install` (28 of these
    80 pairs), `packages` (49), `pip_packages` (62), `no_use_env` (4) and `nano_cpus`
    (3); those are dropped here because nothing in this process builds an image. Do not
    read a matching (install, python) pair as proof that two environments are the same.
    """

    test_cmd: str
    log_parser: str
    eval_commands: tuple[str, ...] = ()
    install: str = ""
    python: str = ""


# (repo, version) -> spec, covering all 80 pairs present in SWE-bench Verified.
# Grouped by repo; instance counts are this dataset's, for reviewer orientation.
SPECS: dict[tuple[str, str], RepoSpec] = {
    # --- astropy/astropy: 6 version(s), 22 instance(s), parser=pytest_v2 ---
    ("astropy/astropy", "1.3"): RepoSpec(
        test_cmd="pytest -rA -vv -o console_output_style=classic --tb=no",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test] --verbose",
        python="3.6",
    ),
    ("astropy/astropy", "3.1"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test] --verbose",
        python="3.9",
    ),
    ("astropy/astropy", "4.3"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test] --verbose",
        python="3.9",
    ),
    ("astropy/astropy", "5.0"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test] --verbose",
        python="3.9",
    ),
    ("astropy/astropy", "5.1"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test] --verbose",
        python="3.9",
    ),
    ("astropy/astropy", "5.2"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test] --verbose",
        python="3.9",
    ),
    # --- django/django: 9 version(s), 231 instance(s), parser=django ---
    ("django/django", "1.11"): RepoSpec(
        test_cmd="./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
        log_parser="django",
        eval_commands=(
            "export LANG=en_US.UTF-8",
            "export LC_ALL=en_US.UTF-8",
            "export PYTHONIOENCODING=utf8",
            "export LANGUAGE=en_US:en",
        ),
        install="python setup.py install",
        python="3.5",
    ),
    ("django/django", "2.2"): RepoSpec(
        test_cmd="./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
        log_parser="django",
        eval_commands=(
            "export LANG=en_US.UTF-8",
            "export LC_ALL=en_US.UTF-8",
            "export PYTHONIOENCODING=utf8",
            "export LANGUAGE=en_US:en",
        ),
        install="python setup.py install",
        python="3.5",
    ),
    ("django/django", "3.0"): RepoSpec(
        test_cmd="./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
        log_parser="django",
        eval_commands=(
            "sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen",
            "export LANG=en_US.UTF-8",
            "export LANGUAGE=en_US:en",
            "export LC_ALL=en_US.UTF-8",
        ),
        install="python -m pip install -e .",
        python="3.6",
    ),
    ("django/django", "3.1"): RepoSpec(
        test_cmd="./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
        log_parser="django",
        eval_commands=(
            "sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen",
            "export LANG=en_US.UTF-8",
            "export LANGUAGE=en_US:en",
            "export LC_ALL=en_US.UTF-8",
        ),
        install="python -m pip install -e .",
        python="3.6",
    ),
    ("django/django", "3.2"): RepoSpec(
        test_cmd="./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
        log_parser="django",
        eval_commands=(
            "sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen",
            "export LANG=en_US.UTF-8",
            "export LANGUAGE=en_US:en",
            "export LC_ALL=en_US.UTF-8",
        ),
        install="python -m pip install -e .",
        python="3.6",
    ),
    ("django/django", "4.0"): RepoSpec(
        test_cmd="./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
        log_parser="django",
        install="python -m pip install -e .",
        python="3.8",
    ),
    ("django/django", "4.1"): RepoSpec(
        test_cmd="./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
        log_parser="django",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("django/django", "4.2"): RepoSpec(
        test_cmd="./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
        log_parser="django",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("django/django", "5.0"): RepoSpec(
        test_cmd="./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
        log_parser="django",
        install="python -m pip install -e .",
        python="3.11",
    ),
    # --- matplotlib/matplotlib: 6 version(s), 34 instance(s), parser=matplotlib ---
    ("matplotlib/matplotlib", "3.0"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="matplotlib",
        install="python -m pip install -e .",
        python="3.7",
    ),
    ("matplotlib/matplotlib", "3.1"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="matplotlib",
        install="python -m pip install -e .",
        python="3.8",
    ),
    ("matplotlib/matplotlib", "3.4"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="matplotlib",
        install="python -m pip install -e .",
        python="3.8",
    ),
    ("matplotlib/matplotlib", "3.5"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="matplotlib",
        install="python -m pip install -e .",
        python="3.11",
    ),
    ("matplotlib/matplotlib", "3.6"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="matplotlib",
        install="python -m pip install -e .",
        python="3.11",
    ),
    ("matplotlib/matplotlib", "3.7"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="matplotlib",
        install="python -m pip install -e .",
        python="3.11",
    ),
    # --- mwaskom/seaborn: 1 version(s), 2 instance(s), parser=seaborn ---
    ("mwaskom/seaborn", "0.12"): RepoSpec(
        test_cmd="pytest --no-header -rA",
        log_parser="seaborn",
        install="python -m pip install -e .[dev]",
        python="3.9",
    ),
    # --- pallets/flask: 1 version(s), 1 instance(s), parser=pytest ---
    ("pallets/flask", "2.3"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest",
        install="python -m pip install -e .",
        python="3.11",
    ),
    # --- psf/requests: 7 version(s), 8 instance(s), parser=pytest_options ---
    ("psf/requests", "1.1"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_options",
        install="python -m pip install .",
        python="3.9",
    ),
    ("psf/requests", "2.0"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_options",
        install="python -m pip install .",
        python="3.9",
    ),
    ("psf/requests", "2.26"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_options",
        install="python -m pip install .",
        python="3.9",
    ),
    ("psf/requests", "2.27"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_options",
        install="python -m pip install .",
        python="3.9",
    ),
    ("psf/requests", "2.3"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_options",
        install="python -m pip install .",
        python="3.9",
    ),
    ("psf/requests", "2.4"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_options",
        install="python -m pip install .",
        python="3.9",
    ),
    ("psf/requests", "2.9"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_options",
        install="python -m pip install .",
        python="3.9",
    ),
    # --- pydata/xarray: 4 version(s), 22 instance(s), parser=pytest ---
    ("pydata/xarray", "0.12"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest",
        install="python -m pip install -e .",
        python="3.10",
    ),
    ("pydata/xarray", "2022.03"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest",
        install="python -m pip install -e .",
        python="3.10",
    ),
    ("pydata/xarray", "2022.06"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest",
        install="python -m pip install -e .",
        python="3.10",
    ),
    ("pydata/xarray", "2022.09"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest",
        install="python -m pip install -e .",
        python="3.10",
    ),
    # --- pylint-dev/pylint: 5 version(s), 10 instance(s), parser=pytest_options ---
    ("pylint-dev/pylint", "2.10"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_options",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("pylint-dev/pylint", "2.14"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_options",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("pylint-dev/pylint", "2.15"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_options",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("pylint-dev/pylint", "2.9"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_options",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("pylint-dev/pylint", "3.0"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_options",
        install="python -m pip install -e .",
        python="3.9",
    ),
    # --- pytest-dev/pytest: 10 version(s), 19 instance(s), parser=pytest ---
    ("pytest-dev/pytest", "4.5"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("pytest-dev/pytest", "4.6"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("pytest-dev/pytest", "5.0"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("pytest-dev/pytest", "5.1"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("pytest-dev/pytest", "5.2"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("pytest-dev/pytest", "5.4"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("pytest-dev/pytest", "6.0"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("pytest-dev/pytest", "6.2"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("pytest-dev/pytest", "6.3"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("pytest-dev/pytest", "7.2"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest",
        install="python -m pip install -e .",
        python="3.9",
    ),
    # --- scikit-learn/scikit-learn: 4 version(s), 32 instance(s), parser=pytest_v2 ---
    ("scikit-learn/scikit-learn", "0.20"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_v2",
        install="python -m pip install -v --no-use-pep517 --no-build-isolation -e .",
        python="3.6",
    ),
    ("scikit-learn/scikit-learn", "0.21"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_v2",
        install="python -m pip install -v --no-use-pep517 --no-build-isolation -e .",
        python="3.6",
    ),
    ("scikit-learn/scikit-learn", "0.22"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_v2",
        install="python -m pip install -v --no-use-pep517 --no-build-isolation -e .",
        python="3.6",
    ),
    ("scikit-learn/scikit-learn", "1.3"): RepoSpec(
        test_cmd="pytest -rA",
        log_parser="pytest_v2",
        install="python -m pip install -v --no-use-pep517 --no-build-isolation -e .",
        python="3.9",
    ),
    # --- sphinx-doc/sphinx: 15 version(s), 44 instance(s), parser=pytest_v2 ---
    ("sphinx-doc/sphinx", "3.0"): RepoSpec(
        test_cmd="tox --current-env -epy39 -v --",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test]",
        python="3.9",
    ),
    ("sphinx-doc/sphinx", "3.1"): RepoSpec(
        test_cmd="tox --current-env -epy39 -v --",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test]",
        python="3.9",
    ),
    ("sphinx-doc/sphinx", "3.2"): RepoSpec(
        test_cmd="tox --current-env -epy39 -v --",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test]",
        python="3.9",
    ),
    ("sphinx-doc/sphinx", "3.3"): RepoSpec(
        test_cmd="tox --current-env -epy39 -v --",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test]",
        python="3.9",
    ),
    ("sphinx-doc/sphinx", "3.4"): RepoSpec(
        test_cmd="tox --current-env -epy39 -v --",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test]",
        python="3.9",
    ),
    ("sphinx-doc/sphinx", "3.5"): RepoSpec(
        test_cmd="tox --current-env -epy39 -v --",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test]",
        python="3.9",
    ),
    ("sphinx-doc/sphinx", "4.0"): RepoSpec(
        test_cmd="tox --current-env -epy39 -v --",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test]",
        python="3.9",
    ),
    ("sphinx-doc/sphinx", "4.1"): RepoSpec(
        test_cmd="tox --current-env -epy39 -v --",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test]",
        python="3.9",
    ),
    ("sphinx-doc/sphinx", "4.2"): RepoSpec(
        test_cmd="tox --current-env -epy39 -v --",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test]",
        python="3.9",
    ),
    ("sphinx-doc/sphinx", "4.3"): RepoSpec(
        test_cmd="tox --current-env -epy39 -v --",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test]",
        python="3.9",
    ),
    ("sphinx-doc/sphinx", "5.0"): RepoSpec(
        test_cmd="tox --current-env -epy39 -v --",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test]",
        python="3.9",
    ),
    ("sphinx-doc/sphinx", "5.1"): RepoSpec(
        test_cmd="tox --current-env -epy39 -v --",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test]",
        python="3.9",
    ),
    ("sphinx-doc/sphinx", "5.2"): RepoSpec(
        test_cmd="tox --current-env -epy39 -v --",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test]",
        python="3.9",
    ),
    ("sphinx-doc/sphinx", "7.1"): RepoSpec(
        test_cmd="tox --current-env -epy39 -v --",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test]",
        python="3.9",
    ),
    ("sphinx-doc/sphinx", "7.2"): RepoSpec(
        test_cmd="tox --current-env -epy39 -v --",
        log_parser="pytest_v2",
        install="python -m pip install -e .[test]",
        python="3.9",
    ),
    # --- sympy/sympy: 12 version(s), 75 instance(s), parser=sympy ---
    ("sympy/sympy", "1.0"): RepoSpec(
        test_cmd="PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose",
        log_parser="sympy",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("sympy/sympy", "1.1"): RepoSpec(
        test_cmd="PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose",
        log_parser="sympy",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("sympy/sympy", "1.10"): RepoSpec(
        test_cmd="PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose",
        log_parser="sympy",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("sympy/sympy", "1.11"): RepoSpec(
        test_cmd="PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose",
        log_parser="sympy",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("sympy/sympy", "1.12"): RepoSpec(
        test_cmd="PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose",
        log_parser="sympy",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("sympy/sympy", "1.2"): RepoSpec(
        test_cmd="PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose",
        log_parser="sympy",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("sympy/sympy", "1.4"): RepoSpec(
        test_cmd="PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose",
        log_parser="sympy",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("sympy/sympy", "1.5"): RepoSpec(
        test_cmd="PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose",
        log_parser="sympy",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("sympy/sympy", "1.6"): RepoSpec(
        test_cmd="PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose",
        log_parser="sympy",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("sympy/sympy", "1.7"): RepoSpec(
        test_cmd="PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose",
        log_parser="sympy",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("sympy/sympy", "1.8"): RepoSpec(
        test_cmd="PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose",
        log_parser="sympy",
        install="python -m pip install -e .",
        python="3.9",
    ),
    ("sympy/sympy", "1.9"): RepoSpec(
        test_cmd="PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose",
        log_parser="sympy",
        install="python -m pip install -e .",
        python="3.9",
    ),
}


def spec_for(repo: str, version: str) -> RepoSpec:
    """Return the harness spec for `repo` at `version`, or raise.

    Never falls back to a default. A silent pytest default is the exact defect this
    table exists to fix: `pytest -rA` invoked against django or sympy exits non-zero
    having collected nothing, the log parser returns an empty status map, and the
    instance is scored unresolved -- a fabricated failure that is indistinguishable
    from a genuine one in the aggregate.
    """
    try:
        return SPECS[(repo, version)]
    except KeyError:
        known = sorted(v for r, v in SPECS if r == repo)
        if known:
            raise UnknownRepoVersionError(
                f"no SWE-bench harness spec for {repo!r} version {version!r}; "
                f"this table covers versions {known} for that repo. "
                f"If SWE-bench Verified genuinely contains this pair, the table is "
                f"incomplete -- re-derive it from upstream rather than guessing."
            ) from None
        raise UnknownRepoVersionError(
            f"no SWE-bench harness spec for repo {repo!r} (version {version!r}); "
            f"this table covers only the {len(set(r for r, _ in SPECS))} repos present "
            f"in SWE-bench Verified: {sorted(set(r for r, _ in SPECS))}"
        ) from None


# Upstream: `diff_pat = r"diff --git a/.* b/(.*)"` in get_test_directives. Greedy `.*`
# on the a-side is intentional upstream behaviour for paths containing " b/".
_DIFF_PAT = re.compile(r"diff --git a/.* b/(.*)")


def test_directives(repo: str, test_patch: str) -> list[str]:
    """The file-wise arguments appended to `test_cmd`, mirroring get_test_directives().

    The official harness runs tests FILE-WISE, not test-wise: it never passes the
    individual FAIL_TO_PASS node ids to the runner. It passes the *files* the gold
    `test_patch` touches and then reconciles the resulting status map against the
    expected node ids. Reproducing that matters because the two are not equivalent --
    running a whole file executes module-scope setup and sibling tests that a targeted
    node-id invocation would skip, which changes both pass/fail outcomes and runtime.

    Only django is path-transformed. That is worth stating explicitly because it is a
    common and reasonable assumption that sympy needs one too -- sympy's `bin/test`
    takes plain relative file paths, so upstream passes its directives through
    untouched. Verified against every upstream tag from v2.0.0 through v4.1.0: the
    body of get_test_directives is byte-stable across all of them and has never
    contained a sympy branch.

    May legitimately return an empty list, and callers must not treat that as an error.
    In SWE-bench Verified exactly one instance does so -- django__django-10097, whose
    test_patch touches only `tests/validators/{invalid,valid}_urls.txt`, both filtered
    out by NON_TEST_EXTS. Upstream appends nothing and the bare `test_cmd` then runs the
    *entire* django suite. That is real harness behaviour and must be preserved, but it
    means this one instance needs a whole-suite timeout budget rather than a per-file
    one; a timeout tuned to single-file runs will score it unresolved for reasons that
    have nothing to do with the patch under test.
    """
    # Upstream special-case, kept for exactness; no such repo is in Verified.
    if repo == "swe-bench/humaneval":
        return ["test.py"]

    directives = _DIFF_PAT.findall(test_patch)
    directives = [d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)]

    # django's runtests.py takes dotted module paths relative to tests/, not file paths.
    if repo == "django/django":
        transformed = []
        for d in directives:
            d = d[: -len(".py")] if d.endswith(".py") else d
            d = d[len("tests/") :] if d.startswith("tests/") else d
            d = d.replace("/", ".")
            transformed.append(d)
        directives = transformed

    return directives


def test_command(repo: str, version: str, test_patch: str) -> str:
    """The full test invocation, mirroring make_eval_script_list_py's assembly.

    Upstream builds it as `" ".join([test_cmd, *get_test_directives(instance)])`; the
    join is reproduced here rather than left to each caller so that no caller invents
    its own quoting or separator and drifts from the official command line.
    """
    return " ".join([spec_for(repo, version).test_cmd, *test_directives(repo, test_patch)])
