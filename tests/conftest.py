"""Shared fixtures for Agent OS tests."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from examples.refactor_demo.bootstrap import bootstrap

#: The verification command every fixture that actually *runs* a check must use.
#:
#: A bare ``pytest`` resolves against PATH, and PATH does not include the
#: venv's ``bin`` when the suite runs as ``python -m pytest`` -- which is how
#: it runs anywhere nobody typed ``activate`` first, CI included. On the
#: machine this was written on an ambient pytest happened to be on PATH and
#: all of these passed. On a clean host fifteen failed, and they failed
#: reading as *harness* defects -- "rollback did not recover", "the merge gate
#: rejected independent work" -- rather than as a command that does not exist.
#: A green suite on one machine was certifying nothing on any other.
#:
#: Same defect class as gate 0's bare ``python``, and the same fix: name the
#: interpreter running this process instead of asking the environment.
PYTEST_CMD = f"{shlex.quote(sys.executable)} -m pytest -q"


@pytest.fixture
def refactor_workspace(tmp_path: Path) -> Path:
    """Fresh git repo with the refactor demo's template files."""
    return bootstrap(tmp_path / "workspace")


@pytest.fixture
def parallel_workspace(tmp_path: Path) -> Path:
    """Small repo with 3 independent modules — the parallel demo target.

    Modules are disjoint by construction: workers can edit all three in
    parallel without any merge conflicts.
    """
    ws = tmp_path / "parallel"
    ws.mkdir()

    (ws / "math_utils.py").write_text("def add(a, b):\n    return a + b\n")
    (ws / "string_utils.py").write_text("def upper(s):\n    return s.upper()\n")
    (ws / "list_utils.py").write_text("def head(xs):\n    return xs[0] if xs else None\n")
    tests = ws / "tests"
    tests.mkdir()
    (tests / "test_math.py").write_text(
        "from math_utils import add\ndef test_add(): assert add(1, 2) == 3\n"
    )
    (tests / "test_string.py").write_text(
        "from string_utils import upper\ndef test_upper(): assert upper('hi') == 'HI'\n"
    )
    (tests / "test_list.py").write_text(
        "from list_utils import head\ndef test_head(): assert head([1]) == 1\n"
    )
    (ws / "conftest.py").write_text(
        "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).parent))\n"
    )

    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    subprocess.run(["git", "add", "."], cwd=ws, check=True)
    subprocess.run(
        ["git", "-c", "user.name=p", "-c", "user.email=p@p.local", "commit", "-q", "-m", "init"],
        cwd=ws,
        check=True,
    )
    return ws


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    """Minimal git repo with a single file — for memory primitives tests."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "hello.txt").write_text("hello\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c", "user.name=test",
            "-c", "user.email=t@test.local",
            "commit", "-q", "-m", "initial",
        ],
        cwd=root,
        check=True,
    )
    return root
