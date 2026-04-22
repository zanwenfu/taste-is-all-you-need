"""Shared fixtures for Agent OS tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from examples.refactor_demo.bootstrap import bootstrap


@pytest.fixture
def refactor_workspace(tmp_path: Path) -> Path:
    """Fresh git repo with the refactor demo's template files."""
    return bootstrap(tmp_path / "workspace")


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
