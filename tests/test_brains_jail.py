"""The worktree jail, which is the only boundary a sub-brain has.

Measured on this SDK: there is no filesystem sandbox. From one worktree, under
both ``acceptEdits`` and ``bypassPermissions``, a brain wrote into the shared
``.git`` and into a sibling's worktree, reporting success every time. So these
are not defence-in-depth tests; the jail is the defence.

A false allow corrupts a sibling brain or the shared repository. A false deny
costs one explained retry. The tests are written with that asymmetry in mind.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from taste.brains.jail import WorktreeJail


@pytest.fixture
def jail(tmp_path: Path) -> WorktreeJail:
    root = tmp_path / "wt"
    (root / "sub").mkdir(parents=True)
    (root / "keep.txt").write_text("mine\n")
    (tmp_path / "sibling").mkdir()
    (tmp_path / "sibling" / "theirs.txt").write_text("theirs\n")
    return WorktreeJail(root)


# ------------------------------------------------------------------ allowed


@pytest.mark.parametrize(
    "path",
    ["keep.txt", "sub/new.txt", "deeper/not/created/yet.txt", "./keep.txt"],
)
def test_work_inside_the_worktree_is_allowed(jail: WorktreeJail, path: str) -> None:
    assert jail.check("Write", {"file_path": path}) is None


def test_an_absolute_path_inside_the_worktree_is_allowed(jail: WorktreeJail) -> None:
    assert jail.check("Write", {"file_path": str(jail.root / "sub" / "x.txt")}) is None


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q",
        "git status",
        "git add -A && git commit -m 'work'",
        "echo hello > out.txt",
        "cat keep.txt",
        "ls -la sub/",
    ],
)
def test_ordinary_commands_are_allowed(jail: WorktreeJail, command: str) -> None:
    assert jail.check("Bash", {"command": command}) is None, command


def test_read_only_tools_are_not_gated(jail: WorktreeJail) -> None:
    """Every gated call pays the hook's latency, and a read escapes nothing."""
    from taste.brains.jail import MUTATING_TOOLS

    assert "Read" not in MUTATING_TOOLS
    assert "Grep" not in MUTATING_TOOLS


# ------------------------------------------------------------------ denied


@pytest.mark.parametrize(
    "path",
    [
        "../sibling/theirs.txt",
        "../../etc/passwd",
        "sub/../../sibling/theirs.txt",
        "/etc/hosts",
        "~/.ssh/id_rsa",
    ],
)
def test_writing_outside_the_worktree_is_denied(jail: WorktreeJail, path: str) -> None:
    reason = jail.check("Write", {"file_path": path})
    assert reason is not None, f"{path} escaped the jail"
    assert "outside your worktree" in reason


def test_a_traversal_through_a_directory_that_does_not_exist_is_denied(
    jail: WorktreeJail,
) -> None:
    """A check on the literal string would miss this: the prefix looks local."""
    assert jail.check("Write", {"file_path": "newdir/../../sibling/theirs.txt"}) is not None


@pytest.mark.parametrize(
    "command",
    [
        "echo pwned > ../sibling/theirs.txt",
        "echo pwned >> ../../etc/passwd",
        "cat ../sibling/theirs.txt",
        "rm -rf ../sibling",
        "cp keep.txt /etc/keep.txt",
    ],
)
def test_shell_escapes_are_denied(jail: WorktreeJail, command: str) -> None:
    """A file_path check cannot see a redirection; this is the bypass that
    made the jail necessary in the first place."""
    reason = jail.check("Bash", {"command": command})
    assert reason is not None, f"{command!r} escaped the jail"


def test_the_shared_git_directory_is_protected(tmp_path: Path) -> None:
    """The measured worst case: a confused brain corrupting the shared .git
    breaks every sibling worktree at once."""
    base = tmp_path / "repo"
    (base / ".git" / "worktrees").mkdir(parents=True)
    wt = tmp_path / "wt"
    wt.mkdir()
    jail = WorktreeJail(wt)

    assert jail.check("Write", {"file_path": "../repo/.git/config"}) is not None
    assert jail.check("Bash", {"command": "echo x > ../repo/.git/HEAD"}) is not None


def test_a_multi_edit_is_checked_per_edit(jail: WorktreeJail) -> None:
    """One bad edit in a batch must not ride in on the good ones."""
    edits = [
        {"file_path": "keep.txt"},
        {"file_path": "../sibling/theirs.txt"},
    ]
    assert jail.check("MultiEdit", {"edits": edits}) is not None


# ------------------------------------------------------------------ the hook


@pytest.mark.parametrize("field", ["file_path", "path", "notebook_path"])
def test_every_path_field_is_checked(jail: WorktreeJail, field: str) -> None:
    assert jail.check("Write", {field: "../sibling/theirs.txt"}) is not None


def test_the_hook_denies_synchronously_with_a_reason(jail: WorktreeJail) -> None:
    """A decision returned alongside {"async_": True} is silently discarded --
    measured, every tool ran and nothing was logged. Decisions must be
    synchronous, and the reason must reach the model so it adapts rather than
    retrying the same call."""
    import asyncio

    out = asyncio.run(
        jail.hook(
            {"tool_name": "Write", "tool_input": {"file_path": "../sibling/theirs.txt"}},
            "tool-1",
            None,
        )
    )
    specific = out["hookSpecificOutput"]
    assert "async_" not in out
    assert specific["hookEventName"] == "PreToolUse"
    assert specific["permissionDecision"] == "deny"
    assert "outside your worktree" in specific["permissionDecisionReason"]
    assert jail.denials == [("Write", specific["permissionDecisionReason"])]


def test_the_hook_stays_out_of_the_way_when_the_call_is_fine(jail: WorktreeJail) -> None:
    import asyncio

    out = asyncio.run(
        jail.hook({"tool_name": "Write", "tool_input": {"file_path": "keep.txt"}}, "t", None)
    )
    assert out == {}
    assert jail.denials == []


def test_the_gate_is_fast_enough_to_sit_in_the_event_loop(jail: WorktreeJail) -> None:
    """Hook cost is added 1:1 to every matched tool call, so the budget is
    ~20ms. An LLM here would cost a median 2.18s -- about 100x."""
    import time

    commands = [
        "pytest -q tests/",
        "git add -A && git commit -m x",
        "echo hello > out.txt",
        "python -c 'print(1)'",
    ]
    start = time.perf_counter()
    for _ in range(100):
        for command in commands:
            jail.check("Bash", {"command": command})
    elapsed_ms = (time.perf_counter() - start) * 1000 / (100 * len(commands))
    assert elapsed_ms < 5.0, f"{elapsed_ms:.2f} ms per check is too slow for the hot path"


def test_a_symlink_cannot_smuggle_a_write_out(tmp_path: Path) -> None:
    """realpath, not string prefixes: a symlink inside the worktree pointing
    out of it is the classic bypass."""
    wt = tmp_path / "wt"
    wt.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (wt / "escape").symlink_to(outside)
    jail = WorktreeJail(wt)

    assert jail.check("Write", {"file_path": "escape/loot.txt"}) is not None


def test_an_explicitly_allowed_path_is_permitted(tmp_path: Path) -> None:
    """A sub-brain may be given read access to a shared input directory."""
    wt = tmp_path / "wt"
    wt.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    jail = WorktreeJail(wt, allow=(str(shared),))

    assert jail.check("Bash", {"command": f"cat {shared}/input.csv"}) is None
    assert jail.check("Write", {"file_path": str(tmp_path / "elsewhere" / "x")}) is not None
