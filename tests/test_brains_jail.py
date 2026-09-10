"""The worktree jail: an advisory gate over the real sandbox.

``cwd`` confines nothing -- from one worktree, under both ``acceptEdits`` and
``bypassPermissions``, a brain wrote into the shared ``.git`` and into a
sibling's worktree, reporting success every time. The conclusion first drawn
from that, that the jail must therefore BE the boundary, was wrong:
``ClaudeAgentOptions.sandbox`` is an OS-level sandbox that refuses those writes
with "operation not permitted", including ``D=../sib; echo x > $D/f`` and
``dd of=../sib/f``, which defeat any command parser.

So these tests hold the gate to what a gate can promise: it catches the common
cases early and explains them in the brain's own language, and it never lets an
odd input through by crashing -- because a hook that raises FAILS OPEN
(measured: the exception is logged, the tool runs, the run reports success).

A false allow here is a bug, not a breach; the sandbox is underneath. A false
deny costs one explained retry. The asymmetry still favours denying.
"""

from __future__ import annotations

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


# ------------------------------------------------ the gate must fail closed


@pytest.mark.parametrize(
    "tool_input",
    [
        pytest.param(["../sibling/x"], id="input-is-a-list"),
        pytest.param("../sibling/x", id="input-is-a-string"),
        pytest.param(None, id="input-is-none"),
        pytest.param({"file_path": "../sibling/x\x00"}, id="nul-byte-in-path"),
        pytest.param({"file_path": ["../sibling/x"]}, id="path-is-a-list"),
        pytest.param({"file_path": 7}, id="path-is-an-int"),
        pytest.param({"edits": "not-a-list"}, id="edits-is-a-string"),
        pytest.param({"edits": ["not-a-dict"]}, id="edit-is-a-string"),
    ],
)
def test_a_malformed_call_is_denied_not_crashed(jail: WorktreeJail, tool_input) -> None:
    """A hook that raises FAILS OPEN.

    Measured against CLI 2.1.257: the exception is logged, the tool executes
    anyway, and the run reports success -- the opposite of
    ``HookMatcher(timeout=)``, which fails closed. So every input that could
    raise is a silent bypass unless the gate turns it into a denial itself.

    A NUL byte is the sharp case: ``os.path.realpath`` raises ``ValueError``
    ("embedded null character"), which is not an ``OSError`` and so escaped the
    original guard entirely.
    """
    import asyncio

    out = asyncio.run(
        jail.hook({"tool_name": "Write", "tool_input": tool_input}, "t", None)
    )
    decision = out.get("hookSpecificOutput", {}).get("permissionDecision")
    assert decision == "deny", f"{tool_input!r} was not denied: {out!r}"


def test_a_nul_byte_in_a_command_is_denied(jail: WorktreeJail) -> None:
    import asyncio

    out = asyncio.run(
        jail.hook(
            {"tool_name": "Bash", "tool_input": {"command": "cat ../sibling/x\x00"}},
            "t",
            None,
        )
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_the_gate_still_allows_ordinary_work_after_hardening(jail: WorktreeJail) -> None:
    """Failing closed must not mean failing on everything."""
    import asyncio

    for tool, ti in [
        ("Write", {"file_path": "keep.txt"}),
        ("Bash", {"command": "pytest -q"}),
        ("Edit", {"file_path": "sub/x.py"}),
        ("Write", {}),
    ]:
        out = asyncio.run(jail.hook({"tool_name": tool, "tool_input": ti}, "t", None))
        assert out == {}, f"{tool} {ti!r} was wrongly denied: {out!r}"


# ------------------------------------- escapes an earlier version let through


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(
            "git --git-dir=../repo/.git update-ref -d refs/heads/mem/s/beta",
            id="git-dir-deletes-a-siblings-branch",
        ),
        pytest.param("git --work-tree=../sibling checkout -- x", id="work-tree"),
        pytest.param("echo x >../sibling/theirs.txt", id="redirect-without-a-space"),
        pytest.param("dd of=../sibling/theirs.txt", id="operand-hidden-behind-of="),
    ],
)
def test_flag_and_operand_forms_do_not_slip_past(jail: WorktreeJail, command: str) -> None:
    """Every one of these passed an earlier version of the gate.

    ``--git-dir`` is the sharpest: skipping any word starting with ``-`` meant
    a brain could point git at the shared repository and delete a sibling's
    branch -- the only ref reaching that brain's states, after which ``gc``
    would prune them permanently.
    """
    assert jail.check("Bash", {"command": command}) is not None, command


def test_a_brain_cannot_switch_off_its_own_sandbox(jail: WorktreeJail) -> None:
    """The Bash tool takes a ``dangerouslyDisableSandbox`` parameter and the
    model can set it. Measured: a model refused by the sandbox re-issued the
    identical command with the flag set, and the write landed.

    The sandbox is the real boundary; this gate is advisory. Letting the flag
    through would let one brain dismantle what protects the others.
    """
    reason = jail.check("Bash", {"command": "pytest -q", "dangerouslyDisableSandbox": True})
    assert reason is not None and "sandbox cannot be disabled" in reason


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q --maxfail=1",
        "git log --format=%H -n 5",
        "git add -A && git commit -m 'work'",
        "echo hi >out.txt",
        "python -m pytest tests/ -k test_x",
        "grep -rn 'def foo' src/",
    ],
)
def test_hardening_did_not_break_ordinary_flags(jail: WorktreeJail, command: str) -> None:
    """Checking the value half of ``--flag=value`` must not deny every flag."""
    assert jail.check("Bash", {"command": command}) is None, command
