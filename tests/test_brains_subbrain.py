"""A sub-brain, and what survives killing it.

The model is stubbed here on purpose. The claim under test is not "the model
does good work" -- it is that whatever the brain did survives a SIGKILL and the
next process is told the truth about it, and a nondeterministic model would
make that claim untestable rather than better tested.

The sharp case throughout is the tool that was started and never finished. Its
outcome is genuinely unknowable, and the whole design rests on that being
*visible* rather than silent: measured, a brain reading silence said "I hadn't
actually executed it yet, so let me run it now" and repeated the command.
"""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import pytest

from taste.memstore import Store

pytest.importorskip("claude_agent_sdk", reason="the brain layer needs claude-agent-sdk")

from taste.brains.contract import CONTRACT_PATH, Contract  # noqa: E402
from taste.brains.subbrain import SubBrain  # noqa: E402
from taste.brains.wal import InFlight, WriteAheadLog, reconcile  # noqa: E402


def a_contract(**kw) -> Contract:
    base = {
        "identity": "worker-1",
        "task": "build the parser",
        "success_criteria": ("the tests pass",),
    }
    return Contract(**{**base, **kw})


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "repo", "s1")
    yield s
    s.close()


def _in_a_doomed_child(work) -> None:
    """Run ``work`` in a forked child SIGKILLed before it can return."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never returns to pytest
        try:
            work()
        finally:
            os.kill(os.getpid(), signal.SIGKILL)
    _, status = os.waitpid(pid, 0)
    assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL


# ------------------------------------------------------------------ reconcile


def test_a_finished_tool_is_not_reported_as_unknown() -> None:
    turns = [
        {"kind": "tool_intent", "tool": "Bash", "tool_use_id": "t1", "request": {}},
        {"kind": "tool_result", "tool": "Bash", "tool_use_id": "t1", "ok": True},
    ]
    assert reconcile(turns) == []


def test_an_unfinished_tool_is_reported() -> None:
    turns = [
        {"kind": "tool_intent", "tool": "Bash", "tool_use_id": "t1",
         "request": {"command": "rm -rf build"}, "at": "now"},
    ]
    out = reconcile(turns)
    assert [f.tool_use_id for f in out] == ["t1"]
    assert "UNKNOWN" in out[0].as_warning()
    assert "rm -rf build" in out[0].as_warning()


def test_overlapping_tools_are_paired_by_id_not_by_order() -> None:
    """Tools can overlap; pairing by position marks the wrong one unknown."""
    turns = [
        {"kind": "tool_intent", "tool": "Bash", "tool_use_id": "a", "request": {}},
        {"kind": "tool_intent", "tool": "Write", "tool_use_id": "b", "request": {}},
        {"kind": "tool_result", "tool": "Write", "tool_use_id": "b", "ok": True},
    ]
    assert [f.tool_use_id for f in reconcile(turns)] == ["a"]


def test_a_call_with_no_id_is_reported_rather_than_assumed_done() -> None:
    """It cannot be paired, so the safe direction is to warn: a spurious
    warning costs one check, a missed one costs a repeated side effect."""
    turns = [{"kind": "tool_intent", "tool": "Bash", "request": {}, "at": "now"}]
    assert len(reconcile(turns)) == 1


# ------------------------------------------------------------------ the WAL


def test_the_wal_is_durable_without_a_clean_exit(tmp_path: Path) -> None:
    """No flush, no close, no checkpoint -- the write itself is the record,
    because a killed process runs none of those."""
    root = tmp_path / "repo"
    Store.open(root, "s1").close()

    def brain() -> None:
        s = Store.open(root, "s1")
        b = s.branch("worker-1")
        wal = WriteAheadLog(b)
        wal.intent("Bash", "t1", {"command": "pytest"})
        wal.result("Bash", "t1", ok=True, summary="3 passed")
        wal.intent("Write", "t2", {"file_path": "out.py"})

    _in_a_doomed_child(brain)

    s = Store.open(root, "s1")
    recovered = s.branch("worker-1").resume().recovered_turns
    unfinished = reconcile(recovered)
    assert len(recovered) == 3
    assert [f.tool_use_id for f in unfinished] == ["t2"], "the in-flight tool was lost"
    s.close()


def test_the_wal_costs_almost_nothing_per_call(store: Store) -> None:
    """It appends from inside a PreToolUse hook, where cost is added 1:1 to
    every tool call against a ~20 ms budget. Reaching this through
    branch.turn() cost 15.1 ms -- two git subprocesses, not the fsync."""
    import time

    branch = store.branch("worker-1")
    branch.checkpoint("base")
    wal = WriteAheadLog(branch)

    times = []
    for i in range(50):
        start = time.perf_counter()
        wal.intent("Bash", f"t{i}", {"command": "pytest"})
        wal.result("Bash", f"t{i}", ok=True)
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    assert times[len(times) // 2] < 5.0, f"{times[len(times) // 2]:.2f} ms per call"


# ------------------------------------------------------------------ waking


def test_a_fresh_brain_is_told_only_its_contract(store: Store) -> None:
    brain = SubBrain(store, a_contract())
    waking = brain.wake()
    assert waking.fresh and not waking.uncertain
    assert waking.briefing() == ""
    assert brain.opening_prompt() == brain.contract.brief()
    brain.close()


def test_a_resumed_brain_is_told_what_it_was_attempting(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    Store.open(root, "s1").close()

    def brain() -> None:
        s = Store.open(root, "s1")
        sb = SubBrain(s, a_contract())
        sb.branch.intend("rewrite the tokenizer")
        sb.wal.intent("Bash", "t1", {"command": "rm -rf build/"})

    _in_a_doomed_child(brain)

    s = Store.open(root, "s1")
    waking = SubBrain(s, a_contract()).wake()
    assert not waking.fresh and waking.uncertain
    briefing = waking.briefing()
    assert "rewrite the tokenizer" in briefing
    assert "UNKNOWN" in briefing and "rm -rf build/" in briefing
    assert "do not repeat it" in briefing
    s.close()


def test_the_briefing_tells_the_brain_to_check_before_repeating(store: Store) -> None:
    """The failure this exists to prevent: a brain reading silence concluded
    the tool never ran and ran it again."""
    brain = SubBrain(store, a_contract())
    brain.wal.intent("Bash", "t1", {"command": "git push"})
    briefing = brain.wake().briefing()
    assert "may or may not have taken effect" in briefing
    assert "establish what is actually true" in briefing
    brain.close()


# ------------------------------------------------------------------ wiring


def test_a_gate_that_would_never_fire_is_refused(store: Store) -> None:
    """Naming a tool in allowed_tools auto-approves it before the permission
    callback is consulted -- a gate that looks wired up and is not."""
    brain = SubBrain(store, a_contract())
    with pytest.raises(ValueError, match="shadow"):
        brain.options(allowed_tools=["Bash"])
    brain.close()


def test_the_sandbox_cannot_be_configured_away(store: Store) -> None:
    brain = SubBrain(store, a_contract())
    with pytest.raises(ValueError, match="sandbox"):
        brain.options(sandbox={"enabled": False})
    with pytest.raises(ValueError, match="sandbox"):
        brain.options(sandbox={"enabled": True, "allowUnsandboxedCommands": True})
    brain.close()


def test_the_config_directory_is_not_inside_the_worktree(store: Store) -> None:
    """Inside, git status reports it and a checkpoint commits the brain's own
    plumbing as though the brain had produced it."""
    brain = SubBrain(store, a_contract())
    assert brain.worktree not in brain.config_dir.parents
    brain.branch.checkpoint("base")
    assert not brain.branch.is_dirty(), brain.branch.dirty_paths()
    brain.close()


def test_the_brain_runs_on_a_streaming_client_with_the_gates_installed(
    store: Store,
) -> None:
    brain = SubBrain(store, a_contract())
    options = brain.options()
    assert options.session_store is brain.sessions
    assert options.session_store_flush == "eager"
    assert set(options.hooks) == {"PreToolUse", "PostToolUse"}
    assert options.sandbox["enabled"] and not options.sandbox["allowUnsandboxedCommands"]
    assert options.cwd == str(brain.worktree)
    assert options.setting_sources == []
    brain.close()


def test_a_refused_call_leaves_no_phantom_unknown(store: Store) -> None:
    """The jail runs before the WAL: a call that never happened must not leave
    a permanent 'outcome unknown' for a side effect that provably never was."""
    import asyncio

    brain = SubBrain(store, a_contract())
    denied = asyncio.run(
        brain._pre_tool(
            {"tool_name": "Write", "tool_input": {"file_path": "../elsewhere/x"}},
            "t1",
            None,
        )
    )
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert reconcile(brain.branch.resume().recovered_turns) == []
    brain.close()


def test_an_allowed_call_records_an_intent_then_a_result(store: Store) -> None:
    import asyncio

    brain = SubBrain(store, a_contract())
    allowed = asyncio.run(
        brain._pre_tool(
            {"tool_name": "Write", "tool_input": {"file_path": "mine.txt"}}, "t1", None
        )
    )
    assert allowed == {}
    assert len(reconcile(brain.branch.resume().recovered_turns)) == 1

    asyncio.run(
        brain._post_tool(
            {"tool_name": "Write", "tool_response": {"ok": True}}, "t1", None
        )
    )
    assert reconcile(brain.branch.resume().recovered_turns) == []
    brain.close()


def test_the_contract_lives_in_the_branch(store: Store) -> None:
    """Worker and monitor read the same document, not two paraphrases."""
    brain = SubBrain(store, a_contract())
    brain.install_contract()
    state = brain.checkpoint("contract issued")
    assert Contract.from_json(state.read(CONTRACT_PATH)) == brain.contract
    brain.close()


def test_two_brains_cannot_share_one_branch(store: Store) -> None:
    from taste.memstore import BranchBusy

    brain = SubBrain(store, a_contract())
    try:
        other = Store.open(store.root, "s1")
        with pytest.raises(BranchBusy):
            SubBrain(other, a_contract())
        other.close()
    finally:
        brain.close()


def test_its_own_contract_does_not_make_a_new_brain_look_interrupted(
    store: Store,
) -> None:
    """The contract is scaffolding the spawner wrote, not work the brain did.

    Counting it as uncommitted work greeted a brand-new brain with "You are
    resuming work that was interrupted" on its first breath -- wrong, and a
    waste of the context it opens with. Caught in a live run, not by reasoning.
    """
    brain = SubBrain(store, a_contract())
    brain.install_contract()

    waking = brain.wake()
    assert waking.fresh, f"a new brain looked interrupted: {waking.dirty_paths}"
    assert waking.briefing() == ""
    assert brain.opening_prompt() == brain.contract.brief()
    brain.close()


def test_real_uncommitted_work_still_counts_as_interrupted(store: Store) -> None:
    """Ignoring the contract must not blind the brain to actual work."""
    brain = SubBrain(store, a_contract())
    brain.install_contract()
    brain.branch.write("half_written.py", "def f(:\n")

    waking = brain.wake()
    assert not waking.fresh
    assert waking.dirty_paths == ("half_written.py",)
    brain.close()


def test_installing_a_contract_twice_does_not_overwrite_a_revision(
    store: Store,
) -> None:
    """A re-plan revises the contract; the brain told to follow the revision
    must not undo it on the way in."""
    brain = SubBrain(store, a_contract())
    brain.install_contract()
    revised = a_contract(task="build the parser, but simpler").to_json()
    brain.branch.write(CONTRACT_PATH, revised)

    brain.install_contract()
    assert brain.branch.path(CONTRACT_PATH).read_text() == revised
    brain.close()
