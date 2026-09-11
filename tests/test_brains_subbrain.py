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

import os
import signal
from pathlib import Path

import pytest

from taste.memstore import Store

pytest.importorskip("claude_agent_sdk", reason="the brain layer needs claude-agent-sdk")

from taste.brains.contract import CONTRACT_PATH, Contract
from taste.brains.subbrain import BUDGETED_WORKER_TOOLS, WORKER_TOOLS, SubBrain
from taste.brains.wal import WriteAheadLog, reconcile
from taste.pricing import ensure_priced, max_call_cost_usd


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
        {
            "kind": "tool_intent",
            "tool": "Bash",
            "tool_use_id": "t1",
            "request": {"command": "rm -rf build"},
            "at": "now",
        },
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


def test_the_opening_prompt_includes_live_inbox_messages(store: Store) -> None:
    store.send(
        "worker-1",
        {"kind": "artifact_request", "need": "parser-fixture"},
        sender="central",
    )
    brain = SubBrain(store, a_contract())

    waking = brain.wake()

    assert not waking.fresh
    assert "New messages have arrived" in waking.briefing()
    assert "central" in brain.opening_prompt()
    assert "parser-fixture" in brain.opening_prompt()
    brain.close()


def test_the_opening_prompt_includes_unresolved_conflicts(store: Store) -> None:
    brain = SubBrain(store, a_contract())
    brain.branch.write("parser.py", "ours\n")
    brain.branch.checkpoint("ours")
    other = store.branch("worker-2")
    other.write("parser.py", "theirs\n")
    other.checkpoint("theirs")

    result = brain.branch.merge(other, reason="combine")

    assert not result.ok
    waking = brain.wake()
    assert not waking.fresh
    assert "merge conflicts are unresolved" in waking.briefing()
    assert "parser.py" in brain.opening_prompt()
    other.close()
    brain.close()


# ------------------------------------------------------------------ wiring


def test_a_gate_that_would_never_fire_is_refused(store: Store) -> None:
    """Naming a tool in allowed_tools auto-approves it before the permission
    callback is consulted -- a gate that looks wired up and is not."""
    brain = SubBrain(store, a_contract())
    with pytest.raises(ValueError, match="shadow"):
        brain.options(allowed_tools=["Bash"])
    with pytest.raises(ValueError, match="allowed_tools"):
        brain.options(allowed_tools=["Read"])
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
    assert options.tools == list(WORKER_TOOLS)
    assert not {"Agent", "Task", "WebSearch", "WebFetch"} & set(options.tools)
    assert options.mcp_servers == {}
    assert options.strict_mcp_config is True
    assert options.fallback_model is None
    assert options.env["CLAUDE_CODE_DISABLE_ADVISOR_TOOL"] == "1"
    brain.close()


@pytest.mark.parametrize(
    "override",
    [
        {"tools": None},
        {"tools": ["Read", "WebSearch"]},
        {"add_dirs": ["../sibling"]},
        {
            "sandbox": {
                "enabled": True,
                "autoAllowBashIfSandboxed": True,
                "allowUnsandboxedCommands": False,
                "excludedCommands": ["git"],
            }
        },
        {"mcp_servers": {"remote": {"type": "http", "url": "https://example.test"}}},
        {"strict_mcp_config": False},
        {"fallback_model": "claude-opus-5"},
    ],
)
def test_paid_or_unaccounted_worker_surfaces_cannot_be_enabled(
    store: Store, override: dict[str, object]
) -> None:
    brain = SubBrain(store, a_contract())
    with pytest.raises(ValueError):
        brain.options(**override)
    brain.close()


def test_worker_budget_holds_back_the_worst_final_provider_call(store: Store) -> None:
    model = "claude-sonnet-5"
    price = ensure_priced(model)
    exposure = max_call_cost_usd(
        model,
        max_output_tokens=price.context_window,
        max_attempts=1,
        cap_on="billed",
    )
    brain = SubBrain(
        store,
        a_contract(budget_usd=2 * (exposure + 0.75)),
        model=model,
    )

    options = brain.options()

    assert options.max_budget_usd == pytest.approx(0.75)
    assert 2 * (options.max_budget_usd + exposure) <= brain.contract.budget_usd
    assert options.tools == list(BUDGETED_WORKER_TOOLS)
    assert "Bash" not in options.tools
    assert "Bash" in options.disallowed_tools
    assert options.env["DISABLE_COMPACT"] == "1"
    assert options.extra_args == {"disable-slash-commands": None}
    brain.close()


def test_too_small_worker_budget_is_refused_before_provider_start(store: Store) -> None:
    model = "claude-sonnet-5"
    price = ensure_priced(model)
    exposure = max_call_cost_usd(
        model,
        max_output_tokens=price.context_window,
        max_attempts=1,
        cap_on="billed",
    )
    brain = SubBrain(store, a_contract(budget_usd=2 * exposure), model=model)

    with pytest.raises(ValueError, match="cannot cover the reset-safe model exposure"):
        brain.options()
    brain.close()


def test_worker_budget_subtracts_durable_spend_from_prior_resets(store: Store) -> None:
    model = "claude-sonnet-5"
    price = ensure_priced(model)
    exposure = max_call_cost_usd(
        model,
        max_output_tokens=price.context_window,
        max_attempts=1,
        cap_on="billed",
    )
    brain = SubBrain(
        store,
        a_contract(budget_usd=0.4 + 2 * (exposure + 0.6)),
        model=model,
    )

    options = brain.options(budget_already_spent_usd=0.4)

    assert options.max_budget_usd == pytest.approx(0.6)
    assert 0.4 + 2 * (options.max_budget_usd + exposure) <= brain.contract.budget_usd
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
        brain._pre_tool({"tool_name": "Write", "tool_input": {"file_path": "mine.txt"}}, "t1", None)
    )
    assert allowed == {}
    assert len(reconcile(brain.branch.resume().recovered_turns)) == 1

    asyncio.run(brain._post_tool({"tool_name": "Write", "tool_response": {"ok": True}}, "t1", None))
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


def test_a_brain_its_monitor_failed_is_not_treated_as_fresh(store: Store) -> None:
    """The verdict channel exists to stop a brain building on work already
    judged wrong. It was durable and delivered -- `unacked` held it -- but the
    briefing ignored it whenever the tree happened to be clean, so the brain
    woke up and carried on. Found by the monitor's own tests.
    """
    from taste.memstore import Verdict

    brain = SubBrain(store, a_contract())
    brain.install_contract()
    state = brain.checkpoint("work the monitor will fail")
    store.judge(state, Verdict(status="fail", by="monitor", detail="B is a dead end"))

    waking = brain.wake()
    assert not waking.fresh, "a brain that has been failed is not fresh"
    assert "B is a dead end" in waking.briefing()
    assert "B is a dead end" in brain.opening_prompt()
    brain.close()


def test_an_acknowledged_verdict_stops_reopening_the_briefing(store: Store) -> None:
    """Once reacted to, an old failure is history, not news -- otherwise every
    wake reruns the same correction."""
    from taste.memstore import Verdict

    brain = SubBrain(store, a_contract())
    state = brain.checkpoint("work")
    store.judge(state, Verdict(status="fail", by="monitor", detail="dead end"))
    assert not brain.wake().fresh

    brain.branch.acknowledge()
    assert brain.wake().fresh
    brain.close()


def test_an_in_flight_tool_survives_a_checkpoint(store: Store) -> None:
    """This is the WAL's headline claim, and a checkpoint used to break it.

    ``reconcile`` read only the journal for the current head, but ``_build``
    folds that journal into the state transcript and ``publish_state`` unlinks
    it. So an intent recorded before a checkpoint became invisible the moment
    the brain committed -- while still provably on disk in the transcript. The
    gap stopped being detectable, which is exactly the difference between
    "nothing is lost" and "nothing is lost silently".
    """
    brain = SubBrain(store, a_contract())
    brain.wal.intent("Bash", "t1", {"command": "psql -c 'TRUNCATE users'"})
    assert brain.wake().uncertain

    brain.checkpoint("preserve the work so far")

    waking = brain.wake()
    assert waking.uncertain, "the checkpoint hid an in-flight tool"
    assert [f.tool_use_id for f in waking.in_flight] == ["t1"]
    assert "TRUNCATE" in waking.briefing()
    brain.close()


def test_a_tool_that_finished_after_a_checkpoint_is_not_reported(store: Store) -> None:
    """An intent and its result can straddle a checkpoint. Reading across the
    boundary must pair them, not warn about a tool that demonstrably finished.
    """
    brain = SubBrain(store, a_contract())
    brain.wal.intent("Bash", "t1", {"command": "pytest"})
    brain.checkpoint("commit between the intent and the result")
    brain.wal.result("Bash", "t1", ok=True, summary="3 passed")

    waking = brain.wake()
    assert waking.in_flight == (), "a finished tool was reported as unknown"
    brain.close()


def test_in_flight_warnings_survive_several_checkpoints(store: Store) -> None:
    """A long run checkpoints many times; the warning must not decay."""
    brain = SubBrain(store, a_contract())
    brain.wal.intent("Bash", "danger", {"command": "deploy --prod"})
    for i in range(3):
        brain.wal.intent("Bash", f"ok{i}", {})
        brain.wal.result("Bash", f"ok{i}", ok=True)
        brain.checkpoint(f"round {i}")

    waking = brain.wake()
    assert [f.tool_use_id for f in waking.in_flight] == ["danger"]
    assert "deploy --prod" in waking.briefing()
    brain.close()


def test_the_wal_follows_the_branch_through_a_rollback(store: Store) -> None:
    """A rollback moves the head without going through SubBrain.checkpoint,
    so nothing calls rebind -- and the log was left writing into an orphaned
    journal that resume() never reads. An ``rm -rf`` recorded there was
    invisible to the next brain.

    Correctness must not depend on a caller remembering, so the log
    re-resolves whenever its journal has vanished -- which is precisely what
    publish_state does to the journal it folded in.
    """
    brain = SubBrain(store, a_contract())
    brain.wal.intent("Bash", "early", {})
    good = brain.checkpoint("a good state")
    brain.wal.intent("Bash", "mistake", {})
    brain.checkpoint("a bad state")

    brain.branch.rollback(good, "that approach was wrong")
    brain.wal.intent("Bash", "after-rollback", {"command": "rm -rf build"})

    in_flight = {f.tool_use_id for f in brain.wake().in_flight}
    assert "after-rollback" in in_flight, "the log wrote to an orphaned journal"
    brain.close()


def test_the_wal_stays_cheap_enough_for_a_hook(store: Store) -> None:
    """Following the branch must not cost a git read per append.

    Resolving the head through git costs ~8 ms, which both hooks together
    would spend most of the ~20 ms gate budget on. A file-existence check is
    0.01 ms and answers the same question, because publish_state unlinks the
    journal it consumed.
    """
    import time

    brain = SubBrain(store, a_contract())
    brain.checkpoint("base")

    times = []
    for i in range(100):
        start = time.perf_counter()
        brain.wal.intent("Bash", f"t{i}", {"command": "pytest"})
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    median = times[len(times) // 2]
    assert median < 2.0, f"{median:.3f} ms per append is too slow for a hook"
    brain.close()
