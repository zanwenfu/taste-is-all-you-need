"""Memory protection: vetoing a tool call before it runs.

The worker runs arbitrary shell in a real repository. These tests pin the
three things that must not happen — the substrate being rewritten by the code
it stores, verification being edited by the thing under verification, and a
step spending without bound — plus the property that matters more than any of
them: a broken guard degrades the run, it never kills it.
"""

from __future__ import annotations

from pathlib import Path

from taste.cores import Step, Verification
from taste.guardrails import GuardConfig, Guardrails
from taste.recovery import ActionKind, FailureClass, RecoveryConfig
from taste.tools import ToolRegistry, make_builtin_tools
from tests.fakes import FakeLLM, FakeTurn


def _guard(ws: Path, **overrides) -> Guardrails:
    config = GuardConfig(enabled=True, **overrides)
    return Guardrails(workspace=ws, config=config)


def _step() -> Step:
    return Step("step-01", "do it", Verification(kind="shell", command="true"))


def _tools(ws: Path) -> ToolRegistry:
    registry = ToolRegistry()
    registry.extend(make_builtin_tools(ws))
    return registry


def _spec():
    from taste.agent import AgentSpec

    return AgentSpec(name="w", description="", system_prompt="p")


# ------------------------------------------------------------------ substrate


def test_git_mutations_are_vetoed(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    for command in (
        "git commit -am wip",
        "git reset --hard HEAD~1",
        "git checkout main",
        "git branch -D taste/session-x",
        "git update-ref refs/heads/x HEAD",
    ):
        decision = guard.before_tool(1, "run_shell", {"command": command})
        assert decision.action == "veto", command
        assert "kernel owns" in (decision.message or "")


def test_read_only_git_is_allowed(tmp_path: Path) -> None:
    """A worker inspecting history is doing its job; only mutation is denied."""
    guard = _guard(tmp_path)
    for command in ("git status", "git log --oneline -5", "git diff HEAD", "git ls-files"):
        assert guard.before_tool(1, "run_shell", {"command": command}).action == "allow", command


def test_git_hidden_in_a_compound_command_is_still_caught(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    for command in ("pytest && git commit -am done", "echo hi; git reset --hard"):
        assert guard.before_tool(1, "run_shell", {"command": command}).action == "veto", command


def test_substrate_paths_are_protected(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    for path in (".git/hooks/pre-commit", ".git/config", ".taste/plan.json"):
        decision = guard.before_tool(1, "write_file", {"path": path, "content": "x"})
        assert decision.action == "veto", path
        assert "harness state" in (decision.message or "")


def test_paths_outside_the_workspace_are_vetoed(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    guard = _guard(workspace)
    decision = guard.before_tool(1, "read_file", {"path": "../../../etc/passwd"})
    assert decision.action == "veto"
    assert "outside" in (decision.message or "")


def test_ordinary_project_files_are_untouched(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    assert guard.before_tool(1, "write_file", {"path": "src/app.py", "content": "x"}).action == "allow"
    assert guard.before_tool(1, "read_file", {"path": "README.md"}).action == "allow"
    assert guard.before_tool(1, "run_shell", {"command": "pytest -q"}).action == "allow"


# ------------------------------------------------------------------ test mutation


def test_test_edits_are_recorded_not_blocked(tmp_path: Path) -> None:
    """Editing tests is sometimes the actual task — but never invisible."""
    guard = _guard(tmp_path)
    decision = guard.before_tool(1, "write_file", {"path": "tests/test_math.py", "content": "x"})

    assert decision.action == "allow"
    assert guard.report.test_files_touched == ["tests/test_math.py"]


def test_test_mutation_emits_an_event(tmp_path: Path) -> None:
    seen = []
    guard = Guardrails(
        workspace=tmp_path,
        config=GuardConfig(enabled=True),
        on_event=lambda kind, **p: seen.append((kind, p)),
    )
    guard.before_tool(1, "write_file", {"path": "test_thing.py", "content": "assert True"})
    assert [k for k, _ in seen] == ["guard.test_mutation"]


# ------------------------------------------------------------------ rewrite


def test_absurd_shell_timeout_is_clamped_not_vetoed(tmp_path: Path) -> None:
    """The command is legitimate; only its timeout is unreasonable."""
    guard = _guard(tmp_path)
    decision = guard.before_tool(1, "run_shell", {"command": "sleep 9999", "timeout": 3600})

    assert decision.action == "rewrite"
    assert decision.payload["timeout"] == 300
    assert decision.payload["command"] == "sleep 9999"


def test_reasonable_timeout_is_left_alone(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    assert guard.before_tool(1, "run_shell", {"command": "pytest", "timeout": 60}).action == "allow"


# ------------------------------------------------------------------ budget


def test_budget_ceiling_interrupts_between_turns(tmp_path: Path) -> None:
    spend = {"usd": 0.0}
    guard = Guardrails(
        workspace=tmp_path,
        config=GuardConfig(enabled=True, step_budget_usd=1.0),
        cost_reader=lambda: spend["usd"],
    )
    assert guard.before_turn(1) is None

    spend["usd"] = 1.5
    interrupt = guard.before_turn(2)
    assert interrupt is not None
    assert interrupt.kind == "budget_ceiling"
    assert interrupt.failure_kind == "budget"


# ------------------------------------------------------------------ fail-open


def test_a_broken_guard_disables_itself_and_the_run_continues(tmp_path: Path) -> None:
    """A bug in a safety check must not become a new way for runs to die."""
    seen = []
    guard = Guardrails(
        workspace=tmp_path,
        config=GuardConfig(enabled=True),
        on_event=lambda kind, **p: seen.append(kind),
    )

    def explode(*a, **k):
        raise RuntimeError("guard is broken")

    guard._inspect = explode
    for _ in range(4):
        assert guard.before_tool(1, "write_file", {"path": "x.py", "content": "y"}).action == "allow"

    assert seen.count("guard.error") == 3
    assert "guard.disabled" in seen
    assert guard.report.disabled is True


# ------------------------------------------------------------------ worker loop


def test_veto_reaches_the_model_as_a_tool_result(tmp_path: Path) -> None:
    """A model not told why it was stopped will simply try again."""
    from taste import cores

    llm = FakeLLM(
        [
            FakeTurn(tool_calls=[("run_shell", {"command": "git commit -am wip"})]),
            FakeTurn(text="understood, I will not touch git"),
        ]
    )
    result = cores.execute(
        llm,
        spec=_spec(),
        step=_step(),
        plan_context="",
        tools=_tools(tmp_path),
        hook=_guard(tmp_path),
    )

    assert result.vetoes == 1
    assert "BLOCKED" in llm.last_user_content()[0]["content"]
    assert "git commit" in llm.last_user_content()[0]["content"]


def test_vetoed_command_never_executes(tmp_path: Path) -> None:
    from taste import cores

    llm = FakeLLM(
        [
            FakeTurn(tool_calls=[("write_file", {"path": ".git/hooks/pre-commit", "content": "evil"})]),
            FakeTurn(text="ok"),
        ]
    )
    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    cores.execute(
        llm,
        spec=_spec(),
        step=_step(),
        plan_context="",
        tools=_tools(tmp_path),
        hook=_guard(tmp_path),
    )
    assert not (tmp_path / ".git" / "hooks" / "pre-commit").exists()


def test_rewrite_changes_what_the_tool_receives(tmp_path: Path) -> None:
    from taste import cores

    llm = FakeLLM(
        [
            FakeTurn(tool_calls=[("run_shell", {"command": "echo hi", "timeout": 9999})]),
            FakeTurn(text="done"),
        ]
    )
    result = cores.execute(
        llm,
        spec=_spec(),
        step=_step(),
        plan_context="",
        tools=_tools(tmp_path),
        hook=_guard(tmp_path),
    )
    # Ran successfully with the clamped timeout rather than being refused.
    assert result.tool_calls == 1
    assert result.vetoes == 0
    assert "hi" in llm.last_user_content()[0]["content"]


def test_interrupt_stops_the_loop_and_is_reported(tmp_path: Path) -> None:
    from taste import cores

    spend = {"usd": 5.0}
    guard = Guardrails(
        workspace=tmp_path,
        config=GuardConfig(enabled=True, step_budget_usd=0.01),
        cost_reader=lambda: spend["usd"],
    )
    guard._start_usd = 0.0

    llm = FakeLLM([FakeTurn(text="never reached")])
    result = cores.execute(
        llm, spec=_spec(), step=_step(), plan_context="", tools=_tools(tmp_path), hook=guard
    )

    assert llm.call_count == 0, "the model must not be called after the ceiling trips"
    assert result.interrupt is not None
    assert result.stopped_reason == "interrupted"


def test_no_hook_means_no_behavior_change(tmp_path: Path) -> None:
    """Guardrails off: the loop is exactly what it was before they existed."""
    from taste import cores

    llm = FakeLLM(
        [
            FakeTurn(tool_calls=[("run_shell", {"command": "git commit -am wip"})]),
            FakeTurn(text="done"),
        ]
    )
    result = cores.execute(
        llm, spec=_spec(), step=_step(), plan_context="", tools=_tools(tmp_path)
    )
    assert result.vetoes == 0
    assert result.tool_calls == 1


# ------------------------------------------------------------------ interrupt -> diagnosis


def test_an_interrupt_halts_instead_of_triggering_a_retry(refactor_workspace: Path) -> None:
    """The wiring that stops a budget guard becoming a retry trigger.

    Without it the trap handler sees an unclassifiable FAIL, falls back to
    the baseline action, and re-runs the very worker the guard just killed.
    """
    from taste.cores import Interrupt, Plan, WorkerResult
    from taste.kernel import Kernel

    ws = refactor_workspace
    plan = Plan(
        task="interrupted",
        steps=[Step("step-01", "spend it all", Verification(kind="shell", command="test -f x.py"))],
    )

    def worker(step, plan_):
        return WorkerResult(
            summary="killed",
            tool_calls=0,
            stopped_reason="interrupted",
            interrupt=Interrupt(kind="budget_ceiling", detail="over", turn=1, failure_kind="budget"),
        )

    events = []
    Kernel(
        workspace=ws,
        max_retries=3,
        on_event=events.append,
        recovery_config=RecoveryConfig.arm("tiered"),
    ).run(task="i", spec=_spec(), session_id="i", plan_override=plan, worker_override=worker)

    diagnoses = [e for e in events if e.kind == "recovery.diagnosis"]
    actions = [e for e in events if e.kind == "recovery.action"]
    assert diagnoses[0].payload["failure_class"] == FailureClass.INTERRUPTED.value
    assert actions[0].payload["action"] == ActionKind.HALT.value
    # One worker invocation, not four.
    assert len([e for e in events if e.kind == "worker.done"]) == 1
