"""How densely a run is observed, and why that is a protocol question.

A regression is a PASS->FAIL transition between *consecutive* observations, so
the grid decides what is detectable at all. Measured on a real 5-instance
sweep, the per-attempt grid produced **10 observations across 5 runs** -- five
adjacent pairs in which a transition could be seen -- against 70 tool calls.

Worse than low power, that grid is treatment-dependent: an arm that retries
emits up to three observations per step where a no-retry arm emits one, so the
recovering arm samples its own timeline more finely purely by recovering. The
finer grid exists to remove that bias, and these tests pin both halves: that
it is genuinely finer, and that turning it off changes nothing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from taste.config import HarnessConfig, kernel_kwargs
from taste.kernel import Kernel, _ObservingHook
from taste.shadow import load_timeline
from tests.golden import rollback_scenario
from tests.test_golden_baseline import EXPECTED_EVENTS

# ------------------------------------------------------------------ the flag


def test_the_grid_is_off_by_default() -> None:
    """It changes a pre-registered quantity, so it cannot be a default."""
    assert HarnessConfig().observe_tools is False
    assert all(not HarnessConfig.arm(a).observe_tools for a in HarnessConfig.arm_names())


def test_the_flag_reaches_the_kernel() -> None:
    assert kernel_kwargs(HarnessConfig(observe_tools=True))["observe_tools"] is True
    assert Kernel(
        workspace=Path("."), config=HarnessConfig(observe_tools=True)
    ).observe_tools is True


def test_the_frozen_signature_is_unchanged_when_off(refactor_workspace: Path) -> None:
    """The ablation argument rests on this: with the grid off, a run must be
    byte-identical to the baseline it was frozen against."""
    signature = rollback_scenario(refactor_workspace).run(
        refactor_workspace, config=HarnessConfig(observe_tools=False)
    )
    assert signature.events == EXPECTED_EVENTS


# ------------------------------------------------------------------ the hook


class _Recorder:
    """An inner hook that records what it saw, to prove composition."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def before_turn(self, turn):
        self.calls.append("before_turn")

    def before_tool(self, turn, name, payload):
        self.calls.append(f"before_tool:{name}")
        from taste.cores import ALLOW

        return ALLOW

    def after_tool(self, turn, name, payload, output, elapsed_s, decision):
        self.calls.append(f"after_tool:{name}")

    def after_turn(self, turn, message, stop_reason):
        self.calls.append("after_turn")


def test_the_observer_wraps_a_guard_rather_than_replacing_it() -> None:
    """Guardrails are arm-defining. An observer bolted onto them would fire
    only in arms that guard, making density follow the treatment -- the exact
    bias the finer grid removes."""
    seen: list[str] = []
    inner = _Recorder()
    hook = _ObservingHook(observe=seen.append, inner=inner)

    hook.before_tool(1, "write_file", {})
    hook.after_tool(1, "write_file", {}, "ok", 0.1, None)

    assert inner.calls == ["before_tool:write_file", "after_tool:write_file"]
    assert seen == ["write_file"], "the observation must still happen"


def test_the_observer_works_with_no_inner_hook() -> None:
    """Arms that disable guardrails must still be observed."""
    from taste.cores import ALLOW

    seen: list[str] = []
    hook = _ObservingHook(observe=seen.append, inner=None)

    assert hook.before_tool(1, "t", {}) is ALLOW
    hook.after_tool(1, "t", {}, "out", 0.0, None)
    assert seen == ["t"]


def test_observation_runs_after_the_inner_hook() -> None:
    """A veto must already be recorded before the tree is sampled, so the
    sample reflects what happened rather than what was attempted."""
    order: list[str] = []

    class _Inner(_Recorder):
        def after_tool(self, *a, **k):
            order.append("inner")

    hook = _ObservingHook(observe=lambda _n: order.append("observe"), inner=_Inner())
    hook.after_tool(1, "t", {}, "o", 0.0, None)
    assert order == ["inner", "observe"]


def test_a_raising_observer_never_halts_the_run() -> None:
    """An instrument that can stop the thing it measures is worse than none."""
    def explode(_name: str) -> None:
        raise RuntimeError("observer is broken")

    hook = _ObservingHook(observe=explode, inner=_Recorder())
    hook.after_tool(1, "t", {}, "o", 0.0, None)  # must not raise


def test_a_raising_inner_hook_does_not_stop_observation() -> None:
    class _Broken(_Recorder):
        def after_tool(self, *a, **k):
            raise RuntimeError("guard is broken")

    seen: list[str] = []
    _ObservingHook(observe=seen.append, inner=_Broken()).after_tool(
        1, "t", {}, "o", 0.0, None
    )
    assert seen == ["t"], "a broken guard must not blind the instrument"


# ------------------------------------------------------------------ density


def _run_with_tools(workspace: Path, *, observe_tools: bool, session: str) -> list:
    """A real worker loop that issues real tool calls.

    Deliberately NOT the golden scenario: that one uses ``worker_override``
    and never calls a tool, so it cannot distinguish the two grids at all --
    which is exactly how the first version of this test managed to pass while
    proving nothing.
    """
    from taste.agent import AgentSpec
    from tests.fakes import FakeLLM, FakeTurn, plan_turn, verdict_turn

    llm = FakeLLM([
        plan_turn([{"id": "step-01", "description": "edit files",
                    "verification": {"kind": "shell", "command": "true"}}]),
        FakeTurn(tool_calls=[("write_file", {"path": "a.py", "content": "a = 1\n"})]),
        FakeTurn(tool_calls=[("write_file", {"path": "b.py", "content": "b = 2\n"})]),
        FakeTurn(tool_calls=[("write_file", {"path": "c.py", "content": "c = 3\n"})]),
        FakeTurn(text="done"),
        verdict_turn(passed=True),
    ])
    kernel = Kernel(
        workspace=workspace, llm=llm,
        config=HarnessConfig(shadow=True, observe_tools=observe_tools),
    )
    kernel.run(
        task="edit files",
        spec=AgentSpec(name="s", description="", system_prompt="p"),
        session_id=session,
    )
    return list(load_timeline(Path(workspace) / ".git" / "taste", session))


def _fresh_repo(root: Path, name: str) -> Path:
    ws = root / name
    ws.mkdir(parents=True)
    (ws / "seed.py").write_text("x = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t.co", "commit", "-qm", "base"],
        cwd=ws, check=True,
    )
    return ws


def test_the_finer_grid_actually_adds_observation_points(tmp_path: Path) -> None:
    """The whole point of the flag, on a run that really calls tools.

    Three mutating tool calls inside one step. The coarse grid sees the step
    boundary only; the fine grid must see each edit, because a regression is a
    transition BETWEEN observations and a step boundary hides everything that
    happened inside it.
    """
    coarse = _run_with_tools(_fresh_repo(tmp_path, "coarse"),
                             observe_tools=False, session="coarse")
    fine = _run_with_tools(_fresh_repo(tmp_path, "fine"),
                           observe_tools=True, session="fine")

    assert not [c for c in coarse if c.trigger == "tool"]
    tool_points = [c for c in fine if c.trigger == "tool"]
    assert len(tool_points) >= 3, f"expected one per mutating tool call, got {len(tool_points)}"
    assert len(fine) > len(coarse), (
        f"the finer grid must add points: {len(fine)} vs {len(coarse)}"
    )


def test_each_tool_observation_names_the_tool(tmp_path: Path) -> None:
    """Without the name the finer points are anonymous, and the granularity
    invariance check cannot say which grid produced which timeline."""
    fine = _run_with_tools(_fresh_repo(tmp_path, "named"),
                           observe_tools=True, session="named")
    tools = [c.tool for c in fine if c.trigger == "tool"]
    assert tools and all(t == "write_file" for t in tools), tools


def test_tool_observations_carry_the_right_attempt(tmp_path: Path) -> None:
    """One hook is built per step and outlives every attempt, so a captured
    attempt number would stamp every observation with the first one."""
    fine = _run_with_tools(_fresh_repo(tmp_path, "attempt"),
                           observe_tools=True, session="attempt")
    for c in (c for c in fine if c.trigger == "tool"):
        assert c.attempt >= 1
        assert c.step_id == "step-01"
