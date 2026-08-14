"""The measurement instrument: observe, then reconstruct.

The claim being tested is that a regression can be located *after the fact*,
including in an arm that never checkpointed, never rolled back and never
noticed — because that is precisely the arm a self-verification baseline
gives you, and the whole study depends on being able to measure it.

Everything here is hermetic. Reconstruction costs no API tokens; its expense
is wall-clock, which is what the bisection exists to bound.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from taste.agent import AgentSpec
from taste.config import HarnessConfig
from taste.cores import Plan, Step, Verification, WorkerResult
from taste.kernel import Kernel
from taste.memory import Memory
from taste.replay import Probe, Replayer, reconstruct
from taste.shadow import SHADOW_PREFIX, ShadowLog, load_timeline
from tests.golden import rollback_scenario
from tests.test_golden_baseline import EXPECTED_EVENTS


def _spec() -> AgentSpec:
    return AgentSpec(name="s", description="", system_prompt="p")


def _shadow(ws: Path, session: str = "sh") -> tuple[Memory, ShadowLog]:
    memory = Memory.open_session(ws, session)
    log = ShadowLog(memory, gitdir=Path(memory.repo.git_dir) / "taste", session=session)
    return memory, log


# ------------------------------------------------------------------ ablation


def test_shadow_disabled_reproduces_the_baseline(refactor_workspace: Path) -> None:
    sig = rollback_scenario(refactor_workspace).run(refactor_workspace, shadow=False)
    assert sig.events == EXPECTED_EVENTS


def test_shadow_enabled_only_adds_shadow_events(refactor_workspace: Path) -> None:
    """An instrument that changes the run it measures is not an instrument."""
    ws = refactor_workspace
    sig = rollback_scenario(ws).run(ws, shadow=True)

    non_shadow = tuple(e for e in sig.events if not e[0].startswith("shadow."))
    assert non_shadow == EXPECTED_EVENTS
    assert any(e[0] == "shadow.observe" for e in sig.events)
    # And the agent-visible outcome is unchanged.
    assert sig.step_results == (("step-01", True, 2, True), ("step-02", True, 1, False))


def test_shadow_does_not_touch_the_session_branch(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    rollback_scenario(ws).run(ws, shadow=True)

    memory = Memory(ws, "taste/session-golden")
    subjects = [s for _sha, s in memory.commit_subjects()]
    assert not any("shadow" in s for s in subjects), "shadow commits must not be on the branch"
    # They exist, just elsewhere.
    assert memory.list_refs(f"{SHADOW_PREFIX}/")


# ------------------------------------------------------------------ observation


def test_observation_records_only_real_changes(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    _memory, log = _shadow(ws)

    first = log.observe(step_id="step-01", attempt=1, trigger="worker")
    assert first is not None, "the initial tree is an observation"

    # Nothing changed: a second observation would be noise.
    assert log.observe(step_id="step-01", attempt=1, trigger="worker") is None

    (ws / "new.py").write_text("x = 1\n")
    third = log.observe(step_id="step-01", attempt=2, trigger="worker")
    assert third is not None
    assert "new.py" in third.files


def test_observations_carry_cumulative_cost(refactor_workspace: Path) -> None:
    """Latency is measured in dollars, so each point must carry a running total."""
    ws = refactor_workspace
    memory = Memory.open_session(ws, "cost")
    spend = {"billed": 0.0, "work": 0.0}
    log = ShadowLog(
        memory,
        gitdir=Path(memory.repo.git_dir) / "taste",
        session="cost",
        cost_reader=lambda: (spend["billed"], spend["work"]),
    )
    log.observe(step_id="s", attempt=1, trigger="worker")
    spend["billed"], spend["work"] = 0.25, 0.40
    (ws / "a.py").write_text("a = 1\n")
    second = log.observe(step_id="s", attempt=2, trigger="worker")

    assert second is not None
    assert second.cost_billed_usd == 0.25
    assert second.cost_work_usd == 0.40


def test_timeline_survives_to_disk(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    memory, log = _shadow(ws, "disk")
    log.observe(step_id="s", attempt=1, trigger="run")
    (ws / "b.py").write_text("b = 1\n")
    log.observe(step_id="s", attempt=1, trigger="worker")

    loaded = load_timeline(Path(memory.repo.git_dir) / "taste", "disk")
    assert [c.seq for c in loaded] == [1, 2]
    assert loaded[1].files == ("b.py",)


def test_a_broken_shadow_never_fails_the_run(refactor_workspace: Path, monkeypatch) -> None:
    ws = refactor_workspace
    monkeypatch.setattr(
        ShadowLog, "_write_tree_commit", lambda self: (_ for _ in ()).throw(RuntimeError("nope"))
    )
    sig = rollback_scenario(ws).run(ws, shadow=True)
    assert sig.status == "completed"


# ------------------------------------------------------------------ reconstruction


def _timeline_with_regression(ws: Path) -> tuple[Memory, ShadowLog]:
    """Six observations; a helper silently breaks at the fourth."""
    (ws / "lib.py").write_text("def value():\n    return 1\n")
    memory, log = _shadow(ws, "regress")
    log.observe(step_id="step-00", attempt=1, trigger="run")

    for i, content in enumerate(
        [
            "def value():\n    return 1\n\n\ndef a():\n    return 'a'\n",
            "def value():\n    return 1\n\n\ndef a():\n    return 'a'\n\n\ndef b():\n    return 'b'\n",
            # Here it breaks: value() now returns 2.
            "def value():\n    return 2\n\n\ndef a():\n    return 'a'\n\n\ndef b():\n    return 'b'\n",
            "def value():\n    return 2\n\n\ndef a():\n    return 'a'\n\n\ndef b():\n    return 'b'\n\n\ndef c():\n    return 'c'\n",
        ],
        start=1,
    ):
        (ws / "lib.py").write_text(content)
        log.observe(step_id=f"step-{i:02d}", attempt=1, trigger="worker")
    return memory, log


PROBE = Probe(
    name="value_is_one",
    command="python -c \"import lib; assert lib.value() == 1\"",
)


def test_bisection_finds_the_exact_onset(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    memory, log = _timeline_with_regression(ws)

    report = reconstruct(memory, list(log.timeline()), [PROBE], session="regress")

    assert report.contaminated
    regression = report.regressions[0]
    # The break was introduced at the 4th observation (1 baseline + 3 edits).
    assert regression.onset_seq == 4
    assert memory.show(regression.onset_sha, "lib.py").count("return 2") == 1


def test_a_silent_regression_is_reported_as_silent(refactor_workspace: Path) -> None:
    """The case the study exists for: nobody noticed, and we can still see it."""
    ws = refactor_workspace
    memory, log = _timeline_with_regression(ws)

    report = reconstruct(
        memory, list(log.timeline()), [PROBE], harness_failed_at=set(), session="regress"
    )
    regression = report.regressions[0]
    assert regression.silent is True
    assert regression.detected_seq is None


def test_detection_latency_is_measured_in_dollars_and_observations(
    refactor_workspace: Path,
) -> None:
    ws = refactor_workspace
    memory, log = _timeline_with_regression(ws)

    # The harness reported failure at the last observation.
    last = log.timeline()[-1].seq
    report = reconstruct(
        memory, list(log.timeline()), [PROBE], harness_failed_at={last}, session="regress"
    )
    regression = report.regressions[0]

    assert regression.silent is False
    assert regression.detected_seq == last
    assert regression.observations_to_detect == last - regression.onset_seq
    assert regression.cost_to_detect_usd is not None


def test_bisection_costs_far_fewer_replays_than_a_scan(refactor_workspace: Path) -> None:
    """Wall-clock is the real budget here; the log-vs-linear gap is the point."""
    ws = refactor_workspace
    (ws / "lib.py").write_text("def value():\n    return 1\n")
    memory, log = _shadow(ws, "wide")
    log.observe(step_id="s", attempt=1, trigger="run")
    for i in range(1, 33):
        broken = i >= 20
        (ws / "lib.py").write_text(
            f"def value():\n    return {2 if broken else 1}\n\n\n# edit {i}\n"
        )
        log.observe(step_id=f"step-{i:02d}", attempt=1, trigger="worker")

    timeline = list(log.timeline())
    report = reconstruct(memory, timeline, [PROBE], session="wide")

    # seq 1 is the baseline observation, so edit i lands at seq i+1; the
    # first broken edit is i=20.
    assert report.regressions[0].onset_seq == 21
    assert report.replays < len(timeline) / 2, (
        f"{report.replays} replays over {len(timeline)} observations — "
        "bisection is not being used"
    )


def test_a_healthy_run_reports_no_regression(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    (ws / "lib.py").write_text("def value():\n    return 1\n")
    memory, log = _shadow(ws, "clean")
    log.observe(step_id="s", attempt=1, trigger="run")
    (ws / "extra.py").write_text("# harmless\n")
    log.observe(step_id="s", attempt=2, trigger="worker")

    report = reconstruct(memory, list(log.timeline()), [PROBE], session="clean")
    assert not report.contaminated
    assert report.final_verdicts["value_is_one"] == "pass"


def test_a_probe_broken_from_the_start_is_located_at_zero(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    (ws / "lib.py").write_text("def value():\n    return 99\n")
    memory, log = _shadow(ws, "prebroken")
    log.observe(step_id="s", attempt=1, trigger="run")
    (ws / "more.py").write_text("# x\n")
    log.observe(step_id="s", attempt=2, trigger="worker")

    report = reconstruct(memory, list(log.timeline()), [PROBE], session="prebroken")
    assert report.regressions[0].onset_seq == log.timeline()[0].seq


def test_verdicts_are_memoized_across_bisection(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    memory, log = _timeline_with_regression(ws)
    replayer = Replayer(memory, [PROBE])
    timeline = list(log.timeline())

    replayer.find_onset(timeline, PROBE)
    first_count = replayer.replays
    replayer.find_onset(timeline, PROBE)
    assert replayer.replays == first_count, "a repeated search must cost nothing"


def test_replay_leaves_the_workspace_untouched(refactor_workspace: Path) -> None:
    """Probes run in throwaway worktrees; they must not leak into the run."""
    ws = refactor_workspace
    memory, log = _timeline_with_regression(ws)
    before = (ws / "lib.py").read_text()

    reconstruct(memory, list(log.timeline()), [PROBE], session="regress")

    assert (ws / "lib.py").read_text() == before
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ws, capture_output=True, text=True
    )
    assert "probe" not in porcelain.stdout


# ------------------------------------------------------------------ end to end


def test_a_self_verifying_arm_still_yields_a_timeline(refactor_workspace: Path) -> None:
    """The arm with no checkpoints of its own is the one that most needs
    uniform instrumentation — otherwise it has no timeline to compare."""
    ws = refactor_workspace
    plan = Plan(
        task="a1",
        steps=[Step("step-01", "edit", Verification(kind="shell", command="test -f made.py"))],
    )

    def worker(step, plan_):
        (ws / "made.py").write_text("# done\n")
        return WorkerResult("done", 1, "end_turn")

    result = Kernel(workspace=ws, config=HarnessConfig.arm("A1")).run(
        task="a1", spec=_spec(), session_id="a1", plan_override=plan, worker_override=worker
    )
    assert result.status == "completed"

    memory = Memory(ws, "taste/session-a1")
    timeline = load_timeline(Path(memory.repo.git_dir) / "taste", "a1")
    assert len(timeline) >= 2, "even a no-checkpoint arm gets observation points"
