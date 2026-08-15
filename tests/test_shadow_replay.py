"""The measurement instrument: observe, then reconstruct.

The claim being tested is that a regression can be located *after the fact*,
including in an arm that never checkpointed, never rolled back and never
noticed — because that is precisely the arm a self-verification baseline
gives you, and the whole study depends on being able to measure it.

Everything here is hermetic. Reconstruction costs no API tokens, and the scan
is exhaustive by design: the arms under study produce non-monotone verdict
sequences on purpose, so a search that assumes monotonicity cannot see the
very recoveries the study is about.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from taste.agent import AgentSpec
from taste.config import HarnessConfig
from taste.cores import Plan, Step, Verification, WorkerResult
from taste.kernel import Kernel
from taste.memory import Memory
from taste.replay import Probe, episodes_from, reconstruct
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


# ------------------------------------------------------------------ episodes


def _verdicts(*seq: str) -> list[str]:
    return list(seq)


def _fake_timeline(n: int):
    from taste.shadow import ShadowCommit

    return [
        ShadowCommit(seq=i, sha=f"sha{i:02d}", session="s", step_id="s", attempt=1,
                     trigger="worker", cost_work_usd=float(i))
        for i in range(1, n + 1)
    ]


def test_a_recovered_regression_is_recorded() -> None:
    """The bug that would have inverted the headline result.

    The previous implementation bisected for a probe failing at the END of a
    run, so a regression that was repaired left no trace. That is arm-
    dependent in the worst possible way: the arm whose entire claim is that it
    recovers would have shown the fewest regressions and no recovery rate.
    """
    timeline = _fake_timeline(6)
    episodes, _unknown, ever = episodes_from(
        _verdicts("pass", "pass", "fail", "fail", "pass", "pass"), timeline, "p"
    )
    assert ever is True
    assert len(episodes) == 1
    assert episodes[0].onset_seq == 3
    assert episodes[0].recovered_seq == 5
    assert episodes[0].recovered is True


def test_multiple_episodes_per_probe() -> None:
    """A probe can break, be fixed, and break again."""
    timeline = _fake_timeline(7)
    episodes, _u, _e = episodes_from(
        _verdicts("pass", "fail", "pass", "fail", "fail", "pass", "pass"), timeline, "p"
    )
    assert [(e.onset_seq, e.recovered_seq) for e in episodes] == [(2, 3), (4, 6)]


def test_an_unrecovered_regression_stays_open() -> None:
    timeline = _fake_timeline(4)
    episodes, _u, _e = episodes_from(
        _verdicts("pass", "pass", "fail", "fail"), timeline, "p"
    )
    assert len(episodes) == 1 and episodes[0].recovered is False


def test_a_probe_that_never_passed_is_not_a_regression() -> None:
    """Nothing broke — the task was simply never done.

    This is why a from-scratch benchmark cannot host this construct: every
    check fails at observation one, so 100% of probes would read as regressions.
    """
    timeline = _fake_timeline(4)
    episodes, _u, ever = episodes_from(
        _verdicts("fail", "fail", "fail", "fail"), timeline, "p"
    )
    assert ever is False
    assert episodes == []


def test_pass_to_skip_is_unknown_not_maintained() -> None:
    """A skipped test is not a passing one; counting it as maintained would
    understate contamination."""
    timeline = _fake_timeline(3)
    episodes, unknown, _e = episodes_from(_verdicts("pass", "skip", "pass"), timeline, "p")
    assert unknown == 1
    assert episodes == []


def test_errors_are_missing_observations_not_evidence() -> None:
    """A probe that could not run says nothing about the tree."""
    timeline = _fake_timeline(4)
    episodes, _u, _e = episodes_from(
        _verdicts("pass", "error", "error", "pass"), timeline, "p"
    )
    assert episodes == []


def test_recovery_rate_is_none_when_there_were_no_regressions(
    refactor_workspace: Path,
) -> None:
    """An empty run is not a 0% recovery rate."""
    ws = refactor_workspace
    (ws / "lib.py").write_text("def value():\n    return 1\n")
    memory, log = _shadow(ws, "norr")
    log.observe(step_id="s", attempt=1, trigger="run")

    report = reconstruct(memory, list(log.timeline()), [PROBE], session="norr")
    assert report.recovery_rate is None


def test_detection_is_flagged_unattributed_without_coverage_data(
    refactor_workspace: Path,
) -> None:
    """Co-occurrence is not attribution.

    Crediting a harness with detecting THIS regression because it reported
    SOME failure afterwards biases toward arms that fail more often.
    """
    ws = refactor_workspace
    memory, log = _timeline_with_regression(ws)
    timeline = list(log.timeline())

    report = reconstruct(
        memory, timeline, [PROBE], harness_failed_at={timeline[-1].seq}, session="regress"
    )
    assert report.episodes
    episode = report.episodes[0]
    assert episode.detected_seq == timeline[-1].seq
    assert episode.attributed is False, "must be marked an upper bound"


def test_attribution_rejects_an_unrelated_failure(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    memory, log = _timeline_with_regression(ws)
    timeline = list(log.timeline())
    last = timeline[-1].seq

    report = reconstruct(
        memory, timeline, [PROBE],
        harness_failed_at={last},
        attribution={last: {"some_other_probe"}},
        session="regress",
    )
    assert report.episodes[0].silent is True, "an unrelated failure is not detection"


def test_wasted_work_spans_onset_to_close(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    memory, log = _timeline_with_regression(ws)
    report = reconstruct(memory, list(log.timeline()), [PROBE], session="regress")
    assert report.episodes[0].wasted_work_usd >= 0.0
    assert report.wasted_work_usd == report.episodes[0].wasted_work_usd


def test_the_scan_is_exhaustive(refactor_workspace: Path) -> None:
    """One replay per observation: no monotonicity assumed anywhere."""
    ws = refactor_workspace
    memory, log = _timeline_with_regression(ws)
    timeline = list(log.timeline())

    report = reconstruct(memory, timeline, [PROBE], session="regress")
    assert report.replays == len(timeline)
