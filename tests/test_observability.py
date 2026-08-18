"""The run, made visible.

Nothing here is on the measurement path, so the bar is different from the rest
of the suite: the job is that the picture is *faithful* and that it still
draws when the run went badly. A dashboard that refuses to render the run you
most need to look at — the one that crashed — is worthless, so the degraded
cases get as much attention as the happy one.

The one property that is not cosmetic: a Monitor verdict must land on the
observation whose tree it actually graded. Getting that wrong would draw a
silent regression as a detected one, or the reverse.
"""

from __future__ import annotations

from pathlib import Path

from taste.observability import (
    build_trace,
    load_shadow,
    render_branches,
    render_dag,
    render_lanes,
    render_models,
    render_regression_timeline,
)


def _ev(kind: str, ts: float, **payload: object) -> dict:
    return {"kind": kind, "ts": ts, "payload": payload}


PLAN = {
    "steps": [
        {"id": "step-01", "description": "first", "depends_on": []},
        {"id": "step-02", "description": "second", "depends_on": ["step-01"]},
    ]
}

EVENTS = [
    _ev("run.start", 0.0, session="s1", task="do a thing", branch="taste/session-s1"),
    _ev("run.manifest", 0.1, harness_git_sha="abc123",
        models={"planner": "gpt-5.6-terra", "worker": "claude-opus-4-7", "monitor": "claude-haiku-4-5"},
        temperature={"planner": 0.0, "worker": 0.2, "monitor": 0.0}),
    _ev("plan.ready", 0.2, steps=2, waves=2, parallel_waves=0),
    _ev("wave.begin", 0.3, steps=["step-01"], size=1),
    _ev("worktree.open", 0.35, branch="taste/step-01", id="step-01"),
    _ev("step.begin", 0.4, id="step-01", attempt=1),
    _ev("worker.done", 1.0, id="step-01", tools=3, stop="end_turn"),
    _ev("monitor.verdict", 1.1, id="step-01", attempt=1, passed=False, reason="broke", sha="aaa"),
    _ev("recovery.diagnosis", 1.2, id="step-01", failure_class="verification_failed"),
    _ev("recovery.action", 1.3, id="step-01", action="ROLLBACK_AND_RETRY"),
    _ev("step.rollback", 1.4, id="step-01", to="aaa", remaining_retries=1),
    _ev("step.begin", 1.5, id="step-01", attempt=2),
    _ev("monitor.verdict", 2.0, id="step-01", attempt=2, passed=True, reason="ok", sha="bbb"),
    _ev("worktree.merge", 2.1, branch="taste/step-01", id="step-01"),
    _ev("wave.done", 2.2, steps=["step-01"], size=1),
    _ev("run.done", 3.0, status="completed", elapsed=3.0, cost_usd=0.5,
        cache_hit_rate=0.4, reason=None, failure_kind=None),
]

SHADOW = [
    {"seq": 1, "sha": "s1", "session": "s1", "step_id": "step-01", "attempt": 1,
     "trigger": "worker", "files": ["a.py"], "cost_work_usd": 0.1},
    {"seq": 2, "sha": "s2", "session": "s1", "step_id": "step-01", "attempt": 2,
     "trigger": "worker", "files": ["a.py"], "cost_work_usd": 0.2},
    {"seq": 3, "sha": "s3", "session": "s1", "step_id": "step-02", "attempt": 1,
     "trigger": "worker", "files": ["b.py"], "cost_work_usd": 0.3},
]

EVIDENCE = {
    "episodes": [
        {"probe": "t::silent", "onset_seq": 2, "recovered_seq": None,
         "detected_seq_attributed": None, "detected_seq_any": None},
        {"probe": "t::caught", "onset_seq": 1, "recovered_seq": 3,
         "detected_seq_attributed": 1, "detected_seq_any": 1},
    ]
}


# ------------------------------------------------------------------ trace


def test_the_manifest_gives_every_role_its_model() -> None:
    """Three roles can run three different models; a single "model" field
    would erase the distinction the arms are defined over."""
    trace = build_trace(EVENTS, plan=PLAN)
    by_role = {m.role: m.model for m in trace.models}
    assert by_role["planner"] == "gpt-5.6-terra"
    assert by_role["worker"] == "claude-opus-4-7"
    assert by_role["monitor"] == "claude-haiku-4-5"


def test_threads_are_separated_not_flattened() -> None:
    """A worker branch, the Monitor judging it and a recovery handler deciding
    what to do next are three threads; one chronological list hides exactly
    the concurrency the architecture provides."""
    trace = build_trace(EVENTS, plan=PLAN)
    assert {"worker", "monitor", "recovery", "merge"} <= set(trace.lanes)


def test_recovery_actions_are_attached_to_their_step() -> None:
    trace = build_trace(EVENTS, plan=PLAN)
    step = next(s for s in trace.steps if s.step_id == "step-01")
    assert step.recovery_actions == ("ROLLBACK_AND_RETRY",)
    assert step.rolled_back is True
    assert step.attempts == 2
    assert step.passed is True, "the last verdict wins, not the first"


def test_a_branch_is_followed_from_open_to_merge() -> None:
    trace = build_trace(EVENTS, plan=PLAN)
    assert [(b.name, b.outcome) for b in trace.branches] == [("taste/step-01", "merged")]


def test_a_conflicted_branch_is_not_reported_as_merged() -> None:
    events = [
        _ev("worktree.open", 0.0, branch="taste/x", id="step-01"),
        _ev("worktree.conflict", 1.0, step="step-01", detail="both modified"),
    ]
    assert build_trace(events).branches[0].outcome == "conflict"


# ------------------------------------------------------------------ the join


def test_a_verdict_lands_on_the_tree_it_graded() -> None:
    """Not on the nearest timestamp, and not on event order. Both verdicts
    here belong to step-01 but to different attempts, so an order-based
    placement would put them on the same observation."""
    trace = build_trace(EVENTS, plan=PLAN, shadow=SHADOW, evidence=EVIDENCE)
    assert trace.verdict_seqs == [(1, False), (2, True)]


def test_a_verdict_with_no_observation_is_simply_not_placed() -> None:
    """The worker changed nothing, so no shadow commit exists. Inventing a
    position would draw a detection that never happened."""
    events = [_ev("monitor.verdict", 1.0, id="step-99", attempt=1, passed=False)]
    assert build_trace(events, shadow=SHADOW).verdict_seqs == []


# ------------------------------------------------------------------ drawing


def test_the_timeline_distinguishes_silent_from_detected() -> None:
    trace = build_trace(EVENTS, plan=PLAN, shadow=SHADOW, evidence=EVIDENCE)
    svg = render_regression_timeline(trace)

    assert "SILENT" in svg
    assert "detected at 1" in svg
    assert "1 silent" in svg, "the count must be visible without hovering"
    assert "recovered at 3" in svg


def test_every_panel_draws_without_a_measured_run() -> None:
    """Most runs are not measured. The dashboard must still be useful."""
    trace = build_trace(EVENTS, plan=PLAN)
    assert "<svg" in render_dag(trace)
    assert "<svg" in render_lanes(trace)
    assert "not measured" in render_regression_timeline(trace)


def test_the_dag_shows_dependencies_and_verdicts() -> None:
    svg = render_dag(build_trace(EVENTS, plan=PLAN))
    assert "step-01" in svg and "step-02" in svg
    assert "<path" in svg, "the declared dependency must be drawn as an edge"
    assert "rollback" in svg


def test_panels_survive_a_crashed_run() -> None:
    """The run you most need to look at is the one that died mid-flight."""
    partial = EVENTS[:6]
    trace = build_trace(partial, plan=PLAN)
    for svg in (render_dag(trace), render_lanes(trace), render_regression_timeline(trace)):
        assert svg
    assert render_models(trace)
    assert render_branches(trace)


def test_nothing_at_all_still_renders() -> None:
    trace = build_trace([])
    assert "No events" in render_lanes(trace)
    assert "No plan" in render_dag(trace)
    assert "No model manifest" in render_models(trace)


def test_labels_are_escaped() -> None:
    """Task text and branch names reach the page; they are not trusted."""
    events = [_ev("run.start", 0.0, session="s", task="<script>alert(1)</script>", branch="b"),
              _ev("worktree.open", 0.1, branch="<img src=x onerror=1>", id="s1")]
    assert "<script>" not in render_branches(build_trace(events))
    assert "&lt;img" in render_branches(build_trace(events))


# ------------------------------------------------------------------ loading


def test_shadow_is_filtered_by_session(tmp_path: Path) -> None:
    """Two sessions can share a gitdir; mixing their timelines would draw one
    run's regressions onto another's."""
    import json

    gitdir = tmp_path / "taste"
    gitdir.mkdir()
    (gitdir / "shadow.jsonl").write_text(
        "\n".join(json.dumps(r) for r in [
            {"seq": 1, "session": "mine", "sha": "a"},
            {"seq": 2, "session": "other", "sha": "b"},
        ])
    )
    assert [r["seq"] for r in load_shadow(gitdir, "mine")] == [1]
    assert len(load_shadow(gitdir)) == 2


def test_a_truncated_shadow_log_still_loads(tmp_path: Path) -> None:
    gitdir = tmp_path / "taste"
    gitdir.mkdir()
    (gitdir / "shadow.jsonl").write_text('{"seq": 1, "session": "s"}\n{"seq": 2, "ses')
    assert len(load_shadow(gitdir)) == 1


def test_a_missing_shadow_log_is_normal_not_an_error(tmp_path: Path) -> None:
    assert load_shadow(tmp_path / "nope") == []
