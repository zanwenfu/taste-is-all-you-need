"""The sweep driver: resumability, isolation, and honest accounting.

A few hundred trials on a laptop over an API will be interrupted. The failure
that ruins a study is not losing work — it is silently doing work twice, or
counting a rate limit as an agent failure. These tests pin both.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from taste.cores import Plan, Step, Verification
from taste.evalrun import Cell, CellResult, Ledger, SweepReport, cells, run_sweep
from taste.kernel import RunResult


def _run_result(
    *, status: str = "completed", failure_kind: str | None = None, passed: bool = True
) -> RunResult:
    step = Step("step-01", "d", Verification(kind="shell", command="true"))
    return RunResult(
        task="t",
        session_id="s",
        branch="b",
        status=status,
        plan=Plan(task="t", steps=[step]),
        outcomes=[],
        final_sha="deadbeef",
        elapsed_seconds=1.0,
        failure_kind=failure_kind,
    )


# ------------------------------------------------------------------ the grid


def test_arms_interleave_within_a_task() -> None:
    """Arm-major order aligns condition with calendar time, so a quiet API
    day or a model update lands entirely on one arm."""
    grid = list(cells(["lib"], ["A1", "A3"], trials=2))
    assert [(c.arm, c.trial) for c in grid] == [
        ("A1", 1), ("A3", 1),
        ("A1", 2), ("A3", 2),
    ]


def test_grid_covers_every_combination() -> None:
    grid = list(cells(["a", "b"], ["A1", "A3", "A3prime"], trials=3))
    assert len(grid) == 2 * 3 * 3
    assert len({c.key for c in grid}) == len(grid)


# ------------------------------------------------------------------ the ledger


def test_a_completed_cell_is_never_re_run(tmp_path: Path) -> None:
    """The property that makes resume the default rather than a mode."""
    executed: list[str] = []

    def execute(cell, ctx):
        executed.append(cell.key)
        return _run_result()

    kwargs = dict(
        tasks=["lib"],
        arms=["A1", "A3"],
        trials=1,
        ledger_dir=tmp_path / "ledger",
        prepare=lambda cell: None,
        execute=execute,
    )
    first = run_sweep(**kwargs)
    assert len(first.results) == 2 and first.skipped == 0

    second = run_sweep(**kwargs)
    assert second.results == [] and second.skipped == 2
    assert len(executed) == 2, "an interrupted sweep must not redo finished work"


def test_a_partial_sweep_resumes_where_it_stopped(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "ledger"
    calls = {"n": 0}

    def flaky(cell, ctx):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("laptop lid")
        return _run_result()

    with contextlib.suppress(KeyboardInterrupt):
        run_sweep(
            tasks=["lib"], arms=["A1", "A3", "A3prime"], trials=1,
            ledger_dir=ledger_dir, prepare=lambda c: None, execute=flaky,
        )

    done = len(list(ledger_dir.glob("*.json")))
    assert done == 1, "only the completed cell should be on disk"

    resumed = run_sweep(
        tasks=["lib"], arms=["A1", "A3", "A3prime"], trials=1,
        ledger_dir=ledger_dir, prepare=lambda c: None, execute=lambda c, x: _run_result(),
    )
    assert resumed.skipped == 1
    assert len(resumed.results) == 2


def test_ledger_writes_are_atomic(tmp_path: Path) -> None:
    """A half-written file would read as a finished cell on resume."""
    ledger = Ledger(tmp_path)
    result = CellResult(task="t", arm="A3", trial=1, status="completed", config_hash="abc")
    ledger.write(result)

    assert not list(tmp_path.glob("*.tmp")), "no temp file may survive"
    raw = json.loads(ledger.path_for(Cell("t", "A3", 1)).read_text())
    assert raw["status"] == "completed"


def test_ledger_round_trips_and_tolerates_extra_keys(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path)
    ledger.write(CellResult(task="t", arm="A1", trial=2, status="failed", config_hash="h"))

    path = ledger.path_for(Cell("t", "A1", 2))
    raw = json.loads(path.read_text())
    raw["a_field_from_the_future"] = 1
    path.write_text(json.dumps(raw))

    restored = ledger.read(Cell("t", "A1", 2))
    assert restored is not None and restored.status == "failed"


# ------------------------------------------------------------------ accounting


def test_infra_and_budget_outcomes_are_excluded_from_success() -> None:
    """A rate limit is not evidence about the agent. Pooling them biases
    against whichever arm makes more calls — the one under test."""
    assert CellResult(task="t", arm="a", trial=1, status="completed", config_hash="").counts_toward_success
    assert CellResult(task="t", arm="a", trial=1, status="failed", config_hash="").counts_toward_success
    for status in ("infra", "budget", "error"):
        record = CellResult(task="t", arm="a", trial=1, status=status, config_hash="")
        assert not record.counts_toward_success, status


def test_a_run_halted_by_infra_is_recorded_as_infra(tmp_path: Path) -> None:
    report = run_sweep(
        tasks=["lib"], arms=["A3"], trials=1, ledger_dir=tmp_path,
        prepare=lambda c: None,
        execute=lambda c, x: _run_result(status="failed", failure_kind="infra"),
    )
    assert report.results[0].status == "infra"
    assert not report.results[0].counts_toward_success


def test_a_crash_is_recorded_rather_than_retried_forever(tmp_path: Path) -> None:
    def boom(cell, ctx):
        raise RuntimeError("adapter exploded")

    report = run_sweep(
        tasks=["lib"], arms=["A3"], trials=1, ledger_dir=tmp_path,
        prepare=lambda c: None, execute=boom,
    )
    assert report.results[0].status == "error"
    assert "adapter exploded" in (report.results[0].error or "")
    # And it is on disk, so a resume does not repeat it.
    assert run_sweep(
        tasks=["lib"], arms=["A3"], trials=1, ledger_dir=tmp_path,
        prepare=lambda c: None, execute=boom,
    ).skipped == 1


def test_both_currencies_are_recorded(tmp_path: Path) -> None:
    """work is what the cap is enforced on; billed is what was actually paid."""
    from taste.llm import RunStats
    from taste.providers.base import Completion, Usage

    stats = RunStats()
    stats.record(
        "claude-sonnet-4-6",
        Completion(
            text_blocks=(), tool_calls=(), stop_reason="end_turn",
            model="claude-sonnet-4-6", provider="fake",
            usage=Usage(input_tokens=1000, cache_read_tokens=99_000),
            transcript_blocks=(),
        ),
    )
    result = _run_result()
    result.stats = stats

    report = run_sweep(
        tasks=["lib"], arms=["A3"], trials=1, ledger_dir=tmp_path,
        prepare=lambda c: None, execute=lambda c, x: result,
    )
    record = report.results[0]
    assert record.work_usd > record.billed_usd
    assert record.cache_delta_usd > 0


def test_the_score_is_carried_through(tmp_path: Path) -> None:
    report = run_sweep(
        tasks=["lib"], arms=["A3"], trials=1, ledger_dir=tmp_path,
        prepare=lambda c: None,
        execute=lambda c, x: _run_result(),
        score=lambda c, x, r: 0.42,
    )
    assert report.results[0].score == 0.42


# ------------------------------------------------------------------ isolation


def test_every_cell_gets_its_own_prepared_context(tmp_path: Path) -> None:
    """Two trials sharing a workspace means one run's edits become another's
    starting state."""
    prepared: list[str] = []

    report = run_sweep(
        tasks=["a", "b"], arms=["A1", "A3"], trials=2, ledger_dir=tmp_path,
        prepare=lambda cell: prepared.append(cell.key) or cell.key,
        execute=lambda c, x: _run_result(),
    )
    assert len(prepared) == len(report.results) == 8
    assert len(set(prepared)) == 8, "each cell must be prepared independently"


# ------------------------------------------------------------------ reporting


def test_summary_separates_usable_trials_from_excluded_ones() -> None:
    report = SweepReport(
        results=[
            CellResult(task="t", arm="A3", trial=1, status="completed", config_hash="", score=1.0),
            CellResult(task="t", arm="A3", trial=2, status="failed", config_hash="", score=0.0),
            CellResult(task="t", arm="A3", trial=3, status="infra", config_hash=""),
        ]
    )
    text = report.summary()
    assert "A3" in text
    # 2 usable of 3 run; the infra trial is named as excluded, not counted.
    assert "excluded from success rates: 1" in text
    assert " 3 " in text and " 2 " in text
