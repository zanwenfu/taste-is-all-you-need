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
    grid = list(cells(["lib"], ["A0", "A3"], trials=2))
    assert [(c.arm, c.trial) for c in grid] == [
        ("A0", 1), ("A3", 1),
        ("A0", 2), ("A3", 2),
    ]


def test_grid_covers_every_combination() -> None:
    grid = list(cells(["a", "b"], ["A0", "A3", "A3prime"], trials=3))
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
        arms=["A0", "A3"],
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
            tasks=["lib"], arms=["A0", "A3", "A3prime"], trials=1,
            ledger_dir=ledger_dir, prepare=lambda c: None, execute=flaky,
        )

    done = len(list(ledger_dir.glob("*.json")))
    assert done == 1, "only the completed cell should be on disk"

    resumed = run_sweep(
        tasks=["lib"], arms=["A0", "A3", "A3prime"], trials=1,
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
    ledger.write(CellResult(task="t", arm="A0", trial=2, status="failed", config_hash="h"))

    path = ledger.path_for(Cell("t", "A0", 2))
    raw = json.loads(path.read_text())
    raw["a_field_from_the_future"] = 1
    path.write_text(json.dumps(raw))

    restored = ledger.read(Cell("t", "A0", 2))
    assert restored is not None and restored.status == "failed"


# ------------------------------------------------------------------ accounting


def test_budget_exhaustion_is_an_outcome_but_infra_is_not() -> None:
    """Intention-to-treat.

    Running out of money is precisely what an expensive policy costs, so a
    budget-exhausted run is a FAILURE, not an exclusion — dropping those would
    flatter whichever arm spends most, which is the arm under test. A rate
    limit says nothing about any arm and is excluded.
    """
    def record(status: str) -> CellResult:
        return CellResult(task="t", arm="a", trial=1, status=status, config_hash="")

    for status in ("completed", "failed", "budget"):
        assert record(status).counts_toward_success, status
    for status in ("infra", "error"):
        assert not record(status).counts_toward_success, status


def test_only_infra_and_crashes_are_retryable() -> None:
    """Retrying a genuine failure until an arm succeeds is the purest form of
    selecting on the dependent variable."""
    def record(status: str) -> CellResult:
        return CellResult(task="t", arm="a", trial=1, status=status, config_hash="")

    for status in ("infra", "error"):
        assert record(status).retryable, status
    for status in ("completed", "failed", "budget"):
        assert not record(status).retryable, status


def test_an_infra_cell_is_retried_within_budget(tmp_path: Path) -> None:
    """Leaving an infra fault on disk as finished silently shrinks the sample."""
    attempts = {"n": 0}

    def flaky(cell, ctx):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _run_result(status="failed", failure_kind="infra")
        return _run_result()

    kwargs = dict(
        tasks=["lib"], arms=["A3"], trials=1, ledger_dir=tmp_path,
        prepare=lambda c: None, execute=flaky, retry_budget=1,
    )
    first = run_sweep(**kwargs)
    assert first.results[0].status == "infra"

    second = run_sweep(**kwargs)
    assert second.skipped == 0, "an infra cell must be retried"
    assert second.results[0].status == "completed"
    assert second.results[0].attempts_made == 2

    third = run_sweep(**kwargs)
    assert third.skipped == 1, "a completed cell is never retried"


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


def _paid_stats():
    """RunStats with real spend on it — billed and work both nonzero."""
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
    return stats


def test_both_currencies_are_recorded(tmp_path: Path) -> None:
    """work is what the cap is enforced on; billed is what was actually paid."""
    result = _run_result()
    result.stats = _paid_stats()

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


def test_a_score_crash_keeps_the_execute_phase_spend(tmp_path: Path) -> None:
    """A crash in score() lands AFTER the agent phase was paid for. The old
    except path wrote billed_usd=0, so the money vanished from the ledger —
    and, the row being retryable, a resume re-executed the PAID agent phase.
    The spend must be read off the context, and the error must say which
    phase died so a human can tell "only the measurement failed" apart from
    "nothing ran"."""
    from types import SimpleNamespace

    context = SimpleNamespace(
        llm_stats=SimpleNamespace(total_cost_usd=1.23, total_work_usd=2.34)
    )

    def exploding_score(cell, ctx, result):
        raise RuntimeError("sidecar write failed")

    report = run_sweep(
        tasks=["lib"], arms=["A3"], trials=1, ledger_dir=tmp_path,
        prepare=lambda c: context,
        execute=lambda c, x: _run_result(),
        score=exploding_score,
    )
    record = report.results[0]
    assert record.status == "error"
    assert (record.error or "").startswith("score:"), record.error
    assert record.billed_usd == 1.23, "paid spend must never vanish from the ledger"
    assert record.work_usd == 2.34


def test_the_error_row_names_the_phase_that_crashed(tmp_path: Path) -> None:
    def boom(cell, ctx):
        raise RuntimeError("adapter exploded")

    report = run_sweep(
        tasks=["lib"], arms=["A3"], trials=1, ledger_dir=tmp_path,
        prepare=lambda c: None, execute=boom,
    )
    assert (report.results[0].error or "").startswith("execute:")


# ------------------------------------------------------------------ breakers


def test_consecutive_zero_progress_failures_trip_the_breaker(tmp_path: Path) -> None:
    """A grid whose cells all fail for one systematic reason — a broken
    image, a revoked key — used to run to the end at full price. That is how
    $99 went to measuring a broken environment."""
    executed: list[str] = []

    def execute(cell, ctx):
        executed.append(cell.key)
        return _run_result(status="failed")

    report = run_sweep(
        tasks=["a", "b", "c"], arms=["A3"], trials=2, ledger_dir=tmp_path,
        prepare=lambda c: None, execute=execute,
        max_consecutive_failures=3,
    )
    assert len(executed) == 3, "the rest of the grid must not run"

    marker = report.results[-1]
    assert marker.status == "aborted"
    assert "circuit breaker" in (marker.failure_reason or "")
    assert not marker.counts_toward_success and not marker.retryable
    # The marker is on disk too, so a resumed sweep sees the stop was
    # deliberate — and exactly one marker, not one per remaining cell.
    assert sum(1 for r in report.results if r.status == "aborted") == 1


def test_any_other_outcome_resets_the_failure_streak(tmp_path: Path) -> None:
    outcomes = iter(["fail", "fail", "ok", "fail", "fail", "ok"])

    def execute(cell, ctx):
        return _run_result() if next(outcomes) == "ok" else _run_result(status="failed")

    report = run_sweep(
        tasks=["a", "b", "c"], arms=["A3"], trials=2, ledger_dir=tmp_path,
        prepare=lambda c: None, execute=execute,
        max_consecutive_failures=3,
    )
    assert len(report.results) == 6
    assert not any(r.status == "aborted" for r in report.results)


def test_the_sweep_budget_stops_before_the_next_paid_cell(tmp_path: Path) -> None:
    """A stop-loss on real spend: once the running billed total crosses the
    cap, no further cell may start — stopping mid-grid loses nothing, since
    every finished cell is already on disk."""
    executed: list[str] = []

    def execute(cell, ctx):
        executed.append(cell.key)
        result = _run_result()
        result.stats = _paid_stats()  # bills ~$0.033 per cell
        return result

    report = run_sweep(
        tasks=["a", "b", "c"], arms=["A3"], trials=1, ledger_dir=tmp_path,
        prepare=lambda c: None, execute=execute,
        sweep_budget_usd=0.03,
    )
    assert len(executed) == 1, "spend crossed the cap; no further cell may start"

    marker = report.results[-1]
    assert marker.status == "aborted"
    assert "budget" in (marker.failure_reason or "")


# ------------------------------------------------------------------ isolation


def test_every_cell_gets_its_own_prepared_context(tmp_path: Path) -> None:
    """Two trials sharing a workspace means one run's edits become another's
    starting state."""
    prepared: list[str] = []

    report = run_sweep(
        tasks=["a", "b"], arms=["A0", "A3"], trials=2, ledger_dir=tmp_path,
        prepare=lambda cell: prepared.append(cell.key) or cell.key,
        execute=lambda c, x: _run_result(),
    )
    assert len(prepared) == len(report.results) == 8
    assert len(set(prepared)) == 8, "each cell must be prepared independently"


# ------------------------------------------------------------------ reporting


def test_summary_shows_attrition_and_refuses_to_infer() -> None:
    """A bare cross-arm mean hides how many trials each arm lost and why."""
    report = SweepReport(
        results=[
            CellResult(task="t", arm="A3", trial=1, status="completed", config_hash="", score=1.0),
            CellResult(task="t", arm="A3", trial=2, status="failed", config_hash="", score=0.0),
            CellResult(task="t", arm="A3", trial=3, status="infra", config_hash=""),
        ]
    )
    text = report.summary()
    assert "A3" in text
    assert "run=  3" in text and "usable=  2" in text
    assert "infra=1" in text
    assert "taste.stats" in text, "inference must be pointed elsewhere"
