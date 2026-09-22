"""Paid retries remain visible across failed sweeps and ledger restart."""
import json

import pytest

from taste.evalrun import Cell, CellResult, Ledger, run_sweep


def record(*, attempt=1, cost=0.75, status="infra"):
    return CellResult(task="task", arm="A", trial=1, status=status, config_hash="h",
                      attempts_made=attempt, billed_usd=cost, work_usd=cost * 2)


def test_later_attempt_retains_all_earlier_outcomes_and_costs(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.write(record())
    ledger.write(record(attempt=2, cost=0.25))
    ledger.write(record(attempt=3, cost=0.5, status="completed"))
    restored = Ledger(tmp_path).read(Cell("task", "A", 1))
    assert restored.total_billed_usd == 1.5
    assert [row["billed_usd"] for row in restored.prior_attempts] == [0.75, 0.25]
    assert [row["status"] for row in restored.prior_attempts] == ["infra", "infra"]
    assert restored.billed_usd == 0.5  # Last-attempt and lifetime costs remain distinct.


def test_resume_counts_paid_retry_before_starting_next_attempt(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.write(record())
    ledger.write(record(attempt=2, cost=0.5))
    calls = []
    report = run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path,
                       retry_budget=3, sweep_budget_usd=1,
                       prepare=lambda cell: calls.append(cell), execute=lambda cell, context: None)
    assert calls == []
    assert len(report.results) == 1 and report.results[0].status == "aborted"
    assert "1.25" in report.results[0].failure_reason


def test_unknown_legacy_retry_cost_and_corrupt_record_cannot_authorize_more_spend(tmp_path):
    ledger = Ledger(tmp_path)
    path = ledger.path_for(Cell("task", "A", 1))
    raw = record(attempt=2).__dict__.copy()
    raw.pop("prior_attempts", None)
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="missing earlier"):
        run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path,
                  retry_budget=3, prepare=lambda cell: pytest.fail("must not prepare"),
                  execute=lambda cell, context: pytest.fail("must not execute"))
    path.write_text("{truncated")
    with pytest.raises(ValueError, match="malformed"):
        ledger.read(Cell("task", "A", 1))


@pytest.mark.parametrize("raw", ["{truncated", "[]", "null", '{"task":"task"}'])
def test_reporting_cannot_silently_drop_corrupt_cells(tmp_path, raw):
    ledger = Ledger(tmp_path)
    ledger.path_for(Cell("task", "A", 1)).write_text(raw)
    with pytest.raises(ValueError, match="malformed"):
        ledger.all_results()


def test_swapped_cell_file_is_rejected_by_resume_and_reporting(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.write(record())
    source = ledger.path_for(Cell("task", "A", 1))
    source.rename(ledger.path_for(Cell("different-task", "A", 1)))
    with pytest.raises(ValueError, match="identity"):
        ledger.read(Cell("different-task", "A", 1))
    with pytest.raises(ValueError, match="identity"):
        ledger.all_results()


def test_abort_cannot_erase_a_paid_cell_attempt(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.write(record())
    with pytest.raises(ValueError, match="abort marker"):
        ledger.write(record(status="aborted", cost=0))
    assert Ledger(tmp_path).read(Cell("task", "A", 1)).total_billed_usd == 0.75
