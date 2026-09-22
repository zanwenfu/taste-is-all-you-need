"""Resume work queues cannot change the lifetime budget's ledger scope."""

import json

import pytest

from taste.evalrun import Cell, CellResult, Ledger, run_sweep
from tests.test_evalrun import _run_result


def completed(task="completed", arm="A", cost=0.75, attempt=1, status="completed"):
    return CellResult(task=task, arm=arm, trial=1, status=status, config_hash="h",
                      billed_usd=cost, attempts_made=attempt)


@pytest.mark.parametrize("filtered_arm", ["A", "earlier-arm"])
def test_resume_counts_completed_tasks_filtered_out_of_work_queue(tmp_path, filtered_arm):
    ledger = Ledger(tmp_path)
    ledger.write(completed(arm=filtered_arm, status="infra"))
    ledger.write(completed(arm=filtered_arm, cost=0.25, attempt=2))
    called = []
    report = run_sweep(tasks=["remaining"], arms=["A"], trials=1, ledger_dir=tmp_path,
                       prepare=lambda cell: called.append(cell),
                       execute=lambda cell, ctx: _run_result(), sweep_budget_usd=0.9)
    assert called == []
    assert report.results[0].status == "aborted"
    assert "$1.00" in report.results[0].failure_reason
    assert ledger.read(Cell("completed", filtered_arm, 1)).total_billed_usd == 1.0


def test_filtered_unknown_cost_blocks_resume_before_preparing_any_cell(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.write(completed())
    path = ledger.path_for(Cell("completed", "A", 1))
    raw = json.loads(path.read_text())
    raw.pop("billed_usd")
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="malformed"):
        run_sweep(tasks=["remaining"], arms=["A"], trials=1, ledger_dir=tmp_path,
                  prepare=lambda cell: pytest.fail("unknown spend cannot admit another cell"),
                  execute=lambda cell, ctx: _run_result(), sweep_budget_usd=1)


def test_duplicate_work_queue_entries_do_not_double_count_historical_spend(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.write(completed())
    called = []
    report = run_sweep(tasks=["completed", "completed", "remaining"], arms=["A"], trials=1,
                       ledger_dir=tmp_path, prepare=lambda cell: called.append(cell),
                       execute=lambda cell, ctx: _run_result(), sweep_budget_usd=1)
    assert [cell.task for cell in called] == ["remaining"]
    assert report.results[0].status == "completed"


@pytest.mark.parametrize("budget", [True, "1.0", -1, float("nan"), float("inf")])
def test_invalid_stop_loss_is_rejected_before_preparation(tmp_path, budget):
    with pytest.raises(ValueError, match="sweep_budget_usd"):
        run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path,
                  prepare=lambda cell: pytest.fail("invalid budget must not start work"),
                  execute=lambda cell, ctx: _run_result(), sweep_budget_usd=budget)


def test_zero_budget_stops_at_zero_and_independent_ledger_has_its_own_budget(tmp_path):
    Ledger(tmp_path / "old").write(completed(cost=99))
    report = run_sweep(tasks=["new"], arms=["A"], trials=1, ledger_dir=tmp_path / "new",
                       prepare=lambda cell: None, execute=lambda cell, ctx: _run_result(),
                       sweep_budget_usd=1)
    assert report.results[0].status == "completed"
    report = run_sweep(tasks=["new"], arms=["A"], trials=1, ledger_dir=tmp_path / "zero",
                       prepare=lambda cell: pytest.fail("zero allowance cannot start work"),
                       execute=lambda cell, ctx: _run_result(), sweep_budget_usd=0)
    assert report.results[0].status == "aborted"
