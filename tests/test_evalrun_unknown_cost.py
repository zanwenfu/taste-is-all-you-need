"""A failed executor cannot turn missing or invalid receipts into free work."""

import json
from types import SimpleNamespace

import pytest

from taste.evalrun import Cell, CellResult, Ledger, run_sweep
from taste.ledger_costs import ledger_billed_usd
from taste.sweep_journal import UnsettledSweepAttempt


@pytest.mark.parametrize("stats", [
    None, SimpleNamespace(),
    SimpleNamespace(total_cost_usd=None, total_work_usd=1.25),
    SimpleNamespace(total_cost_usd=1.25),
    SimpleNamespace(total_cost_usd=float("nan"), total_work_usd=1.25),
    SimpleNamespace(total_cost_usd=True, total_work_usd=1.25),
    SimpleNamespace(total_cost_usd="1.25", total_work_usd=1.25),
    SimpleNamespace(total_cost_usd=-1, total_work_usd=1.25),
])
def test_unknown_failed_execution_fences_resume_and_reporting(tmp_path, stats):
    calls = []

    def execute(cell, context):
        calls.append(cell.task)
        # The synthetic paid effect occurs before the adapter loses its receipt.
        (tmp_path / "paid-effect").write_text("1.25")
        raise RuntimeError("receipt lost after paid work")

    with pytest.raises(UnsettledSweepAttempt, match="execution cost is unknown"):
        run_sweep(tasks=["paid", "must-not-start"], arms=["A"], trials=1,
                  ledger_dir=tmp_path / "ledger", execute=execute,
                  prepare=lambda cell: SimpleNamespace(llm_stats=stats))
    assert calls == ["paid"]
    assert Ledger(tmp_path / "ledger").all_results() == []
    pending = json.loads((tmp_path / "ledger/.sweep-journal/pending.json").read_text())
    receipt_dir = tmp_path / "ledger/.sweep-journal" / pending["attempt_id"]
    failure = json.loads((receipt_dir / "failure.json").read_text())["record"]
    assert failure["billed_usd"] is None and failure["work_usd"] is None
    assert "receipt lost after paid work" in failure["error"]
    assert failure["attempt_id"] == pending["attempt_id"]
    assert not (receipt_dir / "ending.json").exists()
    with pytest.raises(UnsettledSweepAttempt, match="automatic retry is blocked"):
        run_sweep(tasks=["different-task"], arms=["B"], trials=1, retry_budget=100,
                  ledger_dir=tmp_path / "ledger", prepare=lambda cell: pytest.fail("no admission"),
                  execute=execute)
    with pytest.raises(ValueError, match="pending"):
        ledger_billed_usd(tmp_path / "ledger")


def test_unknown_retry_keeps_the_previous_paid_attempt_intact(tmp_path):
    ledger = Ledger(tmp_path)
    first = CellResult(task="task", arm="A", trial=1, status="infra", config_hash="h",
                       billed_usd=0.75, work_usd=1.0)
    ledger.write(first)
    before = ledger.path_for(Cell("task", "A", 1)).read_bytes()

    def execute(cell, context):
        raise RuntimeError("unsettled retry")

    with pytest.raises(UnsettledSweepAttempt):
        run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path,
                  prepare=lambda cell: None, execute=execute, retry_budget=1)
    assert ledger.path_for(Cell("task", "A", 1)).read_bytes() == before
    pending = json.loads((tmp_path / ".sweep-journal/pending.json").read_text())
    assert pending["attempts_made"] == 2


@pytest.mark.parametrize("cost", [0.0, 1.25])
def test_explicit_failed_execution_receipt_can_be_settled(tmp_path, cost):
    def execute(cell, context):
        raise RuntimeError("failure with a complete receipt")

    report = run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path,
                       prepare=lambda cell: SimpleNamespace(llm_stats=SimpleNamespace(
                           total_cost_usd=cost, total_work_usd=cost)), execute=execute)
    assert report.results[0].billed_usd == cost
    assert report.results[0].status == "error"
    assert ledger_billed_usd(tmp_path) == cost
    assert not (tmp_path / ".sweep-journal/pending.json").exists()
