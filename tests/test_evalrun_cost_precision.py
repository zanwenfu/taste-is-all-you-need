"""Ledger serialization must not quantize paid work into free admissions."""

from types import SimpleNamespace

import pytest

from taste.evalrun import Cell, GradingResult, Ledger, resume_grading, run_sweep
from taste.ledger_costs import ledger_billed_usd
from taste.sweep_journal import UnsettledSweepAttempt
from tests.test_evalrun import _run_result


def receipt(cost):
    return SimpleNamespace(total_cost_usd=cost, total_work_usd=cost * 2, cache_delta_usd=cost)


def test_small_positive_cost_still_stops_the_next_admission(tmp_path):
    calls = []

    def execute(cell, ctx):
        calls.append(cell.task)
        result = _run_result()
        result.stats = receipt(0.0000004)
        return result

    run_sweep(tasks=["first", "must-not-start"], arms=["A"], trials=1, ledger_dir=tmp_path,
              prepare=lambda cell: None, execute=execute, sweep_budget_usd=0.0000003)
    assert calls == ["first"]
    assert ledger_billed_usd(tmp_path) == 0.0000004


@pytest.mark.parametrize("outcome", ["returned", "raised", "grading_failed"])
def test_receipt_precision_survives_failure_and_recovery_boundaries(tmp_path, outcome):
    cost = 0.123456789
    ctx = SimpleNamespace(llm_stats=receipt(cost))

    def execute(cell, context):
        if outcome == "raised":
            raise RuntimeError("execution failed with a receipt")
        result = _run_result()
        result.stats = ctx.llm_stats
        return result

    def score(*args):
        if outcome == "grading_failed":
            raise RuntimeError("grade failed")
        return 1.0

    kwargs = dict(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path,
                  prepare=lambda cell: ctx, execute=execute, score=score)
    if outcome == "grading_failed":
        with pytest.raises(UnsettledSweepAttempt):
            run_sweep(**kwargs)
        resume_grading(ledger_dir=tmp_path, score=lambda record: GradingResult(score=1.0))
    else:
        run_sweep(**kwargs)
    record = Ledger(tmp_path).read(Cell("task", "A", 1))
    assert (record.billed_usd, record.work_usd, record.cache_delta_usd) == (cost, cost * 2, cost)
    assert ledger_billed_usd(tmp_path) == cost


@pytest.mark.parametrize("field,value", [("total_cost_usd", True), ("total_work_usd", -1)])
def test_invalid_returned_cost_receipt_is_not_normalized_into_valid_evidence(tmp_path, field, value):
    result = _run_result()
    result.stats = receipt(1.25)
    setattr(result.stats, field, value)
    with pytest.raises(UnsettledSweepAttempt):
        run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path,
                  prepare=lambda cell: None, execute=lambda cell, ctx: result)
    assert Ledger(tmp_path).all_results() == []
