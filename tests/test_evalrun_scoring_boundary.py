"""Returned execution evidence survives failures across the grading boundary."""

import json
from types import SimpleNamespace

import pytest

from taste.evalrun import CellResult, run_sweep
from taste.sweep_journal import SweepJournal, UnsettledSweepAttempt
from tests.test_evalrun import _run_result


def paid_result(cost=1.25):
    result = _run_result()
    result.stats = SimpleNamespace(total_cost_usd=cost, total_work_usd=cost * 2,
                                   cache_delta_usd=cost)
    return result


@pytest.mark.parametrize("context_cost", [None, 0.0, 99.0])
def test_failed_grader_preserves_returned_cost_and_execution_identity(tmp_path, context_cost):
    result = paid_result()
    context = SimpleNamespace(
        workspace=tmp_path / "product", gitdir=tmp_path / "evidence",
        config=SimpleNamespace(hash=lambda: "exact-config"),
        llm_stats=None if context_cost is None else SimpleNamespace(
            total_cost_usd=context_cost, total_work_usd=context_cost),
    )

    def grade(cell, ctx, returned):
        # A later phase can mutate the receipt object. Execution accounting
        # must already have captured the values the executor actually returned.
        returned.stats.total_cost_usd = 0
        ctx.report_path = str(tmp_path / "partial-grade.json")
        raise RuntimeError("grader failed after execution")

    with pytest.raises(UnsettledSweepAttempt, match="grading is incomplete"):
        run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path / "ledger",
                  prepare=lambda cell: context, execute=lambda cell, ctx: result, score=grade)
    with SweepJournal(tmp_path / "ledger") as journal:
        restored = CellResult(**journal.execution())
    assert restored.status == "completed"
    assert restored.score is None
    assert (restored.billed_usd, restored.work_usd, restored.cache_delta_usd) == (1.25, 2.5, 1.25)
    assert restored.config_hash == "exact-config"
    assert restored.session_id == result.session_id and restored.final_sha == result.final_sha
    assert restored.workspace == str(context.workspace) and restored.gitdir == str(context.gitdir)
    failure_path, = (tmp_path / "ledger/.sweep-journal").glob("*/grading-failure-*.json")
    failure = json.loads(failure_path.read_text())["record"]
    assert failure["error"].startswith("score:") and failure["report_path"] == context.report_path


def test_successful_grader_cannot_rewrite_execution_cost_and_can_publish_report(tmp_path):
    result = paid_result()
    context = SimpleNamespace()

    def grade(cell, ctx, returned):
        returned.stats.total_cost_usd = 17
        ctx.report_path = "completed-grade.json"
        return 1.0

    report = run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path,
                       prepare=lambda cell: context, execute=lambda cell, ctx: result, score=grade)
    (record,) = report.results
    assert record.billed_usd == 1.25 and record.score == 1.0
    assert record.report_path == "completed-grade.json"


def test_context_receipt_is_used_when_the_returned_result_has_no_stats(tmp_path):
    context = SimpleNamespace(llm_stats=SimpleNamespace(total_cost_usd=0.75, total_work_usd=1.5))
    report = run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path,
                       prepare=lambda cell: context, execute=lambda cell, ctx: _run_result())
    assert report.results[0].billed_usd == 0.75


def test_failed_next_execution_cannot_inherit_previous_cells_receipt(tmp_path):
    def execute(cell, context):
        if cell.task == "first":
            return paid_result(1.25)
        raise RuntimeError("second execution failed")

    report = run_sweep(tasks=["first", "second"], arms=["A"], trials=1, ledger_dir=tmp_path,
                       prepare=lambda cell: SimpleNamespace(
                           llm_stats=SimpleNamespace(total_cost_usd=0.5, total_work_usd=1.0)),
                       execute=execute)
    assert [record.billed_usd for record in report.results] == [1.25, 0.5]
    assert report.results[1].status == "error"
    assert report.results[1].session_id == "" and report.results[1].final_sha == ""
