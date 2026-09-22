"""Grader retries reuse durable execution; they never readmit the agent."""

import json
import signal
import subprocess
import sys

import pytest

from taste import evalrun
from taste.evalrun import Cell, CellResult, Ledger, run_sweep
from taste.ledger_costs import ledger_billed_usd
from taste.sweep_journal import UnsettledSweepAttempt
from tests.test_sweep_journal import child_script, paid_result


def fail_grade(*args):
    raise RuntimeError("grader unavailable")


def pending_grade(tmp_path, execute=None):
    with pytest.raises(UnsettledSweepAttempt, match="grading is incomplete"):
        run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path,
                  prepare=lambda cell: None, execute=execute or (lambda cell, ctx: paid_result()),
                  score=fail_grade)


def test_grading_failure_cannot_readmit_paid_execution(tmp_path):
    calls = []
    pending_grade(tmp_path, lambda cell, ctx: calls.append(cell) or paid_result())
    with pytest.raises(UnsettledSweepAttempt, match="automatic retry is blocked"):
        run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path, retry_budget=100,
                  prepare=lambda cell: pytest.fail("must not prepare"),
                  execute=lambda cell, ctx: calls.append(cell) or paid_result())
    assert len(calls) == 1


def test_repaired_grader_preserves_attempt_history_and_ignores_input_mutation(tmp_path):
    Ledger(tmp_path).write(CellResult(task="task", arm="A", trial=1, status="infra",
                                     config_hash="old", billed_usd=0.5))
    with pytest.raises(UnsettledSweepAttempt):
        run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path, retry_budget=1,
                  prepare=lambda cell: None, execute=lambda cell, ctx: paid_result(), score=fail_grade)

    def repaired(record):
        assert record.final_sha == "deadbeef" and record.billed_usd == 1.25
        record.billed_usd = 0
        record.prior_attempts[0]["billed_usd"] = 0
        return evalrun.GradingResult(score=1.0, report_path="repaired.json")

    result = evalrun.resume_grading(ledger_dir=tmp_path, score=repaired)
    assert result.status == "completed" and result.score == 1.0
    assert result.billed_usd == 1.25 and result.total_billed_usd == 1.75
    assert result.attempts_made == 2 and result.report_path == "repaired.json"
    assert ledger_billed_usd(tmp_path) == 1.75
    assert evalrun.resume_grading(ledger_dir=tmp_path, score=lambda r: pytest.fail("already settled")) is None


def test_killed_grader_can_resume_without_the_agent(tmp_path):
    killed = subprocess.run([sys.executable, str(child_script(tmp_path, "score"))],
                             capture_output=True, text=True, timeout=20)
    assert killed.returncode == -signal.SIGKILL
    result = evalrun.resume_grading(ledger_dir=tmp_path / "ledger",
                                    score=lambda record: evalrun.GradingResult(score=0.0))
    assert result.billed_usd == 1.25 and result.attempts_made == 1
    assert result.score == 0.0 and (tmp_path / "paid-effect").read_text() == "1.25\n"


def test_repeated_grading_failures_retain_execution_and_each_error(tmp_path):
    pending_grade(tmp_path)
    journal = tmp_path / ".sweep-journal"
    admission = json.loads((journal / "pending.json").read_text())
    directory = journal / admission["attempt_id"]
    before = (directory / "execution.json").read_bytes()
    for _ in range(2):
        with pytest.raises(UnsettledSweepAttempt, match="grading is incomplete"):
            evalrun.resume_grading(ledger_dir=tmp_path, score=fail_grade)
    assert (directory / "execution.json").read_bytes() == before
    assert len(list(directory.glob("grading-failure-*.json"))) == 3
    assert Ledger(tmp_path).read(Cell("task", "A", 1)) is None


@pytest.mark.parametrize("score", [float("nan"), float("inf"), "1.0"])
def test_invalid_recovered_grade_cannot_finalize_the_attempt(tmp_path, score):
    pending_grade(tmp_path)
    with pytest.raises(UnsettledSweepAttempt):
        evalrun.resume_grading(ledger_dir=tmp_path,
                              score=lambda record: evalrun.GradingResult(score=score))
    assert (tmp_path / ".sweep-journal/pending.json").exists()
    assert Ledger(tmp_path).all_results() == []


@pytest.mark.parametrize("after_write", [False, True])
def test_grading_recovery_replays_ending_after_commit_interruption(tmp_path, monkeypatch, after_write):
    pending_grade(tmp_path)
    original = Ledger.write

    def interrupted(self, record):
        if after_write:
            original(self, record)
        raise KeyboardInterrupt("ledger commit interrupted")

    monkeypatch.setattr(Ledger, "write", interrupted)
    with pytest.raises(KeyboardInterrupt):
        evalrun.resume_grading(ledger_dir=tmp_path, score=lambda record: evalrun.GradingResult(score=1.0))
    monkeypatch.setattr(Ledger, "write", original)
    result = evalrun.resume_grading(ledger_dir=tmp_path,
                                    score=lambda record: pytest.fail("ending already has the grade"))
    assert result.score == 1.0 and result.total_billed_usd == 1.25


def test_grading_cannot_recover_unknown_execution(tmp_path):
    subprocess.run([sys.executable, str(child_script(tmp_path, "execute"))],
                   capture_output=True, text=True, timeout=20)
    with pytest.raises(UnsettledSweepAttempt, match="execution cost is unknown"):
        evalrun.resume_grading(ledger_dir=tmp_path / "ledger",
                              score=lambda record: pytest.fail("no completed execution"))


def test_historical_score_error_cannot_authorize_paid_reexecution(tmp_path):
    record = CellResult(task="task", arm="A", trial=1, status="error", config_hash="h",
                        billed_usd=1.25, error="score: RuntimeError: old grader failed")
    Ledger(tmp_path).write(record)
    report = run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path, retry_budget=100,
                       prepare=lambda cell: pytest.fail("historical grade error is not an agent retry"),
                       execute=lambda cell, ctx: paid_result())
    assert report.skipped == 1 and ledger_billed_usd(tmp_path) == 1.25
