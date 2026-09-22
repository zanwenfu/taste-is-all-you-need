"""Real process deaths and competing drivers across paid-attempt boundaries."""

import json
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from taste.evalrun import Cell, Ledger, run_sweep
from tests.test_evalrun import _run_result


def paid_result():
    result = _run_result()
    result.stats = SimpleNamespace(total_cost_usd=1.25, total_work_usd=1.25, cache_delta_usd=0)
    return result


def child_script(tmp_path, mode):
    script = tmp_path / "driver.py"
    script.write_text(f'''
import os, signal, time
from pathlib import Path
from types import SimpleNamespace
from taste.evalrun import run_sweep
from tests.test_evalrun import _run_result
root = Path({str(tmp_path)!r})
mode = {mode!r}
def execute(cell, context):
    with (root / "paid-effect").open("a") as handle:
        handle.write("1.25\\n")
        handle.flush()
        os.fsync(handle.fileno())
    if mode == "execute":
        os.kill(os.getpid(), signal.SIGKILL)
    if mode == "live":
        time.sleep(30)
    result = _run_result()
    result.stats = SimpleNamespace(total_cost_usd=1.25, total_work_usd=1.25, cache_delta_usd=0)
    return result
def score(*args):
    if mode == "score":
        os.kill(os.getpid(), signal.SIGKILL)
    return 1.0
run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=root / "ledger",
          prepare=lambda cell: None, execute=execute, score=score)
''')
    return script


@pytest.mark.parametrize("phase", ["execute", "score"])
def test_killed_attempt_cannot_be_readmitted_even_from_a_filtered_queue(tmp_path, phase):
    result = subprocess.run([sys.executable, str(child_script(tmp_path, phase))],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == -signal.SIGKILL, result.stderr
    called = []
    with pytest.raises(RuntimeError, match="automatic retry is blocked"):
        run_sweep(tasks=["another-task"], arms=["A"], trials=1, ledger_dir=tmp_path / "ledger",
                  prepare=lambda cell: called.append(cell), execute=lambda cell, ctx: paid_result())
    assert called == []
    assert (tmp_path / "paid-effect").read_text() == "1.25\n"
    pending = json.loads((tmp_path / "ledger/.sweep-journal/pending.json").read_text())
    receipt_dir = tmp_path / "ledger/.sweep-journal" / pending["attempt_id"]
    if phase == "score":
        receipt = json.loads((receipt_dir / "execution.json").read_text())
        assert receipt["record"]["billed_usd"] == 1.25
        assert receipt["record"]["final_sha"] == "deadbeef"


def test_a_second_process_cannot_admit_work_in_an_owned_ledger(tmp_path):
    process = subprocess.Popen([sys.executable, str(child_script(tmp_path, "live"))],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 15
        while not (tmp_path / "paid-effect").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (tmp_path / "paid-effect").exists()
        called = []
        with pytest.raises(RuntimeError, match="another sweep owns"):
            run_sweep(tasks=["other"], arms=["A"], trials=1, ledger_dir=tmp_path / "ledger",
                      prepare=lambda cell: called.append(cell), execute=lambda cell, ctx: paid_result())
        assert called == []
    finally:
        process.kill()
        process.communicate(timeout=5)


@pytest.mark.parametrize("after_write", [False, True])
def test_completed_ending_replays_across_ledger_commit_interruption(tmp_path, monkeypatch, after_write):
    write = Ledger.write
    calls = []

    def interrupted(self, record):
        if after_write:
            write(self, record)
        raise KeyboardInterrupt("interrupted result commit")

    monkeypatch.setattr(Ledger, "write", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path,
                  prepare=lambda cell: None, execute=lambda cell, ctx: calls.append(cell) or paid_result())
    monkeypatch.setattr(Ledger, "write", write)
    report = run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path,
                       prepare=lambda cell: pytest.fail("must not execute twice"),
                       execute=lambda cell, ctx: paid_result())
    assert report.skipped == 1 and len(calls) == 1
    stored = Ledger(tmp_path).read(Cell("task", "A", 1))
    assert stored.total_billed_usd == 1.25 and stored.attempts_made == 1
    assert not (tmp_path / ".sweep-journal/pending.json").exists()


def test_pending_attempt_is_not_reported_as_zero_spend(tmp_path):
    from taste.ledger_costs import ledger_billed_usd

    subprocess.run([sys.executable, str(child_script(tmp_path, "execute"))],
                   capture_output=True, text=True, timeout=20)
    with pytest.raises(ValueError, match="pending"):
        ledger_billed_usd(tmp_path / "ledger")


@pytest.mark.parametrize("field,value,error", [
    ("task", "other-task", "not bound"),
    ("billed_usd", 0, "changed execution evidence"),
])
def test_changed_ending_is_rejected_before_any_callback(tmp_path, monkeypatch, field, value, error):
    write = Ledger.write
    monkeypatch.setattr(Ledger, "write", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path,
                  prepare=lambda cell: None, execute=lambda cell, ctx: paid_result())
    monkeypatch.setattr(Ledger, "write", write)
    pending = json.loads((tmp_path / ".sweep-journal/pending.json").read_text())
    path = tmp_path / ".sweep-journal" / pending["attempt_id"] / "ending.json"
    raw = json.loads(path.read_text())
    raw["record"][field] = value
    path.write_text(json.dumps(raw))
    with pytest.raises(RuntimeError, match=error):
        run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path,
                  prepare=lambda cell: pytest.fail("must not prepare"),
                  execute=lambda cell, ctx: paid_result())


def test_retry_history_is_bound_into_recoverable_ending(tmp_path, monkeypatch):
    from taste.evalrun import CellResult

    first = CellResult(task="task", arm="A", trial=1, status="infra", config_hash="h", billed_usd=0.5)
    Ledger(tmp_path).write(first)
    write = Ledger.write

    def interrupted(self, record):
        write(self, record)
        raise KeyboardInterrupt()

    monkeypatch.setattr(Ledger, "write", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path, retry_budget=1,
                  prepare=lambda cell: None, execute=lambda cell, ctx: paid_result())
    monkeypatch.setattr(Ledger, "write", write)
    run_sweep(tasks=["task"], arms=["A"], trials=1, ledger_dir=tmp_path,
              prepare=lambda cell: pytest.fail("must not prepare"), execute=lambda cell, ctx: paid_result())
    restored = Ledger(tmp_path).read(Cell("task", "A", 1))
    assert restored.total_billed_usd == 1.75
    assert restored.prior_attempts == ({key: value for key, value in asdict(first).items()
                                        if key != "prior_attempts"},)
