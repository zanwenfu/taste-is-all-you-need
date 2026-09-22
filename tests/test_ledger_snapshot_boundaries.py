"""Final reports and every ledger writer share one stable snapshot boundary."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from taste.evalrun import Cell, CellResult, Ledger
from taste.ledger_costs import ledger_billed_usd, lifetime_billed_usd, read_cost_rows
from taste.sweep_journal import SweepBusy, SweepJournal
from tests.test_sweep_journal import child_script


def sample():
    return CellResult(task="task", arm="A", trial=1, status="completed", config_hash="h", billed_usd=0.75)


def test_report_snapshot_excludes_admission_during_file_enumeration(tmp_path, monkeypatch):
    directory = tmp_path / "ledger"
    previous = sample()
    previous.task = "previous"
    Ledger(directory).write(previous)
    script = child_script(tmp_path, "execute")
    glob = Path.glob
    launches = []

    def interleaved(path, pattern, *args, **kwargs):
        if path == directory:
            launches.append(subprocess.run([sys.executable, str(script)], capture_output=True,
                                           text=True, timeout=20))
        return glob(path, pattern, *args, **kwargs)

    monkeypatch.setattr(Path, "glob", interleaved)
    assert ledger_billed_usd(directory) == 0.75
    assert len(launches) == 1
    assert "SweepBusy" in launches[0].stderr
    assert not (tmp_path / "paid-effect").exists()
    assert not (directory / ".sweep-journal/pending.json").exists()


def test_direct_ledger_write_cannot_bypass_sweep_ownership(tmp_path):
    ledger = Ledger(tmp_path)
    with SweepJournal(tmp_path), pytest.raises(SweepBusy):
        ledger.write(sample())
    assert ledger.all_results() == []
    ledger.write(sample())
    assert ledger_billed_usd(tmp_path) == 0.75


@pytest.mark.parametrize("reader", ["report", "ledger"])
def test_duplicate_cost_keys_are_rejected_by_all_ledger_readers(tmp_path, reader):
    row = sample()
    path = tmp_path / f"{Cell(row.task, row.arm, row.trial).key}.json"
    path.write_text(json.dumps(asdict(row))[:-1] + ', "billed_usd": 0}')
    with pytest.raises(ValueError, match="duplicate"):
        if reader == "report":
            read_cost_rows(tmp_path)
        else:
            Ledger(tmp_path).all_results()


def test_ledger_cannot_accept_boolean_trial_identity(tmp_path):
    row = asdict(sample())
    row["trial"] = True
    (tmp_path / "task__A__tTrue.json").write_text(json.dumps(row))
    with pytest.raises(ValueError, match="identity"):
        Ledger(tmp_path).all_results()


def test_prior_attempt_numbers_require_integer_receipts():
    row = asdict(sample())
    row["attempts_made"] = 2
    previous = asdict(sample())
    previous["attempts_made"] = 1.0
    row["prior_attempts"] = [previous]
    with pytest.raises(ValueError, match="history"):
        lifetime_billed_usd(row)


def test_reading_a_legacy_ledger_does_not_create_lock_files(tmp_path, monkeypatch):
    (tmp_path / "task__A__t1.json").write_text(json.dumps(asdict(sample())))
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: pytest.fail("reader must not create files"))
    assert ledger_billed_usd(tmp_path) == 0.75
    assert not (tmp_path / ".sweep-journal").exists()
