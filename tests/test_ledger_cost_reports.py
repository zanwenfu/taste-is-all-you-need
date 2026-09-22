"""Persisted paid retries reach the real reporting CLIs without disappearing."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from taste.evalrun import Cell, CellResult, Ledger

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def paid_history(root, *, latest=0.25):
    ledger = Ledger(root / "ledger")
    for attempt, cost in [(1, 0.75), (2, latest)]:
        ledger.write(CellResult(task="task", arm="A", trial=1, status="infra", config_hash="h",
                                attempts_made=attempt, billed_usd=cost))
    return ledger


@pytest.mark.parametrize("script", ["substrate_table.py", "pilotstats.py", "mswe_report.py"])
def test_report_cli_counts_every_paid_attempt(tmp_path, script):
    paid_history(tmp_path)
    args = ["--root", str(tmp_path)] if script == "pilotstats.py" else [f"audit={tmp_path}"]
    result = subprocess.run([sys.executable, str(SCRIPTS / script), *args],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "$1.00" in result.stdout
    assert "$0.25" not in result.stdout
    if script == "mswe_report.py":
        assert "   1.00" in result.stdout  # The detailed row agrees with the summary.


@pytest.mark.parametrize("script", ["substrate_table.py", "pilotstats.py", "mswe_report.py"])
def test_report_cli_refuses_missing_prior_costs(tmp_path, script):
    ledger = paid_history(tmp_path)
    path = ledger.path_for(Cell("task", "A", 1))
    raw = json.loads(path.read_text())
    raw.pop("prior_attempts")
    path.write_text(json.dumps(raw))
    args = ["--root", str(tmp_path)] if script == "pilotstats.py" else [f"audit={tmp_path}"]
    result = subprocess.run([sys.executable, str(SCRIPTS / script), *args],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert "missing earlier attempt costs" in result.stderr


def test_zero_latest_cost_does_not_hide_prior_paid_attempt(tmp_path):
    from taste.ledger_costs import ledger_billed_usd

    paid_history(tmp_path, latest=0)
    assert ledger_billed_usd(tmp_path / "ledger") == 0.75


@pytest.mark.parametrize("bad_cost", [None, True, -1, "0.75", float("nan"), float("inf")])
def test_raw_cost_reader_and_cell_use_the_same_validation(bad_cost):
    from taste.ledger_costs import lifetime_billed_usd

    row = CellResult(task="task", arm="A", trial=1, status="completed", config_hash="h",
                     billed_usd=bad_cost)
    with pytest.raises(ValueError, match="costs must be finite"):
        lifetime_billed_usd(asdict(row))
    with pytest.raises(ValueError, match="costs must be finite"):
        _ = row.total_billed_usd


def test_legacy_single_attempt_and_invalid_nested_history():
    from taste.ledger_costs import lifetime_billed_usd

    raw = {"task": "task", "arm": "A", "trial": 1, "billed_usd": 0.75}
    assert lifetime_billed_usd(raw) == 0.75
    raw.update(attempts_made=2, prior_attempts=[{**raw, "attempts_made": 1, "prior_attempts": [{}]}])
    with pytest.raises(ValueError, match="not flat"):
        lifetime_billed_usd(raw)


def test_malformed_or_renamed_rows_cannot_disappear_from_report(tmp_path):
    from taste.ledger_costs import ledger_billed_usd

    ledger = paid_history(tmp_path)
    path = ledger.path_for(Cell("task", "A", 1))
    renamed = path.with_name("different__A__t1.json")
    path.rename(renamed)
    with pytest.raises(ValueError, match="identity"):
        ledger_billed_usd(ledger.root)
    renamed.write_text("null")
    with pytest.raises(ValueError, match="object"):
        ledger_billed_usd(ledger.root)


def test_missing_ledger_is_not_a_zero_cost_study(tmp_path):
    from taste.ledger_costs import ledger_billed_usd

    with pytest.raises(FileNotFoundError):
        ledger_billed_usd(tmp_path / "absent")
