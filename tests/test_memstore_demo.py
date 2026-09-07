"""The communicator scenario, scripted, is itself a test of the layer."""

from __future__ import annotations

from pathlib import Path

from examples.memstore_demo import run


def test_communicator_scenario(tmp_path: Path) -> None:
    facts = run(tmp_path / "repo")
    assert facts["found"] == [("worker-b", "revenue_summary")]
    assert facts["revenue_total"] == 555
    assert facts["after_rollback"] == "# Report\n\nTotal revenue: 555\n"
    assert facts["failed_state_still_readable"] == "# Report\n\nTotal revenue: 9999999\n"
    assert facts["failed_transcript"] == "tried a shortcut"
    assert facts["history"] == [
        "monitor: total does not match B's record",
        "revised report (wrong)",
        "drafted report",
        "branch worker-a from session root",
    ]
    assert facts["stale_publish_rejected"] is True
    assert facts["head_after_reopen"] == "monitor: total does not match B's record"
    assert facts["head_is_complete"] is True
