"""The recovery study's figures: what each one shows, checked on a report with known answers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tests.test_recovery_report import episodes, mapped, rejected, report

_SPEC = importlib.util.spec_from_file_location(
    "recovery_figures", Path(__file__).resolve().parents[1] / "scripts" / "recovery_figures.py")
figures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(figures)


def a_report():
    """A report as recovery_report.py writes it (through JSON): two mapped runs, four recovered."""
    maps = {
        "r1": mapped("done", 20, 12, {0: (1, 2), 10: (4, 8), 11: (8, 16), 12: (0, 16)},
                     rewind={"highest_v": 11, "latest": 11}),
        "r2": mapped("done", 10, 5, {0: (1, 3), 4: (2, 16), 5: (0, 16)}, rewind={"highest_v": 4, "latest": 4},
                     rules={"visible_tests": 2, "before_large_edit": None, "start": 0, "rules": 2}),
        "r3": mapped("waiting", 16, None, {0: (1, 2)}),
        "r4": {"task": "t", "base": {}, "status": "skipped", "skipped": "1 steps"},
    }
    runs = {f"r{i}": rejected({"retry": episodes(i % 2 == 0, 0.5), "continue": episodes(True, 0.3)})
            for i in range(4)}
    result = report.report(maps, runs, {"r1": {"step": 9, "reason": "x", "confidence": 0.5}}, resamples=200)
    return json.loads(json.dumps(result, sort_keys=True))     # as --json writes it: keys sorted


def test_the_curves_are_the_mapped_runs_probes_with_their_intervals_and_decisive_steps():
    found = figures.curves(a_report())
    assert [panel["run"] for panel in found] == ["r1", "r2"]       # r3 still searching, r4 left out
    first = found[0]
    assert (first["steps"], first["decisive"]) == (20, 12)
    assert [point["step"] for point in first["points"]] == [0, 10, 11, 12]
    assert first["points"][1]["v"] == 0.5 and first["points"][1]["low"] < 0.5 < first["points"][1]["high"]


def test_a_panel_names_the_task_and_the_run():
    assert figures._short("instance_element-hq__element-web__7VHCQdw") == "element-web 7VHCQdw"
    assert figures._short("r1") == "r1"


def test_the_decisive_steps_are_shares_of_their_runs():
    assert figures.decisive_fractions(a_report()) == [0.5, 0.6]


def test_each_recovery_is_shown_with_its_interval_and_its_paired_difference_from_plain_retry():
    rows = {row["view"]: row for row in figures.recoveries(a_report())}
    assert list(rows) == ["retry", "continue"]                   # the report's order, not the file's
    assert figures.VIEWS == report.VIEWS
    assert rows["retry"]["solved"] == 0.5 and rows["retry"]["difference"] is None
    assert rows["continue"]["solved"] == 1.0
    difference = rows["continue"]["difference"]
    assert (difference["mean"], difference["runs"]) == (0.5, 4)
    assert difference["interval"][0] <= 0.5 <= difference["interval"][1]
    assert (rows["continue"]["agent_usd"], rows["continue"]["check_usd"]) == (0.3, 0.0)


def test_a_difference_registered_the_other_way_round_is_turned_to_read_against_retry():
    items = [{"other": "retry", "base": "continue", "runs": 3, "mean_difference": -0.25, "bootstrap_95": [-0.5, 0.1]}]
    assert figures.paired(items, "continue", "retry") == {"mean": 0.25, "interval": [-0.1, 0.5], "runs": 3}
    assert figures.paired(items, "rewind_rules", "retry") is None


def test_the_checker_cells_are_its_verdicts_against_the_hidden_tests():
    # Every episode of the fake has one round, checked "done": 2 x 2 retries solved, 2 x 2 not, 8 continues solved.
    assert figures.checker_cells(a_report()) == [[12, 4], [0, 0]]


def test_each_method_s_distances_from_the_decisive_step_are_shown_and_the_oracle_s_latest_left_out():
    found = figures.distances(a_report())
    assert found["reader"] == [-3]                       # rewind to 8: first wrong step 9, d = 12
    assert found["start"] == [-11, -4] and found["visible_tests"] == [-2]
    assert found["oracle_highest_v"] == [0, 0] and "oracle_latest" not in found
    assert list(found) == [method for method in figures.METHODS if method in found]


def test_the_figures_are_drawn_from_the_report_file(tmp_path):
    pytest.importorskip("matplotlib")
    path = tmp_path / "report.json"
    path.write_text(json.dumps(a_report()))
    out = tmp_path / "figures"
    assert figures.main(["--report", str(path), "--out", str(out)]) == 0
    assert sorted(item.name for item in out.iterdir()) == [f"{name}.png" for name in sorted(figures.FIGURES)]
    assert all(item.stat().st_size > 1000 for item in out.iterdir())


def test_a_report_without_recoveries_draws_only_the_map_s_figures(tmp_path):
    pytest.importorskip("matplotlib")
    result = a_report()
    for key in ("recoveries", "checker", "gaming", "selector"):
        del result[key]
    written = figures.render(result, tmp_path, "png")
    assert sorted(path.name for path in written) == ["choices.png", "curves.png", "decisive.png"]
