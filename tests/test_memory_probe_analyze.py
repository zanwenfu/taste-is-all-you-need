"""The rule-loss tables and the long run's tables, worked out by hand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from studies.long_run import analyze as long_analyze
from studies.memory_probe.analyze import effective_rounds, load_run, render, tables


def _row(case: int, *, obeyed: bool | None, history: str = "h0", rule: str = "sc01", **extra) -> dict:
    return {"id": f"{history}:{case}", "haystack_id": history, "sc_id": rule, "compliant": obeyed, "looked": True, "found": False, "in_summary": False, **extra}


def _write(root: Path, label: str, rows: list[dict], verdicts: list[dict] | None = None, *, name: str = "a5q-i-anthropic-k1-b4096") -> None:
    folder = root / label / name
    folder.mkdir(parents=True)
    (folder / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    if verdicts is not None:
        (folder / "found.served.jsonl").write_text("".join(json.dumps(row) + "\n" for row in verdicts))


def test_every_row_is_read_against_no_rule_the_summary_alone_and_the_rule_restated(tmp_path: Path) -> None:
    pattern = [(0, True), (1, True), (2, False), (3, False)]
    _write(tmp_path, "summary", [_row(n, obeyed=value) for n, value in pattern])
    _write(tmp_path, "restated", [_row(n, obeyed=True) for n, _ in pattern])
    _write(tmp_path, "turns", [_row(n, obeyed=True) for n, _ in pattern[:3]] + [_row(3, obeyed=False)],
           [{"id": f"h0:{n}", "found_by_judge": n < 2} for n in range(4)])
    floor = {f"h0:{n}": n == 0 for n in range(4)}
    entries = {entry["row"]: entry for entry in tables(load_run(tmp_path, "anthropic", 1), floor)}
    turns = entries["turns"]
    assert (turns["obeyed"], turns["no_rule"], turns["restated"]) == (3, 1, 4)
    assert turns["effective_retention"] == pytest.approx(100 * (75 - 25) / (100 - 25))
    assert turns["against_summary"]["better"] == 1 and turns["against_summary"]["worse"] == 0
    assert turns["difference"][0] == pytest.approx(25.0)
    assert (turns["held"], turns["obeyed_if_held"], turns["obeyed_if_not"]) == (2, [2, 2], [1, 2])
    assert "against_summary" not in entries["summary"]
    assert "turns" in render(list(entries.values()), "t")


def test_a_rule_not_obeyed_even_when_restated_cannot_be_lost_so_it_is_read_apart(tmp_path: Path) -> None:
    _write(tmp_path, "summary", [_row(0, obeyed=False), _row(1, obeyed=False, rule="sc02")])
    _write(tmp_path, "restated", [_row(0, obeyed=True), _row(1, obeyed=False, rule="sc02")])
    _write(tmp_path, "turns", [_row(0, obeyed=True), _row(1, obeyed=False, rule="sc02")])
    entries = {entry["row"]: entry for entry in tables(load_run(tmp_path, "anthropic", 1), {"h0:0": False, "h0:1": False})}
    assert entries["turns"]["obeyed_where_restated_is"] == [1, 1]
    assert entries["turns"]["per_rule"] == {"sc01": [1, 1], "sc02": [0, 1]}


def test_whether_a_reversed_rules_old_text_came_back_is_counted(tmp_path: Path) -> None:
    _write(tmp_path, "turns", [_row(0, obeyed=True, delivered_text="[step 1]\nALWAYS draft first.\n\n[step 9]\nnever mind"), _row(1, obeyed=True, delivered_text="nothing")])
    entries = tables(load_run(tmp_path, "anthropic", 1), {}, {"h0:0": "always draft first.", "h0:1": "always draft first."})
    assert entries[0]["old_rule_came_back"] == 1


def test_only_this_prompt_and_depth_are_read(tmp_path: Path) -> None:
    _write(tmp_path, "turns", [_row(0, obeyed=True)])
    _write(tmp_path, "other", [_row(0, obeyed=False)], name="a5q-i-pi_mono-k16-b4096")
    assert set(load_run(tmp_path, "anthropic", 1)) == {"turns"}


def test_a_round_that_folds_almost_nothing_is_not_counted_as_a_round() -> None:
    rows = [{"part_tokens": [5000, 100, 300, 50]}, {"part_tokens": [400, 400, 400, 400]}]
    assert effective_rounds(rows) == {"cases": 2, "median": 3.0, "smallest": 2, "largest": 4}


def _summary(arm: str, by_task: dict, *, valid: bool = True, knows: tuple[int, int] | None = None) -> dict:
    obeyed, applies = sum(c["obeyed"] for c in by_task.values()), sum(c["applies"] for c in by_task.values())
    score = {"obeyed": obeyed, "applies": applies, "by_task": by_task, "looks": 1, "finds": 0}
    if knows:
        score |= {"knows": knows[0], "knows_of": knows[1]}
    return {"arm": arm, "valid": valid, "calls": 10, "compactions": 2, "no_command": 1, "scores": {"rule": score},
            "outcomes": [{"task": task, "exit_status": "Submitted"} for task in by_task]}


def test_an_arm_is_compared_with_compaction_only_on_tasks_both_reached() -> None:
    runs = {"seed": {"compaction": _summary("compaction", {"a": {"obeyed": 0, "applies": 2}, "b": {"obeyed": 1, "applies": 1}, "c": {"obeyed": 1, "applies": 1}}),
                     "taste": _summary("taste", {"a": {"obeyed": 2, "applies": 2}, "b": {"obeyed": 0, "applies": 1}})}}
    pair = long_analyze.compare_rule(runs, "taste", "rule")
    assert (pair["tasks"], pair["better"], pair["worse"], pair["same"]) == (2, 1, 1, 0)
    assert (pair["obeyed"], pair["applies"], pair["base_obeyed"], pair["base_applies"]) == (2, 3, 1, 3)
    assert pair["p"] == 1.0


def test_a_run_the_harness_made_invalid_is_left_out_of_every_comparison() -> None:
    runs = {"seed": {"compaction": _summary("compaction", {"a": {"obeyed": 0, "applies": 1}}), "taste": _summary("taste", {"a": {"obeyed": 1, "applies": 1}}, valid=False)}}
    assert long_analyze.compare_rule(runs, "taste", "rule")["tasks"] == 0


def test_the_arm_table_pools_seeds_and_shows_ran_and_known(tmp_path: Path) -> None:
    for seed in ("s1", "s2"):
        for arm in ("compaction", "taste"):
            folder = tmp_path / seed / arm
            folder.mkdir(parents=True)
            (folder / "summary.json").write_text(json.dumps(_summary(arm, {"a": {"obeyed": 1, "applies": 1}}, knows=(1, 1))))
    table = long_analyze.arm_table(long_analyze.load([tmp_path / "s1", tmp_path / "s2"]))
    taste = next(entry for entry in table if entry["arm"] == "taste")
    assert taste["sessions"] == 2 and taste["rules"]["rule"]["obeyed"] == 2 and taste["rules"]["rule"]["knows"] == 2
    assert taste["rules"]["rule"]["against_compaction"]["tasks"] == 2
    assert "knows 2/2" in long_analyze.render(table)
