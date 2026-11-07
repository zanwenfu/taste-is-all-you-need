"""The rule-loss tables and the long run's tables, worked out by hand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from studies.long_run import analyze as long_analyze
from studies.memory_probe.analyze import (
    effective_rounds,
    floors_from,
    load_run,
    load_stored,
    one_code,
    render,
    render_rules,
    rounds_from,
    tables,
)


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


def test_a_cued_case_is_read_against_its_own_no_rule_rate(tmp_path: Path) -> None:
    baseline = tmp_path / "results.jsonl"
    baseline.write_text("".join(json.dumps({"id": f"h0:{n}:cued", "full_without_sc_compliant": n == 0}) + "\n" for n in range(2)))
    assert floors_from([baseline]) == {"h0:0:cued": True, "h0:1:cued": False}


def test_a_budget_sweep_beside_the_main_rows_does_not_replace_them(tmp_path: Path) -> None:
    _write(tmp_path, "turns", [_row(0, obeyed=True)], name="a5q-i-anthropic-k1-b4096")
    _write(tmp_path, "turns", [_row(0, obeyed=False)], name="a5q-i-anthropic-k1-b8000")
    assert load_run(tmp_path, "anthropic", 1)["turns"][0]["compliant"] is True
    assert load_run(tmp_path, "anthropic", 1, budget=8000)["turns"][0]["compliant"] is False


def test_a_plain_request_has_no_answer_to_compare_with_the_summary_alone(tmp_path: Path) -> None:
    _write(tmp_path, "summary", [_row(0, obeyed=True)])
    _write(tmp_path, "plain-nothing", [_row(0, obeyed=None, action="run a tool")])
    entries = {entry["row"]: entry for entry in tables(load_run(tmp_path, "anthropic", 1), {"h0:0": False})}
    assert "against_summary" not in entries["plain-nothing"] and "difference" not in entries["plain-nothing"]
    assert entries["plain-nothing"]["acted"] == 1


def test_each_rule_is_shown_for_every_row_that_has_answers(tmp_path: Path) -> None:
    _write(tmp_path, "restated", [_row(0, obeyed=True, rule=1), _row(1, obeyed=False, rule=2)])
    _write(tmp_path, "turns", [_row(0, obeyed=False, rule=1), _row(1, obeyed=False, rule=2)])
    _write(tmp_path, "plain-nothing", [_row(0, obeyed=None, rule=1), _row(1, obeyed=None, rule=2)])
    report = render_rules(tables(load_run(tmp_path, "anthropic", 1), {}))
    lines = report.splitlines()
    assert "| restated | 1/1 | 0/1 |" in lines and "| turns | 0/1 | 0/1 |" in lines
    assert "plain-nothing" not in report


def test_rounds_are_read_from_the_compaction_records_of_that_depth(tmp_path: Path) -> None:
    folder = tmp_path / "h00-05"
    folder.mkdir()
    rows = [{"id": "a", "k": 2, "part_tokens": [5, 5]}, {"id": "a", "k": 16, "part_tokens": [500, 10, 300]}]
    (folder / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert rounds_from(tmp_path, 16) == {"cases": 1, "median": 2, "smallest": 2, "largest": 2}


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


def test_obedience_is_shown_against_the_compactions_so_far() -> None:
    def summary(cells: dict[str, list[int]]) -> dict:
        return {"valid": True, "calls": 1, "compactions": 1, "outcomes": [],
                "scores": {"python_B": {"obeyed": 0, "applies": 0, "finds": 0, "looks": 0, "by_task": {}, "by_compactions": cells}}}

    runs = {"seed1": {"compaction": summary({"0": [2, 2], "2": [1, 2], "12": [0, 4]})},
            "seed2": {"compaction": summary({"1": [1, 1], "30": [0, 1]})}}
    table = long_analyze.compaction_table(runs)
    assert table["python_B"]["compaction"] == {"0": [2, 2], "1-2": [2, 3], "11-20": [0, 4], "21-50": [0, 1]}
    text = long_analyze.render_compactions(table)
    assert "| compaction | 2/2 | 2/3 | - | - | 0/4 | 0/1 | - | - |" in text.splitlines()


# ------------------------------------------------------------------ the third review


def test_an_unread_answer_leaves_the_rate_as_the_benchmark_leaves_it(tmp_path: Path) -> None:
    ids = range(20)
    _write(tmp_path, "summary", [_row(n, obeyed=False) for n in ids])
    _write(tmp_path, "restated", [_row(n, obeyed=True) for n in ids])
    _write(tmp_path, "turns", [_row(n, obeyed=None if n < 8 else True) for n in ids])
    entries = {entry["row"]: entry for entry in tables(load_run(tmp_path, "anthropic", 1), {f"h0:{n}": False for n in ids})}
    assert entries["turns"]["effective_retention"] == pytest.approx(100.0)
    assert entries["turns"]["difference"][0] == pytest.approx(100.0), "only the pairs both read"


def test_a_withdrawn_row_is_read_against_the_no_rule_rate_and_has_no_effective_retention(tmp_path: Path) -> None:
    ids = [f"h0:{n}:withdrawn" for n in range(4)]

    def rows(pattern: list[bool]) -> list[dict]:
        return [{**_row(n, obeyed=value), "id": ids[n]} for n, value in enumerate(pattern)]

    _write(tmp_path, "summary", rows([False, False, False, False]))
    _write(tmp_path, "restated", rows([False, False, True, False]))
    _write(tmp_path, "turns", rows([True, True, True, False]))
    entries = {entry["row"]: entry for entry in tables(load_run(tmp_path, "anthropic", 1), dict.fromkeys(ids, False))}
    turns = entries["turns"]
    assert "effective_retention" not in turns and turns["still_obeying"] == 3 and turns["no_rule"] == 0
    assert "better" not in turns["against_summary"] and turns["against_summary"]["more_often"] == 3
    report = render(list(entries.values()), "withdrawn")
    assert "still obeying" in report


def test_the_stored_step_is_shown_for_each_kind_of_row(tmp_path: Path) -> None:
    def row(n: int, arm: str, **extra) -> dict:
        return _row(n, obeyed=True, arm=arm, rule_turns=[1] if n else [], **extra)

    _write(tmp_path, "turns", [row(n, "a5q") for n in range(2)])
    _write(tmp_path, "notes-last-told", [row(n, "a2cq") for n in range(2)], name="a2cq-i-anthropic-k1-b4096")
    _write(tmp_path, "notes-every-told", [row(n, "a2dq") for n in range(2)], name="a2dq-i-anthropic-k1-b4096")
    _write(tmp_path, "rules", [row(n, "a6") for n in range(2)], [{"id": f"h0:{n}", "found_by_judge": n == 1} for n in range(2)], name="a6-i-anthropic-k1-b4096")
    _write(tmp_path, "summary", [row(n, "a1") for n in range(2)], name="a1-i-anthropic-k1-b4096")
    (tmp_path / "stored-told.served.jsonl").write_text("".join(json.dumps(line) + "\n" for line in (
        {"id": "h0:0", "newest": False, "any": True, "asked": 2}, {"id": "h0:1", "newest": False, "any": False, "asked": 2})))
    entries = {entry["row"]: entry for entry in tables(load_run(tmp_path, "anthropic", 1), {}, stored=load_stored(tmp_path))}
    assert entries["turns"]["stored"] == 1, "every turn is there, and the row says how many cases have the rule's whole text in one"
    assert entries["notes-last-told"]["stored"] == 0 and entries["notes-every-told"]["stored"] == 1
    assert entries["rules"]["stored"] == 1
    assert entries["summary"].get("stored") is None


def test_a_verdict_the_judge_could_not_give_is_counted_apart(tmp_path: Path) -> None:
    _write(tmp_path, "turns", [_row(n, obeyed=True) for n in range(3)], [{"id": "h0:0", "found_by_judge": True}, {"id": "h0:1", "found_by_judge": None}, {"id": "h0:2", "found_by_judge": False}])
    entry = tables(load_run(tmp_path, "anthropic", 1), {})[0]
    assert (entry["held"], entry["held_unclear"]) == (1, 1) and entry["obeyed_if_not"] == [1, 1]


def test_a_table_refuses_rows_from_two_versions_of_the_code(tmp_path: Path) -> None:
    for label, digest in (("turns", "one"), ("rules", "two")):
        _write(tmp_path, label, [_row(0, obeyed=True)])
        (tmp_path / label / "a5q-i-anthropic-k1-b4096" / "results.settings.json").write_text(json.dumps({"code_digest": digest, "commit": digest}))
    with pytest.raises(ValueError, match="two versions"):
        one_code(tmp_path, "anthropic", 1)
    (tmp_path / "rules" / "a5q-i-anthropic-k1-b4096" / "results.settings.json").write_text(json.dumps({"code_digest": "one", "commit": "one"}))
    assert one_code(tmp_path, "anthropic", 1) == {"code_digest": "one", "commit": "one"}


def test_the_long_run_totals_leave_out_invalid_sessions_and_say_how_sessions_ended() -> None:
    good = {**_summary("taste", {"a": {"obeyed": 1, "applies": 2}}), "ended_by": None}
    bad = {**_summary("taste", {"b": {"obeyed": 2, "applies": 2}}, valid=False), "ended_by": "RuntimeError: no model"}
    table = {entry["arm"]: entry for entry in long_analyze.arm_table({"s1": {"taste": good}, "s2": {"taste": bad}})}
    assert table["taste"]["rules"]["rule"]["obeyed"] == 1 and table["taste"]["rules"]["rule"]["applies"] == 2
    assert table["taste"]["invalid"] == 1 and table["taste"]["sessions"] == 2 and table["taste"]["ended_early"] == 1


def _search(task: str, told: bool, block: bool, talk: bool) -> dict:
    return {"task": task, "rules": {"rule": {"told_before": told, "in_block": block, "instruction_in_block": block, "in_talk": talk}}}


def test_the_long_runs_search_arm_reads_only_tasks_that_began_after_the_rule_was_told() -> None:
    summary = _summary("taste_search", {"first": {"obeyed": 3, "applies": 3}, "a": {"obeyed": 1, "applies": 2}, "b": {"obeyed": 0, "applies": 3}, "c": {"obeyed": 2, "applies": 2}, "d": {"obeyed": 4, "applies": 5}})
    summary["searches"] = [_search("first", False, False, False), _search("a", True, True, True), _search("b", True, False, False), _search("c", True, False, True),
                           _search("d", True, True, False), _search("gone", True, True, False)]
    entry = long_analyze.arm_table({"s1": {"taste_search": summary}})[0]
    cell = entry["rules"]["rule"]
    assert entry["searched"] == 6
    assert cell["could_be_back"] == 5 and cell["in_block"] == 3  # the first task is where it is told, so its search could not hold it
    assert cell["obeyed_with_block"] == [5, 7] and cell["obeyed_in_talk_only"] == [2, 2] and cell["obeyed_in_neither"] == [0, 3]
    text = long_analyze.render([entry])
    assert "back at 3 of 5 searches after it was told" in text and "obeyed 5/7 with it, 2/2 in the talk only, 0/3 in neither" in text
    other = long_analyze.arm_table({"s1": {"taste": _summary("taste", {"a": {"obeyed": 1, "applies": 1}})}})[0]
    assert other["searched"] == 0 and "in_block" not in other["rules"]["rule"]


def test_the_long_run_compares_sessions_since_tasks_of_one_conversation_are_not_independent() -> None:
    def seed(mine: int, base: int) -> dict:
        return {"taste": _summary("taste", {"a": {"obeyed": mine, "applies": 4}}), "compaction": _summary("compaction", {"a": {"obeyed": base, "applies": 4}})}

    runs = {"s1": seed(3, 1), "s2": seed(4, 2), "s3": seed(1, 1), "s4": seed(4, 0)}
    pair = long_analyze.compare_rule(runs, "taste", "rule")
    assert (pair["sessions"], pair["sessions_better"], pair["sessions_worse"]) == (4, 3, 0)
    assert pair["sessions_p"] == pytest.approx(0.25)


# ------------------------------------------------------------------ the fourth review

def test_every_rate_by_rule_and_by_what_was_held_leaves_out_unread_answers(tmp_path: Path) -> None:
    ids = range(10)
    _write(tmp_path, "summary", [_row(n, obeyed=False) for n in ids])
    _write(tmp_path, "restated", [_row(n, obeyed=True) for n in ids])
    _write(tmp_path, "turns", [_row(n, obeyed=True if n < 3 else False if n < 5 else None, held=True) for n in ids])
    entry = next(e for e in tables(load_run(tmp_path, "anthropic", 1), {f"h0:{n}": False for n in ids}) if e["row"] == "turns")
    assert entry["per_rule"] == {"sc01": [3, 5]} and entry["obeyed_if_held"] == [3, 5] and entry["obeyed_where_restated_is"] == [3, 5]


def test_a_notes_file_the_judge_could_not_read_is_counted_apart_from_one_without_the_rule(tmp_path: Path) -> None:
    ids = range(4)
    _write(tmp_path, "notes-newest-told", [_row(n, obeyed=True, arm="a2cw") for n in ids], name="a2cw-i-anthropic-k1-b4096")
    stored = {"told": {"h0:0": {"id": "h0:0", "newest": True, "any": True}, "h0:1": {"id": "h0:1", "newest": None, "any": None}, "h0:2": {"id": "h0:2", "newest": False, "any": False},
                       "h0:3": {"id": "h0:3", "newest": True, "any": True}}}
    entry = tables(load_run(tmp_path, "anthropic", 1), {f"h0:{n}": False for n in ids}, None, stored)[0]
    assert (entry["stored"], entry["stored_unclear"]) == (2, 1)


def test_a_session_that_died_with_no_summary_is_counted_rather_than_lost(tmp_path: Path) -> None:
    for arm, whole in (("compaction", True), ("taste", False)):
        folder = tmp_path / "s1" / arm
        folder.mkdir(parents=True)
        (folder / "actions.jsonl").write_text("{}\n")
        if whole:
            (folder / "summary.json").write_text(json.dumps(_summary(arm, {"a": {"obeyed": 1, "applies": 1}})))
    table = {entry["arm"]: entry for entry in long_analyze.arm_table(long_analyze.load([tmp_path / "s1"]))}
    assert (table["taste"]["sessions"], table["taste"]["invalid"], table["taste"]["ended_early"], table["taste"]["calls"]) == (1, 1, 1, 0)
    assert table["compaction"]["sessions"] == 1 and table["compaction"]["invalid"] == 0


def _with(summary: dict, **versions) -> dict:
    return {**summary, "versions": versions}


def test_one_long_run_table_holds_one_reader_one_summary_prompt_and_one_version_of_the_code() -> None:
    base = _summary("taste", {"a": {"obeyed": 1, "applies": 1}})
    same = {"s1": {"taste": _with(base, model="m", summary_prompt="pi", commit="c1")}, "s2": {"taste": _with(base, model="m", summary_prompt="pi", commit="c1")}}
    assert long_analyze.one_setup(same) == {"model": "m", "summary_prompt": "pi", "commit": "c1"}
    two_codes = {"s1": {"taste": _with(base, model="m", summary_prompt="pi", commit="c1")}, "s2": {"taste": _with(base, model="m", summary_prompt="pi", commit="c2")}}
    with pytest.raises(ValueError, match="version of the code"):
        long_analyze.one_setup(two_codes)
    assert long_analyze.one_setup(two_codes, mixed=True)["commit"] == "c1, c2"
    for key, other in (("model", {"model": "other", "summary_prompt": "pi", "commit": "c1"}), ("summary_prompt", {"model": "m", "summary_prompt": "other", "commit": "c1"})):
        with pytest.raises(ValueError, match=key):
            long_analyze.one_setup({"s1": {"taste": _with(base, model="m", summary_prompt="pi", commit="c1")}, "s2": {"taste": _with(base, **other)}}, mixed=True)


def test_commands_that_go_back_to_an_earlier_tasks_files_are_counted_arm_by_arm(tmp_path: Path) -> None:
    seed = tmp_path / "s1"
    tasks = [{"name": "circuit", "instruction": "The file /app/sim.c is a simulator. Create /app/gates.txt and write /app/answer.txt."},
             {"name": "count", "instruction": "Count the tokens and write /app/answer.txt, with count.py if you like."},
             {"name": "fix", "instruction": "Fix bottle.py."}]
    seed.mkdir()
    (seed / "session.json").write_text(json.dumps({"tasks": tasks}))

    def arm(name: str, rows: list[tuple[str, str, str]], valid: bool | None = True) -> None:
        (seed / name).mkdir()
        (seed / name / "actions.jsonl").write_text("\n".join(json.dumps({"kind": kind, "task": task, "command": command}) for task, kind, command in rows))
        if valid is not None:
            (seed / name / "summary.json").write_text(json.dumps({"valid": valid}))

    arm("compaction", [("circuit", "bash", "cat /app/sim.c"), ("count", "bash", "python3 count.py"), ("fix", "bash", "sed -n 1,9p bottle.py")])
    arm("taste_search", [("circuit", "bash", "cat /app/gates.txt"),
                         ("count", "bash", "cat /app/sim.c"),  # back at the first task
                         ("count", "bash", "gcc sim.c && python3 count.py"),  # names this task's file too: not counted
                         ("count", "bash", "cat /app/answer.txt"),  # two tasks name it, so it is nobody's own
                         ("count", "no_command", ""),
                         ("fix", "bash", "cd /app && cat gates.txt && cat count.py")])  # back at both

    arm("taste", [("count", "bash", "cat /app/sim.c")], valid=False)
    arm("notes_told", [("count", "bash", "cat /app/sim.c")], valid=None)  # died before it could write a summary

    table = long_analyze.went_back([seed])
    assert set(table) == {"compaction", "taste_search"}, "an invalid session is in no total"
    assert table["compaction"]["commands"] == 3 and table["compaction"]["back"] == 0
    assert table["taste_search"]["commands"] == 5 and table["taste_search"]["back"] == 2
    assert table["taste_search"]["by_task"] == {"count": 1, "fix": 1}
    text = long_analyze.render_went_back(table)
    assert "taste_search" in text and "2/5" in text


def test_a_name_any_agent_might_give_its_own_file_says_nothing_of_going_back(tmp_path: Path) -> None:
    seed = tmp_path / "s1"
    (seed / "taste").mkdir(parents=True)
    (seed / "session.json").write_text(json.dumps({"tasks": [{"name": "one", "instruction": "Write tests in test.py and /app/sim.c."}, {"name": "two", "instruction": "Fix bottle.py."}]}))
    (seed / "taste" / "summary.json").write_text(json.dumps({"valid": True}))
    (seed / "taste" / "actions.jsonl").write_text("\n".join(json.dumps({"kind": "bash", "task": "two", "command": command}) for command in ("python3 test.py", "cat sim.c")))
    assert long_analyze.went_back([seed])["taste"]["back"] == 1


def test_a_folder_with_no_task_list_has_nothing_to_count_for_going_back(tmp_path: Path) -> None:
    (tmp_path / "s1" / "compaction").mkdir(parents=True)
    assert long_analyze.went_back([tmp_path / "s1"]) == {}


def test_the_table_shows_summaries_asked_again_and_folds_that_waited(tmp_path: Path) -> None:
    summary = {**_summary("compaction", {"a": {"obeyed": 1, "applies": 1}}), "summary_replies": {"asked": 9, "not_a_summary": 2}, "folds_waited": [40]}
    entry = long_analyze.arm_table({"s1": {"compaction": summary}})[0]
    assert entry["summaries_asked"] == 9 and entry["not_a_summary"] == 2 and entry["folds_waited"] == 1
    assert "not a summary" in long_analyze.render([entry])
