"""Whether a header list delivered the rule, read line by line: the rule turn's own line, and whether that line alone states the rule."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from studies.memory_probe.cases import _index_lines
from studies.memory_probe.header_line import check_header_lines
from studies.memory_probe.headers import turn_key

RULE = "Don't ask me to confirm before running commands or making changes, just do them."
RULE_LINE = "3 the user says to stop asking before running commands [confirm, commands]"  # the line of the message at place 2: lines count from 1, the system prompt first


def _count(text: str) -> int:
    return len(text.split())


def _row(tmp_path: Path, *, budget: int = 1000, found: bool = False, plain: bool = False, skip_header: bool = False) -> tuple[Path, Path, Path]:
    """One case of a header-list row as the cell writes it: the delivered block is the real index's lines, cut to the budget, the newest kept."""
    rule_at = 2
    turns = [{"role": "system", "content": "be useful"}] + [{"role": "user" if n % 2 else "assistant", "content": f"turn {n} about the build"} for n in range(1, 7)]
    turns[rule_at] = {"role": "user", "content": f"Also, for the rest of this session. {RULE.lower()}"}
    wrote = {n: (f"what turn {n} is about", [] if plain else [f"word{n}"]) for n in range(1, 7)}
    if not plain:
        wrote[rule_at] = ("the user says to stop asking before running commands", ["confirm", "commands"])
    if skip_header:
        wrote[rule_at] = ("", [])
    head_rows = [{"key": turn_key(turns[n]), "ok": True, "what": what, "keywords": words} for n, (what, words) in wrote.items() if not (skip_header and n == rule_at)]
    lines, _ = _index_lines([{"turn": n, "what": wrote[n][0], "keywords": wrote[n][1]} for n in range(1, 7)], budget, _count)
    cases, heads, results = tmp_path / "cases.jsonl", tmp_path / "turn-headers.jsonl", tmp_path / "row" / "results.jsonl"
    cases.write_text(json.dumps({"id": "c1", "sc_text": RULE, "messages": turns}) + "\n")
    heads.write_text("".join(json.dumps(row) + "\n" for row in head_rows))
    results.parent.mkdir()
    results.write_text(json.dumps({"id": "c1", "arm": "a3", "delivered_text": "\n".join(lines), "index_lines": len(lines), "rule_turns": [rule_at],
                                   "found": found, "rule_in_index": False}) + "\n")
    sha = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (("cases_sha256", cases), ("headers_sha256", heads))}
    (results.parent / "results.settings.json").write_text(json.dumps({"arm": "a3", **sha}))
    return results, cases, heads


def _verdicts(said: list) -> Callable[[str, list[dict]], bool | None]:
    def verdict(rule: str, delivered: list[dict]) -> bool | None:
        said.append((rule, delivered))
        return "stop asking" in delivered[0]["content"]
    return verdict


def _read(results: Path) -> dict:
    return json.loads(results.with_name("header-line.served.jsonl").read_text().splitlines()[0])


def test_the_rule_turns_line_delivered_and_stating_the_rule_is_held_and_the_judge_reads_that_line_alone(tmp_path: Path) -> None:
    results, cases, heads = _row(tmp_path)
    said: list = []
    counts = check_header_lines(results, cases, heads, _verdicts(said), workers=1)
    assert said == [(RULE, [{"role": "assistant", "content": RULE_LINE}])], "one line, as the header file gives it, and nothing else of the list"
    row = _read(results)
    assert row["line_delivered"] is True and row["states_rule"] is True and row["held"] is True and row["line"] == RULE_LINE
    assert counts == {"cases": 1, "line_delivered": 1, "states_rule": 1, "fetched": 0, "held": 1, "unclear": 0, "differs": 0, "missing_header": 0, "no_rule_turn": 0,
                      "asked": 1}


def test_a_rule_line_the_budget_cut_is_not_delivered_and_asks_no_judge(tmp_path: Path) -> None:
    # The newest lines stay: at 30 words only the four lines after the rule's fit.
    results, cases, heads = _row(tmp_path, budget=30)
    assert "stop asking" not in json.loads(results.read_text())["delivered_text"]
    said: list = []
    counts = check_header_lines(results, cases, heads, _verdicts(said), workers=1)
    assert said == [] and _read(results)["line_delivered"] is False and _read(results)["held"] is False
    assert counts["line_delivered"] == 0 and counts["held"] == 0 and counts["asked"] == 0


def test_a_fetch_that_brought_the_rule_back_holds_it_whatever_the_line_says(tmp_path: Path) -> None:
    results, cases, heads = _row(tmp_path, found=True, plain=True)
    counts = check_header_lines(results, cases, heads, _verdicts([]), workers=1)
    row = _read(results)
    assert row["line_delivered"] is True and row["states_rule"] is False and row["fetched"] is True and row["held"] is True
    assert counts["held"] == 1 and counts["fetched"] == 1 and counts["states_rule"] == 0


def test_a_line_the_judge_cannot_read_is_unclear_not_held(tmp_path: Path) -> None:
    results, cases, heads = _row(tmp_path)
    counts = check_header_lines(results, cases, heads, lambda rule, delivered: None, workers=1)
    assert _read(results)["held"] is None and counts["unclear"] == 1 and counts["held"] == 0


def test_a_rule_turn_with_no_header_is_counted_and_its_bare_number_asks_no_judge(tmp_path: Path) -> None:
    results, cases, heads = _row(tmp_path, skip_header=True)
    said: list = []
    counts = check_header_lines(results, cases, heads, _verdicts(said), workers=1)
    assert said == [] and counts["missing_header"] == 1 and counts["held"] == 0 and _read(results)["line"] == "3 "


def test_a_delivered_line_other_than_the_header_files_is_counted_and_the_delivered_one_is_judged(tmp_path: Path) -> None:
    results, cases, heads = _row(tmp_path)
    row = json.loads(results.read_text())
    row["delivered_text"] = row["delivered_text"].replace("stop asking before running", "stop asking before  running")
    results.write_text(json.dumps(row) + "\n")
    said: list = []
    counts = check_header_lines(results, cases, heads, _verdicts(said), workers=1)
    assert counts["differs"] == 1 and said[0][1][0]["content"] == RULE_LINE.replace("before running", "before  running")


def test_a_delivered_line_that_only_begins_as_the_header_files_does_is_not_taken_for_it(tmp_path: Path) -> None:
    results, cases, heads = _row(tmp_path)
    row = json.loads(results.read_text())
    row["delivered_text"] = row["delivered_text"].replace(RULE_LINE, RULE_LINE + " and more")
    results.write_text(json.dumps(row) + "\n")
    said: list = []
    counts = check_header_lines(results, cases, heads, _verdicts(said), workers=1)
    assert counts["differs"] == 1 and said[0][1][0]["content"] == RULE_LINE + " and more"


def test_a_cases_or_headers_file_other_than_the_rows_own_is_refused_before_any_call(tmp_path: Path) -> None:
    results, cases, heads = _row(tmp_path)
    heads.write_text(heads.read_text() + json.dumps({"key": "x", "ok": True, "what": "w", "keywords": []}) + "\n")
    said: list = []
    with pytest.raises(ValueError, match="headers"):
        check_header_lines(results, cases, heads, _verdicts(said), workers=1)
    assert said == [] and not results.with_name("header-line.served.jsonl").exists()


def test_the_reader_names_a_turn_as_the_header_writer_does_and_loads_no_git_library() -> None:
    # The judge's environment on the cluster has the benchmark package but no git library; Taste's store needs one.
    from studies.memory_probe import header_line, headers
    for turn in ({"role": "user", "content": "plain"}, {"role": "assistant", "content": "naïve 日本語 \udcff"}, {"role": "tool", "content": ""}):
        assert header_line.turn_key(turn) == headers.turn_key(turn)
    loaded = subprocess.run([sys.executable, "-c", "import sys, studies.memory_probe.header_line; print([m for m in sys.modules if m.split('.')[0] in ('git', 'taste')])"],
                            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1], check=True).stdout.strip()
    assert loaded == "[]"


def test_a_run_again_asks_only_what_has_no_verdict(tmp_path: Path) -> None:
    results, cases, heads = _row(tmp_path)
    check_header_lines(results, cases, heads, _verdicts([]), workers=1)
    said: list = []
    counts = check_header_lines(results, cases, heads, _verdicts(said), workers=1)
    assert said == [] and counts["cases"] == 1 and counts["held"] == 1 and counts["asked"] == 1
