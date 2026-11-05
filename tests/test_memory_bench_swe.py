"""ConPact's SWE-Natural set in the comparison: one token cap, no query, compliance after one compaction."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from studies.memory_bench.adapters.common import QueryFree, run_adapter
from studies.memory_bench.adapters.mem0_arm import Mem0Arm
from studies.memory_bench.adapters.mempalace_arms import ProductArm
from studies.memory_bench.core import fill
from studies.memory_bench.swe import FIXED_DATE, answer_all, read_blocks, views_of
from studies.memory_bench.systems import FlatBm25, TasteGit
from studies.memory_ledger import check
from taste.memstore import CommitHeader

RULE = "the server must not close the connection until QUIT from the client"
PROMPT = 'Assume ...\n\nRequest from the user:\n"Fix the loop."\n\nTwo candidate behaviors:\nA) Close.\nB) Keep open.\n\nOne letter.'


@pytest.fixture(autouse=True)
def _pinned_code(monkeypatch) -> None:
    monkeypatch.setenv("TASTE_COMMIT", "abc123")


def _case(number: int = 0) -> dict:
    return {"id": f"swe:{number:03d}:sc00", "haystack_id": f"swe:{number:03d}", "sc_id": 0, "sc_type": "Action", "sc_text": RULE,
            "compliant_letter": "B", "system_prompt": "You are an agent.", "developer_prompt": "Pick one.", "probe": "Fix the loop.",
            "probe_prompt": PROMPT, "post_sc_probe_prompt": f"{RULE}\n\n{PROMPT}",
            "messages": [{"role": "user", "content": f"Issue. {RULE}."}, {"role": "assistant", "content": "Reading."},
                         {"role": "user", "content": "OBSERVATION: 3 files."}, {"role": "assistant", "content": "Editing."}]}


def test_each_round_of_a_log_is_a_session_of_its_own_and_nothing_marks_the_rule() -> None:
    (unit,) = views_of([_case()])
    assert [[t["content"] for t in s["turns"]] for s in unit["sessions"]] == [[f"Issue. {RULE}.", "Reading."],
                                                                               ["OBSERVATION: 3 files.", "Editing."]]
    assert {s["date"] for s in unit["sessions"]} == {FIXED_DATE}
    assert unit["queries"] == [{"qid": unit["queries"][0]["qid"], "query": "Fix the loop."}]
    text = json.dumps(unit)
    assert "sc00" not in text and "swe:000" not in text and "compliant" not in text and "A) Close" not in text


def test_a_block_fills_in_the_systems_own_order_and_stops_at_the_cap() -> None:
    count = lambda text: len(text.split())  # noqa: E731
    assert fill(["a b", "c d", "e f"], 4, count) == "a b\n\nc d"
    assert fill(["a b", "c d e f g", "h"], 4, count) == "a b\n\nc d", "the one that does not fit is cut, and filling stops"
    assert fill(["one two three"], 2, count) == "one two"
    assert fill([], 10, count) == ""


def _sessions() -> list[dict]:
    return views_of([_case()])[0]["sessions"]


def test_the_keyword_list_gives_its_newest_records_first(tmp_path: Path) -> None:
    system = FlatBm25()
    system.ingest(_sessions())
    first, second = system.memories()
    assert "OBSERVATION" in first and RULE in second


def test_taste_gives_its_newest_records_read_back_from_the_store(tmp_path: Path) -> None:
    system = TasteGit(tmp_path, mode="delivered")
    try:
        system.ingest(_sessions())
        first, second = system.memories()
        assert "OBSERVATION" in first and RULE in second
        assert system.ledger()["mode"] == "delivered"
        check([system.ledger()])
    finally:
        system.close()


def test_taste_with_headers_gives_what_each_commit_says_it_holds_newest_first(tmp_path: Path) -> None:
    writer = lambda session: CommitHeader(what=session["turns"][1]["content"], keywords=("x",))  # noqa: E731
    system = TasteGit(tmp_path, header_writer=writer, mode="delivered")
    try:
        system.ingest(_sessions())
        first, second = system.memories()
        assert 'what: "Editing."' in first and 'what: "Reading."' in second
        assert RULE not in first + second
    finally:
        system.close()


def test_a_system_with_no_query_is_asked_for_what_it_would_put_in_front_of_an_agent(tmp_path: Path) -> None:
    views = tmp_path / "views.jsonl"
    views.write_text(json.dumps(views_of([_case()])[0]) + "\n")
    out = tmp_path / "rankings.jsonl"
    run_adapter(views, out, lambda: QueryFree(FlatBm25()), system="flat_free", code="c", versions={})
    (row,) = [json.loads(line) for line in out.read_text().splitlines()]
    assert [item["turns"][0]["role"] for item in row["items"]] == ["memory", "memory"]
    assert "OBSERVATION" in row["items"][0]["turns"][0]["content"]
    assert row["ledger"]["name"] == "baseline"


def test_an_adapter_with_a_ledger_of_its_own_keeps_it_on_each_line(tmp_path: Path) -> None:
    views = tmp_path / "views.jsonl"
    views.write_text(json.dumps(views_of([_case()])[0]) + "\n")
    out = tmp_path / "rankings.jsonl"
    (tmp_path / "w").mkdir()
    run_adapter(views, out, lambda: QueryFree(TasteGit(tmp_path / "w", mode="delivered")), system="taste_free", code="c",
                versions={})
    (row,) = [json.loads(line) for line in out.read_text().splitlines()]
    assert row["ledger"]["name"] == "Taste memory, study-driven" and row["ledger"]["mode"] == "delivered"
    assert row["ledger"]["memstore_imported"] is True
    check([row["ledger"]])


def test_mem0_gives_all_its_memories_newest_first() -> None:
    class Memory:
        def reset(self) -> None:
            pass

        def add(self, *args, **kwargs) -> dict:
            return {"results": []}

        def get_all(self, *, filters, top_k):
            assert filters == {"user_id": "eval"} and top_k >= 100_000
            return {"results": [{"memory": "old", "created_at": "2026-01-01T00:00:00"},
                                {"memory": "new", "created_at": "2026-01-02T00:00:00"}]}

    arm = Mem0Arm(Memory())
    arm.ingest(_sessions())
    assert arm.memories() == ["new", "old"]


def test_mempalace_gives_its_wake_up_text(tmp_path: Path) -> None:
    class Stack:
        def __init__(self, palace_path, identity_path):
            self.palace_path, self.identity_path = palace_path, identity_path

        def wake_up(self, wing):
            assert wing == "eval" and self.palace_path.endswith("palace")
            return "L0\n\nL1 essential story"

    arm = ProductArm(lambda **kwargs: None, lambda *a, **k: {"results": []}, tmp_path, stack=Stack)
    arm.ingest(_sessions())
    assert arm.memories() == ["L0\n\nL1 essential story"]
    arm.close()


def _reply(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None), finish_reason="stop")],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))


def test_the_answer_stage_asks_with_the_block_in_place_of_the_history_and_grades_the_letter(tmp_path: Path) -> None:
    sent = []

    def chat(messages, tools):
        sent.append((messages, tools))
        return _reply("thinking\nB")

    case = _case()
    out = tmp_path / "answers.jsonl"
    summary = answer_all([case], {case["id"]: {"block": "the memory", "tokens": 2}}, chat, out, workers=1, system="s")
    ((messages, tools),) = sent
    assert tools is None
    assert [m["role"] for m in messages] == ["system", "developer", "assistant", "user"]
    assert messages[2]["content"] == "the memory" and messages[3]["content"] == PROMPT
    assert summary["cases"] == 1 and summary["compliant"] == 1 and summary["unreadable"] == 0
    (row,) = [json.loads(line) for line in out.read_text().splitlines()]
    assert row["compliant"] is True and row["block_tokens"] == 2 and row["final"] == "thinking\nB"


def test_an_unreadable_answer_is_counted_and_a_stopped_run_resumes(tmp_path: Path) -> None:
    cases = [_case(0), _case(1)]
    blocks = {case["id"]: {"block": "m", "tokens": 1} for case in cases}
    out = tmp_path / "answers.jsonl"
    answer_all(cases[:1], blocks, lambda m, t: _reply("no letter here"), out, workers=1, system="s")
    calls = []

    def chat(messages, tools):
        calls.append(1)
        return _reply("A")

    summary = answer_all(cases, blocks, chat, out, workers=1, system="s")
    assert len(calls) == 1 and summary["cases"] == 2 and summary["unreadable"] == 1 and summary["compliant"] == 0


def test_blocks_are_cut_to_the_cap_from_a_rankings_file_with_each_turns_words(tmp_path: Path) -> None:
    path = tmp_path / "rankings.jsonl"
    case = _case()
    (unit,) = views_of([case])
    qid = unit["queries"][0]["qid"]
    rows = [{"qid": qid, "system": "s", "ledger": {"name": "baseline"}, "items": [
        {"session": "a", "date": FIXED_DATE, "rounds": [], "source": "x", "turns": [{"role": "memory", "content": "one two"}]},
        {"session": "b", "date": FIXED_DATE, "rounds": [0], "source": "y",
         "turns": [{"role": "user", "content": "three"}, {"role": "assistant", "content": "four five six"}]}]}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    blocks = read_blocks(path, [case], cap=4, count=lambda text: len(text.split()))
    assert blocks[case["id"]]["block"] == "one two\n\nuser: three"
    assert blocks[case["id"]]["tokens"] == 4
