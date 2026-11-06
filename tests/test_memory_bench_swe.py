"""ConPact's SWE-Natural set in the comparison: one token cap, no query, compliance after one compaction."""

from __future__ import annotations

import hashlib
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


def test_the_header_writer_takes_the_coding_logs_round_by_round(tmp_path: Path) -> None:
    from studies.memory_bench.headers import write_all

    asked: list[str] = []

    def ask(prompt: str) -> dict:
        asked.append(prompt)
        return {"text": '{"what": "a round", "keywords": ["k"]}', "finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1}

    views = views_of([_case()])
    counts = write_all(views, ask, tmp_path / "headers.jsonl", workers=2)
    assert counts["failed"] == 0 and counts["calls"] == len({json.dumps(s["turns"]) for s in views[0]["sessions"]})


# ------------------------------------------------------------------ the set on one commit: what changed, and one whole table


def _block(case: dict, text: str, system: str = "s") -> dict:
    return {"id": case["id"], "block": text, "tokens": len(text.split()), "offered": 1, "system": system, "ledger": {"name": "baseline"}}


def _answer(case: dict, text: str, *, compliant: bool | None = True, system: str = "s") -> dict:
    return {"id": case["id"], "system": system, "sc_type": case["sc_type"], "final": "A", "finish_reason": "stop", "compliant": compliant,
            "block_tokens": len(text.split()), "usage": [1, 1], "prompt_sha256": hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()}


def _lines(rows: list[dict]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def test_an_answer_read_from_the_same_block_is_kept_and_one_whose_block_changed_is_left_to_be_asked_again(tmp_path: Path) -> None:
    from studies.memory_bench.swe import carry

    cases = [_case(0), _case(1), _case(2)]
    blocks = tmp_path / "blocks.jsonl"
    blocks.write_text(_lines([_block(cases[0], "same text"), _block(cases[1], "new text"), _block(cases[2], "third")]))
    old = tmp_path / "old" / "answers.jsonl"
    old.parent.mkdir()
    old.write_text(_lines([_answer(cases[0], "same text"), _answer(cases[1], "old text")]))
    out = tmp_path / "new" / "answers.jsonl"
    assert carry(blocks, old, out) == {"kept": 1, "changed": 1, "never_read": 1}
    assert [json.loads(line)["id"] for line in out.read_text().splitlines()] == [cases[0]["id"]]
    calls = []

    def chat(messages, tools):
        calls.append(messages[2]["content"])
        return _reply("B")

    summary = answer_all(cases, {row["id"]: row for row in map(json.loads, blocks.read_text().splitlines())}, chat, out, workers=1, system="s")
    assert sorted(calls) == ["new text", "third"] and summary["cases"] == 3


def test_carrying_answers_twice_changes_nothing_and_a_verdict_goes_only_with_its_block(tmp_path: Path) -> None:
    from studies.memory_bench.swe import carry, carry_verdicts

    cases = [_case(0), _case(1)]
    blocks = tmp_path / "blocks.jsonl"
    blocks.write_text(_lines([_block(cases[0], "same text"), _block(cases[1], "new text")]))
    old = tmp_path / "old.jsonl"
    old.write_text(_lines([_answer(cases[0], "same text"), _answer(cases[1], "old text")]))
    out = tmp_path / "answers.jsonl"
    carry(blocks, old, out)
    first = out.read_text()
    carry(blocks, old, out)
    assert out.read_text() == first
    judged = tmp_path / "judged-old"
    judged.mkdir()
    (judged / "results.jsonl").write_text(_lines([{"id": cases[0]["id"], "delivered_text": "same text"}, {"id": cases[1]["id"], "delivered_text": "old text"}]))
    (judged / "found.served.jsonl").write_text(_lines([{"id": cases[0]["id"], "found_by_judge": True}, {"id": cases[1]["id"], "found_by_judge": False}]))
    fresh = tmp_path / "judged-new"
    assert carry_verdicts(blocks, judged, fresh) == {"kept": 1, "to_judge": 1}
    assert [json.loads(line)["id"] for line in (fresh / "found.served.jsonl").read_text().splitlines()] == [cases[0]["id"]]
    assert [json.loads(line)["delivered_text"] for line in (fresh / "results.jsonl").read_text().splitlines()] == ["same text", "new text"]


def _whole_run(root: Path, cases: list[dict], *, commit: str = "c1", digest: str = "d1") -> None:
    from studies.memory_bench import swe

    (root / "cases").mkdir(parents=True)
    (root / "cases" / "cases.jsonl").write_text(_lines(cases))
    for name in ("blocks", "answers", "rankings", "judged", "baselines"):
        (root / name).mkdir()
    for row in swe.required_rows():
        name = row.rsplit("-", 1)[0]
        text = "" if name == "none" else f"text of {row}"
        (root / "blocks" / f"{row}.jsonl").write_text(_lines([_block(case, text, name) for case in cases]))
        (root / "answers" / f"{row}.jsonl").write_text(_lines([_answer(case, text, system=name) for case in cases]))
        if name != "none":
            folder = root / "judged" / row
            folder.mkdir()
            (folder / "results.jsonl").write_text(_lines([{"id": case["id"], "delivered_text": text} for case in cases]))
            (folder / "found.served.jsonl").write_text(_lines([{"id": case["id"], "found_by_judge": True} for case in cases]))
    for name in {row.rsplit("-", 1)[0] for row in swe.required_rows()} - {"none"}:
        versions = {"study_commit": swe.MEM0_INGEST_COMMIT, **swe.MEM0_INGEST} if name.startswith("mem0") else {"study_commit": commit, "code_digest": digest}
        (root / "rankings" / f"{name}.jsonl").write_text(_lines([{"qid": swe.neutral("q", case["id"]), "system": name, "items": [], "versions": versions} for case in cases]))
    for prompt in swe.BASELINE_PROMPTS:
        (root / "baselines" / f"baseline-{prompt}.report.json").write_text(json.dumps({"instances": len(cases), "package_summary": [
            {"source": "swe_natural", "sc_type": "all", "n": len(cases), "full_with_sc_compliance_pct": 50.0, "full_without_sc_compliance_pct": 50.0, "compaction_compliance_pct": 100.0, "upper_bound_compliance_pct": 100.0}]}))


def test_one_table_needs_every_row_every_case_each_answer_to_its_block_and_one_commit(tmp_path: Path) -> None:
    from studies.memory_bench import swe

    cases = [_case(n) for n in range(3)]
    _whole_run(tmp_path, cases)
    assert swe.check_coding(tmp_path, commit="c1", digest="d1") == []

    def broken(change, match: str) -> None:
        import shutil

        other = tmp_path.parent / "broken"
        shutil.rmtree(other, ignore_errors=True)
        shutil.copytree(tmp_path, other)
        change(other)
        problems = swe.check_coding(other, commit="c1", digest="d1")
        assert any(match in problem for problem in problems), problems

    broken(lambda r: (r / "answers" / "flat_free-1000.jsonl").unlink(), "flat_free-1000: no answers")
    broken(lambda r: (r / "blocks" / "flat_free-1000.jsonl").unlink(), "flat_free-1000: no blocks")
    broken(lambda r: (r / "answers" / "flat_free-1000.jsonl").write_text(_lines([_answer(cases[0], "text of flat_free-1000", system="flat_free")])), "flat_free-1000: answers hold 1 of 3")
    broken(lambda r: (r / "answers" / "flat_free-1000.jsonl").write_text(_lines([_answer(case, "something else", system="flat_free") for case in cases])), "flat_free-1000: 3 answers were read from another block")
    broken(lambda r: (r / "answers" / "flat_free-1000.jsonl").write_text(_lines([_answer(cases[0], "text of flat_free-1000", system="flat_free")] * 3)), "flat_free-1000: answers hold 1 of 3")
    broken(lambda r: (r / "judged" / "flat_free-1000" / "found.served.jsonl").unlink(), "flat_free-1000: no verdicts")
    broken(lambda r: (r / "judged" / "flat_free-1000" / "results.jsonl").write_text(_lines([{"id": case["id"], "delivered_text": "old"} for case in cases])), "flat_free-1000: the judge read another block")
    broken(lambda r: (r / "rankings" / "flat_free.jsonl").write_text(_lines([{"qid": "q", "system": "flat_free", "items": [], "versions": {"study_commit": "other", "code_digest": "d1"}}])), "flat_free: rankings made by")
    broken(lambda r: (r / "rankings" / "mem0f_free.jsonl").write_text(_lines([{"qid": "q", "system": "mem0_free", "items": [], "versions": {"study_commit": swe.MEM0_INGEST_COMMIT, **swe.MEM0_INGEST, "top_k": 10}}])), "mem0f_free: rankings made with")
    broken(lambda r: (r / "baselines" / "baseline-pi_mono.report.json").unlink(), "no baseline for pi_mono")
    broken(lambda r: (r / "blocks" / "stray-1000.jsonl").write_text(""), "stray-1000: not a row of the protocol")


def test_the_table_counts_obeyed_unread_and_held_and_compares_each_row_with_no_memory_and_with_taste(tmp_path: Path) -> None:
    from studies.memory_bench import swe

    cases = [_case(n) for n in range(4)]
    _whole_run(tmp_path, cases)
    write = lambda row, marks: (tmp_path / "answers" / f"{row}.jsonl").write_text(_lines([_answer(case, "" if row.startswith("none") else f"text of {row}", compliant=mark, system=row.rsplit('-', 1)[0]) for case, mark in zip(cases, marks, strict=True)]))  # noqa: E731
    write("none-1000", [False, False, False, True])
    write("taste_free-1000", [True, False, False, True])
    write("flat_free-1000", [True, True, None, False])
    (tmp_path / "judged" / "flat_free-1000" / "found.served.jsonl").write_text(_lines([{"id": case["id"], "found_by_judge": mark} for case, mark in zip(cases, [True, False, None, False], strict=True)]))
    rows = {row["row"]: row for row in swe.coding_table(tmp_path)}
    flat = rows["flat_free-1000"]
    assert (flat["obeyed"], flat["read"], flat["unread"]) == (2, 3, 1)
    assert (flat["held"], flat["held_unclear"]) == (1, 1)
    assert flat["against_none"] == {"better": 2, "worse": 1, "p": 1.0}
    assert flat["against_taste"] == {"better": 1, "worse": 1, "p": 1.0}
    assert rows["taste_free-1000"]["against_taste"] is None
    assert rows["none-1000"]["against_none"] is None and rows["none-1000"]["held"] is None
    text = swe.render_coding(swe.coding_table(tmp_path), swe.coding_baselines(tmp_path))
    assert "flat_free-1000" in text and "compaction" in text and "pi_mono" in text


def test_the_job_makes_every_row_the_table_needs() -> None:
    from studies.memory_bench import swe

    job = (Path(__file__).parent.parent / "studies" / "memory_bench" / "jobs" / "swe.sbatch").read_text()
    labels = next(line for line in job.splitlines() if line.startswith("for LABEL in ")).removeprefix("for LABEL in ").removesuffix("; do").split()
    assert labels == [*swe.FREE_ROWS, *swe.SEARCH_ROWS]
    assert "row none 1000" in job and swe.NONE_ROW == "none-1000" and swe.CAPS == (1000, 8000) and "for CAP in 1000 8000;" in job


def test_mem0s_adapter_differs_from_the_commit_of_its_one_ingest_only_in_its_docstrings() -> None:
    import ast
    import subprocess

    from studies.memory_bench.swe import MEM0_INGEST_COMMIT

    def code(text: str) -> str:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and ast.get_docstring(node) is not None:
                node.body = node.body[1:] or [ast.Pass()]
        return ast.dump(tree)

    path = "studies/memory_bench/adapters/mem0_arm.py"
    root = Path(__file__).parent.parent
    then = subprocess.run(["git", "show", f"{MEM0_INGEST_COMMIT}:{path}"], cwd=root, capture_output=True, text=True)
    if then.returncode:
        pytest.skip("that commit is not in this clone")
    assert code(then.stdout) == code((root / path).read_text())


def test_every_stage_runs_from_the_command_line_rather_than_only_its_functions(tmp_path: Path) -> None:
    from studies.memory_bench import swe

    cases = [_case(n) for n in range(2)]
    _whole_run(tmp_path, cases)
    out = tmp_path / "report.md"
    assert swe.main(["report", "--root", str(tmp_path), "--commit", "c1", "--digest", "d1", "--out", str(out)]) == 0
    assert "flat_free-1000" in out.read_text() and json.loads(out.with_suffix(".json").read_text())
    assert swe.main(["report", "--root", str(tmp_path), "--commit", "other", "--digest", "d1", "--out", str(tmp_path / "no.md")]) == 1
    assert not (tmp_path / "no.md").exists()
    (tmp_path / "rank").mkdir()
    assert swe.main(["none", "--cases", str(tmp_path / "cases" / "cases.jsonl"), "--out", str(tmp_path / "rank" / "none.jsonl")]) == 0
    blocks = tmp_path / "blocks" / "none-1000.jsonl"
    old = tmp_path / "answers" / "none-1000.jsonl"
    assert swe.main(["carry", "--blocks", str(blocks), "--answers", str(old), "--out", str(tmp_path / "rank" / "answers.jsonl")]) == 0
    assert len((tmp_path / "rank" / "answers.jsonl").read_text().splitlines()) == 2
