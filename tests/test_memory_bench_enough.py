"""The same retrieval judge for every system, in the words Mem0's makers publish, read from their file and never copied."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from studies.memory_bench.core import Item, reader_prompt
from studies.memory_bench.enough import delivered_text, judge_all, load_prompts, parse_judgment

_FILE = '''
import os
RETRIEVAL_JUDGE_SYSTEM = (
    "You are strict. "
    "Return JSON only."
)
RETRIEVAL_JUDGE_PROMPT = """Is it enough?
Question: {question}
Question Date: {question_date}
Expected Answer: {answer}
{profile_section}
Retrieved Memories ({num_memories} total):
{memories_text}
{{"judgment": "PASS or FAIL"}}"""
os.system("this file is only ever read, never run")
'''


def test_the_prompts_are_read_from_the_file_without_running_it(tmp_path: Path) -> None:
    path = tmp_path / "run.py"
    path.write_text(_FILE)
    system, template = load_prompts(path, sha256=None)
    assert system == "You are strict. Return JSON only."
    assert template.startswith("Is it enough?") and "{memories_text}" in template


def test_only_the_file_at_the_commit_named_is_read(tmp_path: Path) -> None:
    path = tmp_path / "run.py"
    path.write_text(_FILE)
    with pytest.raises(SystemExit, match="is not the file at commit"):
        load_prompts(path)
    assert load_prompts(path, sha256=hashlib.sha256(_FILE.encode()).hexdigest())[0] == "You are strict. Return JSON only."


def test_what_was_delivered_is_cut_from_the_readers_prompt() -> None:
    item = Item("s", "2023/05/01 (Mon) 09:00", ({"role": "memory", "content": "User parked on level 3."},), (0,))
    text = delivered_text(reader_prompt([item], "2023/06/01 (Thu) 10:00", "Which level?"))
    assert "User parked on level 3." in text and "Which level?" not in text and "Current Date" not in text
    assert delivered_text("Which level?Answer step by step.") == ""


def test_a_judgment_is_read_from_its_json_or_from_broken_json_or_not_at_all() -> None:
    assert parse_judgment('{"core_intent": "x", "judgment": "PASS", "reason": "y"}') is True
    assert parse_judgment('Sure.\n{"judgment": "FAIL"}') is False
    assert parse_judgment('{"judgment": "PASS", "reason": "cut off') is True
    assert parse_judgment("no idea") is None


def test_each_record_is_judged_once_with_the_question_the_answer_and_the_text(tmp_path: Path) -> None:
    records = [{"question_id": "q1", "qid": "n1", "prompt_sha256": "h", "delivered": [{"session": "s"}],
                "prompt": "History Chats:\n\nabout parking\n\nCurrent Date: d\nQuestion: q\nAnswer (step by step):"}]
    references = {"q1": {"question": "Which level?", "answer": "3", "question_date": "2023/06/01"}}
    asked = []

    def ask(user: str, system: str) -> SimpleNamespace:
        asked.append((user, system))
        return SimpleNamespace(text='{"judgment": "PASS"}', stop_reason="stop")

    out = tmp_path / "verdicts.jsonl"
    prompts = ("sys", "{question}|{question_date}|{answer}|{profile_section}|{num_memories}|{memories_text}")
    assert judge_all(records, references, prompts, ask, out, workers=1) == {"questions": 1, "judged": 1, "errors": 0, "unread": 0, "enough": 1}
    assert asked == [("Which level?|2023/06/01|3||1|about parking", "sys")]
    assert judge_all(records, references, prompts, ask, out, workers=1)["judged"] == 1 and len(asked) == 1
    assert json.loads(out.read_text())["enough"] is True


def test_a_failed_call_leaves_no_row_and_is_asked_again(tmp_path: Path) -> None:
    records = [{"question_id": "q1", "qid": "n1", "prompt_sha256": "h", "delivered": [], "prompt": "x"}]
    references = {"q1": {"question": "Which?", "answer": "3"}}

    def fails(user: str, system: str) -> None:
        raise RuntimeError("down")

    out = tmp_path / "verdicts.jsonl"
    prompts = ("sys", "{question}{question_date}{answer}{profile_section}{num_memories}{memories_text}")
    assert judge_all(records, references, prompts, fails, out, workers=1)["errors"] == 1
    assert not out.exists() or not out.read_text()
