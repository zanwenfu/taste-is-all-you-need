"""The stages after retrieval: headers written once, the reader, the judge, and the command line.

Each stage writes as it goes and picks up where it stopped, and a failed call is written down and asked
again, never skipped: one benchmark's script silently leaves failed answers out of its denominator.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from studies.context_rot.lme_judge import judge_file
from studies.memory_bench import headers, reader
from studies.memory_bench.lme import load
from studies.memory_bench.run import main as run_main
from studies.memory_bench.run import make_system
from studies.memory_bench.systems import FlatBm25, TasteGit
from taste.memstore import CommitHeader


@pytest.fixture(autouse=True)
def _pinned_code(monkeypatch) -> None:
    monkeypatch.setenv("TASTE_COMMIT", "abc123")


def _instance(question_id: str, question_type: str = "single-session-user") -> dict:
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": "Which level did I park the rental on?",
        "answer": "level 3",
        "question_date": "2023/06/01 (Thu) 10:00",
        "haystack_session_ids": ["sess_b", "answer_abc"],
        "haystack_dates": ["2023/05/03 (Wed) 09:00", "2023/05/02 (Tue) 09:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "Any good pasta recipes?"}, {"role": "assistant", "content": "Try carbonara."}],
            [{"role": "user", "content": f"I parked the rental on level 3, {question_id}.", "has_answer": True},
             {"role": "assistant", "content": "Noted, level 3."}],
        ],
        "answer_session_ids": ["answer_abc"],
    }


def _completion(text: str, finish: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=5),
    )


class _Client:
    """Stands in for the served model's OpenAI interface and records what it was asked."""

    def __init__(self, reply=lambda messages: _completion("level 3")) -> None:
        self.calls: list[dict] = []
        self.reply = reply
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply(kwargs["messages"])


# ------------------------------------------------------------------ the header stage


def test_a_header_is_written_once_for_the_same_words_and_the_calls_are_counted(tmp_path: Path) -> None:
    views, _ = load([_instance("a"), _instance("b")])
    client = _Client(lambda m: _completion(json.dumps({"what": "A chat about pasta.", "keywords": ["pasta", "recipe"]})))
    path = tmp_path / "headers.jsonl"

    written = headers.write_all(views, headers.served_ask(client, "aux", max_tokens=300), path, workers=2)

    # Four sessions in two questions, three different contents: the pasta chat is the same in both.
    assert written["sessions"] == 4 and written["calls"] == 3 and written["failed"] == 0
    assert len(client.calls) == 3
    assert headers.write_all(views, headers.served_ask(client, "aux", max_tokens=300), path, workers=2)["calls"] == 0


def test_the_header_prompt_sees_only_answer_stripped_text(tmp_path: Path) -> None:
    views, _ = load([_instance("a")])
    client = _Client(lambda m: _completion('{"what": "x", "keywords": []}'))
    headers.write_all(views, headers.served_ask(client, "aux", max_tokens=300), tmp_path / "h.jsonl", workers=1)
    sent = json.dumps([c["messages"] for c in client.calls])
    for marker in ("has_answer", "answer_abc", "sess_b", "single-session"):
        assert marker not in sent, marker


def test_a_reply_that_is_not_the_right_json_is_counted_and_gives_a_header_with_nothing_said(tmp_path: Path) -> None:
    views, _ = load([_instance("a")])
    client = _Client(lambda m: _completion("Sure! Here you go."))
    path = tmp_path / "h.jsonl"
    written = headers.write_all(views, headers.served_ask(client, "aux", max_tokens=300), path, workers=1)
    assert written["failed"] == 2
    writer = headers.stored_writer(path)
    assert writer(views[0]["sessions"][0]) == CommitHeader()
    assert writer.missing == 0


def test_a_stored_header_reaches_the_commit_and_a_missing_one_is_counted(tmp_path: Path) -> None:
    views, _ = load([_instance("a")])
    client = _Client(lambda m: _completion('{"what": "A chat.", "keywords": ["vehicle", 3]}'))
    path = tmp_path / "h.jsonl"
    headers.write_all(views, headers.served_ask(client, "aux", max_tokens=300), path, workers=1)
    writer = headers.stored_writer(path)
    header = writer(views[0]["sessions"][0])
    assert header == CommitHeader(what="A chat.", keywords=("vehicle",))
    assert writer({"sid": "other", "date": "d", "turns": [{"role": "user", "content": "never seen"}]}) == CommitHeader()
    assert writer.missing == 1


# ------------------------------------------------------------------ the reader stage


def _records(tmp_path: Path, *question_ids: str) -> Path:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "".join(
            json.dumps({"qid": f"q{q}", "question_id": q, "system": "FlatBm25", "prompt": f"prompt for {q}",
                        "prompt_sha256": "x"}) + "\n"
            for q in question_ids
        )
    )
    return path


def test_the_reader_is_asked_as_the_benchmark_asks_and_its_answers_are_kept(tmp_path: Path) -> None:
    client = _Client()
    out = tmp_path / "answers.jsonl"
    summary = reader.read_all(_records(tmp_path, "a", "b"), reader.served_ask(client, "m", max_tokens=800), out, workers=2)

    assert summary == {"questions": 2, "asked": 2, "failed": 0, "empty": 0, "cut_off": 0}
    call = client.calls[0]
    assert call["temperature"] == 0 and call["max_tokens"] == 800 and call["model"] == "m"
    assert call["messages"][0]["role"] == "user" and call["messages"][0]["content"].startswith("prompt for")
    rows = {row["question_id"]: row for row in map(json.loads, out.read_text().splitlines())}
    assert rows["a"]["hypothesis"] == "level 3" and rows["a"]["finish_reason"] == "stop"
    assert rows["a"]["prompt_tokens"] == 11 and rows["a"]["completion_tokens"] == 5


def test_reasoning_effort_is_sent_only_when_asked_for(tmp_path: Path) -> None:
    client = _Client()
    ask = reader.served_ask(client, "m", max_tokens=800, effort="low")
    ask("hello")
    assert client.calls[0]["reasoning_effort"] == "low"
    reader.served_ask(client, "m", max_tokens=800)("hello")
    assert "reasoning_effort" not in client.calls[1]


def test_a_failed_call_is_written_down_and_asked_again_next_time(tmp_path: Path) -> None:
    state = {"fail": True}

    def reply(messages):
        if state["fail"] and "prompt for b" in messages[0]["content"]:
            raise TimeoutError("gone")
        return _completion("fine")

    client = _Client(reply)
    out = tmp_path / "answers.jsonl"
    records = _records(tmp_path, "a", "b")
    first = reader.read_all(records, reader.served_ask(client, "m", max_tokens=8), out, workers=1)
    assert first["failed"] == 1 and first["asked"] == 2

    state["fail"] = False
    second = reader.read_all(records, reader.served_ask(client, "m", max_tokens=8), out, workers=1)
    assert second == {"questions": 2, "asked": 1, "failed": 0, "empty": 0, "cut_off": 0}
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row["hypothesis"] for row in rows if row["question_id"] == "b"] == ["fine"]


def test_empty_answers_and_answers_cut_off_are_counted(tmp_path: Path) -> None:
    client = _Client(lambda m: _completion("" if "prompt for a" in m[0]["content"] else "partial", "length"))
    summary = reader.read_all(
        _records(tmp_path, "a", "b"), reader.served_ask(client, "m", max_tokens=8), tmp_path / "out.jsonl", workers=1
    )
    assert summary["empty"] == 1 and summary["cut_off"] == 2


# ------------------------------------------------------------------ the judge, free on the served model


def test_the_served_judge_fills_the_benchmarks_own_judge_interface(tmp_path: Path) -> None:
    client = _Client(lambda m: _completion("yes"))
    hypotheses = tmp_path / "answers.jsonl"
    hypotheses.write_text(json.dumps({"question_id": "a", "hypothesis": "level 3"}) + "\n")
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps([_instance("a")]))

    out = judge_file(hypotheses, reference, reader.ServedJudge(client, "gpt-oss-120b"), short_name="served", workers=1)

    row = json.loads(Path(out).read_text())
    assert row["autoeval_label"]["label"] is True and row["autoeval_label"]["model"] == "gpt-oss-120b"
    assert "correct answer" in client.calls[0]["messages"][0]["content"].lower()
    assert "level 3" in client.calls[0]["messages"][0]["content"]


# ------------------------------------------------------------------ the command line


def test_systems_are_made_by_name(tmp_path: Path) -> None:
    views, golds = load([_instance("a")])
    gold = golds[views[0]["qid"]]
    assert isinstance(make_system("flat_bm25", gold, tmp_path), FlatBm25)
    taste = make_system("taste_git", gold, tmp_path)
    assert isinstance(taste, TasteGit) and taste.header_writer is None and taste.commit_per == "session"
    taste.close()
    with pytest.raises(ValueError, match="unknown system"):
        make_system("our memory layer", gold, tmp_path)
    with pytest.raises(ValueError, match="needs stored headers"):
        make_system("taste_git_headers", gold, tmp_path)
    with pytest.raises(ValueError, match="only for LongMemEval"):
        make_system("oracle", None, tmp_path)


def test_a_run_of_the_real_store_end_to_end_with_a_word_count_tokenizer(tmp_path: Path, capsys) -> None:
    data = tmp_path / "data.json"
    data.write_text(json.dumps([_instance("a"), _instance("b"), _instance("c")]))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": ["a", "b"], "held_out": ["c"]}))
    out = tmp_path / "run" / "records.jsonl"
    arguments = ["retrieve", "--data", str(data), "--split", str(split), "--part", "dev", "--system", "taste_git",
                 "--budget", "500", "--tokenizer", "words", "--workdir", str(tmp_path / "work"), "--out", str(out),
                 "--workers", "2"]

    assert run_main(arguments) == 0

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert sorted(row["question_id"] for row in rows) == ["a", "b"]
    assert all(row["ledger"]["search_calls"] == 1 for row in rows)
    summary = json.loads((out.parent / "records.summary.json").read_text())
    assert summary["questions"] == 2 and summary["ledger"]["names"] == {"Taste memory, study-driven": 2}
    assert summary["means"]["session_recall_any@budget"] == 1.0
    assert not any((tmp_path / "work").iterdir())

    # Run again: nothing is repeated, and nothing is lost.
    assert run_main(arguments) == 0
    assert len(out.read_text().splitlines()) == 2


def test_the_held_out_part_refuses_to_run_without_saying_so(tmp_path: Path) -> None:
    data = tmp_path / "data.json"
    data.write_text(json.dumps([_instance("c")]))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": [], "held_out": ["c"]}))
    arguments = ["retrieve", "--data", str(data), "--split", str(split), "--part", "held_out", "--system", "flat_bm25",
                 "--budget", "500", "--tokenizer", "words", "--workdir", str(tmp_path / "w"),
                 "--out", str(tmp_path / "o.jsonl"), "--workers", "1"]
    with pytest.raises(SystemExit, match="held-out"):
        run_main(arguments)
    assert run_main([*arguments, "--frozen-config", "abc123"]) == 0
