"""The stages after retrieval: headers written once, the reader, the judge, and the command line.

Each stage writes as it goes and picks up where it stopped, and a failed call is written down and asked
again, never skipped: one benchmark's script silently leaves failed answers out of its denominator.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from studies.context_rot.lme_judge import judge_file
from studies.memory_bench import headers, reader
from studies.memory_bench import run as run_module
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


def test_an_answer_to_a_prompt_that_has_since_changed_stops_the_stage(tmp_path: Path) -> None:
    records = _records(tmp_path, "a")
    out = tmp_path / "answers.jsonl"
    reader.read_all(records, reader.served_ask(_Client(), "m", max_tokens=8), out, workers=1)
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    records.write_text("".join(json.dumps({**row, "prompt_sha256": "changed"}) + "\n" for row in rows))
    with pytest.raises(ValueError, match="stale"):
        reader.read_all(records, reader.served_ask(_Client(), "m", max_tokens=8), out, workers=1)


def test_every_answer_says_which_model_and_cap_wrote_it(tmp_path: Path) -> None:
    out = tmp_path / "answers.jsonl"
    reader.read_all(_records(tmp_path, "a"), reader.served_ask(_Client(), "m", max_tokens=8), out, workers=1,
                    settings={"model": "m", "max_tokens": 8, "effort": "low"})
    row = json.loads(out.read_text().splitlines()[0])
    assert (row["model"], row["max_tokens"], row["effort"]) == ("m", 8, "low")


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
    assert client.calls[0]["temperature"] == 0, "the official judge's setting: the model's default would put sampling noise in every verdict"


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


def test_the_held_out_part_refuses_to_run_without_saying_so(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "data.json"
    data.write_text(json.dumps([_instance("c")]))
    monkeypatch.setitem(run_module.DATA_SHA256, "lme", hashlib.sha256(data.read_bytes()).hexdigest())
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": [], "held_out": ["c"]}))
    monkeypatch.setattr(run_module, "_now", lambda: run_module.OPENS_AT)
    arguments = ["retrieve", "--data", str(data), "--split", str(split), "--part", "held_out", "--system", "flat_bm25",
                 "--budget", "500", "--tokenizer", "words", "--workdir", str(tmp_path / "w"),
                 "--out", str(tmp_path / "o.jsonl"), "--workers", "1"]
    with pytest.raises(SystemExit, match="held-out"):
        run_main(arguments)
    with pytest.raises(SystemExit, match="held-out"):
        run_main([*arguments, "--frozen-config", "another"])
    with pytest.raises(SystemExit, match="not that commit's"):
        run_main([*arguments, "--frozen-config", "abc123"])
    with pytest.raises(SystemExit, match="not that commit's"):
        run_main([*arguments, "--frozen-config", "abc123", "--frozen-digest", "another"])
    assert run_main([*arguments, "--frozen-config", "abc123", "--frozen-digest", run_module.code_digest()]) == 0


def test_the_held_out_part_does_not_open_before_its_date_however_right_the_rest_is(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TASTE_COMMIT", "abc123")
    data = tmp_path / "data.json"
    data.write_text(json.dumps([_instance("c")]))
    monkeypatch.setitem(run_module.DATA_SHA256, "lme", hashlib.sha256(data.read_bytes()).hexdigest())
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": [], "held_out": ["c"]}))
    arguments = ["retrieve", "--data", str(data), "--split", str(split), "--part", "held_out", "--system", "flat_bm25", "--budget", "500", "--tokenizer", "words",
                 "--workdir", str(tmp_path / "w"), "--out", str(tmp_path / "o.jsonl"), "--workers", "1", "--frozen-config", "abc123", "--frozen-digest", run_module.code_digest()]
    monkeypatch.setattr(run_module, "_now", lambda: run_module.OPENS_AT - timedelta(seconds=1))
    with pytest.raises(SystemExit, match="opens at"):
        run_main(arguments)
    assert not (tmp_path / "o.jsonl").exists()
    monkeypatch.setattr(run_module, "_now", lambda: run_module.OPENS_AT)
    assert run_main(arguments) == 0
    assert run_module.OPENS_AT.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M") == "2026-11-10 12:00"


def test_a_copy_of_the_code_is_held_to_its_commit_by_what_git_holds(tmp_path: Path) -> None:
    from studies.memory_ledger import code_digest, digest_of_commit, ingest_digest

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *args], capture_output=True, text=True, check=True).stdout.strip()

    files = {"taste/a.py": "a", "studies/x/b.py": "b", "studies/x/jobs/run.sbatch": "job", "studies/memory_bench/adapters/m.py": "m", "studies/memory_bench/core.py": "c",
             "data/splits/s.json": "{}", "docs/note.md": "not code"}
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "x")
    commit = git("rev-parse", "HEAD")
    assert code_digest(tmp_path) == digest_of_commit(commit, tmp_path)
    (tmp_path / "docs" / "note.md").write_text("changed")
    (tmp_path / "taste" / "__pycache__").mkdir()
    (tmp_path / "taste" / "__pycache__" / "a.pyc").write_text("compiled")
    assert code_digest(tmp_path) == digest_of_commit(commit, tmp_path), "words and compiled files are not code"
    before = ingest_digest(tmp_path)
    for name in ("studies/x/jobs/run.sbatch", "data/splits/s.json", "taste/a.py"):
        original = (tmp_path / name).read_text()
        (tmp_path / name).write_text(original + "!")
        assert code_digest(tmp_path) != digest_of_commit(commit, tmp_path), name
        assert ingest_digest(tmp_path) == before, name
        (tmp_path / name).write_text(original)
    (tmp_path / "studies/memory_bench/core.py").write_text("c!")
    assert ingest_digest(tmp_path) == before, "the budget's code is not the ingest's"
    (tmp_path / "studies/memory_bench/adapters/m.py").write_text("m!")
    assert ingest_digest(tmp_path) != before


def test_the_held_out_part_refuses_any_file_but_the_benchmarks_own(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TASTE_COMMIT", "abc123")
    monkeypatch.setattr(run_module, "_now", lambda: run_module.OPENS_AT)
    data = tmp_path / "data.json"
    data.write_text(json.dumps([_instance("c")]))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": [], "held_out": ["c"]}))
    with pytest.raises(SystemExit, match="not the benchmark's own file"):
        run_main(["retrieve", "--data", str(data), "--split", str(split), "--part", "held_out", "--system", "flat_bm25",
                  "--budget", "500", "--tokenizer", "words", "--workdir", str(tmp_path / "w"), "--out", str(tmp_path / "o.jsonl"),
                  "--workers", "1", "--frozen-config", "abc123", "--frozen-digest", run_module.code_digest()])


def test_a_run_carries_on_only_with_the_settings_it_started_with(tmp_path: Path) -> None:
    data = tmp_path / "data.json"
    data.write_text(json.dumps([_instance("c")]))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": ["c"], "held_out": []}))
    arguments = ["retrieve", "--data", str(data), "--split", str(split), "--part", "dev", "--system", "flat_bm25",
                 "--tokenizer", "words", "--workdir", str(tmp_path / "w"), "--out", str(tmp_path / "o.jsonl"), "--workers", "1"]
    assert run_main([*arguments, "--budget", "500"]) == 0
    assert run_main([*arguments, "--budget", "500"]) == 0
    with pytest.raises(SystemExit, match="budget"):
        run_main([*arguments, "--budget", "400"])


def test_a_row_at_the_readers_window_charges_what_the_prompt_will_hold(tmp_path: Path) -> None:
    data = tmp_path / "data.json"
    data.write_text(json.dumps([_instance("c")]))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": ["c"], "held_out": []}))
    tokens = {}
    for charge in ("text", "prompt"):
        out = tmp_path / f"{charge}.jsonl"
        assert run_main(["retrieve", "--data", str(data), "--split", str(split), "--part", "dev", "--system", "full", "--budget", "500",
                         "--charge", charge, "--tokenizer", "words", "--workdir", str(tmp_path / "w"), "--out", str(out), "--workers", "1"]) == 0
        row = json.loads(out.read_text())
        assert row["charge"] == charge
        tokens[charge] = row["tokens_delivered"]
    assert tokens["prompt"] > tokens["text"]


def test_the_budget_counts_with_the_readers_own_tokenizer_file(tmp_path: Path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    from studies.memory_bench.run import _counter

    word_level = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]"))
    word_level.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    word_level.save(str(tmp_path / "tokenizer.json"))
    assert _counter(str(tmp_path))("a b c") == 3
    assert _counter(str(tmp_path / "tokenizer.json"))("a b") == 2


def test_the_headers_stage_also_reads_locomo_conversations(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    data = tmp_path / "locomo.json"
    data.write_text(json.dumps([{"sample_id": "conv-1", "qa": [], "conversation": {
        "speaker_a": "A", "speaker_b": "B", "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "My dog is Biscuit."}]}}]))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": ["conv-1"], "held_out": []}))
    asked = []

    def create(**kwargs):
        asked.append(kwargs["messages"][0]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"what": "A talks about a dog.", "keywords": ["dog"]}'),
                                                        finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(run_module, "_client", lambda url: client)
    out = tmp_path / "headers.jsonl"
    assert run_module.main(["headers", "--benchmark", "locomo", "--data", str(data), "--split", str(split), "--part", "dev",
                            "--url", "u", "--model", "m", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["what"] == "A talks about a dog."
    assert "A: My dog is Biscuit." in asked[0] and "user and an assistant" not in asked[0]
