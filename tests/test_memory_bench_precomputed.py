"""Rivals that live in environments of their own: one shared ingest file in, one rankings file out.

MemPalace and Mem0 cannot share a Python environment with the study, so each runs in its own and reads the same
memory-visible file, and its ranked lists come back as data. The study then budgets, prompts and scores them
exactly as it does every other system. These tests hold that path to the in-process one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from studies.memory_bench import lme, locomo
from studies.memory_bench.run import main as run_main
from studies.memory_bench.runner import retrieve_conversation, retrieve_one
from studies.memory_bench.systems import FlatBm25, Precomputed, rankings_line
from studies.memory_ledger import BASELINE, ClaimRefused, stand_in_record
from tests.test_memory_bench import _instance
from tests.test_memory_bench_locomo import _sample


@pytest.fixture(autouse=True)
def _pinned_code(monkeypatch) -> None:
    monkeypatch.setenv("TASTE_COMMIT", "abc123")


def _count(text: str) -> int:
    return len(text.split())


def _views_file(tmp_path: Path, benchmark: str) -> tuple[Path, Path, Path]:
    if benchmark == "lme":
        data = [_instance("a"), _instance("b")]
        split = {"dev": ["a", "b"], "held_out": []}
    else:
        data = [_sample("conv-1"), _sample("conv-2")]
        split = {"dev": ["conv-1", "conv-2"], "held_out": []}
    (tmp_path / "data.json").write_text(json.dumps(data))
    (tmp_path / "split.json").write_text(json.dumps(split))
    out = tmp_path / "views.jsonl"
    assert run_main(["views", "--benchmark", benchmark, "--data", str(tmp_path / "data.json"),
                     "--split", str(tmp_path / "split.json"), "--part", "dev", "--out", str(out)]) == 0
    return tmp_path / "data.json", tmp_path / "split.json", out


# ------------------------------------------------------------------ the shared ingest file


def test_the_views_file_holds_only_what_a_memory_may_see_for_longmemeval(tmp_path: Path) -> None:
    _, _, views = _views_file(tmp_path, "lme")
    text = views.read_text()
    units = [json.loads(line) for line in text.splitlines()]
    assert len(units) == 2
    assert set(units[0]) == {"unit", "sessions", "queries"}
    assert units[0]["queries"] == [{"qid": units[0]["unit"], "query": "Which level did I park the rental on?"}]
    for marker in ("has_answer", "answer_", "noans_", "single-session", "question_type", "level 3\""):
        assert marker not in text, marker


def test_the_views_file_holds_only_what_a_memory_may_see_for_locomo(tmp_path: Path) -> None:
    _, _, views = _views_file(tmp_path, "locomo")
    text = views.read_text()
    unit = json.loads(text.splitlines()[0])
    assert unit["unit"] == "conv-1" and len(unit["queries"]) == 5
    assert all(set(q) == {"qid", "query"} for q in unit["queries"])
    for marker in ("evidence", "category", "Select the correct answer", "a sunset", "GENERATED TEXT", "D1:1"):
        assert marker not in text, marker
    assert unit["sessions"][0]["each_turn_is_a_round"] is True


def test_the_views_file_names_its_own_hash(tmp_path: Path) -> None:
    import hashlib

    _, _, views = _views_file(tmp_path, "lme")
    recorded = json.loads((tmp_path / "views.summary.json").read_text())
    assert recorded["sha256"] == hashlib.sha256(views.read_bytes()).hexdigest()
    assert recorded["units"] == 2 and recorded["queries"] == 2


# ------------------------------------------------------------------ rankings from another environment


def _rankings_from(system_factory, views: Path, out: Path, *, name: str) -> Path:
    """What an adapter does in its own environment: one fresh store per unit, a ranked list per query."""
    with out.open("w") as written:
        for line in views.read_text().splitlines():
            unit = json.loads(line)
            system = system_factory()
            system.ingest(unit["sessions"])
            for query in unit["queries"]:
                items = system.retrieve(query["query"])
                ledger = stand_in_record(BASELINE, code=f"tests:{name}")
                written.write(rankings_line(query["qid"], name, items, ingest_seconds=1.5,
                                            retrieve_seconds=0.25, ledger=ledger) + "\n")
    return out


def test_a_precomputed_ranking_gives_the_same_record_as_the_system_run_in_process(tmp_path: Path) -> None:
    data, _, views_path = _views_file(tmp_path, "lme")
    rankings = _rankings_from(FlatBm25, views_path, tmp_path / "rankings.jsonl", name="flat")
    views, golds = lme.load(json.loads(data.read_text()))
    for view in views:
        direct = retrieve_one(view, golds[view["qid"]], FlatBm25(), budget=60, count=_count)
        stored = retrieve_one(view, golds[view["qid"]], Precomputed(rankings), budget=60, count=_count)
        for key in ("ranked", "delivered", "prompt", "prompt_sha256", "tokens_delivered", "scores"):
            assert stored[key] == direct[key], key
        assert (stored["ingest_seconds"], stored["retrieve_seconds"]) == (1.5, 0.25)
        assert stored["system"] == "flat" and stored["ledger"]["code"] == "tests:flat"


def test_the_same_holds_for_a_locomo_conversation(tmp_path: Path) -> None:
    data, _, views_path = _views_file(tmp_path, "locomo")
    rankings = _rankings_from(FlatBm25, views_path, tmp_path / "rankings.jsonl", name="flat")
    views, golds = locomo.load(json.loads(data.read_text()))
    direct = retrieve_conversation(views[0], golds, FlatBm25(), budget=60, count=_count)
    stored = retrieve_conversation(views[0], golds, Precomputed(rankings), budget=60, count=_count)
    assert [r["prompt"] for r in stored] == [r["prompt"] for r in direct]
    assert [r["ranked"] for r in stored] == [r["ranked"] for r in direct]


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_a_ranking_that_names_a_session_the_question_never_had_is_refused(tmp_path: Path) -> None:
    data, _, views_path = _views_file(tmp_path, "lme")
    rankings = _rankings_from(FlatBm25, views_path, tmp_path / "rankings.jsonl", name="flat")
    rows = [json.loads(line) for line in rankings.read_text().splitlines()]
    rows[0]["items"][0]["session"] = "s000000000000"
    bad = _write(tmp_path / "bad.jsonl", rows)
    views, golds = lme.load(json.loads(data.read_text()))
    view = next(v for v in views if v["qid"] == rows[0]["qid"])
    with pytest.raises(ValueError, match="not one of this question's sessions"):
        retrieve_one(view, golds[view["qid"]], Precomputed(bad), budget=60, count=_count)


def test_a_ranking_whose_rounds_do_not_exist_is_refused(tmp_path: Path) -> None:
    data, _, views_path = _views_file(tmp_path, "lme")
    rankings = _rankings_from(FlatBm25, views_path, tmp_path / "rankings.jsonl", name="flat")
    rows = [json.loads(line) for line in rankings.read_text().splitlines()]
    rows[0]["items"][0]["rounds"] = [99]
    bad = _write(tmp_path / "bad.jsonl", rows)
    views, golds = lme.load(json.loads(data.read_text()))
    view = next(v for v in views if v["qid"] == rows[0]["qid"])
    with pytest.raises(ValueError, match="no round 99"):
        retrieve_one(view, golds[view["qid"]], Precomputed(bad), budget=60, count=_count)


def test_a_question_with_no_ranking_is_refused_and_one_ranked_twice_too(tmp_path: Path) -> None:
    data, _, views_path = _views_file(tmp_path, "lme")
    rankings = _rankings_from(FlatBm25, views_path, tmp_path / "rankings.jsonl", name="flat")
    rows = [json.loads(line) for line in rankings.read_text().splitlines()]
    views, golds = lme.load(json.loads(data.read_text()))
    missing = _write(tmp_path / "missing.jsonl", rows[1:])
    view = next(v for v in views if v["qid"] == rows[0]["qid"])
    with pytest.raises(ValueError, match="no ranking"):
        retrieve_one(view, golds[view["qid"]], Precomputed(missing), budget=60, count=_count)
    with pytest.raises(ValueError, match="twice"):
        Precomputed(_write(tmp_path / "twice.jsonl", [rows[0], rows[0]]))


def test_a_ranking_must_carry_an_allowed_ledger_name(tmp_path: Path) -> None:
    data, _, views_path = _views_file(tmp_path, "lme")
    rankings = _rankings_from(FlatBm25, views_path, tmp_path / "rankings.jsonl", name="flat")
    rows = [json.loads(line) for line in rankings.read_text().splitlines()]
    for row in rows:
        row["ledger"]["name"] = "our memory layer"
    bad = _write(tmp_path / "bad.jsonl", rows)
    out = tmp_path / "records.jsonl"
    split = tmp_path / "split.json"
    with pytest.raises(ClaimRefused):
        run_main(["retrieve", "--data", str(data), "--split", str(split), "--part", "dev", "--system", "precomputed",
                  "--rankings", str(bad), "--budget", "60", "--tokenizer", "words", "--workdir", str(tmp_path / "w"),
                  "--out", str(out), "--workers", "1"])


def test_the_command_line_runs_a_precomputed_system(tmp_path: Path) -> None:
    data, split, views_path = _views_file(tmp_path, "lme")
    rankings = _rankings_from(FlatBm25, views_path, tmp_path / "rankings.jsonl", name="mempalace_raw")
    out = tmp_path / "records.jsonl"
    assert run_main(["retrieve", "--data", str(data), "--split", str(split), "--part", "dev", "--system", "precomputed",
                     "--rankings", str(rankings), "--budget", "60", "--tokenizer", "words",
                     "--workdir", str(tmp_path / "w"), "--out", str(out), "--workers", "1"]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {row["system"] for row in rows} == {"mempalace_raw"} and len(rows) == 2
