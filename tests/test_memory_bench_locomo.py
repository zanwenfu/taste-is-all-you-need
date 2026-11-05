"""LoCoMo as the comparison uses it: one store per conversation, its own prompt, its own judge-free score.

Scoring is ported from the benchmark's task_eval/evaluation.py and checked here against answers worked out
by hand from that code, so that a score can be compared with anyone else's who used the same script.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from studies.memory_bench.core import Item
from studies.memory_bench.locomo import (
    adversarial_score,
    conversation_split,
    f1_multi,
    f1_score,
    load,
    locomo_scores,
    normalize_answer,
    qa_score,
    reader_prompt,
    reference,
)
from studies.memory_bench.run import main as run_main
from studies.memory_bench.runner import retrieve_conversation
from studies.memory_bench.systems import FlatBm25, FullContext, TasteGit
from taste.memstore.keyword import stem


@pytest.fixture(autouse=True)
def _pinned_code(monkeypatch) -> None:
    monkeypatch.setenv("TASTE_COMMIT", "abc123")


def _count(text: str) -> int:
    return len(text.split())


def _sample(sample_id: str = "conv-1") -> dict:
    return {
        "sample_id": sample_id,
        "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Caroline", "dia_id": "D1:1", "text": "I went to the support group yesterday."},
                {"speaker": "Melanie", "dia_id": "D1:2", "text": "How was the pottery class?",
                 "blip_caption": "a photo of a bowl"},
            ],
            "session_2_date_time": "1:14 pm on 25 May, 2023",
            "session_2": [
                {"speaker": "Melanie", "dia_id": "D2:1", "text": "We ran a charity race last Saturday."},
                {"speaker": "Caroline", "dia_id": "D2:2", "text": "Well done, Melanie."},
            ],
        },
        "observation": {"session_1_observation": "GENERATED TEXT NEVER FED TO A MEMORY"},
        "session_summary": {"session_1_summary": "GENERATED TEXT NEVER FED TO A MEMORY"},
        "qa": [
            {"question": "When did Caroline go to the support group?", "answer": "7 May 2023",
             "evidence": ["D1:1"], "category": 2},
            {"question": "What race did Melanie run?", "answer": "charity race", "evidence": ["D2:1"], "category": 4},
            {"question": "What activities does Melanie do?", "answer": "pottery, running",
             "evidence": ["D1:2", "D2:1"], "category": 1},
            {"question": "Would Melanie enjoy art?", "answer": "Likely yes; she does pottery", "evidence": [],
             "category": 3},
            {"question": "What did Caroline paint?", "adversarial_answer": "a sunset", "evidence": [], "category": 5},
        ],
    }


@pytest.fixture
def conversation():
    views, golds = load([_sample()])
    return views[0], golds


# ------------------------------------------------------------------ what a memory may see


def test_a_memory_sees_the_dialogue_and_nothing_generated_or_asked(conversation) -> None:
    view, golds = conversation
    sessions = json.dumps(view["sessions"])
    assert "GENERATED TEXT" not in sessions and "support group" in sessions
    for marker in ("evidence", "D1:1", "category", "7 May 2023"):
        assert marker not in sessions, marker
    assert len(view["questions"]) == 5 and len(golds) == 5


def test_each_turn_is_a_round_and_an_image_caption_is_kept(conversation) -> None:
    view, _ = conversation
    first = view["sessions"][0]
    assert first["date"] == "1:56 pm on 8 May, 2023"
    assert [t["role"] for t in first["turns"]] == ["Caroline", "Melanie"]
    assert first["turns"][1]["content"] == "How was the pottery class? and shared a photo of a bowl."


def test_a_temporal_question_asks_for_a_date_and_the_query_does_not(conversation) -> None:
    view, _ = conversation
    temporal = view["questions"][0]
    assert temporal["query"] == "When did Caroline go to the support group?"
    assert temporal["asked"] == temporal["query"] + " Use DATE of CONVERSATION to answer with an approximate date."


def test_an_adversarial_question_offers_two_options_in_a_fixed_order(conversation) -> None:
    view, golds = conversation
    question = view["questions"][4]
    assert "Select the correct answer: (a)" in question["asked"]
    assert "Not mentioned in the conversation" in question["asked"] and "a sunset" in question["asked"]
    assert golds[question["qid"]].adversarial is True
    again, _ = load([_sample()])
    assert again[0]["questions"][4]["asked"] == question["asked"]


def test_the_answer_of_an_open_domain_question_is_cut_at_its_semicolon_only_when_scored(conversation) -> None:
    view, golds = conversation
    assert golds[view["questions"][3]["qid"]].answer == "Likely yes; she does pottery"
    assert qa_score(3, "likely yes", "Likely yes; she does pottery", stem) == 1.0


# ------------------------------------------------------------------ the reader's prompt


def test_the_prompt_is_the_benchmarks_own_with_retrieved_turns_grouped_in_date_order(conversation) -> None:
    view, _ = conversation
    second, first = view["sessions"][1], view["sessions"][0]
    items = [
        Item(second["sid"], second["date"], (second["turns"][0],), (0,)),
        Item(first["sid"], first["date"], (first["turns"][1],), (1,)),
        Item(first["sid"], first["date"], (first["turns"][0],), (0,)),
    ]
    assert reader_prompt(items, view, view["questions"][1]) == (
        "Below is a conversation between two people: Caroline and Melanie. The conversation takes place over "
        "multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"
        "DATE: 1:56 pm on 8 May, 2023\nCONVERSATION:\n"
        'Caroline said, "I went to the support group yesterday."\n'
        'Melanie said, "How was the pottery class? and shared a photo of a bowl."\n\n\n'
        "DATE: 1:14 pm on 25 May, 2023\nCONVERSATION:\n"
        'Melanie said, "We ran a charity race last Saturday."\n\n\n'
        "\n\n\nBased on the above context, write an answer in the form of a short phrase for the following "
        "question. Answer with exact words from the context whenever possible.\n\n"
        "Question: What race did Melanie run? Short answer:\n"
    )


def test_an_adversarial_question_uses_the_benchmarks_other_prompt(conversation) -> None:
    view, _ = conversation
    prompt = reader_prompt([], view, view["questions"][4])
    assert "Based on the above context, answer the following question." in prompt
    assert "short phrase" not in prompt


# ------------------------------------------------------------------ the judge-free score, by hand from its code


def test_normalising_drops_articles_punctuation_and_commas() -> None:
    assert normalize_answer("The Quick, Brown fox and a Dog!") == "quick brown fox dog"


def test_f1_counts_stemmed_words() -> None:
    assert f1_score("running races", "run race", stem) == 1.0
    assert f1_score("charity race last Saturday", "charity race", stem) == pytest.approx(2 * 0.5 * 1 / 1.5)
    assert f1_score("nothing alike", "charity race", stem) == 0


def test_a_multi_hop_answer_is_scored_part_by_part() -> None:
    assert f1_multi("pottery, running", "pottery, running", stem) == 1.0
    assert f1_multi("pottery", "pottery, running", stem) == 0.5


def test_each_category_goes_to_its_own_rule() -> None:
    assert qa_score(1, "pottery", "pottery, running", stem) == 0.5
    assert qa_score(2, "7 May 2023", "7 May 2023", stem) == 1.0
    assert qa_score(4, "charity race", "charity race", stem) == 1.0
    with pytest.raises(ValueError):
        qa_score(9, "x", "y", stem)


def test_an_adversarial_answer_scores_one_only_for_saying_it_is_not_there() -> None:
    key = {"a": "Not mentioned in the conversation", "b": "a sunset"}
    assert adversarial_score("(a)", key) == 1
    assert adversarial_score("a", key) == 1
    assert adversarial_score("b", key) == 0
    assert adversarial_score("It was not mentioned at all.", key) == 1
    assert adversarial_score("No information available", key) == 1
    assert adversarial_score("a sunset, I think", key) == 0


# ------------------------------------------------------------------ retrieval scores


def test_retrieval_is_the_share_of_evidence_found_and_no_evidence_counts_as_found(conversation) -> None:
    view, golds = conversation
    multi = golds[view["questions"][2]["qid"]]
    first, second = view["sessions"]
    ranked = [Item(first["sid"], first["date"], (), (1,)), Item(second["sid"], second["date"], (), (1,))]
    scores = locomo_scores(ranked, ranked[:1], multi)
    assert scores["session_recall@1"] == 0.5 and scores["session_recall@10"] == 1.0
    assert scores["turn_recall@1"] == 0.5 and scores["turn_recall@10"] == 0.5
    assert scores["session_recall@budget"] == 0.5 and scores["turn_recall@budget"] == 0.5
    open_domain = golds[view["questions"][3]["qid"]]
    assert locomo_scores([], [], open_domain)["session_recall@10"] == 1.0


def test_an_evidence_id_that_names_no_turn_counts_as_not_found() -> None:
    sample = _sample()
    sample["qa"][1]["evidence"] = ["D2:1", "D9:9"]
    views, golds = load([sample])
    gold = golds[views[0]["questions"][1]["qid"]]
    item = Item(views[0]["sessions"][1]["sid"], "d", (), (0,))
    assert locomo_scores([item], [item], gold)["turn_recall@10"] == 0.5


# ------------------------------------------------------------------ one store per conversation


def test_one_store_serves_every_question_of_a_conversation(conversation, tmp_path: Path) -> None:
    view, golds = conversation
    system = TasteGit(tmp_path)
    records = retrieve_conversation(view, golds, system, budget=400, count=_count)

    assert len(records) == 5
    assert records[-1]["ledger"]["search_calls"] == 5
    assert all(r["ingest_seconds"] == records[0]["ingest_seconds"] for r in records)
    race = records[1]
    assert race["ranked"][0] == {"session": view["sessions"][1]["sid"], "rounds": [0]}
    assert "We ran a charity race" in race["prompt"]
    assert race["scores"]["turn_recall@budget"] == 1.0
    assert not any(tmp_path.iterdir())


def test_flat_and_taste_rank_alike_here_too(conversation, tmp_path: Path) -> None:
    view, golds = conversation
    flat = retrieve_conversation(view, golds, FlatBm25(), budget=400, count=_count)
    taste = retrieve_conversation(view, golds, TasteGit(tmp_path), budget=400, count=_count)
    assert [r["ranked"] for r in flat] == [r["ranked"] for r in taste]
    assert [r["prompt"] for r in flat] == [r["prompt"] for r in taste]


def test_full_context_serves_a_conversation_without_searching(conversation) -> None:
    view, golds = conversation
    records = retrieve_conversation(view, golds, FullContext(), budget=4000, count=_count)
    assert all("We ran a charity race" in r["prompt"] and "support group yesterday" in r["prompt"] for r in records)


# ------------------------------------------------------------------ splits and the judge's reference


def test_conversations_split_whole_and_repeatably() -> None:
    ids = [f"conv-{n}" for n in range(10)]
    dev, held_out = conversation_split(ids, 2, seed=7)
    assert len(dev) == 2 and len(held_out) == 8 and not set(dev) & set(held_out)
    assert conversation_split(ids, 2, seed=7) == (dev, held_out)


def test_the_judges_reference_has_only_the_scored_questions_under_the_plain_prompt() -> None:
    entries = reference([_sample()])
    assert len(entries) == 4  # the adversarial question is scored apart, by its own rule
    assert {entry["question_type"] for entry in entries} == {"multi-session"}  # routes to the plain prompt
    assert entries[0]["question_id"] == "conv-1:0" and entries[0]["answer"] == "7 May 2023"


# ------------------------------------------------------------------ the command line


def test_the_command_line_runs_a_conversation_end_to_end(tmp_path: Path) -> None:
    data = tmp_path / "locomo10.json"
    data.write_text(json.dumps([_sample("conv-1"), _sample("conv-2")]))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": ["conv-1"], "held_out": ["conv-2"]}))
    out = tmp_path / "run" / "records.jsonl"
    arguments = ["retrieve", "--benchmark", "locomo", "--data", str(data), "--split", str(split), "--part", "dev",
                 "--system", "taste_git", "--budget", "400", "--tokenizer", "words",
                 "--workdir", str(tmp_path / "work"), "--out", str(out), "--workers", "1"]
    assert run_main(arguments) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 5 and {r["question_id"].split(":")[0] for r in rows} == {"conv-1"}
    assert run_main(arguments) == 0
    assert len(out.read_text().splitlines()) == 5
