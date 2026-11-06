"""Paper 2's shared machinery: one ingest, one budget, one reader prompt, for every system.

What each test guards is a real incident from other people's comparisons: a label leaking into a store,
a budget counted in items and not tokens, a full-context arm that dropped the newest sessions, and a
number reported for a system that never ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from studies.memory_bench.core import (
    Item,
    chunk_text,
    deliver,
    item_text,
    no_memory_prompt,
    reader_prompt,
)
from studies.memory_bench.lme import Gold, load, rounds, unit_text
from studies.memory_bench.metrics import retrieval_scores
from studies.memory_bench.runner import retrieve_one, validate
from studies.memory_bench.splits import grouped_split
from studies.memory_bench.systems import (
    FlatBm25,
    FullContext,
    NoMemory,
    Oracle,
    TasteGit,
)
from studies.memory_ledger import BASELINE, STUDY_DRIVEN, check
from taste.memstore import CommitHeader


@pytest.fixture(autouse=True)
def _pinned_code(monkeypatch) -> None:
    """A job records its commit this way; here it keeps the tests free of the working tree's state."""
    monkeypatch.setenv("TASTE_COMMIT", "abc123")


def _count(text: str) -> int:
    return len(text.split())


def _instance(question_id: str = "q1", question_type: str = "single-session-user") -> dict:
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": "Which level did I park the rental on?",
        "answer": "level 3",
        "question_date": "2023/06/01 (Thu) 10:00",
        "haystack_session_ids": ["sess_b", "answer_abc", "noans_c"],
        "haystack_dates": ["2023/05/03 (Wed) 09:00", "2023/05/02 (Tue) 09:00", "2023/05/01 (Mon) 09:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "Any good pasta recipes?"},
             {"role": "assistant", "content": "Try carbonara."},
             {"role": "user", "content": "Thanks a lot."},
             {"role": "assistant", "content": "Enjoy it."}],
            [{"role": "user", "content": "I parked the rental on level 3.", "has_answer": True},
             {"role": "assistant", "content": "Noted, level 3."},
             {"role": "user", "content": "The garage closes at nine."},
             {"role": "assistant", "content": "Good to know."}],
            [{"role": "assistant", "content": "Hello there."},
             {"role": "user", "content": "My dog is called Biscuit."},
             {"role": "assistant", "content": "Lovely name."}],
        ],
        "answer_session_ids": ["answer_abc"],
    }


@pytest.fixture
def question() -> tuple[dict, Gold]:
    views, golds = load([_instance()])
    return views[0], golds[views[0]["qid"]]


# ------------------------------------------------------------------ what a memory may see


def test_no_marker_of_the_answer_reaches_what_a_memory_sees(question) -> None:
    view, gold = question
    text = json.dumps(view)
    for marker in ("has_answer", "answer_", "noans_", "single-session", "question_type"):
        assert marker not in text, marker
    assert "answer" not in view and "answer_session_ids" not in view
    assert gold.question_id == "q1" and gold.answer == "level 3"


def test_sessions_arrive_oldest_first(question) -> None:
    view, _ = question
    assert [s["date"] for s in view["sessions"]] == [
        "2023/05/01 (Mon) 09:00", "2023/05/02 (Tue) 09:00", "2023/05/03 (Wed) 09:00",
    ]


def test_a_round_is_a_user_turn_and_the_replies_after_it(question) -> None:
    view, _ = question
    first, parked, _ = view["sessions"]
    assert [[t["role"] for t in r] for r in rounds(first)] == [["assistant"], ["user", "assistant"]]
    assert [r[0]["content"] for r in rounds(parked)] == ["I parked the rental on level 3.", "The garage closes at nine."]


def test_gold_names_the_answer_session_and_round_by_neutral_id(question) -> None:
    view, gold = question
    parked = view["sessions"][1]
    assert gold.sessions == frozenset({parked["sid"]})
    assert gold.rounds == frozenset({(parked["sid"], 0)})
    assert gold.evidence == ("I parked the rental on level 3.",)
    assert gold.abstention is False
    assert load([_instance("x_abs")])[1][neutral_qid("x_abs")].abstention is True


def neutral_qid(question_id: str) -> str:
    from studies.context_rot.lme_clean import neutral

    return neutral("q", question_id)


def test_a_units_text_carries_the_date_line_and_both_roles(question) -> None:
    view, _ = question
    parked = view["sessions"][1]
    assert unit_text(parked["date"], rounds(parked)[0]) == (
        "Session date: 2023/05/02 (Tue) 09:00\n"
        "user: I parked the rental on level 3.\n"
        "assistant: Noted, level 3.\n"
    )


# ------------------------------------------------------------------ the reader's prompt


def test_the_reader_prompt_is_the_benchmarks_own_and_sorts_by_date() -> None:
    newer = Item("s2", "2023/05/03 (Wed) 09:00", ({"role": "user", "content": "B"},))
    older = Item("s1", "2023/05/01 (Mon) 09:00", ({"role": "user", "content": "A"},))
    assert reader_prompt([newer, older], "2023/06/01 (Thu) 10:00", "Why?") == (
        "I will give you several history chats between you and a user. Please answer the question based on the "
        "relevant chat history. Answer the question step by step: first extract all the relevant information, and "
        "then reason over the information to get the answer.\n\n\nHistory Chats:\n\n"
        '\n### Session 1:\nSession Date: 2023/05/01 (Mon) 09:00\nSession Content:\n\n[{"role": "user", "content": "A"}]\n'
        '\n### Session 2:\nSession Date: 2023/05/03 (Wed) 09:00\nSession Content:\n\n[{"role": "user", "content": "B"}]\n'
        "\n\nCurrent Date: 2023/06/01 (Thu) 10:00\nQuestion: Why?\nAnswer (step by step):"
    )


def test_what_came_from_one_session_sits_under_one_heading_in_its_own_order() -> None:
    late = Item("s1", "2023/05/01 (Mon) 09:00", ({"role": "user", "content": "B"},), (4,))
    early = Item("s1", "2023/05/01 (Mon) 09:00", ({"role": "user", "content": "A"},), (1,))
    other = Item("s2", "2023/05/03 (Wed) 09:00", ({"role": "memory", "content": "C"},), (0,))
    prompt = reader_prompt([other, late, early], "2023/06/01 (Thu) 10:00", "Why?")
    assert prompt.count("### Session") == 2
    assert '[{"role": "user", "content": "A"}, {"role": "user", "content": "B"}]' in prompt
    assert prompt.index("### Session 1:") < prompt.index('"A"') < prompt.index("### Session 2:") < prompt.index('"C"')


def test_a_piece_with_no_round_number_follows_the_rest_of_its_session() -> None:
    cut = Item("s1", "d", ({"role": "user", "content": "Z"},))
    whole = Item("s1", "d", ({"role": "user", "content": "A"},), (7,))
    assert '[{"role": "user", "content": "A"}, {"role": "user", "content": "Z"}]' in reader_prompt([cut, whole], "d", "Why?")


def test_with_no_memory_the_prompt_is_the_question_alone() -> None:
    assert no_memory_prompt("Why?") == "Why?Answer step by step."


# ------------------------------------------------------------------ the budget


def _item(session: str, *words: str) -> Item:
    return Item(session, "d", tuple({"role": "user", "content": w} for w in words))


def test_the_budget_fills_in_rank_order_and_counts_tokens_not_items() -> None:
    a, b, c = _item("a", "one two"), _item("b", "three"), _item("c", "four")
    delivered = deliver([a, b, c], 3, _count)
    assert [i.session for i in delivered.items] == ["a", "b"]
    assert delivered.tokens == 3
    assert delivered.offered == 3


def test_the_budget_counts_a_memorys_own_words_so_small_pieces_pay_no_heading() -> None:
    pieces = [_item(f"s{number}", "one two") for number in range(4)]
    whole = _item("s", "one two", "one two", "one two", "one two")
    assert deliver(pieces, 8, _count).tokens == deliver([whole], 8, _count).tokens == 8
    assert len(deliver(pieces, 8, _count).items) == 4
    assert item_text(whole).split() == ["one", "two"] * 4


def test_a_row_that_must_fit_the_readers_window_can_charge_the_whole_prompt() -> None:
    item = _item("a", "one two")
    cost = _count(chunk_text(item))
    assert deliver([item], cost, _count, chunk_text).items == (item,)
    assert deliver([item], cost - 1, _count, chunk_text).items == ()


def test_an_item_that_does_not_fit_is_cut_between_turns_and_filling_stops() -> None:
    first = _item("a", "alpha", "beta beta beta beta beta beta beta beta")
    later = _item("b", "x")
    delivered = deliver([first, later], 3, _count)
    assert [i.turns for i in delivered.items] == [({"role": "user", "content": "alpha"},)]
    assert delivered.cut == 1


def test_nothing_fits_means_nothing_delivered() -> None:
    assert deliver([_item("a", "long text here")], 1, _count).items == ()


# ------------------------------------------------------------------ the systems


def test_flat_bm25_ranks_rounds_by_the_shared_keyword_code(question) -> None:
    view, _ = question
    system = FlatBm25()
    system.ingest(view["sessions"])
    first = system.retrieve("park the rental on which level")[0]
    assert first.session == view["sessions"][1]["sid"] and first.rounds == (0,)
    assert first.turns[0]["content"] == "I parked the rental on level 3."
    assert system.ledger()["name"] == BASELINE


@pytest.mark.parametrize("commit_per", ["session", "round"])
def test_taste_with_headers_off_ranks_exactly_as_the_flat_file(question, tmp_path: Path, commit_per: str) -> None:
    """The same text under the same code. A difference here is a bug, and is never reported as a finding."""
    view, _ = question
    flat, taste = FlatBm25(), TasteGit(tmp_path, commit_per=commit_per)
    flat.ingest(view["sessions"])
    taste.ingest(view["sessions"])
    try:
        for query in ("park the rental", "dog name", "pasta recipes thanks", "garage level"):
            flat_items, taste_items = flat.retrieve(query), taste.retrieve(query)
            assert flat_items and [(i.session, i.rounds) for i in flat_items] == [
                (i.session, i.rounds) for i in taste_items
            ], query
            assert [i.turns for i in flat_items] == [i.turns for i in taste_items]
    finally:
        taste.close()


@pytest.mark.parametrize("commit_per", ["session", "round"])
def test_identical_rounds_are_held_once_by_both_so_they_still_rank_alike(tmp_path: Path, commit_per: str) -> None:
    """Git stores the same bytes once. Dev holds no such round and 27 held-out questions do, so this has its own test."""
    instance = _instance()
    pasta = [{"role": "user", "content": "Any good pasta recipes?"}, {"role": "assistant", "content": "Try carbonara."}]
    instance["haystack_sessions"][0] = [*pasta, *pasta, {"role": "user", "content": "Thanks a lot."},
                                        {"role": "assistant", "content": "Enjoy it."}]
    views, _ = load([instance])
    flat, taste = FlatBm25(), TasteGit(tmp_path, commit_per=commit_per)
    flat.ingest(views[0]["sessions"])
    taste.ingest(views[0]["sessions"])
    try:
        for query in ("pasta recipes", "thanks", "park the rental"):
            flat_items, taste_items = flat.retrieve(query), taste.retrieve(query)
            assert flat_items and [(i.session, i.rounds) for i in flat_items] == [(i.session, i.rounds) for i in taste_items], query
        assert len([i for i in flat.retrieve("pasta") if i.rounds in ((0,), (1,))]) == 1
    finally:
        taste.close()


def test_the_flat_list_still_hands_over_every_record_with_no_question_asked() -> None:
    instance = _instance()
    pasta = [{"role": "user", "content": "Any good pasta recipes?"}, {"role": "assistant", "content": "Try carbonara."}]
    instance["haystack_sessions"][0] = [*pasta, *pasta]
    views, _ = load([instance])
    flat = FlatBm25()
    flat.ingest(views[0]["sessions"])
    assert len(flat.memories()) == sum(len(rounds(session)) for session in views[0]["sessions"])


def test_taste_stores_one_commit_per_session_and_still_points_at_the_round(question, tmp_path: Path) -> None:
    view, _ = question
    taste = TasteGit(tmp_path, commit_per="session")
    taste.ingest(view["sessions"])
    try:
        assert taste.commits == 3
        assert taste.retrieve("park the rental")[0].rounds == (0,)
    finally:
        taste.close()
    per_round = TasteGit(tmp_path, commit_per="round")
    per_round.ingest(view["sessions"])
    try:
        assert per_round.commits == 6
    finally:
        per_round.close()


def test_taste_really_ran_and_the_ledger_can_prove_it(question, tmp_path: Path) -> None:
    view, _ = question
    taste = TasteGit(tmp_path)
    taste.ingest(view["sessions"])
    assert taste.ledger()["search_calls"] == 0
    taste.retrieve("park the rental")
    record = taste.ledger()
    taste.close()
    assert record["name"] == STUDY_DRIVEN and record["mode"] == "single_query" and record["search_calls"] == 1
    check([record])


def test_closing_removes_the_store_from_disk(question, tmp_path: Path) -> None:
    view, _ = question
    taste = TasteGit(tmp_path)
    taste.ingest(view["sessions"])
    assert any(tmp_path.iterdir())
    taste.close()
    assert not any(tmp_path.iterdir())


def test_a_header_word_brings_back_that_sessions_rounds(question, tmp_path: Path) -> None:
    """In this mode a header is index text: a word only it holds must still reach the reader."""
    view, _ = question
    parked = view["sessions"][1]

    def writer(session: dict) -> CommitHeader:
        words = ("automobile",) if session["sid"] == parked["sid"] else ()
        return CommitHeader(what="a chat", keywords=words)

    taste = TasteGit(tmp_path, header_writer=writer)
    taste.ingest(view["sessions"])
    try:
        hits = taste.retrieve("automobile")
        assert [(i.session, i.rounds) for i in hits] == [(parked["sid"], (0,)), (parked["sid"], (1,))]
        # A round that matched by its own text is not offered twice.
        both = taste.retrieve("automobile garage")
        assert len({(i.session, i.rounds) for i in both}) == len(both)
    finally:
        taste.close()


def test_with_headers_off_a_commit_message_never_ranks(question, tmp_path: Path) -> None:
    view, _ = question
    taste = TasteGit(tmp_path)
    taste.ingest(view["sessions"])
    try:
        assert taste.retrieve(view["sessions"][0]["sid"]) == []
    finally:
        taste.close()


def test_a_session_can_be_the_unit(question, tmp_path: Path) -> None:
    view, _ = question
    flat, taste = FlatBm25(unit="session"), TasteGit(tmp_path, unit="session")
    flat.ingest(view["sessions"])
    taste.ingest(view["sessions"])
    try:
        top = flat.retrieve("park the rental")[0]
        assert top.session == view["sessions"][1]["sid"] and top.rounds == (0, 1)
        assert len(top.turns) == 4
        assert [(i.session, i.rounds) for i in flat.retrieve("garage dog")] == [
            (i.session, i.rounds) for i in taste.retrieve("garage dog")
        ]
    finally:
        taste.close()


def test_no_memory_offers_nothing(question) -> None:
    view, _ = question
    system = NoMemory()
    system.ingest(view["sessions"])
    assert system.retrieve("anything") == []


def test_full_context_keeps_the_newest_sessions_when_it_must_cut(question) -> None:
    view, _ = question
    system = FullContext()
    system.ingest(view["sessions"])
    items = system.retrieve("anything")
    assert [i.date for i in items] == sorted((s["date"] for s in view["sessions"]), reverse=True)
    newest = _count(item_text(items[0]))
    delivered = deliver(items, newest + 1, _count)
    assert [i.session for i in delivered.items] == [view["sessions"][2]["sid"]]
    assert delivered.dropped == 2


def test_oracle_offers_the_answer_sessions_only(question) -> None:
    view, gold = question
    system = Oracle(gold)
    system.ingest(view["sessions"])
    assert [i.session for i in system.retrieve("anything")] == [view["sessions"][1]["sid"]]


# ------------------------------------------------------------------ retrieval scores


def test_retrieval_scores_at_a_cut_and_at_the_budget(question) -> None:
    view, gold = question
    parked = view["sessions"][1]["sid"]
    other = view["sessions"][0]["sid"]
    ranked = [Item(other, "d", (), (0,)), Item(parked, "d", (), (1,)), Item(parked, "d", (), (0,))]
    scores = retrieval_scores(ranked, ranked[:2], gold)
    assert scores["session_recall_any@1"] == 0.0
    assert scores["session_recall_any@5"] == 1.0
    assert scores["session_recall_all@5"] == 1.0
    assert scores["session_recall_any@budget"] == 1.0
    assert scores["round_recall_any@budget"] == 0.0  # the session arrived, its evidence round did not
    assert scores["session_ndcg@10"] == pytest.approx(0.6309, abs=1e-3)


def test_round_scores_exist_only_at_the_budget_where_every_system_has_paid_the_same(question) -> None:
    """Five whole sessions hold far more rounds than five rounds do, so a round score at a rank compares unlike things."""
    view, gold = question
    parked = view["sessions"][1]["sid"]
    item = Item(parked, "d", (), (0,))
    scores = retrieval_scores([item], [item], gold)
    assert scores["round_recall_any@budget"] == 1.0
    assert not [key for key in scores if key.startswith("round_") and not key.endswith("@budget")]


def test_a_system_that_cannot_say_which_round_gets_no_round_score_rather_than_a_zero(question) -> None:
    view, gold = question
    drawer = Item(view["sessions"][1]["sid"], "d", ({"role": "memory", "content": "text"},))
    scores = retrieval_scores([drawer], [drawer], gold)
    assert scores["session_recall_any@budget"] == 1.0
    assert "round_recall_any@budget" not in scores


def test_abstention_questions_have_no_retrieval_score() -> None:
    views, golds = load([_instance("x_abs")])
    assert retrieval_scores([], [], golds[views[0]["qid"]]) == {}


def test_evidence_survival_looks_for_the_answers_own_words_in_what_was_delivered(question) -> None:
    view, gold = question
    parked = view["sessions"][1]
    kept = Item(parked["sid"], "d", tuple(rounds(parked)[0]), (0,))
    assert retrieval_scores([kept], [kept], gold)["evidence_survived"] == 1.0
    other = Item(parked["sid"], "d", tuple(rounds(parked)[1]), (1,))
    assert retrieval_scores([other], [other], gold)["evidence_survived"] == 0.0


def test_evidence_survival_is_not_asked_of_a_memory_written_in_other_words(question) -> None:
    """A whole turn can never be a substring of a memory, so this would read zero by construction."""
    view, gold = question
    summary = Item(view["sessions"][1]["sid"], "d", ({"role": "memory", "content": "User parked on level 3."},), (0,))
    assert "evidence_survived" not in retrieval_scores([summary], [summary], gold)


# ------------------------------------------------------------------ one question, start to finish


def test_one_question_produces_a_complete_record(question, tmp_path: Path) -> None:
    view, gold = question
    system = TasteGit(tmp_path)
    record = retrieve_one(view, gold, system, budget=200, count=_count)

    assert record["qid"] == view["qid"] and record["question_id"] == "q1"
    assert record["ranked"][0] == {"session": view["sessions"][1]["sid"], "rounds": [0]}
    assert "I parked the rental on level 3." in record["prompt"]
    assert record["tokens_delivered"] <= 200 and record["offered"] >= 1
    assert record["scores"]["session_recall_any@budget"] == 1.0
    assert record["prompt_sha256"] and record["ingest_seconds"] >= 0 and record["retrieve_seconds"] >= 0
    assert record["ledger"]["search_calls"] == 1
    assert not any(tmp_path.iterdir())  # the store is gone, the record stays
    validate([record], expected={view["qid"]})


def test_with_no_memory_a_record_has_the_questions_own_prompt(question) -> None:
    view, gold = question
    record = retrieve_one(view, gold, NoMemory(), budget=200, count=_count)
    assert record["prompt"] == no_memory_prompt(view["question"])
    assert record["tokens_delivered"] == 0
    validate([record], expected={view["qid"]})


def test_validation_refuses_missing_duplicate_empty_and_unledgered_records(question, tmp_path: Path) -> None:
    view, gold = question
    good = retrieve_one(view, gold, FlatBm25(), budget=200, count=_count)
    with pytest.raises(ValueError, match="missing"):
        validate([], expected={view["qid"]})
    with pytest.raises(ValueError, match="twice"):
        validate([good, good], expected={view["qid"]})
    with pytest.raises(ValueError, match="empty prompt"):
        validate([{**good, "prompt": ""}], expected={view["qid"]})
    with pytest.raises(ValueError, match="not one of the allowed names"):
        validate([{**good, "ledger": {"name": "our memory layer"}}], expected={view["qid"]})


def test_validation_refuses_an_arm_that_delivered_nothing(question) -> None:
    view, gold = question
    good = retrieve_one(view, gold, FlatBm25(), budget=200, count=_count)
    with pytest.raises(ValueError, match="delivered nothing"):
        validate([{**good, "tokens_delivered": 0}], expected={view["qid"]})


def test_validation_refuses_a_row_of_headers_that_lost_its_headers(question) -> None:
    view, gold = question
    good = retrieve_one(view, gold, FlatBm25(), budget=200, count=_count)
    with pytest.raises(ValueError, match="headers"):
        validate([{**good, "headers_missing": 3}], expected={view["qid"]})


def test_validation_refuses_records_from_two_commits(question) -> None:
    view, gold = question
    good = retrieve_one(view, gold, FlatBm25(), budget=200, count=_count)
    other = {**good, "qid": "other", "ledger": {**good["ledger"], "taste_commit": "def456"}}
    with pytest.raises(ValueError, match="commit"):
        validate([good, other], expected={view["qid"], "other"})


# ------------------------------------------------------------------ splits


def _questions() -> list[dict]:
    items = []
    for number in range(120):
        qtype = ("multi-session", "temporal-reasoning", "single-session-user")[number % 3]
        items.append({"question_id": f"id{number:03d}", "question_type": qtype})
        if number % 6 == 0:
            items.append({"question_id": f"id{number:03d}_abs", "question_type": qtype})
    return items


def test_a_split_keeps_a_question_and_its_abstention_twin_together() -> None:
    dev, held_out = grouped_split(_questions(), dev_count=20, seed=7)
    assert len(dev) == 20 and len(dev) + len(held_out) == 140
    bases = {q.removesuffix("_abs") for q in dev}
    assert not bases & {q.removesuffix("_abs") for q in held_out}


def test_abstention_questions_reach_dev_in_their_share() -> None:
    """Otherwise no system can be tuned on saying that it does not know."""
    dev, _ = grouped_split(_questions(), dev_count=28, seed=7)
    assert 2 <= sum(q.endswith("_abs") for q in dev) <= 6  # 20 of 140 overall, a fifth of 28 is about 4


def test_a_split_is_repeatable_and_covers_every_type() -> None:
    assert grouped_split(_questions(), 20, 7) == grouped_split(_questions(), 20, 7)
    assert grouped_split(_questions(), 20, 7) != grouped_split(_questions(), 20, 8)
    dev, _ = grouped_split(_questions(), 20, 7)
    kinds = {next(q["question_type"] for q in _questions() if q["question_id"] == d) for d in dev}
    assert kinds == {"multi-session", "temporal-reasoning", "single-session-user"}
