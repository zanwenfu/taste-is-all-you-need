"""The loop rival adapters share, driven here by a stand-in adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

from studies.memory_bench.adapters.common import run_adapter
from studies.memory_bench.core import Item
from studies.memory_bench.systems import Precomputed


@pytest.fixture(autouse=True)
def _pinned_code(monkeypatch) -> None:
    monkeypatch.setenv("TASTE_COMMIT", "abc123")


def _views(tmp_path: Path) -> Path:
    units = [{"unit": f"u{n}", "sessions": [{"sid": f"s{n}", "date": "2023/05/01 (Mon) 09:00",
                                            "turns": [{"role": "user", "content": f"fact {n}"}]}],
              "queries": [{"qid": f"q{n}a", "query": "fact"}, {"qid": f"q{n}b", "query": "other"}]} for n in range(3)]
    path = tmp_path / "views.jsonl"
    path.write_text("".join(json.dumps(u) + "\n" for u in units))
    return path


class _Stand:
    made: ClassVar[list] = []

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on, self.closed = fail_on, False
        _Stand.made.append(self)

    def ingest(self, sessions: list[dict]) -> None:
        self.sessions = sessions

    def retrieve(self, query: str, *, date: str | None = None) -> list[Item]:
        if query == self.fail_on:
            raise RuntimeError("stand-in failure")
        session = self.sessions[0]
        return [Item(session["sid"], session["date"], tuple(session["turns"]), (0,), "src")]

    def close(self) -> None:
        self.closed = True


def test_each_unit_gets_a_fresh_store_and_each_question_one_line(tmp_path: Path) -> None:
    _Stand.made = []
    out = tmp_path / "rankings.jsonl"
    counts = run_adapter(_views(tmp_path), out, _Stand, system="stand", code="tests:stand", versions={"stand": "1"})
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert counts == {"units": 3, "ran": 3, "questions": 6} and len(rows) == 6
    assert len(_Stand.made) == 3 and all(s.closed for s in _Stand.made)
    assert rows[0]["ledger"]["name"] == "baseline" and rows[0]["versions"] == {"stand": "1"}
    assert rows[0]["items"][0]["session"] == "s0" and rows[0]["items"][0]["rounds"] == [0]


def test_a_second_run_skips_finished_units_and_reruns_a_partial_one_whole(tmp_path: Path) -> None:
    views, out = _views(tmp_path), tmp_path / "rankings.jsonl"
    run_adapter(views, out, _Stand, system="stand", code="c", versions={})
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows if r["qid"] != "q1b"))  # u1 half written
    _Stand.made = []
    counts = run_adapter(views, out, _Stand, system="stand", code="c", versions={})
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert counts["ran"] == 1 and len(_Stand.made) == 1
    assert sorted(r["qid"] for r in rows) == sorted(f"q{n}{x}" for n in range(3) for x in "ab")


def test_a_failure_writes_nothing_for_that_unit_and_closes_the_store(tmp_path: Path) -> None:
    views, out = _views(tmp_path), tmp_path / "rankings.jsonl"
    _Stand.made = []
    with pytest.raises(RuntimeError):
        run_adapter(views, out, lambda: _Stand(fail_on="other"), system="stand", code="c", versions={})
    assert (_Stand.made[0].closed and not out.exists()) or out.read_text() == ""


def test_what_an_adapter_writes_is_read_back_by_the_study(tmp_path: Path) -> None:
    out = tmp_path / "rankings.jsonl"
    run_adapter(_views(tmp_path), out, _Stand, system="stand", code="c", versions={})
    system = Precomputed(out)
    system.ingest([{"sid": "s1", "date": "2023/05/01 (Mon) 09:00", "turns": [{"role": "user", "content": "fact 1"}]}])
    (item,) = system.retrieve("fact", qid="q1a")
    assert item.session == "s1" and system.ledger()["name"] == "baseline"


# ------------------------------------------------------------------ MemPalace, with its functions stood in for

from studies.memory_bench.adapters.mempalace_arms import (  # noqa: E402
    BenchArm,
    LocomoArm,
    ProductArm,
    claude_code_jsonl,
    iso_date,
    lme_entry,
)


def _sessions() -> list[dict]:
    return [{"sid": "sA", "date": "2023/05/01 (Mon) 09:00",
             "turns": [{"role": "user", "content": "My dog is Biscuit."}, {"role": "assistant", "content": "Nice."}]},
            {"sid": "sB", "date": "2023/05/02 (Tue) 10:30",
             "turns": [{"role": "user", "content": "Parked on level 3."}, {"role": "assistant", "content": "Noted."},
                       {"role": "user", "content": "Garage closes at nine."}]}]


def test_the_entry_has_what_mempalace_reads_and_nothing_else() -> None:
    entry = lme_entry(_sessions(), "Where did I park?", "2023/06/01 (Thu) 10:00")
    assert set(entry) == {"question", "question_date", "haystack_session_ids", "haystack_dates", "haystack_sessions"}
    assert entry["haystack_session_ids"] == ["sA", "sB"] and entry["haystack_sessions"][1][0]["content"] == "Parked on level 3."


class _Bench:
    def __init__(self) -> None:
        self.calls = []

    def build_palace_and_retrieve(self, entry, granularity, n_results):
        self.calls.append(("raw", entry["question"], granularity, n_results))
        return [1, 0], ["c1", "c0"], ["sA", "sB"], ["t0", "t1"]

    def build_palace_and_retrieve_hybrid_v4(self, entry, granularity, n_results, hybrid_weight):
        self.calls.append(("v4", entry["question_date"], n_results, hybrid_weight))
        return [0, 1], ["c0", "c1"], ["sA", "sB"], ["t0", "t1"]


def test_the_benchmark_arm_hands_each_ranked_session_whole_with_every_round() -> None:
    bench = _Bench()
    arm = BenchArm(bench, mode="raw")
    arm.ingest(_sessions())
    first, second = arm.retrieve("Where did I park?", date=None)
    assert (first.session, second.session) == ("sB", "sA")
    assert first.rounds == (0, 1) and len(first.turns) == 3
    assert bench.calls == [("raw", "Where did I park?", "session", 50)]
    v4 = BenchArm(bench, mode="v4")
    v4.ingest(_sessions())
    assert [i.session for i in v4.retrieve("Where?", date="2023/06/01 (Thu) 10:00")] == ["sA", "sB"]
    assert bench.calls[-1] == ("v4", "2023/06/01 (Thu) 10:00", 50, 0.30)


def test_session_files_are_in_the_shape_the_miner_reads_with_both_roles_and_the_date() -> None:
    lines = [json.loads(line) for line in claude_code_jsonl(_sessions()[1]).splitlines()]
    assert [line["type"] for line in lines] == ["user", "assistant", "user"]
    assert lines[0]["message"] == {"role": "user", "content": "Parked on level 3."}
    assert {line["timestamp"] for line in lines} == {"2023-05-02T10:30:00"} and lines[0]["sessionId"] == "sB"
    two = {"sid": "s1", "date": "1:56 pm on 8 May, 2023",
           "turns": [{"role": "Caroline", "content": "Hi."}, {"role": "Melanie", "content": "Hello."}]}
    people = [json.loads(line) for line in claude_code_jsonl(two, ("Caroline", "Melanie")).splitlines()]
    assert [p["type"] for p in people] == ["user", "assistant"] and people[1]["message"]["content"] == "Melanie: Hello."
    assert iso_date("1:56 pm on 8 May, 2023") == "2023-05-08T13:56:00"


def test_the_product_arm_mines_the_files_and_maps_each_drawer_to_its_session(tmp_path: Path) -> None:
    mined = {}

    def mine(convo_dir, palace_path, wing, agent):
        mined.update(files=sorted(p.name for p in Path(convo_dir).iterdir()), palace=palace_path)

    def search(query, palace_path, n_results):
        assert palace_path == mined["palace"] and n_results == 200
        return {"results": [{"text": "> Parked on level 3.\nNoted.", "source_path": "/x/sB.jsonl", "drawer_id": "d1"},
                            {"text": "> My dog is Biscuit.\nNice.", "source_path": "/x/sA.jsonl", "drawer_id": "d2"}]}

    arm = ProductArm(mine, search, tmp_path)
    arm.ingest(_sessions())
    items = arm.retrieve("Where did I park?")
    arm.close()
    assert mined["files"] == ["sA.jsonl", "sB.jsonl"]
    assert [(i.session, i.date) for i in items] == [("sB", "2023/05/02 (Tue) 10:30"), ("sA", "2023/05/01 (Mon) 09:00")]
    assert items[0].turns == ({"role": "memory", "content": "> Parked on level 3.\nNoted."},) and items[0].rounds == ()
    assert not any(tmp_path.iterdir())


def test_a_drawer_mempalace_folded_from_several_sessions_says_which(tmp_path: Path) -> None:
    def search(query, palace_path, n_results):
        return {"results": [{"text": "shared", "source_path": "/x/sB.jsonl", "drawer_id": "d1",
                             "also_in": [{"drawer_id": "d2", "source_path": "/x/sA.jsonl"}]}]}

    arm = ProductArm(lambda **kwargs: None, search, tmp_path)
    arm.ingest(_sessions())
    (item,) = arm.retrieve("x")
    arm.close()
    assert item.session == "sB" and item.source == "mempalace:product:d1|also:sA"


def test_the_product_arm_finds_the_two_people_of_a_talk_itself(tmp_path: Path) -> None:
    written = {}

    def mine(convo_dir, palace_path, wing, agent):
        written.update({f.stem: f.read_text() for f in Path(convo_dir).iterdir()})

    two = [{"sid": "s1", "date": "1:56 pm on 8 May, 2023", "each_turn_is_a_round": True,
            "turns": [{"role": "Caroline", "content": "Hi."}, {"role": "Melanie", "content": "Hello."}]}]
    arm = ProductArm(mine, lambda *a, **k: {"results": []}, tmp_path)
    arm.ingest(two)
    arm.close()
    lines = [json.loads(line) for line in written["s1"].splitlines()]
    assert [line["type"] for line in lines] == ["user", "assistant"]
    assert lines[1]["message"]["content"] == "Melanie: Hello."


# ------------------------------------------------------------------ MemPalace on LoCoMo, its helpers stood in for


class _Collection:
    def __init__(self) -> None:
        self.added, self.queries = {}, []

    def add(self, documents, ids, metadatas):
        self.added = {"documents": documents, "ids": ids, "metadatas": metadatas}


class _Client:
    collection = None

    def __init__(self, path):
        self.path = path

    def create_collection(self, name):
        assert name == "mempal_drawers"
        _Client.collection = _Collection()
        return _Client.collection


class _Chroma:
    PersistentClient = _Client


class _LocomoBench:
    """Rank by a fixed order; the real helpers are checked on Duke against MemPalace's committed lists."""

    def __init__(self, order: list[str], distances: dict[str, float] | None = None) -> None:
        self.order, self.distances = order, distances or {}

    def load_conversation_sessions(self, conversation, session_summaries=None):
        self.conversation = conversation
        numbers = [int(k.split("_")[1]) for k in conversation if k.endswith("_date_time")]
        return [{"session_num": n, "date": conversation[f"session_{n}_date_time"],
                 "dialogs": conversation[f"session_{n}"], "summary": ""} for n in sorted(numbers)]

    def build_corpus_from_sessions(self, sessions, granularity="dialog"):
        corpus, ids, stamps = [], [], []
        for session in sessions:
            if granularity == "session":
                corpus.append("\n".join(f'{d["speaker"]} said, "{d["text"]}"' for d in session["dialogs"]))
                ids.append(f"session_{session['session_num']}")
                stamps.append(session["date"])
            else:
                for dialog in session["dialogs"]:
                    corpus.append(f'{dialog["speaker"]} said, "{dialog["text"]}"')
                    ids.append(dialog["dia_id"])
                    stamps.append(session["date"])
        return corpus, ids, stamps

    def _query(self, collection, question, n_results, embed_model, include=None, where=None):
        assert embed_model == "default"
        collection.queries.append(n_results)
        ids = self.order[:n_results]
        documents = {m["corpus_id"]: d for m, d in zip(collection.added["metadatas"], collection.added["documents"], strict=True)}
        return {"metadatas": [[{"corpus_id": c} for c in ids]], "documents": [[documents[c] for c in ids]],
                "distances": [[self.distances.get(c, 0.5) for c in ids]]}

    def _person_names(self, text):
        return []

    def _kw(self, text):
        return [word for word in text.lower().split() if len(word) > 2]

    def _kw_overlap(self, keywords, document):
        return sum(k in document.lower() for k in keywords) / len(keywords) if keywords else 0.0

    def _quoted_phrases(self, text):
        return []

    def _quoted_boost(self, phrases, document):
        return 0.0

    def _name_boost(self, names, document):
        return 0.0


def _talk() -> list[dict]:
    return [{"sid": "s1", "date": "1:56 pm on 8 May, 2023", "each_turn_is_a_round": True,
             "turns": [{"role": "Caroline", "content": "Hi."}, {"role": "Melanie", "content": "I run races."}]},
            {"sid": "s2", "date": "2:00 pm on 9 May, 2023", "each_turn_is_a_round": True,
             "turns": [{"role": "Melanie", "content": "My dog barks."}]}]


def test_the_locomo_arm_gives_mempalace_the_conversation_in_its_own_shape(tmp_path: Path) -> None:
    bench = _LocomoBench(["session_2", "session_1"])
    arm = LocomoArm(bench, _Chroma, tmp_path, granularity="session", mode="raw")
    arm.ingest(_talk())
    assert bench.conversation["session_1"][1] == {"speaker": "Melanie", "text": "I run races.", "dia_id": "D1:2"}
    assert bench.conversation["session_2_date_time"] == "2:00 pm on 9 May, 2023"
    added = _Client.collection.added
    assert added["ids"] == ["doc_0", "doc_1"]
    assert added["metadatas"][0] == {"corpus_id": "session_1", "timestamp": "1:56 pm on 8 May, 2023", "room": "general"}
    first, second = arm.retrieve("Who barks?")
    assert (first.session, first.rounds, len(first.turns)) == ("s2", (0,), 1)
    assert (second.session, second.rounds, len(second.turns)) == ("s1", (0, 1), 2)
    arm.close()
    assert not any(tmp_path.iterdir())


def test_at_dialog_level_each_hit_is_its_turn(tmp_path: Path) -> None:
    bench = _LocomoBench(["D1:2", "D2:1", "D1:1"])
    arm = LocomoArm(bench, _Chroma, tmp_path, granularity="dialog", mode="raw")
    arm.ingest(_talk())
    items = arm.retrieve("Who runs?")
    arm.close()
    assert [(i.session, i.rounds) for i in items] == [("s1", (1,)), ("s2", (0,)), ("s1", (0,))]
    assert items[0].turns == ({"role": "Melanie", "content": "I run races."},)


def test_the_top_ten_is_asked_as_published_and_the_rest_follow_in_its_ranking(tmp_path: Path) -> None:
    """The published run asks for ten; the reader may need more, so the whole list follows those ten, unchanged."""
    order = [f"D1:{n}" for n in range(1, 21)]
    session = {"sid": "s1", "date": "1:56 pm on 8 May, 2023", "each_turn_is_a_round": True,
               "turns": [{"role": "A", "content": f"turn {n}"} for n in range(1, 21)]}
    arm = LocomoArm(_LocomoBench(order), _Chroma, tmp_path, granularity="dialog", mode="raw")
    arm.ingest([session])
    items = arm.retrieve("anything")
    arm.close()
    assert _Client.collection.queries == [10, 20]
    assert [i.rounds[0] for i in items] == list(range(20))


def test_hybrid_rescores_only_its_own_pool_with_its_own_formula(tmp_path: Path) -> None:
    """Three times the top ten by distance, re-scored by keyword overlap; the rest stay in plain order behind them."""
    order = [f"D1:{n}" for n in range(1, 41)]
    session = {"sid": "s1", "date": "1:56 pm on 8 May, 2023", "each_turn_is_a_round": True,
               "turns": [{"role": "A", "content": "zebra" if n == 30 else f"turn {n}"} for n in range(1, 41)]}
    arm = LocomoArm(_LocomoBench(order, {c: 0.5 for c in order}), _Chroma, tmp_path, granularity="dialog", mode="hybrid")
    arm.ingest([session])
    items = arm.retrieve("zebra")
    arm.close()
    assert _Client.collection.queries == [30, 40]
    ranks = [i.rounds[0] for i in items]
    assert ranks[0] == 29, "the turn holding the keyword rises to the top of the pool"
    assert ranks[30:] == list(range(30, 40)), "turns outside the pool keep their plain order"
    assert sorted(ranks) == list(range(40))


from studies.memory_bench.adapters.check_mempalace import compare_locomo  # noqa: E402


def test_the_locomo_check_compares_each_question_with_mempalaces_committed_list() -> None:
    published = [{"sample_id": "conv-1", "retrieved_ids": ["session_2", "session_1"], "recall": 1.0},
                 {"sample_id": "conv-1", "retrieved_ids": ["session_1", "session_2"], "recall": 0.0},
                 {"sample_id": "conv-2", "retrieved_ids": ["session_1"], "recall": 0.5}]
    s = lambda sample, n: neutral("s", f"{sample}/{n}")  # noqa: E731
    rankings = [{"qid": neutral("q", "conv-1:0"), "items": [{"session": s("conv-1", 2)}, {"session": s("conv-1", 1)}]},
                {"qid": neutral("q", "conv-1:1"), "items": [{"session": s("conv-1", 2)}, {"session": s("conv-1", 1)}]},
                {"qid": neutral("q", "conv-2:0"), "items": [{"session": s("conv-2", 1)}]}]
    result = compare_locomo(published, rankings)
    assert result["compared"] == 3 and result["same_top10"] == 2
    assert result["published_mean_recall"] == pytest.approx(0.5)
    assert result["differing_top10"] == [neutral("q", "conv-1:1")]


# ------------------------------------------------------------------ the shared meaning-based baseline

from studies.memory_bench.adapters.flat_dense import FlatDenseArm  # noqa: E402


def test_flat_dense_ranks_every_unit_by_similarity_with_the_same_text_as_the_keyword_list() -> None:
    seen = []

    def embed(texts: list[str]) -> list[list[float]]:
        seen.extend(texts)
        return [[1.0, 0.0] if "park" in text.lower() else [0.0, 1.0] for text in texts]

    arm = FlatDenseArm(embed)
    arm.ingest(_sessions())
    items = arm.retrieve("park")
    assert seen[0] == "Session date: 2023/05/01 (Mon) 09:00\nuser: My dog is Biscuit.\nassistant: Nice.\n"
    assert (items[0].session, items[0].rounds) == ("sB", (0,))
    assert len(items) == 3 and items[0].turns[0]["content"] == "Parked on level 3."
    arm.close()


def test_a_tie_goes_to_the_newest_as_it_does_in_taste_search() -> None:
    arm = FlatDenseArm(lambda texts: [[1.0, 0.0] for _ in texts])
    arm.ingest(_sessions())
    assert [(i.session, i.rounds) for i in arm.retrieve("x")] == [("sB", (1,)), ("sB", (0,)), ("sA", (0,))]


def test_a_search_error_is_raised_not_ranked_as_nothing(tmp_path: Path) -> None:
    arm = ProductArm(lambda **kwargs: None, lambda *a, **k: {"error": "broken", "results": []}, tmp_path)
    arm.ingest(_sessions())
    with pytest.raises(RuntimeError, match="broken"):
        arm.retrieve("x")
    arm.close()


# ------------------------------------------------------------------ Mem0, with its Memory stood in for

from studies.memory_bench.adapters.mem0_arm import Counted, Mem0Arm, mem0_config  # noqa: E402


class _Memory:
    def __init__(self) -> None:
        self.adds, self.resets, self.searches = [], 0, []

    def reset(self) -> None:
        self.resets += 1

    def add(self, messages, *, user_id, infer, metadata, **extra):
        self.adds.append({"messages": messages, "user_id": user_id, "infer": infer, "metadata": metadata, **extra})
        return {"results": []}

    def search(self, query, *, filters, top_k, threshold):
        self.searches.append({"query": query, "filters": filters, "top_k": top_k, "threshold": threshold})
        return {"results": [
            {"id": "m1", "memory": "User parked on level 3.", "metadata": {"session_id": "sB", "round": "0"}},
            {"id": "m2", "memory": "User's dog is Biscuit.", "metadata": {"session_id": "sA", "round": "0"}}]}


def test_mem0_gets_one_add_per_round_with_the_date_and_the_session_in_metadata() -> None:
    memory = _Memory()
    arm = Mem0Arm(memory)
    arm.ingest(_sessions())
    assert memory.resets == 1 and len(memory.adds) == 3  # sA one round, sB two
    first = memory.adds[1]
    assert first["messages"][0] == {"role": "system", "content": "Session date: 2023/05/02 (Tue) 10:30"}
    assert first["messages"][1:] == [{"role": "user", "content": "Parked on level 3."},
                                     {"role": "assistant", "content": "Noted."}]
    assert first["metadata"] == {"session_id": "sB", "round": "0", "session_date": "2023/05/02 (Tue) 10:30"}
    assert first["infer"] is True and memory.adds[2]["metadata"]["round"] == "1"


def test_the_session_date_goes_through_adds_prompt_only_when_asked_for() -> None:
    plain, dated = _Memory(), _Memory()
    Mem0Arm(plain).ingest(_sessions())
    Mem0Arm(dated, date_prompt=True).ingest(_sessions())
    assert all("prompt" not in add for add in plain.adds)
    assert "2023/05/02 (Tue) 10:30" in dated.adds[1]["prompt"] and "Observation Date" in dated.adds[1]["prompt"]
    assert [{k: v for k, v in add.items() if k != "prompt"} for add in dated.adds] == plain.adds


def test_mem0_asks_for_more_memories_than_any_budget_here_can_hold() -> None:
    memory = _Memory()
    arm = Mem0Arm(memory)
    arm.ingest(_sessions())
    arm.retrieve("Where did I park?")
    assert memory.searches[0]["top_k"] == 1000


def test_a_conversation_between_two_people_maps_them_to_user_and_assistant_with_names_kept() -> None:
    memory = _Memory()
    sessions = [{"sid": "s1", "date": "1:56 pm on 8 May, 2023", "each_turn_is_a_round": True,
                 "turns": [{"role": "Caroline", "content": "Hi."}, {"role": "Melanie", "content": "Hello."}]}]
    Mem0Arm(memory).ingest(sessions)
    assert [a["messages"][1] for a in memory.adds] == [{"role": "user", "content": "Caroline: Hi."},
                                                       {"role": "assistant", "content": "Melanie: Hello."}]


def test_every_memory_found_reaches_the_reader_tagged_with_its_session_and_round() -> None:
    memory = _Memory()
    arm = Mem0Arm(memory, top_k=150)
    arm.ingest(_sessions())
    first, second = arm.retrieve("Where did I park?")
    assert memory.searches == [{"query": "Where did I park?", "filters": {"user_id": "eval"}, "top_k": 150,
                                "threshold": 0.0}]
    assert (first.session, first.rounds, first.turns) == ("sB", (0,), ({"role": "memory",
                                                                       "content": "User parked on level 3."},))
    assert second.date == "2023/05/01 (Mon) 09:00"


def test_a_memory_from_no_session_of_the_unit_is_refused() -> None:
    memory = _Memory()
    memory.search = lambda *a, **k: {"results": [{"id": "x", "memory": "?", "metadata": {"session_id": "zz", "round": "0"}}]}
    arm = Mem0Arm(memory)
    arm.ingest(_sessions())
    with pytest.raises(ValueError, match="no session of this unit"):
        arm.retrieve("x")


def test_the_served_models_json_replies_are_counted() -> None:
    replies = iter(['{"memory": []}', "```json\n{\"memory\": [1]}\n```", "Sure! Here are the facts."])
    counted = Counted(type("L", (), {"generate_response": lambda self, *a, **k: next(replies), "config": 7})())
    for _ in range(3):
        counted.generate_response([], response_format={"type": "json_object"})
    assert (counted.calls, counted.unparsed) == (3, 1) and counted.config == 7


def test_every_reply_is_logged_whole_so_it_can_be_read(tmp_path: Path) -> None:
    log = tmp_path / "replies.jsonl"
    replies = iter(['{"memory": []}', "Sure!"])
    counted = Counted(type("L", (), {"generate_response": lambda self, *a, **k: next(replies)})(), log=log)
    counted.generate_response([])
    counted.generate_response([])
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(row["parsed"], row["reply"]) for row in rows] == [(True, '{"memory": []}'), (False, "Sure!")]
    assert all(row["seconds"] >= 0 for row in rows)


def test_a_setting_mem0_has_no_field_for_reaches_the_model_on_every_call() -> None:
    """Its vLLM client passes extra arguments through. gpt-oss-120b counts its reasoning against the output cap, and at
    Mem0's defaults 21 of 181 real replies ran out, so every call here says the effort the rest of the study uses."""
    seen = []

    class Llm:
        def generate_response(self, messages, response_format=None, **kwargs):
            seen.append(kwargs)
            return '{"memory": []}'

    counted = Counted(Llm(), extra={"reasoning_effort": "low"})
    counted.generate_response([], response_format={"type": "json_object"})
    assert seen == [{"reasoning_effort": "low"}]
    plain = Counted(Llm())
    plain.generate_response([])
    assert seen[-1] == {}


def test_the_output_cap_is_a_setting_of_the_run() -> None:
    config = mem0_config("http://127.0.0.1:8000/v1", "m", "e", 384, "cpu", max_tokens=4096)
    assert config["llm"]["config"]["max_tokens"] == 4096


def test_the_configuration_names_the_served_model_and_a_local_embedder_only() -> None:
    config = mem0_config("http://127.0.0.1:8000/v1", "openai/gpt-oss-120b", "sentence-transformers/all-MiniLM-L6-v2",
                         384, "cpu")
    assert config["llm"]["provider"] == "vllm" and config["embedder"]["provider"] == "huggingface"
    assert config["vector_store"]["config"]["embedding_model_dims"] == 384 == config["embedder"]["config"]["embedding_dims"]
    assert config["vector_store"]["config"]["path"] == ":memory:" and config["history_db_path"] == ":memory:"
    assert "openai" not in json.dumps(config).replace("openai/gpt-oss-120b", "")


# ------------------------------------------------------------------ the check against MemPalace's own published run

from studies.context_rot.lme_clean import neutral  # noqa: E402
from studies.memory_bench.adapters.check_mempalace import compare  # noqa: E402


def test_the_check_maps_published_ids_to_neutral_ones_and_counts_agreement() -> None:
    data = [{"question_id": "q1", "answer_session_ids": ["answer_x"]}, {"question_id": "q2", "answer_session_ids": ["b"]}]
    published = [
        {"question_id": "q1", "retrieval_results": {"ranked_items": [{"corpus_id": c} for c in ("answer_x", "a", "c")]}},
        {"question_id": "q2", "retrieval_results": {"ranked_items": [{"corpus_id": c} for c in ("a", "b")]}},
    ]
    rankings = [
        {"qid": neutral("q", "q1"), "items": [{"session": neutral("s", f"q1/{c}")} for c in ("answer_x", "a", "c")]},
        {"qid": neutral("q", "q2"), "items": [{"session": neutral("s", f"q2/{c}")} for c in ("b", "a")]},
    ]
    result = compare(data, published, rankings)
    assert result["compared"] == 2 and result["same_top10"] == 1 and result["same_top5_set"] == 2
    assert result["recall_any"] == {"published@5": 1.0, "published@10": 1.0, "adapter@5": 1.0, "adapter@10": 1.0}
    assert result["differing_top5"] == ["q2"]
    q2 = [neutral("s", f"q2/{c}") for c in ("a", "b")]
    assert result["differing_top10"] == [{"question": "q2", "first_rank": 1, "published": q2, "adapter": q2[::-1]}]


def test_shards_split_the_units_between_processes(tmp_path: Path) -> None:
    views, out = _views(tmp_path), tmp_path / "a.jsonl"
    run_adapter(views, out, _Stand, system="s", code="c", versions={}, shard=(1, 2))
    assert sorted({json.loads(line)["qid"][:2] for line in out.read_text().splitlines()}) == ["q1"]


class _WithMemories(_Stand):
    def memories(self) -> list[str]:
        return ["newest memory", "older memory"]


def test_one_ingest_can_write_both_the_search_and_what_is_handed_over_with_no_query(tmp_path: Path) -> None:
    """A system whose ingest is a model call per round costs too much to run twice for the same store."""
    _Stand.made = []
    out, free = tmp_path / "rankings.jsonl", tmp_path / "free.jsonl"
    run_adapter(_views(tmp_path), out, _WithMemories, system="stand", code="tests:stand", versions={"stand": "1"},
                also=(free, "stand_free"))
    rows = [json.loads(line) for line in free.read_text().splitlines()]
    assert len(_Stand.made) == 3, "one store per unit, not two"
    assert [row["qid"] for row in rows] == [json.loads(line)["qid"] for line in out.read_text().splitlines()]
    assert rows[0]["system"] == "stand_free"
    assert [item["turns"][0]["content"] for item in rows[0]["items"]] == ["newest memory", "older memory"]
    run_adapter(_views(tmp_path), out, _WithMemories, system="stand", code="tests:stand", versions={"stand": "1"},
                also=(free, "stand_free"))
    assert len(free.read_text().splitlines()) == 6, "a second run adds nothing"
