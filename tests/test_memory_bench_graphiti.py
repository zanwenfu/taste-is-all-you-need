"""The Graphiti arm, with Graphiti's graph, its model client and the served model stood in for."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from pydantic import BaseModel

from studies.memory_bench.adapters.common import run_adapter
from studies.memory_bench.adapters.graphiti_arm import (
    GraphitiArm,
    LocalEmbedder,
    Logged,
    NoReranker,
    Record,
    check_search,
    episode_body,
    fact_text,
    named,
    reference_time,
    search_config,
)


@pytest.fixture(autouse=True)
def _pinned_code(monkeypatch) -> None:
    monkeypatch.setenv("TASTE_COMMIT", "abc123")


def _sessions() -> list[dict]:
    """Given newest first, so the arm's own order is what is tested."""
    return [{"sid": "sB", "date": "2023/05/02 (Tue) 10:30",
             "turns": [{"role": "user", "content": "Parked on level 3."}, {"role": "assistant", "content": "Noted."},
                       {"role": "user", "content": "Garage closes at nine."}]},
            {"sid": "sA", "date": "2023/05/01 (Mon) 09:00",
             "turns": [{"role": "user", "content": "My dog is Biscuit."}, {"role": "assistant", "content": "Nice."}]}]


class _Graph:
    """One unit's graph: keeps what it is given and answers a search with ``edges``, or with one fact per episode,
    newest first, when none are set."""

    made: ClassVar[list] = []

    def __init__(self, path: Path, fail_on: tuple[int, ...] = ()) -> None:
        self.path, self.fail_on, self.calls = path, fail_on, 0
        self.added, self.queries, self.edges = [], [], None
        self.ready = self.closed = False
        _Graph.made.append(self)

    async def setup(self) -> None:
        self.ready = True

    async def add(self, name, body, description, when) -> str:
        assert self.ready and asyncio.get_running_loop()
        self.calls += 1
        if self.calls - 1 in self.fail_on:
            raise ValueError("the model's reply did not fit the schema")
        uuid = f"{self.path.parent.name}/ep{len(self.added)}"
        self.added.append({"name": name, "body": body, "description": description, "when": when, "uuid": uuid})
        return uuid

    async def search(self, query: str) -> list:
        self.queries.append(query)
        if self.edges is not None:
            return self.edges
        return [_edge(f"e{n}", f"fact {n}", [a["uuid"]], a["when"]) for n, a in reversed(list(enumerate(self.added)))]

    async def close(self) -> None:
        self.closed = True


def _edge(uuid: str, fact: str, episodes: list[str], valid_at=None, invalid_at=None) -> SimpleNamespace:
    return SimpleNamespace(uuid=uuid, fact=fact, episodes=episodes, valid_at=valid_at, invalid_at=invalid_at)


def _arm(tmp_path: Path, **kwargs) -> GraphitiArm:
    return GraphitiArm(_Graph, tmp_path, **kwargs)


# ------------------------------------------------------------------ ingest


def test_one_episode_per_round_oldest_session_first_dated_by_its_session(tmp_path: Path) -> None:
    arm = _arm(tmp_path)
    arm.ingest(_sessions())
    graph = arm.graph
    assert [a["name"] for a in graph.added] == ["sA round 0", "sB round 0", "sB round 1"]
    assert [a["when"] for a in graph.added] == [datetime(2023, 5, 1, 9, 0, tzinfo=UTC),
                                                datetime(2023, 5, 2, 10, 30, tzinfo=UTC),
                                                datetime(2023, 5, 2, 10, 30, tzinfo=UTC)]
    assert graph.added[1]["body"] == "user: Parked on level 3.\nassistant: Noted."
    assert graph.added[2]["body"] == "user: Garage closes at nine."
    assert graph.added[1]["description"] == "chat session sB"
    assert arm.episodes == {graph.added[0]["uuid"]: ("sA", 0), graph.added[1]["uuid"]: ("sB", 0),
                            graph.added[2]["uuid"]: ("sB", 1)}
    assert graph.path.name == "graph.kuzu" and graph.path.parent.parent == tmp_path
    arm.close()


def test_a_talk_between_two_people_keeps_their_names_one_turn_per_episode(tmp_path: Path) -> None:
    arm = _arm(tmp_path)
    arm.ingest([{"sid": "s1", "date": "1:56 pm on 8 May, 2023", "each_turn_is_a_round": True,
                 "turns": [{"role": "Caroline", "content": "Hi."}, {"role": "Melanie", "content": "Hello."}]}])
    assert [a["body"] for a in arm.graph.added] == ["Caroline: Hi.", "Melanie: Hello."]
    assert arm.graph.added[0]["when"] == datetime(2023, 5, 8, 13, 56, tzinfo=UTC)
    arm.close()


def test_dates_are_read_as_each_benchmark_writes_them_and_any_other_form_is_refused() -> None:
    assert reference_time("2023/05/20 (Sat) 02:21") == datetime(2023, 5, 20, 2, 21, tzinfo=UTC)
    assert reference_time("1:56 pm on 8 May, 2023") == datetime(2023, 5, 8, 13, 56, tzinfo=UTC)
    assert reference_time(" 2023/05/20 (Sat) 23:59 ").tzinfo is UTC
    with pytest.raises(ValueError):
        reference_time("May 20th, 2023")


def test_episode_bodies_are_one_line_per_turn() -> None:
    assert episode_body([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b\nc"}]) == "user: a\nassistant: b\nc"


def test_a_lost_episode_is_logged_and_skipped_until_too_many_are_lost(tmp_path: Path) -> None:
    failures = tmp_path / "failed.jsonl"
    record = Record(failures=failures)
    folder = tmp_path / "work"
    folder.mkdir()
    arm = GraphitiArm(lambda path: _Graph(path, fail_on=(1,)), folder, record=record, max_failed=0.5)
    arm.ingest(_sessions())
    assert sorted(arm.episodes.values()) == [("sA", 0), ("sB", 1)]
    assert record.episodes == 2 and record.episodes_failed == 1
    (row,) = [json.loads(line) for line in failures.read_text().splitlines()]
    assert (row["session"], row["round"]) == ("sB", 0) and "did not fit the schema" in row["error"]
    arm.close()
    strict = GraphitiArm(lambda path: _Graph(path, fail_on=(1,)), folder)  # 5%: one lost round of three is too many
    with pytest.raises(RuntimeError, match="lost 1 of this unit's 3 rounds"):
        strict.ingest(_sessions())
    strict.close()
    assert not any(folder.iterdir())


# ------------------------------------------------------------------ retrieve


def test_facts_come_back_in_graphitis_order_tagged_with_the_round_they_came_from(tmp_path: Path) -> None:
    arm = _arm(tmp_path)
    arm.ingest(_sessions())
    a0, b0, b1 = (added["uuid"] for added in arm.graph.added)
    arm.graph.edges = [_edge("e1", "User's garage closes at nine", [b1], datetime(2023, 5, 2, 10, 30, tzinfo=UTC)),
                       _edge("e2", "User has a dog named Biscuit", [a0, b0], datetime(2023, 5, 1, tzinfo=UTC)),
                       _edge("e3", "User parked on level 3", ["not-in-this-unit", b0])]
    first, second, third = arm.retrieve("When does the garage close?", date="2023/06/01 (Thu) 10:00")
    assert arm.graph.queries == ["When does the garage close?"]
    assert first == first.__class__("sB", "2023/05/02 (Tue) 10:30",
                                    ({"role": "memory", "content": "User's garage closes at nine (from 2023-05-02)"},),
                                    (1,), "graphiti:e1")
    assert (second.session, second.rounds, second.date) == ("sA", (0,), "2023/05/01 (Mon) 09:00"), "the first episode"
    assert (third.session, third.rounds, third.turns[0]["content"]) == ("sB", (0,), "User parked on level 3")
    arm.close()


def test_a_fact_from_no_episode_of_the_unit_is_refused(tmp_path: Path) -> None:
    arm = _arm(tmp_path)
    arm.ingest(_sessions())
    arm.graph.edges = [_edge("e9", "?", ["elsewhere"])]
    with pytest.raises(ValueError, match="no episode of this unit: e9"):
        arm.retrieve("x")
    arm.close()


def test_a_fact_shows_the_dates_graphiti_gives_it() -> None:
    may, june = datetime(2023, 5, 20, 2, 21, tzinfo=UTC), datetime(2023, 6, 1, tzinfo=UTC)
    assert fact_text("User lives in Paris", may, None) == "User lives in Paris (from 2023-05-20)"
    assert fact_text("User lives in Paris", may, june) == "User lives in Paris (from 2023-05-20 to 2023-06-01)"
    assert fact_text("User lives in Paris", None, june) == "User lives in Paris (until 2023-06-01)"
    assert fact_text("User lives in Paris", None, None) == "User lives in Paris"
    assert fact_text("x", "2023-05-20T02:21:00Z", None) == "x (from 2023-05-20)"


def test_each_unit_has_a_graph_folder_and_event_loop_of_its_own_both_gone_on_close(tmp_path: Path) -> None:
    _Graph.made = []
    one, two = _arm(tmp_path), _arm(tmp_path)
    one.ingest(_sessions())
    two.ingest(_sessions()[1:])
    assert one.folder != two.folder and {one.folder.parent, two.folder.parent} == {tmp_path}
    assert one.loop is not two.loop and not set(one.episodes) & set(two.episodes)
    two.graph.edges = [_edge("e1", "from the other unit", [one.graph.added[0]["uuid"]])]
    with pytest.raises(ValueError, match="no episode of this unit"):
        two.retrieve("x")
    assert [item.session for item in one.retrieve("x")] == ["sB", "sB", "sA"]
    one.close()
    two.close()
    assert all(graph.closed for graph in _Graph.made) and one.loop.is_closed() and two.loop.is_closed()
    assert not any(tmp_path.iterdir())


def test_a_unit_whose_graph_never_opened_still_leaves_nothing_behind(tmp_path: Path) -> None:
    def broken(path: Path):
        raise OSError("no room on the disk")

    arm = GraphitiArm(broken, tmp_path)
    with pytest.raises(OSError):
        arm.ingest(_sessions())
    arm.close()
    assert not any(tmp_path.iterdir()) and arm.loop.is_closed()


def test_the_shared_loop_writes_one_ranked_list_per_question_and_clears_every_graph(tmp_path: Path) -> None:
    views, out, work = tmp_path / "views.jsonl", tmp_path / "rankings.jsonl", tmp_path / "work"
    work.mkdir()
    units = [{"unit": "u0", "sessions": _sessions(), "queries": [{"qid": "q0", "query": "park"}]},
             {"unit": "u1", "sessions": _sessions()[1:], "queries": [{"qid": "q1", "query": "dog"}]}]
    views.write_text("".join(json.dumps(unit) + "\n" for unit in units))
    counts = run_adapter(views, out, lambda: _arm(work), system="graphiti", code="tests:graphiti",
                         versions={"graphiti_core": "0.31.2"})
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert counts == {"units": 2, "ran": 2, "questions": 2}
    assert [item["session"] for item in rows[0]["items"]] == ["sB", "sB", "sA"]
    assert rows[0]["items"][0]["rounds"] == [1] and rows[0]["items"][0]["source"] == "graphiti:e2"
    assert rows[1]["system"] == "graphiti" and rows[1]["versions"]["graphiti_core"] == "0.31.2"
    assert not any(work.iterdir())


# ------------------------------------------------------------------ the model, the embedder, the search recipe


class _Completions:
    def __init__(self, replies: list[tuple[str, str]]) -> None:
        self.replies, self.seen = iter(replies), []

    async def create(self, **kwargs):
        self.seen.append(kwargs)
        text, reason = next(self.replies)
        if reason == "raise":
            raise ConnectionError(text)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=reason)],
                               usage=SimpleNamespace(prompt_tokens=900, completion_tokens=40))


def _client(replies: list[tuple[str, str]]) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=_Completions(replies)))


def test_every_call_says_the_effort_and_is_capped_and_every_reply_is_logged_whole(tmp_path: Path) -> None:
    """gpt-oss-120b counts its reasoning against the output cap; Graphiti asks for 16,384 tokens on its fact step."""
    log = tmp_path / "replies.jsonl"
    record = Record(replies=log)
    client = _client([('{"edges": []}', "stop"), ('```json\n{"entities": []}\n```', "stop"), ("", "length")])
    logged = Logged(client, record, effort="low", cap=4096)
    schema = {"type": "json_schema", "json_schema": {"name": "ExtractedEdges", "schema": {}}}

    async def calls() -> None:
        await logged.chat.completions.create(model="m", messages=[], temperature=1, max_tokens=16384, response_format=schema)
        await logged.chat.completions.create(model="m", messages=[], temperature=1, max_tokens=1000,
                                             response_format={"type": "json_object"})
        await logged.chat.completions.create(model="m", messages=[], temperature=1, response_format=schema)

    asyncio.run(calls())
    seen = client.chat.completions.seen
    assert [call["max_tokens"] for call in seen] == [4096, 1000, 4096]
    assert all(call["reasoning_effort"] == "low" and call["temperature"] == 1 and call["model"] == "m" for call in seen)
    assert seen[0]["response_format"] is schema, "Graphiti's output mode is passed on unchanged"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(row["parsed"], row["finish_reason"]) for row in rows] == [(True, "stop"), (True, "stop"), (False, "length")]
    assert rows[0]["reply"] == '{"edges": []}' and rows[0]["usage"] == {"prompt": 900, "completion": 40}
    assert [(row["format"], row["asked_max_tokens"]) for row in rows] == [("json_schema", 16384), ("json_object", 1000),
                                                                           ("json_schema", None)]
    assert all(row["seconds"] >= 0 for row in rows)
    assert record.counts() == {"llm_calls": 3, "llm_unparsed": 1, "llm_truncated": 1, "llm_errors": 0,
                               "episodes": 0, "episodes_failed": 0}


def test_a_call_that_fails_is_logged_and_raised_and_no_effort_is_sent_when_none_is_set() -> None:
    record = Record()
    client = _client([("connection refused", "raise"), ('{"a": 1}', "stop")])
    logged = Logged(client, record, effort=None, cap=4096)
    with pytest.raises(ConnectionError):
        asyncio.run(logged.chat.completions.create(model="m", messages=[], max_tokens=50))
    asyncio.run(logged.chat.completions.create(model="m", messages=[], max_tokens=50))
    assert record.errors == 1 and record.calls == 2 and record.unparsed == 0
    assert "reasoning_effort" not in client.chat.completions.seen[1]


def test_each_logged_reply_names_the_step_that_asked_for_it(tmp_path: Path) -> None:
    log = tmp_path / "replies.jsonl"
    logged = Logged(_client([('{"a": 1}', "stop")] * 3), Record(replies=log), effort="low", cap=4096)

    class Base:  # Graphiti's client, as far as the name goes
        async def generate_response(self, messages, response_model=None, **kwargs):
            await asyncio.sleep(0)
            await logged.chat.completions.create(model="m", messages=messages)
            return {}

    client = named(Base)()

    async def calls() -> None:
        await asyncio.gather(client.generate_response([], prompt_name="extract_nodes.extract_message"),
                             client.generate_response([], prompt_name="dedupe_edges.resolve_edge"))
        await client.generate_response([])

    asyncio.run(calls())
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert sorted(row["prompt"] or "" for row in rows) == ["", "dedupe_edges.resolve_edge", "extract_nodes.extract_message"]


def test_the_local_embedder_embeds_the_first_text_for_create_and_all_for_create_batch() -> None:
    seen = []

    def encode(texts: list[str]) -> list[list[float]]:
        seen.append(texts)
        return [[float(len(text)), 1.0] for text in texts]

    embedder = LocalEmbedder(encode)
    assert asyncio.run(embedder.create(["where did I park"])) == [16.0, 1.0]
    assert asyncio.run(embedder.create("dog")) == [3.0, 1.0]
    assert asyncio.run(embedder.create_batch(["a", "bb"])) == [[1.0, 1.0], [2.0, 1.0]]
    assert asyncio.run(embedder.create_batch([])) == []
    assert seen == [["where did I park"], ["dog"], ["a", "bb"]]
    with pytest.raises(TypeError):
        asyncio.run(embedder.create([101, 2023]))


def test_the_search_probe_runs_on_a_scratch_graph_it_removes_and_stops_the_run_when_a_half_finds_nothing(
        tmp_path: Path) -> None:
    class Probed(_Graph):
        result: ClassVar[Exception | None] = None

        async def probe(self) -> None:
            assert self.ready, "the indexes are made first"
            if Probed.result is not None:
                raise Probed.result

    _Graph.made = []
    check_search(Probed, tmp_path)
    assert _Graph.made[0].closed and _Graph.made[0].path.parent.name.startswith("graphiti-probe-")
    Probed.result = RuntimeError("Graphiti's bm25 search on Kuzu does not find a stored fact")
    with pytest.raises(RuntimeError, match="bm25"):
        check_search(Probed, tmp_path)
    assert _Graph.made[1].closed and not any(tmp_path.iterdir())


def test_no_reranking_model_is_ever_called() -> None:
    with pytest.raises(RuntimeError, match="no reranking model"):
        asyncio.run(NoReranker().rank("q", ["p"]))


def test_the_search_asks_for_its_own_copy_of_graphitis_recipe_and_leaves_the_shared_one_alone() -> None:
    """Graphiti's ``search`` writes its limit into the recipe its ingest uses to look for duplicate facts."""

    class EdgeConfig(BaseModel):
        search_methods: list[str]
        sim_min_score: float = 0.6

    class SearchConfig(BaseModel):
        edge_config: EdgeConfig | None = None
        limit: int = 10

    recipe = SearchConfig(edge_config=EdgeConfig(search_methods=["bm25", "cosine_similarity"]))
    config = search_config(recipe, 1000, -1.0)
    assert (config.limit, config.edge_config.sim_min_score, config.edge_config.search_methods) == (
        1000, -1.0, ["bm25", "cosine_similarity"])
    assert (recipe.limit, recipe.edge_config.sim_min_score) == (10, 0.6)
