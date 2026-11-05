"""Paper 1's harness: a conversation kept turn by turn in Taste's real memory, and a probe that may search it.

The rule-loss benchmark gives a long history, a summary that replaced it, and a question the rule should decide.
Here the whole history is also committed into a fresh memory store, one turn per commit, and the model answering
the question is offered search and fetch. What it does with them is the measurement: whether it looks, whether
what it finds holds the rule, and whether its answer obeys it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from studies.memory_ledger import STUDY_DRIVEN, check
from studies.memory_probe.stores import FlatTurns, GitTurns
from studies.memory_probe.tool_loop import ask_with_tools, found_in, grade
from taste.memstore import CommitHeader
from taste.memstore.tools import FETCH, SEARCH, definitions

RULE = "For the rest of this session, always show me a draft and wait for my go-ahead before sending anything."


@pytest.fixture(autouse=True)
def _pinned_code(monkeypatch) -> None:
    monkeypatch.setenv("TASTE_COMMIT", "abc123")


def _history() -> list[dict]:
    return [
        {"role": "user", "content": f"Help me plan the offsite. {RULE}"},
        {"role": "assistant", "content": "Sure. What dates are you considering?"},
        {"role": "user", "content": "The second week of June, near the coast."},
        {"role": "assistant", "content": "I found three venues by the coast with rooms that week."},
        {"role": "user", "content": "Book the one with the best reviews for the kickoff dinner."},
        {"role": "assistant", "content": "The Harbor House has the best reviews; it seats forty."},
    ]


def _count(text: str) -> int:
    return len(text.split())


# ------------------------------------------------------------------ the stores


def test_every_turn_is_its_own_commit_and_a_rule_is_found_by_its_words(tmp_path: Path) -> None:
    store = GitTurns(tmp_path)
    try:
        store.ingest(_history())
        assert store.commits == 6
        results = json.loads(store.answer(SEARCH, {"description": "draft go-ahead before sending"})[0])["results"]
        assert results[0]["kind"] == "transcript"
        fetched = json.loads(store.answer(FETCH, {"pointer": results[0]["pointer"]})[0])["text"]
        assert RULE in fetched
    finally:
        store.close()
    assert not any(tmp_path.iterdir())


def test_a_header_is_written_per_turn_when_asked_for(tmp_path: Path) -> None:
    store = GitTurns(tmp_path, header=lambda number, turn: CommitHeader(
        what=f"turn {number} by {turn['role']}", keywords=("teambuilding",) if number == 0 else ()))
    try:
        store.ingest(_history())
        (result,) = json.loads(store.answer(SEARCH, {"description": "teambuilding"})[0])["results"]
        # The word is only in the header, and the hit stands for the turn that commit added.
        assert result["kind"] == "transcript" and "turn 0 by user" in result["header"]
        assert store.turn_of(result["pointer"]) == 0
        fetched = json.loads(store.answer(FETCH, {"pointer": result["pointer"]})[0])["text"]
        assert RULE in fetched and result["snippet"] in fetched
        # A turn found both by its words and by its header is listed once.
        both = json.loads(store.answer(SEARCH, {"description": "teambuilding offsite", "limit": 20})[0])["results"]
        assert len({r["pointer"] for r in both}) == len(both)
    finally:
        store.close()


def test_the_flat_list_ranks_exactly_as_the_git_store_with_headers_off(tmp_path: Path) -> None:
    """The same turns under the same keyword code: a difference here is a bug, never a finding."""
    git, flat = GitTurns(tmp_path), FlatTurns()
    try:
        git.ingest(_history())
        flat.ingest(_history())
        for query in ("draft go-ahead", "coast venues June", "reviews dinner", "Harbor House seats", "user"):
            results = json.loads(git.answer(SEARCH, {"description": query, "limit": 20})[0])["results"]
            assert flat.rank(query) == [git.turn_of(r["pointer"]) for r in results], query
    finally:
        git.close()


def test_the_ledger_says_which_code_ran_and_how_often_it_looked(tmp_path: Path) -> None:
    store = GitTurns(tmp_path)
    try:
        store.ingest(_history())
        store.answer(SEARCH, {"description": "draft"})
        record = store.ledger()
        assert record["name"] == STUDY_DRIVEN and record["mode"] == "agent_tools" and record["search_calls"] == 1
        check([record])
    finally:
        store.close()


# ------------------------------------------------------------------ the tool loop


def _call(name: str, arguments: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def _reply(content: str | None = None, calls: list | None = None, finish: str = "stop") -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3))


class _Script:
    """A stand-in model that replies from a script and records what it was sent."""

    def __init__(self, replies: list) -> None:
        self.replies, self.sent = list(replies), []

    def __call__(self, messages: list[dict], tools: list | None) -> SimpleNamespace:
        self.sent.append({"messages": json.loads(json.dumps(messages)), "tools": tools})
        return self.replies.pop(0)


def _probe() -> list[dict]:
    return [{"role": "system", "content": "You are an assistant."},
            {"role": "assistant", "content": "Summary: we are planning an offsite by the coast."},
            {"role": "user", "content": "Email the venue to confirm. A) Show me a draft first B) Send it now"}]


def test_a_model_that_searches_then_fetches_then_answers(tmp_path: Path) -> None:
    store = GitTurns(tmp_path)
    store.ingest(_history())
    try:
        pointer = json.loads(store.answer(SEARCH, {"description": "draft"})[0])["results"][0]["pointer"]
        store.store.memory_counts.clear()
        model = _Script([
            _reply(calls=[_call(SEARCH, json.dumps({"description": "draft before sending"}), "c1")], finish="tool_calls"),
            _reply(calls=[_call(FETCH, json.dumps({"pointer": pointer}), "c2")], finish="tool_calls"),
            _reply("I will show a draft first.\nA"),
        ])
        run = ask_with_tools(model, _probe(), definitions("chat"), store.answer, max_rounds=4, budget=10_000,
                             count=_count)
    finally:
        store.close()

    assert run.final == "I will show a draft first.\nA" and grade(run.final, "A") is True
    assert (run.searches, run.fetches, run.rounds, run.tool_errors) == (1, 1, 2, 0)
    assert found_in(run, RULE) is True
    second = model.sent[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "c1"
    assert second[-1] == {"role": "tool", "tool_call_id": "c1", "content": run.calls[0]["output"]}
    assert model.sent[0]["tools"] == definitions("chat")


def test_a_model_that_never_looks_is_counted_as_not_looking(tmp_path: Path) -> None:
    model = _Script([_reply("B")])
    run = ask_with_tools(model, _probe(), definitions("chat"), lambda n, a: ("{}", False), max_rounds=4,
                         budget=100, count=_count)
    assert (run.searches, run.fetches, run.rounds) == (0, 0, 0)
    assert found_in(run, RULE) is False and grade(run.final, "A") is False


def test_arguments_that_are_not_json_come_back_as_an_error_and_are_counted() -> None:
    model = _Script([_reply(calls=[_call(SEARCH, "{not json", "c1")], finish="tool_calls"), _reply("A")])
    asked = []
    run = ask_with_tools(model, _probe(), definitions("chat"), lambda n, a: asked.append(a) or ("{}", False),
                         max_rounds=4, budget=100, count=_count)
    assert run.tool_errors == 1 and run.unparsed_arguments == 1 and asked == []
    assert "error" in json.loads(run.calls[0]["output"])


def test_tool_output_past_the_memory_budget_is_refused() -> None:
    long = json.dumps({"text": "word " * 80})
    model = _Script([
        _reply(calls=[_call(FETCH, json.dumps({"pointer": "p"}), "c1")], finish="tool_calls"),
        _reply(calls=[_call(FETCH, json.dumps({"pointer": "p"}), "c2")], finish="tool_calls"),
        _reply("A"),
    ])
    run = ask_with_tools(model, _probe(), definitions("chat"), lambda n, a: (long, False), max_rounds=4,
                         budget=100, count=_count)
    assert run.calls[0]["output"] == long
    assert "budget" in json.loads(run.calls[1]["output"])["error"]
    assert run.tool_tokens <= 100 and run.budget_refusals == 1


def test_a_model_that_keeps_calling_tools_is_asked_once_more_without_them() -> None:
    looping = [_reply(calls=[_call(SEARCH, json.dumps({"description": "x"}), f"c{n}")], finish="tool_calls")
               for n in range(3)]
    model = _Script([*looping, _reply("A")])
    run = ask_with_tools(model, _probe(), definitions("chat"), lambda n, a: ('{"results": []}', False),
                         max_rounds=3, budget=1000, count=_count)
    assert run.rounds == 3 and run.round_limit_hit is True
    assert model.sent[-1]["tools"] is None and run.final == "A"


def test_grading_reads_the_last_non_empty_line_as_the_package_does() -> None:
    assert grade("Thinking...\n\nA\n", "A") is True
    assert grade("B", "A") is False
    assert grade("A or B", "A") is None
    assert grade("", "A") is None
