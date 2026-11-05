"""Search and fetch as tools: two tools, one implementation, for whichever agent runs.

The agent reads headers and snippets first and fetches only what it wants. A wrong call comes back to
the agent as an error it can read; it never raises into the agent's loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taste.memstore import CommitHeader, Store
from taste.memstore.tools import FETCH, MAX_FETCH, MAX_RESULTS, SEARCH, answer, definitions


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "repo", "s1")
    yield s
    s.close()


def _search(store: Store, **arguments) -> dict:
    text, failed = answer(store, SEARCH, arguments)
    assert failed is False, text
    return json.loads(text)


def test_a_search_returns_pointers_headers_and_snippets_and_a_fetch_brings_back_the_bytes(store: Store) -> None:
    a = store.branch("a")
    a.write("rule.md", "before\nNever deploy on Fridays.\nafter\n")
    state = a.checkpoint("step", header=CommitHeader(what="a rule from the user"))

    results = _search(store, description="deploy on Friday", limit=3)["results"]

    assert results[0]["snippet"] == "Never deploy on Fridays."
    assert results[0]["header"] == state.header
    assert results[0]["kind"] == "file"
    text, failed = answer(store, FETCH, {"pointer": results[0]["pointer"]})
    assert failed is False
    page = json.loads(text)
    assert page["text"] == "before\nNever deploy on Fridays.\nafter\n"
    assert page["eof"] is True and page["size"] == len(page["text"])


def test_a_result_says_which_version_is_current_and_where_the_current_one_is(store: Store) -> None:
    a = store.branch("a")
    a.write("rules.md", "deploy to port 8080 on every release\n")
    a.checkpoint("old")
    a.write("rules.md", "deploy to port 9090 on every release\n")
    a.checkpoint("new")

    new, old = _search(store, description="deploy port release")["results"][:2]

    assert new["current"] is True and "replaced_by" not in new
    assert old["current"] is False
    assert old["order"] < new["order"] and old["recorded"] <= new["recorded"]
    page = json.loads(answer(store, FETCH, {"pointer": old["replaced_by"]})[0])
    assert page["text"] == "deploy to port 9090 on every release\n"


def test_a_turn_or_message_has_no_mark_to_show(store: Store) -> None:
    a = store.branch("a")
    a.turn(role="user", content="the stilt rule")
    a.checkpoint("the stilt in a message")
    results = _search(store, description="stilt")["results"]
    assert {r["kind"] for r in results} == {"message", "transcript"}
    assert all("current" not in r and "replaced_by" not in r for r in results)


def test_fetch_pages_with_offset_and_limit(store: Store) -> None:
    a = store.branch("a")
    body = "the marker\n" + "x" * 30_000
    a.write("long.md", body)
    a.checkpoint("step")
    pointer = _search(store, description="marker")["results"][0]["pointer"]

    pages, offset = [], 0
    while True:
        text, failed = answer(store, FETCH, {"pointer": pointer, "offset": offset, "limit": 8_000})
        assert failed is False
        page = json.loads(text)
        pages.append(page["text"])
        offset = page["next_offset"]
        if page["eof"]:
            break
    assert "".join(pages) == body


def test_a_fetch_with_no_limit_is_one_bounded_page(store: Store) -> None:
    a = store.branch("a")
    a.write("long.md", "the marker\n" + "x" * 100_000)
    a.checkpoint("step")
    pointer = _search(store, description="marker")["results"][0]["pointer"]
    page = json.loads(answer(store, FETCH, {"pointer": pointer})[0])
    assert len(page["text"]) == MAX_FETCH and page["eof"] is False


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("memory_delete", {"pointer": "x"}),
        (SEARCH, {}),
        (SEARCH, {"description": ""}),
        (SEARCH, {"description": 3}),
        (SEARCH, {"description": "x", "limit": 0}),
        (SEARCH, {"description": "x", "limit": MAX_RESULTS + 1}),
        (SEARCH, {"description": "x", "limit": True}),
        (SEARCH, {"description": "x", "path": "y"}),
        (FETCH, {}),
        (FETCH, {"pointer": 7}),
        (FETCH, {"pointer": "not a pointer"}),
        (FETCH, {"pointer": f"{'0' * 40}:file:1:x.md"}),
        (FETCH, {"pointer": f"{'0' * 40}:file:1:x.md", "offset": -1}),
        (FETCH, {"pointer": f"{'0' * 40}:file:1:x.md", "limit": MAX_FETCH + 1}),
        (FETCH, {"pointer": f"{'0' * 40}:file:1:x.md", "content": "y"}),
    ],
)
def test_a_wrong_call_is_an_error_the_agent_can_read(store: Store, name: str, arguments: dict) -> None:
    text, failed = answer(store, name, arguments)
    assert failed is True
    assert json.loads(text)["error"]


def test_a_search_that_finds_nothing_is_not_an_error(store: Store) -> None:
    a = store.branch("a")
    a.write("x.md", "body\n")
    a.checkpoint("step")
    assert _search(store, description="zanzibar") == {"results": []}


def test_a_result_is_valid_json_text_whatever_the_bytes(store: Store) -> None:
    a = store.branch("a")
    a.write("latin.txt", "the plover café\n".encode("latin-1"))
    a.checkpoint("step")
    text, failed = answer(store, SEARCH, {"description": "plover"})
    assert failed is False
    pointer = json.loads(text)["results"][0]["pointer"]
    fetched, failed = answer(store, FETCH, {"pointer": pointer})
    assert failed is False
    # Both go to a model as UTF-8, so nothing in them may fail to encode.
    text.encode("utf-8")
    fetched.encode("utf-8")
    assert "the plover" in json.loads(fetched)["text"]


def test_the_counts_the_ledger_reads_move_with_the_tools(store: Store) -> None:
    a = store.branch("a")
    a.write("x.md", "the marker\n")
    a.checkpoint("step")
    pointer = _search(store, description="marker")["results"][0]["pointer"]
    answer(store, FETCH, {"pointer": pointer})
    answer(store, SEARCH, {})  # refused before it looks
    assert store.memory_counts["search"] == 1 and store.memory_counts["fetch"] == 1
    assert store.memory_counts["tool_errors"] == 1


def test_search_can_be_narrowed_to_some_kinds_and_hits_resolved_and_deduplicated(store: Store) -> None:
    a = store.branch("a")
    a.turn(role="user", content="the godwit rule")
    a.write("x.md", "the godwit file\n")
    a.checkpoint("the godwit commit")

    kinds = {r["kind"] for r in _search_with(store, kinds=("transcript",))}
    assert kinds == {"transcript"}

    first = store.search_contents("godwit", kinds=("transcript",))[0]
    resolved = _search_with(store, resolve=lambda hit: first)
    assert [r["pointer"] for r in resolved] == [first.pointer.to_text()]
    assert store.memory_counts["search"] == 3


def _search_with(store: Store, **options) -> list[dict]:
    text, failed = answer(store, SEARCH, {"description": "godwit"}, **options)
    assert failed is False
    return json.loads(text)["results"]


def test_the_definitions_are_neutral_and_avoid_the_argument_names_benchmarks_read() -> None:
    """A benchmark reads a call with ``path`` and ``content`` as an edit of its repository."""
    for api, unwrap in (("responses", lambda d: d), ("chat", lambda d: d["function"])):
        tools = [unwrap(d) for d in definitions(api)]
        assert [t["name"] for t in tools] == [SEARCH, FETCH]
        for tool in tools:
            names = set(tool["parameters"]["properties"])
            assert not names & {"path", "content", "file", "command"}
            assert tool["parameters"]["additionalProperties"] is False
            # No instruction to use them: whether an agent looks is the thing under test.
            wording = tool["description"].lower()
            for urged in ("always", "must", "should", "first", "before", "remember", "forgot"):
                assert urged not in wording, urged
    assert definitions("responses")[0]["type"] == definitions("chat")[0]["type"] == "function"
    with pytest.raises(ValueError):
        definitions("something else")
