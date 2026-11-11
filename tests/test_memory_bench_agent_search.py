"""The coding-agent search arm: what it may read, what it is charged, and that its list is what the records stage keeps whole."""

from __future__ import annotations

import json

from studies.memory_bench.adapters.agent_search import AgentSearch, Reading, command, files_of, search
from studies.memory_bench.core import deliver, item_text

WORDS = lambda text: len(text.split())  # noqa: E731

SESSIONS = [
    {"sid": "a1", "date": "2023/05/01 (Mon) 10:00", "turns": [
        {"role": "user", "content": "I adopted a cat named Miso yesterday"},
        {"role": "assistant", "content": "Congratulations on Miso"},
        {"role": "user", "content": "What food is best for kittens"},
        {"role": "assistant", "content": "Wet food with high protein is a good start for most kittens"},
    ]},
    {"sid": "b2", "date": "2023/06/02 (Fri) 09:00", "turns": [
        {"role": "user", "content": "My dentist appointment moved to Friday"},
        {"role": "assistant", "content": "Noted, Friday it is"},
    ]},
]


def scripted(*replies: str):
    asked: list[list[dict]] = []

    def ask(messages: list[dict]) -> dict:
        asked.append([dict(message) for message in messages])
        return {"text": replies[min(len(asked) - 1, len(replies) - 1)], "finish_reason": "stop"}

    return ask, asked


def test_files_are_one_per_session_in_rounds_oldest_first():
    files = files_of(SESSIONS)
    assert list(files) == ["session-001", "session-002"]
    assert [part.number for part in files["session-001"]] == [0, 1]
    assert files["session-001"][1].turns[0]["content"] == "What food is best for kittens"
    assert files["session-002"][0].item().rounds == (0,)


def test_search_is_case_insensitive_shows_excerpts_and_falls_back_to_literal_text():
    files = files_of(SESSIONS)
    out = search(files, "miso")
    assert out.startswith("1 rounds match") and "session-001" in out and "round 0" in out
    assert "No round matches" in search(files, "giraffe")
    assert "No round matches 'Friday('" in search(files, "Friday(")  # not a valid pattern: searched as text, no error
    capped = search(files, "e", hits=1)
    assert "and 2 more" in capped


def test_reading_charges_each_round_once_as_every_system_is_charged():
    files = files_of(SESSIONS)
    reading = Reading(100, WORDS)
    reading.read(files, "session-001", 0, 1)
    expected = sum(WORDS(item_text(part.item())) for part in files["session-001"])
    assert reading.used == expected
    again = reading.read(files, "session-001", 1, 1)
    assert "already read" in again and reading.used == expected
    assert [item.source for item in reading.items] == ["session-001/r0", "session-001/r1"]


def test_a_round_that_does_not_fit_is_skipped_and_reading_goes_on():
    files = files_of(SESSIONS)
    first = WORDS(item_text(files["session-001"][0].item()))
    reading = Reading(first + 3, WORDS)
    out = reading.read(files, "session-001", 0, 1)
    assert "does not fit" in out and reading.used == first
    reading.read(files, "session-002", 0, 0)  # 10 words: does not fit in 3 either
    assert reading.used == first


def test_bad_reads_say_why():
    files = files_of(SESSIONS)
    reading = Reading(100, WORDS)
    assert "no file" in reading.read(files, "session-009", 0, 0)
    assert "rounds 0 to 0" in reading.read(files, "session-002", 1, 1)
    assert reading.used == 0


def test_command_is_the_last_command_line():
    assert command("I will look.\nSEARCH: cat") == ("SEARCH", "cat")
    assert command("READ: session-001 0-1\nthen\nDONE") == ("DONE", "")
    assert command("`READ: session-002 0`") == ("READ", "session-002 0")
    assert command("no command here") is None


def test_the_loop_reads_what_it_is_told_and_stops_on_done(tmp_path):
    ask, asked = scripted("SEARCH: cat", "READ: session-001 0", "DONE")
    agent = AgentSearch(ask, WORDS, 100, trace=tmp_path / "trace.jsonl")
    agent.ingest(SESSIONS)
    items = agent.retrieve("What is my cat's name?", date="2023/07/01", qid="q1")
    assert [item.source for item in items] == ["session-001/r0"]
    assert "Question: What is my cat's name?" in asked[0][1]["content"] and "session-002" in asked[0][1]["content"]
    assert "1 rounds match" in asked[1][-1]["content"]
    trace = json.loads((tmp_path / "trace.jsonl").read_text())
    assert trace["stop"] == "done" and trace["qid"] == "q1" and trace["read"] == ["session-001/r0"]


def test_the_loop_stops_at_its_step_cap_and_when_the_budget_is_spent():
    ask, asked = scripted("thinking, no command")
    agent = AgentSearch(ask, WORDS, 100, steps=3)
    agent.ingest(SESSIONS)
    assert agent.retrieve("q") == [] and len(asked) == 3
    assert "No command found" in asked[-1][-1]["content"]
    whole = WORDS(item_text(files_of(SESSIONS)["session-002"][0].item()))
    ask, asked = scripted("READ: session-002 0", "SEARCH: cat")
    agent = AgentSearch(ask, WORDS, whole, steps=5)
    agent.ingest(SESSIONS)
    assert len(agent.retrieve("q")) == 1 and len(asked) == 1


def test_the_records_stage_keeps_the_whole_list_at_the_same_budget():
    ask, _ = scripted("READ: session-001 0-1", "READ: session-002 0", "DONE")
    agent = AgentSearch(ask, WORDS, 40)
    agent.ingest(SESSIONS)
    items = agent.retrieve("q")
    delivered = deliver(items, 40, WORDS)
    assert delivered.items == tuple(items) and delivered.dropped == 0 and delivered.cut == 0
