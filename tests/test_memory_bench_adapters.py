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


def test_a_search_error_is_raised_not_ranked_as_nothing(tmp_path: Path) -> None:
    arm = ProductArm(lambda **kwargs: None, lambda *a, **k: {"error": "broken", "results": []}, tmp_path)
    arm.ingest(_sessions())
    with pytest.raises(RuntimeError, match="broken"):
        arm.retrieve("x")
    arm.close()
