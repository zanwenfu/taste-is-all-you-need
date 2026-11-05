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

    def retrieve(self, query: str) -> list[Item]:
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
