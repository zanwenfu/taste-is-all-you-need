"""The systems that need no outside package: the baselines, and Taste's real memory.

Each takes the same sessions, oldest first, and offers a ranked list of items. ``FlatBm25`` and
``TasteGit`` with headers off hold the same text under the same keyword code, so they must rank the same;
a test holds them to it, and any gain they show belongs to keyword search and not to git. Only the header
can move Taste's row. MemPalace and Mem0 have adapters of their own, in their own environments.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from studies.memory_bench.core import Item
from studies.memory_bench.lme import Gold, rounds, unit_text
from studies.memory_ledger import BASELINE, stand_in_record, taste_record
from taste.memstore import CommitHeader, Pointer, Store
from taste.memstore.keyword import Bm25, tokens


@dataclass(frozen=True)
class _Unit:
    path: str  # neutral: a hash of the session and a number, so a file name can leak nothing
    session: str
    date: str
    numbers: tuple[int, ...]
    turns: tuple[dict, ...]
    text: str


def _units(session: dict, unit: str) -> Iterator[_Unit]:
    parts = rounds(session)
    if unit == "session":
        turns = [turn for part in parts for turn in part]
        yield _Unit(f"{session['sid']}/session.md", session["sid"], session["date"],
                    tuple(range(len(parts))), tuple(turns), unit_text(session["date"], turns))
        return
    for number, part in enumerate(parts):
        yield _Unit(f"{session['sid']}/r{number:03d}.md", session["sid"], session["date"],
                    (number,), tuple(part), unit_text(session["date"], part))


def _item(unit: _Unit) -> Item:
    return Item(unit.session, unit.date, unit.turns, unit.numbers, unit.path)


class System:
    no_retrieval = False

    def ingest(self, sessions: list[dict]) -> None:
        self.sessions = sessions

    def retrieve(self, query: str) -> list[Item]:
        raise NotImplementedError

    def ledger(self) -> dict[str, Any]:
        return stand_in_record(BASELINE, code=f"studies/memory_bench/systems.py:{type(self).__name__}")

    def close(self) -> None:
        pass


class NoMemory(System):
    no_retrieval = True

    def retrieve(self, query: str) -> list[Item]:
        return []


class FullContext(System):
    """Every session, newest first, so a budget too small for all of them loses the oldest. The benchmark's
    own truncation keeps the start and silently loses the newest."""

    def retrieve(self, query: str) -> list[Item]:
        return [_item(next(_units(session, "session"))) for session in reversed(self.sessions)]


class Oracle(System):
    def __init__(self, gold: Gold) -> None:
        self.gold = gold

    def retrieve(self, query: str) -> list[Item]:
        return [_item(next(_units(s, "session"))) for s in self.sessions if s["sid"] in self.gold.sessions]


class FlatBm25(System):
    """The same units in a plain list, ranked by the code Taste's search uses."""

    def __init__(self, unit: str = "round") -> None:
        self.unit = unit

    def ingest(self, sessions: list[dict]) -> None:
        self.units = [unit for session in sessions for unit in _units(session, self.unit)]
        self.index = Bm25()
        for unit in self.units:
            self.index.add(tokens(unit.text))

    def retrieve(self, query: str) -> list[Item]:
        return [_item(self.units[doc]) for doc, _ in self.index.rank(tokens(query))]


class FlatDense(System):
    """The same units ranked by cosine similarity, with the one embedder every system shares."""

    def __init__(self, embed: Callable[[list[str]], Sequence[Sequence[float]]], unit: str = "round") -> None:
        self.embed, self.unit = embed, unit

    def ingest(self, sessions: list[dict]) -> None:
        self.units = [unit for session in sessions for unit in _units(session, self.unit)]
        self.vectors = self.embed([unit.text for unit in self.units])

    def retrieve(self, query: str) -> list[Item]:
        wanted = self.embed([query])[0]
        scored = [(_cosine(wanted, vector), number) for number, vector in enumerate(self.vectors)]
        scored.sort(key=lambda pair: (-pair[0], -pair[1]))
        return [_item(self.units[number]) for _, number in scored]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


class TasteGit(System):
    """Taste's real memory store, driven by study code: "Taste memory, study-driven".

    One file per unit, so a pointer resolves to the round even with one commit per session. Every item
    the reader gets is fetched back from the store and compared with what went in. With a header writer
    the commit messages are searched too, and a hit on one brings back the files of that commit.
    """

    def __init__(self, workdir: Path, *, unit: str = "round", commit_per: str = "session",
                 header_writer: Callable[[dict], CommitHeader] | None = None) -> None:
        self.unit, self.commit_per, self.header_writer = unit, commit_per, header_writer
        self.folder = Path(tempfile.mkdtemp(prefix="store-", dir=workdir))
        self.store = Store.open(self.folder / "repo", "memory-bench")
        self.branch = self.store.branch("memory", producer="memory-bench")
        self.units: dict[str, _Unit] = {}
        self.held_at: dict[str, str] = {}
        self.files_of: dict[str, list[str]] = {}
        self.commits = 0

    def _commit(self, session: dict, paths: list[str], header: CommitHeader | None) -> None:
        state = self.branch.checkpoint(session["sid"], header=header)
        self.commits += 1
        self.files_of[state.id] = paths
        for path in paths:
            self.held_at[path] = state.id

    def ingest(self, sessions: list[dict]) -> None:
        for session in sessions:
            header = self.header_writer(session) if self.header_writer else None
            waiting: list[str] = []
            for unit in _units(session, self.unit):
                self.branch.write(unit.path, unit.text.encode("utf-8", "surrogateescape"))
                self.units[unit.path] = unit
                waiting.append(unit.path)
                if self.commit_per == "round":
                    self._commit(session, waiting, header)
                    waiting = []
            if waiting:
                self._commit(session, waiting, header)

    def _fetched(self, path: str) -> Item:
        unit = self.units[path]
        pointer = Pointer("memory-bench", "memory", self.held_at[path], "file", path, 1)
        if self.store.fetch(pointer).text != unit.text:
            raise ValueError(f"the store does not hold what went in at {path}")
        return _item(unit)

    def retrieve(self, query: str) -> list[Item]:
        kinds = ("file", "message") if self.header_writer else ("file",)
        hits = self.store.search_contents(query, limit=None, kinds=kinds)
        paths = [
            path
            for hit in hits
            for path in (self.files_of[hit.pointer.state] if hit.pointer.kind == "message" else [hit.pointer.where])
        ]
        return [self._fetched(path) for path in dict.fromkeys(paths)]

    def ledger(self) -> dict[str, Any]:
        return taste_record(self.store, mode="single_query")

    def close(self) -> None:
        self.store.close()
        shutil.rmtree(self.folder, ignore_errors=True)
