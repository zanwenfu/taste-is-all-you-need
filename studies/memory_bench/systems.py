"""The systems that need no outside package: the baselines, and Taste's real memory.

Each takes the same sessions, oldest first, and offers a ranked list of items. ``FlatBm25`` and
``TasteGit`` with headers off hold the same text under the same keyword code, so they must rank the same;
a test holds them to it, and any gain they show belongs to keyword search and not to git. Only the header
can move Taste's row. MemPalace, Mem0 and the dense baseline run as adapters, in environments of their own.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from studies.memory_bench.core import Item, rankings_line  # noqa: F401 - rankings_line is re-exported
from studies.memory_bench.lme import Gold, Unit, rounds, units_of
from studies.memory_ledger import BASELINE, stand_in_record, taste_record
from taste.memstore import CommitHeader, Pointer, Store
from taste.memstore.keyword import Bm25, tokens


class System:
    """``ingest`` gets the sessions oldest first, and ``retrieve`` the question's own words. The unit and question
    ids are passed only so a system made elsewhere can be looked up; no system here reads them."""

    no_retrieval = False

    def ingest(self, sessions: list[dict], unit: str | None = None) -> None:
        self.sessions = sessions

    def retrieve(self, query: str, qid: str | None = None) -> list[Item]:
        raise NotImplementedError

    def ledger(self) -> dict[str, Any]:
        return stand_in_record(BASELINE, code=f"studies/memory_bench/systems.py:{type(self).__name__}")

    def close(self) -> None:
        pass


class NoMemory(System):
    no_retrieval = True

    def retrieve(self, query: str, qid: str | None = None) -> list[Item]:
        return []


class FullContext(System):
    """Every session, newest first, so a budget too small for all of them loses the oldest. The benchmark's
    own truncation keeps the start and silently loses the newest."""

    def retrieve(self, query: str, qid: str | None = None) -> list[Item]:
        return [next(units_of(session, "session")).item() for session in reversed(self.sessions)]


class Oracle(System):
    def __init__(self, gold: Gold) -> None:
        self.gold = gold

    def retrieve(self, query: str, qid: str | None = None) -> list[Item]:
        return [next(units_of(s, "session")).item() for s in self.sessions if s["sid"] in self.gold.sessions]


class FlatBm25(System):
    """The same units in a plain list, ranked by the code Taste's search uses. Git stores identical bytes once, so this
    searches the first copy of identical text only, or the two could not rank alike. With no question asked it hands over
    every record, as Taste's store does."""

    def __init__(self, unit: str = "round") -> None:
        self.unit = unit

    def ingest(self, sessions: list[dict], unit: str | None = None) -> None:
        self.every = [unit for session in sessions for unit in units_of(session, self.unit)]
        self.units = list({unit.text: unit for unit in reversed(self.every)}.values())[::-1]
        self.index = Bm25()
        for unit in self.units:
            self.index.add(tokens(unit.text))

    def retrieve(self, query: str, qid: str | None = None) -> list[Item]:
        return [self.units[doc].item() for doc, _ in self.index.rank(tokens(query))]

    def memories(self) -> list[str]:
        return [unit.text for unit in reversed(self.every)]


class TasteGit(System):
    """Taste's real memory store, driven by study code: "Taste memory, study-driven".

    One file per unit, so a pointer resolves to the round even with one commit per session. Every item
    the reader gets is fetched back from the store and compared with what went in. With a header writer
    the commit messages are searched too, and a hit on one brings back the files of that commit.
    """

    def __init__(self, workdir: Path, *, unit: str = "round", commit_per: str = "session",
                 header_writer: Callable[[dict], CommitHeader] | None = None, mode: str = "single_query") -> None:
        self.unit, self.commit_per, self.header_writer, self.mode = unit, commit_per, header_writer, mode
        self.folder = Path(tempfile.mkdtemp(prefix="store-", dir=workdir))
        self.store = Store.open(self.folder / "repo", "memory-bench")
        self.branch = self.store.branch("memory", producer="memory-bench")
        self.units: dict[str, Unit] = {}
        self.held_at: dict[str, str] = {}
        self.files_of: dict[str, list[str]] = {}
        self.commits = 0

    def _commit(self, session: dict, paths: list[str], header: CommitHeader | None) -> None:
        state = self.branch.checkpoint(session["sid"], header=header)
        self.commits += 1
        self.files_of[state.id] = paths
        for path in paths:
            self.held_at[path] = state.id

    def ingest(self, sessions: list[dict], unit: str | None = None) -> None:
        for session in sessions:
            header = self.header_writer(session) if self.header_writer else None
            waiting: list[str] = []
            for unit in units_of(session, self.unit):
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
        return unit.item()

    def retrieve(self, query: str, qid: str | None = None) -> list[Item]:
        kinds = ("file", "message") if self.header_writer else ("file",)
        hits = self.store.search_contents(query, limit=None, kinds=kinds)
        paths = [
            path
            for hit in hits
            for path in (self.files_of[hit.pointer.state] if hit.pointer.kind == "message" else [hit.pointer.where])
        ]
        return [self._fetched(path) for path in dict.fromkeys(paths)]

    def memories(self) -> list[str]:
        """With no question asked, newest first: what each commit's header says it holds or, with headers off, the
        records themselves, read back from the store."""
        if self.header_writer:
            return [self.store.state(state).header for state in reversed(self.files_of)]
        return [self.store.fetch(Pointer("memory-bench", "memory", state, "file", path, 1)).text
                for state in reversed(self.files_of) for path in reversed(self.files_of[state])]

    @property
    def headers_missing(self) -> int:
        return getattr(self.header_writer, "missing", 0)

    def ledger(self) -> dict[str, Any]:
        return taste_record(self.store, mode=self.mode)

    def close(self) -> None:
        self.store.close()
        shutil.rmtree(self.folder, ignore_errors=True)


@lru_cache(maxsize=4)
def _rankings(path: str, stamp: tuple[int, int]) -> dict[str, dict]:
    """Read once per process, and again if the file changes (``stamp`` is its time and size)."""
    rows: dict[str, dict] = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row["qid"] in rows:
            raise ValueError(f"{row['qid']} is ranked twice")
        rows[row["qid"]] = row
    return rows


class Precomputed(System):
    """Ranked lists a system made in an environment of its own (MemPalace, Mem0), read back as data.

    Each item is checked against the unit's own sessions: the session must be one of them, its date must be that
    session's date, and its rounds must exist. Timings and the ledger entry are the adapter's own.
    """

    def __init__(self, path: Path) -> None:
        stat = Path(path).stat()
        self.rows = _rankings(str(path), (stat.st_mtime_ns, stat.st_size))
        self.label: str | None = None
        self._last: dict | None = None

    def ingest(self, sessions: list[dict], unit: str | None = None) -> None:
        self.sessions = {session["sid"]: session for session in sessions}

    def _row(self, qid: str | None) -> dict:
        if qid not in self.rows:
            raise ValueError(f"no ranking for {qid}")
        return self.rows[qid]

    def retrieve(self, query: str, qid: str | None = None) -> list[Item]:
        row = self._row(qid)
        items = []
        for raw in row["items"]:
            session = self.sessions.get(raw["session"])
            if session is None:
                raise ValueError(f"{raw['session']} is not one of this question's sessions")
            if raw["date"] != session["date"]:
                raise ValueError(f"{raw['session']} is dated {session['date']}, not {raw['date']}")
            known = len(rounds(session))
            for number in raw["rounds"]:
                if not 0 <= number < known:
                    raise ValueError(f"{raw['session']} has no round {number}")
            items.append(Item(raw["session"], raw["date"], tuple(raw["turns"]), tuple(raw["rounds"]),
                              raw.get("source", "")))
        self.label, self._last = row["system"], row
        return items

    def measured(self, qid: str) -> tuple[float, float]:
        row = self._row(qid)
        return row["ingest_seconds"], row["retrieve_seconds"]

    def ledger(self) -> dict[str, Any]:
        if self._last is None:
            raise ValueError("nothing was retrieved")
        return self._last["ledger"]
