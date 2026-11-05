"""The loop every adapter shares: a fresh store per unit, its sessions ingested oldest first, each query ranked.

An adapter supplies ``ingest(sessions)``, ``retrieve(query) -> list[Item]`` and ``close()``. This loop times both
steps, writes one rankings line per question as it goes, and on a second run skips every unit already complete.
A unit whose questions are only partly written is run again whole, and its old lines are dropped first, so no
question is ever ranked twice. Every line carries a ledger entry: a rival is always a baseline, and an adapter that
keeps its own (Taste's store does) gives that one.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from studies.memory_bench.core import Item, rankings_line
from studies.memory_ledger import BASELINE, stand_in_record


class Adapter(Protocol):
    def ingest(self, sessions: list[dict]) -> None: ...

    def retrieve(self, query: str, *, date: str | None = None) -> list[Item]: ...

    def close(self) -> None: ...


class QueryFree:
    """What a system would put in front of an agent with no question asked: its ``memories()``, in its own order. The
    question is ignored, because the rule-loss cases have none to give."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    def ingest(self, sessions: list[dict]) -> None:
        self.adapter.ingest(sessions)
        self.first = sessions[0]

    def retrieve(self, query: str, *, date: str | None = None) -> list[Item]:
        return [Item(self.first["sid"], self.first["date"], ({"role": "memory", "content": text},), (), "query-free")
                for text in self.adapter.memories()]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.adapter, name)


def run_adapter(views: Path, out: Path, make: Callable[[], Adapter], *, system: str, code: str,
                versions: dict[str, Any], limit: int | None = None, shard: tuple[int, int] = (0, 1)) -> dict[str, int]:
    """``shard`` (i, n) runs every n-th unit from the i-th, so n processes can share a views file."""
    units = [json.loads(line) for line in Path(views).read_text().splitlines()]
    units = (units[:limit] if limit else units)[shard[0]::shard[1]]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    done = {row["qid"] for row in rows}
    partial = {q["qid"] for unit in units for q in unit["queries"]
               if any(x["qid"] in done for x in unit["queries"]) and not all(x["qid"] in done for x in unit["queries"])}
    if partial:
        out.write_text("".join(json.dumps(row) + "\n" for row in rows if row["qid"] not in partial))
        done -= partial
    default = stand_in_record(BASELINE, code=code)
    ran = 0
    for unit in units:
        if all(q["qid"] in done for q in unit["queries"]):
            continue
        adapter = make()
        try:
            started = time.perf_counter()
            adapter.ingest(unit["sessions"])
            ingest_seconds = time.perf_counter() - started
            lines = []
            for query in unit["queries"]:
                asked = time.perf_counter()
                items = adapter.retrieve(query["query"], date=query.get("date"))
                seconds = time.perf_counter() - asked
                own = getattr(adapter, "ledger", None)
                lines.append(rankings_line(query["qid"], system, items, ingest_seconds=ingest_seconds,
                                           retrieve_seconds=seconds, ledger=own() if own else default,
                                           versions=versions))
        finally:
            adapter.close()
        with out.open("a") as written:
            written.write("".join(line + "\n" for line in lines))
        ran += 1
    return {"units": len(units), "ran": ran, "questions": sum(len(u["queries"]) for u in units)}
