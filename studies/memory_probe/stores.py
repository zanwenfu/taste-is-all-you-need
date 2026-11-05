"""Where the probe can look: a history kept turn by turn in Taste's real memory, or the same turns in a flat list.

``GitTurns`` is "Taste memory, study-driven": one fresh store per case, each turn appended to the transcript and
checkpointed on its own, so a hit is one turn. With a header writer, search also reads each commit's header, and a
header that matches stands for the turn its commit added. Without one, it reads the turns alone, and ``FlatTurns``,
the same turns under the same keyword code, must rank them identically: a difference is a bug, never a finding.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from studies.memory_ledger import taste_record
from taste.memstore import CommitHeader, ContentHit, Pointer, Store
from taste.memstore import tools as memory_tools
from taste.memstore.keyword import Bm25, tokens
from taste.memstore.objects import header_fields
from taste.memstore.store import NOTES

SUBJECT = "conversation turn"  # the same in every commit, so it can never rank one above another


class GitTurns:
    def __init__(self, workdir: Path, *, header: Callable[[int, dict], CommitHeader] | None = None) -> None:
        self.folder = Path(tempfile.mkdtemp(prefix="turns-", dir=workdir))
        self.store = Store.open(self.folder / "repo", "memory-probe")
        self.branch = self.store.branch("conversation", producer="memory-probe")
        self.header = header
        self.kinds = ("transcript", "message") if header else ("transcript",)
        self.turn_at: dict[str, int] = {}
        self.commits = 0

    def ingest(self, messages: list[dict]) -> None:
        for turn in messages:
            self.add(turn)

    def add(self, turn: dict) -> None:
        """One turn, committed on its own, as it happens."""
        number = self.commits
        self.branch.turn(role=turn["role"], content=turn["content"])
        state = self.branch.checkpoint(SUBJECT, header=self.header(number, turn) if self.header else None)
        self.turn_at[state.id] = number
        self.commits += 1

    def _turn(self, hit: ContentHit) -> ContentHit:
        """A header's commit stands for the one turn it added: the last line of its transcript note."""
        if hit.pointer.kind != "message":
            return hit
        lines = (self.store.backend.note_get(NOTES["transcript"], hit.pointer.state) or "").split("\n")
        line = max(number for number, text in enumerate(lines, 1) if text.strip())
        pointer = replace(hit.pointer, kind="transcript", where=NOTES["transcript"], line=line)
        first = self.store.fetch(pointer).text.split("\n", 1)[0]
        return replace(hit, pointer=pointer, snippet=first[:240])

    def answer(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        return memory_tools.answer(self.store, name, arguments, kinds=self.kinds, resolve=self._turn)

    def index(self) -> list[dict[str, Any]]:
        """Each commit, oldest first: what its header says, read back from the store, and a pointer to its turn."""
        rows = []
        for state, number in sorted(self.turn_at.items(), key=lambda item: item[1]):
            lines = (self.store.backend.note_get(NOTES["transcript"], state) or "").split("\n")
            line = max(position for position, text in enumerate(lines, 1) if text.strip())
            fields = header_fields(self.store.state(state).header)
            rows.append({"turn": number, "what": fields.get("what", ""), "keywords": fields.get("keywords", []),
                         "pointer": Pointer("memory-probe", "conversation", state, "transcript", NOTES["transcript"],
                                            line).to_text()})
        return rows

    def turn_of(self, pointer: str | Pointer) -> int | None:
        chosen = Pointer.from_text(pointer) if isinstance(pointer, str) else pointer
        return self.turn_at.get(chosen.state)

    def ledger(self, mode: str = "agent_tools") -> dict[str, Any]:
        return taste_record(self.store, mode=mode)

    def close(self) -> None:
        self.store.close()
        shutil.rmtree(self.folder, ignore_errors=True)


class NotesStore:
    """A notes file in Taste's memory: every version committed in turn (history), or only the last one."""

    PATH = "notes.md"

    def __init__(self, workdir: Path, versions: list[str], *, history: bool) -> None:
        self.folder = Path(tempfile.mkdtemp(prefix="notes-", dir=workdir))
        self.store = Store.open(self.folder / "repo", "memory-probe")
        self.branch = self.store.branch("notes", producer="memory-probe")
        self.commits = 0
        for text in versions if history else versions[-1:]:
            self.branch.write(self.PATH, text)
            self.branch.checkpoint("notes updated")
            self.commits += 1

    def answer(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        return memory_tools.answer(self.store, name, arguments, kinds=("file",))

    def turn_of(self, pointer: str | Pointer) -> int | None:
        return None

    def ledger(self, mode: str = "agent_tools") -> dict[str, Any]:
        return taste_record(self.store, mode=mode)

    def close(self) -> None:
        self.store.close()
        shutil.rmtree(self.folder, ignore_errors=True)


class FlatTurns:
    """The same turns in a plain list: the words search reads of a turn are its content and its role."""

    def ingest(self, messages: list[dict]) -> None:
        self.index = Bm25()
        for turn in messages:
            self.index.add(tokens(f"{turn['content']}\n{turn['role']}"))

    def rank(self, query: str) -> list[int]:
        return [doc for doc, _ in self.index.rank(tokens(query))]
