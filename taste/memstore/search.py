"""Two ways to look something up: a directory of what was published, and search over all of it.

``search`` is the communicator's lookup. Every branch head carries a manifest of what it has
published, and this scans them, scoring an entry by how many query words appear in its name,
description or path. **That is a directory, not retrieval**: cheap enough that a brain can read the
whole thing. ``Store.catalog`` is that whole-index read, and is the right first move for a brain
that does not yet know the words to search for.

``search_contents`` is retrieval by keyword over everything a session recorded, in every state and
not only the newest: the files of each state's tree, commit messages, the turns of its transcript
and the verdicts on it. It answers with pointers, and ``fetch`` brings back exactly the bytes a
pointer names. Matching and scoring live in ``keyword``, which knows nothing of git.

Not searched, because they are not in git or are not this session's: the turn journal until the next
checkpoint, call ledgers, inbox messages, other sessions, and ordinary git history beneath the
session's root. Identical bytes are one hit at the state that first held them, since a record that
repeats the world on every state would otherwise fill the list; the same goes for a turn said twice
in the same words. Binary files and files over ``MAX_DOCUMENT_BYTES`` are left out and counted in
``Store.memory_counts``.

Keeping old versions is the point of a version-controlled memory, and it is also a trap: search can
hand an agent a rule that was changed later. So every hit carries its place in the session's order,
and a file's hit says whether the newest state still holds those bytes and, if not, where that path
stands now. An equal score goes to the current version, then to the newest. A better match in an old
version keeps its rank, because a question about the past must still find the past.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from taste.memstore.backend import _dec, _is_full_object_id
from taste.memstore.keyword import WORD, Bm25, stem, tokens
from taste.memstore.objects import BadPointer, Meta, ObjectType, header_text
from taste.memstore.store import NOTES, ContentHit, Fetched, Hit, Pointer, Store

KINDS = ("file", "message", "transcript", "verdict")
MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
_SNIPPET_CHARS = 240
_SNIPPET_BEFORE = 60
_TRAILER = re.compile(r"Taste-[A-Za-z-]+: ")
# What a header says about the content. Every header holds the others, so they would match every commit.
_HEADER_SEARCHED = ("what", "keywords", "paths")


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def search(
    store: Store,
    query: str,
    *,
    types: Iterable[ObjectType] | None = None,
    branches: Iterable[str] | None = None,
) -> list[Hit]:
    wanted = _tokens(query)
    allowed = set(types) if types else None
    hits: list[Hit] = []
    for name in branches or store.branches():
        sha = store.backend.ref_sha(store.ref_for(name))
        if sha is None:
            continue
        head = store.state(sha)
        for entry in head.manifest.entries.values():
            if allowed and entry.type not in allowed:
                continue
            haystack = set(_tokens(f"{entry.name} {entry.description} {entry.path}"))
            score = sum(1 for w in wanted if w in haystack)
            if score:
                hits.append(Hit(branch=name, state=head, entry=entry, score=score))
    hits.sort(key=lambda h: (-h.score, h.branch, h.entry.name))
    return hits


# ------------------------------------------------------------------ what a document's text is


def _values(value: Any) -> Iterator[str]:
    """The words a person or model wrote in a record: never its field names."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _values(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _values(item)
    elif isinstance(value, int | float) and not isinstance(value, bool):
        yield str(value)


def _rendered(record: dict[str, Any]) -> str:
    """A turn or verdict as text: a line per field, its strings exactly as recorded."""
    return "\n".join(
        f"{key}: {value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)}"
        for key, value in sorted(record.items())
    )


def _turn_lines(store: Store, state: str) -> list[str]:
    raw = store.backend.note_get(NOTES["transcript"], state)
    return raw.split("\n") if raw else []


def _text(store: Store, state: str, kind: str, where: str, line: int) -> str:
    """The document a pointer names, read again from its state. ``BadPointer`` if there is none."""
    if kind == "file":
        data = store.state(state).read_bytes(where)
        if data is None:
            raise BadPointer(f"no file {where!r} in state {state}")
        return _dec(data)
    if kind == "message":
        if where:
            raise BadPointer("a commit message has no path")
        return store.backend.commit_message(state)
    if kind == "transcript":
        turns = _turn_lines(store, state)
        if where != NOTES["transcript"] or not 1 <= line <= len(turns) or not turns[line - 1].strip():
            raise BadPointer(f"no turn {line} in state {state}")
        return _rendered(json.loads(turns[line - 1]))
    if kind == "verdict":
        verdicts = store.state(state).verdicts
        if where != NOTES["verdicts"] or not 1 <= line <= len(verdicts):
            raise BadPointer(f"no verdict {line} in state {state}")
        return _rendered(verdicts[line - 1].to_dict())
    raise BadPointer(f"unknown kind {kind!r}")


def _searched_lines(kind: str, text: str) -> list[str]:
    """Which lines of a document count. A commit message's trailers and most of its header do not:
    every message carries them, and matching them would rank every commit."""
    lines = text.split("\n")
    if kind != "message":
        return lines
    header = set(header_text(text).split("\n"))
    return [
        line
        for line in lines
        if not _TRAILER.match(line)
        and (line not in header or line.partition(": ")[0] in _HEADER_SEARCHED)
    ]


def _searched_words(kind: str, text: str) -> str:
    """The words a document is indexed by: a header line's value, without its label."""
    header = set(header_text(text).split("\n")) if kind == "message" else set()
    return "\n".join(
        line.partition(": ")[2] if line in header else line for line in _searched_lines(kind, text)
    )


# ------------------------------------------------------------------ the index


@dataclass(frozen=True)
class _Document:
    state: str
    branch: str
    kind: str
    where: str
    line: int  # a turn or verdict's place in its note; 0 for a file or message
    order: int
    created_at: str
    blob: str = ""


class _Index:
    """What one choice of kinds and branches can see, grown as the store grows.

    A state's contents never change except its verdicts, which only accumulate, so a search pays for
    what is new since the last one. If a commit it holds is no longer reachable, it starts again.
    """

    def __init__(self, kinds: tuple[str, ...]) -> None:
        self.kinds = kinds
        self._reset()

    def _reset(self) -> None:
        self.scorer = Bm25()
        self.documents: list[_Document] = []
        self.commits: set[str] = set()
        self.states: list[tuple[str, Meta]] = []
        self.heads: list[tuple[str, str]] = []
        self.head_blobs: set[str] = set()
        self.trees: set[str] = set()
        self.blobs: set[str] = set()
        self.turns: set[bytes] = set()
        self.verdicts: dict[str, int] = {}
        self.too_large: set[str] = set()
        self.binary: set[str] = set()
        self.mark: tuple[Any, ...] | None = None

    def _add(self, document: _Document, terms: Iterable[str]) -> None:
        self.scorer.add(terms)
        self.documents.append(document)

    def refresh(self, store: Store, heads: list[tuple[str, str]]) -> None:
        mark = (tuple(sorted(heads)), store.backend.ref_sha(NOTES["verdicts"]))
        if mark == self.mark:
            return
        reachable = store.backend.commits_oldest_first([sha for _, sha in heads])
        if not self.commits.issubset(reachable):
            self._reset()
        for commit in reachable:
            if commit in self.commits:
                continue
            self.commits.add(commit)
            raw = store.backend.note_get(NOTES["meta"], commit)
            if raw is None:
                continue  # ordinary git history beneath the session's root
            meta = Meta.from_json(raw)
            if meta.session == store.session:
                self.states.append((commit, meta))
                self._add_state(store, commit, meta)
        if "verdict" in self.kinds:
            self._add_verdicts(store)
        if "file" in self.kinds:
            self.head_blobs = set().union(*(store.backend.blobs_under(sha) for _, sha in heads))
        self.heads = heads
        self.mark = mark

    def _add_state(self, store: Store, commit: str, meta: Meta) -> None:
        if "file" in self.kinds:
            self._add_files(store, commit, meta)
        # The store's own structural commits say nothing anyone wrote.
        if "message" in self.kinds and meta.kind not in ("root", "branch"):
            words = _searched_words("message", store.backend.commit_message(commit))
            self._add(self._document(commit, meta, "message", "", 0), tokens(words))
        if "transcript" in self.kinds:
            self._add_turns(store, commit, meta)

    def _document(self, commit: str, meta: Meta, kind: str, where: str, line: int, blob: str = "") -> _Document:
        order = next(n for n, (state, _) in enumerate(reversed(self.states)) if state == commit)
        return _Document(commit, meta.branch, kind, where, line, len(self.states) - order, meta.created_at, blob)

    def _add_files(self, store: Store, commit: str, meta: Meta) -> None:
        for path, blob, size in store.backend.unseen_files(commit, self.trees, self.blobs):
            if size > MAX_DOCUMENT_BYTES:
                self.too_large.add(blob)
                continue
            data = store.backend.object_bytes(blob)
            if b"\0" in data[:8000]:
                self.binary.add(blob)
                continue
            self._add(self._document(commit, meta, "file", path, 0, blob), tokens(_dec(data)))

    def _add_turns(self, store: Store, commit: str, meta: Meta) -> None:
        # Every so often a state stores the whole transcript, not the turns it added.
        for number, line in enumerate(_turn_lines(store, commit), 1):
            if not line.strip():
                continue
            key = hashlib.sha256(line.encode("utf-8", "surrogateescape")).digest()
            if key in self.turns:
                continue
            self.turns.add(key)
            words = "\n".join(_values(json.loads(line)))
            self._add(self._document(commit, meta, "transcript", NOTES["transcript"], number), tokens(words))

    def _add_verdicts(self, store: Store) -> None:
        for commit, meta in self.states:
            verdicts = store.state(commit).verdicts
            for number in range(self.verdicts.get(commit, 0), len(verdicts)):
                words = "\n".join(_values(verdicts[number].to_dict()))
                self._add(self._document(commit, meta, "verdict", NOTES["verdicts"], number + 1), tokens(words))
            self.verdicts[commit] = len(verdicts)


def _clip(line: str, wanted: set[str]) -> str:
    if len(line) <= _SNIPPET_CHARS:
        return line
    start = 0
    for match in WORD.finditer(line):
        if stem(match.group().lower()) in wanted:
            start = max(0, match.start() - _SNIPPET_BEFORE)
            break
    return line[start : start + _SNIPPET_CHARS]


def _best_line(kind: str, text: str, wanted: set[str]) -> tuple[int, str]:
    """The first of the lines holding the most query words: its number, and that line cut short."""
    best_number, best, most = 1, "", -1
    searched = set(_searched_lines(kind, text))
    for number, line in enumerate(text.split("\n"), 1):
        if line not in searched:
            continue
        found = len(wanted.intersection(tokens(line)))
        if found > most:
            best_number, best, most = number, line, found
    return best_number, _clip(best, wanted)


def _replaced_by(store: Store, index: _Index, document: _Document) -> str:
    """A pointer to this path in the newest state that has it, this document's own branch first."""
    for name, head in sorted(index.heads, key=lambda item: (item[0] != document.branch, item[0])):
        blob = store.backend.blob_at(head, document.where)
        if blob is not None:
            return Pointer(store.session, name, head, "file", document.where, 1, blob).to_text()
    return ""


def _hit(store: Store, index: _Index, document: _Document, score: float, wanted: set[str]) -> ContentHit:
    text = _text(store, document.state, document.kind, document.where, document.line)
    number, snippet = _best_line(document.kind, text, wanted)
    pointer = Pointer(
        session=store.session,
        branch=document.branch,
        state=document.state,
        kind=document.kind,
        where=document.where,
        line=document.line or number,
        blob=document.blob,
    )
    current = document.blob in index.head_blobs if document.kind == "file" else None
    return ContentHit(
        pointer=pointer,
        score=score,
        snippet=snippet,
        header=header_text(store.backend.commit_message(document.state)),
        order=document.order,
        created_at=document.created_at,
        current=current,
        replaced_by=_replaced_by(store, index, document) if current is False else "",
    )


def search_contents(
    store: Store,
    description: str,
    *,
    limit: int | None = 10,
    kinds: Iterable[str] | None = None,
    branches: Iterable[str] | None = None,
) -> list[ContentHit]:
    """What this session recorded that holds the words of ``description``, best first."""
    chosen = set(KINDS) if kinds is None else set(kinds)
    if not chosen.issubset(KINDS):
        raise ValueError(f"kinds must be among {KINDS}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    names = tuple(sorted(branches)) if branches is not None else None
    selected = tuple(kind for kind in KINDS if kind in chosen)
    with store._content_lock:
        store.memory_counts["search"] += 1
        index = store._content_indexes.setdefault((selected, names), _Index(selected))
        heads = [(name, store.backend.ref_sha(store.ref_for(name))) for name in (names or store.branches())]
        index.refresh(store, [(name, sha) for name, sha in heads if sha])
        indexes = store._content_indexes.values()
        store.memory_counts["documents_too_large"] = len(set().union(*(i.too_large for i in indexes)))
        store.memory_counts["documents_binary"] = len(set().union(*(i.binary for i in indexes)))
        query = tokens(description)

        def replaced(ranked: tuple[int, float]) -> bool:
            document = index.documents[ranked[0]]
            return document.kind == "file" and document.blob not in index.head_blobs

        # The sort is stable, and the scorer gives ties to the newest.
        ordered = sorted(index.scorer.rank(query), key=lambda ranked: (-ranked[1], replaced(ranked)))
        return [
            _hit(store, index, index.documents[doc], score, set(query))
            for doc, score in (ordered if limit is None else ordered[:limit])
        ]


def fetch(
    store: Store, pointer: Pointer | str, *, offset: int = 0, limit: int | None = None
) -> Fetched:
    """The document a pointer names, whole or one page. Checked as an assignment's input is."""
    if offset < 0 or (limit is not None and limit < 1):
        raise ValueError("fetch needs an offset of at least 0 and a limit of at least 1")
    with store._content_lock:
        store.memory_counts["fetch"] += 1
    chosen = Pointer.from_text(pointer) if isinstance(pointer, str) else pointer
    if not _is_full_object_id(chosen.state):
        raise BadPointer("a pointer names its state by exact id")
    raw = store.backend.note_get(NOTES["meta"], chosen.state)
    if raw is None:
        raise BadPointer(f"no state {chosen.state}")
    meta = Meta.from_json(raw)
    if meta.session != store.session or chosen.session not in ("", store.session):
        raise BadPointer("that state belongs to another session")
    if chosen.branch not in ("", meta.branch):
        raise BadPointer("that state belongs to another branch")
    if chosen.kind == "file" and chosen.blob and store.state(chosen.state).blob(chosen.where) != chosen.blob:
        raise BadPointer("those bytes are not at that path in that state")
    text = _text(store, chosen.state, chosen.kind, chosen.where, chosen.line)
    page = text[offset:] if limit is None else text[offset : offset + limit]
    next_offset = offset + len(page)
    return Fetched(text=page, offset=offset, next_offset=next_offset, size=len(text), eof=next_offset >= len(text))
