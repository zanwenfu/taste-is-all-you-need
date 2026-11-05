"""Search and fetch as tools an agent can call: two tools, one implementation, for whichever agent runs.

The agent reads headers and snippets first and fetches only what it wants. Whether it looks at all is
the thing a study measures, so nothing here tells it to. The arguments are not named ``path`` and
``content``, because a benchmark that reads a run's tool calls takes a call carrying those for an edit
of its repository. A wrong call comes back as an error the agent can read; it never raises into the
agent's loop.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

from taste.memstore.objects import BadPointer
from taste.memstore.store import ContentHit, Store

SEARCH = "memory_search"
FETCH = "memory_fetch"
DEFAULT_RESULTS = 5
MAX_RESULTS = 20
MAX_FETCH = 8_192

_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": SEARCH,
        "description": (
            "Search a record of this session by keywords. Returns a ranked list. Each result has a "
            "pointer, the header of the commit it came from, a short snippet, and its order in the "
            "session. A file's result also says whether the newest state still holds those bytes and, "
            "if not, gives a pointer to that file as it is now."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Free text saying what is wanted."},
                "limit": {"type": "integer", "description": f"How many results, 1 to {MAX_RESULTS}."},
            },
            "required": ["description"],
            "additionalProperties": False,
        },
    },
    {
        "name": FETCH,
        "description": f"Read what a pointer from {SEARCH} names, one page at a time.",
        "parameters": {
            "type": "object",
            "properties": {
                "pointer": {"type": "string", "description": "A pointer, exactly as it was returned."},
                "offset": {"type": "integer", "description": "Where to start, in characters."},
                "limit": {"type": "integer", "description": f"How many characters, 1 to {MAX_FETCH}."},
            },
            "required": ["pointer"],
            "additionalProperties": False,
        },
    },
)


class _Refused(ValueError):
    pass


def definitions(api: str = "responses") -> list[dict[str, Any]]:
    """The two tools in the form an API takes them: "responses" or "chat" (chat completions)."""
    if api == "responses":
        return [{"type": "function", **json.loads(json.dumps(item))} for item in _DEFINITIONS]
    if api == "chat":
        return [{"type": "function", "function": json.loads(json.dumps(item))} for item in _DEFINITIONS]
    raise ValueError(f"unknown api {api!r}")


def _text(value: dict[str, Any]) -> str:
    """JSON a model can be sent: bytes that were not UTF-8 become a replacement mark here, and only here."""
    return json.dumps(value, ensure_ascii=False).encode("utf-8", "replace").decode("utf-8")


def _whole_number(arguments: dict[str, Any], name: str, default: int, low: int, high: int) -> int:
    value = arguments.get(name, default)
    if type(value) is not int or not low <= value <= high:
        raise _Refused(f"{name} must be a whole number from {low} to {high}")
    return value


def _only(arguments: dict[str, Any], allowed: tuple[str, ...]) -> None:
    unknown = sorted(set(arguments) - set(allowed))
    if unknown:
        raise _Refused(f"unknown arguments: {', '.join(unknown)}")


def _result(hit: ContentHit) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pointer": hit.pointer.to_text(),
        "kind": hit.pointer.kind,
        "order": hit.order,
        "recorded": hit.created_at,
    }
    if hit.current is not None:
        result["current"] = hit.current
    if hit.replaced_by:
        result["replaced_by"] = hit.replaced_by
    if hit.header:
        result["header"] = hit.header
    result["snippet"] = hit.snippet
    return result


def _search(store: Store, arguments: dict[str, Any], kinds: Iterable[str] | None,
            resolve: Callable[[ContentHit], ContentHit] | None) -> str:
    _only(arguments, ("description", "limit"))
    description = arguments.get("description")
    if not isinstance(description, str) or not description.strip():
        raise _Refused("description must be text saying what is wanted")
    limit = _whole_number(arguments, "limit", DEFAULT_RESULTS, 1, MAX_RESULTS)
    if resolve is None:
        hits = store.search_contents(description, limit=limit, kinds=kinds)
    else:
        # A caller may turn a hit into the thing it stands for (a header's commit into what that commit added);
        # two hits that come to the same thing are shown once, at the better rank.
        hits, seen = [], set()
        for hit in map(resolve, store.search_contents(description, limit=None, kinds=kinds)):
            if hit.pointer.to_text() not in seen:
                seen.add(hit.pointer.to_text())
                hits.append(hit)
            if len(hits) == limit:
                break
    return _text({"results": [_result(hit) for hit in hits]})


def _fetch(store: Store, arguments: dict[str, Any]) -> str:
    _only(arguments, ("pointer", "offset", "limit"))
    pointer = arguments.get("pointer")
    if not isinstance(pointer, str):
        raise _Refused("pointer must be text, exactly as a search returned it")
    offset = _whole_number(arguments, "offset", 0, 0, 1 << 40)
    limit = _whole_number(arguments, "limit", MAX_FETCH, 1, MAX_FETCH)
    page = store.fetch(pointer, offset=offset, limit=limit)
    return _text(
        {"text": page.text, "offset": page.offset, "next_offset": page.next_offset, "size": page.size, "eof": page.eof}
    )


def answer(store: Store, name: str, arguments: dict[str, Any], *, kinds: Iterable[str] | None = None,
           resolve: Callable[[ContentHit], ContentHit] | None = None) -> tuple[str, bool]:
    """(what the agent is told, whether the call failed). ``kinds`` narrows what search reads, and ``resolve``
    maps each hit before results are cut to the limit."""
    try:
        if name == SEARCH:
            return _search(store, arguments, kinds, resolve), False
        if name == FETCH:
            return _fetch(store, arguments), False
        raise _Refused(f"unknown tool {name!r}")
    except (_Refused, BadPointer) as error:
        store.memory_counts["tool_errors"] += 1
        return _text({"error": str(error)}), True
