"""A directory over published manifests: the communicator's lookup.

Every branch head carries a manifest of what it has published, and this
scans them, scoring an entry by how many query words appear in its name,
description or path.

**This is a directory, not retrieval.** There are no embeddings and no
ranking beyond token overlap, deliberately: semantic recall over what an
agent *knows* is a different layer and a solved problem elsewhere (mem0 and
friends). What belongs here is the index of what exists and where, cheap
enough that a brain can read the whole thing. ``Store.catalog`` is that
whole-index read, and is the right first move for a brain that does not yet
know the words to search for.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from taste.memstore.objects import ObjectType
from taste.memstore.store import Hit, Store


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
