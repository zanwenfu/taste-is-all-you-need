"""Finding what any branch has: the communicator's question, answered from manifests.

Every branch head carries a manifest of what it has published. A search
scans them and scores each entry by how many query words appear in its
name, description or path. Deliberately simple: the point of the manifest
is that the answer to "does anyone have X?" is a scan of a few small JSON
notes, not a crawl of every tree.
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
