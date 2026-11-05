"""Dev and held-out, grouped: a question and its ``_abs`` twin share their sessions, so they stay together.

MemPalace's own 50/450 split separates such a pair (``6456829e`` in dev, ``6456829e_abs`` in held-out).
Groups are drawn by a keyed hash, one type at a time, with abstention as a type of its own, so every kind
of question reaches dev.
"""

from __future__ import annotations

import hashlib


def grouped_split(questions: list[dict], dev_count: int, seed: int) -> tuple[list[str], list[str]]:
    """(dev ids, held-out ids), each sorted. Dev has exactly ``dev_count`` questions."""
    groups: dict[str, list[str]] = {}
    kinds: dict[str, str] = {}
    for question in questions:
        base = question["question_id"].removesuffix("_abs")
        groups.setdefault(base, []).append(question["question_id"])
        kinds[base] = question["question_type"]
    for base, members in groups.items():
        if any(member.endswith("_abs") for member in members):
            kinds[base] += " with abstention"
    by_kind: dict[str, list[str]] = {}
    for base in sorted(groups, key=lambda b: hashlib.sha256(f"{seed}:{b}".encode()).hexdigest()):
        by_kind.setdefault(kinds[base], []).append(base)
    sizes = {kind: sum(len(groups[b]) for b in bases) for kind, bases in by_kind.items()}
    taken = dict.fromkeys(by_kind, 0)
    dev: list[str] = []
    while len(dev) < dev_count:
        # The type furthest below its share goes next; a pair that would overshoot waits for a single.
        chosen = None
        for kind in sorted(by_kind, key=lambda k: (taken[k] / sizes[k], k)):
            for base in by_kind[kind]:
                if len(dev) + len(groups[base]) <= dev_count:
                    chosen = (kind, base)
                    break
            if chosen:
                break
        if chosen is None:
            raise ValueError(f"no grouping reaches exactly {dev_count} questions")
        kind, base = chosen
        by_kind[kind].remove(base)
        taken[kind] += len(groups[base])
        dev.extend(groups[base])
    held_out = [q["question_id"] for q in questions if q["question_id"] not in set(dev)]
    return sorted(dev), sorted(held_out)
