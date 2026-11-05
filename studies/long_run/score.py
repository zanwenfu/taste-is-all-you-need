"""A long run's score, read from its record alone: which commands a rule applied to, and which obeyed it.

A rule counts from the model call it was told at. Each command is matched by the rule's own patterns, written before
any run, and is counted under the number of compactions that came before it. A memory call is a look, and a find only
if the rule's whole text came back in it, whatever the case.
"""

from __future__ import annotations

import re
from typing import Any

MEMORY_CALLS = ("memory_search", "memory_fetch")


def score_session(actions: list[dict], rules: list[dict]) -> dict[str, dict[str, Any]]:
    scores = {}
    for rule in rules:
        told_at = 0 if rule["say"] == "first" else int(rule["say"])
        trigger, obeys = re.compile(rule["trigger"]), re.compile(rule["obeys"])
        applies = obeyed = 0
        by_compactions: dict[str, list[int]] = {}
        for action in actions:
            if action["kind"] != "bash" or action["call"] < told_at or not trigger.search(action["command"]):
                continue
            ok = bool(obeys.search(action["command"]))
            applies += 1
            obeyed += ok
            cell = by_compactions.setdefault(str(action["compactions"]), [0, 0])
            cell[0] += ok
            cell[1] += 1
        text = rule.get("text", "").casefold()
        looks = [action for action in actions if action["kind"] in MEMORY_CALLS]
        scores[rule["name"]] = {"applies": applies, "obeyed": obeyed, "by_compactions": by_compactions,
                                "looks": len(looks),
                                "finds": sum(bool(text) and text in str(action.get("output", "")).casefold() for action in looks)}
    return scores
