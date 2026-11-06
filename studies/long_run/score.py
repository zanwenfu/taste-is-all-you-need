"""A long run's score, read from its record alone: which commands a rule applied to, and which obeyed it.

A rule counts from the model call it was told at. Each command is matched by the rule's own patterns, written before
any run, and is counted under the number of compactions that came before it. A rule marked ``per_task`` counts each task
once, from the ``from_task``-th on, obeyed if any of its commands or their output matches, under the compactions at the task's
last command. It is also read for whether the agent still knows the fact, whether or not it ran anything: a reply with no command
can hold it too, but a search result merely shown to it is not its knowing. A memory call is a look, and a find only if the rule's whole text, or the fact it watches, came back in it,
whatever the case. ``by_task`` says the same task by task, because arms can end early, and only tasks both arms reached compare.
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
        marks: list[tuple[bool, int]] = []
        by_task: dict[str, dict[str, int]] = {}
        knows: dict[str, bool] = {}
        if rule.get("per_task"):
            order = list(dict.fromkeys(action["task"] for action in actions))
            later = [action for action in actions if action["task"] not in order[:rule.get("from_task", 0)]]
            tasks: dict[str, list] = {}
            for action in later:
                shown = action["kind"] in MEMORY_CALLS
                written = [action["command"], str(action.get("said", "")), *(() if shown else (str(action.get("output", "")),))]
                knows[action["task"]] = knows.get(action["task"], False) or any(obeys.search(text) for text in written)
                if action["kind"] == "bash" and action["call"] >= told_at:
                    cell = tasks.setdefault(action["task"], [False, 0])
                    cell[0] = cell[0] or bool(obeys.search(action["command"])) or bool(obeys.search(str(action.get("output", ""))))
                    cell[1] = action["compactions"]
            marks = [(ok, compactions) for ok, compactions in tasks.values()]
            by_task = {task: {"obeyed": int(ok), "applies": 1} for task, (ok, _) in tasks.items()}
        else:
            for action in actions:
                if action["kind"] == "bash" and action["call"] >= told_at and trigger.search(action["command"]):
                    ok = bool(obeys.search(action["command"]))
                    marks.append((ok, action["compactions"]))
                    cell = by_task.setdefault(action.get("task", ""), {"obeyed": 0, "applies": 0})
                    cell["obeyed"] += ok
                    cell["applies"] += 1
        for ok, compactions in marks:
            applies += 1
            obeyed += ok
            cell = by_compactions.setdefault(str(compactions), [0, 0])
            cell[0] += ok
            cell[1] += 1
        text = (rule.get("watch") or rule.get("text", "")).casefold()
        looks = [action for action in actions if action["kind"] in MEMORY_CALLS]
        scores[rule["name"]] = {"applies": applies, "obeyed": obeyed, "by_compactions": by_compactions, "by_task": by_task,
                                **({"knows": sum(knows.values()), "knows_of": len(knows)} if rule.get("per_task") else {}),
                                "looks": len(looks),
                                "finds": sum(bool(text) and text in str(action.get("output", "")).casefold() for action in looks)}
    return scores
