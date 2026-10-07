"""Rewind points chosen by rules over a failed run's steps, without reruns.

Each rule names the step r to branch from (the state after step r; 0 is the
start) or abstains. The rules are the trajectory reader's
(``taste.agents.trajectory_reader``):

- ``visible_tests``: the last step after which the visible tests passed, read
  from what the agent's test commands printed (``last_passing_tests``);
- ``before_large_edit``: the step before the last edit of at least
  ``LARGE_EDIT_LINES`` lines (``before_last_large_edit``);
- ``start``: step 0, which is a retry.

``rewind_points`` gives every rule's choice and the combined rule's: the first
of them, in that order, that does not abstain. The map scores each against the
measured point of no return; the recovery driver rewinds to the combined
rule's choice.
"""

from __future__ import annotations

from taste.agents.trajectory_reader import before_last_large_edit, edit_lines, last_passing_tests

LARGE_EDIT_LINES = 20
RULES = ("visible_tests", "before_large_edit", "start")


def rewind_points(steps, large=LARGE_EDIT_LINES):
    """Each rule's rewind point, within [0, T-1], and the combined rule's (with the rule that chose it)."""
    if not steps:
        return {"visible_tests": None, "before_large_edit": None, "start": 0, "rules": 0, "rule": "start"}
    last = len(steps) - 1
    chosen = {"visible_tests": last_passing_tests(steps), "before_large_edit": before_last_large_edit(steps, min_lines=large),
              "start": 0}
    chosen = {name: None if step is None else max(0, min(step, last)) for name, step in chosen.items()}
    rule = next(name for name in RULES if chosen[name] is not None)
    return {**chosen, "rules": chosen[rule], "rule": rule}


def features(steps):
    """What a harness sees of a failed run without rerunning it."""
    return {"run_length": len(steps),
            "visible_tests_passed": bool(steps) and last_passing_tests(steps) is not None,
            "edits": sum(1 for step in steps if edit_lines(step.command) > 0)}
