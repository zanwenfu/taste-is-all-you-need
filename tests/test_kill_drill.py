"""A real SIGKILL at any durable step leaves a goal recoverable, and no paid call is made twice.

The windows are found by scripts/kill_drill.py, which walks every step. These
points are spread across a planner call: a stale restart first showed up in the
window around the request's commit, and again just after a paid reply.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "kill_drill", Path(__file__).resolve().parents[1] / "scripts" / "kill_drill.py")
kill_drill = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kill_drill)


@pytest.mark.parametrize("kill_at", [30, 77, 110, 135, 150])
def test_a_killed_planner_call_recovers_and_is_not_paid_twice(kill_at):
    record = kill_drill.one_point("planner", kill_at)
    assert record["reached"], "the child must die by the signal at this point"
    assert kill_drill.verdict(record) == [], record.get("recovery")
    assert record["recovery"]["outcome"] in ("planned", "refused_to_repeat")


def test_a_goal_that_stays_stale_after_a_restart_counts_as_wrong():
    record = {"reached": True, "recovery_exit": 0, "calls": 0, "git": "clean",
              "recovery": {"outcome": "stale_every_time"}}
    assert kill_drill.verdict(record) == ["the goal stayed stuck after the restart"]
