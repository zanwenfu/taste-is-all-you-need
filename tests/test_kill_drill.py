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


@pytest.mark.parametrize("kill_at", [3, 12, 25, 40])
def test_a_killed_delivery_leaves_the_result_whole_or_refused(kill_at):
    record = kill_drill.one_point("delivery", kill_at)
    assert record["reached"], "the child must die by the signal at this point"
    assert kill_drill.verdict(record) == [], record.get("recovery")
    assert record["recovery"]["outcome"] in ("delivered", "refused_to_repeat")


@pytest.mark.parametrize("recovery, problem", [
    ({"outcome": "delivered", "file": "half\n", "kept": kill_drill.KEPT},
     "the delivered file is not what the worker made"),
    ({"outcome": "delivered", "file": kill_drill.DELIVERED, "kept": None},
     "delivery lost a file the target already held"),
])
def test_a_torn_delivery_counts_as_wrong(recovery, problem):
    record = {"reached": True, "recovery_exit": 0, "calls": 0, "git": "clean", "recovery": recovery}
    assert kill_drill.verdict(record) == [problem]


@pytest.mark.parametrize("path, kill_at", [("terminal", 5), ("terminal", 12), ("worker", 7), ("worker", 14)])
def test_a_killed_command_or_worker_call_is_not_repeated(path, kill_at):
    record = kill_drill.one_point(path, kill_at)
    assert record["reached"], "the child must die by the signal at this point"
    assert kill_drill.verdict(record) == [], record.get("recovery")


@pytest.mark.parametrize("kill_at", [85, 104, 120, 140])
def test_a_restart_into_a_moved_world_keeps_what_the_interrupted_call_may_have_cost(kill_at):
    """A real runtime writes to control before it plans again. A call that had gone out must stay on
    record as uncertain: writing "not called, cost zero" over it stopped the goal at its next audit."""
    record = kill_drill.one_point("planner-moved", kill_at)
    assert record["reached"], "the child must die by the signal at this point"
    assert kill_drill.verdict(record) == [], record.get("recovery")
    assert record["recovery"]["outcome"] in ("planned", "refused_to_repeat")


def test_after_a_moved_world_every_paid_call_must_be_on_the_record():
    recovery = {"outcome": "planned", "attempts": [{"status": "rejected"}, {"status": "accepted"}]}
    asked_again = {"reached": True, "recovery_exit": 0, "calls": 2, "git": "clean", "recovery": recovery,
                   "world_moves": True}
    assert kill_drill.verdict(asked_again) == []
    assert kill_drill.verdict({**asked_again, "calls": 3}) == ["a paid call is missing from the record"]
    assert kill_drill.verdict({**asked_again, "world_moves": False}) == ["a paid call was made twice"]


@pytest.mark.parametrize("path", ["git-index", "git-ref", "git-notes"])
def test_a_kill_inside_a_git_command_leaves_nothing_that_stops_the_restart(path):
    record = kill_drill.one_point(path, 1)
    assert record["reached"], "the child must die by the signal at this point"
    assert kill_drill.verdict(record) == [], record.get("recovery")
    assert record["recovery"]["outcome"] == "saved"


@pytest.mark.parametrize("kill_at", range(1, 40, 3))
def test_a_record_caught_between_its_write_and_its_commit_does_not_ride_into_a_later_commit(kill_at):
    record = kill_drill.one_point("record", kill_at)
    assert record["reached"], "the child must die by the signal at this point"
    assert kill_drill.verdict(record) == [], record.get("recovery")


@pytest.mark.parametrize("read_back, wrong", [("the work", False), ("the work\n", False), ("the wor", True), (None, True)])
def test_saved_work_is_compared_with_what_was_written(read_back, wrong):
    record = {"reached": True, "recovery_exit": 0, "calls": 0, "git": "clean",
              "recovery": {"outcome": "saved", "read_back": read_back}}
    assert bool(kill_drill.verdict(record)) is wrong

