"""The task environment's checkpoints and restores, recorded on the control branch.

The coordinator's runtime asks the controller for a checkpoint of the task's
files before the first worker starts and after worker runs end, and for a
restore when a plan names a checkpoint (``rollback_to``). Each is recorded
here, append-only and never retired: the history of the task's files beside
the history of the plans. The planner reads these records from the control
state its request names, shows the model the checkpoints it may return to, and
checks a proposal's rollback against them, so a replayed decision sees the
same list.

A checkpoint has two names. The model calls it ``cp1``, ``cp2``, ... in the
order taken. The controller knows it by an ID made from the runs it follows
(``initial`` before any run), so a coordinator that restarts asks for the same
checkpoint again and reads the controller's record instead of taking another.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

ENVIRONMENT_ROOT = ".taste/environment"
CHECKPOINT_SCHEMA = "taste.brains/EnvironmentCheckpoint/1"
RESTORE_SCHEMA = "taste.brains/EnvironmentRestore/1"
INITIAL = "initial"
ROLLBACK_FIELDS = frozenset({"rollback_to", "rollback_reason"})
MAX_REASON_CHARS = 4000
# Fields of a controller's checkpoint summary that tell the model nothing.
_UNSHOWN = frozenset({"checkpoint_id", "tar_sha256", "taken_at"})

ROLLBACK_RULE = (
    "task_environment_checkpoints lists checkpoints of the task environment's files: one taken "
    "before any worker ran and one after worker runs ended, each with what the files then held "
    "against the image (counts and the first paths). Running processes are not in a checkpoint. "
    "When the evidence shows that later work broke what an earlier state had right, for example a "
    "check that passed after one run fails after a later one, you may return the files to that "
    "earlier checkpoint: set rollback_to to its checkpoint (\"cp2\") and rollback_reason to that "
    "evidence. It is done before this plan's workers start, or before the goal ends when this "
    "proposal is complete or closing. Everything changed since that checkpoint is undone, good "
    "work included; a service started since keeps running. Reports and records of the undone runs "
    "stay, but no longer describe the files. When you roll back, tell the next worker that the "
    "environment was returned to the state after a given run, and why. Otherwise leave rollback_to "
    "null and rollback_reason \"\"."
)


class TaskEnvironment(Protocol):
    """The controller's checkpoints of the task's files, as the coordinator reaches them.

    Each call returns the controller's record: a summary, or ``failed`` with
    the error's class. Asking again with the same ID reads that record.
    """

    def checkpoint(self, checkpoint_id: str, *, timeout_seconds: float) -> Mapping[str, Any]: ...

    def restore(self, operation_id: str, checkpoint_id: str, *,
                timeout_seconds: float) -> Mapping[str, Any]: ...


def _key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


def goal_root(goal_id: str) -> str:
    return f"{ENVIRONMENT_ROOT}/goals/{_key(goal_id)}"


def checkpoint_path(goal_id: str, number: int) -> str:
    return f"{goal_root(goal_id)}/checkpoints/{number:04d}.json"


def restore_path(goal_id: str, plan_id: str) -> str:
    return f"{goal_root(goal_id)}/restores/{_key(plan_id)}.json"


def checkpoint_id(runs_ended) -> str:
    """The controller's ID for the checkpoint after these runs: the same runs, the same ID."""
    runs = sorted(runs_ended)
    return INITIAL if not runs else "after_" + _key("\n".join(runs))[:32]


def restore_operation_id(plan_id: str) -> str:
    """At most one restore per plan."""
    return "undo_" + _key(plan_id)[:32]


def label(number: int) -> str:
    """What the model calls the checkpoint taken ``number``-th."""
    return f"cp{number}"


def _check(record: Any, schema: str, goal_id: str, path: str) -> dict[str, Any]:
    if (not isinstance(record, dict) or record.get("schema") != schema or record.get("goal_id") != goal_id
            or not isinstance(record.get("result"), dict)):
        raise ValueError(f"environment record {path!r} is malformed")
    return record


@dataclass(frozen=True)
class EnvironmentHistory:
    """Every checkpoint (taken or failed) in order, and every restore in order."""

    checkpoints: tuple[dict[str, Any], ...] = ()
    restores: tuple[dict[str, Any], ...] = ()

    def usable(self) -> dict[str, dict[str, Any]]:
        """The checkpoints a plan may roll back to, by the model's name for them."""
        return {record["label"]: record for record in self.checkpoints if not record["result"].get("failed")}

    def covered_runs(self) -> set[str]:
        """Runs whose end some checkpoint already follows."""
        return {run for record in self.checkpoints for run in record["runs_ended"]}

    def for_planner(self) -> dict[str, list[dict[str, Any]]]:
        checkpoints = [
            {"checkpoint": record["label"], "generation": record["generation"],
             "taken": "after these runs ended" if record["runs_ended"] else "before any worker ran",
             "runs_ended": list(record["runs_ended"]),
             "files": {name: value for name, value in record["result"].items() if name not in _UNSHOWN}}
            for record in self.checkpoints if not record["result"].get("failed")]
        rollbacks = []
        for record in self.restores:
            result = record["result"]
            outcome = ("failed: the files may be partly restored" if result.get("failed")
                       else "exact" if result.get("exact") else "done, with differences")
            rollbacks.append({"plan_generation": record["generation"], "to": record["label"],
                              "reason": record["reason"], "outcome": outcome,
                              **({"differences": result["mismatches"]} if result.get("mismatches") else {})})
        return {"checkpoints": checkpoints, "rollbacks": rollbacks}


def read_history(state: Any, goal_id: str) -> EnvironmentHistory:
    """The environment records in ``state`` for ``goal_id``; empty when there are none."""
    root = goal_root(goal_id) + "/"
    checkpoints, restores = [], []
    for path in sorted(item for item in state.files() if item.startswith(root)):
        record = state.record(path)
        if path.startswith(root + "checkpoints/"):
            checkpoints.append(_check(record, CHECKPOINT_SCHEMA, goal_id, path))
        elif path.startswith(root + "restores/"):
            restores.append(_check(record, RESTORE_SCHEMA, goal_id, path))
        else:
            raise ValueError(f"unexpected environment record {path!r}")
    checkpoints.sort(key=lambda item: item["number"])
    if [item["number"] for item in checkpoints] != list(range(1, len(checkpoints) + 1)):
        raise ValueError("environment checkpoints are not numbered 1, 2, ... in order")
    restores.sort(key=lambda item: (item["generation"], item["at"]))
    return EnvironmentHistory(tuple(checkpoints), tuple(restores))


def validate_rollback(target: Any, reason: Any, history: EnvironmentHistory) -> dict[str, str] | None:
    """A proposal's rollback, checked against the checkpoints its request listed.

    Null means none (a reason given anyway is dropped). A named checkpoint must
    be one listed, and needs its evidence.
    """
    if target is None:
        return None
    usable = history.usable()
    if not isinstance(target, str) or target not in usable:
        raise ValueError("rollback_to must be null or one of the listed checkpoints: "
                         + ", ".join(sorted(usable, key=lambda name: int(re.sub(r"\D", "", name) or 0))))
    if not isinstance(reason, str) or not reason.strip() or len(reason) > MAX_REASON_CHARS:
        raise ValueError(f"a rollback needs rollback_reason: its evidence, at most {MAX_REASON_CHARS} characters")
    return {"checkpoint": target, "checkpoint_id": usable[target]["checkpoint_id"], "reason": reason}
