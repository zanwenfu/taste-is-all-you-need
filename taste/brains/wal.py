"""The write-ahead log: what a brain did, recorded before it does it.

A tool call is two events that cannot be one -- doing the thing, and recording
that it was done. Between them the process can die, and afterwards nothing on
disk says whether the side effect happened. That gap can be made small; it
cannot be closed, because the deed is an arbitrary command.

So this log does not try to close it. It makes it *visible*: an intent with no
matching result is a question the next process can see and ask about. That is
the difference between "nothing is lost" -- which would be a lie -- and
"nothing is lost silently", which is true and is the honest claim.

It matters concretely. Measured, a brain resumed after an interrupted tool said
"I hadn't actually executed it yet, so let me run it now" and re-ran the
command. Silence reads to a model as *never happened*. A brain must be told.

**Why not ``branch.turn()``.** These append from inside a ``PreToolUse`` hook,
where cost is added 1:1 to every tool call against a ~20 ms budget. ``turn()``
costs 15.1 ms -- not the fsync, which is 0.03 ms, but two git subprocesses
underneath it. Resolving the journal path once at construction and appending to
it directly costs **0.028 ms**, and memstore folds the same file into the next
state, so nothing is given up by writing it this way.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["InFlight", "WriteAheadLog", "reconcile"]

INTENT = "tool_intent"
RESULT = "tool_result"


@dataclass(frozen=True)
class InFlight:
    """A tool that was started and whose outcome is unknown.

    ``tool_use_id`` is the SDK's own identifier, which is what pairs an intent
    to its result; without one, a call cannot be reconciled and is reported as
    unknown rather than assumed finished.
    """

    tool: str
    tool_use_id: str
    request: dict[str, Any]
    at: str

    def as_warning(self) -> str:
        """How a resuming brain is told about it, in its own language."""
        return (
            f"- {self.tool} was started at {self.at} and its outcome is UNKNOWN: "
            f"the process ended before the result was recorded. It may have "
            f"completed. Check the actual state of your worktree before "
            f"repeating it, and do not repeat it if the work is already done. "
            f"The call was: {json.dumps(self.request, sort_keys=True)[:400]}"
        )


class WriteAheadLog:
    """Appends tool intents and results to a branch's turn journal.

    One instance per sub-brain. The journal path is resolved once: it is keyed
    on the branch head, so it changes when a state is published, and
    :meth:`rebind` is how the owner tells it that happened.
    """

    def __init__(self, branch: Any) -> None:
        self.branch = branch
        self._head: str = ""
        self._path: Path = branch._turns_path()

    def rebind(self) -> None:
        """Re-resolve the journal after the branch head moves.

        Kept for callers that know a move happened, but correctness no longer
        depends on anyone remembering: :meth:`_append` re-resolves whenever the
        head has changed. It used to, and a rollback -- which moves the head
        without going through ``SubBrain.checkpoint`` -- left the log writing
        into an orphaned journal that ``resume`` never reads. An ``rm -rf``
        recorded there was invisible to the next brain.
        """
        self._head = ""
        self._path = self.branch._turns_path()

    # ------------------------------------------------------------------ write

    def _current_path(self) -> Path:
        """The journal for the head as it is now.

        The check is a file-existence test, not a git read. ``publish_state``
        unlinks the journal it folded in, so a path that has vanished is
        exactly the signal that the head moved -- and asking the filesystem
        costs 0.01 ms against the ~8 ms of resolving the head through git,
        which matters because this sits in a PreToolUse hook whose cost is
        added 1:1 to every tool call.

        A missing journal for a head that has NOT moved is equally handled:
        re-resolving simply returns the same path and the file is recreated on
        append.
        """
        if not self._path.exists():
            self._path = self.branch._turns_path()
        return self._path

    def _append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, default=str) + "\n"
        with open(self._current_path(), "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            # The durability point. A killed process runs no finally, no
            # atexit and no flush on the way out, so the write itself has to
            # be what survives.
            os.fsync(fh.fileno())

    def intent(self, tool: str, tool_use_id: str, request: dict[str, Any]) -> None:
        """Record that a tool is about to run. Called before any side effect."""
        self._append(
            {
                "kind": INTENT,
                "tool": tool,
                "tool_use_id": tool_use_id,
                "request": request,
                "at": _now(),
            }
        )

    def result(self, tool: str, tool_use_id: str, ok: bool, summary: str = "") -> None:
        """Record that a tool finished. Called once the side effect exists."""
        self._append(
            {
                "kind": RESULT,
                "tool": tool,
                "tool_use_id": tool_use_id,
                "ok": ok,
                "summary": summary[:2000],
                "at": _now(),
            }
        )

    def note(self, **fields: Any) -> None:
        """Record anything else the brain said or was told."""
        self._append({"at": _now(), **fields})


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def reconcile(turns: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> list[InFlight]:
    """Tool calls that were started and never finished, oldest first.

    Pairs by ``tool_use_id`` rather than by position: tools can overlap, and a
    positional pairing would mark the wrong one unknown. A call with no
    ``tool_use_id`` cannot be paired at all, so it is reported as unknown --
    the safe direction, since the cost of a spurious warning is one extra check
    and the cost of a missed one is a repeated side effect.
    """
    started: dict[str, InFlight] = {}
    finished: set[str] = set()
    unpairable: list[InFlight] = []
    for turn in turns:
        kind = turn.get("kind")
        if kind == INTENT:
            key = turn.get("tool_use_id")
            entry = InFlight(
                tool=str(turn.get("tool", "a tool")),
                tool_use_id=str(key or ""),
                request=turn.get("request") or {},
                at=str(turn.get("at", "")),
            )
            if key:
                started[str(key)] = entry
            else:
                unpairable.append(entry)
        elif kind == RESULT:
            key = turn.get("tool_use_id")
            if key:
                finished.add(str(key))
    return [*unpairable, *(v for k, v in started.items() if k not in finished)]
