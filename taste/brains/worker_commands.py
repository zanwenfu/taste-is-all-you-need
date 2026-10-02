"""A mechanical record of the terminal commands a worker ran.

A worker states what it did in its report. A worker that makes no claim, or
that is killed and writes no report at all, leaves only its recorded turns.
These lines are read from those: each command, its exit and the last line it
printed. They are a record, not a judgement and not a verification.
"""

from __future__ import annotations

from typing import Any

COMMAND_TOOL = "terminal_exec"


def clip(text: Any, size: int) -> str:
    """One line of at most ``size`` characters, marked where it was cut."""
    text = " ".join(str(text).split())
    return text if len(text) <= size else text[:size] + " [cut]"


def ran(command: Any, output: Any) -> str:
    """One finished command: what was run, its first result line and its last."""
    rows = [row for row in str(output).splitlines() if row.strip()]
    line = f"ran: {clip(command, 300)} -> {clip(rows[0] if rows else 'no result text', 120)}"
    return line + (f"; last line printed: {clip(rows[-1], 200)}" if len(rows) > 1 else "")


def unfinished(command: Any) -> str:
    return f"started, result not recorded: {clip(command, 300)}"


def bounded(lines: list[str], *, last: int = 40, limit: int = 12_000) -> list[str]:
    """The most recent lines that fit: the end of a run says what state it left."""
    lines = lines[-last:]
    while lines and sum(map(len, lines)) > limit:
        lines.pop(0)
    return lines


def recorded_commands(turns: Any, *, run_id: str, last: int = 40, limit: int = 12_000) -> list[str]:
    """The commands one run's recorded turns show, oldest first; [] if there are none.

    Read from memory, not from the worker: its branch keeps every turn it
    recorded, so this works for a worker that was killed. A command is listed
    once its intent was recorded, which is before it runs, and as unfinished
    if no result followed.
    """
    started: dict[Any, Any] = {}
    lines: list[str] = []
    ours = False
    for turn in turns:
        kind = turn.get("kind")
        if kind == "responses_binding":
            binding = turn.get("binding")
            ours = isinstance(binding, dict) and binding.get("run_id") == run_id
            continue
        call = turn.get("call")
        if not ours or not isinstance(call, dict) or call.get("name") != COMMAND_TOOL:
            continue
        effect = turn.get("effect_id")
        if kind == "responses_tool_intent":
            started[effect] = (call.get("arguments") or {}).get("command", "")
        elif kind == "responses_tool_result" and effect in started:
            lines.append(ran(started.pop(effect), (turn.get("result") or {}).get("content", "")))
    lines.extend(unfinished(command) for command in started.values())
    return bounded(lines, last=last, limit=limit)


UNREPORTED = ("No report was written. Mechanical record of this worker's terminal commands, "
              "read from its recorded turns, oldest first:")


def unreported_commands(store: Any, run: Any) -> tuple[str, ...]:
    """What a run that left no report is recorded as having run; () when that is nothing.

    Read from the worker's branch without taking its lease: the turns it had
    checkpointed and the ones still in its journal. A worker killed after its
    grace period recorded each command before running it, so the command it
    was killed in is here too, as started with no result.
    """
    view = store.view(run.assignment.worker)
    if not view.exists():
        return ()
    lines = recorded_commands([*view.head.transcript.turns, *view.pending_turns()], run_id=run.run_id)
    return (UNREPORTED, *lines) if lines else ()

