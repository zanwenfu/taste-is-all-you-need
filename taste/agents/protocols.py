"""How each hosted agent hands in its work, for the code that reads and replays its record.

Branch replay (``taste.brains.branch_replay``) and the recovery study's record
reader (``taste.recovery_study.records``) need three facts about an agent's
submission protocol, without importing the agent itself:

- which of a step's commands ended the run as its submission, so that a branch
  can show the agent a rejection there instead and let it work on;
- which command's output a note goes after, so that the agent reads it with
  the result of the action its step took;
- the submission's text in that command's output.

A step is one model reply and every command the agent ran before its next
call. mini-swe-agent runs one command per action and submits by printing a
sentinel as its first line. SWE-agent runs its action, then its tool bundles'
state command, then reads the state file (and, after a submission, the patch
file) through its runtime; it submits when the action's output holds its
marker. A record of an agent with no name is mini-swe-agent's: the records
from before a second agent was hosted are all of it.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from typing import Any

MINI_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SWE_AGENT_MARKER = "<<SWE_AGENT_SUBMISSION>>"
# Taste's SWE-ReX runtime shows what it runs for SWE-agent itself (starting
# the shell session, reading, writing and uploading files, a command outside
# the session) under this prefix; the agent's own commands are shown as it
# wrote them.
SWE_AGENT_RUNTIME = "# swe-rex: "
# The state commands of the configuration Taste hosts (Scale's ``tool_use``):
# run after every action, never chosen by the model.
SWE_AGENT_STATE_COMMANDS = frozenset({"_state_anthropic"})
# Its editor tool, and the subcommands that write the file they name.
SWE_AGENT_EDITOR = "str_replace_editor"
EDITOR_WRITES = frozenset({"create", "str_replace", "insert", "undo_edit"})
_EDITOR_VALUES = {"--file_text": 1, "--old_str": 1, "--new_str": 1, "--insert_line": 1, "--view_range": 2}


def editor_call(command):
    """SWE-agent's editor call as (subcommand, path, {option: value}); None for any other command.

    Its function-calling parser writes a call as ``str_replace_editor <command> <path>
    [--file_text v] [--old_str v] [--new_str v] [--insert_line n] [--view_range a b]``, each
    string value shell-quoted (``sweagent.tools.parsing``, ``_should_quote``).
    """
    text = str(command).strip()
    if not text.startswith(SWE_AGENT_EDITOR + " "):
        return None
    try:
        words = shlex.split(text)
    except ValueError:
        return None
    if len(words) < 3 or words[0] != SWE_AGENT_EDITOR:
        return None
    options, index = {}, 3
    while index < len(words):
        count = _EDITOR_VALUES.get(words[index])
        if count is None or index + count >= len(words):
            return None
        options[words[index][2:]] = " ".join(words[index + 1:index + 1 + count])
        index += 1 + count
    return words[1], words[2], options


class MiniSweAgentProtocol:
    name = "mini-swe-agent"

    @staticmethod
    def internal(command: str) -> bool:
        """Every command mini-swe-agent runs is an action its model chose."""
        return False

    @staticmethod
    def submitted(output: str, returncode: int) -> bool:
        lines = str(output).lstrip().splitlines()
        return bool(lines) and lines[0].strip() == MINI_SENTINEL and returncode == 0

    def submission_index(self, runs: Sequence[Any]) -> int | None:
        """The last command, when its result is the sentinel and a clean exit."""
        if runs and self.submitted(runs[-1].output, runs[-1].returncode):
            return len(runs) - 1
        return None

    def note_index(self, runs: Sequence[Any]) -> int | None:
        return len(runs) - 1 if runs else None

    def submission_text(self, output: str) -> str:
        lines = str(output).lstrip().splitlines(keepends=True)
        return "".join(lines[1:]) if lines and lines[0].strip() == MINI_SENTINEL else ""


class SweAgentProtocol:
    name = "swe-agent"

    @staticmethod
    def internal(command: str) -> bool:
        """Whether SWE-agent ran this for itself rather than as the action its model chose."""
        return str(command).startswith(SWE_AGENT_RUNTIME) or str(command).strip() in SWE_AGENT_STATE_COMMANDS

    def action_index(self, runs: Sequence[Any]) -> int | None:
        return next((index for index, run in enumerate(runs) if not self.internal(run.command)), None)

    def submission_index(self, runs: Sequence[Any]) -> int | None:
        """The action, when its output holds the marker (``ToolHandler.check_for_submission_cmd``)."""
        index = self.action_index(runs)
        return index if index is not None and SWE_AGENT_MARKER in str(runs[index].output) else None

    def note_index(self, runs: Sequence[Any]) -> int | None:
        """The action: its output is the observation the model reads; the state commands' are not shown."""
        return self.action_index(runs)

    def submission_text(self, output: str) -> str:
        """The patch the submit tool printed between its two markers."""
        parts = str(output).split(SWE_AGENT_MARKER)
        if len(parts) < 3:
            return ""
        return parts[1][1:] if parts[1].startswith("\n") else parts[1]


PROTOCOLS = {protocol.name: protocol for protocol in (MiniSweAgentProtocol(), SweAgentProtocol())}


def protocol(agent_name: str | None):
    """The submission protocol of the agent a record names; mini-swe-agent's for a record that names none."""
    return PROTOCOLS.get(agent_name or "", PROTOCOLS["mini-swe-agent"])
