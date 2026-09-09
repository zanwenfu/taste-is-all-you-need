"""The worktree jail: the only thing keeping a sub-brain inside its own work.

There is no filesystem sandbox. Measured, from one worktree and under both
``acceptEdits`` and ``bypassPermissions``, a brain wrote into the shared
``.git`` directory and into a sibling brain's worktree -- reporting success
every time. Corrupting the shared ``.git`` breaks every sibling worktree at
once. ``cwd`` is a working directory, not a boundary; this hook is the
boundary.

Two properties it has to have, both learned the hard way:

**It must be synchronous and fast.** A ``PreToolUse`` hook runs in the
sub-brain's own event loop and the agent awaits it, so cost is added 1:1 to
every matched call -- measured 42.9 / 228.4 / 1034.2 / 3035.2 ms for hook
sleeps of 0 / 200ms / 1s / 3s. There is no LLM in here and there never can be:
one model verdict costs a median 2.18s, ~100x the budget.

**It must cover Bash, not just file paths.** A ``file_path`` check is trivially
bypassed by ``echo x > ../sibling/file``, so shell commands are checked for
redirections and for paths that leave the jail.

A hook that returns ``{"async_": True}`` alongside a deny has its decision
**silently discarded** -- measured, all tools executed, nothing logged. Every
decision here is returned synchronously.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Any

__all__ = ["WorktreeJail", "MUTATING_TOOLS", "JailDenial"]

MUTATING_TOOLS = ("Bash", "Write", "Edit", "MultiEdit", "NotebookEdit")
"""Tools that can change the world, and so must pass the jail.

Read-only tools are deliberately ungated: every gated call pays the hook's
latency, and a read cannot escape anything.
"""

# Redirections and the shell metacharacters that can start a new command.
_REDIRECT = re.compile(r"(?:^|\s)(?:\d*>>?|<)\s*(\S+)")
_PATHLIKE = re.compile(r"(?:^|[\s=:'\"])((?:\.\.|~|/)[^\s'\";|&)]*)")


class JailDenial(Exception):
    """Raised by :meth:`WorktreeJail.check` when a call leaves the jail."""


class WorktreeJail:
    """Confines a sub-brain to its own worktree.

    ``root`` is resolved once with ``realpath``: on macOS the CLI rewrites
    ``cwd`` to its realpath, so a brain given ``/var/folders/...`` reports
    writes at ``/private/var/folders/...`` and a naive string prefix check
    rejects every legitimate write.
    """

    def __init__(self, root: Path, *, allow: tuple[str, ...] = ()) -> None:
        self.root = Path(os.path.realpath(root))
        self.allow = tuple(Path(os.path.realpath(p)) for p in allow)
        self.denials: list[tuple[str, str]] = []

    # ------------------------------------------------------------------ core

    def contains(self, path: str | Path) -> bool:
        """True if ``path`` resolves inside the jail.

        Resolution walks up to the nearest existing ancestor, so a not-yet
        created file is judged by where it *would* land -- a check on the
        literal string would miss ``newdir/../../escape``.
        """
        text = str(path)
        # Expand ~ before anything else. Path("~/x") is not absolute, so it
        # would otherwise be joined to the worktree and realpath would leave
        # the tilde literal -- while any tool that expands ~ itself writes to
        # the real home directory. That is how ~/.ssh/id_rsa walked straight
        # through an earlier version of this check.
        if text.startswith("~"):
            text = os.path.expanduser(text)
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        # realpath resolves symlinks even when the leaf does not exist yet, so
        # a not-yet-created file is judged by where it *would* land.
        try:
            resolved = Path(os.path.realpath(candidate))
        except OSError:  # pragma: no cover - defensive
            return False
        for permitted in (self.root, *self.allow):
            if resolved == permitted or permitted in resolved.parents:
                return True
        return False

    def check(self, tool_name: str, tool_input: dict[str, Any]) -> str | None:
        """The reason to deny this call, or None to let it through."""
        if tool_name == "Bash":
            return self._check_command(str(tool_input.get("command", "")))
        for field in ("file_path", "path", "notebook_path"):
            value = tool_input.get(field)
            if isinstance(value, str) and value and not self.contains(value):
                return (
                    f"{value} is outside your worktree. You may only read and write "
                    f"inside {self.root}. Work on your own files."
                )
        for edit in tool_input.get("edits", []) or []:
            value = edit.get("file_path") if isinstance(edit, dict) else None
            if isinstance(value, str) and value and not self.contains(value):
                return f"{value} is outside your worktree."
        return None

    def _check_command(self, command: str) -> str | None:
        """Deny a shell command that writes outside the jail.

        A ``file_path`` check cannot see ``echo x > ../sibling/file``, so the
        command string is inspected for redirection targets and for paths that
        leave the jail. This is a coarse check on purpose: it runs on the hot
        path, and the cost of a false deny is one explained retry, while the
        cost of a false allow is a corrupted sibling.
        """
        if not command.strip():
            return None
        for target in _REDIRECT.findall(command):
            cleaned = target.strip("'\"")
            if cleaned and not cleaned.startswith("/dev/") and not self.contains(cleaned):
                return (
                    f"that command redirects into {cleaned}, which is outside your "
                    f"worktree. Write only inside {self.root}."
                )
        try:
            words = shlex.split(command)
        except ValueError:
            words = command.split()
        for word in words:
            if word.startswith("-"):
                continue
            if word.startswith(("/", "~", "..")) or "/.." in word:
                if word.startswith("/dev/") or word.startswith("/tmp/"):
                    continue
                if not self.contains(word):
                    return (
                        f"that command touches {word}, which is outside your worktree. "
                        f"Work only inside {self.root}."
                    )
        return None

    # ------------------------------------------------------------------ hook

    async def hook(
        self, input_data: dict[str, Any], tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        """A ``PreToolUse`` hook. Synchronous decisions only, no LLM, no I/O."""
        tool_name = str(input_data.get("tool_name", ""))
        reason = self.check(tool_name, input_data.get("tool_input") or {})
        if reason is None:
            return {}
        self.denials.append((tool_name, reason))
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                # The reason reaches the model verbatim as an is_error result,
                # so it reads as guidance and the model adapts instead of
                # retrying the same call.
                "permissionDecisionReason": reason,
            }
        }
