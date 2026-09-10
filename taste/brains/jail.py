"""The worktree jail: a fast advisory gate, layered over the real sandbox.

**Read this before trusting it.** This hook is NOT the security boundary. An
earlier version of this file claimed it was, on the strength of a measurement
that ``cwd`` confines nothing -- true, but the conclusion drawn from it was
wrong. ``ClaudeAgentOptions.sandbox`` enables a real OS-level sandbox
(macOS/Linux) and it holds where this cannot: with
``{"enabled": True, "allowUnsandboxedCommands": False}``, every one of
``echo x > ../sibling/f``, ``D=../sibling; echo x > $D/f``,
``python3 -c "open('../sibling/f','w')"`` and ``dd of=../sibling/f`` was
refused with "operation not permitted".

Three of those four defeated the command parser below, and they are the kind of
thing a *confused* brain writes, not only a hostile one. Deciding whether a
shell command is safe by reading it is unwinnable: every blocked spelling has
another spelling.

So the division of labour is:

* the **OS sandbox** is the boundary -- it is what actually stops a write;
* this **hook** is an advisory gate that catches the common cases early, in the
  brain's own language, so the model gets "that is outside your worktree, work
  on your own files" instead of an opaque "operation not permitted", and so
  path-bearing tool calls (Write/Edit/MultiEdit) are refused before they run.

A denial here is therefore a *nudge*, and an escape past it is a bug worth
fixing but not a breach. Never remove the sandbox on the strength of this file.

Two properties it still has to have:

**It must be synchronous and fast.** A ``PreToolUse`` hook runs in the
sub-brain's own event loop and the agent awaits it, so cost is added 1:1 to
every matched call -- measured 42.9 / 228.4 / 1034.2 / 3035.2 ms for hook
sleeps of 0 / 200ms / 1s / 3s. There is no LLM in here and there never can be:
one model verdict costs a median 2.18s, ~100x the budget.

**Its Bash check is best-effort and known-incomplete.** A ``file_path`` check
cannot see ``echo x > ../sibling/file`` at all, so commands are scanned for
redirections and escaping paths -- but variable indirection, paths inside
quoted arguments, and ``of=``-style operands all get through. That is
acceptable only because the sandbox is underneath; it would not be acceptable
alone.

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
# A redirection needs no whitespace before it: `echo x >../sib/f` is valid.
_REDIRECT = re.compile(r"(?:\d*>>?|<)\s*([^\s;|&<>]+)")
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
        except (OSError, ValueError):
            # ValueError, not just OSError: a NUL byte in a path raises
            # "embedded null character" from lstat, and ValueError is not an
            # OSError. Unresolvable means not provably inside.
            return False
        for permitted in (self.root, *self.allow):
            if resolved == permitted or permitted in resolved.parents:
                return True
        return False

    def check(self, tool_name: str, tool_input: dict[str, Any]) -> str | None:
        """The reason to deny this call, or None to let it through."""
        if not isinstance(tool_input, dict):
            return f"{tool_name} was called with a malformed input, so it cannot be checked."
        # The model can ask for its own sandbox to be lifted: the Bash tool
        # takes a `dangerouslyDisableSandbox` parameter. Measured, a model
        # refused by the sandbox re-issued the identical command with the flag
        # set and the write landed. Since the sandbox is the actual boundary
        # and this gate is only advisory, letting the flag through would let a
        # brain dismantle the thing protecting everyone else.
        if tool_input.get("dangerouslyDisableSandbox"):
            return (
                "the sandbox cannot be disabled. Run the command normally, "
                "inside your worktree."
            )
        if tool_name == "Bash":
            return self._check_command(str(tool_input.get("command", "")))
        for field in ("file_path", "path", "notebook_path"):
            value = tool_input.get(field)
            if value is None:
                continue
            # Deny anything that is present but not a string rather than
            # skipping it. An `isinstance(value, str)` guard that falls
            # through to "allow" means a list or a Path sails past unchecked,
            # and a brain has no legitimate reason to send one.
            if not isinstance(value, str):
                return f"{field} must be a path, not {type(value).__name__}."
            if value and not self.contains(value):
                return (
                    f"{value} is outside your worktree. You may only read and write "
                    f"inside {self.root}. Work on your own files."
                )
        edits = tool_input.get("edits") or []
        if not isinstance(edits, list):
            return "edits must be a list."
        for edit in edits:
            if not isinstance(edit, dict):
                return "each edit must be an object."
            value = edit.get("file_path")
            if value is None:
                continue
            if not isinstance(value, str):
                return f"file_path must be a path, not {type(value).__name__}."
            if value and not self.contains(value):
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
                # A flag is not automatically safe: `--git-dir=../repo/.git`
                # and `--work-tree=../sibling` are how git is pointed at
                # another repository, and skipping every "-" word let a brain
                # delete a sibling's branch outright -- the exact worst case
                # this gate exists to make hard. Check the value half.
                _, sep, value = word.partition("=")
                if not sep or not value:
                    continue
                word = value
            # `of=`, `output=`, `if=` and friends hide a path behind an
            # operand name rather than a flag.
            if "=" in word and not word.startswith(("/", "~", "..")):
                _, _, tail = word.partition("=")
                if tail:
                    word = tail
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
        try:
            tool_name = str(input_data.get("tool_name", "")) if isinstance(input_data, dict) else ""
            reason = self.check(tool_name, input_data.get("tool_input") if isinstance(input_data, dict) else None)
        except Exception as exc:  # noqa: BLE001 - see below; this must be total
            # A hook that raises FAILS OPEN: measured against CLI 2.1.257, the
            # exception is logged, the tool runs anyway, and the run reports
            # success. That is the opposite of HookMatcher(timeout=), which
            # fails closed. So an unexpected input must never escape as an
            # exception -- it has to become a denial here.
            tool_name = "a tool"
            reason = f"the guard could not evaluate this call ({type(exc).__name__}), so it was refused."
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
