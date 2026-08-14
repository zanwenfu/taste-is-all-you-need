"""Memory-protection bits: a veto on tool calls before they run.

The worker executes arbitrary shell in a real repository. Three things must
not happen, and prompting is not a mechanism for preventing any of them:

* **The substrate must not be writable by the code it stores.** ``run_shell``
  advertises itself for git operations while the worker prompt merely *asks*
  the model not to touch git. A worker that runs ``git reset`` or
  ``git checkout`` mid-step corrupts the checkpoint semantics under
  measurement, silently, for that trial.
* **Verification must not be editable by the thing being verified.** A worker
  that weakens a failing test earns a PASS, a commit, and a green result. The
  test file edit is detected and reported rather than blocked, because
  sometimes editing tests is the legitimate task — but it is never invisible.
* **A step must not spend without bound.** A pathological loop can burn the
  budget of a whole sweep overnight.

**On the strength of this boundary.** A regex denylist over a shell string is
a speed bump, not a sandbox: ``sh -c``, ``eval``, a variable, or a here-doc
defeats it. That is stated plainly rather than papered over. The real
boundary is a container with no network and no ``.env`` in scope. What this
module buys is that the *common accidental* case is caught, the intent is
documented in executable form, and every attempt is recorded.

Everything here fails open. A guard that raises disables itself and the run
continues unguarded — a bug in a safety check must not become a new way for
runs to die.
"""

from __future__ import annotations

import contextlib
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from taste.cores import ALLOW, Interrupt, ToolDecision

# Commands that mutate the memory substrate. The kernel owns git; a worker
# that moves refs invalidates the checkpoint/rollback semantics for its trial.
_GIT_MUTATORS = (
    "commit", "reset", "checkout", "switch", "merge", "rebase", "cherry-pick",
    "revert", "branch", "tag", "push", "pull", "fetch", "clone", "stash",
    "update-ref", "worktree", "gc", "prune", "reflog", "filter-branch", "am",
    "apply", "restore", "clean", "notes", "submodule", "remote", "config",
)

# Read-only git that is genuinely useful to a worker and harmless to allow.
_GIT_READERS = (
    "status", "log", "diff", "show", "ls-files", "blame", "grep",
    "rev-parse", "describe", "cat-file", "for-each-ref", "shortlog",
)

# Global flags that consume the following token, so the subcommand is not the
# next word: `git -C . reset` must resolve to "reset", not "-C" or ".".
_GIT_FLAGS_WITH_ARG = frozenset(
    {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path", "--super-prefix"}
)


# Shell separators that begin a new command. A newline is one of them, which
# a start-of-string anchor misses.
_SEGMENTS = re.compile(r"\|\||&&|[;&|\n]|\$\(|`")


def git_subcommands(command: str) -> list[str]:
    """Every git subcommand invoked in a shell string.

    Two properties, and both were wrong before. ``git`` must be in *command*
    position — otherwise ``echo use git commit`` reads as a git invocation —
    and global flags must be consumed with their arguments, so
    ``git -C . reset`` resolves to ``reset`` rather than to ``-C`` or ``.``.
    Newline counts as a separator; the previous pattern anchored only on the
    start of the string, so ``pytest\\ngit reset`` slipped through entirely.
    """
    found: list[str] = []
    for segment in _SEGMENTS.split(command):
        try:
            tokens = shlex.split(segment, comments=False, posix=True)
        except ValueError:
            tokens = segment.split()
        if not tokens:
            continue

        index = 0
        # Leading VAR=value assignments precede the command itself.
        while index < len(tokens) and "=" in tokens[index] and not tokens[index].startswith("-"):
            index += 1
        # Strip any path prefix, so /usr/bin/git is still git.
        if index >= len(tokens) or tokens[index].rsplit("/", 1)[-1] != "git":
            continue

        cursor = index + 1
        while cursor < len(tokens):
            token = tokens[cursor]
            if token in _GIT_FLAGS_WITH_ARG:
                cursor += 2
                continue
            if token.startswith("-"):
                cursor += 1
                continue
            found.append(token.lower())
            break
    return found

# Paths a worker has no business writing to, relative to the workspace.
_PROTECTED_PREFIXES = (".git", ".taste")

_MAX_SHELL_TIMEOUT = 300


@dataclass
class GuardConfig:
    """Which protections are active. All off by default: with every flag
    false the hook is never installed and the loop is unguarded, exactly as
    it was before this module existed."""

    enabled: bool = False
    deny_git_mutations: bool = True
    protect_substrate_paths: bool = True
    flag_test_mutations: bool = True
    clamp_shell_timeout: bool = True
    max_shell_timeout: int = _MAX_SHELL_TIMEOUT
    step_budget_usd: float | None = None
    max_turns_hint: int | None = None


@dataclass
class GuardReport:
    """What the guard saw during one step."""

    vetoes: list[tuple[str, str]] = field(default_factory=list)
    rewrites: list[tuple[str, str]] = field(default_factory=list)
    test_files_touched: list[str] = field(default_factory=list)
    interrupt: Interrupt | None = None
    faults: int = 0
    disabled: bool = False


class Guardrails:
    """A :class:`~taste.cores.TurnHook` that vetoes before, not after.

    Parameters
    ----------
    workspace:
        The only directory this worker may write to.
    cost_reader:
        Returns dollars spent so far, for the budget ceiling. Called between
        turns, never inside one.
    on_event:
        Optional sink for ``guard.*`` events.
    """

    #: Consecutive internal faults before the guard switches itself off.
    FAULT_LIMIT = 3

    def __init__(
        self,
        *,
        workspace: Path,
        config: GuardConfig,
        cost_reader: Any = None,
        on_event: Any = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.config = config
        self.cost_reader = cost_reader
        self.on_event = on_event or (lambda *a, **k: None)
        self.report = GuardReport()
        self._start_usd = self._read_cost()

    # ------------------------------------------------------------ hook API

    def before_turn(self, turn: int) -> Interrupt | None:
        if self.report.disabled:
            return None
        budget = self.config.step_budget_usd
        if budget is None:
            return None
        spent = self._read_cost() - self._start_usd
        if spent >= budget:
            interrupt = Interrupt(
                kind="budget_ceiling",
                detail=f"step spent ${spent:.4f} of its ${budget:.4f} ceiling",
                turn=turn,
                failure_kind="budget",
            )
            self.report.interrupt = interrupt
            self._emit("guard.interrupt", kind=interrupt.kind, turn=turn, detail=interrupt.detail)
            return interrupt
        return None

    def before_tool(self, turn: int, name: str, payload: dict[str, Any]) -> ToolDecision:
        if self.report.disabled:
            return ALLOW
        try:
            return self._inspect(turn, name, payload)
        except Exception as exc:
            self._fault(exc)
            return ALLOW

    def after_tool(
        self,
        turn: int,
        name: str,
        payload: dict[str, Any],
        output: str,
        elapsed_s: float,
        decision: ToolDecision,
    ) -> None:
        return None

    def after_turn(self, turn: int, message: Any, stop_reason: str) -> None:
        return None

    # ------------------------------------------------------------ checks

    def _inspect(self, turn: int, name: str, payload: dict[str, Any]) -> ToolDecision:
        if name in ("write_file", "read_file"):
            return self._inspect_path(turn, name, payload)
        if name == "run_shell":
            return self._inspect_shell(turn, payload)
        return ALLOW

    def _inspect_path(self, turn: int, name: str, payload: dict[str, Any]) -> ToolDecision:
        raw = str(payload.get("path", ""))
        if not raw:
            return ALLOW
        target = (self.workspace / raw).resolve()

        if not target.is_relative_to(self.workspace):
            return self._veto(
                turn,
                name,
                "path escapes the workspace",
                f"BLOCKED: {raw} is outside this step's workspace. Work only inside it.",
            )

        parts = target.relative_to(self.workspace).parts
        if self.config.protect_substrate_paths and parts and parts[0] in _PROTECTED_PREFIXES:
            return self._veto(
                turn,
                name,
                f"{parts[0]} belongs to the kernel",
                f"BLOCKED: {raw} is harness state, not project code. The kernel owns it.",
            )

        # Test edits are recorded, not blocked: sometimes editing tests IS the
        # step. What matters is that it can never happen unnoticed.
        if (
            self.config.flag_test_mutations
            and name == "write_file"
            and _looks_like_test(target.name)
        ):
            path_str = "/".join(parts)
            if path_str not in self.report.test_files_touched:
                self.report.test_files_touched.append(path_str)
                self._emit("guard.test_mutation", turn=turn, path=path_str)
        return ALLOW

    def _inspect_shell(self, turn: int, payload: dict[str, Any]) -> ToolDecision:
        command = str(payload.get("command", ""))

        if self.config.deny_git_mutations:
            for verb_lower in git_subcommands(command):
                if verb_lower in _GIT_READERS:
                    continue
                if verb_lower in _GIT_MUTATORS:
                    return self._veto(
                        turn,
                        "run_shell",
                        f"git {verb_lower} mutates the memory substrate",
                        (
                            f"BLOCKED: `git {verb_lower}` is not available to workers. "
                            "The kernel owns commits, branches and resets — just edit "
                            "files and the kernel will checkpoint your work."
                        ),
                    )

        # Clamping is a rewrite, not a veto: the command is legitimate, only
        # its timeout is unreasonable.
        if self.config.clamp_shell_timeout:
            timeout = payload.get("timeout")
            if isinstance(timeout, int | float) and timeout > self.config.max_shell_timeout:
                clamped = dict(payload)
                clamped["timeout"] = self.config.max_shell_timeout
                self.report.rewrites.append(("run_shell", f"timeout {timeout}->{clamped['timeout']}"))
                self._emit(
                    "guard.rewrite",
                    turn=turn,
                    tool="run_shell",
                    detail=f"timeout clamped to {clamped['timeout']}s",
                )
                return ToolDecision(action="rewrite", payload=clamped, reason="timeout clamped")

        return ALLOW

    # ------------------------------------------------------------ helpers

    def _veto(self, turn: int, tool: str, reason: str, message: str) -> ToolDecision:
        self.report.vetoes.append((tool, reason))
        self._emit("guard.veto", turn=turn, tool=tool, reason=reason)
        return ToolDecision(action="veto", message=message, reason=reason)

    def _fault(self, exc: Exception) -> None:
        """A guard that keeps failing switches itself off rather than
        degrading every tool call it touches."""
        self.report.faults += 1
        self._emit("guard.error", error=type(exc).__name__, detail=str(exc)[:200])
        if self.report.faults >= self.FAULT_LIMIT:
            self.report.disabled = True
            self._emit("guard.disabled", faults=self.report.faults)

    def _read_cost(self) -> float:
        if self.cost_reader is None:
            return 0.0
        try:
            return float(self.cost_reader())
        except Exception:
            return 0.0

    def _emit(self, kind: str, /, **payload: Any) -> None:
        # Positional-only: a payload key named "kind" is natural to want and
        # would otherwise collide with this parameter, raising a TypeError
        # that the fail-open wrapper silently swallows.
        with contextlib.suppress(Exception):
            self.on_event(kind, **payload)


def _looks_like_test(filename: str) -> bool:
    return filename.startswith("test_") or filename.endswith(("_test.py", "_test.js", ".test.ts"))
