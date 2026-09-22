"""One sub-brain: one process, one worktree, one branch, one client.

A sub-brain is given a :class:`~taste.brains.contract.Contract` -- an identity,
a task, the inputs it can expect, the outputs it owes, and the criteria it will
be judged against -- and works until those criteria are met or its budget runs
out. It never receives a bare goal, because its monitor is built from the same
contract and a criterion that exists only in the monitor's prompt is one the
worker was never told.

Everything it does lands in memstore: its reasoning through the session-store
adapter, its tool calls through the write-ahead log, its artifacts in the
worktree, all folded into a state at each checkpoint. Kill it at any point and
the next process resumes from that -- knowing what it intended, what it had
recorded, and, for any tool that was in flight, that the outcome is unknown.

The configuration here is not stylistic. Each line of it is something that was
measured going wrong:

* ``ClaudeSDKClient``, never ``query()`` -- ``interrupt()`` is streaming-only,
  so a brain hosted the other way is silently unstoppable by its monitor.
* ``sandbox`` enabled with ``allowUnsandboxedCommands=False`` -- this is the
  real boundary. ``cwd`` confines nothing; the jail hook over it is advisory.
* ``allowed_tools`` left empty -- naming a tool there silently shadows the
  permission callback for it, so a gate can appear wired up and never fire.
* the config directory outside the worktree -- inside, ``git status`` reports
  it and a checkpoint commits the brain's own plumbing as if it were work.
* ``ANTHROPIC_AUTH_TOKEN``, not ``ANTHROPIC_API_KEY`` -- a per-brain config
  directory has no approved-key list, and an unapproved key is ignored with a
  failure that reads as ``api_error`` rather than as a configuration problem.
* ``setting_sources=[]`` -- otherwise sibling brains share CLAUDE.md and
  auto-memory, which reads as model misbehaviour rather than config leakage.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from taste.brains.contract import CONTRACT_PATH, Contract
from taste.brains.jail import MUTATING_TOOLS, WorktreeJail
from taste.brains.session_store import MemstoreSessionStore
from taste.brains.wal import InFlight, WriteAheadLog, reconcile
from taste.brains.worker_environment import PROVIDER_OVERRIDE_ENV
from taste.memstore import Store
from taste.pricing import ensure_priced, max_call_cost_usd

__all__ = ["SubBrain", "SubBrainResult", "Waking"]

# Keep the worker on local, token-free builtins.  In particular, WebSearch is
# separately billed and Agent/Task can create model calls outside this
# runtime's one-stream accounting boundary.
WORKER_TOOLS = (
    "Bash",
    "Glob",
    "Grep",
    "Read",
    "Edit",
    "MultiEdit",
    "Write",
    "NotebookEdit",
)
BUDGETED_WORKER_TOOLS = tuple(tool for tool in WORKER_TOOLS if tool != "Bash")




@dataclass(frozen=True)
class Waking:
    """What a sub-brain knows the moment it starts, before it thinks.

    A fresh brain and a resumed one differ only in what is in here, which is
    what makes resume a normal path rather than a special case.
    """

    fresh: bool
    intent: str | None = None
    in_flight: tuple[InFlight, ...] = ()
    dirty_paths: tuple[str, ...] = ()
    unacked: tuple[Any, ...] = ()
    inbox: tuple[dict[str, Any], ...] = ()
    open_conflicts: tuple[Any, ...] = ()

    @property
    def uncertain(self) -> bool:
        """True if something was in flight and its outcome is unknown."""
        return bool(self.in_flight)

    def briefing(self) -> str:
        """What the brain is told about its own interrupted past.

        Silence reads to a model as "it never happened" -- measured, a brain
        resumed after an interrupted tool said "I hadn't actually executed it
        yet, so let me run it now" and repeated the command. So the unknown is
        stated explicitly rather than left to inference.
        """
        if self.fresh:
            return ""
        parts = (
            ["Your monitor has judged your work; read this before continuing."]
            if self.unacked and not (self.intent or self.in_flight or self.dirty_paths)
            else ["You are resuming work that was interrupted."]
        )
        if self.intent:
            parts.append(f"\nWhen you stopped you were about to: {self.intent}")
        if self.in_flight:
            parts.append(
                "\nThese tool calls were started and their outcomes were never "
                "recorded. They may or may not have taken effect:"
            )
            parts += [f.as_warning() for f in self.in_flight]
            parts.append(
                "\nBefore redoing any of them, look at your worktree and "
                "establish what is actually true."
            )
        if self.dirty_paths:
            parts.append(
                "\nThese files have uncommitted changes: " + ", ".join(self.dirty_paths[:20])
            )
        if self.unacked:
            parts.append("\nYour monitor has raised:")
            parts += [f"- {v.status}: {v.detail}" for v in self.unacked]
        if self.open_conflicts:
            parts.append(
                "\nThese merge conflicts are unresolved. Do not treat the affected "
                "paths as settled:"
            )
            for conflict in self.open_conflicts:
                detail = f" ({conflict.detail})" if conflict.detail else ""
                parts.append(
                    f"- {conflict.path}: ours={conflict.ours_state[:10]} "
                    f"theirs={conflict.theirs_state[:10]}{detail}"
                )
        if self.inbox:
            parts.append("\nNew messages have arrived for you:")
            for message in self.inbox:
                sender = message.get("sender") or "unknown sender"
                message_id = str(message.get("id") or "unknown")[:10]
                body = json.dumps(message.get("body"), sort_keys=True)
                parts.append(f"- [{message_id}] from {sender}: {body}")
        return "\n".join(parts)


@dataclass
class SubBrainResult:
    """What the run produced, for the spawner and the central brain."""

    identity: str
    completed: bool
    terminal_reason: str = ""
    turns: int | None = None
    # A killed SDK process may never emit its final accounting message.  None
    # means unknown; zero is reserved for a run the SDK explicitly priced at
    # zero, so supervision never turns missing evidence into a reassuring
    # number.
    cost_usd: float | None = None
    denials: tuple[tuple[str, str], ...] = ()
    error: str = ""
    states: list[str] = field(default_factory=list)


class SubBrain:
    """A worker brain bound to one memstore branch and one worktree.

    Construction takes the branch lease, so a second sub-brain on the same
    branch is refused rather than quietly interleaving with the first.
    """

    def __init__(
        self,
        store: Store,
        contract: Contract,
        *,
        model: str = "claude-sonnet-5",
        allow_paths: tuple[str, ...] = (),
        config_root: Path | None = None,
    ) -> None:
        self.store = store
        self.contract = contract
        self.model = model
        self.branch = store.branch(contract.identity, producer=contract.identity)
        self.worktree = Path(os.path.realpath(self.branch.worktree))
        self.jail = WorktreeJail(self.worktree, allow=allow_paths)
        self.sessions = MemstoreSessionStore(
            store, contract.identity, project_key=f"brain/{contract.identity}"
        )
        self.wal = WriteAheadLog(self.branch)
        # Outside the worktree: a config directory inside it shows up in
        # `git status` and a checkpoint commits the brain's own plumbing as
        # though the brain had produced it.
        self.config_dir = (
            Path(config_root or (self.store.backend.common_dir / "brain-config"))
            / contract.identity
        )
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._session_id: str | None = None

    # ------------------------------------------------------------------ waking

    def wake(self) -> Waking:
        """What this brain knows before it thinks, fresh or resumed."""
        resume = self.branch.resume()
        # Committed turns AND the live journal. `recovered_turns` is only the
        # journal for the current head, but a checkpoint folds the journal into
        # the state and unlinks it -- so an intent recorded before a checkpoint
        # became invisible to `reconcile` the moment the brain committed, even
        # though it was provably still on disk in the state transcript. That
        # turned "the gap is detectable" into "the gap is silent", which is the
        # whole distinction this layer claims to hold.
        history = self._recorded_turns()
        in_flight = tuple(reconcile(history))
        # The contract is scaffolding the spawner wrote, not work the brain
        # did. Counting it made a brand-new brain look interrupted and greeted
        # it with "You are resuming work that was interrupted" on its first
        # breath -- wrong, and a waste of the context it opens with.
        dirty = tuple(p for p in resume.dirty_paths if p != CONTRACT_PATH)
        # An unacknowledged verdict means the monitor has already judged this
        # brain and it has not reacted. A brain with a clean tree in that
        # position is not fresh -- it is a brain that has been told it went
        # wrong and does not know yet, which is precisely what the verdict
        # channel exists to prevent.
        fresh = not (
            resume.intent
            or in_flight
            or dirty
            or resume.unacked
            or resume.inbox
            or resume.open_conflicts
        )
        return Waking(
            fresh=fresh,
            intent=resume.intent,
            in_flight=in_flight,
            dirty_paths=dirty,
            unacked=resume.unacked,
            inbox=resume.inbox,
            open_conflicts=resume.open_conflicts,
        )

    def _recorded_turns(self) -> list[dict[str, Any]]:
        """Every turn this brain has recorded, committed or not.

        The two halves are the same sequence: `Branch._build` folds the
        journal into the state transcript and `publish_state` then unlinks the
        journal, so reading either alone sees only part of the run.
        """
        view = self.store.view(self.contract.identity)
        if not view.exists():
            return list(self.branch.resume().recovered_turns)
        return list(view.head.transcript.turns) + list(view.pending_turns())

    def install_contract(self) -> None:
        """Put the contract in the branch, so every brain reads the same one.

        Idempotent: a resumed brain finds its contract already there and must
        not rewrite it, or a re-plan that revised the contract would be undone
        by the very brain that was told to follow the revision.
        """
        if self.branch.read(CONTRACT_PATH) is None and not (
            self.branch.path(CONTRACT_PATH).exists()
        ):
            self.branch.write(CONTRACT_PATH, self.contract.to_json())

    ENVIRONMENT_BRIEFING = """\
About where you are working:

  - This directory is a git worktree owned by the harness, and it *is* a
    branch head in the memory layer. The harness checkpoints your work for
    you -- you never commit.
  - So `git commit`, `reset`, `checkout`, `rebase`, `stash` and friends are
    refused. They would move the branch out from under the layer that owns
    it. Reading git is fine and often useful: `status`, `diff`, `log`, `show`.
  - Edit files directly. That is the whole job; the harness handles history.
  - Stay inside this directory. Writes outside it are refused.

If something is refused, the refusal says why. Read it and take another
route rather than reissuing the same call -- a second identical attempt
fails the same way and costs you a turn."""

    def opening_prompt(self, waking: Waking | None = None) -> str:
        """The contract, the environment, plus anything an interrupted past requires.

        The contract leads: the task is the job and the rest is context.

        The environment paragraph exists because the worker ran with no system
        prompt at all. Its entire briefing was the contract, so it learned the
        rules only by breaking one and reading the denial -- which costs a turn
        each time, and for a model that does not generalise from the first
        refusal, costs all of them. The jail's message is good feedback; it
        should not be the first time a worker hears the rule.
        """
        waking = waking or self.wake()
        parts = [self.contract.brief(), self.ENVIRONMENT_BRIEFING]
        if not waking.fresh:
            parts.append(waking.briefing())
        return "\n\n".join(part for part in parts if part.strip()).strip()

    # ------------------------------------------------------------------ hooks

    async def _pre_tool(
        self, input_data: dict[str, Any], tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        """Gate the call, then record the intent -- in that order.

        The jail runs first because a refused call has no side effect to
        record, and recording an intent for it would leave a permanent unknown
        in the log for something that provably never happened.
        """
        verdict = await self.jail.hook(input_data, tool_use_id, context)
        if verdict:
            return verdict
        if isinstance(input_data, dict):
            self.wal.intent(
                str(input_data.get("tool_name", "?")),
                str(tool_use_id or ""),
                input_data.get("tool_input")
                if isinstance(input_data.get("tool_input"), dict)
                else {},
            )
        return {}

    async def _post_tool(
        self, input_data: dict[str, Any], tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        """Record that the side effect now exists."""
        if isinstance(input_data, dict):
            response = input_data.get("tool_response")
            ok = not (isinstance(response, dict) and response.get("is_error"))
            self.wal.result(
                str(input_data.get("tool_name", "?")),
                str(tool_use_id or ""),
                ok=ok,
                summary=str(response)[:2000],
            )
        return {}

    # ------------------------------------------------------------------ options

    def options(
        self,
        *,
        resume: str | None = None,
        budget_already_spent_usd: float = 0.0,
        **extra: Any,
    ) -> Any:
        """The SDK options this brain runs under. See the module docstring."""
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

        if (
            isinstance(budget_already_spent_usd, bool)
            or not isinstance(budget_already_spent_usd, (int, float))
            or not math.isfinite(float(budget_already_spent_usd))
            or float(budget_already_spent_usd) < 0
        ):
            raise ValueError("budget_already_spent_usd must be finite and non-negative")
        already_spent = float(budget_already_spent_usd)

        env = dict(os.environ)
        for name in PROVIDER_OVERRIDE_ENV:
            env.pop(name, None)
        env["CLAUDE_CONFIG_DIR"] = str(self.config_dir)
        env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
        env["CLAUDE_CODE_DISABLE_ADVISOR_TOOL"] = "1"
        env["CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK"] = "1"
        env["CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK"] = "1"
        env["CLAUDE_CODE_MAX_RETRIES"] = "0"
        env["CLAUDE_CODE_NO_MODEL_FALLBACK"] = "1"
        if self.contract.budget_usd is not None:
            # A compact or slash-command clear can replace the conversation
            # and its CLI-local cost scope. Budgeted workers fail at the
            # context boundary instead of silently acquiring a fresh ledger.
            env["DISABLE_COMPACT"] = "1"
        if env.get("ANTHROPIC_API_KEY") and not env.get("ANTHROPIC_AUTH_TOKEN"):
            env["ANTHROPIC_AUTH_TOKEN"] = env["ANTHROPIC_API_KEY"]
            env.pop("ANTHROPIC_API_KEY", None)

        hooks = {
            "PreToolUse": [
                HookMatcher(matcher=tool, hooks=[self._pre_tool], timeout=5.0)
                for tool in MUTATING_TOOLS
            ],
            "PostToolUse": [
                HookMatcher(matcher=tool, hooks=[self._post_tool], timeout=5.0)
                for tool in MUTATING_TOOLS
            ],
        }
        # Defaults a caller may override -- and which are then checked, so an
        # override that would disarm the boundary is refused rather than
        # quietly accepted.
        defaults: dict[str, Any] = {
            "permission_mode": "acceptEdits",
            "tools": list(
                BUDGETED_WORKER_TOOLS if self.contract.budget_usd is not None else WORKER_TOOLS
            ),
            "disallowed_tools": [
                "Agent",
                "Task",
                "WebSearch",
                "WebFetch",
                *(["Bash"] if self.contract.budget_usd is not None else []),
            ],
            "mcp_servers": {},
            "strict_mcp_config": True,
            "extra_args": (
                {"disable-slash-commands": None} if self.contract.budget_usd is not None else {}
            ),
            "sandbox": {
                "enabled": True,
                "autoAllowBashIfSandboxed": True,
                "allowUnsandboxedCommands": False,
            },
        }
        defaults.update(extra)
        extra = defaults
        options = ClaudeAgentOptions(
            model=self.model,
            cwd=str(self.worktree),
            env=env,
            setting_sources=[],
            hooks=hooks,
            session_store=self.sessions,
            session_store_flush="eager",
            include_partial_messages=True,
            max_turns=self.contract.max_turns,
            resume=resume,
            **extra,
        )
        if self.contract.budget_usd is not None:
            budget = self.contract.budget_usd
            if (
                isinstance(budget, bool)
                or not isinstance(budget, (int, float))
                or not math.isfinite(float(budget))
                or float(budget) < 0
            ):
                raise ValueError("contract budget_usd must be finite and non-negative")

            # Claude Code's ``--max-budget-usd`` guard compares accumulated
            # spend *before* dispatching the next model request. Passing the
            # contract cap through verbatim therefore permits the last request
            # to cross it.  A conversation reset also zeroes that CLI counter,
            # and a prompt already in its input pipe can spend another whole
            # threshold before this host observes the reset and kills the CLI.
            # For remaining allowance R, threshold T and one-call overhang E,
            # the safe one-reset inequality is 2*(T + E) <= R.
            #
            # The runtime treats the first reset as terminal and synchronously
            # kills the local CLI before awaiting anything. Durable spend from
            # earlier processes is subtracted before solving the inequality.
            price = ensure_priced(self.model)
            final_call_exposure = max_call_cost_usd(
                self.model,
                max_output_tokens=price.context_window,
                max_attempts=1,
                cap_on="billed",
            )
            remaining = float(budget) - already_spent
            provider_threshold = remaining / 2.0 - final_call_exposure
            # Claude Code rejects a zero max-budget flag. More importantly, a
            # remaining cap no larger than the protected requests cannot admit
            # a request without risking a contract overrun. Refuse before the
            # provider process exists, rather than turning missing accounting
            # after an infrastructure exit into an ambiguous charge.
            if provider_threshold <= 0:
                raise ValueError(
                    "remaining contract budget cannot cover the reset-safe model exposure"
                )
            options.max_budget_usd = provider_threshold
        self._assert_gates_are_live(options)
        return options

    def _assert_gates_are_live(self, options: Any) -> None:
        """Refuse a configuration whose gates would silently never fire.

        Naming a tool in ``allowed_tools`` auto-approves it before the
        permission callback is consulted -- measured, the callback logged zero
        events while the denied write went through. A gate that appears wired
        up and is not is worse than no gate, so this is checked rather than
        remembered.
        """
        named = set(getattr(options, "allowed_tools", None) or ())
        if named:
            raise ValueError(
                "allowed_tools must remain empty; auto-approval would shadow the "
                "ordinary permission boundary"
            )
        sandbox = getattr(options, "sandbox", None) or {}
        expected_sandbox = {
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
        }
        if sandbox != expected_sandbox:
            raise ValueError(
                "the sandbox boundary must remain enabled without exclusions, "
                "ignored violations, weaker nesting, or unsandboxed commands"
            )
        expected_tools = (
            BUDGETED_WORKER_TOOLS if self.contract.budget_usd is not None else WORKER_TOOLS
        )
        tools = getattr(options, "tools", None)
        if not isinstance(tools, list) or not set(tools) <= set(expected_tools):
            raise ValueError("worker tools must be an explicit subset of local token-free builtins")
        forbidden = {"Agent", "Task", "WebSearch", "WebFetch"}
        if self.contract.budget_usd is not None:
            forbidden.add("Bash")
        if not forbidden <= set(getattr(options, "disallowed_tools", None) or ()):
            raise ValueError("worker disallowed_tools removed a cost boundary")
        if (
            getattr(options, "mcp_servers", None) != {}
            or getattr(options, "strict_mcp_config", None) is not True
        ):
            raise ValueError("worker MCP configuration must be strict and empty")
        if getattr(options, "fallback_model", None) is not None:
            raise ValueError("worker fallback_model would break exact model cost accounting")
        if getattr(options, "agents", None) or getattr(options, "skills", None):
            raise ValueError("worker agents and skills bypass the one-stream cost boundary")
        if getattr(options, "plugins", None):
            raise ValueError("worker plugins bypass the fixed local-tool boundary")
        if getattr(options, "env", {}).get("CLAUDE_CODE_DISABLE_ADVISOR_TOOL") != "1":
            raise ValueError("the separately billed advisor tool must remain disabled")
        required_env = {
            "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "1",
            "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK": "1",
            "CLAUDE_CODE_MAX_RETRIES": "0",
            "CLAUDE_CODE_NO_MODEL_FALLBACK": "1",
        }
        if any(
            getattr(options, "env", {}).get(key) != value for key, value in required_env.items()
        ):
            raise ValueError("worker retry and model fallback controls must remain disabled")
        if (
            self.contract.budget_usd is not None
            and getattr(options, "env", {}).get("DISABLE_COMPACT") != "1"
        ):
            raise ValueError("budgeted worker compaction must remain disabled")
        if PROVIDER_OVERRIDE_ENV & set(getattr(options, "env", {})):
            raise ValueError("worker environment contains a provider or model override")
        if getattr(options, "permission_mode", None) != "acceptEdits":
            raise ValueError("worker permission_mode must remain acceptEdits")
        if (
            getattr(options, "extra_args", None)
            != ({"disable-slash-commands": None} if self.contract.budget_usd is not None else {})
            or getattr(options, "settings", None) is not None
            or getattr(options, "cli_path", None) is not None
            or getattr(options, "betas", None)
            or getattr(options, "task_budget", None) is not None
            or getattr(options, "add_dirs", None)
            or getattr(options, "permission_prompt_tool_name", None) is not None
            or getattr(options, "session_id", None) is not None
            or getattr(options, "continue_conversation", False)
            or getattr(options, "fork_session", False)
            or getattr(options, "resume_session_at", None) is not None
            or getattr(options, "resume_drops_turn", None) is not None
            or getattr(options, "enable_file_checkpointing", False)
        ):
            raise ValueError("worker options contain an unaudited CLI escape hatch")
        if self.contract.budget_usd is not None and getattr(options, "resume", None):
            raise ValueError("budgeted workers cannot resume a prior provider connection")

    # ------------------------------------------------------------------ state

    def checkpoint(self, reason: str) -> Any:
        """Turn everything written so far into a state.

        The only place a state is published. The transcript adapter and the
        WAL both write into the worktree and let this commit them, so a state
        exists exactly where the tree is coherent and the reason is true.
        """
        state = self.branch.checkpoint(reason)
        self.wal.rebind()
        return state

    def close(self) -> None:
        self.branch.release()
