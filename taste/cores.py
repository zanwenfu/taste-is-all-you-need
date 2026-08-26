"""The three CPU cores: Planner, Worker, Monitor.

Each core is a function, not a class. They share no mutable state. The
Kernel feeds them inputs, they return results, and every state transition
goes through :class:`taste.memory.Memory`. That's the separation that lets
us delete the Planner or Monitor on a future model generation without
bringing down the rest of the system — per the blog's *build to delete*
principle.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from taste.agent import AgentSpec
from taste.llm import (
    LLM,
    MODEL_MONITOR,
    MODEL_PLANNER,
    MODEL_WORKER,
    cached,
    static_system,
)
from taste.memory import Checkpoint, Memory
from taste.tools import ToolRegistry


class PlannerError(RuntimeError):
    """Raised when the Planner returns a malformed or missing plan."""


# ============================================================== Plan schema


@dataclass
class Verification:
    """How the monitor proves a step succeeded.

    ``shell`` runs a command in the workspace and passes iff exit 0 — cheap,
    deterministic, the default. ``llm`` asks the Monitor model to judge the
    diff against natural-language criteria — reserved for milestones the
    planner flags as not mechanically checkable.
    """

    kind: Literal["shell", "llm"]
    command: str | None = None
    criteria: str | None = None


@dataclass
class Step:
    id: str
    description: str
    verification: Verification
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Plan:
    task: str
    steps: list[Step] = field(default_factory=list)

    def to_summary(self) -> str:
        lines = [f"Task: {self.task}", "", "Plan:"]
        for s in self.steps:
            deps = f"  (depends on: {', '.join(s.depends_on)})" if s.depends_on else ""
            lines.append(f"  {s.id}: {s.description}{deps}")
            if s.verification.kind == "shell":
                lines.append(f"    check: `{s.verification.command}`")
            else:
                lines.append(f"    check (llm): {s.verification.criteria}")
        return "\n".join(lines)

    def waves(self) -> list[list[Step]]:
        """Group steps into parallel waves by ``depends_on`` (topological).

        Steps in the same wave have all of their dependencies satisfied by
        earlier waves and no dependencies on each other — so they can run
        concurrently, each on its own worktree, and be merged back at the
        end of the wave.

        If no step declares a dependency, the plan is treated as a linear
        chain (sequential, one per wave). This keeps plans without explicit
        parallelism hints safe by default: a Planner that ignores the
        ``depends_on`` field ends up with the old single-threaded behavior
        rather than surprise parallelism across steps that may conflict.
        """
        if not any(s.depends_on for s in self.steps):
            return [[s] for s in self.steps]

        by_id = {s.id: s for s in self.steps}
        remaining = dict(by_id)
        done: set[str] = set()
        waves: list[list[Step]] = []

        while remaining:
            ready = [
                s for s in remaining.values() if set(s.depends_on).issubset(done)
            ]
            if not ready:
                unresolved = list(remaining)
                raise ValueError(
                    f"unresolvable dependency cycle or dangling ref among: {unresolved}"
                )
            ready.sort(key=lambda s: [t.id for t in self.steps].index(s.id))
            waves.append(ready)
            for s in ready:
                done.add(s.id)
                del remaining[s.id]
        return waves


# ============================================================== Planner


PLANNER_SYSTEM = """You are the Planner in an Agent OS — the first core of a multi-core agent harness.

Your single job: decompose the user's task into the smallest viable sequence of steps such that each step can be verified by running a command that actually exercises behavior — not by checking that a string appears in a file.

Verification taste (this is the part models get wrong):
  * STRONG verifications actually execute the code: `pytest`, `pytest -k <pattern>`, `python -c '<assertion>'`, `ruff check`, `mypy`.
  * WEAK verifications only confirm text presence: `grep`, `find`, `ls`, `cat | grep`. These are BANNED — if you catch yourself reaching for grep as a verification, the step is not actually verifiable and you should merge it into the next real check.
  * If a step cannot be verified by executing something, either (a) merge it into the next step that CAN be verified, or (b) add an assertion to the existing test suite.
  * For a workspace with a pytest suite, default every verification to `pytest -q` unless you have a concrete reason to use something narrower.

Dependencies and parallelism:
  * Every step has a `depends_on` list (step IDs it strictly requires before it can run).
  * Steps with the same `depends_on` set (or both empty) will be executed **in parallel**, each on its own git worktree, and merged back into the session branch when both succeed.
  * Use parallelism when subtasks touch **disjoint files** — e.g. "add type hints to module_a" and "add type hints to module_b". Do NOT parallelize steps that edit the same file; git will conflict and the run will halt.
  * When unsure, prefer sequential (`depends_on: [<previous-step-id>]`). Safety beats speedup.

Principles:
  1. Each step is the smallest unit of progress that can be independently committed and reverted. If a step cannot be verified, split or merge it.
  2. Prefer `shell` verifications over `llm` verifications. Deterministic checks cannot flatter mediocre work.
  3. Only use `llm` verification when the outcome is genuinely subjective (e.g. "docstrings are clear"). Keep LLM checks rare.
  4. Keep plans tight — 3 to 8 steps. If you catch yourself planning step 15, you are over-decomposing.
  5. Step IDs must be zero-padded: step-01, step-02, ... so they sort lexicographically.

Output: call the `submit_plan` tool exactly once. Do not emit prose.
"""


PLAN_TOOL = {
    "name": "submit_plan",
    "description": "Submit the final decomposition. Call exactly once.",
    "input_schema": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "pattern": "^step-\\d{2,}$"},
                        "description": {"type": "string"},
                        "verification": {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string", "enum": ["shell", "llm"]},
                                "command": {"type": "string"},
                                "criteria": {"type": "string"},
                            },
                            "required": ["kind"],
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Step IDs that must complete before this step. "
                                "Steps with no shared dependencies run in parallel on separate worktrees. "
                                "Leave empty ([]) for steps that can run first or concurrently."
                            ),
                        },
                    },
                    "required": ["id", "description", "verification"],
                },
            }
        },
        "required": ["steps"],
    },
}


def plan(
    llm: LLM,
    task: str,
    spec: AgentSpec,
    workspace_summary: str,
    *,
    model: str | None = None,
    attempts: int = 3,
) -> Plan:
    """Ask the Planner model to decompose ``task`` into steps with verifications.

    ``spec.model`` deliberately does NOT reach the Planner: the agent spec's
    model field configures the Worker only. Letting it override every role
    (the pre-Wave-0 behavior) made the model-size role split inexpressible —
    an audited validity threat. Pass ``model=`` explicitly to override.

    **A malformed plan is retried, with the complaint fed back.** Measured on
    real tasks, a single bad tool call killed **2 of 6 runs** outright — once
    with steps as JSON strings, once with the ``steps`` field missing entirely
    — and each was recorded as a *task* failure, so it would have counted as
    evidence about whichever arm happened to draw it. A schema violation on
    one sample is a transient model failure, not an outcome, and telling the
    model exactly what arrived fixes it far more often than asking again
    blindly. Genuine refusals still fail after the last attempt.
    """
    system = [
        static_system(
            PLANNER_SYSTEM,
            f"Agent capability:\n{spec.description}\n\nAgent instructions:\n{spec.system_prompt}",
        ),
    ]
    messages = [
        {
            "role": "user",
            "content": (
                f"Task: {task}\n\n"
                f"Workspace summary (git ls-files, head of key files):\n{workspace_summary}\n\n"
                "Produce a plan. Call submit_plan with the steps."
            ),
        }
    ]
    last: PlannerError | None = None
    for attempt in range(1, max(1, attempts) + 1):
        response = llm.call(
            model=model or MODEL_PLANNER,
            system=system,
            messages=messages,
            tools=[PLAN_TOOL],
            max_tokens=4096,
            role="planner",
        )
        try:
            return _plan_from(task, response)
        except PlannerError as exc:
            last = exc
            if attempt == attempts:
                break
            # Feed the complaint back. A model told *what arrived* corrects it;
            # one asked again blindly usually repeats itself.
            messages = [
                *messages,
                {"role": "assistant", "content": "(previous submit_plan call)"},
                {
                    "role": "user",
                    "content": (
                        f"That plan could not be read: {exc}. "
                        "Call submit_plan again. `steps` must be a JSON array of "
                        "objects, each with `id`, `description`, and a "
                        "`verification` object — not strings, and not omitted."
                    ),
                },
            ]
    raise last or PlannerError("planner produced no usable plan")


def _plan_from(task: str, response: Any) -> Plan:
    """One planner response, parsed. Raises :class:`PlannerError` on anything
    it cannot read, so the retry loop above has a single thing to catch."""
    try:
        payload = _extract_tool_input(response, "submit_plan")
    except RuntimeError as exc:
        raise PlannerError(str(exc)) from exc

    try:
        steps = [
            Step(
                id=s["id"],
                description=s["description"],
                verification=Verification(**s["verification"]),
                depends_on=list(s.get("depends_on") or []),
            )
            for s in (_coerce_step(raw) for raw in payload["steps"])
        ]
    except (KeyError, TypeError, ValueError) as exc:
        shape = _describe_shape(payload)
        raise PlannerError(f"malformed plan payload: {exc} (got {shape})") from exc

    if not steps:
        raise PlannerError("planner returned an empty step list")
    return Plan(task=task, steps=steps)


# ============================================================== Worker


@dataclass(frozen=True)
class ToolDecision:
    """What the harness says about a tool call the worker wants to make."""

    action: Literal["allow", "rewrite", "veto"] = "allow"
    payload: dict[str, Any] | None = None
    """Replacement arguments, for ``rewrite``."""
    message: str | None = None
    """Returned to the model as the tool result, for ``veto``. The model must
    learn *why* it was stopped, or it will simply try again."""
    reason: str = ""


ALLOW = ToolDecision()


@dataclass(frozen=True)
class Interrupt:
    """A reason to stop a worker mid-flight."""

    kind: str
    detail: str
    turn: int
    failure_kind: Literal["task", "infra", "budget"] = "task"


class TurnHook(Protocol):
    """The only callback surface on the worker loop. A closed set of four
    methods, deliberately.

    Every implementation must be fail-open: it catches its own exceptions and
    returns the no-op value. A broken hook degrades the run to unguarded
    behavior; it must never be the reason a run halts.
    """

    def before_turn(self, turn: int) -> Interrupt | None: ...
    def before_tool(self, turn: int, name: str, payload: dict[str, Any]) -> ToolDecision: ...
    def after_tool(
        self,
        turn: int,
        name: str,
        payload: dict[str, Any],
        output: str,
        elapsed_s: float,
        decision: ToolDecision,
    ) -> None: ...
    def after_turn(self, turn: int, message: Any, stop_reason: str) -> None: ...


WORKER_SYSTEM = """You are a Worker core in an Agent OS. You execute exactly one planned step.

Rules:
  1. Your only job is the current step. Do not drift into later steps.
  2. Use the minimum number of tool calls. If you can do the step in one `write_file`, do it in one.
  3. When the step is complete, stop. The kernel will checkpoint your changes and hand control to the Monitor.
  4. If you are uncertain how to verify your own work, do nothing extra — the Monitor will run the step's verification.
  5. Do not create commits, branches, or touch git. The kernel owns the memory layer.
"""


@dataclass
class WorkerResult:
    summary: str
    tool_calls: int
    stopped_reason: str
    tool_errors: int = 0
    tool_error_kinds: tuple[str, ...] = ()
    vetoes: int = 0
    interrupt: Interrupt | None = None
    turns: int = 0


#: Output ceiling for one worker turn. 4096 was the original cap, and a
#: whole-file `write_file` sails through it: the provider truncates the tool
#: call's JSON arguments at the ceiling, and the failure forks on a coin —
#: invalid JSON dies as a ProtocolFailure, while VALID-but-truncated JSON
#: writes half a file into the tree. The second fork was measured: a
#: truncated write produced a 78-event regression storm in one calibration
#: run. The ceiling is generous AND the truncation guard below still refuses
#: to execute any capped turn, because no ceiling is unreachable.
WORKER_MAX_TOKENS = 32_000

TRUNCATION_FEEDBACK = (
    "Your previous response exceeded the output limit and was discarded — "
    "none of its tool calls were executed, because their contents may have "
    "been cut off mid-way. Re-issue the work in smaller pieces: write large "
    "files in sections (write the file with the first section, then use "
    "read_file and write_file to append the rest), and keep each tool call "
    "well under the limit."
)


def execute(
    llm: LLM,
    *,
    spec: AgentSpec,
    step: Step,
    plan_context: str,
    tools: ToolRegistry,
    max_turns: int = 12,
    hook: TurnHook | None = None,
    guidance: str | None = None,
    mode: Literal["fresh", "repair"] = "fresh",
) -> WorkerResult:
    """Run the model's tool-use loop for a single step.

    Returns a :class:`WorkerResult` describing what happened. The kernel is
    responsible for turning the filesystem changes into a commit.

    ``hook`` is the only extension point: it may veto or rewrite a tool call
    before it runs, and may interrupt between turns. ``guidance`` is prior
    feedback to carry in, and ``mode`` frames the task as fresh work or as a
    repair of what is already on disk.
    """
    # One consolidated static block (crosses the cache minimum) + the plan
    # context on its own breakpoint (stable across the turns of a step, so the
    # tool-loop turns hit cache; varies across steps/retries).
    system = [
        static_system(
            WORKER_SYSTEM,
            f"Agent capability:\n{spec.description}\n\nAgent instructions:\n{spec.system_prompt}",
        ),
        cached(f"Plan so far:\n{plan_context}"),
    ]
    framing = (
        "Execute only this step. Stop when it's done."
        if mode == "fresh"
        else (
            "A previous attempt at this step is already on disk and did not pass. "
            "Repair it in place — build on what is there rather than starting over. "
            "Stop when it's done."
        )
    )
    opening = f"Current step ({step.id}): {step.description}\n\n{framing}"
    if guidance:
        opening += f"\n\nWhat went wrong before:\n{guidance}"
    messages: list[dict[str, Any]] = [{"role": "user", "content": opening}]

    tool_calls = 0
    tool_errors = 0
    error_kinds: list[str] = []
    vetoes = 0
    summary = ""
    stop_reason = "unknown"
    interrupt: Interrupt | None = None
    turns = 0

    for turn in range(1, max_turns + 1):
        turns = turn
        if hook is not None:
            interrupt = _safe_before_turn(hook, turn)
            if interrupt is not None:
                stop_reason = "interrupted"
                break

        completion = llm.call(
            model=spec.model or MODEL_WORKER,
            system=system,
            messages=messages,
            tools=tools.to_anthropic(),
            max_tokens=WORKER_MAX_TOKENS,
            role="worker",
        )
        stop_reason = completion.stop_reason
        messages.append({"role": "assistant", "content": list(completion.transcript_blocks)})

        if stop_reason == "max_tokens":
            # A capped turn is discarded wholesale, tool calls unexecuted. A
            # call that arrived complete before the cap would be safe, but
            # the last one may be truncated-yet-parseable, and executing it
            # writes a mangled artifact the Monitor then diagnoses as the
            # agent's incompetence. The model is told, and the turn retries.
            messages.append({"role": "user", "content": TRUNCATION_FEEDBACK})
            continue

        tool_uses = completion.tool_calls
        if completion.text_blocks:
            summary = completion.text_blocks[-1]
        if hook is not None:
            _safe_after_turn(hook, turn, completion, stop_reason)

        if stop_reason == "end_turn" or not tool_uses:
            break

        tool_results = []
        for call in tool_uses:
            tool_calls += 1
            payload = dict(call.arguments)
            decision = _safe_before_tool(hook, turn, call.name, payload) if hook else ALLOW

            if decision.action == "veto":
                vetoes += 1
                # The refusal goes back as the tool's result: a model that is
                # not told why it was stopped simply tries again.
                output = decision.message or f"BLOCKED: {decision.reason}"
            else:
                if decision.action == "rewrite" and decision.payload is not None:
                    payload = decision.payload
                started = time.monotonic()
                try:
                    output = tools.invoke(call.name, payload)
                except Exception as exc:  # surface to the model, don't crash the loop
                    tool_errors += 1
                    error_kinds.append(type(exc).__name__)
                    output = f"TOOL ERROR ({type(exc).__name__}): {exc}"
                if hook is not None:
                    _safe_after_tool(
                        hook,
                        turn,
                        call.name,
                        payload,
                        output,
                        time.monotonic() - started,
                        decision,
                    )

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": output,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return WorkerResult(
        summary=summary or "(no summary emitted)",
        tool_calls=tool_calls,
        stopped_reason=stop_reason,
        tool_errors=tool_errors,
        tool_error_kinds=tuple(dict.fromkeys(error_kinds)),
        vetoes=vetoes,
        interrupt=interrupt,
        turns=turns,
    )


def _safe_after_turn(hook: TurnHook, turn: int, message: Any, stop_reason: str) -> None:
    """Observation callbacks are advisory; a hook never breaks the loop."""
    with contextlib.suppress(Exception):
        hook.after_turn(turn, message, stop_reason)


def _safe_after_tool(
    hook: TurnHook,
    turn: int,
    name: str,
    payload: dict[str, Any],
    output: str,
    elapsed_s: float,
    decision: ToolDecision,
) -> None:
    with contextlib.suppress(Exception):
        hook.after_tool(turn, name, payload, output, elapsed_s, decision)


def _safe_before_turn(hook: TurnHook, turn: int) -> Interrupt | None:
    try:
        return hook.before_turn(turn)
    except Exception:
        return None


def _safe_before_tool(
    hook: TurnHook, turn: int, name: str, payload: dict[str, Any]
) -> ToolDecision:
    try:
        return hook.before_tool(turn, name, payload)
    except Exception:
        return ALLOW  # fail open: a broken guard must not block real work


# ============================================================== Monitor


@dataclass
class MonitorResult:
    passed: bool
    reason: str
    evidence: str = ""

    def format(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.reason}"


# The Monitor judges deterministically on purpose — its verdicts gate commits.
# Recorded in the run manifest; keep manifest and call site in sync via this
# constant, never an inline literal.
MONITOR_TEMPERATURE = 0.0

MONITOR_SYSTEM = """You are the Monitor core in an Agent OS.

A Worker just completed a step. Your job is to decide: did it actually succeed,
or should the kernel roll back and retry?

You will receive the step description, the evaluation criteria, and the diff
that the Worker produced. Judge **only** whether the diff satisfies the
criteria. Do not suggest improvements; do not second-guess the plan.

Be strict. Agents tend to confidently praise their own work — you are the
counterweight. If you are uncertain, err on the side of FAIL.

Respond by calling the `report_verdict` tool with your decision.
"""


VERDICT_TOOL = {
    "name": "report_verdict",
    "description": "Report whether the step passed its verification.",
    "input_schema": {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "reason": {"type": "string", "description": "One sentence justification."},
        },
        "required": ["passed", "reason"],
    },
}


def evaluate(
    *,
    step: Step,
    memory: Memory,
    workspace: Path,
    llm: LLM | None = None,
    monitor_model: str = MODEL_MONITOR,
    before: Checkpoint | None = None,
    shell_runner: Callable[..., Any] | None = None,
) -> MonitorResult:
    """Evaluate the current step's verification.

    Shell verifications are executed in ``workspace`` and pass iff exit 0. LLM
    verifications delegate to the Monitor model, which judges the worker's
    *uncommitted* changes against natural-language criteria. Deterministic
    checks are preferred because they are cheap, reproducible, and immune to
    the self-praise failure mode.

    ``before`` is the pre-step checkpoint the kernel anchored on. It defines
    what "the worker's changes" means: everything between that commit and the
    current working tree. Without it we fall back to HEAD, which is correct
    but attributes any pre-existing uncommitted state to this step.
    """
    v = step.verification
    if v.kind == "shell":
        return _run_shell_check(v.command or "true", workspace, runner=shell_runner)
    if v.kind == "llm":
        if llm is None:
            raise ValueError("LLM verification requested but no LLM provided.")
        return _run_llm_check(v.criteria or "", step, memory, llm, monitor_model, before=before)
    raise ValueError(f"unknown verification kind: {v.kind}")


#: Ceiling for one deterministic Monitor check. 180s was enough for the
#: host-side module probes the Planner writes; a routed check that runs a
#: real slice of an instance's suite inside the image can legitimately take
#: longer, and a timeout here renders as the step failing — a verdict, not a
#: hole — so the ceiling errs generous.
MONITOR_CHECK_TIMEOUT = 600


def _run_shell_check(
    command: str, workspace: Path, *, runner: Callable[..., Any] | None = None
) -> MonitorResult:
    """Run the step's deterministic check and turn its exit into a verdict.

    ``runner`` is the one-environment seam. When present the check executes
    wherever the agent's own commands execute — the task's pinned container —
    instead of on the host. The Monitor grading a different environment than
    the Worker worked in is bug 20: on a bare host checkout the project
    imports as an uncompiled namespace package, every real check fails, and
    the failure is indistinguishable from the agent's work being wrong.
    26 of 28 zero-step pilot runs traced here.
    """
    if runner is not None:
        result = runner(command, timeout=MONITOR_CHECK_TIMEOUT)
        if result.timed_out:
            return MonitorResult(passed=False, reason=f"check timed out: {command}")
        passed = result.exit_code == 0
        reason = (
            f"`{command}` exited 0" if passed
            else f"`{command}` exited {result.exit_code}"
        )
        evidence = _tail(result.stdout, 40) + (
            "\n[stderr]\n" + _tail(result.stderr, 40) if result.stderr else ""
        )
        return MonitorResult(passed=passed, reason=reason, evidence=evidence)
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=MONITOR_CHECK_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return MonitorResult(passed=False, reason=f"check timed out: {command}")

    passed = proc.returncode == 0
    reason = (
        f"`{command}` exited 0" if passed
        else f"`{command}` exited {proc.returncode}"
    )
    evidence = _tail(proc.stdout, 40) + ("\n[stderr]\n" + _tail(proc.stderr, 40) if proc.stderr else "")
    return MonitorResult(passed=passed, reason=reason, evidence=evidence)


def _run_llm_check(
    criteria: str,
    step: Step,
    memory: Memory,
    llm: LLM,
    model: str,
    *,
    before: Checkpoint | None = None,
) -> MonitorResult:
    # The judge must see what the worker JUST did. Verification runs before
    # the kernel commits, so the changes are still uncommitted — diffing the
    # last commit against its parent would grade the *previous* step's work.
    anchor = before.sha if before is not None else memory.head().sha
    diff = memory.diff_pending(anchor)
    # Single consolidated block for consistency with the other cores. Note the
    # monitor prompt alone sits below the cache minimum on Haiku-class models,
    # so this block may not cache — shell monitors (the default) dominate runs.
    system = [static_system(MONITOR_SYSTEM)]
    messages = [
        {
            "role": "user",
            "content": (
                f"Step: {step.id} — {step.description}\n\n"
                f"Criteria:\n{criteria}\n\n"
                f"The Worker's changes (uncommitted, since {anchor[:7]}):\n"
                f"```diff\n{_tail(diff, 400)}\n```"
            ),
        }
    ]
    response = llm.call(
        model=model,
        system=system,
        messages=messages,
        tools=[VERDICT_TOOL],
        max_tokens=1024,
        temperature=MONITOR_TEMPERATURE,
        role="monitor",
    )
    verdict = _extract_tool_input(response, "report_verdict")
    return MonitorResult(
        passed=bool(verdict["passed"]),
        reason=str(verdict["reason"]),
        evidence=f"judged on pending changes since {anchor[:7]}",
    )


# ============================================================== helpers


def _coerce_step(raw: Any) -> dict[str, Any]:
    """One plan step, whatever shape the model actually emitted.

    The tool schema asks for objects, and models mostly comply — but not
    always: a real planner call returned ``steps`` as a list of JSON *strings*,
    which failed with "string indices must be integers" and killed the whole
    run before a single step executed. Nested-object stringification differs
    across providers, so this is a portability issue rather than a one-off.

    Decoding a string here is not the same as accepting anything: a value that
    is neither an object nor an object-encoding string still raises, because a
    plan we cannot read must fail loudly rather than run half of itself.
    """
    if isinstance(raw, str):
        decoded = json.loads(raw)
        if not isinstance(decoded, dict):
            raise TypeError(f"plan step decoded to {type(decoded).__name__}, not an object")
        return decoded
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"plan step is {type(raw).__name__}, not an object")


def _describe_shape(payload: Any) -> str:
    """A short description of what the planner sent, for the error message.

    Without it the failure reads "string indices must be integers" with no
    hint of which layer was wrong, and diagnosing it costs a run.
    """
    if not isinstance(payload, dict):
        return type(payload).__name__
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return f"steps={type(steps).__name__}"
    kinds = sorted({type(s).__name__ for s in steps})
    return f"steps=list[{'|'.join(kinds) or 'empty'}] n={len(steps)}"


def _extract_tool_input(completion: Any, tool_name: str) -> dict[str, Any]:
    """The arguments of the named tool call, whichever provider produced it."""
    for call in completion.tool_calls:
        if call.name == tool_name:
            return dict(call.arguments)
    raise RuntimeError(
        f"model did not call `{tool_name}` (stop_reason={completion.stop_reason})"
    )


def _tail(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "... (truncated) ...\n" + "\n".join(lines[-max_lines:])
