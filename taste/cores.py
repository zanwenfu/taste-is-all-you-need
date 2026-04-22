"""The three CPU cores: Planner, Worker, Monitor.

Each core is a function, not a class. They share no mutable state. The
Kernel feeds them inputs, they return results, and every state transition
goes through :class:`taste.memory.Memory`. That's the separation that lets
us delete the Planner or Monitor on a future model generation without
bringing down the rest of the system — per the blog's *build to delete*
principle.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from taste.agent import AgentSpec
from taste.llm import LLM, MODEL_MONITOR, MODEL_PLANNER, MODEL_WORKER, cached
from taste.memory import Memory
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


def plan(llm: LLM, task: str, spec: AgentSpec, workspace_summary: str) -> Plan:
    """Ask the Planner model to decompose ``task`` into steps with verifications."""
    system = [
        cached(PLANNER_SYSTEM),
        cached(f"Agent capability:\n{spec.description}\n\nAgent instructions:\n{spec.system_prompt}"),
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
    response = llm.call(
        model=spec.model or MODEL_PLANNER,
        system=system,
        messages=messages,
        tools=[PLAN_TOOL],
        max_tokens=4096,
    )
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
            for s in payload["steps"]
        ]
    except (KeyError, TypeError) as exc:
        raise PlannerError(f"malformed plan payload: {exc}") from exc

    if not steps:
        raise PlannerError("planner returned an empty step list")
    return Plan(task=task, steps=steps)


# ============================================================== Worker


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


def execute(
    llm: LLM,
    *,
    spec: AgentSpec,
    step: Step,
    plan_context: str,
    tools: ToolRegistry,
    max_turns: int = 12,
) -> WorkerResult:
    """Run the Anthropic tool-use loop for a single step.

    Returns a :class:`WorkerResult` describing what happened. The kernel is
    responsible for turning the filesystem changes into a commit.
    """
    system = [
        cached(WORKER_SYSTEM),
        cached(f"Agent capability:\n{spec.description}\n\nAgent instructions:\n{spec.system_prompt}"),
        cached(f"Plan so far:\n{plan_context}"),
    ]
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"Current step ({step.id}): {step.description}\n\n"
                "Execute only this step. Stop when it's done."
            ),
        }
    ]

    tool_calls = 0
    summary = ""
    stop_reason = "unknown"

    for _ in range(max_turns):
        response = llm.call(
            model=spec.model or MODEL_WORKER,
            system=system,
            messages=messages,
            tools=tools.to_anthropic(),
            max_tokens=4096,
        )
        stop_reason = response.stop_reason or "unknown"
        assistant_blocks = [_block_to_dict(b) for b in response.content]
        messages.append({"role": "assistant", "content": assistant_blocks})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]
        if text_blocks:
            summary = text_blocks[-1].text

        if stop_reason == "end_turn" or not tool_uses:
            break

        tool_results = []
        for call in tool_uses:
            tool_calls += 1
            try:
                output = tools.invoke(call.name, dict(call.input))
            except Exception as exc:  # surface to the model, don't crash the loop
                output = f"TOOL ERROR ({type(exc).__name__}): {exc}"
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
    )


# ============================================================== Monitor


@dataclass
class MonitorResult:
    passed: bool
    reason: str
    evidence: str = ""

    def format(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.reason}"


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
) -> MonitorResult:
    """Evaluate the current step's verification.

    Shell verifications are executed in ``workspace`` and pass iff exit 0. LLM
    verifications delegate to the Monitor model, which sees the diff produced
    by the latest checkpoint and is asked to judge against natural-language
    criteria. Deterministic checks are preferred because they are cheap,
    reproducible, and immune to the self-praise failure mode.
    """
    v = step.verification
    if v.kind == "shell":
        return _run_shell_check(v.command or "true", workspace)
    if v.kind == "llm":
        if llm is None:
            raise ValueError("LLM verification requested but no LLM provided.")
        return _run_llm_check(v.criteria or "", step, memory, llm, monitor_model)
    raise ValueError(f"unknown verification kind: {v.kind}")


def _run_shell_check(command: str, workspace: Path) -> MonitorResult:
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=180,
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
) -> MonitorResult:
    head = memory.head()
    diff = memory.diff(head.parent_sha or head.sha, head.sha) if head.parent_sha else ""
    system = [cached(MONITOR_SYSTEM)]
    messages = [
        {
            "role": "user",
            "content": (
                f"Step: {step.id} — {step.description}\n\n"
                f"Criteria:\n{criteria}\n\n"
                f"Diff of the Worker's checkpoint ({head.short_sha}):\n"
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
        temperature=0.0,
    )
    verdict = _extract_tool_input(response, "report_verdict")
    return MonitorResult(
        passed=bool(verdict["passed"]),
        reason=str(verdict["reason"]),
        evidence=f"judged on diff {head.short_sha}",
    )


# ============================================================== helpers


def _extract_tool_input(response, tool_name: str) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            return dict(block.input)
    raise RuntimeError(
        f"model did not call `{tool_name}` (stop_reason={response.stop_reason})"
    )


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Convert an Anthropic SDK content block back into the raw dict form."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    # Thinking and other block types pass through via model_dump
    return block.model_dump()


def _tail(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "... (truncated) ...\n" + "\n".join(lines[-max_lines:])
