"""What a benchmark trial needs decided before it starts, without importing Harbor.

A Harbor agent receives one instruction and one live task container. Everything
else here is this system's own choice and is disclosed with the run: which
served model each role uses, how hard workers reason, how much a trial may
spend, and how its fixed time is divided between work, the coordinator's
closing reply and the handoff to the benchmark's verifier.

Budgets are written as what a role may actually spend. Admission reserves a
call's worst case before dispatching it, so each cap is that allowance plus one
worst-case call: the role stops once its real spending passes the allowance.
"""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from taste.brains import benchmark_reply
from taste.brains.azure_execution_policy import WORKER_EFFORTS, AzureExecutionPolicy
from taste.brains.central_planner import Goal
from taste.brains.terminal_broker import TerminalBinding
from taste.brains.terminal_worker_policy import TerminalWorkerPolicy
from taste.pricing import max_call_cost_usd, table_sha
from taste.providers.azure_openai import AZURE_PLANNER_MODEL, AZURE_WORKER_MODEL

# Names a benchmark command line uses, and the dated served model behind each.
SERVED_MODELS = {"gpt-6-astra": AZURE_PLANNER_MODEL, "gpt-6-sol": AZURE_WORKER_MODEL}
CRITERION = ("The developer's most recent request in the task is addressed in the task "
             "environment, and the final reply says accurately what was done, what was "
             "verified and what was not.")


def served_model(name):
    """``provider/model`` or ``model`` as a benchmark names it -> (short name, served model)."""
    short = str(name or "").rsplit("/", 1)[-1]
    if short not in SERVED_MODELS:
        raise ValueError("model must be one of: " + ", ".join(sorted(SERVED_MODELS)))
    return short, SERVED_MODELS[short]


@dataclass(frozen=True)
class TrialSettings:
    model: str = "gpt-6-astra"
    worker_model: str = ""          # empty: every role uses ``model``
    deployment: str = ""            # empty: the deployment is named after its model
    worker_deployment: str = ""
    worker_effort: str = "low"
    # The coordinator writes every contract and the final reply. Its effort is
    # named so that a run discloses it. Measured on gpt-6-astra the level moves
    # little: 47 to 70 reasoning tokens on one small puzzle from low to high,
    # and 0 to 87 per planner call at both the default and medium.
    planner_effort: str = "medium"
    spend_cap_usd: float = 15.0
    worker_spend_cap_usd: float = 6.0
    monitor_spend_cap_usd: float = 2.0
    lost_workers: int = 2           # killed workers whose full caps the budget can still carry
    worker_max_calls: int = 150
    monitor_max_calls: int = 80
    # Reasoning counts against these. A worker writes whole files in one
    # command; a reply that reaches its limit runs nothing and is redone.
    worker_max_output_tokens: int = 16384
    monitor_max_output_tokens: int = 4096
    planner_max_output_tokens: int = 16384
    max_request_bytes: int = 1_048_576
    monitor_batch_size: int = 8
    max_generations: int = 12
    max_planner_failures: int = 4
    max_assignments: int = 1
    max_commands: int = 600
    command_seconds: float = 600.0
    worker_grace_seconds: float = 45.0
    # Held back for the closing reply: a stopped worker's grace, then one
    # reasoning planner call (measured at 40 to 60 seconds without reasoning).
    reply_reserve_seconds: float = 210.0
    # No new plan is started with less working time than this: it could not be
    # answered and acted on, and the reserve is better spent on the reply.
    plan_seconds: float = 90.0
    handoff_seconds: float = 150.0

    @classmethod
    def from_options(cls, options):
        """Agent keyword arguments, as strings from a command line or typed values."""
        known = {item.name: item.type for item in fields(cls)}
        values = {}
        for name, raw in options.items():
            if name not in known:
                raise ValueError(f"unknown trial setting {name!r}")
            kind = known[name]
            try:
                values[name] = (str(raw) if kind == "str" else int(raw) if kind == "int"
                                else float(raw))
            except (TypeError, ValueError):
                raise ValueError(f"trial setting {name!r} has the wrong type") from None
        return cls(**values)

    def __post_init__(self):
        served_model(self.model)
        served_model(self.worker_model or self.model)
        if served_model(self.model)[1] != AZURE_PLANNER_MODEL:
            raise ValueError("the coordinator runs on gpt-6-astra; name it as the trial's model")
        for name in ("spend_cap_usd", "worker_spend_cap_usd", "monitor_spend_cap_usd",
                     "command_seconds", "worker_grace_seconds", "reply_reserve_seconds",
                     "plan_seconds", "handoff_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.lost_workers < 0 or self.max_assignments < 1:
            raise ValueError("lost_workers and max_assignments must be nonnegative and positive")
        # Refused when the agent is built, not when the first plan is made.
        if self.worker_effort not in WORKER_EFFORTS:
            raise ValueError("worker reasoning effort must be low, medium or high")
        if self.planner_effort not in ("", *WORKER_EFFORTS):
            raise ValueError("planner reasoning effort must be low, medium or high, or empty for the default")

    @property
    def models(self):
        """(coordinator short name, its served model, worker short name, its served model)."""
        coordinator = served_model(self.model)
        return (*coordinator, *served_model(self.worker_model or self.model))

    def budgets(self):
        """(worker cap, monitor cap, goal budget) in USD, each including its worst-case call."""
        _, planner, _, worker = self.models
        worker_cap = self.worker_spend_cap_usd + max_call_cost_usd(
            worker, max_output_tokens=self.worker_max_output_tokens, cap_on="billed")
        monitor_cap = self.monitor_spend_cap_usd + max_call_cost_usd(
            worker, max_output_tokens=self.monitor_max_output_tokens, cap_on="billed")
        planner_call = max_call_cost_usd(
            planner, max_output_tokens=self.planner_max_output_tokens, cap_on="billed")
        # One live worker's reservation, room for the killed ones that stay
        # charged at their caps, and one planner call beyond the spend cap.
        goal = (self.spend_cap_usd + (1 + self.lost_workers) * (worker_cap + monitor_cap)
                + planner_call)
        return round(worker_cap, 6), round(monitor_cap, 6), round(goal, 6)

    def deadlines(self, started_unix, agent_timeout_seconds):
        """(goal deadline, container deadline) inside a benchmark's fixed agent time."""
        goal_seconds = agent_timeout_seconds - self.handoff_seconds
        if goal_seconds <= self.reply_reserve_seconds + max(60.0, self.plan_seconds):
            raise ValueError("the agent's time is too short for work, a closing reply and handoff")
        return started_unix + goal_seconds, started_unix + agent_timeout_seconds + 3600

    def policy(self, endpoint, deadline_unix, *, owner_token, container_id, workdir):
        coordinator, _, worker_name, worker = self.models
        worker_cap, monitor_cap, _ = self.budgets()
        binding = TerminalBinding(owner_token, container_id, deadline_unix, self.max_commands)
        planner_deployment = self.deployment or coordinator
        # One served model has one route: a worker on the coordinator's model
        # shares its deployment unless another is named for it.
        worker_deployment = self.worker_deployment or (
            planner_deployment if worker_name == coordinator else worker_name)
        return AzureExecutionPolicy(
            endpoint=endpoint, planner_deployment=planner_deployment,
            worker_deployment=worker_deployment,
            deadline_unix=deadline_unix, worker_budget_usd=worker_cap, monitor_budget_usd=monitor_cap,
            worker_max_calls=self.worker_max_calls, monitor_max_calls=self.monitor_max_calls,
            worker_max_output_tokens=self.worker_max_output_tokens,
            monitor_max_output_tokens=self.monitor_max_output_tokens,
            planner_max_output_tokens=self.planner_max_output_tokens,
            max_request_bytes=self.max_request_bytes, monitor_batch_size=self.monitor_batch_size,
            pricing_sha=table_sha(),
            terminal=TerminalWorkerPolicy(binding, self.command_seconds, workdir),
            worker_model=worker, worker_effort=self.worker_effort,
            planner_effort=self.planner_effort,
            worker_grace_seconds=self.worker_grace_seconds,
            # A worker may use all the working time; the runtime clamps it to what is left.
            worker_wall_seconds=604800.0, max_assignments=self.max_assignments,
        )

    def goal(self, goal_id, instruction):
        return Goal(goal_id=goal_id, task=instruction, success_criteria=(CRITERION,),
                    budget_usd=self.budgets()[2],
                    metadata={benchmark_reply.KEY: benchmark_reply.SCHEMA,
                              benchmark_reply.RESERVE_KEY: self.reply_reserve_seconds,
                              benchmark_reply.PLAN_KEY: self.plan_seconds})

    def disclosure(self):
        """The configuration a result should be reported with."""
        _, served, _, worker_served = self.models
        worker_cap, monitor_cap, goal = self.budgets()
        return {"coordinator_model": served, "worker_model": worker_served,
                "monitor_model": worker_served, "worker_effort": self.worker_effort,
                "coordinator_effort": self.planner_effort or "provider default",
                "monitor_effort": "low",
                "spend_cap_usd": self.spend_cap_usd, "admission_budgets_usd": {
                    "worker": worker_cap, "monitor": monitor_cap, "goal": goal},
                "max_assignments_per_plan": self.max_assignments,
                "reply_reserve_seconds": self.reply_reserve_seconds,
                "plan_seconds": self.plan_seconds,
                "handoff_seconds": self.handoff_seconds, "command_seconds": self.command_seconds,
                "max_request_bytes": self.max_request_bytes}


def agent_timeout_seconds(task_toml, override=None):
    """The benchmark's published agent time, or an explicit override of it."""
    if override is not None:
        value = float(override)
    else:
        with open(task_toml, "rb") as handle:
            value = tomllib.load(handle).get("agent", {}).get("timeout_sec")
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("the task publishes no agent timeout and none was supplied")
    return float(value)


def compose_project(session_id):
    """The Compose project Harbor gives an environment (its own sanitising rule)."""
    name = str(session_id).lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    return re.sub(r"[^a-z0-9_-]", "-", name)


def safe_parent(path):
    """Where trial directories are made: writable by the owner alone, traversable by all.

    The goal's unprivileged processes reach their own state inside a trial
    directory, so its ancestors must let them through; only the owner may
    create or replace anything there. The trial owner checks this again.
    """
    path = Path(path)
    path.mkdir(mode=0o755, parents=True, exist_ok=True)
    return path.resolve(strict=True)
