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

from taste.agents import HOSTED_AGENTS
from taste.brains import benchmark_reply
from taste.brains.azure_execution_policy import WORKER_EFFORTS, AzureExecutionPolicy
from taste.brains.branch_replay import MODES, OVERRIDES, BranchPolicy
from taste.brains.central_planner import SPEND_CAP_KEY, Goal
from taste.brains.single_run import FIXED_PLAN_MODEL
from taste.brains.terminal_broker import TerminalBinding
from taste.brains.terminal_worker_policy import TerminalWorkerPolicy
from taste.pricing import max_call_cost_usd, table_sha
from taste.providers.azure_openai import (
    AZURE_LUNA6_MODEL,
    AZURE_LUNA_MODEL,
    AZURE_PLANNER_MODEL,
    AZURE_WORKER_MODEL,
)

# Names a benchmark command line uses, and the dated served model behind each.
# Any of them may run every role; the trial's model is the coordinator's.
SERVED_MODELS = {"gpt-6-astra": AZURE_PLANNER_MODEL, "gpt-6-sol": AZURE_WORKER_MODEL,
                 "gpt-5.6-luna": AZURE_LUNA_MODEL, "gpt-6-luna": AZURE_LUNA6_MODEL}
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
    # An agent written by others that every worker runs unchanged (taste.agents);
    # empty for Taste's own worker.
    agent: str = ""
    # "none" runs that agent alone: the task as given, with no planner (a
    # fixed rule plans), no monitor and no certification.
    services: str = "all"
    # With services off: "once" runs the agent one time; "continue" (#34)
    # runs it again in the environment the last run left, given the task as
    # given each time, until the generation bound or the task's time runs
    # out: the supervised arm's attempts without its supervision.
    alone: str = "once"
    # "on": the coordinator may return the task's files to a checkpoint taken
    # before the first worker or after a run, when the evidence shows later
    # work broke them (taste.brains.environment_records).
    rollback: str = "off"
    # A branch of an earlier run of the agent alone (taste.brains.branch_replay):
    # the replay script (scripts/replay_script.py) and the step k it starts
    # from. "rebuild" runs the recorded commands 1..k again in the fresh task
    # container; "restore" puts back the checkpoint in branch_checkpoint, which
    # a rebuild with branch_live=off saved there. The agent's first k steps are
    # answered from the record; then it goes on live, or (off) is stopped so
    # that the verifier grades the files of step k. branch_override changes
    # what step k showed: "append" adds the text in branch_note after its
    # output; "reject", at the submission step, shows that text with exit 1
    # instead. branch_tolerance rebuilt commands may diverge from the record.
    # The agent's context is replayed when the trial's agent is the one the
    # record is of (branch_replay=auto; on or off to say so). Otherwise only
    # the files of step k come back and the trial's own agent starts fresh
    # with its own task text, as a checker given a run's final files does
    # (step k the run's last, its files rebuilt or restored). A checkpoint is
    # a directory path or an ID under the checkpoints root.
    branch: str = ""
    branch_step: int = 0
    branch_mode: str = "rebuild"
    branch_checkpoint: str = ""
    branch_live: str = "on"
    branch_note: str = ""
    branch_override: str = ""
    branch_tolerance: int = 0
    branch_replay: str = "auto"
    # For a fresh trial (or an agent given only a run's files): the task's
    # text replaced by this file's, and this file's text added after it (after
    # a blank line).
    task_text: str = ""
    task_suffix: str = ""
    # Where the container's files are saved when the trial ends, before its
    # verifier runs (a directory path or an ID): the run's final state, which
    # a checker's trial restores (branch_mode=restore, branch_step = the run's
    # last step).
    final_checkpoint: str = ""
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
    # The longest any one model request may take. In 257 recorded calls the
    # slowest worker reply took 36 seconds and the slowest plan 40; a reply of
    # the full output allowance takes about 290 at the slowest speed seen.
    request_seconds: float = 300.0
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
        for name in ("spend_cap_usd", "worker_spend_cap_usd", "monitor_spend_cap_usd",
                     "command_seconds", "request_seconds", "worker_grace_seconds",
                     "reply_reserve_seconds", "plan_seconds", "handoff_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.lost_workers < 0 or self.max_assignments < 1:
            raise ValueError("lost_workers and max_assignments must be nonnegative and positive")
        # Refused when the agent is built, not when the first plan is made.
        if self.worker_effort not in WORKER_EFFORTS:
            raise ValueError("worker reasoning effort must be low, medium or high")
        if self.agent and self.agent not in HOSTED_AGENTS:
            raise ValueError("agent must be one of: " + ", ".join(sorted(HOSTED_AGENTS)))
        if self.services not in ("all", "none"):
            raise ValueError("services must be all or none")
        if self.services == "none" and not self.agent:
            raise ValueError("services can be off only for a hosted agent")
        if self.alone not in ("once", "continue"):
            raise ValueError("alone must be once or continue")
        if self.alone == "continue" and self.services != "none":
            raise ValueError("alone=continue runs the agent alone: it needs services none")
        if self.rollback not in ("off", "on"):
            raise ValueError("rollback must be off or on")
        if self.rollback == "on" and self.services != "all":
            raise ValueError("rollback=on is decided by the coordinator: it needs services all")
        if self.planner_effort not in ("", *WORKER_EFFORTS):
            raise ValueError("planner reasoning effort must be low, medium or high, or empty for the default")
        self._check_branch()

    def _check_branch(self):
        given = {name for name, default in _BRANCH_DEFAULTS.items() if getattr(self, name) != default}
        if not self.branch:
            if given:
                raise ValueError(", ".join(sorted(given)) + " needs branch=<replay script>")
            return
        if self.services != "none" or self.alone != "once":
            raise ValueError("a branch continues a run of the agent alone: it needs services none, alone once")
        if self.branch_step < 0 or self.branch_tolerance < 0:
            raise ValueError("branch_step and branch_tolerance must be nonnegative")
        if self.branch_mode not in MODES:
            raise ValueError("branch_mode must be rebuild or restore")
        if self.branch_live not in ("on", "off"):
            raise ValueError("branch_live must be on or off")
        if self.branch_replay not in ("auto", "on", "off"):
            raise ValueError("branch_replay must be auto, on or off")
        if self.branch_override not in ("", *OVERRIDES):
            raise ValueError("branch_override must be append or reject")
        if bool(self.branch_override) != bool(self.branch_note):
            raise ValueError("branch_override and branch_note go together")
        if self.branch_override and self.branch_live == "off":
            raise ValueError("an override changes what the agent sees next: it needs branch_live=on")
        if self.branch_override and self.branch_replay == "off":
            raise ValueError("an override changes what the replayed agent sees: it needs branch_replay=on")
        if self.branch_mode == "restore" and not self.branch_checkpoint:
            raise ValueError("branch_mode=restore needs branch_checkpoint=<directory>")
        if self.branch_mode == "rebuild" and self.branch_checkpoint and self.branch_live != "off":
            raise ValueError("a rebuild saves its checkpoint of step k only with branch_live=off")
        if (self.task_text or self.task_suffix) and self.branch_replay == "on":
            raise ValueError("a replayed agent is given its record's task; task_text and task_suffix are for "
                             "fresh trials and for an agent given only the files")

    @property
    def models(self):
        """(coordinator short name, its served model, worker short name, its served model).

        An agent run alone has the fixed plan as its coordinator, and the
        trial's model is the agent's.
        """
        coordinator = ((FIXED_PLAN_MODEL, FIXED_PLAN_MODEL) if self.services == "none"
                       else served_model(self.model))
        return (*coordinator, *served_model(self.worker_model or self.model))

    @property
    def generations(self):
        """The goal's generation bound: one run when the agent runs alone once."""
        return 1 if self.services == "none" and self.alone == "once" else self.max_generations

    def budgets(self):
        """(worker cap, monitor cap, goal budget) in USD, each including its worst-case call."""
        _, planner, _, worker = self.models
        # Arms are compared at equal cost: an agent run alone may spend what a
        # whole supervised trial may, which there the planner, the workers and
        # their monitors share.
        allowance = self.spend_cap_usd if self.services == "none" else self.worker_spend_cap_usd
        worker_cap = allowance + max_call_cost_usd(
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

    def branch_policy(self, script, script_sha256, note="", *, replay=True):
        """The worker's branch: the replay script as the worker reads it (in the trial) and its digest.

        ``replay`` is the decision ``branch_trial.load_inputs`` made from the
        record: whether the agent's context is replayed or only the files return.
        """
        return BranchPolicy(script=str(script), script_sha256=script_sha256, step=self.branch_step,
                            mode=self.branch_mode, live=self.branch_live == "on",
                            override=self.branch_override, note=note, tolerance=self.branch_tolerance,
                            replay=replay)

    def policy(self, endpoint, deadline_unix, *, owner_token, container_id, workdir, branch=None):
        coordinator, planner, worker_name, worker = self.models
        worker_cap, monitor_cap, _ = self.budgets()
        binding = TerminalBinding(owner_token, container_id, deadline_unix, self.max_commands)
        planner_deployment = ("fixed-plan" if self.services == "none"
                              else self.deployment or coordinator)
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
            planner_model=planner, worker_model=worker, worker_effort=self.worker_effort,
            worker_agent=self.agent, services=self.services, rollback=self.rollback == "on",
            planner_effort=self.planner_effort, request_seconds=self.request_seconds,
            worker_grace_seconds=self.worker_grace_seconds,
            # A worker may use all the working time; the runtime clamps it to what is left.
            worker_wall_seconds=604800.0, max_assignments=self.max_assignments, branch=branch,
        )

    def goal(self, goal_id, instruction):
        return Goal(goal_id=goal_id, task=instruction, success_criteria=(CRITERION,),
                    budget_usd=self.budgets()[2],
                    metadata={benchmark_reply.KEY: benchmark_reply.SCHEMA,
                              benchmark_reply.RESERVE_KEY: self.reply_reserve_seconds,
                              benchmark_reply.PLAN_KEY: self.plan_seconds,
                              SPEND_CAP_KEY: self.spend_cap_usd})

    def disclosure(self):
        """The configuration a result should be reported with."""
        _, served, _, worker_served = self.models
        worker_cap, monitor_cap, goal = self.budgets()
        return {"coordinator_model": served, "worker_model": worker_served,
                "monitor_model": worker_served, "worker_effort": self.worker_effort,
                "worker_agent": self.agent or "taste", "services": self.services,
                **({"alone": self.alone} if self.services == "none" else {}),
                **({"rollback": self.rollback} if self.rollback != "off" else {}),
                **({"branch": {"script": self.branch, "step": self.branch_step, "mode": self.branch_mode,
                               "live": self.branch_live, "tolerance": self.branch_tolerance,
                               **({"replay": self.branch_replay} if self.branch_replay != "auto" else {}),
                               **({"checkpoint": self.branch_checkpoint} if self.branch_checkpoint else {}),
                               **({"override": self.branch_override, "note": self.branch_note}
                                  if self.branch_override else {})}} if self.branch else {}),
                **({"task_text": self.task_text} if self.task_text else {}),
                **({"task_suffix": self.task_suffix} if self.task_suffix else {}),
                **({"final_checkpoint": self.final_checkpoint} if self.final_checkpoint else {}),
                "coordinator_effort": self.planner_effort or "provider default",
                "monitor_effort": "low",
                "spend_cap_usd": self.spend_cap_usd, "admission_budgets_usd": {
                    "worker": worker_cap, "monitor": monitor_cap, "goal": goal},
                "max_assignments_per_plan": self.max_assignments,
                "reply_reserve_seconds": self.reply_reserve_seconds,
                "plan_seconds": self.plan_seconds,
                "handoff_seconds": self.handoff_seconds, "command_seconds": self.command_seconds,
                "request_seconds": self.request_seconds,
                "max_request_bytes": self.max_request_bytes}


# The branch's own settings, which mean nothing without a branch.
_BRANCH_DEFAULTS = {item.name: item.default for item in fields(TrialSettings) if item.name.startswith("branch_")}


def task_instruction(settings, instruction):
    """The task's text as this trial gives it: replaced by task_text, then task_suffix added after it."""
    text = Path(settings.task_text).read_text(encoding="utf-8") if settings.task_text else instruction
    if settings.task_suffix:
        suffix = Path(settings.task_suffix).read_text(encoding="utf-8")
        if not suffix.strip():
            raise ValueError("the task suffix is empty")
        text = text.rstrip("\n") + "\n\n" + suffix
    if not text.strip():
        raise ValueError("the task's text is empty")
    return text


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
