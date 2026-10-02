"""Trusted, non-secret Azure goal and assignment policy.

The complete policy is part of the immutable Goal. Planner output must echo
the admitted worker limits; it cannot choose a provider route or renew a
deadline. Credentials are read separately from the trusted host environment.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace

from taste.brains.azure_worker_policy import AZURE_WORKER_POLICY_SCHEMA, AzureWorkerPolicy
from taste.brains.responses_session import ResponsesBinding
from taste.brains.terminal_worker_policy import TERMINAL_POLICY_KEY, TerminalWorkerPolicy
from taste.brains.worker_admission import EntrypointInputError
from taste.pricing import max_call_cost_usd, table_sha
from taste.providers.azure_openai import (
    AZURE_MODELS,
    AZURE_PLANNER_MODEL,
    AZURE_WORKER_MODEL,
    AzureDeployment,
    AzureOpenAIConfig,
)

WORKER_EFFORTS = ("low", "medium", "high")

POLICY_KEY = "azure_execution"
_ORIGINAL_CHOICES = {"worker_model": AZURE_WORKER_MODEL, "worker_effort": "low",
                     "worker_grace_seconds": 2.0, "worker_wall_seconds": 900.0, "max_assignments": None,
                     "planner_effort": "", "request_seconds": None}


@dataclass(frozen=True)
class AzureExecutionPolicy:
    endpoint: str
    planner_deployment: str
    worker_deployment: str
    deadline_unix: float
    worker_budget_usd: float
    monitor_budget_usd: float
    worker_max_calls: int
    monitor_max_calls: int
    worker_max_output_tokens: int = 8192
    monitor_max_output_tokens: int = 2048
    planner_max_output_tokens: int = 8192
    max_request_bytes: int = 196608
    monitor_batch_size: int = 8
    pricing_sha: str = ""
    terminal: TerminalWorkerPolicy | None = None
    # The served model workers and their monitors run on, and how hard workers
    # reason. Both are part of the trial's disclosed configuration. A worker
    # on the planner's model uses the planner's deployment: one model, one route.
    worker_model: str = AZURE_WORKER_MODEL
    worker_effort: str = "low"
    # How long a stopped worker may take to settle its active model call and
    # write its report before it is killed. A killed worker's exact cost is
    # lost, so a trial that can afford the wait should allow one call's length.
    worker_grace_seconds: float = 2.0
    # A worker's allowance when its assignment names none, inside the goal's.
    worker_wall_seconds: float = 900.0
    # The most assignments one plan may hold; None leaves it to the planner.
    max_assignments: int | None = None
    # The coordinator's reasoning effort. Empty names none and leaves the
    # provider's default.
    planner_effort: str = ""
    # The longest any one model request may take, for every role. None leaves
    # each request all the time its goal has left: one the service never
    # answers then holds its worker, or the coordinator, until the goal ends.
    request_seconds: float | None = None

    def __post_init__(self):
        if self.terminal is not None and (
                not isinstance(self.terminal, TerminalWorkerPolicy)
                or self.terminal.binding.deadline_unix != self.deadline_unix):
            raise ValueError("terminal policy must share the original Azure goal deadline")
        if self.worker_model not in AZURE_MODELS:
            raise ValueError("worker model has no verified Azure deployment and price")
        if (self.worker_model == AZURE_PLANNER_MODEL) != (self.worker_deployment == self.planner_deployment):
            raise ValueError("one served model must use exactly one deployment")
        if self.worker_effort not in WORKER_EFFORTS:
            raise ValueError("worker reasoning effort must be low, medium or high")
        if self.planner_effort not in ("", *WORKER_EFFORTS):
            raise ValueError("planner reasoning effort must be low, medium or high, or empty for the default")
        for name, ceiling in (("worker_grace_seconds", 600), ("worker_wall_seconds", 604800)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 < value <= ceiling:
                raise ValueError(f"{name} must be positive and at most {ceiling} seconds")
            object.__setattr__(self, name, float(value))
        if self.max_assignments is not None and (
                type(self.max_assignments) is not int or not 1 <= self.max_assignments <= 64):
            raise ValueError("max_assignments must be between 1 and 64")
        if self.request_seconds is not None:
            if type(self.request_seconds) not in (int, float) or not 0 < self.request_seconds <= 3600:
                raise ValueError("request_seconds must be positive and at most 3600 seconds")
            object.__setattr__(self, "request_seconds", float(self.request_seconds))
        route = self.azure_config({"AZURE_OPENAI_BASE_URL": self.endpoint,
                                   "AZURE_OPENAI_API_KEY": "policy-validation-only"})
        if route.base_url != self.endpoint:
            raise ValueError("Azure policy endpoint must have its canonical trailing slash")
        if self.pricing_sha != table_sha():
            raise ValueError("Azure policy requires the exact admitted pricing table")
        if type(self.monitor_batch_size) is not int or not 1 <= self.monitor_batch_size <= 1000:
            raise ValueError("monitor batch size must be between 1 and 1000")
        for role in ("worker", "monitor", "planner"):
            model = AZURE_PLANNER_MODEL if role == "planner" else self.worker_model
            budget = self.worker_budget_usd if role == "planner" else getattr(self, f"{role}_budget_usd")
            tokens = getattr(self, f"{role}_max_output_tokens")
            ResponsesBinding(
                run_id="policy-validation", model=model, endpoint=self.endpoint,
                deployment=self.planner_deployment if role == "planner" else self.worker_deployment,
                budget_usd=budget, max_calls=1 if role == "planner" else getattr(self, f"{role}_max_calls"),
                max_output_tokens=tokens, max_request_bytes=self.max_request_bytes,
                deadline_unix=self.deadline_unix, role=role, request_seconds=self.request_seconds,
            )
            if role != "planner" and budget < max_call_cost_usd(model, max_output_tokens=tokens, cap_on="billed"):
                raise ValueError(f"{role} budget cannot admit even one bounded call")
        for name in ("deadline_unix", "worker_budget_usd", "monitor_budget_usd"):
            object.__setattr__(self, name, float(getattr(self, name)))

    def to_dict(self):
        value = asdict(self)
        value.pop("terminal")
        # Written only when they differ from the original fixed choices, so
        # earlier policies keep their exact wire form and digests.
        for name, original in _ORIGINAL_CHOICES.items():
            if value[name] == original:
                value.pop(name)
        if self.terminal is None:
            return {"schema": "taste.brains/AzureExecutionPolicy/1", **value}
        return {"schema": "taste.brains/AzureExecutionPolicy/2", **value,
                "terminal": self.terminal.to_dict()}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping):
            raise ValueError("invalid Azure execution policy fields or schema")
        chosen = set(_ORIGINAL_CHOICES) & set(value)
        if any(value[name] == _ORIGINAL_CHOICES[name] for name in chosen):
            raise ValueError("invalid Azure execution policy fields or schema")
        names = ({item.name for item in fields(cls)} - {"terminal"} - set(_ORIGINAL_CHOICES)) | chosen
        if value.get("schema") == "taste.brains/AzureExecutionPolicy/1" and set(value) == {"schema", *names}:
            return cls(**{key: value[key] for key in names})
        if value.get("schema") == "taste.brains/AzureExecutionPolicy/2" and set(value) == {"schema", "terminal", *names}:
            return cls(**{key: value[key] for key in names}, terminal=TerminalWorkerPolicy.from_dict(value["terminal"]))
        raise ValueError("invalid Azure execution policy fields or schema")

    def bind_goal(self, goal):
        from taste.brains.central_planner import Goal

        if not isinstance(goal, Goal) or goal.budget_usd is None or goal.budget_usd <= 0:
            raise ValueError("Azure execution requires a Goal with a positive global budget")
        prior = goal.metadata.get(POLICY_KEY)
        if prior is not None and prior != self.to_dict():
            raise ValueError("goal already binds another Azure execution policy")
        return replace(goal, metadata={**goal.metadata, POLICY_KEY: self.to_dict()})

    def azure_config(self, environment):
        result = AzureOpenAIConfig.from_environment(environment, deployments=(
            AzureDeployment(AZURE_PLANNER_MODEL, self.planner_deployment),
            *(() if self.worker_model == AZURE_PLANNER_MODEL
              else (AzureDeployment(self.worker_model, self.worker_deployment),)),
        ))
        if result.base_url != self.endpoint:
            raise ValueError("host Azure endpoint differs from the execution policy")
        return result

    def worker_resources(self):
        return {
            "schema": AZURE_WORKER_POLICY_SCHEMA, "endpoint": self.endpoint,
            "worker_deployment": self.worker_deployment, "monitor_deployment": self.worker_deployment,
            "deadline_unix": self.deadline_unix, "worker_max_calls": self.worker_max_calls,
            "worker_max_output_tokens": self.worker_max_output_tokens,
            "monitor_max_calls": self.monitor_max_calls, "monitor_max_output_tokens": self.monitor_max_output_tokens,
            "max_request_bytes": self.max_request_bytes, "monitor_batch_size": self.monitor_batch_size,
            "pricing_sha": self.pricing_sha,
            **({} if self.worker_effort == "low" else {"worker_effort": self.worker_effort}),
            **({} if self.request_seconds is None else {"request_seconds": self.request_seconds}),
        }

    def configure_prompt(self, payload):
        exemplar = payload["required_output_shape"]["assignments"][0]
        exemplar["model"] = self.worker_model
        exemplar["contract"].update(budget_usd=self.worker_budget_usd, max_turns=self.worker_max_calls)
        exemplar["resources"].update(monitor_budget_usd=self.monitor_budget_usd, azure_openai=self.worker_resources())
        payload["rules"].update(
            minimum_contract_budget_usd=self.worker_budget_usd,
            minimum_monitor_budget_usd=self.monitor_budget_usd,
            budget_floor_reason="Worker and monitor caps, call limits and Azure routing must exactly match the exemplar.",
            worker_capabilities={
                "can": ["read declared input/output artifacts", "write or remove declared output artifacts"],
                "cannot": ["run shell commands", "read undeclared files", "access a task terminal"],
                "harness_does_for_it": "checkpoint, certify and deliver the named outputs to integration",
            },
            azure_execution_policy=self.to_dict(),
            worker_context=(
                "Every worker is shown the goal's original task, verbatim, before its "
                "assignment. contract.task should say what that worker must do and check; "
                "it need not restate the task."
            ),
        )
        # The runtime bounds every worker by the goal's own remaining time. A
        # shorter allowance copied from an example only cuts good work short.
        exemplar["resources"].pop("wall_timeout_seconds", None)
        if self.max_assignments is not None:
            payload["rules"]["most_assignments_per_plan"] = self.max_assignments
        if self.terminal is not None:
            exemplar["resources"][TERMINAL_POLICY_KEY] = self.terminal.to_dict()
            capabilities = payload["rules"]["worker_capabilities"]
            capabilities["can"].extend(["run bounded commands in the shared task container",
                                         "read this worker's retained terminal output pages"])
            capabilities["cannot"] = ["run commands on the controller host", "read undeclared memory artifacts"]
            payload["rules"]["unreported_work"] = (
                "A trigger's work_record, and the closing trigger's unreported_work, list the terminal "
                "commands a worker ran before it ended without a report: each command, its exit and the "
                "last line it printed, read from that worker's recorded turns. A command marked as started "
                "with no result may or may not have finished. These commands did run in the task container "
                "and may have changed it. The record is mechanical: it is not a claim by the worker and "
                "verifies nothing beyond what its lines show.")
            capabilities["terminal_effects"] = (
                "All workers share one task container and its terminal runs one command at a time. What a "
                "command changes there (files, packages, services) persists: later workers see it, and "
                "rolling memory back does not undo it. A command that runs past its timeout is killed with "
                "its child processes; the container and everything else in it stay as they are. The "
                "benchmark's own grading happens after this goal ends and is not a worker's job.")

    def validate_plan(self, assignments):
        if self.max_assignments is not None and len(assignments) > self.max_assignments:
            raise ValueError(
                f"this goal admits at most {self.max_assignments} assignment(s) per plan; "
                "issue the next one in a later plan revision")

    def validate_assignment(self, assignment):
        allowed = {"azure_openai", "monitor_budget_usd", "wall_timeout_seconds"}
        if self.terminal is not None:
            allowed.add(TERMINAL_POLICY_KEY)
            if assignment.resources.get(TERMINAL_POLICY_KEY) != self.terminal.to_dict():
                raise ValueError("assignment differs from the goal's terminal policy")
        if (assignment.resources.get("azure_openai") != self.worker_resources()
                or assignment.model != self.worker_model
                or assignment.resources.get("monitor_budget_usd") != self.monitor_budget_usd
                or assignment.contract.budget_usd != self.worker_budget_usd
                or assignment.contract.max_turns != self.worker_max_calls
                or set(assignment.resources) - allowed):
            raise ValueError("assignment differs from the goal's exact Azure execution policy")
        try:
            AzureWorkerPolicy.from_assignment(assignment)
        except EntrypointInputError as exc:
            raise ValueError("assignment cannot be admitted by the Azure worker") from exc
