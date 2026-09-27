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
    AZURE_PLANNER_MODEL,
    AZURE_WORKER_MODEL,
    AzureDeployment,
    AzureOpenAIConfig,
)

POLICY_KEY = "azure_execution"


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

    def __post_init__(self):
        if self.terminal is not None and (
                not isinstance(self.terminal, TerminalWorkerPolicy)
                or self.terminal.binding.deadline_unix != self.deadline_unix):
            raise ValueError("terminal policy must share the original Azure goal deadline")
        route = self.azure_config({"AZURE_OPENAI_BASE_URL": self.endpoint,
                                   "AZURE_OPENAI_API_KEY": "policy-validation-only"})
        if route.base_url != self.endpoint:
            raise ValueError("Azure policy endpoint must have its canonical trailing slash")
        if self.pricing_sha != table_sha():
            raise ValueError("Azure policy requires the exact admitted pricing table")
        if type(self.monitor_batch_size) is not int or not 1 <= self.monitor_batch_size <= 1000:
            raise ValueError("monitor batch size must be between 1 and 1000")
        for role in ("worker", "monitor", "planner"):
            model = AZURE_PLANNER_MODEL if role == "planner" else AZURE_WORKER_MODEL
            budget = self.worker_budget_usd if role == "planner" else getattr(self, f"{role}_budget_usd")
            tokens = getattr(self, f"{role}_max_output_tokens")
            ResponsesBinding(
                run_id="policy-validation", model=model, endpoint=self.endpoint,
                deployment=self.planner_deployment if role == "planner" else self.worker_deployment,
                budget_usd=budget, max_calls=1 if role == "planner" else getattr(self, f"{role}_max_calls"),
                max_output_tokens=tokens, max_request_bytes=self.max_request_bytes,
                deadline_unix=self.deadline_unix, role=role,
            )
            if role != "planner" and budget < max_call_cost_usd(model, max_output_tokens=tokens, cap_on="billed"):
                raise ValueError(f"{role} budget cannot admit even one bounded call")
        for name in ("deadline_unix", "worker_budget_usd", "monitor_budget_usd"):
            object.__setattr__(self, name, float(getattr(self, name)))

    def to_dict(self):
        value = asdict(self)
        value.pop("terminal")
        if self.terminal is None:
            return {"schema": "taste.brains/AzureExecutionPolicy/1", **value}
        return {"schema": "taste.brains/AzureExecutionPolicy/2", **value,
                "terminal": self.terminal.to_dict()}

    @classmethod
    def from_dict(cls, value):
        names = {item.name for item in fields(cls)} - {"terminal"}
        if not isinstance(value, Mapping):
            raise ValueError("invalid Azure execution policy fields or schema")
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
            AzureDeployment(AZURE_WORKER_MODEL, self.worker_deployment),
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
        }

    def configure_prompt(self, payload):
        exemplar = payload["required_output_shape"]["assignments"][0]
        exemplar["model"] = AZURE_WORKER_MODEL
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
        )
        if self.terminal is not None:
            exemplar["resources"][TERMINAL_POLICY_KEY] = self.terminal.to_dict()
            capabilities = payload["rules"]["worker_capabilities"]
            capabilities["can"].extend(["run bounded commands in the shared task container",
                                         "read this worker's retained terminal output pages"])
            capabilities["cannot"] = ["run commands on the controller host", "read undeclared memory artifacts"]
            capabilities["terminal_effects"] = (
                "All workers share one task container with serial terminal actions. Files, packages and services "
                "persist across successful commands and memory rollback. Active command timeout/cancellation "
                "ends the task environment. Official benchmark grading remains the outside lifecycle owner's job.")

    def validate_assignment(self, assignment):
        allowed = {"azure_openai", "monitor_budget_usd", "wall_timeout_seconds"}
        if self.terminal is not None:
            allowed.add(TERMINAL_POLICY_KEY)
            if assignment.resources.get(TERMINAL_POLICY_KEY) != self.terminal.to_dict():
                raise ValueError("assignment differs from the goal's terminal policy")
        if (assignment.resources.get("azure_openai") != self.worker_resources()
                or assignment.resources.get("monitor_budget_usd") != self.monitor_budget_usd
                or assignment.contract.budget_usd != self.worker_budget_usd
                or assignment.contract.max_turns != self.worker_max_calls
                or set(assignment.resources) - allowed):
            raise ValueError("assignment differs from the goal's exact Azure execution policy")
        try:
            AzureWorkerPolicy.from_assignment(assignment)
        except EntrypointInputError as exc:
            raise ValueError("assignment cannot be admitted by the Azure worker") from exc
