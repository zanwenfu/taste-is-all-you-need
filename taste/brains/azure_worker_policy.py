"""Bind Azure worker spending and routing to its immutable Assignment.

The launch environment supplies the Azure credential, never model selection,
budgets, call limits or a new relative deadline. A reopened run reconstructs
the same Responses bindings from the assignment that was admitted originally.
This is configuration admission; it does not launch a worker or make a call.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from taste.agents import HOSTED_AGENTS
from taste.brains.records import Assignment
from taste.brains.responses_session import ResponsesBinding
from taste.brains.worker_admission import EntrypointConfig, EntrypointInputError
from taste.brains.worker_protocol import _assignment_monitor_budget_usd, assignment_run_id
from taste.pricing import table_sha
from taste.providers.azure_openai import AZURE_MODELS, AzureDeployment, AzureOpenAIConfig
from taste.providers.base import ProtocolFailure

AZURE_WORKER_POLICY_SCHEMA = "taste.brains/AzureWorkerPolicy/1"
_FIELDS = frozenset({
    "schema", "endpoint", "worker_deployment", "monitor_deployment", "deadline_unix",
    "worker_max_calls", "worker_max_output_tokens", "monitor_max_calls", "monitor_max_output_tokens",
    "max_request_bytes", "monitor_batch_size", "pricing_sha",
})


_EFFORTS = ("medium", "high")  # "low" is the original choice and is not written.
# Written only when chosen, so assignments made before them keep their form.
_OPTIONAL = frozenset({"worker_effort", "request_seconds", "worker_agent", "services"})


@dataclass(frozen=True)
class AzureWorkerPolicy:
    worker: ResponsesBinding
    monitor: ResponsesBinding
    monitor_batch_size: int
    effort: str = "low"
    # A hosted agent's name; empty for Taste's own worker.
    agent: str = ""
    # False for an agent run alone: no monitor judges it and nothing certifies it.
    supervised: bool = True

    @classmethod
    def from_assignment(cls, assignment: Assignment) -> AzureWorkerPolicy:
        raw = assignment.resources.get("azure_openai")
        if not isinstance(raw, Mapping) or set(raw) - _OPTIONAL != _FIELDS:
            raise EntrypointInputError("assignment requires one exact Azure worker policy")
        if raw["schema"] != AZURE_WORKER_POLICY_SCHEMA or raw["pricing_sha"] != table_sha():
            raise EntrypointInputError("Azure policy schema or admitted pricing table changed")
        if assignment.model not in AZURE_MODELS:
            raise EntrypointInputError("this Azure worker policy requires a verified dated worker model")
        if "worker_effort" in raw and raw["worker_effort"] not in _EFFORTS:
            raise EntrypointInputError("Azure worker reasoning effort is not admitted")
        if "request_seconds" in raw and type(raw["request_seconds"]) not in (int, float):
            raise EntrypointInputError("Azure request ceiling is not admitted")
        if "worker_agent" in raw and raw["worker_agent"] not in HOSTED_AGENTS:
            raise EntrypointInputError("the assignment names no hosted agent this worker can run")
        if "services" in raw and (raw["services"] != "none" or "worker_agent" not in raw):
            raise EntrypointInputError("only a hosted agent can run without services")
        ceiling = raw.get("request_seconds")
        try:
            monitor_budget = _assignment_monitor_budget_usd(assignment)
            # Worker and monitor currently share one served model. They must
            # therefore use the same deployment, rather than inventing two
            # conflicting routes for one priced model identity.
            if raw["worker_deployment"] != raw["monitor_deployment"]:
                raise ValueError("worker and monitor deployment routes disagree")
            route = AzureOpenAIConfig(raw["endpoint"], "policy-validation-only", (
                AzureDeployment(assignment.model, raw["worker_deployment"]),
            ))
            if route.base_url != raw["endpoint"]:
                raise ValueError("Azure endpoint must use its canonical trailing slash")
            batch = raw["monitor_batch_size"]
            if type(batch) is not int or not 1 <= batch <= 1000:
                raise ValueError("monitor batch size must be an integer from 1 to 1000")
            worker = ResponsesBinding(
                run_id=assignment_run_id(assignment), model=assignment.model,
                endpoint=route.base_url, deployment=raw["worker_deployment"],
                budget_usd=assignment.contract.budget_usd,
                max_calls=raw["worker_max_calls"], max_output_tokens=raw["worker_max_output_tokens"],
                max_request_bytes=raw["max_request_bytes"], deadline_unix=raw["deadline_unix"],
                request_seconds=ceiling,
            )
            monitor = ResponsesBinding(
                run_id=worker.run_id + ".monitor", model=assignment.model, role="monitor",
                endpoint=route.base_url, deployment=raw["monitor_deployment"], budget_usd=monitor_budget,
                max_calls=raw["monitor_max_calls"], max_output_tokens=raw["monitor_max_output_tokens"],
                max_request_bytes=raw["max_request_bytes"], deadline_unix=raw["deadline_unix"],
                request_seconds=ceiling,
            )
            if assignment.contract.max_turns is not None and worker.max_calls > assignment.contract.max_turns:
                raise ValueError("worker call allowance exceeds the contract turn limit")
        except (ValueError, TypeError, ProtocolFailure) as exc:
            # Do not echo arbitrary policy values into launch diagnostics.
            raise EntrypointInputError("Azure routing or spending limits are invalid") from exc
        return cls(worker, monitor, batch, raw.get("worker_effort", "low"), raw.get("worker_agent", ""),
                   raw.get("services") != "none")

    def validate_launch(self, config: EntrypointConfig) -> None:
        if (config.expected_model != self.worker.model or config.monitor_model != self.monitor.model
                or config.monitor_max_tokens != self.monitor.max_output_tokens
                or config.monitor_batch_size != self.monitor_batch_size
                or config.monitor_budget_usd != self.monitor.budget_usd):
            raise EntrypointInputError("launch options differ from the assignment's Azure policy")

    def azure_config(self, environment: Mapping[str, str]) -> AzureOpenAIConfig:
        """Read only explicitly named Azure values, and require the bound route."""
        try:
            result = AzureOpenAIConfig.from_environment(environment, deployments=(
                AzureDeployment(self.worker.model, self.worker.deployment),
            ))
            if result.base_url != self.worker.endpoint:
                raise ProtocolFailure("host Azure endpoint differs from the assignment")
        except (ValueError, TypeError, ProtocolFailure) as exc:
            raise EntrypointInputError("host must supply the assignment's Azure route and credential") from exc
        return result
