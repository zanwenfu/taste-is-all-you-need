"""Trusted, non-secret Azure goal and assignment policy.

The complete policy is part of the immutable Goal. Planner output must echo
the admitted worker limits; it cannot choose a provider route or renew a
deadline. Credentials are read separately from the trusted host environment.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace

from taste.agents import HOSTED_AGENTS
from taste.brains.azure_worker_policy import AZURE_WORKER_POLICY_SCHEMA, AzureWorkerPolicy
from taste.brains.branch_replay import BranchPolicy
from taste.brains.records import ArtifactSpec
from taste.brains.responses_session import ResponsesBinding
from taste.brains.single_run import FIXED_PLAN_MODEL
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
                     "planner_effort": "", "request_seconds": None, "planner_model": AZURE_PLANNER_MODEL,
                     "worker_agent": "", "services": "all", "rollback": False, "branch": None}
SERVICES = ("all", "none")


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
    # The served model the coordinator plans and replies on.
    planner_model: str = AZURE_PLANNER_MODEL
    # An agent written by others that every worker runs, unchanged (see
    # taste.agents); empty for Taste's own worker.
    worker_agent: str = ""
    # "none" runs that agent alone through the same machinery: the fixed plan
    # as its planner (taste.brains.single_run), no monitor, no certification.
    services: str = "all"
    # The coordinator may return the task's files to a checkpoint taken before
    # the first worker or after a run (taste.brains.environment_records).
    rollback: bool = False
    # The agent run alone continues an earlier run of it from step k
    # (taste.brains.branch_replay): the replay script, k, and what changes.
    branch: BranchPolicy | None = None

    def __post_init__(self):
        if self.terminal is not None and (
                not isinstance(self.terminal, TerminalWorkerPolicy)
                or self.terminal.binding.deadline_unix != self.deadline_unix):
            raise ValueError("terminal policy must share the original Azure goal deadline")
        if self.worker_model not in AZURE_MODELS:
            raise ValueError("worker model has no verified Azure deployment and price")
        if self.services not in SERVICES:
            raise ValueError("services must be all or none")
        if self.planner_model not in AZURE_MODELS and self.planner_model != FIXED_PLAN_MODEL:
            raise ValueError("planner model has no verified Azure deployment and price")
        if (self.planner_model == FIXED_PLAN_MODEL) != (self.services == "none"):
            raise ValueError("the fixed plan is the planner of an agent run alone, and only that")
        if self.services == "none" and not self.worker_agent:
            raise ValueError("services can be off only for a hosted agent")
        if type(self.rollback) is not bool:
            raise ValueError("rollback must be true or false")
        if self.rollback and (self.terminal is None or self.services != "all"):
            raise ValueError("rollback needs a task terminal and a coordinator that plans")
        if isinstance(self.branch, Mapping):
            object.__setattr__(self, "branch", BranchPolicy.from_dict(self.branch))
        if self.branch is not None and (not isinstance(self.branch, BranchPolicy)
                                        or self.services != "none" or self.terminal is None):
            raise ValueError("a branch continues a hosted agent run alone, in the task's terminal")
        if (self.worker_model == self.planner_model) != (self.worker_deployment == self.planner_deployment):
            raise ValueError("one served model must use exactly one deployment")
        if self.worker_effort not in WORKER_EFFORTS:
            raise ValueError("worker reasoning effort must be low, medium or high")
        if self.worker_agent and self.worker_agent not in HOSTED_AGENTS:
            raise ValueError(f"no hosted agent named {self.worker_agent!r}")
        if self.worker_agent and self.terminal is None:
            raise ValueError("a hosted agent works in the task's terminal, and this goal has none")
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
            model = self.planner_model if role == "planner" else self.worker_model
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
        if self.branch is not None:
            value["branch"] = self.branch.to_dict()
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
        planner = (() if self.planner_model == FIXED_PLAN_MODEL
                   else (AzureDeployment(self.planner_model, self.planner_deployment),))
        result = AzureOpenAIConfig.from_environment(environment, deployments=(
            *planner,
            *(() if self.worker_model == self.planner_model
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
            **({"worker_agent": self.worker_agent} if self.worker_agent else {}),
            **({"services": self.services} if self.services != "all" else {}),
            **({"branch": self.branch.to_dict()} if self.branch is not None else {}),
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
            # Measured on GPT-5.6 Luna: a run that did the task was certified,
            # and the coordinator still assessed every criterion not met
            # because "the integration state contains only report.md", then
            # wrote contracts requiring the files "in the resulting durable
            # state", which no worker can satisfy.
            payload["rules"]["task_environment"] = (
                "The task is done in its container, not in this system's memory. Files, packages and "
                "services the task asks for are made in the container and never appear in the "
                "integration state, a worker's state or any manifest, so their absence there says "
                "nothing about the container. The evidence about the container is in the world's "
                "outcomes: the commands each worker ran there, what they printed, and whether its "
                "report was certified. Assess each criterion about the container from that evidence, "
                "latest run first, since a later run may have changed what an earlier one showed. "
                "Never ask for a container file to be committed, delivered, kept in memory or listed in "
                "a manifest: no worker can do that. Write each success criterion as an effect in the "
                "container that a worker checks with commands and shows the output of.")
        if self.worker_agent:
            self._configure_hosted(payload, exemplar)

    def _configure_hosted(self, payload, exemplar):
        """Every worker is an agent written by others: say what it can and cannot be asked."""
        rules = payload["rules"]
        rules["worker_capabilities"] = {
            "can": ["run shell commands in the shared task container, one at a time"],
            "cannot": ["read or write memory artifacts", "take inbox messages or verdicts while it runs",
                       "produce any output but its report"],
            "harness_does_for_it": (
                "writes its exit, final message, submission and a record of its commands into the "
                "assignment's one output, then certifies and delivers that report"),
            "terminal_effects": rules["worker_capabilities"].get("terminal_effects", ""),
        }
        rules["hosted_workers"] = (
            f"Every worker is {self.worker_agent}, an agent written by others and run unchanged. It is "
            "given contract.task, the contract's success criteria and the goal's original task as one "
            "text, and works in the task container through shell commands until it decides it is done. "
            "It reads no memory artifacts and takes no feedback while it runs: when its monitor judges "
            "it wrong or lost it is stopped, and the next plan sees why. Each assignment declares no "
            "inputs and exactly one output, its report (contract.outputs equal to that one path); the "
            "harness writes it. What the agent changed in the container stays there for the next worker.")
        rules["worker_context"] = (
            "Every worker is shown the goal's original task after its own task and criteria. "
            "contract.task should say what that worker must do and check.")
        # Measured: contracts asked the agent to "write a literal accurate
        # report"; it wrote reports in the container (one overwritten by a
        # broken heredoc after the task was done) and its monitor judged those.
        rules["hosted_contracts"] = (
            "contract.task and success_criteria ask the agent only for the task's effects in the "
            "container, each checked with commands it runs. Never ask it to write a report, summary "
            "or evidence file, or to quote evidence in one: the harness writes its report from the "
            "record of its commands, and a report the agent writes is not its output. Ask for a "
            "file only when the task itself asks for that file.")
        exemplar["contract"].update(inputs=[], outputs=["report.md"])
        exemplar["inputs"] = []
        exemplar["outputs"] = [{**exemplar["outputs"][0], "artifact_id": "<unique id for this report>",
                                "path": "report.md", "kind": "report",
                                "description": "the agent's report, written by the harness"}]

    def assignment_resources(self):
        """The resources every assignment of this goal carries, exactly."""
        return {"azure_openai": self.worker_resources(), "monitor_budget_usd": self.monitor_budget_usd,
                **({} if self.terminal is None else {TERMINAL_POLICY_KEY: self.terminal.to_dict()})}

    def fill_assignment(self, item, where, filled):
        """Write the routing, model and caps this policy fixes, recording what changed.

        The model chooses the work; these it can only copy. A changed contract
        cap changes the contract, so its digest is derived again, never the
        model's left in place.
        """
        resources = item.get("resources") if isinstance(item.get("resources"), Mapping) else {}
        wanted = {**{key: value for key, value in resources.items() if key == "wall_timeout_seconds"},
                  **self.assignment_resources()}
        if resources != wanted:
            item["resources"] = wanted
            filled.append(f"{where}.resources")
        if item.get("model") != self.worker_model:
            item["model"] = self.worker_model
            filled.append(f"{where}.model")
        contract = item.get("contract")
        if self.worker_agent and (item.get("inputs") or (isinstance(contract, Mapping) and contract.get("inputs"))):
            # A hosted agent reads no memory artifacts. Measured on GPT-5.6
            # Luna: a planner handed the next run the last report as an input,
            # twice, each a refused plan. What it must know goes in its task.
            item["inputs"] = []
            filled.append(f"{where}.inputs")
            if isinstance(contract, Mapping):
                contract = {**contract, "inputs": []}
                item["contract"] = contract
                item.pop("contract_digest", None)
        if self.worker_agent:
            contract = self._fill_hosted_outputs(item, contract, where, filled)
        if isinstance(contract, Mapping):
            changed = {name: wanted for name, wanted in (("budget_usd", self.worker_budget_usd),
                                                         ("max_turns", self.worker_max_calls))
                       if contract.get(name) != wanted}
            if changed:
                item["contract"] = {**contract, **changed}
                item.pop("contract_digest", None)
                filled.extend(f"{where}.contract.{name}" for name in sorted(changed))
        return item

    @staticmethod
    def _fill_hosted_outputs(item, contract, where, filled):
        """A hosted agent's one output is its report, which the harness writes.

        Measured on GPT-5.6 Luna: plans declared the task's own file as a
        second output beside the report, three times in one goal, each a
        refused plan. That file belongs in the container, where the task asks
        for it; the report the model declared, if any, is kept.
        """
        outputs = item.get("outputs") if isinstance(item.get("outputs"), list) else []
        declared = [spec for spec in outputs if isinstance(spec, Mapping) and spec.get("kind") == "report"]
        report = {"schema": ArtifactSpec.SCHEMA, "artifact_id": f"{item.get('assignment_id') or 'agent'}-report",
                  "path": "report.md", "kind": "report", "description": "the agent's report, written by the harness",
                  "metadata": {}, **(declared[0] if declared else {}), "required": True, "disposition": "present"}
        if outputs != [report]:
            item["outputs"] = [report]
            filled.append(f"{where}.outputs")
        if isinstance(contract, Mapping) and contract.get("outputs") != [report["path"]]:
            contract = {**contract, "outputs": [report["path"]]}
            item["contract"] = contract
            item.pop("contract_digest", None)
            filled.append(f"{where}.contract.outputs")
        return contract

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
        if self.worker_agent and (assignment.inputs or len(assignment.outputs) != 1
                                  or assignment.outputs[0].disposition != "present"
                                  or not assignment.outputs[0].required):
            raise ValueError("a hosted agent's assignment declares no inputs and exactly one "
                             "required output, its report")
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
