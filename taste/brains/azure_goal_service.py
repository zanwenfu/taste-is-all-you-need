"""Bind durable Azure handoffs to one externally owned Linux process scope.

The outside trial controller owns the scope directory and immutable input;
it must protect their ancestors and keep exchanges outside model write access.
This composes process drainage with result validation. It does not create a
Harbor environment, restart a paid operation, or convert completion to reward.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

from taste.brains.azure_goal_credentials import GOAL_CREDENTIAL_NAME
from taste.brains.azure_goal_entrypoint import _policy
from taste.brains.azure_goal_handoff import (
    _preparation,
    _read,
    handoff_command,
    read_handoff,
    settled_outcome,
)
from taste.brains.central_planner import Goal
from taste.brains.central_runtime import GoalOutcome
from taste.brains.goal_entrypoint import (
    GoalInputError,
    GoalProcessInput,
    _canonical,
    _decode,
    load_goal_input,
)
from taste.brains.process_credentials import ScopeCredential
from taste.brains.process_scope import OwnedProcessScope, ScopeSpec


@dataclass(frozen=True)
class GoalServiceResult:
    value: GoalProcessInput | GoalOutcome
    result_sha256: str
    termination: dict


class AzureGoalService:
    """One launch only; reopen to drain and inspect, never to retry dispatch.

    A missing report after proved drainage requires a separate, credential-free
    settlement service. An ambiguous scope cannot be settled or graded yet.
    Recovery of a prepare operation never reinitializes its workspace.
    """

    def __init__(self, scope: OwnedProcessScope):
        self.scope = scope
        argv = scope.spec.argv
        if len(argv) != 13:
            raise GoalInputError("scope does not bind the Azure handoff entrypoint")
        self.input_path, self.input_sha256, self.output_dir, self.operation = argv[6], argv[8], argv[10], argv[12]
        expected = handoff_command(self.input_path, self.input_sha256, self.output_dir,
                                    operation=self.operation, python_executable=argv[0])
        if (argv != tuple(expected) or scope.spec.python_path is not None
                or scope.spec.cwd != str(Path(self.input_path).parent)):
            raise GoalInputError("scope differs from its exact Azure handoff command")
        names = {item.name for item in scope.spec.credentials}
        if names != ({GOAL_CREDENTIAL_NAME} if self.operation == "run" else set()):
            raise GoalInputError("only a run service may receive its private Azure credential")

    @classmethod
    def create(cls, directory, input_path, input_sha256, output_dir, *, operation,
               uid, python_executable, runtime_seconds, grace_seconds=2, credential=None, manager=None):
        command = handoff_command(input_path, input_sha256, output_dir, operation=operation,
                                  python_executable=python_executable)
        if os.path.lexists(output_dir):
            raise GoalInputError("goal service requires a fresh exchange destination")
        config = cls._input(input_path, input_sha256, operation)
        if operation != "settle":
            policy = config[2] if operation == "prepare" else _policy(config)
            remaining = policy.deadline_unix - time.time()
            if remaining <= 0:
                raise GoalInputError("goal service cannot renew an expired deadline")
            # Validate before min() so bool/NaN cannot become valid durations.
            spec = ScopeSpec(tuple(command), str(Path(input_path).parent), uid, runtime_seconds, grace_seconds)
            runtime_seconds = min(spec.runtime_seconds, remaining)
        credentials = ()
        values = None
        if operation == "run":
            credentials = (ScopeCredential.from_bytes(GOAL_CREDENTIAL_NAME, credential),)
            values = {GOAL_CREDENTIAL_NAME: credential}
        elif credential is not None:
            raise GoalInputError("preparation and settlement cannot receive provider credentials")
        spec = ScopeSpec(tuple(command), str(Path(input_path).parent), uid, runtime_seconds,
                         grace_seconds, credentials=credentials)
        scope = OwnedProcessScope.create(directory, spec, manager=manager, credentials=values)
        return cls(scope)

    @staticmethod
    def _input(path, digest, operation):
        if operation != "prepare":
            config = load_goal_input(path, digest)
            _policy(config)
            return config
        raw = _read(None, path, 65536)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise GoalInputError("preparation input digest differs")
        value = _decode(raw)
        root, goal, policy = _preparation(value)
        return root, goal, policy, value

    async def run(self, *, timeout_seconds):
        """Launch once and retain ownership through cancellation and drainage."""
        await self.scope.run_async(timeout_seconds=timeout_seconds)
        return self._observe()

    async def recover(self):
        """Drain the original scope without launching, then inspect its result."""
        await self.scope.stop_async("recover Azure handoff service")
        return self._observe()

    def _observe(self):
        # A second stop is idempotent and checks the private durable receipt.
        # Never accept an arbitrary caller-supplied 'processes_stopped' flag.
        termination = self.scope.stop("observe Azure handoff result")
        if not termination["goal_settlement_required"]:
            raise GoalInputError("an unlaunched scope cannot have a service result")
        value, digest = read_handoff(self.output_dir, self.input_sha256, self.operation,
                                    service_uid=self.scope.spec.uid)
        if self.operation == "prepare":
            # The original deadline may expire while the outside owner is
            # absent. Observation must not renew it or reject cleanup solely
            # because it is old; preparation freshness is enforced at launch.
            raw = _read(None, self.input_path, 65536)
            if hashlib.sha256(raw).hexdigest() != self.input_sha256:
                raise GoalInputError("preparation input digest differs")
            source = _decode(raw)
            config = GoalProcessInput.from_bytes((_canonical(value) + "\n").encode())
            policy = _policy(config)
            if (config.repo_root != source["repo_root"] or config.session != source["session"]
                    or _canonical(config.goal.to_dict()) != _canonical(policy.bind_goal(Goal.from_dict(source["goal"])).to_dict())
                    or policy.to_dict() != source["policy"]
                    or config.python_source_sha256 != source["python_source_sha256"]
                    or any(config.limits[key] != source[key] for key in
                           ("max_generations", "wall_clock_seconds", "max_planner_failures"))):
                raise GoalInputError("prepared report differs from its original admission")
            result = config
        else:
            config = self._input(self.input_path, self.input_sha256, self.operation)
            result = settled_outcome(value, config)
        return GoalServiceResult(result, digest, termination)
