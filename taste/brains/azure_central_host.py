"""Explicit Azure planner and worker composition; legacy composition is intact."""
from __future__ import annotations

import os

from taste.brains.azure_execution_policy import AzureExecutionPolicy
from taste.brains.azure_worker_launch import worker_command_factory
from taste.brains.central_host import compose_central_runtime
from taste.brains.supervisor import SubprocessLauncher
from taste.llm import LLM
from taste.providers.azure_openai import AZURE_MONITOR_MODEL, AZURE_PLANNER_MODEL
from taste.providers.base import ProtocolFailure


class _SettlementLLM(LLM):
    def ensure_ready(self, *models):
        raise ProtocolFailure("settlement cannot admit a model call")

    def call(self, **kwargs):
        raise ProtocolFailure("settlement cannot admit a model call")


class _SettlementLauncher(SubprocessLauncher):
    def launch(self, spec):
        raise RuntimeError("settlement cannot launch a worker")


def compose_azure_central_runtime(
    repo_root, session, goal, *, policy: AzureExecutionPolicy,
    environment=None, launcher=None, store=None, python_executable=None,
    launcher_handshake_timeout=5.0, default_wall_timeout_seconds=900.0,
    supervisor_poll_interval=0.05, supervisor_termination_grace=2.0,
    settlement_only=False, terminal_credential_provider=None,
):
    """Bind non-secret policy into host.goal before any planning or launch.

    Composition has no network effect. The caller must retain ``host.goal``
    as the admitted goal; the default Claude composition refuses that goal
    because it does not carry the matching Azure policy. The injected launcher
    is a Python-only seam for deterministic process tests.
    """
    if not isinstance(policy, AzureExecutionPolicy):
        raise TypeError("policy must be an AzureExecutionPolicy")
    if type(settlement_only) is not bool or (settlement_only and launcher is not None):
        raise ValueError("settlement requires its fixed recovery-only launcher")
    if policy.terminal is not None and not settlement_only and launcher is None and terminal_credential_provider is None:
        raise ValueError("terminal execution requires a trusted per-assignment credential provider")
    bound_goal = policy.bind_goal(goal)
    environment = dict(os.environ if environment is None else environment)
    # Settlement needs the original route identity for receipt audit, never
    # provider authentication. Both paid-call methods and process admission
    # are disabled, so even a mistaken run() cannot dispatch with this marker.
    azure = policy.azure_config(
        {"AZURE_OPENAI_BASE_URL": policy.endpoint, "AZURE_OPENAI_API_KEY": "settlement-no-dispatch"}
        if settlement_only else environment)
    llm_type = _SettlementLLM if settlement_only else LLM
    llm = llm_type(azure_openai=azure, budget_usd=bound_goal.budget_usd, cap_on="billed",
                  max_attempts=1, load_env_file=False, run_id=f"central-planner.{bound_goal.goal_id}")
    if launcher is None:
        launcher_type = _SettlementLauncher if settlement_only else SubprocessLauncher
        launcher = launcher_type(
            worker_command_factory(repo_root, session, python_executable=python_executable,
                                   terminal_credential_provider=terminal_credential_provider),
            env={} if settlement_only else {"AZURE_OPENAI_BASE_URL": azure.base_url, "AZURE_OPENAI_API_KEY": azure.api_key},
            handshake_timeout=launcher_handshake_timeout,
        )
    host = compose_central_runtime(
        repo_root, session, bound_goal, store=store, planner_llm=llm, launcher=launcher,
        planner_model=AZURE_PLANNER_MODEL, monitor_model=AZURE_MONITOR_MODEL,
        planner_max_tokens=policy.planner_max_output_tokens, azure_policy=policy,
        owns_planner_sdk=True,
        default_wall_timeout_seconds=default_wall_timeout_seconds,
        supervisor_poll_interval=supervisor_poll_interval,
        supervisor_termination_grace=supervisor_termination_grace,
    )
    try:
        # Refuse policy drift even when composition is only reopening a
        # finished goal; no provider call or worker launch is needed to check.
        host.planner.bind_goal(bound_goal)
    except BaseException:
        host.close()
        raise
    return host
