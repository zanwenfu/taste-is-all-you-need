"""Goal policy -> planner -> prepared actor grant -> actual Azure worker process."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from taste.brains import azure_worker_launch
from taste.brains.azure_central_host import compose_azure_central_runtime
from taste.brains.azure_execution_policy import AzureExecutionPolicy
from taste.brains.records import contract_digest
from taste.brains.terminal_broker import TerminalBinding, TerminalBroker
from taste.brains.terminal_service import TerminalCredential, TerminalGrant, TerminalService
from taste.brains.terminal_worker_policy import TERMINAL_POLICY_KEY, TerminalWorkerPolicy
from tests.test_azure_central_host import environment, install_planner
from tests.test_azure_central_host import goal as _goal
from tests.test_azure_central_host import policy as _policy
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_azure_worker_policy import assignment
from tests.test_azure_worker_process import BOOTSTRAP
from tests.test_terminal_broker import Environment

goal = _goal
policy = _policy
sdk_transport = _sdk_transport


def bound(policy):
    return replace(policy, terminal=TerminalWorkerPolicy(
        TerminalBinding("trial", Environment.environment_id, policy.deadline_unix, 20), 5))


def configured(policy):
    source = assignment()
    contract = replace(source.contract, budget_usd=policy.worker_budget_usd, max_turns=policy.worker_max_calls)
    resources = {"azure_openai": policy.worker_resources(), "monitor_budget_usd": policy.monitor_budget_usd}
    if policy.terminal is not None:
        resources[TERMINAL_POLICY_KEY] = policy.terminal.to_dict()
    return replace(source, contract=contract, contract_digest=contract_digest(contract), resources=resources)


def test_optional_terminal_policy_preserves_old_wire_format_and_pins_new_scope(policy):
    old = policy.to_dict()
    assert old["schema"] == "taste.brains/AzureExecutionPolicy/1" and "terminal" not in old
    assert AzureExecutionPolicy.from_dict(old).to_dict() == old
    new = bound(policy)
    raw = new.to_dict()
    assert raw["schema"] == "taste.brains/AzureExecutionPolicy/2"
    assert AzureExecutionPolicy.from_dict(raw) == new
    new.validate_assignment(configured(new))
    with pytest.raises(ValueError):
        policy.validate_assignment(configured(new))
    with pytest.raises(ValueError):
        new.validate_assignment(configured(policy))


@pytest.mark.parametrize("field,value", [("environment_id", "different_container"), ("max_commands", 21),
                                        ("deadline_unix", 4000000000.0), ("trial_id", "different_trial")])
def test_planner_cannot_change_terminal_environment_or_allowance(policy, field, value):
    admitted = bound(policy)
    proposal = configured(admitted)
    resources = proposal.to_dict()["resources"]
    resources[TERMINAL_POLICY_KEY]["binding"][field] = value
    with pytest.raises(ValueError):
        admitted.validate_assignment(replace(proposal, resources=resources))


def test_terminal_goal_requires_credential_owner_before_composition_or_provider_calls(tmp_path, policy, goal, sdk_transport):
    sent, _ = install_planner(sdk_transport)
    with pytest.raises(ValueError, match="credential provider"):
        compose_azure_central_runtime(tmp_path / "repo", "terminal", goal,
                                     policy=bound(policy), environment=environment())
    assert not sent


def test_default_azure_coordinator_assigns_grant_to_real_worker_and_replays_without_effects(tmp_path, policy, goal, sdk_transport, monkeypatch):
    sent, payloads = install_planner(sdk_transport)
    admitted = bound(policy)
    original_command = azure_worker_launch.worker_command
    bootstrap = BOOTSTRAP.replace("install(network)",
        "from tests.terminal_worker_wire import replies\ninstall(network, worker_reply=replies)")
    bootstrap += "\nsys.modules.pop('taste.brains.azure_worker_entrypoint', None)\n"

    def command(*args, **kwargs):
        argv = list(original_command(*args, **kwargs))
        argv[3] = argv[3].replace("import runpy;", bootstrap + "\nimport runpy;", 1)
        return tuple(argv)

    monkeypatch.setattr(azure_worker_launch, "worker_command", command)

    async def scenario():
        env = Environment()
        owner = TerminalBroker.create(tmp_path / "terminal-ledger", admitted.terminal.binding, env)
        with tempfile.TemporaryDirectory(prefix="taste-central-rpc-", dir="/tmp") as directory:
            socket_path = str(Path(directory) / "service" / "terminal.sock")
            seed = TerminalCredential(socket_path, os.geteuid(), os.geteuid(),
                                      TerminalGrant(owner.binding, "controller_bootstrap", 5), "a" * 64)
            service = TerminalService(owner, [seed])
            await service.start()
            loop = asyncio.get_running_loop()
            issued = []

            def issue(spec):
                credential = TerminalCredential(socket_path, os.geteuid(), os.geteuid(),
                    admitted.terminal.grant(spec.assignment), secrets.token_hex(32))

                async def register():
                    service.authorize(credential)

                asyncio.run_coroutine_threadsafe(register(), loop).result(timeout=5)
                issued.append(credential)
                return credential

            root = tmp_path / "repo"
            root.mkdir()
            try:
                with compose_azure_central_runtime(root, "terminal-central", goal, policy=admitted,
                    environment=environment(), terminal_credential_provider=issue) as runtime:
                    result = await runtime.run_async(max_generations=3, wall_clock_seconds=60)
                    assert result.complete and result.budget.enforceable, result.to_dict()
                    assert runtime.integration.head.read("output.txt") == "correct"
                    assert len(env.calls) == len(issued) == 1
                    assert env.calls[0].actor_id == issued[0].grant.actor_id
                    assert issued[0].token not in json.dumps(payloads)
                    assert "run bounded commands in the shared task container" in payloads[0]["rules"]["worker_capabilities"]["can"]
                    assert all(run.reaped for run in runtime.supervisor.runs())
                with compose_azure_central_runtime(root, "terminal-central", goal, policy=admitted,
                    environment={}, settlement_only=True) as recovered:
                    assert recovered.stop_and_drain("verify completed terminal goal settlement") == result
                assert len(env.calls) == 1 and len(sent) == 2
            finally:
                await service.close()
                owner.close()
    asyncio.run(scenario())
