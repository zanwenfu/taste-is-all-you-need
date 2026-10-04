"""Goal policy -> planner -> prepared actor grant -> actual Azure worker process."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import tempfile
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from taste.brains import azure_worker_launch, benchmark_reply
from taste.brains.azure_central_host import compose_azure_central_runtime
from taste.brains.azure_execution_policy import AzureExecutionPolicy
from taste.brains.azure_worker_policy import AzureWorkerPolicy
from taste.brains.central_planner import Goal
from taste.brains.records import ArtifactSpec, contract_digest
from taste.brains.terminal_broker import TerminalBinding, TerminalBroker, TerminalResult
from taste.brains.terminal_service import TerminalCredential, TerminalGrant, TerminalService
from taste.brains.terminal_worker_policy import TERMINAL_POLICY_KEY, TerminalWorkerPolicy
from taste.providers.azure_openai import AZURE_PLANNER_MODEL
from tests.test_azure_central_host import environment, install_planner, proposal
from tests.test_azure_central_host import goal as _goal
from tests.test_azure_central_host import policy as _policy
from tests.test_azure_openai import httpx
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_azure_worker_policy import assignment
from tests.test_azure_worker_process import BOOTSTRAP
from tests.test_openai_responses import message, response
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


def test_a_request_ceiling_is_part_of_the_policy_only_when_named(policy):
    # A policy made before ceilings existed names none and keeps its wire form.
    assert "request_seconds" not in policy.to_dict()
    assert "request_seconds" not in policy.worker_resources()
    assert AzureWorkerPolicy.from_assignment(configured(policy)).worker.request_seconds is None

    bounded = replace(policy, request_seconds=45)
    raw = bounded.to_dict()
    assert raw["request_seconds"] == 45 and AzureExecutionPolicy.from_dict(raw) == bounded
    # The worker and its monitor are bound to it through the assignment.
    admitted = AzureWorkerPolicy.from_assignment(configured(bounded))
    assert admitted.worker.request_seconds == admitted.monitor.request_seconds == 45
    bounded.validate_assignment(configured(bounded))
    for other in (policy, replace(policy, request_seconds=60)):
        with pytest.raises(ValueError):
            bounded.validate_assignment(configured(other))
    with pytest.raises(ValueError):
        AzureExecutionPolicy.from_dict({**policy.to_dict(), "request_seconds": None})


LUNA = "gpt-5.6-luna-2026-07-09"


def on_luna(policy, **changes):
    return replace(policy, **{"planner_model": LUNA, "worker_model": LUNA,
                              "planner_deployment": "gpt-5.6-luna",
                              "worker_deployment": "gpt-5.6-luna", **changes})


def test_the_coordinator_model_is_part_of_the_policy_only_when_not_the_original(policy):
    # A policy made before the coordinator could change keeps its wire form.
    assert policy.planner_model == AZURE_PLANNER_MODEL and "planner_model" not in policy.to_dict()
    luna = on_luna(policy)
    raw = luna.to_dict()
    assert raw["planner_model"] == LUNA and AzureExecutionPolicy.from_dict(raw) == luna
    route = luna.azure_config({"AZURE_OPENAI_BASE_URL": policy.endpoint, "AZURE_OPENAI_API_KEY": "k"})
    assert [(item.model, item.deployment) for item in route.deployments] == [(LUNA, "gpt-5.6-luna")]
    with pytest.raises(ValueError):
        AzureExecutionPolicy.from_dict({**policy.to_dict(), "planner_model": AZURE_PLANNER_MODEL})


def test_a_coordinator_and_its_workers_on_two_models_keep_two_routes(policy):
    mixed = replace(policy, planner_model=LUNA, planner_deployment="gpt-5.6-luna")
    route = mixed.azure_config({"AZURE_OPENAI_BASE_URL": policy.endpoint, "AZURE_OPENAI_API_KEY": "k"})
    assert {(item.model, item.deployment) for item in route.deployments} == {
        (LUNA, "gpt-5.6-luna"), (mixed.worker_model, mixed.worker_deployment)}


@pytest.mark.parametrize("changes,match", [
    ({"planner_model": "gpt-5.6-luna"}, "verified Azure deployment"),
    ({"planner_model": "gpt-6.1-sol-2026-09-29"}, "verified Azure deployment"),
    ({"worker_deployment": "another-route"}, "exactly one deployment"),
])
def test_an_unpriced_or_doubly_routed_coordinator_is_refused(policy, changes, match):
    with pytest.raises(ValueError, match=match):
        on_luna(policy, **changes) if "planner_model" not in changes else replace(policy, **changes)


def test_a_hosted_agent_is_part_of_the_policy_only_when_named(policy):
    assert "worker_agent" not in policy.to_dict() and "worker_agent" not in policy.worker_resources()
    with pytest.raises(ValueError, match="terminal"):
        replace(policy, worker_agent="mini-swe-agent")
    with pytest.raises(ValueError, match="no hosted agent"):
        replace(bound(policy), worker_agent="claude-code")
    hosted = replace(bound(policy), worker_agent="mini-swe-agent")
    raw = hosted.to_dict()
    assert raw["worker_agent"] == "mini-swe-agent" and AzureExecutionPolicy.from_dict(raw) == hosted
    assert hosted.worker_resources()["worker_agent"] == "mini-swe-agent"


def test_services_are_off_only_for_a_hosted_agent_with_the_fixed_plan(policy):
    hosted = replace(bound(policy), worker_agent="mini-swe-agent")
    alone = replace(hosted, services="none", planner_model="taste-fixed-plan", planner_deployment="fixed-plan")
    assert alone.worker_resources()["services"] == "none" and "services" not in hosted.worker_resources()
    assert AzureExecutionPolicy.from_dict(alone.to_dict()) == alone
    route = alone.azure_config({"AZURE_OPENAI_BASE_URL": policy.endpoint, "AZURE_OPENAI_API_KEY": "k"})
    assert [item.model for item in route.deployments] == [alone.worker_model]  # the fixed plan has no route
    for source, changes, match in (
            (hosted, {"services": "none"}, "fixed plan"),
            (hosted, {"planner_model": "taste-fixed-plan", "planner_deployment": "fixed-plan"}, "fixed plan"),
            (alone, {"worker_agent": ""}, "hosted agent"),
            (hosted, {"services": "some"}, "all or none")):
        with pytest.raises(ValueError, match=match):
            replace(source, **changes)


def test_the_policy_writes_the_routing_model_and_caps_an_assignment_must_carry(policy):
    from taste.brains.central_planner import CentralPlanner
    from taste.brains.records import Assignment

    admitted = bound(policy)
    item = json.loads(configured(admitted).to_json())
    item["resources"] = {"wall_timeout_seconds": 30, "azure_openai": {"endpoint": "elsewhere"}}
    item["model"] = "gpt-6-astra-2026-09-03"
    item["contract"]["budget_usd"] = 999
    filled = []
    fixed = admitted.fill_assignment(item, "assignments[0]", filled)
    assert fixed["resources"] == {"wall_timeout_seconds": 30, **admitted.assignment_resources()}
    assert fixed["model"] == admitted.worker_model
    assert fixed["contract"]["budget_usd"] == admitted.worker_budget_usd and "contract_digest" not in fixed
    assert filled == ["assignments[0].resources", "assignments[0].model", "assignments[0].contract.budget_usd"]
    admitted.validate_assignment(Assignment.from_dict(CentralPlanner._with_derived_digest(fixed)))
    # An assignment that already carries them is left as it is.
    again = []
    assert admitted.fill_assignment(json.loads(configured(admitted).to_json()), "a", again) and again == []


def test_a_hosted_agents_assignment_is_given_no_inputs_whatever_the_planner_writes(policy):
    hosted = replace(bound(policy), worker_agent="mini-swe-agent")
    item = json.loads(configured(hosted).to_json())
    reference = {"schema": "taste.brains/ArtifactRef/1", "artifact_id": "last-report", "branch": "agent-1",
                 "state_id": "a" * 40, "path": "report.md", "blob_id": "b" * 40}
    item["inputs"] = [reference]
    item["contract"]["inputs"] = ["report.md"]
    filled = []
    fixed = hosted.fill_assignment(item, "assignments[0]", filled)
    assert fixed["inputs"] == [] and fixed["contract"]["inputs"] == [] and "contract_digest" not in fixed
    assert filled[0] == "assignments[0].inputs"
    # Taste's own worker reads inputs; nothing is taken from its assignments.
    own = json.loads(configured(bound(policy)).to_json())
    own["inputs"], own["contract"]["inputs"] = [reference], ["report.md"]
    assert bound(policy).fill_assignment(own, "a", [])["inputs"] == [reference]


def with_outputs(assignment, *outputs, inputs=()):
    contract = replace(assignment.contract, inputs=tuple(item.path for item in inputs),
                       outputs=tuple(item.path for item in outputs))
    return replace(assignment, contract=contract, contract_digest=contract_digest(contract),
                   inputs=tuple(inputs), outputs=tuple(outputs))


def test_a_hosted_agents_assignment_declares_one_report_and_no_inputs(policy):
    hosted = replace(bound(policy), worker_agent="mini-swe-agent")
    report = ArtifactSpec("report", "report.md", kind="report")
    hosted.validate_assignment(with_outputs(configured(hosted), report))
    for bad in (with_outputs(configured(hosted)),
                with_outputs(configured(hosted), report, ArtifactSpec("extra", "extra.md")),
                with_outputs(configured(hosted), replace(report, required=False))):
        with pytest.raises(ValueError, match="exactly one"):
            hosted.validate_assignment(bad)


def test_the_planner_is_told_what_a_hosted_agent_can_be_asked(policy):
    hosted = replace(bound(policy), worker_agent="mini-swe-agent")
    exemplar = {"model": "", "contract": {"inputs": ["in.txt"], "outputs": ["out.txt"]},
                "resources": {"wall_timeout_seconds": 600}, "inputs": [{"path": "in.txt"}],
                "outputs": [{"artifact_id": "out", "path": "out.txt", "kind": "file", "required": True,
                             "disposition": "present"}]}
    payload = {"required_output_shape": {"assignments": [exemplar]}, "rules": {}}
    hosted.configure_prompt(payload)
    rules = payload["rules"]
    assert "mini-swe-agent" in rules["hosted_workers"] and "exactly one output" in rules["hosted_workers"]
    assert "run shell commands in the shared task container, one at a time" in rules["worker_capabilities"]["can"]
    assert rules["worker_capabilities"]["terminal_effects"]
    assert exemplar["inputs"] == [] and exemplar["contract"]["outputs"] == ["report.md"]
    assert [(item["path"], item["kind"]) for item in exemplar["outputs"]] == [("report.md", "report")]


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
                    effects = payloads[0]["rules"]["worker_capabilities"]["terminal_effects"]
                    # A timeout ends the command. The planner must not be told it ends the container.
                    assert "killed with its child processes" in effects and "ends the task environment" not in effects
                    assert all(run.reaped for run in runtime.supervisor.runs())
                with compose_azure_central_runtime(root, "terminal-central", goal, policy=admitted,
                    environment={}, settlement_only=True) as recovered:
                    assert recovered.stop_and_drain("verify completed terminal goal settlement") == result
                assert len(env.calls) == 1 and len(sent) == 2
            finally:
                await service.close()
                owner.close()
    asyncio.run(scenario())


KILLED_MID_COMMAND = BOOTSTRAP.replace("install(network)", """
import json
from tests.test_openai_responses import function_call
def two_commands(number, payload):
    command = "make test" if number == 1 else "make install"
    return [function_call(json.dumps({"command": command, "cwd": "/tmp", "timeout_seconds": 60}),
                          name="terminal_exec", call_id="terminal_call_%d" % number)]
install(network, worker_reply=two_commands)
""")


def test_a_worker_killed_mid_command_leaves_what_it_ran_for_the_closing_reply(
        tmp_path, policy, sdk_transport, monkeypatch):
    # Planner, a real worker process and the terminal service, with only the
    # network and the task container scripted. The worker's second command
    # never returns. When working time ends the worker cannot settle inside
    # its grace period and is killed, so it writes no report. The coordinator
    # used to close knowing nothing of it; now the closing reply is asked for
    # with the commands that worker's own recorded turns show.
    class Stuck(Environment):
        def __init__(self):
            super().__init__()
            self.result = TerminalResult(0, b"12 passed\n", b"")
            self.hold = threading.Event()

        def execute(self, request):
            self.calls.append(request)
            if len(self.calls) > 1:
                self.hold.wait(60)
            return self.result

    goal = Goal(goal_id="azure-goal", task="Run the tests, then install.",
                success_criteria=("the tests pass and the install is done",), budget_usd=100,
                metadata={benchmark_reply.KEY: benchmark_reply.SCHEMA, benchmark_reply.RESERVE_KEY: 10})
    payloads = []

    def planner(wire):
        payload = json.loads(json.loads(wire.content)["input"][0]["content"])
        payloads.append(payload)
        closing = "closing_proposal" in payload["rules"]
        if closing:
            result = payload["required_output_shape"]
            result.update(complete=False, completion_reason="", rationale="the run ended before the plan did")
            result["assessment"] = [
                {"criterion_id": item["criterion_id"], "verdict": "not_met",
                 "evidence": "the install was started and no result was recorded"}
                for item in payload["standing_criteria"]]
        else:
            result = proposal(payload)
        result["metadata"] = {"final_reply": "The tests passed. The install did not finish." if closing else ""}
        return httpx.Response(200, json=response(
            model=AZURE_PLANNER_MODEL, output=[message(json.dumps(result))]))

    sdk_transport(planner)
    # A command here may run for a minute, far past the end of working time.
    admitted = replace(policy, terminal=TerminalWorkerPolicy(
        TerminalBinding("trial", Environment.environment_id, policy.deadline_unix, 20), 60))
    original_command = azure_worker_launch.worker_command
    bootstrap = KILLED_MID_COMMAND + "\nsys.modules.pop('taste.brains.azure_worker_entrypoint', None)\n"

    def command(*args, **kwargs):
        argv = list(original_command(*args, **kwargs))
        argv[3] = argv[3].replace("import runpy;", bootstrap + "\nimport runpy;", 1)
        return tuple(argv)

    monkeypatch.setattr(azure_worker_launch, "worker_command", command)

    async def scenario():
        env = Stuck()
        owner = TerminalBroker.create(tmp_path / "terminal-ledger", admitted.terminal.binding, env)
        with tempfile.TemporaryDirectory(prefix="taste-central-rpc-", dir="/tmp") as directory:
            socket_path = str(Path(directory) / "service" / "terminal.sock")
            seed = TerminalCredential(socket_path, os.geteuid(), os.geteuid(),
                                      TerminalGrant(owner.binding, "controller_bootstrap", 5), "a" * 64)
            service = TerminalService(owner, [seed])
            await service.start()
            loop = asyncio.get_running_loop()

            def issue(spec):
                credential = TerminalCredential(socket_path, os.geteuid(), os.geteuid(),
                    admitted.terminal.grant(spec.assignment), secrets.token_hex(32))

                async def register():
                    service.authorize(credential)

                asyncio.run_coroutine_threadsafe(register(), loop).result(timeout=5)
                return credential

            root = tmp_path / "repo"
            root.mkdir()
            try:
                with compose_azure_central_runtime(root, "terminal-central", goal, policy=admitted,
                        environment=environment(), terminal_credential_provider=issue,
                        supervisor_termination_grace=1.0) as runtime:
                    result = await runtime.run_async(max_generations=3, wall_clock_seconds=20)
                    assert result.stop_reason == "wall_clock" and not result.complete, result.to_dict()
                    assert runtime.runtime.closing_failure is None, runtime.runtime.closing_failure
                    (run,) = runtime.supervisor.runs()
                    assert run.terminal and run.report_id is None, "the worker was killed and left no report"
                    assert [call.command for call in env.calls] == ["make test", "make install"]
                    asked = json.dumps(payloads[-1])
                    assert "closing_proposal" in payloads[-1]["rules"] and "unreported_work" in asked
                    assert "ran: make test -> exit 0; last line printed: 12 passed" in asked
                    assert "started, result not recorded: make install" in asked
                    # The planner was told from the start what such a record is.
                    assert "may or may not have finished" in payloads[0]["rules"]["unreported_work"]
                    plan = runtime.planner.current_plan(goal.goal_id)
                    assert plan.metadata["proposal"]["final_reply"] == "The tests passed. The install did not finish."
            finally:
                env.hold.set()
                await service.close()
                owner.close()
    asyncio.run(scenario())


def test_task_working_directory_is_public_scope_with_its_own_wire_version(policy):
    from taste.brains.terminal_worker_policy import TerminalWorkerPolicy

    plain = bound(policy).terminal
    assert "workdir" not in plain.to_dict() and plain.to_dict()["schema"].endswith("/1")
    placed = replace(plain, workdir="/Users/alex/workspace/cli")
    wire = placed.to_dict()
    assert wire["schema"].endswith("/2") and wire["workdir"] == "/Users/alex/workspace/cli"
    assert TerminalWorkerPolicy.from_dict(wire) == placed
    assert TerminalWorkerPolicy.from_dict(plain.to_dict()) == plain
    for damaged in ({**wire, "workdir": None}, {**plain.to_dict(), "workdir": "/tmp"},
                    {key: value for key, value in wire.items() if key != "workdir"}):
        with pytest.raises(ValueError, match="terminal worker policy"):
            TerminalWorkerPolicy.from_dict(damaged)
    for unsafe in ("relative/path", "/a/../b", "/trailing/", "/nul\x00byte", ""):
        with pytest.raises(ValueError, match="working directory"):
            replace(plain, workdir=unsafe)
