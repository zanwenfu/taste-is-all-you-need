"""Azure planning, exact assignment admission, actual worker and goal replay."""
from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import replace

import pytest

from taste.brains.azure_central_host import compose_azure_central_runtime
from taste.brains.azure_execution_policy import POLICY_KEY, AzureExecutionPolicy
from taste.brains.azure_worker_entrypoint import run_directory
from taste.brains.azure_worker_launch import worker_command
from taste.brains.azure_worker_policy import AzureWorkerPolicy
from taste.brains.central_host import compose_central_runtime
from taste.brains.central_planner import Goal, InvalidPlannerOutput, PlannerIdentityConflict
from taste.brains.records import WorkerReport
from taste.brains.supervisor import SubprocessLauncher
from taste.brains.worker_protocol import WORKER_REPORT_PATH
from taste.pricing import call_cost, table_sha
from taste.providers.azure_openai import AZURE_PLANNER_MODEL, AZURE_WORKER_MODEL
from taste.providers.base import ProtocolFailure
from tests.test_azure_openai import config, httpx
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_azure_worker_process import BOOTSTRAP
from tests.test_brains_central_host import NoLaunchLauncher
from tests.test_openai_responses import message, response

sdk_transport = _sdk_transport


@pytest.fixture
def policy():
    return AzureExecutionPolicy(
        endpoint=config().base_url, planner_deployment="gpt-6-astra", worker_deployment="gpt-6-sol",
        deadline_unix=time.time() + 120, worker_budget_usd=20, monitor_budget_usd=20,
        worker_max_calls=8, monitor_max_calls=16, worker_max_output_tokens=256,
        monitor_max_output_tokens=512, planner_max_output_tokens=512,
        monitor_batch_size=1, pricing_sha=table_sha(),
    )


@pytest.fixture
def goal():
    return Goal(goal_id="azure-goal", task="Write output.txt with the exact text correct",
                success_criteria=("output.txt contains correct",), budget_usd=100)


def environment():
    return {"AZURE_OPENAI_BASE_URL": config().base_url, "AZURE_OPENAI_API_KEY": "azure-test-only"}


def host(tmp_path, goal, policy, **kwargs):
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    return compose_azure_central_runtime(root, "azure-central", goal, policy=policy,
                                         environment=environment(), **kwargs)


def proposal(payload, *, complete=False):
    result = payload["required_output_shape"]
    if complete:
        result["assignments"] = []
    else:
        assignment = result["assignments"][0]
        assignment["assignment_id"] = "write-output"
        assignment["contract"].update(identity="azure-worker", task="Write output.txt containing correct",
                                      outputs=["output.txt"], success_criteria=["output.txt contains correct"])
        assignment["outputs"][0].update(artifact_id="output-artifact", path="output.txt")
    result.update(complete=complete, completion_reason="verified delivered output" if complete else "",
                  rationale="one declared artifact")
    result["assessment"] = [
        {"criterion_id": item["criterion_id"], "verdict": "met" if complete else "not_met",
         "evidence": "delivered output contains correct" if complete else "no output yet"}
        for item in payload["standing_criteria"]]
    return result


def install_planner(sdk_transport, transform=None):
    payloads = []

    def handler(wire):
        request = json.loads(wire.content)
        assert request["model"] == "gpt-6-astra"
        assert str(wire.url) == config().base_url + "responses"
        assert wire.headers["authorization"] == "Bearer azure-test-only"
        assert "tools" not in request
        payload = json.loads(request["input"][0]["content"])
        payloads.append(payload)
        result = proposal(payload, complete=len(payloads) > 1)
        if transform is not None:
            transform(result)
        return httpx.Response(200, json=response(model=AZURE_PLANNER_MODEL, output=[message(json.dumps(result))]))

    sent, _ = sdk_transport(handler)
    return sent, payloads


def test_real_azure_planner_to_worker_to_certified_delivery_and_reopen(tmp_path, goal, policy, sdk_transport):
    sent, payloads = install_planner(sdk_transport)
    root = tmp_path / "repo"

    def command(spec):
        argv = list(worker_command(spec, repo_root=root, session="azure-central"))
        bootstrap = BOOTSTRAP + "\nsys.modules.pop('taste.brains.azure_worker_entrypoint', None)\n"
        argv[3] = argv[3].replace("import runpy;", bootstrap + "\nimport runpy;", 1)
        return argv

    launcher = SubprocessLauncher(command, env=environment())
    with host(tmp_path, goal, policy, launcher=launcher) as runtime:
        assert not sent
        assert runtime.goal.metadata[POLICY_KEY] == policy.to_dict()
        result = runtime.run(max_generations=3, wall_clock_seconds=60)
        assert result.complete and result.budget.enforceable, result.to_dict()
        assert result.budget.known_spent_usd > 0
        assert runtime.integration.head.read("output.txt") == "correct"
        runs = runtime.supervisor.runs()
        assert len(runs) == 1 and runs[0].reaped and runs[0].phase == "delivered"
        paid = []
        for journal in run_directory(runtime.store, runs[0].run_id).glob("*/calls.sqlite3"):
            connection = sqlite3.connect(f"file:{journal}?mode=ro", uri=True)
            try:
                for (raw,) in connection.execute("SELECT result FROM calls WHERE status='completed'"):
                    receipt = json.loads(raw)
                    usage = receipt["usage"]
                    paid.append(call_cost(receipt["model"], **{
                        key: usage[key] for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
                    })[0])
            finally:
                connection.close()
        assert len(paid) >= 4
        assert result.budget.worker_spent_usd == pytest.approx(math.fsum(paid))
        admitted = AzureWorkerPolicy.from_assignment(runs[0].assignment)
        assert admitted.worker.model == AZURE_WORKER_MODEL
        assert admitted.worker.deadline_unix == policy.deadline_unix
        assert admitted.worker.budget_usd == policy.worker_budget_usd
        assert len(sent) == 2
        assert "run shell commands" in payloads[0]["rules"]["worker_capabilities"]["cannot"]
        bound_goal = runtime.goal
        client = runtime.planner_llm.provider_for(AZURE_PLANNER_MODEL)._client
        assert not client.is_closed()
    assert client.is_closed()
    with host(tmp_path, goal, policy, launcher=NoLaunchLauncher()) as reopened:
        assert reopened.run(max_generations=3, wall_clock_seconds=60) == result
        assert len(sent) == 2
    # The historical factory cannot accidentally execute an Azure-bound goal.
    with compose_central_runtime(root, "azure-central", bound_goal, launcher=NoLaunchLauncher()) as legacy:
        with pytest.raises(PlannerIdentityConflict, match="policies differ"):
            legacy.cycle()
        assert len(sent) == 2


@pytest.mark.parametrize("field,value", [
    ("endpoint", "https://other.openai.azure.com/openai/v1/"),
    ("worker_deployment", "other-worker"), ("planner_deployment", "other-planner"),
    ("deadline_unix", 4000000000.0), ("worker_budget_usd", 21.0),
])
def test_policy_drift_is_rejected_before_new_call_or_worker(tmp_path, goal, policy, sdk_transport, field, value):
    sent, _ = install_planner(sdk_transport)
    with host(tmp_path, goal, policy, launcher=NoLaunchLauncher()) as first:
        original_head = first.control.head.id
    altered = replace(policy, **{field: value})
    env = {**environment(), "AZURE_OPENAI_BASE_URL": altered.endpoint}
    with pytest.raises(PlannerIdentityConflict):
        compose_azure_central_runtime(tmp_path / "repo", "azure-central", goal,
                                     policy=altered, environment=env, launcher=NoLaunchLauncher())
    with host(tmp_path, goal, policy, launcher=NoLaunchLauncher()) as reopened:
        assert reopened.control.head.id == original_head
        assert not reopened.supervisor.runs() and not sent


@pytest.mark.parametrize("change", ["route", "deadline", "worker_budget", "monitor_budget", "model", "calls", "extra"])
def test_untrusted_planner_cannot_change_execution_policy(tmp_path, goal, policy, sdk_transport, change):
    def transform(result):
        assignment = result["assignments"][0]
        if change == "worker_budget":
            assignment["contract"]["budget_usd"] = 21
        elif change == "monitor_budget":
            assignment["resources"]["monitor_budget_usd"] = 21
        elif change == "model":
            assignment["model"] = "claude-sonnet-4-6"
        elif change == "extra":
            assignment["resources"]["terminal"] = True
        else:
            key, value = {"route": ("worker_deployment", "other"), "deadline": ("deadline_unix", 4000000000),
                          "calls": ("worker_max_calls", 100)}[change]
            assignment["resources"]["azure_openai"][key] = value

    sent, _ = install_planner(sdk_transport, transform)
    with host(tmp_path, goal, policy, launcher=NoLaunchLauncher()) as runtime:
        with pytest.raises(InvalidPlannerOutput):
            runtime.cycle()
        assert runtime.planner.planner_cost(goal.goal_id, currency="billed") > 0
        assert len(sent) == 1 and not runtime.supervisor.runs()


def test_a_lost_planner_reply_is_charged_at_its_ceiling_and_the_goal_goes_on(tmp_path, goal, policy, sdk_transport):
    # One server error on a planner call used to end the goal: the planner's
    # client refused every later call, the budget was called unprovable, and
    # not even a closing reply could be asked for.
    payloads = []

    def handler(wire):
        payload = json.loads(json.loads(wire.content)["input"][0]["content"])
        payloads.append(payload)
        if len(payloads) == 1:
            return httpx.Response(500, json={"error": {"message": "server error"}})
        return httpx.Response(200, json=response(
            model=AZURE_PLANNER_MODEL, output=[message(json.dumps(proposal(payload, complete=True)))]))

    sent, _ = sdk_transport(handler)
    with host(tmp_path, goal, policy, launcher=NoLaunchLauncher()) as runtime:
        result = runtime.run(max_generations=3, wall_clock_seconds=60)
        assert result.complete and result.stop_reason == "complete", result.to_dict()
        # The lost call was not sent again; the next plan was a new call.
        assert len(sent) == 2
        ceiling = runtime.planner.transport.max_billed_call_usd()
        budget = result.budget
        # Its exact cost is unknown. It cannot exceed what the call was admitted
        # against, and that much stays set aside.
        assert budget.reserved_usd == pytest.approx(ceiling) and ceiling > 0
        assert budget.enforceable and not budget.unknown_planner_attempt_ids
        assert budget.known_spent_usd > 0


def test_a_plan_the_service_never_answers_ends_at_its_ceiling_and_the_goal_goes_on(tmp_path, goal, policy, sdk_transport):
    # A request was given all the time its goal had left. One the service
    # accepted and never answered held the coordinator until the trial ended.
    payloads = []

    def handler(wire):
        payload = json.loads(json.loads(wire.content)["input"][0]["content"])
        payloads.append(payload)
        if len(payloads) == 1:
            raise httpx.ReadTimeout("no answer")
        return httpx.Response(200, json=response(
            model=AZURE_PLANNER_MODEL, output=[message(json.dumps(proposal(payload, complete=True)))]))

    sent, _ = sdk_transport(handler)
    with host(tmp_path, goal, replace(policy, request_seconds=45), launcher=NoLaunchLauncher()) as runtime:
        result = runtime.run(max_generations=3, wall_clock_seconds=60)
        assert result.complete and result.stop_reason == "complete", result.to_dict()
        # Each request had 45 seconds, not the minute the run had left. The
        # unanswered one was not sent again; the next plan was a new call.
        assert [wire.extensions["timeout"]["read"] for wire in sent] == [45, 45]
        assert result.budget.enforceable and result.budget.reserved_usd > 0


# A worker process whose second request is accepted and never answered, once:
# the worker that follows it finds the marker and is answered as usual.
LOSES_ONE_REPLY = BOOTSTRAP.replace("install(network)", """
import pathlib
marker = pathlib.Path(MARKER)
allowed = []
def network(handler):
    def observed(wire):
        allowed.append(wire.extensions["timeout"]["read"])
        return handler(wire)
    class Client(OriginalClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, transport=httpx.MockTransport(observed))
    httpx.Client = Client
    return [], []
def lose_one(role, number, payload):
    if role == "worker" and number == 2 and not marker.exists():
        marker.write_text(repr(allowed[-1]))
        raise httpx.ReadTimeout("no answer")
install(network, hook=lose_one)
""")


def test_a_worker_whose_request_is_never_answered_ends_at_its_ceiling_and_the_goal_goes_on(
        tmp_path, goal, policy, sdk_transport):
    # Planner to worker process to coordinator, with only the network scripted.
    # The first worker's second request is never answered. It ends at the
    # request ceiling, not at the goal's deadline; it reports what it paid and
    # what that one request can have cost; and the next worker finishes the goal.
    payloads = []

    def handler(wire):
        payload = json.loads(json.loads(wire.content)["input"][0]["content"])
        payloads.append(payload)
        result = proposal(payload, complete=len(payloads) > 2)
        if not result["complete"]:
            assignment = result["assignments"][0]
            assignment["assignment_id"] = f"write-output-{len(payloads)}"
            assignment["contract"]["identity"] = f"azure-worker-{len(payloads)}"
        return httpx.Response(200, json=response(
            model=AZURE_PLANNER_MODEL, output=[message(json.dumps(result))]))

    sdk_transport(handler)
    root, marker = tmp_path / "repo", tmp_path / "lost-reply"
    script = LOSES_ONE_REPLY.replace("MARKER", repr(str(marker)))

    def command(spec):
        argv = list(worker_command(spec, repo_root=root, session="azure-central"))
        bootstrap = script + "\nsys.modules.pop('taste.brains.azure_worker_entrypoint', None)\n"
        argv[3] = argv[3].replace("import runpy;", bootstrap + "\nimport runpy;", 1)
        return argv

    bounded = replace(policy, request_seconds=45)
    with host(tmp_path, goal, bounded, launcher=SubprocessLauncher(command, env=environment())) as runtime:
        result = runtime.run(max_generations=4, wall_clock_seconds=90)
        assert result.complete and result.stop_reason == "complete", result.to_dict()
        assert runtime.integration.head.read("output.txt") == "correct"
        # The unanswered request had 45 seconds, not the two minutes the goal had.
        assert marker.read_text() == "45.0"

        lost, finished = sorted(runtime.supervisor.runs(), key=lambda run: run.assignment.generation)
        assert finished.phase == "delivered"
        first, second = (WorkerReport.from_json(runtime.store.view(run.assignment.worker).head.read(
            WORKER_REPORT_PATH)) for run in (lost, finished))
        assert first.cost_usd is None and not first.completed
        account = first.metadata["model_cost"]
        assert account["known_usd"] > 0
        # One request of a few kilobytes: cents. Its worker and monitor caps,
        # which the run was charged before, come to $40.
        assert 0 < account["unknown_exposure_usd"] < 1
        budget = result.budget
        assert budget.enforceable
        assert budget.reserved_usd == pytest.approx(account["unknown_exposure_usd"])
        assert budget.worker_spent_usd == pytest.approx(account["known_usd"] + second.cost_usd)


def test_expired_policy_does_not_refresh_planner_deadline_or_dispatch(tmp_path, goal, policy, sdk_transport):
    sent, _ = install_planner(sdk_transport)
    expired = replace(policy, deadline_unix=1)
    with host(tmp_path, goal, expired, launcher=NoLaunchLauncher()) as runtime:
        result = runtime.run(max_generations=1, wall_clock_seconds=30, max_planner_failures=1)
        assert not result.complete and result.budget.enforceable
        assert result.budget.known_spent_usd == 0
        assert not sent and not runtime.supervisor.runs()


@pytest.mark.parametrize("field,value", [
    ("worker_max_calls", True), ("monitor_max_calls", 0), ("max_request_bytes", 0),
    ("worker_budget_usd", 0.001), ("monitor_budget_usd", 0.001),
    ("monitor_batch_size", 0), ("deadline_unix", float("nan")), ("pricing_sha", "changed"),
    ("request_seconds", 0), ("request_seconds", True), ("request_seconds", 3601),
])
def test_unusable_policy_is_rejected_before_composition(policy, field, value):
    with pytest.raises(ValueError):
        replace(policy, **{field: value})


def test_sdk_close_failure_blocks_host_until_cleanup_retry(tmp_path, goal, policy, sdk_transport, monkeypatch):
    sent, _ = install_planner(sdk_transport)
    runtime = host(tmp_path, goal, policy, launcher=NoLaunchLauncher())
    runtime.planner_llm.ensure_ready(AZURE_PLANNER_MODEL)
    client = runtime.planner_llm.provider_for(AZURE_PLANNER_MODEL)._client
    original = client.close

    def fail_close():
        raise OSError("SDK close failed")

    try:
        monkeypatch.setattr(client, "close", fail_close)
        with pytest.raises(OSError, match="SDK close"):
            runtime.close()
        assert not runtime.closed and not client.is_closed()
        with pytest.raises(RuntimeError, match="closing"):
            runtime.cycle()
    finally:
        monkeypatch.setattr(client, "close", original)
        runtime.close()
    assert runtime.closed and client.is_closed() and not sent


def test_settlement_composition_refuses_even_direct_model_and_launch_calls(tmp_path, goal, policy, sdk_transport):
    sent, _ = install_planner(sdk_transport)
    root = tmp_path / "repo"
    root.mkdir()
    with compose_azure_central_runtime(root, "settlement", goal, policy=policy,
                                      environment={}, settlement_only=True) as runtime:
        with pytest.raises(ProtocolFailure, match="settlement"):
            runtime.planner_llm.ensure_ready(AZURE_PLANNER_MODEL)
        with pytest.raises(ProtocolFailure, match="settlement"):
            runtime.planner_llm.call(model=AZURE_PLANNER_MODEL)
        with pytest.raises(RuntimeError, match="settlement"):
            runtime.launcher.launch(None)
        assert runtime.planner_llm._providers == {} and not sent


def test_every_role_can_run_on_the_planner_model_through_one_deployment(policy, goal):
    """A benchmark run reported as one model uses that model for every role."""
    from taste.providers.azure_openai import AZURE_PLANNER_MODEL, AZURE_WORKER_MODEL

    # The larger model's worst-case call is dearer, so its caps must admit one.
    single = replace(policy, worker_model=AZURE_PLANNER_MODEL, worker_deployment=policy.planner_deployment,
                     worker_effort="medium", worker_budget_usd=40, monitor_budget_usd=40)
    with pytest.raises(ValueError, match="cannot admit even one bounded call"):
        replace(policy, worker_model=AZURE_PLANNER_MODEL, worker_deployment=policy.planner_deployment)
    route = single.azure_config(environment())
    assert [(item.model, item.deployment) for item in route.deployments] == [
        (AZURE_PLANNER_MODEL, "gpt-6-astra")]
    wire = single.to_dict()
    assert wire["worker_model"] == AZURE_PLANNER_MODEL and wire["worker_effort"] == "medium"
    assert AzureExecutionPolicy.from_dict(wire) == single
    assert single.worker_resources()["worker_effort"] == "medium"
    # The original choices keep their original wire form and digest.
    assert "worker_model" not in policy.to_dict() and "worker_effort" not in policy.to_dict()
    assert "worker_effort" not in policy.worker_resources()
    assert AzureExecutionPolicy.from_dict(policy.to_dict()) == policy
    assert single.bind_goal(goal).metadata != policy.bind_goal(goal).metadata

    payload = {"required_output_shape": {"assignments": [{"model": "", "contract": {}, "resources": {}}]},
               "rules": {}}
    single.configure_prompt(payload)
    assert payload["required_output_shape"]["assignments"][0]["model"] == AZURE_PLANNER_MODEL
    assert "worker_context" in payload["rules"]

    for damaged in ({**wire, "worker_model": AZURE_WORKER_MODEL},  # the original choice is never written
                    {**wire, "worker_effort": "low"}):
        with pytest.raises(ValueError, match="invalid Azure execution policy"):
            AzureExecutionPolicy.from_dict(damaged)


@pytest.mark.parametrize("changes,match", [
    ({"worker_model": "gpt-6-luna"}, "no verified Azure deployment"),
    ({"worker_model": "gpt-6-astra-2026-09-03"}, "exactly one deployment"),
    ({"worker_deployment": "gpt-6-astra"}, "exactly one deployment"),
    ({"worker_effort": "maximum"}, "reasoning effort"),
])
def test_worker_model_and_effort_are_admitted_not_assumed(policy, changes, match):
    with pytest.raises(ValueError, match=match):
        replace(policy, **changes)
