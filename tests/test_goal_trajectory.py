"""Settled coordinator/worker evidence crosses real SDK, process and file boundaries."""

import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from taste.benchmarks.goal_trajectory import encode_trajectory, goal_trajectory
from taste.brains import benchmark_reply
from taste.brains.azure_goal_handoff import read_handoff, read_trajectory, settled_outcome
from taste.brains.azure_worker_entrypoint import run_directory
from taste.brains.azure_worker_launch import worker_command
from taste.brains.goal_entrypoint import (
    GoalInputError,
    GoalProcessInput,
    _canonical,
    python_source_digest,
)
from taste.brains.supervisor import SubprocessLauncher
from taste.providers.azure_openai import AZURE_MONITOR_MODEL, AZURE_PLANNER_MODEL
from tests.test_azure_central_host import environment, host, proposal
from tests.test_azure_central_host import goal as _goal
from tests.test_azure_central_host import policy as _policy
from tests.test_azure_goal_entrypoint import BOOTSTRAP as GOAL_BOOTSTRAP
from tests.test_azure_goal_handoff import preparation, process, write_input
from tests.test_azure_openai import httpx
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_azure_worker_process import BOOTSTRAP
from tests.test_openai_responses import message, response

goal, policy, sdk_transport = _goal, _policy, _sdk_transport
FINAL = 'Updated output.txt.\r\nOnly the recorded checks were run. ☃\n'


@pytest.fixture
def benchmark_goal(goal):
    return replace(goal, metadata={benchmark_reply.KEY: benchmark_reply.SCHEMA})


@pytest.mark.parametrize("failed_first", [False, True])
def test_all_real_worker_attempts_and_monitor_costs_survive_settlement(
        tmp_path, benchmark_goal, policy, sdk_transport, failed_first, monkeypatch):
    requests = []

    def handle(wire):
        payload = json.loads(json.loads(wire.content)["input"][0]["content"])
        requests.append(payload)
        result = proposal(payload, complete=len(requests) == 3)
        result["metadata"]["final_reply"] = FINAL if result["complete"] else ""
        for assignment in result["assignments"]:
            assignment["assignment_id"] += "-" + str(len(requests))
            assignment["contract"]["identity"] += "-" + str(len(requests))
            assignment["outputs"][0]["artifact_id"] += "-" + str(len(requests))
        return httpx.Response(200, json=response(model=AZURE_PLANNER_MODEL, output=[message(json.dumps(result))]))

    sent, _ = sdk_transport(handle)
    root = tmp_path / "repo"

    def command(spec):
        argv = list(worker_command(spec, repo_root=root, session="azure-central"))
        bootstrap = BOOTSTRAP
        if failed_first and spec.assignment.generation == 1:
            bootstrap = bootstrap.replace("install(network)",
                "from tests.test_openai_responses import message\n"
                "install(network, worker_reply=lambda *_: [message('invalid worker completion')])")
        bootstrap += "\nsys.modules.pop('taste.brains.azure_worker_entrypoint', None)\n"
        argv[3] = argv[3].replace("import runpy;", bootstrap + "\nimport runpy;", 1)
        return argv

    with host(tmp_path, benchmark_goal, policy, launcher=SubprocessLauncher(command, env=environment())) as runtime:
        limits = runtime.prepare_run(max_generations=4, wall_clock_seconds=120,
                                     deadline_at=datetime.fromtimestamp(policy.deadline_unix, UTC))
        config = GoalProcessInput(str(root), "azure-central", runtime.goal, runtime.control.head.id,
            _canonical(limits), python_source_digest(), AZURE_PLANNER_MODEL, AZURE_MONITOR_MODEL,
            policy.planner_max_output_tokens)
        outcome = runtime.run(max_generations=4, wall_clock_seconds=120)
        assert outcome.complete, outcome.to_dict()
        runs = runtime.supervisor.runs()
        assert len(runs) == 2
        if failed_first:
            assert sum(r.phase == "delivered" for r in runs) == 1
        assert runtime.stop_and_drain("export") == outcome
        trace = goal_trajectory(runtime, config, outcome)
        assert len(sent) == 3
        assert trace["steps"][-1]["message"] == FINAL
        assert trace["extra"]["evidence_complete"], trace["extra"]["gaps"]
        assert len(trace["subagent_trajectories"]) == 5  # coordinator, two workers, two monitors
        assert trace["final_metrics"]["total_cost_usd"] == outcome.budget.known_spent_usd
        assert {r["run_id"] for r in trace["extra"]["runs"]} == {r.run_id for r in runs}
        assert all(s.get("extra", {}).get("is_sidechain") for sub in trace["subagent_trajectories"]
                   for s in sub["steps"] if s["source"] == "agent")
        calls = [{"args": c["arguments"], "result": next(r["content"] for r in s["observation"]["results"]
                    if r["source_call_id"] == c["tool_call_id"])}
                 for sub in trace["subagent_trajectories"] for s in sub["steps"] for c in s.get("tool_calls", [])]
        assert len(calls) == (1 if failed_first else 2)
        (tmp_path / "trajectory.goal-case.json").write_bytes(encode_trajectory({"trajectory": trace,
            "expected_reply": FINAL.strip(), "expected_calls": calls}))
        # A missing abandoned worker must be noticed as well as a delivered one.
        lost = run_directory(runtime.store, runs[0].run_id) / "worker"
        lost.rename(lost.with_name("retained-worker"))
        damaged = goal_trajectory(runtime, config, outcome)
        assert not damaged["extra"]["evidence_complete"]
        assert any(g.startswith("missing_worker_journal:") for g in damaged["extra"]["gaps"])
        assert "total_cost_usd" not in damaged["final_metrics"]
        # The original transport hashes, not a new prompt, authorize export.
        original = runtime.planner._prompt
        monkeypatch.setattr(runtime.planner, "_prompt", lambda request: original(request) + "changed")
        with pytest.raises(GoalInputError, match="original transport"):
            goal_trajectory(runtime, config, outcome)
        assert len(sent) == 3


@pytest.mark.parametrize("lost_reply", [False, True])
def test_real_process_handoff_binds_trace_and_retains_unknown_cost_without_credentials(
        tmp_path, benchmark_goal, policy, lost_reply):
    from taste.brains.azure_goal_credentials import (
        GOAL_CREDENTIAL_NAME,
        encode_azure_goal_credentials,
    )

    path, digest = preparation(tmp_path, benchmark_goal, policy)
    env = {k: v for k, v in os.environ.items() if k not in
           {"CREDENTIALS_DIRECTORY", "AZURE_OPENAI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}}
    counter = tmp_path / "calls"
    env["TASTE_TEST_WIRE_COUNT"] = str(counter)
    bootstrap = GOAL_BOOTSTRAP.replace("result = payload['required_output_shape']",
        "result = payload['required_output_shape']\n    result['metadata']['final_reply'] = " + repr(FINAL))
    if lost_reply:
        bootstrap = bootstrap.replace("return httpx.Response(200, json={",
            "raise httpx.ReadTimeout('lost reply', request=wire)\n    return httpx.Response(200, json={")
    prepared = process(path, digest, tmp_path / "prepared", "prepare", env, bootstrap)
    assert prepared.returncode == 0, prepared.stderr
    value, _ = read_handoff(tmp_path / "prepared", digest, "prepare", service_uid=os.geteuid())
    config = GoalProcessInput.from_bytes(json.dumps(value).encode())
    path = tmp_path / "goal.json"
    digest = write_input(path, config.to_bytes())
    directory = tmp_path / "credentials"
    directory.mkdir(mode=0o700)
    secret = directory / GOAL_CREDENTIAL_NAME
    secret.write_bytes(encode_azure_goal_credentials(config, "azure-test-only"))
    secret.chmod(0o400)
    env["CREDENTIALS_DIRECTORY"] = str(directory)
    for operation in ("run", "settle"):
        if operation == "settle":
            secret.unlink()
        child = process(path, digest, tmp_path / operation, operation, env, bootstrap)
        assert child.returncode == 0, child.stderr
    value, _ = read_handoff(tmp_path / "settle", digest, "settle", service_uid=os.geteuid())
    outcome = settled_outcome(value, config)
    raw, trace, sha = read_trajectory(tmp_path / "settle", config, outcome, service_uid=os.geteuid())
    assert sha == hashlib.sha256(raw).hexdigest()
    assert "azure-test-only" not in raw.decode()
    assert not (tmp_path / "run/trajectory.json").exists()
    assert trace["extra"]["evidence_complete"] == (not lost_reply)
    assert trace["extra"]["final_reply_present"] == (not lost_reply)
    assert counter.read_text().splitlines() == ["one Azure planner request"]
    if lost_reply:
        assert len(trace["steps"]) == 1 and "total_cost_usd" not in trace["final_metrics"]
    else:
        assert trace["steps"][-1]["message"] == FINAL
    (tmp_path / "trajectory.goal-case.json").write_bytes(encode_trajectory({"trajectory": trace,
        "expected_reply": "" if lost_reply else FINAL.strip(), "expected_calls": []}))
    saved = (tmp_path / "settle/trajectory.json").read_bytes()
    for damage in (b"{}", saved + b" "):
        (tmp_path / "settle/trajectory.json").write_bytes(damage)
        with pytest.raises(GoalInputError):
            read_trajectory(tmp_path / "settle", config, outcome, service_uid=os.geteuid())
    assert counter.read_text().splitlines() == ["one Azure planner request"]


def test_export_limit_never_silently_truncates(monkeypatch):
    monkeypatch.setattr("taste.benchmarks.goal_trajectory.MAX_TRAJECTORY_BYTES", 20)
    with pytest.raises(GoalInputError, match="no truncated"):
        encode_trajectory({"message": "a" * 20})
