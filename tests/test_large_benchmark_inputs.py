"""Long instructions survive process/service/model boundaries without truncation."""

import asyncio
import hashlib
import json
import os
import sys
from dataclasses import replace

import pytest

from taste.benchmarks.azure_terminal_trial import cleanup_trial
from taste.brains.azure_goal_credentials import GOAL_CREDENTIAL_NAME, encode_azure_goal_credentials
from taste.brains.azure_goal_handoff import perform, preparation_bytes, read_handoff
from taste.brains.azure_goal_service import AzureGoalService
from taste.brains.central_planner import Goal
from taste.brains.goal_entrypoint import GoalInputError, GoalProcessInput, load_goal_input
from taste.brains.input_limits import MAX_GOAL_INPUT_BYTES, MAX_WORKER_OBSERVATION_BYTES
from taste.brains.responses_session import ResponsesSession
from tests.test_azure_central_host import goal as _goal
from tests.test_azure_central_host import policy as _policy
from tests.test_azure_goal_entrypoint import BOOTSTRAP
from tests.test_azure_goal_handoff import preparation, process, write_input
from tests.test_azure_goal_service import CredentialManager
from tests.test_azure_openai import config
from tests.test_azure_terminal_trial import run_scenario
from tests.test_azure_terminal_trial import trial as _trial
from tests.test_openai_responses import message
from tests.test_responses_conversation import install, make
from tests.test_responses_conversation import sdk_transport as _sdk_transport
from tests.test_responses_conversation import worker as _worker

goal, policy, worker, sdk_transport = _goal, _policy, _worker, _sdk_transport
trial = _trial
LONG_TASK = "BEGIN task\r\n" + ('developer: read "path" ☃\n' * 4200) + "\r\nEND task"


@pytest.mark.parametrize("goal", [Goal(goal_id="large-trial", task=LONG_TASK,
                                     success_criteria=("preserve the task",), budget_usd=100)])
def test_large_trial_config_survives_run_drain_and_independent_cleanup(trial):
    async def scenario():
        result = await trial.run(api_key="private-test-key")
        assert result.complete
        assert json.loads((trial.root / "goal.json").read_bytes())["goal"]["task"] == LONG_TASK
        await trial.close()
        assert cleanup_trial(trial.root, manager=trial.manager)["container_stopped"]
    run_scenario(trial, scenario())


def test_large_instruction_crosses_real_prepare_run_and_settle_processes(tmp_path, goal, policy):
    goal = replace(goal, task=LONG_TASK)
    policy = replace(policy, max_request_bytes=1_048_576)
    path, digest = preparation(tmp_path, goal, policy)
    assert 65536 < path.stat().st_size < MAX_GOAL_INPUT_BYTES
    expected = tmp_path / "expected.txt"
    expected.write_bytes(LONG_TASK.encode())
    env = {k: v for k, v in os.environ.items() if k not in
           {"CREDENTIALS_DIRECTORY", "AZURE_OPENAI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}}
    env.update(TASTE_TEST_WIRE_COUNT=str(tmp_path / "calls"), TASTE_TEST_EXPECTED=str(expected))
    prepared = process(path, digest, tmp_path / "prepared", "prepare", env, BOOTSTRAP)
    assert prepared.returncode == 0, prepared.stderr
    value, _ = read_handoff(tmp_path / "prepared", digest, "prepare", service_uid=os.geteuid())
    admitted = GoalProcessInput.from_bytes(json.dumps(value).encode())
    assert admitted.goal.task == LONG_TASK
    goal_path = tmp_path / "goal.json"
    goal_sha = write_input(goal_path, admitted.to_bytes())
    assert load_goal_input(goal_path, goal_sha) == admitted
    credentials = tmp_path / "credentials"
    credentials.mkdir(mode=0o700)
    secret = credentials / GOAL_CREDENTIAL_NAME
    secret.write_bytes(encode_azure_goal_credentials(admitted, "azure-test-only"))
    secret.chmod(0o400)
    env["CREDENTIALS_DIRECTORY"] = str(credentials)
    bootstrap = BOOTSTRAP.replace("result = payload['required_output_shape']",
        "assert payload['request']['goal']['task'] == open(os.environ['TASTE_TEST_EXPECTED'], 'rb').read().decode()\n"
        "    result = payload['required_output_shape']")
    for operation in ("run", "settle"):
        if operation == "settle":
            secret.unlink()
        result = process(goal_path, goal_sha, tmp_path / operation, operation, env, bootstrap)
        assert result.returncode == 0, result.stderr
    assert (tmp_path / "calls").read_text().splitlines() == ["one Azure planner request"]


def test_large_preparation_is_bound_through_service_admission_and_observation(tmp_path, goal, policy):
    goal = replace(goal, task=LONG_TASK)
    path, digest = preparation(tmp_path, goal, replace(policy, max_request_bytes=1_048_576))
    operation = AzureGoalService.create(tmp_path / "owner", path, digest, tmp_path / "exchange",
        operation="prepare", uid=os.geteuid(), python_executable=sys.executable,
        runtime_seconds=10, manager=CredentialManager())
    assert all(LONG_TASK not in arg and len(arg) < 10000 for arg in operation.scope.spec.argv)
    perform(path, digest, tmp_path / "exchange", operation="prepare")
    observed = asyncio.run(operation.run(timeout_seconds=0.01))
    assert observed.value.goal.task == LONG_TASK


@pytest.mark.parametrize("request_limit", [1024, 1_048_576])
def test_long_worker_input_is_exact_and_never_bypasses_provider_admission(worker, sdk_transport, request_limit):
    worker.session.close()
    worker.session = ResponsesSession.create(worker.store.backend.common_dir / "large-responses",
        replace(worker.session.binding, max_request_bytes=request_limit), config())
    sent = install(sdk_transport, [message("done")])
    conversation = make(worker)
    conversation.observe("instruction", LONG_TASK)
    if request_limit == 1024:
        with pytest.raises(ValueError, match="request exceeds"):
            asyncio.run(conversation.step())
        assert not sent and worker.session.known_cost_usd == 0
    else:
        asyncio.run(conversation.step())
        assert json.loads(sent[0].content)["input"][0]["content"] == LONG_TASK
        assert conversation.messages[0]["content"] == LONG_TASK


def test_oversize_inputs_remain_rejected_before_launch_or_memory_publication(tmp_path, goal, policy, worker):
    root = tmp_path / "workspace"
    root.mkdir()
    with pytest.raises(GoalInputError, match="512 KiB"):
        preparation_bytes(root, "large", replace(goal, task="x" * MAX_GOAL_INPUT_BYTES), policy,
                          max_generations=1, wall_clock_seconds=120)
    assert not list(root.iterdir())
    path = tmp_path / "oversize.json"
    raw = b" " * (MAX_GOAL_INPUT_BYTES + 1)
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    with pytest.raises(GoalInputError):
        AzureGoalService._input(path, digest, "prepare")
    with pytest.raises(GoalInputError):
        load_goal_input(path, digest)
    with pytest.raises(GoalInputError):
        perform(path, digest, tmp_path / "exchange", operation="prepare")
    assert not (tmp_path / "exchange").exists()
    conversation = make(worker)
    before = worker.session.conversation_audit()
    with pytest.raises(ValueError, match="512 KiB"):
        conversation.observe("too-big", "☃" * (MAX_WORKER_OBSERVATION_BYTES // 3 + 1))
    assert worker.session.conversation_audit() == before
