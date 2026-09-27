"""Exact Azure goal admission and real outer-process run/settle/replay."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import time
from dataclasses import replace
from datetime import datetime

import pytest

from taste.brains.azure_execution_policy import POLICY_KEY, AzureExecutionPolicy
from taste.brains.azure_goal_credentials import GOAL_CREDENTIAL_NAME, encode_azure_goal_credentials
from taste.brains.azure_goal_entrypoint import (
    azure_goal_command,
    execute_azure_goal,
    prepare_azure_goal_process,
)
from taste.brains.goal_entrypoint import GoalInputError, load_goal_input
from tests.test_azure_central_host import environment
from tests.test_azure_central_host import goal as _goal
from tests.test_azure_central_host import policy as _policy
from tests.test_azure_openai import sdk_transport as _sdk_transport

goal = _goal
policy = _policy
sdk_transport = _sdk_transport


@pytest.fixture
def prepared(tmp_path, goal, policy, sdk_transport):
    sent, _ = sdk_transport(lambda wire: pytest.fail("preparation cannot dispatch"))
    root = tmp_path / "repo"
    root.mkdir()
    config = prepare_azure_goal_process(
        root, "azure-entrypoint", goal, policy=replace(policy, deadline_unix=time.time() + 25), environment=environment(),
        max_generations=2, wall_clock_seconds=30, max_planner_failures=1,
    )
    assert not sent
    return config


BOOTSTRAP = r'''
import importlib.abc, json, os, httpx
class NoClaude(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('claude_agent_sdk'):
            raise AssertionError('Azure goal imported the Claude SDK')
sys.meta_path.insert(0, NoClaude())
OriginalClient = httpx.Client
def handle(wire):
    request = json.loads(wire.content)
    assert str(wire.url) == 'https://test-resource.openai.azure.com/openai/v1/responses'
    assert wire.headers['authorization'] == 'Bearer azure-test-only'
    assert request['model'] == 'gpt-6-astra'
    payload = json.loads(request['input'][0]['content'])
    result = payload['required_output_shape']
    result.update(assignments=[], complete=True, completion_reason='test completion', rationale='test witness')
    result['assessment'] = [dict(criterion_id=item['criterion_id'], verdict='met', evidence='test witness')
                            for item in payload['standing_criteria']]
    with open(os.environ['TASTE_TEST_WIRE_COUNT'], 'a') as output:
        output.write('one Azure planner request\n')
    return httpx.Response(200, json={
        'id': 'resp_test', 'object': 'response', 'created_at': 1,
        'model': 'gpt-6-astra-2026-09-03', 'status': 'completed',
        'output': [{'type': 'message', 'id': 'msg_test', 'role': 'assistant', 'status': 'completed',
                    'content': [{'type': 'output_text', 'text': json.dumps(result), 'annotations': []}]}],
        'usage': {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120,
                  'input_tokens_details': {'cached_tokens': 30, 'cache_write_tokens': 40},
                  'output_tokens_details': {'reasoning_tokens': 10}},
    })
class Client(OriginalClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, transport=httpx.MockTransport(handle))
httpx.Client = Client
'''


@pytest.mark.parametrize("systemd_credentials", [False, True])
def test_real_azure_goal_process_and_settlement_never_fall_back_or_repeat_paid_request(prepared, tmp_path, systemd_credentials):
    raw = prepared.to_bytes()
    path = tmp_path / "input.json"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert load_goal_input(path, digest) == prepared
    counter = tmp_path / "wire-count"
    env = {**os.environ, **environment(), "TASTE_TEST_WIRE_COUNT": str(counter),
           "ANTHROPIC_API_KEY": "claude-do-not-use", "OPENAI_API_KEY": "personal-do-not-use",
           "OPENAI_BASE_URL": "https://wrong.invalid/v1"}
    if systemd_credentials:
        credentials = tmp_path / "service-credentials"
        credentials.mkdir(mode=0o700)
        secret = credentials / GOAL_CREDENTIAL_NAME
        secret.write_bytes(encode_azure_goal_credentials(prepared, "azure-test-only"))
        secret.chmod(0o400)
        env.pop("AZURE_OPENAI_API_KEY")
        env.pop("AZURE_OPENAI_BASE_URL")
        env["CREDENTIALS_DIRECTORY"] = str(credentials)
    for mode in ("run", "run", "settle"):
        argv = azure_goal_command(path, digest, mode=mode, systemd_credentials=systemd_credentials)
        argv[3] = argv[3].replace("from taste.brains.azure_goal_entrypoint", BOOTSTRAP + "\nfrom taste.brains.azure_goal_entrypoint", 1)
        child_env = dict(env)
        if mode == "settle":
            child_env.pop("AZURE_OPENAI_API_KEY", None)
            child_env.pop("AZURE_OPENAI_BASE_URL", None)
            if systemd_credentials:
                secret.unlink()
        result = subprocess.run(argv, cwd=tmp_path, env=child_env, capture_output=True, text=True, timeout=40)
        assert result.returncode == 0, (mode, result.stdout, result.stderr)
        assert counter.read_text().splitlines() == ["one Azure planner request"]
        assert "personal-do-not-use" not in result.stdout + result.stderr
    assert "azure-test-only" not in raw.decode()


@pytest.mark.parametrize("damage", ["missing", "mode", "hardlink", "symlink", "fifo", "large", "input",
                                     "extra", "key", "duplicate", "directory_mode"])
def test_bad_private_goal_credentials_fail_before_host_or_paid_admission(prepared, tmp_path, monkeypatch, damage):
    directory = tmp_path / "private-credentials"
    directory.mkdir(mode=0o700)
    path = directory / GOAL_CREDENTIAL_NAME
    value = json.loads(encode_azure_goal_credentials(prepared, "azure-test-only"))
    if damage == "input":
        value["input_sha256"] = "f" * 64
    elif damage == "extra":
        value["fallback"] = "https://api.openai.com/v1"
    elif damage == "key":
        value["api_key"] = "invalid\x00key"
    raw = json.dumps(value).encode()
    if damage == "large":
        raw = b"x" * 65537
    elif damage == "duplicate":
        raw = raw[:-1] + b', "api_key": "replaced"}'
    path.write_bytes(raw)
    path.chmod(0o400)
    if damage == "missing":
        path.unlink()
    elif damage == "mode":
        path.chmod(0o644)
    elif damage == "hardlink":
        os.link(path, directory / "alias")
    elif damage in {"symlink", "fifo"}:
        path.rename(directory / "saved")
        if damage == "symlink":
            path.symlink_to(directory / "saved")
        else:
            os.mkfifo(path)
    elif damage == "directory_mode":
        directory.chmod(0o777)
    async def cannot_compose(*_args, **_kwargs):
        pytest.fail("credential rejection must precede host composition")
    monkeypatch.setattr("taste.brains.azure_goal_entrypoint.execute_goal", cannot_compose)
    with pytest.raises(GoalInputError, match="private Azure goal credentials were rejected"):
        asyncio.run(execute_azure_goal(prepared, systemd_credentials=True,
                                     environment={"CREDENTIALS_DIRECTORY": str(directory)}))


def test_settlement_before_first_plan_needs_no_credential_or_paid_request(prepared):
    result = asyncio.run(execute_azure_goal(prepared, mode="settle", environment={}))
    assert not result.complete and result.stop_reason == "cancelled"
    assert result.budget.enforceable and result.budget.known_spent_usd == 0


@pytest.mark.parametrize("field,value", [
    ("planner_model", "claude-opus-4-6"), ("monitor_model", "claude-haiku-4-5"),
    ("planner_max_tokens", 513),
])
def test_changed_launch_options_fail_before_any_host_or_provider(prepared, field, value):
    with pytest.raises(GoalInputError, match="exact Azure"):
        asyncio.run(execute_azure_goal(replace(prepared, **{field: value}), environment={}))


def test_prepared_policy_and_absolute_deadline_roundtrip(prepared):
    policy = AzureExecutionPolicy.from_dict(prepared.goal.metadata[POLICY_KEY])
    assert policy.to_dict() == prepared.goal.metadata[POLICY_KEY]
    expected = datetime.fromtimestamp(policy.deadline_unix).timestamp()
    assert datetime.fromisoformat(prepared.limits["deadline_at"]).timestamp() == expected
    policy.to_dict()["worker_max_calls"] = 500
    assert prepared.goal.metadata[POLICY_KEY]["worker_max_calls"] == 8


@pytest.mark.parametrize("change", ["missing", "extra", "schema", "endpoint"])
def test_policy_decode_refuses_missing_or_unadmitted_fields(policy, change):
    raw = policy.to_dict()
    if change == "missing":
        raw.pop("worker_max_calls")
    elif change == "extra":
        raw["api_key"] = "must-not-be-accepted"
    elif change == "schema":
        raw["schema"] = "foreign/1"
    else:
        raw["endpoint"] = "https://api.openai.com/v1"
    with pytest.raises((ValueError, RuntimeError)):
        AzureExecutionPolicy.from_dict(raw)
