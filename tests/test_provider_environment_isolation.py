"""Task .env files cannot become provider configuration in brain processes."""

from __future__ import annotations

import asyncio
import os

import pytest

from taste.brains.central_host import compose_central_runtime
from taste.brains.worker_entrypoint import WorkerExitCode, execute_worker
from taste.llm import LLM, MODEL_MONITOR, MODEL_PLANNER
from taste.memstore import Store
from tests.test_brains_central_runtime import FakeLauncher, simple_goal
from tests.test_brains_worker_entrypoint import (
    FakeMonitorLLM,
    ScriptedClient,
    _assignment,
    _client_factory,
    _config,
    _environment,
)


@pytest.fixture
def provider_environment(monkeypatch):
    # Register every name before a failing baseline loads it, so that failure
    # cannot contaminate a later test's environment.
    for name in ("ANTHROPIC_BASE_URL", "TASTE_TASK_ENV_PROBE", "TASTE_TRUSTED_ENV_PROBE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only-inherited-key")


def task_env(directory):
    (directory / ".env").write_text(
        "ANTHROPIC_BASE_URL=https://task-provider.invalid\n"
        "TASTE_TASK_ENV_PROBE=task-controlled\n"
    )


def client_configuration(llm, model):
    """Build the installed provider client without issuing any HTTP request."""
    llm.ensure_ready(model)
    client = llm.provider_for(model)._client
    try:
        return str(client.base_url).rstrip("/"), client.api_key
    finally:
        client.close()


@pytest.mark.usefixtures("provider_environment")
@pytest.mark.parametrize("location", ["repository", "parent"])
def test_central_host_does_not_load_task_or_parent_env(tmp_path, location):
    repo = tmp_path / "repo"
    repo.mkdir()
    task_env(repo if location == "repository" else tmp_path)
    host = compose_central_runtime(repo, "provider-env", simple_goal(), launcher=FakeLauncher())
    try:
        endpoint, key = client_configuration(host.planner_llm, MODEL_PLANNER)
        assert endpoint == "https://api.anthropic.com"
        assert key == "test-only-inherited-key"
        assert "TASTE_TASK_ENV_PROBE" not in os.environ
        assert "ANTHROPIC_BASE_URL" not in os.environ
    finally:
        host.close()


@pytest.mark.usefixtures("provider_environment")
def test_worker_monitor_uses_host_configuration_before_starting_the_sdk(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    task_env(repo)
    store = Store.open(repo, "session-1")
    observed = []

    def factory(**kwargs):
        # Retain the real facade, dotenv behavior and installed provider client.
        # Actual model responses come from the existing deterministic double.
        real = LLM(**kwargs)
        observed.append(client_configuration(real, MODEL_MONITOR))
        return FakeMonitorLLM(**kwargs)

    try:
        assignment, prepared = _assignment(store)
        code = asyncio.run(execute_worker(
            _config(repo, prepared), store=store, environ=_environment(store, assignment),
            llm_factory=factory, client_factory=_client_factory(ScriptedClient()),
            ready_callback=lambda: None,
        ))
        assert code is WorkerExitCode.COMPLETED
        assert observed == [("https://api.anthropic.com", "test-only-inherited-key")]
        assert "TASTE_TASK_ENV_PROBE" not in os.environ
        assert "ANTHROPIC_BASE_URL" not in os.environ
    finally:
        store.close()


@pytest.mark.usefixtures("provider_environment")
def test_disabled_file_loading_never_searches_or_parses_dotenv(tmp_path, monkeypatch):
    task_env(tmp_path)

    def forbidden(*_args, **_kwargs):
        pytest.fail("disabled dotenv loading searched or parsed task files")

    monkeypatch.setattr("taste.llm._find_env", forbidden)
    monkeypatch.setattr("taste.llm.load_dotenv", forbidden)
    llm = LLM(env_dir=tmp_path, load_env_file=False)
    assert client_configuration(llm, MODEL_MONITOR)[1] == "test-only-inherited-key"
    assert "TASTE_TASK_ENV_PROBE" not in os.environ


@pytest.mark.usefixtures("provider_environment")
def test_explicit_host_endpoint_and_api_key_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://host-provider.invalid")
    task_env(tmp_path)
    llm = LLM(env_dir=tmp_path, load_env_file=False, api_key="test-only-explicit-key")
    assert client_configuration(llm, MODEL_MONITOR) == (
        "https://host-provider.invalid", "test-only-explicit-key",
    )


@pytest.mark.usefixtures("provider_environment")
def test_explicit_trusted_dotenv_loading_remains_available(tmp_path):
    (tmp_path / ".env").write_text("TASTE_TRUSTED_ENV_PROBE=trusted\n")
    LLM(env_dir=tmp_path)
    assert os.environ["TASTE_TRUSTED_ENV_PROBE"] == "trusted"


@pytest.mark.parametrize("invalid", [0, 1, None, "false"])
def test_nonboolean_loading_flags_are_rejected_before_reading_files(monkeypatch, invalid):
    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid loading configuration inspected the environment")

    monkeypatch.setattr("taste.llm._find_env", forbidden)
    with pytest.raises(ValueError, match="boolean"):
        LLM(load_env_file=invalid)
