"""The Harbor agent's own logic, with Harbor itself replaced by the two names it imports.

Harbor runs only in its own environment on the benchmark host, so the full
agent is exercised there by drills; here are its settings, its refusals before
any work, and the record it leaves when there is no settled evidence.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
from types import SimpleNamespace

import pytest


@pytest.fixture
def harbor_agent(monkeypatch):
    base, capabilities = types.ModuleType("harbor.agents.base"), types.ModuleType("harbor.agents.capabilities")

    class BaseAgent:
        def __init__(self, logs_dir, model_name=None, **kwargs):
            self.logs_dir, self.model_name, self.extra = logs_dir, model_name, kwargs

    class AgentCapabilities:
        def __init__(self, **flags):
            self.flags = flags

    base.BaseAgent, capabilities.AgentCapabilities = BaseAgent, AgentCapabilities
    for name, module in (("harbor", types.ModuleType("harbor")), ("harbor.agents", types.ModuleType("harbor.agents")),
                         ("harbor.agents.base", base), ("harbor.agents.capabilities", capabilities)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "taste.benchmarks.harbor_agent", raising=False)
    module = importlib.import_module("taste.benchmarks.harbor_agent")
    yield module
    sys.modules.pop("taste.benchmarks.harbor_agent", None)


def test_settings_come_from_harbors_model_and_agent_options(harbor_agent, tmp_path):
    agent = harbor_agent.TasteAgent(tmp_path, model_name="azure/gpt-5.6-luna", agent="mini-swe-agent",
                                    services="none", spend_cap_usd="2", worker_python="/usr/bin/python3",
                                    unrelated_option="kept for Harbor")
    assert (agent.settings.model, agent.settings.agent, agent.settings.services) == (
        "gpt-5.6-luna", "mini-swe-agent", "none")
    assert agent.settings.spend_cap_usd == 2.0 and agent.settings.generations == 1
    assert agent.extra == {"unrelated_option": "kept for Harbor"}
    assert harbor_agent.TasteAgent.name() == "taste"


@pytest.mark.parametrize("options,match", [({"model_name": "azure/gpt-4o"}, "model must be one of"),
                                           ({"services": "none"}, "hosted agent"),
                                           ({"agent": "claude-code"}, "agent must be one of")])
def test_bad_settings_are_refused_when_the_agent_is_made(harbor_agent, tmp_path, options, match):
    with pytest.raises(ValueError, match=match):
        harbor_agent.TasteAgent(tmp_path, **options)


def test_setup_refuses_any_environment_but_harbors_local_docker(harbor_agent, tmp_path):
    agent = harbor_agent.TasteAgent(tmp_path, model_name="azure/gpt-6-astra", worker_python="/usr/bin/python3")

    class ModalEnvironment:
        pass

    with pytest.raises(RuntimeError, match="local Docker environment"):
        asyncio.run(agent.setup(ModalEnvironment()))


def test_a_trial_without_settled_evidence_still_leaves_a_flagged_record(harbor_agent, tmp_path):
    agent = harbor_agent.TasteAgent(tmp_path, model_name="azure/gpt-6-astra", worker_python="/usr/bin/python3")
    agent.owner = SimpleNamespace(audit_flags=("settlement_failed:OSError",), outcome=None, trajectory_path=None,
                                  root=tmp_path / "trial", sealed=True)
    context = SimpleNamespace(metadata={"taste": {"trial": "t", "configuration": {}}})
    agent._record("goal-1", "Fix the build.", None, context, started=0.0)
    record = json.loads((tmp_path / "trajectory.json").read_text())
    assert "settlement_failed:OSError" in record["extra"]["audit_flags"]
    summary = context.metadata["taste"]
    assert summary["sealed"] is True and summary["audit_flags"] == record["extra"]["audit_flags"]
    assert "stop_reason" not in summary  # no outcome, so none is claimed


def _settled(tmp_path, *runs):
    """A settled record holding the coordinator's run and these (name, step sources) runs."""
    nested = {"steps": [], "subagent_trajectories": [
        {"agent": {"name": "taste-coordinator"}, "steps": [{"source": "system"}, {"source": "agent"}]},
        *({"agent": {"name": name}, "steps": [{"source": source} for source in sources]} for name, sources in runs)]}
    path = tmp_path / "settled.json"
    path.write_text(json.dumps(nested))
    return path


@pytest.mark.parametrize("runs,steps", [
    ((), 0),                                                        # the launch failed: no run of the agent
    ((("taste-azure-worker", ["system"]),), 0),                     # a run that never reached the agent
    ((("taste-monitor", ["agent"]), ("mini-swe-agent", ["user"])), 0),  # the agent got its task, no reply
    ((("mini-swe-agent", ["user", "agent", "agent"]),), 2),
])
def test_the_record_counts_the_hosted_agents_steps_and_a_run_without_one_is_not_graded(
        harbor_agent, tmp_path, runs, steps):
    agent = harbor_agent.TasteAgent(tmp_path, model_name="azure/gpt-6-luna", agent="mini-swe-agent",
                                    services="none", worker_python="/usr/bin/python3")
    agent.owner = SimpleNamespace(audit_flags=(), outcome=None, trajectory_path=_settled(tmp_path, *runs),
                                  root=tmp_path / "trial", sealed=True)
    context = SimpleNamespace(metadata={"taste": {"trial": "t", "configuration": {}}})
    agent._record("goal-1", "Fix the build.", None, context, started=0.0)
    assert context.metadata["taste"]["agent_steps"] == steps
    ended = SimpleNamespace(stop_reason="generation_bound")
    if steps:
        agent._check_started(ended, context)
    else:
        # The harness failed the agent, not the agent the task: Harbor retries such a trial.
        with pytest.raises(harbor_agent.HostedAgentNotStarted, match="never took a step"):
            agent._check_started(ended, context)
    # A run cut short by the benchmark's own time limit is graded as it stands.
    agent._check_started(None, context)


def test_without_a_settled_record_or_a_hosted_agent_no_step_count_is_claimed(harbor_agent, tmp_path):
    context = SimpleNamespace(metadata={"taste": {"trial": "t", "configuration": {}}})
    agent = harbor_agent.TasteAgent(tmp_path, model_name="azure/gpt-6-luna", agent="mini-swe-agent",
                                    services="none", worker_python="/usr/bin/python3")
    agent.owner = SimpleNamespace(audit_flags=(), outcome=None, trajectory_path=None,
                                  root=tmp_path / "trial", sealed=True)
    agent._record("goal-1", "Fix the build.", None, context, started=0.0)
    assert context.metadata["taste"]["agent_steps"] is None
    agent._check_started(SimpleNamespace(stop_reason="complete"), context)
    # Taste's own worker is no hosted agent: its runs are not counted as one.
    native = harbor_agent.TasteAgent(tmp_path / "native", model_name="azure/gpt-6-luna",
                                     worker_python="/usr/bin/python3")
    native.owner = SimpleNamespace(audit_flags=(), outcome=None, root=tmp_path / "trial", sealed=True,
                                   trajectory_path=_settled(tmp_path, ("taste-azure-worker", ["agent"])))
    (tmp_path / "native").mkdir()
    native._record("goal-1", "Fix the build.", None, context, started=0.0)
    assert context.metadata["taste"]["agent_steps"] is None
    native._check_started(SimpleNamespace(stop_reason="complete"), context)
