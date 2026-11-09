"""A study can give mini-swe-agent a longer limit on each of its commands, for the tasks whose builds need one.

mini-swe-agent's environment ends a command after 30 seconds (its class default; the "mini" configuration sets none).
Luna's Go builds, vets and tests were ended by it 52 to 60% of the time (ISSTA plan, item 6). ``agent_command_seconds``
sets the limit for one trial: it reaches the agent's environment through the policies, as ``agent_uname`` does, and is
disclosed with the trial's settings. It is no part of the agent's identity, so a branch from a run recorded at 30 s
replays that run as it was and only its live commands get the new limit.
"""

from __future__ import annotations

import json

import pytest

from taste.agents import hosted_agent
from taste.benchmarks.harbor_settings import TrialSettings
from taste.brains.azure_execution_policy import AzureExecutionPolicy
from taste.brains.azure_worker_policy import AzureWorkerPolicy
from taste.brains.branch_replay import ReplayHost
from taste.brains.worker_admission import EntrypointInputError
from taste.recovery_study import jobs
from tests.test_agent_uname import (
    ALONE,
    OUTPUTS,
    SENTINEL,
    assignment,
    mini_swe_agent,
    policy_for,
    reply,
)
from tests.test_branch_replay import Live, Recorder


class Timed(Live):
    """The live side, writing down the time limit each command was given."""

    def __init__(self, replies=(), outputs=None):
        super().__init__(replies, outputs)
        self.limits = []

    def run(self, command, *, cwd, timeout_seconds, shown=None):
        self.limits.append(timeout_seconds)
        return super().run(command, cwd=cwd, timeout_seconds=timeout_seconds, shown=shown)


def two_steps(first="c1", second="c2"):
    return [reply("make test", first), reply(f"echo {SENTINEL}", second)]


# -- the settings, the policies, the worker


def test_the_limit_is_off_by_default_and_reaches_the_policy_the_worker_and_the_disclosure_only_when_set():
    off = TrialSettings.from_options(ALONE)
    assert off.agent_command_seconds == 0 and "agent_command_seconds" not in off.disclosure()
    given = TrialSettings.from_options({**ALONE, "agent_command_seconds": "120"})
    assert given.agent_command_seconds == 120 and given.disclosure()["agent_command_seconds"] == 120
    on, before = policy_for(given), policy_for(off)
    assert on.worker_command_seconds == 120 and before.worker_command_seconds == 0
    # Off, the policy keeps its exact earlier wire form; on, it round-trips.
    assert "worker_command_seconds" not in before.to_dict() and "worker_command_seconds" not in before.worker_resources()
    assert on.to_dict()["worker_command_seconds"] == on.worker_resources()["worker_command_seconds"] == 120
    assert type(on.worker_resources()["worker_command_seconds"]) is int
    assert AzureExecutionPolicy.from_dict(json.loads(json.dumps(on.to_dict()))) == on
    with pytest.raises(ValueError, match="fields or schema"):
        AzureExecutionPolicy.from_dict({**before.to_dict(), "worker_command_seconds": 0})
    worker = AzureWorkerPolicy.from_assignment(assignment(worker_agent="mini-swe-agent", services="none",
                                                          worker_command_seconds=120))
    assert worker.command_seconds == 120 and worker.agent == "mini-swe-agent"
    assert AzureWorkerPolicy.from_assignment(assignment(worker_agent="mini-swe-agent")).command_seconds == 0


def test_the_limit_is_refused_where_it_cannot_apply():
    for options in ({**ALONE, "agent_command_seconds": "-1"},
                    # Above the terminal's own limit on any command, which would cut it short silently.
                    {**ALONE, "agent_command_seconds": "601"},
                    {**ALONE, "agent": "swe-agent", "agent_command_seconds": "120"},
                    {"model": "gpt-5.6-luna", "agent_command_seconds": "120"}):
        with pytest.raises(ValueError, match="agent_command_seconds"):
            TrialSettings.from_options(options)
    raised = TrialSettings.from_options({**ALONE, "command_seconds": "900", "agent_command_seconds": "700"})
    assert policy_for(raised).worker_command_seconds == 700
    on = policy_for(TrialSettings.from_options({**ALONE, "agent_command_seconds": "120"}))
    for damage in ({"worker_command_seconds": "120"}, {"worker_command_seconds": True},
                   {"worker_command_seconds": 99999}, {"worker_agent": "swe-agent"}):
        with pytest.raises(ValueError, match=r"agent_command_seconds|fields or schema"):
            AzureExecutionPolicy.from_dict({**on.to_dict(), **damage})
    for damage in ({"worker_command_seconds": 120},
                   {"worker_agent": "swe-agent", "services": "none", "worker_command_seconds": 120},
                   {"worker_agent": "mini-swe-agent", "services": "none", "worker_command_seconds": 0},
                   {"worker_agent": "mini-swe-agent", "services": "none", "worker_command_seconds": "120"},
                   {"worker_agent": "mini-swe-agent", "services": "none", "worker_command_seconds": True}):
        with pytest.raises(EntrypointInputError):
            AzureWorkerPolicy.from_assignment(assignment(**damage))


def test_the_worker_builds_its_hosted_agent_with_the_limit_its_assignment_holds(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from taste.brains import azure_worker_entrypoint

    built = []
    monkeypatch.setattr("taste.agents.hosted_agent", lambda name, **options: built.append((name, options)) or name)
    monkeypatch.setattr("taste.brains.hosted_worker.HostedWorkerRuntime", lambda *args, **kwargs: kwargs)
    monkeypatch.setenv("MSWEA_GLOBAL_CONFIG_DIR", str(tmp_path))  # the worker sets both; the test puts them back
    monkeypatch.setenv("MSWEA_SILENT_STARTUP", "1")
    for given, limit in (({"worker_command_seconds": 120}, 120), ({}, 0)):
        policy = AzureWorkerPolicy.from_assignment(assignment(worker_agent="mini-swe-agent", services="none", **given))
        runtime = azure_worker_entrypoint._hosted_runtime(policy, None, SimpleNamespace(model="m"), None, None, None,
                                                          tmp_path)
        assert runtime["agent"] == "mini-swe-agent" and built[-1][1]["command_seconds"] == limit


def test_a_recorded_limit_goes_back_to_the_command_line_as_harbor_reads_it():
    # A later round inherits its settings from a trial's config.json, where Harbor kept the number it parsed.
    spec = jobs.harbor_job("p1-x-ct-e1r2-a1", tasks_dir="/t", task="t1", attempts=1, model="gpt-6-luna",
                           jobs_dir="/j", settings={"agent": "mini-swe-agent", "agent_command_seconds": 120})
    [value] = [part.split("=", 1)[1] for part in spec.argv if part.startswith("agent_command_seconds=")]
    assert value == "120" and TrialSettings.from_options({**ALONE, "agent_command_seconds": value}) == \
        TrialSettings.from_options({**ALONE, "agent_command_seconds": 120})


# -- the agent


def test_only_mini_swe_agent_takes_a_command_limit():
    # Refused before the agent's module is imported.
    for name in ("swe-agent", "checker"):
        with pytest.raises(ValueError, match="agent_command_seconds"):
            hosted_agent(name, effort="low", command_seconds=120)
    MiniSweAgent = mini_swe_agent()
    agent = hosted_agent("mini-swe-agent", effort="low", command_seconds=120)
    assert agent.command_seconds == 120 and hosted_agent("mini-swe-agent", effort="low").command_seconds == 0
    # Its identity, which a branch compares with the record's, is the configuration's alone.
    assert agent.identity() == MiniSweAgent(effort="low").identity()
    for bad in (-1, 1.5, "120", True):
        with pytest.raises(ValueError, match="command limit"):
            MiniSweAgent(effort="low", command_seconds=bad)


def test_every_command_the_agent_runs_gets_the_limit_and_without_one_the_configurations_30_seconds():
    MiniSweAgent = mini_swe_agent()
    longer = Timed(two_steps(), OUTPUTS)
    assert MiniSweAgent(effort="low", command_seconds=120).run("Fix it.", longer, cwd="/app",
                                                               model_name="m").exit_status == "Submitted"
    assert longer.limits == [120, 120]
    own = Timed(two_steps(), OUTPUTS)
    MiniSweAgent(effort="low").run("Fix it.", own, cwd="/app", model_name="m")
    assert own.limits == [30, 30]


def test_a_command_the_longer_limit_ends_is_reported_as_the_agent_reports_any_timeout():
    MiniSweAgent = mini_swe_agent()
    from taste.agents import ShellResult

    class Slow(Timed):
        def run(self, command, *, cwd, timeout_seconds, shown=None):
            self.limits.append(timeout_seconds)
            if shown == "make test":
                return ShellResult("building...\n", -1, timed_out=True)
            return super().run(command, cwd=cwd, timeout_seconds=timeout_seconds, shown=shown)

    slow = Slow(two_steps(), OUTPUTS)
    agent = MiniSweAgent(effort="low", command_seconds=120)
    result = agent.run("Fix it.", slow, cwd="/app", model_name="m")
    observations = json.dumps(list(result.messages), default=str)
    assert "timed out after 120 seconds" in observations and "after 30 seconds" not in observations


def test_a_branch_from_a_run_recorded_at_30_seconds_replays_it_and_gives_its_live_commands_the_new_limit():
    MiniSweAgent = mini_swe_agent()
    recorder = Recorder(Live(two_steps(), OUTPUTS))
    assert MiniSweAgent(effort="low").run("Fix it.", recorder, cwd="/app", model_name="m").exit_status == "Submitted"
    script = recorder.script()
    assert [run["timeout_seconds"] for step in script.to_dict()["steps"] for run in step["runs"]] == [30, 30]
    live, events = Timed([reply(f"echo {SENTINEL}", "c3")], OUTPUTS), []
    host = ReplayHost(live, script, 1, record=lambda kind, payload: events.append((kind, payload)))
    result = MiniSweAgent(effort="low", command_seconds=120).run("Fix it.", host, cwd="/app", model_name="m")
    # Step 1 is the record's (its output answered from it); the first request after it is the recorded one.
    assert result.exit_status == "Submitted" and host.faithful and host.replayed == 1
    assert events[-1][1] == {"live": True, "context": "matched"} and live.limits == [120]
