"""A study can show a hosted mini-swe-agent the machine its recorded runs had, so its prompts stay the same.

mini-swe-agent writes the host's ``uname`` into its task message. A run recorded on one host and branched on another
would otherwise differ at its first request, and the branch's fidelity check (rightly) stops it.
"""

from __future__ import annotations

import json
import platform
import time

import pytest

from taste.agents import HostedStop, ModelReply, hosted_agent, uname_override
from taste.benchmarks.harbor_settings import TrialSettings
from taste.brains.azure_execution_policy import AzureExecutionPolicy
from taste.brains.azure_worker_policy import AZURE_WORKER_POLICY_SCHEMA, AzureWorkerPolicy
from taste.brains.branch_replay import ReplayHost
from taste.brains.contract import Contract
from taste.brains.records import Assignment, contract_digest
from taste.brains.worker_admission import EntrypointInputError
from taste.pricing import table_sha
from taste.providers.azure_openai import AZURE_WORKER_MODEL
from taste.recovery_study import jobs
from tests.test_branch_replay import Live, Recorder

ENDPOINT = "https://test-resource.openai.azure.com/openai/v1/"
SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
AZURE = {"system": "Linux", "release": "7.0.0-1014-azure",
         "version": "#14-Ubuntu SMP PREEMPT Fri Aug 14 16:21:58 UTC 2026", "machine": "x86_64"}
AWS = platform.uname_result("Linux", "ip-172-31-3-184", "7.0.0-1014-aws",
                            "#14-Ubuntu SMP PREEMPT Thu Sep 17 19:08:36 UTC 2026", "x86_64")
AZURE_HOST = platform.uname_result("Linux", "taste-test-eus2", AZURE["release"], AZURE["version"], "x86_64")
AZURE_LINE = "Linux 7.0.0-1014-azure #14-Ubuntu SMP PREEMPT Fri Aug 14 16:21:58 UTC 2026 x86_64"
ALONE = {"model": "gpt-5.6-luna", "agent": "mini-swe-agent", "services": "none"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def call(command, call_id):
    return {"type": "function_call", "id": "fc_" + call_id, "call_id": call_id, "name": "bash",
            "arguments": json.dumps({"command": command}), "status": "completed"}


def reply(command, call_id):
    return ModelReply(output=(call(command, call_id),), status="completed", incomplete_reason=None,
                      usage={"input_tokens": 10, "output_tokens": 5}, model="gpt-5.6-luna-2026-07-09",
                      cost_usd=0.01)


OUTPUTS = {"make test": ("FAILED test_parse\n", 1), f"echo {SENTINEL}": (f"{SENTINEL}\n", 0)}


def mini_swe_agent():
    """The hosted mini-swe-agent; its tests run where mini-swe-agent is installed (the study host)."""
    pytest.importorskip("minisweagent")
    from taste.agents.mini_swe_agent import MiniSweAgent

    return MiniSweAgent


def on_host(monkeypatch, uname):
    monkeypatch.setattr("taste.agents.mini_swe_agent.platform.uname", lambda: uname)


def assignment(**changes):
    """A worker's assignment with an Azure worker policy (as tests.test_azure_worker_policy writes one)."""
    brief = Contract("worker", "produce the artifact", success_criteria=("output is correct",), budget_usd=30,
                     max_turns=4)
    raw = {"schema": AZURE_WORKER_POLICY_SCHEMA, "endpoint": ENDPOINT, "worker_deployment": "gpt-6-sol",
           "monitor_deployment": "gpt-6-sol", "deadline_unix": time.time() + 1500, "worker_max_calls": 4,
           "worker_max_output_tokens": 128, "monitor_max_calls": 3, "monitor_max_output_tokens": 256,
           "max_request_bytes": 196_608, "monitor_batch_size": 10, "pricing_sha": table_sha(), **changes}
    return Assignment("task", 1, 0, brief, contract_digest(brief), "a" * 40, model=AZURE_WORKER_MODEL,
                      resources={"azure_openai": raw, "monitor_budget_usd": 20})


def policy_for(settings):
    return settings.policy(ENDPOINT, time.time() + 1500, owner_token="a" * 32, container_id="c" * 64,
                           workdir="/app")


# -- the value


def test_a_system_line_is_one_exact_json_object_of_unames_fields():
    assert uname_override("") == {}
    assert uname_override(canonical(AZURE)) == AZURE
    assert uname_override(AZURE) == AZURE  # as Harbor's --ak hands a JSON value over: already parsed
    assert uname_override({**AZURE, "node": "taste-test-eus2"})["node"] == "taste-test-eus2"
    for bad in ("Linux 7.0.0", "[]", "{}", canonical({**AZURE, "processor": "x"}),
                canonical({key: value for key, value in AZURE.items() if key != "version"}),
                canonical({**AZURE, "release": 7}), canonical({**AZURE, "release": ""}),
                canonical({**AZURE, "version": "two\nlines"}), 7):
        with pytest.raises(ValueError, match="agent_uname"):
            uname_override(bad)


# -- the agent


def first_task_message(agent):
    """The task message the agent sends with its first request."""
    seen = []

    class First(Live):
        def ask(self, *, messages, tools, effort):
            seen.append(messages[1]["content"])
            raise HostedStop("enough")

    with pytest.raises(HostedStop):
        agent.run("Fix it.", First(), cwd="/app", model_name="m")
    return seen[0]


def test_the_agents_task_message_carries_the_given_system_line_not_the_hosts(monkeypatch):
    MiniSweAgent = mini_swe_agent()
    on_host(monkeypatch, AWS)
    given = first_task_message(MiniSweAgent(effort="low", uname=canonical(AZURE)))
    assert AZURE_LINE in given and AWS.release not in given
    own = first_task_message(MiniSweAgent(effort="low"))
    assert " ".join((AWS.system, AWS.release, AWS.version, AWS.machine)) in own and AZURE["release"] not in own
    # Its identity, which a branch compares with the record's, is the configuration's alone.
    assert MiniSweAgent(effort="low", uname=canonical(AZURE)).identity() == MiniSweAgent(effort="low").identity()


def test_a_run_recorded_on_one_host_branches_faithfully_on_another_given_that_hosts_line(monkeypatch):
    MiniSweAgent = mini_swe_agent()
    on_host(monkeypatch, AZURE_HOST)
    recorder = Recorder(Live([reply("make test", "c1"), reply(f"echo {SENTINEL}", "c2")], OUTPUTS))
    assert MiniSweAgent(effort="low").run("Fix it.", recorder, cwd="/app", model_name="m").exit_status == "Submitted"
    script = recorder.script()
    on_host(monkeypatch, AWS)
    # Shown the recorded host's line, the first request is the recorded one: step 1 replays, then it goes live.
    live, events = Live([reply(f"echo {SENTINEL}", "c3")], OUTPUTS), []
    host = ReplayHost(live, script, 1, record=lambda kind, payload: events.append((kind, payload)))
    result = MiniSweAgent(effort="low", uname=canonical(AZURE)).run("Fix it.", host, cwd="/app", model_name="m")
    assert result.exit_status == "Submitted" and host.faithful and host.replayed == 1
    assert events[-1][1] == {"live": True, "context": "matched"} and live.asked == [script.steps[1].request_sha]
    # Shown this host's own line, the first request differs and the branch stops before any model call.
    other = Live([], OUTPUTS)
    stopped = ReplayHost(other, script, 1, record=lambda kind, payload: None)
    with pytest.raises(HostedStop):
        MiniSweAgent(effort="low").run("Fix it.", stopped, cwd="/app", model_name="m")
    assert not stopped.faithful and other.asked == []


def test_only_mini_swe_agent_takes_a_system_line():
    # Refused before the agent's module is imported.
    for name in ("swe-agent", "checker"):
        with pytest.raises(ValueError, match="agent_uname"):
            hosted_agent(name, effort="low", uname=canonical(AZURE))
    agent = hosted_agent("mini-swe-agent", effort="low", uname=json.dumps(AZURE)) if mini_swe_agent() else None
    assert agent.uname == AZURE


# -- the settings, the policies, the worker


def test_the_setting_reaches_the_policy_the_worker_and_the_disclosure_only_when_set():
    given = TrialSettings.from_options({**ALONE, "agent_uname": canonical(AZURE)})
    # Harbor parses a JSON --ak value before the agent sees it: the same setting either way.
    assert TrialSettings.from_options({**ALONE, "agent_uname": dict(AZURE)}) == given
    assert given.agent_uname == canonical(AZURE)
    assert given.disclosure()["agent_uname"] == AZURE
    on, off = policy_for(given), policy_for(TrialSettings.from_options(ALONE))
    assert on.worker_uname == canonical(AZURE) and off.worker_uname == ""
    # Off, the policy keeps its exact earlier wire form; on, it round-trips.
    assert "worker_uname" not in off.to_dict() and "worker_uname" not in off.worker_resources()
    assert "agent_uname" not in TrialSettings.from_options(ALONE).disclosure()
    assert on.to_dict()["worker_uname"] == on.worker_resources()["worker_uname"] == canonical(AZURE)
    assert AzureExecutionPolicy.from_dict(json.loads(json.dumps(on.to_dict()))) == on
    with pytest.raises(ValueError, match="fields or schema"):
        AzureExecutionPolicy.from_dict({**off.to_dict(), "worker_uname": ""})
    worker = AzureWorkerPolicy.from_assignment(assignment(worker_agent="mini-swe-agent", services="none",
                                                          worker_uname=canonical(AZURE)))
    assert worker.uname == canonical(AZURE) and worker.agent == "mini-swe-agent"
    assert AzureWorkerPolicy.from_assignment(assignment(worker_agent="mini-swe-agent")).uname == ""


def test_a_system_line_is_refused_where_it_cannot_apply():
    for options in ({**ALONE, "agent_uname": "Linux"}, {**ALONE, "agent": "swe-agent", "agent_uname": canonical(AZURE)},
                    {"model": "gpt-5.6-luna", "agent_uname": canonical(AZURE)}):
        with pytest.raises(ValueError, match="agent_uname"):
            TrialSettings.from_options(options)
    on = policy_for(TrialSettings.from_options({**ALONE, "agent_uname": canonical(AZURE)}))
    # Only the exact canonical form is admitted, so one setting has one digest.
    spaced = json.dumps(AZURE)
    with pytest.raises(ValueError, match="agent_uname"):
        AzureExecutionPolicy.from_dict({**on.to_dict(), "worker_uname": spaced})
    for damage in ({"worker_uname": canonical(AZURE)},
                   {"worker_agent": "swe-agent", "services": "none", "worker_uname": canonical(AZURE)},
                   {"worker_agent": "mini-swe-agent", "services": "none", "worker_uname": spaced},
                   {"worker_agent": "mini-swe-agent", "services": "none", "worker_uname": ""}):
        with pytest.raises(EntrypointInputError):
            AzureWorkerPolicy.from_assignment(assignment(**damage))


def test_a_recorded_setting_goes_back_to_the_command_line_as_the_json_harbor_reads():
    # A later round inherits its settings from a trial's config.json, where Harbor kept the parsed object.
    spec = jobs.harbor_job("p1-x-ct-e1r2-a1", tasks_dir="/t", task="t1", attempts=1, model="gpt-6-luna",
                           jobs_dir="/j", settings={"agent": "mini-swe-agent", "agent_uname": dict(AZURE)})
    [value] = [part.split("=", 1)[1] for part in spec.argv if part.startswith("agent_uname=")]
    assert value == canonical(AZURE) and json.loads(value) == AZURE
