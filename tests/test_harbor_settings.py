"""A trial's disclosed settings become one exact policy, goal and time split."""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

from taste.benchmarks.harbor_settings import (
    CRITERION,
    TrialSettings,
    agent_timeout_seconds,
    compose_project,
    served_model,
)
from taste.brains import benchmark_reply
from taste.brains.central_planner import SPEND_CAP_KEY
from taste.pricing import max_call_cost_usd
from taste.providers.azure_openai import AZURE_PLANNER_MODEL, AZURE_WORKER_MODEL

ENDPOINT = "https://test-resource.openai.azure.com/openai/v1/"


def policy_for(settings, deadline=None):
    return settings.policy(ENDPOINT, deadline or time.time() + 1500, owner_token="a" * 32,
                           container_id="c" * 64, workdir="/Users/alex/workspace/cli")


def test_one_named_model_runs_every_role_through_one_deployment():
    settings = TrialSettings.from_options({"model": "gpt-6-astra"})
    policy = policy_for(settings)
    assert policy.worker_model == AZURE_PLANNER_MODEL
    assert policy.planner_deployment == policy.worker_deployment == "gpt-6-astra"
    assert settings.disclosure()["monitor_model"] == AZURE_PLANNER_MODEL
    assert policy.max_assignments == 1 and policy.worker_grace_seconds == 45.0
    assert policy.terminal.workdir == "/Users/alex/workspace/cli"
    assert policy.terminal.max_timeout_seconds == 600 and policy.max_request_bytes == 1_048_576
    # The coordinator reasons unless a trial says otherwise, and the run says which.
    assert policy.planner_effort == "medium" and settings.disclosure()["coordinator_effort"] == "medium"
    default = policy_for(TrialSettings.from_options({"planner_effort": ""}))
    assert default.planner_effort == "" and "planner_effort" not in default.to_dict()
    assert TrialSettings(planner_effort="").disclosure()["coordinator_effort"] == "provider default"


def test_every_request_has_a_time_ceiling_which_the_run_discloses():
    # In 257 recorded calls the slowest worker reply took 36 seconds and the
    # slowest plan 40. Five minutes cuts off no reply that is still coming.
    settings = TrialSettings()
    assert settings.request_seconds == 300.0
    policy = policy_for(settings)
    assert policy.request_seconds == 300.0 and policy.worker_resources()["request_seconds"] == 300.0
    assert settings.disclosure()["request_seconds"] == 300.0
    chosen = TrialSettings.from_options({"request_seconds": "120"})
    assert policy_for(chosen).request_seconds == 120.0


@pytest.mark.parametrize("name,served", [("gpt-5.6-luna", "gpt-5.6-luna-2026-07-09"),
                                         ("gpt-6-sol", AZURE_WORKER_MODEL)])
def test_any_admitted_model_can_run_every_role_including_the_coordinator(name, served):
    settings = TrialSettings.from_options({"model": "azure/" + name})
    policy = policy_for(settings)
    assert policy.planner_model == policy.worker_model == served
    assert policy.planner_deployment == policy.worker_deployment == name
    disclosed = settings.disclosure()
    assert disclosed["coordinator_model"] == disclosed["worker_model"] == disclosed["monitor_model"] == served
    worker_cap, _, _ = settings.budgets()
    assert worker_cap == pytest.approx(settings.worker_spend_cap_usd + max_call_cost_usd(
        served, max_output_tokens=settings.worker_max_output_tokens, cap_on="billed"))


def test_a_trial_can_host_an_agent_in_every_worker():
    settings = TrialSettings.from_options({"model": "gpt-5.6-luna", "agent": "mini-swe-agent"})
    assert policy_for(settings).worker_agent == "mini-swe-agent"
    assert settings.disclosure()["worker_agent"] == "mini-swe-agent"
    assert TrialSettings().disclosure()["worker_agent"] == "taste"
    assert policy_for(TrialSettings()).worker_agent == ""


def test_an_agent_run_alone_has_the_fixed_plan_one_generation_and_says_so():
    settings = TrialSettings.from_options({"model": "gpt-5.6-luna", "agent": "mini-swe-agent",
                                           "services": "none"})
    policy = policy_for(settings)
    assert (policy.planner_model, policy.planner_deployment, policy.services) == (
        "taste-fixed-plan", "fixed-plan", "none")
    assert policy.worker_model == "gpt-5.6-luna-2026-07-09" and policy.worker_deployment == "gpt-5.6-luna"
    assert settings.generations == 1 and TrialSettings().generations == settings.max_generations
    disclosed = settings.disclosure()
    assert disclosed["coordinator_model"] == "taste-fixed-plan" and disclosed["services"] == "none"
    with pytest.raises(ValueError, match="hosted agent"):
        TrialSettings.from_options({"services": "none"})


def test_both_arms_are_held_to_the_same_cap_per_trial():
    worst = max_call_cost_usd("gpt-5.6-luna-2026-07-09", max_output_tokens=TrialSettings().worker_max_output_tokens,
                              cap_on="billed")
    supervised = TrialSettings.from_options({"model": "gpt-5.6-luna", "agent": "mini-swe-agent",
                                             "spend_cap_usd": "2", "worker_spend_cap_usd": "1"})
    alone = replace(supervised, services="none")
    # Supervised: the trial's $2 is shared, each worker held to its own $1.
    assert supervised.budgets()[0] == pytest.approx(1 + worst)
    assert policy_for(supervised).worker_budget_usd == pytest.approx(1 + worst)
    # Alone: the agent may spend the whole $2 the supervised trial may.
    assert alone.budgets()[0] == pytest.approx(2 + worst)
    assert alone.spend_cap_usd == supervised.spend_cap_usd == 2


def test_the_continue_control_has_the_supervised_arms_generations():
    """#34: the agent alone, run again until the supervised arm's bound or the task's time."""
    once = TrialSettings.from_options({"model": "gpt-5.6-luna", "agent": "mini-swe-agent", "services": "none"})
    again = replace(once, alone="continue")
    assert once.generations == 1 and again.generations == again.max_generations == 12
    assert again.disclosure()["alone"] == "continue" and "alone" not in replace(once, services="all").disclosure()


def test_a_cheaper_worker_model_keeps_its_own_route():
    settings = TrialSettings.from_options({"model": "gpt-6-astra", "worker_model": "gpt-6-sol",
                                           "worker_effort": "medium"})
    policy = policy_for(settings)
    assert policy.worker_model == AZURE_WORKER_MODEL and policy.worker_deployment == "gpt-6-sol"
    assert policy.worker_effort == "medium" and policy.planner_deployment == "gpt-6-astra"


def test_caps_are_what_a_role_may_spend_plus_one_worst_case_call():
    settings = TrialSettings(spend_cap_usd=10, worker_spend_cap_usd=4, monitor_spend_cap_usd=1,
                             lost_workers=2)
    worker, monitor, goal = settings.budgets()
    call = max_call_cost_usd(AZURE_PLANNER_MODEL, max_output_tokens=settings.worker_max_output_tokens,
                             cap_on="billed")
    assert worker == pytest.approx(4 + call)
    # A role is admitted another call only while its real spending is under its allowance.
    assert worker - call == pytest.approx(4) and monitor > 1
    planner_call = max_call_cost_usd(AZURE_PLANNER_MODEL, max_output_tokens=16384, cap_on="billed")
    assert goal == pytest.approx(10 + 3 * (worker + monitor) + planner_call)
    policy = policy_for(settings)
    assert (policy.worker_budget_usd, policy.monitor_budget_usd) == (worker, monitor)
    assert settings.goal("trial-1", "task").budget_usd == goal
    # The admission budget bounds the worst case. The cap is what the goal is
    # held to: it takes on no more work once its known spending reaches it.
    assert settings.goal("trial-1", "task").metadata[SPEND_CAP_KEY] == 10


def test_goal_reserves_its_closing_reply_and_names_the_generic_criterion():
    goal = TrialSettings(reply_reserve_seconds=120).goal("trial-1", "Developer: fix it.\n")
    assert goal.task == "Developer: fix it.\n" and goal.success_criteria == (CRITERION,)
    assert benchmark_reply.required(goal.metadata)
    assert benchmark_reply.closing_reserve(goal.metadata) == 120.0
    # No plan is started with less working time than it needs to be acted on.
    assert benchmark_reply.planning_minimum(goal.metadata) == 90.0


def test_fixed_agent_time_is_split_into_work_reply_and_handoff():
    settings = TrialSettings(reply_reserve_seconds=150, handoff_seconds=150)
    goal_deadline, container_deadline = settings.deadlines(1000.0, 1800.0)
    # 25 minutes of work, 2.5 for the closing reply, 2.5 to settle and hand over.
    assert goal_deadline == 1000.0 + 1650.0 and container_deadline > 1000.0 + 1800.0
    with pytest.raises(ValueError, match="too short"):
        settings.deadlines(1000.0, 300.0)


def test_agent_time_is_the_task_published_value_unless_overridden(tmp_path):
    task = tmp_path / "task.toml"
    task.write_text('schema_version = "1.4"\n[agent]\ntimeout_sec = 1800.0\n[verifier]\ntimeout_sec = 600.0\n')
    assert agent_timeout_seconds(task) == 1800.0
    assert agent_timeout_seconds(task, "900") == 900.0
    task.write_text('schema_version = "1.4"\n[agent]\nuser = "root"\n')
    with pytest.raises(ValueError, match="publishes no agent timeout"):
        agent_timeout_seconds(task)


@pytest.mark.parametrize("options,match", [
    ({"model": "gpt-6-luna"}, "model must be one of"),
    ({"model": "gpt-6.1-sol"}, "model must be one of"),
    ({"agent": "claude-code"}, "agent must be one of"),
    ({"agent": "mini-swe-agent", "services": "some"}, "services must be all or none"),
    ({"agent": "mini-swe-agent", "alone": "twice"}, "alone must be once or continue"),
    ({"agent": "mini-swe-agent", "alone": "continue"}, "needs services none"),
    ({"worker_model": "claude"}, "model must be one of"),
    ({"spend_cap_usd": "0"}, "must be positive"),
    ({"handoff_seconds": "nan"}, "must be positive"),
    ({"max_generations": "many"}, "wrong type"),
    ({"planner_effort": "extreme"}, "planner reasoning effort"),
    ({"request_seconds": "0"}, "must be positive"),
    ({"request_seconds": "inf"}, "must be positive"),
    ({"workers": 3}, "unknown trial setting"),
])
def test_settings_are_admitted_not_guessed(options, match):
    with pytest.raises(ValueError, match=match):
        TrialSettings.from_options(options)


def test_command_line_strings_are_typed():
    settings = TrialSettings.from_options({"model": "gpt-6-astra", "worker_max_calls": "40",
                                           "spend_cap_usd": "7.5", "worker_effort": "high"})
    assert (settings.worker_max_calls, settings.spend_cap_usd, settings.worker_effort) == (40, 7.5, "high")
    assert served_model("azure/gpt-6-astra") == ("gpt-6-astra", AZURE_PLANNER_MODEL)


@pytest.mark.parametrize("session,project", [
    ("hutusi-amytis-15__AbCdEfG", "hutusi-amytis-15__abcdefg"),
    ("_starts.with punctuation", "0_starts-with-punctuation"),
])
def test_compose_project_matches_harbor_naming(session, project):
    assert compose_project(session) == project
