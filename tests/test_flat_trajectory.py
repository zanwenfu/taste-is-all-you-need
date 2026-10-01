"""The record a benchmark reads is in the order things happened, with one answer."""

from __future__ import annotations

import json

import pytest

from taste.benchmarks.flat_trajectory import (
    WORKER_AGENT,
    encode,
    flat_trajectory,
    ledger_trajectory,
)
from taste.brains.terminal_broker import TerminalResult

TASK = "Developer: the importer drops rows.\n(Respond to the developer's most recent message above.)\n"
REPLY = "I fixed the importer and ran its tests; the migration was not checked."


def agent_step(step_id, at, *calls, text="", request="r"):
    step = {"step_id": step_id, "timestamp": at, "source": "agent", "message": text,
            "model_name": "gpt-6-astra-2026-09-03", "llm_call_count": 1,
            "metrics": {"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 0, "cost_usd": 0.01},
            "extra": {"is_sidechain": True, "request_id": request, "tool_execution": {}}}
    for name, arguments, result_at, content in calls:
        identifier = f"call_{step_id}_{len(step.get('tool_calls', []))}"
        step.setdefault("tool_calls", []).append(
            {"tool_call_id": identifier, "function_name": name, "arguments": arguments})
        if content is not None:
            step.setdefault("observation", {"results": []})["results"].append(
                {"source_call_id": identifier, "content": content,
                 "extra": {"is_error": content.startswith("exit 1"), "timestamp": result_at}})
    return step


def worker(run_id, *steps):
    return {"schema_version": "ATIF-v1.7", "session_id": run_id, "trajectory_id": run_id,
            "agent": {"name": WORKER_AGENT, "version": "1"},
            "steps": [{"step_id": 1, "source": "system", "message": "worker system prompt"},
                      # A worker is shown the task again; that is not the task's own conversation.
                      {"step_id": 2, "source": "user", "message": "For reference...\n\n" + TASK},
                      *steps],
            "extra": {"incomplete_worker_trace": False}}


def nested(*, reply=True):
    first = worker(
        "worker-run.a",
        agent_step(3, "2026-10-01T02:00:01.000000+00:00",
                   ("terminal_exec", {"command": "pytest -q"}, "2026-10-01T02:00:05.000000+00:00", "exit 1\n1 failed\n"),
                   ("terminal_exec", {"command": "sed -i s/a/b/ importer.py"},
                    "2026-10-01T02:00:20.000000+00:00", "exit 0\n"), text="Reproducing, then fixing."),
        agent_step(4, "2026-10-01T02:00:40.000000+00:00", text='{"status":"completed"}', request="r2"))
    second = worker(
        "worker-run.b",
        # Ran between the first worker's two commands, in the shared container.
        agent_step(3, "2026-10-01T02:00:02.000000+00:00",
                   ("terminal_exec", {"command": "git log -1"}, "2026-10-01T02:00:10.000000+00:00", "exit 0\nabc\n")),
        agent_step(4, "2026-10-01T02:00:30.000000+00:00",
                   ("write_artifact", {"artifact": "notes.md", "body": "x", "executable": False},
                    "2026-10-01T02:00:31.000000+00:00", "Artifact saved."),
                   ("terminal_exec", {"command": "sleep 999"}, None, None)))
    planner = {"schema_version": "ATIF-v1.7", "session_id": "g.planner", "trajectory_id": "g.planner",
               "agent": {"name": "taste-coordinator", "version": "1"},
               "steps": [{"step_id": 1, "source": "system", "message": "planner"},
                         {"step_id": 2, "source": "user", "message": "prompt holding " + TASK},
                         {"step_id": 3, "source": "agent", "message": "{}", "llm_call_count": 1,
                          "model_name": "gpt-6-astra-2026-09-03", "extra": {"is_sidechain": True}}]}
    steps = [{"step_id": 1, "source": "user", "message": TASK}]
    if reply:
        steps.append({"step_id": 2, "source": "agent", "message": REPLY, "llm_call_count": 0,
                      "timestamp": "2026-10-01T02:01:00.000000+00:00",
                      "extra": {"accepted_planner_attempt_id": "sha256:1", "is_sidechain": False}})
    return {"schema_version": "ATIF-v1.7", "session_id": "g", "trajectory_id": "g",
            "agent": {"name": "taste", "version": "1"}, "steps": steps,
            "extra": {"evidence_complete": True, "gaps": [], "final_reply_present": reply},
            "subagent_trajectories": [planner, first, second],
            "final_metrics": {"total_prompt_tokens": 30, "extra": {}}}


def answer(record):
    """A reader's rule: what the main agent said to end its work, never a subagent's words."""
    for step in reversed(record["steps"][1:]):
        if step["source"] != "agent" or step["extra"].get("is_sidechain"):
            continue
        return "" if step.get("tool_calls") else step["message"]
    return ""


def test_worker_calls_are_one_list_in_the_order_the_container_saw_them():
    record = flat_trajectory(nested(), audit_flags=["accounting_unsettled"])
    steps = record["steps"]
    assert [step["step_id"] for step in steps] == list(range(1, len(steps) + 1))
    assert steps[0] == {"step_id": 1, "source": "user", "message": TASK}
    commands = [(step["tool_calls"][0]["arguments"].get("command"), step["extra"]["worker_run"])
                for step in steps if step.get("tool_calls")]
    assert commands == [
        ("pytest -q", "worker-run.a"), ("git log -1", "worker-run.b"),
        ("sed -i s/a/b/ importer.py", "worker-run.a"),
        # A call whose result was never recorded still follows the call before it.
        (None, "worker-run.b"), ("sleep 999", "worker-run.b"),
    ]
    assert all(len(step["tool_calls"]) == 1 for step in steps if step.get("tool_calls"))
    # Each observation answers the call in its own step, as the format requires.
    for step in steps:
        for result in (step.get("observation") or {}).get("results", []):
            assert result["source_call_id"] == step["tool_calls"][0]["tool_call_id"]
    unanswered = next(step for step in steps if step.get("tool_calls")
                      and step["tool_calls"][0]["arguments"].get("command") == "sleep 999")
    assert "observation" not in unanswered and unanswered["extra"]["executed"] is False
    assert steps[1]["message"] == "Reproducing, then fixing." and steps[1]["llm_call_count"] == 1
    split = next(step for step in steps if step.get("tool_calls")
                 and step["tool_calls"][0]["arguments"].get("command", "").startswith("sed"))
    assert split["message"] == "" and split["llm_call_count"] == 0 and "metrics" not in split
    assert record["extra"]["audit_flags"] == ["accounting_unsettled"]
    assert record["extra"]["record"] == "chronological"
    assert record["extra"]["worker_runs"] == ["worker-run.a", "worker-run.b"]
    assert record["final_metrics"] == {"total_prompt_tokens": 30, "extra": {}}
    assert record["agent"]["model_name"] == "gpt-6-astra-2026-09-03"
    assert json.loads(encode(record)) == record


def test_only_the_coordinator_reply_can_be_read_as_the_answer():
    record = flat_trajectory(nested())
    assert answer(record) == REPLY and record["steps"][-1]["extra"]["is_sidechain"] is False
    # A worker's own closing words are in the record, and are not the answer.
    assert any(step["message"] == '{"status":"completed"}' for step in record["steps"])
    stopped = flat_trajectory(nested(reply=False))
    assert answer(stopped) == "" and all(
        step["extra"]["is_sidechain"] for step in stopped["steps"][1:])


def test_the_task_is_carried_once_so_the_attempt_starts_after_it():
    record = flat_trajectory(nested())
    carrying = [step for step in record["steps"] if step["source"] == "user"]
    assert carrying == [record["steps"][0]], "a worker's copy of the task must not restart the attempt"
    assert "subagent_trajectories" not in record
    encoded = encode(record).decode()
    assert "worker system prompt" not in encoded and "prompt holding" not in encoded
    kept = flat_trajectory(nested(), include_internal=True)
    assert [item["agent"]["name"] for item in kept["subagent_trajectories"]] == ["taste-coordinator"]
    assert not any(step.get("tool_calls") for item in kept["subagent_trajectories"]
                   for step in item["steps"])


@pytest.mark.parametrize("damage", [
    lambda value: value.update(schema_version="ATIF-v1.5"),
    lambda value: value["steps"].pop(0),
    lambda value: value["steps"].append({"step_id": 3, "source": "agent", "message": "second answer"}),
    lambda value: value["steps"].insert(1, {"step_id": 2, "source": "system", "message": "x"}),
])
def test_an_unknown_shape_is_refused_rather_than_guessed(damage):
    value = nested()
    damage(value)
    with pytest.raises(ValueError):
        flat_trajectory(value)


def test_the_terminal_ledger_alone_still_tells_what_the_container_ran():
    rows = [
        {"request_id": "effect_1", "command": "make test", "cwd": "/src", "timeout_seconds": 60.0,
         "status": "completed", "result": TerminalResult(2, b"1 failed\n", b"warn\n")},
        {"request_id": "effect_2", "command": "make slow", "cwd": "/src", "timeout_seconds": 5.0,
         "status": "completed", "result": TerminalResult(137, b"partial", b"", terminated="timeout")},
        {"request_id": "effect_3", "command": "make deploy", "cwd": "/src", "timeout_seconds": 5.0,
         "status": "uncertain", "result": None},
    ]
    record = ledger_trajectory("trial-1", TASK, rows, audit_flags=["settlement_failed:OSError"])
    steps = record["steps"]
    assert [step["step_id"] for step in steps] == [1, 2, 3, 4] and answer(record) == ""
    assert steps[1]["observation"]["results"][0]["content"] == "exit 2\n1 failed\n[stderr]\nwarn\n"
    assert steps[2]["observation"]["results"][0]["content"].startswith("timed out after 5s:")
    # No result is claimed for a command whose outcome was never confirmed.
    assert "observation" not in steps[3] and steps[3]["extra"]["ledger_status"] == "uncertain"
    assert steps[3]["tool_calls"][0]["arguments"] == {"command": "make deploy", "cwd": "/src",
                                                     "timeout_seconds": 5.0}
    assert record["extra"] == {"record": "terminal_ledger_only", "evidence_complete": False,
                               "final_reply_present": False,
                               "audit_flags": ["settlement_failed:OSError"]}
