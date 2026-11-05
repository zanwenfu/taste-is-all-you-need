"""One coding agent's long run: compaction on, a message planted mid-run, and Taste's memory in the memory arms.

The agent is mini-swe-agent 2.4.6's DefaultAgent with three hooks in ``query``. These tests drive it with a
scripted model and a stand-in shell, so every hook is checked before any container or paid call.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("minisweagent")

from minisweagent.exceptions import Submitted

from studies.long_run.agent import SUMMARY_LEAD, LongRunAgent, MemoryEnvironment, ToolModel, render
from studies.memory_probe.stores import GitTurns
from taste.memstore.tools import FETCH, SEARCH

RULE = "Always run the test suite with -x before you report a fix."
LATE = "From now on, never edit files under vendor/."


@pytest.fixture(autouse=True)
def _pinned_code(monkeypatch) -> None:
    monkeypatch.setenv("TASTE_COMMIT", "abc123")


def _count(text: str) -> int:
    return len(text.split())


def _call(name: str, arguments: dict, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def _reply(calls: list, content: str = "") -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls or None),
                                                    finish_reason="tool_calls")],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))


class _Shell:
    """Stands in for the task container: echoes commands, and submits like mini-swe-agent's environments."""

    def __init__(self) -> None:
        self.ran: list[str] = []

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        command = action["command"]
        self.ran.append(command)
        if command.startswith("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"):
            raise Submitted({"role": "exit", "content": "", "extra": {"exit_status": "Submitted", "submission": ""}})
        return {"output": f"ran: {command}", "returncode": 0, "exception_info": ""}

    def get_template_vars(self, **kwargs) -> dict:
        return {}

    def serialize(self) -> dict:
        return {}


class _Script:
    def __init__(self, replies: list) -> None:
        self.replies, self.sent = list(replies), []

    def __call__(self, messages: list[dict], tools: list | None):
        self.sent.append({"messages": json.loads(json.dumps(messages)), "tools": tools})
        return self.replies.pop(0)


def _agent(script, *, store=None, summarize=None, **config) -> tuple[LongRunAgent, _Shell]:
    shell = _Shell()
    model = ToolModel(script, memory_tools=store is not None)
    env = MemoryEnvironment(shell, store.answer if store else None)
    agent = LongRunAgent(model, env, count=_count, summarize=summarize, record=store.add if store else None,
                         system_template="You are a coding agent.", instance_template="{{task}}",
                         step_limit=40, cost_limit=0, **config)
    return agent, shell


def _bash(command: str, call_id: str) -> SimpleNamespace:
    return _reply([_call("bash", {"command": command}, call_id)])


DONE = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


# ------------------------------------------------------------------ tools


def test_bash_runs_in_the_shell_and_the_memory_tools_are_answered_on_the_host(tmp_path: Path) -> None:
    store = GitTurns(tmp_path)
    script = _Script([_bash("ls", "c1"), _reply([_call(SEARCH, {"description": "test suite before report"}, "c2")]),
                      _bash(DONE, "c3")])
    agent, shell = _agent(script, store=store)
    try:
        agent.run(task=f"Fix the parser. {RULE}")
    finally:
        store.close()
    assert shell.ran == ["ls", DONE]  # the search never reached the container
    assert [t["function"]["name"] for t in script.sent[0]["tools"]] == ["bash", SEARCH, FETCH]
    search_result = next(m for m in agent.messages if m.get("tool_call_id") == "c2")
    assert search_result["role"] == "tool" and RULE.split()[0] in search_result["content"]
    assert json.loads(search_result["content"])["results"]


def test_without_memory_only_bash_is_offered_and_a_memory_call_is_a_format_error() -> None:
    script = _Script([_reply([_call(SEARCH, {"description": "x"}, "c1")]), _bash(DONE, "c2")])
    agent, _ = _agent(script)
    agent.run(task="Fix it.")
    assert [t["function"]["name"] for t in script.sent[0]["tools"]] == ["bash"]
    assert any("Unknown tool" in str(m.get("content")) for m in agent.messages if m["role"] == "user")


def test_the_model_is_sent_plain_chat_messages_with_tool_calls_and_results() -> None:
    script = _Script([_bash("ls", "c1"), _bash(DONE, "c2")])
    agent, _ = _agent(script)
    agent.run(task="Fix it.")
    second = script.sent[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    assert second[2]["tool_calls"][0]["id"] == "c1" and second[3]["tool_call_id"] == "c1"
    assert all("extra" not in m for m in second)


# ------------------------------------------------------------------ the planted message


def test_a_planted_message_arrives_once_just_before_its_call() -> None:
    script = _Script([_bash("ls", "c1"), _bash("pwd", "c2"), _bash("ls", "c3"), _bash(DONE, "c4")])
    agent, _ = _agent(script, planted={3: LATE})
    agent.run(task="Fix it.")
    assert [m["content"] for m in script.sent[2]["messages"] if m["role"] == "user"][-1] == LATE
    assert sum(LATE in json.dumps(sent["messages"]) for sent in script.sent[:2]) == 0
    assert sum(m.get("content") == LATE for m in agent.messages) == 1


# ------------------------------------------------------------------ compaction


def _summarize(log: list):
    def summarize(older: list[dict], previous: str | None) -> str:
        log.append({"older": older, "previous": previous})
        return f"summary {len(log)}"
    return summarize


def test_compaction_replaces_the_older_part_with_a_summary_and_keeps_the_recent_turns() -> None:
    log: list = []
    script = _Script([_bash(f"step {n} " + "word " * 30, f"c{n}") for n in range(6)] + [_bash(DONE, "c9")])
    agent, _ = _agent(script, summarize=_summarize(log), compact_at_tokens=150, keep_recent_tokens=80)
    agent.run(task=f"Fix it. {RULE}")

    assert agent.events, "no compaction happened"
    assert all(event["kept"] >= 2 for event in agent.events)  # the newest call and its result fit 80 tokens
    event = agent.events[0]
    assert event["tokens_before"] > 150 >= event["tokens_after"] or event["tokens_after"] < event["tokens_before"]
    first_older = log[0]["older"]
    assert first_older[0]["content"] == f"Fix it. {RULE}"  # the task with its rule went into the summary
    after = script.sent[event["call"] - 1]["messages"]
    assert after[0]["role"] == "system" and after[1] == {"role": "user", "content": SUMMARY_LEAD + "summary 1"}
    assert RULE not in json.dumps(after)
    assert after[2]["role"] != "tool"  # a kept turn never starts with a result whose call was summarized
    if len(agent.events) > 1:
        assert log[1]["previous"] == "summary 1"
        assert all(SUMMARY_LEAD not in str(m.get("content")) for m in log[1]["older"])


def test_each_compaction_says_whether_the_watched_texts_survived() -> None:
    script = _Script([_bash(f"step {n} " + "word " * 30, f"c{n}") for n in range(6)] + [_bash(DONE, "c9")])
    agent, _ = _agent(script, summarize=lambda older, previous: "kept: Always run the test suite with -x",
                      compact_at_tokens=150, keep_recent_tokens=60, watch={"rule": RULE, "kept_part": "test suite"})
    agent.run(task=f"Fix it. {RULE}")
    event = agent.events[0]
    assert event["watched_in_summary"] == {"rule": False, "kept_part": True}
    assert event["watched_in_kept_turns"] == {"rule": False, "kept_part": False}


def test_a_call_and_its_result_that_do_not_fit_together_are_both_summarized() -> None:
    script = _Script([_bash(f"step {n} " + "word " * 30, f"c{n}") for n in range(6)] + [_bash(DONE, "c9")])
    agent, _ = _agent(script, summarize=lambda older, previous: "s", compact_at_tokens=150, keep_recent_tokens=60)
    agent.run(task="Fix it.")
    assert agent.events and agent.events[0]["kept"] == 0  # a result alone would have no call to answer


def test_no_compaction_below_the_trigger() -> None:
    script = _Script([_bash("ls", "c1"), _bash(DONE, "c2")])
    agent, _ = _agent(script, summarize=_summarize([]), compact_at_tokens=10_000, keep_recent_tokens=100)
    agent.run(task="Fix it.")
    assert agent.events == []


# ------------------------------------------------------------------ the memory keeps what compaction drops


def test_every_message_is_kept_in_the_memory_and_compaction_does_not_touch_it(tmp_path: Path) -> None:
    store = GitTurns(tmp_path)
    steps = [_bash(f"step {n} " + "word " * 30, f"c{n}") for n in range(5)]
    script = _Script([*steps, _reply([_call(SEARCH, {"description": "test suite -x report fix"}, "s1")]),
                      _bash(DONE, "c9")])
    agent, _ = _agent(script, store=store, summarize=_summarize([]), compact_at_tokens=150, keep_recent_tokens=60)
    try:
        agent.run(task=f"Fix it. {RULE}")
        assert agent.events
        result = next(m for m in agent.messages if m.get("tool_call_id") == "s1")
        assert RULE in json.dumps(json.loads(result["content"]))  # found again after it was compacted away
        assert store.commits == agent.recorded and agent.recorded >= 2 + 2 * 5
    finally:
        store.close()


def test_render_shows_tool_calls_as_text() -> None:
    message = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": "ls -la"})}}]}
    assert "bash" in render(message) and "ls -la" in render(message)
