"""One coding agent's long run: compaction on, a message planted mid-run, and Taste's memory in the memory arms.

The agent is mini-swe-agent 2.4.6's DefaultAgent with three hooks in ``query``. These tests drive it with a
scripted model and a stand-in shell, so every hook is checked before any container or paid call.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
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


# ------------------------------------------------------------------ several tasks in one conversation


def test_the_next_task_is_a_new_user_message_in_its_own_container_and_the_conversation_goes_on() -> None:
    script = _Script([_bash("first", "c1"), _bash(DONE, "c2"), _bash("second", "c3"), _bash(DONE, "c4")])
    agent, _ = _agent(script)
    shells = {"one": _Shell(), "two": _Shell()}

    outcomes = agent.run_tasks([{"name": "one", "instruction": "Do one."}, {"name": "two", "instruction": "Do two."}],
                               lambda task: shells[task["name"]])

    assert shells["one"].ran == ["first", DONE] and shells["two"].ran == ["second", DONE]
    assert [o["task"] for o in outcomes] == ["one", "two"] and all(o["exit_status"] == "Submitted" for o in outcomes)
    users = [m["content"] for m in script.sent[-1]["messages"] if m["role"] == "user"]
    assert users == ["Do one.", "Do two."], "the second task follows the first; no exit message is ever sent to the model"


def test_memory_tools_are_answered_in_every_task_and_a_task_change_is_kept_in_the_memory(tmp_path: Path) -> None:
    store = GitTurns(tmp_path)
    script = _Script([_bash(DONE, "c1"), _reply([_call(SEARCH, {"description": "parser one"}, "s1")]), _bash(DONE, "c2")])
    agent, _ = _agent(script, store=store)
    shells = {"one": _Shell(), "two": _Shell()}
    try:
        agent.run_tasks([{"name": "one", "instruction": "Fix parser one."}, {"name": "two", "instruction": "Do two."}],
                        lambda task: shells[task["name"]])
        result = next(m for m in agent.messages if m.get("tool_call_id") == "s1")
        assert "Fix parser one." in result["content"]
        assert store.commits == agent.recorded
    finally:
        store.close()


# ------------------------------------------------------------------ the record the scores are read from


def test_every_action_is_logged_with_its_call_its_task_and_how_many_compactions_came_before(tmp_path: Path) -> None:
    log: list = []
    store = GitTurns(tmp_path)
    script = _Script([*[_bash(f"step {n} " + "word " * 30, f"c{n}") for n in range(5)],
                      _reply([_call(SEARCH, {"description": "the rule"}, "s1")]), _bash(DONE, "c9")])
    agent, _ = _agent(script, store=store, summarize=_summarize([]), compact_at_tokens=150, keep_recent_tokens=60,
                      on_action=log.append)
    try:
        agent.run_tasks([{"name": "one", "instruction": f"Fix it. {RULE}"}], lambda task: _Shell())
    finally:
        store.close()
    commands = [row for row in log if row["kind"] == "bash"]
    assert commands[0]["command"].startswith("step 0") and commands[0]["call"] == 1 and commands[0]["compactions"] == 0
    assert commands[0]["task"] == "one"
    assert log[-2]["kind"] == SEARCH and RULE in log[-2]["output"], "a memory call is logged whole, with what it brought back"
    assert log[-1]["compactions"] == len(agent.events) > 0


# ------------------------------------------------------------------ the summary, from the two published prompts

from studies.long_run.session import load_session, make_summarizer, run_session  # noqa: E402


def _prompt(option: str, messages: list[dict]) -> list[dict]:
    return [{"role": "system", "content": f"summarize with {option}"}, *messages, {"role": "user", "content": "now summarize"}]


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None),
                                                    finish_reason="stop")],
                           usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))


def test_the_summary_is_the_published_prompt_asked_of_the_served_model_with_no_tools() -> None:
    sent = []

    def chat(messages, tools):
        sent.append((messages, tools))
        return _text("the summary")

    summarize = make_summarizer(chat, "anthropic", _prompt)
    older = [{"role": "user", "content": "the task"}]
    assert summarize(older, None) == "the summary" and sent[0][1] is None
    assert sent[0][0][0]["content"] == "summarize with anthropic"
    summarize(older, "the summary")
    assert {"role": "assistant", "content": "the summary"} in sent[1][0], "the previous summary goes in with the new part"


def test_an_empty_summary_is_asked_again_and_stops_the_run_if_it_stays_empty() -> None:
    replies = iter(["", "fine"])
    assert make_summarizer(lambda m, t: _text(next(replies)), "anthropic", _prompt)([], None) == "fine"
    with pytest.raises(RuntimeError, match="summary"):
        make_summarizer(lambda m, t: _text(""), "anthropic", _prompt)([], None)


def test_a_summary_cut_off_once_is_asked_again() -> None:
    cut = _text("half a summ")
    cut.choices[0].finish_reason = "length"
    replies = iter([cut, _text("a whole summary")])
    assert make_summarizer(lambda m, t: next(replies), "anthropic", _prompt)([], None) == "a whole summary"


def test_a_summary_cut_off_every_time_stops_the_run() -> None:
    reply = _text("half a summ")
    reply.choices[0].finish_reason = "length"
    with pytest.raises(RuntimeError, match="cut off"):
        make_summarizer(lambda m, t: reply, "anthropic", _prompt)([], None)


# ------------------------------------------------------------------ one whole session, on a stand-in


SESSION = {
    "tasks": [{"name": "one", "instruction": "Do one."}, {"name": "two", "instruction": "Do two."}],
    "rules": [{"name": "bytes", "say": "first", "text": "Whenever you run a Python script, pass -B to Python.",
               "trigger": r"\bpython3?\s", "obeys": r"\bpython3?\s+-B\b"},
              {"name": "ls", "say": 3, "text": "From now on, list a folder with ls -1.",
               "trigger": r"(^|[;&|]\s*)ls\b", "obeys": r"\bls\s+-1\b"}],
    "compact_at_tokens": 150, "keep_recent_tokens": 60, "summary_prompt": "anthropic", "step_limit": 40,
}


def _session_chat() -> _Script:
    return _Script([_bash("python3 a.py " + "word " * 30, "c1"), _bash("ls " + "word " * 30, "c2"),
                    _bash("python3 -B a.py " + "word " * 30, "c3"), _bash(DONE, "c4"),
                    _reply([_call(SEARCH, {"description": "python script"}, "s1")]) , _bash("ls -1", "c5"), _bash(DONE, "c6")])


def _summaries(messages, tools):
    return _text("a summary")


def _routed(script):
    def chat(messages, tools):
        return script(messages, tools) if tools else _summaries(messages, tools)
    return chat


def test_a_memory_session_writes_its_record_and_its_ledger_counts_the_real_store(tmp_path: Path) -> None:
    session = load_session(SESSION)
    script = _session_chat()
    summary = run_session(session, arm="taste", chat=_routed(script), make_env=lambda task: _Shell(), count=_count,
                          out=tmp_path / "run", workdir=tmp_path / "w", package_prompt=_prompt)
    rows = [json.loads(line) for line in (tmp_path / "run" / "actions.jsonl").read_text().splitlines()]
    assert any(row["kind"] == SEARCH for row in rows)
    assert summary["arm"] == "taste" and summary["ledger"]["name"] == "Taste memory, study-driven"
    assert summary["ledger"]["search_calls"] == 1
    assert summary["memory_turns"] > 0 and summary["compactions"] == len(summary["events"]) > 0
    assert [o["exit_status"] for o in summary["outcomes"]] == ["Submitted", "Submitted"]
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["arm"] == "taste"


def test_the_first_rule_is_told_with_the_first_task_and_the_later_one_at_its_call(tmp_path: Path) -> None:
    session = load_session(SESSION)
    script = _session_chat()
    run_session(session, arm="taste", chat=_routed(script), make_env=lambda task: _Shell(), count=_count,
                out=tmp_path / "run", workdir=tmp_path / "w", package_prompt=_prompt)
    first = script.sent[0]["messages"]
    assert [m["content"] for m in first if m["role"] == "user"] == [f"{session['rules'][0]['text']}\n\nDo one."]
    assert session["rules"][1]["text"] in json.dumps(script.sent[2]["messages"])
    assert session["rules"][1]["text"] not in json.dumps(script.sent[1]["messages"])


def test_with_no_memory_only_bash_is_offered_and_the_ledger_says_baseline(tmp_path: Path) -> None:
    script = _Script([_bash("python3 a.py", "c1"), _bash(DONE, "c2"), _bash(DONE, "c3")])
    summary = run_session(load_session(SESSION), arm="compaction", chat=_routed(script), make_env=lambda task: _Shell(),
                          count=_count, out=tmp_path / "run", workdir=tmp_path / "w", package_prompt=_prompt)
    assert [t["function"]["name"] for t in script.sent[0]["tools"]] == ["bash"]
    assert summary["ledger"]["name"] == "baseline" and summary["memory_turns"] == 0


def test_a_ceiling_with_no_compaction_never_asks_for_a_summary(tmp_path: Path) -> None:
    script = _Script([_bash("word " * 200, "c1"), _bash(DONE, "c2"), _bash(DONE, "c3")])
    summary = run_session(load_session(SESSION), arm="no_compaction", chat=_routed(script), make_env=lambda task: _Shell(),
                          count=_count, out=tmp_path / "run", workdir=tmp_path / "w", package_prompt=_prompt)
    assert summary["compactions"] == 0


def test_the_restated_ceiling_has_every_rule_told_so_far_in_front_of_the_model_at_every_call(tmp_path: Path) -> None:
    script = _Script([_bash("echo a", "c1"), _bash("echo b", "c2"), _bash(DONE, "c3"), _bash(DONE, "c4")])
    summary = run_session(load_session(SESSION), arm="restated", chat=_routed(script), make_env=lambda task: _Shell(), count=_count,
                          out=tmp_path / "run", workdir=tmp_path / "w", package_prompt=_prompt)
    first, later = SESSION["rules"][0]["text"], SESSION["rules"][1]["text"]
    systems = [call["messages"][0]["content"] for call in script.sent]
    assert all(first in system for system in systems)
    assert later not in systems[1] and later in systems[2], "the later rule from the call it is told at"
    assert [m["content"] for m in script.sent[-1]["messages"] if m["role"] == "user"][-1] == "Do two."
    assert summary["valid"] is True, "this arm delivers its rules on purpose, in the system prompt"


def test_a_session_with_a_rule_it_cannot_score_is_refused() -> None:
    with pytest.raises(ValueError, match="trigger"):
        load_session({**SESSION, "rules": [{"name": "x", "say": "first", "text": "t", "obeys": "y"}]})
    with pytest.raises(ValueError, match="say"):
        load_session({**SESSION, "rules": [{"name": "x", "say": "soon", "text": "t", "trigger": "a", "obeys": "y"}]})
    with pytest.raises(ValueError, match="tasks"):
        load_session({**SESSION, "tasks": []})


# ------------------------------------------------------------------ the score, from the record alone

from studies.long_run.score import score_session  # noqa: E402


def test_a_rule_is_scored_only_after_it_was_told_and_only_on_commands_it_applies_to() -> None:
    rules = [{"name": "bytes", "say": "first", "trigger": r"\bpython3?\s", "obeys": r"\bpython3?\s+-B\b"},
             {"name": "ls", "say": 3, "trigger": r"(^|[;&|]\s*)ls\b", "obeys": r"\bls\s+-1\b"}]
    actions = [{"kind": "bash", "call": 1, "compactions": 0, "command": "python3 a.py"},
               {"kind": "bash", "call": 2, "compactions": 0, "command": "ls"},
               {"kind": "bash", "call": 3, "compactions": 1, "command": "cd x && python3 -B a.py"},
               {"kind": "bash", "call": 4, "compactions": 1, "command": "cat f; ls -1"},
               {"kind": "bash", "call": 5, "compactions": 2, "command": "ls -la"},
               {"kind": "memory_search", "call": 6, "compactions": 2, "command": "[memory_search] {}", "output": "x"}]
    scored = score_session(actions, rules)
    assert scored["bytes"]["applies"] == 2 and scored["bytes"]["obeyed"] == 1
    assert scored["ls"]["applies"] == 2 and scored["ls"]["obeyed"] == 1, "call 2 came before the rule was told"
    assert scored["ls"]["by_compactions"] == {"1": [1, 1], "2": [0, 1]}
    assert scored["bytes"]["by_compactions"] == {"0": [0, 1], "1": [1, 1]}


def test_a_memory_call_counts_as_a_look_and_a_find_only_when_the_rule_came_back() -> None:
    rules = [{"name": "ls", "say": "first", "text": "From now on, list a folder with ls -1.", "trigger": r"\bls\b", "obeys": r"\bls\s+-1\b"}]
    actions = [{"kind": "memory_search", "call": 1, "compactions": 0, "command": "q", "output": "nothing here"},
               {"kind": "memory_search", "call": 2, "compactions": 0, "command": "q", "output": "FROM NOW ON, list a folder with ls -1. more"},
               {"kind": "memory_fetch", "call": 3, "compactions": 0, "command": "q", "output": "other"}]
    scored = score_session(actions, rules)
    assert scored["ls"]["looks"] == 3 and scored["ls"]["finds"] == 1


# ------------------------------------------------------------------ the command line, on a stand-in server and a real shell

import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402

import studies.long_run.session as session_module  # noqa: E402


def _stand_in_server(script: list[dict]) -> HTTPServer:
    """Answers each tool-offering call with the next scripted call, and any call without tools with a summary."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body.get("tools"):
                call = script.pop(0)
                message = {"role": "assistant", "content": "working", "tool_calls": [
                    {"id": f"c{len(script)}", "type": "function",
                     "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}]}
            else:
                message = {"role": "assistant", "content": "a summary of the work so far"}
            payload = json.dumps({"choices": [{"message": message, "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_a_whole_session_from_the_command_line_with_a_real_shell(tmp_path: Path, monkeypatch) -> None:
    work = tmp_path / "project"
    work.mkdir()
    bash = lambda command: {"name": "bash", "arguments": {"command": command}}  # noqa: E731
    long = " ".join(["word"] * 60)
    server = _stand_in_server([bash(f"echo {long}"), bash(f"echo python3 -B a.py {long}"),
                               bash("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"),
                               {"name": SEARCH, "arguments": {"description": "Python script"}},
                               bash(f"echo {long}"), bash("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")])
    file = tmp_path / "session.json"
    file.write_text(json.dumps({**SESSION, "tasks": [{"name": "one", "instruction": "Do one.", "cwd": str(work)},
                                                     {"name": "two", "instruction": "Do two.", "cwd": str(work)}]}))
    monkeypatch.setattr(session_module, "package_prompt", lambda: _prompt)
    try:
        code = session_module.main(["--session", str(file), "--arm", "taste", "--url", f"http://127.0.0.1:{server.server_port}/v1",
                                    "--model", "stand-in", "--tokenizer", "words", "--env", "local", "--out", str(tmp_path / "run"),
                                    "--workdir", str(tmp_path / "w")])
    finally:
        server.shutdown()
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert code == 0 and [o["exit_status"] for o in summary["outcomes"]] == ["Submitted", "Submitted"]
    assert summary["compactions"] > 0 and summary["ledger"]["search_calls"] == 1
    assert summary["scores"]["bytes"]["looks"] == 1 and summary["scores"]["bytes"]["finds"] == 1
    assert summary["versions"]["mini_swe_agent"] == "2.4.6"


# ------------------------------------------------------------------ the real sessions, on Duke


def test_a_task_past_its_call_limit_is_left_and_the_next_task_follows(tmp_path: Path) -> None:
    script = _Script([_bash("echo a", "c1"), _bash("echo b", "c2"), _bash(DONE, "c3")])
    session = load_session({**SESSION, "rules": [], "task_calls": 2, "compact_at_tokens": 0})
    summary = run_session(session, arm="compaction", chat=_routed(script), make_env=lambda task: _Shell(), count=_count,
                          out=tmp_path / "run", workdir=tmp_path / "w", package_prompt=_prompt)
    assert [o["exit_status"] for o in summary["outcomes"]] == ["TaskCallLimit", "Submitted"]
    last = script.sent[-1]["messages"]
    assert last[-1]["role"] == "user" and "Do two." in last[-1]["content"]
    assert all(m["role"] != "exit" for m in last)


def test_the_ceiling_with_no_compaction_ends_when_the_window_is_full_and_keeps_its_record(tmp_path: Path) -> None:
    script = _Script([_bash("word " * 200, "c1"), _bash(DONE, "c2")])
    session = load_session({**SESSION, "rules": [], "stop_at_tokens": 100})
    summary = run_session(session, arm="no_compaction", chat=_routed(script), make_env=lambda task: _Shell(), count=_count,
                          out=tmp_path / "run", workdir=tmp_path / "w", package_prompt=_prompt)
    assert [o["exit_status"] for o in summary["outcomes"]] == ["WindowFull"]
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["calls"] == 1


def test_a_model_failure_ends_the_session_but_not_its_record(tmp_path: Path) -> None:
    calls = [0]

    def chat(messages, tools):
        calls[0] += 1
        if calls[0] == 2:
            raise TimeoutError("the server went away")
        return _bash("python3 a.py", "c1")

    summary = run_session(load_session({**SESSION, "compact_at_tokens": 0}), arm="compaction", chat=chat,
                          make_env=lambda task: _Shell(), count=_count, out=tmp_path / "run", workdir=tmp_path / "w",
                          package_prompt=_prompt)
    assert "the server went away" in summary["ended_by"]
    assert summary["scores"]["bytes"]["applies"] == 1
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["ended_by"] == summary["ended_by"]


def test_a_fact_in_the_first_machine_is_found_by_the_agent_and_never_sent_by_the_harness(tmp_path: Path) -> None:
    shells: list[_Shell] = []

    def make_env(task):
        shells.append(_Shell())
        return shells[-1]

    tag = {"name": "tag", "say": "first", "text": "This machine has a file /opt/tag. Read it. Run echo TAG in every task.",
           "per_task": True, "trigger": "", "obeys": "tag-7f3a", "watch": "tag-7f3a",
           "restate": "The tag is tag-7f3a. Run echo tag-7f3a in every task."}
    session = load_session({**SESSION, "rules": [tag], "compact_at_tokens": 0,
                            "tasks": [{"name": "one", "instruction": "Do one.", "prepare": ["echo tag-7f3a > /opt/tag"]},
                                      {"name": "two", "instruction": "Do two."}]})
    script = _Script([_bash("cat /opt/tag", "c1"), _bash("echo tag-7f3a", "c2"), _bash(DONE, "c3"), _bash(DONE, "c4")])
    summary = run_session(session, arm="compaction", chat=_routed(script), make_env=make_env, count=_count, out=tmp_path / "run",
                          workdir=tmp_path / "w", package_prompt=_prompt)
    assert shells[0].ran[0] == "echo tag-7f3a > /opt/tag" and shells[1].ran == [DONE]
    assert "tag-7f3a" not in json.dumps(script.sent[0]["messages"])
    assert summary["scores"]["tag"]["applies"] == 2 and summary["scores"]["tag"]["obeyed"] == 1
    rows = [json.loads(line) for line in (tmp_path / "run" / "actions.jsonl").read_text().splitlines()]
    assert "echo tag-7f3a > /opt/tag" not in [row["command"] for row in rows], "the harness's own step is not the agent's"
    restated = _Script([_bash(DONE, "c1"), _bash(DONE, "c2")])
    run_session(session, arm="restated", chat=_routed(restated), make_env=make_env, count=_count, out=tmp_path / "r2",
                workdir=tmp_path / "w", package_prompt=_prompt)
    assert "The tag is tag-7f3a" in json.dumps(restated.sent[-1]["messages"])


def test_the_memory_arms_differ_only_by_what_the_system_prompt_says_about_the_record(tmp_path: Path) -> None:
    from studies.long_run.session import ARMS
    from studies.memory_probe.cases import LOOK, SENTENCE

    seen = {}
    for arm in ("taste", "taste_exists", "taste_check"):
        assert arm in ARMS
        script = _Script([_bash(DONE, "c1"), _bash(DONE, "c2")])
        run_session(load_session({**SESSION, "compact_at_tokens": 0}), arm=arm, chat=_routed(script), make_env=lambda task: _Shell(),
                    count=_count, out=tmp_path / arm, workdir=tmp_path / "w", package_prompt=_prompt)
        seen[arm] = script.sent[0]["messages"][0]["content"]
        assert [t["function"]["name"] for t in script.sent[0]["tools"]] == ["bash", SEARCH, FETCH]
    assert SENTENCE not in seen["taste"] and LOOK not in seen["taste"]
    assert seen["taste_exists"] == f"{seen['taste']} {SENTENCE}"
    assert seen["taste_check"] == f"{seen['taste']} {SENTENCE} {LOOK}"


def _stand_in_apptainer(tmp_path: Path, monkeypatch) -> Path:
    """Stands in for Apptainer: it builds a folder, keeps a log, and runs a command given to an instance on this machine's shell."""
    log = tmp_path / "apptainer.log"
    folder = tmp_path / "bin"
    folder.mkdir()
    (folder / "apptainer").write_text(f"""#!/bin/bash
echo "$*" >> {log}
[ "$1" = --silent ] && shift
case "$1 $2" in
  "build --sandbox") mkdir -p "$3"; exit 0;;
  "instance start") [ -n "$STAND_IN_WILL_NOT_START" ] && {{ echo "no user namespace"; exit 1; }}; exit 0;;
  "instance stop") [ -n "$STAND_IN_WILL_NOT_STOP" ] && {{ echo "not stopped"; exit 1; }}; exit 0;;
esac
case "$*" in *"/proc/[0-9]*"*) printf '%s\\n' "${{STAND_IN_PROCESSES:-4}}" "${{STAND_IN_FILES:-0}}" "${{STAND_IN_CODE_FILES:-0}}"; exit 0;; esac
while [ "$1" != bash ]; do shift; done
exec "$@"
""")
    (folder / "apptainer").chmod(0o755)
    monkeypatch.setenv("PATH", f"{folder}:{os.environ['PATH']}")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    tempfile.tempdir = None
    return log


def test_a_machine_has_its_own_process_list_and_says_so_in_how_it_starts(tmp_path: Path, monkeypatch) -> None:
    from studies.long_run.session import _environment

    log = _stand_in_apptainer(tmp_path, monkeypatch)
    machine = _environment("machine", 30)({"image": "x.sif", "cwd": "/"})
    machine.execute({"command": "echo hello"})
    name, lines = machine.name, log.read_text().splitlines()
    start = next(line for line in lines if "instance start" in line)
    assert "--contain --cleanenv --fakeroot --ignore-fakeroot-command" in start and "--writable" in start and "--workdir" in start and start.endswith(name)
    command = next(line for line in lines if "echo hello" in line)
    assert command.startswith("--silent exec --cleanenv --ignore-fakeroot-command") and f"instance://{name}" in command
    sandbox, scratch = machine.sandbox_dir, machine.scratch
    machine.cleanup()
    assert f"instance stop {name}" in log.read_text() and not sandbox.exists() and not scratch.exists()


def test_a_machine_gives_a_commands_output_and_a_submission_as_the_agent_expects(tmp_path: Path, monkeypatch) -> None:
    from studies.long_run.machine import Machine

    _stand_in_apptainer(tmp_path, monkeypatch)
    machine = Machine(image="x.sif")
    done = machine.execute({"command": "echo out; echo err >&2; exit 3"})
    assert (done["returncode"], sorted(done["output"].split())) == (3, ["err", "out"]) and done["exception_info"] == ""
    with pytest.raises(Submitted):
        machine.execute({"command": DONE})
    machine.cleanup()


def test_work_left_in_the_background_does_not_hold_a_command_open_and_a_slow_one_times_out(tmp_path: Path, monkeypatch) -> None:
    from studies.long_run.machine import Machine

    _stand_in_apptainer(tmp_path, monkeypatch)
    machine = Machine(image="x.sif", timeout=2)
    started = time.monotonic()
    assert machine.execute({"command": "sleep 20 & echo started"})["output"].strip() == "started"
    assert time.monotonic() - started < 2
    slow = machine.execute({"command": "echo begun; sleep 20"})
    assert slow["returncode"] == -1 and "begun" in slow["output"] and "executing the command" in slow["exception_info"]
    machine.cleanup()


def test_a_machine_that_will_not_start_says_why_and_leaves_nothing_behind(tmp_path: Path, monkeypatch) -> None:
    from studies.long_run.machine import Machine

    _stand_in_apptainer(tmp_path, monkeypatch)
    monkeypatch.setenv("STAND_IN_WILL_NOT_START", "1")
    with pytest.raises(RuntimeError, match="no user namespace"):
        Machine(image="x.sif")
    assert not list(tmp_path.glob("minisweagent-*"))


def test_a_machine_reads_its_wall_before_anything_else_runs_on_it_and_keeps_what_it_saw(tmp_path: Path, monkeypatch) -> None:
    from studies.long_run.machine import Machine

    log = _stand_in_apptainer(tmp_path, monkeypatch)
    machine = Machine(image="x.sif")
    assert machine.wall["processes"] == 4 and set(machine.wall["files_outside"]) == {str(Path.home()), str(Path.cwd())}
    assert not any(machine.wall["files_outside"].values())
    run_on_it = [line for line in log.read_text().splitlines() if " exec " in f" {line}"]
    assert len(run_on_it) == 1 and "/proc/[0-9]*" in run_on_it[0], "the check is the first thing a machine runs"
    assert " kill" not in run_on_it[0], "the check sends no signal"
    machine.cleanup()


@pytest.mark.parametrize(("setting", "value", "says"), [("STAND_IN_PROCESSES", "500", "sees 500 processes"), ("STAND_IN_PROCESSES", "0", "sees 0 processes"),
                                                       ("STAND_IN_PROCESSES", "no /proc", "would not say"), ("STAND_IN_FILES", "12", "this account's files"),
                                                       ("STAND_IN_CODE_FILES", "7", "this account's files")])
def test_a_machine_whose_wall_does_not_hold_is_refused_and_leaves_nothing_behind(tmp_path: Path, monkeypatch, setting: str, value: str, says: str) -> None:
    from studies.long_run.machine import Machine

    log = _stand_in_apptainer(tmp_path, monkeypatch)
    monkeypatch.setenv(setting, value)
    with pytest.raises(RuntimeError, match=says):
        Machine(image="x.sif")
    assert "instance stop" in log.read_text() and not list(tmp_path.glob("minisweagent-*"))


def test_a_session_goes_on_past_a_machine_with_no_wall_and_keeps_what_each_machine_saw(tmp_path: Path, monkeypatch) -> None:
    from studies.long_run.session import _environment

    _stand_in_apptainer(tmp_path, monkeypatch)
    session = load_session({"tasks": [{"name": "one", "instruction": "A", "image": "x.sif"}, {"name": "two", "instruction": "B", "image": "x.sif"}],
                            "instance_template": "{{task}}"})
    broken = {"one"}

    def make(task: dict):
        monkeypatch.setenv("STAND_IN_PROCESSES", "500" if task["name"] in broken else "4")
        return _environment("machine", 30)(task)

    summary = run_session(session, arm="compaction", chat=_routed(_Script([_bash(DONE, "c1")])), make_env=make, count=_count,
                          out=tmp_path / "out", workdir=tmp_path / "work", package_prompt=lambda: _prompt)
    assert [o["exit_status"] for o in summary["outcomes"]] == ["MachineFailed", "Submitted"]
    assert "wall does not hold" in summary["outcomes"][0]["reason"] and "sees 500 processes" in summary["outcomes"][0]["reason"]
    assert [row["task"] for row in summary["walls"]] == ["two"] and summary["walls"][0]["wall"]["processes"] == 4
    assert not list(tmp_path.glob("minisweagent-*")), "every machine is closed by the session's end"


def test_each_machine_is_closed_before_the_next_opens_and_the_last_when_the_session_ends(tmp_path: Path) -> None:
    events: list[str] = []

    class Shell:
        def __init__(self, name: str) -> None:
            self.name = name
            events.append(f"open {name}")

        def execute(self, action: dict, cwd: str = "", *, timeout=None) -> dict:
            if DONE in action["command"]:
                raise Submitted({"role": "exit", "content": "", "extra": {"exit_status": "Submitted", "submission": ""}})
            return {"output": "", "returncode": 0, "exception_info": ""}

        def get_template_vars(self, **kwargs) -> dict:
            return {}

        def serialize(self) -> dict:
            return {}

        def cleanup(self) -> None:
            events.append(f"close {self.name}")

    session = load_session({"tasks": [{"name": "one", "instruction": "A"}, {"name": "two", "instruction": "B"}]})
    run_session(session, arm="compaction", chat=_routed(_Script([_bash(DONE, "c1"), _bash(DONE, "c2")])), make_env=lambda task: Shell(task["name"]), count=_count,
                out=tmp_path / "out", workdir=tmp_path / "work", package_prompt=lambda: _prompt)
    assert events == ["open one", "close one", "open two", "close two"]


def test_output_left_in_the_background_stays_out_of_the_next_command_and_holds_nothing_open(tmp_path: Path, monkeypatch) -> None:
    from studies.long_run.machine import SETTLE, Machine

    _stand_in_apptainer(tmp_path, monkeypatch)
    machine = Machine(image="x.sif")
    started = time.monotonic()
    first = machine.execute({"command": "(for i in $(seq 1 40); do echo noise; sleep 0.05; done) & echo started"})
    assert first["output"].startswith("started") and time.monotonic() - started < SETTLE + 1
    assert machine.execute({"command": "echo next"})["output"].strip() == "next", "what was left running writes to nothing anyone reads"
    machine.cleanup()


def test_only_the_end_of_a_huge_output_is_kept(tmp_path: Path, monkeypatch) -> None:
    from studies.long_run.machine import Machine

    _stand_in_apptainer(tmp_path, monkeypatch)
    monkeypatch.setattr("studies.long_run.machine.MOST_OUTPUT", 1000)
    machine = Machine(image="x.sif")
    done = machine.execute({"command": "for i in $(seq 1 3000); do echo line$i; done"})
    assert len(done["output"]) <= 1000 and done["output"].rstrip().endswith("line3000")
    machine.cleanup()


def test_a_command_that_waits_for_input_gets_none_rather_than_its_whole_time_limit(tmp_path: Path, monkeypatch) -> None:
    from studies.long_run.machine import Machine

    _stand_in_apptainer(tmp_path, monkeypatch)
    machine = Machine(image="x.sif", timeout=20)
    started = time.monotonic()
    assert machine.execute({"command": "cat; echo ended"})["output"].strip() == "ended" and time.monotonic() - started < 5
    machine.cleanup()


def test_a_machine_that_would_not_stop_is_on_the_record(tmp_path: Path, monkeypatch) -> None:
    from studies.long_run.session import _environment

    _stand_in_apptainer(tmp_path, monkeypatch)
    monkeypatch.setenv("STAND_IN_WILL_NOT_STOP", "1")
    session = load_session({"tasks": [{"name": "one", "instruction": "A", "image": "x.sif"}], "instance_template": "{{task}}"})
    summary = run_session(session, arm="compaction", chat=_routed(_Script([_bash(DONE, "c1")])), make_env=_environment("machine", 30), count=_count,
                          out=tmp_path / "out", workdir=tmp_path / "work", package_prompt=lambda: _prompt)
    assert len(summary["machines_not_stopped"]) == 1 and "not stopped" in summary["machines_not_stopped"][0]


@pytest.mark.parametrize("bad", [["bash", "-lc", "ls"], "echo a\0b", 7])
def test_a_command_that_is_not_one_string_is_told_so_and_the_session_goes_on(tmp_path: Path, bad) -> None:
    session = load_session({"tasks": [{"name": "one", "instruction": "A"}], "instance_template": "{{task}}"})
    script = _Script([_reply([_call("bash", {"command": bad}, "c0")]), _bash(DONE, "c1")])
    summary = run_session(session, arm="compaction", chat=_routed(script), make_env=lambda task: _Shell(), count=_count,
                          out=tmp_path / "out", workdir=tmp_path / "work", package_prompt=lambda: _prompt)
    assert summary["ended_by"] is None and [o["exit_status"] for o in summary["outcomes"]] == ["Submitted"]
    assert summary["no_command"] == 1 and (tmp_path / "out" / "summary.json").exists()


def test_a_tool_call_that_would_not_parse_is_kept_as_the_model_wrote_it(tmp_path: Path) -> None:
    session = load_session({"tasks": [{"name": "one", "instruction": "A"}], "instance_template": "{{task}}"})
    broken = SimpleNamespace(id="c0", type="function", function=SimpleNamespace(name="bash", arguments='{"command": "echo "hi""}'))
    summary = run_session(session, arm="compaction", chat=_routed(_Script([_reply([broken]), _bash(DONE, "c1")])), make_env=lambda task: _Shell(), count=_count,
                          out=tmp_path / "out", workdir=tmp_path / "work", package_prompt=lambda: _prompt)
    rows = [json.loads(line) for line in (tmp_path / "out" / "actions.jsonl").read_text().splitlines()]
    assert summary["no_command"] == 1 and rows[0]["kind"] == "no_command" and rows[0]["arguments"] == '{"command": "echo "hi""}'


def test_a_session_stopped_by_its_job_still_writes_its_record_and_closes_its_machine(tmp_path: Path) -> None:
    from studies.long_run.agent import Stopped

    closed: list[bool] = []

    class Closing(_Shell):
        def cleanup(self) -> None:
            closed.append(True)

    replies = iter([_bash("ls", "c1")])

    def chat(messages: list[dict], tools: list | None):
        try:
            return next(replies)
        except StopIteration:
            raise Stopped("signal 15") from None

    disarmed: list[bool] = []
    session = load_session({"tasks": [{"name": "one", "instruction": "A"}, {"name": "two", "instruction": "B"}], "instance_template": "{{task}}"})
    summary = run_session(session, arm="compaction", chat=chat, make_env=lambda task: Closing(), count=_count, out=tmp_path / "out", workdir=tmp_path / "work",
                          package_prompt=lambda: _prompt, disarm=lambda: disarmed.append(True))
    assert summary["ended_by"] == "Stopped: signal 15" and (tmp_path / "out" / "summary.json").exists()
    assert closed == [True] and disarmed == [True], "one machine had opened, and the second task never began"


def test_the_stop_signal_stops_a_session_once_and_after_that_changes_nothing() -> None:
    code = """
import signal
from studies.long_run.agent import Stopped
from studies.long_run.session import _stop_on_term
_stop_on_term()
try:
    signal.raise_signal(signal.SIGTERM)
except Stopped as error:
    print("stopped:", error)
signal.raise_signal(signal.SIGTERM)
print("went on")
"""
    done = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and "stopped: signal 15\nwent on" in done.stdout, done.stdout + done.stderr


def test_every_model_call_and_its_tokens_are_on_the_record(tmp_path: Path) -> None:
    session = load_session({**SESSION, "rules": [], "compact_at_tokens": 5, "keep_recent_tokens": 0})
    script = _Script([_bash("ls", "c1"), _bash(DONE, "c2"), _bash(DONE, "c3")])
    summary = run_session(session, arm="compaction", chat=_routed(script), make_env=lambda task: _Shell(), count=_count,
                          out=tmp_path / "run", workdir=tmp_path / "w", package_prompt=_prompt)
    rows = [json.loads(line) for line in (tmp_path / "run" / "actions.jsonl").read_text().splitlines()]
    assert summary["model_use"]["calls"] >= summary["calls"] and summary["model_use"]["prompt_tokens"] > 0
    assert next(row["usage"] for row in rows if "usage" in row) == [10, 2]


def test_tasks_that_open_a_port_are_left_out_since_sessions_share_a_network(tmp_path: Path) -> None:
    from studies.long_run.make_session import _eligible

    for name, words in (("quiet", "Sort the file."), ("web", "Serve the files on port 8080."), ("rpc", "Build a gRPC service."), ("nginx", "Set up a web server.")):
        folder = tmp_path / name
        (folder / "environment").mkdir(parents=True)
        (folder / "task.toml").write_text(f'[environment]\ndocker_image = "owner/{name}:1"\ncpus = 1\nmemory_mb = 2048\n')
        (folder / "instruction.md").write_text(words)
        (folder / "environment" / "Dockerfile").write_text("FROM python:3.13\nWORKDIR /app\n")
    assert [task["name"] for task in _eligible(tmp_path)] == ["quiet"]


def test_the_session_file_is_built_the_same_way_from_the_public_tasks(tmp_path: Path) -> None:
    from studies.long_run.make_session import build

    root = tmp_path / "tb"
    for name, cpus, memory in (("alpha", 1, 2048), ("beta", 1, 2048), ("gamma", 1, 2048), ("delta", 4, 8192), ("qemu-startup", 1, 2048)):
        folder = root / name
        folder.mkdir(parents=True)
        (folder / "task.toml").write_text(f'[environment]\ndocker_image = "owner/{name}:1"\ncpus = {cpus}\nmemory_mb = {memory}\n')
        (folder / "instruction.md").write_text(f"Do {name}.\n")
        (folder / "environment").mkdir()
        (folder / "environment" / "Dockerfile").write_text("FROM python:3.13\nWORKDIR /app\n")
    no_python = root / "epsilon"
    (no_python / "environment").mkdir(parents=True)
    (no_python / "task.toml").write_text('[environment]\ndocker_image = "owner/epsilon:1"\ncpus = 1\nmemory_mb = 2048\n')
    (no_python / "instruction.md").write_text("Do epsilon.\n")
    (no_python / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nWORKDIR /work\n")
    template = {"agent": {"system_template": "You help.", "instance_template": "Solve: {{task}}"},
                "model": {"observation_template": "{{output.output}}", "format_error_template": "{{error}}"}}
    one = build(root, template, seed=7, tasks=2, compact_at=16000, keep_recent=4000, prompt="anthropic", later_rule_at=20, task_calls=60)
    assert one == build(root, template, seed=7, tasks=2, compact_at=16000, keep_recent=4000, prompt="anthropic", later_rule_at=20, task_calls=60)
    names = [task["name"] for task in one["tasks"]]
    assert len(names) == 2 and not set(names) & {"delta", "qemu-startup"}, "too big or needs a machine inside the machine"
    assert "epsilon" not in names, "a machine with no Python never meets the rule about Python"
    assert one["tasks"][0]["image"].startswith("docker://owner/") and one["tasks"][0]["cwd"] == "/app"
    tag = next(rule for rule in one["rules"] if rule["name"] == "session_tag")
    assert tag["obeys"] in one["first_prepare"][0]
    assert tag["obeys"] not in json.dumps(one["tasks"]), "the tag is in no message, and on the first machine that opens only"
    assert all("APT::Sandbox::User" in task["prepare"][0] for task in one["tasks"]), "the package installer works on every machine"
    assert tag["obeys"] not in tag["text"] and tag["obeys"] in tag["restate"]
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in one["system_template"], "a summary may lose the first message"
    assert one["instance_template"] == "Solve: {{task}}" and one["tasks"][1]["instruction"].endswith("\n")
    assert load_session(one)["task_calls"] == 60
    assert build(root, template, seed=8, tasks=2, compact_at=16000, keep_recent=4000, prompt="anthropic", later_rule_at=20,
                 task_calls=60)["rules"] != one["rules"]


def test_a_rule_scored_per_task_counts_each_task_once_whether_or_not_it_was_obeyed() -> None:
    rules = [{"name": "tag", "say": "first", "per_task": True, "trigger": "", "obeys": "tag-7f3a"}]
    actions = [{"kind": "bash", "call": 1, "task": "one", "compactions": 0, "command": "cat /opt/tag"},
               {"kind": "bash", "call": 2, "task": "one", "compactions": 0, "command": "echo tag-7f3a"},
               {"kind": "bash", "call": 3, "task": "two", "compactions": 1, "command": "echo x"},
               {"kind": "bash", "call": 4, "task": "two", "compactions": 2, "command": "echo y"}]
    scored = score_session(actions, rules)["tag"]
    assert scored["applies"] == 2 and scored["obeyed"] == 1
    assert scored["by_compactions"] == {"0": [1, 1], "2": [0, 1]}, "counted under the compactions when the task ended"


def test_a_model_that_stops_issuing_commands_leaves_its_task_and_says_so_in_the_record(tmp_path: Path) -> None:
    silent = _reply([], content="I think this is done.")
    script = _Script([silent, silent, silent, _bash(DONE, "c1")])
    session = load_session({**SESSION, "rules": [], "compact_at_tokens": 0})
    summary = run_session(session, arm="compaction", chat=_routed(script), make_env=lambda task: _Shell(), count=_count,
                          out=tmp_path / "run", workdir=tmp_path / "w", package_prompt=_prompt)
    assert [o["exit_status"] for o in summary["outcomes"]] == ["RepeatedFormatError", "Submitted"]
    rows = [json.loads(line) for line in (tmp_path / "run" / "actions.jsonl").read_text().splitlines()]
    silent_rows = [row for row in rows if row["kind"] == "no_command"]
    assert len(silent_rows) == 3 and silent_rows[0]["output"] == "I think this is done." and silent_rows[0]["task"] == "one"
    assert score_session(rows, [{"name": "x", "say": "first", "per_task": True, "trigger": "", "obeys": "y"}])["x"]["applies"] == 1


def test_a_fact_is_scored_from_the_task_after_it_was_read_in_commands_or_their_output() -> None:
    rule = {"name": "tag", "say": "first", "per_task": True, "from_task": 1, "trigger": "", "obeys": "tag-7f3a"}
    actions = [{"kind": "bash", "call": 1, "task": "one", "compactions": 0, "command": "cat /opt/tag", "output": "tag-7f3a\n"},
               {"kind": "bash", "call": 2, "task": "two", "compactions": 0, "command": "echo tag-7f3a", "output": "tag-7f3a\n"},
               {"kind": "bash", "call": 3, "task": "three", "compactions": 1, "command": "echo $(cat notes)", "output": "tag-7f3a\n"},
               {"kind": "bash", "call": 4, "task": "four", "compactions": 2, "command": "cat /opt/tag", "output": "no such file\n"}]
    scored = score_session(actions, [rule])["tag"]
    assert scored["applies"] == 3 and scored["obeyed"] == 2
    assert scored["by_compactions"] == {"0": [1, 1], "1": [1, 1], "2": [0, 1]}


# ------------------------------------------------------------------ memory that delivers, since the agent does not look

from studies.context_rot.criteria import READER_SYSTEM  # noqa: E402


def _reader_route(script, quotes):
    """The served model: the reader's call, the summaries and notes with no tools, the agent with them."""
    asked = {"reader": 0, "notes": []}

    def chat(messages, tools):
        if tools:
            return script(messages, tools)
        if messages[0]["content"] == READER_SYSTEM:
            asked["reader"] += 1
            found = [quote for quote in quotes if quote in messages[1]["content"]]
            return _text(json.dumps({"standing": found}))
        if "notes file" in messages[0]["content"]:
            asked["notes"].append(messages[0]["content"])
            return _text("Notes: pass -B to Python.")
        return _text("a summary")

    return chat, asked


def test_rules_read_as_each_message_arrives_live_in_commit_headers_and_reach_every_call(tmp_path: Path) -> None:
    rule = SESSION["rules"][0]["text"]
    script = _Script([_bash("python3 a.py " + "word " * 30, "c1"), _bash("ls " + "word " * 30, "c2"), _bash(DONE, "c3"), _bash(DONE, "c4")])
    chat, asked = _reader_route(script, [rule])
    session = load_session({**SESSION, "rules": SESSION["rules"][:1]})
    summary = run_session(session, arm="taste_rules", chat=chat, make_env=lambda task: _Shell(), count=_count, out=tmp_path / "run",
                          workdir=tmp_path / "w", package_prompt=_prompt)
    assert [t["function"]["name"] for t in script.sent[0]["tools"]] == ["bash"], "delivered, so no search is offered"
    assert all(rule in sent["messages"][0]["content"] for sent in script.sent), "in the system prompt at every call"
    assert summary["compactions"] > 0 and rule in script.sent[-1]["messages"][0]["content"], "a summary cannot lose it"
    assert asked["reader"] >= 1 and summary["reader_calls"] == asked["reader"]
    assert summary["ledger"]["name"] == "Taste memory, study-driven" and summary["ledger"]["search_calls"] == 0
    assert summary["events"][0]["delivered"].count(rule) == 1


def test_a_notes_file_is_rewritten_at_each_compaction_kept_in_git_and_reaches_every_later_call(tmp_path: Path) -> None:
    for arm, clause in (("notes_told", True), ("notes_neutral", False)):
        script = _Script([_bash("python3 a.py " + "word " * 30, "c1"), _bash("ls " + "word " * 30, "c2"), _bash(DONE, "c3"), _bash(DONE, "c4")])
        chat, asked = _reader_route(script, [])
        summary = run_session(load_session({**SESSION, "rules": SESSION["rules"][:1]}), arm=arm, chat=chat, make_env=lambda task: _Shell(),
                              count=_count, out=tmp_path / arm, workdir=tmp_path / "w", package_prompt=_prompt)
        assert len(asked["notes"]) == summary["compactions"] > 0
        assert ("anything the user asked it to keep doing" in asked["notes"][0]) is clause
        assert "python3 a.py" in asked["notes"][0], "the note-taker sees what the compaction is about to fold away"
        assert "Notes: pass -B to Python." in script.sent[-1]["messages"][0]["content"]
        assert "Notes: pass -B to Python." not in script.sent[0]["messages"][0]["content"]
        assert summary["ledger"]["name"] == "Taste memory, study-driven" and summary["note_versions"] == summary["compactions"]
        assert [t["function"]["name"] for t in script.sent[0]["tools"]] == ["bash"]


def test_the_reader_does_not_read_the_harnesss_own_messages_as_the_users_rules(tmp_path: Path) -> None:
    rule = SESSION["rules"][0]["text"]
    silent = _reply([], content="done")
    script = _Script([silent, _bash(DONE, "c1"), _bash(DONE, "c2")])
    nagging = "Every response MUST include at least one tool call."
    chat, _ = _reader_route(script, [rule, nagging])
    session = load_session({**SESSION, "rules": SESSION["rules"][:1], "compact_at_tokens": 0})
    run_session(session, arm="taste_rules", chat=chat, make_env=lambda task: _Shell(), count=_count, out=tmp_path / "run",
                workdir=tmp_path / "w", package_prompt=_prompt)
    system = script.sent[-1]["messages"][0]["content"]
    assert rule in system and system.count(rule) == 1
    assert nagging not in system
    assert "\n\n- " not in system.split("\n", 1)[1], "one record to a line"


def test_a_machine_that_will_not_build_leaves_its_task_and_the_session_goes_on(tmp_path: Path) -> None:
    def make_env(task):
        if task["name"] == "one":
            raise RuntimeError("the image could not be fetched")
        return _Shell()

    script = _Script([_bash(DONE, "c1")])
    summary = run_session(load_session({**SESSION, "rules": [], "compact_at_tokens": 0}), arm="compaction", chat=_routed(script), make_env=make_env,
                          count=_count, out=tmp_path / "run", workdir=tmp_path / "w", package_prompt=_prompt)
    assert [o["exit_status"] for o in summary["outcomes"]] == ["MachineFailed", "Submitted"]
    assert "the image could not be fetched" in summary["outcomes"][0]["reason"]
    assert summary["ended_by"] is None
    first = script.sent[0]["messages"]
    assert [m["role"] for m in first] == ["system", "user"], "the next task opens the conversation"
    assert "Do two." in first[1]["content"]


def test_the_session_file_can_point_at_images_fetched_once(tmp_path: Path) -> None:
    from studies.long_run.make_session import build

    root = tmp_path / "tb"
    folder = root / "alpha"
    (folder / "environment").mkdir(parents=True)
    (folder / "task.toml").write_text('[environment]\ndocker_image = "owner/alpha:1"\ncpus = 1\nmemory_mb = 2048\n')
    (folder / "instruction.md").write_text("Do alpha.\n")
    (folder / "environment" / "Dockerfile").write_text("FROM python:3.13\n")
    template = {"agent": {"system_template": "You help.", "instance_template": "Solve: {{task}}"},
                "model": {"observation_template": "{{output.output}}", "format_error_template": "{{error}}"}}
    session = build(root, template, seed=1, tasks=1, compact_at=8000, keep_recent=2000, prompt="anthropic", later_rule_at=20, task_calls=40,
                    images=tmp_path / "images")
    assert session["tasks"][0]["image"] == str(tmp_path / "images" / "alpha.sif")


# ------------------------------------------------------------------ what the second review found


_TEMPLATE = ("{% if finish_reason is defined and finish_reason == 'length' %}Cut off.{% else %}Tool call error: {{error}} "
             "Every response needs to use the bash tool. To end the task run the submit command.{% endif %}")


def test_a_reply_with_no_command_gets_the_agents_published_guidance_and_the_record_says_why(tmp_path: Path) -> None:
    script = _Script([_reply([], "All done, I think."), _bash(DONE, "c1")])
    rows = []
    shell = _Shell()
    model = ToolModel(script, memory_tools=False, format_error_template=_TEMPLATE)
    agent = LongRunAgent(model, MemoryEnvironment(shell, None), count=_count, on_action=rows.append, system_template="s",
                         instance_template="{{task}}", step_limit=10, cost_limit=0)
    agent.run_tasks([{"name": "one", "instruction": "Do it."}], lambda task: shell)
    said = [m["content"] for m in script.sent[1]["messages"] if m["role"] == "user"][-1]
    assert "Every response needs to use the bash tool" in said and "No tool calls found" in said
    silent = next(row for row in rows if row["kind"] == "no_command")
    assert silent["finish_reason"] == "tool_calls" and "No tool calls found" in silent["error"] and silent["output"] == "All done, I think."


def test_a_reply_cut_off_by_the_output_cap_is_told_so() -> None:
    model = ToolModel(lambda messages, tools: None, memory_tools=False, format_error_template=_TEMPLATE)
    model.last_finish = "length"
    assert model._refuse("x", has_tool_calls=False).messages[0]["content"] == "Cut off."


def test_a_delivered_block_does_not_bring_compaction_forward() -> None:
    def run(standing):
        script = _Script([_bash("word " * 8 + str(n), f"c{n}") for n in range(6)] + [_bash(DONE, "z")])
        agent, _ = _agent(script, summarize=lambda older, previous: "short", compact_at_tokens=60, keep_recent_tokens=10)
        agent.standing = standing
        agent.run(task="Do it.")
        return len(agent.events)

    assert run(lambda: "block " * 200) == run(None) > 0


def test_the_system_prompt_is_not_recorded_since_it_is_in_front_of_the_model_anyway(tmp_path: Path) -> None:
    store = GitTurns(tmp_path)
    agent, _ = _agent(_Script([_bash(DONE, "c1")]), store=store)
    try:
        agent.run(task="Fix it.")
        assert agent.recorded == 2  # the task and the reply with its command
    finally:
        store.close()


def test_when_the_first_machine_fails_the_rules_and_the_tag_go_to_the_first_one_that_opens(tmp_path: Path) -> None:
    session = load_session({**SESSION, "rules": SESSION["rules"][:1], "first_prepare": ["plant the tag"],
                            "tasks": [{"name": "broken", "instruction": "Do broken."}, {"name": "one", "instruction": "Do one."},
                                      {"name": "two", "instruction": "Do two."}]})
    script = _Script([_bash(DONE, "c1"), _bash(DONE, "c2")])
    shells = []

    def make_env(task):
        if task["name"] == "broken":
            raise RuntimeError("no machine")
        shells.append(_Shell())
        return shells[-1]

    summary = run_session(session, arm="compaction", chat=_routed(script), make_env=make_env, count=_count, out=tmp_path / "run",
                          workdir=tmp_path / "w", package_prompt=_prompt)
    first_user = next(m["content"] for m in script.sent[0]["messages"] if m["role"] == "user")
    assert SESSION["rules"][0]["text"] in first_user and "Do one." in first_user
    assert shells[0].ran[0] == "plant the tag" and "plant the tag" not in shells[1].ran
    assert summary["outcomes"][0]["exit_status"] == "MachineFailed"


def test_a_rule_the_harness_sends_twice_makes_the_run_invalid(tmp_path: Path) -> None:
    rule = SESSION["rules"][0]
    session = load_session({**SESSION, "rules": [rule], "tasks": [{"name": "one", "instruction": "Do one."},
                                                                  {"name": "two", "instruction": f"Do two. {rule['text']}"}]})
    script = _Script([_bash(DONE, "c1"), _bash(DONE, "c2")])
    summary = run_session(session, arm="compaction", chat=_routed(script), make_env=lambda task: _Shell(), count=_count,
                          out=tmp_path / "run", workdir=tmp_path / "w", package_prompt=_prompt)
    assert summary["valid"] is False and summary["resent"] == {"bytes": 1}


def test_a_rule_told_once_is_a_valid_run_and_the_prompt_after_each_compaction_is_kept(tmp_path: Path) -> None:
    summary = run_session(load_session(SESSION), arm="compaction", chat=_routed(_session_chat()), make_env=lambda task: _Shell(), count=_count,
                          out=tmp_path / "run", workdir=tmp_path / "w", package_prompt=_prompt)
    assert summary["valid"] is True and summary["compactions"] > 0
    kept = [json.loads(line) for line in (tmp_path / "run" / "prompts.jsonl").read_text().splitlines()]
    assert len(kept) == summary["compactions"] and kept[0]["messages"][0]["role"] == "system"


def test_a_summary_that_puts_a_rule_in_its_own_words_still_counts_as_holding_it() -> None:
    script = _Script([_bash("word " * 30, "c1"), _bash(DONE, "c2")])
    agent, _ = _agent(script, summarize=lambda older, previous: "Use `ls -1` as the user asked.", compact_at_tokens=20, keep_recent_tokens=2,
                      watch={"ls": "From now on, when you list a folder, use ls -1 and no other form of ls."}, mentions={"ls": r"ls\s+-\w*1\b"})
    agent.run(task="Do it.")
    assert agent.events[0]["watched_in_summary"]["ls"] is False and agent.events[0]["mentioned_in_summary"]["ls"] is True


def test_a_task_where_the_agent_says_the_tag_but_never_runs_it_counts_as_known_not_ran() -> None:
    rules = [{"name": "tag", "say": "first", "per_task": True, "from_task": 1, "trigger": "", "obeys": "tag-7f3a"}]
    actions = [{"kind": "bash", "call": 1, "task": "one", "compactions": 0, "command": "cat /opt/tag", "output": "tag-7f3a"},
               {"kind": "bash", "call": 2, "task": "two", "compactions": 0, "command": "echo x", "said": ""},
               {"kind": "no_command", "call": 3, "task": "two", "compactions": 0, "command": "", "output": "The tag is tag-7f3a."},
               {"kind": "bash", "call": 4, "task": "three", "compactions": 1, "command": "echo tag-7f3a"}]
    scored = score_session(actions, rules)["tag"]
    assert (scored["obeyed"], scored["applies"]) == (1, 2)
    assert (scored["knows"], scored["knows_of"]) == (2, 2)
    assert scored["by_task"] == {"two": {"obeyed": 0, "applies": 1}, "three": {"obeyed": 1, "applies": 1}}


def test_every_rule_is_also_scored_task_by_task_so_arms_that_end_early_can_be_paired() -> None:
    rules = [{"name": "bytes", "say": "first", "trigger": r"\bpython3?\s", "obeys": r"\bpython3?\s+-B\b"}]
    actions = [{"kind": "bash", "call": 1, "task": "one", "compactions": 0, "command": "python3 a.py"},
               {"kind": "bash", "call": 2, "task": "one", "compactions": 0, "command": "python3 -B a.py"},
               {"kind": "bash", "call": 3, "task": "two", "compactions": 1, "command": "python3 -B b.py"}]
    assert score_session(actions, rules)["bytes"]["by_task"] == {"one": {"obeyed": 1, "applies": 2}, "two": {"obeyed": 1, "applies": 1}}


def test_a_folder_that_already_holds_a_run_is_refused_rather_than_appended_to(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "actions.jsonl").write_text("{}\n")
    with pytest.raises(FileExistsError, match="no resume"):
        run_session(load_session(SESSION), arm="compaction", chat=_routed(_session_chat()), make_env=lambda task: _Shell(), count=_count,
                    out=tmp_path / "run", workdir=tmp_path / "w", package_prompt=_prompt)


def test_an_empty_reply_from_the_note_taker_leaves_the_notes_as_they_were(tmp_path: Path) -> None:
    from studies.long_run.session import make_note_taker
    from studies.memory_probe.stores import NotesStore

    store = NotesStore(tmp_path, ["the rule is -B"], history=True)
    skipped = [0]
    try:
        replies = iter(["", "", ""])
        make_note_taker(lambda messages, tools: _text(next(replies)), "told", store, skipped)([{"role": "user", "content": "x"}])
        assert store.newest() == "the rule is -B" and store.commits == 1 and skipped == [1]
        replies = iter(["", "kept and more"])
        make_note_taker(lambda messages, tools: _text(next(replies)), "told", store, skipped)([{"role": "user", "content": "x"}])
        assert store.newest() == "kept and more" and store.commits == 2 and skipped == [1]
    finally:
        store.close()


def test_a_notes_version_cut_off_by_the_output_cap_is_never_kept_and_the_session_goes_on(tmp_path: Path) -> None:
    from studies.long_run.session import make_note_taker
    from studies.memory_probe.stores import NotesStore

    cut = _text("half a no")
    cut.choices[0].finish_reason = "length"
    store = NotesStore(tmp_path, ["the rule is -B"], history=True)
    skipped = [0]
    try:
        make_note_taker(lambda messages, tools: cut, "told", store, skipped)([{"role": "user", "content": "x"}])
        assert store.newest() == "the rule is -B" and skipped == [1]
    finally:
        store.close()


# ------------------------------------------------------------------ the third review


def test_the_memory_holds_the_talk_and_not_its_own_searches(tmp_path: Path) -> None:
    store = GitTurns(tmp_path)
    script = _Script([_bash("ls deploy", "c1"), _reply([_call(SEARCH, {"description": "deploy"}, "s1")]), _reply([_call(SEARCH, {"description": "deploy"}, "s2")]),
                      _bash(DONE, "c2")])
    agent, _ = _agent(script, store=store)
    agent.run_tasks([{"name": "one", "instruction": "Do the deploy."}], lambda task: _Shell())
    last = [message for message in script.sent[-1]["messages"] if message["role"] == "tool"][-1]
    results = json.loads(last["content"])["results"]
    assert results and not any(SEARCH in json.dumps(result) or "pointer" in result["snippet"] for result in results)
    assert agent.recorded == 4, "the task, a command, its output and the last command"
    store.close()


def test_when_a_message_holds_a_command_and_a_search_only_the_command_goes_in_the_memory(tmp_path: Path) -> None:
    store = GitTurns(tmp_path)
    kept: list[dict] = []
    script = _Script([_reply([_call("bash", {"command": "ls deploy"}, "c1"), _call(SEARCH, {"description": "deploy"}, "s1")], "I will look."), _bash(DONE, "c2")])
    agent, _ = _agent(script, store=store)
    agent.record = kept.append
    agent.run_tasks([{"name": "one", "instruction": "Do it."}], lambda task: _Shell())
    both = next(row for row in kept if "ls deploy" in row["content"] and row["role"] == "assistant")
    assert "I will look." in both["content"] and SEARCH not in both["content"]
    store.close()


def test_a_search_result_merely_shown_is_not_knowing_but_the_agents_own_words_are() -> None:
    rule = {"name": "tag", "say": "first", "per_task": True, "from_task": 0, "trigger": "", "obeys": "tag-7f3a"}
    shown = [{"kind": SEARCH, "call": 1, "task": "two", "compactions": 0, "command": f"[{SEARCH}] {{}}", "output": "the tag is tag-7f3a", "said": ""},
             {"kind": "bash", "call": 2, "task": "two", "compactions": 0, "command": "ls", "output": "x", "said": ""}]
    assert score_session(shown, [rule])["tag"]["knows"] == 0
    said = [shown[0], {**shown[1], "said": "the tag is tag-7f3a"}]
    assert score_session(said, [rule])["tag"]["knows"] == 1
    silent = [{"kind": "no_command", "call": 1, "task": "two", "compactions": 0, "command": "", "output": "tag-7f3a"}]
    assert score_session(silent, [rule])["tag"]["knows"] == 1


def test_a_plant_that_failed_leaves_that_machine_and_the_next_one_gets_it(tmp_path: Path) -> None:
    class Full(_Shell):
        def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
            if "/opt/tag" in action["command"]:
                return {"output": "No space left on device", "returncode": 1, "exception_info": ""}
            return super().execute(action, cwd, timeout=timeout)

    shells = [Full(), _Shell()]
    machines = iter(shells)
    session = load_session({**SESSION, "rules": [], "compact_at_tokens": 0, "first_prepare": ["echo tag-7f3a > /opt/tag"]})
    summary = run_session(session, arm="compaction", chat=_routed(_Script([_bash(DONE, "c1")])), make_env=lambda task: next(machines), count=_count,
                          out=tmp_path / "run", workdir=tmp_path / "w", package_prompt=_prompt)
    assert [o["exit_status"] for o in summary["outcomes"]] == ["MachineFailed", "Submitted"]
    assert "No space left" in summary["outcomes"][0]["reason"]
    assert shells[1].ran[0] == "echo tag-7f3a > /opt/tag" and summary["planted_on"] == "two"


def test_a_session_where_nothing_could_be_planted_is_not_valid(tmp_path: Path) -> None:
    class Full(_Shell):
        def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
            return {"output": "No space left on device", "returncode": 1, "exception_info": ""}

    session = load_session({**SESSION, "rules": [], "compact_at_tokens": 0, "first_prepare": ["echo tag-7f3a > /opt/tag"]})
    summary = run_session(session, arm="compaction", chat=_routed(_Script([])), make_env=lambda task: Full(), count=_count,
                          out=tmp_path / "run", workdir=tmp_path / "w", package_prompt=_prompt)
    assert summary["planted_on"] is None and summary["valid"] is False


def test_the_whole_output_of_a_long_command_is_scored(tmp_path: Path) -> None:
    class Long(_Shell):
        def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
            if action["command"].startswith("echo COMPLETE"):
                return super().execute(action, cwd, timeout=timeout)
            return {"output": "x" * 5000 + " tag-7f3a", "returncode": 0, "exception_info": ""}

    tag = {"name": "tag", "say": "first", "text": "Print the tag.", "per_task": True, "trigger": "", "obeys": "tag-7f3a"}
    session = load_session({**SESSION, "rules": [tag], "compact_at_tokens": 0, "tasks": [{"name": "one", "instruction": "Do one."}]})
    summary = run_session(session, arm="compaction", chat=_routed(_Script([_bash("build", "c1"), _bash(DONE, "c2")])), make_env=lambda task: Long(), count=_count,
                          out=tmp_path / "run", workdir=tmp_path / "w", package_prompt=_prompt)
    assert summary["scores"]["tag"]["obeyed"] == 1


def test_a_record_the_ledger_refuses_makes_the_session_invalid(tmp_path: Path, monkeypatch) -> None:
    from studies import memory_ledger

    monkeypatch.setattr("studies.long_run.session.check", lambda records: (_ for _ in ()).throw(memory_ledger.ClaimRefused("no commit recorded")))
    summary = run_session(load_session(SESSION), arm="taste", chat=_routed(_session_chat()), make_env=lambda task: _Shell(), count=_count,
                          out=tmp_path / "run", workdir=tmp_path / "w", package_prompt=_prompt)
    assert summary["valid"] is False and "no commit recorded" in summary["ledger_refused"]


def test_the_ledger_of_every_arm_passes_its_check(tmp_path: Path) -> None:
    session = load_session(SESSION)
    summary = run_session(session, arm="taste", chat=_routed(_session_chat()), make_env=lambda task: _Shell(), count=_count,
                          out=tmp_path / "run", workdir=tmp_path / "w", package_prompt=_prompt)
    assert summary["valid"] is True and summary["ledger_refused"] is None


def test_the_command_line_says_when_a_session_ended_badly(tmp_path: Path, monkeypatch) -> None:
    from studies.long_run import session as module

    def ended(*args, **kwargs) -> dict:
        return {"arm": "taste", "outcomes": [], "calls": 0, "compactions": 0, "memory_turns": 0, "scores": {}, "ended_by": "RuntimeError: no model", "valid": True}

    monkeypatch.setattr(module, "run_session", ended)
    monkeypatch.setattr(module, "package_prompt", lambda: _prompt)
    (tmp_path / "s.json").write_text(json.dumps(SESSION))
    code = module.main(["--session", str(tmp_path / "s.json"), "--url", "http://x", "--model", "m", "--tokenizer", "words", "--out", str(tmp_path / "o"), "--workdir",
                        str(tmp_path / "w"), "--arm", "taste", "--env", "local"])
    assert code == 1

