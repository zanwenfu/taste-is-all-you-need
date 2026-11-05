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


def test_a_summary_cut_off_by_the_output_cap_stops_the_run() -> None:
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


def test_a_ceiling_with_each_rule_told_again_at_every_task_says_so(tmp_path: Path) -> None:
    script = _Script([_bash(DONE, "c1"), _bash(DONE, "c2")])
    run_session(load_session({**SESSION, "rules": SESSION["rules"][:1]}), arm="restated", chat=_routed(script),
                make_env=lambda task: _Shell(), count=_count, out=tmp_path / "run", workdir=tmp_path / "w",
                package_prompt=_prompt)
    users = [m["content"] for m in script.sent[-1]["messages"] if m["role"] == "user"]
    assert users[-1] == f"{SESSION['rules'][0]['text']}\n\nDo two."


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
