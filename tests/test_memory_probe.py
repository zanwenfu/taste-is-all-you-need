"""Paper 1's harness: a conversation kept turn by turn in Taste's real memory, and a probe that may search it.

The rule-loss benchmark gives a long history, a summary that replaced it, and a question the rule should decide.
Here the whole history is also committed into a fresh memory store, one turn per commit, and the model answering
the question is offered search and fetch. What it does with them is the measurement: whether it looks, whether
what it finds holds the rule, and whether its answer obeys it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from studies.memory_ledger import STUDY_DRIVEN, check
from studies.memory_probe.stores import FlatTurns, GitTurns
from studies.memory_probe.tool_loop import ask_with_tools, found_in, grade
from taste.memstore import CommitHeader
from taste.memstore.tools import FETCH, SEARCH, definitions

RULE = "For the rest of this session, always show me a draft and wait for my go-ahead before sending anything."


@pytest.fixture(autouse=True)
def _pinned_code(monkeypatch) -> None:
    monkeypatch.setenv("TASTE_COMMIT", "abc123")


def _history() -> list[dict]:
    return [
        {"role": "user", "content": f"Help me plan the offsite. {RULE}"},
        {"role": "assistant", "content": "Sure. What dates are you considering?"},
        {"role": "user", "content": "The second week of June, near the coast."},
        {"role": "assistant", "content": "I found three venues by the coast with rooms that week."},
        {"role": "user", "content": "Book the one with the best reviews for the kickoff dinner."},
        {"role": "assistant", "content": "The Harbor House has the best reviews; it seats forty."},
    ]


def _count(text: str) -> int:
    return len(text.split())


# ------------------------------------------------------------------ the stores


def test_every_turn_is_its_own_commit_and_a_rule_is_found_by_its_words(tmp_path: Path) -> None:
    store = GitTurns(tmp_path)
    try:
        store.ingest(_history())
        assert store.commits == 6
        results = json.loads(store.answer(SEARCH, {"description": "draft go-ahead before sending"})[0])["results"]
        assert results[0]["kind"] == "transcript"
        fetched = json.loads(store.answer(FETCH, {"pointer": results[0]["pointer"]})[0])["text"]
        assert RULE in fetched
    finally:
        store.close()
    assert not any(tmp_path.iterdir())


def test_a_header_is_written_per_turn_when_asked_for(tmp_path: Path) -> None:
    store = GitTurns(tmp_path, header=lambda number, turn: CommitHeader(
        what=f"turn {number} by {turn['role']}", keywords=("teambuilding",) if number == 0 else ()))
    try:
        store.ingest(_history())
        (result,) = json.loads(store.answer(SEARCH, {"description": "teambuilding"})[0])["results"]
        # The word is only in the header, and the hit stands for the turn that commit added.
        assert result["kind"] == "transcript" and "turn 0 by user" in result["header"]
        assert store.turn_of(result["pointer"]) == 0
        fetched = json.loads(store.answer(FETCH, {"pointer": result["pointer"]})[0])["text"]
        assert RULE in fetched and result["snippet"] in fetched
        # A turn found both by its words and by its header is listed once.
        both = json.loads(store.answer(SEARCH, {"description": "teambuilding offsite", "limit": 20})[0])["results"]
        assert len({r["pointer"] for r in both}) == len(both)
    finally:
        store.close()


def test_the_flat_list_ranks_exactly_as_the_git_store_with_headers_off(tmp_path: Path) -> None:
    """The same turns under the same keyword code: a difference here is a bug, never a finding."""
    git, flat = GitTurns(tmp_path), FlatTurns()
    try:
        git.ingest(_history())
        flat.ingest(_history())
        for query in ("draft go-ahead", "coast venues June", "reviews dinner", "Harbor House seats", "user"):
            results = json.loads(git.answer(SEARCH, {"description": query, "limit": 20})[0])["results"]
            assert flat.rank(query) == [git.turn_of(r["pointer"]) for r in results], query
    finally:
        git.close()


def test_the_ledger_says_which_code_ran_and_how_often_it_looked(tmp_path: Path) -> None:
    store = GitTurns(tmp_path)
    try:
        store.ingest(_history())
        store.answer(SEARCH, {"description": "draft"})
        record = store.ledger()
        assert record["name"] == STUDY_DRIVEN and record["mode"] == "agent_tools" and record["search_calls"] == 1
        check([record])
    finally:
        store.close()


# ------------------------------------------------------------------ the tool loop


def _call(name: str, arguments: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def _reply(content: str | None = None, calls: list | None = None, finish: str = "stop") -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3))


class _Script:
    """A stand-in model that replies from a script and records what it was sent."""

    def __init__(self, replies: list) -> None:
        self.replies, self.sent = list(replies), []

    def __call__(self, messages: list[dict], tools: list | None) -> SimpleNamespace:
        self.sent.append({"messages": json.loads(json.dumps(messages)), "tools": tools})
        return self.replies.pop(0)


def _probe() -> list[dict]:
    return [{"role": "system", "content": "You are an assistant."},
            {"role": "assistant", "content": "Summary: we are planning an offsite by the coast."},
            {"role": "user", "content": "Email the venue to confirm. A) Show me a draft first B) Send it now"}]


def test_a_model_that_searches_then_fetches_then_answers(tmp_path: Path) -> None:
    store = GitTurns(tmp_path)
    store.ingest(_history())
    try:
        pointer = json.loads(store.answer(SEARCH, {"description": "draft"})[0])["results"][0]["pointer"]
        store.store.memory_counts.clear()
        model = _Script([
            _reply(calls=[_call(SEARCH, json.dumps({"description": "draft before sending"}), "c1")], finish="tool_calls"),
            _reply(calls=[_call(FETCH, json.dumps({"pointer": pointer}), "c2")], finish="tool_calls"),
            _reply("I will show a draft first.\nA"),
        ])
        run = ask_with_tools(model, _probe(), definitions("chat"), store.answer, max_rounds=4, budget=10_000,
                             count=_count)
    finally:
        store.close()

    assert run.final == "I will show a draft first.\nA" and grade(run.final, "A") is True
    assert (run.searches, run.fetches, run.rounds, run.tool_errors) == (1, 1, 2, 0)
    assert found_in(run, RULE) is True
    second = model.sent[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "c1"
    assert second[-1] == {"role": "tool", "tool_call_id": "c1", "content": run.calls[0]["output"]}
    assert model.sent[0]["tools"] == definitions("chat")


def test_a_model_that_never_looks_is_counted_as_not_looking(tmp_path: Path) -> None:
    model = _Script([_reply("B")])
    run = ask_with_tools(model, _probe(), definitions("chat"), lambda n, a: ("{}", False), max_rounds=4,
                         budget=100, count=_count)
    assert (run.searches, run.fetches, run.rounds) == (0, 0, 0)
    assert found_in(run, RULE) is False and grade(run.final, "A") is False


def test_arguments_that_are_not_json_come_back_as_an_error_and_are_counted() -> None:
    model = _Script([_reply(calls=[_call(SEARCH, "{not json", "c1")], finish="tool_calls"), _reply("A")])
    asked = []
    run = ask_with_tools(model, _probe(), definitions("chat"), lambda n, a: asked.append(a) or ("{}", False),
                         max_rounds=4, budget=100, count=_count)
    assert run.tool_errors == 1 and run.unparsed_arguments == 1 and asked == []
    assert "error" in json.loads(run.calls[0]["output"])


def test_tool_output_past_the_memory_budget_is_refused() -> None:
    long = json.dumps({"text": "word " * 80})
    model = _Script([
        _reply(calls=[_call(FETCH, json.dumps({"pointer": "p"}), "c1")], finish="tool_calls"),
        _reply(calls=[_call(FETCH, json.dumps({"pointer": "p"}), "c2")], finish="tool_calls"),
        _reply("A"),
    ])
    run = ask_with_tools(model, _probe(), definitions("chat"), lambda n, a: (long, False), max_rounds=4,
                         budget=100, count=_count)
    assert run.calls[0]["output"] == long
    assert "budget" in json.loads(run.calls[1]["output"])["error"]
    assert run.tool_tokens <= 100 and run.budget_refusals == 1


def test_a_model_that_keeps_calling_tools_is_asked_once_more_without_them() -> None:
    looping = [_reply(calls=[_call(SEARCH, json.dumps({"description": "x"}), f"c{n}")], finish="tool_calls")
               for n in range(3)]
    model = _Script([*looping, _reply("A")])
    run = ask_with_tools(model, _probe(), definitions("chat"), lambda n, a: ('{"results": []}', False),
                         max_rounds=3, budget=1000, count=_count)
    assert run.rounds == 3 and run.round_limit_hit is True
    assert model.sent[-1]["tools"] is None and run.final == "A"


def test_grading_reads_the_last_non_empty_line_as_the_package_does() -> None:
    assert grade("Thinking...\n\nA\n", "A") is True
    assert grade("B", "A") is False
    assert grade("A or B", "A") is None
    assert grade("", "A") is None


# ------------------------------------------------------------------ one case, start to finish

from studies.memory_probe.cases import SENTENCE, case_messages, run_case  # noqa: E402


def _case() -> dict:
    return {"id": "h00:sc01", "haystack_id": "h00", "sc_id": "sc01", "sc_type": "process", "sc_text": RULE,
            "compliant_letter": "A", "system_prompt": "You are an assistant.", "developer_prompt": "Be helpful.",
            "probe_prompt": "Email the venue to confirm. A) Show me a draft first B) Send it now",
            "post_sc_probe_prompt": f"{RULE}\nEmail the venue to confirm. A) Show me a draft first B) Send it now",
            "messages": _history()}


def test_the_probe_sees_the_summary_and_the_question_and_the_second_condition_adds_one_sentence() -> None:
    first = case_messages(_case(), "Summary of the plan.", condition="i", upper=False)
    assert [m["role"] for m in first] == ["system", "developer", "assistant", "user"]
    assert first[1]["content"] == "Be helpful." and first[2]["content"] == "Summary of the plan."
    second = case_messages(_case(), "Summary of the plan.", condition="ii", upper=False)
    assert second[1]["content"] == "Be helpful.\n\n" + SENTENCE
    assert "rule" not in SENTENCE.lower() and "instruction" not in SENTENCE.lower()
    assert case_messages(_case(), "S", condition="i", upper=True)[-1]["content"].startswith(RULE)
    with pytest.raises(ValueError):
        case_messages(_case(), "S", condition="iii", upper=False)


def test_a_case_record_says_whether_the_model_looked_found_and_obeyed(tmp_path: Path) -> None:
    case = _case()

    def chat(messages, tools):
        if tools and not any(m["role"] == "tool" for m in messages):
            return _reply(calls=[_call(SEARCH, json.dumps({"description": "draft go-ahead sending"}), "c1")],
                          finish="tool_calls")
        if tools and sum(m["role"] == "tool" for m in messages) == 1:
            pointer = json.loads(messages[-1]["content"])["results"][0]["pointer"]
            return _reply(calls=[_call(FETCH, json.dumps({"pointer": pointer}), "c2")], finish="tool_calls")
        return _reply("A")

    row = run_case(case, "Summary of the plan.", chat=chat, arm="a5", condition="ii", workdir=tmp_path,
                   max_rounds=4, budget=5000, count=_count, prompt="anthropic", k=1)

    assert (row["looked"], row["found"], row["compliant"]) == (True, True, True)
    assert (row["searches"], row["fetches"], row["commits"]) == (1, 1, 6)
    assert row["found_turns"] == [0]
    assert row["ledger"]["search_calls"] == 1 and row["ledger"]["fetch_calls"] == 1
    check([row["ledger"]])
    assert {"id", "arm", "condition", "prompt", "k", "final", "calls", "usage", "ingest_seconds"} <= set(row)
    assert not any(tmp_path.iterdir())


def test_a_case_where_the_model_does_not_look_is_still_complete(tmp_path: Path) -> None:
    row = run_case(_case(), "S", chat=lambda m, t: _reply("B"), arm="a5", condition="i", workdir=tmp_path,
                   max_rounds=4, budget=5000, count=_count, prompt="pi_mono", k=16)
    assert (row["looked"], row["found"], row["compliant"]) == (False, False, False)
    assert row["ledger"]["search_calls"] == 0 and row["found_turns"] == []
    check([row["ledger"]])


def test_the_header_arm_needs_its_headers_and_uses_them(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="headers"):
        run_case(_case(), "S", chat=lambda m, t: _reply("A"), arm="a4", condition="i", workdir=tmp_path,
                 max_rounds=4, budget=5000, count=_count, prompt="pi_mono", k=1)
    headers = {n: CommitHeader(what=f"turn {n}") for n in range(6)}
    row = run_case(_case(), "S", chat=lambda m, t: _reply("A"), arm="a4", condition="i", workdir=tmp_path,
                   max_rounds=4, budget=5000, count=_count, prompt="pi_mono", k=1,
                   header=lambda number, turn: headers[number])
    assert row["arm"] == "a4" and row["compliant"] is True


# ------------------------------------------------------------------ inputs and whole runs

from studies.memory_probe.export import summaries as export_summaries  # noqa: E402
from studies.memory_probe.run import run_all  # noqa: E402


def _baseline(folder: Path, prompt: str, ids: list[str], *, twice: bool = False) -> None:
    part = folder / "full" / f"hermes-{prompt}" / "h00-05"
    part.mkdir(parents=True)
    rows = [{"id": i, "compacted": [{"role": "assistant", "content": f"summary of {i}"}], "compaction_compliant": True,
             "upper_bound_compliant": True, "full_without_sc_compliant": False} for i in ids]
    (part / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows + (rows[:1] if twice else [])))
    (folder / "full" / f"hermes-{prompt}" / "h00-05-user_record").mkdir()
    deep = folder / "repeated" / prompt / "h00-05"
    deep.mkdir(parents=True)
    deep_rows = [{"id": i, "k": k, "summaries": [f"round {n} of {i}" for n in range(k)],
                  "compaction_compliant": False, "compaction_upper_compliant": True} for i in ids for k in (2, 16)]
    (deep / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in deep_rows))


def test_the_summaries_come_from_the_baseline_runs_at_both_depths(tmp_path: Path) -> None:
    for prompt in ("pi_mono", "anthropic"):
        _baseline(tmp_path, prompt, ["a", "b"])
    rows = export_summaries(tmp_path / "full", tmp_path / "repeated", {"a", "b"})
    assert len(rows) == 8
    one = next(r for r in rows if r["id"] == "a" and r["prompt"] == "anthropic" and r["k"] == 1)
    deep = next(r for r in rows if r["id"] == "a" and r["prompt"] == "anthropic" and r["k"] == 16)
    assert one["summary"] == "summary of a" and one["a1_compliant"] is True and one["floor_compliant"] is False
    assert deep["summary"] == "round 15 of a" and deep["a1_compliant"] is False


def test_a_case_missing_or_doubled_in_a_baseline_run_is_refused(tmp_path: Path) -> None:
    for prompt in ("pi_mono", "anthropic"):
        _baseline(tmp_path, prompt, ["a"])
    with pytest.raises(ValueError, match="missing"):
        export_summaries(tmp_path / "full", tmp_path / "repeated", {"a", "b"})
    doubled = tmp_path / "doubled"
    for prompt in ("pi_mono", "anthropic"):
        _baseline(doubled, prompt, ["a"], twice=True)
    with pytest.raises(ValueError, match="twice"):
        export_summaries(doubled / "full", doubled / "repeated", {"a"})


def test_a_whole_run_resumes_and_counts_the_steps(tmp_path: Path) -> None:
    cases = [{**_case(), "id": f"h00:sc0{n}"} for n in range(3)]
    summaries = {case["id"]: "S" for case in cases}
    out = tmp_path / "run" / "results.jsonl"
    settings = dict(arm="a5", condition="i", workdir=tmp_path / "w", max_rounds=2, budget=500, count=_count,
                    prompt="anthropic", k=1)
    (tmp_path / "w").mkdir()
    summary = run_all(cases[:2], summaries, chat=lambda m, t: _reply("A"), out=out, workers=2, **settings)
    assert summary["cases"] == 2
    summary = run_all(cases, summaries, chat=lambda m, t: _reply("B"), out=out, workers=2, **settings)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 3 and summary["compliant"]["true"] == 2
    assert summary["looked"]["true"] == 0 and summary["ledger"]["looks"]["looked"] == 0
    assert json.loads((out.parent / "results.summary.json").read_text())["cases"] == 3


# ------------------------------------------------------------------ the plain-HTTP client

import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402

from studies.memory_probe.tool_loop import http_chat  # noqa: E402


def _serve(replies: list[tuple[int, dict]]) -> tuple[HTTPServer, list[dict]]:
    received: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            received.append({"path": self.path,
                             "body": json.loads(self.rfile.read(int(self.headers["Content-Length"])))})
            status, body = replies.pop(0)
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, received


def test_the_http_client_sends_the_tools_and_reads_tool_calls_back() -> None:
    reply = {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": SEARCH, "arguments": "{\"description\": \"x\"}"}}]}}],
             "usage": {"prompt_tokens": 12, "completion_tokens": 4}}
    server, received = _serve([(200, reply)])
    try:
        chat = http_chat(f"http://127.0.0.1:{server.server_port}/v1", "m", max_tokens=99)
        answer = chat([{"role": "user", "content": "hi"}], definitions("chat"))
    finally:
        server.shutdown()
    body = received[0]["body"]
    assert received[0]["path"] == "/v1/chat/completions"
    assert body["tools"] == definitions("chat") and body["tool_choice"] == "auto" and body["max_tokens"] == 99
    assert body["reasoning_effort"] == "low"
    call = answer.choices[0].message.tool_calls[0]
    assert (call.id, call.function.name, json.loads(call.function.arguments)) == ("c1", SEARCH, {"description": "x"})
    assert answer.usage.prompt_tokens == 12 and answer.choices[0].finish_reason == "tool_calls"


def test_the_http_client_tries_again_after_a_server_error_and_not_after_a_refusal() -> None:
    plain = {"choices": [{"finish_reason": "stop", "message": {"content": "A"}}], "usage": {}}
    server, received = _serve([(503, {"error": "busy"}), (200, plain), (400, {"error": "too long"})])
    try:
        chat = http_chat(f"http://127.0.0.1:{server.server_port}/v1", "m", max_tokens=9)
        answer = chat([{"role": "user", "content": "hi"}], None)
        assert answer.choices[0].message.content == "A" and answer.choices[0].message.tool_calls is None
        assert "tools" not in received[1]["body"]
        import urllib.error

        with pytest.raises(urllib.error.HTTPError):
            chat([{"role": "user", "content": "again"}], None)
    finally:
        server.shutdown()
    assert len(received) == 3


# ------------------------------------------------------------------ headers for the header arm

from studies.memory_probe.headers import StoredTurnHeaders, write_turn_headers  # noqa: E402


def test_each_distinct_turn_gets_one_header_call_and_the_store_reads_them_back(tmp_path: Path) -> None:
    asked: list[str] = []

    def ask(prompt: str) -> dict:
        asked.append(prompt)
        return {"text": '{"what": "A turn about the offsite.", "keywords": ["offsite", 7]}', "prompt_tokens": 5,
                "completion_tokens": 2}

    cases = [{**_case(), "id": "a"}, {**_case(), "id": "b"}]
    cases[1]["messages"] = [{"role": "user", "content": "A different first turn."}, *_history()[1:]]
    path = tmp_path / "headers.jsonl"
    counts = write_turn_headers(cases, ask, path, workers=2)
    assert counts["turns"] == 12 and counts["calls"] == 7 and counts["failed"] == 0
    assert write_turn_headers(cases, ask, path, workers=2)["calls"] == 0
    assert all("Turn:\nuser: " in p or "Turn:\nassistant: " in p for p in asked)

    headers = StoredTurnHeaders(path)
    header = headers(0, _history()[0])
    assert header == CommitHeader(what="A turn about the offsite.", keywords=("offsite",))
    assert headers(0, {"role": "user", "content": "never seen"}) == CommitHeader() and headers.missing == 1


def test_a_long_turn_is_cut_before_the_header_model_sees_it(tmp_path: Path) -> None:
    asked: list[str] = []
    case = {**_case(), "messages": [{"role": "tool", "content": "x" * 100_000}]}
    write_turn_headers([case], lambda p: asked.append(p) or {"text": "{}", "prompt_tokens": 1,
                                                             "completion_tokens": 1}, tmp_path / "h.jsonl", workers=1)
    assert len(asked[0]) < 30_000


# ------------------------------------------------------------------ the delivered and flat-file arms

from studies.memory_probe.cases import DELIVERED_HEADER, FLAT_HEADER, flat_file  # noqa: E402


def test_the_rules_arm_delivers_the_records_beside_the_summary_and_offers_no_tools(tmp_path: Path) -> None:
    sent = []

    def chat(messages, tools):
        sent.append((json.loads(json.dumps(messages)), tools))
        return _reply("A")

    row = run_case(_case(), "S", chat=chat, arm="a6", condition="i", workdir=tmp_path, max_rounds=4, budget=500,
                   count=_count, prompt="pi_mono", k=1, delivered=f"- {RULE}")
    (messages, tools), = sent
    assert tools is None and messages[2]["content"] == f"S\n\n{DELIVERED_HEADER}\n- {RULE}"
    assert row["compliant"] is True and row["looked"] is False and row["ledger"]["name"] == "baseline"
    check([row["ledger"]])
    with pytest.raises(ValueError, match="delivered"):
        run_case(_case(), "S", chat=chat, arm="a6", condition="i", workdir=tmp_path, max_rounds=4, budget=500,
                 count=_count, prompt="pi_mono", k=1)


def test_the_strongest_arm_has_the_tools_the_headers_and_the_delivered_records(tmp_path: Path) -> None:
    sent = []

    def chat(messages, tools):
        sent.append((messages[2]["content"], tools))
        return _reply("A")

    headers = {n: CommitHeader(what=f"turn {n}") for n in range(6)}
    row = run_case(_case(), "S", chat=chat, arm="a7", condition="ii", workdir=tmp_path, max_rounds=4, budget=500,
                   count=_count, prompt="pi_mono", k=1, delivered=f"- {RULE}", header=lambda n, t: headers[n])
    content, tools = sent[0]
    assert content.endswith(f"- {RULE}") and [t["function"]["name"] for t in tools] == [SEARCH, FETCH]
    assert row["ledger"]["name"] == STUDY_DRIVEN and row["commits"] == 6


def test_the_flat_file_arm_is_the_whole_conversation_cut_to_the_budget_from_the_chosen_end() -> None:
    whole = flat_file(_history(), budget=10_000, keep="newest", count=_count)
    assert whole.startswith("[user]: Help me plan") and whole.endswith("it seats forty.")
    newest = flat_file(_history(), budget=12, keep="newest", count=_count)
    oldest = flat_file(_history(), budget=12, keep="oldest", count=_count)
    assert _count(newest) <= 12 and _count(oldest) <= 12
    assert newest.endswith("it seats forty.") and RULE not in newest
    assert oldest.startswith("[user]: Help me plan")
    with pytest.raises(ValueError):
        flat_file(_history(), budget=12, keep="middle", count=_count)


def test_the_flat_file_arm_offers_no_tools_and_is_a_baseline(tmp_path: Path) -> None:
    sent = []

    def chat(messages, tools):
        sent.append((messages[2]["content"], tools))
        return _reply("B")

    row = run_case(_case(), "S", chat=chat, arm="a2a", condition="i", workdir=tmp_path, max_rounds=4, budget=12,
                   count=_count, prompt="pi_mono", k=1, keep="newest")
    content, tools = sent[0]
    assert tools is None and content.startswith(f"S\n\n{FLAT_HEADER}\n") and RULE not in content
    assert row["ledger"]["name"] == "baseline" and row["compliant"] is False


from studies.memory_probe.export import delivered as export_delivered  # noqa: E402
from studies.memory_probe.run import delivered_blocks  # noqa: E402


def _records(root: Path, prompt: str, records: dict, part: str = "h00-05") -> None:
    folder = root / prompt / part
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "records.json").write_text(json.dumps({"records": records, "reads": []}))


def test_each_summary_prompt_keeps_the_rule_records_its_own_run_delivered(tmp_path: Path) -> None:
    """The reader ran once per prompt's run and the readings differ, so one prompt's can't stand for the other's."""
    _records(tmp_path, "pi_mono", {"a": [{"text": RULE}], "b": []})
    _records(tmp_path, "anthropic", {"a": [{"text": "other"}, {"text": RULE}], "b": []})

    rows = {(row["id"], row["prompt"]): row for row in export_delivered(tmp_path, {"a", "b"})}

    assert len(rows) == 4
    assert rows["a", "pi_mono"]["block"] == f"- {RULE}" and rows["a", "pi_mono"]["records"] == 1
    assert rows["a", "anthropic"]["block"] == f"- other\n- {RULE}" and rows["a", "anthropic"]["records"] == 2
    assert rows["b", "pi_mono"]["block"] == "(none)" and rows["b", "anthropic"]["records"] == 0


def test_a_case_with_no_rule_records_or_found_twice_in_a_run_is_refused(tmp_path: Path) -> None:
    _records(tmp_path, "pi_mono", {"a": [{"text": RULE}]})
    _records(tmp_path, "anthropic", {"b": [{"text": RULE}]})
    with pytest.raises(ValueError, match="no rule records"):
        export_delivered(tmp_path, {"a"})
    _records(tmp_path, "anthropic", {"a": [{"text": RULE}]})
    _records(tmp_path, "anthropic", {"a": [{"text": RULE}]}, part="h05-10")
    with pytest.raises(ValueError, match="twice"):
        export_delivered(tmp_path, {"a"})


def test_a_run_reads_only_its_own_prompts_rule_records(tmp_path: Path) -> None:
    path = tmp_path / "delivered.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in [
        {"id": "a", "prompt": "pi_mono", "records": 1, "block": "- pi"},
        {"id": "a", "prompt": "anthropic", "records": 1, "block": "- anthropic"},
    ]))
    assert delivered_blocks(path, "anthropic") == {"a": "- anthropic"}
    assert delivered_blocks(path, "pi_mono") == {"a": "- pi"}


# ------------------------------------------------------------------ a notes file the model keeps, with and without history

from studies.memory_probe.notes import chunks, take_notes, write_notes  # noqa: E402
from studies.memory_probe.stores import NotesStore  # noqa: E402


def test_the_conversation_is_read_in_rounds_and_each_round_rewrites_the_notes() -> None:
    assert [len(c) for c in chunks(_history())] == [2, 2, 2]
    asked: list[str] = []

    def ask(prompt: str) -> str:
        asked.append(prompt)
        return f"notes after {len(asked)}"

    versions = take_notes(_history(), ask)
    assert versions == ["notes after 1", "notes after 2", "notes after 3"]
    assert RULE in asked[0] and "(no notes yet)" in asked[0]
    assert "notes after 1" in asked[1] and RULE not in asked[1]


def test_notes_are_written_once_per_case_and_resume(tmp_path: Path) -> None:
    calls = []
    cases = [{**_case(), "id": "a"}, {**_case(), "id": "b"}]
    path = tmp_path / "notes.jsonl"
    counts = write_notes(cases, lambda prompt: calls.append(prompt) or "n", path, workers=2)
    assert counts == {"cases": 2, "written": 2, "failed": 0} and len(calls) == 6
    assert write_notes(cases, lambda prompt: "x", path, workers=2)["written"] == 0


def _versions() -> list[str]:
    return [f"Offsite planning. {RULE}", "Offsite: coast, second week of June.",
            "Offsite: Harbor House booked for the kickoff dinner."]


def test_without_history_only_the_current_notes_can_be_found(tmp_path: Path) -> None:
    store = NotesStore(tmp_path, _versions(), history=False)
    try:
        assert store.commits == 1
        assert json.loads(store.answer(SEARCH, {"description": "draft go-ahead"})[0])["results"] == []
    finally:
        store.close()


def test_with_history_an_overwritten_rule_is_found_and_marked_replaced(tmp_path: Path) -> None:
    store = NotesStore(tmp_path, _versions(), history=True)
    try:
        assert store.commits == 3
        (hit,) = json.loads(store.answer(SEARCH, {"description": "draft go-ahead"})[0])["results"]
        assert hit["current"] is False and hit["kind"] == "file"
        assert RULE in json.loads(store.answer(FETCH, {"pointer": hit["pointer"]})[0])["text"]
        now = json.loads(store.answer(FETCH, {"pointer": hit["replaced_by"]})[0])["text"]
        assert now == _versions()[-1]
    finally:
        store.close()


def test_the_two_notes_arms_differ_only_in_history(tmp_path: Path) -> None:
    for arm, commits in (("a2c", 1), ("a2d", 3)):
        row = run_case(_case(), "S", chat=lambda m, t: _reply("A"), arm=arm, condition="ii", workdir=tmp_path,
                       max_rounds=4, budget=500, count=_count, prompt="pi_mono", k=1, notes=_versions())
        assert row["commits"] == commits and row["ledger"]["name"] == STUDY_DRIVEN
    with pytest.raises(ValueError, match="notes"):
        run_case(_case(), "S", chat=lambda m, t: _reply("A"), arm="a2d", condition="ii", workdir=tmp_path,
                 max_rounds=4, budget=500, count=_count, prompt="pi_mono", k=1)
