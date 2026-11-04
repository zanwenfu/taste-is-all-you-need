"""Judges, cleaner and bridge for the context-rot reproduction runs (studies/context_rot).

Nothing here calls a model: Bedrock is replaced by a mock transport. The tests that
read the upstream code (LongMemEval's prompts, the conpact package) run only where
that code is on disk, named by LME_REPO and CONPACT_SRC.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path

import httpx
import pytest

from studies.context_rot import bedrock, conpact_bridge, lme_clean, lme_judge, lme_metrics

# -- the Bedrock client ---------------------------------------------------------------


def _reply(text, *, inputs=100, outputs=7, reasoning=None, status=200):
    content = ([{"reasoningContent": {"reasoningText": {"text": reasoning}}}] if reasoning else [])
    content.append({"text": text})
    return httpx.Response(status, json={
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": inputs, "outputTokens": outputs, "totalTokens": inputs + outputs}},
        headers={"x-amzn-requestid": "req-1"})


def _client(handler, model="global.openai.gpt-5.4", **options):
    return bedrock.Converse(model, token="test-only", transport=httpx.MockTransport(handler), **options)


def test_a_converse_call_sends_the_effort_and_reads_only_the_text_block():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return _reply("yes", reasoning="thinking it over")

    reply = _client(handler, effort="low", max_tokens=256).ask("Is it?", system="Judge.")
    assert reply.text == "yes" and reply.input_tokens == 100 and reply.output_tokens == 7
    assert seen["url"].endswith("/model/global.openai.gpt-5.4/converse")
    assert seen["auth"] == "Bearer test-only"
    body = seen["body"]
    assert body["system"] == [{"text": "Judge."}]
    assert body["messages"] == [{"role": "user", "content": [{"text": "Is it?"}]}]
    assert body["inferenceConfig"] == {"maxTokens": 256}
    assert body["additionalModelRequestFields"] == {"reasoning": {"effort": "low"}}


def test_a_throttled_call_is_sent_again_and_a_refused_one_is_not():
    calls = []

    def throttled_once(request):
        calls.append(1)
        return httpx.Response(429, json={"message": "slow down"}) if len(calls) == 1 else _reply("no")

    assert _client(throttled_once, wait_seconds=0).ask("q").text == "no" and len(calls) == 2

    def refused(request):
        return httpx.Response(400, json={"message": "bad request"})

    with pytest.raises(bedrock.BedrockError):
        _client(refused, wait_seconds=0).ask("q")


def test_the_token_is_never_part_of_what_a_reply_or_an_error_shows():
    def refused(request):
        return httpx.Response(403, json={"message": "denied"})

    with pytest.raises(bedrock.BedrockError) as caught:
        _client(refused, wait_seconds=0).ask("q")
    assert "test-only" not in str(caught.value) and "test-only" not in repr(_client(refused))


# -- the LongMemEval judge ------------------------------------------------------------


@pytest.mark.parametrize("text, label", [
    ("yes", True), ("Yes.", True), ("**No**", False), ("no", False), (" YES ", True), ('"no"', False),
    ("yes, the response is correct", None), ("Yes or no", None), ("", None), ("maybe", None),
    ("I would say yes", None)])
def test_a_verdict_must_be_a_clear_yes_or_no(text, label):
    assert lme_judge.parse_verdict(text) is label


def test_the_official_rule_is_kept_beside_the_strict_one():
    assert lme_judge.official_label("I would say yes") is True
    assert lme_judge.official_label("No, yesterday is wrong") is True  # the substring test the paper used
    assert lme_judge.official_label("no") is False


def test_abstention_questions_get_the_abstention_prompt_whatever_their_type():
    prompt = lme_judge.anscheck_prompt("multi-session", "Q?", "None given.", "I don't know.", abstention=True)
    assert prompt.startswith("I will give you an unanswerable question")
    with pytest.raises(NotImplementedError):
        lme_judge.anscheck_prompt("no-such-type", "q", "a", "r")


@pytest.mark.skipif(not os.environ.get("LME_REPO"), reason="needs LongMemEval's code, named by LME_REPO")
def test_the_prompts_are_the_official_ones_byte_for_byte(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(os.environ["LME_REPO"]) / "src" / "evaluation"))
    stubs = {name: types.ModuleType(name) for name in ("backoff", "openai", "tqdm", "numpy")}
    stubs["backoff"].on_exception = lambda *a, **k: (lambda f: f)
    stubs["backoff"].expo = None
    stubs["openai"].RateLimitError = stubs["openai"].APIError = Exception
    stubs["openai"].OpenAI = object
    stubs["tqdm"].tqdm = iter
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "evaluate_qa", raising=False)
    evaluate_qa = importlib.import_module("evaluate_qa")

    cases = [(task, False) for task in lme_judge.TYPES] + [("multi-session", True)]
    for task, abstention in cases:
        assert lme_judge.anscheck_prompt(task, "Q", "A", "R", abstention) == \
            evaluate_qa.get_anscheck_prompt(task, "Q", "A", "R", abstention)


REFS = [
    {"question_id": "a1", "question_type": "single-session-user", "question": "Where?", "answer": "Paris"},
    {"question_id": "a2", "question_type": "temporal-reasoning", "question": "When?", "answer": 3},
    {"question_id": "a3_abs", "question_type": "multi-session", "question": "Who?", "answer": "Not said."},
]


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_judging_writes_the_official_file_and_keeps_every_unclear_verdict(tmp_path):
    hyp = _write(tmp_path / "hyp.jsonl", [{"question_id": q, "hypothesis": h} for q, h in
                                          (("a1", "Paris"), ("a2", "five"), ("a3_abs", "unknown"))])
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps(REFS))
    answers = {"Paris": "yes", "five": "no", "unknown": "It depends"}

    def handler(request):
        prompt = json.loads(request.content)["messages"][0]["content"][0]["text"]
        return _reply(next(v for k, v in answers.items() if f"Model Response: {k}" in prompt))

    out = lme_judge.judge_file(hyp, ref, _client(handler), short_name="gpt-5.4", workers=2)
    assert out.name == "hyp.jsonl.eval-results-gpt-5.4"
    rows = {row["question_id"]: row for row in map(json.loads, out.read_text().splitlines())}
    assert rows["a1"]["autoeval_label"]["label"] is True
    assert rows["a2"]["autoeval_label"]["label"] is False
    unclear = rows["a3_abs"]["autoeval_label"]
    assert unclear["label"] is None and unclear["parse"] == "unparseable"
    assert unclear["judge_output"] == "It depends" and unclear["model"] == "global.openai.gpt-5.4"
    assert unclear["input_tokens"] == 100 and unclear["effort"] == "low"


def test_judging_again_skips_what_is_done(tmp_path):
    hyp = _write(tmp_path / "hyp.jsonl", [{"question_id": "a1", "hypothesis": "Paris"}])
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps(REFS))
    calls = []

    def handler(request):
        calls.append(1)
        return _reply("yes")

    lme_judge.judge_file(hyp, ref, _client(handler), short_name="j")
    lme_judge.judge_file(hyp, ref, _client(handler), short_name="j")
    assert len(calls) == 1


def _row(qid, label):
    return {"question_id": qid, "autoeval_label": {"model": "m", "label": label}}


def test_metrics_follow_the_official_aggregates_and_count_what_could_not_be_judged():
    refs = {r["question_id"]: r for r in REFS}
    summary = lme_metrics.summarize([_row("a1", True), _row("a2", None), _row("a3_abs", True)], refs)
    assert summary["judged"] == 3 and summary["unparseable"] == 1 and summary["missing"] == 0
    assert summary["overall_as_wrong"] == pytest.approx(2 / 3)
    assert summary["overall_excluding"] == pytest.approx(1.0)
    assert summary["abstention"] == pytest.approx(1.0)
    # The official task average: the mean over the six type means, each over its own questions.
    assert summary["task_averaged_as_wrong"] == pytest.approx((1.0 + 0.0 + 1.0) / 3)


def test_metrics_say_how_many_questions_have_no_verdict():
    refs = {r["question_id"]: r for r in REFS}
    assert lme_metrics.summarize([_row("a1", True)], refs)["missing"] == 2


def test_two_judges_are_compared_question_by_question():
    a = [_row("a1", True), _row("a2", False), _row("a3_abs", True)]
    b = [_row("a1", True), _row("a2", True), _row("a3_abs", None)]
    agreement = lme_metrics.agreement(a, b)
    assert agreement["compared"] == 2 and agreement["agree"] == 1 and agreement["either_unparseable"] == 1


# -- the LongMemEval cleaner ---------------------------------------------------------

RAW = [{
    "question_id": "7d1a_abs", "question_type": "knowledge-update", "question": "What car?",
    "answer": "A red one", "question_date": "2023/05/30 (Tue) 10:00",
    "haystack_session_ids": ["answer_7d1a_1", "sharegpt_9", "noans_4"],
    "haystack_dates": ["2023/05/22 (Mon) 09:00", "2023/05/20 (Sat) 08:00", "2023/05/25 (Thu) 12:00"],
    "haystack_sessions": [
        [{"role": "user", "content": "I bought a red car", "has_answer": True},
         {"role": "assistant", "content": "Nice!"}],
        [{"role": "user", "content": "Write a poem"}, {"role": "assistant", "content": "Roses"}],
        [{"role": "user", "content": "Weather?", "has_answer": False}]],
    "answer_session_ids": ["answer_7d1a_1"]}]


def _strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def test_nothing_that_marks_the_answer_reaches_the_memory():
    cleaned, private = lme_clean.clean(RAW)
    seen = list(_strings(cleaned))
    for marker in ("has_answer", "answer_session_ids", "question_type", "knowledge-update", "A red one",
                   "7d1a", "_abs", "answer_", "noans_", "sharegpt"):
        assert not any(marker in text for text in seen), marker
    assert set(cleaned[0]) == {"qid", "question", "question_date", "sessions"}
    assert all(set(turn) == {"role", "content"} for s in cleaned[0]["sessions"] for turn in s["turns"])
    assert private[cleaned[0]["qid"]]["question_id"] == "7d1a_abs"


def test_sessions_come_in_date_order_and_map_back_to_their_original_ids():
    cleaned, private = lme_clean.clean(RAW)
    sessions = cleaned[0]["sessions"]
    assert [s["date"] for s in sessions] == sorted(s["date"] for s in sessions)
    back = private[cleaned[0]["qid"]]["sessions"]
    assert [back[s["sid"]] for s in sessions] == ["sharegpt_9", "answer_7d1a_1", "noans_4"]
    assert sessions[1]["turns"][0]["content"] == "I bought a red car"


def test_neutral_ids_do_not_depend_on_file_order():
    first, _ = lme_clean.clean(RAW)
    again, _ = lme_clean.clean([*RAW[:0], *RAW])
    assert first[0]["qid"] == again[0]["qid"] == lme_clean.neutral("q", "7d1a_abs")


# -- the rule-loss bridge ------------------------------------------------------------


def test_the_retention_verdict_follows_their_rule_and_an_unclear_one_is_counted():
    replies = iter(["maybe", "Probably", "YES, it is there"])

    def handler(request):
        return _reply(next(replies))

    judge = conpact_bridge.RetentionJudge(_client(handler))
    assert judge.verdict("System.", "User.") is True and judge.calls == 3
    never = conpact_bridge.RetentionJudge(_client(lambda request: _reply("unsure")))
    assert never.verdict("S", "U") is None and never.unparseable == 1


def test_counts_show_what_their_summary_drops():
    results = [
        {"source": "hermes", "sc_type": "Action", "retention": True, "full_with_sc_compliant": True,
         "full_without_sc_compliant": None, "compaction_compliant": False, "upper_bound_compliant": True,
         "compaction_output": "A\nB"},
        {"source": "hermes", "sc_type": "Action", "retention": None, "full_with_sc_compliant": None,
         "full_without_sc_compliant": False, "compaction_compliant": True, "upper_bound_compliant": True,
         "compaction_output": "B"},
    ]
    counts = conpact_bridge.unparseable_counts(results)
    assert counts == {"retention": 1, "full_with_sc": 1, "full_without_sc": 1, "compaction": 0,
                      "upper_bound": 0, "instances": 2}


@pytest.mark.parametrize("output, letter, graded", [("B", "B", True), ("A", "B", False), ("A\nB", "B", None),
                                                    (" b ", "B", True), ("B.", "B", None)])
def test_the_paper_rule_grades_only_an_exact_single_letter(output, letter, graded):
    assert conpact_bridge.paper_grade(output, letter) is graded


def test_the_first_contexts_are_chosen_whole():
    rows = [types.SimpleNamespace(haystack_id=h, id=f"{h}-{i}") for h in ("h3", "h1", "h2") for i in range(15)]
    chosen = conpact_bridge.first_contexts(rows, 2)
    assert len(chosen) == 30 and {row.haystack_id for row in chosen} == {"h3", "h1"}


def test_a_history_range_is_chosen_whole_in_file_order():
    rows = [types.SimpleNamespace(haystack_id=h, id=f"{h}-{i}") for h in ("h3", "h1", "h2", "h4") for i in range(15)]
    chosen = conpact_bridge.history_range(rows, 1, 3)
    assert len(chosen) == 30 and {row.haystack_id for row in chosen} == {"h1", "h2"}
    assert conpact_bridge.history_range(rows, 0, 2) == conpact_bridge.first_contexts(rows, 2)


def _slice(folder, rows):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (folder / "judge_inputs.jsonl").write_text(
        "".join(json.dumps({"id": row["id"], "sc_text": "rule " + row["id"]}) + "\n" for row in rows))


def test_slices_merge_into_one_run_with_earlier_verdicts_kept(tmp_path):
    _slice(tmp_path / "a", [{"id": "h1-1", "retention": None}, {"id": "h1-2", "retention": None}])
    _slice(tmp_path / "b", [{"id": "h2-1", "retention": None}])
    judged = tmp_path / "earlier.jsonl"
    judged.write_text(json.dumps({"id": "h1-2", "retention": True}) + "\n")
    rows = conpact_bridge.merge_slices([tmp_path / "a", tmp_path / "b"], tmp_path / "all", verdicts_from=judged,
                                       expected=3)
    assert [(row["id"], row["retention"]) for row in rows] == [("h1-1", None), ("h1-2", True), ("h2-1", None)]
    inputs = [json.loads(line)["id"] for line in (tmp_path / "all" / "judge_inputs.jsonl").read_text().splitlines()]
    assert inputs == ["h1-1", "h1-2", "h2-1"]
    with pytest.raises(SystemExit, match="3 cases, expected 4"):
        conpact_bridge.merge_slices([tmp_path / "a", tmp_path / "b"], tmp_path / "x", expected=4)
    _slice(tmp_path / "c", [{"id": "h2-1", "retention": None}])
    with pytest.raises(SystemExit, match="twice"):
        conpact_bridge.merge_slices([tmp_path / "b", tmp_path / "c"], tmp_path / "y")


def _deferred(folder, count):
    rows = [{"id": f"h1-{i}", "retention": None, "compacted": [{"role": "assistant", "content": "kept"}]}
            for i in range(count)]
    _slice(folder, rows)


def test_judging_that_stops_part_way_resumes_without_asking_again(tmp_path):
    _deferred(tmp_path, 3)
    asked = []

    def handler(request):
        asked.append(1)
        return _reply("YES")

    prompts = lambda sc_text, compacted: ("System.", sc_text)  # noqa: E731
    conpact_bridge.judge_deferred(tmp_path, conpact_bridge.RetentionJudge(_client(handler)), prompts, workers=1)
    assert len(asked) == 3 and len((tmp_path / "verdicts.jsonl").read_text().splitlines()) == 3
    again = conpact_bridge.judge_deferred(tmp_path, conpact_bridge.RetentionJudge(_client(handler)), prompts,
                                          workers=1)
    assert len(asked) == 3 and [row["retention"] for row in again] == [True, True, True]


def test_judging_stops_at_the_spend_cap(tmp_path):
    _deferred(tmp_path, 5)
    # each call: 1,000,000 input tokens at $2.50/M = $2.50
    judge = conpact_bridge.RetentionJudge(_client(lambda request: _reply("NO", inputs=1_000_000, outputs=0)))
    rows = conpact_bridge.judge_deferred(tmp_path, judge, lambda s, c: ("S", s), workers=1, spend_cap=5.0)
    assert [row["retention"] for row in rows] == [False, False, None, None, None]
    assert judge.cost() == pytest.approx(5.0)


def test_a_prompt_too_long_for_the_window_is_found_before_any_call():
    rows = [types.SimpleNamespace(id="h1-1", text=100), types.SimpleNamespace(id="h1-2", text=131_073)]
    over = conpact_bridge.over_window(rows, {"compaction": lambda row: row.text,
                                             "full_with_sc": lambda row: row.text - 50},
                                      window=131_072, output_cap=4_096)
    assert over == [("h1-2", "compaction", 131_073)]  # the server refuses only a prompt that cannot fit


def test_a_prompt_that_fits_with_less_than_the_cap_is_counted_not_refused():
    counts = {"h1-1": {"compaction": 100}, "h1-2": {"compaction": 126_977, "full_with_sc": 130_688},
              "h1-3": {"compaction": 131_072}}
    assert conpact_bridge.short_room(counts, window=131_072, output_cap=4_096) == \
        [("h1-2", "compaction", 4_095), ("h1-2", "full_with_sc", 384)]


def test_prompt_tokens_are_counted_from_the_rendered_text_not_the_encoding_object():
    class Tokenizer:  # transformers 5 returns a two-key encoding from apply_chat_template(tokenize=True)
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            assert tokenize is False and add_generation_prompt is True
            return " ".join(m["role"] + " " + m["content"] for m in messages) + " assistant"

        def __call__(self, text, add_special_tokens=True):
            assert add_special_tokens is False
            return {"input_ids": text.split(), "attention_mask": [1] * len(text.split())}

    row = types.SimpleNamespace(system_prompt="sys", developer_prompt="dev", probe_prompt="which one",
                                messages=[{"role": "user", "content": "one two three"}],
                                messages_without_sc=[{"role": "user", "content": "two three"}])
    counters = conpact_bridge.window_counters(Tokenizer(), lambda messages: [{"role": "user", "content": "x"}])
    assert counters["compaction"](row) == 3
    assert counters["full_with_sc"](row) == 12
    assert counters["full_without_sc"](row) == 11


def test_deferred_verdicts_are_judged_later_and_only_once(tmp_path):
    rows = [{"id": "h1-1", "source": "hermes", "sc_type": "Action", "retention": None,
             "compacted": [{"role": "assistant", "content": "use metric units"}]},
            {"id": "h1-2", "source": "hermes", "sc_type": "Action", "retention": None,
             "compacted": [{"role": "assistant", "content": "nothing"}]},
            {"id": "h1-3", "source": "hermes", "sc_type": "Action", "retention": True,
             "compacted": [{"role": "assistant", "content": "kept"}]}]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "judge_inputs.jsonl").write_text(
        "".join(json.dumps({"id": i, "sc_text": "Use metric units."}) + "\n" for i in ("h1-1", "h1-2", "h1-3")))
    asked = []

    def handler(request):
        user = json.loads(request.content)["messages"][0]["content"][0]["text"]
        asked.append(user)
        return _reply("YES" if "use metric units" in user else "NO")

    def prompts(sc_text, compacted):
        return "System.", sc_text + " | " + compacted[0]["content"]

    judge = conpact_bridge.RetentionJudge(_client(handler))
    judged = conpact_bridge.judge_deferred(tmp_path, judge, prompts, workers=2)
    assert [row["retention"] for row in judged] == [True, False, True]
    assert len(asked) == 2  # the row judged already is not asked again
    written = [json.loads(line) for line in (tmp_path / "results.judged.jsonl").read_text().splitlines()]
    assert [row["retention"] for row in written] == [True, False, True]
    assert [json.loads(line)["retention"] for line in (tmp_path / "results.jsonl").read_text().splitlines()] == \
        [None, None, True]  # the raw file is kept as the GPU run wrote it


@pytest.mark.skipif(not os.environ.get("CONPACT_SRC"), reason="needs the conpact package, named by CONPACT_SRC")
def test_their_scoring_runs_with_the_bedrock_judge_in_place_of_theirs(tmp_path, monkeypatch):
    stubs = {name: types.ModuleType(name) for name in ("datasets", "openai", "tqdm", "tqdm.auto")}
    stubs["datasets"].load_dataset = None
    stubs["tqdm.auto"].tqdm = lambda iterable, **kwargs: iterable

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=self)

        def create(self, **kwargs):
            message = types.SimpleNamespace(content="A")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    stubs["openai"].OpenAI = FakeOpenAI
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.syspath_prepend(os.environ["CONPACT_SRC"])
    # The package's __init__ re-exports a function named evaluate over the submodule.
    evaluation = importlib.import_module("conpact.evaluate")
    Instance = importlib.import_module("conpact.instance").Instance

    instance = Instance(
        id="h1-1", source="hermes", haystack_id="h1", target_length=100000, token_length=10, position="top",
        repeat=1, strict=False, direct=True, sc_id=1, sc_type="Action", sc_text="Use metric units.",
        sc_rendered="For the rest of this session. use metric units.", sc_message_indices=[0],
        system_prompt="s", developer_prompt="d", messages=[{"role": "user", "content": "hi"}],
        messages_without_sc=[{"role": "user", "content": "hi"}], probe="p", correct_answer="c",
        incorrect_answer="i", swap_seed=1, compliant_letter="A", probe_prompt="pp", post_sc_probe_prompt="pp2",
        meta="{}")
    prompts = []

    def handler(request):
        body = json.loads(request.content)
        prompts.append((body["system"][0]["text"], body["messages"][0]["content"][0]["text"]))
        return _reply("NO")

    judge = conpact_bridge.RetentionJudge(_client(handler))
    with conpact_bridge.bedrock_judge(judge):
        results = evaluation.score([instance], [[{"role": "assistant", "content": "summary"}]],
                                   out_dir=tmp_path, probe=evaluation.Endpoint("m", "u", "k"),
                                   judge=evaluation.Endpoint("unused", api_key="unused"), workers=1)
    assert results[0]["retention"] is False and results[0]["compaction_compliant"] is True
    expected = evaluation.judge_prompts("Use metric units.", [{"role": "assistant", "content": "summary"}])
    assert prompts == [expected]
    assert evaluation._judge.__module__ == "conpact.evaluate"  # put back afterwards
    with conpact_bridge.deferred_judge():
        later = evaluation.score([instance.__class__(**{**instance.to_dict(), "id": "h1-2"})],
                                 [[{"role": "assistant", "content": "summary"}]],
                                 out_dir=tmp_path, probe=evaluation.Endpoint("m", "u", "k"),
                                 judge=evaluation.Endpoint("unused", api_key="unused"), workers=1)
    assert later[0]["retention"] is None and later[0]["compaction_compliant"] is True
    assert evaluation._judge.__module__ == "conpact.evaluate"


# -- calls the server refuses as longer than its window -------------------------------


class _FakeCompletions:
    def __init__(self, error=None):
        self.error, self.calls = error, 0

    def create(self, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="A"), finish_reason="stop")],
            usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=1))


def _fake_client(error=None):
    completions = _FakeCompletions(error)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions)), completions


def _too_long():
    import openai

    response = httpx.Response(400, request=httpx.Request("POST", "http://server/v1/chat/completions"))
    return openai.BadRequestError("Error code: 400 - {'error': {'message': \"Input length (133595) exceeds model's "
                                  "maximum context length (131072).\"}}", response=response, body=None)


def test_a_prompt_the_server_refuses_as_too_long_is_recorded_as_not_served(tmp_path):
    inner, completions = _fake_client(_too_long())
    log = tmp_path / "not_served.jsonl"
    client = conpact_bridge.WindowGuard(inner, log)
    reply = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "long"}])
    assert reply.choices[0].message.content == conpact_bridge.NOT_SERVED
    assert json.loads(log.read_text())["input_tokens"] == 133595


def test_another_refusal_still_stops_the_run(tmp_path):
    import openai

    response = httpx.Response(400, request=httpx.Request("POST", "http://server/v1/chat/completions"))
    inner, _ = _fake_client(openai.BadRequestError("bad model name", response=response, body=None))
    with pytest.raises(openai.BadRequestError):
        conpact_bridge.WindowGuard(inner, tmp_path / "log").chat.completions.create(model="m", messages=[])


def test_nothing_is_asked_on_a_context_built_from_an_unserved_summary(tmp_path):
    inner, completions = _fake_client()
    client = conpact_bridge.WindowGuard(inner, tmp_path / "log")
    reply = client.chat.completions.create(model="m", messages=[
        {"role": "assistant", "content": conpact_bridge.NOT_SERVED + "\n\nkept block"}, {"role": "user", "content": "P?"}])
    assert completions.calls == 0 and reply.choices[0].message.content == conpact_bridge.NOT_SERVED
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "fine"}])
    assert completions.calls == 1


def test_unserved_cases_are_counted_and_never_sent_to_the_judge(tmp_path):
    rows = [{"id": "h1-1", "retention": None, "compacted": [{"role": "assistant", "content": conpact_bridge.NOT_SERVED}],
             "full_with_sc_output": conpact_bridge.NOT_SERVED, "compaction_output": conpact_bridge.NOT_SERVED},
            {"id": "h1-2", "retention": None, "compacted": [{"role": "assistant", "content": "kept"}],
             "full_with_sc_output": "A", "compaction_output": "B"}]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "judge_inputs.jsonl").write_text(
        "".join(json.dumps({"id": row["id"], "sc_text": "rule"}) + "\n" for row in rows))
    asked = []
    judge = conpact_bridge.RetentionJudge(_client(lambda request: asked.append(1) or _reply("YES")))
    judged = conpact_bridge.judge_deferred(tmp_path, judge, lambda s, c: ("S", s), workers=1)
    assert len(asked) == 1 and [row["retention"] for row in judged] == [None, True]
    counts = conpact_bridge.not_served_counts(rows)
    assert counts["compaction"] == 1 and counts["full_with_sc"] == 1 and counts["summary"] == 1


# -- reusing the full-history probes across summary prompts --------------------------


def test_full_history_probes_are_taken_from_an_earlier_run_and_only_summaries_are_probed(monkeypatch):
    evaluate = pytest.importorskip("conpact.evaluate")
    inst = types.SimpleNamespace(id="h1:sc01", messages=[{"role": "user", "content": "with"}],
                                 messages_without_sc=[{"role": "user", "content": "without"}])
    stored = {"h1:sc01": {"full_with_sc_output": "A", "full_with_sc_compliant": True,
                          "full_without_sc_output": "B", "full_without_sc_compliant": False}}
    asked = []
    monkeypatch.setattr(evaluate, "_probe", lambda client, endpoint, i, context, turn: asked.append(context) or ("B", None))
    with conpact_bridge.reuse_full_history(stored):
        assert evaluate._probe(None, None, inst, inst.messages, "P?") == ("A", True)
        assert evaluate._probe(None, None, inst, inst.messages_without_sc, "P?") == ("B", False)
        summary = [{"role": "assistant", "content": "S"}]
        assert evaluate._probe(None, None, inst, summary, "P?") == ("B", None)
    assert asked == [[{"role": "assistant", "content": "S"}]]


def test_a_case_missing_from_the_earlier_run_is_probed_in_full(monkeypatch):
    evaluate = pytest.importorskip("conpact.evaluate")
    inst = types.SimpleNamespace(id="h9:sc01", messages=[{"role": "user", "content": "with"}],
                                 messages_without_sc=[])
    monkeypatch.setattr(evaluate, "_probe", lambda client, endpoint, i, context, turn: ("A", True))
    with conpact_bridge.reuse_full_history({}):
        assert evaluate._probe(None, None, inst, inst.messages, "P?") == ("A", True)


def test_dropping_the_cue_changes_only_the_rule_text_where_it_was_injected():
    pytest.importorskip("conpact")
    from conpact.instance import Instance

    rendered = "For the rest of this session. use metric units."
    fields = {f: None for f in Instance.__dataclass_fields__}
    fields.update(id="h1:sc01", source="hermes", haystack_id="h1", target_length=100000, token_length=9,
                  position="top", repeat=1, strict=False, direct=True, sc_id=1, sc_type="Output",
                  sc_text="Use metric units.", sc_rendered=rendered, sc_message_indices=[1],
                  system_prompt="S", developer_prompt="D",
                  messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": f"{rendered}\nBook a room."},
                            {"role": "assistant", "content": rendered}],
                  messages_without_sc=[{"role": "system", "content": "sys"}, {"role": "user", "content": "Book a room."}],
                  probe="p", correct_answer="a", incorrect_answer="b", swap_seed=1, compliant_letter="A",
                  probe_prompt="P?", post_sc_probe_prompt=f"{rendered}\n\nP?", meta="{}")
    inst = Instance(**fields)
    [plain] = conpact_bridge.drop_cue([inst])
    assert plain.direct is False and plain.sc_rendered == "Use metric units."
    assert plain.messages[1]["content"] == "Use metric units.\nBook a room."
    assert plain.messages[2]["content"] == rendered  # only the injected turn changes
    assert plain.post_sc_probe_prompt == "Use metric units.\n\nP?" and plain.probe_prompt == "P?"
    assert plain.messages_without_sc is inst.messages_without_sc and plain.id == inst.id


def test_only_the_named_full_history_probe_is_reused(monkeypatch):
    evaluate = pytest.importorskip("conpact.evaluate")
    inst = types.SimpleNamespace(id="h1:sc01", messages=[{"role": "user", "content": "with"}],
                                 messages_without_sc=[{"role": "user", "content": "without"}])
    stored = {"h1:sc01": {"full_with_sc_output": "A", "full_with_sc_compliant": True,
                          "full_without_sc_output": "B", "full_without_sc_compliant": False}}
    monkeypatch.setattr(evaluate, "_probe", lambda client, endpoint, i, context, turn: ("fresh", None))
    with conpact_bridge.reuse_full_history(stored, conditions=("full_without_sc",)):
        assert evaluate._probe(None, None, inst, inst.messages, "P?") == ("fresh", None)
        assert evaluate._probe(None, None, inst, inst.messages_without_sc, "P?") == ("B", False)
