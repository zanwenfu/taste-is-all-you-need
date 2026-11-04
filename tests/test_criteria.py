"""Criterion records for the rule-loss benchmark (studies/context_rot/criteria.py). No model is called."""

from __future__ import annotations

import json

from studies.context_rot import criteria

HISTORY = [{"role": "system", "content": "You are an agent."},
           {"role": "user", "content": "For the rest of this session, use metric units. Book a room in Oslo."},
           {"role": "assistant", "content": "Found three hotels."},
           {"role": "user", "content": "Pick the cheapest. Always answer in one sentence from now on."},
           {"role": "assistant", "content": "Booked."}]


def test_the_reader_sees_one_user_message_and_nothing_after_it():
    prompt = criteria.reader_prompt(HISTORY[1]["content"])
    assert prompt[0]["role"] == "system" and prompt[-1]["role"] == "user"
    assert HISTORY[1]["content"] in prompt[-1]["content"]
    assert "Pick the cheapest" not in json.dumps(prompt)


def test_only_quotes_found_word_for_word_in_the_message_are_kept():
    reply = json.dumps({"standing": ["use metric units", "always be polite", "  Book a room in Oslo.  "]})
    kept, dropped = criteria.parse_reply(reply, HISTORY[1]["content"])
    assert kept == ["use metric units", "Book a room in Oslo."] and dropped == ["always be polite"]
    assert criteria.parse_reply("not json", HISTORY[1]["content"]) == ([], [])
    fenced = '```json\n{"standing": ["use metric units"]}\n```'
    assert criteria.parse_reply(fenced, HISTORY[1]["content"])[0] == ["use metric units"]


def test_records_point_at_their_message_and_each_message_is_read_once():
    asked = []

    def read(text):
        asked.append(text)
        return {"For the rest of this session, use metric units. Book a room in Oslo.": ["use metric units"],
                "Pick the cheapest. Always answer in one sentence from now on.":
                    ["Always answer in one sentence from now on."]}[text]

    cache = {}
    records = criteria.build_records(HISTORY, read, cache)
    assert records == [{"message": 1, "text": "use metric units"},
                       {"message": 3, "text": "Always answer in one sentence from now on."}]
    criteria.build_records(HISTORY, read, cache)
    assert len(asked) == 2  # the second case with the same messages asks nothing


def test_the_block_lists_the_records_verbatim_in_order():
    block = criteria.records_block([{"message": 1, "text": "use metric units"},
                                    {"message": 3, "text": "Always answer in one sentence."}])
    assert block.splitlines() == ["- use metric units", "- Always answer in one sentence."]
    assert criteria.records_block([]) == "(none)"


def test_every_distinct_user_message_is_read_once_across_cases():
    import types

    first = dict(HISTORY[1], content="Rule A. " + HISTORY[1]["content"])
    cases = [types.SimpleNamespace(id="h1:sc01", messages=HISTORY),
             types.SimpleNamespace(id="h1:sc02", messages=[HISTORY[0], first, *HISTORY[2:]])]
    asked = []

    def read(text):
        asked.append(text)
        return ["Rule A."] if text.startswith("Rule A.") else []

    records = criteria.records_for(cases, read, workers=2)
    assert sorted(asked) == sorted({HISTORY[1]["content"], first["content"], HISTORY[3]["content"]})
    assert records == {"h1:sc01": [], "h1:sc02": [{"message": 1, "text": "Rule A."}]}


def test_repeated_rounds_probe_an_added_block_after_the_last_summary():
    import types

    from studies.context_rot import repeated

    inst = types.SimpleNamespace(id="h1:sc01", haystack_id="h1", sc_id=1, sc_type="Action", sc_text="r",
                                 messages=HISTORY, probe_prompt="P?", post_sc_probe_prompt="r P?",
                                 compliant_letter="A")
    seen = []
    row = repeated.run_one(inst, k=2, option="anthropic", sizes=[len(m["content"].split()) for m in HISTORY],
                           summarize=lambda messages: "S", judge=lambda sc, summary: False,
                           probe=lambda i, context, turn: (seen.append(context[0]["content"]) or ("B", False)),
                           count=lambda text: len(text.split()),
                           package_prompt=lambda option, messages: [{"role": "user", "content": "x"}],
                           blocks={"criteria": ("- use metric units", "Standing:")})
    assert "S\n\nStanding:\n- use metric units" in seen and row["criteria_compliant"] is False


def test_an_instruction_repeated_word_for_word_is_one_record_at_its_first_message():
    history = [{"role": "user", "content": "Always answer in English. First task."},
               {"role": "assistant", "content": "Done."},
               {"role": "user", "content": "Always answer in English. Second task."}]
    records = criteria.build_records(history, lambda text: ["Always answer in English."], {})
    assert records == [{"message": 0, "text": "Always answer in English."}]
