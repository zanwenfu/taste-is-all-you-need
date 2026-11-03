"""Repeated compaction on the rule-loss benchmark (studies/context_rot/repeated.py). No model is called."""

from __future__ import annotations

import json
import types

import pytest

from studies.context_rot import repeated

PI_SYSTEM = "You are a context summarization assistant."
TRANSCRIPT = [{"role": "user", "content": "For the rest of this session, use metric units. Book a room."},
              {"role": "assistant", "content": "a " * 30},
              {"role": "user", "content": "Pick the cheapest."},
              {"role": "assistant", "content": "b " * 30},
              {"role": "tool", "content": "c " * 30},
              {"role": "assistant", "content": "Booked."}]


def words(text: str) -> int:
    return len(text.split())


def test_parts_are_consecutive_whole_messages_of_about_equal_size():
    sizes = [words(m["content"]) for m in TRANSCRIPT]
    for k in (1, 2, 3):
        parts = repeated.split_parts(TRANSCRIPT, sizes, k)
        assert len(parts) == k and all(parts)
        assert [m for part in parts for m in part] == TRANSCRIPT
    two = repeated.split_parts(TRANSCRIPT, sizes, 2)
    assert two[0][-1] == TRANSCRIPT[2]  # 45 of 106 words: the message boundary nearest the half-way point
    with pytest.raises(ValueError):
        repeated.split_parts(TRANSCRIPT[:2], sizes[:2], 3)


def _package_prompt(option, messages):
    if option == "pi_mono":
        return [{"role": "system", "content": PI_SYSTEM}, *messages, {"role": "user", "content": "SUMMARIZE"}]
    return [{"role": "user", "content": "<transcript>" + "|".join(m["content"] for m in messages)},
            {"role": "user", "content": "ANTHROPIC SUMMARIZE"}]


def test_the_first_round_is_the_package_prompt_as_published():
    part = TRANSCRIPT[:2]
    for option in ("pi_mono", "anthropic"):
        assert repeated.round_prompt(option, None, part, _package_prompt) == _package_prompt(option, part)


def test_a_later_pi_round_is_pis_own_update_step():
    prompt = repeated.round_prompt("pi_mono", "EARLIER SUMMARY", TRANSCRIPT[2:4], _package_prompt)
    assert prompt[0] == {"role": "system", "content": PI_SYSTEM}
    assert prompt[1:3] == TRANSCRIPT[2:4]
    last = prompt[-1]["content"]
    assert last.startswith("<previous-summary>\nEARLIER SUMMARY\n</previous-summary>\n\n")
    assert last.endswith(repeated.PI_UPDATE_PROMPT) and "PRESERVE all existing information" in last


def test_a_later_anthropic_round_summarizes_the_earlier_summary_with_the_new_part():
    prompt = repeated.round_prompt("anthropic", "EARLIER SUMMARY", TRANSCRIPT[2:4], _package_prompt)
    assert prompt == _package_prompt("anthropic", [{"role": "assistant", "content": "EARLIER SUMMARY"},
                                                   *TRANSCRIPT[2:4]])


def test_each_round_carries_the_summary_forward_and_every_round_is_judged(tmp_path):
    inst = types.SimpleNamespace(id="h1:sc01", haystack_id="h1", sc_id=1, sc_type="Action",
                                 sc_text="Use metric units.", messages=TRANSCRIPT,
                                 probe_prompt="P?", post_sc_probe_prompt="Rule. P?", compliant_letter="A")
    seen = []

    def summarize(messages):
        seen.append(messages)
        return f"S{len(seen)}"

    def judge(sc_text, summary):
        return summary == "S1"

    probed = []

    def probe(inst, context, turn):
        probed.append((context[0]["content"], turn))
        return "A", True

    row = repeated.run_one(inst, k=3, option="pi_mono", sizes=[words(m["content"]) for m in TRANSCRIPT],
                           summarize=summarize, judge=judge, probe=probe, count=words, package_prompt=_package_prompt)
    assert row["summaries"] == ["S1", "S2", "S3"] and row["retention"] == [True, False, False]
    assert "<previous-summary>\nS1\n</previous-summary>" in seen[1][-1]["content"]
    assert "<previous-summary>\nS2\n</previous-summary>" in seen[2][-1]["content"]
    contexts = {turn_context for turn_context, _ in probed}
    assert "S3" in contexts  # summary only
    record = next(c for c in contexts if "own messages" in c)
    assert record.startswith("S3") and "use metric units. Book a room." in record and "Pick the cheapest." in record
    control = next(c for c in contexts if "most recent part" in c)
    assert "Booked." in control and "metric" not in control
    assert {turn for _, turn in probed} == {"P?", "Rule. P?"} and len(probed) == 6
    for arm in ("compaction", "user_record", "recent_control"):
        assert row[f"{arm}_compliant"] is True and row[f"{arm}_upper_compliant"] is True
    assert row["k"] == 3 and row["part_tokens"] and len(row["part_tokens"]) == 3
    json.dumps(row)


def test_finished_cases_are_not_run_again(tmp_path):
    (tmp_path / "results.jsonl").write_text(json.dumps({"id": "h1:sc01", "k": 2}) + "\n")
    assert repeated.done_keys(tmp_path) == {("h1:sc01", 2)}
