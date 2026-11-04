"""Record arms for the rule-loss benchmark (studies/context_rot/record_arms.py). No model is called."""

from __future__ import annotations

import json
import types

from studies.context_rot import record_arms

HISTORY = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "For the rest of this session, use metric units. Book a room."},
    {"role": "assistant", "content": "Searching hotels... found three."},
    {"role": "user", "content": "Pick the cheapest."},
    {"role": "assistant", "content": "Booked the cheapest room."},
]


def words(text: str) -> int:
    return len(text.split())


def test_the_user_record_keeps_every_user_turn_verbatim_in_order_and_nothing_else():
    block = record_arms.user_record(HISTORY)
    assert block.index("use metric units. Book a room.") < block.index("Pick the cheapest.")
    assert "Searching hotels" not in block and "You are an agent." not in block


def test_the_recent_control_takes_the_newest_text_up_to_the_same_size():
    block = record_arms.recent_block(HISTORY, budget=6, count=words)
    assert words(block) <= 6 and "Booked the cheapest room." in block
    assert "metric" not in block


def test_an_arm_returns_one_assistant_message_holding_the_summary_then_the_block():
    context = record_arms.arm_context([{"role": "assistant", "content": "SUMMARY"}], "BLOCK", "Kept from the record")
    assert len(context) == 1 and context[0]["role"] == "assistant"
    text = context[0]["content"]
    assert text.startswith("SUMMARY") and text.index("Kept from the record") < text.index("BLOCK")


def test_paired_scoring_reuses_the_baseline_full_history_probes_and_reprobes_only_the_new_context(tmp_path):
    instance = types.SimpleNamespace(id="h1-1", source="hermes", haystack_id="h1", target_length=100000,
                                     token_length=9, position="top", repeat=1, strict=False, direct=True, sc_id=1,
                                     sc_type="Action", compliant_letter="A", sc_text="Use metric units.",
                                     probe_prompt="P?", post_sc_probe_prompt="P with rule?")
    baseline = {"id": "h1-1", "full_with_sc_output": "A", "full_with_sc_compliant": True,
                "full_without_sc_output": "B", "full_without_sc_compliant": False}
    asked = []

    def probe(inst, context, user_turn):
        asked.append((context[0]["content"], user_turn))
        return "A", True

    rows = record_arms.score_paired([instance], {"h1-1": baseline}, {"h1-1": [{"role": "assistant", "content": "C"}]},
                                    probe, tmp_path, workers=1)
    assert asked == [("C", "P?"), ("C", "P with rule?")]
    row = rows[0]
    assert row["full_with_sc_compliant"] is True and row["full_without_sc_compliant"] is False
    assert row["compaction_compliant"] is True and row["upper_bound_compliant"] is True
    assert row["retention"] is None and row["compacted"] == [{"role": "assistant", "content": "C"}]
    written = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert written[0]["id"] == "h1-1"
    inputs = [json.loads(line) for line in (tmp_path / "judge_inputs.jsonl").read_text().splitlines()]
    assert inputs == [{"id": "h1-1", "sc_text": "Use metric units."}]


def test_stored_records_are_reused_so_arm_and_control_share_one_reading(tmp_path):
    stored = tmp_path / "records.json"
    stored.write_text(json.dumps({"records": {"h1:sc01": [{"message": 1, "text": "Use metric units."}]}, "reads": []}))
    assert record_arms.stored_records(stored) == {"h1:sc01": [{"message": 1, "text": "Use metric units."}]}
