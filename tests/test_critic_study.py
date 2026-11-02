"""The record both critics see, which runs are in the sample, and how each critic is scored."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "critic_study", Path(__file__).resolve().parents[1] / "scripts" / "critic_study.py")
critic_study = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(critic_study)


def test_a_record_keeps_the_latest_work_when_it_must_be_cut():
    plans = [{"generation": 1, "rationale": "first " * 40, "assignments": [], "assessment": []},
             {"generation": 2, "rationale": "second", "assignments": [{"assignment_id": "a", "contract": {"task": "do"}}],
              "assessment": [{"verdict": "not_met", "evidence": "still failing"}]}]
    text = critic_study.record_text(plans, "the report", limit=120)
    assert text.startswith("(earlier record cut)") and "the report" in text and "still failing" in text
    assert "Plan 1" in critic_study.record_text(plans, "the report")


def test_only_runs_whose_files_show_what_the_verifier_saw_are_sampled():
    rows = [{"trial": "kept", "kind": "final", "reward": 1.0, "state_reward": 1.0},
            {"trial": "live-process", "kind": "final", "reward": 1.0, "state_reward": 0.0},
            {"trial": "no-reward", "kind": "final", "reward": None, "state_reward": 0.0},
            {"trial": "ungraded", "kind": "final", "reward": 0.0, "state_reward": None},
            {"trial": "kept", "kind": "checkpoint", "reward": 1.0, "state_reward": 1.0}]
    assert [row["trial"] for row in critic_study.sample(rows)] == ["kept"]


def test_a_critic_is_scored_in_both_directions_and_a_non_answer_is_counted():
    rows = [{"trial": "p1", "reward": 1.0}, {"trial": "p2", "reward": 1.0},
            {"trial": "f1", "reward": 0.0}, {"trial": "f2", "reward": 0.0}]
    verdicts = [{"trial": "p1", "verdict": "done"}, {"trial": "p2", "verdict": "not_done"},
                {"trial": "f1", "verdict": "done"}, {"trial": "f2", "verdict": None}]
    summary = critic_study.summarize(rows, verdicts)
    assert summary["judged"] == 3 and summary["unjudged"] == 1
    assert summary["false_reject"] == 1 and summary["false_reject_rate"] == 0.5
    assert summary["false_accept"] == 1 and summary["false_accept_rate"] == 1.0


def test_a_reply_that_is_not_a_verdict_is_not_read_as_one():
    assert critic_study.parse_verdict('{"verdict": "done", "confidence": 0.8}') == ("done", 0.8)
    assert critic_study.parse_verdict('{"verdict": "maybe"}') == (None, None)
    assert critic_study.parse_verdict("not json") == (None, None)
