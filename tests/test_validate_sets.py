"""Gate between a trial and its full run (studies/context_rot/validate_sets.py). No model is called."""

from __future__ import annotations

import json

from studies.context_rot import validate_sets

ARMS = ("user_record", "recent_control", "criteria", "criteria_control")


def _write(folder, rows):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _case(i, **over):
    row = {"id": f"h:000:sc{i:02d}", "compacted": [{"role": "assistant", "content": "summary"}],
           "full_with_sc_output": "A", "full_with_sc_compliant": True, "full_without_sc_output": "B",
           "full_without_sc_compliant": False, "compaction_output": "A", "compaction_compliant": True,
           "upper_bound_output": "A", "upper_bound_compliant": True}
    row.update(over)
    return row


def _trial(root, n=15, **over):
    for prompt in ("pi_mono", "anthropic"):
        _write(root / prompt / "h00-01", [_case(i, **over) for i in range(n)])
        for arm in ARMS:
            _write(root / prompt / f"h00-01-{arm}", [{"id": f"h:000:sc{i:02d}", "compaction_compliant": True,
                                                       "upper_bound_compliant": True} for i in range(n)])


def test_a_clean_trial_passes(tmp_path):
    _trial(tmp_path)
    assert validate_sets.problems(tmp_path, "h00-01", expected=15) == []


def test_missing_cases_empty_summaries_and_refusals_are_problems(tmp_path):
    _trial(tmp_path, n=14)
    assert any("14 of 15" in p for p in validate_sets.problems(tmp_path, "h00-01", expected=15))
    _trial(tmp_path / "b", compacted=[{"role": "assistant", "content": ""}])
    assert any("empty summaries" in p for p in validate_sets.problems(tmp_path / "b", "h00-01", expected=15))
    _trial(tmp_path / "c", compaction_output="[not served: the prompt is longer than the model's window]",
           compaction_compliant=None)
    found = validate_sets.problems(tmp_path / "c", "h00-01", expected=15)
    assert any("not served" in p for p in found) and any("unreadable" in p for p in found)


def test_a_missing_arm_is_a_problem(tmp_path):
    _trial(tmp_path)
    (tmp_path / "anthropic" / "h00-01-criteria" / "results.jsonl").unlink()
    assert any("criteria" in p for p in validate_sets.problems(tmp_path, "h00-01", expected=15))
