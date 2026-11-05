"""Cases built from the benchmark's own, one change each, so that they pair with the published case case by case."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pytest

from studies.memory_probe.variants import (
    CUES,
    RENDER,
    REVISED,
    apply,
    cued,
    moved,
    revised,
    withdrawn,
)


@dataclass(frozen=True)
class _Instance:
    id: str = "hermes_cat_100k:000:sc03"
    sc_id: int = 3
    position: str = "top"
    sc_text: str = "Don't send any messages or emails on my behalf, draft them and let me send them myself."
    sc_rendered: str = "For the rest of this session. don't send any messages or emails on my behalf, draft them and let me send them myself."
    sc_message_indices: tuple = (1,)
    messages: tuple = ()
    messages_without_sc: tuple = ()
    probe: str = "Email Sarah and let her know I'll be late."
    correct_answer: str = "draft"
    incorrect_answer: str = "send"
    compliant_letter: str = "B"
    probe_prompt: str = ('Assume ...\n\nRequest from the user:\n"Email Sarah and let her know I\'ll be late."\n\n'
                         "Two candidate behaviors:\nA) Send it.\nB) Draft it.\n\nOne letter.")
    post_sc_probe_prompt: str = ""


def _instance(**changes) -> _Instance:
    plain = [{"role": "system", "content": "tools"}, {"role": "user", "content": "first"}, {"role": "assistant", "content": "a"},
             {"role": "user", "content": "second"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "third"},
             {"role": "assistant", "content": "c"}, {"role": "user", "content": "fourth"}]
    base = _Instance()
    with_rule = [dict(m) for m in plain]
    with_rule[1]["content"] = f"{base.sc_rendered}\nfirst"
    return dataclasses.replace(base, messages=with_rule, messages_without_sc=plain,
                               post_sc_probe_prompt=f"{base.sc_rendered}\n\n{base.probe_prompt}", **changes)


def test_there_is_a_revision_and_a_cue_for_each_of_the_fifteen_rules() -> None:
    assert sorted(REVISED) == sorted(CUES) == list(range(1, 16))
    assert len(set(REVISED.values())) == len(set(CUES.values())) == 15


def test_a_rule_moved_to_the_middle_or_the_end_is_in_the_package_s_place_and_nowhere_else() -> None:
    middle = moved(_instance(), "middle")
    assert middle.messages[5]["content"] == f"{middle.sc_rendered}\nthird", "the median of four user turns"
    assert middle.messages[1]["content"] == "first" and middle.sc_message_indices == [5]
    assert (middle.position, middle.id) == ("middle", "hermes_cat_100k:000:sc03:middle")
    bottom = moved(_instance(), "bottom")
    assert bottom.messages[7]["content"] == f"{bottom.sc_rendered}\nfourth" and bottom.sc_message_indices == [7]
    assert bottom.probe_prompt == _Instance().probe_prompt and bottom.compliant_letter == "B"
    with pytest.raises(ValueError):
        moved(_instance(), "top")


def test_a_revised_rule_comes_later_flips_the_right_answer_and_keeps_the_old_one_in_the_history() -> None:
    case = revised(_instance())
    assert case.sc_text == REVISED[3]
    assert case.messages[1]["content"].startswith("For the rest of this session. don't send")
    assert case.messages[5]["content"] == f"{RENDER(REVISED[3])}\nthird"
    assert case.sc_message_indices == [5] and case.sc_rendered == RENDER(REVISED[3])
    assert case.compliant_letter == "A"
    assert (case.correct_answer, case.incorrect_answer) == ("send", "draft")
    assert case.post_sc_probe_prompt == f"{RENDER(REVISED[3])}\n\n{case.probe_prompt}"
    assert case.id.endswith(":revised")
    assert case.messages_without_sc == _instance().messages_without_sc


def test_a_withdrawn_rule_keeps_the_letter_of_the_rule_so_obeying_it_reads_as_a_stale_answer() -> None:
    case = withdrawn(_instance())
    assert "no longer applies" in case.messages[5]["content"] and CUES[3] in case.messages[5]["content"]
    assert case.compliant_letter == "B", "the old rule's option: choosing it now is obeying what was withdrawn"
    assert case.messages[5]["content"] == f"{RENDER(case.sc_text)}\nthird"
    assert case.id.endswith(":withdrawn")


def test_a_cue_names_the_subject_inside_the_request_and_changes_nothing_else() -> None:
    case = cued(_instance())
    assert f"be late. Keep in mind what I told you earlier about {CUES[3]}.\"\n\nTwo candidate" in case.probe_prompt
    assert case.post_sc_probe_prompt.endswith(case.probe_prompt)
    assert case.messages == _instance().messages and case.compliant_letter == "B"
    assert case.id.endswith(":cued")


def test_a_cue_gives_the_subject_and_not_what_the_rule_says() -> None:
    for number, cue in CUES.items():
        assert len(cue.split()) <= 7, number
    assert "draft" not in CUES[3] and "metric" not in CUES[11] and "arXiv" not in CUES[10]
    assert "bullet" not in CUES[15] and "web search" not in CUES[9]


def test_a_history_too_short_to_hold_a_later_statement_is_refused() -> None:
    plain = [{"role": "user", "content": "only"}, {"role": "assistant", "content": "a"}]
    case = dataclasses.replace(_Instance(), messages=plain, messages_without_sc=plain)
    with pytest.raises(ValueError, match="later"):
        revised(case)


def test_variants_are_chosen_by_name_and_an_unknown_one_is_refused() -> None:
    instances = [_instance()]
    assert apply(instances, None) == instances
    assert apply(instances, "middle")[0].position == "middle"
    assert apply(instances, "revised")[0].id.endswith(":revised")
    assert apply(instances, "withdrawn")[0].id.endswith(":withdrawn")
    assert apply(instances, "cued")[0].id.endswith(":cued")
    with pytest.raises(ValueError):
        apply(instances, "sideways")
