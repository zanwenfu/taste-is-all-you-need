"""The serving-path check (studies/context_rot/engine_check.py). No model is called."""

from __future__ import annotations

import types

from studies.context_rot import engine_check


def test_probe_messages_are_the_packages_order():
    inst = types.SimpleNamespace(system_prompt="SYS", developer_prompt="DEV")
    messages = engine_check.probe_messages(inst, [{"role": "assistant", "content": "SUMMARY"}], "P?")
    assert [m["role"] for m in messages] == ["system", "developer", "assistant", "user"]
    assert messages[2]["content"] == "SUMMARY" and messages[3]["content"] == "P?"


def test_answers_are_graded_by_both_rules_after_the_engines_channel_label():
    assert engine_check.grade("[final]\nB", "B") == (True, True)
    assert engine_check.grade("A\nB", "B") == (True, None)  # package: last line; paper runner: whole reply
    assert engine_check.grade("[commentary]\nmaybe", "A") == (None, None)
