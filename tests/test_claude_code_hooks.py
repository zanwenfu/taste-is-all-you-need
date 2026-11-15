"""Tests for the study's Claude Code hooks and the settings that wire them: hook JSON shape, the 10,000-character cap, fake readers and clients."""

import io
import json
import shlex
from pathlib import Path

import pytest

from studies.claude_code import driver, settings
from studies.claude_code.hooks import compaction_log, mem0_hook, rule_list_hook
from studies.claude_code.hooks.common import CONTEXT_CAP, fit
from studies.claude_code.hooks.mem0_hook import FakeMemory
from studies.claude_code.hooks.rule_list_hook import stub_reader

STUB = "studies.claude_code.hooks.rule_list_hook:stub_reader"
FIRST_PROMPT = f"{driver.PYTHON_B} {driver.tag_rule()}\n\nIn t01_fizz/, write fizz.py and run it."


def _context(output: str, event_name: str) -> str:
    """The added context, after checking the whole output is exactly the documented shape."""
    data = json.loads(output)
    assert list(data) == ["hookSpecificOutput"]
    assert set(data["hookSpecificOutput"]) == {"hookEventName", "additionalContext"}
    assert data["hookSpecificOutput"]["hookEventName"] == event_name
    return data["hookSpecificOutput"]["additionalContext"]


# The rule list

def test_stub_reader_keeps_the_cue_sentences_word_for_word():
    found = stub_reader(FIRST_PROMPT)
    assert found == [driver.PYTHON_B, driver.tag_rule().split(" so read it now. ")[1]]
    assert stub_reader(driver.LS_ONE) == [driver.LS_ONE]
    assert stub_reader("Write fizz.py and run it. Then check the folder.") == []


def test_prompt_mode_lists_new_rules_once_and_returns_them_all(tmp_path):
    rules, log = tmp_path / "project" / ".claude" / "standing_rules.md", tmp_path / "rule_list.jsonl"
    first = rule_list_hook.on_prompt({"prompt": FIRST_PROMPT, "session_id": "s1"}, rules_path=rules, reader=stub_reader, log=log)
    text = _context(first, "UserPromptSubmit")
    assert text.startswith(rule_list_hook.HEADER) and driver.PYTHON_B in text
    again = rule_list_hook.on_prompt({"prompt": FIRST_PROMPT}, rules_path=rules, reader=stub_reader, log=log)
    later = rule_list_hook.on_prompt({"prompt": driver.LS_ONE}, rules_path=rules, reader=stub_reader, log=log)
    assert _context(again, "UserPromptSubmit") == text
    assert rule_list_hook.listed(rules) == [*stub_reader(FIRST_PROMPT), driver.LS_ONE]
    assert _context(later, "UserPromptSubmit").index(driver.PYTHON_B) < _context(later, "UserPromptSubmit").index(driver.LS_ONE)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [len(record["added"]) for record in records] == [2, 0, 1]


def test_no_rules_yet_means_no_output(tmp_path):
    assert rule_list_hook.on_prompt({"prompt": "Write a script."}, rules_path=tmp_path / "r.md", reader=stub_reader) is None
    assert rule_list_hook.on_compact({"source": "compact"}, rules_path=tmp_path / "r.md") is None


def test_compact_mode_gives_the_list_back_after_compaction_only(tmp_path):
    rules = tmp_path / "r.md"
    rule_list_hook.add(rules, [driver.PYTHON_B, "Use ls -1."])
    text = _context(rule_list_hook.on_compact({"hook_event_name": "SessionStart", "source": "compact"}, rules_path=rules), "SessionStart")
    assert driver.PYTHON_B in text and "Use ls -1." in text
    assert rule_list_hook.on_compact({"source": "startup"}, rules_path=rules) is None


def test_a_failing_reader_is_logged_and_the_list_so_far_still_goes_out(tmp_path):
    rules, log = tmp_path / "r.md", tmp_path / "log.jsonl"
    rule_list_hook.add(rules, [driver.PYTHON_B])

    def broken(_):
        raise TimeoutError("model did not answer")

    output = rule_list_hook.on_prompt({"prompt": "anything"}, rules_path=rules, reader=broken, log=log)
    assert driver.PYTHON_B in _context(output, "UserPromptSubmit")
    assert "TimeoutError" in json.loads(log.read_text())["error"]


def test_main_reads_stdin_and_loads_the_reader_named_in_the_environment(tmp_path):
    out = io.StringIO()
    code = rule_list_hook.main(["prompt", "--rules", str(tmp_path / "r.md")], stdin=io.StringIO(json.dumps({"prompt": driver.LS_ONE})),
                               stdout=out, environ={rule_list_hook.READER_ENV: STUB})
    assert code == 0 and driver.LS_ONE in _context(out.getvalue(), "UserPromptSubmit")


def test_main_without_a_reader_says_so_and_never_blocks_the_prompt(tmp_path):
    out, log = io.StringIO(), tmp_path / "log.jsonl"
    code = rule_list_hook.main(["prompt", "--rules", str(tmp_path / "r.md"), "--log", str(log)], stdin=io.StringIO("{}"), stdout=out,
                               environ={})
    assert code == 1 and code != 2 and out.getvalue() == ""
    assert "error" in json.loads(log.read_text())


def test_the_rule_list_stays_under_the_cap_and_keeps_the_newest(tmp_path):
    rules = tmp_path / "r.md"
    rule_list_hook.add(rules, [f"Rule {n}: " + "x" * 100 for n in range(300)])
    text = _context(rule_list_hook.on_compact({}, rules_path=rules), "SessionStart")
    assert len(text) < CONTEXT_CAP
    assert "Rule 299:" in text and "Rule 0:" not in text and "left out" in text
    assert text.index("Rule 298:") < text.index("Rule 299:")


def test_one_rule_longer_than_the_cap_is_cut_to_fit(tmp_path):
    rules = tmp_path / "r.md"
    rule_list_hook.add(rules, ["y" * 25_000])
    assert len(_context(rule_list_hook.on_compact({}, rules_path=rules), "SessionStart")) < CONTEXT_CAP


def test_rule_list_hook_source_is_short():
    assert len(Path(rule_list_hook.__file__).read_text()) < 10_000


def test_fit_shows_kept_entries_sorted_by_key():
    assert fit("H", [(2, "b"), (1, "a")], limit=1000) == "H\n- a\n- b"


# Mem0

def test_fake_memory_finds_by_shared_words_and_keeps_a_file(tmp_path):
    memory = FakeMemory(tmp_path / "fake.json")
    memory.add([{"role": "user", "content": "Pass python the -B flag."}, {"role": "assistant", "content": "Will do."}], user=True)
    assert [hit["memory"] for hit in memory.search("run python here", 10)] == ["Pass python the -B flag."]
    assert [hit["memory"] for hit in FakeMemory(tmp_path / "fake.json").search("python", 10)] == ["Pass python the -B flag."]


def test_mem0_prompt_then_stop_then_prompt(tmp_path):
    memory, state, log = FakeMemory(), tmp_path / "state", tmp_path / "mem0.jsonl"
    assert mem0_hook.on_prompt({"prompt": FIRST_PROMPT}, client=memory, state=state, log=log) is None
    mem0_hook.on_stop({"last_assistant_message": "Tag is abc. Done."}, client=memory, state=state, log=log)
    assert memory.adds == [([{"role": "user", "content": FIRST_PROMPT}, {"role": "assistant", "content": "Tag is abc. Done."}], True)]
    assert not (state / mem0_hook.PENDING).exists()
    text = _context(mem0_hook.on_prompt({"prompt": "Now run the python tests."}, client=memory, state=state, log=log), "UserPromptSubmit")
    assert text.startswith(mem0_hook.SEARCH_HEADER) and "-B flag" in text
    assert [json.loads(line)["mode"] for line in log.read_text().splitlines()] == ["prompt", "stop", "prompt"]


def test_mem0_hits_are_shown_oldest_first_and_capped(tmp_path):
    class Ranked:
        def search(self, query, limit):  # best first: the newest memories rank highest here
            return [{"memory": f"memory {n} " + "z" * 200, "created_at": f"2026-11-15T22:{n:02d}"} for n in reversed(range(60))]

        def add(self, messages, user=False):
            pass

    text = _context(mem0_hook.on_prompt({"prompt": "q"}, client=Ranked(), state=tmp_path), "UserPromptSubmit")
    assert len(text) < CONTEXT_CAP and "memory 59 " in text and "memory 0 " not in text
    assert text.index("memory 58 ") < text.index("memory 59 ")


def test_mem0_stop_reads_the_transcript_when_given_no_last_message(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("\n".join(json.dumps(line) for line in [
        {"type": "user", "message": {"role": "user", "content": "hi"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "First reply."}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "Bash", "input": {}}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Last reply."}]}},
    ]))
    memory = FakeMemory()
    mem0_hook.on_stop({"transcript_path": str(transcript)}, client=memory, state=tmp_path)
    assert memory.adds == [([{"role": "assistant", "content": "Last reply."}], False)]


def test_mem0_search_failure_is_logged_and_adds_nothing(tmp_path):
    class Down:
        def search(self, query, limit):
            raise ConnectionError("sidecar gone")

        def add(self, messages, user=False):
            pass

    log = tmp_path / "log.jsonl"
    assert mem0_hook.on_prompt({"prompt": "q"}, client=Down(), state=tmp_path, log=log) is None
    assert "ConnectionError" in json.loads(log.read_text())["error"]


def test_mem0_main_across_processes_with_the_file_fake(tmp_path, monkeypatch):
    monkeypatch.setenv(mem0_hook.FAKE_ENV, str(tmp_path / "fake.json"))
    environ = {mem0_hook.CLIENT_ENV: "studies.claude_code.hooks.mem0_hook:fake_from_env"}
    state = ["--state", str(tmp_path / "state")]
    assert mem0_hook.main(["prompt", *state], stdin=io.StringIO(json.dumps({"prompt": "Pass python -B."})), stdout=io.StringIO(),
                          environ=environ) == 0
    assert mem0_hook.main(["stop", *state], stdin=io.StringIO(json.dumps({"last_assistant_message": "OK."})), stdout=io.StringIO(),
                          environ=environ) == 0
    out = io.StringIO()
    assert mem0_hook.main(["prompt", *state], stdin=io.StringIO(json.dumps({"prompt": "run python"})), stdout=out, environ=environ) == 0
    assert "Pass python -B." in _context(out.getvalue(), "UserPromptSubmit")


def test_mem0_main_without_a_client_says_so(tmp_path):
    out = io.StringIO()
    assert mem0_hook.main(["prompt", "--state", str(tmp_path)], stdin=io.StringIO("{}"), stdout=out, environ={}) == 1
    assert out.getvalue() == ""


def test_mem0_header_is_the_long_run_studys():
    source = (Path(__file__).resolve().parents[1] / "studies" / "long_run" / "session.py").read_text()
    first, second = mem0_hook.SEARCH_HEADER.split("stands. ")
    assert f"{first}stands. " in source and second in source


# Compaction log

def test_compaction_log_writes_one_line_and_prints_nothing(tmp_path, capsys):
    log = tmp_path / "compaction.jsonl"
    event = {"hook_event_name": "PreCompact", "trigger": "auto", "session_id": "s1"}
    assert compaction_log.main(["--event", "PreCompact", "--log", str(log)], stdin=io.StringIO(json.dumps(event))) == 0
    assert compaction_log.main(["--event", "PostCompact", "--log", str(log)], stdin=io.StringIO("not json")) == 0
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(line["event"], line["trigger"]) for line in lines] == [("PreCompact", "auto"), ("PostCompact", None)]
    assert capsys.readouterr().out == ""


def test_compaction_log_never_fails_the_compaction(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("")
    assert compaction_log.main(["--log", str(blocked / "under_a_file.jsonl")], stdin=io.StringIO("{}")) == 0


# Settings

def _settings(tmp_path, arm):
    path = settings.write_settings(tmp_path / "project", arm, state=tmp_path / "hooks", python="/usr/bin/python3")
    return json.loads(path.read_text())["hooks"]


def test_every_arm_logs_compaction_and_only_the_memory_arms_add_hooks(tmp_path):
    expected = {"floor": set(), "native": set(), "restated": set(), "rules": {"UserPromptSubmit", "SessionStart"},
                "mem0": {"UserPromptSubmit", "Stop"}}
    for arm, extra in expected.items():
        hooks = _settings(tmp_path / arm, arm)
        assert set(hooks) == {"PreCompact", "PostCompact"} | extra, arm


def test_rules_arm_reinjects_after_compaction_only(tmp_path):
    hooks = _settings(tmp_path, "rules")
    assert hooks["SessionStart"][0]["matcher"] == "compact"
    assert "matcher" not in hooks["UserPromptSubmit"][0]
    assert hooks["UserPromptSubmit"][0]["hooks"][0]["timeout"] > 30


def test_hook_commands_run_the_studys_scripts_without_bytecode(tmp_path):
    for arm in ("rules", "mem0"):
        for entries in _settings(tmp_path / arm, arm).values():
            for hook in entries[0]["hooks"]:
                argv = shlex.split(hook["command"])
                assert hook["type"] == "command" and argv[:2] == ["/usr/bin/python3", "-B"]
                assert Path(argv[2]).exists() and Path(argv[2]).parent == settings.HOOKS


def test_rules_file_is_in_the_project(tmp_path):
    command = _settings(tmp_path, "rules")["UserPromptSubmit"][0]["hooks"][0]["command"]
    assert str(tmp_path / "project" / settings.RULES_FILE) in shlex.split(command)


def test_unknown_arm_is_refused(tmp_path):
    with pytest.raises(ValueError):
        settings.hooks_block("other", project=tmp_path, state=tmp_path, python="python3")
