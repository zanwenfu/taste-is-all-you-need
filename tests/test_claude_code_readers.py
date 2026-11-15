"""Tests for studies.claude_code.readers: the reader's command line and how its reply becomes the rule list (no real Claude call)."""

import json
import subprocess

import pytest

from studies.claude_code import readers

MESSAGE = "For the rest of this session, whenever you run Python, pass it the -B flag, as in python3 -B script.py. Now fix the parser."


def _fake(reply, code=0, seen=None):
    def run(argv, *, input, capture_output, text, timeout, check):
        if seen is not None:
            seen.update(argv=argv, input=input, timeout=timeout)
        return subprocess.CompletedProcess(argv, code, stdout=json.dumps({"type": "result", "result": reply}), stderr="boom" if code else "")
    return run


def test_the_reader_runs_without_hooks_on_a_small_model_and_no_tools():
    seen = {}
    readers.claude_reader(MESSAGE, run=_fake("[]", seen=seen))
    argv = seen["argv"]
    assert "--bare" in argv and argv[argv.index("--model") + 1] == readers.READER_MODEL
    assert "Bash" in argv[argv.index("--disallowedTools") + 1] and "--system-prompt" in argv
    assert MESSAGE in seen["input"] and seen["timeout"] < 120  # inside the hook's own timeout


def test_only_quotes_found_word_for_word_are_kept():
    quote = "For the rest of this session, whenever you run Python, pass it the -B flag, as in python3 -B script.py."
    reply = json.dumps({"standing": [quote, "Always use pytest."]})  # the reply shape the study's reader prompt asks for
    assert readers.claude_reader(MESSAGE, run=_fake(reply)) == [quote]


def test_a_failed_reader_call_raises_so_the_hook_logs_it():
    with pytest.raises(RuntimeError, match="exited 1"):
        readers.claude_reader(MESSAGE, run=_fake("", code=1))
