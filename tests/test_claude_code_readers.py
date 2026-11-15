"""Tests for studies.claude_code.readers: the reader's command line, where it runs, and how its reply becomes the rule list (no real Claude call)."""

import json
import subprocess

import pytest

from studies.claude_code import readers

MESSAGE = "For the rest of this session, whenever you run Python, pass it the -B flag, as in python3 -B script.py. Now fix the parser."


def _fake(reply, code=0, seen=None, is_error=False):
    def run(argv, *, input, capture_output, text, timeout, check, cwd, env):
        if seen is not None:
            seen.update(argv=argv, input=input, timeout=timeout, cwd=cwd, env=env)
        body = json.dumps({"type": "result", "result": reply, "is_error": is_error})
        return subprocess.CompletedProcess(argv, code, stdout=body, stderr="boom" if code else "")
    return run


def test_the_reader_loads_nothing_of_the_session(tmp_path, monkeypatch):
    monkeypatch.setenv(readers.READER_HOME_ENV, str(tmp_path / "reader"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "session-config"))
    seen = {}
    readers.claude_reader(MESSAGE, run=_fake('{"standing": []}', seen=seen))
    argv = seen["argv"]
    assert "--bare" not in argv  # it refuses the subscription token
    assert argv[argv.index("--model") + 1] == readers.READER_MODEL
    assert json.loads(argv[argv.index("--settings") + 1]) == {"disableAllHooks": True}
    assert "Bash" in argv[argv.index("--disallowedTools") + 1]
    assert seen["cwd"] == str(tmp_path / "reader" / "work")
    assert seen["env"]["CLAUDE_CONFIG_DIR"] == str(tmp_path / "reader" / "config")  # not the session's own config folder
    assert seen["env"]["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1" and seen["env"]["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert MESSAGE in seen["input"] and seen["timeout"] < 120  # inside the hook's own timeout


def test_only_quotes_found_word_for_word_are_kept(tmp_path, monkeypatch):
    monkeypatch.setenv(readers.READER_HOME_ENV, str(tmp_path / "reader"))
    quote = "For the rest of this session, whenever you run Python, pass it the -B flag, as in python3 -B script.py."
    reply = json.dumps({"standing": [quote, "Always use pytest."]})  # the reply shape the study's reader prompt asks for
    assert readers.claude_reader(MESSAGE, run=_fake(reply)) == [quote]


def test_a_failed_or_refused_reader_call_raises_so_the_hook_logs_it(tmp_path, monkeypatch):
    monkeypatch.setenv(readers.READER_HOME_ENV, str(tmp_path / "reader"))
    with pytest.raises(RuntimeError, match="exited 1"):
        readers.claude_reader(MESSAGE, run=_fake("", code=1))
    with pytest.raises(RuntimeError, match="Not logged in"):
        readers.claude_reader(MESSAGE, run=_fake("Not logged in", is_error=True))
