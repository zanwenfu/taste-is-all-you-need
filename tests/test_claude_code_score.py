"""Tests for studies.claude_code.score: Claude Code's Bash calls read from small synthetic stream-json sessions."""

import json

import pytest

from studies.claude_code.score import (
    LS_LA,
    LS_ONE,
    LS_OTHER,
    ls_form,
    program,
    python_options,
    score_session,
    split_commands,
    summarize,
    words,
)

TAG = "abc123def456"


def at(minute: int, second: int = 0) -> str:
    return f"2026-11-15T22:{minute:02d}:{second:02d}.000+00:00"


def marker(turn, kind="task", tells=(), time=None):
    return {"type": "study_turn", "turn": turn, "kind": kind, "tells": list(tells), "time": time or at(0)}


def bash(call_id, command, parent=None, context=None):
    message = {"id": f"m-{call_id}", "role": "assistant",
               "content": [{"type": "tool_use", "id": call_id, "name": "Bash", "input": {"command": command, "description": "x"}}]}
    if context:
        message["usage"] = {"input_tokens": context, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": 5}
    return {"type": "assistant", "message": message, "parent_tool_use_id": parent, "session_id": "s1"}


def output(call_id, text):
    return {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": text}]},
            "parent_tool_use_id": None, "session_id": "s1"}


def said(text):
    return {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}, "parent_tool_use_id": None}


COMPACT = {"type": "system", "subtype": "compact_boundary", "session_id": "s1", "compact_metadata": {"trigger": "auto", "pre_tokens": 90000}}


def result(cost=0.1):
    return {"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": cost, "num_turns": 4, "session_id": "s1",
            "permission_denials": [], "usage": {"input_tokens": 10, "cache_read_input_tokens": 80, "cache_creation_input_tokens": 10,
                                                "output_tokens": 50}}


def session():
    """Seven turns: the rules told with the first task, the ls override before task 4, two compactions, a reversion in task 5."""
    return [
        marker("task_01", tells=("python_B", "session_tag"), time=at(0)), {"type": "system", "subtype": "init", "session_id": "s1"},
        bash("b1", "cat .session_tag"), output("b1", TAG),
        bash("b2", "cd t01 && python3 -B a.py; ls -la"), output("b2", "ok"), result(0.1),
        marker("task_02", time=at(1)), bash("b3", "python3 t02/b.py"), bash("b4", f"echo {TAG}"), output("b4", TAG), result(0.2),
        marker("task_03", time=at(2)), COMPACT, bash("b5", "ls -la t03"), said("Done."), result(0.3),
        marker("rule_ls", kind="rule", tells=("ls_one",), time=at(3)), said("Understood."), result(0.4),
        marker("task_04", time=at(4)), bash("b6", "ls -1 t04 && python3 -B -m unittest"), bash("b7", f"echo {TAG}"), result(0.5),
        marker("task_05", time=at(5)), COMPACT, bash("b8", "ls -la t05", context=42000), bash("b9", "echo $(cat notes/tag.txt)"),
        output("b9", TAG), result(0.6),
        marker("task_06", time=at(6)), bash("b10", "ls t06"), result(0.7),
    ]


LOG = [{"event": "PreCompact", "trigger": "auto", "time": at(2, 30)}, {"event": "PostCompact", "trigger": "auto", "time": at(2, 31)},
       {"event": "PreCompact", "trigger": "auto", "time": at(5, 10)}]


# Splitting commands

@pytest.mark.parametrize("command, expected", [
    ("cd x && python3 -B a.py; ls -la", ["cd x", "python3 -B a.py", "ls -la"]),
    ("python3 a.py 2>&1 | tail -5", ["python3 a.py 2>&1", "tail -5"]),
    ("ls -la &> out.txt", ["ls -la &> out.txt"]),
    ("python3 server.py & sleep 1", ["python3 server.py", "sleep 1"]),
    ("false || ls -1", ["false", "ls -1"]),
    ("(cd t01 && python3 a.py)", ["cd t01", "python3 a.py"]),
    ("echo $(ls -1 src)", ["echo", "ls -1 src"]),
    ("echo `ls -la`", ["echo", "ls -la"]),
    ("for f in *.py; do python3 $f; done", ["for f in *.py", "do python3 $f", "done"]),
    ('echo "a; ls -la" && python3 -c "import os; print(1)"', ['echo "a; ls -la"', 'python3 -c "import os; print(1)"']),
    ("# run it\npython3 a.py  # note\nls -1", ["python3 a.py", "ls -1"]),
    ("python3 \\\n  -B a.py", ["python3    -B a.py"]),
])
def test_split_commands(command, expected):
    assert split_commands(command) == expected


def test_here_document_bodies_are_not_commands():
    command = "cat > a.py << 'EOF'\npython3 b.py\nls -la\nEOF\npython3 -B a.py"
    assert split_commands(command) == ["cat > a.py << 'EOF'", "python3 -B a.py"]
    assert split_commands("cat <<-EOF\n\tls -la\n\tEOF\nls -1") == ["cat <<-EOF", "ls -1"]
    assert split_commands("python3 - <<EOF\nimport os\nEOF") == ["python3 - <<EOF"]


@pytest.mark.parametrize("segment, name, args", [
    ("PYTHONDONTWRITEBYTECODE=1 python3 a.py", "python3", ["a.py"]),
    ("timeout 60 python3 -B a.py", "python3", ["-B", "a.py"]),
    ("sudo -E env FOO=1 /usr/bin/python3.11 -B x.py", "python3.11", ["-B", "x.py"]),
    ("nice -n 5 python3 x.py", "python3", ["x.py"]),
    ("time python3 -B x.py", "python3", ["-B", "x.py"]),
    ("do python3 $f", "python3", ["$f"]),
    ("", "", []),
])
def test_program_looks_past_assignments_keywords_and_wrappers(segment, name, args):
    assert program(words(segment)) == (name, args)


@pytest.mark.parametrize("args, expected", [
    (["-B", "a.py"], (True, False)),
    (["a.py", "-B"], (False, False)),
    (["-u", "-B", "a.py"], (True, False)),
    (["-Bc", "print(1)"], (True, True)),
    (["-c", "x"], (False, True)),
    (["-m", "unittest", "-B"], (False, False)),
    (["-B", "-m", "unittest"], (True, False)),
    (["-W", "ignore", "-B", "a.py"], (True, False)),
    (["-", "-B"], (False, False)),
    ([], (False, False)),
])
def test_python_options(args, expected):
    assert python_options(args) == expected


@pytest.mark.parametrize("args, form", [
    (["-1"], LS_ONE), (["-1", "src"], LS_ONE), (["-1a"], LS_ONE), (["-la"], LS_LA), (["-al", "src"], LS_LA), (["-l", "-a"], LS_LA),
    (["-lah"], LS_LA), (["-l", "--all"], LS_LA), ([], LS_OTHER), (["-l"], LS_OTHER), (["--", "-1"], LS_OTHER),
])
def test_ls_form(args, form):
    assert ls_form(args) == form


def test_names_that_only_start_like_python_are_not_python():
    lines = [marker("task_01", tells=("python_B",)), bash("a", "python3-config --prefix; pytest -q; pythonista run; python3.12 -B x.py"),
             result()]
    assert score_session(lines, TAG)["scores"]["python_B"]["applies"] == 1


# Scoring a session

def test_python_rule_counts_each_run_from_the_turn_it_was_told():
    scores = score_session(session(), TAG)["scores"]["python_B"]
    assert (scores["obeyed"], scores["applies"]) == (2, 3)
    assert scores["told_at"] == "task_01"
    assert scores["by_task"] == {"task_01": {"obeyed": 1, "applies": 1}, "task_02": {"obeyed": 0, "applies": 1},
                                 "task_04": {"obeyed": 1, "applies": 1}}
    assert scores["by_compactions"] == {"0": [1, 2], "1": [1, 1]}


def test_ls_rule_applies_only_after_its_own_message():
    scores = score_session(session(), TAG)["scores"]["ls_one"]
    assert (scores["obeyed"], scores["applies"]) == (1, 3)
    assert scores["told_at"] == "rule_ls"
    assert set(scores["by_task"]) == {"task_04", "task_05", "task_06"}


def test_tag_rule_counts_every_later_task_and_reads_the_echo_or_its_output():
    scores = score_session(session(), TAG)["scores"]["session_tag"]
    assert (scores["obeyed"], scores["applies"]) == (3, 5)
    assert {task: cell["obeyed"] for task, cell in scores["by_task"].items()} == {
        "task_02": 1, "task_03": 0, "task_04": 1, "task_05": 1, "task_06": 0}


def test_a_task_with_no_bash_call_does_not_obey_the_tag_rule():
    lines = [marker("task_01", tells=("session_tag",)), result(), marker("task_02", time=at(1)), said("All done."), result()]
    scores = score_session(lines, TAG)["scores"]["session_tag"]
    assert (scores["obeyed"], scores["applies"]) == (0, 1)


def test_per_task_counts():
    tasks = score_session(session(), TAG)["tasks"]
    assert tasks["task_01"]["python_with_B"] == 1 and tasks["task_01"]["ls"] == {LS_ONE: 0, LS_LA: 1, LS_OTHER: 0}
    assert tasks["task_01"]["tag_echoed"] is False
    assert tasks["task_02"]["python_without_B"] == 1 and tasks["task_02"]["tag_echoed"] is True
    assert tasks["task_04"]["ls"][LS_ONE] == 1 and tasks["task_04"]["python_with_B"] == 1
    assert tasks["rule_ls"]["kind"] == "rule" and tasks["rule_ls"]["bash_calls"] == 0


def test_override_reversion_point():
    found = score_session(session(), TAG, LOG)["override"]
    assert found["told"] is True and found["told_at"] == "rule_ls"
    first = found["first_ls_la"]
    assert first == found["first_not_ls_1"]
    assert first["turn"] == "task_05" and first["turns_after"] == 2
    assert first["bash_calls_before"] == 2 and first["ls_calls_before"] == 1
    assert first["compactions_since"] == 1 and first["log_compactions_by_turn_end"] == 1
    assert first["context_tokens"] == 42000 and first["command"] == "ls -la t05"
    assert found["ls_calls_after"] == 3 and found["compactions_after"] == 1


def test_a_plain_ls_breaks_the_rule_before_the_file_form_comes_back():
    lines = [marker("rule_ls", kind="rule", tells=("ls_one",)), result(), marker("task_04", time=at(1)), bash("a", "ls"), COMPACT,
             bash("b", "ls -al"), result()]
    found = score_session(lines, TAG)["override"]
    assert found["first_not_ls_1"]["command"] == "ls" and found["first_not_ls_1"]["compactions_since"] == 0
    assert found["first_ls_la"]["command"] == "ls -al" and found["first_ls_la"]["compactions_since"] == 1


def test_no_override_without_its_message():
    summary = score_session([marker("task_01", tells=("python_B",)), bash("a", "ls -la"), result()], TAG)
    assert summary["override"] == {"told": False}
    assert summary["scores"]["ls_one"]["applies"] == 0 and summary["scores"]["ls_one"]["told_at"] is None


def test_compactions_from_the_stream_and_the_log_agree_turn_by_turn():
    summary = score_session(session(), TAG, LOG)
    assert summary["compactions"] == {"stream": 2, "log_pre": 2, "log_post": 1}
    assert {turn: row["compactions"] for turn, row in summary["tasks"].items()} == {
        turn: row["compactions_in_log"] for turn, row in summary["tasks"].items()}
    assert summary["tasks"]["task_03"]["compactions_in_log"] == 1


def test_a_task_stopped_before_it_finished_is_not_scored_for_the_tag():
    lines = [marker("task_01", tells=("session_tag",)), result(0.1), marker("task_02", time=at(1)), bash("b1", f"echo {TAG}"),
             result(0.2), marker("task_03", time=at(2)), bash("b2", "ls -1")]  # task_03 was cut off: no result line
    scores = score_session(lines, TAG)["scores"]["session_tag"]
    assert (scores["obeyed"], scores["applies"]) == (1, 1)


def test_session_totals():
    summary = score_session(session(), TAG)
    assert summary["valid"] is True and summary["turns"] == 7 == summary["turns_with_result"]
    assert summary["cost_usd"] == pytest.approx(0.7)  # the running total at the end, not the sum 2.8 of the totals
    assert summary["cache_read_share"] == pytest.approx(0.8)
    assert summary["tools"] == {"Bash": 10} and summary["bash_calls"] == 10


def test_a_tool_call_seen_twice_counts_once_and_subagents_are_counted_apart():
    lines = [marker("task_01", tells=("python_B",)), bash("a", "python3 x.py"), bash("a", "python3 x.py"),
             bash("b", "python3 -B y.py", parent="task-tool-1"), result()]
    summary = score_session(lines, TAG)
    assert summary["scores"]["python_B"]["applies"] == 2
    assert summary["bash_calls"] == 2 and summary["subagent_bash_calls"] == 1


def test_lines_before_any_turn_and_broken_lines_are_kept_apart():
    lines = [json.dumps(bash("a", "ls -la")), "not json", "", json.dumps(marker("task_01")), json.dumps(result())]
    summary = score_session(lines, TAG)
    assert list(summary["tasks"]) == ["unmarked", "task_01"]
    assert summary["broken_lines"] == 1 and summary["valid"] is False


def test_summarize_reads_a_session_folder(tmp_path):
    (tmp_path / "hooks").mkdir()
    (tmp_path / "session.json").write_text(json.dumps({"arm": "floor", "seed": 1, "tag": TAG, "ended_by": None}))
    (tmp_path / "stream.jsonl").write_text("".join(json.dumps(line) + "\n" for line in session()))
    (tmp_path / "hooks" / "compaction.jsonl").write_text("".join(json.dumps(line) + "\n" for line in LOG))
    summary = summarize(tmp_path)
    assert summary["arm"] == "floor" and summary["compactions"]["log_pre"] == 2
    assert json.loads((tmp_path / "summary.json").read_text())["scores"]["python_B"]["applies"] == 3
