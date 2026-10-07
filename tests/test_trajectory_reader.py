"""Where a failed run went wrong: the reader's prompt and reply check, the two rules, and steps from a record."""

from __future__ import annotations

import json

import pytest

from taste.agents.checker import SCHEMA
from taste.agents.trajectory_reader import (
    READER_SYSTEM,
    InvalidReading,
    Step,
    before_last_large_edit,
    edit_lines,
    final_message,
    last_passing_tests,
    read_trajectory,
    reader_prompt,
    render_steps,
    rewind_point,
    shows_passing_tests,
    steps_from_trajectory,
    validate_reading,
)
from taste.benchmarks.worker_trajectory import hosted_trajectory
from taste.brains.responses_audit import ROOT, event_id

FINDINGS = {"schema": SCHEMA, "verdict": "not_done",
            "unmet_requirement": "Blank lines must not be counted; they are.",
            "evidence": {"check": "Counted a file with a blank line.", "command": "python count.py /tmp/blank.txt",
                         "output": "3", "expected": "2", "observed": "3"},
            "confidence": 0.8, "ended": "verdict", "steps": 5, "commands": 4, "refused": 0, "cost_usd": 0.2,
            "tokens": {"input": 1000, "output": 100}}

STEPS = [Step(1, "ls", "count.py\nREADME.md\n", 0),
         Step(2, "cat count.py", "\n".join(f"line {n}" for n in range(1, 31)), 0),
         Step(3, "sed -i 's/strip()/rstrip()/' count.py", "", 0),
         Step(4, "python -m pytest -q", "3 passed in 0.02s", 0)]


class Model:
    """A model whose replies are scripted, keeping every conversation it was sent."""

    def __init__(self, *replies):
        self.replies, self.sent = list(replies), []

    def __call__(self, messages):
        self.sent.append(messages)
        return self.replies.pop(0)


# -- the reader -------------------------------------------------------------------


def test_the_reader_names_the_first_wrong_step_from_the_record_it_is_shown():
    model = Model('{"step": 3, "reason": "It changed strip to rstrip, so blank lines count.", "confidence": 0.7}')
    reading = read_trajectory("Count the non-blank lines of a file.", STEPS, "Done: count.py counts lines.",
                              FINDINGS, ask=model)
    assert reading == {"step": 3, "reason": "It changed strip to rstrip, so blank lines count.", "confidence": 0.7}
    assert rewind_point(reading) == 2
    [messages] = model.sent
    assert messages[0] == {"role": "system", "content": READER_SYSTEM}
    prompt = messages[1]["content"]
    assert "<task>\nCount the non-blank lines of a file.\n</task>" in prompt
    assert "<final_message>\nDone: count.py counts lines.\n</final_message>" in prompt
    assert "[step 3] $ sed -i 's/strip()/rstrip()/' count.py\nexit 0" in prompt
    # The checker's findings as the coding agent would be shown them.
    assert "<findings>\nBlank lines must not be counted; they are.\n\nThe reviewer's check:" in prompt
    assert '"step": <a step number from 1 to 4>' in prompt


def test_a_reply_that_names_no_valid_step_is_asked_again_once_with_what_was_wrong():
    model = Model("I think step 9 is where it broke.",
                  'Sure:\n```json\n{"step": 2, "reason": "It read\\nthe wrong file.", "confidence": 1}\n```')
    reading = read_trajectory("Count lines.", STEPS, "", "The tests the task names fail.", ask=model)
    assert reading == {"step": 2, "reason": "It read the wrong file.", "confidence": 1.0}
    retry = model.sent[1]
    assert retry[2] == {"role": "assistant", "content": "I think step 9 is where it broke."}
    assert retry[3]["role"] == "user" and retry[3]["content"].startswith(
        "That reply was not accepted: the reply must be one JSON object.")
    with pytest.raises(InvalidReading, match="no valid reading in 2 replies"):
        read_trajectory("Count lines.", STEPS, "", "x", ask=Model('{"step": 0}', '{"step": 5}'))
    with pytest.raises(TypeError):
        read_trajectory("Count lines.", STEPS, "", "x", ask=lambda messages: {"step": 1})


@pytest.mark.parametrize("value, problem", [
    ({"step": 5, "reason": "r", "confidence": 0.5}, "step must be one of the step numbers, 1 to 4"),
    ({"step": True, "reason": "r", "confidence": 0.5}, "step must be one of"),
    ({"step": "2", "reason": "r", "confidence": 0.5}, "step must be one of"),
    ({"step": 2.5, "reason": "r", "confidence": 0.5}, "step must be one of"),
    ({"step": 2, "reason": " \n ", "confidence": 0.5}, "reason must be one line of text"),
    ({"step": 2, "reason": "x" * 301, "confidence": 0.5}, "at most 300 characters"),
    ({"step": 2, "reason": "r", "confidence": 1.2}, "confidence must be a number from 0 to 1"),
    ({"step": 2, "reason": "r", "confidence": float("nan")}, "confidence must be a number from 0 to 1"),
    (["step", 2], "one JSON object"),
])
def test_a_reading_is_checked_against_the_steps_shown(value, problem):
    with pytest.raises(InvalidReading, match=problem):
        validate_reading(value, {1, 2, 3, 4})


def test_a_reading_keeps_a_whole_step_number_and_one_line():
    assert validate_reading({"step": 2.0, "reason": "  a\n wrong  turn ", "confidence": 0, "extra": 1}, {2}) == {
        "step": 2, "reason": "a wrong turn", "confidence": 0.0}


def test_each_step_is_shown_by_its_command_exit_code_and_the_ends_of_its_output():
    shown = render_steps([
        Step(1, "\n".join(f"echo {n}" for n in range(1, 10)), "\n".join(f"out {n}" for n in range(1, 21)), 0),
        Step(2, "", "", None),
        Step(3, "true", "", 0),
        Step(5, "cat wide.txt", "w" * 250, 1)])
    assert shown == "\n".join([
        "[step 1] $ echo 1", "echo 2", "echo 3", "echo 4", "echo 5", "echo 6", "[... 3 lines not shown ...]",
        "exit 0", "out 1", "out 2", "out 3", "out 4", "[... 10 lines not shown ...]",
        "out 15", "out 16", "out 17", "out 18", "out 19", "out 20", "",
        "[step 2] (no command)", "",
        "[step 3] $ true", "exit 0", "(no output)", "",
        "[step 5] $ cat wide.txt", "exit 1", "w" * 200 + " [...]"])


def test_the_findings_are_shown_as_given():
    assert "<findings>\nThe reviewer judged the work done.\n</findings>" in reader_prompt(
        "t", STEPS, "", {**FINDINGS, "verdict": "done"})
    assert "<findings>\nBlank lines must not be counted" in reader_prompt("t", STEPS, "", json.dumps(FINDINGS))
    assert "<findings>\nThe hidden tests failed.\n</findings>" in reader_prompt("t", STEPS, "", "The hidden tests failed.")


def test_steps_are_numbered_in_order():
    with pytest.raises(ValueError):
        render_steps([Step(2, "ls"), Step(1, "ls")])
    with pytest.raises(ValueError):
        render_steps([])


# -- the rules -----------------------------------------------------------------------


@pytest.mark.parametrize("output, code, passed", [
    ("============ 12 passed, 1 skipped in 0.40s ============", 0, True),
    ("12 passed, 2 warnings in 0.40s", 0, True),
    ("======= 1 failed, 11 passed in 0.40s =======", 1, False),
    ("======= 11 passed, 1 error in 0.40s =======", 0, False),
    ("collected 0 items\n\n============ no tests ran in 0.01s ============", 5, False),
    ("....\n----------------------------------------------------------------------\nRan 4 tests in 0.003s\n\nOK", 0, True),
    ("Ran 4 tests in 0.003s\n\nFAILED (failures=1)", 1, False),
    ("Ran 0 tests in 0.000s\n\nOK", 0, False),
    ("ok  \texample.com/count\t0.012s\nok  \texample.com/cli\t(cached)", 0, True),
    ("--- FAIL: TestCount (0.00s)\nFAIL\texample.com/count\t0.01s", 1, False),
    ("test result: ok. 7 passed; 0 failed; 0 ignored", 0, True),
    ("Tests:       9 passed, 9 total", 0, True),
    ("Tests:       1 failed, 8 passed, 9 total", 1, False),
    ("  9 passing (20ms)", 0, True),
    ("  8 passing (20ms)\n  1 failing", 1, False),
    ("9 examples, 0 failures", 0, True),
    ("OK (9 tests, 20 assertions)", 0, True),
    ("Tests run: 9, Failures: 0, Errors: 0, Skipped: 0", 0, True),
    ("3 passed in 0.02s", 2, False),
    ("3 passed in 0.02s", None, True),
    ("Build finished.", 0, False),
])
def test_visible_tests_pass_when_a_runner_says_so_and_none_says_otherwise(output, code, passed):
    assert shows_passing_tests(Step(1, "run the tests", output, code)) is passed


def test_the_last_step_after_which_the_visible_tests_passed():
    steps = [Step(1, "python -m pytest -q", "1 failed, 2 passed in 0.1s", 1),
             Step(2, "sed -i 's/a/b/' x.py", "", 0),
             Step(3, "python -m pytest -q", "3 passed in 0.1s", 0),
             Step(4, "cat x.py", "def f(): pass", 0),
             Step(5, "python -m pytest -q", "1 failed, 2 passed in 0.1s", 1),
             Step(6, "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "", 0)]
    assert last_passing_tests(steps) == 3
    assert last_passing_tests(steps[:2]) is None


def heredoc(target, lines, *, opener="cat <<'EOF' > {target}"):
    return "\n".join([opener.format(target=target), *(f"line {n}" for n in range(lines)), "EOF"])


@pytest.mark.parametrize("command, lines", [
    (heredoc("/app/count.py", 25), 25),
    (heredoc("count.py", 3, opener="cat > {target} << EOF"), 3),
    (heredoc("count.py", 4, opener="tee {target} <<-EOF"), 4),
    (heredoc("/tmp/scratch.py", 40), 0),
    ("cat <<'EOF' | python\nprint(1)\nEOF", 0),
    ("python3 - <<'PY'\nfrom pathlib import Path\np = Path('count.py')\np.write_text(p.read_text().replace('a', 'b'))\nPY", 3),
    ("git apply <<'EOF'\n--- a/count.py\n+++ b/count.py\n@@ -1,2 +1,2 @@\n-old\n+new\n context\nEOF", 2),
    ("sed -i 's/strip()/rstrip()/' count.py", 1),
    ("sed -i -e 's/a/b/' -e 's/c/d/' count.py", 2),
    ("printf 'a\\nb\\n' > notes.txt", 3),
    ("echo done > /dev/null", 0),
    ("python -m pytest -q 2>&1 | tail -5", 0),
    ("cat count.py && grep -n strip count.py", 0),
])
def test_an_edit_is_measured_by_the_lines_it_writes(command, lines):
    assert edit_lines(command) == lines


def test_the_step_before_the_last_large_edit():
    steps = [Step(1, "ls"),
             Step(2, heredoc("/app/count.py", 30)),
             Step(3, "python -m pytest -q"),
             Step(4, heredoc("/app/count.py", 22)),
             Step(5, "sed -i 's/a/b/' count.py"),
             Step(6, heredoc("/tmp/check.py", 50))]
    assert before_last_large_edit(steps) == 3
    assert before_last_large_edit(steps, min_lines=25) == 1
    assert before_last_large_edit(steps, min_lines=100) is None
    assert before_last_large_edit([Step(1, heredoc("a.py", 20))]) == 0


# -- steps from a hosted agent's record ---------------------------------------------------


def _rows(events):
    rows, parent = [], ROOT
    for number, event in enumerate(events):
        identifier = event_id(parent, event)
        rows.append({"id": identifier, "parent": parent, "event": event, "published": True,
                     "at": f"2026-10-07T00:00:{number:02d}Z"})
        parent = identifier
    return rows


def _completion(number, text, calls, **extra):
    return [{"kind": "hosted_request", "id": f"hosted.{number}", "request_sha": "0" * 64},
            {"kind": "hosted_completion", "id": f"hosted.{number}", "text": [text], "stop_reason": "tool_use",
             "model": "gpt-5.6-luna-2026-07-09", "cost_usd": 0.001,
             "calls": [{"id": f"c{number}{index}", "name": "bash", "arguments": {"command": command}}
                       for index, command in enumerate(calls)], **extra}]


def _ran(effect, command, returncode, output):
    return [{"kind": "hosted_command", "effect_id": effect, "command": command, "cwd": "/app", "timeout_seconds": 30},
            {"kind": "hosted_output", "effect_id": effect, "returncode": returncode, "terminated": "",
             "output": output, "output_chars": len(output), "dropped_bytes": 0}]


def test_the_steps_of_a_hosted_run_are_read_from_its_record():
    events = [{"kind": "hosted_binding", "binding": {"run_id": "run-1", "model": "gpt-5.6-luna-2026-07-09",
                                                      "agent": {"name": "mini-swe-agent", "version": "2.4.6"}}},
              {"kind": "hosted_task", "content": "Count the non-blank lines."},
              *_completion(1, "Let me look.", ["ls"]), *_ran("e1", "ls", 0, "count.py\n"),
              *_completion(2, "I think it is done.", []),
              *_completion(3, "", ["python -m pytest -q", "git diff"]),
              *_ran("e2", "python -m pytest -q", 1, "1 failed in 0.1s\n"), *_ran("e3", "git diff", 0, "+x\n"),
              *_completion(4, "The fix is in count.py.", ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]),
              *_ran("e4", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", 0, "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"),
              # Paid for, but the run ended before the agent received it: not a step it took.
              *_completion(5, "One more thing.", ["ls"], cut_off=True),
              {"kind": "hosted_exit", "exit_status": "Submitted", "stopped_by": "", "submission": ""}]
    trajectory = hosted_trajectory(_rows(events), run_id="run-1")
    assert steps_from_trajectory(trajectory) == [
        Step(1, "ls", "count.py\n", 0),
        Step(2, "", "", None),
        Step(3, "python -m pytest -q\ngit diff", "1 failed in 0.1s\n\n+x\n", 0),
        Step(4, "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", 0)]
    assert final_message(trajectory) == "The fix is in count.py."
    assert final_message({"steps": []}) == ""
