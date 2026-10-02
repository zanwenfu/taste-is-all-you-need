"""What a worker's recorded turns say it ran, read by someone other than the worker."""

from __future__ import annotations

from taste.brains.worker_commands import recorded_commands


def binding(run_id):
    return {"kind": "responses_binding", "binding": {"run_id": run_id}}


def intent(effect, command, *, name="terminal_exec", call_id="call_1"):
    return {"kind": "responses_tool_intent", "effect_id": effect,
            "call": {"id": call_id, "name": name, "arguments": {"command": command}}}


def result(effect, content, *, name="terminal_exec", call_id="call_1"):
    return {"kind": "responses_tool_result", "effect_id": effect,
            "call": {"id": call_id, "name": name, "arguments": {}},
            "result": {"content": content, "is_error": False}}


def test_the_record_is_each_command_its_exit_and_the_last_line_it_printed():
    turns = [
        binding("run-1"),
        {"kind": "responses_input", "id": "assignment", "content": "fix the parser"},
        {"kind": "responses_request", "id": "response.a", "request_sha": "0" * 64},
        intent("effect_1", "go test ./..."),
        result("effect_1", "exit 1\n--- FAIL: TestParse (0.00s)\n\nFAIL"),
        # A provider may reuse a call id after a rollback; the effect id is its own.
        intent("effect_2", "git   status\n--short"),
        result("effect_2", "exit 0"),
        # Saving a note in memory is not a command in the task environment.
        intent("effect_3", "", name="write_artifact"),
        result("effect_3", "Saved in the memory workspace.", name="write_artifact"),
        # The worker was killed while this one ran.
        intent("effect_4", "npm run build"),
    ]
    assert recorded_commands(turns, run_id="run-1") == [
        "ran: go test ./... -> exit 1; last line printed: FAIL",
        "ran: git status --short -> exit 0",
        "started, result not recorded: npm run build",
    ]


def test_only_the_named_run_is_read_and_a_worker_that_ran_nothing_has_no_record():
    turns = [binding("run-1"), intent("effect_1", "make"), result("effect_1", "exit 0"),
             binding("run-2"), intent("effect_2", "make test"), result("effect_2", "exit 2\nfailed")]
    assert recorded_commands(turns, run_id="run-2") == ["ran: make test -> exit 2; last line printed: failed"]
    assert recorded_commands(turns, run_id="run-3") == []
    assert recorded_commands([binding("run-1")], run_id="run-1") == []


def test_the_record_is_bounded_in_lines_and_in_length():
    many = [binding("run-1")]
    for number in range(60):
        many += [intent(f"effect_{number}", f"step {number}"), result(f"effect_{number}", "exit 0")]
    lines = recorded_commands(many, run_id="run-1")
    assert len(lines) == 40 and lines[0] == "ran: step 20 -> exit 0" and lines[-1] == "ran: step 59 -> exit 0"
    long = recorded_commands([binding("run-1"), intent("effect_1", "x" * 1000),
                              result("effect_1", "exit 0\n" + "y" * 1000)], run_id="run-1")
    assert long == ["ran: " + "x" * 300 + " [cut] -> exit 0; last line printed: " + "y" * 200 + " [cut]"]
    assert sum(map(len, recorded_commands(many, run_id="run-1", limit=100))) <= 100
