"""The checker: its verdict, its tool layer, its limits, its feedback, and its run as a hosted agent."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import secrets
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from taste.agents import HOSTED_AGENTS, AgentExit, HostedStop, ModelReply, ShellResult, hosted_agent
from taste.agents.checker import (
    BARE_FEEDBACK,
    BASH_TOOL,
    CUT_OFF,
    FORBIDDEN_PATHS,
    LAST_STEP,
    NO_ACTION,
    SCHEMA,
    SUBMISSION_CHARS,
    SUBMIT_TOOL,
    VERDICT_CHARS,
    Checker,
    CheckerSettings,
    InvalidVerdict,
    checker_task,
    feedback,
    forbidden_reference,
    json_object,
    parse_submission,
    validate_verdict,
)
from taste.benchmarks.harbor_settings import TrialSettings
from taste.brains import azure_worker_launch, benchmark_reply
from taste.brains.azure_central_host import compose_azure_central_runtime
from taste.brains.central_planner import Goal
from taste.brains.hosted_worker import SUMMARY_CHARS
from taste.brains.records import WorkerReport
from taste.brains.terminal_broker import TerminalBroker
from taste.brains.terminal_service import TerminalCredential, TerminalGrant, TerminalService
from taste.brains.worker_protocol import WORKER_REPORT_PATH
from taste.providers._openai import OpenAIProvider
from tests.test_agent_alone import Scripted, alone
from tests.test_azure_central_host import environment, install_planner
from tests.test_azure_central_host import policy as _policy
from tests.test_azure_openai import sdk_transport as _sdk_transport
from tests.test_azure_worker_process import BOOTSTRAP

policy, sdk_transport = _policy, _sdk_transport

NOT_DONE = {
    "verdict": "not_done",
    "unmet_requirement": "An empty input must print 0; the program prints nothing and fails.",
    "evidence": {"check": "Ran the counter on an empty file, as the task's example does.",
                 "command": "printf '' > /tmp/empty.txt && python count.py /tmp/empty.txt",
                 "output": "Traceback (most recent call last):\nValueError: empty file",
                 "expected": "0 on standard output, exit 0",
                 "observed": "a ValueError traceback, exit 1"},
    "confidence": 0.9,
}


def reply(*items, cost=0.01, status="completed"):
    return ModelReply(output=tuple(items), status=status,
                      incomplete_reason=None if status == "completed" else "max_output_tokens",
                      usage={"input_tokens": 100, "cache_read_tokens": 50, "output_tokens": 20},
                      model="gpt-6-sol-2026-09-22", cost_usd=cost)


def call(name, arguments, call_id):
    return {"type": "function_call", "id": "fc_" + call_id, "call_id": call_id, "name": name,
            "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments), "status": "completed"}


def bash(command, call_id):
    return call("bash", {"command": command}, call_id)


def submit(verdict, call_id="s1"):
    return call("submit_verdict", verdict, call_id)


def text(words):
    return {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": words, "annotations": []}]}


class Host:
    """A host whose model replies are scripted and whose shell answers by command."""

    def __init__(self, replies=(), outputs=None):
        self.replies = list(replies)
        self.outputs = dict(outputs or {})
        self.asked, self.ran = [], []

    def ask(self, *, messages, tools, effort):
        self.asked.append({"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools), "effort": effort})
        return self.replies.pop(0)

    def run(self, command, *, cwd, timeout_seconds, shown=None):
        self.ran.append({"command": command, "cwd": cwd, "timeout": timeout_seconds, "shown": shown})
        for key, result in self.outputs.items():
            if key in shown:
                return result
        return ShellResult("", 0)


def review(host, task="Count the lines of a file.", **settings):
    return Checker(effort="medium", settings=CheckerSettings(**settings)).run(
        task, host, cwd="/app", model_name="gpt-6-sol")


def seen(request):
    """What the model was shown after its calls: each call's output, and any message of ours."""
    outputs, notes = {}, []
    for message in request["messages"][2:]:
        if isinstance(message["content"], str):
            notes.append(message["content"])
            continue
        item = message["content"][0]["item"]
        if item.get("type") == "function_call_output":
            outputs[item["call_id"]] = item["output"]
    return outputs, notes


# -- the verdict ------------------------------------------------------------------


def test_a_not_done_verdict_keeps_its_requirement_evidence_and_confidence():
    assert validate_verdict(NOT_DONE) == NOT_DONE
    done = validate_verdict({"verdict": "done", "confidence": 1, "evidence": {"check": "all passed"},
                             "notes": "dropped"})
    assert done == {"verdict": "done", "confidence": 1.0}


@pytest.mark.parametrize("change, problem", [
    ({"verdict": "maybe"}, 'verdict must be "done" or "not_done"'),
    ({"confidence": 1.5}, "confidence must be a number from 0 to 1"),
    ({"confidence": True}, "confidence must be a number from 0 to 1"),
    ({"confidence": "high"}, "confidence must be a number from 0 to 1"),
    ({"unmet_requirement": "  "}, "unmet_requirement must be text, not empty"),
    ({"evidence": None}, "evidence must be an object"),
    ({"evidence": {**NOT_DONE["evidence"], "expected": None}}, "evidence.expected must be text"),
    ({"evidence": {**NOT_DONE["evidence"], "output": "\n".join(["line"] * 31)}}, "at most 30 lines"),
    ({"evidence": {**NOT_DONE["evidence"], "check": "x" * 301}}, "evidence.check is 301 characters"),
    ({"unmet_requirement": "Fix it like this:\n```python\nreturn 0\n```"}, "without code blocks or a patch"),
    ({"evidence": {**NOT_DONE["evidence"], "observed": "--- a/count.py\n+++ b/count.py\n@@ -1 +1 @@"}},
     "without code blocks or a patch"),
    ({"evidence": {**NOT_DONE["evidence"], "command": "git apply <<EOF\n@@ -1 +1 @@\nEOF"}}, "not a patch"),
])
def test_a_verdict_that_breaks_the_schema_is_refused_with_what_to_correct(change, problem):
    with pytest.raises(InvalidVerdict, match=problem):
        validate_verdict({**NOT_DONE, **change})


def test_a_done_verdict_names_no_unmet_requirement():
    with pytest.raises(InvalidVerdict, match="names no unmet requirement"):
        validate_verdict({"verdict": "done", "confidence": 0.7, "unmet_requirement": "the edge case"})


def test_a_verdict_too_long_to_carry_whole_is_refused_and_one_within_fits_the_trials_reply():
    long_lines = "\n".join(["e" * 79] * 30)
    with pytest.raises(InvalidVerdict, match="shorten the evidence"):
        validate_verdict({**NOT_DONE, "evidence": {**NOT_DONE["evidence"], "output": long_lines[:2400],
                                                   "command": "c" * 1200}})
    fits = {**NOT_DONE, "evidence": {**NOT_DONE["evidence"], "output": long_lines[:2400]}}
    assert len(json.dumps(validate_verdict(fits))) <= VERDICT_CHARS
    host = Host([reply(submit(fits))])
    result = review(host)
    assert len(result.submission) <= SUBMISSION_CHARS < SUMMARY_CHARS


def test_a_verdict_written_as_text_is_read_from_the_reply():
    assert json_object('Here it is:\n```json\n{"verdict": "done", "confidence": 0.8}\n```') == {
        "verdict": "done", "confidence": 0.8}
    assert json_object('My verdict: {"verdict": "done", "confidence": 0.8}. Thanks.') == {
        "verdict": "done", "confidence": 0.8}
    assert json_object("no object here") is None


# -- the tool layer -------------------------------------------------------------------


@pytest.mark.parametrize("command, cwd, named", [
    ("cat /tests/test_outputs.py", "/app", "/tests"),
    ("ls -la /tests/", "/app", "/tests"),
    ("cat '/tests/test_outputs.py'", "/app", "/tests"),
    ('cat /te"st"s/test_outputs.py', "/app", "/tests"),
    ("cat //tests//test_outputs.py", "/app", "/tests"),
    ("cat /app/../tests/test_outputs.py", "/app", "/tests"),
    ("cat ../tests/test_outputs.py", "/app", "/tests"),
    ("ls /te*", "/app", "/tests"),
    ("ls /t?sts /[t]ests", "/app", "/tests"),
    ("cat /{tests,app}/x.py", "/app", "/tests"),
    ("ls /*", "/app", "/tests"),
    ("cd / && cat tests/test_outputs.py", "/app", "/tests"),
    ("python -m pytest tests", "/", "/tests"),
    ("PYTHONPATH=/tests python -m pytest", "/app", "/tests"),
    ("python -m pytest --rootdir=/tests", "/app", "/tests"),
    ("python -c \"print(open('/tests/test_outputs.py').read())\"", "/app", "/tests"),
    ("cat /solution/solve.sh", "/app", "/solution"),
    ("bash /solutions/run.sh", "/app", "/solutions"),
    ("ls /oracle", "/app", "/oracle"),
    ("cat /grader/grade.py", "/app", "/grader"),
    ("cat /logs/verifier/reward.txt", "/app", "/logs/verifier"),
    ("cd /logs && cat verifier/ctrf.json", "/app", "/logs/verifier"),
    ("ls /logs/*", "/app", "/logs/verifier"),
    ("sh /eval.sh", "/app", "/eval.sh"),
    ("git clone https://github.com/laude-institute/terminal-bench", "/app", "laude-institute"),
    ("pip download swebench", "/app", "swebench"),
    ("git clone https://github.com/someone/tbench-tasks", "/app", "tbench"),
    ("curl -s https://raw.githubusercontent.com/x/SWE-bench/main/x.json", "/app", "SWE-bench"),
])
def test_a_command_that_names_a_forbidden_path_or_source_is_caught(command, cwd, named):
    assert forbidden_reference(command, cwd) == named


@pytest.mark.parametrize("command, cwd", [
    ("git status && git diff", "/app"),
    ("python -m pytest tests/test_count.py -q", "/app"),
    ("cat /app/tests/test_count.py", "/app"),
    ("cd /testbed && python -m pytest tests", "/"),
    ("ls /testbed /", "/"),
    ("cat /solution.txt /app/solution.py /tests.md", "/app"),
    ("curl -s https://example.com/tests/solution", "/app"),
    ("ls /logs /logs/agent", "/app"),
    ("cat > /tmp/check_count.py <<'EOF'\nimport subprocess\nassert run('tests') == 0\nEOF\npython /tmp/check_count.py",
     "/app"),
    ("grep -rn 'def test' tests/ ${HOME}/tests", "/app"),
    ("echo testbench > /tmp/x && sed -i 's/tests/checks/' /tmp/x", "/app"),
])
def test_the_work_and_the_repositorys_own_tests_are_not_caught(command, cwd):
    assert forbidden_reference(command, cwd) is None


def test_the_forbidden_list_names_the_benchmarks_hidden_places():
    assert {"/tests", "/solution", "/grader", "/logs/verifier", "/oracle"} <= set(FORBIDDEN_PATHS)


def test_a_forbidden_command_is_refused_before_it_reaches_the_container():
    host = Host([reply(bash("cat /tests/test_outputs.py", "c1"), bash("ls /app", "c2")),
                 reply(bash("ls /solution", "c3")),
                 reply(submit({"verdict": "done", "confidence": 0.6}))],
                outputs={"ls /app": ShellResult("count.py\n", 0)})
    result = review(host)
    assert [run["shown"] for run in host.ran] == ["ls /app"]
    outputs, _ = seen(host.asked[1])
    assert outputs["c1"].startswith("refused: the command names /tests") and "Nothing was run" in outputs["c1"]
    assert outputs["c2"] == "exit code: 0\noutput:\ncount.py\n"
    value = json.loads(result.submission)
    assert (value["refused"], value["commands"], value["steps"]) == (2, 1, 3)


# -- the run -------------------------------------------------------------------------


def test_a_review_runs_its_checks_in_the_container_and_ends_with_its_verdict_as_the_submission():
    host = Host([reply(text("I will look at the change."), bash("git diff", "c1"), cost=0.02),
                 reply(bash("python count.py /tmp/empty.txt", "c2"), cost=0.03),
                 reply(submit(NOT_DONE), cost=0.04)],
                outputs={"git diff": ShellResult("+def count(path):\n", 0),
                         "count.py": ShellResult("Traceback (most recent call last):\nValueError: empty file\n", 1)})
    result = review(host, command_seconds=60)
    assert isinstance(result, AgentExit)
    assert (result.exit_status, result.model_calls) == ("Submitted", 3)
    # Each command in a fresh shell, standard error folded in, as the agent wrote it on record.
    assert host.ran[0] == {"command": ("export PAGER=cat MANPAGER=cat GIT_PAGER=cat LESS=-R PIP_PROGRESS_BAR=off "
                                       "TQDM_DISABLE=1; {\ngit diff\n} 2>&1"),
                           "cwd": "/app", "timeout": 60, "shown": "git diff"}
    outputs, _ = seen(host.asked[2])
    assert outputs["c2"] == "exit code: 1\noutput:\nTraceback (most recent call last):\nValueError: empty file\n"
    value = json.loads(result.submission)
    assert value == {"schema": SCHEMA, **NOT_DONE, "ended": "verdict", "steps": 3, "commands": 2, "refused": 0,
                     "cost_usd": 0.09, "tokens": {"input": 450, "output": 60}}
    assert parse_submission(result.submission) == value


def test_the_request_is_the_checkers_own_prompt_and_tools_sent_as_written():
    host = Host([reply(submit({"verdict": "done", "confidence": 0.5}))])
    review(host, task=checker_task("Count the lines.", "Done: count.py counts lines.", "count.py"))
    request = host.asked[0]
    assert request["effort"] == "medium"
    assert request["tools"] == [BASH_TOOL, SUBMIT_TOOL]
    system, task = request["messages"]
    assert system["role"] == "system" and "the working directory is /app" in system["content"]
    assert "a limit of 120 seconds" in system["content"] and "at most 30 replies" in system["content"]
    assert "/tests" in system["content"] and "solution code" in system["content"]
    assert task["role"] == "user" and "<task>\nCount the lines.\n</task>" in task["content"]
    # The provider replays the input items exactly as the checker wrote them.
    later = Host([reply(bash("ls", "c1")), reply(submit({"verdict": "done", "confidence": 0.5}))])
    review(later)
    items = OpenAIProvider(api_key="unused")._to_input(later.asked[1]["messages"])
    assert items[2] == bash("ls", "c1")
    assert items[3] == {"type": "function_call_output", "call_id": "c1", "output": "exit code: 0\noutput:\n(none)"}


def test_the_task_text_holds_the_task_verbatim_and_the_agents_own_account():
    task = "Fix the parser.\n\n  Keep {braces} and trailing spaces.  \n"
    built = checker_task(task, "  I fixed it.\n", "", changed_paths=["/app/parser.py", "/app/new.py"])
    assert f"<task>\n{task}\n</task>" in built
    assert "<final_message>\nI fixed it.\n</final_message>" in built
    assert "<submission>\n(empty)\n</submission>" in built
    assert "<changed_paths>\n/app/parser.py\n/app/new.py\n</changed_paths>" in built
    assert "check what they claim, do not take them as evidence" in built
    assert "<changed_paths>" not in checker_task(task, "", "")
    with pytest.raises(ValueError):
        checker_task(" ", "", "")


def test_at_its_step_limit_it_is_told_to_give_its_verdict_and_runs_nothing_more():
    host = Host([reply(bash("ls", "c1")), reply(bash("git diff", "c2")), reply(bash("cat count.py", "c3"))])
    result = review(host, max_steps=3)
    assert [run["shown"] for run in host.ran] == ["ls", "git diff"]
    assert seen(host.asked[2])[1] == [LAST_STEP] and seen(host.asked[1])[1] == []
    value = json.loads(result.submission)
    assert result.exit_status == "LimitsExceeded"
    assert (value["verdict"], value["confidence"], value["ended"], value["steps"]) == (None, None, "steps", 3)
    assert "evidence" not in value and parse_submission(result.submission)["ended"] == "steps"


def test_a_verdict_given_at_the_last_step_is_accepted():
    host = Host([reply(bash("ls", "c1")), reply(submit(NOT_DONE))])
    result = review(host, max_steps=2)
    assert result.exit_status == "Submitted" and json.loads(result.submission)["verdict"] == "not_done"


def test_it_stops_before_a_call_its_dollars_could_not_pay_for():
    # Each call costs 0.3 of 1.0: the third is the last one a fourth could not follow.
    host = Host([reply(bash("ls", f"c{n}"), cost=0.3) for n in range(1, 5)])
    result = review(host, max_usd=1.0)
    assert len(host.asked) == 3 and seen(host.asked[2])[1] == [LAST_STEP]
    value = json.loads(result.submission)
    assert (value["ended"], value["cost_usd"], value["commands"]) == ("usd", 0.9, 2)
    # A call far dearer than the ones before it leaves no room for another.
    dear = Host([reply(bash("ls", "c1"), cost=0.2), reply(bash("ls", "c2"), cost=1.5)])
    assert json.loads(review(dear, max_usd=1.0).submission)["ended"] == "usd" and len(dear.asked) == 2


def test_it_stops_before_a_step_its_time_could_not_hold():
    now = [0.0]

    class Slow(Host):
        def run(self, command, **options):
            now[0] += 100  # each command takes 100 seconds
            return super().run(command, **options)

    host = Slow([reply(bash("make test", f"c{n}")) for n in range(1, 6)])
    result = Checker(effort="low", settings=CheckerSettings(max_seconds=350), clock=lambda: now[0]).run(
        "Make the tests pass.", host, cwd="/app", model_name="gpt-6-sol")
    # After two steps of 100 seconds, a third fits but a fourth would not: the third is the last.
    assert len(host.asked) == 3 and seen(host.asked[2])[1] == [LAST_STEP] and len(host.ran) == 2
    value = json.loads(result.submission)
    assert (result.exit_status, value["ended"], value["verdict"]) == ("LimitsExceeded", "seconds", None)


def test_replies_that_neither_run_nor_decide_end_it_after_three():
    host = Host([reply(text("Let me think.")), reply(status="incomplete"), reply(text("Still thinking."))])
    result = review(host)
    assert result.exit_status == "FormatError" and json.loads(result.submission)["ended"] == "format_errors"
    assert seen(host.asked[1])[1] == [NO_ACTION] and seen(host.asked[2])[1] == [NO_ACTION, CUT_OFF]


def test_a_refused_verdict_is_answered_with_what_to_correct_and_a_corrected_one_accepted():
    host = Host([reply(submit({**NOT_DONE, "confidence": 2}, "s1")), reply(submit(NOT_DONE, "s2"))])
    result = review(host)
    outputs, _ = seen(host.asked[1])
    assert outputs["s1"].startswith("not accepted: confidence must be a number from 0 to 1")
    assert result.exit_status == "Submitted" and json.loads(result.submission)["steps"] == 2


def test_a_verdict_written_in_the_reply_text_ends_the_review():
    host = Host([reply(text("```json\n" + json.dumps(NOT_DONE) + "\n```"))])
    assert json.loads(review(host).submission)["unmet_requirement"] == NOT_DONE["unmet_requirement"]
    wrong = Host([reply(text(json.dumps({"verdict": "done", "confidence": 9}))), reply(submit(NOT_DONE))])
    review(wrong)
    assert seen(wrong.asked[1])[1][0].startswith("Your verdict was not accepted: confidence")


def test_a_malformed_or_unknown_call_is_answered_and_the_review_goes_on():
    host = Host([reply(call("bash", "{not json", "c1"), call("python", {"code": "1"}, "c2"),
                       call("bash", {"command": " "}, "c3")),
                 reply(submit({"verdict": "done", "confidence": 0.5}))])
    review(host)
    outputs, _ = seen(host.asked[1])
    assert outputs == {"c1": "error: the arguments must be one JSON object",
                       "c2": "error: there is no tool named 'python'; the tools are bash and submit_verdict",
                       "c3": "error: bash needs a command, as text"}
    assert host.ran == []


def test_a_command_past_its_time_and_a_long_output_are_shown_as_such():
    host = Host([reply(bash("sleep 999", "c1"), bash("cat big.log", "c2")),
                 reply(submit({"verdict": "done", "confidence": 0.5}))],
                outputs={"sleep": ShellResult("partial\n", -9, timed_out=True),
                         "big.log": ShellResult("a" * 2000 + "b" * 3000, 0)})
    review(host, command_seconds=30, output_chars=1000)
    outputs, _ = seen(host.asked[1])
    assert outputs["c1"] == "exit code: none (ended after 30 seconds; its output so far is below)\noutput:\npartial\n"
    assert outputs["c2"] == ("exit code: 0\noutput:\n" + "a" * 400 + "\n[... 4,000 characters cut ...]\n" + "b" * 600)


def test_a_stopped_host_ends_the_review_and_is_not_swallowed():
    class Stopped(Host):
        def run(self, command, **_):
            raise HostedStop("budget")

    with pytest.raises(HostedStop):
        review(Stopped([reply(bash("ls", "c1"))]))


def test_settings_are_checked():
    for bad in ({"max_steps": 0}, {"max_usd": float("nan")}, {"command_seconds": -1}, {"max_steps": 2.0},
                {"output_chars": 10}, {"max_seconds": 0}):
        with pytest.raises(ValueError):
            CheckerSettings(**bad)


# -- feedback ---------------------------------------------------------------------------


def test_the_evidence_feedback_shows_the_unmet_requirement_and_the_check_that_shows_it():
    submission = json.dumps({"schema": SCHEMA, **NOT_DONE, "ended": "verdict", "steps": 4, "commands": 3,
                             "refused": 0, "cost_usd": 0.1, "tokens": {"input": 1, "output": 1}})
    expected = (
        "An empty input must print 0; the program prints nothing and fails.\n\n"
        "The reviewer's check: Ran the counter on an empty file, as the task's example does.\n"
        "$ printf '' > /tmp/empty.txt && python count.py /tmp/empty.txt\n"
        "Traceback (most recent call last):\nValueError: empty file\n"
        "Expected: 0 on standard output, exit 0\n"
        "Observed: a ValueError traceback, exit 1")
    assert feedback(submission) == feedback(submission, "evidence") == expected
    assert feedback(json.loads(submission)) == feedback(NOT_DONE) == expected
    report = f"# Report of taste-checker 1\n\n## Its submission\n\n{submission}\n\n## Commands it ran\n\n(none)\n"
    assert feedback(report) == expected
    assert feedback(submission, "bare") == BARE_FEEDBACK == "A reviewer rejected the submission; keep working."


def test_feedback_is_only_for_a_not_done_verdict():
    done = Host([reply(submit({"verdict": "done", "confidence": 0.9}))])
    submission = review(done).submission
    for variant in ("evidence", "bare"):
        with pytest.raises(ValueError, match="not_done"):
            feedback(submission, variant)
    with pytest.raises(ValueError, match="variant"):
        feedback(NOT_DONE, "hint")


def test_a_submission_that_is_not_the_checkers_is_refused():
    good = json.loads(review(Host([reply(submit(NOT_DONE))])).submission)
    for broken in ({**good, "schema": "other"}, {**good, "ended": "steps"}, {**good, "steps": -1},
                   {**good, "cost_usd": float("inf")}, {**good, "verdict": None},
                   {**good, "verdict": None, "confidence": None, "ended": "verdict"}):
        with pytest.raises(ValueError):
            parse_submission(json.dumps(broken))
    with pytest.raises(ValueError):
        parse_submission("not json")


# -- hosting ----------------------------------------------------------------------------


def test_the_checker_is_a_hosted_agent_a_trial_can_name():
    assert HOSTED_AGENTS["checker"] == ("taste.agents.checker", "Checker")
    agent = hosted_agent("checker", effort="high")
    assert isinstance(agent, Checker) and agent.effort == "high" and agent.settings == CheckerSettings()
    identity = agent.identity()
    assert (identity["name"], identity["version"], identity["schema"]) == ("taste-checker", "1", SCHEMA)
    assert identity["settings"] == {"max_steps": 30, "max_usd": 2.0, "max_seconds": 900.0, "command_seconds": 120.0,
                                    "output_chars": 10_000, "max_format_errors": 3}
    assert len(identity["prompts_sha256"]) == 64
    settings = TrialSettings.from_options({"model": "gpt-6-sol", "agent": "checker", "services": "none",
                                           "worker_max_calls": "32", "spend_cap_usd": "2.5"})
    assert settings.disclosure()["worker_agent"] == "checker" and settings.generations == 1


def test_its_messages_hold_no_final_text_so_the_report_summary_is_its_verdict():
    from taste.brains.hosted_worker import HostedWorkerRuntime

    result = review(Host([reply(text("Looks wrong."), bash("ls", "c1")), reply(submit(NOT_DONE))]))
    assert HostedWorkerRuntime._final_text(result.messages) == ""
    assert all(isinstance(message, dict) for message in result.messages)
    assert result.messages[0]["role"] == "system" and result.messages[-2]["name"] == "submit_verdict"
    assert result.messages[-1] == {"type": "function_call_output", "call_id": "s1", "output": "accepted"}


# -- the checker alone, through the goal machinery a trial uses ---------------------------


CHECKER_ALONE = BOOTSTRAP.replace("install(network)", f"""
import json
from tests.test_openai_responses import function_call
def agent(number, payload):
    if number == 1:
        return [function_call(json.dumps({{"command": "make test"}}), name="bash", call_id="c1", id="fc_c1")]
    return [function_call({json.dumps(NOT_DONE)!r}, name="submit_verdict", call_id="c2", id="fc_c2")]
install(network, worker_reply=agent)
""")


def test_the_checker_alone_reviews_once_and_its_verdict_is_the_trials_final_reply(
        tmp_path, policy, sdk_transport, monkeypatch):
    sent, _ = install_planner(sdk_transport)
    admitted = replace(alone(policy), worker_agent="checker")
    task = checker_task("Make the tests pass.", "All tests pass now.", "")
    goal = Goal(goal_id="azure-goal", task=task, success_criteria=("the review is done",), budget_usd=100,
                metadata={benchmark_reply.KEY: benchmark_reply.SCHEMA, benchmark_reply.RESERVE_KEY: 10})
    original = azure_worker_launch.worker_command

    def command(*args, **kwargs):
        argv = list(original(*args, **kwargs))
        argv[3] = argv[3].replace("import runpy;", CHECKER_ALONE + "\nsys.modules.pop("
                                  "'taste.brains.azure_worker_entrypoint', None)\nimport runpy;", 1)
        return tuple(argv)

    monkeypatch.setattr(azure_worker_launch, "worker_command", command)

    async def scenario():
        env = Scripted()
        owner = TerminalBroker.create(tmp_path / "terminal-ledger", admitted.terminal.binding, env)
        with tempfile.TemporaryDirectory(prefix="taste-checker-rpc-", dir="/tmp") as directory:
            socket_path = str(Path(directory) / "service" / "terminal.sock")
            seed = TerminalCredential(socket_path, os.geteuid(), os.geteuid(),
                                      TerminalGrant(owner.binding, "controller_bootstrap", 5), "a" * 64)
            service = TerminalService(owner, [seed])
            await service.start()
            loop = asyncio.get_running_loop()

            def issue(spec):
                credential = TerminalCredential(socket_path, os.geteuid(), os.geteuid(),
                                                admitted.terminal.grant(spec.assignment), secrets.token_hex(32))

                async def register():
                    service.authorize(credential)

                asyncio.run_coroutine_threadsafe(register(), loop).result(timeout=5)
                return credential

            root = tmp_path / "repo"
            root.mkdir()
            try:
                with compose_azure_central_runtime(root, "checker", goal, policy=admitted,
                                                   environment=environment(),
                                                   terminal_credential_provider=issue) as runtime:
                    result = await runtime.run_async(max_generations=1, wall_clock_seconds=60)
                    assert result.stop_reason == "generation_bound", result.detail
                    (run,) = runtime.supervisor.runs()
                    # The checker's task text, verbatim, was its whole assignment.
                    assert run.assignment.contract.task == task
                    report = WorkerReport.from_json(
                        runtime.store.view(run.assignment.worker).head.read(WORKER_REPORT_PATH))
                    assert report.completed and report.metadata["agent"]["name"] == "taste-checker"
                    # Its verdict is the run's closing reply, whole: a driver reads it there.
                    reply = runtime.planner.current_plan(goal.goal_id).metadata["proposal"]["final_reply"]
                    value = parse_submission(reply)
                    assert (value["verdict"], value["steps"], value["commands"]) == ("not_done", 2, 1)
                    assert value["evidence"] == NOT_DONE["evidence"]
                    assert feedback(reply).startswith(NOT_DONE["unmet_requirement"] + "\n\nThe reviewer's check:")
                    assert [call.command.rsplit("{\n", 1)[1] for call in env.calls] == ["make test\n} 2>&1"]
                assert sent == []
            finally:
                await service.close()
                owner.close()
    asyncio.run(scenario())
