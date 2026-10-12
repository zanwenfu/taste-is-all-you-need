"""The hidden tests' own verdict on a graded trial: which required tests did not pass, and what they printed."""

from __future__ import annotations

import json

import pytest

from taste.recovery_study import test_failures
from taste.recovery_study.test_failures import failures, message, required

GO_LOG = """=== RUN   TestLoad
--- PASS: TestLoad (0.01s)
=== RUN   TestAuditUnaryInterceptor_CreateNamespace
    middleware_test.go:1512:
        \tError Trace:\t/app/internal/server/middleware/grpc/middleware_test.go:1512
        \tError:      \tNot equal:
        \t            \texpected: 0
        \t            \tactual  : 1
        \tTest:       \tTestAuditUnaryInterceptor_CreateNamespace
--- FAIL: TestAuditUnaryInterceptor_CreateNamespace (0.00s)
=== RUN   TestAuditUnaryInterceptor_DeleteNamespace
--- FAIL: TestAuditUnaryInterceptor_DeleteNamespace (0.00s)
FAIL
FAIL\tgo.flipt.io/flipt/internal/server/middleware/grpc\t0.015s
"""

RUST_LOG = """+ git checkout 59d4cfb tests/integration_tests.rs
: '>>>>> Start Test Output'
running 1 test
test ignored_suffix_arg ... FAILED

failures:

---- ignored_suffix_arg stdout ----
thread 'ignored_suffix_arg' panicked at tests/integration_tests.rs:1012:5:
Unexpected stdout: "abc"

failures:
    ignored_suffix_arg

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 79 filtered out; finished in 0.71s
: '>>>>> End Test Output'
SWEBench Multilingual results starts here
FAILED
SWEBench Multilingual results ends here
"""


def pro_trial(tmp_path, passed, *, stdout=GO_LOG, fail_to_pass=("TestLoad", "TestAuditUnaryInterceptor_CreateNamespace",
                                                                   "TestAuditUnaryInterceptor_DeleteNamespace"),
              pass_to_pass=()):
    """A graded SWE-bench Pro trial (Harbor's adapter) and its task: config.json keeps its lists as text."""
    task, trial = tmp_path / "tasks" / "instance_flipt", tmp_path / "jobs" / "rec-ct-f1" / "instance_flipt__abc"
    (task / "tests").mkdir(parents=True)
    (task / "tests" / "config.json").write_text(json.dumps(
        {"fail_to_pass": repr(list(fail_to_pass)), "pass_to_pass": repr(list(pass_to_pass)),
         "selected_test_files_to_run": json.dumps(list(fail_to_pass))}))
    verifier = trial / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "output.json").write_text(json.dumps({"tests": [{"name": n, "status": "PASSED"} for n in passed]}))
    (verifier / "run-script-stdout.txt").write_text(stdout)
    (verifier / "run-script-stderr.txt").write_text("")
    (verifier / "reward.txt").write_text("0")
    return trial, task


def harness_trial(tmp_path, *, log=RUST_LOG, repo="sharkdp/bat"):
    """A graded SWE-bench Multilingual trial (the SWE-bench harness) and its task."""
    task, trial = tmp_path / "tasks" / "sharkdp__bat-1892", tmp_path / "jobs" / "mlrec-ct-f1" / "sharkdp__bat-1892__x"
    (task / "tests").mkdir(parents=True)
    (task / "tests" / "config.json").write_text(json.dumps(
        {"repo": repo, "instance_id": "sharkdp__bat-1892", "version": "1892",
         "FAIL_TO_PASS": ["ignored_suffix_arg"], "PASS_TO_PASS": ["basic", "tabs_4"]}))
    verifier = trial / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "test-stdout.txt").write_text(log)
    (verifier / "reward.txt").write_text("0")
    return trial, task


def rust_parser(log, spec):
    """Stands in for swebench's cargo parser: "test <name> ... ok|FAILED"."""
    found = {}
    for line in log.splitlines():
        if line.startswith("test ") and " ... " in line:
            name, _, status = line[5:].partition(" ... ")
            found[name] = "PASSED" if status.strip() == "ok" else "FAILED"
    return found


def test_required_tests_come_from_either_graders_task_config(tmp_path):
    _, pro = pro_trial(tmp_path / "pro", [])
    assert required(pro) == ["TestLoad", "TestAuditUnaryInterceptor_CreateNamespace",
                             "TestAuditUnaryInterceptor_DeleteNamespace"]
    _, harness = harness_trial(tmp_path / "ml")
    assert required(harness) == ["ignored_suffix_arg", "basic", "tabs_4"]
    assert required(tmp_path / "nowhere") is None


def test_a_pro_trial_fails_the_required_tests_its_results_do_not_show_passing(tmp_path):
    # Harbor's Pro adapter lists the tests it saw pass; a required test that failed or never ran is missing.
    trial, task = pro_trial(tmp_path, ["TestLoad", "TestJSONSchema"])
    found = failures(trial, task)
    assert found["grader"] == "swebench-pro" and found["required"] == 3
    assert found["failing"] == ["TestAuditUnaryInterceptor_CreateNamespace", "TestAuditUnaryInterceptor_DeleteNamespace"]
    assert "--- FAIL: TestAuditUnaryInterceptor_CreateNamespace" in found["excerpt"]
    assert "actual  : 1" in found["excerpt"] and "--- PASS: TestLoad" not in found["excerpt"]


def test_a_harness_trial_is_read_by_the_repositorys_own_log_parser_between_the_markers(tmp_path):
    trial, task = harness_trial(tmp_path)
    seen = []

    def parser(log, spec):
        seen.append((log, spec.repo, spec.instance_id))
        return rust_parser(log, spec)

    found = failures(trial, task, parsers={"sharkdp/bat": parser})
    assert found["grader"] == "swebench-harness" and found["required"] == 3
    # The required tests the log does not show passing: the failed one, and two that never ran.
    assert found["failing"] == ["ignored_suffix_arg", "basic", "tabs_4"]
    [(log, repo, instance)] = seen
    assert log.startswith("'\nrunning 1 test") and "results starts here" not in log
    assert (repo, instance) == ("sharkdp/bat", "sharkdp__bat-1892")
    assert "panicked at tests/integration_tests.rs:1012:5" in found["excerpt"]


def test_a_test_named_by_its_parts_is_found_by_its_last_part_and_passing_lines_are_left_out():
    log = ("PASS test/other-test.ts\n  ✓ should render the expected pill (31 ms)\n"
           "FAIL test/stores/Store-test.ts\n  ● Store \u203a and membership events \u203a should invalidate the cache\n"
           "    expect(received).toBe(expected)\n    Expected: 2\n    Received: 1\n")
    found = test_failures.excerpt(log, ["test/stores/Store-test.ts | and membership events | should invalidate the cache"])
    assert "should invalidate the cache" in found and "Received: 1" in found
    assert "✓" not in found and "expected pill" not in found


def test_without_a_parser_for_the_repository_the_failing_tests_are_unknown_but_the_output_is_kept(tmp_path):
    trial, task = harness_trial(tmp_path, repo="someone/else")
    found = failures(trial, task, parsers={})
    assert found["failing"] is None and found["grader"] == "swebench-harness"
    assert "panicked at" in found["excerpt"]


def test_without_the_tasks_list_of_required_tests_the_failing_ones_are_unknown_not_none(tmp_path):
    trial, _ = pro_trial(tmp_path, ["TestLoad"])
    found = failures(trial, tmp_path / "no-such-task")
    assert found["failing"] is None and found["required"] is None
    assert "--- FAIL: TestAuditUnaryInterceptor_CreateNamespace" in found["excerpt"]


def test_a_trial_with_no_grader_output_has_no_failures_to_report(tmp_path):
    (tmp_path / "trial").mkdir()
    assert failures(tmp_path / "trial", tmp_path / "task") is None


def test_the_message_names_the_failing_tests_then_their_output_within_the_cap(tmp_path):
    trial, task = pro_trial(tmp_path, ["TestLoad"])
    text = message(failures(trial, task), cap=400)
    assert len(text) <= 400
    assert text.startswith("These required tests did not pass: TestAuditUnaryInterceptor_CreateNamespace, "
                           "TestAuditUnaryInterceptor_DeleteNamespace.\n\nTest output:\n")
    assert "Not equal" in text
    many = {"grader": "swebench-pro", "required": 40, "failing": [f"test_{i}" for i in range(40)], "excerpt": "x"}
    assert "test_0, test_1" in message(many, cap=400) and "(and 3" not in message(many, cap=5000)
    assert message(many, cap=200).count("test_") < 40 and "more)" in message(many, cap=200)
    unknown = {"grader": "swebench-harness", "required": 3, "failing": None, "excerpt": "boom"}
    assert message(unknown, cap=400) == "Test output:\nboom"
    assert message(None, cap=400) == "" and message({**unknown, "excerpt": ""}) == ""


def test_the_swebench_parsers_load_alone_when_swebench_is_installed():
    parsers = test_failures.swebench_parsers()
    if parsers is None:
        pytest.skip("swebench is not installed")
    assert "sharkdp/bat" in parsers and callable(parsers["sharkdp/bat"])
    log = RUST_LOG.split(">>>>> Start Test Output'")[1].split(": '>>>>> End Test Output'")[0]
    spec = test_failures.Spec(repo="sharkdp/bat", instance_id="sharkdp__bat-1892", version="1892")
    assert parsers["sharkdp/bat"](log, spec).get("ignored_suffix_arg") == "FAILED"
