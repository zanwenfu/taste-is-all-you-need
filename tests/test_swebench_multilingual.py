"""SWE-bench Multilingual: upstream's slice, parsers and verdict rule, and the script as the grader ran it."""

from __future__ import annotations

import pytest

from taste.benchmarks import swebench_multilingual as ml
from taste.benchmarks.swebench import SWEInstance
from taste.execution import ExecResult, ScriptedSandbox


def _instance(repo: str = "briannesbitt/carbon", **kw) -> SWEInstance:
    fields = dict(
        instance_id="briannesbitt__carbon-3005", repo=repo, base_commit="0" * 40, problem_statement="",
        test_patch="", version="3.8", fail_to_pass=("Tests\\Foo > New works",),
        pass_to_pass=("Tests\\Foo > Old works", "Tests\\Foo > Old skipped"),
    )
    fields.update(kw)
    return SWEInstance(**fields)


# Under the script's `set -x`, each marker appears first in the trace of the line that prints it.
_LOG = "\n".join([
    "+ git apply --verbose --reject -",
    "+ echo '>>>>> Start Test Output'",
    ">>>>> Start Test Output",
    "+ vendor/bin/phpunit --testdox tests/FooTest.php",
    "Tests\\Foo (Tests\\Foo)",
    " ✔ New works",
    " ✔ Old works",
    " ↩ Old skipped",
    "+ echo '>>>>> End Test Output'",
    ">>>>> End Test Output",
])


def test_slice_is_upstreams_first_start_to_the_next_end() -> None:
    # The first start marker is the trace's, so the slice is the rest of that line.
    assert ml.test_output(_LOG) == "'\n"
    body = ml.test_output("x\n>>>>> Start Test Output\n ✔ New works\n>>>>> End Test Output\n")
    assert body == "\n ✔ New works\n"


def test_an_empty_slice_falls_back_to_the_whole_log_as_upstream_does() -> None:
    assert ml.parse_output(_instance(), _LOG)["Tests\\Foo > New works"] == "PASSED"


def test_a_harness_failure_code_means_no_results() -> None:
    assert ml.parse_output(_instance(), _LOG + "\n>>>>> Tests Timed Out") == {}


def test_no_markers_no_results() -> None:
    assert ml.test_output("bash: vendor/bin/phpunit: not found") is None
    assert ml.parse_output(_instance(), "bash: vendor/bin/phpunit: not found") == {}


def test_parse_uses_the_repos_official_parser() -> None:
    assert ml.parse_output(_instance(), _LOG) == {
        "Tests\\Foo > New works": "PASSED",
        "Tests\\Foo > Old works": "PASSED",
        "Tests\\Foo > Old skipped": "SKIPPED",
    }


def test_a_language_without_a_vendored_parser_raises() -> None:
    with pytest.raises(KeyError):
        ml.parse_output(_instance(repo="gin-gonic/gin"), _LOG)


def test_verdict_is_upstreams_skipped_blocks_nothing_missing_and_error_fail() -> None:
    instance = _instance()
    passing = {"Tests\\Foo > New works": "PASSED", "Tests\\Foo > Old works": "XFAIL",
               "Tests\\Foo > Old skipped": "SKIPPED"}
    assert ml.resolved(instance, passing)
    assert not ml.resolved(instance, {**passing, "Tests\\Foo > Old works": "ERROR"})
    missing = dict(passing)
    del missing["Tests\\Foo > New works"]
    assert not ml.resolved(instance, missing)


def test_suite_runs_the_script_in_a_plain_shell_with_the_image_env() -> None:
    probe = ml.suite(_instance(), "#!/bin/bash\nset -uxo pipefail\necho hi\n", image_env={"PATH": "/usr/local/bin:/usr/bin"})
    assert probe.command.startswith("env -i PATH=/usr/local/bin:/usr/bin HOME=/root ")
    assert "bash -c " in probe.command and "exec 2>&1" in probe.command
    assert probe.members == ("Tests\\Foo > Old works", "Tests\\Foo > Old skipped", "Tests\\Foo > New works")


def test_grade_resolved_from_the_official_output() -> None:
    sandbox = ScriptedSandbox().on("Start Test Output", ExecResult(0, _LOG, ""))
    report = ml.grade_in_sandbox(sandbox, _instance(), "diff --git a/x b/x\n", eval_script="echo '>>>>> Start Test Output'")
    assert report is not None and report.resolved is True
    assert report.fail_to_pass_passed == 1 and report.pass_to_pass_passed == 1


def test_grade_is_none_when_the_script_prints_no_markers_even_at_the_base() -> None:
    sandbox = ScriptedSandbox().on("phpunit", ExecResult(127, "phpunit: not found", ""))
    assert ml.grade_in_sandbox(sandbox, _instance(), "", eval_script="vendor/bin/phpunit") is None
