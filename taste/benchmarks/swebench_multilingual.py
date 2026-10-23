"""SWE-bench Multilingual: the official test scripts and log parsers, for replaying its published runs.

The dataset carries each instance's official evaluation script (``eval_script``)
and the leaderboard publishes the script each submission was graded with, so
nothing here re-derives a test command: a state is tested by running that
script on it, exactly as the official harness ran it on the final patch. What
this module adds is the reading of the output, by upstream's own rule:

- **The slice.** Upstream's ``get_logs_eval`` (4.1.0) parses the text between
  the first ``>>>>> Start Test Output`` and the next ``>>>>> End Test Output``,
  and when that yields no result, the whole log. Under the script's
  ``set -x`` the first start marker is usually the trace of the ``echo`` that
  prints it, and the two streams interleave, so the slice is often a line of
  trace and the whole-log fallback is what actually grades. A log holding one
  of the harness's failure codes, or missing either marker, has no results.
- **The verdict.** Upstream's ``get_eval_tests_report`` counts a graded test
  as failed when it is missing from the log or FAILED/ERROR, and as passed
  when PASSED/XFAIL; a SKIPPED test is neither and blocks nothing. An
  instance is resolved when no FAIL_TO_PASS or PASS_TO_PASS test failed.
  This is upstream's rule, not Verified's adapter's (which counts SKIPPED as
  not passing); mixing the two would make the replay's grades incomparable
  with the leaderboard's.

Only the JavaScript/TypeScript, PHP and Ruby parsers are vendored
(:mod:`taste.benchmarks.swebench_log_multilingual`); an instance in another
language raises rather than reading its log with the wrong grammar.
"""

from __future__ import annotations

import functools
from pathlib import Path
from types import SimpleNamespace

from taste.benchmarks.swebench import GradeReport, SWEInstance
from taste.benchmarks.swebench_log_multilingual import MAP_REPO_TO_PARSER
from taste.replay import SuiteProbe

START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"
#: Upstream's APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT: a log holding one has no results.
BAD_CODES = (">>>>> Patch Apply Failed", ">>>>> Reset Failed", ">>>>> Tests Errored", ">>>>> Tests Timed Out")
FAILING_STATUSES = ("FAILED", "ERROR")
PASSING_STATUSES = ("PASSED", "XFAIL")


def load_scripts(directory: Path) -> dict[str, str]:
    """``{instance_id: eval script}`` from ``<directory>/<instance_id>.sh``."""
    return {path.stem: path.read_text() for path in sorted(Path(directory).glob("*.sh"))}


def suite_command(eval_script: str, image_env: dict[str, str] | None = None) -> str:
    """The official script as the grader ran it: ``docker exec ... /bin/bash /eval.sh``.

    A non-login shell with the image's own environment, both streams on one.
    The sandbox runs every command under ``bash -lc``, and a login shell's
    ``/etc/profile`` resets ``PATH`` on Debian-based images, where these
    projects' runtimes (``bundle``, ``composer``, ``node``) are put on it by the
    image's ``ENV``.
    """
    from taste.benchmarks.public_replay import plain_shell

    return plain_shell("exec 2>&1\n" + eval_script, image_env or {}, {})


def test_output(log: str) -> str | None:
    """Upstream's slice of a run's output, or None when either marker is missing."""
    if START_TEST_OUTPUT not in log or END_TEST_OUTPUT not in log:
        return None
    return log.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]


def parse_output(instance: SWEInstance, log: str) -> dict[str, str]:
    """Per-test statuses from a run's output, as upstream's ``get_logs_eval`` reads them."""
    parser = MAP_REPO_TO_PARSER.get(instance.repo)
    if parser is None:
        raise KeyError(f"no vendored log parser for {instance.repo}")
    body = test_output(log)
    if body is None or any(code in log for code in BAD_CODES):
        return {}
    spec = SimpleNamespace(instance_id=instance.instance_id)
    return dict(parser(body, spec) or parser(log, spec))


def failed(test: str, statuses: dict[str, str]) -> bool:
    """Upstream's ``test_failed``: missing from the log, or FAILED/ERROR."""
    return test not in statuses or statuses[test] in FAILING_STATUSES


def resolved(instance: SWEInstance, statuses: dict[str, str]) -> bool:
    """Upstream's FULL resolution: no FAIL_TO_PASS or PASS_TO_PASS test failed."""
    return not any(failed(t, statuses) for t in (*instance.fail_to_pass, *instance.pass_to_pass))


def suite(instance: SWEInstance, eval_script: str, *, image_env: dict[str, str] | None = None,
          timeout: int = 1800) -> SuiteProbe:
    """The per-state probe: the official script, with the target tests as members too.

    FAIL_TO_PASS tests fail at the base by definition, so they never enter the
    set of previously passing tests a break is measured against; naming them
    records how far the fix had got.
    """
    return SuiteProbe(
        name=f"p2p::{instance.instance_id}",
        command=suite_command(eval_script, image_env),
        members=(*instance.pass_to_pass, *instance.fail_to_pass),
        parse=functools.partial(parse_output, instance),
        timeout=timeout,
    )


def parity_check(sandbox, instance: SWEInstance) -> str | None:
    """The image's checkout is a git tree at /testbed: the one thing the replay relies on."""
    result = sandbox.exec("cd /testbed && git rev-parse --is-inside-work-tree", timeout=120)
    if result.exit_code != 0 or result.stdout.strip() != "true":
        return f"/testbed is not a git tree in {instance.image}: {(result.stderr or result.stdout)[-300:]}"
    return None


def grade_in_sandbox(sandbox, instance: SWEInstance, model_patch: str, *, eval_script: str,
                     image_env: dict[str, str] | None = None, timeout: int = 1800) -> GradeReport | None:
    """The official verdict on one final patch, in the instance's image.

    None when no verdict could be produced: the tree would not reset, or the
    script printed no markers at the base either (a dead environment, not a
    verdict). A patch that does not apply, or that leaves the suite printing
    nothing, is graded unresolved, as upstream grades it.
    """
    from taste.routing import prepare_container_tree

    workdir = sandbox.workdir
    target = prepare_container_tree(sandbox, workdir=workdir, hide_upstream=False)

    def reset() -> bool:
        return sandbox.exec(f"cd {workdir} && git checkout -q {target} -- . && git clean -qfd", timeout=300).exit_code == 0

    if not reset():
        return None
    apply_error = ""
    if model_patch.strip():
        sandbox.put_text("/tmp/taste_pred.diff", model_patch + "\n")
        applied = sandbox.exec(f"cd {workdir} && git apply -v /tmp/taste_pred.diff", timeout=300)
        if applied.exit_code != 0:
            apply_error = (applied.stderr or applied.stdout)[-400:]
    command = suite_command(eval_script, image_env)
    statuses: dict[str, str] = {}
    if not apply_error:
        run = sandbox.exec(command, timeout=timeout)
        if test_output(run.stdout) is None:
            if not reset():
                return None
            baseline = sandbox.exec(command, timeout=timeout)
            if test_output(baseline.stdout) is None:
                return None
        else:
            statuses = parse_output(instance, run.stdout)
    return GradeReport(
        instance_id=instance.instance_id,
        resolved=not apply_error and resolved(instance, statuses),
        fail_to_pass_passed=sum(1 for t in instance.fail_to_pass if statuses.get(t) in PASSING_STATUSES),
        fail_to_pass_total=len(instance.fail_to_pass),
        pass_to_pass_passed=sum(1 for t in instance.pass_to_pass if statuses.get(t) in PASSING_STATUSES),
        pass_to_pass_total=len(instance.pass_to_pass),
        per_test=statuses,
    )
