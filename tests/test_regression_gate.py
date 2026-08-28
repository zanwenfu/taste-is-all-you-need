"""The regression gate: roll back on what broke, never on a guessed check.

Nine of the rollback arm's eighteen failures were false rejections by a
planner-written check. These tests pin the gate's semantics with a fake
executor, and pin the wiring: an arm that claims the gate must use it, and
must refuse to run without a container to run the suite in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taste.benchmarks import swebench
from taste.benchmarks.swebench import END_MARKER, START_MARKER
from taste.config import HarnessConfig
from taste.execution import ExecResult
from taste.llm import InfraFailure
from taste.regression_gate import RegressionGate


def _instance() -> swebench.SWEInstance:
    return swebench.SWEInstance(
        instance_id="pytest-dev__pytest-1000", repo="pytest-dev/pytest",
        base_commit="0" * 40, problem_statement="", test_patch="diff", version="7.2",
        fail_to_pass=("testing/test_a.py::test_target",),
        pass_to_pass=("testing/test_a.py::test_old", "testing/test_a.py::test_other"),
    )


def _log(*lines: str) -> str:
    return "\n".join([START_MARKER, *lines, END_MARKER])


class _Runner:
    """Answers each suite invocation with the next scripted log."""

    def __init__(self, *logs):
        self.logs = list(logs)
        self.commands: list[str] = []

    def __call__(self, command: str, *, timeout: int = 0) -> ExecResult:
        self.commands.append(command)
        out = self.logs.pop(0)
        if out == "TIMEOUT":
            return ExecResult(124, "", "", timed_out=True)
        return ExecResult(0, out, "")


def test_baseline_records_what_passes_and_refuses_a_dead_suite() -> None:
    gate = RegressionGate(_instance(), _Runner(_log(
        "PASSED testing/test_a.py::test_old", "FAILED testing/test_a.py::test_other")))
    gate.establish_baseline()
    assert gate.baseline_pass == frozenset({"testing/test_a.py::test_old"})

    dead = RegressionGate(_instance(), _Runner("conda: command not found"))
    with pytest.raises(InfraFailure):
        dead.establish_baseline()


def test_a_step_is_rejected_only_for_a_genuine_regression() -> None:
    runner = _Runner(
        _log("PASSED testing/test_a.py::test_old", "PASSED testing/test_a.py::test_other"),
        _log("PASSED testing/test_a.py::test_old", "PASSED testing/test_a.py::test_other"),
        _log("FAILED testing/test_a.py::test_old", "PASSED testing/test_a.py::test_other"),
    )
    gate = RegressionGate(_instance(), runner)
    gate.establish_baseline()
    clean = gate.check()
    assert clean.passed, clean.reason
    broken = gate.check()
    assert not broken.passed
    assert "testing/test_a.py::test_old" in broken.evidence
    assert "1 previously-passing" in broken.reason


def test_a_test_that_failed_at_baseline_cannot_cause_a_rejection() -> None:
    """The gate never demands more than the starting tree delivered: a test
    that was already failing is not the agent's regression."""
    runner = _Runner(
        _log("PASSED testing/test_a.py::test_old", "FAILED testing/test_a.py::test_other"),
        _log("PASSED testing/test_a.py::test_old", "FAILED testing/test_a.py::test_other"),
    )
    gate = RegressionGate(_instance(), runner)
    gate.establish_baseline()
    assert gate.check().passed


def test_a_suite_that_no_longer_runs_is_a_regression_of_everything() -> None:
    """A tree that kills collection (the `$(cat …)` line) is what the grader
    scores as every test failed; the gate must reject it, not pass it for
    lack of evidence."""
    runner = _Runner(_log("PASSED testing/test_a.py::test_old"), "SyntaxError: invalid syntax")
    gate = RegressionGate(_instance(), runner)
    gate.establish_baseline()
    verdict = gate.check()
    assert not verdict.passed and "no longer runs" in verdict.reason


def test_a_timed_out_suite_is_conservative() -> None:
    runner = _Runner(_log("PASSED testing/test_a.py::test_old"), "TIMEOUT")
    gate = RegressionGate(_instance(), runner)
    gate.establish_baseline()
    assert not gate.check().passed


def test_the_gate_runs_the_repository_tests_without_the_hidden_patch() -> None:
    """The command must run the graded test files as they are, with no
    checkout of the base commit and no application of the benchmark's test
    patch — that patch is the hidden target and must never reach the agent's
    environment."""
    runner = _Runner(_log("PASSED testing/test_a.py::test_old"))
    gate = RegressionGate(_instance(), runner)
    gate.establish_baseline()
    cmd = runner.commands[0]
    assert "testing/test_a.py" in cmd
    assert "git checkout" not in cmd and "git apply" not in cmd and "TASTE_TEST_PATCH" not in cmd
    assert START_MARKER in cmd and END_MARKER in cmd


def test_django_directives_are_dotted_module_paths() -> None:
    assert swebench.directives_for("django/django", ["tests/expressions/tests.py"]) == ["expressions.tests"]
    assert swebench.directives_for("pytest-dev/pytest", ["testing/test_a.py"]) == ["testing/test_a.py"]


def test_the_gated_arm_is_a_different_harness() -> None:
    assert HarnessConfig.arm("A3reg").regression_gate is True
    assert HarnessConfig.arm("A3").regression_gate is False
    assert HarnessConfig.arm("A3reg").hash() != HarnessConfig.arm("A3").hash()
    assert "A3reg" in HarnessConfig.arm_names()


def test_the_kernel_uses_the_gate_instead_of_the_step_check(refactor_workspace: Path) -> None:
    """With a gate present, the step's own verification must not run: a
    command that would FAIL the step is ignored, and the gate's verdict
    decides. Reverse the gate's answer and the run must fail."""
    from taste.agent import AgentSpec
    from taste.cores import Plan, Step, Verification, WorkerResult
    from taste.kernel import Kernel

    class _Gate:
        def __init__(self, answer: bool):
            self.answer = answer
            self.baseline_pass = frozenset({"t::a"})
            self.calls = 0

        def establish_baseline(self) -> None:
            self.established = True

        def check(self):
            from taste.cores import MonitorResult
            self.calls += 1
            return MonitorResult(passed=self.answer, reason="fake gate")

    plan = Plan(task="gated", steps=[
        Step(id="step-01", description="noop",
             verification=Verification(kind="shell", command="false")),  # would fail
    ])
    spec = AgentSpec(name="g", description="", model=None, system_prompt="")

    def worker(step, _p):
        return WorkerResult(summary="noop", tool_calls=0, stopped_reason="end_turn")

    accept = _Gate(True)
    result = Kernel(workspace=refactor_workspace, max_retries=0, regression_gate=accept).run(
        task="g", spec=spec, plan_override=plan, worker_override=worker)
    assert result.status == "completed", "the step's own `false` check must not have run"
    assert accept.calls == 1

    reject = _Gate(False)
    result = Kernel(workspace=refactor_workspace, max_retries=0, regression_gate=reject).run(
        task="g", spec=spec, plan_override=plan, worker_override=worker)
    assert result.status == "failed"


def test_a_gated_arm_refuses_to_run_unrouted(tmp_path: Path) -> None:
    """Without a container the gate cannot run the suite, and falling back to
    the planner's check would test the wrong verifier under the right label."""
    from taste.benchmarks.swebench_run import CellContext, make_execute
    from taste.evalrun import Cell

    ctx = CellContext(instance=_instance(), config=HarnessConfig.arm("A3reg"),
                      workspace=tmp_path, gitdir=tmp_path / ".git" / "taste")
    with pytest.raises(RuntimeError, match="routed"):
        make_execute()(Cell(task="x", arm="A3reg", trial=1), ctx)


def test_member_files_come_from_the_ids_not_only_the_patch() -> None:
    pyt = _instance()
    assert swebench.member_test_files(pyt) == {"testing/test_a.py"}
    dj = swebench.SWEInstance(
        instance_id="django__django-1", repo="django/django", base_commit="0" * 40,
        problem_statement="", test_patch="", version="4.0", fail_to_pass=(),
        pass_to_pass=("test_x (expressions.tests.BasicExpressionsTests)",),
    )
    assert swebench.member_test_files(dj) == {"tests/expressions/tests.py"}



def test_the_half_split_is_deterministic_and_leaves_a_held_out_half() -> None:
    """The split-oracle arm answers the circularity objection only if the
    gate genuinely never sees the held-out files, and only if the split is
    reproducible across runs and arms."""
    inst = swebench.SWEInstance(
        instance_id="x__x-1", repo="pytest-dev/pytest", base_commit="0" * 40,
        problem_statement="", test_patch="", version="7.2", fail_to_pass=(),
        pass_to_pass=tuple(f"testing/test_{k}.py::test_it" for k in "abcdefgh"),
    )
    a = RegressionGate(inst, _Runner(), split="half").watched_files()
    b = RegressionGate(inst, _Runner(), split="half").watched_files()
    everything = RegressionGate(inst, _Runner(), split="all").watched_files()
    assert a == b, "the split must be deterministic"
    assert 0 < len(a) < len(everything), "half must hold something out and keep something"
    assert set(a) < set(everything)


def test_an_empty_watched_split_is_refused_not_passed() -> None:
    inst = swebench.SWEInstance(
        instance_id="x__x-2", repo="pytest-dev/pytest", base_commit="0" * 40,
        problem_statement="", test_patch="", version="7.2", fail_to_pass=(),
        pass_to_pass=("testing/test_only.py::test_it",),
    )
    gate = RegressionGate(inst, _Runner(_log("PASSED testing/test_only.py::test_it")), split="half")
    if not gate.watched_files():
        with pytest.raises(InfraFailure):
            gate.establish_baseline()
    else:
        gate.establish_baseline()
        assert gate.baseline_pass


def test_the_split_arm_is_a_different_harness() -> None:
    assert HarnessConfig.arm("A3reg2").gate_split == "half"
    assert HarnessConfig.arm("A3reg2").hash() != HarnessConfig.arm("A3reg").hash()
