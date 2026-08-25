"""Hermetic walkthrough of the refactor demo — no API key required.

Runs the Kernel against the demo workspace with a scripted worker that
reproduces the ``step-87`` failure mode:

    step-01  ->  benign edit, tests green
    step-02  ->  *first attempt introduces a regression*
                 -> Monitor catches it via pytest
                 -> Kernel rolls back to the pre-step-02 commit
                 -> second attempt fixes it, tests green
    step-03  ->  benign edit, tests green

Use this to see the harness working end-to-end before you wire up a real
model. The output is what the CI-stable test asserts on.

    python examples/refactor_demo/simulate.py
"""

from __future__ import annotations

import shlex
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.refactor_demo.bootstrap import bootstrap
from taste.agent import AgentSpec
from taste.cli import _print_event, _print_result
from taste.cores import Plan, Step, Verification, WorkerResult
from taste.kernel import Kernel


BROKEN_STEP2 = """\
def run(items):
    total = 0
    for x in items:
        if x > 0:
            total += x  # REGRESSION: should be x * 2
        else:
            total -= x
    return total


def fmt(total):
    return f"total is {total}"


def main(items):
    return fmt(run(items))
"""

CORRECT_STEP2 = """\
def run(items: list[int]) -> int:
    total = 0
    for x in items:
        if x > 0:
            total += x * 2
        else:
            total -= x
    return total


def fmt(total: int) -> str:
    return f"total is {total}"


def main(items: list[int]) -> str:
    return fmt(run(items))
"""


def main() -> None:
    workspace = bootstrap(Path(tempfile.mkdtemp(prefix="taste-demo-")))
    math = workspace / "legacy_math.py"
    attempts: dict[str, int] = {}

    def scripted_worker(step: Step, plan: Plan) -> WorkerResult:
        attempts[step.id] = attempts.get(step.id, 0) + 1
        n = attempts[step.id]
        if step.id == "step-01":
            math.write_text(math.read_text() + "\n# refactor: legacy_math\n")
        elif step.id == "step-02":
            math.write_text(BROKEN_STEP2 if n == 1 else CORRECT_STEP2)
        elif step.id == "step-03":
            math.write_text(math.read_text().rstrip() + "\n")
        return WorkerResult(summary=f"{step.id} attempt {n}", tool_calls=0, stopped_reason="end_turn")

    check = Verification(kind="shell", command=f"{shlex.quote(sys.executable)} -m pytest -q")
    plan = Plan(
        task="refactor legacy_math.py preserving behavior",
        steps=[
            Step(id="step-01", description="annotate module with a header comment", verification=check),
            Step(id="step-02", description="add type hints to run / fmt / main", verification=check),
            Step(id="step-03", description="normalize trailing whitespace", verification=check),
        ],
    )

    kernel = Kernel(
        workspace=workspace,
        max_retries=2,
        on_event=_print_event,
    )
    result = kernel.run(
        task=plan.task,
        spec=AgentSpec(
            name="scripted_refactor",
            description="scripted demo worker — no LLM calls",
            model=None,
            system_prompt="",
        ),
        plan_override=plan,
        worker_override=scripted_worker,
    )
    _print_result(result)

    print()
    print(f"workspace: {workspace}")
    print(f"inspect:   git -C {workspace} log {result.branch} --oneline")


if __name__ == "__main__":
    main()
