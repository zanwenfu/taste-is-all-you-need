"""Descriptors must not accumulate across runs.

This is the guard for a defect that no single-run test can see. A ``Repo``
holds a ``cat-file --batch`` process pair and mmaps every pack it touches; one
leaked instance is invisible, and at the default limit of 1024 the *four
hundredth* cell of a sweep starts failing with ``Too many open files`` while
every earlier one passed.

What made it worth a dedicated test is the shape of the damage. ``run_sweep``
records such a cell as ``status="error"`` and drops it from the denominator,
so the loss lands entirely on the back half of a sweep -- ordered, not random,
and perfectly capable of looking like a result. It surfaced here only because
a fixture was made to assert on the sweep's status instead of using its output
blind; before that it read as an unrelated ``IsADirectoryError`` in pathlib.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taste.agent import AgentSpec
from taste.cores import Plan, Step, Verification, WorkerResult
from taste.kernel import Kernel
from taste.memory import Memory
from tests.conftest import PYTEST_CMD

FD_DIR = Path("/proc/self/fd")


def _open_fds() -> int:
    if not FD_DIR.exists():
        pytest.skip("descriptor counting needs /proc (Linux)")
    return len(list(FD_DIR.iterdir()))


def _spec() -> AgentSpec:
    return AgentSpec(name="fd", description="no LLM", model=None, system_prompt="")


def _plan() -> Plan:
    return Plan(
        task="touch a file",
        steps=[
            Step(
                id="step-01",
                description="append a line",
                verification=Verification(kind="shell", command=PYTEST_CMD),
            )
        ],
    )


def _worker(step: Step, plan: Plan) -> WorkerResult:
    return WorkerResult(summary="noop", tool_calls=0, stopped_reason="end_turn")


def test_repeated_runs_do_not_accumulate_descriptors(refactor_workspace: Path) -> None:
    """Ten runs in one process must not cost ten descriptors' worth of drift.

    The threshold is deliberately loose -- caches and lazily-opened logs make
    a few descriptors of steady-state growth normal. What it rules out is
    growth *proportional to the number of runs*, which is the only kind that
    kills a sweep.
    """
    ws = refactor_workspace
    warmup = Kernel(workspace=ws, max_retries=0)
    warmup.run(task="warmup", spec=_spec(), plan_override=_plan(), worker_override=_worker)

    before = _open_fds()
    for _ in range(10):
        Kernel(workspace=ws, max_retries=0).run(
            task="again", spec=_spec(), plan_override=_plan(), worker_override=_worker
        )
    growth = _open_fds() - before
    assert growth < 10, f"{growth} descriptors leaked across 10 runs"


def _working_session(ws: Path, name: str) -> Memory:
    """A Memory that has actually read objects.

    An idle ``Repo`` costs nothing measurable -- GitPython starts its
    ``cat-file --batch`` pair lazily, on first object access. A test that only
    constructs Memories therefore measures nothing and passes whether or not
    ``close()`` does anything at all. Checkpointing forces the handles open,
    which is the state a real run leaves them in.
    """
    memory = Memory.open_session(ws, name)
    (ws / f"{name}.txt").write_text(name)
    memory.checkpoint("step-01", f"work for {name}")
    return memory


def test_memory_releases_its_handles_on_close(refactor_workspace: Path) -> None:
    """The lifecycle exists and is not a no-op."""
    ws = refactor_workspace
    before = _open_fds()
    memories = [_working_session(ws, f"fd-{i}") for i in range(8)]
    peak = _open_fds()
    for memory in memories:
        memory.close()
    after = _open_fds()

    assert peak > before, "eight working sessions should cost descriptors"
    assert after - before < peak - before, "close() released nothing"


def test_memory_is_a_context_manager(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    before = _open_fds()
    for i in range(8):
        with _working_session(ws, f"ctx-{i}") as memory:
            assert memory.branch.endswith(f"ctx-{i}")
    assert _open_fds() - before < 8
