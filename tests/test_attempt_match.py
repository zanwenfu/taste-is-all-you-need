"""The attempt-matched control, and the ways it could quietly not be one.

A3' exists to answer "A3 only wins because it re-samples more". It answers it
only if the matching actually binds. Every test here is written against a
specific way the matching could be configured, reported, and inert -- which
is the failure mode this project keeps finding, and the reason none of these
assert merely that the code ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taste import recovery
from taste.agent import AgentSpec
from taste.attempts import RetryPool, harvest_by_instance, harvest_retries, retries_in
from taste.config import HarnessConfig
from taste.cores import Plan, Step, Verification, WorkerResult
from taste.kernel import Event, Kernel
from tests.conftest import PYTEST_CMD

LEGACY_MATH = "legacy_math.py"

BROKEN = """\
def run(items):
    return sum(items)  # REGRESSION


def fmt(total):
    return f"total is {total}"


def main(items):
    return fmt(run(items))
"""

CORRECT = """\
def run(items):
    total = 0
    for x in items:
        if x > 0:
            total += x * 2
        else:
            total -= x
    return total


def fmt(total):
    return f"total is {total}"


def main(items):
    return fmt(run(items))
"""


def _spec() -> AgentSpec:
    return AgentSpec(name="matched", description="no LLM", model=None, system_prompt="")


def _plan() -> Plan:
    check = Verification(kind="shell", command=PYTEST_CMD)
    return Plan(
        task="two steps, both of which fail at least once",
        steps=[
            Step(id="step-01", description="fails once, then succeeds", verification=check),
            Step(id="step-02", description="never succeeds", verification=check),
        ],
    )


def _worker_for(ws: Path):
    """step-01 fails once then recovers; step-02 is broken forever.

    Chosen so the two steps compete for one pool. A worker where only one step
    ever retried could not distinguish a run-level allowance from a per-step
    one, and would pass against either.
    """
    math = ws / LEGACY_MATH
    seen: dict[str, int] = {}

    def worker(step: Step, plan: Plan) -> WorkerResult:
        seen[step.id] = seen.get(step.id, 0) + 1
        n = seen[step.id]
        if step.id == "step-01":
            math.write_text(BROKEN if n == 1 else CORRECT)
        else:
            math.write_text(BROKEN)
        return WorkerResult(summary=f"{step.id}#{n}", tool_calls=0, stopped_reason="end_turn")

    return worker


def _retries(result) -> int:
    return sum(o.attempts - 1 for o in result.outcomes)


# ============================================================ the pool binds


def test_unpooled_run_establishes_the_baseline(refactor_workspace: Path) -> None:
    """Without a pool: step-01 retries once, step-02 exhausts its own ceiling."""
    ws = refactor_workspace
    kernel = Kernel(workspace=ws, max_retries=2)
    result = kernel.run(
        task="baseline", spec=_spec(), plan_override=_plan(), worker_override=_worker_for(ws)
    )
    by_step = {o.step.id: o.attempts for o in result.outcomes}
    assert by_step == {"step-01": 2, "step-02": 3}
    assert _retries(result) == 3


def test_pool_binds_across_steps_not_within_one(refactor_workspace: Path) -> None:
    """The whole point: step-01's retry leaves step-02 with fewer.

    With a per-step cap of 2 retries, step-02 would take 2 regardless of what
    step-01 did. It takes 1, because step-01 already spent one from a shared
    allowance. This assertion is what a per-step implementation fails.
    """
    ws = refactor_workspace
    pool = RetryPool(total=2)
    kernel = Kernel(workspace=ws, max_retries=2, retry_pool=pool)
    result = kernel.run(
        task="matched", spec=_spec(), plan_override=_plan(), worker_override=_worker_for(ws)
    )
    by_step = {o.step.id: o.attempts for o in result.outcomes}
    assert by_step == {"step-01": 2, "step-02": 2}
    assert _retries(result) == 2, "the run must use exactly the allowance, no more"
    assert pool.spent == 2 and pool.exhausted


def test_pool_larger_than_demand_changes_nothing(refactor_workspace: Path) -> None:
    """A matched arm is capped, never forced to spend. Guards against a pool
    that pads runs up to its total and inflates the very quantity it controls."""
    ws = refactor_workspace
    pool = RetryPool(total=99)
    kernel = Kernel(workspace=ws, max_retries=2, retry_pool=pool)
    result = kernel.run(
        task="slack", spec=_spec(), plan_override=_plan(), worker_override=_worker_for(ws)
    )
    assert {o.step.id: o.attempts for o in result.outcomes} == {"step-01": 2, "step-02": 3}
    assert pool.spent == 3


def test_zero_pool_allows_first_attempts_but_no_retries(refactor_workspace: Path) -> None:
    """A paired run that never retried yields a pool of zero. Every step must
    still get its one structural attempt, or the matched arm answers a
    different question by silently truncating its own plan."""
    ws = refactor_workspace
    pool = RetryPool(total=0)
    kernel = Kernel(workspace=ws, max_retries=2, retry_pool=pool)
    result = kernel.run(
        task="none", spec=_spec(), plan_override=_plan(), worker_override=_worker_for(ws)
    )
    assert result.outcomes, "a zero pool must not stop the run from starting"
    assert all(o.attempts == 1 for o in result.outcomes)
    assert pool.spent == 0


# ============================================ exhaustion keeps arm discipline


def test_exhausted_pool_halts_with_the_arm_s_reset_discipline(refactor_workspace: Path) -> None:
    """Running out of tries is not a licence to leave unverified work behind.

    A resetting arm discards its last failed attempt even when the reason it
    stopped was the allowance rather than its own ceiling.
    """
    ws = refactor_workspace
    events: list[Event] = []
    pool = RetryPool(total=1)
    kernel = Kernel(
        workspace=ws,
        retry_pool=pool,
        on_event=events.append,
        config=HarnessConfig.arm("A3"),
    )
    result = kernel.run(
        task="discipline", spec=_spec(), plan_override=_plan(), worker_override=_worker_for(ws)
    )
    last = result.outcomes[-1]
    assert last.step.id == "step-02"
    assert last.rolled_back is True, "the resetting arm must still reset on exhaustion"
    reasons = [e.payload.get("reason", "") for e in events if e.kind == "recovery.action"]
    assert any("exhaust" in r for r in reasons), reasons


def test_pool_binds_even_with_recovery_disabled(refactor_workspace: Path) -> None:
    """The legacy fault path never consults Budget. A pool configured there
    and ignored would be reported as matching while matching nothing."""
    ws = refactor_workspace
    pool = RetryPool(total=1)
    kernel = Kernel(workspace=ws, max_retries=2, retry_pool=pool, recovery_config=None)
    result = kernel.run(
        task="legacy", spec=_spec(), plan_override=_plan(), worker_override=_worker_for(ws)
    )
    assert _retries(result) == 1
    assert pool.exhausted


def test_reverify_costs_an_action_but_not_a_retry(refactor_workspace: Path) -> None:
    """Re-running the Monitor re-samples nothing, so it must not draw down an
    allowance whose entire purpose is to hold sampling fixed."""
    ws = refactor_workspace
    pool = RetryPool(total=5)
    cfg = recovery.RecoveryConfig(
        enabled=True, policy="fixed", fixed_action=recovery.ActionKind.REVERIFY
    )
    kernel = Kernel(workspace=ws, max_retries=2, retry_pool=pool, recovery_config=cfg)
    kernel.run(
        task="reverify", spec=_spec(), plan_override=_plan(), worker_override=_worker_for(ws)
    )
    assert pool.spent == 0, "re-verification consumed the sampling allowance"


def test_spend_equals_what_harvest_would_report(refactor_workspace: Path) -> None:
    """The allowance and the measurement must be the same unit.

    A3' is matched by harvesting A3's retries from its event log and handing
    that number back as a pool. If the pool were charged for retries the loop
    refuses, or the harvest counted something the pool does not, the control
    would be matched against a number no run ever produced -- and it would
    still report itself as matched. This is the assertion that makes the two
    definitions impossible to drift apart.
    """
    ws = refactor_workspace
    events: list[Event] = []
    pool = RetryPool(total=99)
    kernel = Kernel(workspace=ws, max_retries=2, retry_pool=pool, on_event=events.append)
    result = kernel.run(
        task="unit", spec=_spec(), plan_override=_plan(), worker_override=_worker_for(ws)
    )
    as_dicts = [{"type": e.kind, **e.payload} for e in events]
    assert retries_in(as_dicts) == pool.spent
    assert pool.spent == _retries(result)


# ================================================================ harvesting


def test_retries_in_counts_only_worker_attempts() -> None:
    events = [
        {"type": "step.begin", "attempt": 1},
        {"type": "step.begin", "attempt": 2},
        {"type": "step.begin", "attempt": 3},
        {"type": "step.begin", "attempt": 1},
        {"type": "monitor.verdict", "attempt": 7},
        {"type": "recovery.action", "attempt": 4},
    ]
    assert retries_in(events) == 2


def test_retries_in_ignores_malformed_attempt_fields() -> None:
    assert retries_in([{"type": "step.begin", "attempt": "2"}]) == 0
    assert retries_in([{"type": "step.begin"}]) == 0


def test_harvest_skips_a_truncated_final_line(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    log.write_text(
        json.dumps({"type": "step.begin", "attempt": 2})
        + "\n"
        + json.dumps({"type": "step.begin", "attempt": 3})
        + "\n"
        + '{"type": "step.beg'
    )
    assert harvest_retries(log) == 2


def test_harvest_by_instance_omits_missing_logs_rather_than_zeroing(tmp_path: Path) -> None:
    """Zero retries and no record are different facts. Collapsing them hands
    the matched arm an allowance of zero exactly where the paired run failed
    to record -- a silent way to make the control look worse than it is."""
    for name, retries in (("inst-a", 2), ("inst-b", 0)):
        d = tmp_path / name / ".git" / "taste"
        d.mkdir(parents=True)
        (d / "events.jsonl").write_text(
            "".join(
                json.dumps({"type": "step.begin", "attempt": i + 2}) + "\n" for i in range(retries)
            )
        )
    (tmp_path / "inst-c").mkdir()

    found = harvest_by_instance(tmp_path)
    assert found == {"inst-a": 2, "inst-b": 0}
    assert "inst-c" not in found


def test_negative_pool_is_rejected() -> None:
    with pytest.raises(ValueError):
        RetryPool(total=-1)
