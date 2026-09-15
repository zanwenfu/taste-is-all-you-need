"""One goal, driven to a recorded ending by ``run()``, against real models.

The rest of the suite proves the seams with scripted transports: a planner
that returns a canned proposal, a launcher that starts ``sleep``, a judge that
answers from a script. Each half is exercised honestly and the whole path is
not, which is exactly where nine consecutive live failures hid -- among them a
budgeted worker that could never finish, because the CLI bills a little Haiku
for a session title and the accounting check read that as an identity
violation, and a terminal certification that failed closed on every budgeted
run because the runtime appends its own ledger entry after the monitor stops
looking.

So this is the claim the suite structurally cannot make: a real planner
decomposes, a real worker executes in its own process on its own worktree, a
real monitor judges beside it, delivery projects the artifact into the
integration branch, and ``run()`` stops on a promoted complete plan and writes
a ``GoalOutcome`` whose assessment accounts for every standing criterion.

**It spends real money and is off by default.** Set ``TASTE_LIVE_E2E=1`` to run
it; roughly $0.25 and a minute against the pinned models. CI runs a bare
``pytest``, so the gate is a whole-file skip rather than a marker -- a marker
would still be collected and would still spend.

Budgeted deliberately. A budgeted goal forces per-assignment worker and monitor
caps, and that is the path the central planner actually validates; an
unbudgeted run would skip the accounting seam these failures lived in.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("claude_agent_sdk", reason="the brain layer needs claude-agent-sdk")

from taste.brains.central_host import compose_central_runtime
from taste.brains.central_planner import Goal

pytestmark = pytest.mark.skipif(
    os.environ.get("TASTE_LIVE_E2E") != "1",
    reason="live end-to-end run against real models; set TASTE_LIVE_E2E=1",
)

# Per model, and not a round number by accident: a brain refuses to start
# unless its cap clears twice its one-call exposure, so a worker on the pinned
# Sonnet needs more than $5 and its monitor more than $2.50. A goal budget that
# cannot cover both never spawns anything, and the failure surfaces as an
# unprovable budget rather than as a missing cap.
GOAL_BUDGET_USD = 40.0


def test_a_budgeted_goal_reaches_a_recorded_ending(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    goal = Goal(
        goal_id="live-adder",
        task=(
            "Create a single file adder.py containing exactly one function: "
            "def add(a, b): return a + b -- nothing else, no tests, no __main__."
        ),
        success_criteria=(
            "adder.py exists in the integration branch",
            "adder.py defines add(a, b) returning a + b",
        ),
        budget_usd=GOAL_BUDGET_USD,
    )

    host = compose_central_runtime(root, "live-1", goal)
    try:
        outcome = host.run(max_generations=4, wall_clock_seconds=600.0)

        assert outcome.stop_reason == "complete", (
            f"stopped on {outcome.stop_reason!r}: {outcome.detail}"
        )
        assert outcome.complete
        assert outcome.completion_reason.strip(), "a completion must say why"
        assert outcome.delivered_assignment_ids

        # Completion is an accounting claim, not an assertion: every standing
        # obligation is named and met, with evidence attached to each.
        assert outcome.criteria is not None
        standing = {item.criterion_id for item in outcome.criteria.criteria}
        assessed = {item["criterion_id"] for item in outcome.assessment}
        assert assessed == standing, "a complete plan must assess every standing criterion"
        assert all(item["verdict"] == "met" for item in outcome.assessment)
        assert all(str(item["evidence"]).strip() for item in outcome.assessment)

        # The product is in the integration branch, not merely claimed.
        delivered = host.integration.head.read("adder.py")
        assert delivered is not None, "adder.py never reached the integration branch"
        assert "def add(a, b)" in delivered

        # And the ending is durable: a restart reads the same answer from git.
        assert host.outcome() == outcome

        # Real spend was accounted for. Zero worker cost would mean the worker
        # never billed, which is how several of the earlier failures looked.
        assert outcome.budget.known_spent_usd > 0
        assert outcome.budget.worker_spent_usd > 0
        assert outcome.budget.known_spent_usd <= GOAL_BUDGET_USD
    finally:
        host.close()
