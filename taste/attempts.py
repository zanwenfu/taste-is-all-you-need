"""Attempt matching: separating "reset helps" from "more tries help".

The first objection any reviewer raises about a rollback arm is that it is
not one intervention but two. A3 resets the tree *and* re-samples the model;
A2 does neither. If A3 leaves less collateral damage, the credit could belong
entirely to the extra sampling, and nothing about rollback would be shown.

A3' is the control that closes this. It retries with the same guidance and
never resets, and it is allowed exactly as many retries as its paired A3 run
actually used on the same instance. A3 vs A3' is then a clean contrast in one
variable -- the reset -- because the sampling is held fixed by construction.

Two decisions are load-bearing and neither is obvious.

**The pool counts retries, not attempts.** A step's first attempt is
structural: without it the step does no work at all, and a matched arm that
truncated its own plan would be answering a different question. The quantity
the objection is about is the *extra* tries, so that is the quantity matched.
Pool = sum over steps of (attempts - 1).

**The pool is per run, not per step.** Matching per step would require the
two arms to produce the same plan for the same instance, and they do not have
to -- the planner is sampled, and a reset changes what the planner sees next.
Attributing a per-step budget across plans that disagree about what the steps
are would be a false precision. A run-level pool needs no such assumption; it
gives the arm the same number of shots and lets it spend them where its own
plan calls for. How the two arms actually allocated those retries across
steps is reported, not assumed.

The pool is a *runtime* limit, deliberately not part of HarnessConfig: it
takes a different value on every instance, and folding it into the config
would give one arm a different config_hash per instance and destroy the
property that an arm is one configuration.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = ["RetryPool", "harvest_by_instance", "harvest_retries", "retries_in"]


@dataclass
class RetryPool:
    """A run-level allowance of retries, consumed as steps use them."""

    total: int
    spent: int = 0

    def __post_init__(self) -> None:
        if self.total < 0:
            raise ValueError(f"retry pool cannot be negative: {self.total}")

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.spent)

    @property
    def exhausted(self) -> bool:
        return self.remaining == 0

    def spend(self, n: int = 1) -> None:
        self.spent += n


def retries_in(events: Iterable[Mapping[str, object]]) -> int:
    """How many retries a recorded run actually used.

    Counted from ``step.begin``, which the kernel emits once per worker
    attempt and *not* on a re-verification -- re-running the Monitor costs an
    action but no sampling, so it must not consume a matched arm's pool. Any
    attempt past the first is a retry.
    """
    total = 0
    for event in events:
        if event.get("type") != "step.begin":
            continue
        attempt = event.get("attempt")
        if isinstance(attempt, int) and attempt >= 2:
            total += 1
    return total


def harvest_retries(events_path: Path) -> int:
    """Read one run's event log and total its retries.

    A truncated final line is ignored rather than fatal: a log being appended
    to while it is read is normal, and a half-written record is not evidence
    of anything.
    """
    events: list[Mapping[str, object]] = []
    with open(events_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                events.append(record)
    return retries_in(events)


def harvest_by_instance(root: Path) -> dict[str, int]:
    """Map instance id -> retries used, over a completed sweep's cells.

    A cell whose event log is missing is omitted rather than recorded as
    zero. Zero is a real measurement -- a run that never retried -- and a
    missing log is the absence of one; collapsing them would hand the matched
    arm a pool of zero on exactly the instances where the paired run failed
    to record, which is a silent way to make it look worse.
    """
    found: dict[str, int] = {}
    for events_path in sorted(Path(root).glob("*/.git/taste/events.jsonl")):
        instance = events_path.parent.parent.parent.name
        found[instance] = harvest_retries(events_path)
    return found
