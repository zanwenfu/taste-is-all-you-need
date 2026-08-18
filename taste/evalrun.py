"""The sweep driver: many trials, resumable, and honest about what it cost.

A sweep is a grid of (task x arm x trial). Running one is mostly an exercise
in surviving interruption: a few hundred trials on a laptop over an API will
be stopped by a rate limit, a laptop lid, or a mistake, and the failure mode
that ruins a study is not losing work — it is *silently double-counting* the
work done twice.

So the unit of progress is a **cell**, keyed by (task, arm, trial), and the
ledger is the filesystem: one JSON file per completed cell. A cell that has a
file is done and is never re-run; a cell that does not is pending. Resume is
therefore the default behaviour rather than a mode, and it needs no state
beyond what is already on disk.

Three things this refuses to do, each because it would quietly corrupt the
resulting numbers:

* **Pool failure kinds.** A trial killed by a rate limit is not evidence about
  the agent. Infra and budget outcomes are recorded and excluded from success
  rates rather than counted as failures — which would bias against whichever
  arm makes more API calls, i.e. the one under test.
* **Share a workspace.** Every trial materializes its own copy. Two trials in
  one directory means one run's edits become another's starting state.
* **Report a cap as a cost.** What a trial was *allowed* to spend and what it
  *did* spend are different numbers, and only the second is data.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from taste.config import HarnessConfig, kernel_kwargs
from taste.kernel import Kernel, RunResult

CellStatus = str  # "completed" | "failed" | "infra" | "budget" | "error"


@dataclass(frozen=True)
class Cell:
    """One trial: a task, an arm, a repetition index."""

    task: str
    arm: str
    trial: int

    @property
    def key(self) -> str:
        return f"{self.task}__{self.arm}__t{self.trial}"


@dataclass
class CellResult:
    """What one trial produced. Written once, then never recomputed."""

    task: str
    arm: str
    trial: int
    status: CellStatus
    config_hash: str
    session_id: str = ""
    final_sha: str = ""
    steps_passed: int = 0
    steps_total: int = 0
    attempts: int = 0
    rollbacks: int = 0
    score: float | None = None
    """The primary endpoint — fractional, filled in by the scorer."""
    billed_usd: float = 0.0
    work_usd: float = 0.0
    """Both currencies. A cap is enforced on work; billed is what was paid."""
    cache_delta_usd: float = 0.0
    elapsed_s: float = 0.0
    failure_reason: str | None = None
    split_id: str = ""
    error: str | None = None

    # --- where this cell's evidence lives.
    # Without these the ledger records a number but not the artifacts behind
    # it, and the replay stage cannot find shadow.jsonl from the ledger alone
    # — which makes the primary outcome unrecomputable from what was written.
    workspace: str = ""
    gitdir: str = ""
    session_branch: str = ""
    shadow_ref: str = ""
    report_path: str = ""
    """Sidecar JSON: the full verdict matrix and silence report. A scalar
    score is not the dependent variable; contamination events are."""

    attempts_made: int = 1
    """How many times this cell has been executed, including retries after
    infrastructure faults. Reported as attrition, never hidden."""
    ts: float = field(default_factory=time.time)

    @property
    def counts_toward_success(self) -> bool:
        """Whether this trial is evidence about the recovery policy.

        Budget exhaustion COUNTS, as a failure — intention-to-treat. Running
        out of money is precisely what an expensive policy costs, so dropping
        those runs would flatter whichever arm spends most, which is the arm
        under test. Only infrastructure faults and harness errors are
        excluded, because those say nothing about any arm.
        """
        return self.status in ("completed", "failed", "budget")

    @property
    def retryable(self) -> bool:
        """Whether re-running this cell could produce evidence.

        An infra fault or a harness crash is not an outcome; leaving it on
        disk as a finished cell would permanently shrink the sample. A budget
        or task failure IS an outcome and must never be retried — retrying
        until an arm succeeds is the purest form of selecting on the
        dependent variable.
        """
        return self.status in ("infra", "error")


@dataclass
class SweepReport:
    results: list[CellResult] = field(default_factory=list)
    skipped: int = 0
    """Cells already on disk — resumed, not re-run."""

    def by_arm(self) -> dict[str, list[CellResult]]:
        out: dict[str, list[CellResult]] = {}
        for r in self.results:
            out.setdefault(r.arm, []).append(r)
        return out

    def summary(self) -> str:
        """Per-arm description with attrition explicit.

        Deliberately not a bare cross-arm mean: an unweighted mean over
        whatever survived listwise deletion hides both how many trials each
        arm lost and why, and it is the first thing a reviewer asks for.
        Inference belongs in :mod:`taste.stats`, not here.
        """
        from taste.stats import summarise_arm

        lines = [f"{len(self.results)} cells run, {self.skipped} resumed from disk"]
        for arm, records in sorted(self.by_arm().items()):
            lines.append(summarise_arm(arm, records).render())
        lines.append(
            "note: means are descriptive only; the pre-registered contrasts "
            "are computed by taste.stats on paired blocks."
        )
        return "\n".join(lines)


class Ledger:
    """One file per completed cell. The filesystem is the state."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, cell: Cell) -> Path:
        return self.root / f"{cell.key}.json"

    def done(self, cell: Cell, *, retry_budget: int = 0) -> bool:
        """Whether this cell is finished for good.

        A recorded infra fault or crash is retryable up to a declared budget:
        treating it as done would silently shrink the sample, and the number
        of retries spent is itself reportable attrition.
        """
        result = self.read(cell)
        if result is None:
            return False
        return not (result.retryable and result.attempts_made <= retry_budget)

    def read(self, cell: Cell) -> CellResult | None:
        path = self.path_for(cell)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
        except ValueError:
            return None
        known = {f for f in CellResult.__dataclass_fields__}
        return CellResult(**{k: v for k, v in raw.items() if k in known})

    def write(self, result: CellResult) -> None:
        """Atomic: a partial file on interruption would read as a done cell."""
        cell = Cell(result.task, result.arm, result.trial)
        target = self.path_for(cell)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(result), indent=2, sort_keys=True))
        tmp.replace(target)

    def all_results(self) -> list[CellResult]:
        out = []
        for path in sorted(self.root.glob("*.json")):
            try:
                raw = json.loads(path.read_text())
            except ValueError:
                continue
            known = {f for f in CellResult.__dataclass_fields__}
            out.append(CellResult(**{k: v for k, v in raw.items() if k in known}))
        return out


def cells(tasks: list[str], arms: list[str], trials: int) -> Iterator[Cell]:
    """The grid, ordered so arms interleave within a task.

    Deliberately not arm-major: running every A1 trial and then every A3 trial
    aligns condition with calendar time, so a model update or a quiet API day
    lands entirely on one arm. Interleaving spreads that noise across all of
    them.
    """
    for task in tasks:
        for trial in range(1, trials + 1):
            for arm in arms:
                yield Cell(task=task, arm=arm, trial=trial)


def run_sweep(
    *,
    tasks: list[str],
    arms: list[str],
    trials: int,
    ledger_dir: Path,
    prepare: Callable[[Cell], Any],
    execute: Callable[[Cell, Any], RunResult],
    score: Callable[[Cell, Any, RunResult], float | None] | None = None,
    on_cell: Callable[[CellResult], None] | None = None,
    retry_budget: int = 0,
) -> SweepReport:
    """Run the grid, skipping cells already on disk.

    ``prepare`` materializes an isolated workspace for a cell and returns
    whatever ``execute`` and ``score`` need. Splitting it out keeps the
    per-trial isolation guarantee in one place rather than scattered through
    the driver.
    """
    ledger = Ledger(ledger_dir)
    report = SweepReport()
    notify = on_cell or (lambda _r: None)

    for cell in cells(tasks, arms, trials):
        if ledger.done(cell, retry_budget=retry_budget):
            report.skipped += 1
            continue
        previous = ledger.read(cell)
        attempt_number = (previous.attempts_made + 1) if previous else 1

        started = time.time()
        try:
            context = prepare(cell)
            result = execute(cell, context)
            value = score(cell, context, result) if score else None
            record = _record_from(cell, result, value, context)
            record.attempts_made = attempt_number
        except Exception as exc:
            # A crash is data too — recorded so the cell is not silently
            # retried forever, and excluded from success rates.
            record = CellResult(
                task=cell.task,
                arm=cell.arm,
                trial=cell.trial,
                status="error",
                config_hash="",
                elapsed_s=round(time.time() - started, 2),
                error=f"{type(exc).__name__}: {exc}",
                attempts_made=attempt_number,
            )
            record.failure_reason = traceback.format_exc(limit=3)

        ledger.write(record)
        report.results.append(record)
        notify(record)

    return report


def _record_from(cell: Cell, result: RunResult, score: float | None, context: Any) -> CellResult:
    stats = result.stats
    status: CellStatus = result.status
    if result.failure_kind in ("infra", "budget"):
        status = result.failure_kind

    return CellResult(
        task=cell.task,
        arm=cell.arm,
        trial=cell.trial,
        status=status,
        config_hash=getattr(getattr(context, "config", None), "hash", lambda: "")(),
        session_id=result.session_id,
        final_sha=result.final_sha,
        steps_passed=sum(1 for o in result.outcomes if o.verdict.passed),
        steps_total=len(result.plan.steps),
        attempts=sum(o.attempts for o in result.outcomes),
        rollbacks=sum(1 for o in result.outcomes if o.rolled_back),
        score=score,
        billed_usd=round(stats.total_cost_usd, 6) if stats else 0.0,
        work_usd=round(stats.total_work_usd, 6) if stats else 0.0,
        cache_delta_usd=round(stats.cache_delta_usd, 6) if stats else 0.0,
        elapsed_s=result.elapsed_seconds,
        failure_reason=result.failure_reason,
        split_id=getattr(context, "split_id", "") or "",
        workspace=str(getattr(context, "workspace", "") or ""),
        gitdir=str(getattr(context, "gitdir", "") or ""),
        session_branch=result.session_id,
        shadow_ref=str(getattr(context, "shadow_ref", "") or ""),
        report_path=str(getattr(context, "report_path", "") or ""),
    )


def kernel_for(arm: str, workspace: Path, llm: Any, **overrides: Any) -> Kernel:
    """A kernel configured for one arm. One place, so arms cannot drift."""
    config = HarnessConfig.arm(arm, **overrides)
    return Kernel(workspace=workspace, llm=llm, **kernel_kwargs(config), config=config)
