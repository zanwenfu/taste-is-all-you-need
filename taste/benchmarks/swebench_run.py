"""One SWE-bench instance, end to end: run it, then reconstruct what was true.

This is the seam between the harness and the measurement. Everything either
side of it was already built and tested; what was missing was the code that
actually connects them, so nothing had ever produced a contamination number
from a real run.

The shape is the three callables :func:`taste.evalrun.run_sweep` expects, and
the ordering inside them is load-bearing:

1. **A fresh workspace per cell.** Not an optimisation to skip. The event log
   is truncated at every run, so a reused workspace destroys the previous
   cell's evidence; and two trials sharing a tree makes one run's edits the
   next run's starting state, which breaks the pairing the analysis rests on.

2. **The run is observed, not interrogated.** Shadow commits are written as
   the agent works. Nothing in this module asks the agent anything, and
   nothing it does is visible to the agent.

3. **Reconstruction happens afterwards, out of band**, against the pinned
   image. This is where the cost lives, and it is CPU rather than tokens.

The dependent variable is the *episode list*, not the score. ``run_sweep``
wants a float, so ``score`` returns one — but the float is the official
resolve verdict, and everything the paper actually claims is in the sidecar
JSON written next to the ledger entry.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from taste.agent import AgentSpec
from taste.attribution import (
    CoverageMap,
    attribution_map,
    failed_at,
    harness_failures,
    read_events,
    summarise_silence,
)
from taste.benchmarks import swebench
from taste.config import HarnessConfig, kernel_kwargs
from taste.evalrun import Cell
from taste.execution import SandboxProvider
from taste.kernel import Kernel, RunResult
from taste.memory import Memory
from taste.replay import SandboxProbeExecutor, reconstruct
from taste.shadow import SHADOW_HEAD, load_timeline


@dataclass
class CellContext:
    """Everything one cell needs, and everywhere its evidence landed.

    A real dataclass rather than a bag: the driver reads ``config.hash()``
    through ``getattr`` with a silent default, so an untyped context that
    happened to lack ``.config`` would record an empty hash and the manifest
    could not distinguish two arms.
    """

    instance: swebench.SWEInstance
    config: HarnessConfig
    workspace: Path
    gitdir: Path
    split_id: str = ""
    budget_usd: float = 1.50
    shadow_ref: str = ""
    report_path: str = ""
    probe_coverage: CoverageMap | None = None
    monitor_coverage: CoverageMap | None = None
    provider: SandboxProvider | None = None
    session: str = ""


@dataclass
class CellEvidence:
    """The sidecar. What a scalar score cannot carry."""

    instance_id: str
    arm: str
    trial: int
    session: str
    observations: int
    episodes: list[dict[str, Any]] = field(default_factory=list)
    never_passed: list[str] = field(default_factory=list)
    unknown_transitions: int = 0
    replays: int = 0
    monitor_failures: int = 0
    monitor_failures_unindexed: int = 0
    """Failures on a tree that produced no observation. Reported rather than
    dropped: they are real detections at points the timeline cannot index."""
    silence: dict[str, Any] = field(default_factory=dict)
    resolved: bool | None = None

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        return path


def make_prepare(
    *,
    instances: dict[str, swebench.SWEInstance],
    root: Path,
    budget_usd: float = 1.50,
    source_root: Path | None = None,
    repo_cache: Path | None = None,
    coverage: dict[str, tuple[CoverageMap, CoverageMap]] | None = None,
    provider: SandboxProvider | None = None,
    observe_tools: bool = False,
):
    """Build the ``prepare`` callable for a sweep over these instances.

    ``repo_cache`` switches materialization to the real repository at
    ``base_commit``, cloned once per repo and reused. ``source_root`` is the
    fixture path, for tests. Neither means an empty workspace, which is only
    ever right for a synthetic task.
    """

    def prepare(cell: Cell) -> CellContext:
        instance = instances[cell.task]
        # Fresh per cell — see the module docstring; this is not reusable.
        workspace = Path(root) / cell.task / cell.arm / f"t{cell.trial}"
        if repo_cache is not None:
            swebench.materialize_from_repo(instance, workspace, cache=Path(repo_cache))
        else:
            source = Path(source_root) / instance.instance_id if source_root else None
            swebench.materialize(instance, workspace, source=source)

        # max_parallel=1 for every measured arm. The shadow log is bound to
        # the primary session memory, so a parallel worker's edits are not
        # visible to the observation stamped with that worker's step -- which
        # would make the recorded file set wrong exactly where attribution
        # reads it.
        config = HarnessConfig.arm(
            cell.arm, max_parallel=1, observe_tools=observe_tools
        )
        probe_cov, monitor_cov = (coverage or {}).get(instance.instance_id, (None, None))
        return CellContext(
            instance=instance,
            config=config,
            workspace=workspace,
            gitdir=workspace / ".git" / "taste",
            split_id=instance.repo_short,
            budget_usd=budget_usd,
            probe_coverage=probe_cov,
            monitor_coverage=monitor_cov,
            # No fallback provider. Defaulting to a local sandbox would point
            # probes at an empty directory, every execution would fail to
            # start, and the resulting all-holes timeline reads as a clean
            # run rather than as a broken one. Absent means "replay in a
            # worktree"; the image must be passed deliberately.
            provider=provider,
        )

    return prepare


def make_execute(
    *,
    llm_factory=None,
    spec: AgentSpec | None = None,
    run_overrides=None,
):
    """Build the ``execute`` callable.

    ``llm_factory`` is called once per cell. One LLM per cell, never shared:
    the budget cap and the telemetry both live on it, so a shared instance
    would cap the sweep rather than the run, and attribute one cell's spend
    to another.

    ``run_overrides`` supplies the kernel's ``plan_override`` /
    ``worker_override`` hooks. That is what makes the whole pipeline testable
    without a model: the measurement path is identical, only the thing
    producing edits is scripted.
    """

    def execute(cell: Cell, ctx: CellContext) -> RunResult:
        llm = llm_factory(ctx) if llm_factory else None
        # The kernel runs the config `prepare` built, not one rebuilt from the
        # arm name. Rebuilding drops anything prepare decided -- the
        # observation grid, the parallelism pin -- while the ledger still
        # records prepare's config_hash, so the manifest would describe a run
        # that never happened. A reproducibility claim rests on those being
        # the same object.
        kernel = Kernel(
            workspace=ctx.workspace, llm=llm,
            **kernel_kwargs(ctx.config), config=ctx.config,
        )
        agent = spec or AgentSpec(
            name="swe",
            description="Resolve the reported issue.",
            system_prompt=(
                "You are fixing a bug in an existing repository. Make the "
                "smallest change that resolves the report without breaking "
                "behaviour that already works."
            ),
        )
        extra = run_overrides(cell, ctx) if run_overrides else {}
        result = kernel.run(
            task=swebench.task_text(ctx.instance), spec=agent, base_ref="HEAD", **extra
        )
        ctx.session = result.session_id
        ctx.shadow_ref = f"{SHADOW_HEAD}_{result.session_id.upper().replace('-', '_')}"
        return result

    return execute


def make_score(*, ledger_dir: Path, grade=None, suite_factory=None):
    """Build the ``score`` callable: replay, attribute, and write the sidecar.

    Two things are injected rather than imported, and for the same reason:
    both need a real image. ``grade`` runs the official grader on the final
    patch, and ``suite_factory`` builds the probe. Neither the ``swebench``
    package nor Docker exists on a development machine, and the whole
    timeline is computable without either — so the seam is where the
    dependency is, not scattered through the driver.
    """
    build_suite = suite_factory or swebench.pass_to_pass_suite

    def score(cell: Cell, ctx: CellContext, result: RunResult) -> float | None:
        memory = Memory.open_session(ctx.workspace, ctx.session or result.session_id)
        session = ctx.session or result.session_id
        timeline = load_timeline(ctx.gitdir, session)

        events_path = ctx.gitdir / "events.jsonl"
        events = read_events(events_path) if events_path.exists() else []
        failures = harness_failures(events, timeline, session=session)

        suite = build_suite(ctx.instance)
        sandbox = None
        executor = None
        if ctx.provider is not None:
            sandbox = ctx.provider.open(key=ctx.instance.instance_id, image=ctx.instance.image)
            executor = SandboxProbeExecutor(sandbox, memory, ctx.instance.base_commit)

        attribution = None
        if ctx.probe_coverage and ctx.monitor_coverage:
            attribution = attribution_map(
                failures=failures,
                probe_tests=list(ctx.instance.pass_to_pass),
                monitor_coverage=ctx.monitor_coverage,
                probe_coverage=ctx.probe_coverage,
                modified_files_at={c.seq: frozenset(c.files) for c in timeline},
            )

        report = reconstruct(
            memory,
            timeline,
            [suite],
            harness_failed_at=failed_at(failures),
            attribution=attribution.by_seq if attribution else None,
            session=session,
            executor=executor,
        )

        silence = summarise_silence(
            report.episodes,
            attribution
            or attribution_map(
                failures=[], probe_tests=[],
                monitor_coverage=CoverageMap("", "", "none"),
                probe_coverage=CoverageMap("", "", "none"),
                modified_files_at={},
            ),
            method=ctx.probe_coverage.method if ctx.probe_coverage else "none",
        )

        graded = grade(ctx, result) if grade else None
        evidence = CellEvidence(
            instance_id=ctx.instance.instance_id,
            arm=cell.arm,
            trial=cell.trial,
            session=session,
            observations=report.observations,
            episodes=[asdict(e) for e in report.episodes],
            never_passed=list(report.never_passed),
            unknown_transitions=report.unknown_transitions,
            replays=report.replays,
            monitor_failures=len(failures),
            monitor_failures_unindexed=sum(1 for f in failures if f.seq is None),
            silence=asdict(silence),
            resolved=graded,
        )
        ctx.report_path = str(
            evidence.write(Path(ledger_dir) / "evidence" / f"{cell.key}.json")
        )

        if executor is not None:
            executor.close()
        return None if graded is None else float(graded)

    return score
