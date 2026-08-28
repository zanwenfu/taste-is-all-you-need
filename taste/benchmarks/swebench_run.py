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
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from taste.agent import AgentSpec
from taste.attempts import RetryPool
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
    agent_sandbox: Any = None
    """The cell's own container, when execution is routed. Per cell, never
    per instance: arms interleave within a task, and a shared container would
    hand the second arm the first arm's tree (the dead-container lesson, one
    level up)."""
    routed: bool = False
    router: Any = None
    """The live SandboxRouter for this cell, set by execute before any
    override runs. Scripted workers use it to drive the REAL tool path
    (write_file -> push -> in-container exec -> pull) instead of silently
    bypassing the seam under test."""
    grade_report: Any = None
    gate_watched: Any = None
    """The test files the regression gate was allowed to run, when one ran;
    the sidecar keeps them so held-out contamination can be computed."""
    """The official per-test grade, when a grader ran. The sidecar keeps the
    counts; the scalar keeps only resolved."""
    session: str = ""
    llm_stats: Any = None
    """The run's RunStats, written by ``execute`` the moment the kernel hands
    back control. The sweep driver's crash path reads it: a crash in score()
    lands after the agent phase was paid for, and a ledger row with
    billed_usd=0 at that point erases real spend."""


@dataclass
class CellEvidence:
    """The sidecar. What a scalar score cannot carry."""

    instance_id: str
    arm: str
    trial: int
    session: str
    observations: int
    episodes: list[dict[str, Any]] = field(default_factory=list)
    contamination_events_declared: int = 0
    """``len(episodes)`` counts one event per raw test id; the pre-declared
    unit is (test function, onset) with parametrised variants collapsed, and
    this is the count in that unit. Both are always written — a reader is
    entitled to see how much the collapse moved the number."""
    never_passed: list[str] = field(default_factory=list)
    unknown_transitions: int = 0
    replays: int = 0
    monitor_failures: int = 0
    monitor_failures_unindexed: int = 0
    """Failures on a tree that produced no observation. Reported rather than
    dropped: they are real detections at points the timeline cannot index."""
    silence: dict[str, Any] = field(default_factory=dict)
    resolved: bool | None = None
    grade: dict[str, str] = field(default_factory=dict)
    grade_failed: list[str] = field(default_factory=list)
    """PASS_TO_PASS ids the official grade scored as not passing — the
    per-test detail that lets contamination be restricted to tests the
    gate never saw."""
    gate_watched: list[str] = field(default_factory=list)
    routed: bool = False
    """Whether the agent executed inside the pinned image. A number from an
    unrouted run of a real instance is a number about bug 20, and the sidecar
    must say so rather than leave the reader to guess from dates."""

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
    route_execution: bool = False,
    planner_model: str | None = None,
    parity_check=None,
):
    """Build the ``prepare`` callable for a sweep over these instances.

    ``repo_cache`` switches materialization to the real repository at
    ``base_commit``, cloned once per repo and reused. ``source_root`` is the
    fixture path, for tests. Neither means an empty workspace, which is only
    ever right for a synthetic task.

    ``route_execution`` is the one-environment mode: the cell gets its own
    container, the workspace is materialised FROM that container's tree, and
    the parity check must pass before a single model call is paid for. This
    is the only mode in which a real benchmark run is valid — the host path
    exists for synthetic tasks and Gate 0.
    """
    if route_execution and provider is None:
        raise ValueError(
            "route_execution requires a sandbox provider: routing the agent "
            "into no container would silently fall back to the host — bug 20 "
            "with a flag claiming otherwise"
        )

    def prepare(cell: Cell) -> CellContext:
        instance = instances[cell.task]
        # Fresh per cell — see the module docstring; this is not reusable.
        workspace = Path(root) / cell.task / cell.arm / f"t{cell.trial}"
        agent_sandbox = None
        if route_execution:
            agent_sandbox = provider.open(key=f"agent:{cell.key}", image=instance.image)
            swebench.materialize_from_image(agent_sandbox, instance, workspace)
            check = parity_check or swebench.environment_parity_check
            reason = check(agent_sandbox, instance)
            if reason is not None:
                # Refused at $0. The alternative is paying for a full agent
                # run whose every command fails somewhere the benchmark never
                # grades, then reading that as the agent's incompetence.
                agent_sandbox.close()
                raise RuntimeError(f"environment parity: {reason}")
        elif repo_cache is not None:
            swebench.materialize_from_repo(instance, workspace, cache=Path(repo_cache))
        else:
            source = Path(source_root) / instance.instance_id if source_root else None
            swebench.materialize(instance, workspace, source=source)

        # max_parallel=1 for every measured arm. The shadow log is bound to
        # the primary session memory, so a parallel worker's edits are not
        # visible to the observation stamped with that worker's step -- which
        # would make the recorded file set wrong exactly where attribution
        # reads it.
        overrides: dict = {"max_parallel": 1, "observe_tools": observe_tools}
        if planner_model is not None:
            # A different planner is a different harness configuration, and
            # the config hash says so — two model families must never share
            # an identity in the ledger.
            overrides["planner_model"] = planner_model
        config = HarnessConfig.arm(cell.arm, **overrides)
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
            agent_sandbox=agent_sandbox,
            routed=agent_sandbox is not None,
        )

    return prepare


def make_execute(
    *,
    llm_factory=None,
    spec: AgentSpec | None = None,
    run_overrides=None,
    retry_allowance: Mapping[str, int] | None = None,
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

    ``retry_allowance`` maps instance id -> retries, and is what makes A3' an
    attempt-matched control rather than a differently-budgeted arm. An
    instance absent from the mapping runs unmatched on the arm's own ceiling:
    a missing paired run is not the same fact as a paired run that never
    retried, and defaulting it to zero would silently strangle the control on
    exactly the instances where the pairing failed. A caller that wants every
    cell matched should check the mapping's coverage, not rely on this.
    """

    def execute(cell: Cell, ctx: CellContext) -> RunResult:
        llm = llm_factory(ctx) if llm_factory else None
        # The kernel runs the config `prepare` built, not one rebuilt from the
        # arm name. Rebuilding drops anything prepare decided -- the
        # observation grid, the parallelism pin -- while the ledger still
        # records prepare's config_hash, so the manifest would describe a run
        # that never happened. A reproducibility claim rests on those being
        # the same object.
        allowance = (retry_allowance or {}).get(ctx.instance.instance_id)
        router = None
        if ctx.agent_sandbox is not None:
            from taste.routing import SandboxRouter

            # The sandbox's own workdir, never the default: a router aimed
            # at a path the sandbox does not use syncs into nowhere while the
            # commands run somewhere else — and the very first dry run of
            # this wiring proved it, by silently creating a real /testbed on
            # the development host.
            router = SandboxRouter(
                ctx.agent_sandbox, ctx.workspace, workdir=ctx.agent_sandbox.workdir
            )
            ctx.router = router
        gate = None
        if getattr(ctx.config, "regression_gate", False):
            if router is None:
                # An arm that claims the gate must not quietly run the
                # planner's check instead: that is the verifier being
                # compared against, under the label of the one being tested.
                raise RuntimeError(
                    "regression_gate requires routed execution (a container to run the suite in)"
                )
            from taste.regression_gate import RegressionGate

            gate = RegressionGate(
                instance=ctx.instance, run=router.exec,
                split=getattr(ctx.config, "gate_split", "all"),
            )
            ctx.gate_watched = gate.watched_files()
        kernel = Kernel(
            workspace=ctx.workspace, llm=llm,
            **kernel_kwargs(ctx.config), config=ctx.config,
            retry_pool=RetryPool(total=allowance) if allowance is not None else None,
            router=router,
            regression_gate=gate,
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
        try:
            result = kernel.run(
                task=swebench.task_text(ctx.instance), spec=agent, base_ref="HEAD", **extra
            )
        finally:
            # Onto the context the moment the kernel is done — including a
            # crash mid-run. Anything that throws after this point (score,
            # the sidecar write, a dead container) happens with the agent
            # phase already paid, and the sweep driver's error row reads the
            # spend from here so it cannot vanish from the ledger.
            ctx.llm_stats = llm.stats if llm is not None else None
            if ctx.agent_sandbox is not None:
                # The cell's container dies with the cell. Scoring opens its
                # own probe container under a different key; keeping this one
                # alive would only offer the next cell a stale tree to
                # inherit.
                ctx.agent_sandbox.close()
                ctx.agent_sandbox = None
        ctx.session = result.session_id
        ctx.shadow_ref = f"{SHADOW_HEAD}_{result.session_id.upper().replace('-', '_')}"
        return result

    return execute


def make_grade(*, timeout: int = 1800):
    """Build the ``grade`` callable for make_score: the official resolve verdict.

    This is the number that was never wired: every pilot recorded
    ``resolved=None`` on every cell because make_score's ``grade`` parameter
    had zero call sites. "Completed 8/40" was the Monitor's opinion of
    itself; this is the benchmark's opinion of the patch.
    """

    def grade(ctx: CellContext, result) -> bool | None:
        if ctx.provider is None:
            return None
        root = subprocess.run(
            ["git", "rev-list", "--max-parents=0", "HEAD"],
            cwd=ctx.workspace, capture_output=True, text=True,
        ).stdout.split()
        if not root:
            return None
        # The workspace's own root commit, never instance.base_commit: the
        # upstream sha is deliberately unresolvable here, and diffing against
        # it yields an empty patch — a grader that would have scored every
        # run "unresolved, empty prediction" while looking perfectly healthy.
        patch = swebench.patch_for(ctx.workspace, root[0])
        sandbox = ctx.provider.open(
            key=f"grade:{ctx.instance.instance_id}", image=ctx.instance.image,
            # The official harness grades with the network up; four of
            # psf/requests' graded connect-timeout tests literally need a
            # stack to time out on. The MEASUREMENT containers stay severed —
            # this exception is for comparability with the leaderboard, and
            # for nothing else.
            network_mode="bridge",
        )
        try:
            report = swebench.grade_in_sandbox(
                sandbox, ctx.instance, patch, timeout=timeout
            )
        finally:
            sandbox.close()
        if report is None:
            return None
        ctx.grade_report = report
        return report.resolved

    return grade


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
            routed=ctx.routed,
            trial=cell.trial,
            session=session,
            observations=report.observations,
            episodes=[asdict(e) for e in report.episodes],
            contamination_events_declared=report.contamination_events_declared,
            never_passed=list(report.never_passed),
            unknown_transitions=report.unknown_transitions,
            replays=report.replays,
            monitor_failures=len(failures),
            monitor_failures_unindexed=sum(1 for f in failures if f.seq is None),
            silence=asdict(silence),
            resolved=graded,
            grade={
                "fail_to_pass": f"{ctx.grade_report.fail_to_pass_passed}/{ctx.grade_report.fail_to_pass_total}",
                "pass_to_pass": f"{ctx.grade_report.pass_to_pass_passed}/{ctx.grade_report.pass_to_pass_total}",
            } if ctx.grade_report is not None else {},
            grade_failed=sorted(
                t for t in ctx.instance.pass_to_pass
                if ctx.grade_report.per_test.get(t) not in swebench.PASSING_STATUSES
            ) if ctx.grade_report is not None else [],
            gate_watched=list(ctx.gate_watched or []),
        )
        ctx.report_path = str(
            evidence.write(Path(ledger_dir) / "evidence" / f"{cell.key}.json")
        )

        if executor is not None:
            executor.close()
        memory.close()
        return None if graded is None else float(graded)

    return score
