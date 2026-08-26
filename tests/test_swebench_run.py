"""One instance, end to end, for zero dollars.

This is the test that proves the seam actually connects. Every part of it was
individually tested before this file existed and none of it had ever run
together, so nothing had produced a contamination number from an actual run.

A toy "instance" in the published schema and a scripted worker stand in for
the real thing; the probe runs locally instead of in a pinned image. What is genuinely exercised is the wiring: the
kernel runs, shadow observes, the events join onto the timeline, replay
reconstructs episodes, attribution classifies them, and the sidecar lands on
disk with pointers recorded in the ledger.

No Docker, no network, no API.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from taste.agent import AgentSpec
from taste.benchmarks import swebench
from taste.benchmarks.swebench_run import (
    CellEvidence,
    make_execute,
    make_prepare,
    make_score,
)
from taste.cores import Plan, Step, Verification, WorkerResult
from taste.evalrun import Cell, run_sweep
from taste.replay import SuiteProbe

TEST_PATCH = """diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1,2 +1,3 @@
 def test_old(): pass
+def test_new(): pass
"""


@pytest.fixture
def instance() -> swebench.SWEInstance:
    return swebench.SWEInstance(
        instance_id="toy__toy-1",
        repo="pytest-dev/pytest",
        base_commit="0" * 40,
        problem_statement="value() should return 1.",
        test_patch=TEST_PATCH,
        version="7.0",
        fail_to_pass=("tests/test_core.py::test_new",),
        pass_to_pass=("tests/test_core.py::test_old",),
        image="",
    )


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A two-file repository standing in for a real checkout."""
    root = tmp_path / "source" / "toy__toy-1"
    (root / "tests").mkdir(parents=True)
    (root / "lib.py").write_text("def value():\n    return 1\n")
    (root / "tests" / "test_core.py").write_text("def test_old(): pass\n")
    return tmp_path / "source"


# A P2P probe that runs for real on a local tree -- the stand-in for a suite
# that would otherwise need the pinned image.
def _local_suite(_instance: swebench.SWEInstance) -> SuiteProbe:
    return SuiteProbe(
        name="p2p::toy",
        command=f'{sys.executable} -c "import lib; assert lib.value() == 1"',
        members=("tests/test_core.py::test_old",),
        timeout=30,
    )


def _breaking_run(workspace: Path):
    """Two steps: the first quietly breaks the P2P invariant, the second does
    unrelated work. Exactly the shape the study measures -- a regression the
    step's own verification does not catch."""
    plan = Plan(
        task="toy",
        steps=[
            Step(id="step-01", description="add a helper",
                 verification=Verification(kind="shell", command="test -f helper.py")),
            Step(id="step-02", description="add a note",
                 verification=Verification(kind="shell", command="test -f note.txt")),
        ],
    )

    def worker(step: Step, _plan: Plan) -> WorkerResult:
        if step.id == "step-01":
            (workspace / "helper.py").write_text("# helper\n")
            # The silent regression: verification only checks helper.py exists.
            (workspace / "lib.py").write_text("def value():\n    return 2\n")
            return WorkerResult("added helper", 1, "end_turn")
        (workspace / "note.txt").write_text("note\n")
        return WorkerResult("added note", 1, "end_turn")

    return {"plan_override": plan, "worker_override": worker}


# ------------------------------------------------------------------ workspace


def test_materialize_gives_each_cell_its_own_tree(
    instance: swebench.SWEInstance, source: Path, tmp_path: Path
) -> None:
    """Two trials sharing a tree makes one run's edits the next run's starting
    state, which destroys the pairing the analysis rests on."""
    a = swebench.materialize(instance, tmp_path / "a", source=source / "toy__toy-1")
    b = swebench.materialize(instance, tmp_path / "b", source=source / "toy__toy-1")

    (a / "lib.py").write_text("def value():\n    return 999\n")
    assert "999" not in (b / "lib.py").read_text()


def test_the_upstream_history_is_not_carried_in(
    instance: swebench.SWEInstance, source: Path, tmp_path: Path
) -> None:
    """Read-only git is available to workers, so an inherited history would
    let an agent read the upstream fix for its own instance."""
    repo = source / "toy__toy-1"
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t.co", "commit", "-qm", "THE FIX"],
        cwd=repo, check=True,
    )

    workspace = swebench.materialize(instance, tmp_path / "ws", source=repo)
    log = subprocess.run(
        ["git", "log", "--all", "--oneline"], cwd=workspace, capture_output=True, text=True
    ).stdout
    assert "THE FIX" not in log
    assert log.count("\n") == 1, "exactly one commit, not an imported history"


def test_gc_is_disabled_so_the_timeline_cannot_vanish(
    instance: swebench.SWEInstance, tmp_path: Path
) -> None:
    """Losing the shadow chain does not look like a failure. It looks like a
    clean run."""
    workspace = swebench.materialize(instance, tmp_path / "ws")
    setting = subprocess.run(
        ["git", "config", "--get", "gc.auto"], cwd=workspace, capture_output=True, text=True
    ).stdout.strip()
    assert setting == "0"


# ------------------------------------------------------------------ the sweep


def test_one_cell_runs_end_to_end_and_leaves_evidence(
    instance: swebench.SWEInstance, source: Path, tmp_path: Path
) -> None:
    """The whole point of this file: harness in, measurement out, $0."""
    root = tmp_path / "runs"
    ledger = tmp_path / "ledger"
    instances = {instance.instance_id: instance}

    report = run_sweep(
        tasks=[instance.instance_id],
        arms=["A0"],
        trials=1,
        ledger_dir=ledger,
        prepare=make_prepare(
            instances=instances, root=root, source_root=source, provider=None,
        ),
        execute=make_execute(run_overrides=lambda _c, ctx: _breaking_run(ctx.workspace)),
        score=make_score(ledger_dir=ledger, suite_factory=_local_suite),
    )

    assert len(report.results) == 1
    record = report.results[0]
    assert record.error is None, record.error
    assert record.status in ("completed", "failed"), record.status

    # The ledger must point at the artifacts, or the primary outcome cannot be
    # recomputed from what was written.
    assert record.workspace and Path(record.workspace).exists()
    assert record.session_branch
    assert record.shadow_ref.startswith("TASTE_SHADOW_HEAD")
    assert record.report_path and Path(record.report_path).exists()

    evidence = json.loads(Path(record.report_path).read_text())
    assert evidence["instance_id"] == instance.instance_id
    assert evidence["observations"] > 0, "the run was never observed"

    # The actual claim: the instrument found the regression the worker hid.
    # step-01's verification only checks that helper.py exists, so the Monitor
    # passed and the harness noticed nothing -- which is the phenomenon.
    assert len(evidence["episodes"]) == 1, evidence["episodes"]
    # Both units travel in the sidecar. With one unparametrised probe the
    # counts coincide; what matters is that the declared count is written.
    assert evidence["contamination_events_declared"] == 1
    episode = evidence["episodes"][0]
    assert episode["probe"] == "tests/test_core.py::test_old"
    assert episode["onset_seq"] == 2, "onset must land on the observation that broke it"
    assert episode["recovered_seq"] is None, "A0 does not recover"
    assert episode["detected_seq_attributed"] is None
    assert episode["detected_seq_any"] is None, "nothing failed, so nothing co-occurred"
    assert evidence["silence"]["silent_attributed"] == 1

    # One preparation per observation, not one per member: the cost property
    # that makes an exhaustive scan affordable instead of bisecting.
    assert evidence["replays"] == evidence["observations"]


def test_the_config_hash_is_recorded_not_silently_empty(
    instance: swebench.SWEInstance, source: Path, tmp_path: Path
) -> None:
    """The driver reads config.hash() through a getattr default, so an
    untyped context records an empty hash and two arms become
    indistinguishable in the manifest."""
    ledger = tmp_path / "ledger"
    report = run_sweep(
        tasks=[instance.instance_id], arms=["A0"], trials=1, ledger_dir=ledger,
        prepare=make_prepare(
            instances={instance.instance_id: instance}, root=tmp_path / "runs",
            source_root=source, provider=None,
        ),
        execute=make_execute(run_overrides=lambda _c, ctx: _breaking_run(ctx.workspace)),
        score=make_score(ledger_dir=ledger, suite_factory=_local_suite),
    )
    assert report.results[0].config_hash, "an empty hash makes arms indistinguishable"


def test_arms_get_separate_workspaces(
    instance: swebench.SWEInstance, source: Path, tmp_path: Path
) -> None:
    ledger = tmp_path / "ledger"
    report = run_sweep(
        tasks=[instance.instance_id], arms=["A0", "A2"], trials=1, ledger_dir=ledger,
        prepare=make_prepare(
            instances={instance.instance_id: instance}, root=tmp_path / "runs",
            source_root=source, provider=None,
        ),
        execute=make_execute(run_overrides=lambda _c, ctx: _breaking_run(ctx.workspace)),
        score=make_score(ledger_dir=ledger, suite_factory=_local_suite),
    )
    paths = {r.workspace for r in report.results}
    assert len(paths) == 2, "arms must never share a tree"


def test_evidence_carries_what_a_scalar_cannot(tmp_path: Path) -> None:
    """The dependent variable is the episode list, not the score."""
    evidence = CellEvidence(
        instance_id="i", arm="A3", trial=1, session="s", observations=7,
        episodes=[{"probe": "t", "onset_seq": 3}],
        monitor_failures=2, monitor_failures_unindexed=1,
    )
    written = evidence.write(tmp_path / "e.json")
    loaded = json.loads(written.read_text())

    assert loaded["episodes"][0]["onset_seq"] == 3
    assert loaded["monitor_failures_unindexed"] == 1, (
        "failures on an unobserved tree are real detections and must be "
        "reported, not dropped"
    )


def test_measured_arms_are_pinned_to_one_worker(
    instance: swebench.SWEInstance, source: Path, tmp_path: Path
) -> None:
    """The shadow log is bound to the primary session memory, so a parallel
    worker's edits are invisible to the observation stamped with its step —
    making the recorded file set wrong exactly where attribution reads it."""
    prepare = make_prepare(
        instances={instance.instance_id: instance}, root=tmp_path / "runs",
        source_root=source,
    )
    ctx = prepare(Cell(task=instance.instance_id, arm="A3", trial=1))
    assert ctx.config.max_parallel == 1


def test_the_agent_is_given_the_problem_and_not_the_oracle(
    instance: swebench.SWEInstance,
) -> None:
    text = swebench.task_text(instance)
    assert instance.problem_statement in text
    for node_id in instance.pass_to_pass + instance.fail_to_pass:
        assert node_id not in text


def test_a_spec_can_be_supplied_without_touching_the_driver() -> None:
    execute = make_execute(spec=AgentSpec(name="custom", description="d", system_prompt="p"))
    assert callable(execute)


def test_a_partial_mirror_is_replaced_not_reused(tmp_path: Path, monkeypatch) -> None:
    """An interrupted clone leaves a directory that is not a repository.

    Left alone it poisons every later cell on that repo: cat-file fails, the
    top-up fetch fails, and instances die for a reason that has nothing to do
    with the arm under test. Found when a clone was killed mid-run.
    """
    import subprocess as sp

    cache = tmp_path / "mirrors"
    poisoned = cache / "psf__requests.git"
    poisoned.mkdir(parents=True)
    (poisoned / "junk").write_text("not a repo")

    real_run = sp.run
    cloned: list[list[str]] = []

    def fake_run(args, **kwargs):
        # `git rev-parse --git-dir` must really run so the validity check is
        # the thing under test; only the network clone is stubbed.
        if isinstance(args, list) and args[:2] == ["git", "clone"]:
            cloned.append(args)
            Path(args[-1]).mkdir(parents=True, exist_ok=True)
            return sp.CompletedProcess(args, 0, b"", b"")
        return real_run(args, **kwargs)

    monkeypatch.setattr(swebench.subprocess, "run", fake_run)

    inst = swebench.SWEInstance(
        instance_id="psf__requests-1", repo="psf/requests", base_commit="0" * 40,
        problem_statement="x", test_patch="", version="2.27",
        fail_to_pass=(), pass_to_pass=("t::a",),
    )
    swebench.fetch_repo(inst, cache)

    assert cloned, "an invalid mirror must be re-cloned, not reused"
    assert not (poisoned / "junk").exists(), "the poisoned directory survived"


def test_a_mirror_that_is_a_git_dir_but_has_no_commits_is_rejected(tmp_path: Path) -> None:
    """Being a git directory is not enough, and the gap is not academic.

    An interrupted `git clone --bare` leaves a directory that satisfies
    `rev-parse --git-dir` while containing zero commits. Observed on two of
    five mirrors during a dry run. Accepting one skips the top-up fetch,
    `git archive` then fails, and the instance dies far from the cause.
    """
    import subprocess as sp

    cache = tmp_path / "mirrors"
    empty = cache / "psf__requests.git"
    empty.mkdir(parents=True)
    sp.run(["git", "init", "--bare", "-q", str(empty)], check=True)
    assert sp.run(["git", "rev-parse", "--git-dir"], cwd=empty,
                  capture_output=True).returncode == 0, "fixture must look like a git dir"

    assert swebench._mirror_is_usable(empty) is False


def test_the_kernel_runs_the_config_prepare_built(
    instance: swebench.SWEInstance, source: Path, tmp_path: Path
) -> None:
    """Caught while wiring the observation grid through a sweep.

    execute() rebuilt the config from the arm name, dropping everything
    prepare had decided -- the observation grid, the parallelism pin -- while
    the ledger still recorded prepare's config_hash. The manifest would have
    described a run that never happened, which is precisely what a
    reproducibility claim rests on.

    Asserted end to end rather than by inspection: a sweep configured for the
    fine grid must actually produce tool-triggered observations.
    """
    from taste.shadow import load_timeline

    ledger = tmp_path / "ledger"
    report = run_sweep(
        tasks=[instance.instance_id], arms=["A0"], trials=1, ledger_dir=ledger,
        prepare=make_prepare(
            instances={instance.instance_id: instance}, root=tmp_path / "runs",
            source_root=source, provider=None, observe_tools=True,
        ),
        execute=make_execute(run_overrides=lambda _c, ctx: _breaking_run(ctx.workspace)),
        score=make_score(ledger_dir=ledger, suite_factory=_local_suite),
    )
    record = report.results[0]
    assert record.error is None, record.error

    workspace = Path(record.workspace)
    timeline = list(load_timeline(workspace / ".git" / "taste", record.session_branch))
    assert timeline, "the run was never observed"
    # worker_override bypasses the tool loop, so no tool points here -- what
    # this pins is that the flag survived into the kernel's own config.
    assert record.config_hash == ctx_hash(workspace, observe_tools=True)


def ctx_hash(_workspace: Path, *, observe_tools: bool) -> str:
    from taste.config import HarnessConfig

    return HarnessConfig.arm("A0", max_parallel=1, observe_tools=observe_tools).hash()


def test_a_score_crash_leaves_the_paid_agent_phase_on_the_ledger(
    instance: swebench.SWEInstance, source: Path, tmp_path: Path
) -> None:
    """End to end through the real seam: the kernel runs and spends, score()
    blows up, and the ledger row must still carry the spend — read from the
    stats ``execute`` stashed on the context the moment the kernel finished —
    with an error naming the phase. Without that the money vanished and a
    resume re-executed the paid agent phase."""
    from types import SimpleNamespace

    ledger = tmp_path / "ledger"
    fake_llm = SimpleNamespace(stats=_paid_stats())

    def exploding_score(cell, ctx, result):
        raise RuntimeError("evidence write failed")

    report = run_sweep(
        tasks=[instance.instance_id], arms=["A0"], trials=1, ledger_dir=ledger,
        prepare=make_prepare(
            instances={instance.instance_id: instance}, root=tmp_path / "runs",
            source_root=source, provider=None,
        ),
        execute=make_execute(
            llm_factory=lambda _ctx: fake_llm,
            run_overrides=lambda _c, ctx: _breaking_run(ctx.workspace),
        ),
        score=exploding_score,
    )
    record = report.results[0]
    assert record.status == "error"
    assert (record.error or "").startswith("score:"), record.error
    assert record.billed_usd > 0, "the agent phase was paid; the ledger must say so"
    assert record.work_usd > 0


def _paid_stats():
    """RunStats carrying real spend, so billed and work are both nonzero."""
    from taste.llm import RunStats
    from taste.providers.base import Completion, Usage

    stats = RunStats()
    stats.record(
        "claude-sonnet-4-6",
        Completion(
            text_blocks=(), tool_calls=(), stop_reason="end_turn",
            model="claude-sonnet-4-6", provider="fake",
            usage=Usage(input_tokens=1000, cache_read_tokens=99_000),
            transcript_blocks=(),
        ),
    )
    return stats


# ---------------------------------------------------------- routed execution


class _BedProvider:
    """A provider whose 'containers' are LocalSandboxes over per-key dirs,
    each seeded like an image: a source tree plus a build artifact that only
    the image has. Existence of that artifact after materialisation is the
    proof the workspace came from the image and not from a mirror."""

    def __init__(self, root: Path):
        self.root = root
        self.opened: list[str] = []
        self.closed: list[str] = []

    def open(self, *, key: str, image: str):
        from taste.execution import LocalSandbox

        bed = self.root / key.replace(":", "_").replace("/", "_")
        if not bed.exists():
            bed.mkdir(parents=True)
            (bed / "lib.py").write_text("def rate():\n    return 1\n")
            (bed / "built_extension.so").write_bytes(b"\x7fELF-fake")
        box = LocalSandbox(bed)
        provider = self

        class _Tracked:
            workdir = str(bed)

            def __getattr__(self, name):  # delegate everything else
                return getattr(box, name)

            def close(self):
                provider.closed.append(key)
                box.close()

        self.opened.append(key)
        return _Tracked()


def _routed_sweep(tmp_path: Path, *, parity_ok: bool = True):
    from taste.benchmarks.swebench_run import make_execute, make_prepare, make_score
    from taste.evalrun import run_sweep
    from taste.replay import SuiteProbe

    instance = swebench.SWEInstance(
        instance_id="psf__requests-9999", repo="psf/requests", base_commit="0" * 40,
        problem_statement="rate() must stay 1.", test_patch="", version="2.0",
        fail_to_pass=(), pass_to_pass=("t::rate",),
    )
    if not parity_ok:
        # An unmapped repo forces the parity check to refuse the cell.
        object.__setattr__(instance, "repo", "nobody/unknown-project")

    provider = _BedProvider(tmp_path / "beds")

    def scripted(_cell, ctx):
        ws = ctx.workspace
        plan = Plan(task="toy", steps=[
            # lib.py comes from the image baseline, so the routed Monitor
            # (which runs in the container) can pass it. note.txt is written
            # host-side by the scripted worker and is asserted on the host —
            # the environment distinction itself is pinned in test_routing.
            Step(id="step-01", description="edit",
                 verification=Verification(kind="shell", command="test -f lib.py")),
        ])

        def worker(step, _p):
            (ws / "note.txt").write_text("note\n")
            return WorkerResult("note", 1, "end_turn")

        return {"plan_override": plan, "worker_override": worker}

    def suite(_i):
        return SuiteProbe(name="p", command="true", members=("t::rate",), timeout=30)

    ledger = tmp_path / "ledger"
    report = run_sweep(
        tasks=[instance.instance_id], arms=["A0"], trials=1, ledger_dir=ledger,
        prepare=make_prepare(
            instances={instance.instance_id: instance}, root=tmp_path / "runs",
            provider=provider, route_execution=True,
        ),
        execute=make_execute(run_overrides=scripted),
        score=make_score(ledger_dir=ledger, suite_factory=suite),
    )
    return report.results[0], provider, tmp_path


def test_routed_prepare_materialises_from_the_image(tmp_path: Path, monkeypatch) -> None:
    """The workspace must carry the image's build artifacts. A mirror gives
    tracked source only; the first sync would then overwrite the container's
    compiled extensions with clean checkouts — bug 20's ghost."""
    monkeypatch.setattr(
        swebench, "environment_parity_check", lambda sandbox, instance: None
    )
    cell, provider, _root = _routed_sweep(tmp_path)
    assert cell.status == "completed", (
        f"{cell.status}: {cell.error}\n{cell.failure_reason}"
    )
    ws = Path(cell.workspace)
    assert (ws / "built_extension.so").exists(), (
        "the workspace was not materialised from the image"
    )
    assert (ws / "note.txt").exists()
    agent_keys = [k for k in provider.opened if k.startswith("agent:")]
    assert agent_keys, "no per-cell agent container was opened"
    assert agent_keys[0] in provider.closed, "the agent container leaked past the cell"

    import json
    evidence = json.loads(Path(cell.report_path).read_text())
    assert evidence["routed"] is True, "the sidecar must say the run was routed"


def test_parity_failure_refuses_the_cell_before_any_model_call(tmp_path: Path) -> None:
    """A cell whose environment cannot be verified is infrastructure, refused
    at $0 — not an agent run whose every command fails somewhere the
    benchmark never grades."""
    cell, provider, _ = _routed_sweep(tmp_path, parity_ok=False)
    assert cell.status == "error"
    assert "parity" in (cell.error or ""), cell.error
    assert cell.billed_usd == 0.0
    agent_keys = [k for k in provider.opened if k.startswith("agent:")]
    assert agent_keys and agent_keys[0] in provider.closed, (
        "the refused cell's container must still be closed"
    )


def test_route_execution_without_a_provider_is_refused_loudly() -> None:
    from taste.benchmarks.swebench_run import make_prepare

    with pytest.raises(ValueError, match="provider"):
        make_prepare(instances={}, root=Path("/tmp/x"), route_execution=True)
