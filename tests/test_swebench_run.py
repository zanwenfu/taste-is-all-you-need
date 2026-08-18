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
        command='python -c "import lib; assert lib.value() == 1"',
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
