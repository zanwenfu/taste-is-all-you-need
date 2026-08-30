"""The one-environment seam: the agent executes where it is measured.

Bug 20 in miniature: the Worker's shell and the Monitor's check ran on the
host in a bare checkout, while every probe and every graded verdict ran
inside the pinned image. These tests drive the router with a LocalSandbox
over a separate directory standing in for /testbed, so the property under
test — two trees, one truth — is provable with the Docker daemon down.

The three tests the audit demanded by name are here: a file mutated only by
a container-side command appears in the next shadow observation; a
write_file edit is visible to the next in-container command; and the
Monitor's check runs in the container's environment, not the host's.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from taste.agent import AgentSpec
from taste.cores import Plan, Step, Verification, WorkerResult
from taste.execution import LocalSandbox
from taste.kernel import Kernel
from taste.routing import SandboxRouter, prepare_container_tree
from taste.shadow import load_timeline


@pytest.fixture
def testbed(tmp_path: Path) -> Path:
    """A directory standing in for the container's /testbed."""
    bed = tmp_path / "bed"
    bed.mkdir()
    (bed / "lib.py").write_text("def rate():\n    return 1\n")
    return bed


def _router(bed: Path, ws: Path) -> SandboxRouter:
    return SandboxRouter(LocalSandbox(bed), ws, workdir=str(bed))


def _git(bed: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(bed), *args], capture_output=True, text=True)
    return proc.stdout


# ------------------------------------------------------------ the baseline


def test_prepare_container_tree_is_idempotent(testbed: Path) -> None:
    box = LocalSandbox(testbed)
    first = prepare_container_tree(box, workdir=str(testbed))
    second = prepare_container_tree(box, workdir=str(testbed))
    assert first and first == second
    assert (testbed / ".git" / "info" / "exclude").read_text().strip()


def test_the_baseline_commits_the_image_state_not_an_empty_tree(testbed: Path) -> None:
    """The baseline must BE the image's tree — pre_install edits included.

    Resetting replay to the upstream base_commit instead of to the image's
    actual state is bug B8: a whole repo family's build-time source edits
    reverted before every probe, oracle dead, disguised as flake.
    """
    (testbed / "edited_by_image_build.py").write_text("PATCHED = True\n")
    prepare_container_tree(LocalSandbox(testbed), workdir=str(testbed))
    tracked = _git(testbed, "ls-files")
    assert "edited_by_image_build.py" in tracked


# ------------------------------------------------------------ push and pull


def test_a_write_file_edit_is_visible_to_the_next_container_command(
    testbed: Path, tmp_path: Path
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    router = _router(testbed, ws)

    (ws / "new_module.py").write_text("VALUE = 7\n")
    router.mark_dirty("new_module.py")
    result = router.exec("cat new_module.py")
    assert result.exit_code == 0
    assert "VALUE = 7" in result.stdout, "the container ran against a tree missing the edit"


def test_a_container_side_mutation_reaches_the_host_before_exec_returns(
    testbed: Path, tmp_path: Path
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    router = _router(testbed, ws)

    result = router.exec("printf 'born in the box' > box_born.txt")
    assert result.exit_code == 0
    assert (ws / "box_born.txt").read_text() == "born in the box", (
        "the observation after this exec would have seen an unchanged tree"
    )


def test_a_container_side_deletion_reaches_the_host(testbed: Path, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "doomed.py").write_text("x = 1\n")
    router = _router(testbed, ws)
    router.mark_dirty("doomed.py")
    router.exec("true")  # push it
    router.exec("rm doomed.py")
    assert not (ws / "doomed.py").exists()


# ------------------------------------------------------------ transparent mode


def test_transparent_mode_keeps_git_diff_meaningful_for_the_agent(testbed: Path, tmp_path: Path) -> None:
    """An agent that verifies its edit with git must see it there.

    Against an advancing baseline the edit is committed by the sync the
    moment the command returns, so the agent's next ``git diff`` is empty
    and ``git status`` is clean — mini-swe-agent's first paid pilot spent
    eight steps on exactly that. Transparent mode leaves HEAD alone.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "lib.py").write_text((testbed / "lib.py").read_text())
    router = SandboxRouter(LocalSandbox(testbed), ws, workdir=str(testbed), advance_baseline=False)

    router.exec("printf 'def rate():\\n    return 2\\n' > lib.py")
    assert (ws / "lib.py").read_text().endswith("return 2\n"), "the host must still receive the edit"
    diff = router.exec("git diff")
    assert "+    return 2" in diff.stdout, "the agent's own git diff must show its edit"
    status = router.exec("git status --porcelain")
    assert status.stdout.strip().startswith("M"), status.stdout
    assert _git(testbed, "log", "--oneline").count("\n") == 1, "no sync commits may appear"

    # The host keeps following later edits too — the pull is idempotent.
    router.exec("printf 'def rate():\\n    return 3\\n' > lib.py")
    assert (ws / "lib.py").read_text().endswith("return 3\n")


def test_transparent_mode_pulls_work_the_agent_committed(testbed: Path, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "lib.py").write_text((testbed / "lib.py").read_text())
    router = SandboxRouter(LocalSandbox(testbed), ws, workdir=str(testbed), advance_baseline=False)
    router.exec("printf 'x = 1\\n' > committed.py && git add committed.py && git -c user.name=a -c user.email=a@b commit -qm agent")
    assert (ws / "committed.py").read_text() == "x = 1\n", "a commit must not hide work from the pull"
    router.exec("rm lib.py")
    assert not (ws / "lib.py").exists(), "a deletion against the baseline must reach the host"
    # A pull on an unchanged tree changes nothing and raises nothing.
    assert router.pull() == ("committed.py",)


def test_junk_from_a_test_run_never_crosses_the_boundary(
    testbed: Path, tmp_path: Path
) -> None:
    """__pycache__ pulled to the host is bug D3 all over again: thousands of
    generated files absorbed into shadow observations, attribution vacuous."""
    ws = tmp_path / "ws"
    ws.mkdir()
    router = _router(testbed, ws)
    router.exec("mkdir -p __pycache__ && printf x > __pycache__/lib.cpython-311.pyc && printf y > real.py")
    assert (ws / "real.py").exists()
    assert not (ws / "__pycache__").exists()


def test_a_host_deletion_is_pushed(testbed: Path, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "temp.py").write_text("x\n")
    router = _router(testbed, ws)
    router.mark_dirty("temp.py")
    router.exec("true")
    (ws / "temp.py").unlink()
    router.mark_dirty("temp.py")
    result = router.exec("test -f temp.py")
    assert result.exit_code != 0, "the container still had a file the host deleted"


def test_mark_reset_pushes_exactly_the_rollback_delta(tmp_path: Path) -> None:
    """A rollback is a host-side mutation of many files at once; the delta —
    and only the delta — must reach the container."""
    from git import Repo

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.py").write_text("A = 1\n")
    (ws / "b.py").write_text("B = 1\n")
    repo = Repo.init(ws)
    with repo.git.custom_environment(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@l",
                                     GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@l"):
        repo.git.add("-A")
        repo.git.commit("-qm", "one")
        good = repo.head.commit.hexsha
        (ws / "a.py").write_text("A = 2\n")
        repo.git.add("-A")
        repo.git.commit("-qm", "two")
        bad = repo.head.commit.hexsha

    bed = tmp_path / "bed"
    bed.mkdir()
    router = _router(bed, ws)
    router.mark_dirty("a.py")
    router.mark_dirty("b.py")
    router.exec("true")  # container now has A=2

    repo.git.reset("--hard", good)  # the rollback
    router.mark_reset(repo, bad, good)
    result = router.exec("cat a.py")
    assert "A = 1" in result.stdout, "the container kept executing the discarded attempt"
    repo.close()


# ---------------------------------------------------- the kernel, end to end


def _spec() -> AgentSpec:
    return AgentSpec(name="routed", description="no LLM", model=None, system_prompt="")


def test_container_only_mutations_appear_in_the_shadow_timeline(
    refactor_workspace: Path, tmp_path: Path
) -> None:
    """The audit's named test: a file mutated ONLY by a container-side shell
    command must appear in the next ShadowCommit.files. Without the pull
    ordered before the observation, the finer per-tool grid records empty
    exactly where the mutations happen."""
    bed = tmp_path / "bed"
    bed.mkdir()
    ws = refactor_workspace
    router = SandboxRouter(LocalSandbox(bed), ws, workdir=str(bed))

    def worker(step: Step, plan: Plan) -> WorkerResult:
        # The worker's edit happens purely container-side.
        router.exec("printf 'MUTATED = True' > container_edit.py")
        return WorkerResult(summary="edited in the box", tool_calls=1, stopped_reason="end_turn")

    plan = Plan(task="routed", steps=[
        Step(id="step-01", description="edit via container",
             verification=Verification(kind="shell", command="test -f container_edit.py")),
    ])
    kernel = Kernel(workspace=ws, shadow=True, router=router)
    result = kernel.run(task="routed", spec=_spec(), plan_override=plan, worker_override=worker)

    assert result.status == "completed", result.failure_reason
    assert (ws / "container_edit.py").exists()
    timeline = load_timeline(Path(kernel._runtime_dir), result.session_id)
    observed = {f for c in timeline for f in c.files}
    assert "container_edit.py" in observed, (
        "the timeline never saw the container-side edit; the instrument is "
        "measuring an empty grid where the work happened"
    )


def test_the_monitor_grades_the_container_environment_not_the_host(
    refactor_workspace: Path, tmp_path: Path
) -> None:
    """The audit's named test, Monitor half. The marker exists only in the
    container's baseline; a host-side check cannot see it. If this fails,
    the Monitor is grading a different machine than the one the agent works
    on — which is bug 20, verbatim."""
    bed = tmp_path / "bed"
    bed.mkdir()
    (bed / "installed_only_in_image.marker").write_text("present\n")
    ws = refactor_workspace
    router = SandboxRouter(LocalSandbox(bed), ws, workdir=str(bed))

    plan = Plan(task="parity", steps=[
        Step(id="step-01", description="noop",
             verification=Verification(kind="shell", command="test -f installed_only_in_image.marker")),
    ])
    def worker(step: Step, plan: Plan) -> WorkerResult:
        return WorkerResult(summary="noop", tool_calls=0, stopped_reason="end_turn")

    routed = Kernel(workspace=ws, router=router).run(
        task="p", spec=_spec(), plan_override=plan, worker_override=worker)
    assert routed.status == "completed", (
        "the routed Monitor could not see the container environment: "
        f"{routed.failure_reason}"
    )

    unrouted = Kernel(workspace=ws, max_retries=0).run(
        task="p", spec=_spec(), plan_override=plan, worker_override=worker)
    assert unrouted.status == "failed", (
        "the control leg is broken: a host-side check saw the container marker, "
        "so this test can no longer distinguish the two environments"
    )


def test_a_router_forces_sequential_execution(tmp_path: Path) -> None:
    """One /testbed, one worker at a time. Two worktree workers syncing into
    the same container tree is cross-arm contamination within a single run."""
    bed = tmp_path / "bed"
    bed.mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "seed.py").write_text("x\n")
    router = SandboxRouter(LocalSandbox(bed), ws, workdir=str(bed))
    kernel = Kernel(workspace=ws, max_parallel=8, router=router)
    assert kernel.max_parallel == 1


def test_rollback_delta_reaches_the_container(refactor_workspace: Path, tmp_path: Path) -> None:
    """After A3 resets the host tree, the container must not keep the
    discarded attempt: the next attempt's commands would run against work
    the harness already threw away."""
    bed = tmp_path / "bed"
    bed.mkdir()
    ws = refactor_workspace
    router = SandboxRouter(LocalSandbox(bed), ws, workdir=str(bed))
    seen: dict[str, int] = {}

    def worker(step: Step, plan: Plan) -> WorkerResult:
        n = seen[step.id] = seen.get(step.id, 0) + 1
        if n == 1:
            (ws / "attempt.py").write_text("BROKEN = True\n")
            router.mark_dirty("attempt.py")
            router.exec("true")  # container now holds the broken attempt
        else:
            check = router.exec("test -f attempt.py")
            assert check.exit_code != 0, (
                "the container kept the discarded attempt across the rollback"
            )
            (ws / "fixed.py").write_text("OK = True\n")
            router.mark_dirty("fixed.py")
        return WorkerResult(summary=f"attempt {n}", tool_calls=1, stopped_reason="end_turn")

    plan = Plan(task="reset", steps=[
        Step(id="step-01", description="fails once",
             verification=Verification(kind="shell", command="test -f fixed.py")),
    ])
    result = Kernel(workspace=ws, max_retries=1, router=router).run(
        task="reset", spec=_spec(), plan_override=plan, worker_override=worker)
    assert result.status == "completed", result.failure_reason
    assert seen["step-01"] == 2


def test_an_untransportable_path_is_skipped_and_counted_not_fatal(
    testbed: Path, tmp_path: Path
) -> None:
    """Bug 29: sphinx's test fixtures are full of symlinks; the transport's
    tar carries no content for a broken one, and the resulting
    FileNotFoundError killed two paid cells mid-sweep. The pull must skip
    the irregular path, record it, and still deliver everything else."""
    ws = tmp_path / "ws"
    ws.mkdir()
    router = _router(testbed, ws)
    router.exec("ln -s does-not-exist broken_link && printf ok > real_file.py")
    assert (ws / "real_file.py").exists(), "the regular file must still arrive"
    assert not (ws / "broken_link").exists()
    assert "broken_link" in router.skipped, "the skip must be visible, not silent"
