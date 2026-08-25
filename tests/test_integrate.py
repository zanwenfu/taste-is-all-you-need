"""Two-phase integration and the union gate.

Two failures are being prevented. A conflict partway through a merge loop
leaving the session branch holding a partial integration no step produced —
and two steps that merge cleanly as text while breaking each other, which
nobody can catch because each verified its own work alone.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests.conftest import PYTEST_CMD
from taste.agent import AgentSpec
from taste.cores import Plan, Step, Verification, WorkerResult
from taste.integrate import (
    Proposal,
    accumulate,
    integrate,
    preview_merge,
    supports_merge_tree,
    union_gate,
)
from taste.kernel import Kernel, current_step
from taste.memory import Memory


def _spec() -> AgentSpec:
    return AgentSpec(name="s", description="", system_prompt="p")


def _commit(ws: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t.co", "commit", "-m", message], cwd=ws, check=True, capture_output=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ws, check=True, capture_output=True, text=True
    ).stdout.strip()


def _branch_with(memory: Memory, name: str, path: str, content: str) -> Proposal:
    """A branch off the session head containing one file."""
    wt = memory.add_worktree(name)
    (wt.repo_path / path).write_text(content)
    checkpoint = wt.checkpoint("step-x", f"add {path}")
    return Proposal(step_id=name, branch=name, sha=checkpoint.sha, files=(path,))


# ------------------------------------------------------------------ primitives


def test_git_supports_merge_tree(refactor_workspace: Path) -> None:
    memory = Memory.open_session(refactor_workspace, "mt")
    assert supports_merge_tree(memory) is True


def test_preview_merge_touches_nothing(refactor_workspace: Path) -> None:
    """Phase 1 must be free: no refs move, the working tree is untouched."""
    ws = refactor_workspace
    memory = Memory.open_session(ws, "preview")
    before_head = memory.head().sha
    proposal = _branch_with(memory, "prev-a", "alpha.py", "a = 1\n")

    tree, _detail = preview_merge(memory, memory.head().sha, proposal.sha)

    assert tree is not None
    assert memory.head().sha == before_head, "no ref may move during a preview"
    assert not (ws / "alpha.py").exists(), "the working tree must be untouched"


def test_preview_merge_reports_a_conflict_without_raising(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    memory = Memory.open_session(ws, "conf")
    (ws / "shared.py").write_text("original\n")
    memory.checkpoint("base", "base")

    a = _branch_with(memory, "conf-a", "shared.py", "version A\n")
    b = _branch_with(memory, "conf-b", "shared.py", "version B\n")

    tree_a, _ = preview_merge(memory, memory.head().sha, a.sha)
    assert tree_a is not None
    merged_a = memory.repo.git.commit_tree(tree_a, "-p", memory.head().sha, "-m", "a").strip()
    tree_b, detail = preview_merge(memory, merged_a, b.sha)

    assert tree_b is None
    assert "shared.py" in detail


def test_accumulate_folds_disjoint_proposals(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    memory = Memory.open_session(ws, "acc")
    a = _branch_with(memory, "acc-a", "one.py", "one = 1\n")
    b = _branch_with(memory, "acc-b", "two.py", "two = 2\n")

    tree, conflicts = accumulate(memory, memory.head().sha, [a, b])

    assert conflicts == []
    assert tree is not None
    listing = memory.repo.git.ls_tree("-r", "--name-only", tree)
    assert "one.py" in listing and "two.py" in listing


# ------------------------------------------------------------------ atomicity


def test_a_late_conflict_leaves_the_branch_untouched(refactor_workspace: Path) -> None:
    """The bug two-phase exists to fix: no partial integration, ever."""
    ws = refactor_workspace
    memory = Memory.open_session(ws, "atomic")
    (ws / "shared.py").write_text("original\n")
    memory.checkpoint("base", "base")
    head_before = memory.head().sha

    good = _branch_with(memory, "atom-good", "safe.py", "safe = 1\n")
    a = _branch_with(memory, "atom-a", "shared.py", "version A\n")
    b = _branch_with(memory, "atom-b", "shared.py", "version B\n")

    result = integrate(memory, [good, a, b], gate=False)

    assert not result.ok
    assert result.conflicted
    assert memory.head().sha == head_before, "a conflict must strand nothing on the branch"
    assert not (ws / "safe.py").exists()


def test_a_clean_set_is_merged(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    memory = Memory.open_session(ws, "clean")
    a = _branch_with(memory, "clean-a", "one.py", "one = 1\n")
    b = _branch_with(memory, "clean-b", "two.py", "two = 2\n")

    result = integrate(memory, [a, b], gate=False)

    assert result.ok
    assert result.merged == ["clean-a", "clean-b"]
    assert memory.show("HEAD", "one.py").strip() == "one = 1"
    assert memory.show("HEAD", "two.py").strip() == "two = 2"


# ------------------------------------------------------------------ union gate


def test_union_gate_catches_a_semantic_conflict(refactor_workspace: Path) -> None:
    """Both merge cleanly as text; together they break. Only the gate sees it.

    One step renames a helper, another adds a caller for the old name — git
    reports success and the combined tree is broken.
    """
    ws = refactor_workspace
    memory = Memory.open_session(ws, "semantic")
    (ws / "helper.py").write_text("def old_name():\n    return 1\n")
    (ws / "test_combined.py").write_text(
        "import helper\n\n\ndef test_it():\n    assert helper.old_name() == 1\n"
    )
    memory.checkpoint("base", "base")

    # Step A renames the helper and updates the one caller it knows about.
    a_wt = memory.add_worktree("sem-a")
    (a_wt.repo_path / "helper.py").write_text("def new_name():\n    return 1\n")
    (a_wt.repo_path / "test_combined.py").write_text(
        "import helper\n\n\ndef test_it():\n    assert helper.new_name() == 1\n"
    )
    a_sha = a_wt.checkpoint("step-a", "rename helper").sha

    # Step B adds a brand-new module calling the OLD name. No textual overlap.
    b_wt = memory.add_worktree("sem-b")
    (b_wt.repo_path / "consumer.py").write_text("import helper\n\n\ndef go():\n    return helper.old_name()\n")
    (b_wt.repo_path / "test_consumer.py").write_text(
        "import consumer\n\n\ndef test_go():\n    assert consumer.go() == 1\n"
    )
    b_sha = b_wt.checkpoint("step-b", "add consumer").sha

    a = Proposal("step-a", "sem-a", a_sha, ("helper.py",), PYTEST_CMD)
    b = Proposal("step-b", "sem-b", b_sha, ("consumer.py",), PYTEST_CMD)

    tree, conflicts = accumulate(memory, memory.head().sha, [a, b])
    assert conflicts == [], "git considers these disjoint — that is the point"

    passed, detail = union_gate(memory, tree, [a, b])
    assert passed is False
    assert "exited" in (detail or "")


def test_gate_failure_stops_the_merge(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    memory = Memory.open_session(ws, "gatestop")
    (ws / "helper.py").write_text("def old_name():\n    return 1\n")
    memory.checkpoint("base", "base")
    head_before = memory.head().sha

    a_wt = memory.add_worktree("gs-a")
    (a_wt.repo_path / "helper.py").write_text("def new_name():\n    return 1\n")
    a_sha = a_wt.checkpoint("step-a", "rename").sha

    b_wt = memory.add_worktree("gs-b")
    (b_wt.repo_path / "test_uses_old.py").write_text(
        "import helper\n\n\ndef test_old():\n    assert helper.old_name() == 1\n"
    )
    b_sha = b_wt.checkpoint("step-b", "use old").sha

    result = integrate(
        memory,
        [
            Proposal("step-a", "gs-a", a_sha, ("helper.py",), PYTEST_CMD),
            Proposal("step-b", "gs-b", b_sha, ("test_uses_old.py",), PYTEST_CMD),
        ],
        gate=True,
    )

    assert result.gate_passed is False
    assert not result.ok
    assert memory.head().sha == head_before, "a broken combination must never land"


def test_gate_passes_for_genuinely_independent_work(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    memory = Memory.open_session(ws, "gateok")
    a = _branch_with(memory, "ok-a", "alpha.py", "alpha = 1\n")
    b = _branch_with(memory, "ok-b", "beta.py", "beta = 2\n")
    a = Proposal(a.step_id, a.branch, a.sha, a.files, PYTEST_CMD)
    b = Proposal(b.step_id, b.branch, b.sha, b.files, PYTEST_CMD)

    result = integrate(memory, [a, b], gate=True)
    assert result.gate_passed is True
    assert result.ok


def test_gate_without_commands_is_a_no_op(refactor_workspace: Path) -> None:
    memory = Memory.open_session(refactor_workspace, "nogate")
    a = _branch_with(memory, "ng-a", "x.py", "x = 1\n")
    passed, detail = union_gate(memory, memory.head().sha, [a])
    assert passed is True and detail is None


# ------------------------------------------------------------------ kernel wiring


def _parallel_plan() -> Plan:
    return Plan(
        task="parallel",
        steps=[
            Step("step-01", "seed", Verification(kind="shell", command="true")),
            Step("step-02", "module a", Verification(kind="shell", command="true"),
                 depends_on=["step-01"]),
            Step("step-03", "module b", Verification(kind="shell", command="true"),
                 depends_on=["step-01"]),
        ],
    )


def _worker(step: Step, plan: Plan) -> WorkerResult:
    path = current_step().workspace
    if step.id == "step-01":
        (path / "seed.py").write_text("seed = 1\n")
    else:
        (path / f"{step.id}.py").write_text(f"# {step.id}\n")
    return WorkerResult("", 0, "end_turn")


def test_kernel_two_phase_merges_a_parallel_wave(parallel_workspace: Path) -> None:
    ws = parallel_workspace
    result = Kernel(workspace=ws, max_retries=0, two_phase_merge=True, union_gate=False).run(
        task="tp",
        spec=_spec(),
        session_id="tp",
        plan_override=_parallel_plan(),
        worker_override=_worker,
    )

    assert result.status == "completed", result.failure_reason
    memory = Memory(ws, "taste/session-tp")
    tracked = memory.repo.git.ls_files()
    assert "step-02.py" in tracked and "step-03.py" in tracked


def test_kernel_default_still_uses_the_sequential_path(parallel_workspace: Path) -> None:
    """Off by default: the existing merge behavior is unchanged."""
    ws = parallel_workspace
    events = []
    result = Kernel(workspace=ws, max_retries=0, on_event=events.append).run(
        task="seq",
        spec=_spec(),
        session_id="seq",
        plan_override=_parallel_plan(),
        worker_override=_worker,
    )

    assert result.status == "completed"
    assert not [e for e in events if e.kind.startswith("merge.")]
    assert len([e for e in events if e.kind == "worktree.merge"]) == 2
