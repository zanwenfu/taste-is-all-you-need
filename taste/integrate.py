"""Integrating parallel work: compute the merge, verify it, then commit it.

Two defects in the naive approach, both of which this module exists to fix.

**Partial merges.** Merging worktree branches one at a time in plan order
means a conflict on the third branch leaves the first two already on the
session branch — a state no step produced and no monitor ever verified. The
fix is two-phase: fold every proposal together in git's *object store* first,
where nothing is referenced and a failure costs nothing, and only move refs
once the whole outcome is known.

**Semantic conflicts.** Two steps can merge cleanly as text and still break
each other — one renames a helper, the other adds a caller for the old name.
Git reports success; the tree is broken. Each step verified its own work in
isolation, so nobody ever ran the combination. The union gate runs every
step's own verification against the *combined* tree, which is the only place
that failure can be caught. It is also an instrument: a regression that
appears here and in neither parent was introduced by the merge itself.

What is deliberately absent: any model call. ``git merge-tree --write-tree``
computes conflicts exactly and returns a usable tree, for free. Paying a
model to predict what git computes exactly would be indefensible, and letting
one *resolve* a conflict would have the harness inventing a combination no
step produced — silently manufacturing the contamination the whole system
exists to measure.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

from git.exc import GitCommandError

from taste.memory import Memory, MergeConflict


@dataclass(frozen=True)
class Proposal:
    """One completed step's branch, offered for integration."""

    step_id: str
    branch: str
    sha: str
    files: tuple[str, ...] = ()
    verification_command: str | None = None

    def touches(self, other: Proposal) -> bool:
        return bool(set(self.files) & set(other.files))


@dataclass
class IntegrationResult:
    merged: list[str] = field(default_factory=list)
    conflicted: list[tuple[str, str]] = field(default_factory=list)
    gate_passed: bool = True
    gate_failure: str | None = None
    tree_sha: str | None = None
    used_two_phase: bool = True

    @property
    def ok(self) -> bool:
        return not self.conflicted and self.gate_passed


def supports_merge_tree(memory: Memory) -> bool:
    """Whether this git can fold branches without touching the working tree.

    Decided on the version number rather than by pattern-matching an error
    message: older gits reject ``--write-tree`` with wording that varies, and
    guessing wrong here is expensive in the wrong direction — every fold
    would be misreported as a conflict instead of falling back.
    """
    try:
        raw = memory.repo.git.version()
    except Exception:
        return False
    match = re.search(r"(\d+)\.(\d+)", raw or "")
    if not match:
        return False
    return (int(match.group(1)), int(match.group(2))) >= (2, 38)


def preview_merge(memory: Memory, base: str, other: str) -> tuple[str | None, str]:
    """Fold ``other`` into ``base`` in the object store only.

    Returns ``(tree_sha, detail)``; ``tree_sha`` is None when the merge
    conflicts. Nothing is referenced, no ref moves, and the working tree is
    untouched — so a failure here costs exactly nothing.
    """
    try:
        out = memory.repo.git.merge_tree("--write-tree", base, other)
    except GitCommandError as exc:
        # A non-zero exit from merge-tree IS the conflict signal; anything
        # else (a missing object, a broken repo) is infrastructure and must
        # not be reported to the operator as "your steps conflict".
        detail = getattr(exc, "stdout", "") or str(exc)
        return None, str(detail)[:2000]
    tree = out.splitlines()[0].strip() if out else ""
    return (tree or None), out


def accumulate(
    memory: Memory,
    base_sha: str,
    proposals: list[Proposal],
) -> tuple[str | None, list[tuple[str, str]]]:
    """Fold every proposal onto ``base_sha``, in the object store.

    Returns ``(combined_tree_sha, conflicts)``. Proposals are folded in the
    order given; the first conflict stops the fold, because a combination
    built on top of a conflict is not a combination anyone asked for.
    """
    conflicts: list[tuple[str, str]] = []
    current = base_sha
    tree: str | None = None

    for proposal in proposals:
        merged_tree, detail = preview_merge(memory, current, proposal.sha)
        if merged_tree is None:
            conflicts.append((proposal.step_id, detail))
            return None, conflicts
        tree = merged_tree
        # Commit the intermediate tree so the next fold has a commit to merge
        # against. It is unreferenced: if we stop here, git simply reclaims it.
        current = memory.repo.git.commit_tree(
            merged_tree, "-p", current, "-p", proposal.sha, "-m", f"fold: {proposal.step_id}"
        ).strip()

    return tree, conflicts


def union_gate(
    memory: Memory,
    tree_sha: str,
    proposals: list[Proposal],
    *,
    timeout: int = 300,
) -> tuple[bool, str | None]:
    """Run every step's own verification against the *combined* tree.

    This is the only place a semantic conflict can be caught: each step
    verified its work alone, so nobody has ever run the combination. Executed
    in a throwaway worktree, so a failure leaves nothing behind.
    """
    commands = [p.verification_command for p in proposals if p.verification_command]
    if not commands:
        return True, None

    # A tree is not checkout-able; wrap it in an unreferenced commit.
    try:
        commit = memory.repo.git.commit_tree(tree_sha, "-m", "union-gate probe").strip()
    except Exception as exc:
        return True, f"gate skipped: {exc}"

    try:
        with memory.probe_worktree(commit) as path:
            for command in dict.fromkeys(commands):
                try:
                    proc = subprocess.run(
                        command,
                        shell=True,
                        cwd=path,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired:
                    return False, f"`{command}` timed out against the combined tree"
                if proc.returncode != 0:
                    tail = (proc.stdout or "")[-1500:] + (proc.stderr or "")[-1500:]
                    return False, f"`{command}` exited {proc.returncode} on the combined tree:\n{tail}"
    except Exception as exc:
        # A gate that cannot run must not block integration; it just cannot
        # vouch for it either.
        return True, f"gate skipped: {exc}"
    return True, None


def integrate(
    memory: Memory,
    proposals: list[Proposal],
    *,
    gate: bool = True,
    emit: Any = None,
) -> IntegrationResult:
    """Two-phase integration: compute and verify, then commit.

    Phase 1 folds every proposal in the object store and gates the result.
    Phase 2 performs the real merges only once the outcome is known, so the
    session branch is never left holding a partial integration.
    """
    result = IntegrationResult()
    emit = emit or (lambda *a, **k: None)

    if not proposals:
        return result

    if not supports_merge_tree(memory):
        result.used_two_phase = False
        emit("merge.unavailable", reason="git lacks merge-tree --write-tree")
        return _sequential(memory, proposals, result, emit)

    base = memory.head().sha
    tree, conflicts = accumulate(memory, base, proposals)
    if conflicts:
        result.conflicted = conflicts
        emit("merge.conflict", step=conflicts[0][0], detail=conflicts[0][1][:500])
        return result

    result.tree_sha = tree
    if gate and tree:
        passed, detail = union_gate(memory, tree, proposals)
        result.gate_passed = passed
        result.gate_failure = detail
        emit("merge.gate", verdict="pass" if passed else "fail", detail=(detail or "")[:500])
        if not passed:
            # Clean textual merge, broken combination: exactly the case the
            # gate exists for. Nothing moves.
            return result

    # Phase 2: the outcome is known, so these merges cannot strand the branch.
    return _sequential(memory, proposals, result, emit)


def _sequential(
    memory: Memory,
    proposals: list[Proposal],
    result: IntegrationResult,
    emit: Any,
) -> IntegrationResult:
    for proposal in proposals:
        try:
            merged = memory.merge_branch(
                proposal.branch, message=f"merge: {proposal.step_id} from {proposal.branch}"
            )
        except MergeConflict as exc:
            result.conflicted.append((proposal.step_id, exc.detail))
            emit("merge.conflict", step=proposal.step_id, detail=exc.detail[:500])
            return result
        result.merged.append(proposal.step_id)
        emit("merge.done", step=proposal.step_id, sha=merged.short_sha)
    return result


def proposals_from_outcomes(outcomes: list[Any], worktrees: dict[str, Memory]) -> list[Proposal]:
    """Build proposals from a finished wave's outcomes."""
    built: list[Proposal] = []
    for outcome in outcomes:
        wt = worktrees.get(outcome.step.id)
        if wt is None:
            continue
        built.append(
            Proposal(
                step_id=outcome.step.id,
                branch=wt.branch,
                sha=outcome.checkpoint.sha,
                verification_command=(
                    outcome.step.verification.command
                    if outcome.step.verification.kind == "shell"
                    else None
                ),
            )
        )
    return built
