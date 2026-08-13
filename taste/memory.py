"""Git-backed memory for the Agent OS.

The thesis in one file: branches are execution contexts, commits are
checkpoints, `git show` is demand paging, `git reset` is rollback. Nothing
about the agent loop lives in here — this file only knows how to persist,
retrieve, and revert state.

For parallel execution (Milestone B), this module also exposes ``add_worktree``
/ ``remove_worktree`` / ``merge_branch`` primitives. Each worktree is a
real physical working tree at a sibling path — the blog's *processes have
their own address spaces* analogy made literal in git.
"""

from __future__ import annotations

import contextlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from git import Repo
from git.exc import GitCommandError


@dataclass(frozen=True)
class Checkpoint:
    """A named commit on the session branch."""

    step_id: str
    sha: str
    message: str
    parent_sha: str | None

    @property
    def short_sha(self) -> str:
        return self.sha[:7]


class Memory:
    """Git-backed persistent memory for a single kernel run.

    One Memory instance corresponds to one session branch in a git repo. Every
    state transition the kernel wants to preserve becomes a commit; every
    rollback is a `git reset --hard` back to an earlier commit.
    """

    CHECKPOINT_TRAILER = "Taste-Checkpoint"

    def __init__(self, repo_path: Path, branch: str) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.branch = branch
        self.repo = Repo(self.repo_path)
        if self.repo.bare:
            raise ValueError(f"{self.repo_path} is a bare repository; need a working tree.")

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    def open_session(
        cls,
        repo_path: Path,
        session_id: str,
        *,
        base_ref: str = "HEAD",
    ) -> Memory:
        """Create (or resume) a session branch rooted at ``base_ref``.

        The branch name is ``taste/session-{session_id}``. If the branch
        already exists we resume on it; otherwise we fork it from ``base_ref``.

        Also installs a handful of local-only git excludes so that generated
        artifacts (Python bytecode, pytest caches) don't end up in the
        per-step checkpoints — they'd otherwise conflict at merge time when
        parallel workers each produce their own copies.
        """
        repo_path = Path(repo_path).resolve()
        repo = Repo(repo_path)
        branch = f"taste/session-{session_id}"

        if branch in [h.name for h in repo.heads]:
            repo.git.checkout(branch)
        else:
            repo.git.checkout("-b", branch, base_ref)

        _install_local_excludes(repo_path)
        return cls(repo_path, branch)

    # ------------------------------------------------------------------ writes

    def checkpoint(
        self,
        step_id: str,
        message: str,
        *,
        allow_empty: bool = False,
    ) -> Checkpoint:
        """Stage everything under the working tree and commit it as a checkpoint.

        The checkpoint's ``step_id`` is embedded as a git trailer on the commit
        message, so the raw git log is enough to reconstruct the execution
        trajectory (no sidecar file needed).
        """
        self._ensure_on_branch()
        self.repo.git.add("--all", ".")

        if not allow_empty and not self._has_staged_changes():
            head = self.head()
            return Checkpoint(
                step_id=step_id,
                sha=head.sha,
                message=f"(no-op) {message}",
                parent_sha=head.parent_sha,
            )

        commit_msg = f"{message}\n\n{self.CHECKPOINT_TRAILER}: {step_id}"
        args = ["-m", commit_msg]
        if allow_empty:
            args.insert(0, "--allow-empty")
        self.repo.git.commit(*args)

        head_commit = self.repo.head.commit
        return Checkpoint(
            step_id=step_id,
            sha=head_commit.hexsha,
            message=message,
            parent_sha=head_commit.parents[0].hexsha if head_commit.parents else None,
        )

    def tag_checkpoint(self, checkpoint: Checkpoint, tag: str) -> None:
        """Tag a checkpoint for quick named reference (e.g. 'pre-step-03')."""
        self.repo.git.tag("-f", tag, checkpoint.sha)

    # ------------------------------------------------------------------ rollback

    def rollback_to(self, checkpoint: Checkpoint) -> None:
        """Hard-reset the branch back to ``checkpoint``.

        This is the move that makes the step-87 problem tractable: once a
        checkpoint is tagged, discarding every commit on top of it is a single
        reversible operation.
        """
        self._ensure_on_branch()
        self.repo.git.reset("--hard", checkpoint.sha)

    def rollback_last(self) -> Checkpoint:
        """Undo the most recent checkpoint; return the new HEAD checkpoint."""
        history = self.log(limit=2)
        if len(history) < 2:
            raise RuntimeError("Cannot rollback: no prior checkpoint on this branch.")
        previous = history[1]
        self.rollback_to(previous)
        return previous

    # ------------------------------------------------------------------ reads (demand paging)

    def show(self, ref: str, path: str) -> str:
        """`git show <ref>:<path>` — load a single file from any commit on demand.

        This is the paging primitive: an agent loads exactly the 2K tokens it
        needs from another branch's output instead of checking out the whole
        working tree.
        """
        try:
            return self.repo.git.show(f"{ref}:{path}")
        except GitCommandError as e:
            raise FileNotFoundError(f"{path} at {ref}: {e.stderr.strip()}") from e

    def diff(self, from_ref: str, to_ref: str = "HEAD", *, path: str | None = None) -> str:
        args = [f"{from_ref}..{to_ref}"]
        if path:
            args += ["--", path]
        return self.repo.git.diff(*args)

    def diff_pending(self, from_ref: str) -> str:
        """Diff the *uncommitted* working tree against ``from_ref``.

        The Monitor runs before the kernel checkpoints, so at verification
        time the worker's changes exist only in the working tree. Staging
        first (``git add --all``) is what makes untracked files visible to
        ``git diff``; it commits nothing, and the kernel's own checkpoint
        stages identically a moment later.
        """
        self.repo.git.add("--all", ".")
        return self.repo.git.diff("--cached", from_ref)

    def changed_files(self, from_ref: str) -> list[str]:
        """Paths that differ between ``from_ref`` and the working tree."""
        self.repo.git.add("--all", ".")
        out = self.repo.git.diff("--cached", "--name-only", from_ref)
        return [line for line in out.splitlines() if line.strip()]

    def log(self, *, limit: int | None = None, branch: str | None = None) -> list[Checkpoint]:
        """Return checkpoints (most-recent first) for the session branch."""
        target = branch or self.branch
        commits = list(self.repo.iter_commits(target, max_count=limit))
        return [self._commit_to_checkpoint(c) for c in commits]

    def head(self) -> Checkpoint:
        return self._commit_to_checkpoint(self.repo.head.commit)

    def working_tree_dirty(self) -> bool:
        return self.repo.is_dirty(untracked_files=True)

    # ------------------------------------------------------------------ worktrees

    def add_worktree(
        self,
        branch: str,
        *,
        base: str | Checkpoint | None = None,
    ) -> Memory:
        """Spawn a new worktree on a fresh branch, return a Memory bound to it.

        Workers in a parallel wave each get their own worktree so filesystem
        writes cannot collide. The worktree's physical path lives under
        ``<parent>/.taste-worktrees/<slug>`` — outside the main working tree
        so it's invisible to the primary session.
        """
        base_ref: str
        if isinstance(base, Checkpoint):
            base_ref = base.sha
        elif base is None:
            base_ref = self.branch
        else:
            base_ref = base

        if branch in [h.name for h in self.repo.heads]:
            raise ValueError(f"branch already exists: {branch}")

        wt_root = self.repo_path.parent / ".taste-worktrees" / _slug(self.branch)
        wt_root.mkdir(parents=True, exist_ok=True)
        wt_path = wt_root / _slug(branch)
        if wt_path.exists():
            raise FileExistsError(f"worktree path already exists: {wt_path}")

        self.repo.git.worktree("add", "-b", branch, str(wt_path), base_ref)
        return Memory(wt_path, branch)

    def remove_worktree(self, other: Memory, *, force: bool = True) -> None:
        """Prune the worktree that ``other`` is bound to. Safe to call twice."""
        args = ["remove"]
        if force:
            args.append("--force")
        args.append(str(other.repo_path))
        try:
            self.repo.git.worktree(*args)
        except GitCommandError:
            # Already gone, or git refused — fall back to manual cleanup.
            if other.repo_path.exists():
                shutil.rmtree(other.repo_path, ignore_errors=True)
            self.repo.git.worktree("prune")

    def merge_branch(
        self,
        source: str,
        *,
        message: str | None = None,
    ) -> Checkpoint:
        """Merge ``source`` into the current (session) branch. --no-ff to keep
        the topology visible in ``git log --graph``.

        Raises :class:`MergeConflict` if git reports conflicts — the blog's
        *merge conflict = coordination signal* made explicit as a typed
        exception the Orchestrator can catch.
        """
        self._ensure_on_branch()
        msg = message or f"merge: {source} into {self.branch}"
        try:
            self.repo.git.merge("--no-ff", "-m", msg, source)
        except GitCommandError as exc:
            # Leave the working tree in its conflicted state; abort and raise.
            with contextlib.suppress(GitCommandError):
                self.repo.git.merge("--abort")
            raise MergeConflict(source=source, target=self.branch, detail=str(exc)) from exc
        return self.head()

    # ------------------------------------------------------------------ helpers

    def _ensure_on_branch(self) -> None:
        current = self.repo.active_branch.name
        if current != self.branch:
            self.repo.git.checkout(self.branch)

    def _has_staged_changes(self) -> bool:
        return bool(self.repo.git.diff("--cached", "--name-only"))

    def _commit_to_checkpoint(self, commit) -> Checkpoint:  # type: ignore[no-untyped-def]
        message, step_id = _parse_commit_message(commit.message)
        return Checkpoint(
            step_id=step_id or commit.hexsha[:7],
            sha=commit.hexsha,
            message=message,
            parent_sha=commit.parents[0].hexsha if commit.parents else None,
        )


def _parse_commit_message(raw: str) -> tuple[str, str | None]:
    """Split a commit message into (subject, step_id) using the trailer."""
    lines = raw.rstrip().splitlines()
    step_id: str | None = None
    subject_lines: list[str] = []
    for line in lines:
        if line.startswith(f"{Memory.CHECKPOINT_TRAILER}:"):
            step_id = line.split(":", 1)[1].strip()
        else:
            subject_lines.append(line)
    subject = "\n".join(subject_lines).strip()
    return subject, step_id


def _slug(s: str) -> str:
    """Branch name → filesystem-safe slug."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


_LOCAL_EXCLUDE_MARKER = "# taste-os: local excludes (never committed)"
_LOCAL_EXCLUDES = [
    _LOCAL_EXCLUDE_MARKER,
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    ".taste/dashboard.html",
]


def _install_local_excludes(repo_path: Path) -> None:
    """Write common Python generated-artifact patterns to .git/info/exclude.

    Uses git's per-repo local exclude file so we don't touch a tracked
    .gitignore. This keeps workers from committing bytecode/pytest caches
    that would otherwise conflict at merge time in parallel waves.
    """
    exclude = repo_path / ".git" / "info" / "exclude"
    if not exclude.parent.is_dir():
        return
    current = exclude.read_text() if exclude.exists() else ""
    if _LOCAL_EXCLUDE_MARKER in current:
        return
    with exclude.open("a") as f:
        if current and not current.endswith("\n"):
            f.write("\n")
        f.write("\n".join(_LOCAL_EXCLUDES) + "\n")


class MergeConflict(RuntimeError):
    """The Orchestrator-visible form of a git merge conflict.

    Raised by :meth:`Memory.merge_branch` when ``git merge`` reports conflicts.
    The merge is aborted before the exception surfaces, so the session branch
    remains clean — the Orchestrator can route the conflict or escalate.
    """

    def __init__(self, *, source: str, target: str, detail: str) -> None:
        super().__init__(f"conflict merging {source} into {target}: {detail}")
        self.source = source
        self.target = target
        self.detail = detail
