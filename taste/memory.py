"""Git-backed memory for the Agent OS.

The thesis in one file: branches are execution contexts, commits are
checkpoints, `git show` is demand paging, `git reset` is rollback. Nothing
about the agent loop lives in here — this file only knows how to persist,
retrieve, and revert state.
"""

from __future__ import annotations

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
        """
        repo = Repo(Path(repo_path).resolve())
        branch = f"taste/session-{session_id}"

        if branch in [h.name for h in repo.heads]:
            repo.git.checkout(branch)
        else:
            repo.git.checkout("-b", branch, base_ref)

        return cls(Path(repo_path), branch)

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

    def log(self, *, limit: int | None = None, branch: str | None = None) -> list[Checkpoint]:
        """Return checkpoints (most-recent first) for the session branch."""
        target = branch or self.branch
        commits = list(self.repo.iter_commits(target, max_count=limit))
        return [self._commit_to_checkpoint(c) for c in commits]

    def head(self) -> Checkpoint:
        return self._commit_to_checkpoint(self.repo.head.commit)

    def working_tree_dirty(self) -> bool:
        return self.repo.is_dirty(untracked_files=True)

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
