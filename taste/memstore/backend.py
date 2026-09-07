"""Git as the backend of the memory layer: plumbing only.

Nothing above this file knows a git command. Everything here is a thin,
named wrapper around one plumbing operation, chosen so that the store can
compose them into atomic sequences. The two that matter:

* ``commit_tree`` makes a commit object that no ref points at. It is
  invisible until a ref moves to it, which is what lets the store build a
  whole state (commit plus notes) before publishing it.
* ``cas_update_ref`` moves a ref only if it still points where we think it
  does. That is the publish step, and it is the only step that can fail
  because someone else got there first.

Notes are written under a repo-wide lock because ``git notes`` is a
read-modify-write of the notes tree and two writers can lose an update.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from git import Repo
from git.exc import GitCommandError

from taste.memory import _install_local_excludes

# One held description per lock path, process-wide; see ``GitBackend.lock``.
_HELD: dict[Path, tuple[IO[str], int]] = {}
_REGISTRY_LOCK = threading.RLock()

ZERO_SHA = "0" * 40
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

IDENTITY_NAME = "taste"
IDENTITY_EMAIL = "taste@localhost"


@dataclass(frozen=True)
class MergeTreeResult:
    tree: str
    conflicted_paths: tuple[str, ...]
    messages: str


class GitBackend:
    """One backend per working tree; all of them share the object store."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        self.repo = Repo(self.path)
        if self.repo.bare:
            raise ValueError(f"{self.path} is bare; the memory layer needs a working tree")
        self._ensure_identity()
        _install_local_excludes(self.common_dir.parent if self.common_dir.name == ".git" else self.path)

    # ---------------------------------------------------------------- basics

    @classmethod
    def init(cls, path: Path) -> GitBackend:
        path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        if not (path / ".git").exists():
            Repo.init(path)
        return cls(path)

    def close(self) -> None:
        self.repo.close()

    @property
    def common_dir(self) -> Path:
        raw = self.repo.git.rev_parse("--git-common-dir")
        p = Path(raw)
        return (self.path / p).resolve() if not p.is_absolute() else p

    def _ensure_identity(self) -> None:
        reader = self.repo.config_reader()
        have_name = reader.has_option("user", "name")
        have_email = reader.has_option("user", "email")
        if have_name and have_email:
            return
        with self.repo.config_writer() as w:
            if not have_name:
                w.set_value("user", "name", IDENTITY_NAME)
            if not have_email:
                w.set_value("user", "email", IDENTITY_EMAIL)

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        """Repo-wide exclusive lock, shared by every worktree of this repo.

        Re-entrant within a process. ``flock`` is per open file description,
        so a nested ``lock()`` that opened the file again would block on
        itself; instead one description per lock path is held for the
        duration of the outermost holder, and nested holders only count.
        Threads serialise on the registry lock, which is what we want: git
        notes are a read-modify-write of one tree.
        """
        lock_path = self.common_dir / "memstore.lock"
        with _REGISTRY_LOCK:
            fh, depth = _HELD.get(lock_path, (None, 0))
            if fh is None:
                fh = open(lock_path, "a+")  # noqa: SIM115  held past this block by design
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            _HELD[lock_path] = (fh, depth + 1)
        try:
            yield
        finally:
            with _REGISTRY_LOCK:
                fh, depth = _HELD[lock_path]
                if depth == 1:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    fh.close()
                    del _HELD[lock_path]
                else:
                    _HELD[lock_path] = (fh, depth - 1)

    # ---------------------------------------------------------------- refs

    def ref_sha(self, ref: str) -> str | None:
        try:
            return self.repo.git.rev_parse("--verify", "--quiet", f"{ref}^{{commit}}")
        except GitCommandError:
            return None

    def cas_update_ref(self, ref: str, new: str, old: str | None) -> bool:
        """Move ``ref`` to ``new`` only if it currently points at ``old``.

        ``old=None`` means the ref must not exist. Returns False if the
        compare failed; nothing is written in that case.
        """
        expected = old if old is not None else ZERO_SHA
        try:
            self.repo.git.update_ref(ref, new, expected)
            return True
        except GitCommandError:
            return False

    def delete_ref(self, ref: str) -> None:
        with contextlib.suppress(GitCommandError):
            self.repo.git.update_ref("-d", ref)

    def for_each_ref(self, prefix: str) -> list[tuple[str, str]]:
        out = self.repo.git.for_each_ref("--format=%(refname) %(objectname)", prefix)
        pairs: list[tuple[str, str]] = []
        for line in out.splitlines():
            name, _, sha = line.partition(" ")
            if name and sha:
                pairs.append((name, sha))
        return pairs

    def head_commit(self) -> str | None:
        return self.ref_sha("HEAD")

    # ---------------------------------------------------------------- objects

    def stage_all(self) -> None:
        self.repo.git.add("--all", ".")

    def write_tree(self) -> str:
        return self.repo.git.write_tree()

    def commit_tree(self, tree: str, parents: list[str], message: str) -> str:
        args: list[str] = [tree]
        for p in parents:
            args += ["-p", p]
        args += ["-m", message]
        return self.repo.git.commit_tree(*args)

    def tree_of(self, commit: str) -> str:
        return self.repo.git.rev_parse(f"{commit}^{{tree}}")

    def parents_of(self, commit: str) -> list[str]:
        out = self.repo.git.rev_list("--parents", "-n", "1", commit)
        parts = out.split()
        return parts[1:]

    def blob_at(self, commit: str, path: str) -> str | None:
        try:
            return self.repo.git.rev_parse("--verify", "--quiet", f"{commit}:{path}")
        except GitCommandError:
            return None

    def show(self, commit: str, path: str) -> str | None:
        try:
            return self.repo.git.show(f"{commit}:{path}", strip_newline_in_stdout=False)
        except GitCommandError:
            return None

    def cat_blob(self, sha: str) -> str:
        return self.repo.git.cat_file("-p", sha, strip_newline_in_stdout=False)

    def hash_blob(self, content: str) -> str:
        with self._tempfile(content) as path:
            return self.repo.git.hash_object("-w", str(path))

    @contextlib.contextmanager
    def _tempfile(self, content: str) -> Iterator[Path]:
        """A private file holding ``content``, for commands that read a path.

        Lives beside the lock so it is on the same filesystem as the repo and
        is never mistaken for part of a working tree.
        """
        path = self.common_dir / f"memstore.tmp.{os.getpid()}.{id(content)}"
        path.write_text(content)
        try:
            yield path
        finally:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()

    def ls_files(self, commit: str) -> list[str]:
        out = self.repo.git.ls_tree("-r", "--name-only", commit)
        return [line for line in out.splitlines() if line]

    def diff_names(self, a: str, b: str) -> list[tuple[str, str]]:
        """(status, path) for every path that differs between two commits."""
        out = self.repo.git.diff_tree("--name-status", "-r", a, b)
        rows: list[tuple[str, str]] = []
        for line in out.splitlines():
            if not line:
                continue
            status, _, path = line.partition("\t")
            rows.append((status[:1], path))
        return rows

    def is_ancestor(self, maybe_ancestor: str, commit: str) -> bool:
        try:
            self.repo.git.merge_base("--is-ancestor", maybe_ancestor, commit)
            return True
        except GitCommandError:
            return False

    def merge_base(self, a: str, b: str) -> str | None:
        try:
            return self.repo.git.merge_base(a, b)
        except GitCommandError:
            return None

    def rev_list_first_parent(self, commit: str, limit: int | None = None) -> list[str]:
        args = ["--first-parent"]
        if limit is not None:
            args += ["-n", str(limit)]
        out = self.repo.git.rev_list(*args, commit)
        return [line for line in out.splitlines() if line]

    # ---------------------------------------------------------------- notes

    def note_get(self, namespace: str, commit: str) -> str | None:
        try:
            return self.repo.git.notes("--ref", namespace, "show", commit, strip_newline_in_stdout=False)
        except GitCommandError:
            return None

    def note_set(self, namespace: str, commit: str, text: str) -> None:
        with self.lock(), self._tempfile(text) as path:
            self.repo.git.notes("--ref", namespace, "add", "-f", "--allow-empty", "--no-stripspace", "-F", str(path), commit)

    def note_copy(self, namespace: str, src: str, dst: str) -> None:
        text = self.note_get(namespace, src)
        if text is not None:
            self.note_set(namespace, dst, text)

    # ---------------------------------------------------------------- worktree

    def reset_hard_to_head(self) -> None:
        self.repo.git.reset("--hard")
        self.repo.git.clean("-fd")

    def worktree_add(self, path: Path, ref: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.repo.git.worktree("add", str(path), ref)

    def worktree_remove(self, path: Path) -> None:
        with contextlib.suppress(GitCommandError):
            self.repo.git.worktree("remove", "--force", str(path))

    # ---------------------------------------------------------------- merge

    def merge_tree(self, ours: str, theirs: str) -> MergeTreeResult:
        """Three-way merge of two commits into a tree, without touching anything.

        ``git merge-tree --write-tree`` finds the base itself. On conflict it
        exits 1 and lists the conflicted entries; the tree it wrote contains
        the usual conflict markers for those paths.
        """
        status, out, err = self.repo.git.execute(
            ["git", "merge-tree", "--write-tree", ours, theirs],
            with_extended_output=True,
            with_exceptions=False,
        )
        lines = out.splitlines()
        if not lines:
            raise RuntimeError(f"merge-tree produced no output: {err}")
        tree = lines[0].strip()
        conflicted: list[str] = []
        messages: list[str] = []
        in_entries = True
        for line in lines[1:]:
            if in_entries:
                if line.strip() == "":
                    in_entries = False
                    continue
                # "<mode> <sha> <stage>\t<path>"
                _, _, path = line.partition("\t")
                if path and path not in conflicted:
                    conflicted.append(path)
            else:
                messages.append(line)
        if status not in (0, 1):
            raise RuntimeError(f"merge-tree failed ({status}): {err or out}")
        return MergeTreeResult(tree=tree, conflicted_paths=tuple(conflicted), messages="\n".join(messages))

    def tree_with_blob(self, tree: str, path: str, blob: str) -> str:
        """Return a new tree equal to ``tree`` with ``path`` replaced by ``blob``.

        Done through a throwaway index so the real index and working tree
        are never touched.
        """
        index_path = self.common_dir / f"memstore.index.{os.getpid()}"
        env = {"GIT_INDEX_FILE": str(index_path)}
        try:
            with self.repo.git.custom_environment(**env):
                self.repo.git.read_tree(tree)
                self.repo.git.update_index("--add", "--cacheinfo", f"100644,{blob},{path}")
                return self.repo.git.write_tree()
        finally:
            with contextlib.suppress(FileNotFoundError):
                index_path.unlink()
