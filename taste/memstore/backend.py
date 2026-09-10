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

Two rules learned from an audit that broke the first version of this file:

* **Content is bytes.** ``git`` output is decoded with ``surrogateescape`` at
  this boundary and re-encoded the same way on the way in, so a blob that is
  not UTF-8 round-trips unchanged instead of raising deep inside a merge.
* **Nothing is excluded.** The first version imported the legacy harness's
  ``.git/info/exclude`` writer, so caches and bytecode were silently never
  checkpointed by a layer whose first invariant is that nothing is lost. A
  caller that wants exclusions now has to ask for them.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from git import Repo
from git.exc import GitCommandError

# One held description per lock path, process-wide; see ``GitBackend.lock``.
_HELD: dict[Path, tuple[IO[str], int]] = {}
_REGISTRY_LOCK = threading.RLock()

ZERO_SHA = "0" * 40
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

MODE_FILE = "100644"
MODE_EXEC = "100755"
MODE_SYMLINK = "120000"
MODE_GITLINK = "160000"
#: Modes whose content is not a plain file, so a content merge is meaningless.
UNMERGEABLE_MODES = frozenset({MODE_SYMLINK, MODE_GITLINK})

IDENTITY_NAME = "taste"
IDENTITY_EMAIL = "taste@localhost"


class RefUpdateFailed(RuntimeError):
    """A ref write failed for a reason that is not a lost compare-and-swap."""


class NoteConflict(RuntimeError):
    """A note already exists on this object with different content.

    Reachable only if two states share a commit id, which the store prevents
    by making every state's message unique. It is raised rather than
    overwritten because the first version of this file used ``notes add -f``
    and destroyed one agent's transcript when ids did collide.
    """


def _dec(raw: bytes) -> str:
    return raw.decode("utf-8", errors="surrogateescape")


def _enc(text: str) -> bytes:
    return text.encode("utf-8", errors="surrogateescape")


@dataclass(frozen=True)
class MergeTreeResult:
    tree: str
    conflicted_paths: tuple[str, ...]
    messages: str


@dataclass(frozen=True)
class TreeEntry:
    mode: str
    sha: str
    path: str

    @property
    def is_mergeable_file(self) -> bool:
        return self.mode not in UNMERGEABLE_MODES


class GitBackend:
    """One backend per working tree; all of them share the object store."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        self.repo = Repo(self.path)
        if self.repo.bare:
            raise ValueError(f"{self.path} is bare; the memory layer needs a working tree")
        self._ensure_identity()

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
        """The shared ``.git`` of this repository, identical in every worktree.

        Memoized because it is a git subprocess costing ~7.7 ms and the answer
        cannot change for the life of this backend -- a worktree does not
        migrate to another repository. It sits under ``turn()``, which the
        layer above calls from a hook on the hot path, where that was most of
        a 15 ms budget spent re-asking git a constant.
        """
        cached = getattr(self, "_common_dir", None)
        if cached is None:
            raw = self.repo.git.rev_parse("--git-common-dir")
            p = Path(raw)
            cached = (self.path / p).resolve() if not p.is_absolute() else p
            self._common_dir = cached
        return cached

    @property
    def identity(self) -> str:
        """A stable short id for the repository, for keying paths outside it."""
        return hashlib.sha256(str(self.common_dir).encode()).hexdigest()[:12]

    def _ensure_identity(self) -> None:
        reader = self.repo.config_reader()
        have_name = reader.has_option("user", "name")
        have_email = reader.has_option("user", "email")
        # ``core.quotePath`` defaults on, which makes git report café.txt as
        # "caf\303\251.txt" wherever it prints a path. The parsers here read
        # NUL-delimited output and are immune, but ``merge-tree`` has no such
        # form, and a quoted path there produced a conflict on a name that
        # matched nothing. Turning it off at the repository settles it once.
        have_quoting = reader.has_option("core", "quotepath")
        if have_name and have_email and have_quoting:
            return
        with self.repo.config_writer() as w:
            if not have_name:
                w.set_value("user", "name", IDENTITY_NAME)
            if not have_email:
                w.set_value("user", "email", IDENTITY_EMAIL)
            if not have_quoting:
                w.set_value("core", "quotepath", "false")

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        """Repo-wide exclusive lock, shared by every worktree of this repo.

        Re-entrant within a process. ``flock`` is per open file description,
        so a nested ``lock()`` that opened the file again would block on
        itself; instead one description per lock path is held for the
        duration of the outermost holder, and nested holders only count.
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
        except GitCommandError as exc:
            current = self.ref_sha(ref) or ZERO_SHA
            if current != expected:
                return False
            # The compare would have succeeded, so the ref did not move and
            # something else refused the write. Reporting that as a lost race
            # sent callers off to resolve a conflict that never happened --
            # a pruned object, for one, reads exactly like staleness.
            raise RefUpdateFailed(
                f"{ref} still points at {expected[:10]}, so the update failed "
                f"for another reason: {exc}"
            ) from exc

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
        return out.split()[1:]

    def entry_at(self, treeish: str, path: str) -> TreeEntry | None:
        """The mode, sha and path of one entry, or None if it is not there."""
        try:
            out = self.repo.git.ls_tree(
                treeish, "-z", "--", path,
                stdout_as_string=False, strip_newline_in_stdout=False,
            )
        except GitCommandError:
            return None
        for record in out.split(b"\0"):
            if not record:
                continue
            meta, _, name = record.partition(b"\t")
            parts = meta.split()
            if len(parts) >= 3 and _dec(name) == path:
                return TreeEntry(mode=_dec(parts[0]), sha=_dec(parts[2]), path=path)
        return None

    def blob_at(self, treeish: str, path: str) -> str | None:
        entry = self.entry_at(treeish, path)
        return entry.sha if entry else None

    def mode_at(self, treeish: str, path: str) -> str | None:
        entry = self.entry_at(treeish, path)
        return entry.mode if entry else None

    def show_bytes(self, commit: str, path: str) -> bytes | None:
        try:
            return self.repo.git.show(
                f"{commit}:{path}", stdout_as_string=False, strip_newline_in_stdout=False
            )
        except GitCommandError:
            return None

    def show(self, commit: str, path: str) -> str | None:
        raw = self.show_bytes(commit, path)
        return None if raw is None else _dec(raw)

    def cat_blob_bytes(self, sha: str) -> bytes:
        return self.repo.git.cat_file(
            "-p", sha, stdout_as_string=False, strip_newline_in_stdout=False
        )

    def cat_blob(self, sha: str) -> str:
        return _dec(self.cat_blob_bytes(sha))

    def hash_blob(self, content: str | bytes) -> str:
        raw = content if isinstance(content, bytes) else _enc(content)
        with self._tempfile(raw) as path:
            return self.repo.git.hash_object("-w", str(path))

    def ls_files(self, commit: str) -> list[str]:
        out = self.repo.git.ls_tree(
            "-r", "--name-only", "-z", commit,
            stdout_as_string=False, strip_newline_in_stdout=False,
        )
        return [_dec(name) for name in out.split(b"\0") if name]

    def diff_names(self, a: str, b: str) -> list[tuple[str, str]]:
        """(status, path) for every path that differs between two commits."""
        out = self.repo.git.diff_tree(
            "--name-status", "-r", "-z", a, b,
            stdout_as_string=False, strip_newline_in_stdout=False,
        )
        fields = [f for f in out.split(b"\0") if f]
        rows: list[tuple[str, str]] = []
        i = 0
        while i < len(fields):
            status = _dec(fields[i])[:1]
            i += 1
            # A rename or copy is followed by two paths, source then
            # destination; every other status by one. The destination is the
            # path that exists in ``b``, which is the one a caller wants.
            wanted = 2 if status in ("R", "C") else 1
            if i + wanted > len(fields):
                break
            rows.append((status, _dec(fields[i + wanted - 1])))
            i += wanted
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

    @contextlib.contextmanager
    def _tempfile(self, content: bytes) -> Iterator[Path]:
        """A private file holding ``content``, for commands that read a path."""
        path = self.common_dir / f"memstore.tmp.{os.getpid()}.{threading.get_ident()}.{id(content)}"
        path.write_bytes(content)
        try:
            yield path
        finally:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()

    # ---------------------------------------------------------------- notes

    def note_get(self, namespace: str, commit: str) -> str | None:
        try:
            return _dec(
                self.repo.git.notes(
                    "--ref", namespace, "show", commit,
                    stdout_as_string=False, strip_newline_in_stdout=False,
                )
            )
        except GitCommandError:
            return None

    def note_set(self, namespace: str, commit: str, text: str, *, overwrite: bool = True) -> None:
        """Attach ``text`` to ``commit``.

        With ``overwrite=False`` an existing note whose content differs raises
        ``NoteConflict`` instead of being replaced. The store uses that for
        every note it writes, so a commit-id collision can never silently
        destroy another agent's record.
        """
        with self.lock():
            if not overwrite:
                current = self.note_get(namespace, commit)
                if current is not None and current != text:
                    raise NoteConflict(
                        f"{namespace} already holds a different note on {commit[:10]}"
                    )
            with self._tempfile(_enc(text)) as path:
                self.repo.git.notes(
                    "--ref", namespace, "add", "-f", "--allow-empty",
                    "--no-stripspace", "-F", str(path), commit,
                )

    def note_prune(self, namespace: str) -> None:
        """Drop notes whose object is gone. Safe to call at any time."""
        with self.lock(), contextlib.suppress(GitCommandError):
            self.repo.git.notes("--ref", namespace, "prune")

    def prune_unreachable(self) -> None:
        """Delete objects no ref reaches.

        Notes must be pruned *after* this: ``git notes prune`` drops a note
        whose object is gone, and an unpublished commit is unreferenced but
        still present until it is pruned.
        """
        with self.lock(), contextlib.suppress(GitCommandError):
            self.repo.git.prune("--expire=now")

    # ---------------------------------------------------------------- worktree

    def reset_hard_to_head(self) -> None:
        self.repo.git.reset("--hard")
        self.repo.git.clean("-fd")

    def dirty_paths(self) -> list[str]:
        """Paths that differ from HEAD, without touching the index.

        ``status --porcelain`` reports tracked modifications and untracked
        files alike and does not stage anything, so asking whether a branch
        is dirty cannot itself change what a checkpoint would capture.
        """
        out = self.repo.git.status(
            "--porcelain", "-z", "--untracked-files=all",
            stdout_as_string=False, strip_newline_in_stdout=False,
        )
        entries = out.split(b"\0")
        paths: list[str] = []
        i = 0
        while i < len(entries):
            entry = entries[i]
            i += 1
            if len(entry) < 4:
                continue
            xy, path = entry[:2], entry[3:]
            if b"R" in xy or b"C" in xy:
                # NUL format puts the destination in this record and the
                # source in the next one; the source is not a dirty path.
                i += 1
            paths.append(_dec(path))
        return paths

    def untracked_paths(self) -> list[str]:
        """Paths git does not track yet, without staging any of them."""
        out = self.repo.git.status(
            "--porcelain", "-z", "--untracked-files=all",
            stdout_as_string=False, strip_newline_in_stdout=False,
        )
        entries = out.split(b"\0")
        paths: list[str] = []
        i = 0
        while i < len(entries):
            entry = entries[i]
            i += 1
            if len(entry) < 4:
                continue
            xy, path = entry[:2], entry[3:]
            if b"R" in xy or b"C" in xy:
                i += 1
            if xy == b"??":
                paths.append(_dec(path))
        return paths

    def _diff_untracked(self, rel: str, *, numstat: bool) -> bytes:
        """A patch for a file git has never seen, as an addition from nothing.

        ``git diff`` against a commit cannot show an untracked file, and the
        alternative -- ``add --intent-to-add`` -- writes to the index, which
        would make asking what changed change what a checkpoint captures.
        ``--no-index`` against /dev/null shows it while touching nothing.
        """
        args = ["git", "diff", "--no-color"]
        if numstat:
            args.append("--numstat")
        args += ["--no-index", "--", os.devnull, rel]
        _status, out, _err = self.repo.git.execute(
            args, with_extended_output=True, with_exceptions=False, stdout_as_string=False,
        )
        return out if isinstance(out, bytes) else _enc(out)

    def diff_pending(self, against: str | None = None, *, numstat: bool = False) -> str:
        """Uncommitted work as patch text, untracked files included.

        This is what a monitor reads. ``TypedDiff`` answers which paths
        changed between two published states; a monitor has to judge work that
        has not been published yet, and it has to see the code.
        """
        args = ["diff", "--no-color"]
        if numstat:
            args.append("--numstat")
        args += [against or "HEAD", "--"]
        chunks = [
            self.repo.git.execute(
                ["git", *args],
                with_extended_output=False, stdout_as_string=False,
            )
        ]
        chunks += [self._diff_untracked(rel, numstat=numstat) for rel in self.untracked_paths()]
        return _dec(b"".join(c for c in chunks if c))

    def worktree_add(self, path: Path, ref: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.repo.git.worktree("add", str(path), ref)
        except GitCommandError:
            # A checkout whose directory was deleted leaves an administrative
            # entry behind that blocks re-adding it. Pruning is the documented
            # repair and is safe: it only drops entries whose path is gone.
            self.repo.git.worktree("prune")
            self.repo.git.worktree("add", str(path), ref)

    def worktree_remove(self, path: Path) -> None:
        with contextlib.suppress(GitCommandError):
            self.repo.git.worktree("remove", "--force", str(path))
        with contextlib.suppress(GitCommandError):
            self.repo.git.worktree("prune")

    def write_excludes(self, patterns: list[str]) -> None:
        """Set this repository's local excludes. Empty means exclude nothing."""
        exclude = self.common_dir / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        exclude.write_text("\n".join(patterns) + ("\n" if patterns else ""))

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
                _, _, path = line.partition("\t")
                if path and path not in conflicted:
                    conflicted.append(path)
            else:
                messages.append(line)
        if status not in (0, 1):
            raise RuntimeError(f"merge-tree failed ({status}): {err or out}")
        return MergeTreeResult(tree=tree, conflicted_paths=tuple(conflicted), messages="\n".join(messages))

    def tree_with_blob(self, tree: str, path: str, blob: str, mode: str = MODE_FILE) -> str:
        """Return a new tree equal to ``tree`` with ``path`` replaced.

        ``mode`` is preserved by the caller rather than assumed: the first
        version hardcoded ``100644`` and silently turned executables into
        plain files. Done through a throwaway index so the real index and
        working tree are never touched.
        """
        index_path = self.common_dir / f"memstore.index.{os.getpid()}.{threading.get_ident()}"
        env = {"GIT_INDEX_FILE": str(index_path)}
        try:
            with self.repo.git.custom_environment(**env):
                self.repo.git.read_tree(tree)
                self.repo.git.update_index("--add", "--cacheinfo", f"{mode},{blob},{path}")
                return self.repo.git.write_tree()
        finally:
            with contextlib.suppress(FileNotFoundError):
                index_path.unlink()
