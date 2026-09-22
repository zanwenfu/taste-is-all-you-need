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
  task's ignore rules remain available to its Git tools, but do not exclude
  files from memory checkpoints.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import stat
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import IO, Concatenate, ParamSpec, TypeVar, cast

from git import Repo
from git.exc import BadName, BadObject, GitCommandError
from git.objects import Blob, Commit, Tree
from git.refs import SymbolicReference

from taste.memstore.objects import UnsafeCapture

# One held description per lock path, process-wide; see ``GitBackend.lock``.
# Keyed by thread as well as path: the description is what ``flock`` acts on,
# so a second thread that found it already open would skip the syscall and
# walk straight into the critical section.
_HELD: dict[tuple[Path, int], tuple[IO[str], int]] = {}
_REGISTRY_LOCK = threading.RLock()
# Held across the ``flock`` so two threads of one process serialise here
# rather than both reaching a syscall that cannot tell them apart.
_PROCESS_LOCKS: dict[Path, threading.RLock] = {}


def _process_lock(path: Path) -> threading.RLock:
    with _REGISTRY_LOCK:
        lock = _PROCESS_LOCKS.get(path)
        if lock is None:
            lock = threading.RLock()
            _PROCESS_LOCKS[path] = lock
        return lock

ZERO_SHA = "0" * 40
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

MODE_FILE = "100644"
MODE_EXEC = "100755"
MODE_SYMLINK = "120000"
MODE_GITLINK = "160000"
BLOB_MODES = frozenset({MODE_FILE, MODE_EXEC, MODE_SYMLINK})
#: Modes whose content is not a plain file, so a content merge is meaningless.
UNMERGEABLE_MODES = frozenset({MODE_SYMLINK, MODE_GITLINK})

IDENTITY_NAME = "taste"
IDENTITY_EMAIL = "taste@localhost"

_CACHE_MISS = object()
_K = TypeVar("_K")
_V = TypeVar("_V")
_P = ParamSpec("_P")
_R = TypeVar("_R")

# Object ids and cache cardinalities are intentionally bounded.  Coordinator
# processes can live for days and encounter an unbounded number of states.
_NOTE_TREE_CACHE_MAX_REFS = 16
_NOTE_TREE_CACHE_MAX_TARGETS = 16_384
_NOTE_BLOB_CACHE_MAX_ENTRIES = 4_096
_NOTE_BLOB_CACHE_MAX_VALUE_BYTES = 256 * 1_024
_NOTE_BLOB_CACHE_MAX_TOTAL_BYTES = 8 * 1_024 * 1_024
_TREE_CACHE_MAX_ENTRIES = 2_048
_PARENTS_CACHE_MAX_ENTRIES = 2_048
_PARENTS_CACHE_MAX_PARENTS = 64
_ENTRY_CACHE_MAX_ENTRIES = 4_096
_CACHE_MAX_PATH_CHARACTERS = 4_096
_SHOW_CACHE_MAX_ENTRIES = 512
_SHOW_CACHE_MAX_VALUE_BYTES = 256 * 1_024
_SHOW_CACHE_MAX_TOTAL_BYTES = 8 * 1_024 * 1_024
_LS_FILES_CACHE_MAX_ENTRIES = 128
_LS_FILES_CACHE_MAX_PATHS = 8_192
_LS_FILES_CACHE_MAX_CHARACTERS = 512 * 1_024
_FIRST_PARENT_CACHE_MAX_ENTRIES = 32
_FIRST_PARENT_CACHE_MAX_COMMITS = 4_096


def _lru_get(cache: OrderedDict[_K, _V], key: _K) -> _V | object:
    try:
        value = cache.pop(key)
    except KeyError:
        return _CACHE_MISS
    cache[key] = value
    return value


def _lru_put(cache: OrderedDict[_K, _V], key: _K, value: _V, *, maximum: int) -> None:
    cache.pop(key, None)
    cache[key] = value
    while len(cache) > maximum:
        cache.popitem(last=False)


def _serialized_object_read(
    method: Callable[Concatenate[GitBackend, _P], _R],
) -> Callable[Concatenate[GitBackend, _P], _R]:
    @wraps(method)
    def wrapped(self: GitBackend, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        with self._object_lock:
            return method(self, *args, **kwargs)

    return wrapped


def _is_full_object_id(value: str) -> bool:
    """Whether ``value`` names one immutable Git object without resolution.

    Refs and abbreviated object names are intentionally excluded: both can
    resolve to a different object later in the same process.  Git object ids
    are currently SHA-1 or SHA-256, hence the two accepted lengths.
    """
    return len(value) in {40, 64} and all(character in "0123456789abcdef" for character in value)


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
        # A full object id must mean its hashed bytes.  Git normally lets a
        # mutable refs/replace entry reinterpret even an explicit SHA, which
        # would invalidate every exact-id cache below.
        self.repo.git.update_environment(GIT_NO_REPLACE_OBJECTS="1")
        # GitPython multiplexes object reads through one persistent
        # ``cat-file --batch`` stream per Repo.  Concurrent reads otherwise
        # consume one another's headers and corrupt the protocol.
        self._object_lock = threading.RLock()
        # Git notes are ordinary content-addressed trees behind mutable refs.
        # Cache only the tree decoded for one exact ref commit; a note write,
        # deletion, or raw ref rollback necessarily selects a different key.
        self._note_tree_cache: OrderedDict[str, tuple[str, dict[str, Blob]]] = OrderedDict()
        self._note_blob_cache: OrderedDict[str, bytes] = OrderedDict()
        self._note_blob_cache_size = 0
        # These plumbing results are immutable only when addressed by a full
        # object id.  Never populate them for a ref or abbreviated id.  Lists
        # are stored as tuples so callers cannot mutate a cached result.
        self._tree_cache: OrderedDict[str, str] = OrderedDict()
        self._parents_cache: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        self._entry_cache: OrderedDict[tuple[str, str], TreeEntry | None] = OrderedDict()
        self._show_bytes_cache: OrderedDict[tuple[str, str], bytes] = OrderedDict()
        self._show_bytes_cache_size = 0
        self._ls_files_cache: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        self._first_parent_cache: OrderedDict[tuple[str, int | None], tuple[str, ...]] = (
            OrderedDict()
        )
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
        with self._object_lock:
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

        Exclusive at both levels, and re-entrant on one thread at either.

        Across processes ``flock`` does the work, which is the architecture:
        one OS process per worker, each with its own worktree. Within a
        process it does nothing on its own -- ``flock`` acts on an open file
        description, so a second thread that found the description already
        open skipped the syscall entirely and counted itself in beside the
        holder. Measured: two threads checkpointing different branches of one
        ``Store`` produced a head whose meta note was missing, which the next
        write refused as a ``ForeignHead``. "Nothing is lost" is this layer's
        first rule, so the gap mattered more than its rarity.

        So a process-local lock is taken first and the description is kept per
        thread. Nesting on one thread still never blocks: the ``RLock``
        re-enters and the depth counter rises, exactly as before.
        """
        lock_path = self.common_dir / "memstore.lock"
        key = (lock_path, threading.get_ident())
        process_lock = _process_lock(lock_path)
        process_lock.acquire()
        try:
            with _REGISTRY_LOCK:
                fh, depth = _HELD.get(key, (None, 0))
                if fh is None:
                    fh = open(lock_path, "a+")  # noqa: SIM115  held past this block by design
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                _HELD[key] = (fh, depth + 1)
            try:
                yield
            finally:
                with _REGISTRY_LOCK:
                    fh, depth = _HELD[key]
                    if depth == 1:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                        fh.close()
                        del _HELD[key]
                    else:
                        _HELD[key] = (fh, depth - 1)
        finally:
            process_lock.release()

    # ---------------------------------------------------------------- refs

    @_serialized_object_read
    def ref_sha(self, ref: str) -> str | None:
        try:
            # GitPython resolves the ref from disk on every call, then checks
            # its object through the serialized batch stream.  This keeps
            # mutable refs live without spawning ``rev-parse`` thousands of
            # times in one coordinator cycle.
            return self.repo.commit(ref).hexsha
        except (BadName, BadObject, TypeError, ValueError):
            return None

    @_serialized_object_read
    def worktree_branch_ref(self) -> str | None:
        """The ref this working tree is checked out on, or ``None`` if detached.

        ``git worktree add`` binds a tree to one branch, and every checkpoint
        assumes that binding still holds: it stages *this* tree and moves
        *that* ref. A detached HEAD breaks the pair without touching the ref,
        so no amount of looking at the branch can see it.
        """
        try:
            return self.repo.git.symbolic_ref("HEAD").strip() or None
        except GitCommandError:
            return None  # Detached HEAD exits non-zero; that is the answer.

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
        self.check_capture_safe()
        self.repo.git.add("--force", "--all", "--", ".")
        # A worker can create a nested repo after the filesystem scan. Never
        # publish the resulting gitlink as though it captured those files.
        self._check_index_capture_safe()

    def _check_index_capture_safe(self) -> None:
        entries = self.repo.git.ls_files(
            "--stage", "-z", stdout_as_string=False, strip_newline_in_stdout=False,
        )
        for entry in entries.split(b"\0"):
            if entry.startswith(b"160000 "):
                relative = _dec(entry.split(b"\t", 1)[1])
                path = self.path / relative
                # Removing a gitlink entirely is representable. An existing
                # checkout behind that pointer is not a file snapshot.
                if path.exists() or path.is_symlink():
                    raise UnsafeCapture(
                        f"nested repository in Git index at {relative}; "
                        "the worktree must be retained"
                    )

    def check_capture_safe(self) -> None:
        """Refuse filesystem shapes that a Git tree would omit or misrepresent.

        A nested repo becomes a gitlink, which does not preserve its working
        files or object database. FIFOs/devices/sockets likewise cannot be
        reconstructed from this store. Keep their original worktree available
        for explicit recovery instead of claiming a complete snapshot.
        """
        self._check_index_capture_safe()

        def unreadable(error: OSError) -> None:
            raise error

        for directory, dirs, files in os.walk(self.path, followlinks=False, onerror=unreadable):
            parent = Path(directory)
            if parent == self.path:
                dirs[:] = [name for name in dirs if name != ".git"]
                files = [name for name in files if name != ".git"]
            elif ".git" in dirs or ".git" in files:
                relative = parent.relative_to(self.path)
                raise UnsafeCapture(
                    f"nested repository at {relative!s} cannot be preserved by a Git file "
                    "snapshot; the worktree must be retained"
                )
            for name in (*dirs, *files):
                path = parent / name
                mode = path.lstat().st_mode
                if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
                    raise UnsafeCapture(
                        f"unsupported filesystem entry at {path.relative_to(self.path)!s}; "
                        "the worktree must be retained"
                    )

    def stage_entry(self, path: str, mode: str, content: bytes) -> None:
        """Pin an adopted file's exact mode, even with core.filemode=false."""
        if mode not in BLOB_MODES:
            raise ValueError(f"cannot stage file mode {mode}")
        blob = self.hash_blob(content)
        self.repo.git.update_index("--add", "--cacheinfo", f"{mode},{blob},{path}")

    def write_tree(self) -> str:
        return self.repo.git.write_tree()

    def commit_tree(self, tree: str, parents: list[str], message: str) -> str:
        args: list[str] = [tree]
        for p in parents:
            args += ["-p", p]
        args += ["-m", message]
        return self.repo.git.commit_tree(*args)

    def _assert_raw_ancestry(self) -> None:
        """Reject local mechanisms that rewrite or truncate commit ancestry."""
        for path in (self.common_dir / "shallow", self.common_dir / "info" / "grafts"):
            try:
                populated = path.stat().st_size > 0
            except FileNotFoundError:
                populated = False
            if populated:
                raise ValueError(f"memstore requires raw, complete ancestry; found {path}")

    def _tree_object(self, treeish: str) -> Tree:
        obj = self.repo.rev_parse(treeish)
        if isinstance(obj, Commit):
            return obj.tree
        if isinstance(obj, Tree):
            # A root tree resolved by id has no path.  Tree.join() expects the
            # root path to be the empty string when walking a nested path.
            return Tree(self.repo, obj.binsha, path="")
        raise ValueError(f"object {treeish} is neither a commit nor a tree")

    @_serialized_object_read
    def tree_of(self, commit: str) -> str:
        if _is_full_object_id(commit):
            cached = _lru_get(self._tree_cache, commit)
            if cached is not _CACHE_MISS:
                return cast(str, cached)
        tree = (
            self.repo.commit(commit).tree.hexsha
            if _is_full_object_id(commit)
            else self.repo.git.rev_parse(f"{commit}^{{tree}}")
        )
        if _is_full_object_id(commit):
            _lru_put(self._tree_cache, commit, tree, maximum=_TREE_CACHE_MAX_ENTRIES)
        return tree

    @_serialized_object_read
    def parents_of(self, commit: str) -> list[str]:
        if _is_full_object_id(commit):
            self._assert_raw_ancestry()
            cached = _lru_get(self._parents_cache, commit)
            if cached is not _CACHE_MISS:
                return list(cast(tuple[str, ...], cached))
            parents = tuple(parent.hexsha for parent in self.repo.commit(commit).parents)
        else:
            out = self.repo.git.rev_list("--parents", "-n", "1", commit)
            parents = tuple(out.split()[1:])
        if _is_full_object_id(commit) and len(parents) <= _PARENTS_CACHE_MAX_PARENTS:
            _lru_put(
                self._parents_cache,
                commit,
                parents,
                maximum=_PARENTS_CACHE_MAX_ENTRIES,
            )
        return list(parents)

    @_serialized_object_read
    def blob_size(self, blob_id: str) -> int:
        """Read object size without loading its contents into memory."""
        if not _is_full_object_id(blob_id):
            raise ValueError("blob size requires an exact object id")
        return int(self.repo.git.cat_file("-s", blob_id))

    @_serialized_object_read
    def entry_at(self, treeish: str, path: str) -> TreeEntry | None:
        """The mode, sha and path of one entry, or None if it is not there."""
        cache_key = (treeish, path)
        cacheable = _is_full_object_id(treeish) and len(path) <= _CACHE_MAX_PATH_CHARACTERS
        if cacheable:
            cached = _lru_get(self._entry_cache, cache_key)
            if cached is not _CACHE_MISS:
                return cast(TreeEntry | None, cached)
        try:
            if cacheable:
                entry = self._tree_object(treeish) / path
                result = TreeEntry(mode=f"{entry.mode:06o}", sha=entry.hexsha, path=path)
                _lru_put(
                    self._entry_cache,
                    cache_key,
                    result,
                    maximum=_ENTRY_CACHE_MAX_ENTRIES,
                )
                return result
            out = self.repo.git.ls_tree(
                treeish,
                "-z",
                "--",
                path,
                stdout_as_string=False,
                strip_newline_in_stdout=False,
            )
        except KeyError:
            if cacheable:
                _lru_put(
                    self._entry_cache,
                    cache_key,
                    None,
                    maximum=_ENTRY_CACHE_MAX_ENTRIES,
                )
            return None
        except (BadName, BadObject, GitCommandError, TypeError, ValueError):
            return None
        for record in out.split(b"\0"):
            if not record:
                continue
            meta, _, name = record.partition(b"\t")
            parts = meta.split()
            if len(parts) >= 3 and _dec(name) == path:
                result = TreeEntry(mode=_dec(parts[0]), sha=_dec(parts[2]), path=path)
                if cacheable:
                    _lru_put(
                        self._entry_cache,
                        cache_key,
                        result,
                        maximum=_ENTRY_CACHE_MAX_ENTRIES,
                    )
                return result
        if cacheable:
            _lru_put(
                self._entry_cache,
                cache_key,
                None,
                maximum=_ENTRY_CACHE_MAX_ENTRIES,
            )
        return None

    def blob_at(self, treeish: str, path: str) -> str | None:
        entry = self.entry_at(treeish, path)
        return entry.sha if entry else None

    def mode_at(self, treeish: str, path: str) -> str | None:
        entry = self.entry_at(treeish, path)
        return entry.mode if entry else None

    @_serialized_object_read
    def show_bytes(self, commit: str, path: str) -> bytes | None:
        cache_key = (commit, path)
        cacheable = _is_full_object_id(commit) and len(path) <= _CACHE_MAX_PATH_CHARACTERS
        if cacheable:
            cached = _lru_get(self._show_bytes_cache, cache_key)
            if cached is not _CACHE_MISS:
                return cast(bytes, cached)
        try:
            if cacheable:
                blob = self._tree_object(commit) / path
                if not isinstance(blob, Blob):
                    return None
                raw = blob.data_stream.read()
            else:
                raw = self.repo.git.show(
                    f"{commit}:{path}", stdout_as_string=False, strip_newline_in_stdout=False
                )
        except (BadName, BadObject, GitCommandError, KeyError, TypeError, ValueError):
            return None
        if cacheable and len(raw) <= _SHOW_CACHE_MAX_VALUE_BYTES:
            previous = self._show_bytes_cache.pop(cache_key, None)
            if previous is not None:
                self._show_bytes_cache_size -= len(previous)
            self._show_bytes_cache[cache_key] = raw
            self._show_bytes_cache_size += len(raw)
            while (
                len(self._show_bytes_cache) > _SHOW_CACHE_MAX_ENTRIES
                or self._show_bytes_cache_size > _SHOW_CACHE_MAX_TOTAL_BYTES
            ):
                _old_key, old_value = self._show_bytes_cache.popitem(last=False)
                self._show_bytes_cache_size -= len(old_value)
        return raw

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

    @_serialized_object_read
    def ls_files(self, commit: str) -> list[str]:
        if _is_full_object_id(commit):
            cached = _lru_get(self._ls_files_cache, commit)
            if cached is not _CACHE_MISS:
                return list(cast(tuple[str, ...], cached))
        if _is_full_object_id(commit):
            files = tuple(
                sorted(
                    entry.path
                    for entry in self._tree_object(commit).traverse()
                    if entry.type != "tree"
                )
            )
        else:
            out = self.repo.git.ls_tree(
                "-r",
                "--name-only",
                "-z",
                commit,
                stdout_as_string=False,
                strip_newline_in_stdout=False,
            )
            files = tuple(_dec(name) for name in out.split(b"\0") if name)
        if (
            _is_full_object_id(commit)
            and len(files) <= _LS_FILES_CACHE_MAX_PATHS
            and sum(map(len, files)) <= _LS_FILES_CACHE_MAX_CHARACTERS
        ):
            _lru_put(
                self._ls_files_cache,
                commit,
                files,
                maximum=_LS_FILES_CACHE_MAX_ENTRIES,
            )
        return list(files)

    def diff_names(self, a: str, b: str) -> list[tuple[str, str]]:
        """(status, path) for every path that differs between two commits."""
        out = self.repo.git.diff_tree(
            "--name-status",
            "-r",
            "-z",
            a,
            b,
            stdout_as_string=False,
            strip_newline_in_stdout=False,
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

    @_serialized_object_read
    def rev_list_first_parent(self, commit: str, limit: int | None = None) -> list[str]:
        cache_key = (commit, limit)
        if _is_full_object_id(commit):
            self._assert_raw_ancestry()
            cached = _lru_get(self._first_parent_cache, cache_key)
            if cached is not _CACHE_MISS:
                return list(cast(tuple[str, ...], cached))
            history: list[str] = []
            current = self.repo.commit(commit)
            while limit is None or len(history) < limit:
                history.append(current.hexsha)
                if not current.parents:
                    break
                current = current.parents[0]
            commits = tuple(history)
        else:
            args = ["--first-parent"]
            if limit is not None:
                args += ["-n", str(limit)]
            out = self.repo.git.rev_list(*args, commit)
            commits = tuple(line for line in out.splitlines() if line)
        if _is_full_object_id(commit) and len(commits) <= _FIRST_PARENT_CACHE_MAX_COMMITS:
            _lru_put(
                self._first_parent_cache,
                cache_key,
                commits,
                maximum=_FIRST_PARENT_CACHE_MAX_ENTRIES,
            )
        return list(commits)

    @_serialized_object_read
    def _clear_object_caches(self) -> None:
        """Forget object reads before an operation that may delete objects."""
        self._tree_cache.clear()
        self._parents_cache.clear()
        self._entry_cache.clear()
        self._show_bytes_cache.clear()
        self._show_bytes_cache_size = 0
        self._ls_files_cache.clear()
        self._first_parent_cache.clear()
        self._note_tree_cache.clear()
        self._note_blob_cache.clear()
        self._note_blob_cache_size = 0

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

    @_serialized_object_read
    def note_get(self, namespace: str, commit: str) -> str | None:
        """Read one note from the exact current notes-ref tree.

        ``git notes show`` starts a subprocess for every property read.  A
        State audit reads the same immutable metadata hundreds of times, so
        that implementation dominated coordinator runtime.  GitPython can
        traverse the same content-addressed tree in-process.  The cache is
        scoped to the current notes-ref commit, not merely ``(namespace,
        target)``: forward writes, deletion, and raw rollback are therefore
        observed on the very next call.
        """
        try:
            # Read the mutable ref itself every time, but resolve its commit
            # and tree only when that exact value changes.
            ref_sha = SymbolicReference.dereference_recursive(self.repo, namespace)
        except ValueError as exc:
            self._note_tree_cache.pop(namespace, None)
            if (self.common_dir / namespace).exists():
                raise ValueError(f"notes ref {namespace!r} is malformed") from exc
            return None
        cached_value = _lru_get(self._note_tree_cache, namespace)
        cached = (
            cast(tuple[str, dict[str, Blob]], cached_value)
            if cached_value is not _CACHE_MISS
            else None
        )
        if cached is None or cached[0] != ref_sha:
            self._note_tree_cache.pop(namespace, None)
            try:
                note_commit = self.repo.commit(ref_sha)
            except (BadName, BadObject, TypeError, ValueError) as exc:
                raise ValueError(f"notes ref {namespace!r} points to non-commit {ref_sha}") from exc
            notes: dict[str, Blob] = {}
            for entry in note_commit.tree.traverse():
                if entry.type == "tree":
                    prefix = entry.path.replace("/", "")
                    if (
                        not prefix
                        or len(prefix) >= len(ref_sha)
                        or any(character not in "0123456789abcdef" for character in prefix)
                        or len(cast(Tree, entry)) == 0
                    ):
                        raise ValueError(
                            f"notes ref {namespace!r} contains invalid tree prefix {entry.path!r}"
                        )
                    continue
                if entry.type != "blob":
                    raise ValueError(
                        f"notes ref {namespace!r} contains non-blob entry {entry.path!r}"
                    )
                target = entry.path.replace("/", "")
                if len(target) != len(ref_sha) or any(
                    character not in "0123456789abcdef" for character in target
                ):
                    raise ValueError(
                        f"notes ref {namespace!r} contains invalid target path {entry.path!r}"
                    )
                blob = cast(Blob, entry)
                previous = notes.setdefault(target, blob)
                if previous != entry:
                    raise ValueError(f"notes ref {namespace!r} contains duplicate target paths")
            cached = (ref_sha, notes)
            if len(notes) <= _NOTE_TREE_CACHE_MAX_TARGETS:
                _lru_put(
                    self._note_tree_cache,
                    namespace,
                    cached,
                    maximum=_NOTE_TREE_CACHE_MAX_REFS,
                )
        blob = cached[1].get(commit)
        if blob is None:
            return None
        cached_raw = _lru_get(self._note_blob_cache, blob.hexsha)
        if cached_raw is not _CACHE_MISS:
            return _dec(cast(bytes, cached_raw))
        raw = blob.data_stream.read()
        if len(raw) <= _NOTE_BLOB_CACHE_MAX_VALUE_BYTES:
            previous = self._note_blob_cache.pop(blob.hexsha, None)
            if previous is not None:
                self._note_blob_cache_size -= len(previous)
            self._note_blob_cache[blob.hexsha] = raw
            self._note_blob_cache_size += len(raw)
            while (
                len(self._note_blob_cache) > _NOTE_BLOB_CACHE_MAX_ENTRIES
                or self._note_blob_cache_size > _NOTE_BLOB_CACHE_MAX_TOTAL_BYTES
            ):
                _old_key, old_value = self._note_blob_cache.popitem(last=False)
                self._note_blob_cache_size -= len(old_value)
        return _dec(raw)

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
                    "--ref",
                    namespace,
                    "add",
                    "-f",
                    "--allow-empty",
                    "--no-stripspace",
                    "-F",
                    str(path),
                    commit,
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
        with self.lock():
            with contextlib.suppress(GitCommandError):
                self.repo.git.prune("--expire=now")
            self._clear_object_caches()

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
            "--porcelain",
            "-z",
            "--untracked-files=all",
            "--ignored=traditional",
            stdout_as_string=False,
            strip_newline_in_stdout=False,
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
            "--porcelain",
            "-z",
            "--untracked-files=all",
            "--ignored=traditional",
            stdout_as_string=False,
            strip_newline_in_stdout=False,
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
            if xy in {b"??", b"!!"}:
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
            args,
            with_extended_output=True,
            with_exceptions=False,
            stdout_as_string=False,
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
                with_extended_output=False,
                stdout_as_string=False,
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
        if path.exists() or path.is_symlink():
            self.repo.git.worktree("remove", "--force", str(path))
            if path.exists() or path.is_symlink():
                raise OSError(f"Git did not remove worktree {path}")
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
        return MergeTreeResult(
            tree=tree, conflicted_paths=tuple(conflicted), messages="\n".join(messages)
        )

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
