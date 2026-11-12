"""Only the records a replay needs, restored from a trial archive.

The study's archives (``archive.sh`` and ``run-stage.sh`` of the Oct 20, Nov 3 and Nov 7 kits) are each one tar
stream of paths relative to ``/``, compressed with zstd and cut into 4 GiB parts (``<set>.tar.zst.part-000``, ...),
with the set's file list beside them (``<set>.files.tsv``: a path, a tab, its size; sorted, which is the tar's own
order, since the tar was made from that list) and ``SHA256SUMS``. A trial's directory holds about 2,300 files; a
replay of it (``taste.benchmarks.replay_export``) needs four, which are the only ones written here:

- ``journal``: the worker's model journal, ``<token>/agent-state/workspace/.git/taste.azure.<run>/worker/calls.sqlite3``;
- ``ledger``: the terminal ledger, ``<token>/controller/terminal/terminal.sqlite3``;
- ``script``: the replay script a branch was given, ``<token>/branch-script.json`` (a branch's only);
- ``trajectory``: the settled record, ``<token>/controller/trajectory.json`` (the task the agent was given, its steps).

Unpacking whole trial archives filled a disk's file slots on Oct 20, so nothing else is written: each wanted member
goes to ``<out>/<token>/<the same path>`` (the trials-root layout every tool reads), a regular file only, under a size
cap, never through a link, written whole or not at all. A file already there with the same bytes is left; one with
other bytes is a conflict, reported and never overwritten. The stream is read from its start (the parts are cuts of
one compressed stream); with the set's file list the wanted members are known in advance and reading stops after the
last of them.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import shutil
import subprocess
import tarfile
import threading
from collections import defaultdict
from pathlib import Path

KINDS = ("journal", "ledger", "script", "trajectory")
# A trial a replay can be exported from: these three; a branch also needs its script.
REQUIRED = ("journal", "ledger", "trajectory")
_TOKEN = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
_WANTED = re.compile(
    r"^(?P<prefix>(?:[^/]+/)*?)(?P<token>" + _TOKEN + r")/(?P<rest>"
    r"(?P<journal>agent-state/workspace/\.git/taste\.azure\.[A-Za-z0-9._-]+/worker/calls\.sqlite3)"
    r"|(?P<ledger>controller/terminal/terminal\.sqlite3)"
    r"|(?P<script>branch-script\.json)"
    r"|(?P<trajectory>controller/trajectory\.json))$")
# No record a replay reads is near this; a member over it is refused, not written.
MAX_FILE_BYTES = 256 * 1024 * 1024
CHUNK = 1024 * 1024
_PART = re.compile(r"\.tar\.zst\.part-(\d+)$")


def tokens_from(path):
    """The trial tokens in a file, one per line (blank lines and lines starting with # are skipped)."""
    found = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            if not re.fullmatch(_TOKEN, line):
                raise ValueError(f"not a trial token: {line[:80]!r}")
            found.append(line)
    return list(dict.fromkeys(found))


def wanted(name, tokens):
    """(token, kind, path under the token) when the archive member ``name`` is one of the four files of a wanted
    trial; else None."""
    name = name[2:] if name.startswith("./") else name
    found = _WANTED.match(name)
    if found is None or found["token"] not in tokens or ".." in found["rest"].split("/"):
        return None
    kind = next(kind for kind in KINDS if found[kind])
    return found["token"], kind, found["rest"]


def read_index(path):
    """A set's file list (``<set>.files.tsv``): [(path, size)], in the archive's order."""
    found = []
    for line in Path(path).read_text().splitlines():
        if not line:
            continue
        name, _, size = line.rpartition("\t")
        found.append((name, int(size)))
    return found


def plan(index, tokens):
    """What a set's file list says of the wanted trials: their members in it, those missing, the last one (where
    reading can stop) and the bytes to write."""
    members, by_token = {}, defaultdict(dict)
    last = None
    for position, (name, size) in enumerate(index):
        hit = wanted(name, tokens)
        if hit is None:
            continue
        token, kind, _ = hit
        members[name] = size
        by_token[token].setdefault(kind, []).append(name)
        last = (position, name)
    return {"members": members, "tokens": {token: {kind: names for kind, names in kinds.items()}
                                           for token, kinds in sorted(by_token.items())},
            "missing": {token: [kind for kind in REQUIRED if kind not in by_token.get(token, {})]
                        for token in sorted(tokens) if any(kind not in by_token.get(token, {}) for kind in REQUIRED)},
            "last": last, "bytes": sum(members.values())}


def parts(directory, name):
    """A set's parts in order (``<name>.tar.zst.part-000``, ...); refused when one is missing between."""
    found = []
    for path in Path(directory).glob(f"{name}.tar.zst.part-*"):
        number = _PART.search(path.name)
        if number and path.name == f"{name}.tar.zst.part-{number[1]}":
            found.append((int(number[1]), path))
    found.sort()
    if not found:
        raise ValueError(f"no parts of {name} in {directory}")
    if [number for number, _ in found] != list(range(len(found))):
        raise ValueError(f"the parts of {name} are not numbered 0 to {len(found) - 1} without a gap")
    return [path for _, path in found]


def sums(path):
    """``SHA256SUMS`` as {file name: digest}."""
    found = {}
    for line in Path(path).read_text().splitlines():
        digest, _, name = line.partition("  ")
        if digest and name:
            found[name.lstrip("*")] = digest.lower()
    return found


def check_parts(paths, digests):
    """Each part's SHA-256 against ``SHA256SUMS``; raises at the first part that is not listed or differs."""
    for path in paths:
        expected = digests.get(Path(path).name)
        if expected is None:
            raise ValueError(f"{Path(path).name} is not in SHA256SUMS")
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"{Path(path).name} differs from its SHA-256 in SHA256SUMS")
    return len(paths)


@contextlib.contextmanager
def decompressed(paths=None, stream=None, command=("zstd", "-q", "-dc")):
    """The uncompressed tar stream of a set: its parts fed in order (or ``stream``, a compressed stream such as
    standard input) through ``zstd -dc``. Leaving early stops the decompressor; nothing is left running."""
    if (paths is None) == (stream is None):
        raise ValueError("give the parts or a stream")
    if shutil.which(command[0]) is None:
        raise RuntimeError(f"{command[0]} is not installed")
    process = subprocess.Popen(list(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    failure = []

    def feed():
        try:
            if paths is not None:
                for path in paths:
                    with open(path, "rb") as handle:
                        shutil.copyfileobj(handle, process.stdin, CHUNK)
            else:
                shutil.copyfileobj(stream, process.stdin, CHUNK)
        except BrokenPipeError:
            pass  # reading stopped early
        except OSError as error:
            failure.append(error)
        finally:
            with contextlib.suppress(OSError):
                process.stdin.close()

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    try:
        yield process.stdout
    finally:
        with contextlib.suppress(OSError):
            process.stdout.close()
        if process.poll() is None:
            process.kill()
        process.wait()
        feeder.join(timeout=60)
    if failure:
        raise failure[0]


def _inside(root, path):
    root, path = os.path.realpath(root), os.path.realpath(path)
    return path == root or path.startswith(root + os.sep)


def _digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(member, source, out, token, rest):
    """One wanted member to ``<out>/<token>/<rest>``: "written", "present" (the same bytes were there) or
    "conflict" (other bytes are there; they are kept)."""
    target = Path(out) / token / rest
    # Checked before and after the folders are made: no link may lead out of the restore folder.
    if not _inside(out, target.parent):
        raise ValueError("the target is not inside the restore folder")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not _inside(out, target.parent) or target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("the target is not a plain file inside the restore folder")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.restore")
    digest, size = hashlib.sha256(), 0
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for chunk in iter(lambda: source.read(CHUNK), b""):
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    raise ValueError("larger than the cap on one record")
                digest.update(chunk)
                handle.write(chunk)
        if size != member.size:
            raise ValueError("the member ended before its size")
        if target.exists():
            same = _digest(target) == digest.hexdigest()
            temporary.unlink()
            return "present" if same else "conflict"
        os.replace(temporary, target)
        return "written"
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def extract(stream, tokens, out, *, members=None):
    """The four files of each wanted trial from an uncompressed tar ``stream``, into ``<out>/<token>/...``.

    ``members`` (from ``plan``): the wanted members by name, as the set's file list has them; reading stops once
    every one has been seen, and a member not on it is not taken. Returns the account: per trial the kinds found,
    what was written, already present or in conflict, the members refused (not a regular file, too large, short)
    and the trials still missing a required file."""
    tokens = set(tokens)
    out = Path(out)
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    left = None if members is None else set(members)
    found = defaultdict(lambda: defaultdict(list))
    account = {"written": 0, "present": 0, "conflicts": [], "refused": [], "read_members": 0, "stopped_early": False}
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            account["read_members"] += 1
            hit = wanted(member.name, tokens)
            if hit is None or (left is not None and member.name not in members):
                continue
            token, kind, rest = hit
            if not member.isreg():
                account["refused"].append({"member": member.name, "why": "not a regular file"})
            elif member.size > MAX_FILE_BYTES:
                account["refused"].append({"member": member.name, "why": "larger than the cap on one record"})
            else:
                try:
                    said = _write(member, archive.extractfile(member), out, token, rest)
                except ValueError as error:
                    account["refused"].append({"member": member.name, "why": str(error)})
                else:
                    if said == "conflict":
                        account["conflicts"].append(member.name)
                    else:
                        account[said] += 1
                        found[token][kind].append(rest)
            if left is not None:
                left.discard(member.name)
                if not left:
                    account["stopped_early"] = True
                    break
    account["trials"] = {token: {kind: sorted(names) for kind, names in kinds.items()}
                         for token, kinds in sorted(found.items())}
    account["missing"] = {token: [kind for kind in REQUIRED if kind not in found.get(token, {})]
                          for token in sorted(tokens) if any(kind not in found.get(token, {}) for kind in REQUIRED)}
    account["several_journals"] = sorted(token for token, kinds in found.items() if len(kinds.get("journal", ())) > 1)
    return account


def on_disk(out, tokens):
    """The trials already restored under ``out``: {token: [required kinds missing]} for those not whole."""
    missing = {}
    for token in tokens:
        root = Path(out) / token
        have = {"journal": any(root.glob("agent-state/workspace/.git/taste.azure.*/worker/calls.sqlite3")),
                "ledger": (root / "controller" / "terminal" / "terminal.sqlite3").is_file(),
                "trajectory": (root / "controller" / "trajectory.json").is_file()}
        lacking = [kind for kind in REQUIRED if not have[kind]]
        if lacking:
            missing[token] = lacking
    return missing
