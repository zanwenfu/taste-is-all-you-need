"""A trial's final change: git's diff of the task's files against the commit they started from, read in its container.

The hidden tests grade the files a trial leaves. A study that must read those files later (did keeping going keep the
gap a reviewer named, or did a restart close it?) keeps them as text. With the trial setting ``final_change_chars``,
the trial's owner reads two things through the task's terminal, as the task's own user, in the task's directory:

- before anything changes the files (a branch's restore, the agent's first command), the commit the repository is at
  (``base_command``): the task's base commit, which its verifier also resets the tests to;
- once the agent has ended, its final files are saved and before the verifier runs, git's account of what the files
  became (``change_command``): the paths changed against that commit (tracked files as ``git diff --name-status``
  lists them, then every file git neither tracks nor ignores, marked ``?``), and the diff (tracked files first, then
  each such new file as a diff from nothing, at most ``NEW_FILES`` of them; binary files are named, not shown).

Neither changes the repository. A diff against a commit reads the index for its cache of the files' stat and, finding a
file written again as it was, writes the index back refreshed (whatever GIT_OPTIONAL_LOCKS says); so these reads use a
copy of the index in a temporary file, removed when they end, and the index stays as the agent left it. No external
diff or text conversion runs, and the diff has one form whatever the configuration (no colour, a/ and b/ prefixes, no
renames). The record (``read_change``) keeps the diff up to ``final_change_chars`` characters, its whole length,
whether it was cut and the cap; the changed paths up to ``PATHS``; and the commit at the end, with whether the agent
moved it (its own commits are in the diff, which is against the base). A directory that is not a git repository, or a
read that fails, is recorded with why; the trial is graded all the same.
"""

from __future__ import annotations

import re
import time

from taste.brains.terminal_broker import TerminalRequest

# The record lists at most this many changed paths, and the diff shows at most this many new files whole.
PATHS = 1000
NEW_FILES = 200
# The reads' time limits: a diff takes seconds even on the largest of these repositories, and the change is read in
# the trial's handoff time (150 s by default), after its final save.
BASE_SECONDS = 30.0
CHANGE_SECONDS = 60.0
# Bounds of the setting: room for a heading and a few files, and at most what the terminal keeps of one output.
LEAST_CHARS, MOST_CHARS = 1000, 1_000_000
ACTOR = "taste-trial-owner"
_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
# git with no optional locks, no pager, paths as they are, in a repository of any owner.
_GIT = ("g() { GIT_OPTIONAL_LOCKS=0 GIT_PAGER=cat PAGER=cat LC_ALL=C git -c safe.directory='*' "
        "-c core.quotepath=off \"$@\"; }\n")
# The same, reading a copy of the index (none if the repository has none), removed when the read ends.
_COPY = ("index=$(g rev-parse --git-path index) && copy=$(mktemp) || exit 3\n"
         "trap 'rm -f \"$copy\"' EXIT\n"
         "cp \"$index\" \"$copy\" 2>/dev/null || rm -f \"$copy\"\n"
         "c() { GIT_INDEX_FILE=\"$copy\" g \"$@\"; }\n")
# One form whatever the repository's or the user's configuration says (colour, prefixes, renames, external tools).
_DIFF = "--no-color --no-ext-diff --no-textconv --no-renames --src-prefix=a/ --dst-prefix=b/"
_NEW = "c ls-files --others --exclude-standard"


def _mark(name):
    return f"@@taste-final-change {name}@@"


def base_command():
    """The command that prints the commit the task's repository is at, and nothing if it is none."""
    return _GIT + "g rev-parse --verify --quiet HEAD"


def change_command(base):
    """The command that prints the commit at the end, the changed paths and the diff against ``base``, in marked
    sections, ``end`` last."""
    if not isinstance(base, str) or not _COMMIT.fullmatch(base):
        raise ValueError("the base is a commit's full hexadecimal name")
    return _GIT + _COPY + "\n".join([
        f"echo '{_mark('head')}'", "g rev-parse --verify --quiet HEAD",
        f"echo '{_mark('new')}'", f"{_NEW} | wc -l",
        f"echo '{_mark('paths')}'", f"c diff {_DIFF} --name-status {base} --",
        f"{_NEW} | head -n {PATHS} | while IFS= read -r f; do printf '?\\t%s\\n' \"$f\"; done",
        f"echo '{_mark('diff')}'", f"c diff {_DIFF} {base} --",
        f"{_NEW} | head -n {NEW_FILES} | while IFS= read -r f; do g diff {_DIFF} --no-index -- /dev/null \"$f\"; done",
        f"echo '{_mark('end')}'"])


def read_base(result):
    """The commit a base read printed, or None."""
    found = result.stdout.decode("utf-8", "replace").strip()
    return found if result.return_code == 0 and _COMMIT.fullmatch(found) else None


def _between(text, first, second):
    start = text.find(first + "\n")
    if start < 0:
        return None
    start += len(first) + 1
    end = text.find(second + "\n", start)
    return text[start:] if end < 0 else text[start:end]


def read_change(result, base, cap_chars, seconds=None):
    """The record of a change read: the diff kept up to ``cap_chars``, its whole length, whether it was cut and the
    cap; the changed paths; the commit at the end. A read whose diff never began says why instead."""
    text = result.stdout.decode("utf-8", "replace")
    start = text.find(_mark("diff") + "\n")
    if start < 0:
        why = result.stderr.decode("utf-8", "replace").strip()[-500:]
        return {"read": False, "reason": why or f"the read ended with exit code {result.return_code} before its diff",
                "base": base, "cap_chars": cap_chars}
    end = text.rfind(_mark("end") + "\n")
    diff = text[start + len(_mark("diff")) + 1:end if end > start else len(text)]
    complete = end > start and not result.stdout_dropped_bytes and not result.terminated
    head = (_between(text, _mark("head"), _mark("new")) or "").strip() or None
    counted = (_between(text, _mark("new"), _mark("paths")) or "").strip()
    new = int(counted) if counted.isdigit() else None
    listed = (_between(text, _mark("paths"), _mark("diff")) or "").splitlines()
    total = sum(1 for line in listed if not line.startswith("?\t")) + (new if new is not None else
                                                                        sum(1 for line in listed if line.startswith("?\t")))
    return {"read": True, "base": base, "head": head, "head_moved": head is not None and head != base,
            "paths": listed[:PATHS], "paths_more": max(0, total - min(len(listed), PATHS)), "new_files": new,
            "diff": diff[:cap_chars], "diff_chars": len(diff), "cap_chars": cap_chars,
            "cut": len(diff) > cap_chars or not complete, "complete": complete,
            "dropped_bytes": result.stdout_dropped_bytes, "terminated": result.terminated or "",
            **({"seconds": round(seconds, 3)} if seconds is not None else {})}


def _run(backend, name, command, cwd, seconds):
    return backend.execute(TerminalRequest(f"final-change-{name}", ACTOR, command, cwd, seconds))


def base_commit(backend, cwd):
    """(the commit the task's files start from, "") or (None, why): read before anything changes them."""
    try:
        commit = read_base(_run(backend, "base", base_command(), cwd, BASE_SECONDS))
    except Exception as failure:
        return None, f"the base read failed: {type(failure).__name__}"
    return (commit, "") if commit else (None, f"{cwd} is not a git repository with a commit when the trial starts")


def final_change(backend, cwd, base, cap_chars):
    """The record of the task's final change against ``base`` (``read_change``)."""
    started = time.monotonic()
    result = _run(backend, "read", change_command(base), cwd, CHANGE_SECONDS)
    return read_change(result, base, cap_chars, time.monotonic() - started)
