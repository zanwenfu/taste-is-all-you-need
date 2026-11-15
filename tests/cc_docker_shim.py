"""A stand-in for ``docker`` in the Claude Code reviewer's tests: it runs nothing.

Installed as an executable named ``docker`` (``install(bin_dir, toy_dir, log)``), it answers the reviewer's
``docker exec -w <cwd> <container> /bin/sh -c <script>`` from a local folder that stands for the container's working
directory: it reads the reviewer's own command back out of the script (``command_from_script``) and answers ``cat``,
``ls``, ``pwd``, ``head`` and ``wc`` from the folder, in the checker's observation format; any other program is "not
found". No command the reviewer writes is ever executed on this machine. Every call is logged, one JSON line each.
"""

from __future__ import annotations

import json
import posixpath
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def install(bin_dir, toy_dir, log, python=sys.executable):
    path = Path(bin_dir) / "docker"
    path.write_text(f"#!/bin/sh\nexec {shlex.quote(python)} -I {shlex.quote(str(Path(__file__).resolve()))} "
                    f"{shlex.quote(str(toy_dir))} {shlex.quote(str(log))} \"$@\"\n")
    path.chmod(0o755)
    return path


def _resolve(toy, cwd, name):
    """A path the reviewer named, inside the folder, or None for anything outside the container's directory."""
    full = posixpath.normpath(name if name.startswith("/") else posixpath.join(cwd, name))
    if full != cwd and not full.startswith(cwd.rstrip("/") + "/"):
        return None
    return Path(toy) / posixpath.relpath(full, cwd)


def answer(toy, cwd, command):
    """(exit code, output) for one command, read from the folder."""
    words = shlex.split(command.strip())
    while words[:1] == ["cd"] and len(words) >= 3 and words[2] in ("&&", ";"):
        words = words[3:]
    if not words:
        return 0, ""
    program, rest = words[0], words[1:]
    if program == "pwd":
        return 0, cwd + "\n"
    if program == "ls":
        names = [word for word in rest if not word.startswith("-")] or ["."]
        out = []
        for name in names:
            path = _resolve(toy, cwd, name)
            if path is None or not path.exists():
                return 2, f"ls: cannot access '{name}': No such file or directory\n"
            out.extend(sorted(item.name for item in path.iterdir()) if path.is_dir() else [name])
        return 0, "".join(f"{name}\n" for name in out)
    if program in ("cat", "head", "wc"):
        lines = None
        if program == "head" and rest[:1] == ["-n"] and len(rest) > 1:
            lines, rest = int(rest[1]), rest[2:]
        if program == "wc":
            rest = [word for word in rest if word != "-l"]
        out = ""
        for name in rest:
            path = _resolve(toy, cwd, name)
            if path is None or not path.is_file():
                return 1, f"{program}: {name}: No such file or directory\n"
            text = path.read_text()
            if program == "wc":
                out += f"{text.count(chr(10))} {name}\n"
            else:
                out += "".join(text.splitlines(True)[:lines]) if lines is not None else text
        return 0, out
    if program == "git":
        return 128, "fatal: not a git repository (or any of the parent directories): .git\n"
    return 127, f"/bin/sh: 1: {program}: not found (this test container has only cat, ls, pwd, head and wc)\n"


def main(argv):
    toy, log, args = argv[1], argv[2], argv[3:]
    entry = {"argv0": args[:1]}
    try:
        if args[:1] != ["exec"]:
            raise ValueError("only exec")
        rest = args[1:]
        cwd = "/"
        if rest[:1] == ["-w"]:
            cwd, rest = rest[1], rest[2:]
        if rest[:1] == ["-u"]:
            rest = rest[2:]
        container, shell, flag, script = rest[0], rest[1], rest[2], rest[3]
        if (shell, flag) != ("/bin/sh", "-c") or len(rest) != 4:
            raise ValueError("not a /bin/sh -c call")
        entry.update(container=container, cwd=cwd)
        if "for t in mktemp" in script:
            entry["preflight"] = True
            print("has:timeout")
            return 0
        sys.path.insert(0, str(REPO))
        from taste.recovery_study.cc_reviewer import command_from_script
        command = command_from_script(script)
        entry["command"] = command
        if command is None:
            code, out = 126, "shim: not a reviewer command\n"
        else:
            code, out = answer(toy, cwd, command)
        sys.stdout.write(f"exit code: {code}\noutput:\n{out or '(none)'}")
        return 0
    except (ValueError, IndexError) as error:
        entry["error"] = str(error)
        print(f"shim: {error}", file=sys.stderr)
        return 1
    finally:
        with open(log, "a") as handle:
            handle.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
