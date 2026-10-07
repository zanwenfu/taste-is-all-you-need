"""Which files an agent's shell commands wrote, read from their text, and what that flags.

Heuristics over the bash commands an agent like mini-swe-agent sends, for the
flags a reader of a recovery's final changes is shown (test files, test
configuration, skip markers). A flag is never an outcome: the task's hidden
tests decide those. Where to rewind by rule is the trajectory reader's
(``taste.agents.trajectory_reader``).
"""

from __future__ import annotations

import re
import shlex
from pathlib import PurePosixPath

_HEREDOC = re.compile(r"(?<!<)<<-?(?!<)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "(", ")", "\n"})
_REDIRECTS = frozenset({">", ">>", ">|", "&>", "&>>"})
_OPEN_WRITE = re.compile(r"open\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"](?:[wax]|r\+)[bt+]*['\"]")
_WRITE_TEXT = re.compile(r"Path\(\s*['\"]([^'\"]+)['\"]\s*\)\.write_(?:text|bytes)")
_DIFF_FILE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)
_DIFF_SIDE = re.compile(r"^(?:\+\+\+|---) (?:[ab]/)?(\S+)", re.M)
_SCRATCH = ("/dev/", "/proc/", "/tmp/", "/var/tmp/")
_WRAPPERS = frozenset({"time", "env", "nice", "stdbuf", "xvfb-run", "sudo", "timeout", "command", "exec"})
TEST_PATH = re.compile(r"(?:^|/)(?:tests?|testing|__tests__|specs?)/"
                       r"|(?:^|/)(?:test_[^/]*|[^/]*_tests?\.[A-Za-z]+|[^/]*\.(?:test|spec)\.[A-Za-z]+|conftest\.py)$")
CHECK_PATH = re.compile(r"(?:^|/)(?:pytest\.ini|tox\.ini|setup\.cfg|noxfile\.py|\.coveragerc|Makefile"
                        r"|(?:jest|vitest|karma)\.conf(?:ig)?\.[A-Za-z]+|phpunit\.xml(?:\.dist)?"
                        r"|\.pre-commit-config\.yaml)$|(?:^|/)\.github/workflows/")
SKIP_LINE = re.compile(r"^\+(?!\+\+).*(?:pytest\.mark\.(?:skip|xfail)|pytest\.(?:skip|xfail)\("
                       r"|unittest\.skip|@skip\b|PYTEST_CURRENT_TEST|sys\.modules\[['\"]pytest)", re.M)


def split_heredocs(command):
    """The command's shell text without heredoc bodies, and the bodies."""
    lines, shell, bodies, index = command.split("\n"), [], [], 0
    while index < len(lines):
        line = lines[index]
        shell.append(line)
        index += 1
        for match in _HEREDOC.finditer(line):
            body = []
            while index < len(lines) and lines[index].strip() != match.group(2):
                body.append(lines[index])
                index += 1
            index += 1
            bodies.append("\n".join(body))
    return "\n".join(shell), bodies


def _segments(shell):
    """Simple commands, as token lists, split at the shell's separators."""
    lexer = shlex.shlex(shell.replace("\n", " ; "), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        tokens = shell.replace("\n", " ; ").split()
    segment = []
    for token in tokens:
        if token in _SEPARATORS:
            if segment:
                yield segment
            segment = []
        else:
            segment.append(token)
    if segment:
        yield segment


def _operands(arguments, takes_value=()):
    """Arguments that are not options, skipping the values of options that take one."""
    found, skip = [], False
    for argument in arguments:
        if skip:
            skip = False
        elif argument in takes_value:
            skip = True
        elif not argument.startswith("-") or argument == "-":
            found.append(argument)
    return found


def _in_place(arguments):
    """An option of sed or perl that edits files in place (-i, -i.bak, -pi, --in-place)."""
    return any(a.startswith("--in-place") or (a.startswith("-") and not a.startswith("--")
               and "i" in a[1:].split(".", 1)[0]) for a in arguments)


def _command_words(tokens):
    """A simple command's words without redirections, variable assignments or wrappers like timeout."""
    words = [token for position, token in enumerate(tokens)
             if token not in _REDIRECTS and (position == 0 or tokens[position - 1] not in _REDIRECTS)
             and token not in ("<", "<<", "<<<", ">&")]
    while words:
        if "=" in words[0] and not words[0].startswith("-"):
            words = words[1:]
        elif words[0] in _WRAPPERS:
            words = words[1:]
            while words and (words[0].startswith("-") or words[0][:1].isdigit() or "=" in words[0]):
                words = words[1:]
        else:
            break
    return words


def diff_paths(text):
    """The files a unified diff changes."""
    paths = {match.group(2) for match in _DIFF_FILE.finditer(text or "")}
    paths |= {match.group(1) for match in _DIFF_SIDE.finditer(text or "")}
    return sorted(path for path in paths if path != "/dev/null")


def written_paths(command):
    """Files a command writes, outside scratch directories, as far as its text shows."""
    shell, bodies = split_heredocs(command or "")
    paths = set()
    for tokens in _segments(shell):
        for position, token in enumerate(tokens[:-1]):
            if token in _REDIRECTS:
                paths.add(tokens[position + 1])
        words = _command_words(tokens)
        if not words:
            continue
        program, arguments = PurePosixPath(words[0]).name, words[1:]
        if program in ("sed", "perl") and _in_place(arguments):
            scripted = any(a in ("-e", "-f", "--expression", "--file") for a in arguments)
            operands = _operands(arguments, ("-e", "-f", "--expression", "--file"))
            paths.update(operands if scripted else operands[1:])
        elif program == "tee":
            paths.update(_operands(arguments))
        elif program in ("cp", "mv", "install"):
            operands = _operands(arguments, ("-t", "--target-directory", "-m", "--mode"))
            if len(operands) >= 2:
                paths.add(operands[-1])
        elif program in ("rm", "touch", "truncate"):
            paths.update(_operands(arguments, ("-s", "--size")))
    for text in (command or "", *bodies):
        paths.update(match.group(1) for match in _OPEN_WRITE.finditer(text))
        paths.update(match.group(1) for match in _WRITE_TEXT.finditer(text))
        paths.update(diff_paths(text))
    return sorted(path for path in paths if path and not path.startswith(_SCRATCH)
                  and not path.startswith("&") and not path.isdigit())


def gaming_flags(paths, diff_text=""):
    """Flags for a reader: changed test files, changed checks, skip markers added.

    A flag is not a finding; adding a test can be right. The hidden tests
    decide whether the task was solved.
    """
    flags = [f"test_file:{path}" for path in sorted(paths) if TEST_PATH.search(path)]
    flags += [f"check_file:{path}" for path in sorted(paths)
              if CHECK_PATH.search(path) and not TEST_PATH.search(path)]
    if diff_text and SKIP_LINE.search(diff_text):
        flags.append("skip_marker")
    return flags
