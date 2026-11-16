"""Claude Code hook for the rules arm: a reader keeps the user's standing instructions in a rules file, and the whole list goes back
to the agent with every prompt and after every compaction.

    prompt   UserPromptSubmit: the reader reads the prompt; each standing instruction it finds that is not listed yet is added to the
             rules file; the whole list is returned as additional context
    compact  SessionStart, matcher "compact": the whole list again, since compaction only summarizes context a hook added earlier

The reader is a function from one message to the instructions in it, word for word, named "module:function" in CC_STUDY_RULE_READER.
The study's reader is a small model; stub_reader knows the study's own cue words and serves tests and plumbing runs only. Each call
is logged with --log; a reader that fails is logged and the list so far still goes out. The hook never exits with 2, which would
block the prompt.

    python -B rule_list_hook.py prompt --rules RULES.md --log rule_list.jsonl < hook-input.json
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TextIO

if __package__ in (None, ""):  # run by Claude Code as a script: make the study's package importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studies.claude_code.hooks.common import CONTEXT_LIMIT, append_jsonl, context_output, fit, load, now, read_event  # noqa: E402

READER_ENV = "CC_STUDY_RULE_READER"
HEADER = "Standing instructions the user gave earlier in this session, oldest first. Where two disagree, the later one stands."
_CUE = re.compile(r"\b(?:for the rest of this session|from now on)\b", re.IGNORECASE)


def stub_reader(message: str) -> list[str]:
    """Each sentence holding "for the rest of this session" or "from now on", word for word. Tests and plumbing runs only."""
    return [part for part in (piece.strip() for piece in re.split(r"(?<=[.!?])\s+", message)) if part and _CUE.search(part)]


def listed(path: Path) -> list[str]:
    path = Path(path)
    return [line[2:].strip() for line in path.read_text().splitlines() if line.startswith("- ")] if path.exists() else []


def add(path: Path, found: list[str]) -> list[str]:
    """Append each instruction not listed yet, as one line; return those added."""
    known, new = set(listed(path)), []
    for text in found:
        line = " ".join(str(text).split())
        if line and line not in known:
            known.add(line)
            new.append(line)
    if new:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a") as file:
            file.writelines(f"- {line}\n" for line in new)
    return new


def commit(path: Path, new: list[str], run=None) -> bool:
    """Our git memory: the rules file lives in a git repository and each reading that adds rules is one commit, so the list has a
    history that can be read, diffed and rolled back like the agent's code. A failure here is returned, never raised."""
    import os
    import subprocess
    import tempfile
    run = run or subprocess.run
    folder = Path(path).parent.resolve()
    # The memory folder is mounted from the host, so its owner differs from the container's root, and git refuses a repository owned
    # by someone else ("dubious ownership"). Newer git takes safe.directory on the command line; the older git of some task images
    # (2.25.1 with the backported check, Nov 16) reads it only from a config file, so git also gets a home of its own holding one.
    home = Path(tempfile.mkdtemp(prefix="taste-memory-git-"))
    (home / ".gitconfig").write_text(f"[safe]\n\tdirectory = {folder}\n\tdirectory = *\n")
    env = {**os.environ, "HOME": str(home)}
    git = ["git", "-c", "safe.directory=*", "-c", "user.name=taste-memory", "-c", "user.email=memory@taste.invalid"]
    try:
        if not (folder / ".git").exists():
            run([*git, "-C", str(folder), "init", "-q"], check=True, capture_output=True, timeout=30, env=env)
        run([*git, "-C", str(folder), "add", Path(path).name], check=True, capture_output=True, timeout=30, env=env)
        message = f"rule: {new[0][:72]}" + (f" (+{len(new) - 1})" if len(new) > 1 else "")
        run([*git, "-C", str(folder), "commit", "-q", "-m", message], check=True, capture_output=True, timeout=30, env=env)
        return True
    except Exception:
        return False


def render(rules: list[str], limit: int = CONTEXT_LIMIT) -> str | None:
    """The list as the agent sees it, oldest first; when it cannot all fit, the newest rules are the ones kept."""
    return fit(HEADER, [(index, text) for index, text in reversed(list(enumerate(rules)))], limit) if rules else None


def on_prompt(event: Mapping[str, Any], *, rules_path: Path, reader: Callable[[str], list[str]], log: Path | None = None,
              git: bool = False) -> str | None:
    prompt = str(event.get("prompt") or "")
    record: dict[str, Any] = {"time": now(), "mode": "prompt", "session_id": event.get("session_id"), "prompt_chars": len(prompt)}
    try:
        found = list(reader(prompt)) if prompt.strip() else []
        added = add(rules_path, found)
        record.update({"found": found, "added": added})  # not |=: some task images have Python 3.8
        if git and added:
            record["committed"] = commit(rules_path, added)
    except Exception as error:  # the list so far still goes out; the driver stops the session on a logged error
        record["error"] = f"{type(error).__name__}: {error}"[:500]
    text = render(listed(rules_path))
    record["context_chars"] = len(text or "")
    if log:
        append_jsonl(log, record)
    return context_output("UserPromptSubmit", text) if text else None


def on_compact(event: Mapping[str, Any], *, rules_path: Path, log: Path | None = None) -> str | None:
    if event.get("source", "compact") != "compact":
        return None
    text = render(listed(rules_path))
    if log:
        append_jsonl(log, {"time": now(), "mode": "compact", "session_id": event.get("session_id"), "context_chars": len(text or "")})
    return context_output("SessionStart", text) if text else None


def main(argv: list[str] | None = None, stdin: TextIO | None = None, stdout: TextIO | None = None,
         environ: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("prompt", "compact"))
    parser.add_argument("--rules", required=True, type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--git", action="store_true", help="keep the rules file in a git repository, one commit per reading that adds")
    args = parser.parse_args(argv)
    stdin, stdout, environ = stdin or sys.stdin, stdout or sys.stdout, os.environ if environ is None else environ
    event = read_event(stdin)
    if args.mode == "compact":
        output = on_compact(event, rules_path=args.rules, log=args.log)
    else:
        try:
            reader = load(environ.get(READER_ENV, ""))
        except Exception as error:  # no reader named, or it cannot be imported: say so, and let the prompt go on
            message = f"{READER_ENV}: {type(error).__name__}: {error}"[:500]
            if args.log:
                append_jsonl(args.log, {"time": now(), "mode": "prompt", "session_id": event.get("session_id"), "error": message})
            print(message, file=sys.stderr)
            return 1
        output = on_prompt(event, rules_path=args.rules, reader=reader, log=args.log, git=args.git)
    if output:
        stdout.write(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
