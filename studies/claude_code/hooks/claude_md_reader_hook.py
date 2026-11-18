"""The capture test's reader condition (agent-os session, Nov 17 2026; notes/capture-test-plan.md): Claude Code as shipped, plus a Stop
hook that, at the end of every turn, has the small reader read the user's last message and copies each standing instruction it finds
into the project's CLAUDE.md, word for word, under one heading. The store stays CLAUDE.md, as in the condition where the user asks
Claude to keep it; only the writer changes.

The user's message comes from Claude Code's own transcript, read the way the git memory's Stop hook reads it (compaction summaries
are stored as user rows and are never taken for the user's words). An instruction close to one already kept (the git memory's own
word-overlap test, at the same threshold) is not added again, unless the two name different numbers: a changed limit is a new line.
Where two lines disagree, the heading says the later one stands.

Standard library only on its side: it runs inside the task container (Python 3.8 or later). It never blocks Claude: it prints
nothing, and any failure is written to its log.

    python3 -B claude_md_reader_hook.py --project /app --log /cc/reader/claude-md-reader.jsonl < stop-hook-input.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HEADING = "## Standing instructions from the user (oldest first; where two disagree, the later one replaces the earlier)"
CODE = os.environ.get("CC_STUDY_CODE", "/cc/code")


def _code_path() -> None:
    if CODE not in sys.path:
        sys.path.insert(0, CODE)


def kept_lines(text: str) -> list[str]:
    """The instructions already under the heading."""
    lines = text.splitlines()
    if HEADING not in lines:
        return []
    out = []
    for line in lines[lines.index(HEADING) + 1:]:
        if line.startswith("#"):
            break
        if line.startswith("- "):
            out.append(line[2:].strip())
    return out


def add_instructions(claude_md: Path, instructions: list[str], similar=None, threshold: float = 0.6) -> dict:
    """Append each instruction under the heading (made at the end of the file the first time), skipping one close to an instruction
    already there; anything Claude wrote in the file stays as it was. Returns what was added and what was skipped."""
    if similar is None:
        _code_path()
        from studies.claude_code.git_memory import SIMILAR, similar  # noqa: E402  (the git memory's own test and threshold)
        threshold = SIMILAR
    text = claude_md.read_text(errors="replace") if claude_md.exists() else ""
    kept = kept_lines(text)
    added, skipped = [], []
    for instruction in (" ".join(i.split()) for i in instructions):
        if not instruction:
            continue
        if any(instruction == k or similar(instruction, k) >= threshold for k in kept + added):
            skipped.append(instruction)
        else:
            added.append(instruction)
    if not added:
        return {"added": [], "skipped": skipped}
    lines = text.splitlines()
    new = [f"- {a}" for a in added]
    if HEADING in lines:
        at = lines.index(HEADING) + 1
        while at < len(lines) and not lines[at].startswith("#"):
            at += 1
        while at > lines.index(HEADING) + 1 and not lines[at - 1].strip():  # after the last instruction, before any blank lines
            at -= 1
        lines[at:at] = new
    else:
        lines += ([""] if lines and lines[-1].strip() else []) + [HEADING, ""] + new
    tmp = claude_md.with_name(claude_md.name + ".reader-tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, claude_md)
    return {"added": added, "skipped": skipped}


def run(event: dict, project: Path, log: Path, read=None, turn_of=None) -> dict:
    """One Stop event: read the user's last message and keep its standing instructions. Returns what it did (also logged)."""
    record = {"time": time.time(), "session": str(event.get("session_id") or "")}
    try:
        if event.get("stop_hook_active"):
            record["skipped"] = "stop hook active"
            return record
        if turn_of is None:
            _code_path()
            from studies.claude_code.hooks.git_memory_hook import last_turn as turn_of  # noqa: E402
        user = turn_of(Path(str(event.get("transcript_path") or ""))).get("user", "")
        record["user_chars"] = len(user)
        if not user.strip():
            record["found"] = 0
            return record
        if read is None:
            _code_path()
            from studies.claude_code.readers import claude_reader as read  # noqa: E402  (the small model, on the subscription)
        found = read(user)
        done = add_instructions(project / "CLAUDE.md", found)
        record.update({"found": len(found), "added": done["added"], "skipped": done["skipped"]})
        return record
    except Exception as error:  # the reader failing must never stop Claude: log it
        record["error"] = f"{type(error).__name__}: {error}"[:300]
        return record
    finally:
        try:
            with log.open("a") as out:
                out.write(json.dumps(record) + "\n")
        except OSError:
            pass


def main(argv=None, stdin=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=Path("/app"))
    parser.add_argument("--log", type=Path, default=Path("/cc/reader/claude-md-reader.jsonl"))
    args = parser.parse_args(argv)
    raw = (stdin or sys.stdin).read()
    try:
        event = json.loads(raw) if raw.strip() else {}
    except ValueError:
        event = {}
    run(event if isinstance(event, dict) else {}, args.project, args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
