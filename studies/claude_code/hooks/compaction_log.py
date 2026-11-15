"""Claude Code hook for every arm: one line per PreCompact or PostCompact event, with its trigger and time, in the compaction log.

It prints nothing, because PreCompact's output becomes instructions for the compaction summary, and it always exits 0, because exit
code 2 would block the compaction. The scorer reads this log beside the stream's compact_boundary lines, as a check that both agree.

    python -B compaction_log.py --event PreCompact --log compaction.jsonl < hook-input.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

if __package__ in (None, ""):  # run by Claude Code as a script: make the study's package importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studies.claude_code.hooks.common import append_jsonl, now, read_event  # noqa: E402


def main(argv: list[str] | None = None, stdin: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--event", default="")
    parser.add_argument("--log", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        event = read_event(stdin or sys.stdin)
        append_jsonl(args.log, {"time": now(), "event": event.get("hook_event_name") or args.event, "trigger": event.get("trigger"),
                                "session_id": event.get("session_id")})
    except Exception as error:  # a lost log line must not touch the session; stderr is not shown to the model
        print(f"compaction log: {type(error).__name__}: {error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
