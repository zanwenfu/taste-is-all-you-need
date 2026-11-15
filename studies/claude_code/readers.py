"""The rules arm's reader: one small Claude model reads each user message and returns its standing instructions, word for word.

It uses the same reader prompt and the same reply rule as the long-session study (studies.context_rot.criteria), so the two studies'
rule lists are written the same way: only quotes found word for word in the message are kept. The model is called through Claude Code
itself (claude -p) on the user's subscription, with --bare so that the call runs no hooks: the reader runs inside the agent's own
UserPromptSubmit hook, and a hook that started the same hook again would never end. Named in CC_STUDY_RULE_READER as
studies.claude_code.readers:claude_reader.
"""

from __future__ import annotations

import json
import os
import subprocess

from studies.context_rot.criteria import parse_reply, reader_prompt

READER_MODEL = os.environ.get("CC_STUDY_READER_MODEL", "claude-haiku-5-5")
READER_TIMEOUT_S = 90  # inside the hook's own timeout, which settings.py sets above the 30-second default


def claude_reader(message: str, *, run=subprocess.run, claude: str = "claude") -> list[str]:
    """The standing instructions in one user message, as the reader quotes them; [] when it finds none."""
    system, user = reader_prompt(message)
    argv = [claude, "-p", "--bare", "--model", READER_MODEL, "--output-format", "json", "--max-turns", "1",
            "--system-prompt", system["content"], "--disallowedTools", "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch"]
    done = run(argv, input=user["content"], capture_output=True, text=True, timeout=READER_TIMEOUT_S, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"reader call exited {done.returncode}: {done.stderr[-300:]}")
    reply = json.loads(done.stdout).get("result") or ""
    kept, _dropped = parse_reply(reply, message)
    return kept
