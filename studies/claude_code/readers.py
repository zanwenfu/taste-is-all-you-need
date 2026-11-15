"""The rules arm's reader: one small Claude model reads each user message and returns its standing instructions, word for word.

It uses the same reader prompt and the same reply rule as the long-session study (studies.context_rot.criteria), so the two studies'
rule lists are written the same way: only quotes found word for word in the message are kept. The model is called through Claude Code
itself (claude -p) on the user's subscription. The reader runs inside the agent's own UserPromptSubmit hook, so its call must load no
hooks, or it would start the same hook again. --bare would do that, but it refuses the subscription token ("Not logged in", tested
Nov 15), so the call instead runs from an empty folder with an empty Claude Code config folder of its own, with CLAUDE.md files and
auto memory off and hooks disabled in its settings: nothing of the session's project, config or memory is loaded. Named in
CC_STUDY_RULE_READER as studies.claude_code.readers:claude_reader.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from studies.context_rot.criteria import parse_reply, reader_prompt

READER_MODEL = os.environ.get("CC_STUDY_READER_MODEL", "claude-haiku-5-5")
READER_TIMEOUT_S = 90  # inside the hook's own timeout, which settings.py sets above the 30-second default
READER_HOME_ENV = "CC_STUDY_READER_HOME"  # where the reader's empty working folder and config folder live; a temp folder if unset


def reader_home() -> Path:
    home = Path(os.environ.get(READER_HOME_ENV) or Path(tempfile.gettempdir()) / "cc-study-reader")
    for sub in ("work", "config"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    return home


def claude_reader(message: str, *, run=subprocess.run, claude: str = "claude") -> list[str]:
    """The standing instructions in one user message, as the reader quotes them; [] when it finds none."""
    system, user = reader_prompt(message)
    home = reader_home()
    argv = [claude, "-p", "--model", READER_MODEL, "--output-format", "json", "--max-turns", "1",
            "--system-prompt", system["content"], "--settings", json.dumps({"disableAllHooks": True}),
            "--disallowedTools", "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch"]
    env = {**os.environ, "CLAUDE_CONFIG_DIR": str(home / "config"), "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
           "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1"}
    done = run(argv, input=user["content"], capture_output=True, text=True, timeout=READER_TIMEOUT_S, check=False,
               cwd=str(home / "work"), env=env)
    if done.returncode != 0:
        raise RuntimeError(f"reader call exited {done.returncode}: {(done.stderr or done.stdout)[-300:]}")
    answer = json.loads(done.stdout)
    if answer.get("is_error"):
        raise RuntimeError(f"reader call failed: {str(answer.get('result'))[:300]}")
    kept, _dropped = parse_reply(answer.get("result") or "", message)
    return kept
