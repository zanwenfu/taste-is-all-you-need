"""The hooks block of the project's .claude/settings.json for each arm, written before a session starts.

Every arm logs compaction: PreCompact and PostCompact each add a line with the trigger and the time to the compaction log, and print
nothing (see hooks/compaction_log.py). On top of that:

    floor, native, restated   nothing more; they differ in the environment and the prompts (see driver.py)
    rules                     UserPromptSubmit runs the rule-list hook; SessionStart with matcher "compact" gives the list back after
                              every compaction
    mem0                      UserPromptSubmit searches Mem0; Stop adds the exchange that just ended

Each hook is a command running the study's own Python on a script in hooks/, with -B so that no bytecode is written into the study's
code. A hook that calls a model gets a longer timeout than UserPromptSubmit's 30 seconds by default.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

ARMS = ("floor", "native", "rules", "mem0", "restated")
HOOKS = Path(__file__).resolve().parent / "hooks"
RULES_FILE = Path(".claude") / "standing_rules.md"  # in the project, beside settings.json
COMPACTION_LOG = "compaction.jsonl"
RULE_LOG = "rule_list.jsonl"
MEM0_LOG = "mem0.jsonl"
TIMEOUTS = {"log": 30, "rules": 120, "compact": 60, "mem0_search": 300, "mem0_add": 120}  # seconds; a Mem0 search waits for its adds


def _entry(command: str, timeout: int, matcher: str | None = None) -> dict[str, Any]:
    hooks = {"hooks": [{"type": "command", "command": command, "timeout": timeout}]}
    return hooks if matcher is None else {"matcher": matcher, **hooks}


def hooks_block(arm: str, *, project: Path, state: Path, python: str) -> dict[str, list[dict[str, Any]]]:
    """The hooks for one arm. ``state`` is the session's folder for hook logs and Mem0's waiting prompt, outside the project."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; the arms are {', '.join(ARMS)}")

    def run(script: str, *arguments: object) -> str:
        return " ".join(shlex.quote(str(part)) for part in (python, "-B", HOOKS / script, *arguments))

    log = Path(state) / COMPACTION_LOG
    hooks = {"PreCompact": [_entry(run("compaction_log.py", "--event", "PreCompact", "--log", log), TIMEOUTS["log"])],
             "PostCompact": [_entry(run("compaction_log.py", "--event", "PostCompact", "--log", log), TIMEOUTS["log"])]}
    if arm == "rules":
        rules, rule_log = Path(project) / RULES_FILE, Path(state) / RULE_LOG
        hooks["UserPromptSubmit"] = [_entry(run("rule_list_hook.py", "prompt", "--rules", rules, "--log", rule_log), TIMEOUTS["rules"])]
        hooks["SessionStart"] = [_entry(run("rule_list_hook.py", "compact", "--rules", rules, "--log", rule_log), TIMEOUTS["compact"],
                                        matcher="compact")]
    elif arm == "mem0":
        mem0_log = Path(state) / MEM0_LOG
        hooks["UserPromptSubmit"] = [_entry(run("mem0_hook.py", "prompt", "--state", state, "--log", mem0_log), TIMEOUTS["mem0_search"])]
        hooks["Stop"] = [_entry(run("mem0_hook.py", "stop", "--state", state, "--log", mem0_log), TIMEOUTS["mem0_add"])]
    return hooks


def write_settings(project: Path, arm: str, *, state: Path, python: str) -> Path:
    """Write <project>/.claude/settings.json with the arm's hooks and nothing else."""
    path = Path(project) / ".claude" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"hooks": hooks_block(arm, project=project, state=state, python=python)}, indent=2) + "\n")
    return path
