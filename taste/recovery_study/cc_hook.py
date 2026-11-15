"""The Claude Code reviewer's hook (``taste.recovery_study.cc_reviewer``): the only way its commands reach anything.

    <python> -I cc_hook.py <hook.json>        (Claude Code runs it; the event arrives on standard input)

PreToolUse, for every tool call:

- the verdict tool (StructuredOutput) passes;
- a Bash command is rewritten into ``docker exec -w <cwd> <container> /bin/sh -c <the checker's wrapper>``, unless
  no steps are left, it asks for the background, it is empty, or it names what a reviewer may not use (the
  checker's ``forbidden_reference``), when it is denied with the checker's message and nothing runs;
- any other tool is denied.

PostToolBatch, once per model reply that called tools: counts the step, and before the last one (the checker's rule:
no room for two more steps, or for two more of the slowest so far within the time) tells the model so, once.

Counts and a short event log are kept in the state file the runner reads. A failure here exits 2 on PreToolUse (the
settings say ``|| exit 2``), which blocks the call.
"""

from __future__ import annotations

import fcntl
import itertools
import json
import os
import sys
import time
from pathlib import Path


def _load(path):
    return json.loads(Path(path).read_text())


def _deny(reason):
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                   "permissionDecisionReason": reason}}


def _event(state, kind, command=""):
    state["events"].append({"t": round(time.time(), 3), "kind": kind, "command": str(command)[:300]})


def pre_tool_use(event, config, state):
    from taste.agents.checker import REFUSED, forbidden_reference
    from taste.recovery_study.cc_reviewer import ReviewSettings, container_command

    name, given = event.get("tool_name"), event.get("tool_input") or {}
    if name == config["verdict_tool"]:
        return None
    if name != "Bash":
        state["other_tools"] = state.get("other_tools", 0) + 1
        _event(state, "other_tool", name)
        return _deny(f"error: there is no tool named {name!r}; the tools are Bash and {config['verdict_tool']}")
    command = given.get("command")
    if state.get("last_step"):
        # The announced last reply: its commands are not run, as the checker's are not.
        state["denied"] += 1
        _event(state, "no_steps", command)
        return _deny(config["no_steps"])
    if given.get("run_in_background"):
        state["denied"] += 1
        _event(state, "background", command)
        return _deny(config["background"])
    if not isinstance(command, str) or not command.strip():
        state["denied"] += 1
        _event(state, "empty")
        return _deny(config["empty"])
    found = forbidden_reference(command, config["cwd"])
    if found is not None:
        state["refused"] += 1
        _event(state, "refused", command)
        return _deny(REFUSED.format(name=found))
    state["commands"] += 1
    # Which calls went into the container: the runner reads each one's result back from the transcript and fails the
    # review if the wrapper's exit line is not in it (the container, the daemon or the connection was lost).
    state.setdefault("rewritten", []).append(event.get("tool_use_id"))
    _event(state, "command", command)
    settings = ReviewSettings(docker=config["docker"], **config["settings"])
    rewritten = {**given, "command": container_command(command, config["container"], config["cwd"], settings,
                                                       user=config.get("user"), timeout=config["timeout"]),
                 "timeout": config["tool_timeout_ms"], "run_in_background": False}
    # The hook approves the command it has just confined to the review's own container (the user's yes, Sun Nov 15
    # 2026, 3:15 pm ET): without it Claude Code's own check refused a 'sh -c' script with rm that it could not read,
    # so a third of one task family's reviews failed, where Luna's checker could run any command in its container.
    # Only this rewritten command is approved; every other tool and every refused command is still denied above.
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow",
                                   "permissionDecisionReason": "confined to the review's own container",
                                   "updatedInput": rewritten}}


def post_tool_batch(event, config, state):
    now = time.time()
    state["batches"] += 1
    state["batch_times"].append(round(now, 3))
    if state.get("last_step"):
        if "Bash" in [call.get("tool_name") for call in event.get("tool_calls") or ()]:
            # The announced last reply ran no command and gave no verdict: the review ends at its limit, as the
            # checker's does.
            _event(state, "ended_at_limit", state["last_step"])
            return {"continue": False, "stopReason": "no steps are left"}
        return None
    times = [config["started"], *state["batch_times"]]
    slowest = max(later - earlier for earlier, later in itertools.pairwise(times))
    reason = None
    if state["batches"] + 2 > config["max_steps"]:
        reason = "steps"
    elif now - config["started"] + 2 * slowest > config["max_seconds"]:
        reason = "seconds"
    if reason is None:
        return None
    state["last_step"] = reason
    _event(state, "last_step", reason)
    return {"hookSpecificOutput": {"hookEventName": "PostToolBatch", "additionalContext": config["last_step"]}}


def main(argv):
    config = _load(argv[1])
    sys.path.insert(0, config["repo"])
    event = json.loads(sys.stdin.read() or "{}")
    kind = event.get("hook_event_name")
    with open(config["state"] + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = _load(config["state"])
        # Whether the subscription's token reached this process (it must not: Claude Code scrubs it); never its value.
        state["token_in_env"] = bool(state.get("token_in_env")) or "CLAUDE_CODE_OAUTH_TOKEN" in os.environ
        if kind == "PreToolUse":
            out = pre_tool_use(event, config, state)
        elif kind == "PostToolBatch":
            out = post_tool_batch(event, config, state)
        else:
            out = None
        temporary = config["state"] + ".tmp"
        Path(temporary).write_text(json.dumps(state))
        Path(temporary).replace(config["state"])
    if out is not None:
        sys.stdout.write(json.dumps(out))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception as error:  # any failure must block the call
        sys.stderr.write(f"reviewer hook failed: {type(error).__name__}\n")
        sys.exit(2)
