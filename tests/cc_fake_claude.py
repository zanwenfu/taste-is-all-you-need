"""A fake ``claude`` for the Claude Code reviewer's unit tests: no model, no network.

Installed as an executable named ``claude`` (``install(bin_dir, scenario)``). Each invocation takes the next run of
the scenario (a JSON list) and plays it as Claude Code would: for each turn it writes the assistant message to the
session's transcript under ``CLAUDE_CONFIG_DIR``, sends each tool call through the PreToolUse hook named in the
``--settings`` file (with the token scrubbed from the hook's environment, as Claude Code does), runs a rewritten
command with ``/bin/sh`` when the run asks (the command is then a ``docker exec`` that the docker stand-in answers),
and sends the batch through the PostToolBatch hook. Then it prints the run's result object and exits with its code.

A run: {"turns": [[{"name": "Bash", "input": {...}}, ...], ...], "execute": bool, "skip_hook": bool, "no_ids": bool,
"no_results": bool,
"result": {...}, "exit": 0, "sleep": 0, "stderr": ""}. Every invocation is logged to ``calls.jsonl`` beside the
scenario: its arguments, whether the token was in its environment, the prompt it read, and what the hooks said.
"""

from __future__ import annotations

import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

TOKEN = "CLAUDE_CODE_OAUTH_TOKEN"


def install(bin_dir, scenario, python=sys.executable):
    bin_dir = Path(bin_dir)
    (bin_dir / "scenario.json").write_text(json.dumps(scenario))
    path = bin_dir / "claude"
    path.write_text(f"#!/bin/sh\nexec {shlex.quote(python)} -I {shlex.quote(str(Path(__file__).resolve()))} "
                    f"{shlex.quote(str(bin_dir))} \"$@\"\n")
    path.chmod(0o755)
    return path


def _value(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


def _hook(command, event, env):
    done = subprocess.run(["/bin/sh", "-c", command], input=json.dumps(event), capture_output=True, text=True,
                          env=env, timeout=60)
    try:
        out = json.loads(done.stdout) if done.stdout.strip() else {}
    except ValueError:
        out = {"unparsed": done.stdout}
    return done.returncode, out, done.stderr


def main(argv):
    here, args = Path(argv[1]), argv[2:]
    if "--version" in args:
        print("9.9.9 (Claude Code)")
        return 0
    # Which run this is: one byte appended per invocation, under a lock, so that reviews running at once each take
    # their own.
    with open(here / "count", "a+b") as count:
        fcntl.flock(count, fcntl.LOCK_EX)
        count.seek(0, os.SEEK_END)
        index = count.tell()
        count.write(b".")
    runs = json.loads((here / "scenario.json").read_text())
    run = runs[min(index, len(runs) - 1)]
    prompt = sys.stdin.read()
    log = {"index": index, "argv": args, "token_present": TOKEN in os.environ,
           "env_keys": sorted(os.environ), "prompt": prompt[:400], "hooks": []}
    session = _value(args, "--session-id") or _value(args, "--resume")
    config = Path(os.environ["CLAUDE_CONFIG_DIR"])
    transcript = config / "projects" / "-fake" / f"{session}.jsonl"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    settings = json.loads(Path(_value(args, "--settings")).read_text())
    pre = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    post = settings["hooks"]["PostToolBatch"][0]["hooks"][0]["command"]
    hook_env = {key: value for key, value in os.environ.items() if key != TOKEN}
    for number, turn in enumerate(run.get("turns") or ()):
        message_id = f"msg_fake_{index}_{number}"
        blocks = [{"type": "tool_use", "id": f"toolu_{index}_{number}_{k}", "name": call["name"],
                   "input": call["input"]} for k, call in enumerate(turn)]
        with transcript.open("a") as handle:
            handle.write(json.dumps({"type": "assistant", "message": {
                "id": message_id, "model": "claude-haiku-5-5", "role": "assistant", "content": blocks,
                "usage": {"input_tokens": 100, "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 9000,
                          "output_tokens": 50}}}) + "\n")
        calls = []
        for block in blocks:
            if run.get("skip_hook"):
                calls.append({"tool_name": block["name"], "tool_input": block["input"], "tool_use_id": block["id"],
                              "tool_response": "ran without the hook"})
                continue
            event = {"hook_event_name": "PreToolUse", "tool_name": block["name"], "tool_input": block["input"],
                     "tool_use_id": block["id"], "session_id": session}
            if run.get("no_ids"):  # a Claude Code that sends its hook no tool-use ID
                del event["tool_use_id"]
            code, out, err = _hook(pre, event, hook_env)
            specific = out.get("hookSpecificOutput") or {}
            entry = {"event": "PreToolUse", "tool": block["name"], "exit": code, "out": out, "stderr": err[-300:]}
            response = "ok"
            if code == 2 or specific.get("permissionDecision") == "deny":
                response = specific.get("permissionDecisionReason") or err
            elif "updatedInput" in specific and run.get("execute"):
                done = subprocess.run(["/bin/sh", "-c", specific["updatedInput"]["command"]], capture_output=True,
                                      text=True, timeout=60, env=hook_env)
                response = done.stdout + done.stderr
            elif "updatedInput" in specific:
                # Not run here: answered as the container wrapper answers a command with no output.
                response = "exit code: 0\noutput:\n(none)"
            entry["response"] = response
            # The result as Claude Code keeps it in the transcript, which the runner reads back.
            if not run.get("no_results"):  # no_results: a transcript that keeps no tool results
                with transcript.open("a") as handle:
                    handle.write(json.dumps({"type": "user", "message": {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": block["id"], "content": response}]}}) + "\n")
            log["hooks"].append(entry)
            calls.append({"tool_name": block["name"], "tool_input": block["input"], "tool_use_id": block["id"],
                          "tool_response": response})
        if not run.get("skip_hook"):
            code, out, err = _hook(post, {"hook_event_name": "PostToolBatch", "tool_calls": calls,
                                          "session_id": session}, hook_env)
            log["hooks"].append({"event": "PostToolBatch", "exit": code, "out": out})
            if out.get("continue") is False:
                break
    time.sleep(run.get("sleep", 0))
    with (here / "calls.jsonl").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(log) + "\n")
    if "result" in run:
        sys.stdout.write(json.dumps({"type": "result", "session_id": session, **run["result"]}))
    if run.get("stderr"):
        sys.stderr.write(run["stderr"])
    return run.get("exit", 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
