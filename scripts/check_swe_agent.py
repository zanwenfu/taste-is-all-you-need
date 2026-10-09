#!/usr/bin/env python3
"""SWE-agent run unchanged in a real task container, then branched from its record, without a model.

    sudo python3 scripts/check_swe_agent.py --image <image> --owner-token <hex> --output out.json
    sudo python3 scripts/check_swe_agent.py --owner-token <hex> --cleanup-only

SWE-agent (taste.agents.swe_agent) runs in a disposable container of a task
image (no network), its commands through the controller's terminal broker as
a trial's go, and its model replaced by a script of replies: look around, make
a file with its editor, change it, set a variable and a directory in its
shell, read them back in the next command, send a command with a syntax error
(SWE-agent asks again), and submit. What crossed the host is the replay
script, the setup included. Then, each in a fresh container of the image:

- a rebuild of steps 1..k must match the record by the fidelity rule and the
  recorded container's files and session state at k, and SWE-agent replayed
  to k there must go on live with the request the record has next, and end
  with the same submission and files;
- the rebuilt files of step k, saved as a branch checkpoint, restored into
  another container exactly, and SWE-agent replayed there finishes too;
- a note appended at step k reaches the agent with the action's output;
- a rejection of the submission keeps the agent working: it asks again,
  submits again, and the patch read its first submission did is left out.

The containers are removed afterwards; --cleanup-only removes anything left
with the same owner token.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taste.agents import ModelReply, ShellResult
from taste.agents.swe_agent import SweAgent
from taste.benchmarks import branch_trial
from taste.benchmarks.harbor_settings import TrialSettings
from taste.benchmarks.replay_export import encode_script
from taste.brains.branch_replay import (
    SCRIPT_SCHEMA,
    ReplayHost,
    ReplayScript,
    digest,
    message_shas,
    rebuild,
    rebuild_timeout,
    reply_to_dict,
    request_sha,
)
from taste.brains.docker_terminal import OWNER_LABEL, DockerTerminalBackend
from taste.brains.terminal_broker import TerminalBinding, TerminalBroker, TerminalRequest

MAX_TIMEOUT = 300.0  # a hosted worker's terminal grant
TASK = ("The repository needs a file taste_check.txt at its root, with the lines alpha and gamma. "
        "Create it and submit.")
NOTE = "Note: an attempt that continued from here was rejected by a reviewer: the file must end with gamma."
REJECTION = "Submission rejected by a reviewer: the file must end with gamma."
# The task's files and SWE-agent's own (tools, session, state); not Python's bytecode caches, which
# hold the time their source was written.
FINGERPRINT = ("cd /app && git status --porcelain=v1 -uall | sort && git diff | sha256sum && "
               "find /root/tools /root/.taste-swe-agent /root/state.json /root/.swe-agent-env -type f "
               "! -name '*.pyc' -print0 2>/dev/null | sort -z | xargs -0 -r sha256sum")
STEP = 5  # after the shell's state is read back, before the syntax error


def call(number, name, arguments, text=""):
    output = ([{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]
              if text else [])
    output.append({"type": "function_call", "call_id": f"call_{number}", "name": name,
                   "arguments": json.dumps(arguments)})
    return ModelReply(tuple(output), "completed", None, {"input_tokens": 1000, "output_tokens": 50}, "scripted", 0.0)


REPLIES = [
    call(1, "bash", {"command": "cd /app && git status --short | head -3 && ls | head -3"}, "Look around."),
    call(2, "str_replace_editor", {"command": "create", "path": "/app/taste_check.txt", "file_text": "alpha\nbeta\n"}),
    call(3, "str_replace_editor", {"command": "str_replace", "path": "/app/taste_check.txt", "old_str": "beta",
                                   "new_str": "gamma"}),
    call(4, "bash", {"command": "export TASTE_CHECK=kept && cd /app && mkdir -p taste_dir && cd taste_dir && pwd"}),
    call(5, "bash", {"command": "echo $TASTE_CHECK; pwd; cat ../taste_check.txt"}),
    call(6, "bash", {"command": "if then fi"}),
    call(7, "bash", {"command": "cd /app && rmdir taste_dir && git status --short"}),
    call(8, "submit", {}),
]


def cli(*args, check=True):
    # Fixture control only; never used to run task output.
    return subprocess.run(["docker", "--host", "unix:///var/run/docker.sock", *args],
                          capture_output=True, check=check, timeout=120)


def cleanup(token):
    listed = cli("ps", "-aq", "--filter", f"label={OWNER_LABEL}={token}").stdout.decode().split()
    for container in listed:
        cli("rm", "-f", container, check=False)
    return listed


def container(image, token):
    made = cli("create", "--pull", "never", "--network", "none", "--memory", "2g",
               "--label", f"{OWNER_LABEL}={token}", "--entrypoint", "/bin/sh", image, "-c", "sleep 1800")
    identifier = made.stdout.decode().strip()
    cli("start", identifier)
    return identifier


class Terminal:
    """One container's backend and broker, as a trial's controller holds them."""

    def __init__(self, identifier, token, directory, name):
        deadline = time.time() + 1800
        self.backend = DockerTerminalBackend.admit("/var/run/docker.sock", identifier, token, deadline)
        binding = TerminalBinding("swe_agent_check_" + name, self.backend.environment_id, deadline, 5000)
        self.broker = TerminalBroker.create(directory / ("ledger-" + name), binding, self.backend)
        self.count = 0

    async def run(self, text, cwd="/", timeout=60.0):
        self.count += 1
        result = await self.broker.execute(TerminalRequest(f"command_{self.count}", "check_agent", text, cwd,
                                                           min(float(timeout), MAX_TIMEOUT)))
        output = result.stdout.decode("utf-8", errors="replace") + result.stderr.decode("utf-8", errors="replace")
        return output, result.return_code, result.terminated == "timeout"

    async def execute(self, _number, run):
        return await self.run(run.executed, run.cwd, rebuild_timeout(run, MAX_TIMEOUT))

    def close(self):
        self.broker.close()


class Live:
    """The agent thread's host: scripted replies, commands to a container's broker on the check's loop."""

    def __init__(self, terminal, loop, replies):
        self.terminal, self.loop, self.replies = terminal, loop, list(replies)
        self.asked, self.stopped = [], None

    def sync(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()

    def ask(self, *, messages, tools, effort):
        self.asked.append(messages)
        return self.replies.pop(0)

    def run(self, command, *, cwd, timeout_seconds, shown=None):
        output, code, timed_out = self.sync(self.terminal.run(command, cwd, timeout_seconds))
        return ShellResult(output, code, timed_out=timed_out)

    def stop(self, reason, *, now=True):
        self.stopped = reason


class Recorder:
    """What crossed the host, as a replay script; the container's fingerprint when step ``at`` is done."""

    def __init__(self, inner, at):
        self.inner, self.at = inner, at
        self.setup, self.steps, self.previous, self.stopped, self.fingerprint = [], [], [], None, None

    def ask(self, *, messages, tools, effort):
        if len(self.steps) == self.at:
            self.fingerprint = self.inner.sync(self.inner.terminal.run(FINGERPRINT))[0]
        answer = self.inner.ask(messages=messages, tools=tools, effort=effort)
        shas = message_shas(messages)
        base = next((index for index, (a, b) in enumerate(zip(self.previous, shas, strict=False)) if a != b),
                    min(len(self.previous), len(shas)))
        self.previous = shas
        calls = [{"id": item["call_id"], "name": item["name"], "arguments": json.loads(item["arguments"])}
                 for item in answer.output if item["type"] == "function_call"]
        self.steps.append({"step": len(self.steps) + 1, "request_sha": request_sha(messages, tools, effort),
                           "messages": {"base": base, "added": shas[base:]}, "reply": reply_to_dict(answer),
                           "text": [], "calls": calls, "runs": [], "replayed": False})
        self.tools, self.effort = tools, effort
        return answer

    def run(self, command, *, cwd, timeout_seconds, shown=None):
        result = self.inner.run(command, cwd=cwd, timeout_seconds=timeout_seconds, shown=shown)
        (self.steps[-1]["runs"] if self.steps else self.setup).append({
            "command": shown, "executed": command, "cwd": cwd, "timeout_seconds": min(timeout_seconds, MAX_TIMEOUT),
            "output": result.output, "returncode": result.returncode, "timed_out": result.timed_out,
            "output_exact": True})
        return result

    def script(self):
        return ReplayScript.from_dict({"schema": SCRIPT_SCHEMA, "task": TASK, "setup": self.setup,
                                       "steps": self.steps, "exit": None,
                                       "source": {"trial": "swe-agent-check", "agent": SweAgent(effort="low").identity(),
                                                  "effort": self.effort, "tools_sha256": digest(self.tools)}})


def agent_run(host):
    return SweAgent(effort="low").run(TASK, host, cwd="/app", model_name="scripted")


async def fresh(image, token, directory, name, terminals):
    terminal = Terminal(container(image, token), token, directory, name)
    terminals.append(terminal)
    return terminal


async def branch(terminal, script, step, replies, **options):
    loop, rows = asyncio.get_running_loop(), []
    live = Live(terminal, loop, replies)
    host = ReplayHost(live, script, step, record=lambda kind, payload: rows.append((kind, payload)), **options)
    result = await asyncio.to_thread(agent_run, host)
    return result, rows, live


async def check(image, token, directory):
    terminals = []
    try:
        loop = asyncio.get_running_loop()
        recorded = await fresh(image, token, directory, "recorded", terminals)
        recorder = Recorder(Live(recorded, loop, REPLIES), STEP)
        result = await asyncio.to_thread(agent_run, recorder)
        final_files = (await recorded.run(FINGERPRINT))[0]
        script = recorder.script()
        path = directory / "script.json"
        path.write_bytes(encode_script(script.to_dict()))
        outputs = [[run.output for run in item.runs] for item in script.steps]

        rebuilt = await fresh(image, token, directory, "rebuilt", terminals)
        account = await rebuild(script, STEP, rebuilt.execute)
        rebuilt_files = (await rebuilt.run(FINGERPRINT))[0]
        replayed, rows, _ = await branch(rebuilt, script, STEP, REPLIES[STEP:])
        replayed_files = (await rebuilt.run(FINGERPRINT))[0]

        saved_to = directory / "checkpoint"
        options = {"model": "gpt-6-luna", "agent": "swe-agent", "services": "none", "branch": str(path),
                   "branch_step": str(STEP), "branch_checkpoint": str(saved_to)}
        inputs = branch_trial.load_inputs(TrialSettings.from_options({**options, "branch_live": "off"}))
        source = await fresh(image, token, directory, "source", terminals)
        saving = await rebuild(script, STEP, source.execute)
        saved = await asyncio.to_thread(branch_trial.save_checkpoint, source.backend, saved_to, inputs, STEP,
                                        {"faithful": saving["faithful"]}, "swe-agent-check")
        restored = await fresh(image, token, directory, "restored", terminals)
        receipt = await asyncio.to_thread(branch_trial.restore_checkpoint, restored.backend, saved_to, inputs, STEP)
        restored_files = (await restored.run(FINGERPRINT))[0]
        from_restore, restore_rows, _ = await branch(restored, script, STEP, REPLIES[STEP:])

        noted = await fresh(image, token, directory, "noted", terminals)
        await rebuild(script, STEP, noted.execute)
        with_note, _, note_live = await branch(noted, script, STEP, REPLIES[STEP:], override="append", note=NOTE)
        seen_note = json.dumps(note_live.asked[0]) if note_live.asked else ""

        last = len(script.steps)
        rejected = await fresh(image, token, directory, "rejected", terminals)
        rejected_account = await rebuild(script, last, rejected.execute)
        after_rejection, reject_rows, reject_live = await branch(rejected, script, last, [REPLIES[-1]],
                                                                 override="reject", note=REJECTION)

        prefix = dict(rows).get("prefix", {})
        checks = {
            "the recorded run submitted the file it made": (
                result.exit_status == "Submitted" and "taste_check.txt" in result.submission
                and "+gamma" in result.submission and len(recorder.steps) == len(REPLIES)),
            "the setup ran before the first model call": len(script.setup) >= 10,
            "the editor tool worked in the task image": "has been edited" in outputs[2][0],
            "the shell kept its variable and directory between commands": (
                outputs[4][0].splitlines()[:2] == ["kept", "/app/taste_dir"]),
            "a syntax error was answered by asking the model again": "syntax" in json.dumps(outputs[5]).lower(),
            "the record follows SWE-agent's submission protocol": script.submission(last)
            and not script.submission(STEP),
            "the rebuild matches the record": account["faithful"] and account["divergent"] == 0,
            "the rebuilt files and session are those of step k": rebuilt_files == recorder.fingerprint,
            "the replayed agent went on live with the recorded next request": (
                ("live", {"live": True, "context": "matched"}) in rows and prefix.get("steps") == STEP),
            "the replayed agent finished the same way": (
                replayed.exit_status == "Submitted" and replayed.submission == result.submission
                and replayed_files == final_files),
            "the checkpoint was saved and restored exactly": saved.get("saved") is True
            and receipt.get("exact") is True and restored_files == recorder.fingerprint,
            "the agent replayed on restored files finished": (
                from_restore.exit_status == "Submitted" and from_restore.submission == result.submission
                and ("live", {"live": True, "context": "matched"}) in restore_rows),
            "a note reached the agent with the action's output": NOTE in seen_note
            and with_note.exit_status == "Submitted",
            "a rejected submission kept the agent working": (
                rejected_account["faithful"] and after_rejection.exit_status == "Submitted"
                and len(reject_live.asked) == 1 and REJECTION in json.dumps(reject_live.asked[0])
                and dict(reject_rows).get("prefix", {}).get("dropped", 0) >= 1),
        }
        return {"checks": checks, "setup_commands": len(script.setup), "steps": len(script.steps),
                "rebuild": {key: account[key] for key in ("commands", "divergent", "faithful")},
                "checkpoint": saved, "restore": receipt, "prefix": prefix,
                "reject_prefix": dict(reject_rows).get("prefix"), "outputs": outputs,
                "fingerprints": {"recorded_at_k": recorder.fingerprint, "rebuilt_at_k": rebuilt_files,
                                 "restored_at_k": restored_files, "recorded_final": final_files,
                                 "replayed_final": replayed_files}}
    finally:
        for terminal in terminals:
            terminal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image")
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cleanup-only", action="store_true")
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{32}", args.owner_token) is None:
        parser.error("owner token must be 32 lowercase hex characters")
    if args.cleanup_only:
        print(json.dumps({"removed": cleanup(args.owner_token)}))
        return 0
    if not args.image or args.output is None:
        parser.error("an image and an output path are required")
    report = {"status": "failed", "checks": {}}
    try:
        with tempfile.TemporaryDirectory(prefix="taste-swe-agent-check-") as directory:
            report.update(asyncio.run(check(args.image, args.owner_token, Path(directory))))
        report["status"] = "passed" if report["checks"] and all(report["checks"].values()) else "failed"
    finally:
        report["removed"] = cleanup(args.owner_token)
        args.output.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, indent=1))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
