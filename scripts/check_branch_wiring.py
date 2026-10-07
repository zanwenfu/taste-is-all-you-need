#!/usr/bin/env python3
"""A recorded run rebuilt at step k in a real container, its checkpoint saved, and restored into another.

    sudo python3 scripts/check_branch_wiring.py --image sha256:<id> --owner-token <hex> --output out.json
    sudo python3 scripts/check_branch_wiring.py --owner-token <hex> --cleanup-only

What a branch trial wires together, without a model: disposable containers
of one cached image (with /bin/sh, no network), each with the controller's
terminal broker. A small run is recorded in the first: commands in the
agent's shell form that make, change and delete files (some in the image),
print timings, fail with an exit code, and submit; the replay script is
written from what they printed. A fresh container rebuilds steps 1..k from the
script (taste.brains.branch_replay.rebuild, the worker's procedure) and must
match the record by the fidelity rule and the first container's files at k;
its files are saved as a branch checkpoint (taste.benchmarks.branch_trial),
as a rebuild with live off saves them; a third container, never touched by
a command, gets them back by restore and must match too. A fourth rebuilds
from a record with one exit code changed and must be found unfaithful at that
command. The containers are removed afterwards, and --cleanup-only removes
anything left with the same owner token.
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

from taste.benchmarks import branch_trial
from taste.benchmarks.harbor_settings import TrialSettings
from taste.benchmarks.replay_export import encode_script
from taste.brains.branch_replay import SCRIPT_SCHEMA, ReplayScript, digest, rebuild
from taste.brains.docker_terminal import OWNER_LABEL, DockerTerminalBackend
from taste.brains.terminal_broker import TerminalBinding, TerminalBroker, TerminalRequest

FINGERPRINT = ("cd / && find taste-branch -print0 2>/dev/null | sort -z | xargs -0 -r ls -ld --time-style=+ | "
               "awk '{print $1, $NF}'; find taste-branch -type f -print0 2>/dev/null | sort -z | xargs -0 -r sha256sum;"
               " for f in etc/issue etc/issue.net etc/debian_version; do"
               " if [ -e $f ]; then sha256sum $f; else echo missing $f; fi; done")
RUN = [
    "mkdir -p /taste-branch/app && printf 'def answer():\\n    return 41\\n' > /taste-branch/app/server.py"
    " && echo created at $(date +%s%N)",
    "sed -i 's/41/42/' /taste-branch/app/server.py && cat /taste-branch/app/server.py",
    "rm /etc/issue.net && printf 'changed\\n' >> /etc/issue && mkdir -p /taste-branch/build"
    " && printf obj > /taste-branch/build/out.o && echo built in $(( $(date +%s%N) % 1000 ))ns",
    "grep -c 43 /taste-branch/app/server.py; exit 3",
    "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
]
STEP = 4  # after the failing command, before the submission


def cli(*args, check=True):
    # Fixture control only; never used to run task output.
    return subprocess.run(["docker", "--host", "unix:///var/run/docker.sock", *args],
                          capture_output=True, check=check, timeout=60)


def cleanup(token):
    listed = cli("ps", "-aq", "--filter", f"label={OWNER_LABEL}={token}").stdout.decode().split()
    for container in listed:
        cli("rm", "-f", container, check=False)
    return listed


def container(image, token):
    made = cli("create", "--pull", "never", "--network", "none", "--memory", "256m",
               "--label", f"{OWNER_LABEL}={token}", "--entrypoint", "/bin/sh", image, "-c", "sleep 900")
    identifier = made.stdout.decode().strip()
    cli("start", identifier)
    return identifier


def shell(command):
    """The command as mini-swe-agent's environment sends it: one group, standard error folded in."""
    return "{\n" + command + "\n} 2>&1"


class Terminal:
    """One container's backend and broker, as a trial's controller holds them."""

    def __init__(self, identifier, token, directory, name):
        deadline = time.time() + 900
        self.backend = DockerTerminalBackend.admit("/var/run/docker.sock", identifier, token, deadline)
        binding = TerminalBinding("branch_check_" + name, self.backend.environment_id, deadline, 100)
        self.broker = TerminalBroker.create(directory / ("ledger-" + name), binding, self.backend)
        self.count = 0

    async def run(self, text, cwd="/", timeout=60.0):
        self.count += 1
        result = await self.broker.execute(TerminalRequest(f"command_{self.count}", "check_agent", text, cwd, timeout))
        output = result.stdout.decode("utf-8", errors="replace") + result.stderr.decode("utf-8", errors="replace")
        return output, result.return_code, result.terminated == "timeout"

    async def execute(self, _number, run):
        return await self.run(run.executed, run.cwd, run.timeout_seconds)

    def close(self):
        self.broker.close()


def script_of(results):
    """A replay script of the recorded commands; the replies are placeholders, as no model ran."""
    steps = []
    for number, (command, (output, code, timed_out)) in enumerate(zip(RUN, results, strict=True), 1):
        call = {"id": f"call_{number}", "name": "bash", "arguments": {"command": command}}
        steps.append({"step": number, "request_sha": digest(["branch-check", number]),
                      "messages": {"base": 0, "added": []}, "text": [], "calls": [call],
                      "reply": {"output": [{"type": "function_call", "call_id": call["id"], "name": "bash",
                                            "arguments": json.dumps(call["arguments"])}],
                                "status": "completed", "incomplete_reason": None, "usage": {},
                                "model": "none", "cost_usd": 0.0},
                      "runs": [{"command": command, "executed": shell(command), "cwd": "/", "timeout_seconds": 60.0,
                                "output": output, "returncode": code, "timed_out": timed_out,
                                "output_exact": True}]})
    return ReplayScript.from_dict({"schema": SCRIPT_SCHEMA, "source": {"trial": "branch-check"},
                                   "task": "Make answer() return 42.", "steps": steps, "exit": None})


async def check(image, token, directory):
    terminals = []
    try:
        recorded = Terminal(container(image, token), token, directory, "recorded")
        terminals.append(recorded)
        results, at_step = [], None
        for number, command in enumerate(RUN, 1):
            results.append(await recorded.run(shell(command)))
            if number == STEP:
                at_step = (await recorded.run(FINGERPRINT))[0]
        script = script_of(results)
        path = directory / "script.json"
        path.write_bytes(encode_script(script.to_dict()))
        saved_to = directory / "checkpoint"
        options = {"model": "gpt-5.6-luna", "agent": "mini-swe-agent", "services": "none",
                   "branch": str(path), "branch_step": str(STEP), "branch_checkpoint": str(saved_to)}
        rebuilding = TrialSettings.from_options({**options, "branch_live": "off"})
        inputs = branch_trial.load_inputs(rebuilding)

        rebuilt = Terminal(container(image, token), token, directory, "rebuilt")
        terminals.append(rebuilt)
        account = await rebuild(inputs.script, STEP, rebuilt.execute)
        rebuilt_files = (await rebuilt.run(FINGERPRINT))[0]
        saved = await asyncio.to_thread(branch_trial.save_checkpoint, rebuilt.backend, rebuilding, inputs,
                                        {"faithful": account["faithful"]}, "branch-check")

        restored = Terminal(container(image, token), token, directory, "restored")
        terminals.append(restored)
        restoring = TrialSettings.from_options({**options, "branch_mode": "restore"})
        receipt = await asyncio.to_thread(branch_trial.restore_checkpoint, restored.backend, restoring, inputs)
        restored_files = (await restored.run(FINGERPRINT))[0]

        changed = script.to_dict()
        changed["steps"][1]["runs"][0]["returncode"] = 1
        divergent = Terminal(container(image, token), token, directory, "divergent")
        terminals.append(divergent)
        refused = await rebuild(ReplayScript.from_dict(changed), STEP, divergent.execute)

        checks = {
            "the recorded run failed where it should and submitted": (
                results[3][1] == 3 and script.steps[-1].submission and not script.steps[STEP - 1].submission),
            "the rebuild matches the record": account["faithful"] and account["divergent"] == 0
            and account["commands"] == STEP,
            "timings in outputs do not count": min(row["similarity"] for row in account["rows"]) == 1.0,
            "rebuilt files are those of step k": rebuilt_files == at_step,
            "the checkpoint was saved": saved.get("saved") is True and saved.get("deleted", 0) >= 1,
            "restored into another container exactly": receipt.get("exact") is True,
            "restored files are those of step k": restored_files == at_step,
            "a changed exit code is unfaithful at its command": (
                not refused["faithful"] and refused["commands"] == 2 and refused["rows"][-1]["divergent"]),
        }
        return {"checks": checks, "rebuild": account, "checkpoint": saved, "restore": receipt,
                "divergent": refused, "fingerprint": at_step}
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
    if not args.image or re.fullmatch(r"sha256:[0-9a-f]{64}", args.image) is None or args.output is None:
        parser.error("an immutable cached image ID and an output path are required")
    report = {"status": "failed", "checks": {}}
    try:
        with tempfile.TemporaryDirectory(prefix="taste-branch-check-") as directory:
            report.update(asyncio.run(check(args.image, args.owner_token, Path(directory))))
        report["status"] = "passed" if report["checks"] and all(report["checks"].values()) else "failed"
    finally:
        report["removed"] = cleanup(args.owner_token)
        args.output.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, indent=1))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
