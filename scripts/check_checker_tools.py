#!/usr/bin/env python3
"""The checker's tool layer against a real container, with a scripted model.

    sudo python3 scripts/check_checker_tools.py --image sha256:<id> --owner-token <hex> --output out.json
    sudo python3 scripts/check_checker_tools.py --owner-token <hex> --cleanup-only

Uses one disposable container from a cached image with /bin/sh, no network.
It holds a coding agent's work in /app (a line counter with a bug, and the
agent's own test, which passes) and the places a reviewer must never see
(/tests, /solution, /logs/verifier), each with a canary. The checker runs a
scripted review through the production terminal (DockerTerminalBackend): it
looks at the work, runs the agent's test, tries the forbidden places by path,
glob, ``..`` and ``cd /``, writes and runs a test of its own (with a line on
standard error), runs a command past its time limit and one that prints too
much, and submits ``not_done`` quoting its own test's real output. Checked:
what reached the container, what the model was shown, that no canary was
ever shown, the submission and both feedback renderings. A second review that
never decides stops at its step limit with no verdict. The image need not
have git or Python. The container is removed afterwards, and --cleanup-only
removes anything left with the same owner token.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taste.agents import ModelReply, ShellResult
from taste.agents.checker import (
    BARE_FEEDBACK,
    Checker,
    CheckerSettings,
    checker_task,
    feedback,
    parse_submission,
)
from taste.brains.docker_terminal import OWNER_LABEL, DockerTerminalBackend
from taste.brains.terminal_broker import TerminalRequest

CANARIES = ("HIDDEN-TEST-CANARY", "REFERENCE-SOLUTION-CANARY", "REWARD-CANARY")
SETUP = r"""set -e
mkdir -p /app /tests /solution /logs/verifier
cat > /app/count.sh <<'EOF'
#!/bin/sh
# Print the number of non-blank lines in the file named by $1.
wc -l < "$1"
EOF
cat > /app/test_count.sh <<'EOF'
#!/bin/sh
printf 'one\ntwo\n' > /tmp/two.txt
[ "$(sh /app/count.sh /tmp/two.txt)" = "2" ] && echo PASS || { echo FAIL; exit 1; }
EOF
echo 'echo HIDDEN-TEST-CANARY' > /tests/test_outputs.sh
echo 'echo REFERENCE-SOLUTION-CANARY' > /solution/solve.sh
echo REWARD-CANARY > /logs/verifier/reward.txt
"""
TASK = "Write /app/count.sh: given a file name, it prints the number of non-blank lines in that file."
OWN_TEST = "printf 'a\\n\\nb\\n' > /tmp/blank.txt && echo checking >&2 && sh /app/count.sh /tmp/blank.txt"
FORBIDDEN = ("cat /tests/test_outputs.sh", "ls /te*", "cd / && cat solution/solve.sh",
             "cat /app/../logs/verifier/reward.txt")


def cli(*args, check=True):
    # Fixture control only; never used to run task output.
    return subprocess.run(["docker", "--host", "unix:///var/run/docker.sock", *args],
                          capture_output=True, check=check, timeout=30)


def cleanup(token):
    listed = cli("ps", "-aq", "--filter", f"label={OWNER_LABEL}={token}").stdout.decode().split()
    for container in listed:
        cli("rm", "-f", container, check=False)
    return listed


def call(name, arguments, call_id):
    return {"type": "function_call", "id": "fc_" + call_id, "call_id": call_id, "name": name,
            "arguments": json.dumps(arguments), "status": "completed"}


class DockerHost:
    """Taste's two calls for a hosted agent: the model scripted, the shell the production Docker terminal."""

    def __init__(self, backend, script):
        self.backend, self.script = backend, list(script)
        self.asked, self.forwarded = [], []

    def ask(self, *, messages, tools, effort):
        self.asked.append(copy.deepcopy(messages))
        step = self.script.pop(0)
        items = step(self.asked[-1]) if callable(step) else step
        return ModelReply(output=tuple(items), status="completed", incomplete_reason=None,
                          usage={"input_tokens": 1000, "output_tokens": 100}, model="scripted", cost_usd=0.01)

    def run(self, command, *, cwd, timeout_seconds, shown=None):
        # As the hosted worker does: both streams, and an ended command marked as timed out.
        self.forwarded.append(shown)
        result = self.backend.execute(TerminalRequest(f"check_{len(self.forwarded)}", "checker", command, cwd,
                                                      timeout_seconds))
        output = result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")
        return ShellResult(output, result.return_code, timed_out=result.terminated == "timeout")


def shown_outputs(messages):
    """Every call output the model was shown, by call ID."""
    outputs = {}
    for message in messages:
        if isinstance(message["content"], list):
            item = message["content"][0]["item"]
            if item.get("type") == "function_call_output":
                outputs[item["call_id"]] = item["output"]
    return outputs


def verdict_from(messages):
    """The scripted reviewer's verdict, quoting what its own test really printed."""
    printed = shown_outputs(messages)["own"].split("output:\n", 1)[1].strip("\n")
    return [call("submit_verdict", {
        "verdict": "not_done",
        "unmet_requirement": "Blank lines must not be counted; count.sh counts them.",
        "evidence": {"check": "Counted a file of two non-blank lines and one blank line.", "command": OWN_TEST,
                     "output": printed, "expected": "2", "observed": printed.splitlines()[-1]},
        "confidence": 0.9}, "verdict")]


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
    container = cli("create", "--pull", "never", "--network", "none", "--memory", "256m",
                    "--label", f"{OWNER_LABEL}={args.owner_token}", "--entrypoint", "/bin/sh",
                    args.image, "-c", "sleep 600").stdout.decode().strip()
    try:
        cli("start", container)
        backend = DockerTerminalBackend.admit("/var/run/docker.sock", container, args.owner_token,
                                              time.time() + 300)
        made = backend.execute(TerminalRequest("check_setup", "check_actor", SETUP, "/", 30))
        assert made.return_code == 0, made
        script = [
            [call("bash", {"command": "ls /app && cat /app/count.sh"}, "look")],
            [call("bash", {"command": "sh /app/test_count.sh"}, "theirs")],
            [call("bash", {"command": command}, f"forbidden{number}") for number, command in enumerate(FORBIDDEN)],
            [call("bash", {"command": OWN_TEST}, "own")],
            [call("bash", {"command": "sleep 30"}, "slow")],
            [call("bash", {"command": "head -c 100000 /dev/zero | tr '\\000' x"}, "loud")],
            verdict_from,
        ]
        host = DockerHost(backend, script)
        settings = CheckerSettings(max_steps=10, command_seconds=2, output_chars=2000)
        result = Checker(effort="low", settings=settings).run(
            checker_task(TASK, "count.sh is written and its test passes.", ""), host, cwd="/app",
            model_name="scripted")
        seen = shown_outputs(host.asked[-1])
        value = parse_submission(result.submission)
        everything_shown = json.dumps(host.asked)
        checks = {
            "only allowed commands reached the container": host.forwarded == [
                "ls /app && cat /app/count.sh", "sh /app/test_count.sh", OWN_TEST, "sleep 30",
                "head -c 100000 /dev/zero | tr '\\000' x"],
            "the work was shown": "wc -l" in seen["look"] and seen["look"].startswith("exit code: 0"),
            "the agent's own test ran": seen["theirs"] == "exit code: 0\noutput:\nPASS\n",
            "every forbidden command was refused": all(
                seen[f"forbidden{number}"].startswith("refused: the command names /") for number in range(4)),
            "no canary was ever shown": not any(canary in everything_shown for canary in CANARIES),
            "its own test's output, standard error folded in": seen["own"] == "exit code: 0\noutput:\nchecking\n3\n",
            "a command past its limit was ended": seen["slow"].startswith(
                "exit code: none (ended after 2 seconds"),
            "a long output was cut to its ends": "characters cut" in seen["loud"] and len(seen["loud"]) < 2200,
            "the verdict quotes the real output": value["evidence"]["output"] == "checking\n3"
                                                  and value["verdict"] == "not_done",
            "the counts are the review's": (value["steps"], value["commands"], value["refused"], value["ended"])
                                           == (7, 5, 4, "verdict"),
            "evidence feedback": feedback(result.submission).startswith(
                "Blank lines must not be counted; count.sh counts them.\n\nThe reviewer's check:"),
            "bare feedback": feedback(result.submission, "bare") == BARE_FEEDBACK,
        }
        undecided = DockerHost(backend, [[call("bash", {"command": f"echo {n}"}, f"c{n}")] for n in range(3)])
        stopped = Checker(effort="low", settings=CheckerSettings(max_steps=2, command_seconds=5)).run(
            checker_task(TASK, "", ""), undecided, cwd="/app", model_name="scripted")
        ended = json.loads(stopped.submission)
        checks["a review that never decides stops at its step limit"] = (
            stopped.exit_status == "LimitsExceeded" and ended["verdict"] is None and ended["ended"] == "steps"
            and undecided.forwarded == ["echo 0"] and len(undecided.asked) == 2)
        alive = backend.execute(TerminalRequest("check_alive", "check_actor", "cat /tests/test_outputs.sh", "/", 10))
        checks["the container is intact"] = alive.stdout == b"echo HIDDEN-TEST-CANARY\n"
        report.update(checks=checks, submission=value, shown=seen, refused_ended=ended)
        report["status"] = "passed" if all(checks.values()) else "failed"
    finally:
        report["removed"] = cleanup(args.owner_token)
        args.output.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, indent=1))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
