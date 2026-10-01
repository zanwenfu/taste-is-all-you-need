#!/usr/bin/env python3
"""Paid, bounded smoke of the complete Azure goal on one disposable container.

Server-only root script: real planner, worker and monitor calls against a tiny
synthetic repository task, with no Harbor and no benchmark data. The secrets
file holds AZURE_OPENAI_BASE_URL and AZURE_OPENAI_API_KEY and is never echoed.
Run inside a bounded systemd unit with --cleanup-only as ExecStopPost for the
same fresh --owner-token. This measures whether the loop works with real
models; it is not a benchmark score.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pwd
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_docker_terminal import cleanup, cli
from taste.benchmarks.azure_terminal_trial import AzureTerminalTrial, cleanup_trial
from taste.brains import benchmark_reply
from taste.brains.azure_execution_policy import AzureExecutionPolicy
from taste.brains.central_planner import Goal
from taste.brains.docker_terminal import OWNER_LABEL, DockerTerminalBackend
from taste.brains.terminal_broker import TerminalBinding
from taste.brains.terminal_worker_policy import TerminalWorkerPolicy
from taste.pricing import table_sha

WORKDIR = "/workspace/calc"
CHECK = f"cd {WORKDIR} && python3 -m unittest discover -s tests -v"
FILES = {
    "calc/__init__.py": "",
    "calc/stats.py": (
        '"""Small statistics helpers."""\n\n\n'
        "def mean(values):\n    return sum(values) / len(values)\n\n\n"
        "def median(values):\n"
        "    ordered = sorted(values)\n"
        "    middle = len(ordered) // 2\n"
        "    if len(ordered) % 2:\n"
        "        return ordered[middle]\n"
        "    return (ordered[middle] + ordered[middle + 1]) / 2\n\n\n"
        "def spread(values):\n    return max(values) - min(values)\n"
    ),
    "tests/__init__.py": "",
    "tests/test_stats.py": (
        "import unittest\n\nfrom calc.stats import mean, median, spread\n\n\n"
        "class StatsTest(unittest.TestCase):\n"
        "    def test_mean(self):\n        self.assertEqual(mean([1, 2, 3]), 2)\n\n"
        "    def test_median_odd(self):\n        self.assertEqual(median([3, 1, 2]), 2)\n\n"
        "    def test_median_even(self):\n        self.assertEqual(median([4, 1, 3, 2]), 2.5)\n\n"
        "    def test_spread(self):\n        self.assertEqual(spread([5, 1, 9]), 8)\n\n\n"
        'if __name__ == "__main__":\n    unittest.main()\n'
    ),
    "README.md": "# calc\n\nRun the tests with `python3 -m unittest discover -s tests -v`.\n",
}
INSTRUCTION = (
    f"You are continuing work in the repository at {WORKDIR}. The developer reports that "
    "`python3 -m unittest discover -s tests -v` fails. Find the cause, fix it without editing "
    "the tests, and confirm the suite passes. When you have finished, reply to the developer "
    "in plain text: say what you changed, what you verified, and anything you did not check.\n"
)
# A second defect whose test never returns: running the suite as asked hangs,
# so the command has to be ended without losing the container.
HANG_FILES = {
    "calc/retry.py": (
        '"""Retry a flaky operation a bounded number of times."""\n\n\n'
        "def retry(operation, attempts=3):\n"
        "    failures = 0\n"
        "    while failures < attempts:\n"
        "        try:\n"
        "            return operation()\n"
        "        except ValueError:\n"
        "            pass\n"
        '    raise RuntimeError("operation kept failing")\n'
    ),
    "tests/test_retry.py": (
        "import unittest\n\nfrom calc.retry import retry\n\n\n"
        "class RetryTest(unittest.TestCase):\n"
        "    def test_returns_first_success(self):\n"
        "        self.assertEqual(retry(lambda: 7), 7)\n\n"
        "    def test_gives_up(self):\n"
        "        def broken():\n            raise ValueError('no')\n\n"
        "        with self.assertRaises(RuntimeError):\n            retry(broken)\n\n\n"
        'if __name__ == "__main__":\n    unittest.main()\n'
    ),
}
HANG_INSTRUCTION = (
    f"You are continuing work in the repository at {WORKDIR}. The developer says the test run "
    "`python3 -m unittest discover -s tests -v` never finishes on their machine and at least one "
    "test fails. Find out why, fix the code without editing the tests, and confirm the whole "
    "suite passes. When you have finished, reply to the developer in plain text: say what you "
    "changed, what you verified, and anything you did not check.\n"
)


def owner_directory(token):
    if re.fullmatch(r"[0-9a-f]{32}", token) is None:
        raise ValueError("a fresh lowercase 32-hex owner token is required")
    return Path("/var/tmp") / f"ta-{token}"


def secrets(path):
    info = os.stat(path)
    if info.st_uid != 0 or info.st_mode & 0o077:
        raise ValueError("the secrets file must be private to root")
    values = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            name, value = line.split("=", 1)
            values[name.strip()] = value.strip().strip('"').strip("'")
    endpoint, key = values.get("AZURE_OPENAI_BASE_URL", ""), values.get("AZURE_OPENAI_API_KEY", "")
    if not endpoint or not key:
        raise ValueError("the secrets file lacks the Azure endpoint or key")
    return endpoint.rstrip("/") + "/", key


def scrub(text, *private):
    for value in private:
        if value:
            text = text.replace(value, "<redacted>")
    return text


def cleanup_all(token):
    root, errors, result = owner_directory(token), [], {}
    if (root / "controller/trial.json").exists():
        try:
            result["trial"] = cleanup_trial(root)
        except BaseException as exc:
            errors.append(type(exc).__name__)
    try:
        result["removed_containers"] = cleanup(token)
    except BaseException as exc:
        errors.append(type(exc).__name__)
    result["errors"] = errors
    return result


def seed(container, files):
    for name, body in files.items():
        target = f"{WORKDIR}/{name}"
        cli("exec", container, "/bin/sh", "-c", f"mkdir -p \"$(dirname '{target}')\"")
        done = subprocess.run(
            ["docker", "--host", "unix:///var/run/docker.sock", "exec", "-i", container,
             "/bin/sh", "-c", f"cat > '{target}'"], input=body.encode(), capture_output=True, timeout=15)
        assert done.returncode == 0, done.stderr.decode()[:300]


def suite(container):
    # Bounded inside the container: a seeded defect can make the suite hang.
    done = subprocess.run(["docker", "--host", "unix:///var/run/docker.sock", "exec", container,
                           "/bin/sh", "-c", f"timeout -s KILL 20 /bin/sh -c '{CHECK}'"],
                          capture_output=True, timeout=40)
    return done.returncode, (done.stdout + done.stderr).decode(errors="replace")[-1500:]


async def run(args):
    assert os.geteuid() == 0
    endpoint, api_key = secrets(args.secrets)
    account = pwd.getpwnam(args.worker_user)
    token, root = args.owner_token, owner_directory(args.owner_token)
    out = args.output
    out.mkdir(mode=0o700, parents=True)
    assert not cli("ps", "-aq", "--filter", f"label={OWNER_LABEL}={token}").stdout.strip()
    container = cli("create", "--pull", "never", "--network", "none", "--memory", "1g", "--cpus", "1",
        "--pids-limit", "256", "--restart", "no", "--label", f"{OWNER_LABEL}={token}",
        "--entrypoint", "/bin/sh", args.image, "-c", "sleep 7200").stdout.decode().strip()
    cli("start", container)
    hang = args.scenario == "hang"
    seed(container, {**FILES, **(HANG_FILES if hang else {})})
    before = suite(container)
    assert before[0] != 0, "the seeded suite must fail before the agent works"
    seconds = args.minutes * 60
    deadline = time.time() + seconds
    backend = DockerTerminalBackend.admit("/var/run/docker.sock", container, token, deadline + 120,
                                          output_limit=65536)
    binding = TerminalBinding(token, backend.environment_id, deadline, 300)
    policy = AzureExecutionPolicy(endpoint=endpoint, planner_deployment=args.planner_deployment,
        worker_deployment=args.worker_deployment, deadline_unix=deadline,
        worker_budget_usd=args.worker_budget, monitor_budget_usd=args.monitor_budget,
        worker_max_calls=args.worker_calls, monitor_max_calls=args.worker_calls,
        worker_max_output_tokens=8192, monitor_max_output_tokens=2048, planner_max_output_tokens=8192,
        monitor_batch_size=8, pricing_sha=table_sha(),
        terminal=TerminalWorkerPolicy(binding, args.command_seconds, WORKDIR))
    goal = Goal(goal_id="smoke-" + token[:12], task=HANG_INSTRUCTION if hang else INSTRUCTION,
        success_criteria=("the unit test suite passes without edits to the tests",),
        budget_usd=args.goal_budget, metadata={benchmark_reply.KEY: benchmark_reply.SCHEMA,
                                               benchmark_reply.RESERVE_KEY: args.reserve})
    report = {"status": "failed", "paid": True, "token": token, "before_exit": before[0]}
    owner, started = None, time.time()
    try:
        owner = AzureTerminalTrial.create(root, backend, goal, policy, service_uid=account.pw_uid,
            python_executable=args.worker_python, max_generations=args.generations,
            wall_clock_seconds=seconds, max_planner_failures=3)
        try:
            outcome = await owner.run(api_key=api_key)
            report["run_error"] = None
        except BaseException as exc:
            outcome = owner.outcome
            leaves = exc.exceptions if isinstance(exc, BaseExceptionGroup) else (exc,)
            report["run_error"] = [type(item).__name__ + ": " + scrub(str(item)[:300], api_key, endpoint)
                                   for item in leaves]
        report["elapsed_seconds"] = round(time.time() - started, 1)
        report["audit_flags"], report["sealed"] = list(owner.audit_flags), owner.sealed
        if outcome is not None:
            report["outcome"] = outcome.to_dict()
        trajectory = root / "controller/trajectory.json"
        if trajectory.exists():
            (out / "trajectory.json").write_bytes(trajectory.read_bytes())
            trace = json.loads(trajectory.read_bytes())
            report["gaps"] = trace["extra"].get("gaps")
            report["final_reply"] = next((step["message"] for step in reversed(trace["steps"])
                                          if step["source"] == "agent"), None)
            report["closing_reply"] = trace["extra"].get("closing_reply")
        ledger = root / "controller/terminal/terminal.sqlite3"
        if ledger.exists():
            with sqlite3.connect(ledger.as_uri() + "?mode=ro", uri=True) as db:
                report["terminal_phase"] = db.execute("SELECT phase FROM state").fetchone()[0]
                report["commands"] = [{"status": status, "code": code, "terminated": terminated,
                                       "command": json.loads(payload)["command"][:400]}
                                      for payload, status, code, terminated in db.execute(
                                          "SELECT payload,status,code,terminated FROM requests ORDER BY rowid")]
        # A benchmark runner verifies in the live container and removes it
        # itself, so this owner releases without stopping it.
        if owner.sealed:
            await owner.release()
        running = json.loads(cli("inspect", container).stdout)[0]["State"]["Running"]
        report["container_running_after_goal"] = running
        if running:
            report["after_exit"], report["after_tail"] = suite(container)
        report["status"] = "completed"
    finally:
        if owner is not None and not owner.closed:
            try:
                await owner.close()
            except BaseException as exc:
                report["close_error"] = type(exc).__name__
        report["cleanup"] = cleanup_all(token)
        (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {name: report.get(name) for name in ("status", "run_error", "elapsed_seconds", "gaps",
               "audit_flags", "sealed", "closing_reply", "terminal_phase", "after_exit",
               "container_running_after_goal")}
    summary["has_final_reply"] = bool(report.get("final_reply"))
    if "outcome" in report:
        summary.update(stop_reason=report["outcome"]["stop_reason"], complete=report["outcome"]["complete"],
                       generations=report["outcome"]["generations"],
                       spent_usd=report["outcome"]["budget"]["known_spent_usd"])
    summary["commands"] = len(report.get("commands", []))
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--image")
    parser.add_argument("--secrets", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-user", default="bugbash")
    parser.add_argument("--worker-python")
    parser.add_argument("--planner-deployment", default="gpt-6-astra")
    parser.add_argument("--worker-deployment", default="gpt-6-sol")
    parser.add_argument("--scenario", choices=("repair", "hang"), default="repair")
    parser.add_argument("--minutes", type=float, default=12)
    parser.add_argument("--reserve", type=float, default=120,
                        help="seconds held back for the coordinator's closing reply")
    parser.add_argument("--command-seconds", type=float, default=120,
                        help="longest allowance one terminal command may ask for")
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--worker-calls", type=int, default=40)
    parser.add_argument("--worker-budget", type=float, default=20.0)
    parser.add_argument("--monitor-budget", type=float, default=10.0)
    parser.add_argument("--goal-budget", type=float, default=150.0)
    parser.add_argument("--cleanup-only", action="store_true")
    args = parser.parse_args()
    owner_directory(args.owner_token)
    if args.cleanup_only:
        print(json.dumps(cleanup_all(args.owner_token)))
        return
    if (not args.image or re.fullmatch(r"sha256:[0-9a-f]{64}", args.image) is None
            or not args.secrets or not args.output or not args.worker_python):
        parser.error("execution requires a cached image ID, secrets, output and worker python")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
