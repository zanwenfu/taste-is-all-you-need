#!/usr/bin/env python3
"""Kill a real process at every durable step of an effect, recover, and check what is on record.

    python3 scripts/kill_drill.py --path planner --out drill.jsonl
    python3 scripts/kill_drill.py --path planner --at 7 --keep /tmp/seven   # one point, kept to look at

The tests in tests/ raise an exception at chosen windows, inside one process.
This runs the same code in a child process that sends itself SIGKILL at the Nth
durable step: before and after each write, rename, fsync and git operation. No
cleanup runs, and nothing in memory survives. A second process then opens the
same store and runs the effect again, as a restart would, and a third reads what
is on record. The model is a stand-in that logs each call it receives and fsyncs
that log before it answers, so a call that reached the "provider" is counted
even when the caller dies before seeing the reply. Nothing here calls a model.

Checks after every kill: the store opens, git finds nothing wrong with it, no
paid call was made twice for one operation, and the operation has one outcome.

A killed process keeps what it handed the kernel, so this does not test what a
power cut does to writes still in the page cache.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

REPOSITORY = Path(__file__).resolve().parents[1]
PATHS = ("planner", "checkpoint", "terminal")
GIT_STEPS = ("commit_tree", "note_set", "cas_update_ref")


class Boundaries:
    """Counts durable steps and dies at the chosen one, before or after it."""

    def __init__(self, kill_at):
        self.kill_at = kill_at
        self.count = 0

    def mark(self):
        self.count += 1
        if self.count == self.kill_at:
            os.kill(os.getpid(), signal.SIGKILL)

    def around(self, function):
        def wrapped(*args, **kwargs):
            self.mark()
            result = function(*args, **kwargs)
            self.mark()
            return result
        return wrapped


def install(boundaries):
    from taste.memstore import backend

    os.fsync = boundaries.around(os.fsync)
    os.replace = boundaries.around(os.replace)
    for name in GIT_STEPS:
        setattr(backend.GitBackend, name, boundaries.around(getattr(backend.GitBackend, name)))
    from taste.brains.terminal_broker import TerminalBroker

    TerminalBroker._event = boundaries.around(TerminalBroker._event)
    TerminalBroker._execute_backend = boundaries.around(TerminalBroker._execute_backend)


class CountingModel:
    """A provider that logs every call it receives, durably, before it answers."""

    max_attempts = 1

    def __init__(self, log, respond):
        self.log, self.respond = Path(log), respond

    def ensure_ready(self, *models):
        return None

    def call(self, **kwargs):
        from taste.providers.base import Usage

        prompt = kwargs["messages"][0]["content"]
        with self.log.open("a") as written:
            written.write("call\n")
            written.flush()
            os.fsync(written.fileno())
        return SimpleNamespace(model=kwargs["model"], provider="anthropic",
                               usage=Usage(input_tokens=120, output_tokens=30),
                               text_blocks=(self.respond(prompt),), tool_calls=(), stop_reason="end_turn")


def planner_effect(root):
    from taste.brains.central_planner import (
        PLANNER_SYSTEM,
        CentralPlanner,
        Goal,
        PlannerTransportError,
        RejectedPlannerOperation,
        StalePlanningWorld,
    )
    from taste.brains.planner_transport import PLANNER_RECEIPT_BRANCH, LLMPlannerTransport
    from taste.memstore import Store
    from tests.test_brains_central_planner import one_assignment_response

    store = Store.open(root / "repo", "drill")
    integration = store.branch("integration", producer="central-integration")
    integration.close()
    lock = threading.RLock()
    control = store.branch("central-control", producer="drill")
    model = CountingModel(root / "calls.log", lambda prompt: one_assignment_response("drill", PLANNER_SYSTEM, prompt))
    transport = LLMPlannerTransport(model, store=store, control=control,
                                    journal=store.branch(PLANNER_RECEIPT_BRANCH, producer="drill"), mutation_lock=lock)
    central = CentralPlanner(store, transport=transport, control=control, mutation_lock=lock)
    goal = Goal(goal_id="ship-parser", task="Build and validate a parser",
                success_criteria=("parser.py exists", "the parser tests pass"), budget_usd=12.5)
    outcome, stale = None, 0
    # The runtime answers a stale world by asking again on its next cycle.
    for _ in range(4):
        try:
            plan = central.plan(goal)
        except StalePlanningWorld:
            stale += 1
        except (PlannerTransportError, RejectedPlannerOperation) as stopped:
            outcome = {"outcome": "refused_to_repeat", "why": type(stopped).__name__}
            break
        else:
            outcome = {"outcome": "planned", "generation": plan.generation}
            break
    outcome = outcome or {"outcome": "stale_every_time"}
    outcome["stale_answers"] = stale
    attempts = central.planner_attempts(goal.goal_id)
    outcome["attempts"] = [{"status": item.status, "cost_known": item.telemetry.cost_known} for item in attempts]
    store.close()
    return outcome


def checkpoint_effect(root):
    from taste.memstore import Store

    store = Store.open(root / "repo", "drill")
    branch = store.branch("worker", producer="drill")
    head = branch.head
    branch.write("result.txt", "the work")
    if head is None or head.read("result.txt") != "the work":
        branch.checkpoint("the work is done")
    saved = branch.head.read("result.txt")
    store.close()
    return {"outcome": "saved", "read_back": saved}


def terminal_effect(root):
    from taste.brains.terminal_broker import (
        TerminalBinding,
        TerminalBroker,
        TerminalConflict,
        TerminalFenced,
        TerminalRequest,
        TerminalResult,
    )

    class Container:
        """A task container whose command leaves one effect, logged durably before it answers."""

        environment_id = "container_instance_123"

        def execute(self, request):
            with (root / "calls.log").open("a") as written:
                written.write("command\n")
                written.flush()
                os.fsync(written.fileno())
            return TerminalResult(0, b"done\n", b"")

        def stop_and_confirm(self):
            return self.environment_id

    deadline = root / "deadline"
    if not deadline.exists():
        deadline.write_text(str(time.time() + 3600))
    binding = TerminalBinding("trial_1", Container.environment_id, float(deadline.read_text()), 100)
    directory = root / "terminal"
    fresh = not (directory / "terminal.sqlite3").exists()
    if fresh and directory.exists():
        shutil.rmtree(directory)  # killed before the ledger existed: nothing was admitted
    open_broker = TerminalBroker.create if fresh else TerminalBroker.open
    try:
        owner = open_broker(directory, binding, Container())
    except (TerminalConflict, sqlite3.OperationalError) as refused:
        # A ledger whose creation was cut short admitted no command. The trial
        # owner never reopens one: it ends the trial, and so does the refusal.
        return {"outcome": "refused_to_repeat", "why": f"unfinished ledger: {type(refused).__name__}"}

    async def scenario():
        request = TerminalRequest("effect_1", "worker_A", "write persistent effect", "/task", 3)
        try:
            result = await owner.execute(request)
        except TerminalFenced as stopped:
            # The owner confirms the container is stopped; only then may it close.
            await owner.abort()
            return {"outcome": "refused_to_repeat", "why": str(stopped)[:80], "phase": owner.phase}
        return {"outcome": "completed", "exit": result.return_code, "phase": owner.phase}

    try:
        return asyncio.run(scenario())
    finally:
        owner.close()


EFFECTS = {"planner": planner_effect, "checkpoint": checkpoint_effect, "terminal": terminal_effect}


def child(path, root, kill_at):
    if kill_at:
        install(Boundaries(kill_at))
    print(json.dumps(EFFECTS[path](Path(root))), flush=True)


def spawn(path, root, kill_at):
    environment = {**os.environ, "PYTHONPATH": str(REPOSITORY), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, __file__, "--child", path, "--root", str(root), "--at", str(kill_at)],
                          capture_output=True, text=True, env=environment, timeout=120, check=False)


def fsck(root):
    repository = root / "repo"
    if not repository.exists():
        return "no store yet"
    found = next((path.parent for path in repository.rglob("HEAD") if path.parent.name.endswith(".git")
                  or (path.parent / "objects").is_dir()), None)
    if found is None:
        return "no repository yet"
    result = subprocess.run(["git", "-C", str(found), "fsck", "--no-dangling", "--no-progress"],
                            capture_output=True, text=True, check=False)
    return "clean" if result.returncode == 0 else result.stderr[-300:]


def one_point(path, kill_at, keep=None):
    root = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="drill-"))
    root.mkdir(parents=True, exist_ok=True)
    try:
        killed = spawn(path, root, kill_at)
        if killed.returncode == 0:
            return {"at": kill_at, "reached": False}
        record = {"at": kill_at, "reached": killed.returncode == -signal.SIGKILL,
                  "kill_exit": killed.returncode}
        recovered = spawn(path, root, 0)
        calls = root / "calls.log"
        record.update(
            recovery_exit=recovered.returncode,
            recovery=json.loads(recovered.stdout.splitlines()[-1]) if recovered.returncode == 0 else
            recovered.stderr[-400:],
            calls=len(calls.read_text().splitlines()) if calls.exists() else 0, git=fsck(root))
        return record
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)


def verdict(record):
    """What went wrong at this point, or an empty list."""
    wrong = []
    if not record["reached"]:
        return ["the child did not die by the signal"]
    if record["recovery_exit"] != 0:
        wrong.append("recovery failed")
    if isinstance(record.get("recovery"), dict) and record["recovery"]["outcome"] == "stale_every_time":
        wrong.append("the goal stayed stuck after the restart")
    if record["calls"] > 1:
        wrong.append("a paid call was made twice")
    if record["git"] not in ("clean", "no store yet", "no repository yet"):
        wrong.append("git found damage")
    return wrong


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", choices=PATHS, default="planner")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--at", type=int)
    parser.add_argument("--keep")
    parser.add_argument("--child", choices=PATHS)
    parser.add_argument("--root")
    parser.add_argument("--limit", type=int, default=2000)
    arguments = parser.parse_args(argv)
    if arguments.child:
        child(arguments.child, arguments.root, arguments.at)
        return 0
    if arguments.at:
        print(json.dumps(one_point(arguments.path, arguments.at, arguments.keep), indent=1))
        return 0
    total = wrong = 0
    outcomes = {}
    with (arguments.out.open("w") if arguments.out else open(os.devnull, "w")) as written:
        for kill_at in range(1, arguments.limit + 1):
            record = one_point(arguments.path, kill_at)
            if not record["reached"] and "kill_exit" not in record:
                break
            total += 1
            problems = verdict(record)
            record["wrong"] = problems
            wrong += bool(problems)
            label = record["recovery"]["outcome"] if isinstance(record.get("recovery"), dict) else "no recovery"
            outcomes[label] = outcomes.get(label, 0) + 1
            written.write(json.dumps(record) + "\n")
            written.flush()
            if problems:
                print(kill_at, problems, str(record.get("recovery"))[:200], flush=True)
    print(json.dumps({"path": arguments.path, "kill_points": total, "wrong": wrong, "outcomes": outcomes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
