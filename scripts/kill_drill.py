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
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

REPOSITORY = Path(__file__).resolve().parents[1]
PATHS = ("planner", "planner-moved", "checkpoint", "terminal", "worker", "delivery", "record",
         "git-index", "git-ref", "git-notes")
# Killed from inside the git command, with its locks held, not at a step of ours.
GIT_PATHS = {"git-index": "", "git-ref": "refs/heads/", "git-notes": "refs/notes/"}
RECORD = "cycles/00000014/result.json"
DIE = '''die() {
    left=$(cat "$KILL_DRILL_COUNT" 2>/dev/null)
    [ -n "$left" ] || return 0
    if [ "$left" -le 1 ]; then
        rm -f "$KILL_DRILL_COUNT"
        kill -9 "$KILL_DRILL_PID" "$PPID"
        sleep 5
    else
        echo $((left - 1)) > "$KILL_DRILL_COUNT"
    fi
}
'''
DELIVERED = "NEW = 1\n"
KEPT = "base file\n"
BEDROCK_URL = "https://bedrock-runtime.us-east-1.amazonaws.com"
GIT_STEPS = ("commit_tree", "note_set", "cas_update_ref")
RECOVERING = False


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
    from taste.brains.responses_session import ResponsesSession

    ResponsesSession._save_receipt = boundaries.around(ResponsesSession._save_receipt)
    ResponsesSession._dispatch = boundaries.around(ResponsesSession._dispatch)


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


def live_model(root):
    """The model facade on the real Bedrock route, logging each call before it is sent."""
    from taste.llm import LLM
    from taste.providers.azure_openai import AZURE_LUNA6_MODEL, AzureDeployment, AzureOpenAIConfig

    class Counted(LLM):
        def call(self, **kwargs):
            with (root / "calls.log").open("a") as written:
                written.write("call\n")
                written.flush()
                os.fsync(written.fileno())
            return super().call(**kwargs)

    route = AzureOpenAIConfig(base_url=BEDROCK_URL, api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
                              deployments=(AzureDeployment(AZURE_LUNA6_MODEL, "global.openai.gpt-6-luna"),))
    return Counted(azure_openai=route, max_attempts=1)


def planner_effect(root, world_moves=False):
    """``world_moves``: as in a real run, the restarted runtime writes to control before it plans again."""
    from taste.brains.central_host import open_coordinator_branch
    from taste.brains.central_planner import (
        PLANNER_SYSTEM,
        CentralPlanner,
        Goal,
        InvalidPlannerOutput,
        PlannerTransportError,
        RejectedPlannerOperation,
        StalePlanningWorld,
    )
    from taste.brains.planner_transport import PLANNER_RECEIPT_BRANCH, LLMPlannerTransport
    from taste.memstore import Store
    from taste.providers.azure_openai import AZURE_LUNA6_MODEL
    from tests.test_brains_central_planner import one_assignment_response

    store = Store.open(root / "repo", "drill")
    integration = store.branch("integration", producer="central-integration")
    integration.close()
    lock = threading.RLock()
    control = open_coordinator_branch(store, "central-control", producer="drill")
    journal = open_coordinator_branch(store, PLANNER_RECEIPT_BRANCH, producer="drill")
    if os.environ.get("KILL_DRILL_LIVE"):
        # The real planner on the real model: recovery runs against a live reply.
        transport = LLMPlannerTransport(live_model(root), store=store, control=control, journal=journal,
                                        mutation_lock=lock, model=AZURE_LUNA6_MODEL, effort="medium")
    else:
        model = CountingModel(root / "calls.log",
                              lambda prompt: one_assignment_response("drill", PLANNER_SYSTEM, prompt))
        transport = LLMPlannerTransport(model, store=store, control=control, journal=journal, mutation_lock=lock)
    central = CentralPlanner(store, transport=transport, control=control, mutation_lock=lock)
    goal = Goal(goal_id="ship-parser", task="Build and validate a parser",
                success_criteria=("parser.py exists", "the parser tests pass"), budget_usd=12.5)
    if world_moves and RECOVERING and control.head is not None:
        with lock:
            control.write(".taste/central-runtime/after-restart.json", "{}\n")
            control.checkpoint("central decision after a restart")
    outcome, stale = None, 0
    # The runtime answers a stale world by asking again on its next cycle, as a new operation.
    for _ in range(4):
        try:
            plan = central.plan(goal, operation_id=f"epoch-{stale}" if stale else "initial")
        except StalePlanningWorld:
            stale += 1
        except InvalidPlannerOutput as refused:
            outcome = {"outcome": "reply_refused", "why": str(refused)[:120]}
            break
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


def worker_effect(root):
    """Two turns of a worker's model journal, on the Bedrock route, against a provider that logs each call."""
    import httpx

    from taste.brains.responses_session import ResponsesBinding, ResponsesFenced, ResponsesSession
    from taste.providers.azure_openai import AZURE_LUNA6_MODEL
    from tests.test_bedrock_provider import ENDPOINT, converse, route

    def provider(wire):
        turn = json.loads(wire.content)["messages"][-1]["content"][0]["text"]
        with (root / "calls.log").open("a") as written:
            written.write(turn + "\n")
            written.flush()
            os.fsync(written.fileno())
        return httpx.Response(200, json=converse({"text": "Done."}), headers={"x-amzn-requestid": "req-drill"})

    real = httpx.Client

    class Client(real):
        def __init__(self, **options):
            super().__init__(**options, transport=httpx.MockTransport(provider))

    httpx.Client = Client
    deadline = root / "deadline"
    if not deadline.exists():
        deadline.write_text(str(time.time() + 3600))
    limits = ResponsesBinding(run_id="worker-run.drill", model=AZURE_LUNA6_MODEL, endpoint=ENDPOINT + "/",
                              deployment="global.openai.gpt-6-luna", budget_usd=30, max_calls=4,
                              max_output_tokens=128, deadline_unix=float(deadline.read_text()))
    directory = root / "session"
    fresh = not (directory / "calls.sqlite3").exists()
    if fresh and directory.exists():
        shutil.rmtree(directory)  # killed before the journal existed: no call was admitted
    session = (ResponsesSession.create if fresh else ResponsesSession.open)(directory, limits, route())

    async def scenario():
        answers = []
        for turn in ("turn-1", "turn-2"):
            try:
                await session.complete(turn, system="Read precisely.", messages=[{"role": "user", "content": turn}])
                answers.append("answered")
            except ResponsesFenced as stopped:
                return {"outcome": "refused_to_repeat", "why": str(stopped)[:70], "answers": answers,
                        "unsettled": session.unsettled}
        return {"outcome": "completed", "answers": answers, "unsettled": session.unsettled}

    try:
        return asyncio.run(scenario())
    finally:
        session.close()


def delivery_prepare(root):
    """Set up before the kill walk, so that every kill point falls in the delivery itself."""
    from taste.memstore import Store

    store = Store.open(root / "repo", "drill")
    integration = store.branch("integration", producer="central")
    integration.write("kept.txt", KEPT)
    base = integration.checkpoint("base")
    worker = store.branch("worker", from_state=base, producer="worker")
    worker.write("new.py", DELIVERED)
    final = worker.checkpoint("worker output")
    (root / "states.json").write_text(json.dumps({"base": base.id, "final": final.id}))
    store.close()


def delivery_effect(root):
    from taste.brains.delivery import DeliveryRecoveryRequired, deliver_product
    from taste.memstore import Store

    states = json.loads((root / "states.json").read_text())
    store = Store.open(root / "repo", "drill")
    target = store.branch("integration", producer="central")
    try:
        try:
            result = deliver_product(store, target, delivery_id="drill", final_state_id=states["final"],
                                     base_state_id=states["base"], artifact_paths=("new.py",))
        except DeliveryRecoveryRequired as stopped:
            return {"outcome": "refused_to_repeat", "why": type(stopped).__name__,
                    "file": target.head.read("new.py"), "kept": target.head.read("kept.txt")}
        return {"outcome": "delivered" if result.ok else "conflict", "reused": bool(result.reused_outcome),
                "file": target.head.read("new.py"), "kept": target.head.read("kept.txt")}
    finally:
        store.close()


def record_effect(root):
    """A cycle's result caught between its write and its commit, then the restart's next record.

    The restart finishes the unfinished cycle, as ``_begin_cycle`` does, with an
    answer of its own. The dead process's file must not ride into a later commit.
    """
    from taste.brains.central_host import open_coordinator_branch
    from taste.brains.central_runtime import CentralRuntime
    from taste.memstore import Store

    store = Store.open(root / "repo", "drill")
    control = open_coordinator_branch(store, "central-control", producer="central-runtime")
    runtime = SimpleNamespace(control=control, control_lock=threading.RLock(), store=store)
    try:
        if control.head.read(RECORD) is not None:
            return {"outcome": "recorded", "by": "the first process"}
        status = "replanned" if RECOVERING else "running"
        if RECOVERING:
            CentralRuntime._immutable(runtime, "cycles/00000014/decision.json", {"after": "restart"}, "a decision")
        CentralRuntime._immutable(runtime, RECORD, {"status": status}, "central cycle")
        return {"outcome": "recorded", "by": "the restart" if RECOVERING else "the first process"}
    finally:
        store.close()


def git_prepare(root):
    """A branch to work on, and the means to die inside git with its locks held."""
    from taste.memstore import Store

    store = Store.open(root / "repo", "drill")
    store.branch("worker", producer="drill").close()
    common = store.backend.common_dir
    (common / "info").mkdir(exist_ok=True)
    (common / "info" / "attributes").write_text("*.die filter=drill\n")
    hooks = root / "hooks"
    hooks.mkdir()
    clean = root / "clean"
    # A clean filter runs while ``git add`` holds index.lock.
    clean.write_text(f"#!/bin/sh\n{DIE}[ -z \"$KILL_DRILL_REF\" ] && die\ncat\n")
    # This hook runs with the transaction's ref locks held.
    hook = hooks / "reference-transaction"
    hook.write_text(f"#!/bin/sh\n{DIE}[ \"$1\" = prepared ] || exit 0\n[ -n \"$KILL_DRILL_REF\" ] || exit 0\n"
                    "grep -q \"$KILL_DRILL_REF\" && die\nexit 0\n")
    clean.chmod(0o755)
    hook.chmod(0o755)
    store.backend.repo.git.config("filter.drill.clean", str(clean))
    store.backend.repo.git.config("core.hooksPath", str(hooks))
    store.close()


def git_effect(root, ref, kill_at):
    from taste.memstore import Store

    if kill_at:
        (root / "armed").write_text(str(kill_at))
        os.environ.update(KILL_DRILL_COUNT=str(root / "armed"), KILL_DRILL_PID=str(os.getpid()), KILL_DRILL_REF=ref)
    store = Store.open(root / "repo", "drill")
    branch = store.branch("worker", producer="drill")
    branch.write("work.die", "the work\n")
    branch.checkpoint("the work is done")
    saved = branch.head.read("work.die")
    store.close()
    return {"outcome": "saved", "read_back": saved}


EFFECTS = {"planner": planner_effect, "planner-moved": lambda root: planner_effect(root, world_moves=True),
           "checkpoint": checkpoint_effect, "terminal": terminal_effect, "worker": worker_effect,
           "delivery": delivery_effect, "record": record_effect}
PREPARE = {"delivery": delivery_prepare, **dict.fromkeys(GIT_PATHS, git_prepare)}


def child(path, root, kill_at):
    global RECOVERING
    RECOVERING = not kill_at
    if path in GIT_PATHS:
        print(json.dumps(git_effect(Path(root), GIT_PATHS[path], kill_at)), flush=True)
        return
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
        if path in PREPARE:
            PREPARE[path](root)
        killed = spawn(path, root, kill_at)
        if killed.returncode == 0:
            return {"at": kill_at, "reached": False}
        record = {"at": kill_at, "reached": killed.returncode == -signal.SIGKILL,
                  "kill_exit": killed.returncode, "world_moves": path == "planner-moved"}
        recovered = spawn(path, root, 0)
        calls = root / "calls.log"
        record.update(
            recovery_exit=recovered.returncode,
            recovery=json.loads(recovered.stdout.splitlines()[-1]) if recovered.returncode == 0 else
            recovered.stderr[-400:],
            calls=max(Counter(calls.read_text().splitlines()).values(), default=0) if calls.exists() else 0,
            git=fsck(root))
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
    if record.get("world_moves"):
        # The old answer is stale, so the question is asked afresh: each call must be on the record.
        attempts = record["recovery"].get("attempts", []) if isinstance(record.get("recovery"), dict) else []
        if record["calls"] > max(1, len(attempts)):
            wrong.append("a paid call is missing from the record")
    elif record["calls"] > 1:
        wrong.append("a paid call was made twice")
    if record["git"] not in ("clean", "no store yet", "no repository yet"):
        wrong.append("git found damage")
    recovery = record.get("recovery")
    if isinstance(recovery, dict) and recovery["outcome"] == "saved" and (
            (recovery.get("read_back") or "").rstrip("\n") != "the work"):
        wrong.append("the saved work is not what was written")
    if isinstance(recovery, dict) and recovery["outcome"] in ("delivered", "conflict", "refused_to_repeat"):
        if recovery["outcome"] == "conflict":
            wrong.append("a clean delivery ended in a conflict")
        allowed = (DELIVERED,) if recovery["outcome"] == "delivered" else (DELIVERED, None)
        if "file" in recovery and recovery["file"] not in allowed:
            wrong.append("the delivered file is not what the worker made")
        if "kept" in recovery and recovery["kept"] != KEPT:
            wrong.append("delivery lost a file the target already held")
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
    parser.add_argument("--start", type=int, default=1, help="first kill point")
    parser.add_argument("--step", type=int, default=1, help="walk every Nth point, to split a slow walk across shards")
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
        for kill_at in range(arguments.start, arguments.limit + 1, arguments.step):
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
