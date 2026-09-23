#!/usr/bin/env python3
"""Server-only entrypoint validation in real, disposable, unprivileged services.

Four serial cases, at most fifteen services, no provider requests. Exercises
expired admission through the actual CLI, then real SIGTERM/SIGKILL of a hung
planner plus detached writer, followed by the actual settlement CLI. The
test-only provider seam is Python injection; launch JSON cannot select it.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pwd
import re
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

from taste.brains import goal_entrypoint
from taste.brains.central_host import compose_central_runtime
from taste.brains.central_runtime import _goal_root
from taste.brains.goal_entrypoint import goal_command, load_goal_input, prepare_goal_process
from taste.brains.process_scope import OwnedProcessScope, ScopeSpec, SystemdManager
from taste.brains.python_process import isolated_python_argv


def require(condition, detail):
    if not condition:
        raise AssertionError(detail)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def script_command(script, *args):
    return isolated_python_argv(
        sys.executable, "import runpy; runpy.run_path(sys.argv.pop(1), run_name='__main__')",
        [str(script), *(str(arg) for arg in args)],
    )


def prepare(root, case):
    from tests.test_brains_central_runtime import simple_goal

    (root / "repo").mkdir()
    config = prepare_goal_process(
        root / "repo", "entrypoint-scope", simple_goal(budget_usd=5),
        max_generations=1, wall_clock_seconds=30,
        deadline_at=datetime.now(UTC) + timedelta(seconds=-1 if case == "expired" else 30),
    )
    (root / "prepared.json").write_bytes(config.to_bytes())


def hang(root, input_path, digest):
    from tests.test_brains_central_runtime import FakeLauncher, ScriptedTransport

    config = load_goal_input(input_path, digest)

    def hung_provider(_request, _prompt):
        descendant = subprocess.Popen(
            script_command(Path(__file__).with_name("check_goal_containment.py"), "writer", root),
            start_new_session=True, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        write_json(root / "entered.json", {
            "driver_pid": os.getpid(), "descendant_pid": descendant.pid,
            "uid": os.getuid(), "cgroup": Path("/proc/self/cgroup").read_text(),
        })
        while True:
            time.sleep(0.05)

    def factory(*args, **kwargs):
        host = compose_central_runtime(
            *args, **kwargs, transport=ScriptedTransport(hung_provider), launcher=FakeLauncher(),
        )
        original = host.request_stop

        def request_stop(detail):
            write_json(root / "signal-observed.json", {"stop_requested": True})
            original(detail)

        host.request_stop = request_stop
        return host

    # Exercise the production signal handler; only the external model and
    # worker-launch boundary is replaced in this disposable fixture process.
    goal_entrypoint.execute_goal = partial(goal_entrypoint.execute_goal, host_factory=factory)
    asyncio.run(goal_entrypoint._run_with_signals(config, "run"))
    raise AssertionError("intentionally hung provider returned")


def verify(root, input_path, digest, case):
    config = load_goal_input(input_path, digest)
    host = compose_central_runtime(config.repo_root, config.session, config.goal)
    try:
        outcome = host.outcome()
        require(outcome is not None and not outcome.complete, "missing or incorrect ending")
        require(not host.supervisor.runs(), "verification observed an unexpected worker")
        require(host.control.head.record(f"{_goal_root(config.goal.goal_id)}/run-limits.json")
                == config.limits, "the admitted deadline changed")
        attempts = host.planner.planner_attempts(config.goal.goal_id)
        if case == "expired":
            require(outcome.stop_reason == "wall_clock" and not attempts, "expired admission executed")
            require(outcome.budget.enforceable, "zero calls left unknown spending")
        else:
            require(outcome.stop_reason == "cancelled", "settlement did not retain cancellation")
            require(len(attempts) == 1 and outcome.budget.unknown_planner_attempt_ids,
                    "recovery retried or forgot the pending provider attempt")
            require(not outcome.budget.enforceable, "missing reply became known spending")
        write_json(root / "verified.json", outcome.to_dict())
    finally:
        host.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--user", default="bugbash")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("expired", "deadline", "explicit_stop", "async_cancel"))
    args = parser.parse_args()
    require(os.geteuid() == 0, "controller must run as root")
    require(re.fullmatch(r"[0-9a-f]{32}", args.owner_token), "invalid owner token")
    account = pwd.getpwnam(args.user)
    root = Path(account.pw_dir) / f"taste-entrypoint-scope-{args.owner_token}"
    root.mkdir(mode=0o755)
    script = Path(__file__).resolve()
    checkout = script.parent.parent
    manager, owned = SystemdManager(), []
    report = {"owner_token": args.owner_token, "cases": [], "cleanup_errors": [],
              "source_sha256": {name: hashlib.sha256((checkout / name).read_bytes()).hexdigest()
                                for name in ("taste/brains/goal_entrypoint.py",
                                             "taste/brains/process_scope.py",
                                             "scripts/check_goal_containment.py",
                                             "scripts/check_goal_entrypoint_scope.py")}}

    def interrupted(signum, _frame):
        raise TimeoutError(f"validation interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)

    def create(case, mode, workdir, argv, duration=15):
        require(len(owned) < 15, "validation exceeded its fifteen-service bound")
        scope = OwnedProcessScope.create(
            root / f"owner-{case}-{mode}",
            ScopeSpec(tuple(argv), str(workdir), account.pw_uid, duration, 0.5),
        )
        owned.append(scope)
        return scope

    def finish(scope):
        scope.start()
        receipt = scope.wait(timeout_seconds=scope.spec.runtime_seconds + 5)
        require(receipt["processes_stopped"] and manager.inspect(scope.unit) is None
                and manager.empty(scope.unit), "service was not drained and released")
        return receipt

    def entered(workdir, scope):
        entry = json.loads((workdir / "entered.json").read_text())
        group = f"/system.slice/{scope.unit}"
        require(entry["uid"] == account.pw_uid and group in entry["cgroup"], "wrong driver scope")
        require(group in Path(f"/proc/{entry['descendant_pid']}/cgroup").read_text(),
                "detached descendant left its process scope")
        return entry

    async def cancel_owned(scope, workdir):
        task = asyncio.create_task(scope.run_async(timeout_seconds=10))
        try:
            deadline = time.monotonic() + 5
            while not (workdir / "entered.json").exists() or not (workdir / "effects.txt").exists():
                require(time.monotonic() < deadline, "async driver did not enter")
                await asyncio.sleep(0.02)
            entry = entered(workdir, scope)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0.02)
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("asynchronous caller cancellation was lost")
            return scope.stop("inspect completed async drainage"), entry
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    try:
        for case in ((args.case,) if args.case else ("expired", "deadline", "explicit_stop", "async_cancel")):
            started = time.monotonic()
            workdir = root / case
            workdir.mkdir(mode=0o700)
            os.chown(workdir, account.pw_uid, account.pw_gid)
            preparation = finish(create(case, "prepare", workdir,
                                        script_command(script, "prepare", workdir, case)))
            raw = (workdir / "prepared.json").read_bytes()
            # Retain the launch input outside the service user's write access.
            input_path = root / f"input-{case}.json"
            input_path.write_bytes(raw)
            input_path.chmod(0o444)
            digest = hashlib.sha256(raw).hexdigest()
            entry = None
            if case == "expired":
                execution = finish(create(case, "run", workdir, goal_command(input_path, digest)))
                settlement = None
            else:
                scope = create(case, "run", workdir,
                               script_command(script, "hang", workdir, input_path, digest), 6)
                if case == "async_cancel":
                    execution, entry = asyncio.run(cancel_owned(scope, workdir))
                else:
                    scope.start()
                    entered_deadline = time.monotonic() + 5
                    while not (workdir / "entered.json").exists() or not (workdir / "effects.txt").exists():
                        require(time.monotonic() < entered_deadline, "hung planner and writer did not enter")
                        time.sleep(0.02)
                    entry = entered(workdir, scope)
                    execution = (scope.stop("test caller cancelled") if case == "explicit_stop"
                                 else scope.wait(timeout_seconds=10))
                require(execution["processes_stopped"] and manager.empty(scope.unit)
                        and manager.inspect(scope.unit) is None, "hung scope was not released")
                require((workdir / "signal-observed.json").is_file(), "production signal handler was not exercised")
                effects = (workdir / "effects.txt").read_bytes()
                time.sleep(0.15)
                require(effects and (workdir / "effects.txt").read_bytes() == effects,
                        "detached effects continued after scope drainage")
                settlement = finish(create(case, "settle", workdir,
                                           goal_command(input_path, digest, mode="settle")))
            verification = finish(create(case, "verify", workdir,
                                         script_command(script, "verify", workdir, input_path, digest, case)))
            outcome = json.loads((workdir / "verified.json").read_text())
            report["cases"].append({"case": case, "input_sha256": digest, "entry": entry,
                                    "preparation": preparation, "execution": execution,
                                    "settlement": settlement, "verification": verification,
                                    "outcome": outcome, "elapsed_seconds": time.monotonic() - started})
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        for scope in owned:
            try:
                scope.stop("validation cleanup")
                require(manager.inspect(scope.unit) is None and manager.empty(scope.unit), "scope remains")
            except BaseException as exc:
                report["cleanup_errors"].append(f"{scope.unit}: {type(exc).__name__}: {exc}")
        write_json(args.output, report)
    return 0 if report.get("status") == "passed" and not report["cleanup_errors"] else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"prepare", "hang", "verify"}:
        mode, root = sys.argv[1], Path(sys.argv[2])
        if mode == "prepare":
            prepare(root, sys.argv[3])
        elif mode == "hang":
            hang(root, Path(sys.argv[3]), sys.argv[4])
        else:
            verify(root, Path(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        raise SystemExit(main())
