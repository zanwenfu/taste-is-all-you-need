#!/usr/bin/env python3
"""Exercise the durable process controller against real disposable services.

Server only, root controller, unprivileged services, no model calls. Includes
actual controller SIGKILL, lost start reply, explicit stop and independent
deadline expiry. The bounded child/recovery fixtures live in
check_goal_containment.py and use the real central runtime and its journal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from taste.brains.process_scope import OwnedProcessScope, ScopeSpec, SystemdManager
from taste.resources import ResourceCleanupError


def require(condition, detail):
    if not condition:
        raise AssertionError(detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--user", default="bugbash")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.geteuid() == 0, "controller must run as root")
    require(re.fullmatch(r"[0-9a-f]{32}", args.owner_token), "invalid owner token")
    account = pwd.getpwnam(args.user)
    require(account.pw_uid > 0, "service must be unprivileged")
    root = Path(account.pw_dir) / f"taste-owned-scope-check-{args.owner_token}"
    root.mkdir(mode=0o755)
    script = Path(__file__).resolve()
    checkout = script.parent.parent
    fixture = checkout / "scripts/check_goal_containment.py"
    manager = SystemdManager()
    owned = []
    report = {"owner_token": args.owner_token, "cases": [], "cleanup_errors": [],
              "source_sha256": {name: hashlib.sha256((checkout / name).read_bytes()).hexdigest()
                                for name in ("taste/brains/process_scope.py",
                                             "scripts/check_goal_containment.py",
                                             "scripts/check_owned_goal_scope.py")}}

    def interrupted(signum, _frame):
        raise TimeoutError(f"scope check interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)

    def create(case, workdir, mode, duration):
        require(len(owned) < 8, "check exceeded its eight-service bound")
        scope = OwnedProcessScope.create(
            root / f"owner-{case}-{mode}",
            ScopeSpec((sys.executable, str(fixture), mode, str(workdir)), str(workdir),
                      account.pw_uid, duration, 0.5, str(checkout)),
        )
        owned.append(scope)
        return scope

    try:
        for case in ("deadline", "explicit_stop", "lost_start_reply", "controller_sigkill"):
            workdir = root / case
            workdir.mkdir(mode=0o700)
            os.chown(workdir, account.pw_uid, account.pw_gid)
            scope = create(case, workdir, "worker", 6)
            started = time.monotonic()
            if case == "lost_start_reply":
                class LostReply(SystemdManager):
                    def start(self, *values):
                        super().start(*values)
                        raise ConnectionError("injected lost service start acknowledgement")

                scope.manager = LostReply()
                try:
                    scope.start()
                except ResourceCleanupError as exc:
                    require("lost service start acknowledgement" in str(exc), f"wrong launch failure: {exc}")
                else:
                    raise AssertionError("lost start reply was not reported")
                scope = OwnedProcessScope(scope.directory)
            elif case == "controller_sigkill":
                child = subprocess.run(
                    [sys.executable, str(script), "start-and-die", str(scope.directory)],
                    capture_output=True, text=True, timeout=20,
                )
                require(child.returncode == -signal.SIGKILL,
                        f"controller was not killed as intended: {child.returncode}: {child.stderr}")
                scope = OwnedProcessScope(scope.directory)
                try:
                    scope.start()
                except RuntimeError as exc:
                    require("already admitted" in str(exc), "restart failed for the wrong reason")
                else:
                    raise AssertionError("replacement controller admitted a second execution")
            else:
                scope.start()
            while not (workdir / "entered.json").exists() or not (workdir / "effects.txt").exists():
                require(time.monotonic() - started < 5, "hung planner and writer did not enter")
                time.sleep(0.02)
            entered = json.loads((workdir / "entered.json").read_text())
            expected = f"/system.slice/{scope.unit}"
            require(entered["uid"] == account.pw_uid and expected in entered["cgroup"], "wrong driver scope")
            require(expected in Path(f"/proc/{entered['descendant_pid']}/cgroup").read_text(),
                    "detached child did not inherit the execution scope")
            if case in {"explicit_stop", "lost_start_reply"}:
                termination = scope.stop("caller cancelled the test goal")
            else:
                termination = scope.wait(timeout_seconds=10)
            require(termination["processes_stopped"] and termination["goal_settlement_required"],
                    "process termination incorrectly settled the goal")
            require(manager.inspect(scope.unit) is None and manager.empty(scope.unit), "scope leaked")
            require(OwnedProcessScope(scope.directory).stop("retry") == termination,
                    "termination receipt was not stable across restart")
            effects = (workdir / "effects.txt").read_bytes()
            require(effects, "detached child made no observable effect")
            time.sleep(0.15)
            require((workdir / "effects.txt").read_bytes() == effects, "effects continued after stop returned")
            recovery = create(case, workdir, "recover", 10)
            recovery.start()
            recovered = recovery.wait(timeout_seconds=12)
            require(recovered["processes_stopped"], "recovery process did not drain")
            outcome = json.loads((workdir / "recovered.json").read_text())
            require(not outcome["complete"], "hard stop was turned into goal success")
            report["cases"].append({
                "case": case, "termination": termination, "entry": entered,
                "elapsed_seconds": time.monotonic() - started,
                "retained_effect_bytes": len(effects), "unknown_cost_preserved": True,
                "recovery_termination": recovered,
            })
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        for scope in owned:
            try:
                OwnedProcessScope(scope.directory).stop("validation cleanup")
                require(manager.inspect(scope.unit) is None and manager.empty(scope.unit), "scope remains")
            except BaseException as exc:
                report["cleanup_errors"].append(f"{scope.unit}: {type(exc).__name__}: {exc}")
                # The test's independent cleanup still verifies exact ownership.
                # This never converts a failed controller check into a pass.
                try:
                    observed = manager.inspect(scope.unit)
                    if observed is not None:
                        require(observed["Description"] == scope.description, "refuse foreign cleanup")
                        manager.stop(scope.unit, scope.spec.grace_seconds)
                        require(manager.empty(scope.unit), "fallback cleanup left live descendants")
                        manager.release(scope.unit)
                except BaseException as cleanup:
                    report["cleanup_errors"].append(f"independent cleanup: {type(cleanup).__name__}: {cleanup}")
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report.get("status") == "passed" and not report["cleanup_errors"] else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "start-and-die":
        OwnedProcessScope(sys.argv[2]).start()
        os.kill(os.getpid(), signal.SIGKILL)
    else:
        raise SystemExit(main())
