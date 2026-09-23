#!/usr/bin/env python3
"""Bounded Linux/systemd feasibility check for the outer goal timeout.

Run only on the authorized test server, as the outside controller (root).
The disposable services run as an unprivileged user, call no model, and use
the real central runtime with a deliberately hung synthetic planner. This
checks kernel process containment; it is not the production Harbor adapter
or proof that task commands may safely execute on the host.
"""

from __future__ import annotations

import argparse
import json
import os
import pwd
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def command(argv, *, timeout=15, check=True):
    return subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=check)


def require(condition, detail):
    if not condition:
        raise AssertionError(detail)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def worker(root):
    """Neither the provider nor its detached descendant cooperates with stop."""
    from taste.brains.central_host import compose_central_runtime
    from tests.test_brains_central_runtime import FakeLauncher, ScriptedTransport, simple_goal

    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    def hung_provider(_request, _prompt):
        descendant = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "writer", str(root)],
            start_new_session=True, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        write_json(root / "entered.json", {
            "driver_pid": os.getpid(), "descendant_pid": descendant.pid,
            "uid": os.getuid(), "cgroup": Path("/proc/self/cgroup").read_text(),
        })
        while True:
            time.sleep(0.05)

    repo = root / "repo"
    repo.mkdir()
    host = compose_central_runtime(
        repo, "containment-check", simple_goal(budget_usd=5),
        transport=ScriptedTransport(hung_provider), launcher=FakeLauncher(),
    )
    # The inner deadline cannot preempt this provider. The outside systemd
    # manager must terminate it even after the process starting the unit exits.
    host.run(max_generations=1, wall_clock_seconds=1)
    raise AssertionError("the intentionally hung planner unexpectedly returned")


def writer(root):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        with (root / "effects.txt").open("a") as out:
            out.write("detached effect\n")
        time.sleep(0.02)


def recover(root):
    """Settle the retained real journal without admitting another model call."""
    from taste.brains.central_host import compose_central_runtime
    from tests.test_brains_central_runtime import FakeLauncher, ScriptedTransport, simple_goal

    def forbidden(*_args):
        raise AssertionError("recovery admitted a new provider call")

    launcher, transport = FakeLauncher(), ScriptedTransport(forbidden)
    host = compose_central_runtime(
        root / "repo", "containment-check", simple_goal(budget_usd=5),
        transport=transport, launcher=launcher,
    )
    try:
        outcome = host.stop_and_drain("outer containment terminated the hung driver")
        require(not outcome.complete, "hard termination produced success")
        require(not outcome.budget.enforceable, "missing provider reply became known spending")
        require(outcome.budget.unknown_planner_attempt_ids, "missing provider receipt was lost")
        require(not transport.calls and not launcher.launch_calls, "recovery admitted new execution")
        write_json(root / "recovered.json", outcome.to_dict())
    finally:
        host.close()


def properties(unit):
    result = command([
        "systemctl", "show", unit, "--no-pager",
        "--property=LoadState,ActiveState,SubState,Description,ControlGroup,Result,"
        "InvocationID,KillMode,RuntimeMaxUSec,TimeoutStopUSec,SendSIGKILL,User,Restart",
    ], check=False)
    # systemctl may return nonzero for an unloaded transient unit. Its explicit
    # not-found property is evidence; an arbitrary command failure is not.
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    require(values.get("LoadState") in {"loaded", "not-found"},
            f"unit inspection failed: {result.returncode}: {result.stderr}")
    return values


def empty_group(unit):
    path = Path("/sys/fs/cgroup/system.slice") / unit / "cgroup.events"
    try:
        values = dict(line.split() for line in path.read_text().splitlines())
    except FileNotFoundError:
        return True
    return values.get("populated") == "0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--user", default="bugbash")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.geteuid() == 0, "the outside controller must own the system service")
    require(re.fullmatch(r"[0-9a-f]{32}", args.owner_token), "invalid owner token")
    account = pwd.getpwnam(args.user)
    require(account.pw_uid != 0 and not os.getgrouplist(args.user, account.pw_gid)[1:],
            "the service user must be unprivileged and have no supplementary groups")
    require(Path("/sys/fs/cgroup/cgroup.controllers").is_file(), "cgroup v2 is required")
    root = Path(account.pw_dir) / f"taste-containment-check-{args.owner_token}"
    root.mkdir(mode=0o755)  # Exclusive ownership; never reuse an earlier check.
    script = Path(__file__).resolve()
    checkout = script.parent.parent
    report = {"owner_token": args.owner_token, "cases": [], "cleanup_errors": []}
    owned = {}

    def interrupted(signum, _frame):
        raise TimeoutError(f"containment check interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)

    def launch(unit, workdir, mode, seconds):
        description = f"taste-containment-check:{args.owner_token}:{mode}"
        require(properties(unit)["LoadState"] == "not-found", "unit name already exists")
        # Remember before dispatch: a lost acknowledgement still needs cleanup.
        owned[unit] = description
        argv = [
            "systemd-run", "--quiet", f"--unit={unit}", f"--description={description}",
            "--service-type=exec", "--expand-environment=no",
            f"--uid={args.user}", f"--gid={account.pw_gid}",
            f"--working-directory={workdir}",
            "--property=Slice=system.slice", "--property=KillMode=control-group",
            f"--property=RuntimeMaxSec={seconds}", "--property=TimeoutStopSec=0.5",
            "--property=SendSIGKILL=yes", "--property=FinalKillSignal=SIGKILL",
            "--property=Restart=no", "--property=Delegate=no",
            "--property=NoNewPrivileges=yes", "--property=CapabilityBoundingSet=",
            "--property=ProtectControlGroups=yes", "--property=TasksMax=64",
            "--property=MemoryMax=512M", "--property=CPUQuota=100%",
            "--property=InaccessiblePaths=-/run/user -/run/dbus -/run/systemd/private -/run/docker.sock",
            "--property=StandardOutput=null", "--property=StandardError=journal",
            f"--setenv=PYTHONPATH={checkout}",
            "--", sys.executable, str(script), mode, str(workdir),
        ]
        # This separate submitting process exits after the start reply. The
        # service manager, not that process or Python's driver, owns the timer.
        command(argv)

    def stop_owned(unit):
        state = properties(unit)
        if state["LoadState"] != "not-found":
            require(state["Description"] == owned[unit], "cleanup refused foreign unit")
            command(["systemctl", "stop", unit], timeout=10)
        require(empty_group(unit), "owned cgroup still has live descendants")

    try:
        for case in ("deadline_after_submitter_exit", "explicit_stop"):
            workdir = root / case
            workdir.mkdir(mode=0o700)
            os.chown(workdir, account.pw_uid, account.pw_gid)
            unit = f"taste-containment-{args.owner_token}-{case.replace('_', '-')}.service"
            started = time.monotonic()
            launch(unit, workdir, "worker", 6 if case.startswith("deadline") else 15)
            while not (workdir / "entered.json").exists():
                require(time.monotonic() - started < 5, "planner failed to enter before the deadline")
                time.sleep(0.02)
            entered = json.loads((workdir / "entered.json").read_text())
            while not (workdir / "effects.txt").exists():
                require(time.monotonic() - started < 5, "detached writer failed to start")
                time.sleep(0.02)
            running = properties(unit)
            require(running["Description"] == owned[unit], "service identity changed")
            require(running["KillMode"] == "control-group", "wrong process kill boundary")
            require(running["SendSIGKILL"] == "yes" and running["Restart"] == "no",
                    "service termination configuration differs")
            require(running["ControlGroup"] == f"/system.slice/{unit}", "unexpected cgroup")
            require(entered["uid"] == account.pw_uid and f"/system.slice/{unit}" in entered["cgroup"],
                    "driver did not enter the intended unprivileged scope")
            require(f"/system.slice/{unit}" in Path(
                f"/proc/{entered['descendant_pid']}/cgroup"
            ).read_text(), "detached descendant escaped the cgroup")
            if case == "explicit_stop":
                stop_owned(unit)
            else:
                while not empty_group(unit):
                    require(time.monotonic() - started < 10, "hard deadline left running descendants")
                    time.sleep(0.05)
            stopped = properties(unit)
            require(empty_group(unit), "scope still populated after termination")
            effects = (workdir / "effects.txt").read_bytes()
            require(effects, "detached writer produced no observable effect")
            time.sleep(0.15)
            require((workdir / "effects.txt").read_bytes() == effects, "effects continued after containment")
            recovery = f"taste-containment-{args.owner_token}-{case.replace('_', '-')}-recovery.service"
            launch(recovery, workdir, "recover", 10)
            recovery_started = time.monotonic()
            while not empty_group(recovery):
                require(time.monotonic() - recovery_started < 12, "recovery did not end")
                time.sleep(0.05)
            require((workdir / "recovered.json").is_file(), "bounded journal recovery failed")
            report["cases"].append({
                "case": case, "entry": entered, "running": running, "stopped": stopped,
                "elapsed_seconds": time.monotonic() - started,
                "retained_effect_bytes": len(effects), "unknown_cost_preserved": True,
                "scope_empty": True,
            })
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        for unit in owned:
            try:
                stop_owned(unit)
                state = properties(unit)
                if state["LoadState"] != "not-found":
                    require(state["Description"] == owned[unit], "reset refused foreign unit")
                    command(["systemctl", "reset-failed", unit], check=False)
            except BaseException as exc:
                report["cleanup_errors"].append(f"{unit}: {type(exc).__name__}: {exc}")
        write_json(args.output, report)
    return 0 if report.get("status") == "passed" and not report["cleanup_errors"] else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in {"worker", "writer", "recover"}:
        {"worker": worker, "writer": writer, "recover": recover}[sys.argv[1]](Path(sys.argv[2]))
    else:
        raise SystemExit(main())
