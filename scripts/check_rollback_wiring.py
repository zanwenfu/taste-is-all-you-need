#!/usr/bin/env python3
"""The coordinator's checkpoints and restores of a real container, through the controller.

    sudo python3 scripts/check_rollback_wiring.py --image sha256:<id> --owner-token <hex> --output out.json
    sudo python3 scripts/check_rollback_wiring.py --owner-token <hex> --cleanup-only

What a trial wires together, without a model: the controller's terminal broker
and service bound to one disposable container (cached image with /bin/sh, no
network), the coordinator's private issuer credential, and the adapter its
runtime uses (``IssuerEnvironment``). Commands change the task's files through
the broker as a worker's would. The coordinator checkpoints the container
before them and after them; more commands break things; the coordinator
restores the checkpoint taken after the good work, then the first one. Each
time the files are compared with a fingerprint taken when the checkpoint was.
Asking again with a restore's ID must read the record, not restore again; the
ledger must record each operation before and after. The container is removed
afterwards, and --cleanup-only removes anything left with the same owner token.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import subprocess
import tempfile
import time
from pathlib import Path

from taste.brains.docker_terminal import OWNER_LABEL, DockerTerminalBackend
from taste.brains.terminal_broker import TerminalBinding, TerminalBroker, TerminalRequest
from taste.brains.terminal_issuer import (
    IssuerEnvironment,
    TerminalIssuerClient,
    TerminalIssuerCredential,
)
from taste.brains.terminal_service import TerminalService
from taste.brains.terminal_worker_policy import TerminalWorkerPolicy

FINGERPRINT = ("cd / && find taste-check -print0 2>/dev/null | sort -z | xargs -0 -r ls -ld --time-style=+ | "
               "awk '{print $1, $NF}'; find taste-check -type f -print0 2>/dev/null | sort -z | xargs -0 -r sha256sum;"
               " for f in etc/issue etc/issue.net etc/debian_version etc/shells; do"
               " if [ -e $f ]; then sha256sum $f; else echo missing $f; fi; done")
GOOD = ("mkdir -p /taste-check/app && printf 'def answer():\\n    return 42\\n' > /taste-check/app/server.py"
        " && printf 'ok\\n' > /taste-check/app/status && chmod 600 /taste-check/app/status"
        " && ln -s /taste-check/app/server.py /taste-check/current && printf 'changed\\n' >> /etc/issue"
        " && rm /etc/issue.net")
BREAK = ("printf 'def answer():\\n    return 41\\n' > /taste-check/app/server.py && rm /taste-check/app/status"
         " && mkdir -p /taste-check/build && printf obj > /taste-check/build/out.o && rm /etc/shells"
         " && printf 'patched\\n' >> /etc/debian_version && printf 'back\\n' > /etc/issue.net")


def cli(*args, check=True):
    # Fixture control only; never used to run task output.
    return subprocess.run(["docker", "--host", "unix:///var/run/docker.sock", *args],
                          capture_output=True, check=check, timeout=60)


def cleanup(token):
    listed = cli("ps", "-aq", "--filter", f"label={OWNER_LABEL}={token}").stdout.decode().split()
    for container in listed:
        cli("rm", "-f", container, check=False)
    return listed


async def check(container, token, directory):
    deadline = time.time() + 600
    backend = DockerTerminalBackend.admit("/var/run/docker.sock", container, token, deadline)
    binding = TerminalBinding("rollback_check", backend.environment_id, deadline, 100)
    broker = TerminalBroker.create(directory / "ledger", binding, backend)
    issuer = TerminalIssuerCredential(str(directory / "rpc/terminal.sock"), os.geteuid(), os.geteuid(),
                                      "1" * 64, TerminalWorkerPolicy(binding, 60), secrets.token_hex(32))
    service = TerminalService(broker, issuer=issuer)
    await service.start()
    environment = IssuerEnvironment(TerminalIssuerClient(issuer))
    counter = iter(range(1, 1000))

    async def run(command):
        result = await broker.execute(TerminalRequest(f"command_{next(counter)}", "check_worker", command, "/", 60))
        if result.return_code != 0:
            raise SystemExit(f"command failed: {command!r}: {result.stderr.decode(errors='replace')}")
        return result.stdout.decode(errors="replace")

    async def ask(method, *args):
        return await asyncio.to_thread(getattr(environment, method), *args, timeout_seconds=300)

    try:
        before = await run(FINGERPRINT)
        first = await ask("checkpoint", "initial")
        await run(GOOD)
        good = await run(FINGERPRINT)
        second = await ask("checkpoint", "after_good")
        await run(BREAK)
        broken = await run(FINGERPRINT)
        back_to_good = await ask("restore", "undo_1", "after_good")
        after_good_restore = await run(FINGERPRINT)
        again = await ask("restore", "undo_1", "after_good")
        await run(BREAK)
        back_to_start = await ask("restore", "undo_2", "initial")
        after_start_restore = await run(FINGERPRINT)
        kinds = [kind for kind, _ in broker.events()]
        checks = {
            "checkpoints taken": not first.get("failed") and not second.get("failed"),
            "the good work is in the second checkpoint": second.get("added", 0) >= 1 and second.get("deleted") == 1,
            "breaking changed the files": broken != good,
            "restore to the good checkpoint is exact": back_to_good.get("exact") is True,
            "files as after the good work": after_good_restore == good,
            "asking again reads the record": again == back_to_good,
            "restore to the first checkpoint is exact": back_to_start.get("exact") is True,
            "files as before any work": after_start_restore == before,
            "each operation recorded before and after": (
                kinds.count("checkpoint_intent") == kinds.count("checkpoint") == 2
                and kinds.count("restore_intent") == kinds.count("restored") == 2),
            "no failure recorded": not {"checkpoint_failed", "restore_failed", "fenced"} & set(kinds),
            "still admitting commands": broker.phase == "ready",
        }
        return {"checks": checks, "checkpoints": [first, second], "restores": [back_to_good, back_to_start],
                "fingerprint": good, "events": kinds}
    finally:
        await service.close()
        broker.close()


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
                    args.image, "-c", "sleep 900").stdout.decode().strip()
    try:
        cli("start", container)
        with tempfile.TemporaryDirectory(prefix="taste-rollback-check-") as directory:
            report.update(asyncio.run(check(container, args.owner_token, Path(directory))))
        report["status"] = "passed" if report["checks"] and all(report["checks"].values()) else "failed"
    finally:
        report["removed"] = cleanup(args.owner_token)
        args.output.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, indent=1))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
