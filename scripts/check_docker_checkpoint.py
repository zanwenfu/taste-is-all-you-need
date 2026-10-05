#!/usr/bin/env python3
"""A checkpoint of a real container's changes, against a real Docker daemon.

    sudo python3 scripts/check_docker_checkpoint.py --image sha256:<id> --owner-token <hex> --output out.json
    sudo python3 scripts/check_docker_checkpoint.py --owner-token <hex> --cleanup-only

Uses one disposable container from a cached image with /bin/sh, no network.
Through the production terminal it adds a file, a directory and a symbolic
link, changes a file's mode, changes a file and deletes another that the
image holds; then it takes a checkpoint and checks that its tar holds exactly
the added and changed paths, with their contents, modes and link, and that
its manifest lists the deletion. The container is removed afterwards, and
--cleanup-only removes anything left with the same owner token.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from taste.brains.docker_terminal import OWNER_LABEL, DockerTerminalBackend
from taste.brains.terminal_broker import TerminalRequest

CHANGES = ("mkdir -p /taste-check/pkg && printf 'A = 1\\n' > /taste-check/pkg/a.py && printf 'new\\n' > /taste-check/new.txt"
           " && chmod 600 /taste-check/new.txt && ln -s /taste-check/new.txt /taste-check/link"
           " && printf 'changed\\n' >> /etc/issue && rm /etc/issue.net")


def cli(*args, check=True):
    # Fixture control only; never used to run task output.
    return subprocess.run(["docker", "--host", "unix:///var/run/docker.sock", *args],
                          capture_output=True, check=check, timeout=30)


def cleanup(token):
    listed = cli("ps", "-aq", "--filter", f"label={OWNER_LABEL}={token}").stdout.decode().split()
    for container in listed:
        cli("rm", "-f", container, check=False)
    return listed


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
        present = backend.execute(TerminalRequest("check_1", "check_actor", "test -f /etc/issue && test -f /etc/issue.net",
                                                  "/", 30))
        if present.return_code != 0:
            raise SystemExit("the image lacks /etc/issue or /etc/issue.net; choose a Debian or Ubuntu based image")
        changed = backend.execute(TerminalRequest("check_2", "check_actor", CHANGES, "/", 30))
        assert changed.return_code == 0, changed
        with tempfile.TemporaryDirectory(prefix="taste-checkpoint-") as directory:
            manifest = backend.checkpoint(directory)
            with tarfile.open(Path(directory) / manifest.tar) as archive:
                members = {member.name: member for member in archive.getmembers()}
                checks = {
                    "added and changed paths copied": sorted(members) == [
                        "etc/issue", "taste-check", "taste-check/link", "taste-check/new.txt", "taste-check/pkg",
                        "taste-check/pkg/a.py"],
                    "added file content": archive.extractfile("taste-check/new.txt").read() == b"new\n",
                    "added file mode kept": members["taste-check/new.txt"].mode & 0o777 == 0o600,
                    "symbolic link kept": members["taste-check/link"].issym()
                                          and members["taste-check/link"].linkname == "/taste-check/new.txt",
                    "changed file content": archive.extractfile("etc/issue").read().endswith(b"changed\n"),
                    "deletion listed": "/etc/issue.net" in manifest.deleted,
                    "nothing over the cap": not manifest.partial,
                }
            report.update(checks=checks, manifest=manifest.to_dict())
        report["status"] = "passed" if all(checks.values()) else "failed"
    finally:
        report["removed"] = cleanup(args.owner_token)
        args.output.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, indent=1))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
