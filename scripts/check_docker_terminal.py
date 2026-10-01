#!/usr/bin/env python3
"""Production terminal transport in five serial, disposable server containers.

Use only a cached immutable image with /bin/sh and python3. No network, host
mounts, image pulls or provider calls. The outside runner must bound this
process and independently run --cleanup-only for the same unique owner token
after it exits (including timeout). Docker daemon effects outlive this process.

A command that runs past its timeout, or whose caller is cancelled, is ended
alone: it and every process it started are killed, its partial output is kept
as a completed receipt, and the container, its files and the terminal stay
usable. The two interrupted cases check exactly that against a real daemon.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from taste.brains.docker_terminal import OWNER_LABEL, DockerTerminalBackend, DockerTerminalBinding
from taste.brains.terminal_broker import (
    TerminalBinding,
    TerminalBroker,
    TerminalRequest,
    TerminalResult,
)

WRITER = "setsid sh -c 'while :; do echo tick >> /tmp/ticks; sleep 0.05; done' </dev/null >/dev/null 2>&1 &"
HUNG = WRITER + " echo started > /tmp/started; trap '' TERM; while :; do sleep 1; done"


def cli(*args, check=True):
    # Fixture control only; never used to execute arbitrary task output.
    return subprocess.run(["docker", "--host", "unix:///var/run/docker.sock", *args],
                          capture_output=True, check=check, timeout=15)


def req(identifier, command, timeout=10, cwd="/tmp"):
    return TerminalRequest(identifier, "validation_worker", command, cwd, timeout)


def leaves(exc):
    return [leaf for child in exc.exceptions for leaf in leaves(child)] if isinstance(exc, BaseExceptionGroup) else [exc]


def empty_cgroup(path):
    assert Path("/sys/fs/cgroup/cgroup.controllers").is_file()
    events = Path("/sys/fs/cgroup") / path.lstrip("/") / "cgroup.events"
    try:
        values = dict(line.split() for line in events.read_text().splitlines())
    except FileNotFoundError:
        return True
    return values.get("populated") == "0"


def cleanup(token):
    owned = cli("ps", "--all", "--quiet", "--no-trunc", "--filter", f"label={OWNER_LABEL}={token}")
    removed = []
    for container_id in owned.stdout.decode().splitlines():
        assert re.fullmatch(r"[0-9a-f]{64}", container_id)
        info = json.loads(cli("inspect", container_id).stdout)[0]
        assert info["Id"] == container_id and info["Config"]["Labels"][OWNER_LABEL] == token
        cgroup = None
        if info["State"]["Running"]:
            pid = info["State"]["Pid"]
            cgroup = next(line[3:] for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines()
                          if line.startswith("0::"))
            assert container_id in cgroup
            cli("stop", "--time", "1", container_id)
        assert json.loads(cli("inspect", container_id).stdout)[0]["State"]["Running"] is False
        if cgroup:
            assert empty_cgroup(cgroup), "cleanup found remaining task descendants"
        cli("rm", container_id)
        removed.append(container_id)
    return removed


def crash_child(binding_path, ledger):
    binding = DockerTerminalBinding(**json.loads(binding_path.read_text()))
    backend = DockerTerminalBackend(binding)
    execute = backend.execute

    def die_after_effect(request):
        result = execute(request)
        assert result.return_code == 0
        os._exit(73)

    backend.execute = die_after_effect
    owner = TerminalBroker.create(ledger,
        TerminalBinding(binding.owner_token, binding.container_id, binding.deadline_unix, 20), backend)
    asyncio.run(owner.execute(req("crash_effect", WRITER + " echo effect-before-controller-death")))
    raise AssertionError("crash child returned")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-token")
    parser.add_argument("--image")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cleanup-only", action="store_true")
    parser.add_argument("--crash-binding", type=Path)
    parser.add_argument("--crash-ledger", type=Path)
    args = parser.parse_args()
    if args.crash_binding:
        return crash_child(args.crash_binding, args.crash_ledger)
    if not args.owner_token or re.fullmatch(r"[0-9a-f]{32}", args.owner_token) is None:
        parser.error("unique lowercase 32-hex owner token required")
    if args.cleanup_only:
        print(json.dumps({"removed": cleanup(args.owner_token)}))
        return
    if not args.image or re.fullmatch(r"sha256:[0-9a-f]{64}", args.image) is None or args.output is None:
        parser.error("immutable cached image ID and output path required")
    token, output = args.owner_token, args.output
    directory = output.with_suffix(".artifacts")
    directory.mkdir(mode=0o700)
    report = {"cases": [], "containers": [], "status": "failed", "cleanup_errors": []}
    assert not cli("ps", "-aq", "--filter", f"label={OWNER_LABEL}={token}").stdout.strip()
    assert json.loads(cli("image", "inspect", args.image).stdout)[0]["Id"] == args.image

    def create(name):
        assert len(report["containers"]) < 5
        container_id = cli("create", "--pull", "never", "--network", "none", "--memory", "128m",
            "--cpus", "0.5", "--pids-limit", "64", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--restart", "no",
            "--label", f"{OWNER_LABEL}={token}", "--entrypoint", "/bin/sh",
            args.image, "-c", "sleep 300").stdout.decode().strip()
        report["containers"].append(container_id)
        output.write_text(json.dumps(report, indent=2) + "\n")
        cli("start", container_id)
        backend = DockerTerminalBackend.admit("/var/run/docker.sock", container_id, token,
                                              time.time() + 60, output_limit=4096)
        (directory / f"{name}.binding.json").write_text(json.dumps(asdict(backend.binding)) + "\n")
        return backend

    def owner_for(name, backend):
        return TerminalBroker.create(directory / name,
            TerminalBinding(token, backend.environment_id, backend.binding.deadline_unix, 20), backend)

    def record(name, owner, backend):
        assert owner.phase == "stopped"
        info = json.loads(cli("inspect", backend.environment_id).stdout)[0]
        assert info["State"]["Running"] is False
        assert empty_cgroup(backend.binding.cgroup_path)
        report["cases"].append({"name": name, "binding": asdict(backend.binding),
                                "events": owner.events(), "cgroup_empty": True})
        owner.close()
        cli("rm", backend.environment_id)
        output.write_text(json.dumps(report, indent=2) + "\n")

    async def regular():
        backend = create("regular")
        owner = owner_for("regular", backend)
        await asyncio.gather(
            owner.execute(req("a", "echo A-start >> /tmp/order; sleep 0.15; echo A-end >> /tmp/order")),
            owner.execute(req("b", "echo B >> /tmp/order; " + WRITER + " sleep 0.2")))
        assert await owner.execute(req("read", "cat /tmp/order; test -s /tmp/ticks")) == TerminalResult(0, b"A-start\nA-end\nB\n")
        code = "import sys; sys.stdout.buffer.write(bytes(range(256))*32768); sys.stderr.buffer.write(b'e'*3000000)"
        command = "python3 -c " + shlex.quote(code)
        captured = await owner.execute(req("large", command))
        assert captured == TerminalResult(0, bytes(range(256)) * 16, b"e" * 4096, 8388608 - 4096, 3000000 - 4096)
        for exit_code in (1, 125, 126, 127, 137, 255):
            assert (await owner.execute(req(f"exit_{exit_code}", f"exit {exit_code}"))).return_code == exit_code
        owner.close()
        recovered = TerminalBroker.open(directory / "regular", owner.binding, DockerTerminalBackend(backend.binding))
        assert await recovered.execute(req("large", command)) == captured
        await recovered.abort()
        record("serial, persistent, binary, bounded, exit statuses and replay", recovered, backend)

    async def interrupted(ending):
        backend = create(ending)
        owner = owner_for(ending, backend)
        if ending == "bad_cwd":
            # Docker 29 streams the OCI error and confirms exit 127 via exec
            # inspect. That is a known result, unlike a lost daemon reply.
            result = await owner.execute(req("failed_start", "echo cannot start", cwd="/missing-task-dir"))
            assert result.return_code == 127 and b"/missing-task-dir" in result.stdout + result.stderr
            assert await owner.execute(req("recovered_cwd", "echo recovered")) == TerminalResult(0, b"recovered\n")
            await owner.abort()
            record("confirmed OCI exit and working-directory recovery", owner, backend)
            return
        began = time.monotonic()
        active = asyncio.create_task(owner.execute(req("hung", HUNG, 1 if ending == "timeout" else 30)))
        if ending == "cancel":
            deadline = time.monotonic() + 5
            while (await asyncio.to_thread(cli, "cp", f"{backend.environment_id}:/tmp/started", "-", check=False)).returncode != 0:
                assert time.monotonic() < deadline, "command did not start"
                await asyncio.sleep(0.02)
            active.cancel()
            try:
                await active
            except BaseException as exc:
                assert any(isinstance(e, asyncio.CancelledError) for e in leaves(exc))
            else:
                raise AssertionError("a cancelled caller received a result")
            # The ending is on record: asking again replays it, it does not run again.
            result = await owner.execute(req("hung", HUNG, 30))
        else:
            result = await active
        ended = time.monotonic() - began
        assert result.terminated == ("timeout" if ending == "timeout" else "cancelled"), result
        assert ended < 20, "the command was not ended promptly"
        # It ignored TERM and had started a detached writer. Both are gone...
        assert owner.phase == "ready"
        first = await owner.execute(req("ticks_1", "cat /tmp/started; wc -c < /tmp/ticks"))
        await asyncio.sleep(0.5)
        second = await owner.execute(req("ticks_2", "wc -c < /tmp/ticks; ps -eo pid,args 2>/dev/null || ls /proc"))
        assert first.return_code == 0 and first.stdout.startswith(b"started\n"), first
        assert first.stdout.split()[1] == second.stdout.split()[0], "the command's writer survived it"
        assert b"while :" not in second.stdout, second.stdout
        # ...and the container, its files and the terminal are as they were.
        assert json.loads(cli("inspect", backend.environment_id).stdout)[0]["State"]["Running"] is True
        assert await owner.execute(req("after", "echo still-usable")) == TerminalResult(0, b"still-usable\n")
        await owner.abort()
        record(f"{ending}: the command ended alone after {ended:.1f}s (exit {result.return_code})", owner, backend)

    try:
        asyncio.run(regular())
        for ending in ("timeout", "cancel", "bad_cwd"):
            asyncio.run(interrupted(ending))
        backend = create("crash")
        result = subprocess.run([sys.executable, __file__, "--crash-binding", str(directory / "crash.binding.json"),
                                 "--crash-ledger", str(directory / "crash")], timeout=15, check=False)
        assert result.returncode == 73
        assert json.loads(cli("inspect", backend.environment_id).stdout)[0]["State"]["Running"] is True
        recovered = TerminalBroker.open(directory / "crash",
            TerminalBinding(token, backend.environment_id, backend.binding.deadline_unix, 20),
            DockerTerminalBackend(backend.binding))
        assert recovered.phase == "fenced"
        asyncio.run(recovered.abort())
        record("controller death recovery from original binding", recovered, backend)
        report["status"] = "passed"
    finally:
        try:
            report["fallback_removed"] = cleanup(token)
        except BaseException as exc:
            report["cleanup_errors"].append(repr(exc))
            report["status"] = "failed"
            raise
        finally:
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"status": report["status"], "cases": len(report["cases"]),
                              "containers": len(report["containers"]), "cleanup_errors": report["cleanup_errors"]}))


if __name__ == "__main__":
    main()
