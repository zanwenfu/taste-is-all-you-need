#!/usr/bin/env python3
"""Two serial Docker/RPC checks with an unprivileged worker and root broker.

Cached image only; no model calls, network or host mounts. Run in a bounded
systemd unit whose independent ExecStopPost invokes check_docker_terminal.py
--cleanup-only with this unique owner token. The Docker daemon outlives us.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pwd
import re
import shlex
import socket
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

from scripts.check_docker_terminal import HUNG, cleanup, cli, empty_cgroup
from taste.brains.docker_terminal import OWNER_LABEL, DockerTerminalBackend
from taste.brains.terminal_broker import TerminalBinding, TerminalBroker, TerminalRequest
from taste.brains.terminal_service import (
    TerminalAccessDenied,
    TerminalClient,
    TerminalCredential,
    TerminalGrant,
    TerminalService,
)
from taste.brains.terminal_tools import TerminalTools
from taste.providers.base import ToolCall


async def child(credential_path, mode, ledger_path):
    credential = TerminalCredential.from_dict(json.loads(credential_path.read_text()))
    assert os.geteuid() == credential.worker_uid and os.geteuid() != 0
    assert credential.server_uid == 0
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as daemon:
        try:
            daemon.connect("/var/run/docker.sock")
        except PermissionError:
            pass
        else:
            raise AssertionError("worker can access the Docker daemon")
    try:
        ledger_path.read_bytes()
    except PermissionError:
        pass
    else:
        raise AssertionError("worker can read the controller terminal ledger")
    client = TerminalClient(credential)
    assert await client.ping() == "ready"
    if mode == "killed":
        await client.execute(TerminalRequest("hung", credential.grant.actor_id, HUNG, "/tmp", 10))
        raise AssertionError("worker intended for termination returned")
    code = "import sys; sys.stdout.buffer.write(bytes(range(256))*40); sys.stderr.buffer.write(b'err')"
    call = ToolCall("fixture", "terminal_exec", {
        "command": "python3 -c " + shlex.quote(code), "cwd": "/tmp", "timeout_seconds": 10})
    result = await TerminalTools(client).execute("binary", call)
    parsed = json.loads(result.content)
    assert parsed["return_code"] == 0 and parsed["stdout"]["captured_bytes"] == 8192
    assert parsed["stdout"]["dropped_bytes"] == 2048 and not parsed["stdout"]["captured_eof"]
    full = await client.lookup("binary")
    assert full.stdout == bytes(range(256)) * 32 and full.stderr == b"err"
    assert await TerminalTools(client).execute("binary", call) == result
    forged = TerminalClient(replace(credential, grant=replace(credential.grant, actor_id="another_actor")))
    try:
        await forged.ping()
    except TerminalAccessDenied:
        pass
    else:
        raise AssertionError("worker forged another actor's terminal grant")
    print(json.dumps({"status": "passed", "uid": os.geteuid(), "docker_access": False,
                      "ledger_access": False, "captured_bytes": len(full.stdout),
                      "dropped_bytes": full.stdout_dropped_bytes}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-token")
    parser.add_argument("--image")
    parser.add_argument("--worker-user", default="bugbash")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child-credential", type=Path)
    parser.add_argument("--child-mode", choices=("complete", "killed"))
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    if args.child_credential:
        asyncio.run(child(args.child_credential, args.child_mode, args.ledger))
        return
    if os.geteuid() != 0 or not args.owner_token or re.fullmatch(r"[0-9a-f]{32}", args.owner_token) is None:
        parser.error("root controller and a unique 32-hex owner token required")
    if not args.image or re.fullmatch(r"sha256:[0-9a-f]{64}", args.image) is None or args.output is None:
        parser.error("immutable cached image ID and output path required")
    account = pwd.getpwnam(args.worker_user)
    assert account.pw_uid != 0
    assert not cli("ps", "-aq", "--filter", f"label={OWNER_LABEL}={args.owner_token}").stdout.strip()
    assert json.loads(cli("image", "inspect", args.image).stdout)[0]["Id"] == args.image
    artifacts = args.output.with_suffix(".artifacts")
    artifacts.mkdir(mode=0o700)
    report = {"status": "failed", "cases": [], "containers": [], "cleanup_errors": []}

    async def scenario(mode):
        assert len(report["containers"]) < 2
        container_id = cli("create", "--pull", "never", "--network", "none", "--memory", "128m",
            "--cpus", "0.5", "--pids-limit", "64", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--restart", "no", "--label", f"{OWNER_LABEL}={args.owner_token}", "--entrypoint", "/bin/sh",
            args.image, "-c", "sleep 180").stdout.decode().strip()
        report["containers"].append(container_id)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        cli("start", container_id)
        backend = DockerTerminalBackend.admit("/var/run/docker.sock", container_id, args.owner_token,
                                              time.time() + 45, output_limit=8192)
        binding = TerminalBinding(args.owner_token, container_id, backend.binding.deadline_unix, 10)
        owner = TerminalBroker.create(artifacts / mode, binding, backend)
        endpoint = Path("/tmp") / f"taste-terminal-{args.owner_token}-{mode}" / "terminal.sock"
        # Fixture secret is random, separate from the public container label.
        credential = TerminalCredential(str(endpoint), 0, account.pw_uid,
            TerminalGrant(binding, "worker_" + mode, 10), os.urandom(32).hex())
        service = TerminalService(owner, [credential])
        await service.start(worker_gid=account.pw_gid)
        process = None
        try:
            with tempfile.TemporaryDirectory(prefix="taste-rpc-worker-", dir="/tmp") as private:
                os.chown(private, account.pw_uid, account.pw_gid)
                path = Path(private) / "credential.json"
                path.write_text(json.dumps(credential.to_dict()))
                path.chmod(0o600)
                os.chown(path, account.pw_uid, account.pw_gid)
                process = await asyncio.create_subprocess_exec(sys.executable, __file__,
                    "--child-credential", str(path), "--child-mode", mode,
                    "--ledger", str(artifacts / mode / "terminal.sqlite3"),
                    user=account.pw_uid, group=account.pw_gid, extra_groups=(), start_new_session=True,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                if mode == "killed":
                    deadline = time.monotonic() + 8
                    while (await asyncio.to_thread(cli, "cp", f"{container_id}:/tmp/started", "-", check=False)).returncode != 0:
                        assert time.monotonic() < deadline, "unprivileged command never reached container"
                        await asyncio.sleep(0.02)
                    process.kill()
                stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
                assert process.returncode == (0 if mode == "complete" else -9), stderr.decode(errors="replace")
                if mode == "complete":
                    observed = json.loads(stdout)
                    assert observed["status"] == "passed"
                else:
                    async with asyncio.timeout(15):
                        while owner.phase != "stopped":
                            await asyncio.sleep(0.02)
                    observed = {"worker_killed": True, "service_drained_command": True}
                await service.close()
                assert owner.phase == "stopped" and empty_cgroup(backend.binding.cgroup_path)
                assert json.loads(cli("inspect", container_id).stdout)[0]["State"]["Running"] is False
                assert len([event for kind, event in owner.events() if kind == "intent"]) == 1
                report["cases"].append({"mode": mode, "worker_uid": account.pw_uid, "controller_uid": 0,
                    "observed": observed, "cgroup_empty": True, "binding": asdict(backend.binding),
                    "events": owner.events()})
                cli("rm", container_id)
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.communicate()
            await service.close()
            owner.close()
            endpoint.parent.rmdir()

    try:
        for mode in ("complete", "killed"):
            asyncio.run(scenario(mode))
        report["status"] = "passed"
    finally:
        try:
            report["fallback_removed"] = cleanup(args.owner_token)
        except BaseException as exc:
            report["status"] = "failed"
            report["cleanup_errors"].append(type(exc).__name__)
            raise
        finally:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"status": report["status"], "cases": len(report["cases"]),
                              "cleanup_errors": report["cleanup_errors"]}), flush=True)


if __name__ == "__main__":
    main()
