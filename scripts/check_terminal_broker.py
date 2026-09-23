#!/usr/bin/env python3
"""Serial terminal-broker checks in at most five disposable server containers.

No model calls, network, image pulls, or host mounts. The Docker backend here
is a validation fixture for fixed, tiny-output commands, NOT a production
Harbor adapter. In particular it does not provide bounded arbitrary-output
transport, worker authentication, or automatic containment on controller death.
The last boundary is exercised by an explicit recovery owner in this script.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

from taste.brains.terminal_broker import (
    TerminalBinding,
    TerminalBroker,
    TerminalFenced,
    TerminalRequest,
    TerminalResult,
)

WRITER = "setsid sh -c 'while :; do echo tick >> /tmp/ticks; sleep 0.05; done' </dev/null >/dev/null 2>&1 & echo $! > /tmp/writer.pid"
HUNG = WRITER + "; trap '' TERM; while :; do sleep 1; done"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class DockerFixture:
    def __init__(self, client, container_id, token):
        self.client, self.environment_id, self.token = client, container_id, token
        self.calls, self.stops = 0, 0
        self.lose_stop_reply = False
        self.before_stop = threading.Event()
        self.allow_stop = threading.Event()
        self.allow_stop.set()
        container = self._container()
        pid = container.attrs["State"]["Pid"]
        self.cgroup = next(line[3:] for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines()
                           if line.startswith("0::"))
        require(self.cgroup != "/", "container has no distinct cgroup")

    def _container(self):
        container = self.client.containers.get(self.environment_id)
        require(container.id == self.environment_id, "container ID changed")
        require(container.labels.get("taste.validation") == self.token, "container owner changed")
        return container

    def execute(self, request):
        container = self._container()
        require(container.attrs["State"]["Running"], "fixture never restarts a stopped container")
        self.calls += 1
        result = container.exec_run(["sh", "-c", request.command], workdir=request.cwd, demux=True)
        stdout, stderr = result.output
        return TerminalResult(result.exit_code, stdout or b"", stderr or b"")

    def stopped(self):
        container = self._container()
        require(not container.attrs["State"]["Running"], "container is still running")
        events = Path("/sys/fs/cgroup") / self.cgroup.lstrip("/") / "cgroup.events"
        if events.exists():
            values = dict(line.split() for line in events.read_text().splitlines())
            require(values.get("populated") == "0", "task descendants remain in the cgroup")
        return self.environment_id

    def stop_and_confirm(self):
        self.stops += 1
        self.before_stop.set()
        require(self.allow_stop.wait(10), "fixture stop gate timed out")
        container = self._container()
        if container.attrs["State"]["Running"]:
            container.stop(timeout=1)
        receipt = self.stopped()
        if self.lose_stop_reply:
            self.lose_stop_reply = False
            raise ConnectionError("injected loss of an actual stop acknowledgement")
        return receipt


def req(identifier, command, timeout=10):
    return TerminalRequest(identifier, "fixture_worker", command, "/tmp", timeout)


async def wait_writer(backend):
    # Observe through the daemon's archive API; never start a competing exec.
    from docker.errors import NotFound

    def exists():
        try:
            chunks, _ = backend._container().get_archive("/tmp/ticks")
            return bool(b"".join(chunks))
        except NotFound:
            return False

    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if await asyncio.to_thread(exists):
            return
        await asyncio.sleep(0.02)
    raise AssertionError("detached writer never started")


def child(directory, container_id, token, deadline):
    import docker

    client = docker.from_env(timeout=10)
    backend = DockerFixture(client, container_id, token)
    execute = backend.execute

    def die_after_effect(request):
        result = execute(request)
        require(result.return_code == 0, "crash fixture command failed")
        os._exit(73)  # Daemon-side writer lives; no terminal receipt was committed.

    backend.execute = die_after_effect
    binding = TerminalBinding(token, container_id, float(deadline), 20)
    owner = TerminalBroker.create(Path(directory), binding, backend)
    asyncio.run(owner.execute(req("crashed_effect", WRITER)))
    raise AssertionError("crash fixture returned")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{32}", args.owner_token) is None:
        parser.error("owner token must be 32 lowercase hex characters")

    import docker

    client = docker.from_env(timeout=10)
    token = args.owner_token
    label = {"label": f"taste.validation={token}"}
    report = {"owner_token": token, "python": platform.python_version(), "cases": [],
              "created_containers": [], "cleanup_errors": [], "status": "failed"}
    owned = False

    def create():
        require(len(report["created_containers"]) < 5, "container limit reached")
        container = client.containers.run(
            image, ["sh", "-c", "sleep 300"], detach=True, network_mode="none",
            mem_limit="128m", nano_cpus=500_000_000, pids_limit=64,
            cap_drop=["ALL"], security_opt=["no-new-privileges"],
            labels={"taste.validation": token},
        )
        report["created_containers"].append(container.id)
        return DockerFixture(client, container.id, token)

    def record(name, backend, owner):
        require(owner.phase == "stopped", "broker did not record confirmed termination")
        backend.stopped()
        report["cases"].append({"name": name, "binding": asdict(owner.binding),
                                "exec_calls": backend.calls, "stop_calls": backend.stops,
                                "cgroup": backend.cgroup, "events": owner.events()})
        owner.close()
        backend._container().remove()

    try:
        require(not client.containers.list(all=True, filters=label), "validation token already in use")
        owned = True
        image = client.images.get(args.image).id  # Cached image only.
        report.update(image=image, docker_server=client.version()["Version"],
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        with nullcontext(args.output.with_suffix(".artifacts")) as directory:
            directory.mkdir(mode=0o700)
            report["artifacts_directory"] = str(directory)

            async def persistent():
                backend = create()
                binding = TerminalBinding(token, backend.environment_id, time.time() + 60, 20)
                owner = TerminalBroker.create(directory / "persistent", binding, backend)
                first = req("first", "echo A-start >> /tmp/order; sleep 0.2; echo A-end >> /tmp/order")
                second = req("second", "echo B >> /tmp/order; " + WRITER)
                await asyncio.gather(owner.execute(first), owner.execute(second))
                await wait_writer(backend)
                observed = await owner.execute(req("read", "cat /tmp/order; test -s /tmp/ticks"))
                require(observed == TerminalResult(0, b"A-start\nA-end\nB\n"), "serial order or persistent service missing")
                owner.close()
                recovered = TerminalBroker.open(directory / "persistent", binding, backend)
                await recovered.execute(first)
                require(backend.calls == 3, "completed request was replayed after controller reopen")
                await recovered.abort()
                record("serial effects and persistent service survive controller reopen", backend, recovered)

            asyncio.run(persistent())

            for ending in ("timeout", "cancel", "lost_stop_reply"):
                async def interrupted(ending=ending):
                    backend = create()
                    owner = TerminalBroker.create(directory / ending,
                        TerminalBinding(token, backend.environment_id, time.time() + 60, 20), backend)
                    if ending == "cancel":
                        backend.allow_stop.clear()
                    backend.lose_stop_reply = ending == "lost_stop_reply"
                    active = asyncio.create_task(owner.execute(req("hung", HUNG, 15 if ending == "cancel" else 2)))
                    await wait_writer(backend)
                    if ending == "cancel":
                        for _ in range(3):
                            active.cancel()
                            await asyncio.sleep(0.02)
                            require(not active.done(), "cancellation abandoned active stop")
                        backend.allow_stop.set()
                    try:
                        await active
                    except asyncio.CancelledError:
                        require(ending == "cancel", "unexpected cancellation")
                    except TimeoutError:
                        require(ending == "timeout", "lost stop failure was discarded")
                    except BaseExceptionGroup:
                        require(ending == "lost_stop_reply", "unexpected grouped failure")
                        require(owner.phase == "fenced", "unknown stop acknowledged as complete")
                        await owner.abort()
                    else:
                        raise AssertionError("hung request was reported completed")
                    try:
                        await owner.execute(req("cannot_retry", "echo invalid"))
                    except TerminalFenced:
                        pass
                    else:
                        raise AssertionError("interrupted trial admitted another command")
                    record(ending + " drains a real detached task writer", backend, owner)

                asyncio.run(interrupted())

            backend = create()
            binding = TerminalBinding(token, backend.environment_id, time.time() + 60, 20)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--crash-child",
                                     str(directory / "crash"), backend.environment_id, token, str(binding.deadline_unix)],
                                    timeout=20, capture_output=True, check=False)
            require(result.returncode == 73, f"controller crash fixture failed: {result.stderr.decode()}")
            require(backend._container().attrs["State"]["Running"], "fixture did not leave a live task environment")
            recovered = TerminalBroker.open(directory / "crash", binding, backend)

            async def recover():
                require(recovered.phase == "fenced", "crashed effect was not fenced")
                for request in (req("crashed_effect", WRITER), req("replacement", "echo invalid")):
                    try:
                        await recovered.execute(request)
                    except TerminalFenced:
                        pass
                    else:
                        raise AssertionError("crashed effect was retried")
                await recovered.abort()

            asyncio.run(recover())
            require(backend.calls == 0, "recovery launched a command")
            record("controller death preserves intent and recovery drains without replay", backend, recovered)
        report["status"] = "passed"
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if owned:
            try:
                for container in client.containers.list(all=True, filters=label):
                    container.remove(force=True)
                report["remaining_containers"] = [c.id for c in client.containers.list(all=True, filters=label)]
            except BaseException as exc:
                report["cleanup_errors"].append(f"{type(exc).__name__}: {exc}")
            if report["cleanup_errors"] or report.get("remaining_containers"):
                report["status"] = "failed"
        client.close()
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["--crash-child"]:
        child(*sys.argv[2:])
    raise SystemExit(main())
