#!/usr/bin/env python3
"""Native Harbor Compose identity, agent user, shared verifier and durable drain.

Server-only synthetic fixture; no model calls, benchmark data, extra Compose,
environment subclass or image pulls. Run in a bounded root systemd unit with
--cleanup-only as ExecStopPost for the same fresh --owner-token. This checks
the native terminal boundary, not the production agent or model-only egress.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["HARBOR_TELEMETRY"] = "0"

from harbor.agents.base import BaseAgent
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial

from scripts.check_docker_terminal import cli, empty_cgroup, req
from taste.brains.azure_goal_handoff import _write
from taste.brains.docker_terminal import (
    COMPOSE_PROJECT_LABEL,
    DockerTerminalBackend,
    DockerTerminalBinding,
)
from taste.brains.terminal_broker import TerminalBinding, TerminalBroker


def paths(token):
    if re.fullmatch(r"[0-9a-f]{32}", token) is None:
        raise ValueError("a fresh lowercase 32-hex fixture owner token is required")
    return Path("/root") / f"taste-native-{token}", f"taste-native-{token}__env"


def write(root, name, value):
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        _write(fd, name, (json.dumps(value, sort_keys=True) + "\n").encode())
    finally:
        os.close(fd)


def owned(project):
    return cli("ps", "-aq", "--no-trunc", "--filter",
               f"label={COMPOSE_PROJECT_LABEL}={project}").stdout.decode().splitlines()


def cleanup(token):
    root, project = paths(token)
    removed = []
    binding_path = root / "binding.json"
    if binding_path.exists():
        binding = DockerTerminalBinding(**json.loads(binding_path.read_bytes()))
        assert binding.owner_token == token and binding.compose_project == project
        drained = root / "drain.json"
        if drained.exists():
            assert json.loads(drained.read_bytes()) == asdict(binding)
            assert empty_cgroup(binding.cgroup_path)
        else:
            DockerTerminalBackend(binding).stop_and_confirm()
            write(root, "drain.json", asdict(binding))
    # Before admission, the root runner's unique reserved project is the only
    # available ownership record. Inspect each selected ID again before removal.
    for container_id in owned(project):
        info = json.loads(cli("inspect", container_id).stdout)[0]
        assert info["Id"] == container_id
        assert info["Config"]["Labels"][COMPOSE_PROJECT_LABEL] == project
        cgroup = None
        if info["State"]["Running"]:
            pid = info["State"]["Pid"]
            cgroup = next(line[3:] for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines()
                          if line.startswith("0::"))
            assert container_id in cgroup
            cli("stop", "--time", "1", container_id)
        assert not json.loads(cli("inspect", container_id).stdout)[0]["State"]["Running"]
        if cgroup:
            assert empty_cgroup(cgroup)
        cli("rm", container_id)
        removed.append(container_id)
    networks = cli("network", "ls", "-q", "--no-trunc", "--filter",
                   f"label={COMPOSE_PROJECT_LABEL}={project}").stdout.decode().splitlines()
    for network in networks:
        info = json.loads(cli("network", "inspect", network).stdout)[0]
        assert info["Labels"][COMPOSE_PROJECT_LABEL] == project and not info["Containers"]
        cli("network", "rm", network)
    assert not owned(project)
    return {"removed_containers": removed, "removed_networks": networks,
            "original_binding_drained": binding_path.exists()}


class NativeTerminalProbe(BaseAgent):
    def __init__(self, *, owner_token, image_id, mode, **kwargs):
        super().__init__(**kwargs)
        self.token, self.image_id, self.mode = owner_token, image_id, mode
        self.root, self.project = paths(owner_token)

    @staticmethod
    def name():
        return "taste-native-terminal-probe"

    def version(self):
        return "1"

    async def setup(self, environment):
        assert type(environment) is DockerEnvironment
        assert not environment.extra_docker_compose_paths
        assert environment.default_user == "1000:1000"
        ids = owned(self.project)
        assert len(ids) == 1
        info = json.loads(cli("inspect", ids[0]).stdout)[0]
        assert info["HostConfig"]["NanoCpus"] == 1_000_000_000
        assert info["HostConfig"]["Memory"] == 256 * 1024 * 1024
        assert {m["Destination"] for m in info["Mounts"]} == {
            "/logs/agent", "/logs/verifier", "/logs/artifacts"}
        assert "taste.terminal.owner" not in info["Config"]["Labels"]
        self.backend = DockerTerminalBackend.admit("/var/run/docker.sock", ids[0], self.token,
            time.time() + 150, compose_project=self.project, image_id=self.image_id,
            exec_user=environment.default_user, output_limit=4096)
        write(self.root, "binding.json", asdict(self.backend.binding))
        write(self.root, "inspect.json", info)

    async def run(self, instruction, environment, context):
        assert environment.default_user == self.backend.binding.exec_user
        assert environment.task_env_config.workdir == "/tmp"
        backend = DockerTerminalBackend(DockerTerminalBinding(**json.loads(
            (self.root / "binding.json").read_bytes())))
        binding = TerminalBinding(self.token, backend.environment_id, backend.binding.deadline_unix, 20)
        broker = TerminalBroker.create(self.root / "terminal", binding, backend)
        command = "id -u; id -g; pwd; printf native > /tmp/native-agent-file"
        try:
            result = await broker.execute(req("first", command, cwd=environment.task_env_config.workdir))
            assert result.return_code == 0 and result.stdout == b"1000\n1000\n/tmp\n"
        finally:
            broker.close()
        recovered = TerminalBroker.open(self.root / "terminal", binding, DockerTerminalBackend(backend.binding))
        try:
            assert await recovered.execute(req("first", command, cwd="/tmp")) == result
            if self.mode == "killed-owner":
                active = asyncio.create_task(recovered.execute(req("interrupted",
                    "echo started > /tmp/native-started; sleep 60", timeout=20)))
                deadline = time.monotonic() + 10
                while (await asyncio.to_thread(cli, "cp", f"{backend.environment_id}:/tmp/native-started",
                                                "-", check=False)).returncode != 0:
                    assert time.monotonic() < deadline and not active.done()
                    await asyncio.sleep(0.02)
                write(self.root, "owner-killed.json", {"container_id": backend.environment_id,
                                                       "terminal_command_started": True})
                os.kill(os.getpid(), 9)
            assert (await recovered.execute(req("append", "printf -- '-retained' >> /tmp/native-agent-file"))).return_code == 0
            recovered.seal_for_grading()
            write(self.root, "terminal-events.json", recovered.events())
        finally:
            recovered.close()
        context.metadata = {"fixture_only": True, "paid_calls": 0, "native_container": backend.environment_id}


async def check(args):
    root, project = paths(args.owner_token)
    assert os.geteuid() == 0
    assert not owned(project)
    assert json.loads(cli("image", "inspect", args.image).stdout)[0]["Id"] == args.image
    root.mkdir(mode=0o700)
    task = root / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("Create /tmp/native-agent-file as the configured agent user.\n")
    (task / "task.toml").write_text('schema_version = "1.4"\n[agent]\nuser = "1000:1000"\n'
        'timeout_sec = 120\n[verifier]\ntimeout_sec = 30\n[environment]\n'
        f'docker_image = "{args.image}"\ncpus = 1\nmemory_mb = 256\nworkdir = "/tmp"\n'
        'network_mode = "public"\n')
    (task / "tests/test.sh").write_text('#!/bin/bash\nset -euo pipefail\n'
        'test "$(cat /tmp/native-agent-file)" = native-retained\n'
        'test "$(stat -c %u:%g /tmp/native-agent-file)" = 1000:1000\n'
        'echo 1 > /logs/verifier/reward.txt\n')
    config = TrialConfig(task=TaskConfig(path=task), trial_name=root.name, trials_dir=root / "trials",
        agent=AgentConfig(import_path="scripts.check_native_harbor_terminal:NativeTerminalProbe",
                          kwargs={"owner_token": args.owner_token, "image_id": args.image, "mode": args.mode}),
        environment=EnvironmentConfig(delete=False))
    try:
        trial = await Trial.create(config)
        result = await trial.run()
        assert result.exception_info is None, result.exception_info
        assert result.verifier_result.rewards == {"reward": 1.0}
        # Harbor 0.23 uses Compose down even with delete=False. Drain must use
        # the original binding and cgroup after the daemon removes metadata.
        assert not owned(project)
    finally:
        drained = cleanup(args.owner_token)
    report = {"status": "passed", "fixture_only": True, "paid_calls": 0,
              "reward": result.verifier_result.rewards, "native_environment": True,
              "agent_user": "1000:1000", "extra_compose": [], "cleanup": drained}
    write(root, "report.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--image")
    parser.add_argument("--cleanup-only", action="store_true")
    parser.add_argument("--mode", choices=("complete", "killed-owner"), default="complete")
    args = parser.parse_args()
    paths(args.owner_token)
    if args.cleanup_only:
        print(json.dumps(cleanup(args.owner_token)))
    else:
        if args.image is None or re.fullmatch(r"sha256:[0-9a-f]{64}", args.image) is None:
            parser.error("a cached immutable image ID is required")
        asyncio.run(check(args))
