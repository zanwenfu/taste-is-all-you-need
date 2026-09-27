#!/usr/bin/env python3
"""Server-only official Harbor trial with a scripted, unprivileged terminal worker.

Requires the separately pinned Harbor environment. This is a deterministic
integration fixture, not a Terminal Bench score or a paid-model evaluation.
Run inside a bounded root systemd unit with owner-label-scoped ExecStopPost
cleanup using check_docker_terminal.py. No image pulls or model calls.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pwd
import re
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial

from scripts.check_docker_terminal import cleanup, cli, empty_cgroup
from taste.benchmarks.output_snapshot import OutputSnapshotError, download_snapshot
from taste.brains.docker_terminal import OWNER_LABEL, DockerTerminalBackend
from taste.brains.owned_thread import start_owned_thread
from taste.brains.terminal_broker import TerminalBinding, TerminalBroker, _settle
from taste.brains.terminal_service import TerminalCredential, TerminalGrant, TerminalService


class BrokerDockerEnvironment(DockerEnvironment):
    """Let Harbor grade, then prove broker drainage before deleting metadata."""

    async def prepare_logs_for_host(self):
        service = getattr(self, "taste_service", None)
        if service is None or service.broker.phase != "stopped":
            await super().prepare_logs_for_host()

    async def stop(self, delete):
        service = getattr(self, "taste_service", None)
        if service is not None:
            await self.prepare_logs_for_host()
            await service.close()
            assert service.broker.phase == "stopped"
            self.taste_drained_before_delete = True
        await super().stop(delete)


class SnapshotDockerEnvironment(BrokerDockerEnvironment):
    """Controlled single-container fixture with no host output bind mounts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self._mounts) == 3
        assert {m["target"] for m in self._mounts} == {"/logs/agent", "/logs/verifier", "/logs/artifacts"}
        assert all(set(m) == {"type", "source", "target"} and m["type"] == "bind" for m in self._mounts)
        self.output_targets = {m["target"]: Path(m["source"]) for m in self._mounts}
        self._mounts = []
        self.snapshots = []

    @property
    def capabilities(self):
        return super().capabilities.model_copy(update={"mounted": False, "stream": False})

    async def start(self, *args, **kwargs):
        await super().start(*args, **kwargs)
        await self.ensure_dirs(list(self.output_targets))

    async def prepare_logs_for_host(self):
        # Every exported file is created by the host snapshot reader. There are
        # no mounted guest files to chown or inspect here.
        pass

    async def download_dir(self, source_dir, target_dir):
        if self.output_targets.get(source_dir) != Path(target_dir):
            raise OutputSnapshotError("fixture only admits its three recorded output destinations")
        target = Path(target_dir)
        if target.is_symlink():
            raise OutputSnapshotError("snapshot destination is a link")
        target.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.chmod(0o700)
        operation = start_owned_thread(download_snapshot, self.taste_service.broker.backend,
            source_dir, target, deadline_unix=self.taste_service.broker.binding.deadline_unix)
        try:
            await asyncio.wait((operation,))
            summary = operation.result()
            self.snapshots.append({"source": source_dir, **summary})
        except BaseException:
            # Cancellation cannot leave a host writer running after the owner
            # removes the container or releases its trial directory.
            await _settle(operation)
            raise

    async def download_file(self, *args, **kwargs):
        raise OutputSnapshotError("fixture does not admit individual file downloads")

    async def download_dir_filtered(self, *args, **kwargs):
        raise OutputSnapshotError("fixture does not admit filtered downloads")

    async def download_dir_with_exclusions(self, *args, **kwargs):
        raise OutputSnapshotError("fixture does not admit filtered downloads")

    async def service_download_dir(self, source_dir, target_dir, *, service=None):
        if service not in (None, "main"):
            raise OutputSnapshotError("fixture has no sidecar service")
        await self.download_dir(source_dir, target_dir)

    async def service_download_file(self, *args, **kwargs):
        raise OutputSnapshotError("fixture does not admit individual file downloads")


class ScriptedTerminalAgent(BaseAgent):
    def __init__(self, *, owner_token, controller_dir, worker_python, worker_uid, worker_gid, mode, **kwargs):
        super().__init__(**kwargs)
        self.owner_token = owner_token
        self.controller_dir = Path(controller_dir)
        self.worker_python, self.worker_uid, self.worker_gid = worker_python, worker_uid, worker_gid
        self.mode = mode
        self.owner = self.service = self.backend = None
        self.worker_observation = None

    @staticmethod
    def name():
        return "taste-scripted-terminal-validation"

    def version(self):
        return "1"

    async def setup(self, environment):
        assert isinstance(environment, BrokerDockerEnvironment)
        assert environment.default_user is None
        ids = cli("ps", "-aq", "--no-trunc", "--filter", f"label={OWNER_LABEL}={self.owner_token}").stdout.splitlines()
        assert len(ids) == 1, "fixture must have exactly one authoritative container"
        assert not json.loads(cli("inspect", ids[0].decode()).stdout)[0]["Mounts"], "fixture must not bind host outputs"
        self.backend = DockerTerminalBackend.admit("/var/run/docker.sock", ids[0].decode(), self.owner_token,
                                                   time.time() + 45, output_limit=8192)
        binding = TerminalBinding(self.owner_token, self.backend.environment_id, self.backend.binding.deadline_unix, 10)
        self.owner = TerminalBroker.create(self.controller_dir / "ledger", binding, self.backend)
        self.credential = TerminalCredential(f"/run/taste-harbor-{self.owner_token}/terminal.sock", 0,
            self.worker_uid, TerminalGrant(binding, "scripted_worker", 10), os.urandom(32).hex())
        self.service = TerminalService(self.owner, [self.credential])
        environment.taste_service = self.service
        await self.service.start(worker_gid=self.worker_gid)

    async def run(self, instruction, environment, context):
        assert "agent-result" in instruction
        context.n_input_tokens = context.n_output_tokens = context.n_cache_tokens = 0
        context.cost_usd = 0.0
        context.metadata = {"fixture": "scripted-terminal", "mode": self.mode}
        checkout = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="taste-harbor-worker-", dir="/tmp") as temporary:
            directory = Path(temporary)
            os.chown(directory, self.worker_uid, self.worker_gid)
            credential_path = directory / "credential.json"
            credential_path.write_text(json.dumps(self.credential.to_dict()))
            credential_path.chmod(0o600)
            os.chown(credential_path, self.worker_uid, self.worker_gid)
            # This trusted fixture uses the actual TerminalTools/RPC client and
            # checks denial of Docker and private ledger access in its process.
            child_env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(checkout), "PYTHONDONTWRITEBYTECODE": "1"}
            process = await asyncio.create_subprocess_exec(
                self.worker_python, str(checkout / "scripts/check_terminal_service.py"),
                "--child-credential", str(credential_path), "--child-mode", "killed" if self.mode == "killed" else "complete",
                "--ledger", str(self.controller_dir / "ledger/terminal.sqlite3"),
                user=self.worker_uid, group=self.worker_gid, extra_groups=(), start_new_session=True,
                env=child_env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                if self.mode == "killed":
                    deadline = time.monotonic() + 8
                    while (await asyncio.to_thread(cli, "cp", f"{self.backend.environment_id}:/tmp/started", "-", check=False)).returncode != 0:
                        assert time.monotonic() < deadline, "fixture terminal command did not enter"
                        await asyncio.sleep(0.02)
                    process.kill()
                stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
                if self.mode == "killed":
                    assert process.returncode == -9
                    async with asyncio.timeout(15):
                        while self.owner.phase != "stopped":
                            await asyncio.sleep(0.02)
                    self.worker_observation = {"worker_killed": True, "terminal_command_drained": True}
                    raise RuntimeError("injected worker death; terminal task stopped")
                assert process.returncode == 0, stderr.decode(errors="replace")
                self.worker_observation = json.loads(stdout)
                assert self.worker_observation["status"] == "passed"
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.communicate()
        self.service.seal_for_grading()
        assert self.owner.phase == "sealed"
        context.metadata["agent_admission_sealed"] = True


async def check(args):
    assert os.geteuid() == 0 and os.environ.get("HARBOR_TELEMETRY") == "0"
    assert re.fullmatch(r"[0-9a-f]{32}", args.owner_token)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", args.image)
    account = pwd.getpwnam(args.worker_user)
    assert account.pw_uid > 0
    assert set(os.getgrouplist(account.pw_name, account.pw_gid)) == {account.pw_gid}
    assert json.loads(cli("image", "inspect", args.image).stdout)[0]["Id"] == args.image
    assert not cli("ps", "-aq", "--filter", f"label={OWNER_LABEL}={args.owner_token}").stdout.strip()
    directory = Path("/root") / f"taste-harbor-fixture-{args.owner_token}"
    directory.mkdir(mode=0o700)
    controller = directory / "controller"
    controller.mkdir(mode=0o700)
    task = directory / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("Write the exact bytes correct to /tmp/agent-result.\n")
    (task / "task.toml").write_text(
        'schema_version = "1.4"\n[agent]\ntimeout_sec = 25\n'
        '[verifier]\ntimeout_sec = 15\n[environment]\n'
        f'docker_image = "{args.image}"\nbuild_timeout_sec = 30\ncpus = 1\nmemory_mb = 256\n'
        'network_mode = "public"\n'
    )
    canary = controller / "private-canary"
    canary.write_bytes(b"1\n")
    reward_code = "Path('/logs/verifier/reward.txt').write_text('1\\n')"
    if args.mode == "reward-symlink":
        reward_code = f"Path('/logs/verifier/reward.txt').symlink_to({str(canary)!r})"
    elif args.mode == "reward-fifo":
        reward_code = "import os; os.mkfifo('/logs/verifier/reward.txt')"
    elif args.mode == "reward-oversize":
        reward_code = "Path('/logs/verifier/reward.txt').write_bytes(b'1' * (4 * 1024 * 1024 + 1))"
    (task / "tests/test.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\npython3 - <<'PY'\n"
        "from pathlib import Path\n"
        "assert Path('/tmp/agent-result').read_bytes() == b'correct'\n"
        f"{reward_code}\nPY\n"
    )
    overlay = controller / "compose.json"
    # The fixture needs no network. Record the explicit Docker override; it is
    # not a claim about the network policy of a published benchmark task.
    overlay.write_text(json.dumps({"services": {"main": {
        "labels": {OWNER_LABEL: args.owner_token}, "restart": "no", "pull_policy": "never",
        "network_mode": "none", "pids_limit": 64, "security_opt": ["no-new-privileges:true"],
    }}}))
    config = TrialConfig(
        task=TaskConfig(path=task), trial_name=f"taste-fixture-{args.owner_token}", trials_dir=directory / "trials",
        agent=AgentConfig(import_path="scripts.check_harbor_trial:ScriptedTerminalAgent", kwargs={
            "owner_token": args.owner_token, "controller_dir": str(controller),
            "worker_python": args.worker_python, "worker_uid": account.pw_uid, "worker_gid": account.pw_gid,
            "mode": args.mode}, override_setup_timeout_sec=10),
        environment=EnvironmentConfig(import_path="scripts.check_harbor_trial:SnapshotDockerEnvironment",
                                      extra_docker_compose=[overlay], delete=False),
    )
    report = {"status": "failed", "mode": args.mode, "cleanup_errors": [], "fixture_only": True, "paid_calls": 0,
              "task_files_sha256": {str(p.relative_to(task)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(task.rglob("*")) if p.is_file()},
              "compose_overlay_sha256": hashlib.sha256(overlay.read_bytes()).hexdigest()}
    trial = None
    try:
        trial = await Trial.create(config)
        result = await trial.run()
        if args.mode == "complete":
            assert result.exception_info is None, result.exception_info
            assert result.verifier_result is not None and result.verifier_result.rewards == {"reward": 1.0}
        elif args.mode == "killed":
            assert result.exception_info is not None and "injected worker death" in result.exception_info.exception_message
            assert result.verifier_result is None, "uncertain terminal execution must not produce a reward"
        else:
            assert result.exception_info is not None and result.exception_info.exception_type == "DownloadVerifierDirError"
            assert result.verifier_result is None, "unsafe output must not produce a reward"
            assert not list(trial.paths.verifier_dir.iterdir()), "unsafe snapshot must not publish partial files"
        assert canary.read_bytes() == b"1\n"
        agent = trial.agent
        assert agent.owner.phase == "stopped" and trial.agent_environment.taste_drained_before_delete
        assert empty_cgroup(agent.backend.binding.cgroup_path)
        assert not cli("ps", "-aq", "--filter", f"label={OWNER_LABEL}={args.owner_token}").stdout.strip()
        assert json.loads(cli("image", "inspect", args.image).stdout)[0]["Id"] == args.image
        assert len([event for kind, event in agent.owner.events() if kind == "intent"]) == 1
        report.update(status="passed", official_rewards=result.verifier_result.rewards if result.verifier_result else None,
            worker=agent.worker_observation, broker_events=agent.owner.events(),
            docker_binding=asdict(agent.backend.binding), drained_before_container_delete=True,
            original_cgroup_empty=True, cached_image_retained=True, host_output_mounts=False,
            snapshots=trial.agent_environment.snapshots, harbor_result=str(trial.paths.result_path))
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if trial is not None and getattr(trial.agent, "service", None) is not None:
            try:
                await trial.agent.service.close()
                trial.agent.owner.close()
                Path(trial.agent.credential.socket_path).parent.rmdir()
            except BaseException as exc:
                report["cleanup_errors"].append(f"{type(exc).__name__}: {exc}")
        try:
            report["fallback_removed"] = cleanup(args.owner_token)
        except BaseException as exc:
            report["cleanup_errors"].append(f"{type(exc).__name__}: {exc}")
        if report["cleanup_errors"]:
            report["status"] = "failed"
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    assert report["status"] == "passed", report["cleanup_errors"]
    print(json.dumps({"status": "passed", "mode": args.mode, "official_fixture_reward": report["official_rewards"],
                      "paid_calls": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--worker-user", default="bugbash")
    parser.add_argument("--worker-python", required=True)
    parser.add_argument("--mode", choices=("complete", "killed", "reward-symlink", "reward-fifo", "reward-oversize"),
                        default="complete")
    parser.add_argument("--output", required=True, type=Path)
    asyncio.run(check(parser.parse_args()))


if __name__ == "__main__":
    main()
