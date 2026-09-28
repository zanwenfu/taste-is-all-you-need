#!/usr/bin/env python3
"""Bounded server fixture: actual Azure goal, artifact transfer, separate Harbor verifier.

Only model HTTP is mocked. The trusted verifier script is installed into a
second cached-image container during fixture setup. No image builds/pulls, host
output mounts, task network or paid calls. Use --cleanup-only in an independent
systemd ExecStopPost, and --observe-owner-death after a kill-mode unit ends.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from contextlib import suppress
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["HARBOR_TELEMETRY"] = "0"

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial

from scripts.check_azure_harbor import OwnedAzureEnvironment, cleanup_fixture, owner_directory
from scripts.check_docker_terminal import cleanup, cli, empty_cgroup
from taste.benchmarks.artifact_handoff import ArtifactHandoff, ArtifactTarget
from taste.benchmarks.output_snapshot import (
    OutputSnapshotError,
    _private_directory,
    download_snapshot,
)
from taste.brains.azure_goal_handoff import _read, _write
from taste.brains.docker_terminal import OWNER_LABEL, DockerTerminalBackend, DockerTerminalBinding
from taste.brains.owned_thread import start_owned_thread
from taste.brains.process_scope import SystemdManager, _owned_call
from taste.brains.terminal_broker import TerminalConflict, _settle

MODES = ("complete", "missing", "link", "tamper", "manifest-failure", "reward-zero", "reward-missing", "reward-nan",
         "cleanup-failure", "kill-copy", "kill-verifier", "service-complete", "service-missing", "service-restarted",
         "service-stop-failure", "service-cleanup-failure", "kill-service-copy", "kill-service-verifier")

# Trusted fixture service. The main container shares only this network namespace;
# it cannot access the service's filesystem or the host. No external network.
HELPER_CODE = '''from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ready")
    def do_POST(self):
        assert self.path == "/submit" and self.headers["Content-Length"] == "7"
        value = self.rfile.read(7)
        assert value == b"correct"
        Path("/tmp/service-evidence").mkdir(exist_ok=True)
        Path("/tmp/service-evidence/raw.bin").write_bytes(bytes([0,255]) + value)
        Path("/tmp/service-result").write_bytes(b"helper:" + value)
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *args): pass
HTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
'''


def has_helper(mode):
    return mode.startswith("service-") or mode.startswith("kill-service-")


def fixture_root(token):
    owner_directory(token)  # Validate before constructing any host path.
    return Path("/root") / f"taste-harbor-separate-{token}"


def verifier_token(token):
    owner_directory(token)
    return hashlib.sha256(("separate-verifier:" + token).encode()).hexdigest()[:32]


def helper_token(token):
    owner_directory(token)
    return hashlib.sha256(("artifact-helper:" + token).encode()).hexdigest()[:32]


def save(root, name, value):
    fd = _private_directory(root)
    try:
        _write(fd, name, (json.dumps(value, sort_keys=True) + "\n").encode())
    finally:
        os.close(fd)


def read(root, name):
    fd = _private_directory(root)
    try:
        return json.loads(_read(fd, name, 1024 * 1024, uid=os.geteuid(), mode=0o600))
    finally:
        os.close(fd)


def stop_verifier(root):
    binding_data = read(root, "verifier-binding.json")
    binding = DockerTerminalBinding(**binding_data)
    expected = {"binding": binding_data, "stopped": True}
    if (root / "verifier-drain.json").exists():
        if read(root, "verifier-drain.json") != expected or not empty_cgroup(binding.cgroup_path):
            raise OutputSnapshotError("verifier drainage no longer matches its original identity")
    else:
        DockerTerminalBackend(binding).stop_and_confirm()
        save(root, "verifier-drain.json", expected)
    return expected


def stop_helper(root):
    binding_data = read(root, "helper-binding.json")
    binding = DockerTerminalBinding(**binding_data)
    if (root / "helper-drain.json").exists():
        result = read(root, "helper-drain.json")
        assert result["binding"] == binding_data and result["stopped"] and empty_cgroup(binding.cgroup_path)
        return result
    result = {"binding": binding_data, "stopped": True, "identity_conflict": None, "removed_replacement": []}
    try:
        DockerTerminalBackend(binding).stop_and_confirm()
    except TerminalConflict as exc:
        # A restarted service is never re-admitted for collection. Independent
        # cleanup can remove it under the original ownership label while
        # retaining the failed identity check and proving physical drainage.
        result["identity_conflict"] = str(exc)
        result["removed_replacement"] = cleanup(binding.owner_token)
        assert empty_cgroup(binding.cgroup_path)
    save(root, "helper-drain.json", result)
    return result


def cleanup_separate(token):
    root, errors, report = fixture_root(token), [], {}
    try:
        report["agent"] = cleanup_fixture(token)
    except BaseException as exc:
        errors.append(exc)
    if (root / "verifier-binding.json").exists():
        try:
            report["verifier"] = stop_verifier(root)
        except BaseException as exc:
            errors.append(exc)
    if (root / "helper-binding.json").exists():
        try:
            report["helper"] = stop_helper(root)
        except BaseException as exc:
            errors.append(exc)
    try:
        report["removed_verifiers"] = cleanup(verifier_token(token))
    except BaseException as exc:
        errors.append(exc)
    try:
        report["removed_helpers"] = cleanup(helper_token(token))
    except BaseException as exc:
        errors.append(exc)
    if errors:
        raise BaseExceptionGroup("separate trial cleanup remains incomplete", errors)
    return report


class SeparateAzureEnvironment(OwnedAzureEnvironment):
    """Controlled one-agent/one-verifier topology; no guest-writable host mounts."""
    groups: ClassVar[dict] = {}

    def __init__(self, *args, owner_token, mode, deadline_unix, **kwargs):
        # The shared fixture's three-mount constructor cannot admit the
        # separate verifier's one reward mount. Validate both shapes here.
        DockerEnvironment.__init__(self, *args, **kwargs)
        targets = {m["target"] for m in self._mounts}
        assert targets in ({"/logs/agent", "/logs/verifier", "/logs/artifacts"}, {"/logs/verifier"})
        assert len(targets) == len(self._mounts)
        assert all(set(m) == {"type", "source", "target"} and m["type"] == "bind" for m in self._mounts)
        self.role = "verifier" if len(targets) == 1 else "agent"
        self.output_targets = {m["target"]: Path(m["source"]) for m in self._mounts}
        self._mounts, self.snapshots = [], []
        self.token, self.mode, self.deadline = owner_token, mode, deadline_unix
        self.root = fixture_root(owner_token)
        self.group = type(self).groups.setdefault(owner_token, {})
        assert self.role not in self.group, "a role cannot be restarted within this trial"
        if self.role == "verifier":
            agent = self.group["agent"]
            if not agent.taste_owner.closed:
                raise OutputSnapshotError("agent owner was not drained before separate verification")
            if has_helper(mode) and (not agent.helper_stopped or agent.helper_stop_error):
                raise OutputSnapshotError("helper service has unconfirmed cleanup")
            manifest = agent.trial_paths.artifacts_dir / "manifest.json"
            fd = _private_directory(manifest.parent)
            try:
                info = manifest.lstat()
                if info.st_mode & 0o022:
                    raise OutputSnapshotError("Harbor collection manifest is writable by another user")
                payload = _read(fd, manifest.name, 65536, uid=os.geteuid(), mode=info.st_mode & 0o777)
            finally:
                os.close(fd)
            receipts = agent.handoff.seal_harbor_manifest(payload, manifest.parent)
            save(self.root, "artifacts-sealed.json", receipts)
        self.group[self.role] = self
        self.stopped, self.stop_error, self.death = False, None, None
        self.death_cancelled = False
        self.preparing = True
        self.helper_backend, self.helper_stopped, self.helper_stop_error = None, False, None
        self.handoff = None
        self.overlay = self.root / f"{self.role}-compose.json"
        token = verifier_token(owner_token) if self.role == "verifier" else owner_token
        network = "service:evidence" if self.role == "agent" and has_helper(mode) else "none"
        save(self.root, self.overlay.name, {"services": {"main": {"labels": {OWNER_LABEL: token},
            "restart": "no", "pull_policy": "never", "network_mode": network, "pids_limit": 64,
            "security_opt": ["no-new-privileges:true"]}}})
        if self.role == "verifier":
            save(self.root, "verifier-intent.json", {"owner_token": token, "deadline_unix": deadline_unix})

    @property
    def _docker_compose_paths(self):
        paths = super()._docker_compose_paths
        return [*paths, self.overlay] if hasattr(self, "overlay") else paths

    async def _run_docker_compose_command(self, command, check=True, timeout_sec=None, **kwargs):
        remaining = self.deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("fixture's original container lifetime expired")
        return await super()._run_docker_compose_command(command, check=check,
            timeout_sec=min(timeout_sec or 15, 15, remaining), **kwargs)

    async def start(self, *args, **kwargs):
        try:
            await DockerEnvironment.start(self, *args, **kwargs)
        finally:
            self.preparing = False
        await self.ensure_dirs(list(self.output_targets))
        if self.role == "verifier" and self.mode in {"kill-verifier", "kill-service-verifier"}:
            self.death = asyncio.create_task(self._kill_during_verifier())

    async def _upload_environment_dir_after_start(self):
        if self.role == "agent" and has_helper(self.mode):
            token = helper_token(self.token)
            ids = cli("ps", "-aq", "--no-trunc", "--filter", f"label={OWNER_LABEL}={token}").stdout.splitlines()
            assert len(ids) == 1
            info = json.loads(cli("inspect", ids[0].decode()).stdout)[0]
            assert not info["Mounts"] and info["HostConfig"]["NetworkMode"] == "none"
            self.helper_backend = DockerTerminalBackend.admit("/var/run/docker.sock", ids[0].decode(), token,
                                                            self.deadline, output_limit=8192)
            save(self.root, "helper-binding.json", asdict(self.helper_backend.binding))
        if self.role == "verifier":
            # Save the exact verifier identity before Harbor's trusted setup
            # upload. It is distinct from artifacts produced by the agent.
            token = verifier_token(self.token)
            ids = cli("ps", "-aq", "--no-trunc", "--filter", f"label={OWNER_LABEL}={token}").stdout.splitlines()
            assert len(ids) == 1
            info = json.loads(cli("inspect", ids[0].decode()).stdout)[0]
            assert not info["Mounts"] and info["Config"]["User"] in ("", "root", "0")
            self.taste_backend = DockerTerminalBackend.admit("/var/run/docker.sock", ids[0].decode(), token,
                                                          self.deadline, output_limit=8192)
            save(self.root, "verifier-binding.json", asdict(self.taste_backend.binding))
            await self.ensure_dirs(["/tests"])
        await super()._upload_environment_dir_after_start()

    def _artifacts(self):
        if self.handoff is None:
            root = self.trial_paths.artifacts_dir
            targets = [
                ArtifactTarget("/tmp/agent-result", root / "tmp/agent-result", "file"),
                ArtifactTarget("/tmp/evidence", root / "tmp/evidence", "directory"),
                ArtifactTarget("/logs/artifacts", self.output_targets["/logs/artifacts"], "directory"),
            ]
            services = {}
            if has_helper(self.mode):
                targets += [ArtifactTarget("/tmp/service-result", root / "tmp/service-result", "file", "evidence"),
                            ArtifactTarget("/tmp/service-evidence", root / "tmp/service-evidence", "directory", "evidence")]
                services["evidence"] = self.helper_backend
            self.handoff = ArtifactHandoff(self.taste_backend, targets, service_backends=services)
        return self.handoff

    async def _copy(self, source, target, kind, *, service=None):
        target = Path(target)
        if self.role == "agent" and source in self._artifacts().targets:
            item = self._artifacts()
            if source not in item.targets or item.targets[source].destination != target:
                raise OutputSnapshotError("unadmitted artifact target")
            # Harbor creates its output hierarchy; all parts must be private
            # and outside task write access before our snapshot reader runs.
            target.parent.mkdir(parents=True, exist_ok=True)
            path = target.parent
            while path != self.trial_paths.artifacts_dir.parent:
                assert not path.is_symlink()
                path.chmod(0o700)
                path = path.parent
            if kind == "directory":
                if target.is_symlink():
                    raise OutputSnapshotError("artifact directory is a link")
                target.mkdir(mode=0o700, exist_ok=True)
                target.chmod(0o700)
            receipt = await item.collect(source, target, kind, deadline_unix=self.deadline, service=service)
            save(self.root, "artifact-" + hashlib.sha256(source.encode()).hexdigest()[:16] + ".json", receipt)
            if self.mode == "manifest-failure" and source == "/tmp/agent-result":
                raise OutputSnapshotError("controlled failure after successful artifact capture")
            if self.mode == "tamper" and source == "/tmp/agent-result":
                target.write_bytes(b"changed")
            return receipt
        if kind != "directory" or self.output_targets.get(source) != target:
            raise OutputSnapshotError("unadmitted output destination")
        if target.is_symlink():
            raise OutputSnapshotError("output directory is a link")
        target.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.chmod(0o700)
        operation = start_owned_thread(download_snapshot, self.taste_backend, source, target,
                                       deadline_unix=self.deadline)
        try:
            await asyncio.wait((operation,))
            result = operation.result()
            self.snapshots.append({"source": source, **result})
        except BaseException:
            await _settle(operation)
            raise

    async def download_dir(self, source_dir, target_dir):
        return await self._copy(source_dir, target_dir, "directory")

    async def service_download_file(self, source_path, target_path, *, service=None):
        return await self._copy(source_path, target_path, "file", service=service)

    async def service_download_dir(self, source_dir, target_dir, *, service=None):
        return await self._copy(source_dir, target_dir, "directory", service=service)

    async def service_is_dir(self, path, *, service=None, user=None):
        if self.role == "agent" and path in self._artifacts().targets:
            target = self.handoff.targets[path]
            if target.service != (service or "main"):
                raise OutputSnapshotError("artifact type query changed its admitted service")
            return target.kind == "directory"
        return await super().service_is_dir(path, service=service, user=user)

    async def stop_service(self, service):
        if self.role != "agent" or service != "main" or not has_helper(self.mode):
            raise OutputSnapshotError("unadmitted service stop")
        if self.mode == "service-stop-failure":
            save(self.root, "main-stop-error.json", {"error": "controlled stop failure before main drainage"})
            raise OutputSnapshotError("controlled stop failure before main drainage")
        await self.taste_owner.close()
        binding = await self._artifacts().drain_main()
        save(self.root, "main-before-helper.json", {"binding": binding, "stopped": True})
        if self.mode == "service-restarted":
            cli("restart", self.helper_backend.environment_id)
        elif self.mode == "service-missing":
            cli("exec", self.helper_backend.environment_id, "rm", "/tmp/service-result")

    def _validate_upload(self, source, target, kind):
        handoff = self.group["agent"].handoff
        handoff.verify()
        admitted = handoff.targets.get(target)
        if admitted is None or admitted.destination != Path(source) or admitted.kind != kind:
            raise OutputSnapshotError("upload differs from the captured artifact")

    async def upload_file(self, source_path, target_path):
        if self.role == "verifier":
            self._validate_upload(source_path, target_path, "file")
        return await super().upload_file(source_path, target_path)

    async def upload_dir(self, source_dir, target_dir):
        if self.role == "verifier":
            trusted_setup = (self.preparing and Path(source_dir) == self.environment_dir and target_dir == "/tests")
            if not trusted_setup:
                self._validate_upload(source_dir, target_dir, "directory")
        return await super().upload_dir(source_dir, target_dir)

    async def stop(self, delete):
        if self.stopped:
            return
        if self.role == "agent":
            if self.helper_backend is not None:
                if self.mode == "service-cleanup-failure":
                    self.helper_stop_error = "controlled helper stop failure"
                    save(self.root, "helper-stop-error.json", {"error": self.helper_stop_error})
                    raise OutputSnapshotError(self.helper_stop_error)
                operation = start_owned_thread(_owned_call, partial(stop_helper, self.root))
                try:
                    await asyncio.wait((operation,))
                    operation.result()
                    self.helper_stopped = True
                except BaseException as exc:
                    self.helper_stop_error = type(exc).__name__
                    await _settle(operation)
                    raise
            await super().stop(delete)
        else:
            if self.death is not None:
                self.death_cancelled = True
                self.death.cancel()
                with suppress(asyncio.CancelledError):
                    await _settle(self.death)
            if self.mode == "cleanup-failure":
                self.stop_error = "controlled verifier stop failure"
                save(self.root, "verifier-stop-error.json", {"error": self.stop_error})
                raise RuntimeError(self.stop_error)
            operation = start_owned_thread(_owned_call, partial(stop_verifier, self.root))
            try:
                await asyncio.wait((operation,))
                operation.result()
                await DockerEnvironment.stop(self, delete)
            except BaseException as exc:
                self.stop_error = type(exc).__name__
                await _settle(operation)
                raise
        self.stopped = True

    async def _kill_during_verifier(self):
        while time.time() < self.deadline:
            operation = start_owned_thread(cli, "cp", f"{self.taste_backend.environment_id}:/tmp/verifier-started", "-", check=False)
            result = await _settle(operation)
            if self.death_cancelled:
                return
            if result.returncode == 0:
                save(self.root, "owner-killed.json", {"stage": "verifier", "container_id": self.taste_backend.environment_id})
                os.kill(os.getpid(), 9)
            await asyncio.sleep(0.02)
        raise AssertionError("verifier never reached its kill point")


def guard_verifier(trial):
    original = trial._run_separate_verifier

    async def guarded(**kwargs):
        result = await original(**kwargs)
        verifier = trial.agent_environment.group.get("verifier")
        if verifier is None or not verifier.stopped or verifier.stop_error:
            raise OutputSnapshotError("separate verifier result has unconfirmed cleanup")
        return result
    trial._run_separate_verifier = guarded


async def check(args):
    assert os.geteuid() == 0 and re.fullmatch(r"sha256:[0-9a-f]{64}", args.image)
    root = fixture_root(args.owner_token)
    root.mkdir(mode=0o700)
    assert not cli("ps", "-aq", "--filter", f"label={OWNER_LABEL}={args.owner_token}").stdout.strip()
    assert not cli("ps", "-aq", "--filter", f"label={OWNER_LABEL}={verifier_token(args.owner_token)}").stdout.strip()
    assert not cli("ps", "-aq", "--filter", f"label={OWNER_LABEL}={helper_token(args.owner_token)}").stdout.strip()
    assert json.loads(cli("image", "inspect", args.image).stdout)[0]["Id"] == args.image
    task = root / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("Write correct to /tmp/agent-result and binary evidence to /tmp/evidence.\n")
    artifact_count = 5 if has_helper(args.mode) else 3
    extra_artifacts = (', {source="/tmp/service-result", service="evidence"}, '
                       '{source="/tmp/service-evidence", service="evidence"}' if has_helper(args.mode) else '')
    if has_helper(args.mode):
        (task / "environment/docker-compose.yaml").write_text(json.dumps({"services": {
            "main": {"depends_on": {"evidence": {"condition": "service_healthy"}}},
            "evidence": {"image": args.image, "command": ["python3", "-u", "-c", HELPER_CODE],
                "labels": {OWNER_LABEL: helper_token(args.owner_token)}, "network_mode": "none",
                "restart": "no", "pull_policy": "never", "cpus": 1, "mem_limit": "256m", "pids_limit": 64,
                "security_opt": ["no-new-privileges:true"], "healthcheck": {
                    "test": ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/', timeout=1).read()"],
                    "interval": "1s", "timeout": "2s", "retries": 10}}}}, indent=2))
    (task / "task.toml").write_text('schema_version = "1.4"\n'
        f'artifacts = ["/tmp/agent-result", "/tmp/evidence"{extra_artifacts}]\n'
        '[agent]\ntimeout_sec = 100\n[verifier]\ntimeout_sec = 12\nenvironment_mode = "separate"\n'
        f'[verifier.environment]\ndocker_image = "{args.image}"\ncpus = 1\nmemory_mb = 256\nworkdir = "/tests"\n'
        f'[environment]\ndocker_image = "{args.image}"\ncpus = 1\nmemory_mb = 256\nbuild_timeout_sec = 30\n')
    script = "#!/bin/bash\nset -euo pipefail\n"
    if args.mode in {"kill-verifier", "kill-service-verifier"}:
        script += "touch /tmp/verifier-started\n(sleep 60 &)\nsleep 60\n"
    script += "python3 - <<'PY'\nfrom pathlib import Path\n"
    script += "assert Path('/tmp/agent-result').read_bytes() == b'correct'\n"
    script += "assert Path('/tmp/evidence/raw.bin').read_bytes() == bytes([0,255]) + b'data'\n"
    if has_helper(args.mode):
        script += "assert Path('/tmp/service-result').read_bytes() == b'helper:correct'\n"
        script += "assert Path('/tmp/service-evidence/raw.bin').read_bytes() == bytes([0,255]) + b'correct'\n"
    if args.mode != "reward-missing":
        value = "nan" if args.mode == "reward-nan" else "0" if args.mode == "reward-zero" else "1"
        script += f"Path('/logs/verifier/reward.txt').write_text('{value}\\n')\n"
    script += "PY\n"
    (task / "tests/test.sh").write_text(script)
    deadline = time.time() + 150
    save(root, "plan.json", {"owner_token": args.owner_token, "verifier_token": verifier_token(args.owner_token),
        "mode": args.mode, "deadline_unix": deadline, "base_image": args.image,
        "task_files_sha256": {str(p.relative_to(task)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(task.rglob("*")) if p.is_file()}})
    if args.mode in {"kill-copy", "kill-service-copy"}:
        import taste.benchmarks.artifact_handoff as handoff_module
        original = handoff_module._fingerprint

        def die_before_receipt(target, limits):
            result = original(target, limits)
            expected = "/tmp/service-result" if has_helper(args.mode) else "/tmp/agent-result"
            if target.source == expected:
                stage = "helper-copy-before-receipt" if has_helper(args.mode) else "copy-before-receipt"
                save(root, "owner-killed.json", {"stage": stage, "target": str(target.destination)})
                os.kill(os.getpid(), 9)
            return result
        handoff_module._fingerprint = die_before_receipt
    config = TrialConfig(task=TaskConfig(path=task), trial_name=f"taste-separate-{args.owner_token}",
        trials_dir=root / "trials", agent=AgentConfig(import_path="scripts.check_azure_harbor:AzureTerminalAgent",
            kwargs={"owner_token": args.owner_token, "worker_python": args.worker_python,
                    "worker_uid": 1000, "mode": "separate-" + args.mode}, override_setup_timeout_sec=10),
        environment=EnvironmentConfig(import_path="scripts.check_harbor_separate:SeparateAzureEnvironment",
            kwargs={"owner_token": args.owner_token, "mode": args.mode, "deadline_unix": deadline}, delete=False))
    report = {"status": "failed", "fixture_only": True, "mode": args.mode, "paid_calls": 0, "cleanup_errors": []}
    try:
        trial = await Trial.create(config)
        guard_verifier(trial)
        result = await trial.run()
        success = args.mode in {"complete", "reward-zero", "service-complete"}
        if success:
            assert result.exception_info is None, result.exception_info
            assert result.verifier_result.rewards == {"reward": 0.0 if args.mode == "reward-zero" else 1.0}
            assert len(read(root, "artifacts-sealed.json")) == artifact_count
        else:
            assert result.exception_info is not None and result.verifier_result is None
        failure = result.exception_info
        handoff = trial.agent_environment.handoff
        bad_artifact = args.mode in {"missing", "link", "tamper", "manifest-failure"}
        if args.mode in {"service-missing", "service-restarted", "service-stop-failure", "service-cleanup-failure"}:
            assert failure.exception_type == "OutputSnapshotError", failure
            assert not (root / "verifier-intent.json").exists() and not (root / "artifacts-sealed.json").exists()
            if args.mode == "service-cleanup-failure":
                assert "helper service has unconfirmed cleanup" in failure.exception_message
                assert read(root, "helper-stop-error.json")["error"] == "controlled helper stop failure"
                assert not (root / "helper-drain.json").exists()
            else:
                assert handoff.phase == "failed"
                manifest = json.loads((trial.paths.artifacts_dir / "manifest.json").read_bytes())
                assert any(row["source"] == "/tmp/service-result" and row["status"] == "failed" for row in manifest)
                if args.mode == "service-stop-failure":
                    assert not (root / "main-before-helper.json").exists()
                    assert read(root, "main-stop-error.json")["error"]
                elif args.mode == "service-restarted":
                    proof = read(root, "helper-drain.json")
                    assert "restarted" in proof["identity_conflict"] and len(proof["removed_replacement"]) == 1
        elif bad_artifact:
            assert failure.exception_type == "OutputSnapshotError", failure
            assert handoff.phase == "failed"
            assert not (root / "verifier-intent.json").exists(), "bad artifact started a verifier"
            assert not (root / "artifacts-sealed.json").exists()
            manifest = json.loads((trial.paths.artifacts_dir / "manifest.json").read_bytes())
            if args.mode == "tamper":
                assert all(row["status"] == "ok" for row in manifest)
                assert "changed before verification" in failure.exception_message
            else:
                failed_source = "/tmp/evidence" if args.mode == "link" else "/tmp/agent-result"
                assert any(row["source"] == failed_source and row["status"] != "ok" for row in manifest)
        else:
            assert handoff.phase == "sealed" and len(read(root, "artifacts-sealed.json")) == artifact_count
            verifier = trial.agent_environment.group["verifier"]
            assert read(root, "verifier-binding.json")["container_id"] == verifier.taste_backend.environment_id
            if args.mode == "cleanup-failure":
                assert not verifier.stopped and verifier.stop_error == "controlled verifier stop failure"
                assert read(root, "verifier-stop-error.json")["error"] == verifier.stop_error
                assert not (root / "verifier-drain.json").exists()
                assert failure.exception_type == "OutputSnapshotError"
                assert "unconfirmed cleanup" in failure.exception_message
                # A valid reward existed, but the outside guard must withhold it.
                assert trial.paths.reward_text_path.read_text().strip() == "1"
            else:
                assert verifier.stopped and verifier.stop_error is None
                assert read(root, "verifier-drain.json")["stopped"]
                assert empty_cgroup(verifier.taste_backend.binding.cgroup_path)
                if args.mode == "reward-missing":
                    assert failure.exception_type == "RewardFileNotFoundError", failure
                elif args.mode == "reward-nan":
                    assert failure.exception_type == "VerifierOutputParseError", failure
                    assert "Non-finite reward" in failure.exception_message
        owner = trial.agent.owner
        assert owner.closed and owner.outcome.complete and owner.outcome.budget.enforceable
        assert empty_cgroup(trial.agent.backend.binding.cgroup_path)
        if has_helper(args.mode) and args.mode != "service-cleanup-failure":
            proof = read(root, "helper-drain.json")
            assert proof["stopped"] and empty_cgroup(proof["binding"]["cgroup_path"])
            if args.mode != "service-stop-failure":
                assert read(root, "main-before-helper.json")["binding"] == asdict(trial.agent.backend.binding)
        report.update(status="passed", official_rewards=result.verifier_result.rewards if result.verifier_result else None,
            harbor_result=str(trial.paths.result_path), agent_binding=asdict(trial.agent.backend.binding),
            goal_outcome=owner.outcome.to_dict(), artifact_phase=handoff.phase,
            expected_failure=failure.model_dump(mode="json") if failure else None)
    finally:
        try:
            report["independent_cleanup"] = cleanup_separate(args.owner_token)
        except BaseException as exc:
            report["cleanup_errors"].append(type(exc).__name__)
            report["status"] = "failed"
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    assert not report["cleanup_errors"]
    for token in (args.owner_token, verifier_token(args.owner_token), helper_token(args.owner_token)):
        assert not cli("ps", "-aq", "--filter", f"label={OWNER_LABEL}={token}").stdout.strip()
    assert json.loads(cli("image", "inspect", args.image).stdout)[0]["Id"] == args.image
    print(json.dumps({"status": report["status"], "mode": args.mode, "reward": report["official_rewards"], "paid_calls": 0}))


def observe(args):
    root = fixture_root(args.owner_token)
    marker = read(root, "owner-killed.json")
    report = cleanup_separate(args.owner_token)
    agent_root = owner_directory(args.owner_token)
    assert not (agent_root / "controller/run/credentials").exists()
    drain = read(agent_root / "controller", "drain.json")
    assert all(SystemdManager().empty(row["unit"]) for row in drain["scopes"])
    assert empty_cgroup(read(agent_root / "controller", "trial.json")["backend"]["cgroup_path"])
    mode = read(root, "plan.json")["mode"]
    if marker["stage"] in {"copy-before-receipt", "helper-copy-before-receipt"}:
        assert Path(marker["target"]).read_bytes() == (b"helper:correct" if has_helper(mode) else b"correct")
        assert not (root / "artifacts-sealed.json").exists() and not (root / "verifier-intent.json").exists()
    else:
        assert marker["stage"] == "verifier" and read(root, "verifier-drain.json")["stopped"]
    calls = [json.loads(line) for line in (agent_root / "exchange/calls.jsonl").read_text().splitlines()]
    counts = Counter(row["role"] for row in calls)
    assert counts["planner"] == 2 and counts["worker"] == 3 and counts["terminal"] == 1
    if has_helper(mode):
        assert read(root, "main-before-helper.json")["stopped"]
        assert read(root, "helper-drain.json")["stopped"]
        assert empty_cgroup(read(root, "helper-binding.json")["cgroup_path"])
    for token in (args.owner_token, verifier_token(args.owner_token), helper_token(args.owner_token)):
        assert not cli("ps", "-aq", "--filter", f"label={OWNER_LABEL}={token}").stdout.strip()
    image = read(root, "plan.json")["base_image"]
    assert json.loads(cli("image", "inspect", image).stdout)[0]["Id"] == image
    result = {"status": "passed", "mode": mode,
              "paid_calls": 0, "owner_death": marker, "request_counts": dict(counts),
              "independent_cleanup": report, "all_original_scopes_empty": True, "official_rewards": None}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "stage": marker["stage"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--mode", choices=MODES, default="complete")
    parser.add_argument("--image")
    parser.add_argument("--worker-python")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cleanup-only", action="store_true")
    parser.add_argument("--observe-owner-death", action="store_true")
    args = parser.parse_args()
    if args.cleanup_only:
        print(json.dumps(cleanup_separate(args.owner_token)))
    elif args.observe_owner_death:
        observe(args)
    else:
        asyncio.run(check(args))


if __name__ == "__main__":
    main()
