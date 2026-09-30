#!/usr/bin/env python3
"""Actual Harbor verifier plus Azure goal/worker/monitor/RPC, with mocked HTTP.

Server-only bounded root fixture. Use --cleanup-only as an independent systemd
ExecStopPost command. Uses the separately pinned Harbor environment, no image
pulls, no task network, no paid model calls and no Claude SDK. This controlled
task is integration evidence, not a Terminal Bench score.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pwd
import re
import sqlite3
import sys
import time
from collections import Counter
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["HARBOR_TELEMETRY"] = "0"

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.models.trajectories import Trajectory
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial

from scripts.check_docker_terminal import cleanup, cli, empty_cgroup
from scripts.check_harbor_trial import SnapshotDockerEnvironment
from taste.benchmarks.azure_terminal_trial import AzureTerminalTrial, cleanup_trial
from taste.brains import benchmark_reply
from taste.brains.azure_execution_policy import AzureExecutionPolicy
from taste.brains.azure_goal_handoff import _write
from taste.brains.central_planner import Goal
from taste.brains.docker_terminal import OWNER_LABEL, DockerTerminalBackend
from taste.brains.owned_thread import start_owned_thread
from taste.brains.process_scope import SystemdManager, _owned_call
from taste.brains.terminal_broker import TerminalBinding, _settle
from taste.brains.terminal_worker_policy import TerminalWorkerPolicy
from taste.pricing import table_sha


def owner_directory(token):
    if re.fullmatch(r"[0-9a-f]{32}", token) is None:
        raise ValueError("fixture requires a fresh owner token")
    return Path("/var/tmp") / f"ta-{token}"


def cleanup_fixture(token):
    root = owner_directory(token)
    errors, result = [], {}
    if (root / "controller/trial.json").exists():
        try:
            result["trial"] = cleanup_trial(root)
        except BaseException as exc:
            errors.append(exc)
    try:
        result["removed_containers"] = cleanup(token)
    except BaseException as exc:
        errors.append(exc)
    if errors:
        raise BaseExceptionGroup("Azure Harbor fixture cleanup remains incomplete", errors)
    return result


class FixtureManager(SystemdManager):
    def __init__(self, counter, mode):
        self.code = f"from tests.azure_harbor_wire import install_goal; install_goal({str(counter)!r}, {mode!r}); "

    def start(self, unit, description, spec, *, credential_directory=None):
        argv = list(spec.argv)
        argv[3] = argv[3].replace("from taste.brains.azure_goal_handoff", self.code + "from taste.brains.azure_goal_handoff", 1)
        super().start(unit, description, replace(spec, argv=tuple(argv)), credential_directory=credential_directory)


class OwnedAzureEnvironment(SnapshotDockerEnvironment):
    async def stop(self, delete):
        owner = getattr(self, "taste_owner", None)
        if owner is not None:
            await owner.close()
            assert owner.closed
            self.taste_drained_before_delete = True
            # The complete trial owner has already stopped/released the broker.
            # The base fixture must not close that released broker again.
            self.taste_service = None
        elif getattr(self, "taste_backend", None) is not None:
            operation = start_owned_thread(_owned_call, self.taste_backend.stop_and_confirm)
            await _settle(operation)
            self.taste_drained_before_delete = True
        await super().stop(delete)


class AzureTerminalAgent(BaseAgent):
    capabilities = AgentCapabilities(atif=True)

    def __init__(self, *, owner_token, worker_python, worker_uid, mode, **kwargs):
        super().__init__(**kwargs)
        self.owner_token, self.worker_python, self.worker_uid, self.mode = owner_token, worker_python, worker_uid, mode
        self.owner = self.backend = None

    @staticmethod
    def name():
        return "taste-azure-terminal-validation"

    def version(self):
        return "1"

    async def setup(self, environment):
        assert isinstance(environment, OwnedAzureEnvironment) and environment.default_user is None
        ids = cli("ps", "-aq", "--no-trunc", "--filter", f"label={OWNER_LABEL}={self.owner_token}").stdout.splitlines()
        assert len(ids) == 1
        inspected = json.loads(cli("inspect", ids[0].decode()).stdout)[0]
        assert not inspected["Mounts"] and inspected["Config"]["User"] in ("", "root", "0")
        self.agent_seconds = 35 if self.mode == "late-verifier" else 60
        self.agent_deadline = time.time() + self.agent_seconds
        self.backend = DockerTerminalBackend.admit("/var/run/docker.sock", ids[0].decode(), self.owner_token,
            self.agent_deadline + 45, output_limit=8192)
        environment.taste_backend = self.backend

    async def run(self, instruction, environment, context):
        binding = TerminalBinding(self.owner_token, self.backend.environment_id, self.agent_deadline, 20)
        policy = AzureExecutionPolicy(endpoint="https://test-resource.openai.azure.com/openai/v1/",
            planner_deployment="gpt-6-astra", worker_deployment="gpt-6-sol", deadline_unix=self.agent_deadline,
            worker_budget_usd=20, monitor_budget_usd=20, worker_max_calls=8, monitor_max_calls=16,
            worker_max_output_tokens=256, monitor_max_output_tokens=512, planner_max_output_tokens=512,
            monitor_batch_size=1, pricing_sha=table_sha(), terminal=TerminalWorkerPolicy(binding, 5))
        root = owner_directory(self.owner_token)
        manager = FixtureManager(root / "exchange/calls.jsonl", self.mode)
        self.owner = AzureTerminalTrial.create(root, self.backend,
            Goal(goal_id="harbor-terminal", task=instruction,
                 success_criteria=("/tmp/agent-result contains the exact bytes correct",), budget_usd=100,
                 metadata={benchmark_reply.KEY: benchmark_reply.SCHEMA}),
            policy, service_uid=self.worker_uid, python_executable=self.worker_python,
            max_generations=3, wall_clock_seconds=self.agent_seconds, max_planner_failures=1, manager=manager)
        environment.taste_owner = self.owner
        self._owner_death_cancelled = False
        death = asyncio.create_task(self._kill_owner_during_command()) if self.mode == "killed-owner" else None
        context.metadata = {"fixture": True, "paid_calls": 0, "mode": self.mode,
                            "http_mock_sha256": hashlib.sha256(manager.code.encode()).hexdigest()}
        try:
            outcome = await self.owner.run(api_key="azure-test-only")
            environment.taste_service = self.owner.service
            assert self.owner.broker.phase == "sealed"
            assert json.loads(cli("inspect", self.backend.environment_id).stdout)[0]["State"]["Running"]
            context.cost_usd = outcome.budget.known_spent_usd
            context.metadata["goal_outcome"] = outcome.to_dict()
            # Actual shared-verifier consumption, after the agent process scope
            # drains and terminal requests are sealed. Harbor later snapshots
            # the guest log directory; keep the controller copy authoritative.
            assert self.owner.trajectory_path is not None
            trace = Trajectory.model_validate_json(self.owner.trajectory_path.read_bytes())
            assert trace.extra["evidence_complete"] and len(trace.subagent_trajectories) == 3
            await environment.upload_file(self.owner.trajectory_path, "/logs/agent/trajectory.json")
        finally:
            if death is not None:
                self._owner_death_cancelled = True
                death.cancel()
                with suppress(asyncio.CancelledError):
                    await _settle(death)
            if self.owner.outcome is not None:
                context.metadata["goal_outcome"] = self.owner.outcome.to_dict()
                if not self.owner.outcome.budget.enforceable:
                    context.cost_usd = None

    async def _kill_owner_during_command(self):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            check = start_owned_thread(cli, "cp", f"{self.backend.environment_id}:/tmp/owner-command-started", "-", check=False)
            result = await _settle(check)
            if self._owner_death_cancelled:
                return
            if result.returncode == 0:
                _write(self.owner.fd, "owner-killed.json", (json.dumps({"pid": os.getpid(),
                    "container_id": self.backend.environment_id, "command_entered": True}) + "\n").encode())
                os.kill(os.getpid(), 9)
            await asyncio.sleep(0.02)
        raise AssertionError("fixture terminal command did not start before owner-death injection")


def observe_owner_death(args):
    """Independent observer after the killed controller's ExecStopPost ends."""
    assert os.geteuid() == 0
    root = owner_directory(args.owner_token)
    marker = json.loads((root / "controller/owner-killed.json").read_text())
    assert marker["command_entered"]
    saved = json.loads((root / "controller/trial.json").read_text())
    receipt = cleanup_trial(root)
    assert receipt["container_stopped"] and receipt["goal_settlement_required"] and len(receipt["scopes"]) == 2
    assert all(SystemdManager().empty(row["unit"]) for row in receipt["scopes"])
    assert empty_cgroup(saved["backend"]["cgroup_path"])
    assert not (root / "controller/grading.json").exists()
    assert not (root / "controller/run/credentials").exists()
    assert not cli("ps", "-aq", "--filter", f"label={OWNER_LABEL}={args.owner_token}").stdout.strip()
    calls = [json.loads(line) for line in (root / "exchange/calls.jsonl").read_text().splitlines()]
    counts = Counter(row["role"] for row in calls)
    assert counts["planner"] == counts["worker"] == 1
    with sqlite3.connect((root / "controller/terminal/terminal.sqlite3").as_uri() + "?mode=ro", uri=True) as connection:
        effects = connection.execute("SELECT id,status FROM requests").fetchall()
    assert len(effects) == 1 and effects[0][1] == "pending"
    assert json.loads(cli("image", "inspect", args.image).stdout)[0]["Id"] == args.image
    report = {"status": "passed", "fixture_only": True, "mode": "killed-owner", "paid_calls": 0,
              "official_rewards": None, "request_counts": dict(counts), "terminal_effects": effects,
              "goal_settlement_required": True, "independent_drain": receipt,
              "original_container_cgroup_empty": True, "cached_image_retained": True,
              "private_launch_copy_removed": True}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "mode": "killed-owner", "request_counts": dict(counts)}))


async def check(args):
    assert os.geteuid() == 0 and os.environ.get("HARBOR_TELEMETRY") == "0"
    owner_directory(args.owner_token)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", args.image)
    account = pwd.getpwnam(args.worker_user)
    assert account.pw_uid > 0 and set(os.getgrouplist(account.pw_name, account.pw_gid)) == {account.pw_gid}
    assert json.loads(cli("image", "inspect", args.image).stdout)[0]["Id"] == args.image
    assert not cli("ps", "-aq", "--filter", f"label={OWNER_LABEL}={args.owner_token}").stdout.strip()
    directory = Path("/root") / f"taste-azure-harbor-{args.owner_token}"
    directory.mkdir(mode=0o700)
    task = directory / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("Write the exact bytes correct to /tmp/agent-result using the task terminal.\n")
    (task / "task.toml").write_text('schema_version = "1.4"\n[agent]\ntimeout_sec = 120\n'
        '[verifier]\ntimeout_sec = 55\n[environment]\n'
        f'docker_image = "{args.image}"\nbuild_timeout_sec = 30\ncpus = 1\nmemory_mb = 256\n'
        'network_mode = "public"\n')
    (task / "tests/test.sh").write_text("#!/bin/bash\nset -euo pipefail\n" +
        ("sleep 35\n" if args.mode == "late-verifier" else "") +
        "python3 - <<'PY'\nfrom pathlib import Path\nimport json\n"
        "assert Path('/tmp/agent-result').read_bytes() == b'correct'\n"
        "trace = json.loads(Path('/logs/agent/trajectory.json').read_bytes())\n"
        "assert trace['extra']['evidence_complete']\n"
        "assert trace['steps'][-1]['message'] == 'Wrote correct to /tmp/agent-result and retained the terminal evidence.'\n"
        "assert len(trace['subagent_trajectories']) == 3\n"
        "Path('/logs/verifier/reward.txt').write_text('1\\n')\nPY\n")
    overlay = directory / "compose.json"
    overlay.write_text(json.dumps({"services": {"main": {
        "labels": {OWNER_LABEL: args.owner_token}, "restart": "no", "pull_policy": "never",
        "network_mode": "none", "pids_limit": 64, "security_opt": ["no-new-privileges:true"],
    }}}))
    config = TrialConfig(task=TaskConfig(path=task), trial_name=f"taste-azure-{args.owner_token}",
        trials_dir=directory / "trials",
        agent=AgentConfig(import_path="scripts.check_azure_harbor:AzureTerminalAgent", kwargs={
            "owner_token": args.owner_token, "worker_python": args.worker_python,
            "worker_uid": account.pw_uid, "mode": args.mode}, override_setup_timeout_sec=10),
        environment=EnvironmentConfig(import_path="scripts.check_azure_harbor:OwnedAzureEnvironment",
                                       extra_docker_compose=[overlay], delete=False))
    report = {"status": "failed", "fixture_only": True, "mode": args.mode, "paid_calls": 0, "cleanup_errors": [],
        "task_files_sha256": {str(p.relative_to(task)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(task.rglob("*")) if p.is_file()},
        "compose_sha256": hashlib.sha256(overlay.read_bytes()).hexdigest()}
    trial = None
    try:
        trial = await Trial.create(config)
        result = await trial.run()
        root = owner_directory(args.owner_token)
        calls = [json.loads(line) for line in (root / "exchange/calls.jsonl").read_text().splitlines()]
        counts = Counter(row["role"] for row in calls)
        if args.mode in {"complete", "late-verifier"}:
            assert result.exception_info is None, result.exception_info
            assert result.verifier_result.rewards == {"reward": 1.0}
            assert counts["planner"] == 2 and counts["worker"] == 3 and counts["terminal"] == 1
            assert trial.agent.owner.outcome.complete and trial.agent.owner.outcome.budget.enforceable
            assert trial.agent.owner.outcome.delivered_assignment_ids == ("terminal-result",)
            assert (trial.agent.logs_dir / "trajectory.json").read_bytes() == trial.agent.owner.trajectory_path.read_bytes()
            if args.mode == "late-verifier":
                assert time.time() > trial.agent.agent_deadline
        else:
            assert result.exception_info is not None and result.verifier_result is None
            assert trial.agent.owner.outcome is not None
            assert not trial.agent.owner.outcome.budget.enforceable
            assert counts["planner"] == 1
            assert counts["worker"] == (2 if args.mode == "killed-worker" else 0)
        assert trial.agent.owner.closed and trial.agent_environment.taste_drained_before_delete
        assert empty_cgroup(trial.agent.backend.binding.cgroup_path)
        assert not cli("ps", "-aq", "--filter", f"label={OWNER_LABEL}={args.owner_token}").stdout.strip()
        assert json.loads(cli("image", "inspect", args.image).stdout)[0]["Id"] == args.image
        database = root / "controller/terminal/terminal.sqlite3"
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            effects = connection.execute("SELECT id,status FROM requests").fetchall()
        assert len(effects) == (0 if args.mode == "lost-planner" else 1)
        report.update(status="passed", official_rewards=result.verifier_result.rewards if result.verifier_result else None,
            goal_outcome=trial.agent.owner.outcome.to_dict(), request_counts=dict(counts), terminal_effects=effects,
            goal_services_drained=True, original_container_cgroup_empty=True, cached_image_retained=True,
            snapshot_count=len(trial.agent_environment.snapshots), agent_deadline_unix=trial.agent.agent_deadline,
            container_deadline_unix=trial.agent.backend.binding.deadline_unix, host_output_mounts=False,
            harbor_result=str(trial.paths.result_path))
    finally:
        if trial is not None and getattr(trial.agent, "owner", None) is not None and not trial.agent.owner.closed:
            try:
                await trial.agent.owner.close()
            except BaseException as exc:
                report["cleanup_errors"].append(type(exc).__name__)
        try:
            report["independent_cleanup"] = cleanup_fixture(args.owner_token)
        except BaseException as exc:
            report["cleanup_errors"].append(type(exc).__name__)
        if report["cleanup_errors"]:
            report["status"] = "failed"
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    assert report["status"] == "passed", report
    print(json.dumps({"status": "passed", "mode": args.mode, "reward": report["official_rewards"],
                      "request_counts": report["request_counts"], "paid_calls": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--image")
    parser.add_argument("--worker-user", default="bugbash")
    parser.add_argument("--worker-python")
    parser.add_argument("--mode", choices=("complete", "killed-worker", "lost-planner", "late-verifier", "killed-owner"), default="complete")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cleanup-only", action="store_true")
    parser.add_argument("--observe-owner-death", action="store_true")
    args = parser.parse_args()
    if args.cleanup_only:
        print(json.dumps(cleanup_fixture(args.owner_token)))
    elif args.observe_owner_death:
        if not args.image or not args.output:
            parser.error("observation requires image and output")
        observe_owner_death(args)
    else:
        if not args.image or not args.worker_python or not args.output:
            parser.error("execution requires image, worker-python and output")
        asyncio.run(check(args))


if __name__ == "__main__":
    main()
