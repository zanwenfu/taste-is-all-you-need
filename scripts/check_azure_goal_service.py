#!/usr/bin/env python3
"""Server-only root/unprivileged Azure goal lifecycle fixture; zero paid calls.

Run in a bounded root systemd unit with this script's --cleanup-only invocation
as ExecStopPost. A Python-only manager seam replaces HTTP with the existing SDK
mock; inputs cannot select that seam. No Docker or Harbor environment is started.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pwd
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taste.brains.azure_execution_policy import AzureExecutionPolicy
from taste.brains.azure_goal_credentials import encode_azure_goal_credentials
from taste.brains.azure_goal_handoff import grading_ready, preparation_bytes
from taste.brains.azure_goal_service import AzureGoalService
from taste.brains.central_planner import Goal
from taste.brains.process_scope import OwnedProcessScope, SystemdManager
from taste.pricing import table_sha
from tests.test_azure_goal_entrypoint import BOOTSTRAP


def directory(owner):
    if re.fullmatch(r"[0-9a-f]{32}", owner) is None:
        raise ValueError("fixture owner must be a fresh UUID hex")
    return Path("/var/tmp") / f"taste-azure-goal-service-{owner}"


def cleanup(owner):
    receipts = []
    failures = []
    for operation in ("prepare", "run", "settle"):
        path = directory(owner) / "controller" / operation
        if path.exists():
            try:
                receipts.append(OwnedProcessScope(path).stop("independent fixture owner cleanup"))
            except BaseException as exc:
                failures.append(exc)
    if failures:
        raise BaseExceptionGroup("goal fixture cleanup remains incomplete", failures)
    return receipts


class MockHTTPManager(SystemdManager):
    def __init__(self, counter, mode):
        self.bootstrap = "import os\nos.environ['TASTE_TEST_WIRE_COUNT'] = " + repr(str(counter)) + "\n" + BOOTSTRAP
        if mode == "lost-reply":
            self.bootstrap = self.bootstrap.replace("return httpx.Response(200, json={",
                "raise httpx.ReadTimeout('injected lost reply', request=wire)\n    return httpx.Response(200, json={")
        elif mode == "killed":
            self.bootstrap = self.bootstrap.replace("return httpx.Response(200, json={",
                "os.kill(os.getpid(), 9)\n    return httpx.Response(200, json={")
        # Fixture-only diagnostics: every provider is mocked and the only key
        # is a public test string. Production handoffs retain redacted errors.
        self.bootstrap += '''
import taste.brains.azure_goal_handoff as handoff, traceback
original_perform = handoff.perform
def fixture_perform(*args, **kwargs):
    try:
        return original_perform(*args, **kwargs)
    except BaseException:
        traceback.print_exc()
        raise
handoff.perform = fixture_perform
'''

    def start(self, unit, description, spec, *, credential_directory=None):
        argv = list(spec.argv)
        argv[3] = argv[3].replace("from taste.brains.azure_goal_handoff",
            self.bootstrap + "\nfrom taste.brains.azure_goal_handoff", 1)
        super().start(unit, description, replace(spec, argv=tuple(argv)), credential_directory=credential_directory)


def write_public(path, raw):
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)
    return hashlib.sha256(raw).hexdigest()


async def check(args):
    assert os.geteuid() == 0
    account = pwd.getpwnam(args.worker_user)
    assert account.pw_uid > 0 and set(os.getgrouplist(account.pw_name, account.pw_gid)) == {account.pw_gid}
    root = directory(args.owner_token)
    root.mkdir(mode=0o755)
    control = root / "controller"
    control.mkdir(mode=0o700)
    state = root / "agent-state"
    state.mkdir(mode=0o700)
    os.chown(state, account.pw_uid, account.pw_gid)
    work = state / "workspace"
    work.mkdir(mode=0o700)
    os.chown(work, account.pw_uid, account.pw_gid)
    exchange = root / "exchange"
    exchange.mkdir(mode=0o700)
    os.chown(exchange, account.pw_uid, account.pw_gid)
    counter = exchange / "wire-count"
    manager = MockHTTPManager(counter, args.mode)
    report = {"status": "failed", "fixture_only": True, "paid_calls": 0, "mode": args.mode,
              "mock_http_sha256": hashlib.sha256(manager.bootstrap.encode()).hexdigest(),
              "worker_uid": account.pw_uid, "cleanup_errors": []}
    try:
        goal = Goal(goal_id="azure-service-fixture", task="Confirm the controlled fixture",
                    success_criteria=("controlled fixture completed",), budget_usd=100)
        policy = AzureExecutionPolicy(endpoint="https://test-resource.openai.azure.com/openai/v1/",
            planner_deployment="gpt-6-astra", worker_deployment="gpt-6-sol", deadline_unix=time.time() + 120,
            worker_budget_usd=20, monitor_budget_usd=20, worker_max_calls=8, monitor_max_calls=16,
            worker_max_output_tokens=256, monitor_max_output_tokens=512, planner_max_output_tokens=512,
            monitor_batch_size=1, pricing_sha=table_sha())
        raw = preparation_bytes(work, "service-fixture", goal, policy, max_generations=2,
                                wall_clock_seconds=120, max_planner_failures=1)
        path = root / "prepare.json"
        digest = write_public(path, raw)

        def operation(name, input_path, input_digest, *, credential=None):
            return AzureGoalService.create(control / name, input_path, input_digest, exchange / name,
                operation=name, uid=account.pw_uid, python_executable=args.worker_python,
                runtime_seconds=30, grace_seconds=3, credential=credential, manager=manager)

        prepared = await operation("prepare", path, digest).run(timeout_seconds=40)
        assert not counter.exists()
        config = prepared.value
        assert config.goal.goal_id == goal.goal_id and Path(config.repo_root).stat().st_uid == account.pw_uid
        path = root / "goal.json"
        digest = write_public(path, config.to_bytes())
        runner = operation("run", path, digest, credential=encode_azure_goal_credentials(config, "azure-test-only"))
        missing_result = False
        try:
            run_result = await runner.run(timeout_seconds=40)
        except FileNotFoundError:
            assert args.mode == "killed"
            missing_result = True
            assert not (exchange / "run/result.json").exists()
            receipt = runner.scope.stop("verify killed goal drainage")
            assert receipt["processes_stopped"]
        else:
            assert args.mode != "killed"
            assert run_result.value.complete == (args.mode == "complete")
        assert counter.read_text().splitlines() == ["one Azure planner request"]
        # Remove the private launch copy after proved drainage. Recovery must
        # not need it or refresh the original admitted deadline.
        for secret in (control / "run/credentials").iterdir():
            secret.unlink()
        (control / "run/credentials").rmdir()
        settled = await operation("settle", path, digest).run(timeout_seconds=40)
        assert counter.read_text().splitlines() == ["one Azure planner request"]
        assert grading_ready(settled.value) == (args.mode == "complete")
        assert bool(settled.value.budget.unknown_planner_attempt_ids) == (args.mode != "complete")
        if not missing_result:
            assert run_result.value == settled.value
            reopened = AzureGoalService(OwnedProcessScope(control / "run"))
            assert (await reopened.recover()).value == run_result.value
        assert not (control / "prepare/credentials").exists()
        assert not (control / "settle/credentials").exists()
        receipts = cleanup(args.owner_token)
        assert len(receipts) == 3 and all(row["processes_stopped"] for row in receipts)
        assert all(SystemdManager().empty(row["unit"]) for row in receipts)
        report.update(status="passed", mock_requests=1, missing_run_result=missing_result,
                      source_sha256=config.python_source_sha256, input_sha256=digest,
                      original_deadline_unix=policy.deadline_unix,
                      outcome=settled.value.to_dict(), grading_ready=grading_ready(settled.value),
                      receipts=receipts, result_sha256=settled.result_sha256,
                      original_cgroups_empty=True, private_launch_copy_removed=True)
    finally:
        try:
            cleanup(args.owner_token)
        except BaseException as exc:
            report["cleanup_errors"].append(type(exc).__name__)
            report["status"] = "failed"
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["status"] == "passed", report
    print(json.dumps(report, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--worker-python")
    parser.add_argument("--worker-user", default="bugbash")
    parser.add_argument("--mode", choices=("complete", "lost-reply", "killed"), default="complete")
    parser.add_argument("--cleanup-only", action="store_true")
    args = parser.parse_args()
    if args.cleanup_only:
        print(json.dumps({"cleanup_receipts": cleanup(args.owner_token)}))
    else:
        if not args.worker_python:
            parser.error("--worker-python is required for execution")
        asyncio.run(check(args))


if __name__ == "__main__":
    main()
