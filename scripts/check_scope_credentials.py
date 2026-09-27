#!/usr/bin/env python3
"""Server-only systemd credential delivery check. Dummy bytes, no model calls.

Run as the outside root controller. Disposable services run as an unprivileged
account, read only their systemd credential copy, and retain no secret in their
reports. Root snapshots stay outside the task/worker filesystem access boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import pwd
import re
import secrets
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from taste.brains.azure_execution_policy import AzureExecutionPolicy
from taste.brains.azure_goal_credentials import GOAL_CREDENTIAL_NAME, encode_azure_goal_credentials
from taste.brains.process_credentials import ScopeCredential
from taste.brains.process_scope import OwnedProcessScope, ScopeSpec, SystemdManager
from taste.pricing import table_sha
from taste.resources import ResourceCleanupError

CHILD = r'''
import errno, hashlib, json, os, pathlib, sys, time
source, expected, report, name, checkout, public_policy = sys.argv[1:]
public_policy = json.loads(public_policy)
directory = pathlib.Path(os.environ["CREDENTIALS_DIRECTORY"])
path = directory / name
raw = path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == expected
secret = raw
if public_policy:
    # Exercise the actual credential loader in systemd's mount. The fake public
    # config isolates delivery from goal preparation, covered by process tests.
    sys.path.insert(0, checkout)
    from types import SimpleNamespace
    from taste.brains.azure_goal_credentials import load_azure_goal_credentials
    config = SimpleNamespace(to_bytes=lambda: b"dummy goal input",
        goal=SimpleNamespace(metadata={"azure_execution": public_policy}))
    environment, issuer = load_azure_goal_credentials(config, str(directory))
    assert issuer is None
    secret = json.loads(raw)["api_key"].encode()
    assert environment["AZURE_OPENAI_API_KEY"].encode() == secret
assert secret not in pathlib.Path("/proc/self/cmdline").read_bytes()
assert secret not in pathlib.Path("/proc/self/environ").read_bytes()
try:
    pathlib.Path(source).read_bytes()
except PermissionError:
    pass
else:
    raise AssertionError("worker could read the controller snapshot")
try:
    path.write_bytes(b"changed")
except OSError as exc:
    assert exc.errno in {errno.EACCES, errno.EROFS, errno.EPERM}
else:
    raise AssertionError("systemd credential copy was writable")
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
pathlib.Path(report).write_text(json.dumps({
    "uid": os.geteuid(), "gid": os.getegid(), "groups": os.getgroups(),
    "credential_directory": str(directory), "sha256": expected,
    "source_denied": True, "copy_readonly": True, "argv_and_env_clean": True,
    "azure_loader_verified": bool(public_policy), "credential_uid": path.stat().st_uid,
    "credential_mode": path.stat().st_mode & 0o777,
    "pid": os.getpid(), "cgroup": pathlib.Path("/proc/self/cgroup").read_text(),
}))
time.sleep(60)
'''


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-token", required=True)
    parser.add_argument("--user", default="bugbash")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.geteuid() == 0, "controller must run as root")
    require(re.fullmatch(r"[0-9a-f]{32}", args.owner_token), "invalid owner token")
    account = pwd.getpwnam(args.user)
    require(account.pw_uid > 0, "service must be unprivileged")
    require(set(os.getgrouplist(account.pw_name, account.pw_gid)) == {account.pw_gid},
            "service user has supplementary groups")
    private = Path("/root") / f"taste-scope-credentials-{args.owner_token}"
    private.mkdir(mode=0o700)
    workdir = Path(account.pw_dir) / f"taste-scope-credentials-{args.owner_token}"
    workdir.mkdir(mode=0o700)
    os.chown(workdir, account.pw_uid, account.pw_gid)
    report = {"owner_token": args.owner_token, "cases": [], "cleanup_errors": []}
    owned = []
    manager = SystemdManager()

    def interrupted(signum, _frame):
        raise TimeoutError(f"credential check interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for case in ("deadline", "lost_start_reply", "azure_loader"):
            secret = secrets.token_hex(32).encode()
            name, public_policy = "azure.json", ""
            if case == "azure_loader":
                policy = AzureExecutionPolicy(
                    "https://test-resource.openai.azure.com/openai/v1/", "gpt-6-astra", "gpt-6-sol",
                    time.time() + 60, 20, 20, 2, 2, 256, 256, 256, pricing_sha=table_sha(),
                )
                public_policy = json.dumps(policy.to_dict(), sort_keys=True)
                config = SimpleNamespace(to_bytes=lambda: b"dummy goal input",
                    goal=SimpleNamespace(metadata={"azure_execution": policy.to_dict()}))
                secret = encode_azure_goal_credentials(config, secret.decode())
                name = GOAL_CREDENTIAL_NAME
            descriptor = ScopeCredential.from_bytes(name, secret)
            owner = private / case
            output = workdir / f"{case}.json"
            spec = ScopeSpec(
                (sys.executable, "-I", "-c", CHILD, str(owner / "credentials" / name),
                 descriptor.sha256, str(output), name, str(Path(__file__).resolve().parents[1]),
                 public_policy or "null"), str(workdir), account.pw_uid, 4, 0.5,
                credentials=(descriptor,),
            )
            scope = OwnedProcessScope.create(owner, spec, credentials={name: secret})
            owned.append(scope)
            if case == "lost_start_reply":
                class LostReply(SystemdManager):
                    def start(self, *values, **kwargs):
                        super().start(*values, **kwargs)
                        raise ConnectionError("injected lost credential service acknowledgement")

                scope.manager = LostReply()
                try:
                    scope.start()
                except ResourceCleanupError as exc:
                    require("injected lost" in str(exc), f"unexpected launch failure: {exc}")
                else:
                    raise AssertionError("lost acknowledgement was not reported")
            else:
                scope.start()
            deadline = time.monotonic() + 3
            while not output.exists():
                require(time.monotonic() < deadline, "credential child did not report readiness")
                time.sleep(0.02)
            child = json.loads(output.read_text())
            require(child["uid"] == account.pw_uid and child["groups"] == [account.pw_gid],
                    "credential reader had unexpected authority")
            require(f"/system.slice/{scope.unit}" in child["cgroup"], "reader escaped its scope")
            for source in (owner / "state.json", Path(f"/proc/{child['pid']}/cmdline"),
                           Path(f"/proc/{child['pid']}/environ")):
                require(secret not in source.read_bytes(), "credential leaked into public launch metadata")
            restored = OwnedProcessScope(owner)
            receipt = (restored.wait(timeout_seconds=8) if case == "deadline"
                       else restored.stop("credential check complete"))
            require(receipt["processes_stopped"] and receipt["goal_settlement_required"],
                    "credential scope cleanup incorrectly settled its goal")
            require(manager.empty(scope.unit) and manager.inspect(scope.unit) is None, "scope leaked")
            require(not Path(child["credential_directory"]).exists(), "systemd credential mount remained")
            require(secret not in json.dumps(receipt).encode(), "credential leaked into receipt")
            report["cases"].append({"case": case, "reader": child, "termination": receipt})
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        for scope in owned:
            try:
                OwnedProcessScope(scope.directory).stop("credential validation cleanup")
            except BaseException as exc:
                report["cleanup_errors"].append(f"{scope.unit}: {type(exc).__name__}: {exc}")
        if report["cleanup_errors"]:
            report["status"] = "failed"
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]),
                      "cleanup_errors": len(report["cleanup_errors"])}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
