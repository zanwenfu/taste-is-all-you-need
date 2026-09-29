"""Outside owner of one Azure goal and its authoritative terminal container.

The caller admits the container and enforces the owner's lifetime independently.
After run(), grading uses the still-live, sealed container. close() must finish
before Harbor removes it. After owner death, cleanup_trial() drains the saved
scopes and original container without restarting an issuer, goal or paid call.
"""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import math
import os
import pwd
import secrets
import stat
import time
from dataclasses import asdict
from pathlib import Path

from taste.brains.azure_execution_policy import AzureExecutionPolicy
from taste.brains.azure_goal_credentials import encode_azure_goal_credentials
from taste.brains.azure_goal_handoff import _read, _write, grading_ready, preparation_bytes
from taste.brains.azure_goal_service import AzureGoalService
from taste.brains.central_planner import Goal
from taste.brains.docker_terminal import DockerTerminalBackend, DockerTerminalBinding
from taste.brains.goal_entrypoint import GoalInputError, _canonical, _decode
from taste.brains.input_limits import MAX_GOAL_INPUT_BYTES
from taste.brains.owned_thread import start_owned_thread
from taste.brains.process_scope import OwnedProcessScope, _owned_call
from taste.brains.terminal_broker import TerminalBroker, _settle
from taste.brains.terminal_issuer import TerminalIssuerCredential
from taste.brains.terminal_service import TerminalService

_SCHEMA = "taste.benchmarks/AzureTerminalTrial/1"
_OPERATIONS = ("prepare", "run", "settle")


def _protected(path):
    """No writable or symlink ancestor may replace controller state."""
    path = Path(path)
    if not path.is_absolute() or str(path.resolve(strict=True)) != str(path):
        raise GoalInputError("trial directory must be canonical and absolute")
    for item in (path, *path.parents):
        info = item.stat()
        sticky_root = info.st_uid == 0 and bool(info.st_mode & stat.S_ISVTX)
        if info.st_uid not in {0, os.geteuid()} or (info.st_mode & 0o022 and not sticky_root):
            raise GoalInputError("trial directory can be replaced by another user")
    return path


def _claim_control(root):
    _protected(root)
    descriptor = os.open(root / "controller", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise GoalInputError("trial control directory must be private and owned")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _json_bytes(value):
    return (_canonical(value) + "\n").encode()


def _load_config(fd, root):
    raw = _read(fd, "trial.json", MAX_GOAL_INPUT_BYTES, uid=os.geteuid(), mode=0o600)
    value = _decode(raw)
    binding = _validate_config(value, root)
    return value, binding, hashlib.sha256(raw).hexdigest()


def _validate_config(value, root):
    if (not isinstance(value, dict) or set(value) != {"schema", "directory", "backend", "policy",
            "goal", "service_uid", "service_gid", "python_executable", "max_generations",
            "wall_clock_seconds", "max_planner_failures"} or value["schema"] != _SCHEMA
            or value["directory"] != str(root)):
        raise GoalInputError("invalid terminal trial admission")
    binding = DockerTerminalBinding(**value["backend"])
    policy = AzureExecutionPolicy.from_dict(value["policy"])
    goal = Goal.from_dict(value["goal"])
    policy.bind_goal(goal)
    if (policy.terminal is None or policy.terminal.binding.environment_id != binding.container_id
            or policy.deadline_unix > binding.deadline_unix
            or policy.terminal.binding.trial_id != binding.owner_token
            or type(value["service_uid"]) is not int or value["service_uid"] <= 0
            or type(value["service_gid"]) is not int or value["service_gid"] < 0
            or not isinstance(value["python_executable"], str)
            or not Path(value["python_executable"]).is_absolute()
            or any(c in value["python_executable"] for c in ("\x00", "%"))):
        raise GoalInputError("terminal trial admission has inconsistent ownership")
    if (any(type(value[key]) is not int or not 1 <= value[key] <= 10000
            for key in ("max_generations", "max_planner_failures"))
            or type(value["wall_clock_seconds"]) not in (int, float)
            or not math.isfinite(value["wall_clock_seconds"])
            or not 0 < value["wall_clock_seconds"] <= 604800
            or len(_json_bytes(value)) > MAX_GOAL_INPUT_BYTES):
        raise GoalInputError("invalid terminal trial limits")
    return binding


def _scope_paths(root):
    result = []
    for operation in _OPERATIONS:
        path = root / "controller" / operation
        if os.path.lexists(path):
            result.append(path)
    return result


def _drain_record(fd, config_sha, container_id, receipts):
    expected = {"schema": "taste.benchmarks/TerminalTrialDrain/1", "config_sha256": config_sha,
                "container_id": container_id, "container_stopped": True, "scopes": receipts,
                "goal_settlement_required": any(row["goal_settlement_required"] for row in receipts)}
    try:
        saved = _decode(_read(fd, "drain.json", 65536, uid=os.geteuid(), mode=0o600))
    except FileNotFoundError:
        return expected, False
    if saved != expected:
        raise GoalInputError("terminal trial drain evidence differs from its original ownership")
    return saved, True


def _remove_launch_material(root):
    # All original scopes must be drained before this is called. Retain their
    # public descriptors and durable journals, but remove the private key copy.
    directory = root / "controller/run/credentials"
    if directory.exists():
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise GoalInputError("private launch material ownership changed")
            for name in os.listdir(fd):
                os.unlink(name, dir_fd=fd)
            os.fsync(fd)
        finally:
            os.close(fd)
        directory.rmdir()
    rpc = root / "rpc"
    if rpc.exists():
        info = rpc.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise GoalInputError("terminal socket directory ownership changed")
        socket = rpc / "terminal.sock"
        if os.path.lexists(socket):
            info = socket.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.geteuid():
                raise GoalInputError("terminal socket identity changed")
            socket.unlink()
        rpc.rmdir()


def cleanup_trial(directory, *, manager=None):
    """Independent watchdog, after owner death. Never prepares or runs a goal.

    Absence of a service after a lost launch acknowledgement stays ambiguous.
    The container is still stopped, but no successful drain record is published.
    A prior exact drain record permits cleanup after Harbor removed the container.
    """
    root = Path(directory)
    fd = _claim_control(root)
    try:
        _, binding, digest = _load_config(fd, root)
        errors, receipts = [], []
        for path in _scope_paths(root):
            try:
                scope = OwnedProcessScope(path, manager=manager)
                receipts.append(scope.stop("outside terminal trial owner ended"))
            except BaseException as exc:
                errors.append(exc)
        record, recorded = _drain_record(fd, digest, binding.container_id, receipts) if not errors else (None, False)
        if not recorded:
            try:
                DockerTerminalBackend(binding).stop_and_confirm()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise BaseExceptionGroup("terminal trial cleanup is incomplete", errors)
        if not recorded:
            _write(fd, "drain.json", _json_bytes(record))
        _remove_launch_material(root)
        return record
    finally:
        os.close(fd)


class AzureTerminalTrial:
    """One controller lifetime; recovery drains instead of restarting execution."""

    @classmethod
    def create(cls, directory, backend, goal, policy, *, service_uid, python_executable,
               max_generations, wall_clock_seconds, max_planner_failures=3, manager=None):
        if not isinstance(backend, DockerTerminalBackend) or not isinstance(policy, AzureExecutionPolicy):
            raise TypeError("an admitted Docker backend and Azure policy are required")
        account = pwd.getpwuid(service_uid)
        if (type(service_uid) is not int or service_uid <= 0
                or set(os.getgrouplist(account.pw_name, account.pw_gid)) != {account.pw_gid}):
            raise GoalInputError("goal service must use an unprivileged account without supplementary groups")
        root = Path(directory)
        _protected(root.parent)
        config = {"schema": _SCHEMA, "directory": str(root), "backend": asdict(backend.binding),
                  "policy": policy.to_dict(), "goal": goal.to_dict(), "service_uid": service_uid,
                  "service_gid": account.pw_gid, "python_executable": str(python_executable),
                  "max_generations": max_generations, "wall_clock_seconds": wall_clock_seconds,
                  "max_planner_failures": max_planner_failures}
        _validate_config(config, root)
        if not 0 < policy.deadline_unix - time.time() <= wall_clock_seconds:
            raise GoalInputError("trial deadline is expired or exceeds its original wall allowance")
        root.mkdir(mode=0o755)
        root.chmod(0o755)
        (root / "controller").mkdir(mode=0o700)
        fd = _claim_control(root)
        try:
            # Original container identity is durable before any process launch.
            _write(fd, "trial.json", _json_bytes(config))
            _load_config(fd, root)
            for name in ("agent-state", "exchange"):
                path = root / name
                path.mkdir(mode=0o700)
                os.chown(path, service_uid, account.pw_gid)
            workspace = root / "agent-state/workspace"
            workspace.mkdir(mode=0o700)
            os.chown(workspace, service_uid, account.pw_gid)
            raw = preparation_bytes(workspace, "terminal-trial", goal, policy,
                max_generations=max_generations, wall_clock_seconds=wall_clock_seconds,
                max_planner_failures=max_planner_failures)
            instance = cls.__new__(cls)
            instance.root, instance.fd, instance.config = root, fd, config
            instance.backend, instance.policy, instance.manager = backend, policy, manager
            instance.broker = instance.service = None
            instance.started = instance.closed = False
            instance.outcome = None
            instance.preparation_sha = instance._input("prepare.json", raw)
            return instance
        except BaseException:
            os.close(fd)
            raise

    def _input(self, name, raw):
        parent = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            _write(parent, name, raw)
            os.chmod(name, 0o444, dir_fd=parent, follow_symlinks=False)
            os.fsync(parent)
        finally:
            os.close(parent)
        return hashlib.sha256(raw).hexdigest()

    def _operation(self, name, input_name, digest, *, credential=None):
        runtime = 30 if name == "settle" else min(30 if name == "prepare" else 604800,
                                                  max(0.001, self.policy.deadline_unix - time.time()))
        return AzureGoalService.create(self.root / "controller" / name, self.root / input_name,
            digest, self.root / "exchange" / name, operation=name, uid=self.config["service_uid"],
            python_executable=self.config["python_executable"], runtime_seconds=runtime,
            grace_seconds=3, credential=credential, manager=self.manager)

    async def run(self, *, api_key):
        if self.started or self.closed:
            raise GoalInputError("terminal trial was already admitted; recover without replay")
        self.started = True
        try:
            prepared = await self._operation("prepare", "prepare.json", self.preparation_sha).run(timeout_seconds=40)
            config = prepared.value
            digest = self._input("goal.json", config.to_bytes())
            self.broker = TerminalBroker.create(self.root / "controller/terminal", self.policy.terminal.binding, self.backend)
            issuer = TerminalIssuerCredential(str(self.root / "rpc/terminal.sock"), os.geteuid(),
                self.config["service_uid"], digest, self.policy.terminal, secrets.token_hex(32))
            self.service = TerminalService(self.broker, issuer=issuer)
            await self.service.start(worker_gid=self.config["service_gid"])
            runner = self._operation("run", "goal.json", digest,
                credential=encode_azure_goal_credentials(config, api_key, terminal_issuer=issuer))
            result = None
            try:
                result = await runner.run(timeout_seconds=min(604800,
                    max(0.001, self.policy.deadline_unix - time.time()) + 15))
            except Exception:
                # Never start settlement until original-scope drainage is
                # proved. A missing launch acknowledgement cannot pass here.
                await runner.scope.stop_async("settle terminal goal after execution failure")
            settled = await self._operation("settle", "goal.json", digest).run(timeout_seconds=40)
            if result is not None and result.value != settled.value:
                raise GoalInputError("goal result changed during credential-free settlement")
            self.outcome = settled.value
            _write(self.fd, "outcome.json", _json_bytes({"input_sha256": digest,
                "result_sha256": settled.result_sha256, "outcome": self.outcome.to_dict()}))
            if not grading_ready(self.outcome):
                raise GoalInputError("terminal goal is not settled within its admitted budget")
            self.service.seal_for_grading()
            _write(self.fd, "grading.json", _json_bytes({"input_sha256": digest,
                "result_sha256": settled.result_sha256, "binding": asdict(self.broker.binding)}))
            return self.outcome
        except BaseException as original:
            try:
                await self.close()
            except BaseException as failure:
                raise BaseExceptionGroup("terminal goal and cleanup failed", [original, failure]) from None
            raise

    async def close(self):
        if self.closed:
            return
        errors, cancelled, receipts = [], [], []
        for path in _scope_paths(self.root):
            try:
                scope = OwnedProcessScope(path, manager=self.manager)
            except BaseException as exc:
                errors.append(exc)
                continue
            try:
                await scope.stop_async("terminal trial ended")
            except asyncio.CancelledError as exc:
                cancelled.append(exc)  # stop_async retains its cleanup thread.
            except BaseException as exc:
                errors.append(exc)
            try:
                receipts.append(scope.stop("confirm terminal goal drainage"))
            except BaseException as exc:
                errors.append(exc)
        try:
            if self.service is not None:
                await self.service.close()
            else:
                operation = start_owned_thread(_owned_call, self.backend.stop_and_confirm)
                try:
                    await asyncio.wait((operation,))
                except asyncio.CancelledError as exc:
                    cancelled.append(exc)
                await _settle(operation)
        except asyncio.CancelledError as exc:
            cancelled.append(exc)
        except BaseException as exc:
            errors.append(exc)
        if not errors:
            _, binding, digest = _load_config(self.fd, self.root)
            record, exists = _drain_record(self.fd, digest, binding.container_id, receipts)
            if not exists:
                _write(self.fd, "drain.json", _json_bytes(record))
            _remove_launch_material(self.root)
            if self.broker is not None:
                self.broker.close()
            os.close(self.fd)
            self.closed = True
        if errors:
            raise BaseExceptionGroup("terminal trial cleanup is incomplete", [*errors, *cancelled])
        if cancelled:
            raise cancelled[0]
