"""Run one benchmark task as a goal, in Harbor's standard Docker environment.

    harbor run -p <tasks> -a taste.benchmarks.harbor_agent:TasteAgent -m azure/gpt-6-astra \\
        --ak worker_python=<python of the worker environment> [--ak <setting>=<value> ...]

The task is run as published: Harbor's own Docker environment, its container,
user, working directory, network rule and time limit, with no environment
subclass, extra Compose file, mount or override. The coordinator, its workers
and their monitors run on the host, outside the task container; only shell
commands enter it, one at a time, through the terminal broker.

What the benchmark receives is the sealed container and one trajectory at
``trajectory.json`` in Harbor's agent log directory: the instruction, every
worker step in the order it happened, and the coordinator's final reply. The
trial is handed over for grading whatever this system's own audit found; a
gap is recorded in the trajectory's ``extra.audit_flags`` and in the trial
metadata, never used to withhold the task from its verifier.

Requirements of the machine, not of the task: Harbor runs as root on the
Docker host (cgroup v2, systemd), an unprivileged account runs the goal's
processes, and AZURE_OPENAI_BASE_URL and AZURE_OPENAI_API_KEY are in Harbor's
environment. The key is delivered to the goal as a private credential and
never enters the task container, a prompt or this record.

This module is the only one that imports Harbor. Trial settings live in
``harbor_settings``; they are the run's disclosed configuration.
"""

from __future__ import annotations

import asyncio
import json
import os
import pwd
import secrets
import sqlite3
import subprocess
import time
from dataclasses import fields
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities

from taste import __version__
from taste.benchmarks.azure_terminal_trial import AzureTerminalTrial
from taste.benchmarks.flat_trajectory import encode, flat_trajectory, ledger_trajectory
from taste.benchmarks.harbor_settings import (
    TrialSettings,
    agent_timeout_seconds,
    compose_project,
    safe_parent,
)
from taste.brains.docker_terminal import COMPOSE_PROJECT_LABEL, DockerTerminalBackend
from taste.brains.goal_entrypoint import python_source_digest
from taste.brains.terminal_broker import MAX_TERMINAL_OUTPUT_BYTES, TerminalResult

_SETTINGS = frozenset(item.name for item in fields(TrialSettings))
_DOCKER_SECONDS = 20


def _docker(socket, *arguments):
    done = subprocess.run(["docker", "--host", "unix://" + socket, *arguments],
                          capture_output=True, timeout=_DOCKER_SECONDS)
    if done.returncode != 0:
        raise RuntimeError("Docker could not describe the task's container")
    return done.stdout.decode()


def _ledger(root):
    """Every command the task container was asked to run, in order."""
    database = Path(root) / "controller/terminal/terminal.sqlite3"
    if not database.is_file():
        return []
    rows = []
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        for payload, status, code, out, err, out_dropped, err_dropped, terminated in connection.execute(
                "SELECT payload,status,code,stdout,stderr,stdout_dropped_bytes,stderr_dropped_bytes,"
                "terminated FROM requests ORDER BY rowid"):
            request = json.loads(payload)
            rows.append({**{key: request[key] for key in ("request_id", "command", "cwd", "timeout_seconds")},
                         "status": status,
                         "result": (TerminalResult(code, out or b"", err or b"", out_dropped, err_dropped,
                                                   terminated) if status == "completed" else None)})
    return rows


class TasteAgent(BaseAgent):
    capabilities = AgentCapabilities(atif=True)

    def __init__(self, logs_dir, model_name=None, *, worker_python=None, worker_user=None,
                 trials_root=None, agent_timeout_sec=None, docker_socket="/var/run/docker.sock",
                 **kwargs):
        options = {name: kwargs.pop(name) for name in list(kwargs) if name in _SETTINGS}
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        if model_name is not None:
            options.setdefault("model", str(model_name).rsplit("/", 1)[-1])
        self.settings = TrialSettings.from_options(options)
        self.worker_python = worker_python or os.environ.get("TASTE_WORKER_PYTHON")
        self.worker_user = worker_user or os.environ.get("TASTE_WORKER_USER", "bugbash")
        self.trials_root = Path(trials_root or os.environ.get("TASTE_TRIALS_ROOT", "/var/lib/taste-trials"))
        self.timeout_override = agent_timeout_sec
        self.docker_socket = docker_socket
        self.owner = None

    @staticmethod
    def name():
        return "taste"

    def version(self):
        return f"{__version__}+{python_source_digest()[:12]}"

    async def setup(self, environment):
        """Identify the published container; change nothing in it."""
        if type(environment).__name__ != "DockerEnvironment":
            raise RuntimeError("this agent's terminal needs Harbor's local Docker environment")
        if os.geteuid() != 0:
            raise RuntimeError("the trial owner must run as root on the Docker host")
        if not self.worker_python or not Path(self.worker_python).is_absolute():
            raise RuntimeError("worker_python must name the worker environment's interpreter")
        account = pwd.getpwnam(self.worker_user)
        if account.pw_uid == 0:
            raise RuntimeError("the goal's processes need an unprivileged account")
        self._service_uid = account.pw_uid
        self._project = compose_project(environment.session_id)
        listed = await asyncio.to_thread(
            _docker, self.docker_socket, "ps", "-q", "--no-trunc",
            "--filter", f"label={COMPOSE_PROJECT_LABEL}={self._project}",
            "--filter", "label=com.docker.compose.service=main")
        containers = listed.split()
        if len(containers) != 1:
            raise RuntimeError("the task's Compose project has no single running main container")
        info = json.loads(await asyncio.to_thread(_docker, self.docker_socket, "inspect", containers[0]))[0]
        self._container, self._image = info["Id"], info["Image"]
        workdir = environment.task_env_config.workdir or info["Config"].get("WorkingDir") or "/"
        self._workdir = os.path.normpath(workdir)
        user = environment.default_user
        self._exec_user = None if user is None else str(user)
        self._task_toml = Path(environment.environment_dir).parent / "task.toml"

    async def run(self, instruction, environment, context):
        started = time.time()
        settings, token = self.settings, secrets.token_hex(16)
        deadline, container_deadline = settings.deadlines(
            started, agent_timeout_seconds(self._task_toml, self.timeout_override))
        endpoint = os.environ.get("AZURE_OPENAI_BASE_URL", "").rstrip("/") + "/"
        key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        if not key or endpoint == "/":
            raise RuntimeError("AZURE_OPENAI_BASE_URL and AZURE_OPENAI_API_KEY must be set for Harbor")
        backend = await asyncio.to_thread(
            DockerTerminalBackend.admit, self.docker_socket, self._container, token, container_deadline,
            output_limit=MAX_TERMINAL_OUTPUT_BYTES, compose_project=self._project,
            image_id=self._image, exec_user=self._exec_user)
        policy = settings.policy(endpoint, deadline, owner_token=token,
                                 container_id=backend.environment_id, workdir=self._workdir)
        goal = settings.goal("trial-" + token[:20], instruction)
        root = safe_parent(self.trials_root) / token
        self.owner = AzureTerminalTrial.create(
            root, backend, goal, policy, service_uid=self._service_uid,
            python_executable=self.worker_python, max_generations=settings.generations,
            wall_clock_seconds=deadline - started, max_planner_failures=settings.max_planner_failures)
        context.metadata = {"taste": {"trial": token, "configuration": settings.disclosure()}}
        outcome = None
        try:
            outcome = await self.owner.run(api_key=key)
        finally:
            # Also after the benchmark's own time limit cancelled the run: the
            # owner sealed the container first, and the record must be on disk
            # before the verifier reads it.
            try:
                self._record(goal.goal_id, instruction, outcome, context, started)
            finally:
                if self.owner.sealed:
                    try:
                        await self.owner.release()
                    except Exception as failure:
                        # The record is written and the container is sealed
                        # and running. A resource this owner could not free
                        # is no reason to keep the trial from its verifier.
                        context.metadata["taste"]["release_failed"] = type(failure).__name__

    def _record(self, goal_id, instruction, outcome, context, started):
        owner, flags = self.owner, list(self.owner.audit_flags)
        # A cancelled run returned nothing, but it was settled before it was
        # sealed: what it is known to have spent is still reported.
        outcome = outcome if outcome is not None else owner.outcome
        nested = None
        try:
            if owner.trajectory_path is not None:
                nested = json.loads(owner.trajectory_path.read_bytes())
                record = flat_trajectory(nested, audit_flags=flags)
            else:
                record = ledger_trajectory(goal_id, instruction, _ledger(owner.root),
                                           audit_flags=flags or ["no_settled_evidence"])
        except (OSError, ValueError, KeyError, sqlite3.Error) as failure:
            # Never let our own record hide the task from its verifier.
            record = ledger_trajectory(goal_id, instruction, [], audit_flags=[
                *flags, "record_failed:" + type(failure).__name__])
        target = Path(self.logs_dir) / "trajectory.json"
        temporary = target.with_name(".trajectory.json." + secrets.token_hex(4))
        temporary.write_bytes(encode(record))
        temporary.chmod(0o644)
        os.replace(temporary, target)

        summary = context.metadata["taste"]
        summary.update(audit_flags=record["extra"]["audit_flags"], sealed=owner.sealed,
                       seconds=round(time.time() - started, 1),
                       final_reply=bool(record["extra"].get("final_reply_present")),
                       record=record["extra"]["record"])
        if outcome is not None:
            budget = outcome.budget
            summary.update(stop_reason=outcome.stop_reason, complete=outcome.complete,
                           generations=outcome.generations,
                           unsettled_usd=budget.reserved_usd, exact_cost=budget.enforceable
                           and budget.reserved_usd == 0)
            context.cost_usd = budget.known_spent_usd
        totals = (nested or {}).get("final_metrics") or {}
        if "total_prompt_tokens" in totals:
            context.n_input_tokens = totals["total_prompt_tokens"]
            context.n_cache_tokens = totals.get("total_cached_tokens")
            context.n_output_tokens = totals.get("total_completion_tokens")
