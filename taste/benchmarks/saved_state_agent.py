"""Grade a state an earlier trial saved: its files back in a fresh container, nothing else.

Each saved state of a supervised trial answers one question afterwards: would
the task's own verifier have passed had the run stopped there? This Harbor
agent puts the state's files back in the task's fresh container and ends, and
Harbor then runs the verifier as it does after any agent. No model is called
and no worker runs.

    harbor run -p <task> -a taste.benchmarks.saved_state_agent:SavedStateAgent \\
        --ak saved_state=<a checkpoint's manifest file> ...

A trial's checkpoints are kept as ``<id>.json`` beside their tars, and a final
save as ``manifest.json``; either file names the tar it needs, which is read
from the same directory. A partial save left files out, and a restore that did
not come back exact grades something the run never held: neither is graded.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities

from taste import __version__
from taste.benchmarks.branch_trial import load_saved_state, restore_saved_state  # noqa: F401
from taste.benchmarks.harbor_agent import _docker
from taste.benchmarks.harbor_settings import compose_project
from taste.brains.docker_terminal import COMPOSE_PROJECT_LABEL, DockerTerminalBackend
from taste.brains.terminal_broker import MAX_TERMINAL_OUTPUT_BYTES

# A restore is copies through the Docker API; this bounds one that hangs.
RESTORE_SECONDS = 1800


class SavedStateAgent(BaseAgent):
    capabilities = AgentCapabilities(atif=False)

    def __init__(self, logs_dir, model_name=None, *, saved_state=None,
                 docker_socket="/var/run/docker.sock", **kwargs):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        if not saved_state:
            raise ValueError("saved_state must name a checkpoint's manifest file")
        self.saved_state = Path(saved_state)
        self.docker_socket = docker_socket

    @staticmethod
    def name():
        return "taste-saved-state"

    def version(self):
        return __version__

    async def setup(self, environment):
        """Identify the published container, as the Taste agent does; change nothing in it."""
        if type(environment).__name__ != "DockerEnvironment":
            raise RuntimeError("restoring a saved state needs Harbor's local Docker environment")
        if os.geteuid() != 0:
            raise RuntimeError("restoring a saved state needs the Docker socket: run as root")
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
        user = environment.default_user
        self._exec_user = None if user is None else str(user)

    async def run(self, instruction, environment, context):
        backend = await asyncio.to_thread(
            DockerTerminalBackend.admit, self.docker_socket, self._container, secrets.token_hex(16),
            time.time() + RESTORE_SECONDS, output_limit=MAX_TERMINAL_OUTPUT_BYTES,
            compose_project=self._project, image_id=self._image, exec_user=self._exec_user)
        account = await asyncio.to_thread(restore_saved_state, backend, self.saved_state)
        context.metadata = {"saved_state": {**account, "image_id": self._image}}
