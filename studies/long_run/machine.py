"""A task's machine as an Apptainer instance, so that what an agent runs stays on its machine.

mini-swe-agent's own Singularity environment starts each command in a container of its own that shares the node's process list. On a shared
cluster that let one agent list and kill the study's own sessions, and it could have reached anything else its user runs on the node. Here each
machine has its own process list, so a command can signal only that machine's processes. Files in /tmp, and work left in the background, last from
one command to the next, as on a real machine. A command's output goes to a file, so a server it leaves running holds nothing open.

Each machine reads its own wall when it starts, before anything the agent sends reaches it, and one that can see the node's processes or this
account's files is refused. The check only reads.

The network is the node's: sessions of a run share it, so tasks that open a port are left out of the session file.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from minisweagent.environments.singularity import SingularityEnvironment

# Duke's Apptainer has no user-id map, and its fakeroot helper needs a newer C library than the task images have. Without the
# helper the agent is still root inside and its writes stay; a package whose install changes file owners may only half install.
START_ARGS = ["--contain", "--cleanenv", "--fakeroot", "--ignore-fakeroot-command"]
EXEC_ARGS = ["--cleanenv", "--ignore-fakeroot-command"]
# A warning from the container runtime would come first in a command's output, where the agent looks for its submission.
GLOBAL_ARGS = ["--silent"]
START_TIMEOUT = 300
MOST_OUTPUT = 20_000_000  # bytes read back of one command's output, its end kept: what the model and the record can use is far less
MOST_PROCESSES = 30  # a machine that has just started sees about 4 of its own, and a node has hundreds


class Machine(SingularityEnvironment):
    def __init__(self, *, image: str, executable: str = "apptainer", **settings: Any) -> None:
        self.name: str | None = None
        super().__init__(image=image, executable=executable, global_args=GLOBAL_ARGS, exec_args=EXEC_ARGS, **settings)
        self.scratch = Path(f"{self.sandbox_dir}.tmp")
        self.scratch.mkdir()  # the machine's /tmp, with no small limit on what it holds
        name = f"aos-{uuid.uuid4().hex[:16]}"
        started = subprocess.run([executable, *GLOBAL_ARGS, "instance", "start", *START_ARGS, "--workdir", str(self.scratch), "--writable",
                                  str(self.sandbox_dir), name], capture_output=True, text=True, errors="replace", timeout=START_TIMEOUT)
        if started.returncode:
            self.cleanup()
            raise RuntimeError(f"the machine would not start: {started.stdout[-300:]} {started.stderr[-300:]}")
        self.name = name
        self.wall = self._read_wall()

    def _read_wall(self) -> dict[str, Any]:
        """What this machine can see of the node, or a refusal. Reads only, and sends no signal to anything."""
        outside = [str(Path.home()), str(Path.cwd())]
        counts = "".join(f"; ls -A {shlex.quote(folder)} 2>/dev/null | wc -l" for folder in outside)
        read = self.execute({"command": f"ls -d /proc/[0-9]* 2>/dev/null | wc -l{counts}"}, "/")
        seen = read["output"].split()
        problem = None
        if not all(word.isdigit() for word in seen) or len(seen) != 1 + len(outside):
            problem = f"it would not say what it can see: {read['output'][-300:]}"
        elif not 0 < int(seen[0]) <= MOST_PROCESSES:
            problem = f"it sees {seen[0]} processes"
        elif any(int(word) for word in seen[1:]):
            problem = f"it can see this account's files: {dict(zip(outside, seen[1:], strict=True))}"
        if problem:
            self.cleanup()
            raise RuntimeError(f"the machine's wall does not hold: {problem}")
        return {"processes": int(seen[0]), "files_outside": {folder: int(word) for folder, word in zip(outside, seen[1:], strict=True)}}

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """As mini-swe-agent's, with the command run inside this machine."""
        command = [self.config.executable, *GLOBAL_ARGS, "exec", *EXEC_ARGS]
        folder = cwd or self.config.cwd
        if folder and folder != "/":
            command += ["--pwd", folder]
        for key, value in self.config.env.items():
            command += ["--env", f"{key}={value}"]
        command += [f"instance://{self.name}", "bash", "-c", action.get("command", "")]
        with tempfile.TemporaryFile() as sink:
            failure: dict[str, Any] = {}
            try:
                code = subprocess.run(command, stdout=sink, stderr=subprocess.STDOUT, timeout=timeout or self.config.timeout).returncode
            except subprocess.TimeoutExpired as error:
                code = -1
                failure = {"exception_info": f"An error occurred while executing the command: {error}",
                           "extra": {"exception_type": type(error).__name__, "exception": str(error)}}
            size = sink.seek(0, 2)
            sink.seek(max(0, size - MOST_OUTPUT))
            text = sink.read().decode("utf-8", errors="replace")
        output = {"output": text, "returncode": code, "exception_info": "", **failure}
        self._check_finished(output)
        return output

    def cleanup(self) -> None:
        if getattr(self, "name", None):
            subprocess.run([self.config.executable, *GLOBAL_ARGS, "instance", "stop", self.name], capture_output=True, timeout=START_TIMEOUT)
            self.name = None
        scratch = getattr(self, "scratch", None)
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
        super().cleanup()
