"""Where a command runs, separated from what the command is.

Every probe in this project asks the same question — *did this test pass at
this historical tree?* — and the answer is only trustworthy if the environment
answering it is the one the benchmark built. On SWE-bench that environment is a
pinned image with a conda env, compiled C extensions, and installed
dependencies. It is emphatically not the harness's own interpreter, and it is
not a bare checkout of the repository either: a checkout is missing everything
``.gitignore`` hides, which on astropy or matplotlib means the compiled
extensions, which means every test errors on import.

So execution is a seam, with three implementations:

``LocalSandbox``     a subprocess in a directory. What the hermetic tests and
                     Gate 0 use, and what runs on a machine with no Docker.
``DockerSandbox``    a long-lived container on a pinned image. The real thing.
``ScriptedSandbox``  a recorded double that asserts *which commands we issued*
                     without running any of them, so the eval-script builders
                     and log parsers are testable at zero cost.

**The invariant that matters more than any other here.** A sandbox reports what
happened; it never interprets. If a container will not start, if ``git apply``
conflicts, if the interpreter is missing — that is an *infrastructure* failure
and it must surface as one. The moment such a failure is allowed to render as
"the test failed", the instrument manufactures the very phenomenon it exists to
measure, and no amount of downstream statistics can undo it. Callers convert
``ExecResult`` into verdicts; this module never does.

A sandbox is deliberately long-lived. The whole cost argument for scanning
*every* observation rather than bisecting is that the environment stays warm:
N observations amortise onto one container, not one container per verdict.
Lifetime is therefore explicit and owned by the caller.
"""

from __future__ import annotations

import contextlib
import io
import os
import shlex
import subprocess
import tarfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# Inherited environment is both a correctness hazard and a disclosure one. A
# probe command is built from dataset content and runs whatever the tree
# contains; handing it the harness's own environment leaks ANTHROPIC_API_KEY
# and points it at the harness's interpreter rather than the benchmark's. The
# allowlist is what a POSIX shell needs to function and nothing else.
DEFAULT_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR")

# GNU coreutils `timeout` exit codes. Used by DockerSandbox, which cannot
# interrupt a running exec from outside without leaking the process.
_TIMEOUT_EXIT = 124
_TIMEOUT_KILLED = 137


@dataclass(frozen=True)
class ExecResult:
    """The outcome of one command. Facts only, no interpretation."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def output(self) -> str:
        """Both streams, in the order a terminal would have shown them.

        Log parsers upstream are written against interleaved console output,
        and some runners (django's) report results on stderr.
        """
        if not self.stderr:
            return self.stdout
        if not self.stdout:
            return self.stderr
        return f"{self.stdout}\n{self.stderr}"


@runtime_checkable
class Sandbox(Protocol):
    """One prepared execution environment, alive across many commands."""

    workdir: str

    def exec(
        self, command: str, *, timeout: int = 600, env: dict[str, str] | None = None
    ) -> ExecResult: ...

    def put_text(self, path: str, text: str) -> None: ...

    def get_text(self, path: str) -> str: ...

    def close(self) -> None: ...


@runtime_checkable
class SandboxProvider(Protocol):
    """Opens sandboxes. The only thing that needs to know Docker exists."""

    def open(self, *, key: str, image: str = "") -> Sandbox: ...

    def prune(self) -> None: ...


def _filtered_env(
    allowlist: tuple[str, ...], extra: dict[str, str] | None = None
) -> dict[str, str]:
    env = {name: os.environ[name] for name in allowlist if name in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    if extra:
        env.update(extra)
    return env


# ---------------------------------------------------------------- local


class LocalSandbox:
    """A directory on this machine, entered by a subprocess.

    Correct for synthetic trajectories and for Gate 0, where the "environment"
    is a single Python file with no dependencies. It is *not* correct for a
    real SWE-bench instance, and the difference is not a matter of degree —
    see the module docstring. Nothing here should be mistaken for the real
    measurement path.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        env_allowlist: tuple[str, ...] = DEFAULT_ENV_ALLOWLIST,
    ) -> None:
        self.root = Path(root)
        self.workdir = str(self.root)
        self._allowlist = env_allowlist

    def exec(
        self, command: str, *, timeout: int = 600, env: dict[str, str] | None = None
    ) -> ExecResult:
        started = time.monotonic()
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_filtered_env(self._allowlist, env),
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                exit_code=_TIMEOUT_EXIT,
                stdout=_decode(exc.stdout),
                stderr=_decode(exc.stderr),
                timed_out=True,
                duration_s=time.monotonic() - started,
            )
        return ExecResult(
            exit_code=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            duration_s=time.monotonic() - started,
        )

    def put_text(self, path: str, text: str) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def get_text(self, path: str) -> str:
        return self._resolve(path).read_text(encoding="utf-8")

    def _resolve(self, path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate

    def close(self) -> None:
        return None

    def __enter__(self) -> LocalSandbox:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class LocalProvider:
    """Hands out directories under a root. No images, no daemon."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def open(self, *, key: str, image: str = "") -> LocalSandbox:
        path = self.root / key
        path.mkdir(parents=True, exist_ok=True)
        return LocalSandbox(path)

    def prune(self) -> None:
        return None


# ---------------------------------------------------------------- scripted


@dataclass
class ScriptedSandbox:
    """A sandbox that runs nothing and remembers everything.

    Exists so the parts most likely to be wrong — the eval script we build,
    the order we ship a patch in, how we classify a failure — are provable on
    a laptop with the Docker daemon down. Responses are matched by substring
    in insertion order, so a test states only the commands it cares about.
    """

    workdir: str = "/testbed"
    responses: list[tuple[str, ExecResult]] = field(default_factory=list)
    default: ExecResult = field(default_factory=lambda: ExecResult(0, "", ""))
    commands: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    closed: bool = False

    def on(self, pattern: str, result: ExecResult) -> ScriptedSandbox:
        self.responses.append((pattern, result))
        return self

    def exec(
        self, command: str, *, timeout: int = 600, env: dict[str, str] | None = None
    ) -> ExecResult:
        self.commands.append(command)
        for pattern, result in self.responses:
            if pattern in command:
                return result
        return self.default

    def put_text(self, path: str, text: str) -> None:
        self.files[path] = text

    def get_text(self, path: str) -> str:
        return self.files[path]

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> ScriptedSandbox:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------- docker


class DockerSandbox:
    """A long-lived container on a pinned image.

    Started as ``sleep infinity`` and kept alive, because the cost of an
    exhaustive scan is dominated by container startup: one start per instance
    is affordable, one per observation is not.

    ``docker`` is imported inside ``__init__`` rather than at module scope so
    that importing :mod:`taste.execution` costs nothing on a machine that has
    never installed it — which includes the development machine and every CI
    runner for the hermetic suite.

    Timeouts are enforced *inside* the container with coreutils ``timeout``
    rather than by a watchdog on this side. Docker's exec API gives no way to
    cancel a running exec, so a host-side timer would return control while the
    command kept running and kept writing to the tree — silently corrupting
    the next observation.
    """

    def __init__(
        self,
        *,
        image: str,
        name: str,
        workdir: str = "/testbed",
        platform: str = "linux/amd64",
        client: Any | None = None,
        on_close: Callable[[DockerSandbox], None] | None = None,
    ) -> None:
        if client is None:
            import docker

            client = docker.from_env()
        self._client = client
        self.image = image
        self.workdir = workdir
        # Whoever caches this sandbox must learn when it dies. close() removes
        # the container, so a cache entry that outlives close() is a handle to
        # nothing: every exec throws, the probe's fail-open wrapper renders
        # each throw as a hole, and the cell scores as CLEAN. See
        # DockerProvider for the run this cost.
        self._on_close = on_close
        self.container = client.containers.run(
            image,
            command="sleep infinity",
            name=name,
            detach=True,
            platform=platform,
            # `network_mode="none"`, NOT `network_disabled=True`.
            #
            # Both sever external access, which is the property the
            # measurement needs: a probe reaching a third-party host fails
            # intermittently for reasons unrelated to the agent, and an
            # intermittent pass->fail is indistinguishable by this instrument
            # from a silent regression.
            #
            # But `network_disabled` removes the network namespace outright,
            # including loopback. Measured consequence: matplotlib and sphinx
            # images ship `pytest_rerunfailures`, which binds a localhost
            # socket at pytest_configure time. With no `lo` that raises
            # `socket.gaierror` and pytest dies with INTERNALERROR before
            # collecting a single test -- 812 graded tests reported as holes
            # on one instance alone, and 23.5% of the entire dev-slice oracle
            # across six instances.
            #
            # `none` keeps a loopback interface and still resolves nothing
            # external. Verified both directions in the image.
            network_mode="none",
            auto_remove=False,
        )
        self._assert_workdir()

    def _assert_workdir(self) -> None:
        """Fail loudly now rather than silently on every command later.

        Docker resolves the exec working directory at *start*, so if it does
        not exist every subsequent exec dies with exit 127 and the OCI error
        text on **stdout**. Downstream that parses as "no results for any
        graded test" — an infrastructure hole, which is the correct
        classification but the wrong moment to discover it: the whole
        instance would replay as holes and score as a clean run.
        """
        probe = self.container.exec_run(
            ["test", "-d", self.workdir], demux=False
        )
        code = probe[0] if isinstance(probe, tuple) else probe.exit_code
        if code != 0:
            self.close()
            raise RuntimeError(
                f"workdir {self.workdir!r} does not exist in image {self.image!r}; "
                "every command would fail with exit 127 and be recorded as a hole"
            )

    def exec(
        self, command: str, *, timeout: int = 600, env: dict[str, str] | None = None
    ) -> ExecResult:
        wrapped = f"timeout --signal=KILL {int(timeout)} bash -lc {shlex.quote(command)}"
        started = time.monotonic()
        exit_code, (stdout, stderr) = self.container.exec_run(
            ["bash", "-c", wrapped],
            workdir=self.workdir,
            environment=env or {},
            demux=True,
        )
        duration = time.monotonic() - started
        return ExecResult(
            exit_code=exit_code,
            stdout=_decode(stdout),
            stderr=_decode(stderr),
            timed_out=exit_code in (_TIMEOUT_EXIT, _TIMEOUT_KILLED),
            duration_s=duration,
        )

    def put_text(self, path: str, text: str) -> None:
        """Ship a file in via the archive API.

        Not ``exec("cat > file")``: the payloads here are unified diffs
        containing arbitrary repository content, and routing them through a
        shell would require quoting that is impossible to get right for every
        instance in the dataset.
        """
        directory, _, filename = path.rpartition("/")
        payload = text.encode("utf-8")
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            info = tarfile.TarInfo(name=filename)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
        buffer.seek(0)
        self.container.put_archive(directory or "/", buffer.getvalue())

    def get_text(self, path: str) -> str:
        stream, _ = self.container.get_archive(path)
        buffer = io.BytesIO(b"".join(stream))
        buffer.seek(0)
        with tarfile.open(fileobj=buffer) as archive:
            member = archive.next()
            if member is None:
                raise FileNotFoundError(path)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(path)
            return extracted.read().decode("utf-8", errors="replace")

    def close(self) -> None:
        with contextlib.suppress(Exception):  # teardown must not mask a result
            self.container.remove(force=True)
        callback, self._on_close = self._on_close, None  # close() may run twice
        if callback is not None:
            with contextlib.suppress(Exception):  # eviction must not either
                callback(self)

    def __enter__(self) -> DockerSandbox:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class DockerProvider:
    """Opens one container per key, on digest-pinned images.

    ``prune`` exists because nothing upstream removes evaluation images and
    they run to several gigabytes each; a sweep over a few dozen instances
    fills a disk without it.

    **The cache never outlives what it caches.** Arms interleave per task in
    one process, so the same key is opened once per cell — and the scorer
    closes its sandbox when a cell ends. The cache used to keep the entry
    anyway, so the next cell of the same instance was handed a sandbox whose
    container had been force-removed: every exec threw, the probe executor's
    fail-open wrapper turned each throw into a hole, and the cell scored as
    CLEAN with the agent phase already paid. Ownership is explicit now — a
    closing sandbox evicts its own entry — and a cached entry is served only
    after its container is confirmed live, because a container can also die
    without anyone calling ``close`` (daemon restart, OOM kill).
    """

    def __init__(
        self,
        *,
        prefix: str = "taste",
        platform: str = "linux/amd64",
        client: Any | None = None,
    ) -> None:
        if client is None:
            import docker

            client = docker.from_env()
        self._client = client
        self.prefix = prefix
        self.platform = platform
        self._open: dict[str, DockerSandbox] = {}

    def open(self, *, key: str, image: str = "") -> DockerSandbox:
        if not image:
            raise ValueError(f"no image pinned for {key!r}")
        cached = self._open.get(key)
        if cached is not None:
            if self._alive(cached):
                return cached
            # Dead without a close() — evict, then fall through to reopen.
            self._open.pop(key, None)
            cached.close()
        name = f"{self.prefix}-{key}".replace("/", "_").replace(":", "_")[:200]
        self._remove_stale(name)
        sandbox = DockerSandbox(
            image=image,
            name=name,
            platform=self.platform,
            client=self._client,
            on_close=lambda closing: self._evict(key, closing),
        )
        self._open[key] = sandbox
        return sandbox

    def _alive(self, sandbox: DockerSandbox) -> bool:
        """Whether a cached sandbox's container still exists and runs.

        Asked of the daemon, not the cached handle: a removed container's
        Python object answers exec_run happily and every call throws
        downstream, where the probe's fail-open wrapper renders it as a hole.
        """
        try:
            current = self._client.containers.get(sandbox.container.id)
            return getattr(current, "status", "") == "running"
        except Exception:
            return False

    def _evict(self, key: str, sandbox: DockerSandbox) -> None:
        """Drop the cache entry when *this* sandbox closes — never a successor
        that has since taken the key."""
        if self._open.get(key) is sandbox:
            del self._open[key]

    def _remove_stale(self, name: str) -> None:
        """A crashed sweep leaves containers behind; the name would collide."""
        with contextlib.suppress(Exception):  # absence is the expected case
            self._client.containers.get(name).remove(force=True)

    def close_all(self) -> None:
        for sandbox in list(self._open.values()):
            sandbox.close()
        self._open.clear()

    def prune(self) -> None:
        self.close_all()
        with contextlib.suppress(Exception):  # reclaiming disk is best-effort
            self._client.images.prune(filters={"dangling": False})


def _decode(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)
