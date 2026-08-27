"""One environment: the agent executes where it is measured.

The instrument's probes always ran inside the task's pinned container image.
The agent did not: its tools and its Monitor ran on the host, in a bare
checkout where the project was never built. On the host `import matplotlib`
*succeeds* -- the working directory shadows the installed package and an
uncompiled source tree imports as a namespace package with a garbage version
-- so the failure did not even look like an environment failure. It looked
like the agent being bad at its job, for 26 of 28 zero-step runs, across
$110 of pilots. That is bug 20, and this module is the seam that closes it.

The design keeps the host workspace as the single source of truth for the
*instrument* -- Memory's checkpoints, the shadow timeline, `patch_for`, every
diff -- and makes the container a faithful execution mirror:

- host-side mutations (``write_file``, a rollback's reset) are marked dirty
  and pushed before the next container command runs;
- container-side mutations (anything a shell command writes) are pulled back
  to the host *before* the shadow observation fires, so the timeline sees
  exactly what the agent's command produced.

Neither direction is optional. Push-only, and the Monitor grades a tree the
agent never tested; pull-only, and the agent tests code it never wrote. Both
failure modes are silent, which is why the contract lives in one object
instead of being distributed across call sites as discipline.

The container side keeps its own git repo purely as sync bookkeeping: a
baseline commit advanced after every transfer, so "what changed since the
last sync" is one `git status` rather than a tree walk. It is never the
measurement -- the host's shadow ref is.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from taste.execution import ExecResult

__all__ = ["CONTAINER_EXCLUDES", "SandboxRouter", "prepare_container_tree"]

#: Junk a test run generates that must never cross the sync boundary: pulled
#: into the host tree it would pollute every shadow observation (the npm
#: lesson, bug D3, taught what an install does to attribution's changed-files
#: term). Written to the container's git exclude file so `git status` never
#: reports them, and filtered again on pull for sandboxes without git.
CONTAINER_EXCLUDES = (
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".pytest_cache/",
    "*.egg-info/",
    ".eggs/",
    ".tox/",
    ".coverage",
    ".coverage.*",
    ".hypothesis/",
    "node_modules/",
    ".mypy_cache/",
    ".ruff_cache/",
)

_GIT_ID = "-c user.name=taste -c user.email=taste@local"


def prepare_container_tree(
    sandbox: Any, *, workdir: str = "/testbed", hide_upstream: bool = True
) -> str:
    """Turn the container's tree into a sync-ready baseline. Idempotent.

    Returns the baseline commit sha.

    ``hide_upstream=True`` moves the image's original ``.git`` out of the
    tree first. The images clone the full repository before checking out
    ``base_commit``, so the upstream object store can contain refs *after*
    the task's cut point -- including the fix the agent is being asked to
    write. The host workspace was deliberately materialised with no upstream
    history; an agent whose shell can run ``git log`` against it would have
    more history than the task permits. Probe containers keep it (the replay
    path never exposes a shell to the agent).
    """
    q = shlex.quote(workdir)
    probe = sandbox.exec(f"test -d {q}", timeout=30)
    if probe.exit_code != 0:
        # Refuse loudly. Continuing "works": git and the transport happily
        # create the directory piecemeal wherever the commands land — the
        # first misconfigured run of this code built a real /testbed on the
        # development host as a side effect and reported nothing.
        raise RuntimeError(f"sync workdir {workdir!r} does not exist in this sandbox")
    marker = sandbox.exec(f"git -C {q} rev-parse --verify -q refs/heads/taste-baseline", timeout=60)
    if marker.exit_code == 0 and marker.stdout.strip():
        return marker.stdout.strip()

    if hide_upstream:
        stash = shlex.quote(f"{workdir.rstrip('/')}_upstream_git")
        sandbox.exec(
            f"if [ -e {q}/.git ] && [ ! -e {stash} ]; then mv {q}/.git {stash}; fi",
            timeout=60,
        )
        # Not `init -b`: that flag needs git >= 2.28 and the older instance
        # images ship 2.25. symbolic-ref does the same on any git.
        sandbox.exec(
            f"git -C {q} init -q && git -C {q} symbolic-ref HEAD refs/heads/taste-baseline",
            timeout=60,
        )
    else:
        # Keep upstream objects (probe containers): baseline is a new branch
        # on top of whatever state the image build left, pre_install edits
        # included -- resetting to upstream base_commit instead is bug B8,
        # which killed a whole repo family's oracle.
        sandbox.exec(f"git -C {q} checkout -q -b taste-baseline", timeout=60)

    excludes = "\n".join(CONTAINER_EXCLUDES) + "\n"
    sandbox.put_text(f"{workdir}/.git/info/exclude", excludes)
    sandbox.exec(f"git -C {q} add -A", timeout=300)
    sandbox.exec(
        f"git -C {q} {_GIT_ID} commit -q --allow-empty -m taste-baseline", timeout=300
    )
    sha = sandbox.exec(f"git -C {q} rev-parse HEAD", timeout=60)
    if sha.exit_code != 0 or not sha.stdout.strip():
        raise RuntimeError(
            f"could not establish a sync baseline in {workdir}: {sha.stdout} {sha.stderr}"
        )
    return sha.stdout.strip()


class SandboxRouter:
    """The one object that owns host<->container coherence.

    ``exec`` is the only way routed code runs a command: push what the host
    knows, run activated in the workdir, pull what the container wrote, and
    only then return -- so the caller's next action (a shadow observation, a
    Monitor verdict, another tool call) sees one consistent tree.
    """

    def __init__(
        self,
        sandbox: Any,
        workspace: Path,
        *,
        workdir: str = "/testbed",
    ) -> None:
        self.sandbox = sandbox
        self.workspace = Path(workspace)
        self.workdir = workdir
        self._q = shlex.quote(workdir)
        self._dirty: set[str] = set()
        #: Paths pull could not transport (symlinks, vanished files) — counted
        #: rather than silently dropped.
        self.skipped: list[str] = []
        prepare_container_tree(sandbox, workdir=workdir)

    # ------------------------------------------------------------ marking

    def mark_dirty(self, rel_path: str) -> None:
        """A host-side mutation the container has not seen yet."""
        self._dirty.add(rel_path)

    def mark_reset(self, repo: Any, old_sha: str, new_sha: str) -> None:
        """A rollback moved the host tree wholesale; mark exactly what moved.

        The delta between the two commits, not "everything": a full re-push
        per reset would cost minutes on a real tree, and the reset touches
        only what the discarded attempt touched.
        """
        if old_sha == new_sha:
            return
        names = repo.git.diff("--name-only", old_sha, new_sha)
        for name in names.splitlines():
            if name.strip():
                self._dirty.add(name.strip())

    # ------------------------------------------------------------ transfer

    def push(self) -> None:
        if not self._dirty:
            return
        deletions: list[str] = []
        for rel in sorted(self._dirty):
            host = self.workspace / rel
            if host.is_file():
                self.sandbox.put_bytes(f"{self.workdir}/{rel}", host.read_bytes())
            elif not host.exists():
                deletions.append(rel)
        if deletions:
            quoted = " ".join(shlex.quote(f"{self.workdir}/{d}") for d in deletions)
            self.sandbox.exec(f"rm -f {quoted}", timeout=120)
        self._dirty.clear()
        self._advance_baseline()

    def pull(self) -> tuple[str, ...]:
        """Bring container-side changes home. Returns the pulled paths.

        Runs after *every* routed command, mutating or not: a verification
        command has no obligation to be read-only (``makemigrations`` is a
        legal check), and a pull on an unchanged tree costs one git status.
        """
        status = self.sandbox.exec(
            f"git -C {self._q} status --porcelain -z", timeout=300
        )
        if status.exit_code != 0:
            raise RuntimeError(
                f"sync pull cannot see the container tree: {status.stderr or status.stdout}"
            )
        changed: list[tuple[str, str]] = []
        entries = [e for e in status.stdout.split("\0") if e.strip()]
        i = 0
        while i < len(entries):
            entry = entries[i]
            code, path = entry[:2], entry[3:]
            if code.startswith("R"):
                # rename: -z emits the new name in this entry, old name next.
                old = entries[i + 1] if i + 1 < len(entries) else None
                if old:
                    changed.append(("D", old))
                    i += 1
                changed.append(("A", path))
            else:
                kind = "D" if "D" in code else "A"
                changed.append((kind, path))
            i += 1

        pulled: list[str] = []
        for kind, rel in changed:
            if self._excluded(rel):
                continue
            host = self.workspace / rel
            if kind == "D":
                if host.is_file():
                    host.unlink()
                    pulled.append(rel)
                continue
            try:
                payload = self.sandbox.get_bytes(f"{self.workdir}/{rel}")
            except (FileNotFoundError, KeyError, IsADirectoryError):
                # Symlinks and other irregular members: the tar the transport
                # returns carries no file content for them, and sphinx's test
                # fixtures are full of them — two paid cells died here before
                # this except existed. Skipped, counted, and visible: a
                # symlink in fixtures is not agent work the instrument must
                # mirror, but a skip that vanishes would be D-class.
                self.skipped.append(rel)
                continue
            host.parent.mkdir(parents=True, exist_ok=True)
            host.write_bytes(payload)
            pulled.append(rel)
        if changed:
            self._advance_baseline()
        return tuple(pulled)

    # ------------------------------------------------------------ execution

    def exec(self, command: str, *, timeout: int = 600) -> ExecResult:
        self.push()
        run = getattr(self.sandbox, "exec_in_env", self.sandbox.exec)
        result = run(command, timeout=timeout)
        self.pull()
        return result

    # ------------------------------------------------------------ internals

    def _advance_baseline(self) -> None:
        self.sandbox.exec(f"git -C {self._q} add -A", timeout=300)
        self.sandbox.exec(
            f"git -C {self._q} {_GIT_ID} commit -q --allow-empty -m taste-sync",
            timeout=300,
        )

    @staticmethod
    def _excluded(rel: str) -> bool:
        parts = rel.split("/")
        if any(part == "__pycache__" or part.endswith((".pyc", ".pyo")) for part in parts):
            return True
        junk = {".pytest_cache", ".tox", ".eggs", ".hypothesis", "node_modules",
                ".mypy_cache", ".ruff_cache"}
        if any(part in junk or part.endswith(".egg-info") for part in parts):
            return True
        return parts[-1] == ".coverage" or parts[-1].startswith(".coverage.")
