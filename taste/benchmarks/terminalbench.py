"""Terminal-Bench tasks in the Harbor format, as a second substrate.

A Harbor task is four things on disk::

    task.toml           name, description, authors, keywords
    instruction.md      what the agent is asked to do
    environment/        a Dockerfile and whatever it needs
    tests/test.sh       programmatic verification

**Why this substrate is interesting to us, and how it differs from SWE-bench.**
SWE-bench hands us a *benchmark-defined* previously-passing set: PASS_TO_PASS
was computed by its authors before we existed, which is what makes a silent
regression there an externally-defined event. Harbor tasks give us nothing of
the kind — most are built from scratch, so at the first observation nothing
passes at all. That is exactly why Commit0 was cut from the evidentiary path.

But a long build establishes its own previously-passing state *during the run*.
A test the agent gets passing at observation 20 and breaks at observation 40 is
a regression by the same definition, and :func:`taste.replay.episodes_from`
already handles it correctly: it refuses to open an episode for a check that
never passed, so from-scratch tasks yield episodes only where the agent really
did break its own work. The oracle is within-run rather than external, and the
paper must say so — it is a weaker claim than the SWE-bench one, and a
different one.

**A caution about scale.** The published `terminal-bench-challenges` set is
three tasks, and each is a multi-hour, token-intensive project — one asks for a
faster `rustc` with the full test suite still passing. They are the right shape
for demonstrating long-horizon behaviour and the wrong shape for a quick trial.
Check :attr:`HarborTask.is_long_running` before committing a budget to one.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from taste.replay import SuiteProbe

CANARY = re.compile(r"<!--\s*harbor-canary[^>]*-->\s*", re.I)

DEPENDENCY_IGNORES = """\
# Installed dependencies and build output are not the agent's work, and
# counting them as such defeats attribution rather than merely adding noise.
# Measured on wasm-render: one `npm install` put 3,640 of 3,641 "changed
# files" into a single observation. `modified_files_at` is the set the
# coverage rule intersects against, so when everything reads as modified that
# term goes vacuous and any coverage overlap links -- attributing essentially
# everything and erasing the phenomenon the rule exists to isolate.
node_modules/
__pycache__/
*.pyc
.venv/
venv/
target/
dist/
build/
.pytest_cache/
.mypy_cache/
.tox/
*.egg-info/
"""

# Signals that a task is a multi-hour build rather than a trial. The list grew
# after the first version scored `inference-engine-codegolf` as light: its
# keywords are cuda / moe / multi-gpu / nccl, none of which mentioned building
# anything, and the task needs hardware this project does not have.
_HEAVY = (
    "compiler", "kernel", "renderer", "rasterizer", "emulator", "browser",
    "cuda", "gpu", "nccl", "inference", "serving", "wasm", "rust",
    "operating-system", "database", "distributed",
)


@dataclass(frozen=True)
class HarborTask:
    """One Terminal-Bench task, read from its directory."""

    name: str
    description: str
    instruction: str
    root: Path
    keywords: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()
    schema_version: str = ""

    @property
    def slug(self) -> str:
        return self.name.split("/")[-1]

    @property
    def dockerfile(self) -> Path:
        return self.root / "environment" / "Dockerfile"

    @property
    def test_script(self) -> Path:
        return self.root / "tests" / "test.sh"

    @property
    def runnable(self) -> bool:
        """Whether this task has everything needed to actually execute."""
        return self.dockerfile.is_file() and self.test_script.is_file()

    @property
    def heavy_signals(self) -> tuple[str, ...]:
        """Which terms suggested this is a large build. Facts, not a verdict."""
        haystack = f"{self.name} {self.description} {' '.join(self.keywords)}".lower()
        return tuple(word for word in _HEAVY if word in haystack)

    @property
    def is_long_running(self) -> bool:
        """A heuristic, and one that has already been wrong once.

        The first version scored `inference-engine-codegolf` as light because
        none of its keywords named a thing being built, and it is in fact a
        multi-GPU CUDA task. Treat this as a prompt to check, never as a
        budget guarantee — read :attr:`heavy_signals` and decide.

        Note that every task in the published ``terminal-bench-challenges``
        set is long-running by that set's own definition: it exists precisely
        for "token-intensive, long-running, single task benchmarks". A task
        from that repository should be assumed heavy whatever this returns.
        """
        return bool(self.heavy_signals)


def load_task(root: Path) -> HarborTask:
    """Read one task directory.

    The canary comment in ``instruction.md`` is deliberately **kept**. It is
    the benchmark's own contamination probe — if a model reproduces the GUID,
    it memorised the task — and stripping it to tidy the prompt would disable
    an instrument that belongs to someone else.
    """
    root = Path(root)
    config_path = root / "task.toml"
    if not config_path.is_file():
        raise FileNotFoundError(f"{root} has no task.toml — not a Harbor task")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    task = config.get("task", {})

    instruction_path = root / "instruction.md"
    instruction = (
        instruction_path.read_text(encoding="utf-8") if instruction_path.is_file() else ""
    )
    authors = tuple(
        str(a.get("name", a)) if isinstance(a, dict) else str(a)
        for a in task.get("authors", ())
    )
    return HarborTask(
        name=str(task.get("name") or root.name),
        description=str(task.get("description", "")),
        instruction=instruction,
        root=root,
        keywords=tuple(str(k) for k in task.get("keywords", ())),
        authors=authors,
        schema_version=str(config.get("schema_version", "")),
    )


def discover(root: Path) -> list[HarborTask]:
    """Every Harbor task under a checkout, in a stable order."""
    found: list[HarborTask] = []
    for config in sorted(Path(root).glob("*/task.toml")):
        try:
            found.append(load_task(config.parent))
        except Exception:
            continue
    return found


def task_text(task: HarborTask) -> str:
    """What the agent is given. The instruction, verbatim, and nothing else.

    No hints about the tests, and no restating of the description: the
    instruction is the benchmark's own prompt and rewriting it would forfeit
    comparability with anyone else's result on the same task.
    """
    return task.instruction.strip()


def graded_paths(task: HarborTask) -> set[str]:
    """Files the verification owns, which the agent must not be scored on editing.

    Everything under ``tests/`` belongs to the benchmark. An agent that edits
    its own grader has not solved the task, and a Monitor scoped over these
    files would be grading the thing that scores it.
    """
    tests = task.root / "tests"
    if not tests.is_dir():
        return set()
    return {
        str(p.relative_to(task.root)) for p in tests.rglob("*") if p.is_file()
    }


def build_image(task: HarborTask, *, tag: str = "", platform: str = "linux/amd64") -> str:
    """Build the task's environment image. Returns the tag.

    Harbor ships a Dockerfile rather than a published image, so unlike
    SWE-bench there is nothing to pin by digest — the environment is whatever
    the build produces today. That is a real reproducibility gap and is
    recorded rather than papered over: pin the built image yourself and reuse
    it across arms, or the arms are not running in the same environment.
    """
    if not task.runnable:
        raise FileNotFoundError(f"{task.slug}: needs environment/Dockerfile and tests/test.sh")
    image = tag or f"taste-tb-{task.slug}:local"
    subprocess.run(
        ["docker", "build", "--platform", platform, "-t", image, "."],
        cwd=task.dockerfile.parent, check=True, capture_output=True,
    )
    return image


def materialize(task: HarborTask, dest: Path) -> Path:
    """A workspace holding the task's environment, with the tests withheld.

    ``tests/`` is deliberately not copied. The agent is asked to satisfy a
    grader it cannot read, which is the same discipline SWE-bench enforces by
    running the gold test patch out of band.
    """
    workspace = Path(dest)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    environment = task.root / "environment"
    if environment.is_dir():
        shutil.copytree(environment, workspace, dirs_exist_ok=True)
    (workspace / "INSTRUCTION.md").write_text(task_text(task), encoding="utf-8")
    (workspace / ".gitignore").write_text(DEPENDENCY_IGNORES, encoding="utf-8")

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.name", "taste")
    git("config", "user.email", "taste@localhost")
    git("config", "gc.auto", "0")
    git("config", "gc.pruneExpire", "never")
    git("add", "-A")
    git("commit", "-qm", f"{task.slug} @ environment", "--allow-empty")
    return workspace


@dataclass
class TerminalBenchProbe:
    """The task's own verification, as a suite our replay can execute."""

    task: HarborTask
    workdir: str = "/app"
    timeout: int = 3600
    members: tuple[str, ...] = field(default_factory=tuple)

    def suite(self) -> SuiteProbe:
        """One command, one verdict.

        Harbor's ``test.sh`` reports through its exit code and has no common
        per-test grammar across tasks, so this is an exit-code probe with a
        single member. That is coarser than the SWE-bench path — a suite-level
        verdict cannot say *which* test broke, so coverage attribution is not
        available here and silence can only be reported in its unattributed
        form.
        """
        members = self.members or (f"tb::{self.task.slug}",)
        return SuiteProbe(
            name=f"tb::{self.task.slug}",
            command=f"cd {self.workdir} && bash tests/test.sh",
            members=members,
            timeout=self.timeout,
        )
