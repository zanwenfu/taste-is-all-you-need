"""Commit0 as a task source, with a held-out oracle bolted on.

Commit0 hands an agent a real library whose public function bodies have been
replaced with ``pass``, plus the library's own test suite, and asks for an
implementation. It is the right shape for this project: hours-long, many
interdependent edits, and the agent is *expected* to run the tests as
feedback — which is exactly the loop the harness automates.

**The problem it does not solve, and this module does.** Commit0's tests are
visible by design, and its score is those same tests. An agent that can read
the thing scoring it is being measured on a different task than the one
intended, and a harness whose Monitor runs the scoring suite has made
"monitor passed" and "task solved" the same event by construction — a
circularity that makes any success claim unfalsifiable.

So the suite is split. A **visible** portion stays in the workspace: the
agent reads it, the Monitor runs it, it is the feedback signal the benchmark
intends. A **held-out** portion is physically removed and kept outside the
workspace, restored only by an out-of-band scorer. The agent cannot read it,
cannot satisfy it directly, and cannot know which behaviours it checks.

The split is deterministic given a seed, recorded in the task, and re-drawable
— because a result that depends on which tests happened to be hidden is a
result about the split, and the only way to know is to re-draw it and look.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from taste.replay import Probe

# The lite split: 16 libraries with fewer functions to implement.
LITE = (
    "babel", "cachetools", "chardet", "cookiecutter", "deprecated", "imapclient",
    "jinja", "marshmallow", "minitorch", "parsel", "portalocker", "pyjwt",
    "simpy", "tinydb", "voluptuous", "wcwidth",
)

# Their own pytest config injects coverage plugins we neither have nor want;
# addopts="" ignores it so a run measures the library, not the toolchain.
#
# `sys.executable`, never a bare `python`: a clean Ubuntu host ships `python3`
# and no `python` at all, so this command exited 127 and `fractional_score`
# returned 0.0 -- a benchmark score of zero manufactured by a missing
# interpreter and indistinguishable from an agent that implemented nothing.
# Same defect as gate 0's probe, in the path that produces reported numbers.
PYTEST_BASE = (
    f'{shlex.quote(sys.executable)} -m pytest -q -p no:cacheprovider -o addopts=""'
)

_TEST_DEF = re.compile(r"^\s*def (test_\w+)", re.MULTILINE)


@dataclass(frozen=True)
class Commit0Task:
    """One library, ready to hand to the kernel."""

    name: str
    workspace: Path
    visible_command: str
    """What the Monitor runs — the agent's own feedback signal."""
    probes: tuple[Probe, ...]
    """Held-out checks, restored only by the scorer."""
    visible_tests: tuple[str, ...]
    holdout_tests: tuple[str, ...]
    visible_count: int
    holdout_count: int
    holdout_dir: Path
    seed: int
    task_text: str = ""
    baseline_failures: int | None = None

    @property
    def split_id(self) -> str:
        """Names this exact split, so two runs on 'the same task' that used
        different holdouts cannot be silently pooled."""
        payload = json.dumps(
            {"library": self.name, "seed": self.seed, "holdout": sorted(self.holdout_tests)},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def manifest(self) -> dict:
        return {
            "benchmark": "commit0",
            "library": self.name,
            "split_id": self.split_id,
            "seed": self.seed,
            "visible_tests": list(self.visible_tests),
            "holdout_tests": list(self.holdout_tests),
            "visible_count": self.visible_count,
            "holdout_count": self.holdout_count,
            "baseline_failures": self.baseline_failures,
        }


@dataclass
class TestFile:
    path: str
    count: int
    names: tuple[str, ...] = field(default_factory=tuple)


def discover_tests(repo: Path) -> list[TestFile]:
    """Every test file in a library, with how many tests it holds."""
    found: list[TestFile] = []
    for path in sorted(repo.rglob("test_*.py")) + sorted(repo.rglob("*_test.py")):
        if any(part in {".git", "build", "dist"} for part in path.parts):
            continue
        try:
            names = tuple(_TEST_DEF.findall(path.read_text(errors="ignore")))
        except OSError:
            continue
        if names:
            found.append(TestFile(str(path.relative_to(repo)), len(names), names))
    return found


def choose_holdout(files: list[TestFile], *, fraction: float, seed: int) -> list[TestFile]:
    """Pick whole files totalling about ``fraction`` of the tests.

    Whole files, not individual tests: rewriting a test module to remove some
    of its functions risks breaking shared fixtures and imports, and a probe
    that fails for a reason unrelated to the agent's work is worse than a
    coarser split. Deterministic given the seed, so a split can be reproduced
    and re-drawn.
    """
    if not files:
        return []
    total = sum(f.count for f in files)
    target = total * fraction

    # Seeded shuffle for reproducible randomness across re-draws, then greedy
    # under a ceiling. Admitting any file merely because the running total is
    # still below target overshoots wildly when the next file is large — on a
    # 4-file suite that turned a 30% target into 67% held out.
    order = sorted(files, key=lambda f: hashlib.sha256(f"{seed}:{f.path}".encode()).hexdigest())
    chosen: list[TestFile] = []
    running = 0
    for candidate in order:
        if running + candidate.count > target:
            continue
        # Never hold out everything: the agent needs a feedback signal and the
        # Monitor needs something to run.
        if running + candidate.count >= total:
            continue
        chosen.append(candidate)
        running += candidate.count

    if not chosen:
        # Coarse suites (a handful of large files) can admit nothing under the
        # ceiling. Take the smallest file rather than leaving no oracle at all.
        smallest = min(files, key=lambda f: f.count)
        if smallest.count < total:
            chosen = [smallest]
    return chosen


def materialize(
    library: str,
    *,
    source_root: Path,
    dest: Path,
    holdout_fraction: float = 0.30,
    seed: int = 0,
) -> Commit0Task:
    """Copy a library into an isolated workspace and split its suite.

    ``source_root`` is the directory ``commit0 setup`` populated (it contains
    ``repos/<library>``). The copy is per-trial: two trials must never share a
    working tree, or one run's edits become another's starting state.
    """
    source = Path(source_root) / "repos" / library
    if not source.is_dir():
        raise FileNotFoundError(
            f"{source} not found — run `commit0 setup lite` in {source_root} first"
        )

    workspace = Path(dest) / library
    if workspace.exists():
        shutil.rmtree(workspace)
    # Copy without .git. The library's own history is not merely unnecessary,
    # it is a leak: deleting a held-out test from the working tree leaves it
    # readable at `git show HEAD:<path>`, and read-only git is deliberately
    # allowed to workers. A fresh repo is created below, after the split, so
    # the held-out tests exist in no reachable object.
    shutil.copytree(source, workspace, symlinks=True, ignore=shutil.ignore_patterns(".git"))

    files = discover_tests(workspace)
    holdout = choose_holdout(files, fraction=holdout_fraction, seed=seed)
    holdout_paths = {f.path for f in holdout}

    holdout_dir = Path(dest) / f"{library}__holdout"
    if holdout_dir.exists():
        shutil.rmtree(holdout_dir)
    holdout_dir.mkdir(parents=True)

    # Physically remove the held-out files. Hiding them any other way leaves
    # them readable, and a probe the agent can read is not held out.
    for f in holdout:
        src = workspace / f.path
        target = holdout_dir / f.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(target))

    # Only now, with the held-out files already gone, does this become a git
    # repository. Its first commit is the agent's starting state and its
    # entire history, so there is no object anywhere containing an oracle.
    _init_repo(workspace)

    visible = [f for f in files if f.path not in holdout_paths]
    visible_command = _pytest_for([f.path for f in visible])

    probes = tuple(
        Probe(
            name=f"holdout::{f.path}",
            # Restore the file into the checkout, then run only it. The
            # worktree is thrown away afterwards, so nothing leaks back.
            command=(
                f"mkdir -p $(dirname {f.path}) && "
                f"cp {holdout_dir / f.path} {f.path} && "
                f"{PYTEST_BASE} {f.path}"
            ),
            timeout=300,
        )
        for f in holdout
    )

    return Commit0Task(
        name=library,
        workspace=workspace,
        visible_command=visible_command,
        probes=probes,
        visible_tests=tuple(f.path for f in visible),
        holdout_tests=tuple(f.path for f in holdout),
        visible_count=sum(f.count for f in visible),
        holdout_count=sum(f.count for f in holdout),
        holdout_dir=holdout_dir,
        seed=seed,
        task_text=_task_text(library, visible),
    )


def _init_repo(workspace: Path) -> None:
    """A fresh repository whose only commit is the stubbed starting state."""
    import subprocess

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=taste", "-c", "user.email=taste@local", *args],
            cwd=workspace,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "commit0: stubbed starting state")


def _pytest_for(paths: list[str]) -> str:
    if not paths:
        return f"{PYTEST_BASE} --collect-only"
    return f"{PYTEST_BASE} {' '.join(sorted(paths))}"


def _task_text(library: str, visible: list[TestFile]) -> str:
    suite = ", ".join(sorted(f.path for f in visible)[:6])
    return (
        f"Implement the {library} library. Its public functions have been "
        f"stubbed out with `pass` — restore them so the library behaves as its "
        f"documentation and tests specify. The test suite ({suite}) is your "
        f"feedback: run it, read the failures, and work through them. Do not "
        f"modify the tests; they define the contract."
    )


def fractional_score(memory, sha: str, task: Commit0Task) -> float:
    """Fraction of held-out probes passing at ``sha``.

    Fractional rather than binary because a from-scratch library is almost
    never fully solved: the published state of the art on this benchmark is
    well under half. A binary endpoint would compress every arm onto the same
    value and measure nothing.
    """
    if not task.probes:
        return 0.0
    from taste.replay import Replayer

    replayer = Replayer(memory, list(task.probes))
    passed = sum(1 for p in task.probes if replayer.verdict_at(sha, p) == "pass")
    return passed / len(task.probes)
