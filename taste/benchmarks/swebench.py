"""SWE-bench as a task source, with the benchmark's own oracle left intact.

The property that makes this the right substrate is narrow and decisive: the
official grader is a **pure function of (instance_id, model_patch)**, executed
out of band in a container the agent never touches. Nothing about the run is
an input to it.

That has a consequence worth stating precisely, because it is the paper's
central defence. The grader can be applied to *any* intermediate patch — every
observation the harness records — **without modifying the benchmark**. Our
protocol is a strict *superset of observations* over an unmodified benchmark,
never a modification of one. The score at the final observation remains
bit-for-bit the published-protocol number, so it is comparable to every result
on the leaderboard, and there is no author-defined split for a reviewer to
interrogate.

**PASS_TO_PASS is not a proxy for the phenomenon; it is the phenomenon.**
Tests that passed at ``base_commit`` and must still pass afterwards, computed
by the benchmark's authors by running the gold patch repeatedly and screening
flakes, on a dataset that predates this work. A PASS_TO_PASS test going from
pass to fail *is* a silent regression, defined by people who never heard of
this paper.

**The one split that remains is inside our own harness**, where design choices
belong and a reviewer expects them: the Monitor's scope is every test file in
the repository *minus* the files the instance's evaluation selects. Overlap
with the graded set is therefore zero at file level by construction, so the
Monitor cannot be running the thing that scores it. Our absolute resolve rates
sit below published numbers as a result — a disclosed, arm-symmetric cost of
not letting the harness grade its own work.

Honest bound, stated before a reviewer states it: PASS_TO_PASS covers the one
or two test files the gold patch touched, so it is an *edge-biased lower
bound* on regression, not a repository-wide detector.
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from taste.benchmarks.swebench_log import PARSER_REGISTRY
from taste.benchmarks.swebench_log import parse as parse_log
from taste.benchmarks.swebench_specs import spec_for, test_directives
from taste.replay import SuiteProbe

# The dataset fields this adapter depends on. Named here so a schema change
# surfaces as a clear error rather than a silent KeyError mid-sweep.
REQUIRED_FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "test_patch",
    "version",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
)


@dataclass(frozen=True)
class SWEInstance:
    """One benchmark instance, as published."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str
    version: str
    """Which release line this instance sits on. Required, not defaulted: the
    test *runner* is a function of ``(repo, version)`` — django 3.0 and django
    4.1 do not take the same arguments — so a snapshot without it cannot be
    graded, and should fail at load rather than silently run the wrong
    command halfway through a sweep."""
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    environment_setup_commit: str = ""
    image: str = ""
    """Pinned by digest. A ``:latest`` tag on these images has already moved
    once and will move again; a run graded against a different image is not
    comparable to one graded against ours."""

    @property
    def repo_short(self) -> str:
        """The clustering unit. Django alone is 231/500 of Verified, which is
        why the analysis clusters on repository and a leave-one-repo-out
        sensitivity check is mandatory."""
        return self.repo.split("/")[-1]

    @property
    def has_oracle(self) -> bool:
        """Whether this instance can host the construct at all.

        An instance with no PASS_TO_PASS tests provides no previously-passing
        state, so "something that used to work broke" is not expressible on it
        — it must be excluded *before* any arm runs, not discovered later.
        """
        return bool(self.pass_to_pass)


def load_dataset(path: Path) -> list[SWEInstance]:
    """Read a pinned local JSONL snapshot — never a live download.

    Offline and digest-pinned on purpose: a sweep that silently re-resolves
    its dataset or its images midway is not reproducible, and image tags for
    this benchmark are known to move.
    """
    instances: list[SWEInstance] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        missing = [f for f in REQUIRED_FIELDS if f not in raw]
        if missing:
            raise ValueError(f"{path}:{line_number} is missing {missing}")
        instances.append(
            SWEInstance(
                instance_id=raw["instance_id"],
                repo=raw["repo"],
                base_commit=raw["base_commit"],
                problem_statement=raw["problem_statement"],
                test_patch=raw["test_patch"],
                version=str(raw["version"]),
                fail_to_pass=tuple(_as_list(raw["FAIL_TO_PASS"])),
                pass_to_pass=tuple(_as_list(raw["PASS_TO_PASS"])),
                environment_setup_commit=raw.get("environment_setup_commit", ""),
                image=raw.get("image", ""),
            )
        )
    return instances


def _as_list(value: Any) -> list[str]:
    """The test lists ship as JSON strings in some snapshots, lists in others."""
    if isinstance(value, str):
        try:
            return list(json.loads(value))
        except ValueError:
            return [value] if value else []
    return list(value or [])


# ------------------------------------------------------------------ selection


@dataclass
class Exclusion:
    instance_id: str
    reason: str


@dataclass
class Selection:
    """A frame, its exclusions, and the seed that produced it.

    Published in full. Every exclusion is a deterministic function of dataset
    fields or of gold-patch behaviour — never of an arm's outcome, which is
    not observed until after selection is frozen.
    """

    instances: list[SWEInstance]
    excluded: list[Exclusion] = field(default_factory=list)
    seed: int = 0
    strata: dict[str, int] = field(default_factory=dict)

    def manifest(self) -> dict:
        return {
            "benchmark": "swebench",
            "seed": self.seed,
            "selected": [i.instance_id for i in self.instances],
            "strata": self.strata,
            "excluded": [{"instance_id": e.instance_id, "reason": e.reason} for e in self.excluded],
            "n_selected": len(self.instances),
            "n_excluded": len(self.excluded),
        }


def eligible(instances: list[SWEInstance], monitor_scopes: dict[str, int] | None = None) -> Selection:
    """Apply the pre-registered exclusions, each with its reason recorded.

    E1 no PASS_TO_PASS  — the instance cannot host the construct.
    E3 empty Monitor scope — no test files outside the graded set, so the
       harness would have no verification signal that is not the oracle.

    Neither is a function of contamination outcomes; both limit external
    validity and are reported as such.
    """
    kept: list[SWEInstance] = []
    dropped: list[Exclusion] = []
    for instance in instances:
        if not instance.has_oracle:
            dropped.append(Exclusion(instance.instance_id, "E1: PASS_TO_PASS is empty"))
            continue
        if monitor_scopes is not None and monitor_scopes.get(instance.instance_id, 1) == 0:
            dropped.append(Exclusion(instance.instance_id, "E3: Monitor scope is empty"))
            continue
        kept.append(instance)
    return Selection(instances=kept, excluded=dropped)


def stratified_sample(
    selection: Selection, *, n: int, seed: int, holdout: set[str] | None = None
) -> Selection:
    """A seeded sample stratified by repository, proportional to the frame.

    Proportional rather than balanced: Django dominates Verified, and forcing
    balance would produce a sample that is not the benchmark. Keeping it
    dominant means a leave-one-repo-out sensitivity analysis is mandatory
    rather than optional, which is the honest trade.
    """
    import random

    pool = [i for i in selection.instances if not holdout or i.instance_id not in holdout]
    by_repo: dict[str, list[SWEInstance]] = {}
    for instance in pool:
        by_repo.setdefault(instance.repo_short, []).append(instance)

    total = len(pool)
    rng = random.Random(seed)
    chosen: list[SWEInstance] = []
    strata: dict[str, int] = {}

    for repo in sorted(by_repo):
        group = sorted(by_repo[repo], key=lambda i: i.instance_id)
        rng.shuffle(group)
        take = max(1, round(n * len(group) / total)) if total else 0
        picked = group[: min(take, len(group))]
        chosen.extend(picked)
        strata[repo] = len(picked)

    # Proportional rounding overshoots or undershoots; trim or top up
    # deterministically so N is exactly as pre-registered.
    chosen.sort(key=lambda i: i.instance_id)
    if len(chosen) > n:
        rng.shuffle(chosen)
        chosen = sorted(chosen[:n], key=lambda i: i.instance_id)
        strata = {}
        for instance in chosen:
            strata[instance.repo_short] = strata.get(instance.repo_short, 0) + 1

    return Selection(instances=chosen, excluded=selection.excluded, seed=seed, strata=strata)


# ------------------------------------------------------------------ scoping


def monitor_scope(repo_root: Path, graded_files: set[str]) -> list[str]:
    """Test files the Monitor may run: everything except the graded ones.

    A published deterministic function, not a judgement call. File-level
    overlap with the oracle is zero by construction, so the harness cannot be
    verified by the thing that scores it. Shared code paths remain — that is
    real, and it is handled by clustering rather than pretended away.
    """
    found: list[str] = []
    for path in sorted(repo_root.rglob("test_*.py")) + sorted(repo_root.rglob("*_test.py")):
        if any(part in {".git", "build", "dist", ".tox"} for part in path.parts):
            continue
        relative = str(path.relative_to(repo_root))
        if relative in graded_files:
            continue
        found.append(relative)
    return found


def graded_test_files(instance: SWEInstance) -> set[str]:
    """The files the gold test patch touches — the graded set."""
    files: set[str] = set()
    for line in instance.test_patch.splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            path = line[6:].strip()
            if path and path != "/dev/null":
                files.add(path)
    return files


# ------------------------------------------------------------------ probes


START_MARKER = "TASTE_START_TEST_OUTPUT"
END_MARKER = "TASTE_END_TEST_OUTPUT"


def resolve_parser(name: str) -> str:
    """Reconcile the two vendored tables' naming.

    The spec table records a parser as upstream's registry *value* names it
    (``django``); the parser module keys its registry by upstream's *function*
    names (``parse_log_django``). Both are faithful to their own source, and
    the mismatch sits exactly on the seam between them — so it is resolved
    once, here, loudly, rather than by either table quietly renaming what it
    vendored.
    """
    if name in PARSER_REGISTRY:
        return name
    prefixed = f"parse_log_{name}"
    if prefixed in PARSER_REGISTRY:
        return prefixed
    raise KeyError(
        f"no log parser for {name!r}; the spec table and the parser registry "
        f"disagree. Known: {sorted(PARSER_REGISTRY)}"
    )


def build_eval_script(instance: SWEInstance, *, workdir: str = "/testbed") -> str:
    """The command that evaluates PASS_TO_PASS inside the instance's image.

    This mirrors the official harness rather than approximating it, because
    every approximation available here fails silently. The version this
    replaced ran ``python -m pytest <node_ids>``, which is wrong for 70% of
    SWE-bench Verified in two independent ways: django (46% of the set) runs
    its suite through ``./tests/runtests.py``, not pytest, and its
    PASS_TO_PASS identifiers are unittest reprs like
    ``test_defaults (str.tests.SimpleTests)`` — which pytest reads as a
    *file path* and exits 4 on. The probe therefore failed at every
    observation including the baseline, so the whole stratum was classified
    "never passed" and contributed no episodes at all, which a negative
    control scores as a clean run.

    Four things this does that the old command did not:

    *Activates the environment.* The image puts conda ``base`` on PATH and
    activates ``testbed`` only from ``.bashrc``, which a non-interactive shell
    never reads. Without this the suite runs against an interpreter where the
    repository under test is not installed.

    *Restores the graded tests before applying the gold patch, and again
    afterwards.* The agent may have edited the test files, and the old
    command's ``|| true`` swallowed the resulting conflict and then graded
    whatever the agent had written — in the exact scenario the probe exists
    to detect. There is no ``|| true`` here; a failed apply is an
    infrastructure error and the caller records a hole.

    *Passes file directives, not test ids.* The official harness runs
    test *files* and filters afterwards, so the command stays a fixed size
    whatever the instance holds — one instance names 1,432 PASS_TO_PASS
    tests and another 2,476, which as argv approaches ARG_MAX.

    *Brackets the output.* Everything the setup prints is outside the
    markers, so a line in a traceback cannot be mistaken for a test result.
    """
    spec = spec_for(instance.repo, instance.version)
    directives = " ".join(test_directives(instance.repo, instance.test_patch))
    graded = " ".join(sorted(graded_test_files(instance))) or "."
    setup = "\n".join(spec.eval_commands)
    return "\n".join(
        part
        for part in (
            "source /opt/miniconda3/bin/activate && conda activate testbed",
            f"cd {workdir}",
            setup,
            f"git config --global --add safe.directory {workdir}",
            f"git checkout {instance.base_commit} -- {graded}",
            f"git apply -v - <<'TASTE_TEST_PATCH'\n{instance.test_patch}\nTASTE_TEST_PATCH",
            f": '{START_MARKER}'",
            f"{spec.test_cmd} {directives}".strip(),
            f": '{END_MARKER}'",
            f"git checkout {instance.base_commit} -- {graded}",
        )
        if part
    )


def parse_eval_output(instance: SWEInstance, log: str) -> dict[str, str]:
    """Per-test statuses from the bracketed slice of a run's output.

    Only the slice between the markers is parsed. Setup noise — a conda
    banner, a ``git apply`` diagnostic, a deprecation warning — can otherwise
    match a runner's grammar and be recorded as a test result.
    """
    spec = spec_for(instance.repo, instance.version)
    start = log.find(START_MARKER)
    end = log.find(END_MARKER, start + 1) if start >= 0 else -1
    if start >= 0:
        # Skip past the marker's own line.
        newline = log.find("\n", start)
        body = log[newline + 1 : end] if end > newline >= 0 else log[start:]
    else:
        # No markers: the script died before reaching them. Parsing the whole
        # log here would read setup output as results.
        return {}
    return dict(parse_log(resolve_parser(spec.log_parser), body))


def pass_to_pass_suite(instance: SWEInstance, *, timeout: int = 1800) -> SuiteProbe:
    """The instance's silent-regression oracle, as one executable suite.

    ``members`` is the published PASS_TO_PASS list. The command runs whole
    test *files* and the parser reports everything they contained, so the run
    is file-scoped while the grading stays test-scoped — a member the log
    never mentions is a hole, not a failure.
    """
    return SuiteProbe(
        name=f"p2p::{instance.instance_id}",
        command=build_eval_script(instance),
        members=instance.pass_to_pass,
        parse=functools.partial(parse_eval_output, instance),
        timeout=timeout,
    )


# ------------------------------------------------------------------ grading


@dataclass
class GradeReport:
    """One instance's official verdict at one patch."""

    instance_id: str
    resolved: bool
    fail_to_pass_passed: int
    fail_to_pass_total: int
    pass_to_pass_passed: int
    pass_to_pass_total: int
    per_test: dict[str, str] = field(default_factory=dict)
    """Per-test status. This is what makes a timeline possible rather than an
    end-state check: an aggregate would say only that something broke."""

    @property
    def regressed_tests(self) -> tuple[str, ...]:
        """PASS_TO_PASS tests not passing — the contamination signal."""
        return tuple(
            sorted(t for t, status in self.per_test.items() if status not in ("PASSED", "SKIPPED"))
        )


def parse_report(raw: dict, instance: SWEInstance) -> GradeReport:
    """Read the official report's per-test status into our own shape."""
    entry = raw.get(instance.instance_id, raw)
    status = entry.get("tests_status", {})
    f2p = status.get("FAIL_TO_PASS", {})
    p2p = status.get("PASS_TO_PASS", {})

    per_test = {t: "PASSED" for t in p2p.get("success", [])}
    per_test.update({t: "FAILED" for t in p2p.get("failure", [])})

    return GradeReport(
        instance_id=instance.instance_id,
        resolved=bool(entry.get("resolved", False)),
        fail_to_pass_passed=len(f2p.get("success", [])),
        fail_to_pass_total=len(f2p.get("success", [])) + len(f2p.get("failure", [])),
        pass_to_pass_passed=len(p2p.get("success", [])),
        pass_to_pass_total=len(p2p.get("success", [])) + len(p2p.get("failure", [])),
        per_test=per_test,
    )


def patch_for(repo_root: Path, base_commit: str) -> str:
    """The prediction: a diff from ``base_commit`` to the current tree.

    Test files are excluded. The grader restores them anyway, but emitting
    them would make our prediction differ from what is actually graded, and a
    prediction that does not match the thing scored is a debugging trap.
    """
    result = subprocess.run(
        ["git", "diff", base_commit, "--", ".", ":(exclude)*test_*.py", ":(exclude)*tests/*"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return result.stdout


def task_text(instance: SWEInstance) -> str:
    """What the agent is told. The problem statement, and nothing else.

    No test names, no hints about the graded set — that is the benchmark's
    protocol and departing from it would forfeit comparability.
    """
    return (
        f"Resolve the following issue in the {instance.repo} repository.\n\n"
        f"{instance.problem_statement}\n\n"
        "Make the smallest change that fixes the issue without breaking "
        "existing behaviour. Do not modify existing tests."
    )


# ------------------------------------------------------------------ workspace


def materialize(instance: SWEInstance, dest: Path, *, source: Path | None = None) -> Path:
    """An isolated working tree at ``base_commit`` for one cell.

    Per-cell, never shared. Two trials sharing a tree makes one run's edits
    the next run's starting state, which silently destroys the pairing the
    whole analysis rests on.

    The repository's own history is dropped and a single commit created in its
    place. That is not tidiness: read-only git is deliberately available to
    workers, so an inherited history would let an agent read the upstream fix
    for its own instance straight out of the object database.

    ``gc.auto`` is disabled because the shadow timeline lives in this
    repository, and losing it does not look like a failure — it looks like a
    clean run.
    """
    workspace = Path(dest)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    if source is not None:
        shutil.copytree(
            source, workspace, symlinks=True, dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".git"),
        )

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.name", "taste")
    git("config", "user.email", "taste@localhost")
    git("config", "gc.auto", "0")
    git("config", "gc.pruneExpire", "never")
    git("add", "-A")
    git("commit", "-qm", f"{instance.instance_id} @ {instance.base_commit[:12]}", "--allow-empty")
    return workspace
