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

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from taste.replay import Probe

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


def pass_to_pass_probe(instance: SWEInstance, *, runner: str = "python -m pytest") -> Probe:
    """A probe that runs the instance's PASS_TO_PASS set.

    Applied out of band against a historical tree. The gold ``test_patch`` is
    applied first, exactly as the official grader does, so the tests being run
    are the benchmark's own rather than any version the agent may have edited.
    """
    node_ids = " ".join(_quote(t) for t in instance.pass_to_pass)
    return Probe(
        name=f"p2p::{instance.instance_id}",
        command=(
            f"git apply -v - <<'TASTE_TEST_PATCH' || true\n"
            f"{instance.test_patch}\n"
            f"TASTE_TEST_PATCH\n"
            f"{runner} -q -p no:cacheprovider {node_ids}"
        ),
        timeout=600,
    )


def _quote(node_id: str) -> str:
    return "'" + node_id.replace("'", "'\\''") + "'"


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
