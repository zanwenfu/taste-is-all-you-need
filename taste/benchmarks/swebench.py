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
import re
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
    def published_image(self) -> str:
        """SWE-bench's published image tag for this instance.

        The dataset carries no image column, so every driver has to derive the
        tag, and each one that derives it separately is a chance to derive it
        differently. That failure would not raise: it would quietly evaluate
        the instance in some other environment and report the difference as a
        change in the agent's behaviour. One definition, used by the sweep
        driver and the re-scorer alike.

        The ``_1776_`` separator is upstream's, not ours -- ``__`` is not
        legal in a Docker tag.
        """
        return f"swebench/sweb.eval.x86_64.{self.instance_id.replace('__', '_1776_')}:latest"

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


_UNITTEST_ID = re.compile(r"^\S+ \(([\w.]+)\)$")


def member_test_files(instance: SWEInstance) -> set[str]:
    """The files that hold the instance's previously-passing tests.

    Derived from the ids themselves and unioned with the gold test patch's
    files: pytest node ids carry their path; django's unittest labels carry a
    dotted module that maps back under ``tests/``; sympy's bare function
    names carry nothing and rely on the patch's files. A gate that ran only
    the patch's files would miss every passing test the patch did not touch.
    """
    files = set(graded_test_files(instance))
    for member in instance.pass_to_pass:
        if "::" in member:
            files.add(member.split("::", 1)[0])
            continue
        m = _UNITTEST_ID.match(member.strip())
        if m and instance.repo == "django/django":
            module = m.group(1).rsplit(".", 1)[0]  # drop the TestCase class
            files.add("tests/" + module.replace(".", "/") + ".py")
    return files


def directives_for(repo: str, test_files) -> list[str]:
    """The runner arguments for a set of test files, with the same per-repo
    transform the official harness applies (django wants dotted module paths
    relative to tests/). Mirrors ``test_directives`` but takes paths rather
    than a patch, so a caller can run any subset of the repository's tests."""
    directives = sorted(set(test_files))
    if repo == "django/django":
        out = []
        for d in directives:
            d = d[: -len(".py")] if d.endswith(".py") else d
            d = d[len("tests/"):] if d.startswith("tests/") else d
            out.append(d.replace("/", "."))
        directives = out
    return directives


def plain_suite_command(instance: SWEInstance, test_files) -> str:
    """Run the given test files as they exist in the CURRENT tree.

    Unlike :func:`build_eval_script` this restores nothing and applies no
    test patch: it is what a harness may legitimately run inside the agent's
    own environment — the repository's visible tests, unmodified — without
    leaking the benchmark's hidden fail-to-pass tests. Activation and the
    working directory are supplied by the routed executor.
    """
    spec = spec_for(instance.repo, instance.version)
    directives = " ".join(directives_for(instance.repo, test_files))
    return "\n".join((
        "exec 2>&1",
        f"echo '{START_MARKER}'",
        f"{spec.test_cmd} {directives}".strip(),
        f"echo '{END_MARKER}'",
    ))


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
            # Everything on one stream, in the order a terminal would show it.
            # django's runner reports results on STDERR while the markers below
            # go to STDOUT, so without this the results land *after* the end
            # marker once the streams are concatenated and the slice misses
            # them entirely -- every test reads as a hole.
            "exec 2>&1",
            "source /opt/miniconda3/bin/activate && conda activate testbed",
            f"cd {workdir}",
            setup,
            f"git config --global --add safe.directory {workdir}",
            f"git checkout {instance.base_commit} -- {graded}",
            f"git apply -v - <<'TASTE_TEST_PATCH'\n{instance.test_patch}\nTASTE_TEST_PATCH",
            # `echo`, not `:`. The shell's null command accepts an argument and
            # prints nothing, so the marker never reached the log, the slice
            # found no start, and the parser returned no results at all --
            # which classified every graded test as an infrastructure hole on
            # a run where pytest had in fact reported them perfectly.
            f"echo '{START_MARKER}'",
            f"{spec.test_cmd} {directives}".strip(),
            f"echo '{END_MARKER}'",
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
        """PASS_TO_PASS tests not passing — the contamination signal.

        Passing means the official grader's ``test_passed``: PASSED or XFAIL.
        The previous version counted XFAIL as a regression and SKIPPED as a
        pass — both inverted against upstream. A SKIPPED graded test is NOT
        passing there: a patch that skips its way around the oracle would
        otherwise grade as resolved.
        """
        return tuple(
            sorted(t for t, status in self.per_test.items() if status not in PASSING_STATUSES)
        )


#: The official grader's ``test_passed``: a test counts as passing iff its
#: status is PASSED or XFAIL. SKIPPED is not passing — see
#: :meth:`GradeReport.regressed_tests`.
PASSING_STATUSES = ("PASSED", "XFAIL")


def grade_in_sandbox(
    sandbox, instance: SWEInstance, model_patch: str, *, timeout: int = 1800
) -> GradeReport | None:
    """The official resolve verdict for one final patch, in the pinned image.

    Returns ``None`` when the verdict could not be produced — a tree that
    would not reset, a log with no markers. That is a missing measurement,
    and the caller records it as one; folding it into ``resolved=False``
    would let infrastructure manufacture failure, the mirror image of the
    bug-B invariant.

    A model patch that does not APPLY, by contrast, is a real verdict:
    ``resolved=False`` with the reason recorded, exactly as the official
    harness scores it. The empty patch runs the evaluation at the base tree
    -- FAIL_TO_PASS fails by definition and the verdict is honest.
    """
    from taste.routing import prepare_container_tree

    workdir = sandbox.workdir
    target = prepare_container_tree(sandbox, workdir=workdir, hide_upstream=False)
    reset = sandbox.exec(
        f"cd {workdir} && git checkout -q {target} -- . && git clean -qfd", timeout=300
    )
    if reset.exit_code != 0:
        return None

    apply_error = ""
    if model_patch.strip():
        sandbox.put_text("/tmp/taste_pred.diff", model_patch + "\n")
        applied = sandbox.exec(
            f"cd {workdir} && git apply -v /tmp/taste_pred.diff", timeout=300
        )
        if applied.exit_code != 0:
            apply_error = (applied.stderr or applied.stdout)[-400:]

    per_test: dict[str, str] = {}
    suite_killed = False
    if not apply_error:
        result = sandbox.exec(build_eval_script(instance, workdir=workdir), timeout=timeout)
        per_test = parse_eval_output(instance, result.stdout)
        if not per_test:
            # No result between the markers. Two very different causes share
            # this shape, and the contrast's primary endpoint sits exactly on
            # the difference: the environment died (infrastructure — a hole,
            # never a verdict), or the PATCH made the suite uncollectable (a
            # syntax error in the package, a broken conftest), which the
            # official grader scores as every test failed. Seven no-recovery
            # cells were silently dropped from the first unblinding as
            # "ungradable" — the most catastrophic patches in the sweep,
            # vanishing from the very endpoint they were the strongest
            # evidence for. Disambiguate by asking the baseline: reset the
            # tree, run the same script with no patch. Results there mean the
            # environment is alive and the patch is what killed the suite.
            reset = sandbox.exec(
                f"cd {workdir} && git checkout -q {target} -- . && git clean -qfd", timeout=300
            )
            if reset.exit_code != 0:
                return None
            baseline = sandbox.exec(build_eval_script(instance, workdir=workdir), timeout=timeout)
            if not parse_eval_output(instance, baseline.stdout):
                return None
            suite_killed = True

    def passed(test: str) -> bool:
        return per_test.get(test) in PASSING_STATUSES

    if suite_killed:
        # Official semantics: a graded test the run never reported is a
        # failure. Recorded per test so the sidecar shows what the patch did.
        per_test = dict.fromkeys(
            (*instance.fail_to_pass, *instance.pass_to_pass), "MISSING"
        )
    f2p_passed = sum(1 for t in instance.fail_to_pass if passed(t))
    p2p_passed = sum(1 for t in instance.pass_to_pass if passed(t))
    return GradeReport(
        instance_id=instance.instance_id,
        resolved=(
            not apply_error
            and not suite_killed
            and f2p_passed == len(instance.fail_to_pass)
            and p2p_passed == len(instance.pass_to_pass)
        ),
        fail_to_pass_passed=f2p_passed,
        fail_to_pass_total=len(instance.fail_to_pass),
        pass_to_pass_passed=p2p_passed,
        pass_to_pass_total=len(instance.pass_to_pass),
        per_test=per_test,
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
        # The harness's own artifacts (.taste/plan.json, monitor verdicts) live
        # in the tree and were leaking into the prediction. They apply cleanly
        # and grade harmlessly, which is exactly why nobody noticed; but a
        # prediction is the agent's source change and nothing else.
        ["git", "diff", base_commit, "--", ".", ":(exclude)*test_*.py", ":(exclude)*tests/*",
         ":(exclude).taste/*"],
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


def _mirror_is_usable(mirror: Path) -> bool:
    """Whether a cached mirror can actually answer for a commit.

    Being a git directory is not enough, and the difference is not academic:
    an interrupted `git clone --bare` leaves a directory that satisfies
    `rev-parse --git-dir` while containing **zero commits**. Observed on two
    of five mirrors during a dry run. Accepting one means the top-up fetch is
    skipped, `git archive` fails, and the instance dies far from the cause.
    """
    probe = subprocess.run(
        ["git", "rev-list", "--all", "--count"], cwd=mirror, capture_output=True, text=True
    )
    if probe.returncode != 0:
        return False
    return probe.stdout.strip().isdigit() and int(probe.stdout.strip()) > 0


def fetch_repo(instance: SWEInstance, cache: Path) -> Path:
    """A local mirror of the instance's repository, cloned once and reused.

    Cloning per cell would be absurd — django is hundreds of megabytes and a
    sweep touches the same repo dozens of times — so the clone is cached per
    repository and every cell copies a working tree out of it.

    The mirror is bare and fetched with full history because ``base_commit``
    is usually not on any branch tip; a shallow clone cannot reach it.
    """
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    mirror = cache / f"{instance.repo.replace('/', '__')}.git"
    if mirror.exists() and not _mirror_is_usable(mirror):
        # An interrupted clone leaves a directory that poisons every later
        # cell on this repository: cat-file fails, the top-up fetch fails, and
        # instances die for a reason that has nothing to do with the arm.
        shutil.rmtree(mirror, ignore_errors=True)
    if not mirror.exists():
        subprocess.run(
            ["git", "clone", "--bare", "--quiet",
             f"https://github.com/{instance.repo}.git", str(mirror)],
            check=True, capture_output=True,
        )
    have = subprocess.run(
        ["git", "cat-file", "-e", f"{instance.base_commit}^{{commit}}"],
        cwd=mirror, capture_output=True,
    )
    if have.returncode != 0:
        # The mirror predates this instance's commit; top it up rather than
        # re-cloning several hundred megabytes.
        subprocess.run(["git", "fetch", "--quiet", "origin", "+refs/*:refs/*"],
                       cwd=mirror, check=False, capture_output=True)
    return mirror


def materialize_from_repo(
    instance: SWEInstance, dest: Path, *, cache: Path, mirror: Path | None = None
) -> Path:
    """A working tree at the instance's ``base_commit``, with no history.

    The tree is extracted with ``git archive`` rather than checked out, so the
    upstream history never reaches the workspace at all. That is not tidiness:
    read-only git is deliberately available to workers, and a real clone would
    let an agent read the upstream fix for its own issue — and every later
    commit that references it — straight out of the object database.
    """
    mirror = Path(mirror) if mirror else fetch_repo(instance, cache)
    workspace = Path(dest)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    archive = subprocess.run(
        ["git", "archive", "--format=tar", instance.base_commit],
        cwd=mirror, capture_output=True, check=True,
    )
    subprocess.run(["tar", "-x", "-C", str(workspace)], input=archive.stdout, check=True)
    return _init_workspace(instance, workspace)


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

    return _init_workspace(instance, workspace)


#: Top-level module each benchmark repository imports as. The parity check
#: needs the name to prove the agent's environment resolves the INSTALLED
#: project: on a bare host checkout `import matplotlib` "succeeds" as an
#: uncompiled namespace package with a garbage version, which is exactly the
#: silent form of bug 20 the check exists to refuse.
REPO_MODULES = {
    "astropy/astropy": "astropy",
    "django/django": "django",
    "matplotlib/matplotlib": "matplotlib",
    "mwaskom/seaborn": "seaborn",
    "pallets/flask": "flask",
    "psf/requests": "requests",
    "pydata/xarray": "xarray",
    "pylint-dev/pylint": "pylint",
    "pytest-dev/pytest": "pytest",
    "scikit-learn/scikit-learn": "sklearn",
    "sphinx-doc/sphinx": "sphinx",
    "sympy/sympy": "sympy",
}


def materialize_from_image(
    sandbox, instance: SWEInstance, workspace: Path
) -> Path:
    """Materialise the workspace FROM the container's /testbed.

    Not from a git mirror. The mirror gives tracked files only; the image's
    tree additionally holds everything the build produced -- compiled
    extensions, generated parsers, pre_install edits. An agent container
    synced against a mirror-materialised workspace would have the host
    overwrite those artifacts with clean checkouts on the first push, and
    imports die exactly as they did on the host (bug 20's ghost).

    Ordering is load-bearing: ``prepare_container_tree`` first (it moves the
    upstream ``.git`` out and commits the sync baseline), the archive second,
    so host and container start from the identical tree and the container's
    fresh ``.git`` never reaches the host (the host repo is the instrument's;
    a nested container .git inside it would be bug D3 with git metadata).
    """
    from taste.routing import prepare_container_tree

    prepare_container_tree(sandbox, workdir=sandbox.workdir, hide_upstream=True)
    workspace = Path(workspace)
    if workspace.exists() and any(workspace.iterdir()):
        raise FileExistsError(f"workspace {workspace} already exists; refusing to overwrite")
    workspace.mkdir(parents=True, exist_ok=True)

    tar_path = "/tmp/taste_ws.tar"
    packed = sandbox.exec(
        f"tar -C {sandbox.workdir} --exclude=.git -cf {tar_path} .", timeout=600
    )
    if packed.exit_code != 0:
        raise RuntimeError(f"could not pack /testbed: {packed.stderr or packed.stdout}")
    payload = sandbox.get_bytes(tar_path)
    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        archive.extractall(workspace, filter="data")
    return _init_workspace(instance, workspace)


def environment_parity_check(sandbox, instance: SWEInstance) -> str | None:
    """Prove the agent's environment is the one the benchmark grades. $0.

    Returns None when sound, else a reason string. Run per cell BEFORE any
    model call: a cell that fails here is infrastructure, and the difference
    between catching it now and catching it after the run is the entire cost
    of the run. This is the check whose absence let 26 zero-step runs read
    as the agent failing.
    """
    module = REPO_MODULES.get(instance.repo)
    if module is None:
        return f"no module mapping for {instance.repo}; cannot verify the environment"
    probe = (
        f"python -c \"import {module} as m; import sys; "
        f"assert m.__file__, 'namespace package'; "
        f"print(getattr(m, '__version__', '?')); print(m.__file__)\""
    )
    run = getattr(sandbox, "exec_in_env", sandbox.exec)
    result = run(probe, timeout=120)
    if result.exit_code != 0:
        return f"import {module} failed in the agent environment: {(result.stderr or result.stdout)[-300:]}"
    lines = result.stdout.strip().splitlines()
    version = lines[0] if lines else "?"
    location = lines[-1] if len(lines) > 1 else ""
    if "unknown" in version:
        # The signature of a source tree shadowing the installed package.
        return f"{module} resolves to an uninstalled tree (version {version!r} at {location!r})"
    if "/testbed" not in location and "site-packages" not in location:
        return f"{module} resolves outside the graded environment: {location!r}"
    return None


def _init_workspace(instance: SWEInstance, workspace: Path) -> Path:
    """A fresh single-commit repository over whatever is already in the tree."""

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
