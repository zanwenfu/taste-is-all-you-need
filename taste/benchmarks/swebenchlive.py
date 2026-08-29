"""SWE-bench-Live: the substrate the pre-registered switch names.

Live exists because Verified's contamination made its resolve scores stop
meaning much, and it matters to this project for a sharper reason: its
median held-out oracle is ~37x Verified's (the first lite instance alone
carries 1,220 PASS_TO_PASS tests), and the oracle is the regression net.
The valid pilot measured zero events on Verified's adjacent-file window;
this is the pre-declared response, not a post-hoc hunt.

The adapter is deliberately thinner than Verified's, because the dataset is
better: `test_cmds` and `log_parser` travel per instance, so there is no
vendored spec table to keep line-for-line auditable, and the image name is
one published convention. What still deserves care:

- **Their grading semantics differ from official SWE-bench and we mirror
  them exactly, labelled.** Their parser normalises to pass/skip/fail with
  `'pass' in status.lower()` — so XFAIL counts as *fail* (upstream SWE-bench
  counts it as pass) — and a graded test MISSING from the log blocks
  nothing (upstream counts it failed). Resolve requires: every FAIL_TO_PASS
  present-and-passing, and zero recorded failures in either set. Mixing the
  two rulebooks would make our Live numbers incomparable to their
  leaderboard, which is the only reason to run their benchmark.
- **The probe stays file-scoped.** `test_cmds` is usually the whole suite
  (`pytest -rA`); replaying that at every observation multiplies suite
  runtime by the timeline length. For pytest-parser instances the members'
  file paths are their id prefixes, so the probe runs exactly the files the
  oracle lives in — same trick as Verified's directives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from taste.benchmarks.swebench_log import parse_log_pytest
from taste.replay import SuiteProbe

__all__ = [
    "LiveInstance",
    "build_live_gate_script",
    "build_live_probe_script",
    "grade_live_in_sandbox",
    "live_parity_check",
    "live_suite_factory",
    "load_live_dataset",
    "parse_live_output",
    "probe_files",
]

START_MARKER = "TASTE_START_TEST_OUTPUT"
END_MARKER = "TASTE_END_TEST_OUTPUT"


@dataclass(frozen=True)
class LiveInstance:
    """One Live task. The gold patch is deliberately absent — the loader
    drops it so nothing downstream can leak it to an agent; the golden
    check re-reads the raw file, exactly as on Verified."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    test_cmds: tuple[str, ...]
    log_parser: str
    image: str = ""

    @property
    def repo_short(self) -> str:
        return self.repo.split("/")[-1]

    @property
    def published_image(self) -> str:
        """Their harness's own naming (evaluation.py:get_default_image_name):
        docker.io/starryzhang, `__` -> `_1776_`, lowercased."""
        name = self.instance_id.replace("__", "_1776_").lower()
        return f"starryzhang/sweb.eval.x86_64.{name}"


def load_live_dataset(path: Path) -> list[LiveInstance]:
    instances: list[LiveInstance] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        instances.append(
            LiveInstance(
                instance_id=row["instance_id"],
                repo=row["repo"],
                base_commit=row["base_commit"],
                problem_statement=row.get("problem_statement", ""),
                test_patch=row.get("test_patch", ""),
                fail_to_pass=tuple(_as_list(row.get("FAIL_TO_PASS"))),
                pass_to_pass=tuple(_as_list(row.get("PASS_TO_PASS"))),
                test_cmds=tuple(_as_list(row.get("test_cmds"))),
                log_parser=str(row.get("log_parser", "") or ""),
            )
        )
    return instances


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parsed = json.loads(value) if value.strip().startswith("[") else [value]
        return [str(v) for v in parsed]
    return [str(v) for v in value]


def probe_files(instance: LiveInstance) -> tuple[str, ...]:
    """The files the oracle lives in, from the members' own id prefixes.

    Empty means "cannot scope" — a non-pytest parser, or ids that do not
    look like paths — and the caller falls back to the full test_cmds. A
    wrong guess here is loud (pytest errors on a missing path), never a
    silently narrower oracle: every P2P member's file must be present or we
    refuse to scope.
    """
    if instance.log_parser.strip().lower() != "pytest":
        return ()
    files: set[str] = set()
    for member in instance.pass_to_pass:
        prefix = member.split("::")[0]
        if not prefix.endswith(".py") or prefix.startswith("/"):
            return ()
        files.add(prefix)
    return tuple(sorted(files))


def build_live_probe_script(instance: LiveInstance, *, workdir: str = "/testbed") -> str:
    """The P2P oracle at one observation, bracketed like Verified's.

    The graded files are restored from the sync baseline before AND after:
    an agent that edited a test file would otherwise have its edit graded by
    the very oracle it rewrote. `taste-baseline` is the snapshot branch
    `prepare_container_tree` creates, which is the IMAGE's state — the same
    bug-B8 reasoning as Verified's executor.

    No conda activation: Live images are RepoLaunch-built with the
    environment on the shell's default PATH. The parity check proves that
    per cell rather than assuming it.
    """
    files = probe_files(instance)
    if files:
        joined = " ".join(files)
        commands = [f"python -m pytest -rA {joined}"]
        restore = f"git checkout -q taste-baseline -- {joined} 2>/dev/null || true"
    else:
        commands = list(instance.test_cmds)
        restore = ""
    body = "\n".join(commands)
    return "\n".join(
        part
        for part in (
            "exec 2>&1",
            f"cd {workdir}",
            restore,
            f"echo '{START_MARKER}'",
            body,
            f"echo '{END_MARKER}'",
            restore,
        )
        if part
    )


def build_live_gate_script(instance: LiveInstance, files, *, workdir: str = "/testbed") -> str:
    """The regression gate's suite on Live: the oracle's files, run on the
    agent's tree as it stands (no restore -- the gate reads what the agent
    left; the instrument's replay restores). Bracketed like the probe so
    ``parse_live_output`` reads only the run."""
    files = list(files) or list(probe_files(instance))
    body = f"python -m pytest -rA {' '.join(files)}" if files else "\n".join(instance.test_cmds)
    return "\n".join(("exec 2>&1", f"cd {workdir}", f"echo '{START_MARKER}'", body, f"echo '{END_MARKER}'"))


def parse_live_output(instance: LiveInstance, log: str) -> dict[str, str]:
    """Statuses for the bracketed slice, in upstream vocabulary.

    Live's own harness reuses SWE-bench's pytest grammar and then squashes
    to pass/skip/fail; we keep the upstream statuses so verdicts_from and
    the grade can each apply their own (different, both documented) rules.
    """
    start = log.find(START_MARKER)
    end = log.find(END_MARKER, start + 1) if start >= 0 else -1
    if start < 0:
        return {}
    newline = log.find("\n", start)
    body = log[newline + 1 : end] if end > newline >= 0 else log[start:]
    if instance.log_parser.strip().lower() != "pytest":
        return {}
    return dict(parse_log_pytest(body))


def live_suite_factory(instance: LiveInstance, *, timeout: int = 3600) -> SuiteProbe:
    return SuiteProbe(
        name=f"live-p2p::{instance.instance_id}",
        command=build_live_probe_script(instance),
        members=instance.pass_to_pass,
        parse=partial(parse_live_output, instance),
        timeout=timeout,
    )


def _live_passed(status: str) -> bool:
    """Their normalisation, verbatim: 'pass' in status.lower(). XFAIL is a
    FAIL here — 'xfail' contains no 'pass' — which inverts upstream
    SWE-bench. Mirrored, not endorsed: comparability with their leaderboard
    is the entire point of running their benchmark."""
    return "pass" in status.lower()


def _live_failed(status: str) -> bool:
    return "fail" in status.lower() or "error" in status.lower()


def grade_live_in_sandbox(
    sandbox, instance: LiveInstance, model_patch: str, *, timeout: int = 3600
):
    """Live's resolve verdict for one final patch. None = ungradable.

    Mirrors evaluation.py's rule exactly: resolved iff every FAIL_TO_PASS is
    present-and-passing, and neither set records a failure. A graded test
    absent from the log blocks nothing (upstream SWE-bench would count it
    failed) — their leniency, their leaderboard, their rule.
    """
    from taste.benchmarks.swebench import GradeReport
    from taste.routing import prepare_container_tree

    workdir = sandbox.workdir
    target = prepare_container_tree(sandbox, workdir=workdir, hide_upstream=False)
    reset = sandbox.exec(
        f"cd {workdir} && git checkout -q {target} -- . && git clean -qfd", timeout=300
    )
    if reset.exit_code != 0:
        return None

    for name, patch in (("test", instance.test_patch), ("pred", model_patch)):
        if not patch.strip():
            continue
        sandbox.put_text(f"/tmp/taste_{name}.diff", patch + "\n")
        applied = sandbox.exec(
            f"cd {workdir} && git apply -v /tmp/taste_{name}.diff", timeout=300
        )
        if applied.exit_code != 0:
            if name == "test":
                # Their harness applies the test patch unconditionally; a
                # tree where it will not apply is infrastructure.
                return None
            return GradeReport(
                instance_id=instance.instance_id,
                resolved=False,
                fail_to_pass_passed=0,
                fail_to_pass_total=len(instance.fail_to_pass),
                pass_to_pass_passed=0,
                pass_to_pass_total=len(instance.pass_to_pass),
            )

    script = "\n".join(
        ("exec 2>&1", f"cd {workdir}", f"echo '{START_MARKER}'",
         *instance.test_cmds, f"echo '{END_MARKER}'")
    )
    result = sandbox.exec(script, timeout=timeout)
    statuses = parse_live_output(instance, result.stdout)
    if not statuses:
        return None

    f2p_pass = [t for t in instance.fail_to_pass if _live_passed(statuses.get(t, ""))]
    f2p_fail = [t for t in instance.fail_to_pass if _live_failed(statuses.get(t, ""))]
    p2p_pass = [t for t in instance.pass_to_pass if _live_passed(statuses.get(t, ""))]
    p2p_fail = [t for t in instance.pass_to_pass if _live_failed(statuses.get(t, ""))]
    resolved = (
        len(f2p_pass) == len(instance.fail_to_pass)
        and not f2p_fail
        and not p2p_fail
    )
    return GradeReport(
        instance_id=instance.instance_id,
        resolved=resolved,
        fail_to_pass_passed=len(f2p_pass),
        fail_to_pass_total=len(instance.fail_to_pass),
        pass_to_pass_passed=len(p2p_pass),
        pass_to_pass_total=len(instance.pass_to_pass),
        per_test=dict(statuses),
    )


def live_parity_check(sandbox, instance: LiveInstance) -> str | None:
    """$0 proof the agent's shell sees a working environment. Generic across
    Live's 156 repos, so it proves execution rather than one import: python
    must exist, and pytest must COLLECT the oracle's files without error.
    Collection imports the package under test — a bare uninstalled checkout
    (bug 20's signature) dies here, before any model call."""
    run = getattr(sandbox, "exec_in_env", sandbox.exec)
    python = run("python -c 'import sys; print(sys.version.split()[0])'", timeout=60)
    if python.exit_code != 0:
        return f"no working python on the agent's PATH: {(python.stderr or python.stdout)[-200:]}"
    files = probe_files(instance)
    if files:
        collect = run(
            f"cd {sandbox.workdir} && python -m pytest --collect-only -q {' '.join(files)}",
            timeout=300,
        )
        if collect.exit_code not in (0, 5):
            return (
                "pytest cannot collect the oracle's files in the agent "
                f"environment: {(collect.stdout or collect.stderr)[-300:]}"
            )
    return None
