"""Gate 0: does the instrument measure what it claims, before we pay to use it.

The four metrics this project reports are only as good as the pipeline that
produces them, and that pipeline has already been wrong twice in ways that
would have inverted the headline result. So it is validated first, against
cases whose answers are known by construction, at zero API cost. If Gate 0
fails, no amount of model spend produces an interpretable number.

Five checks, each with a pre-declared threshold:

**Baseline liveness.** Every observation of a clean run must actually answer
`pass`. This exists because zero contamination events is *also* what a totally
dead instrument reports — if every probe errors, no episode can open, and the
negative control scored that as a perfect result. The instrument's death and
its best possible outcome were the same number. It is exactly the shape of the
probe command being wrong for 46% of the benchmark: nothing ran, and every
downstream figure was computed over holes.

**Negative control.** Replay a trajectory that contains no regression. The
pipeline must report zero regression events. A false positive here means the
instrument manufactures the phenomenon it is meant to detect, which is the
worst failure available to it.

**Positive control.** Inject a regression at a known observation. The pipeline
must locate that exact observation. This is what the earlier bisection failed:
it could not see a regression that was later repaired, so a recovering arm
looked clean.

**Flake screen.** Run the same probe against the same tree k times. Any
disagreement is instrument noise that would otherwise be reported as
contamination, and its rate bounds how small an effect can be believed.

**Unknown rate.** Probes that could not run are neither pass nor fail. A high
rate means the timeline has holes, and a metric computed over holes is not a
measurement.

Every threshold is stated up front so the gate cannot be moved after seeing
the numbers.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from taste.memory import Memory
from taste.replay import Probe, Replayer, reconstruct
from taste.shadow import ShadowCommit, ShadowLog

# Pre-declared, before any result is seen.
NEGATIVE_CONTROL_MIN = 0.95
"""Fraction of clean trajectories that must yield zero regression events."""
POSITIVE_CONTROL_MIN = 0.90
"""Fraction of injected regressions whose onset must be located exactly."""
FLAKE_MAX = 0.02
"""Maximum probe-verdict disagreement across repeats."""
UNKNOWN_MAX = 0.05
"""Maximum fraction of probe executions returning neither pass nor fail."""
BASELINE_LIVENESS_MIN = 0.99
"""Minimum fraction of clean-run observations that must answer `pass`.

Zero contamination events is also what a dead instrument reports, so the
gate needs one check that fails loudly when nothing ran at all."""


@dataclass
class CheckResult:
    name: str
    passed: bool
    value: float
    threshold: float
    detail: str = ""
    samples: int = 0

    def render(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return (
            f"[{mark}] {self.name:<20} {self.value:6.3f} "
            f"(threshold {self.threshold:.3f}, n={self.samples}) {self.detail}"
        )


@dataclass
class Gate0Report:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def render(self) -> str:
        lines = [c.render() for c in self.checks]
        lines.append("")
        lines.append(
            "GATE 0 PASSED — the instrument may be trusted with paid runs."
            if self.passed
            else "GATE 0 FAILED — fix the instrument; paid runs would be uninterpretable."
        )
        return "\n".join(lines)


# ------------------------------------------------------------------ scenarios


@dataclass
class Trajectory:
    """A synthetic run: a sequence of tree states, with a known answer."""

    name: str
    states: list[str]
    """File contents at each observation."""
    regression_at: int | None = None
    """1-based observation index where the probe first fails, or None."""


def clean_trajectory(length: int = 6) -> Trajectory:
    """A run that never breaks the probe."""
    return Trajectory(
        name="clean",
        states=[f"def value():\n    return 1\n\n\n# edit {i}\n" for i in range(length)],
    )


def regressed_trajectory(length: int = 8, onset: int = 5) -> Trajectory:
    """A run that breaks the probe at a known point and never repairs it."""
    return Trajectory(
        name=f"regress@{onset}",
        states=[
            f"def value():\n    return {2 if i + 1 >= onset else 1}\n\n\n# edit {i}\n"
            for i in range(length)
        ],
        regression_at=onset,
    )


def recovered_trajectory(length: int = 9, onset: int = 4, recovery: int = 7) -> Trajectory:
    """Breaks, then is repaired — the case the previous instrument could not see.

    Included in the positive control deliberately: an instrument that only
    finds regressions surviving to the end would score 100% on the other
    shapes and still be useless for the arm whose whole claim is recovery.
    """
    def value(i: int) -> int:
        observation = i + 1
        return 2 if onset <= observation < recovery else 1

    return Trajectory(
        name=f"recovered@{onset}-{recovery}",
        states=[f"def value():\n    return {value(i)}\n\n\n# edit {i}\n" for i in range(length)],
        regression_at=onset,
    )


# sys.executable, not "python". A clean Ubuntu ships python3 and no `python`
# at all, so the bare name made every probe exit 127 -- and Gate 0 then failed
# on a fresh host for a reason that had nothing to do with the instrument it
# was validating. Using the running interpreter also keeps the probe inside
# whatever virtualenv invoked it.
PROBE = Probe(
    name="value_is_one",
    command=f'{sys.executable} -c "import lib; assert lib.value() == 1"',
)


# ------------------------------------------------------------------ execution


def materialise(trajectory: Trajectory, workspace: Path) -> tuple[Memory, list[ShadowCommit]]:
    """Play a trajectory into a real repository, observing at each state."""
    import subprocess

    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "lib.py").write_text(trajectory.states[0])
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t.co", "commit", "-qm", "base"],
        cwd=workspace, check=True, capture_output=True,
    )

    memory = Memory.open_session(workspace, "gate0")
    log = ShadowLog(
        memory,
        gitdir=Path(memory.repo.git_dir) / "taste",
        session="gate0",
        cost_pair_reader=lambda: (0.0, 0.0),
    )
    for state in trajectory.states:
        (workspace / "lib.py").write_text(state)
        log.observe(step_id="s", attempt=1, trigger="worker")
    return memory, list(log.timeline())


def negative_control(make_workspace: Callable[[str], Path], *, samples: int = 5) -> CheckResult:
    """Clean trajectories must produce zero regression events *and be alive*.

    The liveness half is not belt-and-braces. Zero events is also what a
    completely dead instrument reports: if every probe execution errors, no
    episode can open, and the naive form of this check scored that as a
    perfect result. The instrument's death and its best possible outcome were
    the same number — so a gate that only counted events would certify a
    harness that could not run a single test.
    """
    clean = 0
    offenders: list[str] = []
    for index in range(samples):
        memory, timeline = materialise(clean_trajectory(), make_workspace(f"neg{index}"))
        report = reconstruct(memory, timeline, [PROBE], session="gate0")
        if report.never_passed:
            offenders.append(f"{index}:dead")
        elif report.contamination_events == 0:
            clean += 1
        else:
            offenders.append(f"{index}:{report.contamination_events}")
    rate = clean / samples if samples else 0.0
    return CheckResult(
        "negative control", rate >= NEGATIVE_CONTROL_MIN, rate, NEGATIVE_CONTROL_MIN,
        detail=f"false positives on {offenders}" if offenders else "no false positives",
        samples=samples,
    )


def baseline_liveness(make_workspace: Callable[[str], Path], *, samples: int = 3) -> CheckResult:
    """Every observation of a clean run must yield a real verdict.

    Stated separately from the negative control so that a broken environment
    reports as a broken environment rather than as a subtly worse score. This
    is the check that would have caught the probe command being wrong for 46%
    of the benchmark: those probes never ran, every verdict was an error, and
    every downstream number was silently computed over holes.

    On a real instance the same property is asserted per instance before it
    enters the frame — an instance whose PASS_TO_PASS does not pass at
    ``base_commit`` is excluded, with the reason recorded like any other.
    """
    live = 0
    total = 0
    dead: list[str] = []
    for index in range(samples):
        memory, timeline = materialise(clean_trajectory(), make_workspace(f"live{index}"))
        replayer = Replayer(memory, [PROBE])
        for verdict in replayer.verdicts_across(timeline, PROBE):
            total += 1
            if verdict == "pass":
                live += 1
            else:
                dead.append(verdict)
    rate = live / total if total else 0.0
    return CheckResult(
        "baseline liveness", rate >= BASELINE_LIVENESS_MIN, rate, BASELINE_LIVENESS_MIN,
        detail="every observation answered" if not dead else f"non-passing verdicts: {dead[:5]}",
        samples=total,
    )


def positive_control(make_workspace: Callable[[str], Path]) -> CheckResult:
    """Injected regressions must be located at exactly the right observation."""
    cases = [
        regressed_trajectory(onset=3),
        regressed_trajectory(onset=5),
        regressed_trajectory(length=12, onset=9),
        recovered_trajectory(onset=4, recovery=7),
        recovered_trajectory(length=11, onset=3, recovery=9),
    ]
    exact = 0
    misses: list[str] = []
    for index, trajectory in enumerate(cases):
        memory, timeline = materialise(trajectory, make_workspace(f"pos{index}"))
        report = reconstruct(memory, timeline, [PROBE], session="gate0")
        located = report.episodes[0].onset_seq if report.episodes else None
        if located == trajectory.regression_at:
            exact += 1
        else:
            misses.append(f"{trajectory.name}: got {located}, want {trajectory.regression_at}")
    rate = exact / len(cases)
    return CheckResult(
        "positive control", rate >= POSITIVE_CONTROL_MIN, rate, POSITIVE_CONTROL_MIN,
        detail="; ".join(misses) if misses else "all onsets located exactly",
        samples=len(cases),
    )


def flake_screen(make_workspace: Callable[[str], Path], *, k: int = 3) -> CheckResult:
    """Repeated probes on identical trees must agree.

    Disagreement here would be reported as contamination, so its rate is the
    floor on any effect the study can believe.
    """
    memory, timeline = materialise(regressed_trajectory(), make_workspace("flake"))
    disagreements = 0
    executions = 0
    for commit in timeline:
        verdicts = set()
        for _ in range(k):
            replayer = Replayer(memory, [PROBE])  # fresh: no memoisation
            verdicts.add(replayer.verdict_at(commit.sha, PROBE))
            executions += 1
        if len(verdicts) > 1:
            disagreements += 1
    rate = disagreements / len(timeline) if timeline else 1.0
    return CheckResult(
        "flake screen", rate <= FLAKE_MAX, rate, FLAKE_MAX,
        detail=f"{disagreements} unstable observations of {len(timeline)}",
        samples=executions,
    )


def unknown_rate(make_workspace: Callable[[str], Path]) -> CheckResult:
    """Probes that could not run leave holes in the timeline."""
    memory, timeline = materialise(regressed_trajectory(), make_workspace("unknown"))
    replayer = Replayer(memory, [PROBE])
    verdicts = replayer.verdicts_across(timeline, PROBE)
    unknown = sum(1 for v in verdicts if v not in ("pass", "fail"))
    rate = unknown / len(verdicts) if verdicts else 1.0
    return CheckResult(
        "unknown rate", rate <= UNKNOWN_MAX, rate, UNKNOWN_MAX,
        detail=f"{unknown} of {len(verdicts)} probe executions",
        samples=len(verdicts),
    )


def run(root: Path) -> Gate0Report:
    """Run every check. Zero API cost; Docker/CPU only."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    def workspace(name: str) -> Path:
        import shutil

        path = root / name
        if path.exists():
            shutil.rmtree(path)
        return path

    return Gate0Report(
        checks=[
            baseline_liveness(workspace),
            negative_control(workspace),
            positive_control(workspace),
            flake_screen(workspace),
            unknown_rate(workspace),
        ]
    )
