"""Regression-gated verification: roll back on what actually broke.

The Monitor's checks were written by the planner — a `python -c` probe, a
guessed pytest invocation — and in the rollback arm they rejected patches the
official grader accepted on 9 of 18 failed instances. Rollback then destroyed
good work on a false verdict. The precision of the verifier set the exchange
rate between a clean final tree and a solved task, and the verifier was the
weakest part of the harness.

This gate replaces the planner's check with the repository's own tests, run
in the agent's environment against the agent's tree: the same held-out
passing set the instrument replays afterwards, used *during* the run. A step
is rejected if and only if a test that passed at the start of the run fails
now. It does not judge whether the task is solved — that is the grader's job
and the model's — only whether the step broke something that worked. The
observation layer becomes the harness's monitor, which is what an OS-level
recovery primitive should be built on.

What it deliberately does not do: apply the benchmark's hidden test patch
(that would leak the graded target into the agent's environment) or restore
test files the agent edited (the instrument's replay does that; a gate that
runs whatever the tree holds can be gamed by deleting tests, and the
disagreement between gate and instrument is then a measurable quantity
rather than a silent one).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from taste.benchmarks import swebench
from taste.benchmarks.swebench import PASSING_STATUSES
from taste.cores import MonitorResult
from taste.llm import InfraFailure


@dataclass
class RegressionGate:
    instance: swebench.SWEInstance
    run: Callable[..., Any]
    """``run(command, timeout=...) -> ExecResult`` in the agent's environment
    (the router's exec)."""
    timeout: int = 1200
    split: str = "all"
    """Which of the instance's previously-passing tests the gate is allowed
    to read. ``all``: every one (the gate's oracle then coincides with the
    grading oracle, and a clean final tree is partly guaranteed by
    construction). ``half``: a deterministic half of the test ids by name
    hash. The suite still runs whole -- most instances hold their tests in
    one file, and a file-level split left 8 of 21 cells with nothing to
    watch -- but results for the other half are dropped before the gate
    reads them, so the grader's verdict on that half is a genuine held-out
    test of whether gating generalises beyond the tests it watches."""
    baseline_pass: frozenset[str] = frozenset()
    established: bool = False
    checks: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    files_fn: Callable[[Any], Any] | None = None
    """Substrate seam. Verified is the default; a Live gate injects
    ``probe_files`` / a bracketed pytest script / ``parse_live_output`` so the
    gate reads the same oracle the instrument replays on that substrate."""
    command_fn: Callable[[Any, list[str]], str] | None = None
    parse_fn: Callable[[Any, str], dict[str, str]] | None = None

    def watched_files(self) -> list[str]:
        if self.files_fn is not None:
            return sorted(self.files_fn(self.instance))
        return sorted(swebench.member_test_files(self.instance))

    def watched_ids(self) -> frozenset[str] | None:
        """Test ids the gate may read; ``None`` means every id."""
        if self.split != "half":
            return None
        import hashlib

        return frozenset(
            t for t in self.instance.pass_to_pass if int(hashlib.sha1(t.encode()).hexdigest(), 16) % 2 == 0
        )

    def _suite(self) -> dict[str, str]:
        files = self.watched_files()
        command = (
            self.command_fn(self.instance, files) if self.command_fn is not None
            else swebench.plain_suite_command(self.instance, files)
        )
        result = self.run(command, timeout=self.timeout)
        if getattr(result, "timed_out", False):
            raise TimeoutError("regression gate suite timed out")
        parse = self.parse_fn or swebench.parse_eval_output
        statuses = parse(self.instance, result.stdout)
        allowed = self.watched_ids()
        if allowed is not None:
            statuses = {t: s for t, s in statuses.items() if t in allowed}
        return statuses

    def establish_baseline(self) -> None:
        """Run the suite once before the agent acts. No results at all means
        the environment cannot run the repository's tests — infrastructure,
        refused before a model call is paid for, never a silent all-pass."""
        allowed = self.watched_ids()
        if not self.watched_files() or (allowed is not None and not allowed):
            # A half-split of a one-test instance leaves nothing to watch; the
            # gate then has no oracle and must say so rather than pass everything.
            raise InfraFailure(
                "regression gate: no tests in the watched split",
                attempts=1,
                last_error="empty watched split",
            )
        statuses = self._suite()
        if not statuses:
            raise InfraFailure(
                "regression gate: the repository's tests produced no results at baseline",
                attempts=1,
                last_error="no parseable test results at baseline",
            )
        self.baseline_pass = frozenset(t for t, s in statuses.items() if s in PASSING_STATUSES)
        self.established = True

    def check(self) -> MonitorResult:
        if not self.established:
            raise RuntimeError("regression gate used before its baseline was established")
        self.checks += 1
        try:
            now = self._suite()
        except TimeoutError:
            # Conservative: an unfinished suite is not evidence the tree is
            # sound, and the reset costs one attempt rather than a run.
            self.history.append({"check": self.checks, "outcome": "timeout"})
            return MonitorResult(passed=False, reason="regression gate: suite timed out")
        if not now:
            # The suite could not even run: the step broke collection or an
            # import. The grader would score every test failed; so do we.
            regressed = sorted(self.baseline_pass)
            reason = "regression gate: the test suite no longer runs"
        else:
            regressed = sorted(t for t in self.baseline_pass if now.get(t) not in PASSING_STATUSES)
            reason = (
                f"regression gate: {len(regressed)} previously-passing test(s) now fail"
                if regressed else "regression gate: no previously-passing test regressed"
            )
        self.history.append({"check": self.checks, "regressed": len(regressed)})
        evidence = "\n".join(regressed[:40])
        return MonitorResult(passed=not regressed, reason=reason, evidence=evidence)
