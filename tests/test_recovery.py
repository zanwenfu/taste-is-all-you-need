"""The trap handler: diagnosis, the action space, and the experimental arms.

The claim being tested is that a failure is no longer answered blindly. The
kernel reads a fault frame, names the fault from a deterministic rule table,
and dispatches a typed action — and every one of those steps costs nothing,
so none of these tests needs a model.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from taste.cores import MonitorResult, Plan, Step, Verification, WorkerResult
from taste.kernel import Kernel
from taste.recovery import (
    ActionKind,
    Budget,
    Diagnosis,
    FailureClass,
    FixedPolicy,
    GuidanceBook,
    Hypothesis,
    RecoveryConfig,
    Signals,
    StepHistory,
    TieredPolicy,
    diagnose,
    fingerprint,
    observe,
    parse_output,
)
from tests.golden import rollback_scenario
from tests.test_golden_baseline import EXPECTED_EVENTS


def _sig(**kwargs) -> Signals:
    return Signals(**kwargs)


def _hist(step_id: str = "step-01") -> StepHistory:
    return StepHistory(step_id=step_id)


def _diag(cls: FailureClass, confidence: float = 0.9) -> Diagnosis:
    return Diagnosis(top=Hypothesis(cls, confidence, "test", "test.rule"))


# ================================================================ ablation


def test_recovery_disabled_reproduces_the_baseline_exactly(refactor_workspace: Path) -> None:
    """Off means off: no diagnosis, no new events, identical behavior."""
    sig = rollback_scenario(refactor_workspace).run(refactor_workspace)
    assert sig.events == EXPECTED_EVENTS


def test_recovery_disabled_emits_no_recovery_events(refactor_workspace: Path) -> None:
    scenario = rollback_scenario(refactor_workspace)
    scenario.run(refactor_workspace)
    assert not [e for e in scenario.events if e.kind.startswith("recovery.")]


# ================================================================ parsing


def test_parse_output_extracts_pytest_failures() -> None:
    parsed = parse_output(
        "FAILED tests/test_a.py::test_one - AssertionError\n"
        "FAILED tests/test_b.py::test_two - ValueError\n"
        "2 failed, 5 passed in 0.3s",
        "",
    )
    assert parsed["failing_tests"] == ("tests/test_a.py::test_one", "tests/test_b.py::test_two")
    assert parsed["failed_count"] == 2
    assert parsed["passed_count"] == 5


def test_parse_output_detects_environment_faults() -> None:
    assert parse_output("", "bash: pytest: command not found")["env_marker"] == "command_not_found"
    assert parse_output("", "Permission denied")["env_marker"] == "permission_denied"


def test_parse_output_detects_missing_module() -> None:
    parsed = parse_output("", "ModuleNotFoundError: No module named 'numpy'")
    assert parsed["missing_module"] == "numpy"


def test_fingerprint_collides_for_identical_failures() -> None:
    """Two runs of the same failure must hash identically, or the
    no-progress detector never fires."""
    a = fingerprint(
        exit_code=1,
        failing_tests=(),
        stdout="error at /tmp/pytest-of-x/run0/thing.py line 42 in 0.31s",
        stderr="",
    )
    b = fingerprint(
        exit_code=1,
        failing_tests=(),
        stdout="error at /tmp/pytest-of-x/run9/thing.py line 88 in 1.42s",
        stderr="",
    )
    assert a == b, "volatile paths/durations/line numbers must be normalized away"


def test_fingerprint_differs_for_different_failures() -> None:
    a = fingerprint(exit_code=1, failing_tests=("t::a",), stdout="", stderr="")
    b = fingerprint(exit_code=1, failing_tests=("t::b",), stdout="", stderr="")
    assert a != b


# ================================================================ diagnosis


@pytest.mark.parametrize(
    ("signals", "expected", "rule"),
    [
        (_sig(interrupt_kind="budget_ceiling"), FailureClass.INTERRUPTED, "R0.interrupt"),
        (
            _sig(tool_error_kinds=("PermissionError",)),
            FailureClass.HARNESS_FAULT,
            "R1.tool_errors",
        ),
        (_sig(baseline_probe="fail_same"), FailureClass.BAD_VERIFICATION, "R2.baseline_fails_same"),
        (_sig(env_marker="command_not_found"), FailureClass.ENV_BROKEN, "R4.env_marker"),
        (_sig(missing_module="numpy"), FailureClass.MISSING_DEPENDENCY, "R5.missing_module"),
        (
            _sig(empty_diff=True, tool_calls=0),
            FailureClass.UNDER_SPECIFIED_STEP,
            "R6.empty_diff",
        ),
        (_sig(stopped_reason="max_tokens"), FailureClass.STEP_TOO_LARGE, "R7.exhaustion"),
        (
            _sig(
                baseline_probe="pass",
                failing_tests=("other/test_z.py::t",),
                changed_files=("src/a.py",),
                in_blast_radius=False,
            ),
            FailureClass.OUT_OF_SCOPE_REGRESSION,
            "R8.out_of_scope",
        ),
        (
            _sig(baseline_probe="pass", progress_delta=3),
            FailureClass.IMPLEMENTATION_BUG,
            "R9.progress",
        ),
    ],
)
def test_rule_table_names_the_fault(signals, expected, rule) -> None:
    diagnosis = diagnose(signals, _hist())
    assert diagnosis.failure_class is expected
    assert diagnosis.top.rule_id == rule
    assert diagnosis.cost_usd == 0.0, "diagnosis must be free"


def test_first_match_wins_orders_by_certainty() -> None:
    """A structural certainty must not be shadowed by a weaker heuristic."""
    signals = _sig(
        interrupt_kind="budget_ceiling",
        env_marker="command_not_found",
        missing_module="numpy",
        empty_diff=True,
    )
    assert diagnose(signals, _hist()).failure_class is FailureClass.INTERRUPTED


def test_unmatched_failure_falls_back_to_residual() -> None:
    diagnosis = diagnose(_sig(baseline_probe="skipped"), _hist())
    assert diagnosis.failure_class is FailureClass.IMPLEMENTATION_BUG
    assert diagnosis.top.rule_id == "residual"
    assert diagnosis.confidence < 0.5


def test_broken_rule_predicate_cannot_break_the_run() -> None:
    from taste import recovery

    bad = recovery.Rule("R.bad", FailureClass.FLAKY, 1.0, lambda s, h: 1 / 0)
    original = recovery.RULES
    try:
        recovery.RULES = (bad, *original)
        assert diagnose(_sig(missing_module="numpy"), _hist()).failure_class is (
            FailureClass.MISSING_DEPENDENCY
        )
    finally:
        recovery.RULES = original


def test_blast_radius_recognises_a_touched_file() -> None:
    signals = observe(
        verdict=MonitorResult(passed=False, reason="`pytest` exited 1",
                              evidence="FAILED tests/test_math.py::test_add\n1 failed"),
        worker=SimpleNamespace(tool_calls=2, stopped_reason="end_turn"),
        changed_files=("math_utils.py",),
    )
    # test_math.py <-> math_utils.py: the failure is in this step's own work.
    assert signals.in_blast_radius is True


def test_blast_radius_flags_an_untouched_file() -> None:
    signals = observe(
        verdict=MonitorResult(passed=False, reason="`pytest` exited 1",
                              evidence="FAILED tests/test_network.py::test_socket\n1 failed"),
        worker=SimpleNamespace(tool_calls=2, stopped_reason="end_turn"),
        changed_files=("docs/readme.md",),
    )
    assert signals.in_blast_radius is False


# ================================================================ policies


def test_fixed_policy_ignores_the_diagnosis() -> None:
    """That is the point: it reproduces one behavior for a controlled arm."""
    policy = FixedPolicy(ActionKind.ROLLBACK_AND_RETRY)
    for cls in FailureClass:
        action = policy.decide(
            diagnosis=_diag(cls),
            history=_hist(),
            budget=Budget(),
            config=RecoveryConfig(enabled=True),
        )
        assert action.kind is ActionKind.ROLLBACK_AND_RETRY
        assert action.resets is True


def test_tiered_policy_halts_on_broken_verification() -> None:
    """Retrying cannot fix a check that was already failing."""
    action = TieredPolicy().decide(
        diagnosis=_diag(FailureClass.BAD_VERIFICATION, 1.0),
        history=_hist(),
        budget=Budget(),
        config=RecoveryConfig(enabled=True, policy="tiered"),
    )
    assert action.kind is ActionKind.HALT
    assert "already failing" in action.reason


def test_tiered_policy_never_gives_guidance_for_our_own_bug() -> None:
    for cls in (FailureClass.HARNESS_FAULT, FailureClass.INTERRUPTED):
        action = TieredPolicy().decide(
            diagnosis=_diag(cls, 1.0),
            history=_hist(),
            budget=Budget(),
            config=RecoveryConfig(enabled=True, policy="tiered"),
        )
        assert action.kind is ActionKind.HALT
        assert action.guidance is None


def test_tiered_policy_repairs_in_place_when_converging() -> None:
    """Fewer failures than last time is evidence the work is worth keeping."""
    action = TieredPolicy().decide(
        diagnosis=_diag(FailureClass.IMPLEMENTATION_BUG, 0.8),
        history=_hist(),
        budget=Budget(),
        config=RecoveryConfig(enabled=True, policy="tiered"),
        signals=_sig(progress_delta=4),
    )
    assert action.kind is ActionKind.REPAIR_IN_PLACE
    assert action.resets is False


def test_tiered_policy_rolls_back_an_out_of_scope_regression() -> None:
    action = TieredPolicy().decide(
        diagnosis=_diag(FailureClass.OUT_OF_SCOPE_REGRESSION, 0.85),
        history=_hist(),
        budget=Budget(),
        config=RecoveryConfig(enabled=True, policy="tiered"),
    )
    assert action.kind is ActionKind.ROLLBACK_AND_RETRY
    assert action.resets is True


def test_tiered_policy_retries_without_reset_when_nothing_was_written() -> None:
    """There is nothing to reset, so resetting would be theatre."""
    action = TieredPolicy().decide(
        diagnosis=_diag(FailureClass.UNDER_SPECIFIED_STEP, 0.9),
        history=_hist(),
        budget=Budget(),
        config=RecoveryConfig(enabled=True, policy="tiered"),
    )
    assert action.kind is ActionKind.RETRY_WITH_GUIDANCE
    assert action.resets is False


def test_tiered_policy_names_the_missing_package() -> None:
    action = TieredPolicy().decide(
        diagnosis=_diag(FailureClass.MISSING_DEPENDENCY, 0.9),
        history=_hist(),
        budget=Budget(),
        config=RecoveryConfig(enabled=True, policy="tiered"),
        signals=_sig(missing_module="scipy"),
    )
    assert action.kind is ActionKind.RETRY_WITH_GUIDANCE
    assert "scipy" in (action.guidance or "")


def test_tiered_falls_back_to_baseline_when_uncertain() -> None:
    """The safety invariant: a bad diagnosis can never beat doing nothing."""
    tiered = TieredPolicy().decide(
        diagnosis=_diag(FailureClass.UNKNOWN, 0.1),
        history=_hist(),
        budget=Budget(),
        config=RecoveryConfig(enabled=True, policy="tiered"),
    )
    baseline = FixedPolicy(ActionKind.ROLLBACK_AND_RETRY).decide(
        diagnosis=_diag(FailureClass.UNKNOWN, 0.1),
        history=_hist(),
        budget=Budget(),
        config=RecoveryConfig(enabled=True),
    )
    assert tiered.kind is baseline.kind
    assert tiered.resets == baseline.resets


# ================================================================ loop prevention


def test_repeated_identical_failures_halt() -> None:
    from taste.recovery import AttemptRecord

    history = _hist()
    for i in (1, 2):
        history.record(
            AttemptRecord(i, "same-fp", FailureClass.IMPLEMENTATION_BUG, ActionKind.ROLLBACK_AND_RETRY)
        )
    assert history.no_progress_streak() >= 2

    action = TieredPolicy().decide(
        diagnosis=_diag(FailureClass.IMPLEMENTATION_BUG, 0.8),
        history=history,
        budget=Budget(),
        config=RecoveryConfig(enabled=True, policy="tiered"),
    )
    assert action.kind is ActionKind.HALT
    assert "no progress" in action.reason


def test_action_cap_stops_one_verb_repeating_forever() -> None:
    history = _hist()
    history.action_counts[ActionKind.ROLLBACK_AND_RETRY] = 3
    action = TieredPolicy().decide(
        diagnosis=_diag(FailureClass.IMPLEMENTATION_BUG, 0.75),
        history=history,
        budget=Budget(),
        config=RecoveryConfig(enabled=True, policy="tiered"),
    )
    assert action.kind is ActionKind.HALT
    assert "cap" in action.reason


def test_exhausted_budget_halts() -> None:
    action = TieredPolicy().decide(
        diagnosis=_diag(FailureClass.IMPLEMENTATION_BUG, 0.8),
        history=_hist(),
        budget=Budget(attempts_used=3, max_attempts=3),
        config=RecoveryConfig(enabled=True, policy="tiered"),
    )
    assert action.kind is ActionKind.HALT


# ================================================================ guidance book


def test_guidance_book_deduplicates_by_fingerprint() -> None:
    book = GuidanceBook()
    book.add("fp1", "the same failure")
    book.add("fp1", "the same failure")
    book.add("fp2", "a different failure")
    assert len(book.items) == 2


def test_guidance_book_is_capped() -> None:
    """Append-only, never rewritten — so it cannot collapse, only evict."""
    book = GuidanceBook(cap=3)
    for i in range(6):
        book.add(f"fp{i}", f"failure {i}")
    assert len(book.items) == 3
    assert "failure 5" in book.render()
    assert "failure 0" not in book.render()


# ================================================================ arms


def test_arms_are_configurations_not_code_paths() -> None:
    assert RecoveryConfig.arm("A0").fixed_action is ActionKind.ACCEPT
    assert RecoveryConfig.arm("A2").fixed_action is ActionKind.REPAIR_IN_PLACE
    assert RecoveryConfig.arm("A3").fixed_action is ActionKind.ROLLBACK_AND_RETRY
    assert RecoveryConfig.arm("A3prime").fixed_action is ActionKind.RETRY_WITH_GUIDANCE
    assert RecoveryConfig.arm("tiered").policy == "tiered"
    with pytest.raises(ValueError, match="unknown arm"):
        RecoveryConfig.arm("A99")


def _failing_scenario(ws: Path):
    plan = Plan(
        task="arms",
        steps=[Step("step-01", "make it pass", Verification(kind="shell", command="test -f done.py"))],
    )

    def worker(step, plan_):
        (ws / "attempted.py").write_text("# tried\n")
        return WorkerResult("tried", 1, "end_turn")

    return plan, worker


def test_arm_A0_accepts_the_failure_without_retrying(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    plan, worker = _failing_scenario(ws)
    result = Kernel(workspace=ws, recovery_config=RecoveryConfig.arm("A0")).run(
        task="a1", spec=_spec(), session_id="a1", plan_override=plan, worker_override=worker
    )
    assert result.outcomes[0].attempts == 1, "A1 must never retry"
    assert result.outcomes[0].rolled_back is False


def test_arm_A3_and_A3prime_are_attempt_matched(refactor_workspace: Path, tmp_path_factory) -> None:
    """The control that separates 'reset helps' from 'more attempts help'.

    Both arms must burn identical attempts; only the reset differs.
    """
    from examples.refactor_demo.bootstrap import bootstrap

    ws_a = refactor_workspace
    ws_b = tmp_path_factory.mktemp("a3prime")
    bootstrap(ws_b)

    plan_a, worker_a = _failing_scenario(ws_a)
    a3 = Kernel(workspace=ws_a, recovery_config=RecoveryConfig.arm("A3")).run(
        task="a3", spec=_spec(), session_id="a3", plan_override=plan_a, worker_override=worker_a
    )
    plan_b, worker_b = _failing_scenario(ws_b)
    a3p = Kernel(workspace=ws_b, recovery_config=RecoveryConfig.arm("A3prime")).run(
        task="a3p", spec=_spec(), session_id="a3p", plan_override=plan_b, worker_override=worker_b
    )

    assert a3.outcomes[0].attempts == a3p.outcomes[0].attempts
    assert a3.outcomes[0].rolled_back is True
    assert a3p.outcomes[0].rolled_back is False
    # A3 discarded the work; A3' kept it. That is the whole difference.
    assert not (ws_a / "attempted.py").exists()
    assert (ws_b / "attempted.py").exists()


def test_recovery_events_narrate_the_decision(refactor_workspace: Path) -> None:
    """You can watch the trap handler work in the event stream."""
    ws = refactor_workspace
    plan, worker = _failing_scenario(ws)
    events = []
    Kernel(
        workspace=ws,
        on_event=events.append,
        recovery_config=RecoveryConfig.arm("tiered"),
    ).run(task="t", spec=_spec(), session_id="t", plan_override=plan, worker_override=worker)

    diagnoses = [e for e in events if e.kind == "recovery.diagnosis"]
    actions = [e for e in events if e.kind == "recovery.action"]
    assert diagnoses and actions
    assert diagnoses[0].payload["failure_class"]
    assert diagnoses[0].payload["rule"]
    assert actions[0].payload["action"] in {a.value for a in ActionKind}


def _spec():
    from taste.agent import AgentSpec

    return AgentSpec(name="scripted", description="", system_prompt="p")


# ================================================================ baseline probe


def test_probe_detects_a_verification_that_was_already_failing(refactor_workspace: Path) -> None:
    """The cheapest useful question in the system, and it costs $0.

    If the check already failed at the pre-step commit, the worker cannot be
    at fault and no amount of retrying can help.
    """
    ws = refactor_workspace
    # Break the suite BEFORE the run starts, and commit it.
    (ws / "test_broken.py").write_text("def test_impossible():\n    assert False\n")
    import subprocess

    subprocess.run(["git", "add", "-A"], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "pre-broken"], cwd=ws, check=True, capture_output=True)

    plan = Plan(
        task="probe",
        steps=[Step("step-01", "unrelated edit", Verification(kind="shell", command="pytest -q"))],
    )

    def worker(step, plan_):
        (ws / "unrelated.py").write_text("# harmless\n")
        return WorkerResult("did something harmless", 1, "end_turn")

    events = []
    result = Kernel(
        workspace=ws,
        on_event=events.append,
        recovery_config=RecoveryConfig.arm("tiered"),
    ).run(task="p", spec=_spec(), session_id="p", plan_override=plan, worker_override=worker)

    probes = [e for e in events if e.kind == "recovery.probe"]
    assert probes and probes[0].payload["result"] == "fail_same"

    diagnoses = [e for e in events if e.kind == "recovery.diagnosis"]
    assert diagnoses[0].payload["failure_class"] == FailureClass.BAD_VERIFICATION.value

    # And crucially: it stopped instead of burning retries on an impossible step.
    assert result.outcomes[0].attempts == 1
    actions = [e for e in events if e.kind == "recovery.action"]
    assert actions[0].payload["action"] == ActionKind.HALT.value


def test_probe_runs_once_per_step_not_once_per_attempt(refactor_workspace: Path) -> None:
    """The pre-step tree cannot change, so the answer cannot either."""
    ws = refactor_workspace
    plan = Plan(
        task="probe-once",
        steps=[Step("step-01", "make it", Verification(kind="shell", command="test -f never.py"))],
    )

    def worker(step, plan_):
        (ws / "something.py").write_text("# nope\n")
        return WorkerResult("tried", 1, "end_turn")

    events = []
    Kernel(
        workspace=ws,
        max_retries=2,
        on_event=events.append,
        recovery_config=RecoveryConfig.arm("tiered"),
    ).run(task="p", spec=_spec(), session_id="p2", plan_override=plan, worker_override=worker)

    assert len([e for e in events if e.kind == "recovery.probe"]) == 1


def test_probe_leaves_no_worktree_behind(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    from taste.memory import Memory

    memory = Memory.open_session(ws, "probe-clean")
    head = memory.head()
    with memory.probe_worktree(head.sha) as path:
        assert path.exists()
        probe_path = path
    assert not probe_path.exists()
    # git must not still consider it registered, or a later prune could
    # collide with a live worktree. Compare exact paths — the pytest tmpdir
    # is itself named after this test, so substring matching is meaningless.
    registered = {
        line.split()[0] for line in memory.repo.git.worktree("list").splitlines() if line
    }
    assert str(probe_path) not in registered


def test_probe_failure_is_never_fatal(refactor_workspace: Path, monkeypatch) -> None:
    """A probe that cannot run tells us nothing; it must not break the step."""
    from taste import recovery as rec
    from taste.memory import Memory

    def explode(*a, **k):
        raise RuntimeError("no worktrees today")

    monkeypatch.setattr(Memory, "probe_worktree", explode)
    memory = Memory.open_session(refactor_workspace, "probe-boom")
    result, seconds = rec.run_baseline_probe(
        memory=memory,
        before=memory.head(),
        command="true",
        failure_fingerprint="x",
    )
    assert result == "skipped"
    assert seconds >= 0


def test_probe_reports_pass_when_the_check_was_healthy(refactor_workspace: Path) -> None:
    from taste import recovery as rec
    from taste.memory import Memory

    memory = Memory.open_session(refactor_workspace, "probe-ok")
    result, _ = rec.run_baseline_probe(
        memory=memory,
        before=memory.head(),
        command="true",
        failure_fingerprint="x",
    )
    assert result == "pass"


# ================================================================ review fixes


def test_reverify_does_not_re_run_the_worker(refactor_workspace: Path) -> None:
    """REVERIFY re-runs the check, not the step.

    Re-invoking the worker would spend API budget, produce new edits, and
    break the attempt-matching that A3 vs A3' depends on.
    """
    ws = refactor_workspace
    (ws / "flaky_marker.txt").write_text("0\n")
    # A check that fails once then passes, without the tree changing.
    plan = Plan(
        task="flaky",
        steps=[
            Step(
                "step-01",
                "do it",
                Verification(
                    kind="shell",
                    command=(
                        "n=$(cat flaky_marker.txt); echo $((n+1)) > flaky_marker.txt; "
                        "test $n -ge 1"
                    ),
                ),
            )
        ],
    )
    calls = {"n": 0}

    def worker(step, plan_):
        calls["n"] += 1
        (ws / "made.py").write_text("# work\n")
        return WorkerResult("did it", 1, "end_turn")

    events = []
    from taste.recovery import RecoveryConfig as RC

    Kernel(
        workspace=ws,
        max_retries=2,
        on_event=events.append,
        recovery_config=RC(enabled=True, policy="tiered"),
    ).run(task="f", spec=_spec(), session_id="f", plan_override=plan, worker_override=worker)

    reverifies = [e for e in events if e.kind == "recovery.action"
                  and e.payload["action"] == ActionKind.REVERIFY.value]
    if reverifies:
        worker_runs = len([e for e in events if e.kind == "worker.done"])
        step_begins = len([e for e in events if e.kind == "step.begin"])
        assert calls["n"] == worker_runs == step_begins, (
            "a reverify must not add a worker invocation"
        )


def test_diagnosis_sees_the_change_set_without_the_journal(refactor_workspace: Path) -> None:
    """Recovery must not depend on journalling being switched on.

    Gating the diff on the journal left every fault frame reporting an empty
    diff, which silently disabled the blast-radius and empty-diff rules.
    """
    ws = refactor_workspace
    plan = Plan(
        task="nojournal",
        steps=[Step("step-01", "edit", Verification(kind="shell", command="test -f nope.py"))],
    )

    def worker(step, plan_):
        (ws / "touched.py").write_text("# edited\n")
        return WorkerResult("edited", 1, "end_turn")

    events = []
    from taste.recovery import RecoveryConfig as RC

    Kernel(
        workspace=ws,
        max_retries=0,
        journal=False,
        on_event=events.append,
        recovery_config=RC(enabled=True, policy="tiered"),
    ).run(task="n", spec=_spec(), session_id="n", plan_override=plan, worker_override=worker)

    diagnoses = [e for e in events if e.kind == "recovery.diagnosis"]
    assert diagnoses
    # The worker DID change a file, so this must not read as an empty diff.
    assert diagnoses[0].payload["failure_class"] != FailureClass.UNDER_SPECIFIED_STEP.value


def test_no_progress_streak_counts_the_current_attempt() -> None:
    """Reading the streak before recording meant two identical failures
    needed a third attempt to be noticed."""
    from taste.recovery import AttemptRecord

    history = _hist()
    history.record(AttemptRecord(1, "same", FailureClass.UNKNOWN, ActionKind.HALT))
    history.amend_last(
        failure_class=FailureClass.IMPLEMENTATION_BUG, action=ActionKind.ROLLBACK_AND_RETRY
    )
    # One failure is not yet a streak.
    assert history.no_progress_streak() == 0

    # The second identical failure completes it — and is visible on the very
    # attempt it happens, because the record precedes the diagnosis.
    history.record(AttemptRecord(2, "same", FailureClass.UNKNOWN, ActionKind.HALT))
    assert history.no_progress_streak() == 2, "the current attempt must count"

    # A different failure breaks it: progress, not repetition.
    history.record(AttemptRecord(3, "different", FailureClass.UNKNOWN, ActionKind.HALT))
    assert history.no_progress_streak() == 1


def test_amend_last_tallies_actions_once() -> None:
    from taste.recovery import AttemptRecord

    history = _hist()
    history.record(AttemptRecord(1, "fp", FailureClass.UNKNOWN, ActionKind.HALT))
    history.amend_last(
        failure_class=FailureClass.IMPLEMENTATION_BUG, action=ActionKind.ROLLBACK_AND_RETRY
    )
    assert history.action_counts == {ActionKind.ROLLBACK_AND_RETRY: 1}
    assert history.reset_count == 1
    assert history.records[-1].failure_class is FailureClass.IMPLEMENTATION_BUG


def test_action_ceiling_agrees_with_the_budget() -> None:
    """L3 and Budget.exhausted must use the same comparison or they disagree
    about when a step is finished."""
    budget = Budget(actions_used=8, max_actions=8)
    assert budget.exhausted is True
    action = TieredPolicy().decide(
        diagnosis=_diag(FailureClass.IMPLEMENTATION_BUG, 0.75),
        history=_hist(),
        budget=budget,
        config=RecoveryConfig(enabled=True, policy="tiered"),
    )
    assert action.kind is ActionKind.HALT

    fine = Budget(actions_used=7, max_actions=8)
    assert fine.exhausted is False


def test_probe_and_failure_fingerprints_use_the_same_slice(refactor_workspace: Path) -> None:
    """A probe hashing full output while the verdict carries a 40-line tail
    could never produce fail_same for a non-pytest check."""
    from taste import recovery as rec
    from taste.memory import Memory

    ws = refactor_workspace
    noisy = "; ".join(f"echo line{i}" for i in range(60))
    (ws / "noisy.sh").write_text("#!/bin/sh\n" + noisy.replace("; ", "\n") + "\nexit 3\n")
    (ws / "noisy.sh").chmod(0o755)
    memory = Memory.open_session(ws, "fp")
    memory.checkpoint("base", "base")

    import subprocess

    proc = subprocess.run("./noisy.sh", shell=True, cwd=ws, capture_output=True, text=True)
    evidence = rec._tail_lines(proc.stdout, 40)
    failure_fp = rec.fingerprint(
        exit_code=proc.returncode, failing_tests=(), stdout=evidence, stderr=""
    )

    result, _ = rec.run_baseline_probe(
        memory=memory,
        before=memory.head(),
        command="./noisy.sh",
        failure_fingerprint=failure_fp,
    )
    assert result == "fail_same", "identical failing output must collide"
