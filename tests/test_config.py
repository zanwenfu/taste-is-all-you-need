"""One config object, one hash, one recoverable run identity.

The property that matters here is comparability: two runs with the same
config hash were the same harness, and two runs with different hashes are not
comparable — which the data should say before anyone plots them together.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taste.config import EVENT_OWNERS, ROLES, HarnessConfig, kernel_kwargs
from taste.cores import Plan, Step, Verification, WorkerResult
from taste.kernel import Kernel
from taste.recovery import ActionKind
from tests.golden import rollback_scenario
from tests.test_golden_baseline import EXPECTED_EVENTS


def _spec():
    from taste.agent import AgentSpec

    return AgentSpec(name="s", description="", system_prompt="p")


# ------------------------------------------------------------------ the floor


def test_default_config_is_every_subsystem_off() -> None:
    config = HarnessConfig()
    assert config.journal is False
    assert config.recovery.enabled is False
    assert config.guardrails.enabled is False
    assert config.two_phase_merge is False


def test_baseline_config_reproduces_the_frozen_signature(refactor_workspace: Path) -> None:
    """"Build to delete" as a tested property, at the top level."""
    sig = rollback_scenario(refactor_workspace).run(
        refactor_workspace, config=HarnessConfig.baseline()
    )
    assert sig.events == EXPECTED_EVENTS


# ------------------------------------------------------------------ identity


def test_hash_distinguishes_arms() -> None:
    hashes = {name: HarnessConfig.arm(name).hash() for name in HarnessConfig.arm_names()}
    assert len(set(hashes.values())) == len(hashes), f"arms collide: {hashes}"


def test_hash_ignores_the_label() -> None:
    """Renaming an arm must not make it look like a different harness."""
    a = HarnessConfig(label="one")
    b = HarnessConfig(label="two")
    assert a.hash() == b.hash()


def test_hash_is_stable_across_construction() -> None:
    assert HarnessConfig.arm("A3").hash() == HarnessConfig.arm("A3").hash()


def test_hash_changes_when_behavior_changes() -> None:
    base = HarnessConfig()
    assert base.hash() != HarnessConfig(journal=True).hash()
    assert base.hash() != HarnessConfig(max_retries=5).hash()
    assert base.hash() != HarnessConfig(two_phase_merge=True).hash()


def test_unknown_arm_is_rejected_with_the_known_ones() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        HarnessConfig.arm("A99")


# ------------------------------------------------------------------ arms


def test_arms_differ_only_in_recovery_policy() -> None:
    """What makes an arm comparison a statement about recovery policy rather
    than about four different codebases."""
    arms = {n: HarnessConfig.arm(n) for n in ("A0", "A2", "A3", "A3prime")}
    for config in arms.values():
        assert config.journal is True
        assert config.guardrails.enabled is False
        assert config.two_phase_merge is False
        assert config.max_retries == 2

    actions = {n: c.recovery.fixed_action for n, c in arms.items()}
    assert actions == {
        "A0": ActionKind.ACCEPT,
        "A2": ActionKind.REPAIR_IN_PLACE,
        "A3": ActionKind.ROLLBACK_AND_RETRY,
        "A3prime": ActionKind.RETRY_WITH_GUIDANCE,
    }


def test_full_arm_turns_everything_on() -> None:
    full = HarnessConfig.arm("full")
    assert full.journal and full.two_phase_merge
    assert full.recovery.enabled and full.recovery.policy == "tiered"
    assert full.guardrails.enabled


def test_arm_accepts_overrides() -> None:
    config = HarnessConfig.arm("A3", max_retries=7)
    assert config.max_retries == 7
    assert config.recovery.fixed_action is ActionKind.ROLLBACK_AND_RETRY
    assert config.hash() != HarnessConfig.arm("A3").hash()


# ------------------------------------------------------------------ wiring


def test_config_drives_the_kernel(refactor_workspace: Path) -> None:
    kernel = Kernel(workspace=refactor_workspace, config=HarnessConfig.arm("full"))
    assert kernel.journal_enabled is True
    assert kernel.recovery_config.enabled is True
    assert kernel.guard_config.enabled is True
    assert kernel.two_phase_merge is True


def test_config_wins_over_loose_kwargs(refactor_workspace: Path) -> None:
    """A run's identity must not be half config and half stray argument."""
    kernel = Kernel(
        workspace=refactor_workspace,
        journal=False,
        config=HarnessConfig.arm("A3"),
    )
    assert kernel.journal_enabled is True


def test_kernel_kwargs_covers_every_switch() -> None:
    kwargs = kernel_kwargs(HarnessConfig.arm("full"))
    assert set(kwargs) == {
        "max_retries",
        "max_parallel",
        "planner_model",
        "journal",
        "recovery_config",
        "guard_config",
        "two_phase_merge",
        "observe_tools",
        "union_gate",
        "shadow",
    }


# ------------------------------------------------------------------ manifest


def test_manifest_records_the_arm(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    config = HarnessConfig.arm("A3prime")
    result = Kernel(workspace=ws, config=config).run(
        task="m",
        spec=_spec(),
        session_id="m",
        plan_override=Plan(
            task="m",
            steps=[Step("step-01", "x", Verification(kind="shell", command="true"))],
        ),
        worker_override=lambda s, p: WorkerResult("", 0, "end_turn"),
    )

    manifest = json.loads(
        (ws / ".git" / "taste" / f"manifest-{result.session_id}.json").read_text()
    )
    harness = manifest["harness"]
    assert harness["label"] == "A3prime-no-reset"
    assert harness["config_hash"] == config.hash()
    assert harness["recovery"]["fixed_action"] == ActionKind.RETRY_WITH_GUIDANCE.value


def test_a_run_is_groupable_from_its_manifest_alone(refactor_workspace: Path, tmp_path_factory) -> None:
    """Two runs of the same arm must be recognisable as such after the fact."""
    from examples.refactor_demo.bootstrap import bootstrap

    second = bootstrap(tmp_path_factory.mktemp("second") / "ws")
    plan = Plan(
        task="g",
        steps=[Step("step-01", "x", Verification(kind="shell", command="true"))],
    )
    worker = lambda s, p: WorkerResult("", 0, "end_turn")  # noqa: E731

    hashes = []
    for ws in (refactor_workspace, second):
        result = Kernel(workspace=ws, config=HarnessConfig.arm("tiered")).run(
            task="g", spec=_spec(), session_id="g", plan_override=plan, worker_override=worker
        )
        manifest = json.loads(
            (ws / ".git" / "taste" / f"manifest-{result.session_id}.json").read_text()
        )
        hashes.append(manifest["harness"]["config_hash"])

    assert hashes[0] == hashes[1]


def test_adhoc_kernel_is_labelled_as_such(refactor_workspace: Path) -> None:
    """A run built from loose kwargs is honestly marked, not given a fake id."""
    ws = refactor_workspace
    result = Kernel(workspace=ws, journal=True).run(
        task="a",
        spec=_spec(),
        session_id="a",
        plan_override=Plan(
            task="a",
            steps=[Step("step-01", "x", Verification(kind="shell", command="true"))],
        ),
        worker_override=lambda s, p: WorkerResult("", 0, "end_turn"),
    )
    manifest = json.loads(
        (ws / ".git" / "taste" / f"manifest-{result.session_id}.json").read_text()
    )
    assert manifest["harness"]["label"] == "adhoc"
    assert manifest["harness"]["config_hash"] is None


# ------------------------------------------------------------------ registries


def test_every_emitted_event_prefix_has_an_owner(refactor_workspace: Path) -> None:
    """A prefix with no owner is a subsystem that skipped the registry."""
    scenario = rollback_scenario(refactor_workspace)
    scenario.run(refactor_workspace, config=HarnessConfig.arm("full"))

    prefixes = {e.kind.split(".", 1)[0] for e in scenario.events}
    assert prefixes <= set(EVENT_OWNERS), f"unowned: {prefixes - set(EVENT_OWNERS)}"


def test_roles_used_by_the_cores_are_registered() -> None:
    assert {"planner", "worker", "monitor"} <= ROLES
