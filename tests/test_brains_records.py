"""The central/worker boundary consists of durable, strict value records."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from taste.brains.contract import Contract
from taste.brains.records import (
    ArtifactRef,
    ArtifactSpec,
    Assignment,
    CriteriaRevision,
    Criterion,
    LifecycleEvent,
    PlanRevision,
    WorkerReport,
    contract_digest,
)

NOW = "2026-09-10T12:00:00+00:00"
STATE_INPUT = "1" * 40
BASE_STATE = "2" * 40
CENTRAL_STATE = "3" * 40
PRODUCT_STATE = "4" * 40
WORKER_STATE = "5" * 40
FINAL_STATE = "6" * 40
OTHER_STATE = "7" * 40
GRAMMAR_BLOB = "a" * 40
PARSER_BLOB = "b" * 64


def contract(identity: str = "worker-1", **changes) -> Contract:
    values = {
        "identity": identity,
        "task": "build the parser",
        "inputs": ("grammar",),
        "outputs": ("parser.py",),
        "success_criteria": ("the parser tests pass",),
        "budget_usd": 2.5,
        "max_turns": 20,
    }
    return Contract(**{**values, **changes})


def ref(
    artifact_id: str = "grammar",
    *,
    branch: str = "producer",
    state_id: str = STATE_INPUT,
    path: str = "artifacts/grammar.json",
    blob_id: str = GRAMMAR_BLOB,
) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        branch=branch,
        state_id=state_id,
        path=path,
        blob_id=blob_id,
        kind="record",
        metadata={"quality": {"checked": True}},
    )


def spec(artifact_id: str = "parser", path: str = "parser.py") -> ArtifactSpec:
    return ArtifactSpec(
        artifact_id=artifact_id,
        path=path,
        description="working parser",
        metadata={"reviewers": ["worker", "monitor"]},
    )


def assignment(
    assignment_id: str = "assignment-1",
    *,
    identity: str = "worker-1",
    generation: int = 1,
    depends_on: tuple[str, ...] = (),
) -> Assignment:
    brief = contract(identity)
    return Assignment(
        assignment_id=assignment_id,
        generation=generation,
        attempt=0,
        contract=brief,
        contract_digest=contract_digest(brief),
        base_state_id=BASE_STATE,
        depends_on=depends_on,
        inputs=(ref(),),
        outputs=(spec(),),
        resources={"memory_mb": 1024, "labels": ["coding"]},
        metadata={"planner": {"confidence": 0.8}},
    )


def plan(*assignments: Assignment, **changes) -> PlanRevision:
    values = {
        "plan_id": "plan-1",
        "generation": 1,
        "goal_id": "goal-1",
        "based_on_state_id": CENTRAL_STATE,
        "observed_heads": {"integrated": PRODUCT_STATE, "worker-1": WORKER_STATE},
        "assignments": assignments or (assignment(),),
        "created_at": NOW,
        "rationale": "split the parser from integration",
        "metadata": {"trigger": "initial"},
    }
    return PlanRevision(**{**values, **changes})


def event(**changes) -> LifecycleEvent:
    values = {
        "event_id": "event-1",
        "run_id": "run-1",
        "assignment_id": "assignment-1",
        "worker": "worker-1",
        "generation": 1,
        "attempt": 0,
        "kind": "worker_started",
        "at": NOW,
        "observed_state_id": WORKER_STATE,
        "pid": 101,
        "process_group_id": 101,
        "metadata": {"host": "local"},
    }
    return LifecycleEvent(**{**values, **changes})


def report(**changes) -> WorkerReport:
    brief = contract()
    output = ArtifactRef(
        artifact_id="parser",
        branch="worker-1",
        state_id=FINAL_STATE,
        path="parser.py",
        blob_id=PARSER_BLOB,
    )
    values = {
        "report_id": "report-1",
        "run_id": "run-1",
        "assignment_id": "assignment-1",
        "worker": "worker-1",
        "generation": 1,
        "attempt": 0,
        "contract_digest": contract_digest(brief),
        "base_state_id": BASE_STATE,
        "final_state_id": FINAL_STATE,
        "at": NOW,
        "completed": True,
        "terminal_reason": "completed",
        "outputs": (output,),
        "turns": 12,
        "cost_usd": 0.42,
        "monitor_severity": "fine",
        "summary": "parser and tests complete",
        "metadata": {"checks": ["pytest"]},
    }
    return WorkerReport(**{**values, **changes})


def criterion(**changes) -> Criterion:
    values = {"goal_id": "goal-1", "text": "the parser tests pass", "parent_id": None}
    return Criterion.derive(**{**values, **changes})


def criteria_revision(**changes) -> CriteriaRevision:
    base = CriteriaRevision(
        revision_id="criteria.goal-1.0",
        goal_id="goal-1",
        sequence=0,
        criteria=(criterion(),),
        reason="derived from the goal",
        at=NOW,
    )
    if not changes:
        return base
    return base.extend(
        (criterion(text="the changelog names the field"),),
        reason=changes.get("reason", "changelog"),
        at=NOW,
    )


@pytest.mark.parametrize(
    "record",
    [
        spec(),
        ref(),
        assignment(),
        plan(),
        event(),
        report(),
        criterion(),
        criteria_revision(),
        criteria_revision(reason="changelog"),
    ],
    ids=lambda value: type(value).__name__,
)
def test_every_record_round_trips_exactly(record) -> None:
    restored = type(record).from_json(record.to_json())
    assert restored == record
    assert restored.to_json() == record.to_json()
    assert json.loads(record.to_json())["schema"] == record.SCHEMA


def test_contract_digest_is_canonical_and_sensitive_to_the_brief() -> None:
    first = contract()
    same = Contract.from_dict(first.to_dict())
    changed = contract(task="build a lexer")
    assert contract_digest(first) == contract_digest(same)
    assert contract_digest(first).startswith("sha256:")
    assert contract_digest(first) != contract_digest(changed)


@pytest.mark.parametrize("width", [40, 64])
def test_object_ids_accept_only_supported_full_hash_widths(width: int) -> None:
    item = ArtifactRef(
        artifact_id="source",
        branch="producer",
        state_id="1" * width,
        path="source.txt",
        blob_id="a" * width,
    )
    assert len(item.state_id) == width
    assert len(item.blob_id) == width


@pytest.mark.parametrize(
    ("field_name", "build"),
    [
        ("ArtifactRef.state_id", lambda value: ref(state_id=value)),
        ("ArtifactRef.blob_id", lambda value: ref(blob_id=value)),
        (
            "Assignment.base_state_id",
            lambda value: replace(assignment(), base_state_id=value),
        ),
        (
            "PlanRevision.based_on_state_id",
            lambda value: plan(based_on_state_id=value),
        ),
        (
            "PlanRevision.observed_heads",
            lambda value: plan(observed_heads={"integrated": value}),
        ),
        (
            "LifecycleEvent.observed_state_id",
            lambda value: event(observed_state_id=value),
        ),
        (
            "WorkerReport.base_state_id",
            lambda value: report(base_state_id=value),
        ),
        (
            "WorkerReport.final_state_id",
            lambda value: report(final_state_id=value),
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
@pytest.mark.parametrize(
    "bad_id",
    [
        pytest.param("HEAD", id="symbolic-HEAD"),
        pytest.param("a" * 12, id="abbreviated-hash"),
        pytest.param("A" * 40, id="uppercase-hash"),
        pytest.param("g" * 40, id="non-hex"),
        pytest.param("a" * 39, id="sha1-too-short"),
        pytest.param("a" * 41, id="sha1-too-long"),
        pytest.param("a" * 63, id="sha256-too-short"),
        pytest.param("a" * 65, id="sha256-too-long"),
    ],
)
def test_every_object_id_field_rejects_non_exact_ids(
    field_name: str, build, bad_id: str
) -> None:
    with pytest.raises(ValueError, match="full lowercase 40- or 64-hex object ID"):
        build(bad_id)


def test_non_object_stable_ids_remain_symbolic() -> None:
    item = assignment("assignment.parser-v1")
    revision = plan(item, plan_id="plan.parser-v1", goal_id="goal.parser")
    lifecycle = event(event_id="event.started", run_id="run.parser")

    assert item.assignment_id == "assignment.parser-v1"
    assert item.inputs[0].artifact_id == "grammar"
    assert revision.plan_id == "plan.parser-v1"
    assert revision.goal_id == "goal.parser"
    assert lifecycle.event_id == "event.started"
    assert lifecycle.run_id == "run.parser"


def test_an_integer_budget_does_not_change_digest_on_wire_round_trip() -> None:
    brief = contract(budget_usd=2)
    item = Assignment(
        assignment_id="integer-budget",
        generation=1,
        attempt=0,
        contract=brief,
        contract_digest=contract_digest(brief),
        base_state_id=BASE_STATE,
    )
    restored = Assignment.from_json(item.to_json())
    assert restored.contract_digest == item.contract_digest
    assert restored.to_json() == item.to_json()


def test_assignment_refuses_a_contract_digest_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        Assignment(
            assignment_id="a",
            generation=1,
            attempt=0,
            contract=contract(),
            contract_digest="sha256:" + "0" * 64,
            base_state_id=BASE_STATE,
        )


@pytest.mark.parametrize("assignment_generation", [1, 3])
def test_plan_refuses_assignments_from_another_generation(
    assignment_generation: int,
) -> None:
    with pytest.raises(ValueError, match="exact plan generation"):
        plan(
            assignment(generation=assignment_generation),
            generation=2,
        )


def test_records_and_their_extension_maps_are_really_immutable() -> None:
    item = assignment()
    with pytest.raises(FrozenInstanceError):
        item.model = "another"  # type: ignore[misc]
    with pytest.raises(TypeError):
        item.resources["memory_mb"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        item.metadata["planner"]["confidence"] = 0  # type: ignore[index]
    assert item.resources["labels"] == ("coding",)


@pytest.mark.parametrize("path", ["/absolute", "../escape", "a/../b", "a//b", "a\\b", "./x"])
def test_artifact_paths_are_normalized_and_relative(path: str) -> None:
    with pytest.raises(ValueError, match="path"):
        spec(path=path)


def test_artifact_spec_can_explicitly_describe_a_deletion() -> None:
    item = ArtifactSpec(artifact_id="old-cache", path="cache.bin", disposition="absent")
    assert ArtifactSpec.from_json(item.to_json()).disposition == "absent"


@pytest.mark.parametrize("bad", [{"a": object()}, {1: "not a string key"}, {"x": float("nan")}])
def test_extension_maps_accept_only_real_json(bad) -> None:
    with pytest.raises(ValueError):
        ArtifactSpec(artifact_id="a", path="a", metadata=bad)


def test_wire_decoder_rejects_missing_unknown_and_duplicate_fields() -> None:
    raw = spec().to_dict()
    raw.pop("path")
    with pytest.raises(ValueError, match="missing required"):
        ArtifactSpec.from_dict(raw)

    raw = spec().to_dict()
    raw["future"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        ArtifactSpec.from_dict(raw)

    duplicate = spec().to_json().replace(
        '"artifact_id": "parser",',
        '"artifact_id": "parser",\n "artifact_id": "other",',
    )
    with pytest.raises(ValueError, match="duplicate key"):
        ArtifactSpec.from_json(duplicate)


def test_wire_decoder_does_not_coerce_wrong_scalar_or_array_types() -> None:
    raw = assignment().to_dict()
    raw["generation"] = "1"
    with pytest.raises(ValueError, match="integer"):
        Assignment.from_dict(raw)

    raw = assignment().to_dict()
    raw["depends_on"] = "assignment-0"
    with pytest.raises(ValueError, match="JSON array"):
        Assignment.from_dict(raw)


def test_assignment_rejects_duplicate_artifacts_and_self_dependency() -> None:
    brief = contract()
    base = dict(
        assignment_id="a",
        generation=1,
        attempt=0,
        contract=brief,
        contract_digest=contract_digest(brief),
        base_state_id=BASE_STATE,
    )
    with pytest.raises(ValueError, match="depend on itself"):
        Assignment(**base, depends_on=("a",))
    with pytest.raises(ValueError, match="duplicate artifact_id"):
        Assignment(**base, inputs=(ref(), ref()))
    with pytest.raises(ValueError, match="duplicate paths"):
        Assignment(**base, outputs=(spec("a", "same.py"), spec("b", "same.py")))


def test_assignment_rejects_invalid_contract_resource_bounds() -> None:
    bad = contract(max_turns=-1)
    with pytest.raises(ValueError, match="max_turns"):
        contract_digest(bad)


def test_plan_requires_known_acyclic_dependencies_and_unique_workers() -> None:
    unknown = assignment(depends_on=("missing",))
    with pytest.raises(ValueError, match="unknown dependencies"):
        plan(unknown)

    a = assignment("a", identity="worker-a", depends_on=("b",))
    b = assignment("b", identity="worker-b", depends_on=("a",))
    with pytest.raises(ValueError, match="cycle"):
        plan(a, b)

    one = assignment("a", identity="worker-a")
    two = assignment("b", identity="worker-a")
    with pytest.raises(ValueError, match="duplicate worker"):
        plan(one, two)


def test_plan_completion_is_an_explicit_paired_claim() -> None:
    with pytest.raises(ValueError, match="set together"):
        plan(complete=True)
    done = plan(complete=True, completion_reason="all required artifacts were integrated")
    assert done.complete


def test_plan_is_fenced_to_exact_observed_heads() -> None:
    item = plan()
    with pytest.raises(TypeError):
        item.observed_heads["worker-1"] = "later"  # type: ignore[index]
    assert item.based_on_state_id == CENTRAL_STATE


def test_lifecycle_terminal_and_uncertainty_are_never_implicit() -> None:
    with pytest.raises(ValueError, match="terminal_reason"):
        event(terminal=True)
    with pytest.raises(ValueError, match="uncertainty_reasons"):
        event(uncertain=True)

    stopped = event(
        kind="worker_stopped",
        terminal=True,
        terminal_reason="signal",
        signal=9,
        uncertain=True,
        uncertainty_reasons=("tool outcome unknown",),
    )
    assert stopped.terminal and stopped.uncertain


def test_lifecycle_does_not_conflate_exit_and_signal() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        event(exit_code=1, signal=9)


def test_timestamps_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError, match="timezone"):
        event(at="2026-09-10T12:00:00")


def test_worker_report_preserves_unknown_cost_as_none_not_zero() -> None:
    item = report(
        completed=False,
        terminal_reason="killed",
        cost_usd=None,
        turns=None,
        uncertain=True,
        uncertainty_reasons=("cost after process death is unknown",),
    )
    restored = WorkerReport.from_json(item.to_json())
    assert restored.cost_usd is None and restored.turns is None
    assert restored.uncertain


def test_worker_outputs_must_point_to_its_exact_final_state() -> None:
    wrong_state = ArtifactRef(
        artifact_id="parser",
        branch="worker-1",
        state_id=OTHER_STATE,
        path="parser.py",
        blob_id=PARSER_BLOB,
    )
    with pytest.raises(ValueError, match="exact final state"):
        report(outputs=(wrong_state,))


def test_report_rejects_invalid_cost_and_monitor_severity() -> None:
    with pytest.raises(ValueError, match="finite"):
        report(cost_usd=float("nan"))
    with pytest.raises(ValueError, match="monitor_severity"):
        report(monitor_severity="probably-fine")
