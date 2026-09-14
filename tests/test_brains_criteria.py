"""Success criteria are append-only: refinement is allowed, regression is not.

A planner that may rewrite the criteria it is judged against can always reach
``complete`` by releasing the obligations it has not met.  Nothing lies; the
bar moves.  That is the failure these records exist to make unrepresentable.

The rule is monotone, not textual.  A refinement may only ever make
satisfaction harder -- every world failing the parent must still fail the
child -- so a refinement *names* its parent and the parent stays live.  There
is no operation that retires an obligation, which is why a mislabelled
regression buys nothing: the child may pass, the parent will not, and the
criterion is still unmet.

Ids are derived, never supplied.  A model choosing its own ids could "refine"
``c3`` while emitting a different ``c3``, and a retention check over ids would
pass on a set that no longer holds the original obligation.
"""

from __future__ import annotations

import json

import pytest

from taste.brains.central_planner import Goal
from taste.brains.records import CriteriaRevision, Criterion

NOW = "2026-09-14T12:00:00+00:00"
GOAL_ID = "goal-1"


def goal(**changes) -> Goal:
    values = {
        "goal_id": GOAL_ID,
        "task": "add a priority field",
        "success_criteria": ("all tests pass", "the API docs are updated"),
    }
    return Goal(**{**values, **changes})


# --------------------------------------------------------------- derivation


def test_sequence_zero_is_derived_verbatim_from_the_immutable_goal() -> None:
    revision = CriteriaRevision.genesis(goal(), at=NOW)
    assert revision.sequence == 0
    assert revision.goal_id == GOAL_ID
    assert tuple(c.text for c in revision.criteria) == goal().success_criteria
    assert all(c.parent_id is None for c in revision.criteria)
    assert revision.reason == "derived from the goal"


def test_the_same_goal_always_derives_the_same_criterion_ids() -> None:
    first = CriteriaRevision.genesis(goal(), at=NOW)
    second = CriteriaRevision.genesis(goal(), at="2027-01-01T00:00:00+00:00")
    assert [c.criterion_id for c in first.criteria] == [c.criterion_id for c in second.criteria]


def test_changed_text_cannot_keep_its_parents_identity() -> None:
    original = CriteriaRevision.genesis(goal(), at=NOW).criteria[0]
    altered = Criterion.derive(goal_id=GOAL_ID, text="the new tests pass", parent_id=None)
    assert altered.criterion_id != original.criterion_id


def test_a_criterion_id_is_not_model_supplied() -> None:
    with pytest.raises(TypeError):
        Criterion(criterion_id="c1", goal_id=GOAL_ID, text="all tests pass")  # type: ignore[call-arg]


# ------------------------------------------------------------ append-only


def test_a_refinement_keeps_its_parent_live() -> None:
    base = CriteriaRevision.genesis(goal(), at=NOW)
    parent = base.criteria[0]
    child = Criterion.derive(
        goal_id=GOAL_ID, text="`pytest -q` exits 0 with zero failures", parent_id=parent.criterion_id
    )
    revised = base.extend((child,), reason="name the exact command", at=NOW)
    ids = {c.criterion_id for c in revised.criteria}
    assert parent.criterion_id in ids
    assert child.criterion_id in ids
    assert revised.sequence == 1
    assert revised.parent_revision_id == base.revision_id


def test_dropping_a_prior_criterion_is_refused() -> None:
    base = CriteriaRevision.genesis(goal(), at=NOW)
    kept = base.criteria[:1]
    with pytest.raises(ValueError, match="append-only"):
        CriteriaRevision(
            revision_id=base.revision_id,
            goal_id=GOAL_ID,
            sequence=1,
            parent_revision_id=base.revision_id,
            criteria=kept,
            reason="the docs turned out to be unnecessary",
            at=NOW,
            retained_ids=tuple(c.criterion_id for c in base.criteria),
        )


def test_a_refinement_of_an_unknown_parent_is_refused() -> None:
    base = CriteriaRevision.genesis(goal(), at=NOW)
    orphan = Criterion.derive(goal_id=GOAL_ID, text="something else", parent_id="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="parent"):
        base.extend((orphan,), reason="orphaned refinement", at=NOW)


def test_a_criterion_from_another_goal_is_refused() -> None:
    base = CriteriaRevision.genesis(goal(), at=NOW)
    foreign = Criterion.derive(goal_id="goal-2", text="unrelated", parent_id=None)
    with pytest.raises(ValueError, match="goal"):
        base.extend((foreign,), reason="foreign criterion", at=NOW)


def test_adding_a_new_top_level_criterion_is_allowed() -> None:
    base = CriteriaRevision.genesis(goal(), at=NOW)
    added = Criterion.derive(goal_id=GOAL_ID, text="the changelog names the field", parent_id=None)
    revised = base.extend((added,), reason="the release needs a changelog entry", at=NOW)
    assert len(revised.criteria) == len(base.criteria) + 1


def test_a_revision_must_say_why() -> None:
    base = CriteriaRevision.genesis(goal(), at=NOW)
    added = Criterion.derive(goal_id=GOAL_ID, text="another", parent_id=None)
    with pytest.raises(ValueError, match="reason"):
        base.extend((added,), reason="   ", at=NOW)


def test_live_criteria_include_every_generation() -> None:
    base = CriteriaRevision.genesis(goal(), at=NOW)
    parent = base.criteria[0]
    child = Criterion.derive(goal_id=GOAL_ID, text="`pytest -q` exits 0", parent_id=parent.criterion_id)
    revised = base.extend((child,), reason="exact command", at=NOW)
    grandchild = Criterion.derive(
        goal_id=GOAL_ID, text="`pytest -q` exits 0 on 3.11 and 3.12", parent_id=child.criterion_id
    )
    final = revised.extend((grandchild,), reason="both interpreters", at=NOW)
    assert len(final.criteria) == 4
    assert final.sequence == 2


# ------------------------------------------------------------------- wire


def test_revision_round_trips_exactly() -> None:
    base = CriteriaRevision.genesis(goal(), at=NOW)
    added = Criterion.derive(goal_id=GOAL_ID, text="the changelog names the field", parent_id=None)
    revision = base.extend((added,), reason="changelog", at=NOW)
    restored = CriteriaRevision.from_json(revision.to_json())
    assert restored == revision
    assert restored.to_json() == revision.to_json()
    assert json.loads(revision.to_json())["schema"] == CriteriaRevision.SCHEMA


def test_wire_decoding_refuses_a_dropped_criterion() -> None:
    base = CriteriaRevision.genesis(goal(), at=NOW)
    added = Criterion.derive(goal_id=GOAL_ID, text="changelog", parent_id=None)
    revision = base.extend((added,), reason="changelog", at=NOW)
    raw = json.loads(revision.to_json())
    raw["criteria"] = raw["criteria"][:1]
    with pytest.raises(ValueError):
        CriteriaRevision.from_dict(raw)


def test_wire_decoding_refuses_a_forged_criterion_id() -> None:
    revision = CriteriaRevision.genesis(goal(), at=NOW)
    raw = json.loads(revision.to_json())
    raw["criteria"][0]["criterion_id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="derived"):
        CriteriaRevision.from_dict(raw)


def test_unknown_fields_are_refused() -> None:
    revision = CriteriaRevision.genesis(goal(), at=NOW)
    raw = json.loads(revision.to_json())
    raw["surprise"] = True
    with pytest.raises(ValueError, match="unknown"):
        CriteriaRevision.from_dict(raw)
