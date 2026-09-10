"""The brief a sub-brain is created with.

A contract is the one document the worker and its monitor both read, so a
defect here is a defect in both at once -- and because ``identity`` becomes a
memstore branch name, a contract that cannot be honoured has to fail while the
planner can still choose differently, not when the spawner opens the branch.
"""

from __future__ import annotations

import pytest

from taste.brains.contract import Contract
from taste.memstore.objects import BadName


def _c(**kw) -> Contract:
    base = {"identity": "worker-1", "task": "do the thing", "success_criteria": ("it works",)}
    return Contract(**{**base, **kw})


def test_a_contract_carries_what_the_monitor_needs_to_judge() -> None:
    c = _c(success_criteria=("tests pass", "no TODOs left"), outputs=("parser.py",))
    judging = c.judging_brief()
    for criterion in c.success_criteria:
        assert criterion in judging
    assert "parser.py" in judging
    # worker and monitor read the same criteria, not two paraphrases
    for criterion in c.success_criteria:
        assert criterion in c.brief()


def test_a_contract_without_success_criteria_is_refused() -> None:
    """A monitor with no criteria falls back to taste, which is the failure
    this architecture exists to remove."""
    with pytest.raises(ValueError, match="success criteria"):
        _c(success_criteria=())


@pytest.mark.parametrize("identity", ["worker/1", "a b", "..", "-lead"])
def test_an_identity_that_cannot_be_a_branch_is_refused_at_planning_time(
    identity: str,
) -> None:
    """The identity IS the branch name. Refusing it here means the planner can
    still pick another; refusing it at spawn means the decomposition is already
    built around a name that cannot exist.
    """
    with pytest.raises((BadName, ValueError)):
        _c(identity=identity)


def test_a_string_criterion_is_not_shredded_into_characters() -> None:
    """``tuple("tests pass")`` gives ten one-character criteria and no error --
    a contract a monitor cannot possibly judge."""
    c = Contract.from_dict(
        {"identity": "w", "task": "t", "success_criteria": "the tests pass"}
    )
    assert c.success_criteria == ("the tests pass",)


def test_a_contract_round_trips_equal(tmp_path) -> None:
    """Fields are declared as tuples; a caller passing lists must still
    produce a contract equal to the one read back from JSON."""
    written = Contract(
        identity="worker-1",
        task="build the parser",
        inputs=["grammar.md"],
        outputs=["parser.py"],
        success_criteria=["tests pass"],
    )
    assert Contract.from_json(written.to_json()) == written


def test_the_brief_reads_as_instructions_not_as_json() -> None:
    c = _c(inputs=("spec.md",), outputs=("out.py",), success_criteria=("it works",))
    brief = c.brief()
    assert brief.startswith("You are worker-1.")
    assert "{" not in brief and '"' not in brief
