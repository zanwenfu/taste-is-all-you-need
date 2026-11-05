"""The claim ledger: what each reported number really ran, checked by machine.

On Nov 3-4 a test with no Taste code in it was reported as the memory under study. Each test here is
one way that could happen again, and the check must refuse it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from studies.memory_ledger import (
    BASELINE,
    REAL_RUN,
    STAND_IN,
    STUDY_DRIVEN,
    ClaimRefused,
    check,
    looks,
    measured_path,
    stand_in_record,
    taste_record,
)
from taste.memstore import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "repo", "s1")
    a = s.branch("memory")
    a.write("rule.md", "the marker\n")
    a.checkpoint("step")
    yield s
    s.close()


def test_a_taste_record_says_which_code_ran_and_how_often_it_looked(store: Store) -> None:
    store.search_contents("marker")
    hit = store.search_contents("marker")[0]
    store.fetch(hit.pointer)

    record = taste_record(store, mode="agent_tools", commit="abc123", dirty=False)

    assert record["name"] == STUDY_DRIVEN
    assert record["memstore_imported"] is True
    assert record["memstore_file"].endswith("taste/memstore/__init__.py")
    assert record["taste_commit"] == "abc123"
    assert record["store_path"] == str(store.root)
    assert record["store_head"] == store.view("memory").head.id
    assert (record["search_calls"], record["fetch_calls"]) == (2, 1)
    check([record])


def test_a_number_from_a_process_that_never_imported_the_memory_is_refused(tmp_path: Path) -> None:
    """Run in a fresh process: this test file imports taste.memstore itself."""
    script = (
        "import json, sys\n"
        "from studies.memory_ledger import process_facts\n"
        "print(json.dumps(process_facts()))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    facts = json.loads(result.stdout)
    assert facts["memstore_imported"] is False
    record = {
        **facts,
        "name": STUDY_DRIVEN,
        "mode": "agent_tools",
        "taste_commit": "abc123",
        "taste_dirty": False,
        "store_path": "/somewhere",
        "store_head": "f" * 40,
        "search_calls": 3,
        "fetch_calls": 1,
    }
    with pytest.raises(ClaimRefused, match="did not import"):
        check([record])


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"name": "our memory layer"}, "not one of the allowed names"),
        ({"taste_commit": ""}, "no commit"),
        ({"taste_dirty": True}, "uncommitted"),
        ({"store_path": ""}, "no store"),
        ({"store_head": ""}, "no store"),
        ({"mode": "anything"}, "unknown mode"),
        ({"mode": "single_query", "search_calls": 0}, "never searched"),
        ({"search_calls": -1}, "count"),
        ({"fetch_calls": None}, "count"),
    ],
)
def test_each_way_a_record_can_be_wrong_is_refused(store: Store, change: dict, reason: str) -> None:
    record = {**taste_record(store, mode="agent_tools", commit="abc123", dirty=False), **change}
    with pytest.raises(ClaimRefused, match=reason):
        check([record])


def test_a_baseline_may_not_carry_memory_counts_or_call_itself_taste(store: Store) -> None:
    record = stand_in_record(BASELINE, code="studies/context_rot/record_arms.py", commit="abc123", dirty=False)
    check([record])
    assert record["name"] == BASELINE
    with pytest.raises(ClaimRefused, match="not one of the allowed names"):
        stand_in_record("our memory arm", code="x", commit="abc123", dirty=False)
    with pytest.raises(ClaimRefused, match="cannot be a stand-in"):
        stand_in_record(STUDY_DRIVEN, code="x", commit="abc123", dirty=False)
    with pytest.raises(ClaimRefused, match="searched"):
        check([{**record, "search_calls": 4}])
    with pytest.raises(ClaimRefused, match="no code path"):
        check([{**record, "code": ""}])


def test_zero_searches_are_allowed_only_where_the_agent_chose_and_count_as_not_looking(store: Store) -> None:
    quiet = taste_record(store, mode="agent_tools", commit="abc123", dirty=False)
    store.search_contents("marker")
    looked = taste_record(store, mode="agent_tools", commit="abc123", dirty=False)
    delivered = taste_record(store, mode="delivered", commit="abc123", dirty=False)

    check([quiet, looked, delivered])
    assert looks([quiet, looked, looked, delivered]) == {
        "cases": 3, "looked": 2, "rate": pytest.approx(2 / 3),
    }


def test_the_rate_of_looking_counts_only_cases_where_the_agent_could_choose() -> None:
    assert looks([]) == {"cases": 0, "looked": 0, "rate": None}


def test_a_real_taste_run_needs_its_own_evidence(store: Store) -> None:
    record = {**taste_record(store, mode="agent_tools", commit="abc123", dirty=False), "name": REAL_RUN}
    check([record])
    assert STAND_IN != BASELINE


def test_the_measured_path_is_written_down_for_each_record(store: Store) -> None:
    assert measured_path(taste_record(store, mode="agent_tools", commit="c", dirty=False)) == "taste/memstore"
    assert (
        measured_path(stand_in_record(STAND_IN, code="scripts/kill_drill.py", commit="c", dirty=False))
        == "scripts/kill_drill.py"
    )


def test_the_command_line_checks_a_results_file(store: Store, tmp_path: Path) -> None:
    good = tmp_path / "good.jsonl"
    good.write_text(
        "".join(
            json.dumps({"id": n, "ledger": taste_record(store, mode="agent_tools", commit="abc123", dirty=False)}) + "\n"
            for n in range(3)
        )
    )
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": 1, "ledger": {"name": "our memory layer"}}) + "\n")
    missing = tmp_path / "missing.jsonl"
    missing.write_text(json.dumps({"id": 1}) + "\n")

    def run(path: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "studies.memory_ledger", str(path)], capture_output=True, text=True
        )

    passed = run(good)
    assert passed.returncode == 0 and json.loads(passed.stdout)["cases"] == 3
    assert run(bad).returncode != 0
    assert run(missing).returncode != 0
