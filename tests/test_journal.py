"""The journal: checkpoint cards, attempt anchors, and the branch index.

Two things are being proved here. That the journal records what a reader
needs in order to decide which checkpoint to page in — and that with the
journal disabled the kernel behaves exactly as it did before it existed.
The second is the load-bearing one: "build to delete" is only true if it is
tested.
"""

from __future__ import annotations

import json
from pathlib import Path

from taste.cores import Plan, Step, Verification, WorkerResult
from taste.journal import (
    ANCHOR_PREFIX,
    NOTES_REF,
    CheckpointCard,
    FileChange,
    Journal,
    load_index,
    parse_numstat,
)
from taste.kernel import Kernel
from taste.memory import Memory
from tests.golden import rollback_scenario
from tests.test_golden_baseline import EXPECTED_EVENTS


def _journal(ws: Path, branch: str = "taste/session-j") -> Journal:
    memory = Memory.open_session(ws, branch.rsplit("-", 1)[-1])
    return Journal(memory, gitdir=Path(memory.repo.git_dir) / "taste")


# ------------------------------------------------------------------ ablation


def test_journal_disabled_reproduces_the_baseline_exactly(refactor_workspace: Path) -> None:
    """The whole build-to-delete claim, as one assertion."""
    sig = rollback_scenario(refactor_workspace).run(refactor_workspace, journal=False)
    assert sig.events == EXPECTED_EVENTS


def test_journal_disabled_writes_no_artifacts(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    rollback_scenario(ws).run(ws, journal=False)

    assert not (ws / ".git" / "taste" / "journal.jsonl").exists()
    memory = Memory(ws, "taste/session-golden")
    assert memory.list_refs(f"{ANCHOR_PREFIX}/") == []
    assert memory.read_note(memory.head().sha, ref=NOTES_REF) is None


def test_journal_enabled_adds_only_journal_events(refactor_workspace: Path) -> None:
    """Enabling it must not perturb any existing event — only add new kinds."""
    ws = refactor_workspace
    sig = rollback_scenario(ws).run(ws, journal=True)

    non_journal = tuple(e for e in sig.events if not e[0].startswith("journal."))
    assert non_journal == EXPECTED_EVENTS
    assert any(e[0] == "journal.card" for e in sig.events)
    assert any(e[0] == "journal.anchor" for e in sig.events)


# ------------------------------------------------------------------ cards


def test_card_records_the_step_and_survives_as_a_note(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    rollback_scenario(ws).run(ws, journal=True)

    memory = Memory(ws, "taste/session-golden")
    journal = Journal(memory, gitdir=Path(memory.repo.git_dir) / "taste")
    card = journal.read(memory.head().sha)

    assert card is not None
    assert card.step_id == "step-02"
    assert card.verdict == "pass"
    assert card.intent == "create the second module"
    assert "also.py" in [f.path for f in card.files]


def test_card_is_not_in_the_working_tree(refactor_workspace: Path) -> None:
    """A tracked card would become part of the diff the Monitor judges."""
    ws = refactor_workspace
    rollback_scenario(ws).run(ws, journal=True)

    tracked = Memory(ws, "taste/session-golden").repo.git.ls_files()
    assert "journal" not in tracked
    assert not (ws / ".taste" / "cards").exists()


def test_failed_attempt_card_survives_the_rollback(refactor_workspace: Path) -> None:
    """The record of a discarded attempt is exactly what must not be discarded."""
    ws = refactor_workspace
    rollback_scenario(ws).run(ws, journal=True)

    lines = [
        json.loads(line)
        for line in (ws / ".git" / "taste" / "journal.jsonl").read_text().splitlines()
        if line
    ]
    failures = [c for c in lines if c["verdict"] == "fail"]
    assert len(failures) == 1
    assert failures[0]["step_id"] == "step-01"
    assert failures[0]["attempt"] == 1
    # And it names the file the doomed attempt wrote.
    assert "wrong.py" in [f["path"] for f in failures[0]["files"]]


def test_anchor_keeps_the_rolled_back_commit_readable(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    rollback_scenario(ws).run(ws, journal=True)

    memory = Memory(ws, "taste/session-golden")
    anchors = memory.list_refs(f"{ANCHOR_PREFIX}/")
    assert len(anchors) == 1

    ref, sha = anchors[0]
    assert "step-01" in ref
    # The discarded attempt is still fully readable at its anchor.
    assert "not what was asked" in memory.show(sha, "wrong.py")
    # ...while the branch itself has no trace of it.
    assert not (ws / "wrong.py").exists()


def test_journal_write_failure_never_fails_the_step(refactor_workspace: Path, monkeypatch) -> None:
    """Bookkeeping is not allowed to break the thing it is bookkeeping."""
    ws = refactor_workspace

    def explode(*a, **k):
        raise OSError("disk on fire")

    monkeypatch.setattr(Memory, "write_note", explode)
    sig = rollback_scenario(ws).run(ws, journal=True)

    assert sig.status == "completed"
    assert sig.step_results == (("step-01", True, 2, True), ("step-02", True, 1, False))


# ------------------------------------------------------------------ index


def test_index_lists_every_checkpoint_oldest_first(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    rollback_scenario(ws).run(ws, journal=True)

    memory = Memory(ws, "taste/session-golden")
    index = load_index(memory)

    steps = [c.step_id for c in index.cards]
    assert steps[-2:] == ["step-01", "step-02"], steps
    assert index.cards[-1].verdict == "pass"


def test_index_degrades_gracefully_without_cards(refactor_workspace: Path) -> None:
    """A run made before journalling still yields a usable index."""
    ws = refactor_workspace
    rollback_scenario(ws).run(ws, journal=False)

    index = load_index(Memory(ws, "taste/session-golden"))
    assert index.degraded == len(index.cards) > 0
    # Reconstructed from the commit trailer, so step ids still resolve.
    assert "step-02" in [c.step_id for c in index.cards]


def test_index_renders_yaml(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    rollback_scenario(ws).run(ws, journal=True)

    text = load_index(Memory(ws, "taste/session-golden")).to_yaml()
    assert "step: step-02" in text
    assert "verdict: pass" in text
    assert "intent:" in text


def test_one_line_is_scannable() -> None:
    card = CheckpointCard(
        session="s",
        step_id="step-03",
        sha="abcdef1234",
        verdict="fail",
        failure_class="implementation_bug",
        action="rollback_and_retry",
        files=(FileChange("a.py", 14, 3),),
        cost_usd=0.0412,
    )
    line = card.one_line()
    assert "abcdef1" in line and "step-03" in line and "FAIL" in line
    assert "implementation_bug->rollback_and_retry" in line
    assert "+14-3" in line and "$0.0412" in line


# ------------------------------------------------------------------ units


def test_parse_numstat_handles_binary_and_totals() -> None:
    changes, total = parse_numstat("3\t1\tsrc/a.py\n-\t-\timg.png\n10\t0\tb.py\n")
    assert [c.path for c in changes] == ["src/a.py", "img.png", "b.py"]
    assert total == 14  # binary contributes no lines


def test_card_roundtrip_tolerates_schema_drift() -> None:
    card = CheckpointCard(session="s", step_id="step-01", sha="deadbeef", intent="do it")
    raw = json.loads(card.to_json())
    raw["a_field_from_the_future"] = 42
    del raw["verdict_reason"]

    restored = CheckpointCard.from_dict(raw)
    assert restored.step_id == "step-01"
    assert restored.intent == "do it"
    assert restored.verdict_reason == ""


def test_cards_are_bounded_in_size(refactor_workspace: Path) -> None:
    """An index entry must stay an index entry."""
    ws = refactor_workspace
    journal = _journal(ws)
    card = CheckpointCard(
        session="s",
        step_id="step-01",
        sha=Memory(ws, "taste/session-j").head().sha,
        files=tuple(FileChange(f"f{i}.py") for i in range(100)),
        summary="x" * 500,
    )
    journal.card(card)
    stored = journal.read(card.sha)

    assert stored is not None
    assert len(stored.files) == 40
    assert stored.files_truncated is True
    assert len(stored.summary) <= 200


def test_prune_anchors_releases_the_attempts(refactor_workspace: Path) -> None:
    ws = refactor_workspace
    rollback_scenario(ws).run(ws, journal=True)

    memory = Memory(ws, "taste/session-golden")
    journal = Journal(memory, gitdir=Path(memory.repo.git_dir) / "taste")
    assert journal.prune_anchors() == 1
    assert journal.anchors() == []


def test_kernel_multi_attempt_cards_are_distinguishable(refactor_workspace: Path) -> None:
    """Attempt N and attempt N+1 of one step must not collapse into one row."""
    ws = refactor_workspace
    rollback_scenario(ws).run(ws, journal=True)

    lines = [
        json.loads(line)
        for line in (ws / ".git" / "taste" / "journal.jsonl").read_text().splitlines()
        if line
    ]
    step_01 = [c for c in lines if c["step_id"] == "step-01"]
    assert [c["attempt"] for c in step_01] == [1, 2]
    assert [c["verdict"] for c in step_01] == ["fail", "pass"]


def test_journal_survives_a_failed_run(refactor_workspace: Path) -> None:
    """Cards for the steps that did run must exist even when the run halts."""
    ws = refactor_workspace
    plan = Plan(
        task="doomed",
        steps=[Step("step-01", "impossible", Verification(kind="shell", command="false"))],
    )
    result = Kernel(workspace=ws, max_retries=0, journal=True).run(
        task="doomed",
        spec=__import__("taste.agent", fromlist=["AgentSpec"]).AgentSpec(
            name="s", description=""
        ),
        session_id="doomed",
        plan_override=plan,
        worker_override=lambda s, p: WorkerResult("tried", 1, "end_turn"),
    )

    assert result.status == "failed"
    text = (ws / ".git" / "taste" / "journal.jsonl").read_text()
    assert '"verdict": "fail"' in text


def test_the_card_records_tool_errors() -> None:
    """The field existed on the card from the start and was never populated,
    so every card in every run reported zero tool errors.

    Caught on a real run where the recovery table diagnosed R1.tool_errors --
    correctly, the tooling really had raised -- while the card beside it said
    there had been none. Anyone diagnosing from the audit trail would have
    concluded tool errors never happen.
    """
    from taste.cores import Step, Verification, WorkerResult
    from taste.journal import card_from_step

    worker = WorkerResult(
        summary="s", tool_calls=12, stopped_reason="end_turn",
        tool_errors=3, tool_error_kinds=("TypeError", "KeyError"),
    )
    card = card_from_step(
        session="s", branch="b", sha="abc", parent_sha=None,
        step=Step(id="step-01", description="d",
                  verification=Verification(kind="shell", command="true")),
        attempt=1, verdict_passed=False, verdict_reason="r",
        files=(), diff_lines=0, worker=worker,
    )

    assert card.tool_errors == 3, "the card must not under-report tool failures"
    assert card.tool_calls == 12
