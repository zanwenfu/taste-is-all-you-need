"""The monitor brain: does it see, judge, and actually stop things.

The judge is injected, so these tests drive the mechanics with a scripted one.
That is deliberate: the claim under test is not "the model judges well" but
that a monitor can watch a branch while it is being written, that its verdict
reaches the worker, and that each rung of the ladder does what it says. A real
model would make those claims harder to check, not better checked.

The client is a fake that records calls, for the same reason -- the rungs are
about *which* SDK call is made and in what order, and a live model cannot
assert that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taste.memstore import Store

pytest.importorskip("claude_agent_sdk", reason="the brain layer needs claude-agent-sdk")

from taste.brains.contract import Contract
from taste.brains.monitor import (
    Judgement,
    MonitorBrain,
    Severity,
    batch_prompt,
)
from taste.brains.subbrain import SubBrain


def a_contract(**kw) -> Contract:
    base = {
        "identity": "worker-1",
        "task": "build the parser",
        "success_criteria": ("the tests pass", "no TODOs are left"),
    }
    return Contract(**{**base, **kw})


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "repo", "s1")
    yield s
    s.close()


class FakeClient:
    """Records the calls a monitor makes, in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def query(self, text: str) -> None:
        self.calls.append(("query", text))

    async def interrupt(self) -> None:
        self.calls.append(("interrupt", ""))

    async def set_permission_mode(self, mode: str) -> None:
        self.calls.append(("set_permission_mode", mode))

    @property
    def rungs(self) -> list[str]:
        return [c[0] for c in self.calls]


def judge_always(severity: Severity, reason: str = "because"):
    return lambda contract, batch, view: Judgement(severity=severity, reason=reason)


# ------------------------------------------------------------------ watching


def test_a_monitor_reads_a_branch_while_the_worker_holds_it(store: Store) -> None:
    """The worker holds the write lease for its whole run. A monitor that
    needed the lease could only watch a brain that had already stopped."""
    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, judge_always(Severity.FINE))

    brain.wal.intent("Bash", "t1", {"command": "pytest"})
    brain.wal.result("Bash", "t1", ok=False, summary="3 failed")

    seen = monitor.observations()
    assert [e["kind"] for e in seen] == ["tool_intent", "tool_result"]
    assert monitor.worker_is_alive()
    brain.close()


def test_liveness_comes_from_the_lease_not_from_a_claim(store: Store) -> None:
    """A status claim from a dead brain is a confident lie, so the monitor
    asks the lease rather than the worker."""
    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, judge_always(Severity.FINE))
    assert monitor.worker_is_alive()

    brain.close()
    assert not monitor.worker_is_alive()


# ------------------------------------------------------------------ batching


def test_a_monitor_waits_for_a_full_batch_while_work_continues(store: Store) -> None:
    """A persistent per-event monitor re-reads its transcript every turn, so
    its cost climbs with the event count. Batching keeps it flat."""
    brain = SubBrain(store, a_contract())
    calls: list[int] = []

    def counting_judge(contract, batch, view):
        calls.append(len(batch))
        return Judgement(severity=Severity.FINE, reason="ok")

    monitor = MonitorBrain(store, brain.contract, counting_judge, batch_size=4)

    for i in range(3):
        brain.wal.intent("Bash", f"t{i}", {"command": "x"})
    assert monitor.tick() is None, "judged a partial batch while work continued"
    assert calls == []

    brain.wal.intent("Bash", "t3", {"command": "x"})
    assert monitor.tick() is not None
    assert calls == [4]
    brain.close()


def test_a_partial_batch_is_judged_once_the_worker_stops(store: Store) -> None:
    """Waiting for a full batch must not mean never judging the last events."""
    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, judge_always(Severity.FINE), batch_size=10)
    brain.wal.intent("Bash", "t0", {"command": "x"})
    assert monitor.tick() is None

    brain.close()  # the worker is gone; the tail is all there will ever be
    assert monitor.tick() is not None


def test_events_are_judged_once(store: Store) -> None:
    brain = SubBrain(store, a_contract())
    seen: list[str] = []

    def recording_judge(contract, batch, view):
        seen.extend(e["tool_use_id"] for e in batch)
        return Judgement(severity=Severity.FINE, reason="ok")

    monitor = MonitorBrain(store, brain.contract, recording_judge, batch_size=2)
    for i in range(4):
        brain.wal.intent("Bash", f"t{i}", {"command": "x"})

    monitor.tick()
    monitor.tick()
    assert seen == ["t0", "t1", "t2", "t3"]
    assert monitor.tick() is None
    brain.close()


# ------------------------------------------------------------------ the ladder


@pytest.mark.parametrize(
    ("severity", "expected"),
    [
        (Severity.FINE, []),
        (Severity.DRIFTING, ["query"]),
        (Severity.WRONG, ["interrupt", "query"]),
        (Severity.LOST, ["interrupt", "set_permission_mode", "query"]),
    ],
)
def test_severity_chooses_the_rung(store: Store, severity, expected) -> None:
    import asyncio

    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, judge_always(severity))
    client = FakeClient()

    asyncio.run(monitor.respond(Judgement(severity=severity, reason="r"), client))
    assert client.rungs == expected
    brain.close()


def test_a_wrong_turn_is_stopped_before_it_is_explained(store: Store) -> None:
    """Order matters: feedback sent to a brain mid-tool-call is read after the
    tool it was meant to prevent."""
    import asyncio

    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, judge_always(Severity.WRONG))
    client = FakeClient()

    asyncio.run(monitor.respond(Judgement(Severity.WRONG, "editing the wrong file"), client))
    assert client.rungs.index("interrupt") < client.rungs.index("query")
    assert "editing the wrong file" in client.calls[-1][1]
    brain.close()


def test_a_lost_brain_is_demoted_not_killed(store: Store) -> None:
    """The reasoning is the expensive part and it survives demotion -- and a
    brain in plan mode can still report what it learned, which is what the
    central brain needs in order to re-plan."""
    import asyncio

    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, judge_always(Severity.LOST))
    client = FakeClient()

    asyncio.run(monitor.respond(Judgement(Severity.LOST, "going in circles"), client))
    assert ("set_permission_mode", "plan") in client.calls
    assert "Report what you tried" in client.calls[-1][1]
    brain.close()


def test_an_interrupted_worker_is_told_to_check_its_worktree(store: Store) -> None:
    """An interrupted tool can still have completed its side effect while the
    transcript records it as rejected."""
    import asyncio

    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, judge_always(Severity.WRONG))
    client = FakeClient()

    asyncio.run(monitor.respond(Judgement(Severity.WRONG, "wrong file"), client))
    assert "may have completed" in client.calls[-1][1]
    brain.close()


# ------------------------------------------------------------------ the verdict


def test_a_verdict_reaches_the_worker_without_the_lease(store: Store) -> None:
    """The whole loop: the monitor speaks while the worker is still writing,
    and the worker hears it on its next wake."""
    import asyncio

    brain = SubBrain(store, a_contract())
    brain.checkpoint("some work")
    monitor = MonitorBrain(store, brain.contract, judge_always(Severity.WRONG))
    client = FakeClient()

    asyncio.run(monitor.respond(Judgement(Severity.WRONG, "B is a dead end"), client))

    waking = brain.wake()
    assert [v.detail for v in waking.unacked] == ["B is a dead end"]
    assert "B is a dead end" in waking.briefing()
    brain.close()


def test_a_wobble_is_not_reported_as_a_failure(store: Store) -> None:
    """A monitor that cries failure at every wobble is one a brain learns to
    ignore."""
    assert Judgement(Severity.DRIFTING, "r").to_verdict("m").status == "unknown"
    assert Judgement(Severity.WRONG, "r").to_verdict("m").status == "fail"
    assert Judgement(Severity.LOST, "r").to_verdict("m").status == "fail"
    assert Judgement(Severity.FINE, "r").to_verdict("m").status == "pass"


def test_the_verdict_carries_the_severity_for_the_central_brain(store: Store) -> None:
    verdict = Judgement(Severity.LOST, "r").to_verdict("monitor/worker-1")
    assert verdict.failure_class == "lost"
    assert verdict.by == "monitor/worker-1"


# ------------------------------------------------------------------ the prompt


def test_the_judge_is_asked_against_the_contracts_own_words(store: Store) -> None:
    """Worker and monitor are held to the same criteria, not two paraphrases."""
    contract = a_contract()
    prompt = batch_prompt(contract, [{"kind": "tool_intent", "tool": "Bash"}])
    for criterion in contract.success_criteria:
        assert criterion in prompt
    for level in ("fine", "drifting", "wrong", "lost"):
        assert level in prompt


def test_the_report_tells_the_central_brain_what_it_needs(store: Store) -> None:
    import asyncio

    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, judge_always(Severity.LOST))
    asyncio.run(monitor.respond(Judgement(Severity.DRIFTING, "a"), FakeClient()))
    asyncio.run(monitor.respond(Judgement(Severity.LOST, "b"), FakeClient()))

    report = monitor.report()
    assert report["worker"] == "worker-1"
    assert report["alive"] is True
    assert [r for r, _ in report["interventions"]] == ["nudge", "demote"]
    brain.close()


def test_severity_is_ordered_so_the_worst_wins(store: Store) -> None:
    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, judge_always(Severity.FINE))
    monitor.state.judgements = [
        Judgement(Severity.FINE, "a"),
        Judgement(Severity.WRONG, "b"),
        Judgement(Severity.DRIFTING, "c"),
    ]
    assert monitor.state.worst is Severity.WRONG
    brain.close()


def test_a_checkpoint_does_not_make_the_monitor_skip_events(store: Store) -> None:
    """The journal is keyed on the branch head, so a checkpoint starts a fresh
    one.

    A single running index carried across that boundary pointed past the new
    journal's start and silently discarded as many fresh events as it had
    already judged -- and the monitor looked perfectly healthy while never
    seeing the work. Position is per state for that reason.
    """
    seen: list[list[str]] = []

    def recording_judge(contract, batch, view):
        seen.append([e["tool_use_id"] for e in batch])
        return Judgement(Severity.FINE, "ok")

    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, recording_judge, batch_size=2)

    for i in range(2):
        brain.wal.intent("Bash", f"a{i}", {})
    monitor.tick()

    brain.checkpoint("the worker commits its work")

    for i in range(2):
        brain.wal.intent("Bash", f"b{i}", {})
    monitor.tick()

    assert seen == [["a0", "a1"], ["b0", "b1"]], "events were skipped across the checkpoint"
    assert not monitor.unjudged(), "nothing should be left unreachable"
    brain.close()


def test_events_are_still_judged_once_across_many_checkpoints(store: Store) -> None:
    """A worker that checkpoints often would otherwise have most of its work
    never looked at."""
    judged: list[str] = []

    def recording_judge(contract, batch, view):
        judged.extend(e["tool_use_id"] for e in batch)
        return Judgement(Severity.FINE, "ok")

    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, recording_judge, batch_size=1)

    expected = []
    for round_no in range(4):
        tool_id = f"t{round_no}"
        brain.wal.intent("Bash", tool_id, {})
        expected.append(tool_id)
        monitor.tick()
        brain.checkpoint(f"round {round_no}")

    assert judged == expected
    brain.close()


def test_a_partial_batch_survives_a_checkpoint(store: Store) -> None:
    """Events held back as a partial batch must not be destroyed by a commit.

    ``tick`` holds back an incomplete batch while the worker is alive, and
    ``publish_state`` unlinks the journal when the worker checkpoints. Reading
    only the live journal therefore lost exactly ``N mod B`` events at every
    checkpoint -- they were in the published transcript, and the monitor could
    never reach them. ``observations`` spans both sources for that reason.
    """
    seen: list[str] = []

    def recording_judge(contract, batch, view):
        seen.extend(e["tool_use_id"] for e in batch)
        return Judgement(Severity.FINE, "ok")

    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, recording_judge, batch_size=3)

    for i in range(5):  # 3 judged, 2 held back
        brain.wal.intent("Bash", f"e{i}", {})
    monitor.tick()
    assert seen == ["e0", "e1", "e2"]

    brain.checkpoint("the worker commits mid-batch")
    brain.wal.intent("Bash", "e5", {})
    monitor.tick()

    # e3 and e4 were the held-back partial batch; e5 completes it.
    assert seen == ["e0", "e1", "e2", "e3", "e4", "e5"], (
        "the partial batch did not survive the checkpoint"
    )

    brain.close()
    monitor.tick()
    assert set(seen) == {f"e{i}" for i in (0, 1, 2, 3, 4, 5)}
    assert not monitor.unjudged()


def test_a_restarted_monitor_still_knows_the_worker_went_wrong(store: Store) -> None:
    """``report()`` is what the central brain reads to decide whether to
    re-plan. Persisting the position but not the judgements meant a restarted
    monitor reported worst="fine" for a worker it had just judged LOST.
    """
    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, judge_always(Severity.LOST), batch_size=1)
    brain.wal.intent("Bash", "t0", {})
    monitor.tick()
    assert monitor.state.worst is Severity.LOST

    restarted = MonitorBrain(store, brain.contract, judge_always(Severity.FINE), batch_size=1)
    assert restarted.state.worst is Severity.LOST
    assert restarted.report()["worst"] == "lost"
    brain.close()


def test_a_restarted_monitor_does_not_re_judge_old_events(store: Store) -> None:
    """Re-judging would re-interrupt a worker for work already handled."""
    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, judge_always(Severity.FINE), batch_size=2)
    for i in range(2):
        brain.wal.intent("Bash", f"t{i}", {})
    monitor.tick()

    restarted = MonitorBrain(store, brain.contract, judge_always(Severity.FINE), batch_size=2)
    assert restarted.unjudged() == []
    brain.close()
