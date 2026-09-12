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

import json
import os
import signal
import threading
from pathlib import Path

import pytest

from taste.memstore import Store

pytest.importorskip("claude_agent_sdk", reason="the brain layer needs claude-agent-sdk")

from taste.brains.contract import Contract
from taste.brains.monitor import (
    Judgement,
    MonitorBrain,
    Severity,
    TerminalDecision,
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
    "severity", [Severity.FINE, Severity.DRIFTING, Severity.WRONG, Severity.LOST]
)
def test_severity_names_an_observation_not_an_action(store: Store, severity) -> None:
    """Severity says how confident the observation is, not what to do.

    Deciding what to do belongs to the central brain, which can see the plan
    this work is part of and the other workers sharing it. The monitor sees a
    partial view of one worker, seconds old.
    """
    import asyncio

    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, judge_always(severity))
    client = FakeClient()

    rung = asyncio.run(monitor.respond(Judgement(severity=severity, reason="r"), client))
    assert client.calls == []
    assert rung == ("none" if severity is Severity.FINE else f"flagged-{severity.value}")
    brain.close()


def test_the_worker_is_never_interrupted_by_its_own_monitor(store: Store) -> None:
    """The measurement that removed the lever.

    An interrupt lands mid-stream and the turn comes back ``aborted_*`` rather
    than ``end_turn``, so the runtime never holds a terminal candidate and the
    run cannot finish. Six interrupts, six aborted results, zero completions --
    and then a verdict blaming the worker for having been interrupted.
    """
    import asyncio

    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, judge_always(Severity.WRONG))
    client = FakeClient()

    asyncio.run(monitor.respond(Judgement(Severity.WRONG, "editing the wrong file"), client))
    assert "interrupt" not in client.rungs
    assert client.calls == []
    brain.close()


def test_a_lost_brain_is_reported_not_demoted(store: Store) -> None:
    """Even ``lost`` only files a verdict.

    Plan-mode demotion was the gentlest rung and still the wrong shape: it
    changes a worker's capabilities mid-run on evidence the monitor cannot
    fully see, and the central brain is the one re-planning anyway.
    """
    import asyncio

    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, judge_always(Severity.LOST))
    client = FakeClient()

    rung = asyncio.run(monitor.respond(Judgement(Severity.LOST, "going in circles"), client))
    assert client.calls == []
    assert rung == "flagged-lost"
    assert monitor.state.worst is Severity.LOST
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
    assert [r for r, _ in report["interventions"]] == ["flagged-drifting", "flagged-lost"]
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


def test_a_broken_client_cannot_cost_the_brain_its_warning(store: Store) -> None:
    """There is no longer an escalation to fail, and that is the point.

    This used to assert ``interrupt-failed``: the verdict was recorded first so
    that a closed client cost the worker its interruption but never its
    warning. With nothing sent, the failure mode is gone entirely -- a client
    that raises on every method changes nothing, because none is called.
    """
    import asyncio

    class BrokenClient:
        async def interrupt(self) -> None:
            raise RuntimeError("client is closed")

        async def query(self, text: str) -> None:
            raise RuntimeError("client is closed")

        async def set_permission_mode(self, mode: str) -> None:
            raise RuntimeError("client is closed")

    brain = SubBrain(store, a_contract())
    brain.checkpoint("base")
    monitor = MonitorBrain(store, brain.contract, judge_always(Severity.WRONG))

    rung = asyncio.run(
        monitor.respond(Judgement(Severity.WRONG, "editing the wrong file"), BrokenClient())
    )

    assert rung == "flagged-wrong"
    assert monitor.state.interventions[0][0] == "flagged-wrong"
    assert [v.detail for v in brain.wake().unacked] == ["editing the wrong file"]
    brain.close()


# ------------------------------------------------------------------ crash-safe cursor and actions


def test_a_rollback_divergence_cannot_make_the_monitor_skip_events(store: Store) -> None:
    """A cumulative integer cursor points past the end after rollback.

    The replacement trajectory shares a prefix with the abandoned one, then
    diverges.  Only that common prefix is already judged; the new suffix is new
    work even when its numeric position is below the old cursor.
    """
    seen: list[str] = []

    def recording_judge(contract, batch, view):
        seen.extend(e["tool_use_id"] for e in batch)
        return Judgement(Severity.FINE, "ok")

    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, recording_judge, batch_size=1)

    brain.wal.intent("Bash", "a", {})
    good = brain.checkpoint("the shared prefix")
    monitor.tick()

    brain.wal.intent("Bash", "abandoned", {})
    brain.checkpoint("the trajectory we abandon")
    monitor.tick()

    brain.branch.rollback(good, "take another route")
    brain.wal.intent("Bash", "replacement", {})
    monitor.tick()

    assert seen == ["a", "abandoned", "replacement"]
    brain.close()


def test_a_legacy_integer_cursor_is_not_trusted_after_upgrade(store: Store) -> None:
    """Without old content identities, re-judge rather than silently skip."""
    brain = SubBrain(store, a_contract())
    brain.install_contract()
    brain.checkpoint("install the contract that owns the legacy monitor state")
    brain.wal.intent("Bash", "replacement", {})
    legacy_path = store.sidecar("monitor", brain.contract.identity)
    legacy_path.write_text(
        json.dumps(
            {
                "judged_through": 1,
                "judgements": [],
                "interventions": [],
            }
        )
    )
    seen: list[str] = []

    def judge(contract, batch, view):
        seen.extend(event["tool_use_id"] for event in batch)
        return Judgement(Severity.FINE, "checked after upgrade")

    monitor = MonitorBrain(store, brain.contract, judge, batch_size=1)
    monitor.tick()

    assert seen == ["replacement"]
    assert len(monitor.state.fingerprints) == 1
    assert not legacy_path.exists()
    assert monitor._state_path().exists()
    migrated = json.loads(monitor._state_path().read_text())
    assert migrated["schema"] == "taste.brains/MonitorState/3"
    assert migrated["contract_digest"] == monitor.contract_digest
    assert migrated["terminal_assessments"] == []
    brain.close()


def test_a_revised_contract_has_distinct_state_and_old_report_remains_readable(
    store: Store,
) -> None:
    """A worker name is not a monitor-state identity.

    Re-planning may keep the worker address while changing the task or success
    criteria.  Its old LOST result remains audit history, but must not poison
    the replacement contract's initial report or cursor.
    """
    original = a_contract(task="try approach A")
    brain = SubBrain(store, original)
    brain.wal.intent("Bash", "old-attempt", {})
    old_monitor = MonitorBrain(
        store,
        original,
        judge_always(Severity.LOST, "approach A cannot work"),
        batch_size=1,
    )
    assert old_monitor.tick() is not None
    old_report = old_monitor.report()
    old_path = old_monitor._state_path()

    revised = a_contract(
        task="try approach B",
        success_criteria=("approach B's tests pass",),
    )
    new_monitor = MonitorBrain(
        store,
        revised,
        judge_always(Severity.FINE, "new contract is on track"),
        batch_size=1,
    )

    assert new_monitor._state_path() != old_path
    assert new_monitor.report()["worst"] == "fine"
    assert new_monitor.report()["judgements"] == 0
    assert new_monitor.unjudged()[0]["tool_use_id"] == "old-attempt"
    assert new_monitor.tick() == Judgement(Severity.FINE, "new contract is on track")
    assert new_monitor._state_path().exists()

    # The old digest-addressed state is not overwritten or hidden by opening
    # the revision; an auditor holding the original contract can still read it.
    reopened_old = MonitorBrain(
        store,
        original,
        judge_always(Severity.FINE),
        batch_size=1,
    )
    assert reopened_old.report()["worst"] == old_report["worst"] == "lost"
    assert reopened_old.report()["judgements"] == old_report["judgements"] == 1
    assert old_path.exists()
    brain.close()


def test_a_judgement_is_pinned_to_the_head_and_batch_it_observed(store: Store) -> None:
    """A slow judge can return after the worker has checkpointed again.

    Its evidence and verdict belong to the snapshot it actually saw, not to
    whichever head happens to be current when ``respond`` eventually runs.
    """
    import asyncio

    brain = SubBrain(store, a_contract())
    brain.wal.intent("Bash", "observed-call", {"command": "pytest"})
    observed_head = brain.branch.head
    heads_seen_by_judge: list[str] = []

    def moving_judge(contract, batch, view):
        heads_seen_by_judge.append(view.head.id)
        brain.checkpoint("the worker moved while judgement was slow")
        heads_seen_by_judge.append(view.head.id)
        return Judgement(Severity.WRONG, "the observed call was wrong")

    monitor = MonitorBrain(store, brain.contract, moving_judge, batch_size=1)
    judgement = monitor.tick()
    assert judgement is not None
    assert brain.branch.head != observed_head
    assert heads_seen_by_judge == [observed_head.id, observed_head.id]

    (pending,) = monitor.pending_actions
    assert pending.observed_head == observed_head.id
    assert [event["tool_use_id"] for event in pending.batch] == ["observed-call"]

    asyncio.run(monitor.respond(judgement, FakeClient()))
    assert [v.detail for v in store.state(observed_head.id).verdicts] == [
        "the observed call was wrong"
    ]
    brain.close()


def test_a_kill_between_judgement_and_response_is_replayed(store: Store) -> None:
    """Persisting the cursor before acting used to omit the action forever.

    The child dies after ``tick`` has persisted its judgement.  A fresh monitor
    must not re-bill/re-judge the batch, but must still deliver the pending
    verdict and intervention exactly once.
    """
    import asyncio

    brain = SubBrain(store, a_contract())
    brain.wal.intent("Bash", "t0", {"command": "wrong"})

    pid = os.fork()
    if pid == 0:  # pragma: no cover - deliberately never returns to pytest
        monitor = MonitorBrain(
            store,
            brain.contract,
            judge_always(Severity.WRONG, "persisted before death"),
            batch_size=1,
        )
        assert monitor.tick() is not None
        os.kill(os.getpid(), signal.SIGKILL)
    _, status = os.waitpid(pid, 0)
    assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL

    def must_not_rejudge(contract, batch, view):
        raise AssertionError("the persisted judgement should be replayed")

    restarted = MonitorBrain(store, brain.contract, must_not_rejudge, batch_size=1)
    assert len(restarted.pending_actions) == 1
    judgement, rung = asyncio.run(restarted.cycle(FakeClient()))
    assert judgement is not None and judgement.reason == "persisted before death"
    assert rung == "flagged-wrong"
    assert restarted.pending_actions == ()
    assert restarted.unjudged() == []
    assert [v.detail for v in brain.wake().unacked] == ["persisted before death"]
    brain.close()


def test_replay_does_not_duplicate_a_verdict_that_landed_before_the_kill(
    store: Store,
) -> None:
    """The other side of the boundary is judge-record-kill-clear.

    The durable action remains pending, but its stable verdict timestamp lets a
    restarted monitor recognize that the state was already annotated.
    """
    import asyncio

    brain = SubBrain(store, a_contract())
    brain.wal.intent("Bash", "t0", {})
    monitor = MonitorBrain(
        store,
        brain.contract,
        judge_always(Severity.WRONG, "one warning, not two"),
        batch_size=1,
    )
    assert monitor.tick() is not None
    (action,) = monitor.pending_actions
    monitor._record_action(action)  # the old process dies before clearing it
    assert len(store.state(action.observed_head).verdicts) == 1

    restarted = MonitorBrain(
        store,
        brain.contract,
        judge_always(Severity.FINE),
        batch_size=1,
    )
    asyncio.run(restarted.cycle(FakeClient()))

    verdicts = store.state(action.observed_head).verdicts
    assert [(v.detail, v.at) for v in verdicts] == [
        ("one warning, not two", action.verdict_at)
    ]
    brain.close()


def test_monitor_state_replacement_is_atomic(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure before replace leaves the previous complete state readable."""
    import taste.brains.monitor as monitor_module

    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, judge_always(Severity.FINE))
    monitor._save_state()
    path = monitor._state_path()
    before = path.read_bytes()

    monitor.state.interventions.append(("nudge", "new but unpublished state"))

    def fail_replace(source, destination):
        raise OSError("simulated death before rename")

    monkeypatch.setattr(monitor_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated death"):
        monitor._save_state()

    assert path.read_bytes() == before
    json.loads(path.read_text())
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))
    brain.close()


def test_cycle_does_not_block_the_workers_event_loop_on_a_slow_judge(store: Store) -> None:
    import asyncio

    brain = SubBrain(store, a_contract())
    brain.wal.intent("Bash", "t0", {})
    entered = threading.Event()
    release = threading.Event()

    def slow_judge(contract, batch, view):
        entered.set()
        assert release.wait(timeout=10)
        return Judgement(Severity.FINE, "ok")

    monitor = MonitorBrain(store, brain.contract, slow_judge, batch_size=1)

    async def exercise() -> None:
        task = asyncio.create_task(monitor.cycle(FakeClient()))
        assert await asyncio.to_thread(entered.wait, 10)
        # This coroutine is the stand-in for the SDK stream consumer.  It must
        # still get scheduled while the model-backed judge is thinking.
        streamed: list[str] = []
        await asyncio.sleep(0)
        streamed.append("message-consumed")
        assert streamed == ["message-consumed"] and not task.done()
        release.set()
        await task

    asyncio.run(exercise())
    brain.close()


# ------------------------------------------------------------------ terminal certification


class ScriptedTerminalJudge:
    def __init__(self, incremental: Severity = Severity.FINE) -> None:
        self.incremental = incremental
        self.terminal_calls: list[tuple[object, dict, list[dict]]] = []

    def __call__(self, contract, batch, view):
        return Judgement(self.incremental, "incremental finding", cost_usd=0.0)

    def judge_terminal(self, contract, state, context, findings):
        self.terminal_calls.append((state, context, findings))
        ids = tuple(item["id"] for item in findings)
        return TerminalDecision(
            Judgement(
                Severity.FINE,
                "the exact final state corrects the finding",
                cost_usd=0.0,
            ),
            resolved_finding_ids=ids,
        )


def test_terminal_assessment_can_be_fine_without_erasing_historical_worst(
    store: Store,
) -> None:
    import asyncio

    brain = SubBrain(store, a_contract())
    brain.install_contract()
    brain.checkpoint("durable contract")
    judge = ScriptedTerminalJudge(Severity.WRONG)
    monitor = MonitorBrain(store, brain.contract, judge, batch_size=1)
    brain.wal.intent("Bash", "bad-start", {"command": "pytest"})
    judgement = monitor.tick()
    assert judgement is not None
    asyncio.run(monitor.respond(judgement, FakeClient()))
    brain.branch.write("parser.py", "def parse(text):\n    return text\n")
    work_state = brain.checkpoint("corrected terminal work")

    assessment = asyncio.run(
        monitor.certify_terminal(
            work_state,
            context={"tests": {"command": "pytest", "passed": True}},
        )
    )

    assert assessment.state_id == work_state.id
    assert assessment.contract_digest == monitor.contract_digest
    assert assessment.acceptable
    assert assessment.judgement.severity is Severity.FINE
    assert assessment.resolved_finding_ids == assessment.finding_ids
    assert monitor.state.worst is Severity.WRONG
    report = monitor.report()
    assert report["worst"] == "wrong"
    assert report["current"] == "fine"
    assert report["current_state"] == work_state.id
    assert report["terminal_assessment"]["acceptable"] is True
    seen_state, seen_context, seen_findings = judge.terminal_calls[0]
    assert seen_state.id == work_state.id
    assert seen_context["terminal"]["tests"]["passed"] is True
    assert [item["severity"] for item in seen_findings] == ["wrong"]
    brain.close()


def test_terminal_assessment_is_durable_and_idempotent_for_the_same_state(
    store: Store,
) -> None:
    import asyncio

    brain = SubBrain(store, a_contract())
    brain.install_contract()
    work_state = brain.checkpoint("terminal work")
    judge = ScriptedTerminalJudge()
    monitor = MonitorBrain(store, brain.contract, judge)
    context = {"structured_status": "completed", "tests_passed": True}

    first = asyncio.run(monitor.certify_terminal(work_state, context=context))
    second = asyncio.run(monitor.certify_terminal(work_state, context=context))
    assert second == first
    assert len(judge.terminal_calls) == 1

    class MustNotRejudge(ScriptedTerminalJudge):
        def judge_terminal(self, contract, state, context, findings):
            raise AssertionError("a persisted State must not be re-billed")

    restarted = MonitorBrain(store, brain.contract, MustNotRejudge())
    restored = asyncio.run(restarted.certify_terminal(work_state, context=context))
    assert restored == first
    assert restarted.report()["terminal_assessment"]["id"] == first.id
    brain.close()


def test_monitor_cost_report_aggregates_persisted_calls_once_across_restart(
    store: Store,
) -> None:
    import asyncio

    class CostedJudge(ScriptedTerminalJudge):
        def __call__(self, contract, batch, view):
            return Judgement(Severity.FINE, "incremental", cost_usd=0.2)

        def judge_terminal(self, contract, state, context, findings):
            self.terminal_calls.append((state, context, findings))
            return TerminalDecision(
                Judgement(Severity.FINE, "terminal", cost_usd=0.3),
                resolved_finding_ids=tuple(item["id"] for item in findings),
            )

    brain = SubBrain(store, a_contract())
    brain.install_contract()
    brain.checkpoint("durable contract")
    judge = CostedJudge()
    monitor = MonitorBrain(store, brain.contract, judge, batch_size=1)
    empty_report = monitor.report()
    assert empty_report["model_calls"] == 0
    assert empty_report["cost_known"] is True
    assert empty_report["cost_usd"] == 0.0

    brain.wal.intent("Bash", "one-call", {"command": "pytest"})
    incremental = monitor.tick()
    assert incremental is not None
    asyncio.run(monitor.respond(incremental, FakeClient()))
    work_state = brain.checkpoint("terminal work")
    context = {"structured_status": "completed"}
    asyncio.run(monitor.certify_terminal(work_state, context=context))
    first_report = monitor.report()
    assert first_report["model_calls"] == 2
    assert first_report["cost_known"] is True
    assert first_report["cost_usd"] == pytest.approx(0.5)

    restarted = MonitorBrain(store, brain.contract, CostedJudge(), batch_size=1)
    replayed = asyncio.run(restarted.certify_terminal(work_state, context=context))
    assert replayed == monitor.state.terminal_assessments[-1]
    assert restarted.report()["model_calls"] == 2
    assert restarted.report()["cost_usd"] == pytest.approx(0.5)
    brain.close()


@pytest.mark.parametrize(
    "invalid_cost",
    [None, -0.1, float("nan"), float("inf"), True, "not-a-cost"],
    ids=["unknown", "negative", "nan", "infinite", "boolean", "text"],
)
def test_monitor_cost_report_fails_closed_for_persisted_unknown_or_invalid_cost(
    store: Store,
    invalid_cost: object,
) -> None:
    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, ScriptedTerminalJudge())
    monitor.state.judgements.append(
        Judgement(Severity.FINE, "persisted", cost_usd=invalid_cost)  # type: ignore[arg-type]
    )
    monitor._save_state()

    restarted = MonitorBrain(store, brain.contract, ScriptedTerminalJudge())
    report = restarted.report()
    assert report["model_calls"] == 1
    assert report["cost_known"] is False
    assert report["cost_usd"] is None
    brain.close()


def test_monitor_cost_report_fails_closed_for_unknown_terminal_call(
    store: Store,
) -> None:
    import asyncio

    class UnknownCostTerminalJudge(ScriptedTerminalJudge):
        def judge_terminal(self, contract, state, context, findings):
            return TerminalDecision(
                Judgement(Severity.FINE, "terminal cost unavailable"),
                resolved_finding_ids=tuple(item["id"] for item in findings),
            )

    brain = SubBrain(store, a_contract())
    brain.install_contract()
    work_state = brain.checkpoint("terminal work")
    monitor = MonitorBrain(store, brain.contract, UnknownCostTerminalJudge())
    asyncio.run(monitor.certify_terminal(work_state, context={}))

    report = monitor.report()
    assert report["model_calls"] == 1
    assert report["cost_known"] is False
    assert report["cost_usd"] is None
    brain.close()


def test_terminal_certification_is_async_safe_and_finishes_persistence_on_cancellation(
    store: Store,
) -> None:
    import asyncio

    entered = threading.Event()
    release = threading.Event()

    class SlowTerminalJudge(ScriptedTerminalJudge):
        def judge_terminal(self, contract, state, context, findings):
            entered.set()
            assert release.wait(timeout=10)
            return super().judge_terminal(contract, state, context, findings)

    brain = SubBrain(store, a_contract())
    brain.install_contract()
    work_state = brain.checkpoint("terminal work")
    monitor = MonitorBrain(store, brain.contract, SlowTerminalJudge())

    async def exercise() -> None:
        task = asyncio.create_task(
            monitor.certify_terminal(work_state, context={"tests_passed": True})
        )
        assert await asyncio.to_thread(entered.wait, 10)
        await asyncio.sleep(0)
        assert not task.done(), "the synchronous judge blocked the event loop"
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    restarted = MonitorBrain(store, brain.contract, ScriptedTerminalJudge())
    restored = asyncio.run(
        restarted.certify_terminal(work_state, context={"tests_passed": True})
    )
    assert restored.acceptable
    assert len(restarted.state.terminal_assessments) == 1
    brain.close()


@pytest.mark.parametrize("failure_mode", ["raise", "wrong-type", "incomplete-partition"])
def test_terminal_judge_failures_are_persisted_fail_closed(
    store: Store,
    failure_mode: str,
) -> None:
    import asyncio

    class BrokenTerminalJudge(ScriptedTerminalJudge):
        def judge_terminal(self, contract, state, context, findings):
            if failure_mode == "raise":
                raise RuntimeError("model unavailable")
            if failure_mode == "wrong-type":
                return Judgement(Severity.FINE, "not a terminal decision")
            return TerminalDecision(
                Judgement(Severity.FINE, "claims the old problem vanished")
            )

    brain = SubBrain(store, a_contract())
    brain.install_contract()
    brain.checkpoint("contract")
    judge = BrokenTerminalJudge(Severity.WRONG)
    monitor = MonitorBrain(store, brain.contract, judge, batch_size=1)
    brain.wal.intent("Bash", "bad", {})
    judgement = monitor.tick()
    assert judgement is not None
    asyncio.run(monitor.respond(judgement, FakeClient()))
    work_state = brain.checkpoint("terminal work")

    assessment = asyncio.run(monitor.certify_terminal(work_state, context={}))
    assert not assessment.acceptable
    assert assessment.failure.startswith("terminal judge failed:")
    assert assessment.judgement.severity is Severity.LOST
    assert assessment.unresolved_finding_ids == assessment.finding_ids

    restarted = MonitorBrain(store, brain.contract, ScriptedTerminalJudge())
    restored = asyncio.run(restarted.certify_terminal(work_state, context={}))
    assert restored == assessment
    assert restarted.report()["current"] == "lost"
    brain.close()


def test_terminal_certification_fails_closed_if_any_state_events_are_unjudged(
    store: Store,
) -> None:
    import asyncio

    brain = SubBrain(store, a_contract())
    brain.install_contract()
    brain.wal.intent("Bash", "never-reviewed", {})
    work_state = brain.checkpoint("unchecked terminal work")
    judge = ScriptedTerminalJudge()
    monitor = MonitorBrain(store, brain.contract, judge, batch_size=10)

    assessment = asyncio.run(monitor.certify_terminal(work_state, context={}))
    assert not assessment.acceptable
    assert "not completely judged" in assessment.failure
    assert judge.terminal_calls == []
    brain.close()


def test_schema_two_sidecar_is_atomically_upgraded_without_inventing_a_current_result(
    store: Store,
) -> None:
    brain = SubBrain(store, a_contract())
    brain.install_contract()
    brain.checkpoint("contract")
    monitor = MonitorBrain(store, brain.contract, ScriptedTerminalJudge())
    monitor._save_state()
    path = monitor._state_path()
    raw = json.loads(path.read_text())
    raw["schema"] = "taste.brains/MonitorState/2"
    raw.pop("terminal_assessments")
    path.write_text(json.dumps(raw))

    restarted = MonitorBrain(store, brain.contract, ScriptedTerminalJudge())
    migrated = json.loads(path.read_text())
    assert migrated["schema"] == "taste.brains/MonitorState/3"
    assert migrated["terminal_assessments"] == []
    assert restarted.report()["current"] is None
    brain.close()


def test_forced_drain_judges_the_final_partial_batch_before_lease_release(
    store: Store,
) -> None:
    import asyncio

    brain = SubBrain(store, a_contract())
    seen: list[str] = []

    def judge(contract, batch, view):
        seen.extend(e["tool_use_id"] for e in batch)
        return Judgement(Severity.FINE, "terminal batch is fine")

    monitor = MonitorBrain(store, brain.contract, judge, batch_size=10)
    brain.wal.intent("Bash", "last", {})
    assert monitor.tick() is None and brain.branch.holder is not None

    drained = asyncio.run(monitor.drain(FakeClient(), final=True))
    assert [j.reason for j, _ in drained] == ["terminal batch is fine"]
    assert seen == ["last"]
    assert brain.branch.holder is not None, "runtime checkpoints before releasing the lease"
    brain.close()


# ------------------------------------------------- the monitor does not act


@pytest.mark.parametrize(
    "severity", [Severity.FINE, Severity.DRIFTING, Severity.WRONG, Severity.LOST]
)
def test_no_severity_touches_the_worker(store: Store, severity) -> None:
    """The monitor observes. The central brain acts. No rung, at any severity.

    With a lever, this was measured breaking the thing it watched: the monitor
    interrupted its worker mid-stream, every turn came back ``aborted_*`` with
    no ``end_turn``, the run could never terminate -- and the monitor then
    judged the worker for the interruption it had itself caused.
    """
    import asyncio

    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, judge_always(severity), batch_size=1)
    client = FakeClient()

    brain.wal.intent("Bash", "t1", {"command": "x"})
    drained = asyncio.run(monitor.drain(client, final=False))

    assert client.calls == [], f"{severity.value} acted on the worker"
    assert [j.severity for j, _ in drained] == [severity]
    expected = "none" if severity is Severity.FINE else f"flagged-{severity.value}"
    assert [r for _, r in drained] == [expected]
    brain.close()


def test_the_verdict_still_reaches_the_worker_and_the_central_brain(store: Store) -> None:
    """Reporting is a different lever, not a weaker one.

    ``store.judge`` needs no lease, so the verdict lands on the observed state
    while the worker still holds its branch: the worker reads it on its next
    wake, and the central brain reads it in the WorkerReport.
    """
    import asyncio

    brain = SubBrain(store, a_contract())
    brain.checkpoint("some work")
    monitor = MonitorBrain(store, brain.contract, judge_always(Severity.WRONG), batch_size=1)

    brain.wal.intent("Bash", "t1", {"command": "x"})
    asyncio.run(monitor.drain(FakeClient(), final=False))

    assert [v.detail for v in brain.wake().unacked] == ["because"]
    report = monitor.report()
    assert report["worst"] == "wrong"
    assert [r for r, _ in report["interventions"]] == ["flagged-wrong"]
    brain.close()


def test_a_drain_ends_because_judging_makes_no_new_work(store: Store) -> None:
    """The old loop was fed by the monitor's own queries to the worker.

    A verdict produced an answer, the answer was new events, the events were
    the next batch. With nothing sent, the unjudged tail only shrinks.
    """
    import asyncio

    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, judge_always(Severity.WRONG), batch_size=1)

    for i in range(5):
        brain.wal.intent("Bash", f"t{i}", {"command": "x"})
    drained = asyncio.run(monitor.drain(FakeClient(), final=True))

    assert len(drained) == 5
    assert monitor.unjudged() == []
    brain.close()


# ------------------------------------------------- what is worth a model call


def _sdk_turn(message_type: str, **fields):
    return {"kind": "sdk_message", "message_type": message_type, **fields}


def test_token_deltas_are_not_worth_a_model_call(store: Store) -> None:
    """A StreamEvent is the transport, not the work.

    Measured on one live run: 2,277 recorded events, 2,160 of them
    ``StreamEvent`` -- 228 model calls, about sixteen minutes of judging for
    seventy-five seconds of work, while the worker's terminal drain (and so its
    exit) waited behind it. The finished ``AssistantMessage`` that follows each
    burst carries the whole tool call anyway.
    """
    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, judge_always(Severity.FINE))

    for i in range(30):
        brain.branch.turn(**_sdk_turn("StreamEvent", seq=i))
    brain.branch.turn(**_sdk_turn("AssistantMessage", seq="done"))
    brain.wal.intent("Bash", "t1", {"command": "pytest"})

    seen = monitor.observations()
    assert [e.get("message_type") or e["kind"] for e in seen] == [
        "AssistantMessage",
        "tool_intent",
    ]
    brain.close()


def test_narrowing_does_not_rewind_a_restarted_monitors_cursor(store: Store) -> None:
    """The cursor is content-addressed, so the filter has to be a constant.

    A monitor that read the wide stream and then restarted reading a narrow one
    would find its saved fingerprints no longer a prefix, call that a rollback,
    and re-judge work it had already paid for.
    """
    brain = SubBrain(store, a_contract())
    monitor = MonitorBrain(store, brain.contract, judge_always(Severity.FINE), batch_size=1)

    brain.branch.turn(**_sdk_turn("AssistantMessage", seq="one"))
    for i in range(10):
        brain.branch.turn(**_sdk_turn("StreamEvent", seq=i))
    assert monitor.tick() is not None
    judged = list(monitor.state.fingerprints)

    restarted = MonitorBrain(
        store, brain.contract, judge_always(Severity.FINE), batch_size=1
    )
    assert restarted.state.fingerprints == judged
    assert restarted.unjudged() == [], "a narrowed stream must not look like a rollback"
    brain.close()
