"""A monitor's judgment has to reach the brain it judged.

``store.judge`` writes a verdict as a note on the state, which is durable and
lease-free -- correct, but it was not on the path a brain actually takes when
it wakes. ``resume`` reported a clean branch, so a brain carried on building
on a state the monitor had already failed. Worse, verdicts are keyed by state:
one more checkpoint moved the head and the failure vanished behind an ancestor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taste.memstore import Store, Verdict


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "repo", "s1")
    yield s
    s.close()


def test_a_brain_wakes_knowing_the_monitor_failed_it(store: Store) -> None:
    b = store.branch("worker")
    st = b.checkpoint("work I think is fine")

    # The monitor is another process; it never touches the branch.
    store.judge(st, Verdict(status="fail", by="monitor", detail="B is a dead end"))

    r = b.resume()
    assert r.failed, "resume looked clean on a state the monitor had failed"
    assert [v.detail for v in r.unacked] == ["B is a dead end"]
    assert [v.detail for v in r.verdicts] == ["B is a dead end"]


def test_a_failure_does_not_vanish_behind_the_next_checkpoint(store: Store) -> None:
    """Verdicts are keyed by state, so a brain that checkpointed once more used
    to lose sight of the failure entirely."""
    b = store.branch("worker")
    failed = b.checkpoint("the state that goes wrong")
    store.judge(failed, Verdict(status="fail", by="monitor", detail="regression"))

    b.checkpoint("carrying on, unaware")

    assert b.head.verdicts == [], "the head itself is unjudged, as expected"
    r = b.resume()
    assert r.failed
    assert [v.detail for v in r.unacked] == ["regression"]


def test_an_acknowledged_failure_stops_being_news(store: Store) -> None:
    b = store.branch("worker")
    st = b.checkpoint("work")
    store.judge(st, Verdict(status="fail", by="monitor", detail="dead end"))

    assert b.resume().failed
    b.acknowledge()

    r = b.resume()
    assert not r.failed
    assert r.unacked == ()
    # still on the record, just not news
    assert [v.detail for v in st.verdicts] == ["dead end"]


def test_a_second_verdict_on_an_acknowledged_state_is_still_delivered(
    store: Store,
) -> None:
    """Counted per state, not watermarked by state: a monitor that judges an
    older state after the brain moved on must still be heard."""
    b = store.branch("worker")
    st = b.checkpoint("work")
    store.judge(st, Verdict(status="pass", by="monitor", detail="looks fine"))
    b.acknowledge()
    assert b.resume().unacked == ()

    store.judge(st, Verdict(status="fail", by="slower-monitor", detail="actually no"))

    r = b.resume()
    assert r.failed
    assert [v.detail for v in r.unacked] == ["actually no"]


def test_acknowledgement_survives_the_death_of_the_brain(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    s = Store.open(root, "s1")
    b = s.branch("worker")
    st = b.checkpoint("work")
    s.judge(st, Verdict(status="fail", by="monitor", detail="dead end"))
    b.acknowledge()
    s.close()

    again = Store.open(root, "s1")
    assert again.branch("worker").resume().unacked == ()
    again.close()


def test_a_monitor_needs_no_lease_to_be_heard(tmp_path: Path) -> None:
    """The whole point: the judge and the judged are different processes."""
    root = tmp_path / "repo"
    writer = Store.open(root, "s1")
    b = writer.branch("worker")
    st = b.checkpoint("work in progress")

    monitor = Store.open(root, "s1")
    view = monitor.view("worker")
    monitor.judge(view.head, Verdict(status="fail", by="monitor", detail="stop"))
    monitor.close()

    assert st.id == view.head.id
    assert b.resume().failed
    writer.close()
