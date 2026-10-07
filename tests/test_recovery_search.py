"""The binary search for the point of no return, as a pure function of what the probes found."""

from __future__ import annotations

import pytest

from taste.recovery_study.search import Probe, lost_bound, plan, result, wilson


def settled(successes, trials=8):
    return Probe(successes=successes, trials=trials)


def test_wilson_intervals_match_known_values():
    low, high = wilson(4, 8)
    assert (round(low, 4), round(high, 4)) == (0.2152, 0.7848)
    low, high = wilson(0, 16)
    assert low == 0.0 and round(high, 4) == 0.1936
    assert wilson(0, 0) == (0.0, 1.0) and wilson(8, 8)[1] == 1.0
    # Sixteen branches without a success: "lost" means v <= 0.17 at 95%, one-sided.
    assert round(lost_bound(16), 3) == 0.171


def test_the_first_probe_is_the_middle_and_the_search_halves_toward_the_boundary():
    first = plan(20, {}, k=8)
    assert (first.status, first.targets, first.low, first.high) == ("searching", {10: 8}, 0, 20)
    assert first.next_steps == (5, 15)
    after = plan(20, {10: settled(3)}, k=8)
    assert after.targets == {15: 8} and (after.low, after.high) == (10, 20)
    after = plan(20, {10: settled(3), 15: settled(0)}, k=8)
    assert after.targets == {12: 8} and after.next_steps == (11, 13)
    waiting = plan(20, {10: settled(3), 15: Probe(pending=True)}, k=8)
    assert waiting.status == "waiting" and waiting.targets == {}


def test_adjacent_sides_are_confirmed_with_extra_branches_then_done():
    probes = {10: settled(4), 15: settled(0), 12: settled(0), 11: settled(4)}
    confirming = plan(20, probes, k=8, extra=8)
    assert confirming.status == "confirming" and confirming.targets == {11: 16, 12: 16}
    probes.update({11: settled(9, 16), 12: settled(0, 16)})
    done = plan(20, probes, k=8, extra=8)
    assert (done.status, done.low, done.high) == ("done", 11, 12)
    summary = result(20, probes, v0=(1, 2), k=8, extra=8)
    assert summary["decisive_step"] == 12 and summary["decisive_fraction"] == 0.6
    assert summary["rewind"] == {"highest_v": 11, "latest": 11} and summary["recoverable"]
    assert summary["monotone"] and summary["earlier_drops"] == []
    assert [point["step"] for point in summary["curve"]] == [0, 10, 11, 12, 15]
    assert summary["curve"][0]["source"] == "calibration" and summary["curve"][3]["lost_upper_95"] == round(lost_bound(16), 6)


def test_a_lost_side_that_wins_with_more_branches_moves_the_search_on():
    probes = {10: settled(4), 15: settled(0), 12: settled(1, 16), 11: settled(8, 16)}
    moved = plan(20, probes, k=8, extra=8)
    assert (moved.status, moved.low, moved.high, moved.targets) == ("searching", 12, 15, {13: 8})


def test_a_curve_that_recovers_after_a_drop_is_searched_past_the_drop_and_the_drop_reported():
    # A scan every 4 steps: winnable at 4 and 16, lost at 8, 12 and 20.
    probes = {4: settled(4), 8: settled(0), 12: settled(0), 16: settled(4), 20: settled(0)}
    found = plan(24, probes, k=8, scan=(4, 8, 12, 16, 20))
    assert (found.low, found.high, found.targets[18]) == (16, 20, 8)
    probes.update({18: settled(0), 17: settled(0, 16), 16: settled(8, 16)})
    summary = result(24, probes, k=8, extra=8, scan=(4, 8, 12, 16, 20))
    assert summary["status"] == "done" and summary["decisive_step"] == 17
    assert summary["earlier_drops"] == [{"after": 4, "by": 8}] and not summary["monotone"]
    # Without the scan the search assumes a monotone curve and stops at the first drop.
    alone = {12: settled(0), 6: settled(8, 16), 9: settled(0), 7: settled(0, 16)}
    assert result(24, alone, k=8, extra=8)["decisive_step"] == 7


def test_the_highest_v_rewind_point_prefers_the_later_step_on_a_tie():
    probes = {5: settled(6), 9: settled(2), 11: settled(6), 12: settled(0, 16)}
    probes[11] = settled(12, 16)
    assert result(20, probes, k=8)["rewind"] == {"highest_v": 11, "latest": 11}
    probes[5] = settled(8, 8)
    assert result(20, probes, k=8)["rewind"]["highest_v"] == 5


def test_a_step_without_faithful_branches_censors_the_search_there():
    probes = {10: Probe(unusable=True), 5: settled(4), 7: settled(4), 8: Probe(unusable=True)}
    found = plan(20, probes, k=8)
    assert (found.status, found.low, found.limit) == ("censored", 7, 8)
    summary = result(20, probes, k=8)
    assert summary["decisive_step"] is None and summary["rewind"]["latest"] == 7


def test_an_unfaithful_step_below_a_settled_probe_is_only_skipped():
    probes = {10: settled(4), 15: settled(0), 12: Probe(unusable=True)}
    found = plan(20, probes, k=8)
    assert found.limit == 20 and found.targets == {13: 8}
    probes.update({13: settled(0), 11: Probe(unusable=True)})
    assert plan(20, probes, k=8).status == "unresolved"


def test_a_probe_that_cannot_get_more_branches_counts_with_what_it_has():
    probes = {10: settled(4), 15: settled(0), 11: Probe(successes=2, trials=5, exhausted=True),
              12: Probe(successes=0, trials=12, exhausted=True)}
    assert plan(20, probes, k=8, extra=8).status == "done"


@pytest.mark.parametrize("steps", [2, 3, 31])
def test_every_search_ends_at_adjacent_steps(steps):
    """Driven by a monotone truth, the search confirms the boundary within log2(T) + 3 rounds."""
    truth = steps // 2
    probes, rounds = {}, 0
    while True:
        found = plan(steps, probes, k=8, extra=8)
        if found.status == "done":
            break
        for step, wanted in found.targets.items():
            probes[step] = settled(wanted // 2 if step <= truth else 0, wanted)
        rounds += 1
        assert rounds <= steps.bit_length() + 3
    assert found.high == truth + 1 and found.low == truth
