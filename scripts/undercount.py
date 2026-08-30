"""The undercount, in units that match.

    python scripts/undercount.py A3=/root/pilot40d A0=/root/contrast40_A0 ...

"The timeline recorded 184 events; the final patch exposed one" is two true
counts of two different objects. An event is a (test function, onset) pair,
so one test broken three times is three events; a final-state failure is one
test failing at one moment, so the same test can contribute at most one. The
ratio of the two is not a rate, and calling it a capture rate invites a
reviewer to divide them and find the units do not cancel.

Three matched statements replace it, each computed per run and then pooled:

  runs      of the runs that broke something, how many still show anything
            when the benchmark reads the tree
  tests     of the distinct test functions broken during a run, how many are
            still failing at the end (pooled over runs, and per run)
  events    the raw event count, reported as exposure only -- never as a
            denominator for the visible failures

Parametrised variants collapse to their base id on both sides
(``test_x[a-1]`` and ``test_x[b-2]`` are one test function), which is the
permissive direction: it can only make the capture number larger, so a small
capture number computed this way is not an artifact of id splitting.

A cell whose whole suite died at grade time (every graded test failing) is
reported on its own line and enters no numerator and no denominator: a dead
suite is an infrastructure failure that would otherwise read as perfect
capture, exactly where contamination is most likely.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path


def base_test_id(test_id: str) -> str:
    """Drop a parametrisation suffix: ``test_x[a-1]`` -> ``test_x``."""
    return re.sub(r"\[.*\]$", "", str(test_id).strip())


def load(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in glob.glob(str(root / "rescored" / "evidence" / "*.json")) + glob.glob(str(root / "ledger" / "evidence" / "*.json")):
        d = json.loads(Path(f).read_text())
        out.setdefault(d["instance_id"], d)  # rescored first
    return out


def _failed_by_count(evidence: dict) -> int:
    """Graded failures the sidecar knows only as a count, net of dead tests."""
    grade = evidence.get("grade") or {}
    p2p = grade.get("pass_to_pass") or ""
    if "/" not in p2p:
        return 0
    passed, total = (int(x) for x in p2p.split("/"))
    return max(0, total - passed - len(evidence.get("never_passed") or []))


def suite_died(evidence: dict) -> bool:
    """Every graded test failing is an environment failure, not contamination."""
    grade = evidence.get("grade") or {}
    p2p = grade.get("pass_to_pass") or ""
    if "/" in p2p:
        passed, total = (int(x) for x in p2p.split("/"))
        if total and passed == 0:
            return True
    failed = {base_test_id(t) for t in (evidence.get("grade_failed") or [])}
    return bool(failed) and "/" in p2p and len(failed) >= int(p2p.split("/")[1])


def analyse(label: str, evidence: dict[str, dict]) -> dict:
    """The three matched statements for one evidence set (a root, or shards)."""
    rows, dead_cells = [], []
    for instance, d in sorted(evidence.items()):
        broke = {base_test_id(e.get("probe")) for e in (d.get("episodes") or []) if e.get("probe")}
        if not broke:
            continue
        if suite_died(d):
            dead_cells.append(instance)
            continue
        never = {base_test_id(t) for t in (d.get("never_passed") or [])}
        survived = {base_test_id(t) for t in (d.get("grade_failed") or [])} - never
        # Older sidecars record the grade as counts only, with no list of
        # failing ids. There the intersection is not computable, and calling
        # it zero would report the most favourable possible answer from an
        # absent field -- the failure mode this whole file exists to avoid.
        unknown = not d.get("grade_failed") and _failed_by_count(d) > 0
        rows.append({
            "instance": instance,
            "events": int(d.get("contamination_events_declared") or 0),
            "broke": len(broke),
            "captured": len(broke & survived),
            "unknown": unknown,
            "failed_at_grade": _failed_by_count(d) if unknown else len(survived),
        })
    known = [r for r in rows if not r["unknown"]]
    return {
        "label": label,
        "bearing": len(rows),
        "runs_captured": sum(1 for r in rows if r["captured"]),
        "runs_unknown": sum(1 for r in rows if r["unknown"]),
        "tests_broken": sum(r["broke"] for r in known),
        "tests_captured": sum(r["captured"] for r in known),
        "events": sum(r["events"] for r in rows),
        "dead_suites": dead_cells,
        "rows": rows,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    results = []
    for spec in sys.argv[1:]:
        label, _, paths = spec.rpartition("=")
        roots = [Path(p) for p in paths.split("+")]
        merged: dict[str, dict] = {}
        for r in roots:
            for k, v in load(r).items():
                merged.setdefault(k, v)
        results.append(analyse(label or roots[0].name, merged))

    for res in results:
        print(f"\n== {res['label']} ==")
        print(f"{'instance':32s} {'events':>6s} {'tests broken':>12s} {'still failing':>13s}")
        for r in res["rows"]:
            still = "? (ids unrecorded)" if r["unknown"] else str(r["captured"])
            print(f"{r['instance']:32s} {r['events']:>6d} {r['broke']:>12d} {still:>13s}")
        if res["dead_suites"]:
            print(f"  excluded, suite dead at grade time: {', '.join(res['dead_suites'])}")
        b, rc, ru = res["bearing"], res["runs_captured"], res["runs_unknown"]
        tb, tc = res["tests_broken"], res["tests_captured"]
        print(f"  runs   : {rc}/{b} bearing runs still show a broken test at the end"
              + (f" ({100 * rc / b:.0f}%)" if b else "")
              + (f"; {ru} more failed a graded test whose id the sidecar did not record" if ru else ""))
        print(f"  tests  : {tc}/{tb} distinct test functions broken during a run are still failing at the end"
              + (f" ({100 * tc / tb:.1f}%)" if tb else "")
              + (f", over the {b - ru} run(s) with recorded ids" if ru else ""))
        print(f"  events : {res['events']} declared (exposure only -- not a denominator for the {tc} above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
