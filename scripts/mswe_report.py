"""What one mini-swe-agent sweep root says, in the paper's own quantities.

    python scripts/mswe_report.py sonnet=/root/mswe40_sonnet_s1+/root/mswe40_sonnet_s2+/root/mswe40_sonnet_s3 [gpt=...]

Per root: the substrate-table row (same code path as the paper's Table 1),
then per cell: the scaffold's exit status, commands issued, observations
committed, declared events, failures visible in the final tree (the
undercount pair from Figure 2), resolve, and spend. Reads only the ledger,
the evidence sidecars, and the per-cell ``manifest.json`` the driver writes.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from substrate_table import load, row_from, spend


def manifests(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    # make_prepare's layout: runs/<task>/<arm>/t<trial>/.git/taste/manifest.json
    for f in glob.glob(str(root / "runs" / "*" / "*" / "t*" / ".git" / "taste" / "manifest.json")):
        out.setdefault(Path(f).parts[-6], json.loads(Path(f).read_text()))
    return out


def ledger(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in glob.glob(str(root / "ledger" / "*.json")):
        d = json.loads(Path(f).read_text())
        if isinstance(d, dict) and d.get("task"):
            out[d["task"]] = d
    return out


def visible_failures(d: dict) -> int | None:
    """Failures the final tree shows the grader, net of baseline-dead tests."""
    g = d.get("grade") or {}
    dead = set(d.get("never_passed") or [])
    if "grade_failed" in d:
        return len(set(d.get("grade_failed") or []) - dead)
    if g.get("pass_to_pass"):
        p, t = (int(x) for x in g["pass_to_pass"].split("/"))
        return max(0, t - p - len(dead))
    return None


def parse_spec(spec: str) -> tuple[str, list[Path]]:
    """``/root/a`` or ``label=/root/a+/root/b`` (shards merged into one set)."""
    label, _, paths = spec.rpartition("=")
    roots = [Path(p) for p in paths.split("+") if p]
    return (label or f"mswe:{roots[0].name}"), roots


def merged(fn, roots: list[Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in roots:
        for k, v in fn(r).items():
            out.setdefault(k, v)
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    print("| substrate: arm | graded | resolved (strict) | resolved (rot-aware) | events | incidents | bearing runs | contaminated trees | spend |")
    print("|---|---|---|---|---|---|---|---|---|")
    for spec in sys.argv[1:]:
        label, roots = parse_spec(spec)
        print(row_from(label, merged(load, roots), sum(spend(r) for r in roots)))
    for spec in sys.argv[1:]:
        label, roots = parse_spec(spec)
        ev, man, led = merged(load, roots), merged(manifests, roots), merged(ledger, roots)
        print(f"\n== {label}: {' + '.join(str(r) for r in roots)} ==")
        print(f"{'instance':32s} {'exit':16s} {'cmds':>5s} {'obs':>4s} {'events':>6s} {'visible':>7s} {'resolved':>8s} {'$':>7s}  status")
        totals = {"cmds": 0, "obs": 0, "events": 0, "visible": 0, "bearing": 0, "resolved": 0, "graded": 0, "usd": 0.0}
        for inst in sorted(set(ev) | set(man) | set(led)):
            d, m, r = ev.get(inst, {}), man.get(inst, {}), led.get(inst, {})
            events = int(d.get("contamination_events_declared") or 0)
            vis = visible_failures(d) if d else None
            res = d.get("resolved")
            usd = float(r.get("billed_usd") or m.get("cost_usd") or 0.0)
            totals["cmds"] += int(m.get("commands") or 0)
            totals["obs"] += int(m.get("observations") or 0)
            totals["events"] += events
            totals["visible"] += int(vis or 0)
            totals["bearing"] += 1 if events else 0
            totals["resolved"] += 1 if res else 0
            totals["graded"] += 1 if res is not None else 0
            totals["usd"] += usd
            print(f"{inst:32s} {m.get('exit_status') or r.get('failure_reason') or ''!s:16s} {m.get('commands', ''):>5} {m.get('observations', ''):>4} {events:>6d} {'' if vis is None else vis:>7} {'' if res is None else str(res):>8s} {usd:>7.2f}  {r.get('status', '')}")
        n = max(1, len(set(ev) | set(man) | set(led)))
        print(f"{'TOTAL':32s} {'':16s} {totals['cmds']:>5d} {totals['obs']:>4d} {totals['events']:>6d} {totals['visible']:>7d} {totals['resolved']:>3d}/{totals['graded']:<4d} {totals['usd']:>7.2f}  "
              f"bearing {totals['bearing']}/{n}; events visible in final state {totals['visible']}/{totals['events']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
