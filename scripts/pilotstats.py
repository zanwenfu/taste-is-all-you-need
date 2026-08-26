"""Read one sweep's ledger and answer the pre-declared questions.

    python scripts/pilotstats.py --root /root/pilot40c

Committed, not improvised: every pilot before this one was analysed by
ad-hoc scripts on the experiment host, and the declared event unit lived
only in those scripts — which is how the pipeline shipped a different unit
than the registration (bug C2). The analysis the gate depends on belongs in
version control with the gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI, pure stdlib — the box's venv has no scipy."""
    if n == 0:
        return (0.0, 1.0)

    def binom_cdf(x: int, n: int, p: float) -> float:
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(x + 1))

    def bisect_decreasing(f, target):
        # f monotone decreasing in p; find p with f(p) = target.
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if f(mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    # Exact CP: lower solves P(X >= k | p) = alpha/2 (increasing in p, so
    # bisect its complement); upper solves P(X <= k | p) = alpha/2
    # (decreasing in p). The first version got the k=0 upper wrong and
    # printed [0%, 100%] for 0/40 — an interval so obviously broken it was
    # caught on sight, which is the lucky kind of wrong.
    lower = 0.0 if k == 0 else bisect_decreasing(
        lambda p: binom_cdf(k - 1, n, p), 1 - alpha / 2
    )
    upper = 1.0 if k == n else bisect_decreasing(
        lambda p: binom_cdf(k, n, p), alpha / 2
    )
    return (lower, upper)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--gate-lambda", type=float, default=0.30)
    ap.add_argument("--gate-bearing", type=float, default=0.25)
    args = ap.parse_args()
    root = Path(args.root)

    rows = []
    for f in sorted((root / "ledger").glob("*.json")):
        d = json.loads(f.read_text())
        if isinstance(d, dict) and "status" in d:
            rows.append(d)
    evidence = {}
    for f in sorted((root / "ledger" / "evidence").glob("*.json")):
        d = json.loads(f.read_text())
        evidence[d["instance_id"]] = d

    n = len(rows)
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    spend = sum(r.get("billed_usd") or 0 for r in rows)

    graded = [r for r in rows if r.get("score") is not None]
    resolved = [r for r in graded if r["score"] == 1.0]
    steps_pos = sum(1 for r in rows if (r.get("steps_passed") or 0) > 0)

    obs = sum(e["observations"] for e in evidence.values())
    raw = sum(len(e["episodes"]) for e in evidence.values())
    declared = sum(e.get("contamination_events_declared", 0) for e in evidence.values())
    bearing = [i for i, e in evidence.items() if e.get("contamination_events_declared", 0) > 0]
    unrouted = [i for i, e in evidence.items() if not e.get("routed")]
    never = sum(len(e["never_passed"]) for e in evidence.values())
    members = {i: e for i, e in evidence.items() if e["never_passed"]}

    sil_att = sum(e["silence"].get("silent_attributed", 0) for e in evidence.values())
    sil_unatt = sum(e["silence"].get("silent_unattributed", 0) for e in evidence.values())
    measured = [i for i, e in evidence.items() if e["silence"].get("method") not in (None, "none")]

    lam = declared / n if n else 0.0
    frac = len(bearing) / n if n else 0.0
    lo, hi = clopper_pearson(len(bearing), n)

    print(f"cells: {n}  {by_status}   spend: ${spend:.2f}")
    print(f"steps>0: {steps_pos}/{n}")
    print(f"RESOLVE RATE: {len(resolved)}/{len(graded)} graded = "
          f"{len(resolved)/len(graded):.1%}" if graded else "RESOLVE RATE: nothing graded")
    if unrouted:
        print(f"!! UNROUTED CELLS (invalid, exclude): {unrouted}")
    print(f"observations: {obs}   episodes raw: {raw}   DECLARED UNIT: {declared}")
    print(f"lambda-hat (declared/run): {lam:.3f}   gate >= {args.gate_lambda}: "
          f"{'PASS' if lam >= args.gate_lambda else 'FAIL'}")
    print(f"bearing runs: {len(bearing)}/{n} = {frac:.1%}  CI95 [{lo:.1%}, {hi:.1%}]   "
          f"gate >= {args.gate_bearing:.0%}: {'PASS' if frac >= args.gate_bearing else 'FAIL'}")
    print(f"bearing instances: {sorted(bearing)}")
    print(f"silence: attributed={sil_att} co-occurrence-bound={sil_unatt} "
          f"(measured on {len(measured)}/{n} cells; the rest typed UNKNOWN)")
    print(f"oracle holes (never_passed): {never} across {len(members)} instances")
    for i, e in sorted(members.items()):
        print(f"    {i}: {len(e['never_passed'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
