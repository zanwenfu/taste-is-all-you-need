"""The pre-declared recovery-policy contrast, computed exactly as registered.

    python scripts/contrast.py --a /root/pilot40d --b /root/contrast40_A0 \
        --label-a A3 --label-b A0

Committed BEFORE the contrast sweeps were unblinded (prereg amendment 1 §E):
the analysis a result gets is the one that was written when nobody knew the
result. Primary endpoint: final-state contamination (graded PASS_TO_PASS
failures in the official verdict), paired per instance, exact two-sided
sign test. Co-primary: distinct onsets (exposure). Everything else printed
here is descriptive.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(root: Path) -> dict[str, dict]:
    """A sweep's evidence, preferring re-scored sidecars where they exist.

    ``rescore.py`` writes corrected measurements of archived runs to
    ``rescored/evidence`` and never overwrites the originals. Reading only
    the originals would silently analyse the pre-fix numbers — the seven
    "ungradable" cells would stay dropped from the primary endpoint after
    the grader that dropped them was fixed. Which cells came from a rescore
    is printed, so the provenance of every row is on the page.
    """
    cells: dict[str, dict] = {}
    for f in sorted((root / "ledger" / "evidence").glob("*.json")):
        d = json.loads(f.read_text())
        cells[d["instance_id"]] = d
    rescored = root / "rescored" / "evidence"
    swapped = []
    if rescored.exists():
        for f in sorted(rescored.glob("*.json")):
            d = json.loads(f.read_text())
            cells[d["instance_id"]] = d
            swapped.append(d["instance_id"])
    if swapped:
        print(f"  [{root.name}] using re-scored sidecars for: {', '.join(swapped)}")
    return cells


def final_state_contamination(evidence: dict) -> int | None:
    """Graded P2P failures in the official verdict. None = ungraded."""
    grade = evidence.get("grade") or {}
    p2p = grade.get("pass_to_pass")
    if not p2p:
        return None
    passed, total = (int(x) for x in p2p.split("/"))
    return total - passed


def onsets(evidence: dict) -> int:
    return len({e.get("onset_seq") for e in evidence.get("episodes", [])})


def sign_test(diffs: list[int]) -> tuple[int, int, float]:
    """Exact two-sided sign test on nonzero paired differences."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(pos, neg) + 1)) / 2**n
    return pos, neg, min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    a, b = load(Path(args.a)), load(Path(args.b))
    shared = sorted(set(a) & set(b))
    print(f"paired instances: {len(shared)}  ({args.label_a}: {len(a)}, {args.label_b}: {len(b)})")

    fs_diffs, on_diffs, rows = [], [], []
    for inst in shared:
        ea, eb = a[inst], b[inst]
        fa, fb = final_state_contamination(ea), final_state_contamination(eb)
        oa, ob = onsets(ea), onsets(eb)
        rows.append((inst, fa, fb, oa, ob, ea.get("resolved"), eb.get("resolved")))
        if fa is not None and fb is not None:
            fs_diffs.append(fb - fa)
        on_diffs.append(ob - oa)

    header = "{:34s} {:>5s} {:>5s} {:>4s} {:>4s} {:>6s} {:>6s}"
    print(header.format("instance", f"fs:{args.label_a}", f"fs:{args.label_b}",
                        f"on:{args.label_a}"[:4], f"on:{args.label_b}"[:4],
                        f"r:{args.label_a}", f"r:{args.label_b}"))
    for inst, fa, fb, oa, ob, ra, rb in rows:
        print(header.format(inst[:34], str(fa), str(fb), str(oa), str(ob), str(ra), str(rb)))

    pos, neg, p = sign_test(fs_diffs)
    print(f"\nPRIMARY  final-state contamination {args.label_b} vs {args.label_a}: "
          f"{args.label_b} worse on {pos}, better on {neg}, ties {len(fs_diffs)-pos-neg}; "
          f"exact sign test p = {p:.4f}")
    pos, neg, p = sign_test(on_diffs)
    print(f"CO-PRIMARY  onset exposure {args.label_b} vs {args.label_a}: "
          f"{args.label_b} more on {pos}, fewer on {neg}, ties {len(on_diffs)-pos-neg}; "
          f"exact sign test p = {p:.4f}")
    ra = sum(1 for r in rows if r[5]) ; rb = sum(1 for r in rows if r[6])
    print(f"resolve: {args.label_a} {ra}/{len(rows)}  {args.label_b} {rb}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
