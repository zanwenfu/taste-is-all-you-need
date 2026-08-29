"""One row per (substrate, arm): the numbers the two-substrate table needs.

Reads evidence sidecars and the ledger, exactly as pilotstats and the paper
figures do, so the table cannot drift from them::

    python scripts/substrate_table.py verified:rollback=/root/pilot40d live:rollback=/root/live40_A3

Columns: cells graded, resolved (of the slice size), declared events,
incidents (distinct onsets), bearing runs, cells with a contaminated final
tree (grade-based, net of baseline-dead tests), total billed USD.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def load(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in glob.glob(str(root / "rescored" / "evidence" / "*.json")) + glob.glob(str(root / "ledger" / "evidence" / "*.json")):
        d = json.loads(Path(f).read_text())
        out.setdefault(d["instance_id"], d)  # rescored first
    return out


def spend(root: Path) -> float:
    total = 0.0
    for f in glob.glob(str(root / "ledger" / "*.json")):
        d = json.loads(Path(f).read_text())
        if isinstance(d, dict) and d.get("task") and d.get("billed_usd") is not None:
            total += d["billed_usd"]
    return total


def row(label: str, root: Path, slice_size: int = 40) -> str:
    ev = load(root)
    graded = [d for d in ev.values() if d.get("resolved") is not None]
    resolved = sum(1 for d in graded if d["resolved"])
    events = sum(int(d.get("contamination_events_declared") or 0) for d in ev.values())
    incidents = sum(len({e.get("onset_seq") for e in (d.get("episodes") or [])}) for d in ev.values())
    bearing = sum(1 for d in ev.values() if (d.get("contamination_events_declared") or 0) > 0)
    contaminated = 0
    for d in graded:
        g = d.get("grade") or {}
        if g.get("pass_to_pass"):
            passed, total = (int(x) for x in g["pass_to_pass"].split("/"))
            if total - passed - len(d.get("never_passed") or []) > 0:
                contaminated += 1
    return (f"| {label} | {len(graded)} | {resolved}/{slice_size} ({100 * resolved / slice_size:.0f}%) | {events} | "
            f"{incidents} | {bearing}/{len(ev)} | {contaminated} | ${spend(root):.2f} |")


def main() -> int:
    print("| substrate: arm | graded | resolved | events | incidents | bearing runs | contaminated trees | spend |")
    print("|---|---|---|---|---|---|---|---|")
    for spec in sys.argv[1:]:
        label, root = spec.split("=", 1)
        print(row(label.replace(":", ": "), Path(root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
