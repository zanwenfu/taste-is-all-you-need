"""Held-out contamination for the split-oracle gate arm (A3reg2).

The gate under ``gate_split="half"`` reads results only for a deterministic
half of the instance's previously-passing test ids; the other half is never
read during the run. The grader's verdict on that unread half is the
held-out test of whether gating generalises beyond the tests it watches,
which is the circularity objection to the full-oracle arm.

Usage::

    python scripts/heldout.py --root /root/contrast40_A3reg2

Reads each cell's evidence sidecar (``gate_watched`` = ids the gate read,
``grade_failed`` = previously-passing ids failing in the graded final tree).
Sidecars from the superseded file-level split hold file paths in
``gate_watched``; those are mapped back to ids by file so the two formats
report the same quantity.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste.benchmarks import swebench


def _file_of(test_id: str, repo: str) -> str:
    """The file a previously-passing id lives in, by the same rule
    ``member_test_files`` applies: pytest ids carry their path, django's
    unittest labels carry a dotted module under ``tests/``."""
    if "::" in test_id:
        return test_id.split("::", 1)[0]
    m = swebench._UNITTEST_ID.match(test_id.strip())
    if m and repo == "django/django":
        module = m.group(1).rsplit(".", 1)[0]
        return "tests/" + module.replace(".", "/") + ".py"
    return ""


def load_evidence(root: Path) -> dict[str, dict]:
    ev: dict[str, dict] = {}
    for pattern in ("ledger/evidence/*.json", "rescored/evidence/*.json"):
        for f in glob.glob(str(root / pattern)):
            d = json.loads(Path(f).read_text())
            ev[d["instance_id"]] = d  # rescored sidecars overwrite originals
    return ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset", default="data/verified.jsonl")
    ap.add_argument("--substrate", choices=["verified", "live"], default="verified")
    args = ap.parse_args()

    if args.substrate == "live":
        from taste.benchmarks import swebenchlive as live

        if args.dataset == "data/verified.jsonl":
            args.dataset = "data/live_lite.jsonl"
        instances = {i.instance_id: i for i in live.load_live_dataset(Path(args.dataset))}
    else:
        instances = {i.instance_id: i for i in swebench.load_dataset(Path(args.dataset))}
    graded = resolved = held_cells = watched_cells = held_total = watched_total = 0
    rows = []
    for inst_id, d in sorted(load_evidence(Path(args.root)).items()):
        if d.get("resolved") is None:
            continue
        graded += 1
        resolved += bool(d["resolved"])
        p2p = list(instances[inst_id].pass_to_pass)
        raw = list(d.get("gate_watched") or [])
        if raw and all("::" not in w and " (" not in w for w in raw):
            watched_files = set(raw)  # superseded file-level split
            watched = {t for t in p2p if _file_of(t, instances[inst_id].repo) in watched_files}
        else:
            watched = set(raw)
        held = [t for t in p2p if t not in watched]
        held_total += len(held)
        watched_total += len(watched)
        failed = set(d.get("grade_failed") or [])
        hf = [t for t in held if t in failed]
        wf = [t for t in p2p if t in watched and t in failed]
        held_cells += bool(hf)
        watched_cells += bool(wf)
        rows.append((inst_id, len(watched), len(held), len(hf), len(wf), d["resolved"]))

    print(f"split-oracle arm: graded={graded} resolved={resolved} ({100 * resolved / max(1, graded):.1f}%)")
    print(f"previously-passing ids: watched by the gate={watched_total}  held out={held_total}")
    print(f"cells with a HELD-OUT failure in the final tree: {held_cells}")
    print(f"cells with a WATCHED failure in the final tree:  {watched_cells}")
    print("(instance, watched, held_out, held_out_failed, watched_failed, resolved) for cells with any failure:")
    for r in rows:
        if r[3] or r[4]:
            print("  ", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
