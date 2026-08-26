"""Fetch a SWE-bench-Live split into a local jsonl, via the datasets server.

    python scripts/fetch_live.py --split lite --out data/live_lite.jsonl

The gold `patch` column is KEPT in the file (the golden check needs it) and
dropped by the loader — the same containment Verified uses.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows"
DATASET = "SWE-bench-Live/SWE-bench-Live"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="lite")
    ap.add_argument("--out", default="data/live_lite.jsonl")
    ap.add_argument("--page", type=int, default=100)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {"dataset": DATASET, "config": "default", "split": args.split,
             "offset": offset, "length": args.page}
        )
        with urllib.request.urlopen(f"{API}?{query}", timeout=120) as response:
            payload = json.load(response)
        batch = [r["row"] for r in payload.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        print(f"  fetched {offset}", flush=True)
        if offset >= payload.get("num_rows_total", 0):
            break
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"{len(rows)} instances -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
