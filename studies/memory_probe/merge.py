"""Join the shards of one row into its results, or refuse: every case exactly once, and every shard run with the same settings.

    python -m studies.memory_probe.merge --inputs runs/probe/inputs --out <row>/results.jsonl <row>/shard0.jsonl <row>/shard1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from studies.memory_probe.run import summarize


def merge_shards(cases: list[dict], shards: list[Path], out: Path) -> dict[str, Any]:
    rows = {}
    every = []
    settings = []
    for path in shards:
        for row in map(json.loads, Path(path).read_text().splitlines()):
            every.append(row["id"])
            rows[row["id"]] = row
        said = json.loads(Path(path).with_name(Path(path).stem + ".settings.json").read_text())
        said.pop("shard", None)
        settings.append(said)
    if sorted(every) != sorted(case["id"] for case in cases):
        raise ValueError(f"the shards do not hold every case exactly once: {len(every)} rows, {len(set(every))} cases, of {len(cases)}")
    for said in settings:
        changed = sorted(key for key in {*said, *settings[0]} if said.get(key) != settings[0].get(key))
        if changed:
            raise ValueError(f"the shards ran with different settings: {changed}")
    ordered = [rows[case["id"]] for case in cases]
    summary = summarize(ordered)
    out = Path(out)
    out.write_text("".join(json.dumps(row) + "\n" for row in ordered))
    out.with_name(out.stem + ".summary.json").write_text(json.dumps(summary, indent=1))
    out.with_name(out.stem + ".settings.json").write_text(json.dumps({**settings[0], "shards": len(shards)}, indent=1))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("shards", nargs="+")
    arguments = parser.parse_args(argv)
    cases = [json.loads(line) for line in (Path(arguments.inputs) / "cases.jsonl").read_text().splitlines()]
    summary = merge_shards(cases[: arguments.limit] if arguments.limit else cases, [Path(path) for path in arguments.shards], Path(arguments.out))
    print(json.dumps({key: summary[key] for key in ("cases", "looked", "found", "compliant", "acted", "unparsed_arguments")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
