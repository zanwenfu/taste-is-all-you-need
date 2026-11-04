"""Gate between a trial run of a case set and its full run.

Checks one slice of studies/context_rot conpact-sets output (both summary prompts, four record
arms): every case present once, no empty summary, no call refused as too long, at most 10%
unreadable answers per condition, and every arm complete. Exit code 0 only when nothing is
wrong; the full run is queued to start only after this passes.

    python -m studies.context_rot.validate_sets <set folder> <slice> --expected 15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROMPTS = ("pi_mono", "anthropic")
ARMS = ("user_record", "recent_control", "criteria", "criteria_control")
CONDITIONS = ("full_with_sc", "full_without_sc", "compaction", "upper_bound")
NOT_SERVED = "[not served: the prompt is longer than the model's window]"


def _rows(path: Path) -> list[dict] | None:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else None


def problems(root: Path, part: str, expected: int, unreadable_limit: float = 0.10) -> list[str]:
    root, found = Path(root), []
    for prompt in PROMPTS:
        rows = _rows(root / prompt / part / "results.jsonl")
        if rows is None:
            found.append(f"{prompt}: no results")
            continue
        ids = [r["id"] for r in rows]
        if len(set(ids)) != expected or len(ids) != expected:
            found.append(f"{prompt}: {len(set(ids))} of {expected} cases ({len(ids)} rows)")
        empty = sum(not str((r.get("compacted") or [{}])[0].get("content") or "").strip() for r in rows)
        if empty:
            found.append(f"{prompt}: {empty} empty summaries")
        for condition in CONDITIONS:
            refused = sum(r.get(f"{condition}_output") == NOT_SERVED for r in rows)
            if refused:
                found.append(f"{prompt}: {refused} {condition} calls not served")
            unreadable = sum(r.get(f"{condition}_compliant") is None for r in rows)
            if rows and unreadable / len(rows) > unreadable_limit:
                found.append(f"{prompt}: {unreadable} of {len(rows)} {condition} answers unreadable")
        for arm in ARMS:
            arm_rows = _rows(root / prompt / f"{part}-{arm}" / "results.jsonl")
            if arm_rows is None or {r["id"] for r in arm_rows} != set(ids):
                found.append(f"{prompt}: arm {arm} incomplete ({0 if arm_rows is None else len(arm_rows)} rows)")
                continue
            unreadable = sum(r.get("compaction_compliant") is None for r in arm_rows)
            if unreadable / len(arm_rows) > unreadable_limit:
                found.append(f"{prompt}: arm {arm} {unreadable} of {len(arm_rows)} answers unreadable")
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    parser.add_argument("part")
    parser.add_argument("--expected", type=int, required=True)
    arguments = parser.parse_args(argv)
    found = problems(Path(arguments.root), arguments.part, arguments.expected)
    print(json.dumps({"root": arguments.root, "part": arguments.part, "problems": found}, indent=1))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
