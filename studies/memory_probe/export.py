"""What the probe harness reads, written once inside the study's container where the rule-loss package lives.

The harness itself runs outside it, because Taste's memory needs git and the container has none. So the cases
(history, rule, question, letters) and the summaries the baseline runs already made are written out as plain
JSON: k = 1 from the full run, k = 16 from the last round of the repeated run, each with the baseline's own
compliance, its restated ceiling and the no-rule floor, for effective retention.

    python -m studies.memory_probe.export --full runs/conpact/full --repeated runs/conpact/repeated \\
        --histories 0:10 --out runs/probe/inputs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CASE_FIELDS = ("id", "haystack_id", "sc_id", "sc_type", "sc_text", "compliant_letter", "system_prompt",
               "developer_prompt", "probe_prompt", "post_sc_probe_prompt", "messages")
PROMPTS = ("pi_mono", "anthropic")


def _rows(paths) -> list[dict]:
    return [json.loads(line) for path in paths for line in Path(path).read_text().splitlines() if line.strip()]


def summaries(full: Path, repeated: Path, ids: set[str]) -> list[dict]:
    """One row per (case, prompt, k) for k = 1 and 16, refusing a case that is missing or found twice."""
    out = []
    for prompt in PROMPTS:
        base = {}
        for row in _rows(sorted(Path(full, f"hermes-{prompt}").glob("h??-??/results.jsonl"))):
            if row["id"] in ids:
                if row["id"] in base:
                    raise ValueError(f"{row['id']} appears twice in the {prompt} full run")
                base[row["id"]] = row
        long = {}
        for row in _rows(sorted(Path(repeated, prompt).rglob("results.jsonl"))):
            if row["id"] in ids and row["k"] == 16:
                if row["id"] in long:
                    raise ValueError(f"{row['id']} appears twice at k = 16 in the {prompt} repeated run")
                long[row["id"]] = row
        for label, found in (("full", base), ("repeated k = 16", long)):
            if set(found) != ids:
                raise ValueError(f"{len(ids - set(found))} cases missing from the {prompt} {label} run")
        for case in sorted(ids):
            row, deep = base[case], long[case]
            floor = row["full_without_sc_compliant"]
            out.append({"id": case, "prompt": prompt, "k": 1,
                        "summary": "\n\n".join(m["content"] for m in row["compacted"]),
                        "a1_compliant": row["compaction_compliant"], "a1_upper_compliant": row["upper_bound_compliant"],
                        "floor_compliant": floor})
            out.append({"id": case, "prompt": prompt, "k": 16, "summary": deep["summaries"][-1],
                        "a1_compliant": deep["compaction_compliant"],
                        "a1_upper_compliant": deep["compaction_upper_compliant"], "floor_compliant": floor})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", required=True)
    parser.add_argument("--repeated", required=True)
    parser.add_argument("--histories", default="0:10")
    parser.add_argument("--source", default="hermes")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)

    from conpact import load_static

    from studies.context_rot.conpact_bridge import DATA_REVISION, history_range

    start, end = (int(part) for part in arguments.histories.split(":"))
    instances = history_range(load_static(arguments.source, revision=DATA_REVISION), start, end)
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = [{key: getattr(inst, key) for key in CASE_FIELDS} for inst in instances]
    (out / "cases.jsonl").write_text("".join(json.dumps(case) + "\n" for case in cases))
    rows = summaries(Path(arguments.full), Path(arguments.repeated), {case["id"] for case in cases})
    (out / "summaries.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(json.dumps({"cases": len(cases), "summaries": len(rows), "revision": DATA_REVISION}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
