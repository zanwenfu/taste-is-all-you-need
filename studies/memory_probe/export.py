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


def summaries(full: Path, repeated: Path, ids: set[str], *, deep: int = 16, prompts: tuple[str, ...] = PROMPTS) -> list[dict]:
    """One row per (case, prompt, k) for k = 1 and ``deep``, refusing a case that is missing or found twice."""
    out = []
    for prompt in prompts:
        base = {}
        for row in _rows(sorted(Path(full, f"hermes-{prompt}").glob("h??-??/results.jsonl"))):
            if row["id"] in ids:
                if row["id"] in base:
                    raise ValueError(f"{row['id']} appears twice in the {prompt} full run")
                base[row["id"]] = row
        long = {}
        for row in _rows(sorted(Path(repeated, prompt).rglob("results.jsonl"))):
            if row["id"] in ids and row["k"] == deep:
                if row["id"] in long:
                    raise ValueError(f"{row['id']} appears twice at k = {deep} in the {prompt} repeated run")
                long[row["id"]] = row
        for label, found in (("full", base), (f"repeated k = {deep}", long)):
            if set(found) != ids:
                raise ValueError(f"{len(ids - set(found))} cases missing from the {prompt} {label} run")
        for case in sorted(ids):
            row, last = base[case], long[case]
            floor = row["full_without_sc_compliant"]
            out.append({"id": case, "prompt": prompt, "k": 1,
                        "summary": "\n\n".join(m["content"] for m in row["compacted"]),
                        "a1_compliant": row["compaction_compliant"], "a1_upper_compliant": row["upper_bound_compliant"],
                        "floor_compliant": floor})
            out.append({"id": case, "prompt": prompt, "k": deep, "summary": last["summaries"][-1],
                        "a1_compliant": last["compaction_compliant"],
                        "a1_upper_compliant": last["compaction_upper_compliant"], "floor_compliant": floor})
    return out


def delivered(repeated: Path, ids: set[str], *, prompts: tuple[str, ...] = PROMPTS) -> list[dict]:
    """One row per (case, prompt): the study reader's rule records as that prompt's own run delivered them. The reader
    ran once per run and its readings differ (17 of the 150 Hermes cases), so one prompt's can't stand for the other's."""
    out = []
    for prompt in prompts:
        readings: dict[str, list] = {}
        for path in sorted(Path(repeated, prompt).glob("h??-??/records.json")):
            for case, records in json.loads(path.read_text())["records"].items():
                if case in ids:
                    if case in readings:
                        raise ValueError(f"{case} appears twice in the {prompt} run's rule records")
                    readings[case] = records
        if set(readings) != ids:
            raise ValueError(f"{len(ids - set(readings))} cases have no rule records in the {prompt} run")
        out += [{"id": case, "prompt": prompt, "records": len(readings[case]), "items": readings[case],
                 "block": "\n".join(f"- {record['text']}" for record in readings[case]) or "(none)"}
                for case in sorted(ids)]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", required=True)
    parser.add_argument("--repeated", required=True)
    parser.add_argument("--histories", default="0:10")
    parser.add_argument("--source", default="hermes")
    parser.add_argument("--variant", help="middle, bottom, revised, withdrawn or cued (studies.memory_probe.variants)")
    parser.add_argument("--cases-only", action="store_true", help="write the cases and stop; no baseline run is read")
    parser.add_argument("--deep", type=int, default=16, help="the deeper depth the repeated run holds")
    parser.add_argument("--prompts", default=",".join(PROMPTS), help="the summary prompts the baseline runs cover")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)

    from conpact import load_static

    from studies.context_rot.conpact_bridge import DATA_REVISION, history_range
    from studies.memory_probe.variants import apply

    start, end = (int(part) for part in arguments.histories.split(":"))
    instances = apply(history_range(load_static(arguments.source, revision=DATA_REVISION), start, end), arguments.variant)
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = [{key: getattr(inst, key) for key in CASE_FIELDS} for inst in instances]
    (out / "cases.jsonl").write_text("".join(json.dumps(case) + "\n" for case in cases))
    if arguments.cases_only:
        print(json.dumps({"cases": len(cases), "variant": arguments.variant, "revision": DATA_REVISION}))
        return 0
    prompts = tuple(arguments.prompts.split(","))
    rows = summaries(Path(arguments.full), Path(arguments.repeated), {case["id"] for case in cases}, deep=arguments.deep,
                     prompts=prompts)
    (out / "summaries.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    blocks = delivered(Path(arguments.repeated), {case["id"] for case in cases}, prompts=prompts)
    (out / "delivered.jsonl").write_text("".join(json.dumps(row) + "\n" for row in blocks))
    print(json.dumps({"cases": len(cases), "summaries": len(rows), "revision": DATA_REVISION}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
