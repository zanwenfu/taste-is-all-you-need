"""How often does each arm look, and how much does it change between looks?

    python scripts/grid_census.py A0=/root/contrast40_A0 mswe=/root/mswe40_sonnet2_s1+...

Event counts are not comparable across arms until this is. An observation is
minted when a tool call changes the working tree, so two things move
together and in opposite directions:

  exposure    how much the run mutates -- more edits, more chances to break
              something. Counted as mutated-file events (the sum of files
              per observation) and as the union of files touched.
  resolution  how finely the run is sampled -- more observations, more
              adjacent pairs in which a PASS->FAIL transition can be seen at
              all. The harness's write_file touches one path per call;
              mini-swe-agent's only tool is bash, so a three-file edit
              delivered as one heredoc is one observation where the harness
              would record three.

If an arm has *more* exposure and *fewer* observations than another, its
event count is biased downward on both terms, and a positive difference in
its favour is identified without adjusting anything. If the two move the
same way, no cross-arm event rate is licensed -- and dividing events by
observations does not rescue it, because a run that breaks a test then
issues more commands investigating it, so the denominator is caused by the
outcome it is meant to normalise.

The ``run`` and ``final`` observations are excluded throughout: neither is
an opportunity to see a transition.
"""

from __future__ import annotations

import contextlib
import glob
import json
import statistics as st
import sys
from pathlib import Path


def runs_of(root: Path) -> list[dict]:
    out = []
    for shadow in sorted(glob.glob(str(root / "runs" / "*" / "*" / "t*" / ".git" / "taste" / "shadow.jsonl"))):
        rows = [json.loads(line) for line in Path(shadow).read_text().splitlines() if line.strip()]
        looks = [r for r in rows if r.get("trigger") not in ("run", "final")]
        if not looks:
            continue
        files_per = [len(r.get("files") or ()) for r in looks]
        union: set[str] = set()
        for r in looks:
            union.update(r.get("files") or ())
        manifest = Path(shadow).with_name("manifest.json")
        commands = None
        if manifest.exists():
            with contextlib.suppress(json.JSONDecodeError):
                commands = json.loads(manifest.read_text()).get("commands")
        out.append({
            "instance": Path(shadow).parts[-6],
            "observations": len(looks),
            "mutated_file_events": sum(files_per),
            "files_touched": len(union),
            "files_per_observation": files_per,
            "commands": commands,
        })
    return out


def census(label: str, roots: list[Path]) -> dict:
    runs: dict[str, dict] = {}
    for root in roots:
        for r in runs_of(root):
            runs.setdefault(r["instance"], r)
    rows = list(runs.values())
    if not rows:
        return {"label": label, "runs": 0}
    per_obs = [n for r in rows for n in r["files_per_observation"]]
    commands = [r["commands"] for r in rows if r["commands"]]
    return {
        "label": label,
        "runs": len(rows),
        "obs_median": st.median(r["observations"] for r in rows),
        "obs_total": sum(r["observations"] for r in rows),
        "mutated_median": st.median(r["mutated_file_events"] for r in rows),
        "mutated_total": sum(r["mutated_file_events"] for r in rows),
        "files_median": st.median(r["files_touched"] for r in rows),
        "multi_file_share": (sum(1 for n in per_obs if n > 1) / len(per_obs)) if per_obs else 0.0,
        "cmd_median": st.median(commands) if commands else None,
        "cmd_per_obs": (sum(commands) / sum(r["observations"] for r in rows if r["commands"])) if commands else None,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    results = []
    for spec in sys.argv[1:]:
        label, _, paths = spec.rpartition("=")
        roots = [Path(p) for p in paths.split("+")]
        results.append(census(label or roots[0].name, roots))

    print(f"{'arm':22s} {'runs':>4s} {'obs/run':>8s} {'obs':>5s} {'mutated files/run':>18s} "
          f"{'files/run':>9s} {'multi-file obs':>14s} {'cmds/run':>8s} {'cmds/obs':>8s}")
    for r in results:
        if not r["runs"]:
            print(f"{r['label']:22s} (no timelines found)")
            continue
        print(f"{r['label']:22s} {r['runs']:>4d} {r['obs_median']:>8.0f} {r['obs_total']:>5d} "
              f"{r['mutated_median']:>18.0f} {r['files_median']:>9.0f} {r['multi_file_share']:>13.0%} "
              f"{(r['cmd_median'] or 0):>8.0f} {(r['cmd_per_obs'] or 0):>8.1f}")
    print("\nobs/run and mutated files/run are per-run medians; obs is the pooled count of")
    print("observations that were opportunities (run and final excluded).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
