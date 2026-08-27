"""Re-measure an archived run with the current instrument. No model calls.

The reason this exists: the measurement code has a bug history, and every fix
to it invalidates every number taken before the fix. Re-running the *agent* to
find out is both expensive and wrong -- a fresh run samples a different
trajectory, so a changed result confounds "the instrument was fixed" with "the
agent did something else". The trees are already on disk and the shadow
timeline is already committed, so the measurement can simply be taken again
against the identical history.

That makes this the control for a specific question: when a re-run reports
fewer events than the run before it, is that the agent or the instrument?
Re-scoring the *old* workspaces answers it, because the trajectory is held
fixed by construction.

    python scripts/rescore.py --root /root/pilot40 --only psf__requests-5414
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste.benchmarks import swebench
from taste.benchmarks.swebench_run import CellContext, make_grade, make_score
from taste.config import HarnessConfig
from taste.evalrun import Cell
from taste.execution import DockerProvider


def _ledger_cells(root: Path) -> list[dict]:
    out = []
    for path in sorted((root / "ledger").glob("*.json")):
        try:
            row = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "status" in row and row.get("workspace"):
            out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root of an archived sweep.")
    ap.add_argument("--dataset", default="data/verified.jsonl")
    ap.add_argument("--only", default="", help="Comma-separated instance ids.")
    ap.add_argument("--out", default="", help="Where to write re-scored evidence.")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "rescored"
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}

    # The dataset carries no image column; the published tag is derived the
    # same way the sweep driver derives it, so a re-score reaches the same
    # image the original measurement used. Getting this wrong would not error
    # -- it would measure a different environment and report the difference as
    # a change in the agent's behaviour.
    instances = {}
    for inst in swebench.load_dataset(Path(args.dataset)):
        object.__setattr__(inst, "image", inst.published_image)
        instances[inst.instance_id] = inst
    rows = [r for r in _ledger_cells(root) if not wanted or r["task"] in wanted]
    if not rows:
        print(f"no cells matched under {root}")
        return 1

    provider = DockerProvider()
    score = make_score(ledger_dir=out, grade=make_grade())
    print(f"re-scoring {len(rows)} cell(s) from {root} with the current instrument\n")

    for row in rows:
        instance = instances.get(row["task"])
        workspace = Path(row["workspace"])
        if instance is None or not workspace.exists():
            print(f"  {row['task']:32s} SKIP (workspace or instance missing)")
            continue

        ctx = CellContext(
            instance=instance,
            config=HarnessConfig.arm(row["arm"]),
            workspace=workspace,
            gitdir=Path(row.get("gitdir") or (workspace / ".git")),
            provider=provider,
            session=row.get("session_id") or "",
        )
        # `score` reads only the workspace, the git dir and the timeline. The
        # RunResult it takes is used for its session id alone when the context
        # carries none, so a stub is honest here rather than convenient.
        stub = type("ArchivedRun", (), {"session_id": ctx.session})()
        cell = Cell(task=row["task"], arm=row["arm"], trial=row.get("trial", 1))
        try:
            score(cell, ctx, stub)
        except Exception as exc:
            print(f"  {row['task']:32s} ERROR {type(exc).__name__}: {exc}")
            continue

        evidence = json.loads(Path(ctx.report_path).read_text())
        # Both units, side by side: `episodes` is one per raw test id, and
        # `declared` is the pre-declared (test function, onset) unit with
        # parametrised variants collapsed. The gap between them is exactly
        # what parametrisation inflates, and it must stay visible.
        print(
            f"  {row['task']:32s} obs={evidence['observations']:2d} "
            f"episodes={len(evidence['episodes']):2d} "
            f"declared={evidence['contamination_events_declared']:2d} "
            f"holes={len(evidence['never_passed']):4d} "
            f"replays={evidence['replays']:3d}"
        )
    print(f"\nre-scored evidence: {out / 'evidence'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
