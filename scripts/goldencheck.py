"""The golden check: the one test that would have caught most of the catalogue.

Runs the ENTIRE routed pipeline on one real instance with zero model spend:
a scripted worker applies the benchmark's own gold patch through the real
tool path (write_file -> push -> in-container git apply -> pull), the routed
Monitor verifies it in the pinned image, the replay scores the timeline, and
the official grader must return resolved=True. A second cell does nothing
and must grade resolved=False.

If either half fails, the harness is not ready to spend money. This script
IS the readiness gate's final rung, and it exercises exactly the seams fakes
cannot prove: conda activation in exec_in_env, put_archive parents, the
container's git version, tar extraction from a real /testbed.

    python scripts/goldencheck.py --instance psf__requests-5414
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dryrun import image_for

from taste.benchmarks import swebench
from taste.benchmarks.swebench_run import (
    make_execute,
    make_grade,
    make_prepare,
    make_score,
)
from taste.cores import Plan, Step, Verification, WorkerResult
from taste.evalrun import run_sweep
from taste.execution import DockerProvider


def gold_patch_of(dataset: Path, instance_id: str) -> str:
    """The dataset's own fix. Read raw: load_dataset drops the column on
    purpose (the agent must never see it), and the golden check is the one
    caller with a legitimate need."""
    for line in dataset.read_text().splitlines():
        row = json.loads(line)
        if row.get("instance_id") == instance_id:
            return row["patch"]
    raise SystemExit(f"{instance_id} not in {dataset}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="psf__requests-5414")
    ap.add_argument("--dataset", default="data/verified.jsonl")
    ap.add_argument("--root", default="/tmp/taste-golden")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    instance = next(
        i for i in swebench.load_dataset(dataset) if i.instance_id == args.instance
    )
    object.__setattr__(instance, "image", image_for(instance))
    gold = gold_patch_of(dataset, args.instance)
    f2p_files = sorted({d for d in swebench.graded_test_files(instance)})
    print(f"instance {instance.instance_id}  |F2P|={len(instance.fail_to_pass)} "
          f"|P2P|={len(instance.pass_to_pass)}  graded files: {f2p_files}")

    root = Path(args.root)
    if root.exists():
        # A check, not a resumable sweep: a stale ledger silently skips both
        # cells and the summary reads as if nothing ran.
        import shutil

        shutil.rmtree(root)
    provider = DockerProvider()

    def scripted(cell, ctx):
        from taste.tools import make_builtin_tools

        def gold_worker(step: Step, plan: Plan) -> WorkerResult:
            # The gold patch travels the REAL tool path — the exact handlers
            # a model-driven worker calls: write_file marks the diff dirty,
            # run_shell pushes it, applies it inside the image, and pulls the
            # modified sources home for the shadow observation and the grade.
            tools = {t.name: t.handler for t in
                     make_builtin_tools(ctx.workspace, router=ctx.router)}
            tools["write_file"]("gold.diff", gold + "\n")
            out = tools["run_shell"]("git apply -v gold.diff", 300)
            if "(exit 0)" not in out:
                raise RuntimeError(f"gold patch did not apply:\n{out[-800:]}")
            return WorkerResult(summary="gold", tool_calls=2, stopped_reason="end_turn")

        def null_worker(step: Step, plan: Plan) -> WorkerResult:
            return WorkerResult(summary="nothing", tool_calls=0, stopped_reason="end_turn")

        plan = Plan(task="golden", steps=[
            Step(id="step-01", description="apply the fix",
                 verification=Verification(kind="shell", command="true")),
        ])
        worker = gold_worker if cell.arm == "A0" else null_worker
        return {"plan_override": plan, "worker_override": worker}
    started = time.time()
    report = run_sweep(
        tasks=[instance.instance_id], arms=["A0", "A2"], trials=1,
        ledger_dir=root / "ledger",
        prepare=make_prepare(
            instances={instance.instance_id: instance}, root=root / "runs",
            provider=provider, route_execution=True,
        ),
        execute=make_execute(run_overrides=scripted),
        score=make_score(ledger_dir=root / "ledger", grade=make_grade()),
        on_cell=lambda r: print(
            f"  [cell] arm={r.arm} status={r.status} score={r.score} {r.error or ''}",
            flush=True,
        ),
    )
    print(f"({time.time() - started:.0f}s)")

    by_arm = {r.arm: r for r in report.results}
    gold_cell, null_cell = by_arm.get("A0"), by_arm.get("A2")
    ok = True
    if gold_cell is None or gold_cell.score != 1.0:
        ok = False
        print(f"GOLD FAILED: {gold_cell and (gold_cell.status, gold_cell.score, gold_cell.error)}")
    else:
        print("gold patch  -> resolved=True   [ok]")
    if null_cell is None or null_cell.score != 0.0:
        ok = False
        print(f"NULL FAILED: {null_cell and (null_cell.status, null_cell.score, null_cell.error)}")
    else:
        print("null worker -> resolved=False  [ok]")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
