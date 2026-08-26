"""The golden check on SWE-bench-Live: the gold patch through the real
routed pipeline must grade resolved=True; a null run must grade False.
Zero model spend. Same contract as scripts/goldencheck.py, Live substrate.

    python scripts/livegolden.py --instance aws-cloudformation__cfn-lint-3798
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste.benchmarks import swebenchlive as live
from taste.benchmarks.swebench_run import make_execute, make_prepare, make_score
from taste.cores import Plan, Step, Verification, WorkerResult
from taste.evalrun import run_sweep
from taste.execution import DockerProvider
from taste.tools import make_builtin_tools


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="aws-cloudformation__cfn-lint-3798")
    ap.add_argument("--dataset", default="data/live_lite.jsonl")
    ap.add_argument("--root", default="/tmp/taste-live-golden")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    instance = next(
        i for i in live.load_live_dataset(dataset) if i.instance_id == args.instance
    )
    object.__setattr__(instance, "image", instance.published_image)
    gold = next(
        json.loads(line)["patch"]
        for line in dataset.read_text().splitlines()
        if json.loads(line).get("instance_id") == args.instance
    )
    print(f"{instance.instance_id}  |F2P|={len(instance.fail_to_pass)} "
          f"|P2P|={len(instance.pass_to_pass)}  parser={instance.log_parser}")

    root = Path(args.root)
    if root.exists():
        shutil.rmtree(root)
    # env_prefix="": RepoLaunch images carry the environment on the default
    # PATH; sourcing a conda that does not exist fails every routed command.
    provider = DockerProvider(env_prefix="")

    def scripted(cell, ctx):
        def gold_worker(step: Step, plan: Plan) -> WorkerResult:
            tools = {t.name: t.handler for t in
                     make_builtin_tools(ctx.workspace, router=ctx.router)}
            tools["write_file"]("gold.diff", gold + "\n")
            out = tools["run_shell"]("git apply -v gold.diff", 300)
            if "(exit 0)" not in out:
                raise RuntimeError(f"gold patch did not apply:\n{out[-800:]}")
            return WorkerResult(summary="gold", tool_calls=2, stopped_reason="end_turn")

        def null_worker(step: Step, plan: Plan) -> WorkerResult:
            return WorkerResult(summary="nothing", tool_calls=0, stopped_reason="end_turn")

        plan = Plan(task="live golden", steps=[
            Step(id="step-01", description="apply the fix",
                 verification=Verification(kind="shell", command="true")),
        ])
        worker = gold_worker if cell.arm == "A0" else null_worker
        return {"plan_override": plan, "worker_override": worker}

    def grade(ctx, result):
        import subprocess

        root_sha = subprocess.run(
            ["git", "rev-list", "--max-parents=0", "HEAD"],
            cwd=ctx.workspace, capture_output=True, text=True,
        ).stdout.split()
        if not root_sha:
            return None
        from taste.benchmarks.swebench import patch_for

        patch = patch_for(ctx.workspace, root_sha[0])
        sandbox = ctx.provider.open(
            key=f"grade:{ctx.instance.instance_id}", image=ctx.instance.image,
            network_mode="bridge",
        )
        try:
            report = live.grade_live_in_sandbox(sandbox, ctx.instance, patch)
        finally:
            sandbox.close()
        if report is None:
            return None
        ctx.grade_report = report
        # Per-test detail for the rot-aware verdict below. Live instances age:
        # cfn-lint's E2533 starts failing the day the calendar passes a
        # runtime's deprecation date, in the raw image, no harness involved —
        # so "gold grades resolved" is not a property the harness can promise
        # on a stale instance. What it CAN promise: gold fixes every F2P and
        # breaks nothing the baseline had not already lost.
        arm = ctx.config.label
        out = root / "grades"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{arm}.json").write_text(json.dumps({
            "per_test": report.per_test,
            "f2p": f"{report.fail_to_pass_passed}/{report.fail_to_pass_total}",
        }))
        return report.resolved

    started = time.time()
    report = run_sweep(
        tasks=[instance.instance_id], arms=["A0", "A2"], trials=1,
        ledger_dir=root / "ledger",
        prepare=make_prepare(
            instances={instance.instance_id: instance}, root=root / "runs",
            provider=provider, route_execution=True,
            parity_check=live.live_parity_check,
        ),
        execute=make_execute(run_overrides=scripted),
        score=make_score(
            ledger_dir=root / "ledger",
            suite_factory=live.live_suite_factory,
            grade=grade,
        ),
        on_cell=lambda r: print(
            f"  [cell] arm={r.arm} status={r.status} score={r.score} {r.error or ''}",
            flush=True,
        ),
    )
    print(f"({time.time() - started:.0f}s)")

    by_arm = {r.arm: r for r in report.results}
    gold_cell, null_cell = by_arm.get("A0"), by_arm.get("A2")

    def failures(arm_label: str) -> set[str] | None:
        path = root / "grades" / f"{arm_label}.json"
        if not path.exists():
            return None
        detail = json.loads(path.read_text())
        return {t for t, status in detail["per_test"].items()
                if "fail" in status.lower() or "error" in status.lower()}

    gold_fail, null_fail = failures("A0-no-recovery"), failures("A2-repair-in-place")
    ok = True
    if null_cell is None or null_cell.score != 0.0:
        ok = False
        print(f"NULL FAILED: {null_cell and (null_cell.status, null_cell.score, null_cell.error)}")
    else:
        print("null worker -> resolved=False  [ok]")

    if gold_cell is None or gold_fail is None or null_fail is None:
        ok = False
        print(f"GOLD FAILED (no grade detail): {gold_cell and (gold_cell.status, gold_cell.score)}")
    elif gold_cell.score == 1.0:
        print("gold patch  -> resolved=True   [ok]")
    else:
        gold_evidence = json.loads((root / "grades" / "A0-no-recovery.json").read_text())
        f2p_ok = gold_evidence["f2p"].split("/")[0] == gold_evidence["f2p"].split("/")[1]
        new_breaks = gold_fail - null_fail
        if f2p_ok and not new_breaks:
            print("gold patch  -> resolved_fresh=True  [ok]  "
                  f"(instance is stale: {len(null_fail)} baseline P2P failures — "
                  "time-rotted oracle, fails identically in the raw image)")
        else:
            ok = False
            print(f"GOLD FAILED: f2p={gold_evidence['f2p']} new_breaks={sorted(new_breaks)[:5]}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
