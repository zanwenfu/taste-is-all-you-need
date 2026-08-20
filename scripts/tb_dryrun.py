"""A bounded dry run on a Terminal-Bench challenge.

These tasks are multi-hour by design -- the published set exists for
"token-intensive, long-running" work -- so this does not attempt completion.
It asks a narrower question: does the harness drive one of them at all, and
what does the Agent OS actually do in the first few dollars.

The agent works in a local workspace holding the task's environment files; the
grader needs the built image and is not run here.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste.agent import AgentSpec
from taste.benchmarks import terminalbench as tb
from taste.evalrun import kernel_for
from taste.llm import LLM
from taste.viz import write_run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="/tmp/tbc/wasm-render")
    ap.add_argument("--arm", default="A3")
    ap.add_argument("--budget", type=float, default=1.00)
    ap.add_argument("--root", default="/tmp/taste-tb")
    args = ap.parse_args()

    task = tb.load_task(Path(args.task))
    print(f"task        : {task.name}", flush=True)
    print(f"long-running: {task.is_long_running}  signals={task.heavy_signals}", flush=True)
    print(f"instruction : {len(tb.task_text(task))} chars", flush=True)
    print(f"grader      : {len(tb.graded_paths(task))} files withheld", flush=True)

    root = Path(args.root)
    workspace = tb.materialize(task, root / "runs" / task.slug / args.arm / "t1")
    print(f"workspace   : {workspace}", flush=True)
    print(f"              {sorted(p.name for p in workspace.iterdir() if p.name != '.git')}",
          flush=True)

    llm = LLM(budget_usd=args.budget, cap_on="work")
    kernel = kernel_for(args.arm, workspace, llm, max_parallel=1)
    spec = AgentSpec(
        name="tb",
        description="Build the artifact the instruction asks for.",
        system_prompt=(
            "You are implementing a substantial piece of software from scratch in "
            "this workspace. Work incrementally: get something minimal running "
            "before adding surface area, and keep what already works working."
        ),
    )

    started = time.time()
    try:
        result = kernel.run(task=tb.task_text(task), spec=spec, base_ref="HEAD")
        status, reason = result.status, result.failure_reason
    except Exception as exc:  # a bounded dry run must still report
        status, reason = "error", f"{type(exc).__name__}: {exc}"
        result = None

    print(f"\nstatus      : {status}", flush=True)
    print(f"reason      : {str(reason)[:160]}", flush=True)
    print(f"elapsed     : {time.time()-started:.0f}s", flush=True)
    print(f"spend       : ${llm.stats.total_cost_usd:.4f} billed / "
          f"${llm.stats.total_work_usd:.4f} work", flush=True)

    report = write_run(workspace, arm=args.arm, instance=task.slug,
                       output=root / "reports" / f"{task.slug}-{args.arm}.html")
    print(f"report      : {report}", flush=True)

    made = sorted(p.name for p in workspace.iterdir() if p.name != ".git")
    print(f"workspace now: {made}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
