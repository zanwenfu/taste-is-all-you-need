"""A dry run on real instances: does the whole thing hold together?

Not an experiment. Nothing here is evidence about any arm — the machine is
arm64 running x86 images under emulation, which can change test outcomes, and
a handful of instances could not support an inference anyway. The question is
narrower and worth answering before spending real money: does an agent run on
a real repository, get observed, get replayed against the real image, and come
out the other side as a report you can read.

Usage::

    python scripts/dryrun.py --instances 5 --arm A3 --budget 1.00

Every instance is skipped politely if its image is absent, because pulling a
4GB image per instance is the slowest thing in the loop and the point is to
exercise the pipeline, not the network.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste.benchmarks import swebench
from taste.benchmarks.swebench_run import (
    make_execute,
    make_prepare,
    make_score,
)
from taste.evalrun import run_sweep
from taste.execution import DockerProvider
from taste.viz import write_index, write_run


def image_for(instance: swebench.SWEInstance) -> str:
    """SWE-bench's published naming. The dataset does not carry the tag."""
    slug = instance.instance_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{slug}:latest"


def local_images() -> set[str]:
    out = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True, text=True, check=False,
    )
    return set(out.stdout.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--arm", default="A3")
    ap.add_argument("--budget", type=float, default=1.00, help="work-cost cap per cell")
    ap.add_argument("--root", default="/tmp/taste-dryrun")
    ap.add_argument("--dataset", default="data/verified.jsonl")
    ap.add_argument("--offline", action="store_true",
                    help="No model calls: exercise materialize + replay only.")
    args = ap.parse_args()

    root = Path(args.root)
    ledger = root / "ledger"
    have = local_images()

    everything = swebench.load_dataset(Path(args.dataset))
    pool = [i for i in everything if image_for(i) in have]
    if not pool:
        print("No SWE-bench images present locally. Pull at least one, e.g.:")
        print(f"  docker pull --platform linux/amd64 {image_for(everything[0])}")
        return 2
    chosen = pool[: args.instances]
    print(f"{len(pool)} instance image(s) available locally; running {len(chosen)}:")
    for i in chosen:
        print(f"  {i.instance_id:32s} {i.repo:26s} |P2P|={len(i.pass_to_pass)}")

    # The dataset carries no image column, so resolve the published tag here.
    for i in chosen:
        object.__setattr__(i, "image", image_for(i))
    instances = {i.instance_id: i for i in chosen}

    provider = DockerProvider()

    def llm_factory(_ctx):
        if args.offline:
            return None
        from taste.llm import LLM

        return LLM(budget_usd=args.budget, cap_on="work")

    started = time.time()
    report = run_sweep(
        tasks=[i.instance_id for i in chosen],
        arms=[args.arm], trials=1, ledger_dir=ledger,
        prepare=make_prepare(
            instances=instances, root=root / "runs",
            budget_usd=args.budget, provider=provider,
            repo_cache=root / "mirrors",
        ),
        execute=make_execute(llm_factory=llm_factory),
        score=make_score(ledger_dir=ledger),
    )

    print(f"\n=== {len(report.results)} cells in {time.time()-started:.0f}s ===")
    for r in report.results:
        print(f"  {r.task:32s} {r.status:10s} ${r.billed_usd:7.4f} "
              f"steps={r.steps_passed}/{r.steps_total} {r.error or ''}")
        if r.workspace and Path(r.workspace).exists():
            ev = {}
            if r.report_path and Path(r.report_path).exists():
                ev = json.loads(Path(r.report_path).read_text())
            write_run(Path(r.workspace), evidence=ev, arm=r.arm, instance=r.task,
                      output=root / "reports" / f"{r.task}-{r.arm}.html")

    index = write_index(ledger, output=root / "index.html")
    provider.close_all()
    print(f"\nreports : {root/'reports'}")
    print(f"index   : {index}")
    print(f"ledger  : {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
