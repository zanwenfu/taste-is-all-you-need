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

from taste.attempts import harvest_by_instance
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
    """SWE-bench's published naming. Defined on the instance so the sweep
    driver and the re-scorer cannot drift into two different tags."""
    return instance.published_image


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
    ap.add_argument("--only", default="", help="Comma-separated instance ids.")
    ap.add_argument("--observe-tools", action="store_true",
                    help="Observe after every tool call, not only at step boundaries.")
    ap.add_argument("--match-retries-from", default="",
                    help="Root of a completed paired sweep. Each cell is capped at the "
                         "retries its paired run actually used, making this an "
                         "attempt-matched arm (A3'). Coverage is reported, never assumed.")
    ap.add_argument("--offline", action="store_true",
                    help="No model calls: exercise materialize + replay only.")
    args = ap.parse_args()

    root = Path(args.root)
    ledger = root / "ledger"
    have = local_images()

    everything = swebench.load_dataset(Path(args.dataset))
    pool = [i for i in everything if image_for(i) in have]
    if args.only:
        wanted = set(args.only.split(","))
        pool = [i for i in pool if i.instance_id in wanted]

    # Prefer repositories whose mirror is already on disk, then small ones.
    # The clone, not the model, is the long pole on a cold cache: astropy is
    # half a gigabyte and requests is fifteen megabytes.
    cache = Path(args.root) / "mirrors"
    small = ("psf/requests", "pallets/flask", "mwaskom/seaborn", "pytest-dev/pytest")

    def cost(instance: swebench.SWEInstance) -> tuple[int, int]:
        mirror = cache / f"{instance.repo.replace('/', '__')}.git"
        return (0 if mirror.exists() else 1, 0 if instance.repo in small else 1)

    pool.sort(key=cost)
    if not pool:
        print("No SWE-bench images present locally. Pull at least one, e.g.:")
        print(f"  docker pull --platform linux/amd64 {image_for(everything[0])}")
        return 2
    chosen = pool[: args.instances]
    print(f"{len(pool)} instance image(s) available locally; running {len(chosen)}:", flush=True)
    for i in chosen:
        print(f"  {i.instance_id:32s} {i.repo:26s} |P2P|={len(i.pass_to_pass)}", flush=True)

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

    def announce(record) -> None:
        print(f"  [cell] {record.task:30s} {record.status:9s} "
              f"${record.billed_usd:6.4f} {record.error or ''}", flush=True)

    allowance: dict[str, int] = {}
    if args.match_retries_from:
        allowance = harvest_by_instance(Path(args.match_retries_from))
        wanted = {i.instance_id for i in chosen}
        matched = wanted & allowance.keys()
        # Printed, not silently tolerated. A partially matched sweep is a
        # different experiment from a matched one, and the difference is
        # invisible in the ledger -- an unmatched cell simply runs on the
        # arm's own ceiling and looks exactly like a matched one.
        print(f"  retry allowance: {len(matched)}/{len(wanted)} instances matched "
              f"from {args.match_retries_from}")
        if missing := sorted(wanted - allowance.keys()):
            print(f"  UNMATCHED (running on the arm's own ceiling): {', '.join(missing)}")

    started = time.time()
    report = run_sweep(
        tasks=[i.instance_id for i in chosen],
        arms=[args.arm], trials=1, ledger_dir=ledger,
        prepare=make_prepare(
            instances=instances, root=root / "runs",
            budget_usd=args.budget, provider=provider,
            repo_cache=root / "mirrors", observe_tools=args.observe_tools,
        ),
        execute=make_execute(llm_factory=llm_factory, retry_allowance=allowance),
        score=make_score(ledger_dir=ledger),
        on_cell=announce,
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
