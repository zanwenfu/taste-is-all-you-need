"""Run mini-swe-agent, unchanged, under the instrument on a benchmark slice.

    python scripts/miniswe_sweep.py --instances 40 --model anthropic/claude-sonnet-4-6 --root /root/mswe40
    python scripts/miniswe_sweep.py --substrate live --candidates data/live_slice_kept.json --instances 40 ...
    python scripts/miniswe_sweep.py --instances 1 --scripted --root /tmp/mswe-smoke      # $0 path check

Everything but the agent is the sweep driver the harness arms use:
selection, materialisation, parity check, ledger, replay, grading.
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
from taste.benchmarks import swebenchlive as live
from taste.benchmarks.miniswe import ScriptedModel, make_miniswe_execute
from taste.benchmarks.swebench_run import make_grade, make_prepare, make_score
from taste.evalrun import run_sweep
from taste.execution import DockerProvider

SMOKE_COMMANDS = [
    "ls",
    "printf 'x = 1\\n' > taste_smoke_note.py",
    "rm -f taste_smoke_note.py",
    "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff",
]


def image_for(instance) -> str:
    image = instance.published_image
    return image if ":" in image.rsplit("/", 1)[-1] else f"{image}:latest"


def local_images() -> set[str]:
    out = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], capture_output=True, text=True, check=False)
    return set(out.stdout.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--substrate", choices=["verified", "live"], default="verified")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--candidates", default="", help="JSON list of instance ids fixing the selection order.")
    ap.add_argument("--only", default="")
    ap.add_argument("--model", default="anthropic/claude-sonnet-4-6",
                    help="litellm model name for the scaffold (default: the paper's Claude worker snapshot).")
    ap.add_argument("--cost-limit", type=float, default=None, help="Override the scaffold's per-instance cost limit (default: its config, $3).")
    ap.add_argument("--scripted", action="store_true", help="No model: replay fixed commands ($0 path check).")
    ap.add_argument("--root", default="/tmp/taste-mswe")
    ap.add_argument("--skip-completed", action="store_true")
    ap.add_argument("--max-consecutive-failures", type=int, default=0)
    ap.add_argument("--sweep-budget", type=float, default=None)
    args = ap.parse_args()

    root = Path(args.root)
    ledger = root / "ledger"
    have = local_images()
    if args.substrate == "live":
        everything = live.load_live_dataset(Path(args.dataset or "data/live_lite.jsonl"))
    else:
        everything = swebench.load_dataset(Path(args.dataset or "data/verified.jsonl"))
    pool = [i for i in everything if image_for(i) in have]
    if args.only:
        wanted = set(args.only.split(","))
        pool = [i for i in pool if i.instance_id in wanted]
    if args.candidates:
        order = json.loads(Path(args.candidates).read_text())
        rank = {iid: k for k, iid in enumerate(order)}
        pool = sorted((i for i in pool if i.instance_id in rank), key=lambda i: rank[i.instance_id])
    else:
        # The harness sweeps' selection: repositories with a mirror first, small ones next.
        cache = root / "mirrors"
        small = ("psf/requests", "pallets/flask", "mwaskom/seaborn", "pytest-dev/pytest")
        pool.sort(key=lambda i: (0 if (cache / f"{i.repo.replace('/', '__')}.git").exists() else 1, 0 if i.repo in small else 1))
    chosen = pool[: args.instances]
    if args.skip_completed:
        done = set()
        for f in (ledger / "evidence").glob("*__MSWE__*.json"):
            ev = json.loads(f.read_text())
            if ev.get("resolved") is not None:
                done.add(ev["instance_id"])
        chosen = [i for i in chosen if i.instance_id not in done]
        print(f"  resuming: {len(done)} completed cell(s) kept, {len(chosen)} to run", flush=True)
    if not chosen:
        print("nothing to run (no images, or all completed)")
        return 2
    print(f"{len(pool)} instance image(s) available; running {len(chosen)} under {'scripted commands' if args.scripted else args.model}:", flush=True)
    for i in chosen:
        object.__setattr__(i, "image", image_for(i))
        print(f"  {i.instance_id:32s} {i.repo:26s} |P2P|={len(i.pass_to_pass)}", flush=True)
    instances = {i.instance_id: i for i in chosen}

    provider = DockerProvider(env_prefix="") if args.substrate == "live" else DockerProvider()
    execute = make_miniswe_execute(
        model_name=args.model, cost_limit=args.cost_limit,
        model_factory=(lambda: ScriptedModel(list(SMOKE_COMMANDS))) if args.scripted else None,
    )

    def announce(record) -> None:
        why = record.error or getattr(record, "failure_reason", None) or ""
        print(f"  [cell] {record.task:30s} {record.status:9s} ${record.billed_usd:6.4f} {str(why)[:160]}", flush=True)

    started = time.time()
    report = run_sweep(
        tasks=[i.instance_id for i in chosen], arms=["MSWE"], trials=1, ledger_dir=ledger,
        prepare=make_prepare(
            instances=instances, root=root / "runs", provider=provider, route_execution=True, observe_tools=True,
            parity_check=live.live_parity_check if args.substrate == "live" else None,
        ),
        execute=execute,
        score=make_score(
            ledger_dir=ledger,
            grade=make_grade(grader=live.grade_live_in_sandbox if args.substrate == "live" else None),
            suite_factory=live.live_suite_factory if args.substrate == "live" else None,
        ),
        on_cell=announce,
        max_consecutive_failures=args.max_consecutive_failures or None,
        sweep_budget_usd=args.sweep_budget,
    )
    print(f"\n=== {len(report.results)} cells in {time.time() - started:.0f}s ===")
    for r in report.results:
        print(f"  {r.task:32s} {r.status:10s} ${r.billed_usd:7.4f} {r.error or ''}")
    print(f"ledger  : {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
