"""Replay public mini-swe-agent runs under the instrument.

    python scripts/public_replay.py cell  --root /data/pubreplay --submission <name> --task <instance_id>
    python scripts/public_replay.py pool  --root /data/pubreplay --cells cells.txt --jobs 28
    python scripts/public_replay.py fetch --root /data/pubreplay --cells cells.txt
    python scripts/public_replay.py pull  --root /data/pubreplay --cells cells.txt
    python scripts/public_replay.py plan  --root /data/pubreplay --submissions subs.txt --tasks 40 --out cells.txt

A cell is one (submission, task): the published run of one model on one task.
Everything a cell produces lands in <root>/cells/<submission>/<task>/:

  ledger/                  the sweep ledger and evidence sidecar, as the live arm writes them
  manifest.json            the replay's own record (commands replayed, observations, recorder errors)
  replay.jsonl             one line per replayed command beside the recorded one
  states.jsonl             one line per tested state (failures, unreported tests, the output's tail)
  shadow.jsonl             the observation timeline
  states.json.gz           every observed tree as a patch against the root (the workspace is deleted)
  fidelity.json            the replay against the record: same submission? same exit statuses?
  status.json              written last; its presence means the cell is finished

<root> holds verified.jsonl, submissions.json (name -> version, trajectory prefix),
details/<submission>.json (the leaderboard's per-instance verdicts) and
trajs/<submission>/<task>.traj.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ARM = "MSWE"


def cell_dir(root: Path, submission: str, task: str) -> Path:
    return root / "cells" / submission / task


def read_cells(path: Path) -> list[tuple[str, str]]:
    cells = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and not line.startswith("#"):
            cells.append((parts[0], parts[1]))
    return cells


def load_submissions(root: Path) -> dict[str, dict]:
    if not (root / "submissions.json").exists():
        return {}
    raw = json.loads((root / "submissions.json").read_text())
    rows = raw if isinstance(raw, list) else raw.get("submissions", raw)
    if isinstance(rows, dict):
        return rows
    return {row["name"]: row for row in rows}


def trajectory_path(root: Path, submission: str, task: str) -> Path:
    return root / "trajs" / submission / f"{task}.traj.json"


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1e9


def build_edits(root: Path, image: str, task: str, base_commit: str) -> tuple[tuple[str, ...], str]:
    """The image build's own edits, cached per task: untracked leftovers by path, and the
    tracked-file edit as a patch against the instance's base commit (committed or not).
    See public_replay.same_patch for how the comparison uses them."""
    from taste.benchmarks.public_replay import build_edit_paths

    cache = root / "build_edits" / f"{task}.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        if "build_patch" in saved:
            return tuple(saved["paths"]), saved["build_patch"]

    def in_image(script: str) -> str:
        done = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "bash", image, "-c", script],
                              capture_output=True, text=True, timeout=300)
        if done.returncode != 0:
            raise RuntimeError(f"{script!r} in {image} failed: {done.stderr[-300:]}")
        return done.stdout

    porcelain = in_image("cd /testbed && git status --porcelain")
    untracked = build_edit_paths("\n".join(line for line in porcelain.splitlines() if line.startswith("??")))
    build_patch = in_image(f"cd /testbed && git diff {base_commit}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Two cells of one task may compute this at once: write whole, then rename.
    tmp = cache.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"image": image, "porcelain": porcelain, "paths": list(untracked),
                               "base_commit": base_commit, "build_patch": build_patch}, indent=1))
    tmp.replace(cache)
    return untracked, build_patch


def leaderboard_resolved(root: Path, submission: str, task: str) -> bool | None:
    path = root / "details" / f"{submission}.json"
    if not path.exists():
        return None
    row = json.loads(path.read_text()).get(task)
    return None if not isinstance(row, dict) or "resolved" not in row else bool(row["resolved"])


def write_status(directory: Path, state: str, **fields) -> dict:
    status = {"state": state, "finished_at": time.time(), **fields}
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / "status.json.tmp"
    tmp.write_text(json.dumps(status, indent=1, sort_keys=True))
    tmp.replace(directory / "status.json")
    return status


def planted_trajectory(path: Path):
    """A hand-written command list as a trajectory: the instrument's known-answer check.

    ``{"instance_id": ..., "commands": [...]}``. Nothing was recorded, so every
    recorded exit status is unknown and the per-command comparison is empty;
    the submission is whatever the sentinel command prints (usually empty).
    """
    from taste.benchmarks.public_trajectories import PublicTrajectory, RecordedCommand

    spec = json.loads(path.read_text())
    commands = list(spec["commands"])
    recorded = tuple(
        RecordedCommand(index=i, command=c, returncode=None, output="", output_truncated=False, timed_out=False)
        for i, c in enumerate(commands)
    )
    submit = next((i for i, c in enumerate(commands) if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in c), None)
    return PublicTrajectory(
        instance_id=spec["instance_id"], mini_version="planted", trajectory_format="planted",
        exit_status="Submitted" if submit is not None else "Planted", submission=spec.get("submission", ""),
        commands=recorded, submit_index=submit, sentinel="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" if submit is not None else None,
        unexecuted_turns=0, timeout=int(spec.get("timeout", 60)), env=dict(spec.get("env", {})), cwd="/testbed",
        environment_class="docker", step_limit=None, cost_limit=None, api_calls=None, cost_usd=None,
    )


def progress_suite(instance):
    """The per-state suite with the task's target tests as members too.

    The suite command already runs the files holding both lists (the official
    test patch is applied first), so naming FAIL_TO_PASS as members costs
    nothing and records, per state, how far the fix had got. They fail at the
    base by definition, so they never enter the set of previously passing
    tests that a break is measured against.
    """
    import dataclasses

    from taste.benchmarks import swebench

    suite = swebench.pass_to_pass_suite(instance)
    return dataclasses.replace(suite, members=tuple(instance.pass_to_pass) + tuple(instance.fail_to_pass))


def run_cell(root: Path, submission: str, task: str, *, force: bool = False, planted: Path | None = None) -> dict:
    from taste.benchmarks import swebench
    from taste.benchmarks.public_replay import (
        NetworkedAgentProvider,
        export_states,
        make_replay_execute,
        make_replay_grade,
        make_state_log_factory,
        patch_files,
        same_patch,
    )
    from taste.benchmarks.public_trajectories import parse_trajectory
    from taste.benchmarks.swebench_run import make_prepare, make_score
    from taste.evalrun import run_sweep
    from taste.execution import DockerProvider
    from taste.shadow import load_timeline

    directory = cell_dir(root, submission, task)
    if (directory / "status.json").exists() and not force:
        return json.loads((directory / "status.json").read_text())
    if directory.exists():
        # An unfinished attempt: replays cost nothing, so it is redone from scratch rather
        # than reconciled (its journal would block a resumed sweep).
        shutil.rmtree(directory)
    directory.mkdir(parents=True)

    meta = load_submissions(root).get(submission, {})
    version = str(meta.get("version") or meta.get("mini_version") or meta.get("mini-swe-agent_version") or "")
    if planted is None and not trajectory_path(root, submission, task).exists():
        # The submission published no run for this task (gemini-3-5-flash has 441 of 500).
        return write_status(directory, "missing", error="no published trajectory for this task")
    try:
        if planted is not None:
            trajectory = planted_trajectory(planted)
        else:
            trajectory = parse_trajectory(json.loads(trajectory_path(root, submission, task).read_text()), mini_version=version)
    except Exception as exc:
        return write_status(directory, "unparseable", error=f"{type(exc).__name__}: {exc}")
    if trajectory.instance_id and trajectory.instance_id != task:
        return write_status(directory, "unparseable", error=f"trajectory is for {trajectory.instance_id}")

    instances = {i.instance_id: i for i in swebench.load_dataset(root / "verified.jsonl")}
    instance = instances[task]
    object.__setattr__(instance, "image", instance.published_image)
    ignore, build_patch = build_edits(root, instance.image, task, instance.base_commit)

    inspected = subprocess.run(["docker", "image", "inspect", "--format", "{{json .Config.Env}}", instance.image],
                               capture_output=True, text=True, timeout=120)
    image_env = dict(item.split("=", 1) for item in json.loads(inspected.stdout or "[]") if "=" in item)
    import docker as docker_sdk

    # A patient client: with two dozen replays starting at once the daemon can take
    # over a minute to remove a container, and an unconfirmed removal blocks the cell.
    docker = DockerProvider(client=docker_sdk.from_env(timeout=600))
    provider = NetworkedAgentProvider(docker)
    started = time.time()
    try:
        report = run_sweep(
            tasks=[task], arms=[ARM], trials=1, ledger_dir=directory / "ledger",
            prepare=make_prepare(instances={task: instance}, root=directory / "runs", provider=provider,
                                 route_execution=True, observe_tools=True),
            execute=make_replay_execute(lambda cell: trajectory, source=submission, image_env=image_env),
            score=make_score(ledger_dir=directory / "ledger", grade=make_replay_grade(),
                             suite_factory=progress_suite, executor_factory=make_state_log_factory()),
        )
    finally:
        # Only this provider's containers, found by the owner label it stamps on each launch.
        owner = getattr(docker, "_owner_id", None)
        if owner:
            leftover = subprocess.run(["docker", "ps", "-aq", "--filter", f"label=taste.owner={owner}"],
                                      capture_output=True, text=True).stdout.split()
            if leftover:
                subprocess.run(["docker", "rm", "-f", *leftover], capture_output=True)

    cell_result = report.results[0] if report.results else None
    workspace = directory / "runs" / task / ARM / "t1"
    gitdir = workspace / ".git" / "taste"
    for name in ("manifest.json", "replay.jsonl", "states.jsonl", "shadow.jsonl", "replay_submission.diff"):
        if (gitdir / name).exists():
            shutil.copy2(gitdir / name, directory / name)
    manifest = json.loads((directory / "manifest.json").read_text()) if (directory / "manifest.json").exists() else {}
    session = manifest.get("session", "")
    timeline = load_timeline(gitdir, session) if session else ()
    n_states = export_states(workspace, timeline, directory / "states.json.gz") if timeline else 0

    evidence_path = directory / "ledger" / "evidence" / f"{task}__{ARM}__t1.json"
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
    replayed_submission = (directory / "replay_submission.diff").read_text() if (directory / "replay_submission.diff").exists() else None
    lines = [json.loads(line) for line in (directory / "replay.jsonl").read_text().splitlines()] if (directory / "replay.jsonl").exists() else []
    compared = [line for line in lines if line.get("rc_match") is not None]
    if trajectory.submit_index is not None and replayed_submission is not None:
        same, differing = same_patch(trajectory.submission, replayed_submission, ignore=ignore, build=build_patch)
    else:
        same, differing = None, []
    fidelity = {
        "submission": submission,
        "task": task,
        "recorded_submitted": trajectory.submit_index is not None,
        "replay_submitted": replayed_submission is not None,
        "same_submission": same,
        "differing_files": differing,
        "ignored_build_edits": list(ignore),
        "build_edited_files": sorted(patch_files(build_patch)),
        "commands_recorded": len(trajectory.commands),
        "commands_replayed": manifest.get("commands_replayed"),
        "rc_compared": len(compared),
        "rc_matched": sum(1 for line in compared if line["rc_match"]),
        "shadow_errors": manifest.get("shadow_errors"),
        "sync_skipped_paths": manifest.get("sync_skipped_paths"),
        "replay_exception": manifest.get("exception"),
        "leaderboard_resolved": leaderboard_resolved(root, submission, task),
        "replay_resolved": evidence.get("resolved"),
        "observations": manifest.get("observations"),
        "states_exported": n_states,
    }
    (directory / "fidelity.json").write_text(json.dumps(fidelity, indent=1, sort_keys=True))
    if workspace.exists():
        shutil.rmtree(directory / "runs", ignore_errors=True)
    return write_status(
        directory, "done" if evidence else "no_evidence",
        cell_status=getattr(cell_result, "status", None), cell_error=getattr(cell_result, "error", None),
        seconds=round(time.time() - started, 1),
        same_submission=same, resolved=evidence.get("resolved"), leaderboard_resolved=fidelity["leaderboard_resolved"],
        observations=manifest.get("observations"), episodes=len(evidence.get("episodes", [])),
    )


def cmd_cell(args) -> int:
    root = Path(args.root)
    try:
        status = run_cell(root, args.submission, args.task, force=args.force,
                          planted=Path(args.planted) if args.planted else None)
    except Exception as exc:
        status = write_status(cell_dir(root, args.submission, args.task), "crashed",
                              error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-4000:])
    print(json.dumps({"submission": args.submission, "task": args.task, **{k: status.get(k) for k in (
        "state", "seconds", "same_submission", "resolved", "leaderboard_resolved", "observations", "episodes", "error")}}),
          flush=True)
    return 0 if status.get("state") in ("done", "unparseable", "missing") else 1


def cmd_pool(args) -> int:
    """Run cells in parallel subprocesses, oldest first, holding a disk floor."""
    root = Path(args.root)
    redo = {state for state in args.redo.split(",") if state}

    def pending(cell: tuple[str, str]) -> bool:
        status = cell_dir(root, *cell) / "status.json"
        if args.force or not status.exists():
            return True
        return json.loads(status.read_text()).get("state") in redo

    cells = [c for c in read_cells(Path(args.cells)) if pending(c)]
    log = (root / "pool.log").open("a", buffering=1)
    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} pool start: {len(cells)} cells, {args.jobs} jobs\n")

    def one(cell: tuple[str, str]) -> None:
        submission, task = cell
        while free_gb(root) < args.min_free_gb:
            log.write(f"{time.strftime('%H:%M:%S')} waiting: {free_gb(root):.0f} GB free < {args.min_free_gb}\n")
            time.sleep(60)
        out = root / "logs" / submission / f"{task}.log"
        out.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, __file__, "cell", "--root", str(root), "--submission", submission, "--task", task]
        if args.force or (cell_dir(root, submission, task) / "status.json").exists():
            command.append("--force")
        with out.open("w") as handle:
            try:
                done = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=args.cell_timeout)
                code = done.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
                write_status(cell_dir(root, submission, task), "timed_out", error=f"over {args.cell_timeout}s")
        log.write(f"{time.strftime('%H:%M:%S')} {submission} {task} exit={code}\n")

    with futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(one, cells))
    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} pool done\n")
    return 0


def cmd_refidelity(args) -> int:
    """Recompute every finished cell's patch comparison from its stored files; nothing is re-run.

    For when the comparison itself changes (a normalisation added after a cell
    finished). Run it with no pool running: it rewrites fidelity.json and the
    status's copy of the verdict.
    """
    from taste.benchmarks.public_replay import same_patch
    from taste.benchmarks.public_trajectories import parse_trajectory

    root = Path(args.root)
    subs = load_submissions(root)
    checked = changed = 0
    for status_path in sorted((root / "cells").glob("*/*/status.json")):
        directory = status_path.parent
        submission, task = directory.parent.name, directory.name
        status = json.loads(status_path.read_text())
        submitted_diff = directory / "replay_submission.diff"
        if submission == "planted" or status.get("state") != "done" or not submitted_diff.exists():
            continue
        fidelity = json.loads((directory / "fidelity.json").read_text())
        if not fidelity.get("recorded_submitted"):
            continue
        meta = subs.get(submission, {})
        version = str(meta.get("version") or "")
        trajectory = parse_trajectory(json.loads(trajectory_path(root, submission, task).read_text()), mini_version=version)
        cache = json.loads((root / "build_edits" / f"{task}.json").read_text())
        same, differing = same_patch(trajectory.submission, submitted_diff.read_text(),
                                     ignore=tuple(cache["paths"]), build=cache.get("build_patch", ""))
        checked += 1
        if same != fidelity.get("same_submission") or differing != fidelity.get("differing_files"):
            changed += 1
            print(f"  {submission} {task}: same {fidelity.get('same_submission')} -> {same} {differing}")
        fidelity.update(same_submission=same, differing_files=differing, fidelity_recomputed_at=time.time())
        (directory / "fidelity.json").write_text(json.dumps(fidelity, indent=1, sort_keys=True))
        write_status(directory, status.pop("state"), **{**status, "same_submission": same})
    print(f"recomputed {checked} cells; {changed} changed")
    return 0


def cmd_fetch(args) -> int:
    """Download each cell's trajectory from the public bucket (anonymous HTTPS, no credentials)."""
    root = Path(args.root)
    subs = load_submissions(root)
    todo = [c for c in read_cells(Path(args.cells)) if not trajectory_path(root, *c).exists()]

    def one(cell: tuple[str, str]) -> str:
        submission, task = cell
        prefix = str(subs[submission].get("trajs") or subs[submission].get("trajs_s3") or "").rstrip("/")
        if prefix.startswith("https://github.com/"):
            # A submission hosted on GitHub: .../{owner}/{repo}/tree/{branch}/{path} -> raw files.
            owner, repo, _tree, branch, *path = prefix.removeprefix("https://github.com/").split("/")
            base = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{'/'.join(path)}"
        else:
            base = "https://swe-bench-submissions.s3.amazonaws.com/" + prefix.removeprefix("s3://swe-bench-submissions/")
        for url in (f"{base}/{task}/{task}.traj.json", f"{base}/{task}.traj.json"):
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    payload = response.read()
            except Exception:
                continue
            target = trajectory_path(root, submission, task)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            return "ok"
        return f"MISSING {submission} {task}"

    with futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(one, todo))
    missing = [r for r in results if r != "ok"]
    print(f"fetched {len(results) - len(missing)} of {len(todo)}; {len(missing)} missing")
    for line in missing:
        print(" ", line)
    return 0 if not missing else 1


def sample_tasks(instances, n: int, seed: int, exclude: dict[str, str] | None = None) -> list[str]:
    """The study's task sample: Verified ids in a seeded random order, keeping those that can host a break.

    An instance with no PASS_TO_PASS tests has no previously passing test to
    break, so it is skipped before sampling (as the live arm did). ``exclude``
    maps an id to the recorded reason it was dropped after being drawn (a
    suite too slow to test every state, say); the next id in the order takes
    its place, so the sample stays a prefix of one fixed random order.
    """
    import random

    order = sorted(i.instance_id for i in instances if i.pass_to_pass)
    random.Random(seed).shuffle(order)
    exclude = exclude or {}
    return [iid for iid in order if iid not in exclude][:n]


def cmd_plan(args) -> int:
    from taste.benchmarks import swebench

    root = Path(args.root)
    instances = swebench.load_dataset(root / "verified.jsonl")
    exclude = json.loads(Path(args.exclude).read_text()) if args.exclude else {}
    tasks = sample_tasks(instances, args.tasks, args.seed, exclude)
    submissions = [line.split()[0] for line in Path(args.submissions).read_text().splitlines() if line.strip() and not line.startswith("#")]
    cells = [(s, t) for t in tasks for s in submissions]
    Path(args.out).write_text("".join(f"{s} {t}\n" for s, t in cells))
    (Path(args.out).with_suffix(".plan.json")).write_text(json.dumps(
        {"seed": args.seed, "tasks": tasks, "submissions": submissions, "excluded": exclude, "cells": len(cells)}, indent=1))
    print(f"{len(tasks)} tasks x {len(submissions)} submissions = {len(cells)} cells -> {args.out}")
    return 0


def cmd_pull(args) -> int:
    from taste.benchmarks import swebench

    root = Path(args.root)
    tasks = sorted({task for _, task in read_cells(Path(args.cells))})
    instances = {i.instance_id: i for i in swebench.load_dataset(root / "verified.jsonl")}
    images = [instances[t].published_image for t in tasks]

    def one(image: str) -> str:
        done = subprocess.run(["docker", "pull", "-q", image], capture_output=True, text=True, timeout=3600)
        return "ok" if done.returncode == 0 else f"FAILED {image}: {done.stderr[-200:]}"

    with futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(one, images))
    failed = [r for r in results if r != "ok"]
    print(f"pulled {len(results) - len(failed)} of {len(images)}")
    for line in failed:
        print(" ", line)
    return 0 if not failed else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cell")
    c.add_argument("--root", required=True)
    c.add_argument("--submission", required=True)
    c.add_argument("--task", required=True)
    c.add_argument("--force", action="store_true")
    c.add_argument("--planted", default="", help="a known-answer command list instead of a published run")
    p = sub.add_parser("pool")
    p.add_argument("--root", required=True)
    p.add_argument("--cells", required=True)
    p.add_argument("--jobs", type=int, default=24)
    p.add_argument("--cell-timeout", type=int, default=4 * 3600)
    p.add_argument("--min-free-gb", type=float, default=60.0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--redo", default="", help="comma-separated finished states to run again, e.g. crashed,timed_out")
    f = sub.add_parser("fetch")
    f.add_argument("--root", required=True)
    f.add_argument("--cells", required=True)
    u = sub.add_parser("pull")
    u.add_argument("--root", required=True)
    u.add_argument("--cells", required=True)
    u.add_argument("--jobs", type=int, default=4)
    r = sub.add_parser("refidelity")
    r.add_argument("--root", required=True)
    n = sub.add_parser("plan")
    n.add_argument("--root", required=True)
    n.add_argument("--submissions", required=True, help="one submission name per line")
    n.add_argument("--tasks", type=int, required=True)
    n.add_argument("--seed", type=int, default=20261023)
    n.add_argument("--exclude", default="", help="JSON {instance_id: reason} for drawn tasks that were dropped")
    n.add_argument("--out", required=True)
    args = ap.parse_args()
    return {"cell": cmd_cell, "pool": cmd_pool, "fetch": cmd_fetch, "pull": cmd_pull, "plan": cmd_plan,
            "refidelity": cmd_refidelity}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
