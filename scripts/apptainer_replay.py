"""Rebuild one recorded trial in Apptainer, grade it there and judge it against its Docker record (no model asked).

    python3 -I scripts/apptainer_replay.py --script <replay script> --recorded recorded.json --token <trial> \\
        --task-dir <SWE-Bench Pro task folder> --sif-dir <SIF cache> --work <node-local folder> --out result.json \\
        [--step N] [--check-restore] [--empty-control]

The faithfulness test of the Apptainer port (``taste.benchmarks.apptainer_replay``), for a host with no Docker and
no root (Duke's compute nodes). ``recorded.json`` holds, per trial token, what the Docker run recorded: ``reward``,
``changed_paths`` (the trial's own final save), ``image_ids`` and ``steps``. Steps:

1. the task image's pinned identity from its registry, compared with the recorded image ID; the SIF pulled by digest;
2. the rebuild to the trial's last step (no network), its changed paths and final change; the grade (public network);
3. the replay driver's judgement (faithful, grade, changed files, no model cost);
4. ``--check-restore``: the rebuilt overlay copied to a fresh one and graded again (saving and restoring a state);
5. ``--empty-control``: the untouched image graded (a check that can fail: the task's tests must not pass on it).

Writes one JSON result; exit 0 if the rebuild is kept (and every asked check agrees), 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste.benchmarks import apptainer_replay as replay
from taste.brains.branch_replay import ReplayScript


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--recorded", type=Path, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--sif-dir", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True, help="node-local folder for overlays (removed after)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--tolerance", type=int, default=0)
    parser.add_argument("--check-restore", action="store_true")
    parser.add_argument("--empty-control", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--keep-owners", action="store_true",
                        help="run the pulled SIF as is (its files keep their owners; root cannot write those of "
                             "another group here)")
    parser.add_argument("--processors", type=int, default=2, help="CPUs for building the root-owned image")
    arguments = parser.parse_args(argv)

    started = time.time()
    recorded = json.loads(arguments.recorded.read_text())[arguments.token]
    script = ReplayScript.from_bytes(arguments.script.read_bytes())
    steps = len(script.steps)
    step = arguments.step or min(int(recorded.get("steps") or steps), steps)
    short = step < int(recorded.get("steps") or step) or int(script.source.get("dropped_steps") or 0) > 0
    workdir = script.source.get("workdir") or "/app"
    config = replay.task_config(arguments.task_dir)
    reference = config["environment"]["docker_image"]
    result = {"token": arguments.token, "task": recorded.get("task"), "step": step, "script_steps": steps,
              "short": short, "workdir": workdir, "image_reference": reference}

    image = replay.resolve_image(reference)
    result["image"] = {key: image[key] for key in image if key != "env"} | {"env_names": [
        item.split("=", 1)[0] for item in image["env"]]}
    result["image_check"] = replay.same_image(recorded.get("image_ids"), image)
    sif, pull_seconds = replay.pull(image, arguments.sif_dir / (image["config_digest"].replace(":", "-") + ".sif"))
    result["sif"] = {"path": str(sif), "bytes": sif.stat().st_size, "pull_seconds": round(pull_seconds, 1)}
    work = arguments.work / arguments.token
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    if not arguments.keep_owners:
        # One uid and one gid are mapped here: the image's files are made root's so that root can write them.
        sif, seconds = replay.root_owned(sif, sif.with_suffix(".root.sqfs"), work, processors=arguments.processors)
        result["root_owned"] = {"path": str(sif), "bytes": sif.stat().st_size, "seconds": round(seconds, 1),
                                "sha256": _sha256(sif)}
    rebuilt = replay.rebuild_trial(arguments.script, step, arguments.task_dir, sif, image, work=work / "rebuild",
                                   trial_dir=work / "trial-rebuild", tolerance=arguments.tolerance)
    result["rebuild"] = rebuilt
    result["judged"] = replay.judge(recorded, rebuilt, workdir=workdir, short=short)
    kept = bool(result["judged"].get("kept"))
    result["test_stdout_tail"] = _tail(work / "trial-rebuild" / "verifier" / "test-stdout.txt")

    if arguments.check_restore and (work / "rebuild" / "overlay").is_dir():
        restored = replay.check_restore(work / "rebuild" / "overlay", arguments.task_dir, sif, image,
                                        work=work / "restore", trial_dir=work / "trial-restore", workdir=workdir)
        restored["same_changed_paths"] = restored.get("changed_paths") == rebuilt.get("changed_paths")
        restored["same_reward"] = restored.get("reward") == rebuilt.get("reward")
        result["restore"] = restored
        kept = kept and restored["same_changed_paths"] and restored["same_reward"]
    if arguments.empty_control:
        (work / "control" / "overlay").mkdir(parents=True)
        control = replay.check_restore(work / "control" / "overlay", arguments.task_dir, sif, image,
                                       work=work / "control2", trial_dir=work / "trial-control", workdir=workdir)
        control["tests_fail_on_image"] = control.get("reward") == 0.0
        control["test_stdout_tail"] = _tail(work / "trial-control" / "verifier" / "test-stdout.txt")
        result["empty_control"] = control
        kept = kept and control["tests_fail_on_image"]

    result["ok"] = kept
    result["seconds"] = round(time.time() - started, 1)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    if not arguments.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    judged = result["judged"]
    print(json.dumps({"token": arguments.token, "ok": kept, "faithful": judged.get("faithful"),
                      "reward": judged.get("reward"), "grade": judged.get("grade"),
                      "files_equal": (judged.get("files") or {}).get("equal"), "why": judged.get("why"),
                      "image_same": result["image_check"]["same"], "seconds": result["seconds"],
                      "times": rebuilt.get("times"), "restore_ok": (result.get("restore") or {}).get("same_reward"),
                      "control_fails": (result.get("empty_control") or {}).get("tests_fail_on_image")}))
    return 0 if kept else 1


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tail(path, lines=30):
    try:
        return Path(path).read_text(errors="replace").splitlines()[-lines:]
    except OSError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
