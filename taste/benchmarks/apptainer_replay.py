"""A recorded trial rebuilt in an Apptainer container and graded there, for a host with no Docker (off by default).

The faithfulness test of the Apptainer port. It does, in one Apptainer instance (``taste.brains.apptainer_terminal``),
what a replay job does in Harbor's Docker container (``taste.recovery_study.replay_driver``): no model is asked.

1. The task's image, pinned: its registry manifest and config are read (anonymous pull rights only) and the SIF is
   pulled by manifest digest. The config's digest is the Docker image ID; it is compared with the recorded one.
2. Agent phase, no network (Harbor's rule for these tasks): the commit the task's files start from is read
   (``final_change.base_commit``); the recorded commands of steps 1..k run again in order and each is compared with
   the record (``branch_replay.rebuild``, the worker's own rule and tolerance); the changed paths are read from the
   overlay as a Docker checkpoint lists them; the final change is read with git (``final_change.final_change``).
3. Verifier phase, public network: the instance restarts on the same overlay; the task's tests are copied to
   ``/tests`` and ``/tests/test.sh`` runs with its output in ``/logs/verifier/test-stdout.txt``, as Harbor's
   verifier runs it; the reward is ``/logs/verifier/reward.txt`` (or ``reward.json``).
4. The result is judged as the replay driver judges a rebuild (``ReplayDriver.judge``): faithful, the same grade,
   the same changed files inside the task's folder (git's folder left out), no model cost.

The overlay can be kept as the rebuilt state (``keep``): it is the saved state a later trial would start from
(an overlay directory given to a new instance), which ``check_restore`` tests by grading it again.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import secrets
import shutil
import subprocess
import tarfile
import time
import tomllib
import urllib.parse
import urllib.request
from dataclasses import replace
from pathlib import Path

from taste.benchmarks import final_change
from taste.brains.apptainer_terminal import (
    ApptainerSpec,
    ApptainerTerminalBackend,
    copy_overlay,
    docker_env,
)
from taste.brains.branch_replay import REBUILD_OUTPUT_CHARS, ReplayScript, rebuild, rebuild_timeout
from taste.brains.terminal_broker import TerminalRequest

ACTOR = "taste-apptainer-replay"
ACCEPT = ", ".join(["application/vnd.oci.image.index.v1+json", "application/vnd.oci.image.manifest.v1+json",
                    "application/vnd.docker.distribution.manifest.list.v2+json",
                    "application/vnd.docker.distribution.manifest.v2+json"])
INDEXES = ("application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json")
# Harbor's mounts in a task's Docker container: the trial's folders.
LOG_MOUNTS = ("/logs/agent", "/logs/verifier", "/logs/artifacts")
COMMAND_SECONDS = 600.0   # the terminal's maximum for one command (TrialSettings.command_seconds)


# -- the image ----------------------------------------------------------------------------------------------------

def _get(url, headers, timeout=60):
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), dict(response.headers)


def resolve_image(reference, *, platform=("linux", "amd64")):
    """A public image's pinned identity from its registry, with anonymous pull rights only: the manifest list's
    digest (if any), the platform manifest's digest, the config's digest (Docker's image ID) and the config's Env."""
    name, _, tag = reference.rpartition(":")
    host, _, repository = name.partition("/")
    token_url = f"https://{host}/token?" + urllib.parse.urlencode(
        {"scope": f"repository:{repository}:pull", "service": host})
    token = json.loads(_get(token_url, {})[0])["token"]
    auth = {"Authorization": f"Bearer {token}", "Accept": ACCEPT}
    body, headers = _get(f"https://{host}/v2/{repository}/manifests/{tag}", auth)
    found = {"reference": reference, "repository": f"{host}/{repository}", "index_digest": None}
    digest = headers.get("Docker-Content-Digest") or headers.get("docker-content-digest")
    manifest = json.loads(body)
    if manifest.get("mediaType") in INDEXES or "manifests" in manifest:
        found["index_digest"] = digest
        chosen = next(item for item in manifest["manifests"]
                      if (item.get("platform") or {}).get("os") == platform[0]
                      and (item.get("platform") or {}).get("architecture") == platform[1])
        digest = chosen["digest"]
        manifest = json.loads(_get(f"https://{host}/v2/{repository}/manifests/{digest}", auth)[0])
    config_digest = manifest["config"]["digest"]
    config = json.loads(_get(f"https://{host}/v2/{repository}/blobs/{config_digest}", auth)[0])
    found.update(manifest_digest=digest, config_digest=config_digest,
                 env=list((config.get("config") or {}).get("Env") or []),
                 user=(config.get("config") or {}).get("User") or "",
                 workdir=(config.get("config") or {}).get("WorkingDir") or "",
                 layers=len(manifest.get("layers") or ()),
                 compressed_bytes=sum(int(layer.get("size") or 0) for layer in manifest.get("layers") or ()))
    return found


def same_image(recorded_ids, image):
    """Whether a recorded Docker image ID names this image: its config digest (Docker's classic store) or a
    manifest digest (the containerd store names an image by its index or manifest)."""
    names = {image.get("config_digest"), image.get("manifest_digest"), image.get("index_digest")} - {None}
    recorded = set(recorded_ids or ())
    return {"recorded": sorted(recorded), "matches": sorted(recorded & names),
            "same": None if not recorded else bool(recorded & names)}


def pull(image, sif, *, apptainer="apptainer"):
    """The SIF of the image, pulled by its manifest digest; (path, seconds). An existing SIF is reused."""
    sif = Path(sif)
    if sif.is_file():
        return sif, 0.0
    sif.parent.mkdir(parents=True, exist_ok=True)
    temporary = sif.with_name(sif.name + f".part-{secrets.token_hex(4)}")
    started = time.monotonic()
    done = subprocess.run([apptainer, "pull", str(temporary), f"docker://{image['repository']}@{image['manifest_digest']}"],
                          capture_output=True, timeout=7200, stdin=subprocess.DEVNULL)
    if done.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("apptainer pull failed: " + done.stderr.decode("utf-8", "replace")[-800:])
    temporary.replace(sif)
    return sif, time.monotonic() - started


def squashfs_offset(sif, *, apptainer="apptainer"):
    """Where a SIF's root file system (its squashfs partition) starts, from ``apptainer sif list``."""
    done = subprocess.run([apptainer, "sif", "list", str(sif)], capture_output=True, timeout=120, check=True)
    for line in done.stdout.decode().splitlines():
        if "FS (Squashfs" in line:
            return int(line.split("|")[3].split("-")[0])
    raise RuntimeError("the SIF has no squashfs partition")


def root_owned(sif, out, scratch, *, processors=2, apptainer="apptainer"):
    """The image's files with every owner and group made root's: a squashfs file Apptainer runs as an image;
    (path, seconds). Needed because one uid and one gid are mapped here: a file the image gives another group (the
    SWE-Bench Pro images give /app to gid 1001) cannot be copied up into the overlay, so root could not write it.
    Modes (setuid and setgid bits too), links and times are kept; extended attributes are not."""
    out = Path(out)
    if out.is_file():
        return out, 0.0
    started = time.monotonic()
    rootfs = Path(scratch) / f"rootfs-{secrets.token_hex(4)}"
    temporary = out.with_name(out.name + f".part-{secrets.token_hex(4)}")
    try:
        subprocess.run(["unsquashfs", "-n", "-no-xattrs", "-o", str(squashfs_offset(sif, apptainer=apptainer)),
                        "-d", str(rootfs), str(sif)], capture_output=True, timeout=3600, check=True)
        subprocess.run(["mksquashfs", str(rootfs), str(temporary), "-all-root", "-noappend", "-no-xattrs", "-quiet",
                        "-processors", str(processors)], capture_output=True, timeout=3600, check=True)
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)
        if rootfs.exists():
            subprocess.run(["chmod", "-R", "u+rwX", str(rootfs)], capture_output=True, timeout=600)
            shutil.rmtree(rootfs, ignore_errors=True)
    return out, time.monotonic() - started


# -- one rebuild -------------------------------------------------------------------------------------------------

def task_config(task_dir):
    with open(Path(task_dir) / "task.toml", "rb") as handle:
        return tomllib.load(handle)


def _tar_of(directory):
    """A tar of a folder's contents, every member owned by root, as ``docker cp`` would place them for root."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as out:
        for path in sorted(Path(directory).rglob("*")):
            member = out.gettarinfo(str(path), arcname=str(path.relative_to(directory)))
            member.uid = member.gid = 0
            member.uname = member.gname = "root"
            if member.isfile():
                with open(path, "rb") as handle:
                    out.addfile(member, handle)
            else:
                out.addfile(member)
    return buffer.getvalue()


HARBOR_SHELL = ("bash", "-c")  # Harbor's exec in a task's main container (docker_unix.exec_shell_args)


def verify(backend, task_dir, trial_dir, timeout_seconds, cwd=None):
    """Harbor's verifier in the task's container: tests copied to /tests (a root-owned tar, as Harbor's upload),
    ``test.sh`` run by ``bash -c`` in the image's working folder with its output in the verifier folder, the reward
    read from the folder. {reward, seconds, exit, ...}."""
    started = time.monotonic()
    tests = Path(task_dir) / "tests"
    cwd = cwd or backend.spec.workdir
    copied = backend.run("mkdir -p /tests && tar -xf - -C /tests", "/", 300, stdin=_tar_of(tests), shell=HARBOR_SHELL)
    if copied.code != 0:
        return {"reward": None, "why": "the tests could not be copied in",
                "detail": (copied.stdout + copied.stderr).decode("utf-8", "replace")[-500:]}
    backend.run("chmod +x /tests/test.sh", cwd, 60, shell=HARBOR_SHELL)
    config = task_config(task_dir)
    env = {**(config.get("environment", {}).get("env") or {}), **(config.get("verifier", {}).get("env") or {})}
    exports = "".join(f"export {key}={_quote(value)}; " for key, value in env.items())
    done = backend.run(exports + "(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1", cwd, timeout_seconds,
                       shell=HARBOR_SHELL)
    found = {"exit": done.code, "terminated": done.terminated, "seconds": round(time.monotonic() - started, 1)}
    verifier = Path(trial_dir) / "verifier"
    if (verifier / "reward.json").is_file():
        found["rewards"] = json.loads((verifier / "reward.json").read_text())
        found["reward"] = found["rewards"].get("reward")
    elif (verifier / "reward.txt").is_file() and (verifier / "reward.txt").read_text().strip():
        found["reward"] = float((verifier / "reward.txt").read_text())
    else:
        found.update(reward=None, why="no reward file")
    return found


def _quote(value):
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def rebuild_trial(script_path, step, task_dir, sif, image, *, work, trial_dir, tolerance=0,
                  command_seconds=COMMAND_SECONDS, final_change_chars=60000, verify_timeout=None, name=None,
                  apptainer="apptainer", save_to=None):
    """One recorded trial rebuilt to ``step`` and graded, in Apptainer. The summary a replay job's trial gives
    (``ReplayDriver.judge`` reads it), with the rebuild's rows and its phases' times."""
    raw = Path(script_path).read_bytes()
    script = ReplayScript.from_bytes(raw)
    config = task_config(task_dir)
    workdir = script.source.get("workdir") or "/app"
    trial_dir = Path(trial_dir)
    for folder in ("agent", "verifier", "artifacts"):
        (trial_dir / folder).mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(16)
    hostname = secrets.token_hex(6)  # a Docker container's host name is its ID's first 12 hex digits
    spec = ApptainerSpec(image=str(sif), overlay=str(Path(work) / "overlay"), name=name or f"replay-{token[:12]}",
                         workdir=workdir, network="none", hostname=hostname,
                         env=docker_env(image.get("env"), hostname),
                         binds=tuple((str(trial_dir / folder), f"/logs/{folder}")
                                     for folder in ("agent", "verifier", "artifacts")), apptainer=apptainer)
    agent_seconds = float(config.get("agent", {}).get("timeout_sec") or 3000)
    started = time.time()
    backend = ApptainerTerminalBackend(spec, deadline_unix=started + agent_seconds + 3600)
    # The image as the record names it: the containerd store names an image by its manifest digest.
    summary = {"trial": token, "token": token, "image_id": image.get("manifest_digest"),
               "config_digest": image.get("config_digest"), "cost_usd": 0.0,
               "step": step, "script_sha256": hashlib.sha256(raw).hexdigest(), "workdir": workdir,
               "host": "apptainer", "times": {}}
    try:
        clock = time.monotonic()
        backend.start()
        summary["times"]["start"] = round(time.monotonic() - clock, 2)
        commit, why = final_change.base_commit(backend, workdir)
        rows, effects = [], []

        async def execute(number, run):
            effect = f"effect_{len(effects) + 1:04d}"
            effects.append(effect)
            result = await asyncio.to_thread(backend.execute, TerminalRequest(
                effect, ACTOR, run.executed, run.cwd, rebuild_timeout(run, command_seconds)))
            text = result.stdout.decode("utf-8", errors="replace") + result.stderr.decode("utf-8", errors="replace")
            return text, result.return_code, result.terminated == "timeout"

        def on_row(row, output):
            rows.append({**row, "output": output[:REBUILD_OUTPUT_CHARS], "output_chars": len(output)})

        clock = time.monotonic()
        account = asyncio.run(rebuild(script, step, execute, tolerance=tolerance, on_row=on_row))
        summary["times"]["rebuild"] = summary["prefix_seconds"] = round(time.monotonic() - clock, 2)
        summary.update(faithful=account["faithful"], commands=account["commands"], divergent=account["divergent"],
                       rows=rows, unfaithful_at=None if account["faithful"] else rows[-1]["step"])
        clock = time.monotonic()
        summary["changed_paths"] = backend.changed_paths(mounts=LOG_MOUNTS)
        summary["checkpoint_partial"] = False
        summary["times"]["changes"] = round(time.monotonic() - clock, 2)
        if final_change_chars:
            summary["final_change"] = (final_change.final_change(backend, workdir, commit, final_change_chars)
                                       if commit else {"read": False, "reason": why, "cap_chars": final_change_chars})
        clock = time.monotonic()
        backend.stop()
        if save_to:
            # The files the verifier is about to grade, saved (as a trial's checkpoint is, before its verifier).
            backend.save(save_to)
            summary["saved"] = str(save_to)
            summary["times"]["save"] = round(time.monotonic() - clock, 2)
            clock = time.monotonic()
        backend.spec = replace(backend.spec, network="host")  # Harbor's verifier phase: public network, same files
        backend.start()
        summary["times"]["restart"] = round(time.monotonic() - clock, 2)
        graded = verify(backend, task_dir, trial_dir,
                        verify_timeout or float(config.get("verifier", {}).get("timeout_sec") or 3000),
                        cwd=image.get("workdir") or workdir)
        summary["verifier"] = graded
        summary["reward"] = graded.get("reward")
        summary["times"]["verify"] = graded.get("seconds")
    except Exception as failure:  # the trial's own failure is recorded, never raised past its record
        summary["exception"] = f"{type(failure).__name__}: {str(failure)[:500]}"
    finally:
        try:
            backend.stop()
        except Exception as failure:
            summary["stop_failed"] = f"{type(failure).__name__}: {str(failure)[:300]}"
        summary["seconds"] = round(time.time() - started, 1)
    return summary


def check_restore(saved, task_dir, sif, image, *, work, trial_dir, workdir="/app", name=None, apptainer="apptainer",
                  verify_timeout=None):
    """A saved state brought back: the saved overlay copied to a fresh one (as the namespace's root), a new instance
    on it, its changed paths read and its tests run again. {changed_paths, reward, seconds}. An empty ``saved``
    folder is the image untouched."""
    started = time.monotonic()
    fresh = Path(work) / "overlay"
    trial_dir = Path(trial_dir)
    for folder in ("agent", "verifier", "artifacts"):
        (trial_dir / folder).mkdir(parents=True, exist_ok=True)
    hostname = secrets.token_hex(6)
    spec = ApptainerSpec(image=str(sif), overlay=str(fresh), name=name or f"restore-{secrets.token_hex(6)}",
                         workdir=workdir, network="host", hostname=hostname, env=docker_env(image.get("env"), hostname),
                         binds=tuple((str(trial_dir / folder), f"/logs/{folder}")
                                     for folder in ("agent", "verifier", "artifacts")), apptainer=apptainer)
    backend = ApptainerTerminalBackend(spec)
    found = {}
    try:
        copy_overlay(spec, saved, fresh)
        found["copy_seconds"] = round(time.monotonic() - started, 2)
        backend.start()
        found["changed_paths"] = backend.changed_paths(mounts=LOG_MOUNTS)
        graded = verify(backend, task_dir, trial_dir, verify_timeout or 3000, cwd=image.get("workdir") or workdir)
        found.update(reward=graded.get("reward"), verifier=graded)
    except Exception as failure:
        found["exception"] = f"{type(failure).__name__}: {str(failure)[:500]}"
    finally:
        try:
            backend.stop()
        except Exception as failure:
            found["stop_failed"] = f"{type(failure).__name__}: {str(failure)[:300]}"
    found["seconds"] = round(time.monotonic() - started, 1)
    return found


# -- the judgement -----------------------------------------------------------------------------------------------

def judge(recorded, summary, *, workdir, short=False, name="apptainer-replay"):
    """The replay driver's own judgement of one rebuild (``ReplayDriver.judge``) against its recorded trial."""
    from taste.recovery_study.replay_driver import ReplayDriver

    item = {"recorded": {"reward": recorded.get("reward"), "changed_paths": recorded.get("changed_paths"),
                         "image_id": (recorded.get("image_ids") or [None])[0]},
            "script": {"workdir": workdir}, "short": short}
    return ReplayDriver.judge(None, item, name, summary)
