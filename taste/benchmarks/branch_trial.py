"""The trial owner's part of a branch: the replay script into the trial, and checkpoints of step k.

The worker serves a branch (``taste.brains.branch_replay``). Before it starts,
the trial's owner reads the replay script and note named in the settings,
checks the branch against the script, and gives the worker a read-only copy
of the script in the trial's directory, held to its digest. In restore mode
the owner puts a saved checkpoint back into the fresh task container before
the agent starts, and refuses the trial if the files do not then match it.

A rebuild with live off ends with the container's files as they were after
step k, and the verifier grades those. When asked, the owner first saves a
checkpoint of them, which later branches from step k restore instead of
rebuilding. It is saved only if the branch was faithful. A checkpoint
directory holds the manifest (``taste.brains.docker_checkpoint``), the tar it
names, and ``branch.json``: the replay script's digest and the step, so that
it is never restored into another branch. A directory that already holds one
is never written over.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from taste.brains.branch_replay import ReplayScript, check_branch, text_sha

BRANCH_SCRIPT = "branch-script.json"
MANIFEST = "manifest.json"
IDENTITY = "branch.json"
IDENTITY_SCHEMA = "taste.branch/Checkpoint/1"


@dataclass(frozen=True)
class BranchInputs:
    """The replay script as read, and the note shown at step k."""

    raw: bytes
    script: ReplayScript
    note: str

    @property
    def sha256(self):
        return hashlib.sha256(self.raw).hexdigest()


def load_inputs(settings):
    """The branch the settings name, checked against its script; None for a fresh trial."""
    if not settings.branch:
        return None
    raw = Path(settings.branch).read_bytes()
    script = ReplayScript.from_bytes(raw)
    note = Path(settings.branch_note).read_text(encoding="utf-8") if settings.branch_note else ""
    if settings.branch_note and not note.strip():
        raise ValueError("the branch note is empty")
    check_branch(script, settings.branch_step, mode=settings.branch_mode, override=settings.branch_override,
                 note=note)
    return BranchInputs(raw, script, note)


def disclosure(settings, inputs):
    """What a branch trial's record says of its branch before it runs."""
    source = inputs.script.source
    return {"script_sha256": inputs.sha256, "step": settings.branch_step, "steps": len(inputs.script.steps),
            "mode": settings.branch_mode, "live": settings.branch_live, "override": settings.branch_override,
            "tolerance": settings.branch_tolerance,
            **({"note_sha256": text_sha(inputs.note)} if inputs.note else {}),
            "source": {key: source.get(key) for key in ("trial", "run_id", "model")}}


def _identity(inputs, step):
    return {"schema": IDENTITY_SCHEMA, "script_sha256": inputs.sha256, "step": step}


def restore_checkpoint(backend, settings, inputs):
    """Put the saved files of step k back into the task container; their summary. Raises unless exact."""
    from taste.brains.docker_checkpoint import CheckpointManifest

    directory = Path(settings.branch_checkpoint)
    identity = json.loads((directory / IDENTITY).read_text(encoding="utf-8"))
    if {key: identity.get(key) for key in ("schema", "script_sha256", "step")} != _identity(
            inputs, settings.branch_step):
        raise ValueError("this checkpoint is of another replay script or step")
    manifest = CheckpointManifest.from_dict(json.loads((directory / MANIFEST).read_text(encoding="utf-8")))
    receipt = backend.restore(manifest, directory, any_container=True)
    summary = receipt.summary()
    if not receipt.exact:
        raise RuntimeError(f"the restored files differ from the checkpoint at {len(receipt.mismatches)} paths")
    return summary


def save_checkpoint(backend, settings, inputs, replay, trial):
    """Save the container's files, which are those of step k after a faithful rebuild with live off."""
    directory = Path(settings.branch_checkpoint)
    if not (isinstance(replay, dict) and replay.get("faithful") is True):
        return {"saved": False, "reason": "the branch was not faithful"}
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if (directory / MANIFEST).exists() or (directory / IDENTITY).exists():
        return {"saved": False, "reason": "the directory already holds a checkpoint"}
    manifest = backend.checkpoint(directory)
    _write(directory / IDENTITY, {**_identity(inputs, settings.branch_step), "trial": trial,
                                  "image": manifest.image, "replay": replay})
    # The manifest last: a directory with one holds a whole checkpoint.
    _write(directory / MANIFEST, manifest.to_dict())
    summary = manifest.summary()
    return {"saved": True, **{key: summary[key] for key in (
        "tar_sha256", "bytes", "added", "changed", "deleted", "partial", "left_out")}}


def _write(path, value):
    temporary = path.with_name("." + path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def replay_outcome(nested):
    """The branch's account in a settled goal trajectory (its hosted worker's), or None."""
    for sub in (nested or {}).get("subagent_trajectories") or ():
        branch = (sub.get("extra") or {}).get("branch")
        if isinstance(branch, dict):
            return branch
    return None
