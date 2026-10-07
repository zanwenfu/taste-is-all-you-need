"""The trial owner's part of a branch: the replay script into the trial, and checkpoints of a run's files.

The worker serves a branch (``taste.brains.branch_replay``). Before it starts,
the trial's owner reads the replay script and note named in the settings,
checks the branch against the script, and gives the worker a read-only copy
of the script in the trial's directory, held to its digest. In restore mode
the owner puts a saved checkpoint back into the fresh task container before
the agent starts, and refuses the trial if the files do not then match it.

Checkpoints are saved two ways, each only into a directory that holds none:

- A rebuild with live off ends with the container's files as they were after
  step k, and the verifier grades those. When asked (``branch_checkpoint``),
  the owner first saves them, which later branches from step k restore
  instead of rebuilding. Only a faithful branch's files are saved.
- Any trial can leave its final files (``final_checkpoint``), saved when it
  ends and before its verifier runs: what a checker is given a copy of. They
  are restored for a branch from the last step of the replay script exported
  from that same trial, when the script left out none of the run's steps.

A checkpoint directory holds the manifest (``taste.brains.docker_checkpoint``),
the tar it names, and ``branch.json``, which says what the files are: the
replay script's digest and the step, or the trial whose final files they are.
A checkpoint is never restored for another branch than the one it names.
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
                 note=note, replay=settings.branch_replay == "on")
    return BranchInputs(raw, script, note)


def disclosure(settings, inputs):
    """What a branch trial's record says of its branch before it runs."""
    source = inputs.script.source
    return {"script_sha256": inputs.sha256, "step": settings.branch_step, "steps": len(inputs.script.steps),
            "mode": settings.branch_mode, "live": settings.branch_live, "replay": settings.branch_replay,
            "override": settings.branch_override, "tolerance": settings.branch_tolerance,
            **({"note_sha256": text_sha(inputs.note)} if inputs.note else {}),
            "source": {key: source.get(key) for key in ("trial", "run_id", "model")}}


def _identity(inputs, step):
    return {"schema": IDENTITY_SCHEMA, "script_sha256": inputs.sha256, "step": step}


def names_branch(identity, inputs, step):
    """Whether a checkpoint's ``branch.json`` says it holds the files of this script's step ``step``."""
    if not isinstance(identity, dict) or identity.get("schema") != IDENTITY_SCHEMA:
        return False
    if identity.get("final") is True:
        # A trial's final files are those after the last step of the script
        # exported from it, if the script holds every step the run took.
        source = inputs.script.source
        return (isinstance(identity.get("trial"), str) and identity["trial"] == source.get("trial")
                and step == len(inputs.script.steps) and not source.get("dropped_steps"))
    return identity.get("script_sha256") == inputs.sha256 and identity.get("step") == step


def restore_checkpoint(backend, settings, inputs):
    """Put the saved files of step k back into the task container; their summary. Raises unless exact."""
    from taste.brains.docker_checkpoint import CheckpointManifest

    directory = Path(settings.branch_checkpoint)
    identity = json.loads((directory / IDENTITY).read_text(encoding="utf-8"))
    if not names_branch(identity, inputs, settings.branch_step):
        raise ValueError("this checkpoint is of another replay script, step or trial")
    manifest = CheckpointManifest.from_dict(json.loads((directory / MANIFEST).read_text(encoding="utf-8")))
    receipt = backend.restore(manifest, directory, any_container=True)
    summary = receipt.summary()
    if not receipt.exact:
        raise RuntimeError(f"the restored files differ from the checkpoint at {len(receipt.mismatches)} paths")
    return summary


def save_checkpoint(backend, settings, inputs, replay, trial):
    """Save the container's files, which are those of step k after a faithful rebuild with live off."""
    if not (isinstance(replay, dict) and replay.get("faithful") is True):
        return {"saved": False, "reason": "the branch was not faithful"}
    return _save(backend, Path(settings.branch_checkpoint),
                 {**_identity(inputs, settings.branch_step), "trial": trial, "replay": replay})


def save_final_checkpoint(backend, directory, trial):
    """Save the container's files as the trial left them, before its verifier runs."""
    return _save(backend, Path(directory), {"schema": IDENTITY_SCHEMA, "trial": trial, "final": True})


def _save(backend, directory, identity):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if (directory / MANIFEST).exists() or (directory / IDENTITY).exists():
        return {"saved": False, "reason": "the directory already holds a checkpoint"}
    manifest = backend.checkpoint(directory)
    _write(directory / IDENTITY, {**identity, "image": manifest.image})
    # The manifest last: a directory with one holds a whole checkpoint.
    _write(directory / MANIFEST, manifest.to_dict())
    summary = manifest.summary()
    return {"saved": True, "directory": str(directory), **{key: summary[key] for key in (
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
