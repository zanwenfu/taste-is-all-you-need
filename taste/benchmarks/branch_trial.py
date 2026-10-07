"""The trial owner's part of a branch: the replay script into the trial, and checkpoints of a run's files.

The worker serves a branch (``taste.brains.branch_replay``). Before it starts,
the trial's owner reads the replay script and note named in the settings,
decides whether the agent's context is replayed, checks the branch against
the script, and gives the worker a read-only copy of the script in the trial's
directory, held to its digest. In restore mode the owner puts a saved
checkpoint back into the fresh task container before the agent starts, and
refuses the trial if the files do not then match it.

The context is replayed when the trial's agent is the one the record is of
(``branch_replay=auto``, the default; ``on`` or ``off`` to say so). A replayed
agent is given the task its record was given, so a suffix an earlier trial
added is carried over and never added twice. An agent given only the files,
a checker say, starts fresh with the trial's own task text.

Checkpoints are saved two ways, each only into a directory that holds none:

- A rebuild with live off ends with the container's files as they were after
  step k, and the verifier grades those. When asked (``branch_checkpoint``),
  the owner first saves them, which later branches from step k restore
  instead of rebuilding. Only a faithful branch's files are saved.
- Any trial can leave its final files (``final_checkpoint``), saved when it
  ends and before its verifier runs: what a checker is given a copy of. They
  are restored for a branch from the last step of the replay script exported
  from that same trial, when the script left out none of the run's steps.

A checkpoint is named by a directory path or by an ID, a directory of that
name under the checkpoints root. It holds the manifest
(``taste.brains.docker_checkpoint``), the tar it names, and ``branch.json``,
which says what the files are: the replay script's digest and the step, or
the trial whose final files they are. A checkpoint is never restored for
another branch than the one it names.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from taste.brains.branch_replay import ReplayScript, check_branch, text_sha

BRANCH_SCRIPT = "branch-script.json"
MANIFEST = "manifest.json"
IDENTITY = "branch.json"
IDENTITY_SCHEMA = "taste.branch/Checkpoint/1"
CHECKPOINTS_ROOT = "/var/lib/taste-checkpoints"
# What a trial's record lists of a checkpoint's changed paths.
CHANGED_PATHS = 1000
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


@dataclass(frozen=True)
class BranchInputs:
    """The replay script as read, the note shown at step k, and whether the context is replayed."""

    raw: bytes
    script: ReplayScript
    note: str
    replay: bool = True

    @property
    def sha256(self):
        return hashlib.sha256(self.raw).hexdigest()

    @property
    def recorded_agent(self):
        return (self.script.source.get("agent") or {}).get("name")


def load_inputs(settings):
    """The branch the settings name, checked against its script; None for a fresh trial."""
    if not settings.branch:
        return None
    raw = Path(settings.branch).read_bytes()
    script = ReplayScript.from_bytes(raw)
    note = Path(settings.branch_note).read_text(encoding="utf-8") if settings.branch_note else ""
    if settings.branch_note and not note.strip():
        raise ValueError("the branch note is empty")
    recorded = (script.source.get("agent") or {}).get("name")
    same = recorded == settings.agent
    replay = {"on": True, "off": False}.get(settings.branch_replay, same)
    if replay and not same:
        raise ValueError(f"the record is of {recorded or 'an unnamed agent'}, not {settings.agent}: its "
                         "context cannot be replayed into another agent (branch_replay=off gives it the files)")
    if replay and (settings.task_text or settings.task_suffix):
        raise ValueError("a replayed agent is given its record's task: task_text and task_suffix are for "
                         "fresh trials and for an agent given only the files")
    if not replay and settings.branch_live == "off":
        raise ValueError("an agent given only the files would be stopped before it starts: branch_live=off "
                         "is for a replayed agent")
    check_branch(script, settings.branch_step, mode=settings.branch_mode, override=settings.branch_override,
                 note=note, replay=replay)
    return BranchInputs(raw, script, note, replay)


def branch_instruction(inputs, instruction):
    """The task text a branch trial gives its agent.

    A replayed agent gets its record's task: the benchmark's instruction, with
    any suffix an earlier trial added after it. Anything else is not this
    task's record and is refused. An agent given only the files gets the
    trial's own instruction.
    """
    if inputs is None or not inputs.replay:
        return instruction
    recorded = inputs.script.task
    if recorded != instruction and not recorded.startswith(instruction.rstrip("\n") + "\n\n"):
        raise ValueError("the replay script's task is not this task's instruction, with or without a suffix")
    return recorded


def disclosure(settings, inputs):
    """What a branch trial's record says of its branch before it runs."""
    source = inputs.script.source
    return {"script_sha256": inputs.sha256, "step": settings.branch_step, "steps": len(inputs.script.steps),
            "mode": settings.branch_mode, "live": settings.branch_live,
            "replay": "on" if inputs.replay else "off", "override": settings.branch_override,
            "tolerance": settings.branch_tolerance,
            **({"note_sha256": text_sha(inputs.note)} if inputs.note else {}),
            "source": {key: source.get(key) for key in ("trial", "run_id", "model")}}


def checkpoint_directory(value, root=CHECKPOINTS_ROOT):
    """A checkpoint's directory: the path given, or an ID's directory under the checkpoints root."""
    value = str(value)
    if "/" in value:
        return Path(value)
    if _ID.fullmatch(value) is None:
        raise ValueError("a checkpoint is a directory path or an ID of letters, digits, '.', '_' and '-'")
    return Path(root) / value


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


def changed_paths(manifest):
    """What a checkpoint holds as changed: the paths it copied, then the paths deleted, and how many more."""
    paths = [*manifest.copied, *(path + " (deleted)" for path in manifest.deleted)]
    return {"changed_paths": paths[:CHANGED_PATHS], "changed_paths_more": max(0, len(paths) - CHANGED_PATHS)}


def restore_checkpoint(backend, directory, inputs, step):
    """Put the saved files of step k back into the task container; their account. Raises unless exact."""
    from taste.brains.docker_checkpoint import CheckpointManifest

    directory = Path(directory)
    identity = json.loads((directory / IDENTITY).read_text(encoding="utf-8"))
    if not names_branch(identity, inputs, step):
        raise ValueError("this checkpoint is of another replay script, step or trial")
    manifest = CheckpointManifest.from_dict(json.loads((directory / MANIFEST).read_text(encoding="utf-8")))
    receipt = backend.restore(manifest, directory, any_container=True)
    if not receipt.exact:
        raise RuntimeError(f"the restored files differ from the checkpoint at {len(receipt.mismatches)} paths")
    return {**receipt.summary(), "directory": str(directory), **changed_paths(manifest)}


def save_checkpoint(backend, directory, inputs, step, replay, trial):
    """Save the container's files, which are those of step k after a faithful rebuild with live off."""
    if not (isinstance(replay, dict) and replay.get("faithful") is True):
        return {"saved": False, "reason": "the branch was not faithful"}
    return _save(backend, Path(directory), {**_identity(inputs, step), "trial": trial, "replay": replay})


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
        "tar_sha256", "bytes", "added", "changed", "deleted", "partial", "left_out")}, **changed_paths(manifest)}


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


def outcome(replay, *, restore_seconds=0.0):
    """What a driver reads first of a branch trial: faithful, where it stopped being so, and its prefix's time.

    ``prefix_seconds`` is the time spent bringing back step k (a restore here,
    a rebuild in the worker), which a wall-clock budget may leave out.
    """
    unfaithful = (replay or {}).get("unfaithful") or None
    rebuild = (replay or {}).get("rebuild") or {}
    return {"faithful": bool(replay and replay.get("faithful") is True),
            "unfaithful_at": None if unfaithful is None else unfaithful.get("step"),
            "unfaithful_reason": None if unfaithful is None else unfaithful.get("reason"),
            "prefix_seconds": round(float(restore_seconds) + float(rebuild.get("seconds") or 0.0), 3)}
