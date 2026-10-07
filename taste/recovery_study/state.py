"""A driver's state: one JSON file, written whole and atomically, one driver at a time.

The state records the study's settings, the runs and every job the driver has
started with the results it has read back. Harbor's job directories stay the
evidence: a job's results are read once it has finished and kept here, so a
driver can be stopped at any point and run again.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
from datetime import UTC, datetime
from pathlib import Path


def now():
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(path, kind):
    """The state at ``path``, or None if there is none yet."""
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("kind") != kind:
        raise ValueError(f"{path} holds {data.get('kind')!r}, not {kind!r}")
    return data


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated"] = now()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(temporary, path)


@contextlib.contextmanager
def locked(path):
    """Hold the state's lock: two drivers on one state would start the same jobs twice."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another driver is running on {path}") from None
        yield


def settings(stored, given, defaults):
    """The study's settings: fixed when it starts; a later run may restate them, not change them."""
    given = {name: raw for name, raw in given.items() if raw is not None}
    unknown = set(given) - set(defaults)
    if unknown:
        raise ValueError("unknown settings: " + ", ".join(sorted(unknown)))
    if stored is None:
        return {**defaults, **given}
    changed = [name for name, raw in given.items() if stored.get(name, defaults[name]) != raw]
    if changed:
        raise ValueError("the study began with other settings for: " + ", ".join(sorted(changed))
                         + " (start a new state file to change them)")
    return {**defaults, **stored}
