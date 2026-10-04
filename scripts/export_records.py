"""Export a study's trial records for anyone to check: no credentials, a manifest with checksums.

    python scripts/export_records.py --job /root/tb/jobs/s-alone --job /root/tb/jobs/s-taste \\
        --trials /var/lib/taste-trials --out /root/tb/export-s [--forbid-env AZURE_OPENAI_API_KEY]
    python scripts/export_records.py --verify /root/tb/export-s

Copied for each trial: Harbor's result.json and its verifier's reward and
output, and Taste's settled record of the trial (controller/trajectory.json
and controller/outcome.json); for each job, Harbor's config.json. Nothing
else: no journals, logs or credentials. Each file's SHA-256 and size are
listed in manifest.json, which --verify checks. With --forbid-env NAME (any
number of times), the export is refused if a copied file contains the value
of that environment variable; the value is never printed. The export keeps
the layout the report scripts read:

    python scripts/study_report.py --arm a=EXPORT/jobs/s-alone ... --trials EXPORT/trials
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

SCHEMA = "taste.export/1"
TRIAL_FILES = ("result.json", "verifier/reward.txt", "verifier/reward.json", "verifier/test-stdout.txt")
RECORD_FILES = ("controller/trajectory.json", "controller/outcome.json")


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source, target, forbidden):
    data = Path(source).read_bytes()
    for name, value in forbidden:
        if value.encode() in data:
            raise ValueError(f"{source} contains the value of {name}; nothing was exported")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def export(jobs, trials_root, out, forbid_env=()):
    """Copy the records of every trial of the given jobs into ``out``; return the manifest."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty")
    forbidden = [(name, os.environ[name]) for name in forbid_env if os.environ.get(name)]
    staging = out.with_name(out.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    copied = []
    try:
        for job in map(Path, jobs):
            if (job / "config.json").is_file():
                _copy(job / "config.json", staging / "jobs" / job.name / "config.json", forbidden)
                copied.append(f"jobs/{job.name}/config.json")
            for result_path in sorted(job.glob("*/result.json")):
                trial = result_path.parent
                for name in TRIAL_FILES:
                    if (trial / name).is_file():
                        relative = f"jobs/{job.name}/{trial.name}/{name}"
                        _copy(trial / name, staging / relative, forbidden)
                        copied.append(relative)
                result = json.loads(result_path.read_text())
                token = (((result.get("agent_result") or {}).get("metadata") or {}).get("taste") or {}).get("trial")
                for name in RECORD_FILES if token else ():
                    source = Path(trials_root) / str(token) / name
                    if source.is_file():
                        relative = f"trials/{token}/{name}"
                        _copy(source, staging / relative, forbidden)
                        copied.append(relative)
        manifest = {"schema": SCHEMA, "jobs": [Path(job).name for job in jobs],
                    "files": {relative: {"sha256": _sha(staging / relative),
                                         "bytes": (staging / relative).stat().st_size}
                              for relative in sorted(copied)}}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        if out.exists():
            out.rmdir()  # empty, checked above
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def verify(out):
    """Every file the manifest lists is present and unchanged, and nothing else is there."""
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    problems = []
    for relative, expected in manifest["files"].items():
        path = out / relative
        if not path.is_file():
            problems.append(f"missing: {relative}")
        elif _sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            problems.append(f"changed: {relative}")
    present = {str(path.relative_to(out)) for path in out.rglob("*") if path.is_file()} - {"manifest.json"}
    problems += [f"not in the manifest: {relative}" for relative in sorted(present - set(manifest["files"]))]
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job", action="append", default=[], help="a Harbor job directory")
    parser.add_argument("--trials", default="/var/lib/taste-trials", help="Taste's trial records")
    parser.add_argument("--out", type=Path, help="the export directory to create")
    parser.add_argument("--forbid-env", action="append", default=[], metavar="NAME",
                        help="refuse if a copied file contains this environment variable's value")
    parser.add_argument("--verify", type=Path, help="check an export against its manifest")
    arguments = parser.parse_args(argv)
    if arguments.verify:
        problems = verify(arguments.verify)
        print("\n".join(problems) if problems else "export verified")
        return 1 if problems else 0
    if not arguments.job or not arguments.out:
        parser.error("an export needs --job and --out")
    manifest = export(arguments.job, arguments.trials, arguments.out, arguments.forbid_env)
    print(f"exported {len(manifest['files'])} files from {len(manifest['jobs'])} jobs to {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
