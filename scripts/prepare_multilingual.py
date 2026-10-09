"""SWE-bench Multilingual, as Harbor publishes it, prepared for the recovery study.

    python scripts/prepare_multilingual.py --dataset <download>/swebench_multilingual --out <dir> \\
        [--tasks wave.txt] [--cpus java=2] [--cpus OWNER/REPO=N] [--verifier-public OWNER/REPO=REASON] \\
        [--drop TASK=REASON]

The tasks (Harbor's registry, `swebench_multilingual@1.0`) are changed only
where the study's rules require it, and every change is recorded:

- The agent works offline, as in SWE-Bench Pro V2: as published, no phase sets
  a network rule, so the agent would have the internet.
- Grading needs no network either. `tests/test.sh` runs `uv run parser.py`,
  whose inline metadata installs swebench 4.1.0, datasets 2.16.1 and a
  Python >= 3.11 when the task is graded. The prepared Dockerfile builds that
  environment into the image (a script that declares the same metadata, run
  once at the same path), and the grader runs `uv run --offline parser.py`.
  A task whose own tests need the network keeps a public grader, by name and
  with the reason (`--verifier-public`, by repository or task); its agent
  stays offline.
- One CPU per task, as for SWE-Bench Pro V2, so that 24 trials fit the
  measurement VM; a language or a task that cannot be graded at one CPU gets
  more (`--cpus`, by language, repository or task).
- The image is built once and pinned: `docker_image` names a local tag that
  follows the prepared build files. Harbor then neither builds the image for
  each trial nor removes it after, and every trial of a task runs the same
  image, which restoring a checkpoint requires (a checkpoint is restored only
  into a container of its own image).

Nothing else changes: the instruction, the tests, the reference solution, the
time limits and the memory are as published. A task whose layout differs from
the published one is refused, and nothing is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste.benchmarks.task_split import dataset_digest

SCHEMA = "taste.recovery/PreparedTasks/1"
DATASET = "swebench_multilingual@1.0"
LANGUAGES = ("c", "cpp", "go", "java", "javascript", "php", "ruby", "rust")
IMAGE_PREFIX = "taste-swebml"
VERSION = "p1"  # of these preparations: part of every image tag
PRIMARY = re.compile(r'^    uv run parser\.py \| tee -a "\$LOG_FILE"$', re.M)
PARSER_HEADER = re.compile(r"^cat > parser\.py <<EOF\n(# /// script\n(?:#.*\n)*?# ///)\n", re.M)
WARM = """
# Taste, recovery study: the grader's Python environment, built into the image so grading needs no
# network. The script declares what tests/test.sh's parser.py declares, at the same path, so
# `uv run --offline parser.py` reuses this environment.
RUN printf '%s\\n' {lines} 'import swebench.harness.grading' > /parser.py \\
    && uv run /parser.py && rm /parser.py
"""


def _insert_after(text, header, line):
    if text.count(header + "\n") != 1:
        raise ValueError(f"task.toml has not one {header} section")
    return text.replace(header + "\n", f"{header}\n{line}\n", 1)


def _read(task):
    """What preparing ``task`` needs from it, checked against the published layout."""
    toml_text = (task / "task.toml").read_text()
    config = tomllib.loads(toml_text)
    for phase in ("agent", "verifier", "environment"):
        if "network_mode" in config.get(phase, {}) or "allow_internet" in config.get(phase, {}):
            raise ValueError(f"{task.name}: its {phase} already sets a network rule")
    if "docker_image" in config.get("environment", {}):
        raise ValueError(f"{task.name}: it already names an image")
    language = (config.get("metadata", {}).get("tags") or [None])[-1]
    if language not in LANGUAGES:
        raise ValueError(f"{task.name}: its last tag is not a language ({language!r})")
    if len(re.findall(r"^cpus = \d+$", toml_text, re.M)) != 1:
        raise ValueError(f"{task.name}: task.toml sets cpus not once")
    script = (task / "tests" / "test.sh").read_text()
    if len(PRIMARY.findall(script)) != 1:
        raise ValueError(f"{task.name}: test.sh does not run `uv run parser.py` once at the top level")
    header = PARSER_HEADER.search(script)
    if header is None or "'" in header.group(1):
        raise ValueError(f"{task.name}: test.sh's parser.py has no inline metadata to build ahead")
    dockerfile = (task / "environment" / "Dockerfile").read_text()
    bases = re.findall(r"^FROM (\S+)", dockerfile, re.M)
    if len(bases) != 1 or sorted(path.name for path in (task / "environment").iterdir()) != ["Dockerfile"]:
        raise ValueError(f"{task.name}: its environment is not one Dockerfile from one image")
    repo = json.loads((task / "tests" / "config.json").read_text())["repo"]
    return {"toml": toml_text, "script": script, "header": header.group(1), "dockerfile": dockerfile,
            "language": language, "repo": repo, "base_image": bases[0]}


def _prepared(name, found, cpus, verifier_network):
    lines = " ".join(f"'{line}'" for line in found["header"].splitlines())
    dockerfile = found["dockerfile"].rstrip("\n") + "\n" + WARM.format(lines=lines)
    image = f"{IMAGE_PREFIX}/{name.lower()}:{VERSION}-{hashlib.sha256(dockerfile.encode()).hexdigest()[:12]}"
    toml_text = found["toml"]
    toml_text = _insert_after(toml_text, "[verifier]", f'network_mode = "{verifier_network}"')
    toml_text = _insert_after(toml_text, "[agent]", 'network_mode = "no-network"')
    toml_text = _insert_after(toml_text, "[environment]", f'docker_image = "{image}"')
    toml_text = re.sub(r"^cpus = \d+$", f"cpus = {cpus}", toml_text, count=1, flags=re.M)
    toml_text = ("# Prepared for Taste's recovery study by scripts/prepare_multilingual.py: the agent offline,"
                 f" the grader {verifier_network}, {cpus} CPU, the image built once and pinned.\n" + toml_text)
    config = tomllib.loads(toml_text)
    assert config["environment"]["cpus"] == cpus and config["agent"]["network_mode"] == "no-network"
    script = PRIMARY.sub('    uv run --offline parser.py | tee -a "$LOG_FILE"', found["script"], count=1)
    return {"toml": toml_text, "dockerfile": dockerfile, "script": script, "image": image}


def prepare(dataset, out, names=None, *, cpus=None, verifier_public=None, drop=None):
    """Write the prepared tasks ``names`` (all when None) of ``dataset`` to ``out``; the record.

    ``cpus`` maps a language, a repository ("owner/name") or a task to its CPUs, and
    ``verifier_public`` a repository or a task to the reason its grader keeps the network: the
    most specific wins (task, then repository, then language; 1 CPU and no network otherwise).
    ``drop`` maps a task to the reason it is left out.
    """
    dataset, out = Path(dataset), Path(out)
    cpus, verifier_public, drop = dict(cpus or {}), dict(verifier_public or {}), dict(drop or {})
    available = sorted(path.name for path in dataset.iterdir() if (path / "task.toml").is_file())
    repos = {json.loads((dataset / name / "tests" / "config.json").read_text())["repo"] for name in available}
    names = available if names is None else list(dict.fromkeys(names))
    unknown = sorted((set(names) | set(drop)) - set(available))
    unknown += sorted(key for key in verifier_public if key not in available and key not in repos)
    unknown += sorted(key for key in cpus if key not in LANGUAGES and key not in available and key not in repos)
    if unknown:
        raise ValueError("not tasks, repositories or languages of the dataset: " + ", ".join(unknown))
    kept = [name for name in names if name not in drop]
    record = {"schema": SCHEMA, "dataset": DATASET, "dataset_digest": dataset_digest(dataset),
              "rule": {"agent_network": "no-network", "verifier_network": "no-network",
                       "verifier_public": {name: verifier_public[name] for name in sorted(verifier_public)},
                       "cpus": {"default": 1, **cpus},
                       "grader": "swebench's parser environment built into the image; `uv run --offline`",
                       "image": f"{IMAGE_PREFIX}/<task>:{VERSION}-<sha256 of the prepared Dockerfile>"},
              "tasks": {}, "excluded": {name: drop[name] for name in names if name in drop}}
    written = {}
    for name in kept:
        found = _read(dataset / name)
        count = cpus.get(name, cpus.get(found["repo"], cpus.get(found["language"], 1)))
        network = "public" if name in verifier_public or found["repo"] in verifier_public else "no-network"
        written[name] = _prepared(name, found, count, network)
        record["tasks"][name] = {"language": found["language"], "repo": found["repo"],
                                 "image": written[name]["image"], "cpus": count, "verifier_network": network,
                                 "base_image": found["base_image"]}
    clashes = [name for name in kept if (out / name).exists()]
    if clashes:
        raise ValueError("already prepared in the output: " + ", ".join(clashes))
    out.mkdir(parents=True, exist_ok=True)
    for name, files in written.items():
        shutil.copytree(dataset / name, out / name)
        (out / name / "task.toml").write_text(files["toml"])
        (out / name / "environment" / "Dockerfile").write_text(files["dockerfile"])
        (out / name / "tests" / "test.sh").write_text(files["script"])
    (out / "prepared.json").write_text(json.dumps(record, indent=1) + "\n")
    (out / "images.txt").write_text("".join(f"{name} {entry['base_image']} {entry['image']}\n"
                                            for name, entry in record["tasks"].items()))
    return record


def _pairs(items, kind=str):
    found = {}
    for item in items:
        key, _, value = item.partition("=")
        if not key or not value:
            raise SystemExit(f"expected NAME=VALUE, got {item!r}")
        found[key] = kind(value)
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="the downloaded swebench_multilingual directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, help="a file naming the tasks to prepare, one a line (default: all)")
    parser.add_argument("--cpus", action="append", default=[], metavar="LANGUAGE_REPO_OR_TASK=N")
    parser.add_argument("--verifier-public", action="append", default=[], metavar="REPO_OR_TASK=REASON",
                        help="let the grader (never the agent) reach the network, for the reason given")
    parser.add_argument("--drop", action="append", default=[], metavar="TASK=REASON")
    arguments = parser.parse_args(argv)
    names = None
    if arguments.tasks:
        names = [line.strip() for line in arguments.tasks.read_text().splitlines() if line.strip()]
    record = prepare(arguments.dataset, arguments.out, names, cpus=_pairs(arguments.cpus, int),
                     verifier_public=_pairs(arguments.verifier_public), drop=_pairs(arguments.drop))
    print(f"{len(record['tasks'])} tasks prepared, {len(record['excluded'])} left out; "
          f"images to build: {arguments.out / 'images.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
