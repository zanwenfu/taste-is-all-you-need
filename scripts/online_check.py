"""List what passing trials fetched from the internet, for review.

    python scripts/online_check.py --arm alone=/root/tb/jobs/x-alone --arm taste=/root/tb/jobs/x-taste \\
        [--trials /var/lib/taste-trials] [--json out.json]

Terminal-Bench tasks run with internet access, and a trial that found the
task's published solution should not count as solved. For every trial its
verifier passed, every command the agent ran is read from Taste's settled
trial record, and each fetch from the internet is listed: a URL, or curl,
wget, git clone and the like run as a command. Installing from a package
source, and requests to the container's own or a private address, are not
fetches here. Each fetch is classed:

- named_by_task: its URL appears in the task's own text (the task asked for it);
- solution_like: it names the benchmark, its known repositories, a solution,
  or the task itself;
- other: anything else, for a reader to judge.

A passing trial with no settled record is reported as unchecked.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

ROLES_NOT_THE_AGENT = {"taste-coordinator", "taste-monitor"}
_URL = re.compile(r"https?://[^\s'\"<>)\]}`\\]+")
# A shell tool counts where it runs as a command, not where a report mentions it.
_FETCH = re.compile(r"(?:^|[\n;&|(`]|\$\()\s*(?:sudo\s+|env\s+\S+\s+)*(?:curl|wget|aria2c|lynx|w3m|"
                    r"git\s+(?:clone|fetch|pull|ls-remote)|svn\s+(?:checkout|co)|hg\s+clone|gh\s+(?:repo|api|release))\b"
                    r"|\b(?:urlopen|urllib\.request|requests\.(?:get|post)|httpx\.(?:get|post|Client)|aiohttp)\b",
                    re.I)
_LOCAL = re.compile(r"^(localhost|127\.\d+\.\d+\.\d+|0\.0\.0\.0|::1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|"
                    r"172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|host\.docker\.internal|[\w.-]+\.local)$", re.I)
# Package indexes and the distributions' archives: installing is not looking for an answer.
_INDEX_HOSTS = ("pypi.org", "files.pythonhosted.org", "pypi.python.org", "download.pytorch.org",
                "archive.ubuntu.com", "security.ubuntu.com", "ports.ubuntu.com", "deb.debian.org",
                "security.debian.org", "dl-cdn.alpinelinux.org", "cloud.r-project.org", "cran.r-project.org",
                "registry.npmjs.org", "crates.io", "static.crates.io", "proxy.golang.org", "rubygems.org",
                "repo.anaconda.com", "conda.anaconda.org", "astral.sh")
_SOLUTION_MARKERS = ("terminal-bench", "terminal_bench", "terminalbench", "tbench", "t-bench", "laude-institute",
                     "harbor-framework", "solution.sh", "solve.sh", "/solution", "oracle")


def _not_a_lookup(url):
    """A package source, or the container's own or a private address: nothing looked up online."""
    host = (urlparse(url).hostname or "").lower()
    return bool(_LOCAL.match(host)) or any(host == item or host.endswith("." + item) for item in _INDEX_HOSTS)


def fetches(command, task_text, task_name):
    """The fetches in one command, each with its class."""
    urls = [url.rstrip(".,;:") for url in _URL.findall(command)]
    looked_up = [url for url in urls if not _not_a_lookup(url)]
    if urls and not looked_up:
        return []  # package sources or local addresses only
    if not looked_up and not _FETCH.search(command):
        return []
    lowered = command.lower()
    if looked_up and all(url in task_text for url in looked_up):
        kind = "named_by_task"
    elif any(marker in lowered for marker in _SOLUTION_MARKERS) or (task_name and task_name.lower() in lowered
                                                                     and (looked_up or _FETCH.search(command))):
        kind = "solution_like"
    else:
        kind = "other"
    return [{"kind": kind, "urls": looked_up, "command": command[:500]}]


def check_trial(result, trials_root):
    taste = ((result.get("agent_result") or {}).get("metadata") or {}).get("taste") or {}
    task = str(result.get("task_name", "")).rsplit("/", 1)[-1]
    row = {"task": task, "trial": result.get("trial_name"), "checked": False, "fetches": []}
    path = Path(trials_root) / str(taste.get("trial")) / "controller" / "trajectory.json"
    if not taste.get("trial") or not path.is_file():
        return row
    record = json.loads(path.read_text())
    steps = record.get("steps") or []
    task_text = next((str(step.get("message", "")) for step in steps if step.get("source") == "user"), "")
    for sub in record.get("subagent_trajectories", ()):
        if sub["agent"]["name"] in ROLES_NOT_THE_AGENT:
            continue
        for step in sub.get("steps", ()):
            for call in step.get("tool_calls") or ():
                command = (call.get("arguments") or {}).get("command")
                if isinstance(command, str):
                    row["fetches"].extend(fetches(command, task_text, task))
    row["checked"] = True
    return row


def check(arms, trials_root):
    result = {}
    for name, job in arms:
        rows = []
        for path in sorted(Path(job).glob("*/result.json")):
            data = json.loads(path.read_text())
            reward = ((data.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            if reward == 1:
                rows.append(check_trial(data, trials_root))
        result[name] = {"passing": len(rows), "unchecked": sum(not row["checked"] for row in rows),
                        "solution_like": sum(any(f["kind"] == "solution_like" for f in row["fetches"]) for row in rows),
                        "other": sum(any(f["kind"] == "other" for f in row["fetches"]) for row in rows),
                        "trials": rows}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", action="append", required=True, help="name=harbor job directory")
    parser.add_argument("--trials", default="/var/lib/taste-trials", help="Taste's trial records")
    parser.add_argument("--json", type=Path, help="also write the full result here")
    arguments = parser.parse_args(argv)
    result = check([tuple(item.split("=", 1)) for item in arguments.arm], arguments.trials)
    if arguments.json:
        arguments.json.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for name, arm in result.items():
        print(f"{name}: {arm['passing']} passing trials, {arm['unchecked']} unchecked, "
              f"{arm['solution_like']} with a solution-like fetch, {arm['other']} with other fetches")
        for row in arm["trials"]:
            for item in row["fetches"]:
                if item["kind"] != "named_by_task":
                    print(f"  {item['kind']:13} {row['trial']}: {item['command'][:160]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
