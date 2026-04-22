"""Create a throwaway git workspace from the demo's template files.

Use this before ``taste run`` so the agent operates on an isolated tmp repo
instead of polluting the Agent OS repo itself. Example::

    python examples/refactor_demo/bootstrap.py /tmp/refactor-demo
    taste run "refactor legacy_math.py" \\
        --agent examples/refactor_demo/agent_desp.md \\
        --workspace /tmp/refactor-demo
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

TEMPLATE = Path(__file__).parent / "template"


def bootstrap(dest: Path) -> Path:
    dest = Path(dest).resolve()
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(TEMPLATE, dest)

    subprocess.run(["git", "init", "-q"], cwd=dest, check=True)
    subprocess.run(["git", "add", "."], cwd=dest, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=taste-demo",
            "-c",
            "user.email=demo@taste.local",
            "commit",
            "-q",
            "-m",
            "initial: legacy_math.py + tests",
        ],
        cwd=dest,
        check=True,
    )
    return dest


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: bootstrap.py <dest>")
    path = bootstrap(Path(sys.argv[1]))
    print(f"bootstrapped fresh workspace: {path}")
