"""Create a throwaway git workspace from the todo_api demo template.

    python examples/todo_api/bootstrap.py /tmp/taste-todo
    taste run "<task>" \\
        --agent examples/todo_api/agent_desp.md \\
        --workspace /tmp/taste-todo
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
            "-c", "user.name=taste-demo",
            "-c", "user.email=demo@taste.local",
            "commit", "-q", "-m", "initial: flask todo api + tests",
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
