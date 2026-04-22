"""Bootstrap a throwaway workspace from the parallel_demo template.

    python examples/parallel_demo/bootstrap.py /tmp/taste-parallel
    taste run "add type hints to every module" \\
        --agent examples/parallel_demo/agent_desp.md \\
        --workspace /tmp/taste-parallel
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
        ["git", "-c", "user.name=taste-demo", "-c", "user.email=demo@taste.local",
         "commit", "-q", "-m", "initial: three independent utility modules + tests"],
        cwd=dest,
        check=True,
    )
    return dest


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: bootstrap.py <dest>")
    print(f"bootstrapped: {bootstrap(Path(sys.argv[1]))}")
