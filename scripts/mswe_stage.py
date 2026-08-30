"""Stage every archived MSWE workspace so a re-grade sees the files the agent created.

    python scripts/mswe_stage.py /root/mswe40_sonnet_s1 /root/mswe40_sonnet_s2 ...

The graded patch is ``git diff <root-commit>`` over the host tree, which
ignores files that were never staged. The harness arms stage through their
checkpoints; the first MSWE sweeps did not (the driver does now), so their
new source files were invisible to the grader. Staging is idempotent and
touches nothing the shadow timeline reads (it has its own index). Run
``scripts/rescore.py --root <root>`` afterwards for the $0 re-measurement.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    staged = skipped = 0
    for spec in sys.argv[1:]:
        root = Path(spec)
        for row_path in sorted((root / "ledger").glob("*.json")):
            try:
                row = json.loads(row_path.read_text())
            except json.JSONDecodeError:
                continue
            ws = Path(row.get("workspace") or "")
            if not (isinstance(row, dict) and row.get("task") and ws.is_dir()):
                skipped += 1
                continue
            subprocess.run(["git", "add", "--all", "--", ".", ":(exclude)patch.txt"], cwd=ws, check=False)
            new = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=A"], cwd=ws, capture_output=True, text=True).stdout.split()
            new = [n for n in new if not n.startswith(".taste/")]
            print(f"  {row['task']:32s} staged; new files now in the patch: {len(new)} {new[:4]}")
            staged += 1
    print(f"{staged} workspace(s) staged, {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
