"""Check synthetic worker traces with pinned Harbor and Errata's real ATIF reader.

Run in the isolated Harbor environment, after worker tests have generated cases.
Only the public, stdlib ATIF reader is loaded from Errata; no dataset, judge,
credentials, gold answers or repository installation is required.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from harbor.models.trajectories import Trajectory

parser = argparse.ArgumentParser()
parser.add_argument("--cases", type=Path, required=True)
parser.add_argument("--errata-atif", type=Path, required=True)
parser.add_argument("--expected-errata-sha256", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
digest = hashlib.sha256(args.errata_atif.read_bytes()).hexdigest()
assert digest == args.expected_errata_sha256, "Errata ATIF reader changed"
spec = importlib.util.spec_from_file_location("pinned_errata_atif", args.errata_atif)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
checked = []
for path in sorted(args.cases.rglob("trajectory.case.json")):
    case = json.loads(path.read_text())
    trajectory = Trajectory.model_validate(case["trajectory"])
    reply, calls = module.record_of(trajectory.to_json_dict())
    assert reply == case["expected_reply"], path
    assert [{"args": c["args"], "result": c["result"]} for c in calls] == case["expected_calls"], path
    assert all(c["name"].startswith("subagent:") for c in calls), path
    checked.append({"case": str(path.relative_to(args.cases)), "calls": len(calls), "reply": reply})
assert len(checked) == 4, f"expected all four compatibility cases, found {len(checked)}"
args.output.write_text(json.dumps({"status": "passed", "paid_calls": 0,
    "errata_atif_sha256": digest, "cases": checked}, indent=2) + "\n")
print(f"Validated {len(checked)} worker traces through Harbor and Errata; no fabricated final replies.")
