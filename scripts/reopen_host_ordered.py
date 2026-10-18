"""Take back the trials a rebuild stopped at a cut-off directory walk, and reopen the episodes they ended.

    sudo python3 scripts/reopen_host_ordered.py --state /data/study/p1pro-rec.json [--code /opt/taste/<sha>] [--apply]

Reads each finished job of the state whose trials all stopped at a cut-off directory walk
(``taste.recovery_study.host_order``), and says which it would void and which episodes it would reopen; with
--apply it does so and records it on the state (its ``amendments``). --code also points the state's later jobs at
that installed code. Stop the state's loop first: this takes the state's lock, and the driver starts the reopened
episodes on its next run. Running it again does nothing more.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste.recovery_study import host_order, state
from taste.recovery_study.recovery_driver import RecoveryDriver


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", required=True, type=Path, help="the recovery state file (JSON)")
    parser.add_argument("--code", type=Path, help="installed code the state's later jobs are to run")
    parser.add_argument("--apply", action="store_true", help="void, reopen and save; without it, only report")
    arguments = parser.parse_args(argv)
    with state.locked(arguments.state):
        driver = RecoveryDriver.open(arguments.state)
        done = host_order.reopen(driver)
        changes = host_order.move_code(driver, arguments.code) if arguments.code else {}
        purposes = collections.Counter(driver.jobs[name].get("purpose") for name in done["voided"])
        where = collections.Counter(item[3] for item in done["reopened"])
        report = {"state": str(arguments.state), "applied": arguments.apply, "voided": dict(purposes),
                  "reopened": dict(where), "left": done["left"], "settings": changes}
        if arguments.apply and (done["voided"] or changes):
            host_order.amend(driver, {"by": "reopen_host_ordered", "reason": host_order.REASON,
                                      "voided": done["voided"], "reopened": done["reopened"], "settings": changes})
            driver.save()
    print(json.dumps(report, indent=1, sort_keys=True))
    for name in done["voided"]:
        print("void", name)
    for run_id, arm, index, part in done["reopened"]:
        print("reopen", run_id, arm, index, part)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
