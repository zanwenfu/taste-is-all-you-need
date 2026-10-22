"""Take back a rewind arm's ungraded rounds back to the start, and reopen the episodes they ended.

    sudo python3 scripts/reopen_rewind_to_start.py --state /data/study/h2-rec.json [--code /opt/taste/<sha>] [--apply]

Reads each rewind episode that ended ``agent_failed`` on a grade-only round from the run's first call that the
harness stopped before grading (``taste.recovery_study.rewind_to_start``), and says which jobs it would void and which
episodes it would reopen; with --apply it does so and records it on the state (its ``amendments``). --code also points
the state's later jobs at that installed code, which must grade such a round. Stop the state's loop first: this takes
the state's lock, and the driver starts the reopened rounds on its next run. Running it again does nothing more.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taste.recovery_study import host_order, rewind_to_start, state
from taste.recovery_study.recovery_driver import RecoveryDriver


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", required=True, type=Path, help="the recovery state file (JSON)")
    parser.add_argument("--code", type=Path, help="installed code the state's later jobs are to run")
    parser.add_argument("--apply", action="store_true", help="void, reopen and save; without it, only report")
    arguments = parser.parse_args(argv)
    with state.locked(arguments.state):
        driver = RecoveryDriver.open(arguments.state)
        done = rewind_to_start.reopen(driver)
        changes = host_order.move_code(driver, arguments.code) if arguments.code else {}
        report = {"state": str(arguments.state), "applied": arguments.apply, "voided": len(done["voided"]),
                  "reopened": len(done["reopened"]), "settings": changes}
        if arguments.apply and (done["voided"] or changes):
            host_order.amend(driver, {"by": "reopen_rewind_to_start", "reason": rewind_to_start.REASON,
                                      "voided": done["voided"], "reopened": done["reopened"], "settings": changes})
            driver.save()
    print(json.dumps(report, indent=1, sort_keys=True))
    for name in done["voided"]:
        print("void", name)
    for run_id, arm, index, number in done["reopened"]:
        print("reopen", run_id, arm, index, number)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
