#!/usr/bin/env python3
"""Write the replay script of a finished trial's hosted agent, for branching it at any step.

    sudo python3 scripts/replay_script.py /var/lib/taste-trials/<token> -o script.json
    sudo python3 scripts/replay_script.py <harbor job>/<trial> -o script.json [--trials /var/lib/taste-trials]
    python3 scripts/replay_script.py <run dir>/worker/calls.sqlite3 --ledger terminal.sqlite3 -o script.json

The same as ``python3 -m taste.benchmarks.replay_export``. The script holds
the run's steps in order: each model request's digest and the reply as the
provider gave it; each command as the agent wrote it, the exact text the
container ran, its directory, time limit, output, exit code and time-out; and
the task text the agent was given. A trial's directory gives the worker's
journal, the terminal ledger and, for a branch trial, the script it branched
(so a branch of a branch carries the replayed steps and the note shown at
their step k); Harbor's trial directory leads to it. Root reads them (they are
private to the trial's owner and its service account); nothing is changed.
A branch trial is then started with

    harbor run ... --ak agent=mini-swe-agent --ak services=none \\
        --ak branch=script.json --ak branch_step=<k> [--ak branch_mode=rebuild|restore] ...

and a checker of the run's final files with

    harbor run ... --ak agent=checker --ak services=none --ak task_text=<checker task> \\
        --ak branch=script.json --ak branch_step=<the last step> ...

(taste.benchmarks.harbor_settings). Steps are numbered as the trajectory reader
numbers them (taste.agents.trajectory_reader.steps_from_trajectory). The
summary printed says how many steps there are, which one submitted, whether
the outputs and command texts are exact (a rebuild needs the command texts),
and the script's SHA-256.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taste.benchmarks.replay_export import main

if __name__ == "__main__":
    raise SystemExit(main())
