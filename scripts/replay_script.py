#!/usr/bin/env python3
"""Write the replay script of a finished trial's hosted agent, for branching it at any step.

    sudo python3 scripts/replay_script.py /var/lib/taste-trials/<token> -o script.json
    sudo python3 scripts/replay_script.py <harbor job>/<trial> -o script.json [--trials /var/lib/taste-trials]
    python3 scripts/replay_script.py <run dir>/worker/calls.sqlite3 --ledger terminal.sqlite3 -o script.json

The script holds the run's steps in order: each model request's digest and the
reply as the provider gave it; each command as the agent wrote it, the exact
text the container ran, its directory, time limit, output, exit code and
time-out; and the task text the agent was given. A trial's directory gives the
worker's journal, the terminal ledger and, for a branch trial, the script it
branched; Harbor's trial directory leads to it. Root reads them (they are
private to the trial's owner and its service account); nothing is changed.
A branch trial is then started with

    harbor run ... --ak agent=mini-swe-agent --ak services=none \\
        --ak branch=script.json --ak branch_step=<k> [--ak branch_mode=rebuild|restore] ...

and a checker of the run's final files with

    harbor run ... --ak agent=checker --ak services=none --ak task_text=<checker task> \\
        --ak branch=script.json --ak branch_step=<the last step> --ak branch_replay=off ...

(taste.benchmarks.harbor_settings). Steps are numbered as the trajectory reader
numbers them (taste.agents.trajectory_reader.steps_from_trajectory). The
summary printed says how many steps there are, which one submitted, and
whether the outputs and command texts are exact (a rebuild needs the command
texts).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taste.benchmarks.replay_export import encode_script, export_script, trial_files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("record", type=Path, help="a trial's directory, Harbor's trial directory or a journal")
    parser.add_argument("-o", "--output", type=Path, required=True, help="where to write the script")
    parser.add_argument("--trials", default="/var/lib/taste-trials", help="Taste's trial directories")
    parser.add_argument("--run", help="the worker run's ID, when the trial holds more than one")
    parser.add_argument("--ledger", type=Path, help="the trial's terminal ledger, with a journal")
    parser.add_argument("--parent", type=Path, help="the script a branch trial branched, with a journal")
    arguments = parser.parse_args(argv)
    journal, ledger, parent, trial = trial_files(arguments.record, trials_root=arguments.trials,
                                                 run_id=arguments.run)
    script = export_script(journal, ledger=arguments.ledger or ledger, parent=arguments.parent or parent,
                           trial=trial)
    raw = encode_script(script)
    arguments.output.write_bytes(raw)
    runs = [run for step in script["steps"] for run in step["runs"]]
    print(json.dumps({"steps": len(script["steps"]), "submission_step": script["submission_step"],
                      "dropped_steps": script["source"]["dropped_steps"], "commands": len(runs),
                      "outputs_exact": sum(run["output_exact"] for run in runs),
                      "rebuildable": all(run["executed"] is not None for run in runs),
                      "exit": script["exit"], "bytes": len(raw), "output": str(arguments.output)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
