"""Ask the trajectory reader where each run the checker rejected went wrong.

    sudo sh -c 'set -a; . /root/taste-secrets/azure.env; set +a; \\
        PYTHONPATH=/opt/taste/<sha> /home/bugbash/taste-openai-20260923.venv/bin/python \\
        scripts/read_runs.py --state /root/study/recoveries.json --out /root/study/readings.json \\
        [--trials /var/lib/taste-trials] [--model gpt-6-sol] [--effort medium] [--budget-usd 5]'

For every run the recovery driver's checker rejected (its first check's reply
is ``not_done``), the reader (``taste.agents.trajectory_reader``) is given the
task, the run's steps, the agent's final message and the checker's findings,
and names the first step where the agent went wrong. The readings go to --out,
the file the driver's --reader reads: ``{"<run>": {"step", "reason",
"confidence", "model", "effort", "usd"}}``, each saved as soon as it is made.
Runs already there are not read again, so this can be run after each round of
the driver's checks; a run that could not be read is printed with why, and is
read next time.

Azure credentials come from AZURE_OPENAI_BASE_URL and AZURE_OPENAI_API_KEY.
Each reading is one paid call, or two when the first reply is not a reading.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from taste.agents import trajectory_reader
from taste.benchmarks.harbor_settings import SERVED_MODELS
from taste.recovery_study import records

MAX_OUTPUT_TOKENS = 4096


def rejected(state):
    """(run, its base trial's token, the checker's reply) for every run the checker rejected."""
    for run_id, run in sorted((state.get("runs") or {}).items()):
        reply = (run.get("check") or {}).get("reply") or {}
        if run.get("status") == "rejected" and reply.get("verdict") == "not_done":
            yield run_id, run["base"]["token"], reply


def read_runs(state, trials_root, ask, known=(), failures=(trajectory_reader.InvalidReading,)):
    """For each rejected run not in ``known``: (run, its reading, None) or (run, None, why it was not read).

    A run whose reading ends in one of ``failures`` is reported and the next is read.
    """
    for run_id, token, reply in rejected(state):
        if run_id in known:
            continue
        view = records.agent_view(records.settled(token, trials_root))
        if view is None or not view["steps"]:
            yield run_id, None, "no settled record of the agent's steps"
            continue
        try:
            yield run_id, trajectory_reader.read_trajectory(view["task"], view["steps"], view["final_message"],
                                                            reply, ask=ask), None
        except failures as failure:
            yield run_id, None, f"{type(failure).__name__}: {failure}"


def azure_llm(budget_usd, model):
    from taste.llm import LLM
    from taste.providers.azure_openai import AzureDeployment, AzureOpenAIConfig

    azure = AzureOpenAIConfig.from_environment(os.environ, deployments=(AzureDeployment(SERVED_MODELS[model], model),))
    return LLM(azure_openai=azure, budget_usd=budget_usd, cap_on="billed", max_attempts=1, load_env_file=False,
               run_id="trajectory-reader")


def _save(path, readings):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(readings, indent=1, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv=None, llm=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", required=True, type=Path, help="the recovery driver's state file")
    parser.add_argument("--out", required=True, type=Path, help="the readings file (read, extended, rewritten)")
    parser.add_argument("--trials", default="/var/lib/taste-trials", help="Taste's settled trial records")
    parser.add_argument("--model", default="gpt-6-sol", choices=sorted(SERVED_MODELS))
    parser.add_argument("--effort", default="medium", choices=("low", "medium", "high"))
    parser.add_argument("--budget-usd", type=float, default=5.0, help="what all of this run's calls may spend")
    arguments = parser.parse_args(argv)
    state = json.loads(arguments.state.read_text())
    found = json.loads(arguments.out.read_text()) if arguments.out.exists() else {}
    llm = llm or azure_llm(arguments.budget_usd, arguments.model)
    served = SERVED_MODELS[arguments.model]

    def ask(messages):
        completion = llm.call(model=served, system=messages[0]["content"],
                              messages=[{"role": item["role"], "content": item["content"]} for item in messages[1:]],
                              max_tokens=MAX_OUTPUT_TOKENS, effort=arguments.effort, temperature=None, role="reader")
        return completion.summary_text

    from taste.llm import InfraFailure
    from taste.providers.base import ProtocolFailure

    # A call that failed costs that run its reading this time; the budget running out stops all.
    failures = (trajectory_reader.InvalidReading, InfraFailure, ProtocolFailure)
    spent = llm.spent_usd()
    for run_id, reading, why in read_runs(state, arguments.trials, ask, known=set(found), failures=failures):
        now = llm.spent_usd()
        if reading is None:
            print(f"{run_id}: not read: {why}")
        else:
            found[run_id] = {**reading, "model": arguments.model, "effort": arguments.effort,
                             "usd": round(now - spent, 6)}
            _save(arguments.out, found)
            print(f"{run_id}: step {reading['step']} ({reading['confidence']:.2f}): {reading['reason']}")
        spent = now
    print(f"{len(found)} readings in {arguments.out}; this run spent ${llm.spent_usd():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
