#!/usr/bin/env python3
"""Can a critic judge a run from its git record alone, or does it need the live environment?

Question 5 of docs/studies/system-layer.md. Both critics get the task, the same
record of the run from git memory, and the same model. The live critic can also
run commands in a fresh container holding the run's final files. Neither sees
the verifier, whose result on the run is the truth they are scored against.

    sudo python3 scripts/critic_study.py inputs --job <job> --trials <root> --tasks <dir> --graded <graded.json> --out <dir>
    python3 scripts/critic_study.py record --in <dir>                 # no machine; needs the model route
    sudo python3 scripts/critic_study.py live --in <dir> --code <dir> --tasks <dir> --jobs <dir> -n 20
    sudo python3 scripts/critic_study.py read-live --in <dir> --jobs <dir>
    python3 scripts/critic_study.py report --in <dir>
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

RECORD_CHARS = 60_000
SETTINGS = {"agent": "checker-v2", "services": "none", "worker_max_calls": 35, "spend_cap_usd": 2.5,
            "agent_timeout_sec": 1500, "worker_effort": "medium"}
RECORD_SYSTEM = """\
You review a coding agent's work. The agent was given a task, worked on it in a container and submitted. You decide \
whether its work does what the task asks.

You cannot reach the container. All you have is the record kept in the run's git memory: the plans, what each worker \
was asked to do, the planner's reasons and assessments where the record holds them, and the workers' reports. The \
agent's and the planner's statements are claims, not proof: weigh what they show.

Judge the task as written: do not add requirements it does not state, and do not reject work for its style. Where the \
wording allows more than one reasonable reading, work that meets one of them meets it.

Reply with one JSON object and nothing else: {"verdict": "done" or "not_done", "confidence": a number from 0 to 1, \
"reason": one or two sentences}."""


def _git(repository, *args):
    result = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(repository), *args],
                            capture_output=True, text=True, timeout=120, check=False)
    return result.stdout if result.returncode == 0 else ""


def plans_of(repository):
    """Every plan on the control branch, by generation; [] when the branch cannot be read."""
    refs = [line for line in _git(repository, "for-each-ref", "--format=%(refname)", "refs/heads/mem").split()
            if line.endswith("/central-control")]
    if len(refs) != 1:
        return []
    plans = []
    for path in _git(repository, "ls-tree", "-r", "--name-only", refs[0], ".taste/planner/plans/").split():
        try:
            plans.append(json.loads(_git(repository, "show", f"{refs[0]}:{path}")))
        except ValueError:
            continue
    return sorted(plans, key=lambda plan: plan.get("generation", 0))


def last_report(repository):
    refs = [line for line in _git(repository, "for-each-ref", "--format=%(refname)", "refs/heads/mem").split()
            if line.endswith("/integration")]
    return _git(repository, "show", f"{refs[0]}:report.md") if len(refs) == 1 else ""


def record_text(plans, report, limit=RECORD_CHARS, conclusions=True):
    """The run as git memory holds it, oldest plan first; cut from the front, so the latest work stays.

    Without ``conclusions`` the planner's reasons and assessments are left out:
    what was asked and what was reported remain.
    """
    parts = []
    for plan in plans:
        lines = [f"Plan {plan.get('generation')}: {plan.get('rationale', '')}" if conclusions
                 else f"Plan {plan.get('generation')}"]
        for item in plan.get("assignments") or []:
            lines.append(f"  Assigned {item.get('assignment_id')}: {((item.get('contract') or {}).get('task') or '')}")
        for item in (plan.get("assessment") or []) if conclusions else []:
            lines.append(f"  Assessed {item.get('verdict') or item.get('status')}: {item.get('evidence', '')}")
        parts.append("\n".join(lines))
    parts.append("Latest report:\n" + (report or "(none)"))
    text = "\n\n".join(parts)
    return text if len(text) <= limit else "(earlier record cut)\n" + text[-limit:]


def sample(graded):
    """Runs with a usable final save whose control agreed: their files show what the verifier saw."""
    return [row for row in graded if row["kind"] == "final" and row["state_reward"] is not None
            and row["reward"] is not None and row["state_reward"] == row["reward"]]


def inputs(arguments):
    from taste.agents.checker import checker_task

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    tokens = {}
    for path in Path(arguments.job).glob("*/result.json"):
        result = json.loads(path.read_text())
        if "trial_name" in result:
            tokens[result["trial_name"]] = result["agent_result"]["metadata"]["taste"]["trial"]
    rows = []
    for row in sample(json.loads(Path(arguments.graded).read_text())):
        repository = Path(arguments.trials) / tokens[row["trial"]] / "agent-state" / "workspace"
        plans = plans_of(repository)
        report = last_report(repository)
        instruction = (Path(arguments.tasks) / row["task"] / "instruction.md").read_text(encoding="utf-8")
        final = plans[-1].get("rationale", "") if plans and not arguments.facts_only else ""
        record = record_text(plans, report, conclusions=not arguments.facts_only)
        task = checker_task(instruction, final, record)
        name = re.sub(r"[^A-Za-z0-9_-]", "-", row["trial"])
        (out / f"{name}.task.txt").write_text(task, encoding="utf-8")
        (out / f"{name}.plans.json").write_text(json.dumps([
            {key: plan.get(key) for key in ("generation", "complete", "rollback_to", "rationale", "assessment")}
            | {"tasks": [(item.get("contract") or {}).get("task") for item in plan.get("assignments") or []]}
            for plan in plans]), encoding="utf-8")
        rows.append({"trial": row["trial"], "name": name, "task": row["task"], "reward": row["reward"],
                     "state": row["state"], "plans": len(plans), "record_chars": len(record)})
    (out / "sample.json").write_text(json.dumps(rows, indent=0), encoding="utf-8")
    print(json.dumps({"runs": len(rows), "passed": sum(row["reward"] == 1.0 for row in rows)}))


def parse_verdict(text):
    """(verdict, confidence) from a reply, or (None, None) when it is not one."""
    try:
        value = json.loads(text)
    except ValueError:
        return None, None
    if not isinstance(value, dict) or value.get("verdict") not in ("done", "not_done"):
        return None, None
    return value["verdict"], value.get("confidence")


def record_critic(arguments):
    from taste.llm import LLM
    from taste.providers.azure_openai import AZURE_LUNA6_MODEL, AzureDeployment, AzureOpenAIConfig

    folder = Path(arguments.input)
    route = AzureOpenAIConfig(base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
                              api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
                              deployments=(AzureDeployment(AZURE_LUNA6_MODEL, "global.openai.gpt-6-luna"),))
    rows = json.loads((folder / "sample.json").read_text())
    if arguments.limit:
        rows = rows[:arguments.limit]
    done = folder / "record-verdicts.jsonl"
    finished = {json.loads(line)["trial"] for line in done.read_text().splitlines()} if done.exists() else set()

    def review(row):
        llm = LLM(azure_openai=route, max_attempts=1)
        reply = llm.call(model=AZURE_LUNA6_MODEL, system=RECORD_SYSTEM,
                         messages=[{"role": "user", "content": (folder / f"{row['name']}.task.txt").read_text()}],
                         tools=None, max_tokens=4096, effort="medium", temperature=None, timeout_seconds=300,
                         json_output=True)
        verdict, confidence = parse_verdict("".join(reply.text_blocks))
        return {"trial": row["trial"], "verdict": verdict, "confidence": confidence,
                "input": reply.usage.input_tokens, "cached": reply.usage.cache_read_tokens,
                "output": reply.usage.output_tokens}

    with concurrent.futures.ThreadPoolExecutor(arguments.n) as pool, done.open("a") as written:
        for result in pool.map(review, [row for row in rows if row["trial"] not in finished]):
            written.write(json.dumps(result) + "\n")
            written.flush()
    print("written", done)


def live_job(row):
    return "critic-" + row["name"]


def live(arguments):
    folder = Path(arguments.input)
    rows = json.loads((folder / "sample.json").read_text())
    if arguments.limit:
        rows = rows[:arguments.limit]
    environment = {**os.environ, "MODEL": "azure/gpt-6-luna", "JOBS": str(arguments.jobs),
                   "SECRETS": "/root/taste-secrets/bedrock.env", "TASTE_TRIALS_ROOT": "/data/taste-trials-critic",
                   "TASTE_CHECKPOINTS_ROOT": "/data/taste-checkpoints"}

    def one(row):
        job = live_job(row)
        if (Path(arguments.jobs) / job).exists():
            return job, "exists"
        options = [item for key, value in {**SETTINGS, "saved_state": row["state"],
                                           "task_text": str(folder / f"{row['name']}.task.txt")}.items()
                   for item in ("--ak", f"{key}={value}")]
        started = subprocess.run([f"{arguments.code}/infra/azure/run-harbor.sh", job, arguments.tasks, "-i", row["task"],
                                  "-k", "1", "-n", "1", "--max-retries", "1", *options],
                                 env=environment, capture_output=True, text=True, check=False)
        if started.returncode != 0:
            return job, started.stderr[-300:]
        while subprocess.run(["systemctl", "is-active", "--quiet", f"taste-harbor-{job}"], check=False).returncode == 0:
            time.sleep(5)
        return job, "ok"

    with concurrent.futures.ThreadPoolExecutor(arguments.n) as pool:
        for job, status in pool.map(one, rows):
            print(job, status, flush=True)


def read_live(arguments):
    from taste.recovery_study.records import checker_submission

    folder = Path(arguments.input)
    with (folder / "live-verdicts.jsonl").open("w") as written:
        for row in json.loads((folder / "sample.json").read_text()):
            trials = sorted(path for path in (Path(arguments.jobs) / live_job(row)).glob("*/") if path.is_dir())
            found = checker_submission(trials[-1]) if trials else None
            written.write(json.dumps({"trial": row["trial"], "verdict": (found or {}).get("verdict"),
                                      "confidence": (found or {}).get("confidence")}) + "\n")


def summarize(rows, verdicts):
    """Each critic's errors in both directions against the verifier, on the runs it gave a verdict for."""
    truth = {row["trial"]: row["reward"] == 1.0 for row in rows}
    seen = [(truth[item["trial"]], item["verdict"]) for item in verdicts if item["verdict"] in ("done", "not_done")]
    passed = [verdict for good, verdict in seen if good]
    failed = [verdict for good, verdict in seen if not good]
    return {"judged": len(seen), "unjudged": len(rows) - len(seen), "passed_runs": len(passed), "failed_runs": len(failed),
            "false_reject": passed.count("not_done"), "false_accept": failed.count("done"),
            "false_reject_rate": passed.count("not_done") / len(passed) if passed else None,
            "false_accept_rate": failed.count("done") / len(failed) if failed else None,
            "accuracy": (passed.count("done") + failed.count("not_done")) / len(seen) if seen else None}


def report(arguments):
    folder = Path(arguments.input)
    rows = json.loads((folder / "sample.json").read_text())
    out = {}
    for name in ("record", "live"):
        path = folder / f"{name}-verdicts.jsonl"
        if path.exists():
            out[name] = summarize(rows, [json.loads(line) for line in path.read_text().splitlines()])
    print(json.dumps(out, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("inputs")
    for name in ("job", "trials", "tasks", "graded", "out"):
        make.add_argument("--" + name, required=True)
    make.add_argument("--facts-only", action="store_true", help="leave out the planner's reasons and assessments")
    record = commands.add_parser("record")
    record.add_argument("--in", dest="input", required=True)
    record.add_argument("-n", type=int, default=16)
    record.add_argument("--limit", type=int, default=0)
    run = commands.add_parser("live")
    run.add_argument("--in", dest="input", required=True)
    for name in ("code", "tasks", "jobs"):
        run.add_argument("--" + name, required=True)
    run.add_argument("-n", type=int, default=20)
    run.add_argument("--limit", type=int, default=0)
    read = commands.add_parser("read-live")
    read.add_argument("--in", dest="input", required=True)
    read.add_argument("--jobs", required=True)
    show = commands.add_parser("report")
    show.add_argument("--in", dest="input", required=True)
    arguments = parser.parse_args(argv)
    {"inputs": inputs, "record": record_critic, "live": live, "read-live": read_live, "report": report}[
        arguments.command](arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
