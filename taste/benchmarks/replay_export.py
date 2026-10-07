"""A replay script (taste.brains.branch_replay) from a finished trial's record of its hosted agent.

Read, never written: the worker's model journal (every event its run
recorded, in order, and every model request and reply in full), the trial's
terminal ledger (the exact text each command ran and its whole output) and,
for a trial that was itself a branch, the replay script it branched, which
its owner left in the trial's directory. No model is asked and nothing runs.

A step is one model reply the agent received and the commands it ran before
its next call. A reply the run's end cut off, which the agent never received,
and a call given up as lost, which it asked again, are not steps. The script
ends before the first step with a command that has no whole result (one the
run's end cut off); ``source.dropped_steps`` counts what was left out.

A command's output is the ledger's (standard output, then standard error, as
the agent was given them), exact when its length is the one recorded in
memory. Without a ledger it is memory's copy, exact only when nothing was cut
from it, and the script has no exact command texts: a branch from it can
restore a checkpoint but not rebuild.

A branch trial's own record is exported as its steps were seen: steps 1..k
from the script it branched, with step k's output as changed there, then the
steps it took live. A branch that was unfaithful is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from taste.brains.branch_replay import (
    REJECT_EXIT,
    SCRIPT_SCHEMA,
    ReplayScript,
    digest,
    message_shas,
    reply_from_completion,
    reply_to_dict,
    request_sha,
    text_sha,
    with_note,
)
from taste.brains.responses_audit import ROOT, event_id

BRANCH_SCRIPT = "branch-script.json"
_COMPLETE = ("", "timeout")


def _connect(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def _text(data):
    return (data or b"").decode("utf-8", errors="replace")


def read_journal(path):
    """The run's events, in order, and its model calls by request ID."""
    with closing(_connect(path)) as db:
        rows = db.execute("SELECT id,parent,event FROM conversation_events ORDER BY rowid").fetchall()
        calls = {identifier: (request, status, result) for identifier, request, status, result in
                 db.execute("SELECT id,request,status,result FROM calls")}
    events, parent = [], ROOT
    for identifier, recorded, raw in rows:
        event = json.loads(raw)
        if recorded != parent or event_id(parent, event) != identifier:
            raise ValueError("the run's record is not one unbroken line of events")
        events.append(event)
        parent = identifier
    if not events or events[0].get("kind") != "hosted_binding":
        raise ValueError("this journal is not a hosted agent's run")
    return events, calls


def read_ledger(path):
    """Each command the task container was asked to run, by request ID."""
    if path is None:
        return {}
    with closing(_connect(path)) as db:
        return {identifier: {"request": json.loads(payload), "status": status, "code": code,
                             "output": _text(stdout) + _text(stderr), "terminated": terminated}
                for identifier, payload, status, code, stdout, stderr, terminated in db.execute(
                    "SELECT id,payload,status,code,stdout,stderr,terminated FROM requests")}


def _recorded_run(command, output, ledger):
    """One command as the agent saw its result, from the ledger where it can be."""
    text, exact = output["output"], output["output_chars"] == len(output["output"])
    row, executed = ledger.get(command["effect_id"]), None
    if row is not None and row["status"] == "completed" and row["request"]["cwd"] == command["cwd"]:
        executed = row["request"]["command"]
        if row["code"] == output["returncode"] and len(row["output"]) == output["output_chars"]:
            text, exact = row["output"], True
    return {"command": command["command"], "executed": executed, "cwd": command["cwd"],
            "timeout_seconds": command["timeout_seconds"], "output": text, "returncode": output["returncode"],
            "timed_out": output["terminated"] == "timeout", "output_exact": exact}


def _parent_script(parent, branch):
    if parent is None:
        raise ValueError("this run is a branch: its parent replay script is needed (" + BRANCH_SCRIPT + ")")
    raw = parent if isinstance(parent, bytes) else Path(parent).read_bytes()
    if hashlib.sha256(raw).hexdigest() != branch["script_sha256"]:
        raise ValueError("the parent replay script is not the one this branch was given")
    return ReplayScript.from_bytes(raw)


def _seen(run, override, note):
    """A replayed command's result as the branch showed it, beside what the container printed."""
    if not override:
        return run
    printed = run.get("printed") or {key: run[key] for key in ("output", "returncode", "timed_out")}
    if override == "append":
        return {**run, "output": with_note(run["output"], note), "printed": printed}
    return {**run, "output": note, "returncode": REJECT_EXIT, "timed_out": False, "printed": printed}


def export_script(journal, *, ledger=None, parent=None, trial=None):
    """The replay script of one hosted agent's run, as a JSON-ready dict."""
    events, calls = read_journal(journal)
    commands = read_ledger(ledger)
    binding, task, branch, parent_script, ended = None, None, None, None, None
    steps, requests, open_runs, broken = [], {}, {}, set()
    previous, effort, tools = [], None, None
    for event in events:
        kind = event["kind"]
        if kind == "hosted_binding":
            binding = event["binding"]
        elif kind == "hosted_task":
            task = event["content"]
        elif kind == "hosted_branch":
            branch, parent_script = event, _parent_script(parent, event)
        elif kind == "hosted_unfaithful":
            raise ValueError(f"this run was a branch that did not match its record (step {event['step']}, "
                             f"{event['reason']}): it is not a run of the agent")
        elif kind == "hosted_replay":
            source = parent_script.steps[event["step"] - 1]
            if event["step"] != len(steps) + 1 or source.request_sha != event["request_sha"]:
                raise ValueError("the branch's replayed steps differ from its parent script")
            value = source.to_dict()
            steps.append({**value, "step": len(steps) + 1, "runs": [], "replayed": True,
                          "_left": [run.to_dict() for run in source.runs]})
            previous = parent_script.message_shas(event["step"])
        elif kind == "hosted_replay_output":
            current = steps[-1]
            seen = _seen(current["_left"].pop(0), event["override"], branch.get("note", ""))
            if text_sha(seen["output"]) != event["output_sha256"]:
                raise ValueError("a replayed output differs from what the branch showed")
            current["runs"].append(seen)
        elif kind == "hosted_request":
            requests[event["id"]] = event["request_sha"]
        elif kind == "hosted_lost":
            requests.pop(event["id"], None)
        elif kind == "hosted_completion":
            if event.get("cut_off"):
                continue  # paid for, never received: not a step
            raw_request, status, result = calls.get(event["id"], (None, None, None))
            if status != "completed" or event["id"] not in requests:
                raise ValueError("a recorded reply has no completed call in the journal")
            request = json.loads(raw_request)
            if request_sha(request["messages"], request["tools"], request["effort"]) != requests.pop(event["id"]):
                raise ValueError("a journaled request differs from the digest the run recorded")
            effort, tools = request["effort"], request["tools"]
            shas = message_shas(request["messages"])
            base = next((index for index, (left, right) in enumerate(zip(previous, shas, strict=False))
                         if left != right), min(len(previous), len(shas)))
            previous = shas
            steps.append({"step": len(steps) + 1, "request_sha": digest({"messages": request["messages"],
                                                                         "tools": tools, "effort": effort}),
                          "messages": {"base": base, "added": shas[base:]},
                          "reply": reply_to_dict(reply_from_completion(json.loads(result), event["cost_usd"])),
                          "text": list(event["text"]), "calls": list(event["calls"]), "runs": [], "replayed": False})
        elif kind == "hosted_command":
            if not steps:
                raise ValueError("a command was recorded before any model reply")
            open_runs[event["effect_id"]] = (len(steps) - 1, event)
        elif kind == "hosted_output":
            index, command = open_runs.pop(event["effect_id"], (None, None))
            if command is None:
                raise ValueError("a command's output has no recorded command")
            if event["terminated"] not in _COMPLETE:
                broken.add(index)
                continue
            steps[index]["runs"].append(_recorded_run(command, event, commands))
        elif kind == "hosted_exit":
            ended = {key: event[key] for key in ("exit_status", "stopped_by", "submission")}
    broken.update(index for index, _ in open_runs.values())
    broken.update(index for index, item in enumerate(steps) if item.pop("_left", None))
    kept = min(broken, default=len(steps))
    source = {"trial": trial, "run_id": binding["run_id"], "agent": binding["agent"], "model": binding["model"],
              "workdir": binding["workdir"], "dropped_steps": len(steps) - kept,
              "effort": effort if effort is not None else (parent_script.source.get("effort") if parent_script else None),
              "tools_sha256": digest(tools) if tools is not None else (
                  parent_script.source.get("tools_sha256") if parent_script else None),
              "parent": None if branch is None else {
                  "script_sha256": branch["script_sha256"], "step": branch["step"],
                  "override": branch.get("override", ""), "trial": parent_script.source.get("trial"),
                  "run_id": parent_script.source.get("run_id")}}
    script = ReplayScript.from_dict({"schema": SCRIPT_SCHEMA, "source": source, "task": task,
                                     "steps": steps[:kept], "exit": ended})
    return script.to_dict()


def trial_files(path, *, trials_root="/var/lib/taste-trials", run_id=None):
    """(journal, ledger, parent script, trial) for a trial's directory, Harbor's trial directory or a journal."""
    path = Path(path)
    if path.is_file():
        return path, None, None, None
    for candidate in (path / "calls.sqlite3", path / "worker" / "calls.sqlite3"):
        if candidate.is_file():
            return candidate, None, None, None
    if (path / "result.json").is_file():
        result = json.loads((path / "result.json").read_text())
        token = (((result.get("agent_result") or {}).get("metadata") or {}).get("taste") or {}).get("trial")
        if not token:
            raise ValueError("this Harbor trial names no Taste trial")
        path = Path(trials_root) / str(token)
    journals = sorted(path.glob("agent-state/workspace/.git/taste.azure.*/worker/calls.sqlite3"))
    if run_id is not None:
        suffix = "." + hashlib.sha256(run_id.encode()).hexdigest()
        journals = [item for item in journals if item.parent.parent.name.endswith(suffix)]
    if len(journals) != 1:
        raise ValueError(f"{path} holds {len(journals)} worker runs; name one with its run ID")
    ledger = path / "controller" / "terminal" / "terminal.sqlite3"
    parent = path / BRANCH_SCRIPT
    return (journals[0], ledger if ledger.is_file() else None, parent if parent.is_file() else None, path.name)


def encode_script(script):
    """The bytes a replay script is written as; a branch is admitted against their SHA-256."""
    return (json.dumps(script, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def main(argv=None):
    """python3 -m taste.benchmarks.replay_export RECORD -o SCRIPT (scripts/replay_script.py says more)."""
    parser = argparse.ArgumentParser(description="Write the replay script of a finished trial's hosted agent")
    parser.add_argument("record", type=Path, nargs="?",
                        help="a trial's directory, Harbor's trial directory or a journal (or --record)")
    parser.add_argument("--record", type=Path, dest="named_record", help=argparse.SUPPRESS)
    parser.add_argument("-o", "--output", "--out", type=Path, required=True, help="where to write the script")
    parser.add_argument("--trials", default="/var/lib/taste-trials", help="Taste's trial directories")
    parser.add_argument("--run", help="the worker run's ID, when the trial holds more than one")
    parser.add_argument("--ledger", type=Path, help="the trial's terminal ledger, with a journal")
    parser.add_argument("--parent", type=Path, help="the script a branch trial branched, with a journal")
    arguments = parser.parse_args(argv)
    record = arguments.record or arguments.named_record
    if record is None or (arguments.record and arguments.named_record):
        parser.error("name one record")
    journal, ledger, parent, trial = trial_files(record, trials_root=arguments.trials, run_id=arguments.run)
    script = export_script(journal, ledger=arguments.ledger or ledger, parent=arguments.parent or parent,
                           trial=trial)
    raw = encode_script(script)
    arguments.output.write_bytes(raw)
    runs = [run for step in script["steps"] for run in step["runs"]]
    print(json.dumps({"steps": len(script["steps"]), "submission_step": script["submission_step"],
                      "dropped_steps": script["source"]["dropped_steps"], "commands": len(runs),
                      "outputs_exact": sum(run["output_exact"] for run in runs),
                      "rebuildable": all(run["executed"] is not None for run in runs),
                      "exit": script["exit"], "trial": trial, "sha256": hashlib.sha256(raw).hexdigest(),
                      "bytes": len(raw), "output": str(arguments.output)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
