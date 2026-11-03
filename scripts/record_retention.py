#!/usr/bin/env python3
"""How much of a trial's own record reaches each planner call. No model is called.

Taste's planner keeps no conversation: each call is built from a planning request
that the coordinator writes into git from the record. This counts, call by call,
what the record held and what of it the request carried, from the trial's git
record and its terminal command log alone.

    python3 scripts/record_retention.py trial <trial dir> [--json out.json]
    python3 scripts/record_retention.py many <root holding trial dirs> --out results.jsonl
    python3 scripts/record_retention.py summarize results.jsonl

Sources, all read-only. The git record is <trial>/agent-state/workspace: on the
central-control branch, planning requests (.taste/planner/operations/<op>/request.json)
and their attempts, plans (.taste/planner/plans), criteria revisions
(.taste/planner/criteria), the goal (.taste/planner/goals) and supervisor runs
(.taste/supervisor/runs); worker reports on each worker branch (worker-report.json);
the planner's receipts on planner-receipts. The command log is
<trial>/controller/terminal/terminal.sqlite3, table requests, one row per command.

A call is one planning request, ordered by its created_at. Its record moment is
the request's world.captured_at, the time the coordinator read the record to build
it; an item recorded after that moment is not held at that call.

Items, and when each is held and sent:
  goal        The goal's task and success criteria. Held at every call; sent when
              the request's goal has the same task and criteria.
  criterion   Each standing criterion of the latest criteria revision dated at or
              before the record moment. Sent when the request carries a criterion
              with the same id and text. Criteria are append-only (a refinement
              never retires its parent), so none is ever superseded.
  plan        Each plan created before the record moment. Sent when it is the
              request's parent_plan. The request carries only the previous plan,
              so older plans are left out by design.
  assessment  Each criterion verdict inside a held plan. Sent with its plan.
  report      Each worker report dated before the record moment. Sent when the
              request's world.outcomes holds the same report_id.
  command     Each terminal command of a run whose report is held. Sent when the
              request's copy of that report has the command's evidence line: the
              line the worker runtime writes, "ran: <command clipped to 300
              characters> -> exit <code>", or "started, result not recorded:
              <command>" for one with no result. Lines are matched in order, each
              at most once. A worker's record keeps only its last 40 commands and
              12,000 characters, so earlier commands are left out by design.
              A command of a run that started before the record moment but has
              no report yet is undecided: the log has no time per command.

Repeats. A command repeats a recorded failure when a worker of a new assignment
runs a command whose text, with whitespace collapsed, equals that of a command an
earlier run had already run and failed (exit code not 0, or ended by the
terminal) by the time of the call that issued the new assignment. It is split by
whether that failure's evidence line was in the request of that call. A run whose
issuing call cannot be found is undecided. This is a narrow mechanical rule:
running a failed command again after fixing its cause counts too.

Superseded. A plan is superseded at a call when a newer plan is held; a report
when a report of the same assignment with a higher attempt is held. A record is
stale when the request carried a version of a run record (record_blob_id) that
the control branch had already replaced when the call was made: the version at
the commit that recorded the call's first attempt.

Size. The prompt's bytes are not recorded, only their hash. Recorded exactly:
the request's bytes, which the prompt contains, so a lower bound; the prompt's
tokens as billed; and any attempt refused for its size, with the sizes it names.

Age. An item's age at a call is the number of calls since the first call at
which it was held.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

MAX_PROMPT_BYTES = 192 * 1024  # taste/brains/planner_transport.py, LLMPlannerTransport max_prompt_bytes
TYPES = ("goal", "criterion", "plan", "assessment", "report", "command")
_WRAPPER = re.compile(r"^export PAGER=cat [^;\n]*; \{\n(.*)\n\} 2>&1$", re.DOTALL)
_SIZE = re.compile(r"planner input is (\d+) bytes; limit is (\d+) bytes")


def clip(text, size):
    """As taste/brains/worker_commands.py clips a command for its evidence line."""
    text = " ".join(str(text).split())
    return text if len(text) <= size else text[:size] + " [cut]"


def strip_wrapper(command):
    """The command the worker asked for, without the terminal's environment wrapper."""
    found = _WRAPPER.match(command)
    return found.group(1) if found else command


def shown_in(line, raw, code):
    """Whether an evidence line records this command with this result."""
    if code is None:
        return line == f"started, result not recorded: {clip(raw, 300)}"
    expected = f"ran: {clip(raw, 300)} -> exit {code}"
    return line == expected or line.startswith(expected + ";")


def size_refusal(detail):
    found = _SIZE.search(detail or "")
    return (int(found.group(1)), int(found.group(2))) if found else None


def _moment(text):
    """A sortable form of a recorded time; text that is not a time is compared as it is."""
    try:
        value = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return str(text)
    return value.astimezone(UTC).isoformat() if value.tzinfo else value.isoformat()


def _normal(command):
    return " ".join(command.split())


def _failed(command):
    return command["status"] == "completed" and (command["terminated"] or command["code"] not in (0, None))


def _match(commands, lines):
    """Which commands (by seq) a worker's evidence lines show, aligning in order."""
    shown, used = set(), set()
    for command in commands:
        raw = strip_wrapper(command["command"])
        code = command["code"] if command["status"] == "completed" else None
        for index, line in enumerate(lines):
            if index not in used and shown_in(line, raw, code):
                used.add(index)
                shown.add(command["seq"])
                break
    return shown


def analyze(trial):
    """Count, per planner call, what the record held and what the request carried."""
    requests = sorted(trial["requests"], key=lambda item: _moment(item["created_at"]))
    reports = trial["reports"]
    runs = {run["run_id"]: run for run in trial["runs"]}
    by_run = defaultdict(list)
    for command in sorted(trial["commands"], key=lambda item: item["seq"]):
        by_run["worker-run." + command["actor_id"].split("_", 1)[-1]].append(command)
    totals = {kind: Counter() for kind in TYPES}
    ages, first_held = Counter(), {}
    superseded = Counter()
    calls, criteria_everywhere, shown_at = [], True, {}
    cap = trial.get("cap") or MAX_PROMPT_BYTES
    without_report = 0
    for number, item in enumerate(requests, 1):
        request = item["request"]
        moment = _moment(item.get("captured_at") or item["created_at"])
        held, sent, undecided = Counter(), Counter(), Counter()

        def count(kind, key, is_sent, number=number, held=held, sent=sent):
            first_held.setdefault((kind, key), number)
            held[kind] += 1
            sent[kind] += bool(is_sent)
            ages[(kind, number - first_held[(kind, key)], bool(is_sent))] += 1

        goal = request.get("goal") or {}
        count("goal", "goal", goal.get("task") == trial["goal"].get("task")
              and list(goal.get("success_criteria") or ()) == list(trial["goal"].get("success_criteria") or ()))

        revisions = [rev for rev in trial["criteria"] if _moment(rev["at"]) <= moment]
        standing = max(revisions, key=lambda rev: _moment(rev["at"]))["criteria"] if revisions else []
        carried = {(c["criterion_id"], c["text"]) for c in ((request.get("criteria") or {}).get("criteria") or ())}
        for criterion in standing:
            is_sent = (criterion["criterion_id"], criterion["text"]) in carried
            criteria_everywhere &= is_sent
            count("criterion", criterion["criterion_id"], is_sent)

        parent = (request.get("parent_plan") or {}).get("plan_id")
        plans = [plan for plan in trial["plans"] if _moment(plan["created_at"]) < moment]
        newest = max((plan["generation"] for plan in plans), default=None)
        for plan in plans:
            is_sent = plan["plan_id"] == parent
            count("plan", plan["plan_id"], is_sent)
            for entry in plan.get("assessment") or ():
                count("assessment", (plan["plan_id"], entry["criterion_id"]), is_sent)
            if plan["generation"] != newest:
                superseded["plans_held"] += 1
                superseded["plans_sent"] += is_sent

        listed = (request.get("world") or {}).get("outcomes") or ()
        # A run that ended without a report (killed, or out of budget) is listed with report None.
        without_report += sum(o.get("report") is None for o in listed)
        outcomes = {o["report"]["report_id"]: o for o in listed if o.get("report") is not None}
        held_reports = [report for report in reports if _moment(report["at"]) < moment]
        latest = {}
        for report in held_reports:
            latest[report["assignment_id"]] = max(latest.get(report["assignment_id"], -1), report["attempt"])
        for report in held_reports:
            count("report", report["report_id"], report["report_id"] in outcomes)
            if report["attempt"] < latest[report["assignment_id"]]:
                superseded["reports_held"] += 1
                superseded["reports_sent"] += report["report_id"] in outcomes
        for outcome in outcomes.values():
            blob = (item.get("blob_at_call") or {}).get(outcome.get("record_path"))
            if blob is None:
                superseded["stale_undecided"] += 1
            elif blob != outcome.get("record_blob_id"):
                superseded["stale_record_sent"] += 1

        reported = {report["run_id"]: report for report in held_reports}
        shown_here = set()
        for run_id, commands in by_run.items():
            if run_id in reported:
                outcome = outcomes.get(reported[run_id]["report_id"])
                lines = (outcome["report"].get("metadata") or {}).get("worker_evidence") or [] if outcome else []
                shown = _match(commands, lines)
                shown_here |= shown
                for command in commands:
                    count("command", command["seq"], command["seq"] in shown)
            elif run_id in runs and _moment(runs[run_id]["created_at"]) < moment:
                undecided["command"] += len(commands)
        shown_at[number] = (item, {c["seq"] for r in reported for c in by_run.get(r, ())}, shown_here)

        for kind in TYPES:
            totals[kind]["held"] += held[kind]
            totals[kind]["sent"] += sent[kind]
            totals[kind]["undecided"] += undecided[kind]
        refusals = [found for found in (size_refusal(a.get("detail")) for a in item.get("attempts") or ()) if found]
        calls.append({"call": number, "generation": request.get("generation"), "operation_id": item["operation_id"],
                      "request_bytes": item["bytes"],
                      "share_of_cap_lower_bound": round(item["bytes"] / cap, 4),
                      "attempts": len(item.get("attempts") or ()),
                      "prompt_tokens": [a.get("prompt_tokens") for a in item.get("attempts") or ()],
                      "size_refusals": refusals, "held": dict(held), "sent": dict(sent),
                      "undecided": dict(undecided)})

    repeats = Counter()
    seqs = {c["seq"]: c for c in trial["commands"]}
    for run_id, commands in by_run.items():
        run = runs.get(run_id)
        issuing = [n for n, (item, _, _) in shown_at.items() if run is not None
                   and item["request"].get("generation") == run["generation"]
                   and _moment(item["created_at"]) <= _moment(run["created_at"])]
        if not issuing:
            repeats["undecided_runs"] += 1
            continue
        _, known, shown = shown_at[max(issuing)]
        failures = defaultdict(list)
        for seq in known:
            if _failed(seqs[seq]) and seqs[seq]["actor_id"] != commands[0]["actor_id"]:
                failures[_normal(strip_wrapper(seqs[seq]["command"]))].append(seq)
        hits = [c for c in commands if _normal(strip_wrapper(c["command"])) in failures]
        for command in hits:
            earlier = failures[_normal(strip_wrapper(command["command"]))]
            repeats["commands"] += 1
            repeats["failure_shown" if any(seq in shown for seq in earlier) else "failure_not_shown"] += 1
        repeats["distinct"] += len({_normal(strip_wrapper(c["command"])) for c in hits})
        repeats["assignments"] += bool(hits)

    return {"calls": calls, "cap_bytes": cap, "outcomes_without_report": without_report,
            "by_type": {kind: {"held": totals[kind]["held"], "sent": totals[kind]["sent"],
                               "undecided": totals[kind]["undecided"]} for kind in TYPES},
            "criteria_kept_at_every_call": criteria_everywhere if calls else None,
            "repeats": {key: repeats[key] for key in ("commands", "distinct", "assignments", "failure_shown",
                                                     "failure_not_shown", "undecided_runs")},
            "superseded": {key: superseded[key] for key in ("plans_held", "plans_sent", "reports_held",
                                                           "reports_sent", "stale_record_sent", "stale_undecided")},
            "ages": [{"type": kind, "age": age, "sent": is_sent, "count": n}
                     for (kind, age, is_sent), n in sorted(ages.items(), key=str)]}


class Git:
    """Read-only access to one trial's git record."""

    def __init__(self, repository):
        self.repository = str(repository)
        self.environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}

    def __call__(self, *arguments):
        result = subprocess.run(["git", "-c", "safe.directory=*", "-C", self.repository, *arguments],
                                capture_output=True, text=True, env=self.environment, check=False)
        return result.stdout if result.returncode == 0 else ""

    def tree(self, ref, prefix=""):
        """{path: blob} under ``prefix`` at ``ref``."""
        out = {}
        for line in self("ls-tree", "-r", ref, "--", prefix).splitlines() if prefix else self(
                "ls-tree", "-r", ref).splitlines():
            meta, _, path = line.partition("\t")
            out[path] = meta.split()[2]
        return out

    def blobs(self, ids):
        """{blob: text}, read in one batch."""
        ids = list(dict.fromkeys(ids))
        if not ids:
            return {}
        result = subprocess.run(["git", "-c", "safe.directory=*", "-C", self.repository, "cat-file", "--batch"],
                                input=("\n".join(ids) + "\n").encode(), capture_output=True,
                                env=self.environment, check=False)
        data, out, position = result.stdout, {}, 0
        for blob in ids:
            end = data.index(b"\n", position)
            header = data[position:end].decode().split()
            if len(header) < 3 or header[1] != "blob":
                position = end + 1
                continue
            size = int(header[2])
            out[blob] = data[end + 1:end + 1 + size].decode("utf-8", "replace")
            position = end + 1 + size + 1
        return out


def load_trial(directory):
    """The trial's record as analyze() reads it, or None when it has no coordinator."""
    directory = Path(directory)
    git = Git(directory / "agent-state" / "workspace")
    refs = git("for-each-ref", "--format=%(refname)", "refs/heads").split()
    control = next((ref for ref in refs if ref.endswith("/central-control")), None)
    if control is None:
        return None
    receipts = next((ref for ref in refs if ref.endswith("/planner-receipts")), None)
    tree = git.tree(control)
    texts = git.blobs(tree[path] for path in tree if path.startswith((
        ".taste/planner/operations/", ".taste/planner/plans/", ".taste/planner/criteria/",
        ".taste/planner/goals/")) or (path.startswith(".taste/supervisor/runs/") and path.endswith("/run.json")))

    def read(path):
        return json.loads(texts[tree[path]])

    tokens = {}
    if receipts:
        receipt_tree = git.tree(receipts)
        receipt_texts = git.blobs(receipt_tree.values())
        for path, blob in receipt_tree.items():
            if path.endswith("/intent.json"):
                outcome = receipt_tree.get(path[:-len("intent.json")] + "outcome.json")
                if outcome is None:
                    continue
                attempt = (json.loads(receipt_texts[blob]).get("scope") or {}).get("attempt_id")
                usage = ((json.loads(receipt_texts[outcome]).get("telemetry") or {}).get("usage") or {})
                if usage:
                    tokens[attempt] = sum(int(usage.get(k) or 0) for k in
                                          ("input_tokens", "cache_read_tokens", "cache_write_tokens"))

    requests = []
    for path in tree:
        if not (path.startswith(".taste/planner/operations/") and path.endswith("/request.json")):
            continue
        root = path[:-len("request.json")]
        request = read(path)
        attempts = []
        for number in sorted({p[len(root + "attempts/"):].split("/")[0] for p in tree
                              if p.startswith(root + "attempts/")}):
            intent = read(f"{root}attempts/{number}/intent.json")
            outcome_path = f"{root}attempts/{number}/outcome.json"
            outcome = read(outcome_path) if outcome_path in tree else {}
            attempts.append({"status": outcome.get("status"), "detail": outcome.get("detail", ""),
                             "prompt_tokens": tokens.get(intent.get("attempt_id")), "intent": f"{root}attempts/{number}/intent.json"})
        blob_at_call = {}
        if attempts:
            added = git("log", "--format=%H", "--diff-filter=A", control, "--", attempts[0]["intent"]).split()
            outcomes = (request.get("world") or {}).get("outcomes") or ()
            paths = [o["record_path"] for o in outcomes if o.get("record_path")]
            if added and paths:
                for line in git("ls-tree", added[-1], "--", *paths).splitlines():
                    meta, _, record = line.partition("\t")
                    blob_at_call[record] = meta.split()[2]
        requests.append({"operation_id": request.get("operation_id"), "created_at": request["created_at"],
                         "captured_at": (request.get("world") or {}).get("captured_at"),
                         "bytes": len(texts[tree[path]].encode()), "attempts": attempts,
                         "blob_at_call": blob_at_call, "request": request})

    reports = {}
    for item in requests:
        for outcome in (item["request"].get("world") or {}).get("outcomes") or ():
            if outcome.get("report") is not None:
                reports.setdefault(outcome["report"]["report_id"], outcome["report"])
    for ref in refs:
        found = git("show", f"{ref}:worker-report.json")
        if found:
            report = json.loads(found)
            reports.setdefault(report["report_id"], report)

    runs = []
    for path in tree:
        if path.startswith(".taste/supervisor/runs/") and path.endswith("/run.json"):
            run = read(path)
            assignment = run.get("assignment") or {}
            runs.append({"run_id": run["run_id"], "assignment_id": assignment.get("assignment_id"),
                         "generation": assignment.get("generation"), "created_at": run.get("created_at")})

    commands = []
    ledger = directory / "controller" / "terminal" / "terminal.sqlite3"
    if ledger.exists():
        connection = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
        try:
            for seq, payload, status, code, terminated in connection.execute(
                    "SELECT rowid, payload, status, code, terminated FROM requests ORDER BY rowid"):
                body = json.loads(payload)
                commands.append({"seq": seq, "actor_id": body.get("actor_id", ""), "command": body.get("command", ""),
                                 "status": status, "code": code, "terminated": bool(terminated)})
        finally:
            connection.close()

    goals = [read(path) for path in tree if path.startswith(".taste/planner/goals/")]
    criteria = [read(path) for path in tree if path.startswith(".taste/planner/criteria/")]
    # The planner's prompt cap is the run policy's max_request_bytes when the trial set one
    # (taste/brains/central_host.py passes it to LLMPlannerTransport), else the transport default.
    cap = MAX_PROMPT_BYTES
    for source in (directory / "controller" / "trial.json", directory / "prepare.json"):
        if source.exists():
            policy = json.loads(source.read_text()).get("policy") or {}
            if policy.get("max_request_bytes"):
                cap = int(policy["max_request_bytes"])
                break
    return {"goal": goals[0] if goals else {}, "criteria": criteria, "cap": cap,
            "plans": [read(path) for path in tree if path.startswith(".taste/planner/plans/")],
            "requests": requests, "reports": list(reports.values()), "runs": runs, "commands": commands,
            "model_planner": receipts is not None}


def summarize(rows):
    """Totals over many trials."""
    out = {"trials": len(rows), "calls": 0, "by_type": {k: Counter() for k in TYPES}, "repeats": Counter(),
           "superseded": Counter(), "criteria_kept_at_every_call": 0, "size_refusals": 0,
           "largest_request_bytes": 0, "ages": Counter(), "share_of_cap_lower_bound_max": 0.0,
           "caps_bytes": Counter(), "outcomes_without_report": 0}
    for row in rows:
        result = row["result"]
        out["calls"] += len(result["calls"])
        out["criteria_kept_at_every_call"] += bool(result["criteria_kept_at_every_call"])
        out["caps_bytes"][str(result.get("cap_bytes"))] += 1
        out["outcomes_without_report"] += result.get("outcomes_without_report", 0)
        for kind, numbers in result["by_type"].items():
            out["by_type"][kind].update(numbers)
        out["repeats"].update(result["repeats"])
        out["superseded"].update(result["superseded"])
        for call in result["calls"]:
            out["size_refusals"] += len(call["size_refusals"])
            out["largest_request_bytes"] = max(out["largest_request_bytes"], call["request_bytes"])
            out["share_of_cap_lower_bound_max"] = max(out["share_of_cap_lower_bound_max"],
                                                      call["share_of_cap_lower_bound"])
        for age in result["ages"]:
            out["ages"][f"{age['type']}|{age['age']}|{'sent' if age['sent'] else 'left out'}"] += age["count"]
    return json.loads(json.dumps(out))


def _one(directory):
    trial = load_trial(directory)
    if trial is None:
        return {"trial": Path(directory).name, "skipped": "no coordinator"}
    if not trial["requests"]:
        return {"trial": Path(directory).name, "skipped": "no planner calls"}
    return {"trial": Path(directory).name, "model_planner": trial["model_planner"], "result": analyze(trial)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("trial")
    one.add_argument("directory")
    one.add_argument("--json")
    many = commands.add_parser("many")
    many.add_argument("root")
    many.add_argument("--out", required=True)
    total = commands.add_parser("summarize")
    total.add_argument("results")
    arguments = parser.parse_args(argv)
    if arguments.command == "trial":
        row = _one(arguments.directory)
        if arguments.json:
            Path(arguments.json).write_text(json.dumps(row, indent=1))
        print(json.dumps(row, indent=1))
    elif arguments.command == "many":
        with open(arguments.out, "w") as written:
            for directory in sorted(p for p in Path(arguments.root).iterdir() if p.is_dir()):
                try:
                    row = _one(directory)
                except Exception as error:  # one unreadable trial must not end the count
                    row = {"trial": directory.name, "skipped": f"{type(error).__name__}: {error}"[:300]}
                written.write(json.dumps(row) + "\n")
                written.flush()
    else:
        rows = [json.loads(line) for line in Path(arguments.results).read_text().splitlines()]
        counted = [row for row in rows if "result" in row]
        result = summarize(counted)
        result["skipped"] = Counter(row["skipped"].split(":")[0] for row in rows if "skipped" in row)
        print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
