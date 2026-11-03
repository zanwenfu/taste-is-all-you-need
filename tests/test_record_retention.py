"""How much of a trial's own record reaches each planner call, counted from the record alone."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "record_retention", Path(__file__).resolve().parents[1] / "scripts" / "record_retention.py")
retention = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(retention)

# A command and its evidence line, copied from a real supervised trial (sparql-university).
WRAPPED = ("export PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1; {\n"
           "ls -la /app && sed -n '1,240p' /app/university_graph.ttl\n} 2>&1")
RAW = "ls -la /app && sed -n '1,240p' /app/university_graph.ttl"
LINE = "ran: ls -la /app && sed -n '1,240p' /app/university_graph.ttl -> exit 0; last line printed: uni:role \"Student\" ."
FAILING = "sed -n '240,520p' /app/university_graph.ttl; command -v rapper || true; python - <<'PY'\nimport rdflib\nPY"
FAILING_LINE = ("ran: sed -n '240,520p' /app/university_graph.ttl; command -v rapper || true; python - <<'PY' "
                "import rdflib PY -> exit 127; last line printed: /bin/sh: 2: python: not found")
HEADER = "Mechanical record of this agent's commands, oldest first:"


def test_the_terminal_wrapper_is_taken_off_and_a_bare_command_is_left_alone():
    assert retention.strip_wrapper(WRAPPED) == RAW
    assert retention.strip_wrapper(RAW) == RAW


def test_a_command_matches_its_evidence_line_only_with_its_own_exit():
    assert retention.shown_in(LINE, RAW, 0)
    assert not retention.shown_in(LINE, RAW, 1)
    assert retention.shown_in(FAILING_LINE, FAILING, 127)
    long = "echo " + "x" * 400
    assert retention.shown_in(f"ran: {retention.clip(long, 300)} -> exit 0", long, 0)
    assert retention.shown_in(f"started, result not recorded: {retention.clip(long, 300)}", long, None)


def test_a_refusal_for_size_is_read_exactly_from_the_attempt_record():
    detail = "planner input is 210000 bytes; limit is 196608 bytes"
    assert retention.size_refusal(detail) == (210000, 196608)
    assert retention.size_refusal("world changed before planning call") is None


def plan(generation, created_at, assignment, verdict="not_met"):
    return {"plan_id": f"plan.{generation}", "generation": generation, "created_at": created_at,
            "assignments": [{"assignment_id": assignment}] if assignment else [],
            "assessment": [{"criterion_id": "c1", "verdict": verdict, "evidence": "e"}],
            "rationale": f"why {generation}"}


def report(assignment, run, at, evidence, attempt=0, blob=None):
    return {"report_id": f"report.{assignment}.{attempt}", "assignment_id": assignment, "attempt": attempt,
            "run_id": f"worker-run.{run}", "at": at, "summary": "s", "completed": True,
            "metadata": {"worker_evidence": evidence}, "record_blob_id": blob or f"blob.{assignment}.{attempt}",
            "record_path": f".taste/supervisor/runs/{assignment}/run.json"}


def request(generation, at, parent, outcomes, criteria=("c1",), stale=None):
    return {"generation": generation, "operation_id": f"op.{generation}", "created_at": at, "captured_at": at,
            "bytes": 1000 * generation, "attempts": [{"status": "accepted", "detail": "", "prompt_tokens": 500}],
            "blob_at_call": {o["record_path"]: (stale or {}).get(o["record_path"], o["record_blob_id"])
                             for o in outcomes},
            "request": {"generation": generation,
                        "goal": {"task": "do it", "success_criteria": ["works"]},
                        "criteria": {"revision_id": "rev.1",
                                     "criteria": [{"criterion_id": c, "text": f"text {c}"} for c in criteria]},
                        "parent_plan": parent,
                        "world": {"outcomes": [{"record_blob_id": o["record_blob_id"],
                                                "record_path": o["record_path"],
                                                "report": {k: v for k, v in o.items()
                                                           if k not in ("record_blob_id", "record_path")}}
                                               for o in outcomes]}}}


def command(seq, run, raw, code):
    return {"seq": seq, "actor_id": f"worker_{run}", "command": raw, "code": code,
            "status": "completed" if code is not None else "pending", "terminated": False}


def trial():
    p1, p2, p3 = plan(1, "t01", "a1"), plan(2, "t03", "a2"), plan(3, "t05", "a3")
    r1 = report("a1", "aaa", "t02", [HEADER, LINE, FAILING_LINE])
    # The second worker re-ran the command that had already failed; its record keeps only its last line.
    r2 = report("a2", "bbb", "t04", [HEADER, FAILING_LINE])
    return {
        "goal": {"task": "do it", "success_criteria": ["works"]},
        "criteria": [{"revision_id": "rev.1", "at": "t00",
                      "criteria": [{"criterion_id": "c1", "text": "text c1"}]}],
        "plans": [p1, p2, p3],
        "requests": [request(1, "t00", None, []),
                     request(2, "t025", p1, [r1]),
                     request(3, "t045", p2, [r1, r2])],
        "reports": [r1, r2],
        "runs": [{"run_id": "worker-run.aaa", "assignment_id": "a1", "generation": 1, "created_at": "t01"},
                 {"run_id": "worker-run.bbb", "assignment_id": "a2", "generation": 2, "created_at": "t03"},
                 {"run_id": "worker-run.ccc", "assignment_id": "a3", "generation": 3, "created_at": "t05"}],
        "commands": [command(1, "aaa", RAW, 0), command(2, "aaa", FAILING, 127),
                     command(3, "bbb", RAW, 0), command(4, "bbb", FAILING, 127)],
    }


def test_the_goal_and_standing_criteria_reach_every_call():
    result = retention.analyze(trial())
    kept = result["by_type"]
    assert kept["goal"] == {"held": 3, "sent": 3, "undecided": 0}
    assert kept["criterion"] == {"held": 3, "sent": 3, "undecided": 0}
    assert result["criteria_kept_at_every_call"] is True


def test_only_the_previous_plan_is_sent_and_older_plans_are_counted_as_left_out():
    result = retention.analyze(trial())
    # Call 2 holds plan 1 (sent); call 3 holds plans 1 and 2 (plan 2 sent, plan 1 left out).
    assert result["by_type"]["plan"] == {"held": 3, "sent": 2, "undecided": 0}
    assert result["by_type"]["assessment"] == {"held": 3, "sent": 2, "undecided": 0}
    left = [row for row in result["ages"] if row["type"] == "plan" and not row["sent"]]
    assert left == [{"type": "plan", "age": 1, "sent": False, "count": 1}]


def test_a_command_bounded_out_of_its_worker_record_is_counted_as_left_out():
    result = retention.analyze(trial())
    # Call 2: run aaa's 2 commands both shown. Call 3: aaa's 2 shown, bbb's first dropped, last shown.
    assert result["by_type"]["command"] == {"held": 6, "sent": 5, "undecided": 0}
    assert result["by_type"]["report"] == {"held": 3, "sent": 3, "undecided": 0}


def test_a_repeat_of_a_recorded_failure_is_split_by_whether_the_failure_was_shown():
    result = retention.analyze(trial())
    assert result["repeats"]["commands"] == 1 and result["repeats"]["assignments"] == 1
    assert result["repeats"]["failure_shown"] == 1 and result["repeats"]["failure_not_shown"] == 0
    hidden = trial()
    hidden["requests"][1]["request"]["world"]["outcomes"][0]["report"]["metadata"]["worker_evidence"] = [HEADER, LINE]
    result = retention.analyze(hidden)
    assert result["repeats"]["failure_shown"] == 0 and result["repeats"]["failure_not_shown"] == 1


def test_a_command_of_a_run_still_going_at_the_call_is_undecided_not_guessed():
    data = trial()
    data["commands"].append(command(5, "ccc", "make", 2))
    data["requests"].append(request(4, "t055", data["plans"][2], data["reports"]))
    result = retention.analyze(data)
    assert result["by_type"]["command"]["undecided"] == 1


def test_a_record_sent_after_it_was_superseded_is_counted():
    data = trial()
    path = data["reports"][0]["record_path"]
    data["requests"][2]["blob_at_call"][path] = "blob.newer"
    result = retention.analyze(data)
    assert result["superseded"]["stale_record_sent"] == 1
    assert result["superseded"]["plans_held"] == 1 and result["superseded"]["plans_sent"] == 0


def test_an_older_attempt_of_the_same_assignment_is_superseded_and_its_sending_counted():
    data = trial()
    newer = report("a1", "ddd", "t042", [HEADER, LINE], attempt=1)
    data["reports"].append(newer)
    data["requests"][2]["request"]["world"]["outcomes"].append(
        {"record_blob_id": newer["record_blob_id"], "record_path": newer["record_path"] + "x",
         "report": {k: v for k, v in newer.items() if k not in ("record_blob_id", "record_path")}})
    result = retention.analyze(data)
    assert result["superseded"]["reports_held"] == 1 and result["superseded"]["reports_sent"] == 1


def test_sizes_are_reported_against_the_cap_and_a_trial_without_planner_calls_is_said_to_have_none():
    result = retention.analyze(trial())
    assert result["calls"][2]["request_bytes"] == 3000
    assert result["calls"][2]["share_of_cap_lower_bound"] == round(3000 / retention.MAX_PROMPT_BYTES, 4)
    assert retention.analyze({**trial(), "requests": []})["calls"] == []


def test_the_cap_is_the_one_the_planner_transport_enforces():
    import inspect

    from taste.brains.planner_transport import LLMPlannerTransport

    default = inspect.signature(LLMPlannerTransport.__init__).parameters["max_prompt_bytes"].default
    assert default == retention.MAX_PROMPT_BYTES
